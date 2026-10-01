"""ISLA-PERF-2 (2026-10-01): where does a REAL ISLA training step spend its time and memory?

Runs under torchrun exactly like a fleet lane (4 ranks, one GB300 each, DDP, batch 1, bf16 autocast,
the recipe's reader, transforms, collate, loss and optimizer; code_v14 + recipe_v6), on the chosen
dataset and model yaml at 200,000 cells, and measures per step and per rank, with a CUDA sync after
each phase:

  wait_s      time blocked in next(iterator): the dataloader (reader, transforms, collate, prefetch)
  h2d_s       recursive_to_device of the batch
  fwd_s       model forward + loss + metrics (autocast)
  bwd_s       loss.backward (includes the DDP all-reduce)
  opt_s       optimizer.step + zero_grad
  step_s      wall time of the whole step

Steps 0-2 are warm-up, 3-11 are timed; steps 6-8 run under torch.profiler on every rank (CPU + CUDA,
shapes recorded) and rank 0 writes the CUDA kernel time bucketed by operator family plus the top-40
kernels and a chrome trace. One extra step on rank 0 runs with torch.autograd saved-tensor hooks that
attribute every saved activation (bytes, shape, dtype) to the innermost physicsnemo frame that created
it, and with torch.cuda memory-history recording for a snapshot pickle. Output: one JSON per task.

Usage (inside the sbatch): torchrun --standalone --nproc-per-node 4 isla_train_profile.py <dataset> <model> <out.json> [k=v hydra overrides...]
"""
from __future__ import annotations

import collections
import json
import os
import re
import statistics as st
import sys
import time

import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

RECIPE = os.environ["RECIPE"]
sys.path.insert(0, f"{RECIPE}/src")

from datasets import build_dataloaders  # noqa: E402
from loss import LossCalculator  # noqa: E402
from metrics import MetricCalculator, resolve_metrics  # noqa: E402
from output_normalize import require_output_type  # noqa: E402
from train import forward_pass  # noqa: E402
from utils import build_muon_optimizer, recursive_to_device, resolve_dict, set_seed  # noqa: E402

import hydra  # noqa: E402
from physicsnemo.distributed import DistributedManager  # noqa: E402

dataset, model_name, out_path, *extra = sys.argv[1:]
N_WARM, N_TIMED, PROF = 3, 9, (6, 7, 8)

overrides = [
    f"model={model_name}", f"dataset={dataset}", "sampling_resolution=200000",
    "run_id=perf_profile", f"output_dir={os.environ['TASK']}/kernel/perf/runs",
    f"checkpoint_dir={os.environ['TASK']}/kernel/perf/runs", "training.seed=42",
    "training.num_epochs=1", "precision=bfloat16", "augment=false", "compile=false",
    "logging.log_every_n_steps=1000", *extra,
]
with initialize_config_dir(config_dir=f"{RECIPE}/conf", version_base=None):
    cfg = compose(config_name="train", overrides=overrides)

DistributedManager.initialize()
dm = DistributedManager()
dev = dm.device
rank, world = dm.rank, dm.world_size
set_seed(42, rank=rank)
### TF32 left at the lanes default (off): with it on, the GPU-side mesh transforms quantize coordinates and about 22% of the DrivAerML surface cells collapse to zero area (2026-10-01)

train_loader, val_loader, normalizer, dataset_info = build_dataloaders(cfg)
target_config = dataset_info["targets"]
model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(dev)
n_params = sum(p.numel() for p in model.parameters())
if world > 1:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dm.local_rank], output_device=dev)
optimizer = build_muon_optimizer(model, cfg, compile_optimizer=False)
loss_calc = LossCalculator(target_config=target_config, loss_type=cfg.training.get("loss_type", "huber"),
                           field_weights=resolve_dict(cfg, "training.field_weights"))
metric_calc = MetricCalculator(target_config=target_config, metrics=resolve_metrics(cfg))
output_type = require_output_type(cfg)
model.train()
train_loader.set_epoch(0)
_m = model.module if hasattr(model, "module") else model
_blocks = [b for b in _m.modules() if type(b).__name__ == "_SliceBlock"]
geo_resolved = {"geo_kernel": sorted({str(b.geo_kernel) for b in _blocks}), "geo_checkpoint": sorted({bool(b.geo_checkpoint) for b in _blocks}), "n_blocks": len(_blocks)}
if rank == 0:
    print("GEO RESOLVED", geo_resolved, flush=True)


def sync():
    torch.cuda.synchronize(dev)


def one_step(it, prof=None):
    t = {}
    t0 = time.perf_counter()
    batch = next(it); t["wait_s"] = time.perf_counter() - t0
    t1 = time.perf_counter(); batch = recursive_to_device(batch, dev); sync(); t["h2d_s"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    loss, _, _ = forward_pass(batch, model, cfg.precision, loss_calc, metric_calc, output_type=output_type, target_config=target_config)
    sync(); t["fwd_s"] = time.perf_counter() - t2
    t3 = time.perf_counter(); optimizer.zero_grad(); loss.backward(); sync(); t["bwd_s"] = time.perf_counter() - t3
    t4 = time.perf_counter(); optimizer.step(); sync(); t["opt_s"] = time.perf_counter() - t4
    t["step_s"] = time.perf_counter() - t0
    t["n_tokens"] = int(_count_tokens(batch))
    t["loss"] = float(loss.detach())
    if prof is not None:
        prof.step()
    return t


def _count_tokens(batch):
    fk = batch["forward_kwargs"]
    for v in _walk(fk):
        if torch.is_tensor(v) and v.dim() >= 2 and v.shape[-1] in (2, 3) and v.numel() > 1000:
            return v.reshape(-1, v.shape[-1]).shape[0]
    return -1


def _walk(x):
    if torch.is_tensor(x):
        yield x
    elif isinstance(x, dict) or hasattr(x, "items"):
        for v in x.values():
            yield from _walk(v)
    elif hasattr(x, "points"):
        yield x.points
    elif hasattr(x, "interior"):
        yield from _walk(x.interior)


BUCKETS = [
    ("geo_fused_triton", r"_geo_|geo_region|fused|triton"),
    ("softmax", r"[Ss]oftmax|softmax"),
    ("gemm", r"gemm|cutlass|Gemm|nvjet|cublas|matmul|bmm|xmma"),
    ("layernorm_gelu", r"layer_norm|LayerNorm|gelu|Gelu"),
    ("elementwise", r"elementwise|vectorized|unrolled_elementwise|fill|copy_|CatArray|cat_|transpose|Transpose|index|gather|scatter|reduce_kernel|sum_|mul_|add_|where|clamp|norm"),
    ("knn_cuml", r"cuml|knn|raft|faiss|brute"),
    ("optimizer", r"adam|Adam|muon|Muon|foreach|multi_tensor|lerp|addcmul|addcdiv|sqrt"),
    ("nccl", r"nccl|ncclKernel|AllReduce|allreduce"),
    ("memcpy", r"Memcpy|memcpy|Memset|memset"),
]


def bucket(name):
    for b, pat in BUCKETS:
        if re.search(pat, name):
            return b
    return "other"


it = iter(train_loader)
timed = []
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                            schedule=torch.profiler.schedule(wait=PROF[0], warmup=0, active=len(PROF), repeat=1),
                            record_shapes=True, profile_memory=False, with_stack=False) as prof:
    for i in range(N_WARM + N_TIMED):
        t = one_step(it, prof)
        t["step"] = i
        if i >= N_WARM:
            timed.append(t)
        if rank == 0:
            print(f"step {i}: " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in t.items()), flush=True)

peak_gib = torch.cuda.max_memory_allocated(dev) / 2**30
reserved_gib = torch.cuda.max_memory_reserved(dev) / 2**30

### kernel buckets from the profiled steps (per rank), total CUDA time per step
### device kernels only (CPU-side op and annotation rows carry the device time of their children and would double count)
ka = prof.key_averages()
cuda_us = collections.Counter(); top = []; ops = []
for e in ka:
    cu = getattr(e, "self_device_time_total", None)
    if cu is None:
        cu = getattr(e, "self_cuda_time_total", 0.0)
    if not cu or cu <= 0:
        continue
    is_kernel = str(getattr(e, "device_type", "")).endswith("CUDA")
    if is_kernel:
        cuda_us[bucket(e.key)] += cu / len(PROF)
        top.append((e.key[:120], cu / len(PROF) / 1000.0, e.count // len(PROF)))
    else:
        ops.append((e.key[:120], cu / len(PROF) / 1000.0, e.count // len(PROF)))
top.sort(key=lambda x: -x[1]); ops.sort(key=lambda x: -x[1])
kernels = {"bucket_ms_per_step": {k: v / 1000.0 for k, v in cuda_us.most_common()}, "top40_kernels_ms_per_step": top[:40],
           "top40_ops_device_ms_per_step": ops[:40], "total_kernel_ms_per_step": sum(cuda_us.values()) / 1000.0}

### saved-activation attribution + memory snapshot on rank 0 (one more step)
saved = None
if rank == 0:
    ### saved-activation attribution by the module whose forward is running (forward hooks set a cursor; the pack
    ### hook reads it): no stack capture, no memory-history recording (both made rank 0 miss a 40-minute wall).
    acc = collections.defaultdict(lambda: [0, 0])  # module path -> [bytes, count]
    shapes = collections.Counter(); cursor = ["<outside modules>"]; handles = []
    for name, mod in _m.named_modules():
        if name == "":
            continue
        handles.append(mod.register_forward_pre_hook(lambda m, inp, _n=name: cursor.__setitem__(0, _n)))
        handles.append(mod.register_forward_hook(lambda m, inp, out, _n=name: cursor.__setitem__(0, _n.rsplit(".", 1)[0] if "." in _n else "<root>")))

    def pack(t):
        if torch.is_tensor(t) and t.is_cuda:
            nb = t.numel() * t.element_size()
            key = re.sub(r"blocks\.\d+", "blocks.*", cursor[0])
            acc[key][0] += nb; acc[key][1] += 1
            shapes[(tuple(t.shape), str(t.dtype).replace("torch.", ""))] += nb
        return t

    torch.cuda.reset_peak_memory_stats(dev)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        t_extra = one_step(it)
    for h in handles:
        h.remove()
    total = sum(v[0] for v in acc.values())
    saved = {"total_saved_gib": total / 2**30, "peak_gib_extra_step": torch.cuda.max_memory_allocated(dev) / 2**30,
             "by_module_gib": sorted(({"module": k, "gib": v[0] / 2**30, "count": v[1]} for k, v in acc.items()), key=lambda d: -d["gib"])[:40],
             "by_shape_gib": [{"shape": list(s[0]), "dtype": s[1], "gib": b / 2**30} for s, b in shapes.most_common(25)]}
    with open(out_path.replace(".json", "_key_averages.txt"), "w") as f:
        f.write(ka.table(sort_by="self_device_time_total", row_limit=150))
else:
    ### every rank must run the extra step: its backward is a DDP all-reduce (a lone rank 0 deadlocked until the NCCL timeout)
    t_extra = one_step(it)

### gather per-rank timings through files (all_gather_object mis-sizes its buffer under NCCL here)
mine = {"rank": rank, "timed": timed, "peak_gib": peak_gib, "reserved_gib": reserved_gib, "kernels": kernels}
json.dump(mine, open(out_path.replace(".json", f".rank{rank}.json"), "w"), default=str)
dist.barrier()
if rank == 0:
    rows = [json.load(open(out_path.replace(".json", f".rank{r}.json"))) for r in range(world)]
    def med(key):
        return {str(r["rank"]): st.median(x[key] for x in r["timed"]) for r in rows}
    summary = {k: med(k) for k in ("wait_s", "h2d_s", "fwd_s", "bwd_s", "opt_s", "step_s")}
    summary["slowest_rank_step_s_median"] = st.median(max(r["timed"][j]["step_s"] for r in rows) for j in range(len(timed)))
    out = {"protocol": __doc__, "dataset": dataset, "model": model_name, "overrides": overrides, "world_size": world,
           "n_params": n_params, "gpu": torch.cuda.get_device_name(dev), "torch": torch.__version__,
           "n_tokens": timed[0]["n_tokens"], "geo_resolved": geo_resolved, "summary_median_by_rank": summary,
           "per_rank": rows, "saved_activations_rank0": saved,
           "resolved_model_cfg": OmegaConf.to_container(cfg.model, resolve=True)}
    json.dump(out, open(out_path, "w"), indent=1, default=str)
    print("SUMMARY", json.dumps(summary, indent=1))
    print("KERNEL BUCKETS rank0 (ms/step)", json.dumps(kernels["bucket_ms_per_step"], indent=1))
    print("TOP KERNELS rank0", json.dumps(kernels["top40_kernels_ms_per_step"][:20], indent=1))
    print("TOP OPS (device time, overlapping) rank0", json.dumps(kernels["top40_ops_device_ms_per_step"][:15], indent=1))
    if saved:
        print("SAVED ACTIVATIONS", json.dumps({k: saved[k] for k in ("total_saved_gib", "peak_gib_extra_step")}), json.dumps(saved["by_module_gib"][:14], indent=1), json.dumps(saved["by_shape_gib"][:10], indent=1))
    print("PROFILE-DONE", out_path, flush=True)
dist.barrier()
