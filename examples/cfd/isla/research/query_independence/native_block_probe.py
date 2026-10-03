# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Region-matched support readout: the support arm scored on the native protocol's own evaluation points.

Each held-out case is read twice: through the protocol's dataset (the native request, one contiguous cyclic block of
the stored point order) and through an every-point variant. The native request's points are located in the
every-point read by their (bitwise identical) coordinates. On exactly those points, float32:

- ``native``: the interacting model on its native request (the protocol's number);
- ``self_support``: the native request as its own support, read passively (equal to ``native`` by construction);
- ``random_support``: a seeded random sample of the training count drawn from the points *outside* the native request
  as the support, the native request's points read passively (query-independent, the support independent of them);
- ``random_support_all``: the same with the random support drawn from every point.

usage: python native_block_probe.py --model M --dataset NATIVE --dataset-all ALL --run-id R --cells N --request N
       --extra "..." --task T --out O.json
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

FIELDS = ("pressure", "velocity", "nut")


def log(msg):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser()
    for k in ("model", "dataset", "dataset-all", "run-id", "task", "out"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--cells", type=int, required=True)
    p.add_argument("--request", type=int, required=True)
    p.add_argument("--extra", default="")
    p.add_argument("--max-cases", type=int, default=0)
    a = p.parse_args()
    for k, v in dict(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(29000 + os.getpid() % 20000)).items():
        os.environ.setdefault(k, v)
    import hydra
    from datasets import build_dataloaders, find_normalizer
    from hydra import compose, initialize_config_dir
    from metrics import MetricCalculator, resolve_metrics
    from output_normalize import normalize_output_to_tensordict, require_output_type
    from utils import get_autocast_context, recursive_to_device

    from physicsnemo import datapipes  # noqa: F401
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint

    runs = f"{a.task}/runs"

    def loader(ds):
        ov = [f"model={a.model}", *[o for o in a.extra.split() if "training." not in o], f"dataset={ds}", f"run_id={a.run_id}",
              "infer_split=val", f"sampling_resolution={a.cells}", "precision=float32", f"checkpoint_dir={runs}", "++model.compile_blocks=false"]
        with initialize_config_dir(config_dir=str(Path.cwd() / "conf"), version_base=None):
            cfg = compose(config_name="infer", overrides=ov)
        return cfg, build_dataloaders(cfg)

    DistributedManager.initialize()
    dev = DistributedManager().device
    cfg, (_, vl, normalizer, info) = loader(a.dataset)
    _, (_, vl_all, normalizer_all, _) = loader(a.dataset_all)
    tc, ot = info["targets"], require_output_type(cfg)
    model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(dev)
    ck = str(Path(runs) / a.run_id / "checkpoints")
    ep = load_checkpoint(path=ck, models=model, device=dev)
    sp = Path(ck) / "norm_stats.pt"
    if sp.exists():
        st = torch.load(sp, weights_only=True)
        for live in {id(n): n for n in (normalizer, normalizer_all, find_normalizer([vl.dataset]), find_normalizer([vl_all.dataset])) if n is not None}.values():
            live.stats = st
    model.eval()
    calc = MetricCalculator(target_config=tc, metrics=resolve_metrics(cfg))
    log(f"{a.run_id}: epoch {ep}; native dataset {a.dataset}, every-point dataset {a.dataset_all}")

    def predict(fk, pts, support=None):
        with torch.no_grad(), get_autocast_context(cfg.precision):
            out = model(**{**fk, "prediction_points": pts, "support_points": support})
        return normalize_output_to_tensordict(out, tc, ot).float()

    def scores(pred, tgt):
        m = calc(pred, tgt, None)
        return {f: float(m[f"{f}_l2"]) for f in FIELDS}

    cases = []
    idxs = list(vl.sampler)
    idxs_all = list(vl_all.sampler)
    if idxs != idxs_all:
        raise ValueError("the two held-out loaders iterate different cases")
    if a.max_cases > 0:
        idxs = idxs[: a.max_cases]
    for ci, idx in enumerate(idxs):
        t0 = time.time()
        sn, sa = vl.dataset[idx], vl_all.dataset[idx]
        if sn[1].get("source_path") != sa[1].get("source_path"):
            raise ValueError(f"case {ci}: different source files {sn[1].get('source_path')} / {sa[1].get('source_path')}")
        bn = recursive_to_device(vl.collate_fn([sn]), dev)
        ba = recursive_to_device(vl_all.collate_fn([sa]), dev)
        fkn, ppn, tn = bn["forward_kwargs"], bn["forward_kwargs"]["prediction_points"], bn["targets"].float()
        fka, ppa = ba["forward_kwargs"], ba["forward_kwargs"]["prediction_points"]
        ### locate the native request's points among every point by their coordinates (bitwise equal: one file); the key
        ### packs the float32 bit patterns of two coordinates into one int64 (exact for planar data, a check in 3-D)
        def key(x):
            b = x.float().contiguous().view(torch.int32).long() & 0xFFFFFFFF
            return (b[:, 0] << 32) | b[:, 1]
        ka, kn = key(ppa.points), key(ppn.points)
        order = torch.argsort(ka)
        pos = torch.searchsorted(ka[order], kn).clamp_max(ka.numel() - 1)
        hit = order[pos]
        exact = bool((ppa.points[hit] == ppn.points).all())
        inside = torch.zeros(ppa.n_points, dtype=torch.bool, device=dev)
        inside[hit] = True
        outside = (~inside).nonzero(as_tuple=True)[0]
        gen = torch.Generator().manual_seed(7000 + ci)
        r_out = outside[torch.randperm(outside.numel(), generator=gen)[: a.request].to(dev)]
        r_all = torch.randperm(ppa.n_points, generator=gen)[: a.request].to(dev)
        q = ppa.slice_points(hit)  # the native request's points, from the every-point read (same coordinates and fields)
        native = predict(fkn, ppn)
        rec = {"case": ci, "n_all": int(ppa.n_points), "n_native": int(ppn.n_points), "n_outside": int(outside.numel()), "exact_match": exact,
               "native": scores(native, tn),
               "self_support": scores(predict(fka, q, support=q), tn),
               "random_support_outside": scores(predict(fka, q, support=ppa.slice_points(r_out)), tn) if outside.numel() >= a.request // 2 else None,
               "random_support_all": scores(predict(fka, q, support=ppa.slice_points(r_all)), tn),
               "seconds": round(time.time() - t0, 2)}
        cases.append(rec)
        log(f"case {ci + 1}/{len(idxs)} all={rec['n_all']} native={rec['n_native']} match={exact} | native " + " ".join(f"{f}={rec['native'][f]:.4f}" for f in FIELDS)
            + " | random support (all) / native " + " ".join(f"{f}={rec['random_support_all'][f] / rec['native'][f]:.3f}" for f in FIELDS)
            + (" | random support (outside) / native " + " ".join(f"{f}={rec['random_support_outside'][f] / rec['native'][f]:.3f}" for f in FIELDS) if rec["random_support_outside"] else "")
            + f" | self-support / native p={rec['self_support']['pressure'] / rec['native']['pressure']:.4f} ({rec['seconds']} s)")
    summ = {}
    for k in ("native", "self_support", "random_support_all", "random_support_outside"):
        cs = [c for c in cases if c[k] is not None]
        if cs:
            summ[k] = {f: sum(c[k][f] for c in cs) / len(cs) for f in FIELDS}
            summ[f"{k}_n_cases"] = len(cs)
    summ["random_support_all_over_native"] = {f: summ["random_support_all"][f] / summ["native"][f] for f in FIELDS}
    if "random_support_outside" in summ:
        cs = [c for c in cases if c["random_support_outside"] is not None]
        nat = {f: sum(c["native"][f] for c in cs) / len(cs) for f in FIELDS}
        summ["random_support_outside_over_native_same_cases"] = {f: summ["random_support_outside"][f] / nat[f] for f in FIELDS}
    summ["all_exact_match"] = all(c["exact_match"] for c in cases)
    out = {"run_id": a.run_id, "epoch": ep, "dataset": a.dataset, "dataset_all": a.dataset_all, "request": a.request,
           "n_cases": len(cases), "summary": summ, "cases": cases}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".tmp").write_text(json.dumps(out, indent=1))
    os.replace(a.out + ".tmp", a.out)
    log(f"wrote {a.out}: " + json.dumps({k: v for k, v in summ.items() if k.endswith("over_native") or k.endswith("same_cases") or k == "all_exact_match"}))


if __name__ == "__main__":
    main()
