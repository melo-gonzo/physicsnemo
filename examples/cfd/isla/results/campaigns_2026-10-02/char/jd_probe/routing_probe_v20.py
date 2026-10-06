"""JEB-PLATEAU routing probe (code_v20 ISLA): per-block slice-routing statistics of trained checkpoints.

For one run id and a list of epochs, load the checkpoint the way recipe_v8/src/infer.py does (Hydra compose of
conf/infer.yaml with the lane's overrides, build_dataloaders, hydra.utils.instantiate(cfg.model), load_checkpoint,
persisted norm stats), run float32 eval-mode forwards over the first K validation cases, and record per slice block:
  * n_eff        : exp of the entropy of the slice mass distribution q_s = mean_n w[n, s], w = softmax over slices;
  * H_slices     : mean over tokens of the entropy of w[n, :] divided by log S (1 = uniform over slices, 0 = one slice);
  * frac_active  : fraction of slices with q_s > 1 / (4 S);
  * anchor_spread: RMS distance of the slice anchors z_s = sum_n a_ns r_n (a = the block's softmax over points,
                   exactly the anchors the block forms) from their mean, over the RMS radius of the token positions.
The first three are reported for two routing distributions: "pre" = softmax over slices of the cosine routing logits
against the centered sample-conditioned prototypes (block.assign output), and "post" = the same after the geometric
bias (the block's point_mix, which the readback actually uses).

Hook points (code_v20 physicsnemo/experimental/models/isla/blocks.py, _SliceBlock.forward):
  logits = self.assign(self.norm_assign(h), log_w)         -> forward hook on block.assign records the logits
  a = _softmax_over_points(logits + log_w)                 -> wrapped: call 1 per block, records input and output
  bias, point_mix, pooled = <geo region>                   -> not hooked (fused Triton kernel)
  a = _softmax_over_points(logits + bias + log_w)          -> wrapped: call 2 per block; bias = input2 - input1
A forward pre-hook on each block records log_w and r. compile_blocks is forced off so the eager forward runs with
the hooks; the arithmetic is identical.

usage: python routing_probe_v20.py --run RUN_ID --epochs 1 4 --k 4 --out-dir DIR
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

import torch

T = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"
RECIPE = f"{T}/recipe_v8"
MANIFEST = f"{T}/kernel/char/cl_smoke.tsv"


def entropy(p, dim=-1, eps=1e-30):
    return -(p * (p + eps).log()).sum(dim)


def routing_stats(w: torch.Tensor) -> dict:
    """w: (N, S) routing weights over slices per token (rows sum to one)."""
    n, s = w.shape
    q = w.mean(0)  # (S,) slice mass distribution
    return {
        "n_eff": float(torch.exp(entropy(q))),
        "H_slices": float((entropy(w, -1) / math.log(s)).mean()),
        "frac_active": float((q > 1.0 / (4 * s)).float().mean()),
        "mean_top_weight": float(w.max(-1).values.mean()),
        "max_slice_mass": float(q.max()),
    }


def anchor_spread(a: torch.Tensor, r: torch.Tensor) -> float:
    """a: (N, S) softmax over points; r: (N, 3) token positions."""
    z = torch.einsum("ns,nc->sc", a, r)
    spread = (z - z.mean(0, keepdim=True)).norm(dim=-1).pow(2).mean().sqrt()
    rms_r = (r - r.mean(0, keepdim=True)).norm(dim=-1).pow(2).mean().sqrt().clamp_min(1e-12)
    return float(spread / rms_r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--epochs", type=int, nargs="*", default=[], help="epochs to probe (default: every ISLA.0.<epoch>.mdlus present)")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--cells", type=int, default=200000)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    import hydra
    from hydra import compose, initialize_config_dir

    from datasets import build_dataloaders, find_normalizer
    from utils import recursive_to_device, set_seed
    from physicsnemo import datapipes  # noqa: F401
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint
    import physicsnemo.experimental.models.isla.blocks as isla_blocks

    rows = {r["run_id"]: r for r in csv.DictReader(open(MANIFEST), delimiter="\t")}
    row = rows[args.run]
    extra = [t for t in (row["extra"] if row["extra"] != "-" else "").split() if not t.startswith("+training.")]

    DistributedManager.initialize()
    device = DistributedManager().device
    set_seed(42, rank=0)

    overrides = [
        f"model={row['model']}",
        f"dataset={row['dataset']}",
        f"run_id={args.run}",
        "infer_split=test_clean",
        "+save_predictions=false",
        f"sampling_resolution={args.cells}",
        "precision=float32",
        f"checkpoint_dir={T}/runs",
        f"output_dir={args.out_dir}/_scratch",
        "model.compile_blocks=false",
        *extra,
    ]
    with initialize_config_dir(config_dir=f"{RECIPE}/conf", version_base=None):
        cfg = compose(config_name="infer", overrides=overrides)
    print("overrides:", overrides, flush=True)

    _train_loader, val_loader, normalizer, _info = build_dataloaders(cfg)
    model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(device)
    ckpt_path = os.path.join(T, "runs", args.run, "checkpoints")
    norm_stats_path = os.path.join(ckpt_path, "norm_stats.pt")
    if normalizer is not None and os.path.exists(norm_stats_path):
        saved = torch.load(norm_stats_path, weights_only=True)
        val_normalizer = find_normalizer([val_loader.dataset])
        for live in {id(n): n for n in (normalizer, val_normalizer) if n is not None}.values():
            live.stats = saved
    model.eval()
    blocks = list(model.blocks)
    nb = len(blocks)
    n_slices = int(blocks[0].assign.weight.shape[0])

    cap = {"assign": [], "sp": [], "pre": []}

    def assign_hook(mod, inputs, output):
        cap["assign"].append(output.detach())

    def block_pre_hook(mod, inputs):
        # _SliceBlock.forward(h, log_w, r, n_hat, g_hat, eps, vecs, tens)
        cap["pre"].append((inputs[1].detach(), inputs[2].detach()))

    handles = [b.assign.register_forward_hook(assign_hook) for b in blocks]
    handles += [b.register_forward_pre_hook(block_pre_hook) for b in blocks]
    orig_sp = isla_blocks._softmax_over_points

    def sp_rec(x):
        a = orig_sp(x)
        cap["sp"].append((x.detach(), a.detach()))
        return a

    isla_blocks._softmax_over_points = sp_rec

    dataset = val_loader.dataset
    collate_fn = val_loader.collate_fn
    idxs = list(val_loader.sampler)[: args.k]
    print(f"{len(list(val_loader.sampler))} {cfg.infer_split} samples; using {idxs}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    available = sorted(int(f.split(".")[2]) for f in os.listdir(ckpt_path) if f.startswith("ISLA.0.") and f.endswith(".mdlus"))
    print("available ISLA.0.<epoch>.mdlus epochs:", available, flush=True)
    epochs = [e for e in args.epochs if e in available] if args.epochs else available
    if not epochs:
        epochs = available[-1:]
    print("probing epochs:", epochs, flush=True)
    for epoch in epochs:
        if os.path.exists(os.path.join(ckpt_path, f"checkpoint.0.{epoch}.pt")):
            loaded = load_checkpoint(path=ckpt_path, models=model, device=device, epoch=epoch)
            assert loaded == epoch, (loaded, epoch)
        else:
            ### the smoke lanes prune the training-state file of earlier epochs; load the weights file directly
            model.load(os.path.join(ckpt_path, f"ISLA.0.{epoch}.mdlus"))
            model.to(device)
        model.eval()
        per_block = [{"pre": [], "post": [], "anchor_spread": [], "logit_std": [], "bias_std": []} for _ in range(nb)]
        n_tokens = []
        t0 = time.time()
        for k, idx in enumerate(idxs):
            for v in cap.values():
                v.clear()
            sample = dataset[idx]
            batch = recursive_to_device(collate_fn([sample]), device)
            with torch.no_grad():
                model(**batch["forward_kwargs"])
            assert len(cap["assign"]) == nb and len(cap["pre"]) == nb and len(cap["sp"]) == 2 * nb, (
                len(cap["assign"]), len(cap["pre"]), len(cap["sp"]), nb)
            for i in range(nb):
                logits = cap["assign"][i][0].float()  # (N, S) cosine logits, pre-bias
                log_w, r = cap["pre"][i]
                log_w = log_w[0].float()  # (N, 1)
                x1, a1 = cap["sp"][2 * i]
                x2, _a2 = cap["sp"][2 * i + 1]
                x1, a1, x2 = x1[0].float(), a1[0].float(), x2[0].float()
                assert torch.allclose(x1, logits + log_w, atol=1e-3), "call bookkeeping mismatch (call 1)"
                bias = x2 - x1  # (N, S) geometric routing bias
                w_pre = torch.softmax(logits, -1)
                w_post = torch.softmax(logits + bias, -1)
                per_block[i]["pre"].append(routing_stats(w_pre))
                per_block[i]["post"].append(routing_stats(w_post))
                per_block[i]["anchor_spread"].append(anchor_spread(a1, r[0].float()))
                per_block[i]["logit_std"].append(float(logits.std()))
                per_block[i]["bias_std"].append(float(bias.std()))
            n_tokens.append(int(logits.shape[0]))
            print(f"epoch {epoch} case {k} idx {idx} tokens {n_tokens[-1]} t={time.time() - t0:.1f}s", flush=True)

        def mean(lst):
            return float(sum(lst) / len(lst))

        result = {
            "run": args.run,
            "epoch": epoch,
            "code": "code_v20",
            "model": row["model"],
            "dataset": row["dataset"],
            "split": cfg.infer_split,
            "n_cases": len(idxs),
            "sample_indices": [int(i) for i in idxs],
            "n_tokens": n_tokens,
            "n_blocks": nb,
            "n_slices": n_slices,
            "hidden": int(blocks[0].assign.weight.shape[1]),
            "active_threshold": 1.0 / (4 * n_slices),
            "blocks": [],
        }
        for i in range(nb):
            pb = per_block[i]
            result["blocks"].append({
                "block": i,
                "pre_bias": {key: mean([d[key] for d in pb["pre"]]) for key in pb["pre"][0]},
                "post_bias": {key: mean([d[key] for d in pb["post"]]) for key in pb["post"][0]},
                "anchor_spread": mean(pb["anchor_spread"]),
                "logit_std": mean(pb["logit_std"]),
                "bias_std": mean(pb["bias_std"]),
                "gain": float(blocks[i].assign.gain.detach()),
            })
        out = os.path.join(args.out_dir, f"routing_{args.run}_e{epoch}.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=1)
        print(f"\n{args.run} epoch {epoch} cases {len(idxs)} S={n_slices} tokens={n_tokens}", flush=True)
        print(f"{'blk':>3} | {'n_eff_pre':>9} {'H_sl_pre':>8} {'act_pre':>7} | {'n_eff_post':>10} {'H_sl_post':>9} {'act_post':>8} | {'anchor':>6} {'lstd':>5} {'bstd':>5}")
        for b in result["blocks"]:
            p, q = b["pre_bias"], b["post_bias"]
            print(f"{b['block']:>3} | {p['n_eff']:9.1f} {p['H_slices']:8.3f} {p['frac_active']:7.3f} | "
                  f"{q['n_eff']:10.1f} {q['H_slices']:9.3f} {q['frac_active']:8.3f} | "
                  f"{b['anchor_spread']:6.3f} {b['logit_std']:5.2f} {b['bias_std']:5.2f}", flush=True)
        print("wrote", out, flush=True)

    for h in handles:
        h.remove()
    isla_blocks._softmax_over_points = orig_sp


if __name__ == "__main__":
    main()
