"""ROUTING-SAT probe: routing statistics of trained ISLA checkpoints on real validation cars.

For each run named on the command line, compose the recipe's inference config with that run's manifest
overrides, build the validation loader, load the checkpoint, and run the first K validation cases in float32
while recording, per slice block:
  * H_points : mean over slices of the entropy of the routing weights over POINTS (the softmax ISLA uses),
               normalized by log N (1 = uniform over points, 0 = one point per slice);
  * H_slices : mean over points of the entropy of the same logits normalized over SLICES, divided by log S
               (1 = a point spread over all slices, 0 = one slice per point: "routing hardness");
  * n_eff    : effective number of slices carrying mass, exp of the entropy of the slice mass distribution
               q_s = mean_n softmax_s(logits)[n, s];
  * anchor_spread : RMS distance of the slice anchors z_s = sum_n a_ns r_n from their mean, divided by the RMS
               radius of the token positions r (1 = anchors spread like the points, 0 = anchors coincide).
Writes <out_dir>/routing_<run>.json with per-layer means over the K cases.

usage: python routing_probe.py <manifest.tsv> <out_dir> <K> <run_id> [<run_id> ...]
"""
from __future__ import annotations

import csv, json, math, sys
from pathlib import Path

import hydra, torch
from omegaconf import OmegaConf

TASK = Path("/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0")
RECIPE = TASK / "recipe_globin"
manifest, out_dir, K = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
runs = sys.argv[4:]
out_dir.mkdir(parents=True, exist_ok=True)
rows = {r["run_id"]: r for r in csv.DictReader(open(manifest), delimiter="\t")}

import physicsnemo.experimental.nn.isla.model as isla_model  # noqa: E402
from physicsnemo.utils import load_checkpoint  # noqa: E402

REC: dict = {"a": [], "r": []}
_orig_softmax = isla_model._softmax_over_points


def _rec_softmax(x, fast=True):
    a = _orig_softmax(x, fast)
    REC["a"].append((x.detach(), a.detach()))
    return a


isla_model._softmax_over_points = _rec_softmax


def _pre_hook(mod, args):
    # _SliceBlock.forward(h, log_w, r, n_hat, g_hat, eps)
    REC["r"].append(args[2].detach())


def layer_stats(x, a, r):
    """x, a: (B, N, S) logits-with-bias and softmax-over-points; r: (B, N, 3)."""
    b, n, s = a.shape
    eps = 1e-30
    h_points = -(a * (a + eps).log()).sum(1) / math.log(n)          # (B, S)
    p = torch.softmax(x, dim=-1)                                       # over slices per point
    h_slices = -(p * (p + eps).log()).sum(-1) / math.log(s)           # (B, N)
    q = p.mean(1)                                                       # (B, S) slice mass share
    n_eff = torch.exp(-(q * (q + eps).log()).sum(-1))                  # (B,)
    z = torch.einsum("bns,bnc->bsc", a, r)                             # anchors
    spread = (z - z.mean(1, keepdim=True)).norm(dim=-1).pow(2).mean(1).sqrt()
    rms_r = (r - r.mean(1, keepdim=True)).norm(dim=-1).pow(2).mean(1).sqrt().clamp_min(1e-12)
    return dict(H_points=h_points.mean().item(), H_slices=h_slices.mean().item(),
                n_eff=n_eff.mean().item(), anchor_spread=(spread / rms_r).mean().item(),
                logit_std=x.std().item(), logit_range=(x.amax() - x.amin()).item())


sys.path.insert(0, str(RECIPE / "src"))
from datasets import build_dataloaders  # noqa: E402

for run in runs:
    row = rows[run]
    extra = [t for t in (row["extra_overrides"] if row["extra_overrides"] != "-" else "").split() if not t.startswith("+training.")]
    overrides = [f"model={row['model']}", f"dataset={row['dataset']}", f"run_id={run}", "infer_split=val",
                 "sampling_resolution=100000", "precision=float32", f"checkpoint_dir={TASK / 'runs'}",
                 f"output_dir={out_dir}", *extra]
    with hydra.initialize_config_dir(config_dir=str(RECIPE / "conf"), version_base=None):
        cfg = hydra.compose(config_name="infer", overrides=overrides)
    _, val_loader, _, _ = build_dataloaders(cfg)
    model = hydra.utils.instantiate(cfg.model, _convert_="partial").cuda()
    epoch = load_checkpoint(path=str(TASK / "runs" / run), models=model, device="cuda")
    model.eval()
    hooks = [blk.register_forward_pre_hook(_pre_hook) for blk in model.blocks]
    per_layer: list[list[dict]] = [[] for _ in model.blocks]
    n_cases = 0
    with torch.no_grad():
        for batch in val_loader:
            REC["a"].clear(); REC["r"].clear()
            fk = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch["forward_kwargs"].items()}
            model(**fk)
            n_layers = len(model.blocks)
            for i in range(n_layers):
                x, a = REC["a"][i]
                per_layer[i].append(layer_stats(x.float(), a.float(), REC["r"][i].float()))
            n_cases += 1
            if n_cases >= K:
                break
    for h in hooks:
        h.remove()
    summary = {"run": run, "epoch": epoch, "n_cases": n_cases, "n_layers": len(model.blocks), "n_slices": model.n_slices,
               "layers": [{k: sum(d[k] for d in L) / len(L) for k in L[0]} for L in per_layer]}
    (out_dir / f"routing_{run}.json").write_text(json.dumps(summary, indent=2))
    print(run, "epoch", epoch, "cases", n_cases)
    for i, L in enumerate(summary["layers"]):
        print(f"  layer {i:2d}: H_points {L['H_points']:.3f}  H_slices {L['H_slices']:.3f}  n_eff {L['n_eff']:6.1f}/{model.n_slices}  anchor_spread {L['anchor_spread']:.3f}  logit_std {L['logit_std']:.2f}")
    del model
    torch.cuda.empty_cache()
