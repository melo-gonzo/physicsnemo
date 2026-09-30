"""TERRITORY probe, REF-BOARD inference-half copy (2026-09-23; original kernel/territory_probe.py, 2026-09-21).
Only change: ``--recipe-dir`` names the recipe directory explicitly (the R1 draws use the $HOME/rp_probes/recipe_rp copy).

TERRITORY probe (2026-09-21): slice territories of trained ISLA and GeoTransolver checkpoints on real
validation cars, under a chosen evaluation draw, with corrected per-block routing statistics.

For each run named on the command line the script composes the recipe's inference config with that run's
manifest overrides (plus an optional dataset override for the bench draws), builds the validation loader,
loads the checkpoint and runs the first K validation cases in float32. Per slice block it records:

  ISLA (hooked at ``_softmax_over_points``; the block's LAST call is the refined routing softmax over points,
  the read-back weights are the softmax over slices of the same refined logits without the measure bias):
    * H_points, H_slices (hardness), n_eff (effective slice count), anchor_spread -- as in the ROUTING-SAT
      probe but indexed per BLOCK (the 2026-09-20 probes indexed the flat call list, so their "layer i" was
      block i//2, pass i%2; see the notebook erratum);
    * dominant slice per token (argmax of the read-back weights), the anchors z_s (S, 3).
  GeoTransolver (hooked at ``PhysicsAttentionBase._compute_slices_from_projections``; one call per layer):
    * hardness and n_eff of the slice softmax (mean over heads), the spread of the weight-averaged slice
      centroids (the analogue of the anchor spread);
    * dominant slice per token and head, the slice centroids (H, S, 3).

With ``--territory <dir>`` it also writes one .npz per (run, case) holding the un-centered token positions
(the bench pipeline stores the subtracted center in ``global_data.center``), the measure weights, and the
dominant-slice arrays, for the territory-agreement reduction and the pictures.

usage: python territory_probe.py --manifest <tsv> --out <dir> --k <K> [--recipe globin|bench]
                                 [--dataset <yaml name>] [--seed <int>] [--tag <str>] [--territory <dir>]
                                 <run_id> [<run_id> ...]
"""
from __future__ import annotations

import argparse, csv, json, math, sys
from pathlib import Path

import hydra, numpy as np, torch

TASK = Path("/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0")
ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--recipe", default="globin", choices=["globin", "bench"])
ap.add_argument("--recipe-dir", default=None, help="absolute recipe directory (overrides --recipe)")
ap.add_argument("--dataset", default=None, help="dataset yaml name overriding the lane's own")
ap.add_argument("--seed", type=int, default=None, help="override the recipe seed (changes the evaluation draw)")
ap.add_argument("--tag", default="", help="suffix for the output files (e.g. the draw name)")
ap.add_argument("--territory", default=None, help="directory for the per-case .npz territory dumps")
ap.add_argument("--cells", type=int, default=100000, help="sampling_resolution for the evaluation draw")
ap.add_argument("runs", nargs="+")
args = ap.parse_args()

RECIPE = Path(args.recipe_dir) if args.recipe_dir else TASK / f"recipe_{args.recipe}"
out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
terr_dir = Path(args.territory) if args.territory else None
if terr_dir: terr_dir.mkdir(parents=True, exist_ok=True)
rows = {r["run_id"]: r for r in csv.DictReader(open(args.manifest), delimiter="\t")}

import physicsnemo.experimental.nn.isla.model as isla_model  # noqa: E402
from physicsnemo.nn.module import physics_attention as pa  # noqa: E402
from physicsnemo.utils import load_checkpoint  # noqa: E402

REC: dict = {"soft": [], "gt": []}
BLK: list = []
_orig_softmax = isla_model._softmax_over_points
_orig_method = pa.PhysicsAttentionBase._compute_slices_from_projections


def _rec_softmax(x, fast=True):
    a = _orig_softmax(x, fast)
    REC["soft"].append((x.detach(), a.detach()))
    return a


def _rec_method(self, slice_projections, fx, measure_weights=None):
    w, tok = _orig_method(self, slice_projections, fx, measure_weights)
    REC["gt"].append(w.detach())
    return w, tok


isla_model._softmax_over_points = _rec_softmax
pa.PhysicsAttentionBase._compute_slices_from_projections = _rec_method


def _pre(mod, a):
    # _SliceBlock.forward(h, log_w, r, n_hat, g_hat, eps)
    BLK.append({"start": len(REC["soft"]), "log_w": a[1].detach(), "r": a[2].detach()})


def _post(mod, a, out):
    BLK[-1]["end"] = len(REC["soft"])


def entropy(p, dim):
    return -(p * (p + 1e-30).log()).sum(dim)


def spread_of(c, r):
    """RMS distance of centroids c (..., S, 3) from their mean over S, divided by the RMS radius of r (N, 3)."""
    rms_r = (r - r.mean(0, keepdim=True)).norm(dim=-1).pow(2).mean().sqrt().clamp_min(1e-12)
    return ((c - c.mean(-2, keepdim=True)).norm(dim=-1).pow(2).mean(-1).sqrt() / rms_r)


def isla_block_stats(x, a, log_w, r):
    """x: refined logits + measure bias (N, S); a: routing softmax over points; r: (N, 3)."""
    n, s = a.shape
    pi = torch.softmax(x - log_w, dim=-1)                       # read-back: over slices, no bias
    q = pi.mean(0)
    z = a.T @ r                                                  # anchors (S, 3)
    return dict(H_points=(entropy(a, 0) / math.log(n)).mean().item(),
                H_slices=(entropy(pi, -1) / math.log(s)).mean().item(),
                n_eff=torch.exp(entropy(q, -1)).item(),
                anchor_spread=spread_of(z, r).item()), pi.argmax(-1).to(torch.int16), z


def gt_layer_stats(w, r):
    """w: slice softmax (N, H, S); r: (N, 3)."""
    n, h, s = w.shape
    q = w.mean(0)                                                # (H, S) slice mass share per head
    c = torch.einsum("nhs,nc->hsc", w, r) / w.sum(0)[..., None].clamp_min(1e-12)   # weight-averaged centroids
    return dict(H_slices=(entropy(w, -1) / math.log(s)).mean().item(),
                n_eff=torch.exp(entropy(q, -1)).mean().item(),
                anchor_spread=spread_of(c, r).mean().item()), w.argmax(-1).to(torch.int16), c


sys.path.insert(0, str(RECIPE / "src"))
from datasets import build_dataloaders  # noqa: E402
from physicsnemo.distributed import DistributedManager  # noqa: E402

DistributedManager.initialize()

for run in args.runs:
    row = rows[run]
    extra = [t for t in (row["extra_overrides"] if row["extra_overrides"] != "-" else "").split() if not t.startswith("+training.")]
    dataset = args.dataset or row["dataset"]
    overrides = [f"model={row['model']}", f"dataset={dataset}", f"run_id={run}", "infer_split=val",
                 f"sampling_resolution={args.cells}", "precision=float32", f"checkpoint_dir={TASK / 'runs'}",
                 f"output_dir={out_dir}", *extra]
    if args.seed is not None:
        overrides.append(f"++training.seed={args.seed}")   # the reader derives its per-index generator from training.seed
    if terr_dir is not None:
        # The batch carries forward_kwargs only; route the subtracted center (bench yamls: store_center_as: center)
        # through the forward-kwargs spec and pop it before the model call.
        overrides.append("+forward_kwargs.center=global_data.center")
    with hydra.initialize_config_dir(config_dir=str(RECIPE / "conf"), version_base=None):
        cfg = hydra.compose(config_name="infer", overrides=overrides)
    _, val_loader, _, _ = build_dataloaders(cfg)
    model = hydra.utils.instantiate(cfg.model, _convert_="partial").cuda()
    epoch = load_checkpoint(path=str(TASK / "runs" / run / "checkpoints"), models=model, device="cuda")
    if epoch == 0:
        raise FileNotFoundError(f"no checkpoint restored for {run}")
    model.eval()
    is_isla = hasattr(model, "blocks") and isinstance(model, isla_model.ISLA)
    hooks = []
    if is_isla:
        for blk in model.blocks:
            hooks += [blk.register_forward_pre_hook(_pre), blk.register_forward_hook(_post)]
    per_layer: list[list[dict]] = []
    n_cases = 0
    tag = f"_{args.tag}" if args.tag else ""
    with torch.no_grad():
        for batch in val_loader:
            REC["soft"].clear(); REC["gt"].clear(); BLK.clear()
            fk = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch["forward_kwargs"].items()}
            center_fk = fk.pop("center", None)
            model(**fk)
            def _find(obj, key, depth=0):
                """Depth-first search of a nested dict / TensorDict / object for a tensor under ``key``."""
                if depth > 6:
                    return None
                try:
                    keys = list(obj.keys())
                except Exception:
                    return None
                for k in keys:
                    try:
                        v = obj[k]
                    except Exception:
                        continue
                    if str(k) == key and torch.is_tensor(v):
                        return v
                    found = _find(v, key, depth + 1)
                    if found is not None:
                        return found
                return None
            center = center_fk if center_fk is not None else _find(batch, "center")
            if terr_dir is not None and center is None:
                raise RuntimeError("center not found in the batch; the territory dumps would not be un-centered")
            if n_cases == 0:
                try:
                    print("batch keys:", list(batch.keys()), "center found:", center is not None, flush=True)
                except Exception:
                    pass
            dom_layers, anchors_layers, stats_layers = [], [], []
            if is_isla:
                r_tok = fk["points"][0].float()
                for i, b in enumerate(BLK):
                    x, a = REC["soft"][b["end"] - 1]      # the block's refined routing softmax over points
                    st, dom, z = isla_block_stats(x[0].float(), a[0].float(), b["log_w"][0].float(), b["r"][0].float())
                    stats_layers.append(st); dom_layers.append(dom.cpu().numpy()); anchors_layers.append(z.cpu().numpy())
                w_tok = fk.get("measure_weights")
                w_tok = w_tok[0].float().cpu().numpy() if w_tok is not None else None
            else:
                r_tok = fk["geometry"][0].float() if "geometry" in fk else fk["local_positions"][0].float()
                for w in REC["gt"]:
                    st, dom, c = gt_layer_stats(w[0].float(), r_tok)
                    stats_layers.append(st); dom_layers.append(dom.cpu().numpy()); anchors_layers.append(c.cpu().numpy())
                w_tok = None
            if not per_layer:
                per_layer = [[] for _ in stats_layers]
            for i, st in enumerate(stats_layers):
                per_layer[i].append(st)
            if terr_dir is not None:
                pos = r_tok.cpu().numpy()
                cen = None
                if center is not None:
                    cen = torch.as_tensor(center).reshape(-1, 3)[0].float().cpu().numpy()
                    pos = pos + cen
                np.savez_compressed(terr_dir / f"{run}{tag}_case{n_cases}.npz", points=pos.astype(np.float32),
                                    center=(cen if cen is not None else np.zeros(3, np.float32)),
                                    measure_weights=(w_tok if w_tok is not None else np.zeros(0, np.float32)),
                                    dominant=np.stack(dom_layers), anchors=np.stack(anchors_layers))
            n_cases += 1
            if n_cases >= args.k:
                break
    for h in hooks:
        h.remove()
    summary = {"run": run, "epoch": epoch, "model": "ISLA" if is_isla else "GeoTransolver", "dataset": dataset,
               "seed_override": args.seed, "n_cases": n_cases, "n_layers": len(per_layer),
               "layers": [{k: sum(d[k] for d in L) / len(L) for k in L[0]} for L in per_layer]}
    (out_dir / f"territory_{run}{tag}.json").write_text(json.dumps(summary, indent=2))
    print(run, tag, "epoch", epoch, "cases", n_cases, "layers", len(per_layer), flush=True)
    for i, L in enumerate(summary["layers"]):
        print(f"  block {i:2d}: " + "  ".join(f"{k} {v:.3f}" for k, v in L.items()), flush=True)
    del model
    torch.cuda.empty_cache()
