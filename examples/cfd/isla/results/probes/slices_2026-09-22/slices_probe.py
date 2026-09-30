"""SLICES probe (2026-09-22): what the trained routing looks like on one validation case.

For one completed ISLA run this script composes the recipe's inference config exactly as the lane did (model yaml,
dataset yaml, splits, cells), builds the validation loader, restores the final checkpoint with the recipe's own
loader (physicsnemo.utils.load_checkpoint) and runs ONE validation case in float32 with the geometry kernel the run
trained with. The routing softmax over points (`_softmax_over_points` in physicsnemo.experimental.nn.isla.model) is
wrapped to record its input (the refined logits plus the measure bias, lambda'_is + l_i) and output (the routing
weights a_is); forward pre/post hooks on every `_SliceBlock` delimit the calls that belong to each block, and the
block's LAST call is its refined routing (the first is the provisional routing that only builds the anchors; the
2026-09-20 ROUTING-SAT probes indexed the flat call list and so mixed the two passes, see the notebook erratum).

From the refined logits the read-back weights pi_is = softmax over slices of (lambda'_is), i.e. the recorded input
minus the measure bias, are the per-token soft assignment; the routing weights a_is (sum to one over tokens for each
slice) are what the slice states average. Written per run and case to <out>/<run>_case<k>.npz (+ .json):

  points        (N, 3)  the model's input positions (boundary tokens first, then query tokens), dataset coordinates
  n_boundary    ()      number of boundary tokens
  measure       (N,)    the tokens' measure weights (query tokens: the learned relative weight, exp of their log-weight)
  keep          (Ns,)   indices of the stored token subset (all tokens when N <= 60k)
  dominant      (L, Ns) int16, argmax_s pi_is per layer
  pi_max        (L, Ns) float16, max_s pi_is per layer (how decided the token is)
  q             (L, S)  slice mass, mean over all tokens of pi_is (the ROUTING-SAT definition)
  qw            (L, S)  measure-weighted slice mass, sum_i w_i pi_is / sum_i w_i
  anchors       (L, S, 3) z_s = sum_i a_is r_i, rescaled by the length unit into the input coordinates
  anchor_normals(L, S, 3) m_s
  top_layers    (3,)    the layers whose individual slice maps are stored
  top_slices    (3, T)  ids of the T slices that dominate the most tokens (largest argmax territories) in those layers
  top_pi        (3, Ns, T) float16, pi_is for those slices
  top_a_density (3, Ns, T) float16, a_is / w_i (the routing weight per unit measure) for those slices
  stats: per layer n_eff = exp(entropy of q), H_slices, H_points, anchor_spread (as in the TERRITORY probe)

usage: python slices_probe.py --run <run_id> --model <yaml> --dataset <yaml> --train-split <s> --val-split <s>
                              --cells <n> --out <dir> [--case k] [--keep 60000] [--top 8] [--layers 0,5,11] [--extra "a=b c=d"]
"""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path

import hydra, numpy as np, torch

TASK = Path("/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0")
RECIPE = TASK / "recipe_globin"
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--model", required=True); ap.add_argument("--dataset", required=True)
ap.add_argument("--train-split", required=True); ap.add_argument("--val-split", required=True)
ap.add_argument("--cells", type=int, required=True); ap.add_argument("--out", required=True)
ap.add_argument("--case", type=int, default=0); ap.add_argument("--keep", type=int, default=60000)
ap.add_argument("--top", type=int, default=8); ap.add_argument("--layers", default="0,5,11"); ap.add_argument("--extra", default="")
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

import physicsnemo.experimental.nn.isla.model as isla_model  # noqa: E402
from physicsnemo.utils import load_checkpoint  # noqa: E402

REC: list = []
BLK: list = []
_orig_softmax = isla_model._softmax_over_points


def _rec_softmax(x, fast=True):
    a = _orig_softmax(x, fast)
    REC.append((x.detach(), a.detach()))
    return a


isla_model._softmax_over_points = _rec_softmax


def _pre(mod, a):
    # _SliceBlock.forward(h, log_w, r, n_hat, g_hat, eps)
    BLK.append({"start": len(REC), "log_w": a[1].detach(), "r": a[2].detach(), "n_hat": a[3].detach()})


def _post(mod, a, o):
    BLK[-1]["end"] = len(REC)


def entropy(p, dim):
    return -(p * (p + 1e-30).log()).sum(dim)


sys.path.insert(0, str(RECIPE / "src"))
from datasets import build_dataloaders  # noqa: E402
from physicsnemo.distributed import DistributedManager  # noqa: E402

DistributedManager.initialize()
overrides = [f"model={args.model}", f"dataset={args.dataset}", f"run_id={args.run}", f"train_split={args.train_split}",
             f"infer_split={args.val_split}", f"sampling_resolution={args.cells}", "precision=float32",
             f"checkpoint_dir={TASK / 'runs'}", f"output_dir={out}", *args.extra.split()]
with hydra.initialize_config_dir(config_dir=str(RECIPE / "conf"), version_base=None):
    cfg = hydra.compose(config_name="infer", overrides=overrides)
_, val_loader, _, _ = build_dataloaders(cfg)
# try to learn the case ids the loader iterates (for the record only)
case_ids = None; case_attr = None
ds = val_loader.dataset
try:
    n_ds = len(ds)
except Exception:
    n_ds = -1
for attr in dir(ds):
    if attr.startswith("__"):
        continue
    try:
        v = getattr(ds, attr)
    except Exception:
        continue
    if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], (str, Path)) and (n_ds < 0 or len(v) == n_ds):
        case_ids = [str(x) for x in v]; case_attr = attr
        break
print("dataset len", n_ds, "case ids via", case_attr, (case_ids[:3] if case_ids else None), flush=True)

model = hydra.utils.instantiate(cfg.model, _convert_="partial").cuda()
epoch = load_checkpoint(path=str(TASK / "runs" / args.run / "checkpoints"), models=model, device="cuda")
if epoch == 0:
    raise FileNotFoundError(f"no checkpoint restored for {args.run}")
model.eval()
hooks = []
for blk in model.blocks:
    hooks += [blk.register_forward_pre_hook(_pre), blk.register_forward_hook(_post)]

layers_keep = [int(s) for s in args.layers.split(",")]
with torch.no_grad():
    for k, batch in enumerate(val_loader):
        if k < args.case:
            continue
        REC.clear(); BLK.clear()
        fk = {kk: (v.cuda() if torch.is_tensor(v) else v) for kk, v in batch["forward_kwargs"].items()}
        model(**fk)
        pts = fk["points"][0].float()
        n_boundary = pts.shape[0]
        if fk.get("query_points") is not None and getattr(model, "query_tokens", False):
            pts = torch.cat([pts, fk["query_points"][0].float()], dim=0)
        n_tok = pts.shape[0]
        r0 = BLK[0]["r"][0].float()
        gauge = (pts.norm(dim=-1).sum() / r0.norm(dim=-1).sum().clamp_min(1e-12)).item()   # the length unit L (r = points / L)
        log_w = BLK[0]["log_w"][0, :, 0].float()
        w = log_w.exp()
        L = len(BLK); S = model.blocks[0].assign.n_slices if hasattr(model.blocks[0].assign, "n_slices") else REC[0][1].shape[-1]
        keep = torch.arange(n_tok, device="cuda")
        if n_tok > args.keep:
            g = torch.Generator(device="cpu").manual_seed(0)
            keep = torch.randperm(n_tok, generator=g)[: args.keep].sort().values.cuda()
        dominant, pi_max, q, qw, anchors, anchor_n, stats = [], [], [], [], [], [], []
        top_slices, top_pi, top_ad = [], [], []
        for li, b in enumerate(BLK):
            x, a = REC[b["end"] - 1]
            x = x[0].float(); a = a[0].float()
            pi = torch.softmax(x - log_w[:, None], dim=-1)          # read-back / soft assignment: over slices, no bias
            qq = pi.mean(0); qwq = (w[:, None] * pi).sum(0) / w.sum()
            z = a.T @ b["r"][0].float(); m = a.T @ b["n_hat"][0].float(); m = m / m.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            rms_r = (b["r"][0].float() - b["r"][0].float().mean(0)).norm(dim=-1).pow(2).mean().sqrt()
            stats.append(dict(n_eff=torch.exp(entropy(qq, -1)).item(), n_eff_w=torch.exp(entropy(qwq, -1)).item(),
                              H_slices=(entropy(pi, -1) / math.log(pi.shape[1])).mean().item(),
                              H_points=(entropy(a, 0) / math.log(a.shape[0])).mean().item(),
                              anchor_spread=((z - z.mean(0)).norm(dim=-1).pow(2).mean().sqrt() / rms_r).item(),
                              pi_max_mean=pi.max(-1).values.mean().item()))
            dominant.append(pi[keep].argmax(-1).to(torch.int16).cpu().numpy()); pi_max.append(pi[keep].max(-1).values.half().cpu().numpy())
            q.append(qq.cpu().numpy()); qw.append(qwq.cpu().numpy()); anchors.append((z * gauge).cpu().numpy()); anchor_n.append(m.cpu().numpy())
            if li in layers_keep:
                # the slices that dominate the most tokens (largest territories), the pictures' subjects
                counts = torch.bincount(pi.argmax(-1), minlength=pi.shape[1])
                ts = torch.argsort(counts, descending=True)[: args.top]
                top_slices.append(ts.cpu().numpy()); top_pi.append(pi[keep][:, ts].half().cpu().numpy())
                top_ad.append((a[keep][:, ts] / w[keep, None]).half().cpu().numpy())
        arrays = dict(points=pts.cpu().numpy().astype(np.float32), n_boundary=np.array(n_boundary), measure=w.cpu().numpy().astype(np.float32),
                      keep=keep.cpu().numpy().astype(np.int64), dominant=np.stack(dominant), pi_max=np.stack(pi_max), q=np.stack(q), qw=np.stack(qw),
                      anchors=np.stack(anchors), anchor_normals=np.stack(anchor_n), top_layers=np.array(layers_keep), top_slices=np.stack(top_slices),
                      top_pi=np.stack(top_pi), top_a_density=np.stack(top_ad), gauge=np.array(gauge))
        # also carry the global vectors, for drawing the far-field / frame arrows on the same picture
        if fk.get("global_vectors") is not None:
            arrays["global_vectors"] = fk["global_vectors"].reshape(-1, 3).float().cpu().numpy()
        tag = f"{args.run}_case{k}"
        np.savez_compressed(out / f"{tag}.npz", **arrays)
        meta = dict(run=args.run, epoch=int(epoch), model_yaml=args.model, dataset_yaml=args.dataset, train_split=args.train_split,
                    val_split=args.val_split, sampling_resolution=args.cells, precision="float32", geo_kernel=str(getattr(model, "geo_kernel", "")),
                    case_index=k, case_id=(case_ids[k] if case_ids and k < len(case_ids) else None), n_tokens=int(n_tok), n_boundary=int(n_boundary),
                    n_stored=int(keep.numel()), n_layers=int(L), n_slices=int(REC[0][1].shape[-1]), gauge=gauge, layers=stats,
                    top_layers=layers_keep, top_slices=[t.tolist() for t in top_slices])
        (out / f"{tag}.json").write_text(json.dumps(meta, indent=2))
        print(tag, "epoch", epoch, "tokens", n_tok, "boundary", n_boundary, "gauge", gauge, "case_id", meta["case_id"], flush=True)
        for i, st in enumerate(stats):
            print(f"  layer {i:2d}: " + "  ".join(f"{kk} {v:.3f}" for kk, v in st.items()), flush=True)
        break
for h in hooks:
    h.remove()
print("DONE", flush=True)
