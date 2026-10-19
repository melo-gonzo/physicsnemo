"""ROUTING (FIG lane, 2026-10-18): the trained routing of an A2 reference ISLA checkpoint on one validation case, in the
schema of results/probes/slices_2026-09-22/slices_probe.py, ported to the current ISLA class (code_a2r / code_a2s:
physicsnemo.experimental.models.isla, Mesh inputs).

The recipe's inference config is composed exactly as the scorer composes it (model yaml, dataset yaml, the lane's extra
overrides, cells, float32), with model.compile_blocks=false so the blocks run as plain modules (inference already has
compile=false); the final checkpoint is restored with physicsnemo.utils.load_checkpoint; ONE validation sample (sampler
index --sample) runs once. Two read-only wrappers:
  * blocks._softmax_over_points records its input (logits + log_w) and output (a_is, normalized over tokens). Each
    _SliceBlock calls it twice; the SECOND call is the refined routing (provisional routing builds only the anchors).
  * geometry.rms_pairwise_distance (as imported by the model module) records the length unit L, so token positions are
    r * L in the model's input coordinates (the dataset's, after the pipeline's CenterMesh where it has one).
Forward pre-hooks on every block record (log_w, r, n_hat) and delimit the block's softmax calls.

Per layer: the soft assignment pi_is = softmax over slices of the refined logits (the recorded input minus log_w), the
dominant slice argmax_s pi_is, its weight max_s pi_is, slice masses q = mean_i pi_is and qw (measure-weighted), anchors
z_s = sum_i a_is r_i (times L) and anchor normals m_s; for three layers the 8 slices with the largest argmax territories:
their pi and a_is / w_i on the stored tokens. Arrays as the old npz (points, n_boundary, measure, keep, dominant, pi_max,
q, qw, anchors, anchor_normals, top_layers, top_slices, top_pi, top_a_density, gauge) plus sample_id; json with per-layer
n_eff, H_slices, H_points, anchor_spread, pi_max_mean.
usage: python slices_probe_a2.py --recipe-dir R --checkpoint-dir C --run RUN --model M --dataset DS --cells N
       --sample IDX --out DIR [--extra "k=v ..."] [--keep 60000] [--top 8] [--layers 0,5,11]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
for k in ("--recipe-dir", "--checkpoint-dir", "--run", "--model", "--dataset", "--out"):
    ap.add_argument(k, required=True)
ap.add_argument("--cells", type=int, required=True)
ap.add_argument("--sample", type=int, required=True)
ap.add_argument("--extra", default="")
ap.add_argument("--keep", type=int, default=60000)
ap.add_argument("--top", type=int, default=8)
ap.add_argument("--layers", default="0,5,11")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(args.recipe_dir) / "src"))

import hydra  # noqa: E402

import physicsnemo.experimental.models.isla.blocks as isla_blocks  # noqa: E402
import physicsnemo.experimental.models.isla.model as isla_model  # noqa: E402
from datasets import build_dataloaders  # noqa: E402
from infer import _sample_id  # noqa: E402
from physicsnemo import datapipes  # noqa: E402,F401
from physicsnemo.distributed import DistributedManager  # noqa: E402
from physicsnemo.utils import load_checkpoint  # noqa: E402
from utils import recursive_to_device  # noqa: E402

REC, BLK, GAUGE = [], [], []
_orig_softmax = isla_blocks._softmax_over_points
_orig_rms = isla_model.rms_pairwise_distance


def _rec_softmax(x):
    a = _orig_softmax(x)
    REC.append((x.detach(), a.detach()))
    return a


def _rec_rms(points, measure, eps):
    g = _orig_rms(points, measure, eps)
    GAUGE.append((g.detach(), int(points.shape[1])))
    return g


isla_blocks._softmax_over_points = _rec_softmax
isla_model.rms_pairwise_distance = _rec_rms


def _pre(mod, a):
    BLK.append({"start": len(REC), "log_w": a[1].detach(), "r": a[2].detach(), "n_hat": a[3].detach()})


def _post(mod, a, o):
    BLK[-1]["end"] = len(REC)


def entropy(p, dim):
    return -(p * (p + 1e-30).log()).sum(dim)


DistributedManager.initialize()
extra = [t for t in args.extra.split() if t and not t.startswith("training.num_epochs")]
overrides = [f"model={args.model}", f"dataset={args.dataset}", f"run_id={args.run}", "infer_split=val",
             f"sampling_resolution={args.cells}", "precision=float32", f"checkpoint_dir={args.checkpoint_dir}",
             f"output_dir={out}/_scratch", *extra, "model.compile_blocks=false"]
with hydra.initialize_config_dir(config_dir=str(Path(args.recipe_dir) / "conf"), version_base=None):
    cfg = hydra.compose(config_name="infer", overrides=overrides)
print("overrides", overrides, flush=True)
_, val_loader, _, _ = build_dataloaders(cfg)
ds = val_loader.dataset
assert args.sample in list(val_loader.sampler), args.sample
dom, meta = ds[args.sample]
sid = _sample_id(meta, args.sample)
batch = recursive_to_device(val_loader.collate_fn([(dom, meta)]), "cuda")

model = hydra.utils.instantiate(cfg.model, _convert_="partial").cuda()
ckdir = str(Path(args.checkpoint_dir) / args.run / "checkpoints")
epoch = load_checkpoint(path=ckdir, models=model, device="cuda")
assert epoch > 0, f"no checkpoint restored from {ckdir}"
model.eval()
hooks = []
for blk in model.blocks:
    hooks += [blk.register_forward_pre_hook(_pre), blk.register_forward_hook(_post)]
with torch.no_grad():
    model(**batch["forward_kwargs"])
for h in hooks:
    h.remove()
assert len(GAUGE) == 1, len(GAUGE)
gauge = float(GAUGE[0][0].reshape(-1)[0])
n_boundary = GAUGE[0][1]
L = len(BLK)
assert all(b["end"] - b["start"] == 2 for b in BLK), [(b["start"], b["end"]) for b in BLK]
log_w = BLK[0]["log_w"][0, :, 0].float()
w = log_w.exp()
r0 = BLK[0]["r"][0].float()
pts = r0 * gauge
n_tok = pts.shape[0]
keep = torch.arange(n_tok, device="cuda")
if n_tok > args.keep:
    g = torch.Generator(device="cpu").manual_seed(0)
    keep = torch.randperm(n_tok, generator=g)[: args.keep].sort().values.cuda()
layers_keep = [int(s) for s in args.layers.split(",")]
active = torch.isfinite(log_w)  # passive (query-independent) tokens carry log_w = -inf
dominant, pi_max, q, qw, anchors, anchor_n, stats = [], [], [], [], [], [], []
top_slices, top_pi, top_ad = [], [], []
for li, b in enumerate(BLK):
    x, a = REC[b["end"] - 1]
    x = x[0].float()
    a = a[0].float()
    lw = torch.where(active, log_w, torch.zeros_like(log_w))
    pi = torch.softmax(x - lw[:, None], dim=-1)
    qq = pi[active].mean(0)
    qwq = (w[active, None] * pi[active]).sum(0) / w[active].sum()
    r = b["r"][0].float()
    z = a.T @ r
    m = a.T @ b["n_hat"][0].float()
    m = m / m.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    rms_r = (r[active] - r[active].mean(0)).norm(dim=-1).pow(2).mean().sqrt()
    stats.append(dict(n_eff=torch.exp(entropy(qq, -1)).item(), n_eff_w=torch.exp(entropy(qwq, -1)).item(),
                      H_slices=(entropy(pi[active], -1) / math.log(pi.shape[1])).mean().item(),
                      H_points=(entropy(a[active], 0) / math.log(int(active.sum()))).mean().item(),
                      anchor_spread=((z - z.mean(0)).norm(dim=-1).pow(2).mean().sqrt() / rms_r).item(),
                      pi_max_mean=pi[active].max(-1).values.mean().item()))
    dominant.append(pi[keep].argmax(-1).to(torch.int16).cpu().numpy())
    pi_max.append(pi[keep].max(-1).values.half().cpu().numpy())
    q.append(qq.cpu().numpy())
    qw.append(qwq.cpu().numpy())
    anchors.append((z * gauge).cpu().numpy())
    anchor_n.append(m.cpu().numpy())
    if li in layers_keep:
        counts = torch.bincount(pi[active].argmax(-1), minlength=pi.shape[1])
        ts = torch.argsort(counts, descending=True)[: args.top]
        top_slices.append(ts.cpu().numpy())
        top_pi.append(pi[keep][:, ts].half().cpu().numpy())
        top_ad.append((a[keep][:, ts] / w[keep, None].clamp_min(1e-30)).half().cpu().numpy())
arrays = dict(points=pts.cpu().numpy().astype(np.float32), n_boundary=np.array(n_boundary),
              measure=w.cpu().numpy().astype(np.float32), keep=keep.cpu().numpy().astype(np.int64),
              dominant=np.stack(dominant), pi_max=np.stack(pi_max), q=np.stack(q), qw=np.stack(qw),
              anchors=np.stack(anchors), anchor_normals=np.stack(anchor_n), top_layers=np.array(layers_keep),
              top_slices=np.stack(top_slices), top_pi=np.stack(top_pi), top_a_density=np.stack(top_ad),
              gauge=np.array(gauge), sample_id=np.array(sid))
tag = f"{args.run}_s{args.sample:05d}"
np.savez_compressed(out / f"{tag}.npz", **arrays)
meta_out = dict(run=args.run, epoch=int(epoch), model_yaml=args.model, dataset_yaml=args.dataset, overrides=overrides,
                sampling_resolution=args.cells, precision="float32", sample_index=args.sample, sample_id=sid,
                geo_kernel=str(getattr(model.blocks[0], "geo_kernel", "")), n_tokens=int(n_tok),
                n_boundary=int(n_boundary), n_active=int(active.sum()), n_stored=int(keep.numel()), n_layers=int(L),
                n_slices=int(REC[0][1].shape[-1]), gauge=gauge, layers=stats, top_layers=layers_keep,
                top_slices=[t.tolist() for t in top_slices])
(out / f"{tag}.json").write_text(json.dumps(meta_out, indent=2))
print(tag, "epoch", epoch, "tokens", n_tok, "boundary", n_boundary, "gauge", gauge, "sample", sid, flush=True)
for i, st in enumerate(stats):
    print(f"  layer {i:2d}: " + "  ".join(f"{kk} {v:.3f}" for kk, v in st.items()), flush=True)
print("DONE", flush=True)
