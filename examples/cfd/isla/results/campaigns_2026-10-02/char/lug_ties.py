"""Lug float32 sensitivity: are the output changes under a translation carried by nearest-boundary ties? (2026-10-04)

Same case/model as lug_equivariance.py. For the native and the translated (1 RMS along x) DomainMesh: the exact two
nearest boundary centroids per interior node (tie = relative gap < 1e-6), the set S of nodes whose exact nearest index
changes under the translation, the normal angle between old and new nearest cells on S, and the prediction change on
S versus its complement.
"""
import math, sys, torch, hydra
from hydra import compose, initialize_config_dir
sys.path.insert(0, "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/recipe_v8/src")
from datasets import build_dataloaders
from domain_transforms import FixedRandomPose
from output_normalize import normalize_output_to_tensordict, require_output_type
from utils import recursive_to_device
from physicsnemo.distributed import DistributedManager
from physicsnemo.mesh import DomainMesh
from physicsnemo.utils import load_checkpoint
T = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"; RECIPE = T + "/recipe_v8"; RUN = "cl_lug_isla_seed42"
DistributedManager.initialize(); dev = torch.device("cuda")
with initialize_config_dir(config_dir=RECIPE + "/conf", version_base=None):
    cfg = compose(config_name="infer", overrides=["dataset=ginot_lug_volume", "model=isla_lug", "infer_split=val", "train_split=train", "val_split=test", "+training.val_interval=10", "model.hidden=768", "sampling_resolution=200000", "dataloader.num_workers=1", f"run_id={RUN}"])
_, vl, _, info = build_dataloaders(cfg); target_config = info["targets"]; output_type = require_output_type(cfg)
model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(dev); load_checkpoint(path=f"{T}/runs/{RUN}/checkpoints", models=model, device=dev); model.eval()
collate = vl.collate_fn
def predict(dm):
    batch = recursive_to_device(collate([(dm, {})]), dev)
    with torch.no_grad(): out = model(**batch["forward_kwargs"])
    return normalize_output_to_tensordict(out, target_config, output_type)["von_mises"].detach().float().reshape(-1)
idx = next(iter(vl.sampler)); dm, _ = vl.dataset[idx]
tf = FixedRandomPose(); pts = dm.interior.points
rms = (pts.double() - pts.double().mean(0)).norm(dim=-1).square().mean().sqrt()
tx = torch.tensor([float(rms), 0.0, 0.0], dtype=pts.dtype, device=pts.device)
dmt = DomainMesh(interior=tf._transform_mesh(dm.interior, torch.eye(3, dtype=pts.dtype, device=pts.device), tx), boundaries={k: tf._transform_mesh(b, torch.eye(3, dtype=pts.dtype, device=pts.device), tx) for k, b in dm.boundaries.items()}, global_data=dm.global_data)
def two_nearest(q, c):
    d = torch.cat([torch.cdist(q[i:i+4096].double(), c.double()) for i in range(0, q.shape[0], 4096)])  # exact in float64 on the float32 coordinates
    v, i = d.topk(2, dim=1, largest=False); return v, i
b0, b1 = dm.boundaries["surface"], dmt.boundaries["surface"]
q0, c0, n0 = dm.interior.points.cuda(), b0.cell_centroids.cuda(), b0.cell_data["normals"].cuda().float()
q1, c1 = dmt.interior.points.cuda(), b1.cell_centroids.cuda()
v0, i0 = two_nearest(q0, c0); v1, i1 = two_nearest(q1, c1)
gap0 = (v0[:, 1] - v0[:, 0]) / v0[:, 0].clamp_min(1e-12)
print(f"nodes {q0.shape[0]}, boundary cells {c0.shape[0]}; coordinate scale |x|~{float(q0.abs().max()):.3f}, |t|={float(rms):.3f}")
for thr in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
    print(f"  exact two-nearest tie (rel gap < {thr:.0e}): {int((gap0 < thr).sum())} nodes ({100*float((gap0 < thr).float().mean()):.2f}%)")
S = i0[:, 0] != i1[:, 0]
print(f"nearest index changes under the translation (exact search on float32 coords): {int(S.sum())} nodes ({100*float(S.float().mean()):.3f}%)")
if S.any():
    nn0, nn1 = n0[i0[S, 0]], n0[i1[S, 0]]
    cosang = (nn0 * nn1).sum(-1) / (nn0.norm(dim=-1) * nn1.norm(dim=-1)).clamp_min(1e-12)
    ang = torch.rad2deg(torch.acos(cosang.clamp(-1, 1)))
    print(f"  normal angle between old and new nearest cell on S: median {float(ang.median()):.1f} deg, >30 deg: {int((ang > 30).sum())}, gap of those nodes: median {float(gap0[S].median()):.1e}")
p0, p1 = predict(dm), predict(dmt)
d = (p1 - p0).abs()
print(f"prediction change |dp|: total rel {float(d.norm()/p0.norm()):.4f}; share of sum|dp|^2 on S: {100*float((d[S]**2).sum()/(d**2).sum()):.1f}% (S is {100*float(S.float().mean()):.3f}% of nodes); median |dp| on S {float(d[S].median()) if S.any() else float('nan'):.3e} vs off S {float(d[~S].median()):.3e}")
