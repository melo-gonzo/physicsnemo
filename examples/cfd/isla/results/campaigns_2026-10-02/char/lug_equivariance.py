"""Lug posed anomaly (2026-10-04): model-level rigid-motion check of cl_lug_isla_seed42 on one validation case.

Native pipeline (ginot_lug_volume, isla_lug, hidden 768, test split). Transforms applied to the post-pipeline DomainMesh
with FixedRandomPose's own machinery (shape-masked rotation of every spatial field; translation):
  full 90-deg rotation about z (exact in float32), full translation by one RMS radius along x, their combination, a
  37-deg rotation (float-inexact), and a bisection: interior points only / stored sdf gradient only / boundary only /
  globals only. von Mises is a scalar: predictions are compared directly (relative L2 of the difference).
Env EXACT_NN=1 forces the exact nearest-boundary search. Usage: python lug_equivariance.py [run_id]
"""
import math, os, sys, torch, hydra
from hydra import compose, initialize_config_dir
sys.path.insert(0, "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/recipe_v8/src")
from datasets import build_dataloaders
from domain_transforms import FixedRandomPose
from output_normalize import normalize_output_to_tensordict, require_output_type
from utils import recursive_to_device
from physicsnemo.distributed import DistributedManager
from physicsnemo.mesh import DomainMesh
from physicsnemo.utils import load_checkpoint
T = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"
RECIPE = T + "/recipe_v8"
RUN = sys.argv[1] if len(sys.argv) > 1 else "cl_lug_isla_seed42"
DistributedManager.initialize()
dev = torch.device("cuda")
if os.environ.get("EXACT_NN"):
    import functools
    import physicsnemo.experimental.models.isla.geometry as geometry
    from physicsnemo.nn.functional.neighbors.knn.knn import knn
    geometry.knn = functools.partial(knn, implementation="torch"); print("nearest-boundary search: EXACT")
else:
    print("nearest-boundary search: default (cuML)")
with initialize_config_dir(config_dir=RECIPE + "/conf", version_base=None):
    cfg = compose(config_name="infer", overrides=["dataset=ginot_lug_volume", "model=isla_lug", "infer_split=val", "train_split=train", "val_split=test",
                                                  "+training.val_interval=10", "model.hidden=768", "sampling_resolution=200000", "dataloader.num_workers=1", f"run_id={RUN}"])
_, vl, _, info = build_dataloaders(cfg)
target_config = info["targets"]; output_type = require_output_type(cfg)
FP64 = bool(os.environ.get("FP64")); NOCOMPILE = bool(os.environ.get("NOCOMPILE"))
if FP64 or NOCOMPILE:
    cfg.model.compile_blocks = False; print("compile_blocks off")
model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(dev)
print("checkpoint epoch", load_checkpoint(path=f"{T}/runs/{RUN}/checkpoints", models=model, device=dev)); model.eval()
if FP64:
    model = model.double(); print("model and data in FLOAT64 (compile_blocks off)")
collate = vl.collate_fn
def _to64(obj):
    if torch.is_tensor(obj):
        return obj.double() if obj.is_floating_point() else obj
    if hasattr(obj, "apply") and hasattr(obj, "batch_size"):   # TensorDict
        return obj.apply(lambda t: t.double() if t.is_floating_point() else t)
    if hasattr(obj, "points") and hasattr(obj, "with_data"):   # Mesh
        return type(obj)(points=obj.points.double(), cells=obj.cells, point_data=_to64(obj.point_data), cell_data=_to64(obj.cell_data), global_data=_to64(obj.global_data))
    if isinstance(obj, dict):
        return {k: _to64(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to64(v) for v in obj)
    return obj
def predict(dm):
    batch = recursive_to_device(collate([(dm, {})]), dev)
    if FP64:
        batch = _to64(batch)
    with torch.no_grad():
        out = model(**batch["forward_kwargs"])
    td = normalize_output_to_tensordict(out, target_config, output_type)
    return td["von_mises"].detach().float().cpu().reshape(-1), batch["targets"]["von_mises"].float().cpu().reshape(-1)
def rel(a, b): return float((a - b).norm() / b.norm().clamp_min(1e-12))
idx = next(iter(vl.sampler)); dm, meta = vl.dataset[idx]
if FP64:
    ### cast BEFORE transforming, so the rigid motions below are exact to float64 roundoff
    dm = DomainMesh(interior=_to64(dm.interior), boundaries={k: _to64(b) for k, b in dm.boundaries.items()}, global_data=_to64(dm.global_data))
    print("DomainMesh cast to float64 before the transforms")
print("case", idx, meta.get("source_path", "?"), "| interior fields:", sorted(dm.interior.point_data.keys()), "| boundary fields:", sorted(dm.boundaries["surface"].cell_data.keys()), "| globals:", sorted(dm.global_data.keys()))
p0, y0 = predict(dm); print(f"native: rel err vs target = {rel(p0, y0):.4f}")
tf = FixedRandomPose()
pts = dm.interior.points
def Rz(theta):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=pts.dtype, device=pts.device)
rms = (pts.double() - pts.double().mean(0)).norm(dim=-1).square().mean().sqrt()
tx = torch.tensor([float(rms), 0.0, 0.0], dtype=pts.dtype, device=pts.device)
I3 = torch.eye(3, dtype=pts.dtype, device=pts.device)
def variant(R_int=None, t_int=None, R_bnd=None, t_bnd=None, rotate_grad=True, rotate_points=True):
    """Build a DomainMesh with the chosen parts transformed. Interior: points and/or stored sdf gradient."""
    interior = dm.interior
    if R_int is not None or t_int is not None:
        R = R_int if R_int is not None else I3
        full = tf._transform_mesh(interior, R, t_int)
        if rotate_grad and rotate_points:
            interior = full
        elif rotate_points:   # points move, stored gradient stays in the lab frame
            interior = full.with_data(point_data=interior.point_data.clone())
        else:                 # only the stored gradient rotates
            interior = interior.with_data(point_data=full.point_data.clone())
    bnds = dict(dm.boundaries)
    if R_bnd is not None or t_bnd is not None:
        R = R_bnd if R_bnd is not None else I3
        bnds = {k: tf._transform_mesh(b, R, t_bnd) for k, b in dm.boundaries.items()}
    return DomainMesh(interior=interior, boundaries=bnds, global_data=dm.global_data)
R90, R37 = Rz(math.pi / 2), Rz(37 * math.pi / 180)
cases = [
    ("full rotation 90 deg about z", dict(R_int=R90, R_bnd=R90)),
    ("full translation 1 RMS along x", dict(t_int=tx, t_bnd=tx)),
    ("full rotation 90 deg + translation", dict(R_int=R90, t_int=tx, R_bnd=R90, t_bnd=tx)),
    ("full rotation 37 deg about z", dict(R_int=R37, R_bnd=R37)),
    ("BISECT interior only (points + sdf gradient) 90 deg", dict(R_int=R90)),
    ("BISECT interior points only (gradient left in lab frame) 90 deg", dict(R_int=R90, rotate_grad=False)),
    ("BISECT stored sdf gradient only 90 deg", dict(R_int=R90, rotate_points=False)),
    ("BISECT boundary only 90 deg", dict(R_bnd=R90)),
    ("BISECT interior translated only", dict(t_int=tx)),
    ("BISECT boundary translated only", dict(t_bnd=tx)),
]
for label, kw in cases:
    d = variant(**kw)
    p1, y1 = predict(d)
    print(f"## {label}: rel diff pred vs native pred = {rel(p1, p0):.4f}   (rel err vs target {rel(p1, y1):.4f}; target unchanged {rel(y1, y0):.1e})")
### No rigid motion at all: a random jitter of every coordinate (interior + boundary) at the stated relative scale.
### If this moves the output as much as the translation did, the sensitivity is the tie-breaking of the nearest
### boundary sample (76% of lug nodes are exactly equidistant from two boundary centroids), not the motion.
scale = float(pts.abs().max())
for eps in (1e-7, 1e-6, 1e-5):
    g = torch.Generator(device="cpu").manual_seed(11)
    def jit(m):
        noise = (torch.randn(m.points.shape, generator=g, dtype=torch.float64) * eps * scale).to(device=m.points.device, dtype=m.points.dtype)
        return m.with_data() if False else type(m)(points=m.points + noise, cells=m.cells, point_data=m.point_data, cell_data=m.cell_data, global_data=m.global_data)
    d = DomainMesh(interior=jit(dm.interior), boundaries={k: jit(b) for k, b in dm.boundaries.items()}, global_data=dm.global_data)
    p1, y1 = predict(d)
    print(f"## JITTER {eps:.0e} x |x|max (no rigid motion): rel diff pred vs native pred = {rel(p1, p0):.4f}   (rel err vs target {rel(p1, y1):.4f})")
### globals: the lug has none spatial (4 standardized scalars) -- confirm by listing shapes
print("global leaves:", {k: tuple(v.shape) for k, v in dm.global_data.items()})
