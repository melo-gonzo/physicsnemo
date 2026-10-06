"""Tensile2D posed anomaly (2026-10-03): model-level equivariance check and pipeline field check.

Part A (pipeline): for one val case, compare the `plaid_tensile2d_2d_posed` sample with the native sample rotated/translated
by FixedRandomPose's own matrix, field by field (points, boundary points, normals, sdf_normals, displacement, sig as a
tensor, BC labels, params); same for the 3-D baseline dataset `plaid_tensile2d_posed` (targets rotated?).
Part B (model): load cl_t2d_isla_seed42 (hidden 768) as infer.py does on the native 2-D pipeline, predict on one val
DomainMesh, rotate THAT DomainMesh in-plane by 90 degrees (points, boundary, normals, sdf_normals, displacement, sig tensor),
predict again, rotate the predictions back (vector / tensor), and report per-field relative differences; also the same with
a pure translation and with the native pipeline's actual posed sample.
"""
import math, sys, torch, hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
sys.path.insert(0, "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/recipe_v8/src")
from datasets import build_dataloaders
from domain_transforms import FixedRandomPose, _pose_rotation_for_key
from output_normalize import normalize_output_to_tensordict
from utils import recursive_to_device
from output_normalize import require_output_type
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint
T = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"
RECIPE = T + "/recipe_v8"
DistributedManager.initialize()
dev = torch.device("cuda")
import os
if os.environ.get("EXACT_NN"):
    ### force ISLA's nearest-boundary search to the exact difference-form torch backend (see infer_exact_nn.py)
    import functools
    import physicsnemo.experimental.models.isla.geometry as geometry
    from physicsnemo.nn.functional.neighbors.knn.knn import knn
    geometry.knn = functools.partial(knn, implementation="torch")
    print("nearest-boundary search: EXACT (torch difference form)")
else:
    print("nearest-boundary search: default (cuML)")
RUN = sys.argv[1] if len(sys.argv) > 1 else "cl_t2d_isla_seed42"
X = ["train_split=train_400", "val_split=val_100", "+training.val_interval=10", "sampling_resolution=200000", "dataloader.num_workers=1", f"run_id={RUN}"]

def build(ds, model, extra=()):
    with initialize_config_dir(config_dir=RECIPE + "/conf", version_base=None):
        cfg = compose(config_name="infer", overrides=[f"dataset={ds}", f"model={model}", "infer_split=val", *X, *extra])
    _, vl, _, info = build_dataloaders(cfg)
    return cfg, vl, info

def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-12))

def rot2(theta):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float64)

def rotate_domain(dm, R2, t=None):
    """In-plane rotation (+translation) of a 2-D DomainMesh with FixedRandomPose's machinery: shape-masked transform,
    sig components as a tensor, then translate."""
    tf = FixedRandomPose(symmetric_tensor_fields=[["sig11", "sig22", "sig12"]])
    pts = dm.interior.points
    Rm = R2.to(device=pts.device, dtype=pts.dtype)
    tt = None if t is None else t.to(device=pts.device, dtype=pts.dtype)
    interior = tf._transform_mesh(dm.interior, Rm, tt)
    bnds = {k: tf._transform_mesh(b, Rm, tt) for k, b in dm.boundaries.items()}
    from physicsnemo.mesh import DomainMesh
    return DomainMesh(interior=interior, boundaries=bnds, global_data=dm.global_data)

print("######## Part A: pipeline field check (posed sample vs my rotation of the native sample)")
for ds2d, model in ([] if len(sys.argv) > 1 else (("plaid_tensile2d_2d", "isla_tensile2d"), ("plaid_tensile2d", "geotransolver_tensile2d"))):
    _, vl_n, _ = build(ds2d, model); _, vl_p, _ = build(ds2d + "_posed", model)
    idx = next(iter(vl_n.sampler)); assert idx == next(iter(vl_p.sampler))
    (dn, _), (dp, _) = vl_n.dataset[idx], vl_p.dataset[idx]
    key = int(dp.global_data["case_key"]); d = dn.interior.n_spatial_dims
    R = _pose_rotation_for_key(key, 0, 3, axis=2)       # the pose is drawn on the stored 3-D points
    Rd = (R[:d, :d] if d == 2 else R).to(dn.interior.points.device)
    tf = FixedRandomPose(salt=0, translate=1.0, axis=2, symmetric_tensor_fields=[["sig11", "sig22", "sig12"]])
    # the translation vector: posed points minus rotated native points (rigid)
    shift = dp.interior.points.double() - dn.interior.points.double() @ Rd.T
    print(f"## {ds2d}: idx={idx} d={d} key={key} translation rigid? spread={float(shift.std(0).max()):.2e} |t|={float(shift.mean(0).norm()):.4f}")
    mine = tf._transform_mesh(dn.interior, Rd.to(dn.interior.points.dtype), shift.mean(0).to(dn.interior.points.dtype))
    for name in sorted(dn.interior.point_data.keys()):
        a, b = dp.interior.point_data[name].double(), mine.point_data[name].double()
        print(f"   interior.{name:14s} rel diff posed vs mine = {rel(a, b):.2e}   (vs unrotated native {rel(a, dn.interior.point_data[name].double()):.2e})")
    if True:
        ### Hypothesis: the posed pipeline rotates RAW fields and then normalizes with lab-frame means, so the posed
        ### normalized target is R u_n + (R m - m)/s, not R u_n. Undo the normalization, rotate, re-normalize.
        m_u = torch.tensor([-3.38e-4, 1.054e-3, 0.0][:d], dtype=torch.float64, device=Rd.device); s_u = 8.22e-4
        u_raw = dn.interior.point_data["displacement"].double() * s_u + m_u
        u_posed_expected = (u_raw @ Rd.double().T - m_u) / s_u
        print(f"   HYPOTHESIS displacement: posed vs (rotate raw, renormalize) rel diff = {rel(dp.interior.point_data['displacement'].double(), u_posed_expected):.2e}; offset (R m - m)/s per component = {((Rd.double() @ m_u - m_u) / s_u).tolist()}")
        ms = {"sig11": (-0.127, 28.4), "sig22": (57.7, 32.0), "sig12": (8.1e-4, 16.9)}
        raw = {k: dn.interior.point_data[k].double() * ms[k][1] + ms[k][0] for k in ms}
        S = torch.stack([torch.stack([raw["sig11"], raw["sig12"]], -1), torch.stack([raw["sig12"], raw["sig22"]], -1)], -2)
        S = torch.einsum("ij,njk,lk->nil", Rd[:2, :2].double(), S, Rd[:2, :2].double())
        exp = {"sig11": (S[:, 0, 0] - ms["sig11"][0]) / ms["sig11"][1], "sig22": (S[:, 1, 1] - ms["sig22"][0]) / ms["sig22"][1], "sig12": (S[:, 0, 1] - ms["sig12"][0]) / ms["sig12"][1]}
        for k in ms:
            print(f"   HYPOTHESIS {k}: posed vs (rotate raw tensor, renormalize) rel diff = {rel(dp.interior.point_data[k].double(), exp[k]):.2e}")
    if "outer" in dn.boundary_names:
        bn, bp = dn.boundaries["outer"], dp.boundaries["outer"]
        mb = tf._transform_mesh(bn, Rd.to(bn.points.dtype), shift.mean(0).to(bn.points.dtype))
        print(f"   outer.points     rel diff = {rel(bp.points.double(), mb.points.double()):.2e}")
        for name in sorted(bn.cell_data.keys()):
            print(f"   outer.{name:14s} rel diff posed vs mine = {rel(bp.cell_data[name].double(), mb.cell_data[name].double()):.2e}   (vs native {rel(bp.cell_data[name].double(), bn.cell_data[name].double()):.2e})")
    else:
        print("   (this model config drops the in-file boundary; interior only)")
    for name in sorted(dn.global_data.keys()):
        if name in dp.global_data.keys() and dn.global_data[name].dtype.is_floating_point:
            print(f"   global.{name:13s} rel diff posed vs native = {rel(dp.global_data[name].double(), dn.global_data[name].double()):.2e}")

print(f"######## Part B: model-level check ({RUN}, native 2-D pipeline)")
cfg, vl, info = build("plaid_tensile2d_2d", "isla_tensile2d", ["model.hidden=768"])
target_config = info["targets"]; output_type = require_output_type(cfg)
model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(dev)
ep = load_checkpoint(path=f"{T}/runs/{RUN}/checkpoints", models=model, device=dev); print("checkpoint epoch", ep)
model.eval(); collate = vl.collate_fn
def predict(dm):
    batch = recursive_to_device(collate([(dm, {})]), dev)
    with torch.no_grad():
        out = model(**batch["forward_kwargs"])
    td = normalize_output_to_tensordict(out, target_config, output_type)
    return {k: v.detach().float().cpu().reshape(v.shape[-2], -1) if v.dim() > 1 else v.detach().float().cpu() for k, v in td.items()}, {k: v.float().cpu() for k, v in batch["targets"].items()}
idx = next(iter(vl.sampler)); dm, _ = vl.dataset[idx]
p0, y0 = predict(dm)
print("fields:", {k: tuple(v.shape) for k, v in p0.items()}, "native rel err per field:", {k: round(rel(p0[k].reshape(-1), y0[k].reshape(-1)), 4) for k in p0})
def back_rotate(p, R2):
    """Rotate predictions back: scalars unchanged, (N,2) vectors by R^T, sig components as a tensor by R^T . R."""
    q = {k: v.clone() for k, v in p.items()}
    Rt = R2.float().T
    for k, v in q.items():
        if v.dim() == 2 and v.shape[-1] == 2:
            q[k] = v @ Rt.T
    if all(k in q for k in ("sig11", "sig22", "sig12")):
        s = torch.stack([torch.stack([q["sig11"].reshape(-1), q["sig12"].reshape(-1)], -1), torch.stack([q["sig12"].reshape(-1), q["sig22"].reshape(-1)], -1)], -2)
        s = torch.einsum("ij,njk,lk->nil", Rt, s, Rt)
        q["sig11"], q["sig22"], q["sig12"] = s[:, 0, 0], s[:, 1, 1], s[:, 0, 1]
    if "stress" in q and q["stress"].shape[-1] == 3:
        ### the tensor target: the recipe returns the listed components [xx, yy, xy] as one (N, 3) field
        v = q["stress"]
        s = torch.stack([torch.stack([v[:, 0], v[:, 2]], -1), torch.stack([v[:, 2], v[:, 1]], -1)], -2)
        s = torch.einsum("ij,njk,lk->nil", Rt, s, Rt)
        q["stress"] = torch.stack([s[:, 0, 0], s[:, 1, 1], s[:, 0, 1]], -1)
    return q
for label, theta, t in (("rotate 90 deg", math.pi / 2, None), ("rotate 37 deg", 37 * math.pi / 180, None), ("translate (1 RMS radius)", 0.0, None), ("rotate 90 deg + translate", math.pi / 2, None)):
    R2 = rot2(theta)
    tt = None
    if "translate" in label:
        p = dm.interior.points.double(); rms = (p - p.mean(0)).norm(dim=-1).square().mean().sqrt()
        tt = torch.tensor([0.6, -0.8], dtype=torch.float64) * float(rms)
    dmr = rotate_domain(dm, R2, tt)
    p1, y1 = predict(dmr)
    back = back_rotate(p1, R2)
    diffs = {k: round(rel(back[k].reshape(-1), p0[k].reshape(-1)), 4) for k in p0}
    tgt_consistency = {k: round(rel(back_rotate(y1, R2)[k].reshape(-1), y0[k].reshape(-1)), 4) for k in y0}
    print(f"## {label}: pred(back-rotated) vs pred(native) rel diff per field: {diffs}")
    print(f"   (sanity: rotated targets back-rotated vs native targets: {tgt_consistency})")
    print(f"   rel err of pred vs target in the rotated frame: {dict((k, round(rel(p1[k].reshape(-1), y1[k].reshape(-1)), 4)) for k in p1)}")
