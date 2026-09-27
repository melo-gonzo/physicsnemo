"""UNIFIED test (a) diagnostic: on 3 validation cars, prediction statistics of the volume model at the surface centroids
under variants (sdf 0 / small positive sdf / flipped normals), plus the volume dataset's near-wall sdf and normal
convention, to separate a genuine no-skill result from a convention mismatch. Run like unified_eval_2026-09-27.py."""
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
import hydra, torch
from hydra import compose, initialize_config_dir
T = os.environ["T"]; RECIPE = Path(os.environ["RECIPE"]); sys.path.insert(0, str(RECIPE / "src"))
from datasets import build_dataloaders
from forward_kwargs import extract_target_measure
from metrics import _relative_l2
from utils import set_seed
from physicsnemo import datapipes  # noqa
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint

ap = argparse.ArgumentParser(); ap.add_argument("--run", default="sg_drvol_provided_seed42"); ap.add_argument("--ncars", type=int, default=3); a = ap.parse_args()
DistributedManager.initialize(); dev = DistributedManager().device; set_seed(42, rank=0)
def loaders(ds, model, cells):
    with initialize_config_dir(config_dir=str(RECIPE / "conf"), version_base=None):
        cfg = compose("infer", overrides=[f"model={model}", f"dataset={ds}", f"run_id={a.run}", "infer_split=val", f"sampling_resolution={cells}",
                                          "precision=float32", f"checkpoint_dir={T}/runs", f"output_dir={T}/unified_eval/diag", "training.seed=42", "+save_predictions=false"])
    _, vl, _, _ = build_dataloaders(cfg); return cfg, vl
cfg, vl = loaders("drivaer_ml_surface", "isla_volume_ref_provided", 200000)
model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(dev); load_checkpoint(path=f"{T}/runs/{a.run}/checkpoints", models=model, device=dev); model.eval()
# volume dataset: near-wall sdf and normal convention
_, vvl = loaders("drivaer_ml_volume_reference", "isla_volume_ref_provided", 200000)
vd = vvl.dataset; d0, _ = vd[next(iter(vvl.sampler))]
sdf = d0.interior.point_data["sdf"].reshape(-1); sn = d0.interior.point_data["sdf_normals"]
veh = d0.boundaries["vehicle"]; c = veh.cell_centroids; cn = veh.cell_normals
# nearest boundary centroid for the 2000 nearest-wall points: does sdf_normal point away from the wall (dot with (x - c_near) > 0)?
idx = torch.argsort(sdf.abs())[:2000]; x = d0.interior.points[idx]
dmin = torch.cdist(x, c).argmin(1); rel = x - c[dmin]; rel = rel / rel.norm(dim=1, keepdim=True).clamp_min(1e-9)
print(json.dumps({"vol_sdf_min": float(sdf.min()), "vol_sdf_q01": float(sdf.quantile(0.01)), "vol_sdf_median": float(sdf.median()), "frac_sdf_negative": float((sdf < 0).float().mean()),
                  "nearwall_dot(sdf_normal, x-c_near)_mean": float((sn[idx] * rel).sum(1).mean()), "nearwall_dot(cell_normal_near, x-c_near)_mean": float((cn[dmin] * rel).sum(1).mean()),
                  "nearwall_|sdf|_median": float(sdf[idx].abs().median())}))
# surface cars: variants
for i, idx in enumerate(vl.sampler):
    if i >= a.ncars: break
    dom, meta = vl.dataset[idx]; v = dom.boundaries["vehicle"]
    pts = v.cell_centroids.to(dev).float()[None]; nrm = v.cell_normals.to(dev).float()[None]
    w = extract_target_measure(dom); w = (w if w is not None else v.cell_data["quadrature_measure"]).to(dev).float().reshape(1, -1)
    g = dom.global_data["U_inf_dir"].to(dev).float().reshape(1, 3); tgt = dom.interior.point_data["pressure"].to(dev).float().reshape(1, -1)
    out = {}
    for name, (qn, s) in {"sdf0": (nrm, 0.0), "sdf0_flip": (-nrm, 0.0), "sdf_small": (nrm, float(sdf[idx_ := torch.argsort(sdf.abs())[:2000]].abs().median())), "sdf_small_flip": (-nrm, float(sdf[idx_].abs().median()))}.items():
        with torch.no_grad():
            o = model(points=pts, normals=nrm, measure_weights=w, global_inputs={"freestream": g}, query_points=pts, query_normals=qn, query_inputs={"sdf": torch.full_like(w, s)})
        p = o["pressure"].float().reshape(1, -1)
        out[name] = {"l2": round(_relative_l2(p, tgt, w).item(), 4), "pred_mean": round(p.mean().item(), 4), "pred_std": round(p.std().item(), 4), "corr": round(torch.corrcoef(torch.stack([p[0], tgt[0]]))[0, 1].item(), 4)}
    out["target"] = {"mean": round(tgt.mean().item(), 4), "std": round(tgt.std().item(), 4), "nrm_dot_outward_proxy": round(((v.cell_centroids - v.cell_centroids.mean(0)) / (v.cell_centroids - v.cell_centroids.mean(0)).norm(dim=1, keepdim=True) * v.cell_normals).sum(1).mean().item(), 3)}
    print(json.dumps({"car": i, **out}))
