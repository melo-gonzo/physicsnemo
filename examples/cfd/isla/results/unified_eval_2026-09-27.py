"""UNIFIED-TOKENS test (a) (2026-09-27, #sec-nb-unified-prereg): evaluate a DrivAerML VOLUME-trained ISLA (interior task,
provided geometry; runs sg_drvol_provided_seed42/43 on code_v3 / recipe_v3) at the SURFACE cell centroids it was never
trained on, and score its surface pressure against the surface dataset's pressure target with the same metric the
fleet scorer writes (measure-weighted relative L2 in pressure-coefficient space; neither run normalizes pressure, so the
numbers are directly comparable with p200k_dr_isla_seed42/43's pressure_l2).

Boundary points as queries: query_points = the vehicle cell centroids, query_normals = the outward cell normals (the
limit of the signed-distance gradient at the wall), the query scalar sdf = 0. Boundary tokens, measure weights and the
freestream direction are the surface dataset's own (the same 200,000-cell draw the fleet scorer uses under the same
seed and split order). Pressure only: the volume model has no wall-shear output.

Run ON the cluster from recipe_v3/src with PYTHONPATH=code_v3:
  python unified_eval_2026-09-27.py --run sg_drvol_provided_seed42 --seed 42 --out $T/unified_eval
Writes <out>/<run>/metrics.jsonl (config row, one infer_step row per car, infer_summary row).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

T = os.environ.get("T", "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0")
RECIPE = Path(os.environ.get("RECIPE", f"{T}/recipe_v3"))
sys.path.insert(0, str(RECIPE / "src"))

from datasets import build_dataloaders  # noqa: E402
from forward_kwargs import extract_target_measure  # noqa: E402
from infer import _sample_id  # noqa: E402
from metrics import _relative_l2  # noqa: E402
from utils import set_seed  # noqa: E402

from physicsnemo import datapipes  # noqa: E402,F401  (registers ${dp:...})
from physicsnemo.distributed import DistributedManager  # noqa: E402
from physicsnemo.utils import load_checkpoint  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model", default="isla_volume_ref_provided")
    ap.add_argument("--dataset", default="drivaer_ml_surface")
    ap.add_argument("--cells", type=int, default=200000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--flip-normals", action="store_true", help="query normals = -cell_normals (the surface dataset's cell normals point INTO the body; the volume model's query normals are the distance gradient pointing into the fluid; measured 2026-09-27 diagnostic)")
    ap.add_argument("--sdf", type=float, default=0.0, help="query sdf value at the wall (0 = exact; near-wall training points sit at |sdf| ~ 1e-7)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    DistributedManager.initialize()
    device = DistributedManager().device
    set_seed(a.seed, rank=0)
    with initialize_config_dir(config_dir=str(RECIPE / "conf"), version_base=None):
        cfg = compose(
            config_name="infer",
            overrides=[
                f"model={a.model}", f"dataset={a.dataset}", f"run_id={a.run}", "infer_split=val",
                f"sampling_resolution={a.cells}", "precision=float32", f"checkpoint_dir={T}/runs",
                f"output_dir={a.out}", f"training.seed={a.seed}", "+save_predictions=false",
            ],
        )
    _train, val_loader, _norm, info = build_dataloaders(cfg)
    dataset, sampler = val_loader.dataset, val_loader.sampler

    model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(device)
    ckpt = f"{T}/runs/{a.run}/checkpoints"
    epoch = load_checkpoint(path=ckpt, models=model, device=device)
    assert epoch > 0, f"no checkpoint restored from {ckpt}"
    model.eval()

    out_dir = Path(a.out) / (a.run + a.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    f = open(out_dir / "metrics.jsonl", "w")
    log = lambda rec: (f.write(json.dumps(rec) + "\n"), f.flush())
    log({"phase": "config", "model": "ISLA", "run": a.run, "dataset": a.dataset, "infer_split": "val", "checkpoint": ckpt,
         "epoch": epoch, "cells": a.cells, "seed": a.seed, "note": "volume model evaluated at surface cell centroids", "flip_normals": a.flip_normals, "sdf": a.sdf})

    tot = 0.0
    tot_cp_unw = 0.0
    n = 0
    for i, idx in enumerate(sampler):
        t0 = time.time()
        domain, meta = dataset[idx]
        veh = domain.boundaries["vehicle"]
        pts = veh.cell_centroids.to(device).float()[None]  # (1, N, 3)
        nrm = veh.cell_normals.to(device).float()[None]  # the boundary tokens' normals, as trained
        qn = -nrm if a.flip_normals else nrm  # query normals: distance gradient into the fluid = -cell_normals on this dataset
        w = extract_target_measure(domain)
        if w is None:
            w = veh.cell_data["quadrature_measure"]
        w = w.to(device).float().reshape(1, -1)  # (1, N)
        g = domain.global_data["U_inf_dir"].to(device).float().reshape(1, 3)
        target = domain.interior.point_data["pressure"].to(device).float().reshape(1, -1)  # Cp, no normalizer on pressure
        assert domain.interior.points.shape[0] == pts.shape[1], "interior points are not the cell centroids"
        with torch.no_grad():
            out = model(points=pts, normals=nrm, measure_weights=w, global_inputs={"freestream": g},
                        query_points=pts, query_normals=qn, query_inputs={"sdf": torch.full_like(w, a.sdf)})
        pred = out["pressure"].float().reshape(1, -1)
        l2 = _relative_l2(pred, target, w).item()
        l2_unw = _relative_l2(pred, target, None).item()
        sid = _sample_id(meta, idx)
        log({"phase": "infer_step", "step": i, "sample_id": sid, "metrics": {"pressure_l2": l2, "pressure_l2_unweighted": l2_unw},
             "n_cells": int(pts.shape[1]), "seconds": round(time.time() - t0, 2)})
        tot += l2
        tot_cp_unw += l2_unw
        n += 1
        print(f"[{i + 1}] {sid} pressure_l2={l2:.4f} (unweighted {l2_unw:.4f}) cells={pts.shape[1]} {time.time() - t0:.1f}s", flush=True)
    log({"phase": "infer_summary", "n_samples": n, "metrics": {"pressure_l2": tot / n, "pressure_l2_unweighted": tot_cp_unw / n}})
    f.close()
    print(f"SUMMARY {a.run}: pressure_l2 mean {tot / n:.5f} over {n} cars")


if __name__ == "__main__":
    main()
