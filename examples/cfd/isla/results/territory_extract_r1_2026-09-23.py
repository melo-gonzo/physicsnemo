"""Picture artifact for the R1 territory maps (2026-09-23, REF-BOARD inference half): territory_extract_2026-09-21.py
generalized so each model's dumps come from their own directory (ISLA: the R1 probe of today; GeoTransolver: the
2026-09-21 dumps of the released baseline, same case, same draws). One validation case, one block, the uniform and
the biased draw, subsampled to a fixed number of points per panel, with the dominant slice of each point and the slice
of the nearest uniform-draw token; plus the per-block agreement table of the reduction.

usage: python territory_extract_r1_2026-09-23.py <reduction.json> <out.npz> --isla <dumps_dir> --gt <dumps_dir> [--case 0] [--block 6] [--n 30000]
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("reduction"); ap.add_argument("out")
ap.add_argument("--isla", required=True); ap.add_argument("--gt", required=True)
ap.add_argument("--case", type=int, default=0); ap.add_argument("--block", type=int, default=6); ap.add_argument("--n", type=int, default=30000)
args = ap.parse_args()
red = json.load(open(args.reduction))
rng = np.random.default_rng(0)
store = {"block": args.block, "case": args.case, "config": "R1"}
for name, key, dumps in (("isla", "ISLA", Path(args.isla)), ("gt", "GeoTransolver", Path(args.gt))):
    run = red["models"][key]["run"]
    u = np.load(dumps / f"{run}_unif1_case{args.case}.npz"); b = np.load(dumps / f"{run}_biased_case{args.case}.npz")
    du, db = u["dominant"][args.block], b["dominant"][args.block]
    if du.ndim == 2:                      # GeoTransolver: (N, H) -> head 0 for the picture
        du, db = du[:, 0], db[:, 0]
    idx_b_in_u = cKDTree(u["points"]).query(b["points"], k=1)[1]
    su = rng.choice(len(du), size=min(args.n, len(du)), replace=False)
    sb = rng.choice(len(db), size=min(args.n, len(db)), replace=False)
    store[f"{name}_run"] = run
    store[f"{name}_unif_points"] = u["points"][su].astype(np.float32); store[f"{name}_unif_dominant"] = du[su].astype(np.int16)
    store[f"{name}_biased_points"] = b["points"][sb].astype(np.float32); store[f"{name}_biased_dominant"] = db[sb].astype(np.int16)
    store[f"{name}_biased_dominant_of_nearest_uniform"] = du[idx_b_in_u][sb].astype(np.int16)
    store[f"{name}_agreement_block"] = float((db == du[idx_b_in_u]).mean())
    pb = red["models"][key]["per_block"]
    store[f"{name}_floor"] = np.array([r["floor"] for r in pb], np.float32)
    store[f"{name}_biased"] = np.array([r["agreement_biased"] for r in pb], np.float32)
    store[f"{name}_displacement"] = np.array([r["displacement"] for r in pb], np.float32)
np.savez_compressed(args.out, **store)
print("wrote", args.out, {k: (v.shape if hasattr(v, "shape") else v) for k, v in store.items() if not k.endswith("points")})
