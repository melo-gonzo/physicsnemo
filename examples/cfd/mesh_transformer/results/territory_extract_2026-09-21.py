"""Extract a small picture artifact from the TERRITORY dumps for the book: one validation car, one block, both
models, the uniform draw and the biased draw, subsampled to a fixed number of points per panel, with the
dominant slice of each point and, for the biased-draw panel, the slice of the nearest uniform-draw token (so the
figure can show where the territory moved). Also stores the per-block agreement table.

usage: python territory_extract.py <dumps_dir> <reduction.json> <out.npz> [case] [block] [n_points]
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

dumps, red, out = Path(sys.argv[1]), json.load(open(sys.argv[2])), Path(sys.argv[3])
case = int(sys.argv[4]) if len(sys.argv) > 4 else 0
block = int(sys.argv[5]) if len(sys.argv) > 5 else 6
npts = int(sys.argv[6]) if len(sys.argv) > 6 else 30000
rng = np.random.default_rng(0)
store = {"block": block, "case": case}
for name, run in (("isla", red["models"]["ISLA"]["run"]), ("gt", red["models"]["GeoTransolver"]["run"])):
    u = np.load(dumps / f"{run}_unif1_case{case}.npz"); b = np.load(dumps / f"{run}_biased_case{case}.npz")
    du, db = u["dominant"][block], b["dominant"][block]
    if du.ndim == 2:                      # GeoTransolver: (N, H) -> head 0 for the picture
        du, db = du[:, 0], db[:, 0]
    idx_b_in_u = cKDTree(u["points"]).query(b["points"], k=1)[1]
    su = rng.choice(len(du), size=min(npts, len(du)), replace=False)
    sb = rng.choice(len(db), size=min(npts, len(db)), replace=False)
    store[f"{name}_unif_points"] = u["points"][su].astype(np.float32); store[f"{name}_unif_dominant"] = du[su].astype(np.int16)
    store[f"{name}_biased_points"] = b["points"][sb].astype(np.float32); store[f"{name}_biased_dominant"] = db[sb].astype(np.int16)
    store[f"{name}_biased_dominant_of_nearest_uniform"] = du[idx_b_in_u][sb].astype(np.int16)
    store[f"{name}_agreement_block"] = float((db == du[idx_b_in_u]).mean())
    pb = red["models"]["ISLA" if name == "isla" else "GeoTransolver"]["per_block"]
    store[f"{name}_floor"] = np.array([r["floor"] for r in pb], np.float32)
    store[f"{name}_biased"] = np.array([r["agreement_biased"] for r in pb], np.float32)
    store[f"{name}_displacement"] = np.array([r["displacement"] for r in pb], np.float32)
np.savez_compressed(out, **store)
print("wrote", out, {k: (v.shape if hasattr(v, 'shape') else v) for k, v in store.items() if not k.endswith('points')})
