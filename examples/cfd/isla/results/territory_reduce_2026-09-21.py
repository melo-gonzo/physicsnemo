"""TERRITORY reduction (2026-09-21, prereg #sec-nb-territory-prereg).

Reads the per-case dumps written by territory_probe.py (<dumps>/<run>_<tag>_case<k>.npz with points (N,3),
dominant (L,N) for ISLA or (L,N,H) for GeoTransolver, anchors (L,S,3) or (L,H,S,3)) and computes, per model and
block: the agreement between draws (fraction of tokens of draw A whose dominant slice equals that of their
nearest token in draw B; for GeoTransolver averaged over heads), the resampling floor A(unif1, unif2), the bias
excess A(unif1, unif2) - A(unif1, biased), and the anchor displacement between unif1 and biased (RMS over slices
of |z_s^A - z_s^B| divided by the RMS token radius). Writes the JSON artifact and prints a per-block table.

usage: python territory_reduce.py <dumps_dir> <out.json>
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

dumps, out = Path(sys.argv[1]), Path(sys.argv[2])
RUNS = {"ISLA": "iso_dr_isla_w192_c100k_seed42", "GeoTransolver": "iso_dr_gt_unit_w256_c100k_seed42"}
TAGS = ("unif1", "unif2", "biased")


def load(run, tag, k):
    d = np.load(dumps / f"{run}_{tag}_case{k}.npz")
    return d["points"], d["dominant"], d["anchors"]


def agreement(pA, dA, pB, dB):
    """Per-block agreement of dominant slices between draw A and draw B, matched by nearest neighbour on the surface."""
    idx = cKDTree(pB).query(pA, k=1)[1]
    dB_at_A = dB[:, idx] if dB.ndim == 2 else dB[:, idx, :]
    eq = (dA == dB_at_A)                       # (L, N) or (L, N, H)
    return eq.reshape(eq.shape[0], -1).mean(1), float(np.median(np.linalg.norm(pB[idx] - pA, axis=1)))


def displacement(pA, zA, zB):
    """RMS over slices (and heads) of the anchor displacement, divided by the RMS token radius of draw A."""
    rms_r = np.sqrt(((pA - pA.mean(0)) ** 2).sum(1).mean())
    d = np.linalg.norm(zA - zB, axis=-1)        # (L, S) or (L, H, S)
    return np.sqrt((d ** 2).reshape(d.shape[0], -1).mean(1)) / rms_r


res = {"study": "TERRITORY: slice territories of the uniform-trained ISLA-192 and GeoTransolver-256 checkpoints (435 DrivAerML cars, 100,000 cells) on four validation cars under two independent uniform draws and the 10:1-biased draw at 100,000 expected cells, float32",
       "definitions": {"agreement": "fraction of draw-A tokens whose dominant slice equals that of the nearest draw-B token (GeoTransolver: mean over heads)",
                       "floor": "agreement(unif1, unif2)", "excess": "floor - agreement(unif1, biased)",
                       "displacement": "RMS over slices of |anchor(unif1) - anchor(biased)| / RMS token radius"},
       "models": {}}
for name, run in RUNS.items():
    ks = sorted({int(p.stem.split("case")[-1]) for p in dumps.glob(f"{run}_unif1_case*.npz")})
    floors, biased, disp, nn_dist, identical = [], [], [], [], []
    for k in ks:
        p1, d1, z1 = load(run, "unif1", k); p2, d2, z2 = load(run, "unif2", k); pb, db, zb = load(run, "biased", k)
        identical.append(bool(p1.shape == p2.shape and np.allclose(p1, p2)))
        a12, nd12 = agreement(p1, d1, p2, d2); a1b, nd1b = agreement(p1, d1, pb, db)
        floors.append(a12); biased.append(a1b); disp.append(displacement(p1, z1, zb)); nn_dist.append((nd12, nd1b))
    floors, biased, disp = np.mean(floors, 0), np.mean(biased, 0), np.mean(disp, 0)
    excess = floors - biased
    res["models"][name] = {"run": run, "n_cases": len(ks), "uniform_draws_identical": identical,
                           "median_nn_distance_unif_unif": float(np.median([d[0] for d in nn_dist])),
                           "median_nn_distance_unif_biased": float(np.median([d[1] for d in nn_dist])),
                           "per_block": [{"block": i, "floor": float(floors[i]), "agreement_biased": float(biased[i]),
                                          "excess": float(excess[i]), "displacement": float(disp[i])} for i in range(len(floors))],
                           "summary": {"floor_median": float(np.median(floors)), "floor_min": float(floors.min()),
                                       "agreement_biased_median": float(np.median(biased)), "agreement_biased_min": float(biased.min()),
                                       "excess_median": float(np.median(excess)), "excess_max": float(excess.max()),
                                       "displacement_median": float(np.median(disp)), "displacement_max": float(disp.max())}}
    print(f"\n== {name} ({run}), {len(ks)} cars; uniform draws identical: {identical}")
    print("block  floor  biased  excess  displacement")
    for i in range(len(floors)):
        print(f"{i:5d}  {floors[i]:.3f}  {biased[i]:.3f}  {excess[i]:+.3f}  {disp[i]:.4f}")
    s = res["models"][name]["summary"]
    print(f"median: floor {s['floor_median']:.3f}  biased {s['agreement_biased_median']:.3f}  excess {s['excess_median']:+.3f}  displacement {s['displacement_median']:.4f}")
out.write_text(json.dumps(res, indent=2))
print("wrote", out)
