"""TERRITORY reduction under the reference configuration R1 (2026-09-23, REF-BOARD inference half,
#sec-nb-refboard-prereg). The statistics of territory_reduce_2026-09-21.py, generalized so that each model's dumps can
come from a different directory: the ISLA rows are the R1 checkpoints probed today, the GeoTransolver rows are the
2026-09-21 dumps of the released baseline (same four cases, same draws, seeds 11 / 22 / 33), re-reduced so both sit
in one artifact with the original schema.

Per model and block: agreement between draws (fraction of draw-A tokens whose dominant slice equals that of their
nearest token in draw B; GeoTransolver averaged over heads), the resampling floor A(unif1, unif2), the bias excess
floor - A(unif1, biased), the anchor displacement between unif1 and biased (RMS over slices of |z_s^A - z_s^B| divided
by the RMS token radius) with its resampling floor (the same displacement between the two uniform draws, added today so
that anchor motion under the bias can be read against anchor motion under any resampling), and the post hoc
distance-controlled agreement (pairs at most the median uniform-uniform
nearest-neighbour distance of the same case apart).

usage: python territory_reduce_r1_2026-09-23.py <out.json> --study "<text>" --cells <int> \
           --model NAME=<dumps_dir>:<run_id>[:<label>] [--model ...]
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--study", required=True)
ap.add_argument("--cells", type=int, required=True)
ap.add_argument("--model", action="append", required=True, help="NAME=<dumps_dir>:<run_id>[:<label>]")
args = ap.parse_args()
TAGS = ("unif1", "unif2", "biased")


def load(dumps, run, tag, k):
    d = np.load(dumps / f"{run}_{tag}_case{k}.npz")
    return d["points"], d["dominant"], d["anchors"]


def agreement(pA, dA, pB, dB):
    """Per-block agreement of dominant slices between draw A and draw B, matched by nearest neighbour on the surface;
    also the per-token match distances (for the floor and the distance control)."""
    dist, idx = cKDTree(pB).query(pA, k=1)
    dB_at_A = dB[:, idx] if dB.ndim == 2 else dB[:, idx, :]
    eq = (dA == dB_at_A)                       # (L, N) or (L, N, H)
    return eq, dist


def displacement(pA, zA, zB):
    """RMS over slices (and heads) of the anchor displacement, divided by the RMS token radius of draw A."""
    rms_r = np.sqrt(((pA - pA.mean(0)) ** 2).sum(1).mean())
    d = np.linalg.norm(zA - zB, axis=-1)        # (L, S) or (L, H, S)
    return np.sqrt((d ** 2).reshape(d.shape[0], -1).mean(1)) / rms_r


def per_block_mean(eq, mask=None):
    e = eq if mask is None else (eq[:, mask] if eq.ndim == 2 else eq[:, mask, :])
    return e.reshape(e.shape[0], -1).mean(1)


res = {"study": args.study, "config": "R1", "cells": args.cells,
       "definitions": {"agreement": "fraction of draw-A tokens whose dominant slice equals that of the nearest draw-B token (GeoTransolver: mean over heads)",
                       "floor": "agreement(unif1, unif2)", "excess": "floor - agreement(unif1, biased)",
                       "displacement": "RMS over slices of |anchor(unif1) - anchor(biased)| / RMS token radius",
                       "displacement_floor": "RMS over slices of |anchor(unif1) - anchor(unif2)| / RMS token radius"},
       "models": {},
       "distance_controlled": {"definition": "agreement restricted to matched pairs whose nearest-neighbour distance is at most the median uniform-uniform nearest-neighbour distance of the same case (equalizes matching quality between the sparse half of the biased draw and the uniform draw); post hoc, labeled"}}
for spec in args.model:
    name, rest = spec.split("=", 1)
    parts = rest.split(":")
    dumps, run = Path(parts[0]), parts[1]
    label = parts[2] if len(parts) > 2 else None
    ks = sorted({int(p.stem.split("case")[-1]) for p in dumps.glob(f"{run}_unif1_case*.npz")})
    if not ks:
        raise SystemExit(f"no dumps for {run} under {dumps}")
    floors, biased, disp, dfloor, nn_dist, identical, fctrl, bctrl, kept = [], [], [], [], [], [], [], [], []
    for k in ks:
        p1, d1, z1 = load(dumps, run, "unif1", k); p2, d2, z2 = load(dumps, run, "unif2", k); pb, db, zb = load(dumps, run, "biased", k)
        identical.append(bool(p1.shape == p2.shape and np.allclose(p1, p2)))
        eq12, dist12 = agreement(p1, d1, p2, d2); eq1b, dist1b = agreement(p1, d1, pb, db)
        floors.append(per_block_mean(eq12)); biased.append(per_block_mean(eq1b)); disp.append(displacement(p1, z1, zb)); dfloor.append(displacement(p1, z1, z2))
        nn_dist.append((float(np.median(dist12)), float(np.median(dist1b))))
        thr = np.median(dist12)
        fctrl.append(per_block_mean(eq12, dist12 <= thr)); bctrl.append(per_block_mean(eq1b, dist1b <= thr)); kept.append(float((dist1b <= thr).mean()))
    floors, biased, disp, dfloor = np.mean(floors, 0), np.mean(biased, 0), np.mean(disp, 0), np.mean(dfloor, 0)
    fctrl, bctrl = np.mean(fctrl, 0), np.mean(bctrl, 0)
    excess = floors - biased
    entry = {"run": run, "n_cases": len(ks), "uniform_draws_identical": identical,
             "median_nn_distance_unif_unif": float(np.median([d[0] for d in nn_dist])),
             "median_nn_distance_unif_biased": float(np.median([d[1] for d in nn_dist])),
             "per_block": [{"block": i, "floor": float(floors[i]), "agreement_biased": float(biased[i]),
                            "excess": float(excess[i]), "displacement": float(disp[i]), "displacement_floor": float(dfloor[i])} for i in range(len(floors))],
             "summary": {"floor_median": float(np.median(floors)), "floor_min": float(floors.min()),
                         "agreement_biased_median": float(np.median(biased)), "agreement_biased_min": float(biased.min()),
                         "excess_median": float(np.median(excess)), "excess_max": float(excess.max()),
                         "displacement_median": float(np.median(disp)), "displacement_max": float(disp.max()),
                         "displacement_floor_median": float(np.median(dfloor)), "displacement_floor_max": float(dfloor.max())}}
    if label:
        entry["label"] = label
    res["models"][name] = entry
    res["distance_controlled"][name] = {"floor_ctrl": [float(x) for x in fctrl], "biased_ctrl": [float(x) for x in bctrl],
                                        "excess_ctrl_median": float(np.median(fctrl - bctrl)), "biased_pairs_kept": float(np.mean(kept))}
    print(f"\n== {name} ({run}), {len(ks)} cases; uniform draws identical: {identical}")
    print("block  floor  biased  excess  displacement  displ.floor | ctrl floor  ctrl biased")
    for i in range(len(floors)):
        print(f"{i:5d}  {floors[i]:.3f}  {biased[i]:.3f}  {excess[i]:+.3f}  {disp[i]:.4f}  {dfloor[i]:.4f} | {fctrl[i]:.3f}  {bctrl[i]:.3f}")
    s = entry["summary"]
    print(f"median: floor {s['floor_median']:.3f}  biased {s['agreement_biased_median']:.3f}  excess {s['excess_median']:+.3f}  "
          f"displacement {s['displacement_median']:.4f} (floor {s['displacement_floor_median']:.4f})  (ctrl excess {np.median(fctrl - bctrl):+.3f}, biased pairs kept {np.mean(kept):.2f})")
Path(args.out).write_text(json.dumps(res, indent=2))
print("wrote", args.out)
