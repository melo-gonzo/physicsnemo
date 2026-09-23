"""TERRITORY metric check (2026-09-23, task: mass-weighted anchor displacement).

The published anchor displacement is the RMS over all S slices of |z_s(unif1) - z_s(biased)| / RMS token radius. With
cosine routing R1 spreads mass over 170-240 effective slices, but slices carrying almost no tokens still enter the RMS
with weight 1/S and can dominate it. Here the same displacement is re-reduced with each slice weighted by the number of
tokens whose dominant slice it is in draw A (unif1) -- under a uniform draw that count is proportional to the surface
area the slice owns, i.e. its measure. Both the biased displacement and its resampling floor (unif1 vs unif2) are
computed, per block, per case; medians over blocks x cases as in territory_reduce_r1_2026-09-23.py.
usage: python territory_massweighted.py out.json NAME=<dumps>:<run> [...]
"""
import json, sys
from pathlib import Path
import numpy as np

def load(dumps, run, tag, k):
    d = np.load(dumps / f"{run}_{tag}_case{k}.npz"); return d["points"], d["dominant"], d["anchors"]

def disp(pA, dA, zA, zB, weighted):
    rms_r = np.sqrt(((pA - pA.mean(0)) ** 2).sum(1).mean())
    d = np.linalg.norm(zA - zB, axis=-1)                       # (L, S) or (L, H, S)
    if d.ndim == 3:                                            # heads: fold into slices
        d = d.reshape(d.shape[0], -1); dA = np.moveaxis(dA, -1, 1).reshape(dA.shape[0], -1, dA.shape[1]) if dA.ndim == 3 else dA
    L, S = d.shape
    out = np.empty(L)
    for l in range(L):
        if weighted:
            dom = dA[l].reshape(-1)
            w = np.bincount(dom, minlength=S)[:S].astype(float)
            out[l] = np.sqrt((w * d[l] ** 2).sum() / w.sum())
        else:
            out[l] = np.sqrt((d[l] ** 2).mean())
    return out / rms_r

res = {"metric": "anchor displacement / RMS token radius, per block; 'uniform' = RMS over slices (published), 'mass' = token-count-weighted RMS (dominant-slice counts of draw unif1)", "models": {}}
for spec in sys.argv[2:]:
    name, rest = spec.split("=", 1); dumps, run = rest.split(":")[:2]; dumps = Path(dumps)
    ks = sorted({int(p.stem.split("case")[-1]) for p in dumps.glob(f"{run}_unif1_case*.npz")})
    rows = {"uniform_biased": [], "uniform_floor": [], "mass_biased": [], "mass_floor": [], "eff_slices": []}
    for k in ks:
        p1, d1, z1 = load(dumps, run, "unif1", k); _, _, z2 = load(dumps, run, "unif2", k); _, _, zb = load(dumps, run, "biased", k)
        rows["uniform_biased"].append(disp(p1, d1, z1, zb, False)); rows["uniform_floor"].append(disp(p1, d1, z1, z2, False))
        rows["mass_biased"].append(disp(p1, d1, z1, zb, True)); rows["mass_floor"].append(disp(p1, d1, z1, z2, True))
        dd = d1.reshape(d1.shape[0], -1); S = z1.shape[-2]
        rows["eff_slices"].append(np.array([1.0 / ((np.bincount(dd[l], minlength=S)[:S] / dd.shape[1]) ** 2).sum() for l in range(dd.shape[0])]))
    summ = {}
    for key, v in rows.items():
        a = np.stack(v)  # (cases, L)
        summ[key] = {"median": float(np.median(a)), "max": float(a.max()), "per_block_median": [float(x) for x in np.median(a, axis=0)]}
    res["models"][name] = {"run": run, "cases": ks, **summ}
    print(f"{name:14s} cases={len(ks)} uniform biased {summ['uniform_biased']['median']:.4f} (floor {summ['uniform_floor']['median']:.4f}) | "
          f"mass-weighted biased {summ['mass_biased']['median']:.4f} (floor {summ['mass_floor']['median']:.4f}) | eff. slices median {summ['eff_slices']['median']:.1f}")
Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
