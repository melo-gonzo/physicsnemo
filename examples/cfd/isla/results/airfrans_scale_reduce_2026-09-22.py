"""AIRFRANS-SCALE reduction (notebook #sec-nb-airfrans-scale-prereg): the training-case axis of the scaling law on
AirfRANS for ISLA cosine 192 (1e-3) and GeoTransolver-volume (1e-3), rungs 50 / 100 / 200 / 400 (af rows 14-29) and the
800-case full-split pair, two seeds, 100,000 queries, float32 on the 200 test cases.

Per field and architecture: seed means per rung, local exponents between adjacent rungs, and the two-parameter fit
eps(n) = eps_inf + a n^-alpha with eps_inf in [0, 0.9 x the 800-case error] (least squares on the relative residual of the
seed means; 400-draw two-seed bootstrap). Bars: same alpha across architectures if the difference is below the bootstrap
spread; transfer if ISLA's pressure alpha is within 0.3 of the DrivAerML surface value (1.28); the ordering (GT / ISLA
below one on pressure, above one on velocity) at every rung with seeds agreeing; a rung is "unresolved" if the larger
seed spread of the two arms exceeds the relative gap to the next rung.

usage: python airfrans_scale_reduce_2026-09-22.py <fetched p200k_fp32 dir> <airfrans_reduction (full split).json> <out.json>
"""
from __future__ import annotations

import glob, json, statistics as st, sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

root, full_path, out_path = (Path(a) for a in sys.argv[1:4])
FULL = json.load(open(full_path))["arms"]
FIELDS = ("pressure_l2", "velocity_l2", "nut_l2")
RUNGS = (50, 100, 200, 400)
ARMS = {"isla": ("af_isla_cos_lr1e3_n{n}_seed{seed}", "isla_cosine_1e-3"), "gt": ("af_gt_vol_lr1e3_n{n}_seed{seed}", "gt_volume_1e-3")}
SEEDS = (42, 43)
DRIVAERML_ALPHA_ISLA = 1.28


def summary(run):
    """Float32 infer_summary metrics of one run, or None if unscored."""
    ps = glob.glob(str(root / run / "**" / "metrics.jsonl"), recursive=True)
    if not ps:
        return None
    rows = [json.loads(l) for l in open(ps[0]) if l.strip()]
    s = [r for r in rows if r.get("phase") == "infer_summary"]
    return {f: s[0]["metrics"][f] for f in FIELDS if f in s[0]["metrics"]} if s else None


def fit(n, y):
    """eps_inf + a n^-alpha on the relative residual, eps_inf bounded to [0, 0.9 x best]."""
    ub = 0.9 * float(min(y))
    res = lambda th: (th[2] + np.exp(th[0]) * n ** (-th[1])) / y - 1
    sols = [least_squares(res, s, bounds=([-20, 0, 0], [20, 4, ub]), max_nfev=20000) for s in ([0.0, 0.5, ub / 2], [np.log(3), 1.2, ub * 0.9], [np.log(0.3), 0.3, 0.0])]
    th = min(sols, key=lambda r: r.cost).x
    return {"a": float(np.exp(th[0])), "alpha": float(th[1]), "eps_inf": float(th[2]), "eps_inf_on_bound": bool(ub - th[2] < 1e-4 * ub), "rms_rel_residual": float(np.sqrt(np.mean(res(th) ** 2)))}


rng = np.random.default_rng(0)
conf = {}
for arm, (pat, full_key) in ARMS.items():
    for n in RUNGS:
        runs = {pat.format(n=n, seed=s): summary(pat.format(n=n, seed=s)) for s in SEEDS}
        got = [v for v in runs.values() if v]
        c = {"arm": arm, "n": n, "runs": runs, "status": "complete" if len(got) == 2 else ("partial" if got else "pending")}
        for f in FIELDS:
            if got:
                vals = [v[f] for v in got]; c[f] = {"seeds": vals, "mean": st.mean(vals), "seed_spread": max(vals) / min(vals) - 1 if len(vals) == 2 else None}
        conf[f"{arm}|n{n}"] = c
    conf[f"{arm}|n800"] = {"arm": arm, "n": 800, "runs": FULL[full_key]["runs"], "status": FULL[full_key]["status"],
                           **{f: {"seeds": FULL[full_key][f]["seeds"], "mean": FULL[full_key][f]["mean"], "seed_spread": FULL[full_key][f]["seed_spread"]} for f in FIELDS}}

fits = {}
for arm in ARMS:
    for f in FIELDS:
        pts = sorted([c for c in conf.values() if c["arm"] == arm and c["status"] == "complete" and f in c], key=lambda c: c["n"])
        F = {"n_rungs": len(pts), "local_alpha": {}, "unresolved_rungs": []}
        for a, b in zip(pts, pts[1:]):
            F["local_alpha"][f"{a['n']}-{b['n']}"] = float(-np.log(b[f]["mean"] / a[f]["mean"]) / np.log(b["n"] / a["n"]))
            gap = abs(a[f]["mean"] / b[f]["mean"] - 1)
            if max(a[f]["seed_spread"] or 0, b[f]["seed_spread"] or 0) > gap:
                F["unresolved_rungs"].append(f"{a['n']}-{b['n']}")
        if len(pts) >= 4:
            n = np.array([c["n"] for c in pts], float); y = np.array([c[f]["mean"] for c in pts])
            F["fit"] = fit(n, y)
            boots = [fit(n, np.array([c[f]["seeds"][rng.integers(len(c[f]["seeds"]))] for c in pts])) for _ in range(400)]
            F["bootstrap_alpha_p16_p50_p84"] = [float(x) for x in np.percentile([b["alpha"] for b in boots], [16, 50, 84])]
            F["bootstrap_eps_inf_p16_p50_p84"] = [float(x) for x in np.percentile([b["eps_inf"] for b in boots], [16, 50, 84])]
        fits[f"{arm}|{f}"] = F

bars = {}
for f in FIELDS:
    fi, fg = fits[f"isla|{f}"], fits[f"gt|{f}"]
    if "fit" in fi and "fit" in fg:
        spread = max(fi["bootstrap_alpha_p16_p50_p84"][2] - fi["bootstrap_alpha_p16_p50_p84"][0], fg["bootstrap_alpha_p16_p50_p84"][2] - fg["bootstrap_alpha_p16_p50_p84"][0])
        bars[f] = {"alpha_isla": fi["fit"]["alpha"], "alpha_gt": fg["fit"]["alpha"], "difference": abs(fi["fit"]["alpha"] - fg["fit"]["alpha"]), "bootstrap_spread": spread,
                   "same_exponent": abs(fi["fit"]["alpha"] - fg["fit"]["alpha"]) < spread}
        if f == "pressure_l2":
            bars[f]["transfer_from_drivaerml"] = {"drivaerml_alpha_isla_surface": DRIVAERML_ALPHA_ISLA, "difference": abs(fi["fit"]["alpha"] - DRIVAERML_ALPHA_ISLA), "within_0.3": abs(fi["fit"]["alpha"] - DRIVAERML_ALPHA_ISLA) <= 0.3}
ordering = {}
for f, side in (("pressure_l2", "<1"), ("velocity_l2", ">1")):
    per_rung = {}
    for n in RUNGS + (800,):
        ci, cg = conf[f"isla|n{n}"], conf[f"gt|n{n}"]
        if ci["status"] == "complete" and cg["status"] == "complete":
            paired = [cg[f]["seeds"][i] / ci[f]["seeds"][i] for i in range(2)]
            holds = all((p < 1) if side == "<1" else (p > 1) for p in paired)
            per_rung[n] = {"gt_over_isla": cg[f]["mean"] / ci[f]["mean"], "paired": paired, "holds": holds}
    ordering[f] = {"expected_side": side, "per_rung": per_rung, "holds_at_every_scored_rung": all(v["holds"] for v in per_rung.values()) if per_rung else None}
bars["ordering"] = ordering

out = {"study": "AIRFRANS-SCALE: interior error against training airfoils (50 / 100 / 200 / 400 / 800), ISLA cosine 192 (1e-3) and GeoTransolver-volume (1e-3), two seeds, 100,000 queries, float32 on the 200 test cases",
       "law": "eps(n) = eps_inf + a n^-alpha, eps_inf in [0, 0.9 x best]; 400-draw two-seed bootstrap", "configurations": conf, "fits": fits, "bars": bars}
out_path.write_text(json.dumps(out, indent=2))
for key in sorted(conf, key=lambda k: (conf[k]["arm"], conf[k]["n"])):
    c = conf[key]; print(f"{key:12s} {c['status']:8s}", {f: (round(c[f]["mean"], 4), [round(v, 4) for v in c[f]["seeds"]]) for f in FIELDS if f in c})
for k, F in fits.items():
    print(k, "local alpha", {r: round(v, 2) for r, v in F["local_alpha"].items()}, "fit", {kk: round(vv, 3) for kk, vv in F.get("fit", {}).items() if isinstance(vv, float)}, "unresolved", F["unresolved_rungs"])
print(json.dumps(bars, indent=1, default=str)[:3000])
