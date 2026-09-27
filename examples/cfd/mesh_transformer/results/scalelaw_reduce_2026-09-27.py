"""SCALE-LAW reduction, full DrivAerML board (notebook #sec-nb-campaigns-2026-09-21-prereg, #sec-nb-scalelaw-interim).

Surface-pressure error against training-set size n and parameter count P for cosine-routing ISLA (192 / 384 / 768 wide)
and GeoTransolver (256 / 512 / 1,024 wide), 500 epochs at 100,000 cells, float32 scoring, two seeds. DrivAerML rungs
27 / 54 / 109 / 218 cars come from the sl lanes; the 435-car points are the ROUTE-NORM cosine ladder at the sl lanes' rates, 1e-3 / 5e-4 / 5e-4 (ISLA) and the
CAP-435 ladder at each width's better rate (GeoTransolver), as in the chapter's figure. HiLiftAeroML rungs (35 / 210 /
510 wings) are listed as pending until scored.

Preregistered fit (#sec-nb-scalelaw-interim): eps(n, P) = eps_inf + a n^-alpha + b P^-beta, least squares on the
relative residual of the seed means, eps_inf bounded to [0, 0.9 x best measured error], two-seed bootstrap (one seed drawn
per configuration, 400 draws, 16th/50th/84th percentiles). Bars: the law is accepted for a dataset if the RMS relative
residual is below 3%; the two architectures' exponents are "the same" if alpha and beta differ by less than the bootstrap
spread; the compute-optimal split is the case count n* at which the fitted data and parameter terms are equal at the
recipe width (ISLA 192, GeoTransolver 256). The reduction also reports whether eps_inf sits on its bound (then the
coefficients are bound-determined and the law is fitted, not identified) and the local exponents between adjacent rungs.

usage: python scalelaw_reduce_2026-09-22.py <fetched p200k_fp32 dir> <sl lanes.tsv> <cap435.json> <routenorm_cars.json> <out.json>
"""

from __future__ import annotations

import csv, glob, json, statistics as st, sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

root, lanes, cap_path, rn_path, out_path = (Path(a) for a in sys.argv[1:6])
CAP = json.load(open(cap_path))["configurations"]
RN = json.load(open(rn_path))["configurations"]
PARAMS = {
    ("isla", 192): CAP["isla|w192|d12|s256|lr1e-3"]["params"],
    ("isla", 384): CAP["isla|w384|d12|s256|lr1e-3"]["params"],
    ("isla", 768): CAP["isla|w768|d12|s256|lr5e-4"]["params"],
    ("gt", 256): CAP["gt|w256|d12|s256|lr3e-3"]["params"],
    ("gt", 512): CAP["gt|w512|d12|s256|lr3e-3"]["params"],
    ("gt", 1024): CAP["gt|w1024|d12|s256|lr3e3"]["params"],
}
N_OF = {
    "n27": 27,
    "n54": 54,
    "n109": 109,
    "n218": 218,
    "super_scarce": 35,
    "scarce": 210,
    "medium": 510,
    "full": 1260,
}
RECIPE_WIDTH = {"isla": 192, "gt": 256}


def score(run):
    """Float32 surface-pressure L2 of one run from its metrics.jsonl, or None if unscored."""
    ps = glob.glob(str(root / run / "**" / "metrics.jsonl"), recursive=True)
    if not ps:
        return None
    rows = [json.loads(l) for l in open(ps[0]) if l.strip()]
    s = [r for r in rows if r.get("phase") == "infer_summary"]
    return s[0]["metrics"].get("pressure_l2") if s else None


conf: dict = {}
for r in csv.DictReader(open(lanes), delimiter="\t"):
    rid = r["run_id"]
    parts = rid.split("_")
    ds = "drivaerml" if parts[1] == "dr" else "hilift"
    model = "isla" if parts[2] == "isla" else "gt"
    w = int([p for p in parts if p.startswith("w")][0][1:])
    rung = [p for p in N_OF if f"_{p}_" in rid][0]
    key = f"{ds}|{model}|w{w}|n{N_OF[rung]}"
    conf.setdefault(
        key,
        {
            "dataset": ds,
            "model": model,
            "width": w,
            "n": N_OF[rung],
            "params": PARAMS[(model, w)],
            "runs": {},
            "source": "sl lanes",
        },
    )
    conf[key]["runs"][rid] = score(rid)
# 435-car points from the routing and capacity ladders
for w, key in (
    (192, "cars|cos|w192|lr1e-3"),
    (384, "cars|cos|w384|lr5e-4"),
    (768, "cars|cos|w768|lr5e-4"),
):
    c = RN[key]
    conf[f"drivaerml|isla|w{w}|n435"] = {
        "dataset": "drivaerml",
        "model": "isla",
        "width": w,
        "n": 435,
        "params": PARAMS[("isla", w)],
        "runs": c["runs"],
        "source": "routenorm_cars (cosine ladder)",
    }
for w, keys in (
    (256, ["gt|w256|d12|s256|lr3e-3"]),
    (512, ["gt|w512|d12|s256|lr3e-3", "gt|w512|d12|s256|lr1e-3"]),
    (1024, ["gt|w1024|d12|s256|lr3e3", "gt|w1024|d12|s256|lr1e3"]),
):
    cands = [(k, CAP[k]) for k in keys if k in CAP and CAP[k].get("pressure_l2_mean")]
    k, c = min(cands, key=lambda kc: kc[1]["pressure_l2_mean"])
    conf[f"drivaerml|gt|w{w}|n435"] = {
        "dataset": "drivaerml",
        "model": "gt",
        "width": w,
        "n": 435,
        "params": PARAMS[("gt", w)],
        "runs": {f"{k}|seed{i}": v for i, v in enumerate(c["pressure_l2_seeds"])},
        "source": f"cap435 ({k}, better rate)",
    }
for c in conf.values():
    vals = [v for v in c["runs"].values() if v is not None]
    c["n_scored"] = len(vals)
    c["pressure_l2_seeds"] = vals
    c["pressure_l2_mean"] = st.mean(vals) if vals else None
    c["status"] = "complete" if len(vals) >= 2 else ("partial" if vals else "pending")


def law(th, n, P):
    """The bounded-floor law eps_inf + a n^-alpha + b P^-beta with th = (log a, alpha, log b, beta, eps_inf)."""
    la, al, lb, be, c = th
    return np.exp(la) * n ** (-al) + np.exp(lb) * P ** (-be) + c


def fit(n, P, y):
    """Least-squares fit of the law on the relative residual from several starts; eps_inf in [0, 0.9 x best]."""
    best = float(min(y))
    ub = 0.9 * best
    res = lambda th: law(th, n, P) / y - 1
    starts = [
        [0.0, 0.5, np.log(0.2), 0.05, 0.02],
        [np.log(3), 0.8, 0.0, 0.1, 0.03],
        [np.log(0.5), 0.3, np.log(0.1), 0.02, 0.0],
        [np.log(10), 1.2, np.log(100), 0.5, ub * 0.9],
    ]
    sols = [
        least_squares(
            res, s, bounds=([-20, 0, -20, 0, 0], [20, 3, 20, 3, ub]), max_nfev=20000
        )
        for s in starts
    ]
    r = min(sols, key=lambda s: s.cost)
    th = r.x
    return {
        "a": float(np.exp(th[0])),
        "alpha": float(th[1]),
        "b": float(np.exp(th[2])),
        "beta": float(th[3]),
        "eps_inf": float(th[4]),
        "eps_inf_upper_bound": ub,
        "eps_inf_on_bound": bool(ub - th[4] < 1e-4 * ub),
        "rms_rel_residual": float(np.sqrt(np.mean(res(th) ** 2))),
    }


fits: dict = {}
rng = np.random.default_rng(0)
for ds in ("drivaerml", "hilift"):
    for model in ("isla", "gt"):
        pts = [
            c
            for c in conf.values()
            if c["dataset"] == ds and c["model"] == model and c["status"] == "complete"
        ]
        f = {
            "n_points": len(pts),
            "distinct_n": len({c["n"] for c in pts}),
            "distinct_P": len({c["params"] for c in pts}),
        }
        # local exponents between adjacent rungs at each width, and the 15x parameter gain at each rung
        f["alpha_local_by_width"] = {}
        f["gain_widest_over_narrowest_by_n"] = {}
        for w in sorted({c["width"] for c in pts}):
            q = sorted([c for c in pts if c["width"] == w], key=lambda c: c["n"])
            f["alpha_local_by_width"][w] = {
                f"{a['n']}-{b['n']}": float(
                    -np.log(b["pressure_l2_mean"] / a["pressure_l2_mean"])
                    / np.log(b["n"] / a["n"])
                )
                for a, b in zip(q, q[1:])
            }
        ws = sorted({c["width"] for c in pts})
        if len(ws) >= 2:
            for nn in sorted({c["n"] for c in pts}):
                lo = [c for c in pts if c["n"] == nn and c["width"] == ws[0]]
                hi = [c for c in pts if c["n"] == nn and c["width"] == ws[-1]]
                if lo and hi:
                    f["gain_widest_over_narrowest_by_n"][nn] = float(
                        1 - hi[0]["pressure_l2_mean"] / lo[0]["pressure_l2_mean"]
                    )
        # non-monotone width at a rung (the separable law has no term for it)
        f["rungs_where_widest_is_not_best"] = []
        for nn in sorted({c["n"] for c in pts}):
            q = sorted([c for c in pts if c["n"] == nn], key=lambda c: c["width"])
            if (
                len(q) == len(ws)
                and min(q, key=lambda c: c["pressure_l2_mean"]) is not q[-1]
            ):
                f["rungs_where_widest_is_not_best"].append(
                    {
                        "n": nn,
                        "by_width": {c["width"]: c["pressure_l2_mean"] for c in q},
                    }
                )
        if f["distinct_n"] >= 4 and f["distinct_P"] >= 3:
            n = np.array([c["n"] for c in pts], float)
            P = np.array([c["params"] for c in pts], float)
            y = np.array([c["pressure_l2_mean"] for c in pts])
            f["fit"] = fit(n, P, y)
            boots = []
            for _ in range(400):
                yb = np.array(
                    [
                        c["pressure_l2_seeds"][
                            rng.integers(len(c["pressure_l2_seeds"]))
                        ]
                        for c in pts
                    ]
                )
                boots.append(fit(n, P, yb))
            f["bootstrap_p16_p50_p84"] = {
                k: [
                    float(x) for x in np.percentile([b[k] for b in boots], [16, 50, 84])
                ]
                for k in ("alpha", "beta", "eps_inf", "rms_rel_residual")
            }
            g = f["fit"]
            Pw = PARAMS[(model, RECIPE_WIDTH[model])]
            f["n_star_recipe_width"] = float(
                (g["a"] / (g["b"] * Pw ** (-g["beta"]))) ** (1 / g["alpha"])
            )
            f["law_accepted_rms_below_3pct"] = g["rms_rel_residual"] < 0.03
        fits[f"{ds}|{model}"] = f

bars = {}
for ds in ("drivaerml", "hilift"):
    fi, fg = fits.get(f"{ds}|isla", {}), fits.get(f"{ds}|gt", {})
    if "fit" in fi and "fit" in fg:
        same = {}
        for k in ("alpha", "beta"):
            spread = max(
                fi["bootstrap_p16_p50_p84"][k][2] - fi["bootstrap_p16_p50_p84"][k][0],
                fg["bootstrap_p16_p50_p84"][k][2] - fg["bootstrap_p16_p50_p84"][k][0],
            )
            same[k] = {
                "isla": fi["fit"][k],
                "gt": fg["fit"][k],
                "difference": abs(fi["fit"][k] - fg["fit"][k]),
                "bootstrap_spread": spread,
                "same": abs(fi["fit"][k] - fg["fit"][k]) < spread,
            }
        bars[ds] = {
            "law_accepted": {
                "isla": fi["law_accepted_rms_below_3pct"],
                "gt": fg["law_accepted_rms_below_3pct"],
                "rms": {
                    "isla": fi["fit"]["rms_rel_residual"],
                    "gt": fg["fit"]["rms_rel_residual"],
                },
            },
            "exponents_same": same,
            "eps_inf_on_bound": {
                "isla": fi["fit"]["eps_inf_on_bound"],
                "gt": fg["fit"]["eps_inf_on_bound"],
            },
            "n_star_recipe_width": {
                "isla": fi["n_star_recipe_width"],
                "gt": fg["n_star_recipe_width"],
            },
        }

out = {
    "study": "SCALE-LAW: surface-pressure error against training cases and parameters, cosine ISLA and GeoTransolver, DrivAerML (27/54/109/218 cars from the sl lanes; 435 from the routing and capacity ladders) and HiLiftAeroML (pending), 500 epochs, float32 at 100,000 cells, two seeds",
    "law": "eps(n, P) = eps_inf + a n^-alpha + b P^-beta; least squares on the relative residual of seed means; eps_inf in [0, 0.9 x best]; 400-draw two-seed bootstrap",
    "configurations": conf,
    "fits": fits,
    "bars": bars,
}
out_path.write_text(json.dumps(out, indent=2))
for ds in ("drivaerml", "hilift"):
    print(f"== {ds}")
    for key, c in sorted(
        conf.items(),
        key=lambda kv: (kv[1]["dataset"], kv[1]["model"], kv[1]["width"], kv[1]["n"]),
    ):
        if c["dataset"] != ds:
            continue
        tag = (
            f"{c['pressure_l2_mean']:.4f} ({' / '.join(f'{v:.4f}' for v in c['pressure_l2_seeds'])})"
            if c["pressure_l2_mean"]
            else "pending"
        )
        print(
            f"  {c['model']:4s} w{c['width']:<5d} n={c['n']:<4d} P={c['params'] / 1e6:6.1f}M  {tag}  [{c['status']}]"
        )
for k, f in fits.items():
    print(
        k,
        json.dumps(
            {kk: vv for kk, vv in f.items() if kk != "rungs_where_widest_is_not_best"},
            default=float,
        )[:1500],
    )
    print("   widest not best at:", f.get("rungs_where_widest_is_not_best"))
print(json.dumps(bars, indent=1))
