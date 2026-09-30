"""ISO-COST / ISO-MATCH reduction (prereg #sec-nb-isocost-prereg, #sec-nb-isomatch-prereg).

Inputs: the fleet manifest (group iso), the fetched float32 evaluations (<fetch>/<run>/**/metrics.jsonl), and the
per-lane cost table written on the cluster by kernel/isocost_cost.py (median step time after warm-up exclusion,
p10/p90, peak logged memory). Output: one JSON with per-configuration seed means, the training cost per
configuration, the two frontiers (iso-memory, iso-time), the ISO-MATCH comparison against GeoTransolver's
recipe default, the budget controls, and the per-model fits error = a*N^-alpha + b*P^-beta + c where the grid
supports them. Configurations with a missing seed are reported as pending and excluded from frontiers and fits.

Reporting rule for step time: the median over the run is the preregistered readout and is kept, but on this
hardware the ISLA lanes at 100k-200k cells, Transolver at 100k and GeoTransolver-256 at 100k all sit near the
same ~0.26 s median with a p10 half that, i.e. the median is set by the input pipeline, not the model. Both the
median and p10 are carried so the chapter can say which axis a comparison is on.

usage: python reduce_isocost.py <manifest.tsv> <fetched p200k_fp32 dir> <isocost_cost.tsv> <out.json>
"""
import csv, glob, json, re, statistics as st, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

manifest, root, cost_tsv, out_path = (Path(a) for a in sys.argv[1:5])
PARAMS = {("isla", 192): 8912796, ("isla", 288): 19532988, ("isla", 384): 34263516, ("isla", 512): 60298076,
          ("isla", 768): 134288988, ("gt_unit", 256): 9046016, ("gt_unit", 512): 35848576, ("transolver_unit", 256): 6021956}
GT_DEFAULT = {"runs": ["p200k_dr_gt_unit_lr3e3_seed42", "p200k_dr_gt_unit_lr3e3_seed43"], "peak_mem_gb": 79.51,
              "step_s_median": None, "note": "GeoTransolver recipe default: width 256, 12 layers, 200k cells, lr 3e-3; "
              "PROTO-200K lanes (RATE-GT pair); cost from ISO-COST acceptance (80 GB, 0.40 s) and the ep1000 control's log"}
T_DEFAULT_RUNS = ["p200k_dr_transolver_unit_seed42", "p200k_dr_transolver_unit_seed43"]
NAME = re.compile(r"iso_dr_(isla|gt_unit|transolver_unit)_w(\d+)_c(\d+)k(?:_lr(\d)e(\d))?(?:_ep(\d+))?_seed(\d+)")


def val(run):
    ps = glob.glob(str(root / run / "**" / "metrics.jsonl"), recursive=True)
    if not ps:
        return None
    rows = [json.loads(l) for l in open(ps[0]) if l.strip()]
    s = [r for r in rows if r.get("phase") == "infer_summary"]
    return {k: s[0]["metrics"][k] for k in ("pressure_l2", "wss_l2")} if s else None


cost = {r["run_id"]: r for r in csv.DictReader(open(cost_tsv), delimiter="\t")}
cfgs = defaultdict(lambda: {"runs": {}, "cost": {}})
for r in csv.DictReader(open(manifest), delimiter="\t"):
    if r["group"] != "iso":
        continue
    m = NAME.fullmatch(r["run_id"])
    model, w, c, lm, le, ep, seed = m.groups()
    WIDTH_RULE = {192: "1e-3", 288: "6.67e-4", 384: "5e-4", 512: "3.75e-4", 768: "2.5e-4"}
    lr = f"{lm}e-{le}" if lm else (WIDTH_RULE[int(w)] if model == "isla" else "3e-3")
    key = f"{model}|w{w}|c{c}k|lr{lr}|ep{ep or 500}"
    e = cfgs[key]
    e.update(model=model, width=int(w), cells=int(c) * 1000, lr=lr, epochs=int(ep or 500), params=PARAMS[(model, int(w))])
    e["runs"][r["run_id"]] = val(r["run_id"])
    cr = cost.get(r["run_id"], {})
    e["cost"][r["run_id"]] = {k: (float(cr[k]) if cr.get(k) not in (None, "") else None)
                              for k in ("step_s_median", "step_s_p10", "step_s_p90", "peak_mem_gb")} | {
        "epochs_logged": int(cr.get("epochs_logged", 0) or 0), "completed": bool(int(cr.get("completed", 0) or 0))}

# the recipe-default 200k lanes for GeoTransolver-256 and Transolver-256 come from PROTO-200K (same protocol, eager
# baselines are unaffected by the kernel); their cost is the acceptance reading / the ep1000 control's log
for model, runs, mem in (("gt_unit", GT_DEFAULT["runs"], 79.51), ("transolver_unit", T_DEFAULT_RUNS, None)):
    key = f"{model}|w256|c200k|lr3e-3|ep500"
    e = cfgs[key]
    e.update(model=model, width=256, cells=200000, lr="3e-3", epochs=500, params=PARAMS[(model, 256)], source="PROTO-200K lanes")
    for run in runs:
        e["runs"][run] = val(run)
        e["cost"][run] = {"step_s_median": None, "step_s_p10": None, "step_s_p90": None, "peak_mem_gb": mem, "epochs_logged": 500, "completed": True}
# GeoTransolver-256 at 200k: borrow the ep1000 control's per-step cost (same configuration, longer schedule)
ctl = [c for k, c in cfgs.items() if k.startswith("gt_unit|w256|c200k") and c["epochs"] == 1000]
if ctl:
    vals = [c for c in ctl[0]["cost"].values() if c["step_s_median"]]
    if vals:
        for run in GT_DEFAULT["runs"]:
            cfgs["gt_unit|w256|c200k|lr3e-3|ep500"]["cost"][run] = {**cfgs["gt_unit|w256|c200k|lr3e-3|ep500"]["cost"][run],
                                                                   "step_s_median": st.mean(v["step_s_median"] for v in vals),
                                                                   "step_s_p10": st.mean(v["step_s_p10"] for v in vals)}

out = {"study": "ISO-COST + ISO-MATCH: error as a function of cells per case and width at 435 DrivAerML cars, 500 epochs "
                "(1000-epoch budget controls), bf16 training on four GB300 GPUs, float32 scoring on the 48 validation cars at "
                "200,000 cells, two seeds; cost = per-lane median step time after warm-up exclusion and peak logged memory",
       "configurations": {}, "frontiers": {}, "isomatch": {}, "budget_controls": {}, "fits": {}, "bars": {}}
for key, e in sorted(cfgs.items()):
    got = {r: v for r, v in e["runs"].items() if v}
    d = {k: e[k] for k in ("model", "width", "cells", "lr", "epochs", "params")} | {"runs": e["runs"], "cost": e["cost"],
                                                                                    "n_evaluated": len(got), "n_runs": len(e["runs"])}
    if "source" in e:
        d["source"] = e["source"]
    if got and len(got) == len(e["runs"]):
        for f in ("pressure_l2", "wss_l2"):
            d[f"{f}_seeds"] = [v[f] for v in got.values()]
            d[f"{f}_mean"] = st.mean(d[f"{f}_seeds"])
        d["status"] = "complete"
    else:
        d["status"] = "pending"
    for k in ("step_s_median", "step_s_p10", "peak_mem_gb"):
        vals = [c[k] for c in e["cost"].values() if c.get(k)]
        d[k] = st.mean(vals) if vals else None
    out["configurations"][key] = d

done = {k: d for k, d in out["configurations"].items() if d["status"] == "complete" and d["epochs"] == 500}


def better_rate(model, width, cells):
    """GeoTransolver-512 keeps its better learning rate (prereg)."""
    cands = [d for d in done.values() if d["model"] == model and d["width"] == width and d["cells"] == cells]
    return min(cands, key=lambda d: d["pressure_l2_mean"]) if cands else None


grid = {}
for d in done.values():
    b = better_rate(d["model"], d["width"], d["cells"])
    if b is d:
        grid[f"{d['model']}|w{d['width']}|c{d['cells'] // 1000}k"] = d

# frontiers: at each budget spanned by the grid, the best complete lane of each model whose measured cost fits
for axis, label in (("peak_mem_gb", "iso_memory"), ("step_s_median", "iso_time_median"), ("step_s_p10", "iso_time_p10")):
    pts = [d for d in grid.values() if d.get(axis)]
    budgets = sorted({round(d[axis], 3) for d in pts})
    rows = []
    for B in budgets:
        row = {"budget": B}
        for model in ("isla", "gt_unit", "transolver_unit"):
            fit = [d for d in pts if d["model"] == model and d[axis] <= B * 1.0001]
            if fit:
                best = min(fit, key=lambda d: d["pressure_l2_mean"])
                row[model] = {"pressure_l2": best["pressure_l2_mean"], "config": f"w{best['width']} x {best['cells'] // 1000}k"}
        if "isla" in row and "gt_unit" in row:
            row["gt_over_isla"] = row["gt_unit"]["pressure_l2"] / row["isla"]["pressure_l2"]
        rows.append(row)
    out["frontiers"][label] = {"axis": axis, "rows": rows}

# ISO-MATCH: ISLA lanes within 15% of GeoTransolver's default on each axis; best against the default pair
gtd = out["configurations"].get("gt_unit|w256|c200k|lr3e-3|ep500")
if gtd and gtd["status"] == "complete":
    ref = gtd["pressure_l2_mean"]
    for axis, budget in (("peak_mem_gb", 79.51), ("step_s_median", gtd.get("step_s_median")), ("step_s_p10", gtd.get("step_s_p10"))):
        if not budget:
            continue
        within = [d for d in grid.values() if d["model"] == "isla" and d.get(axis) and abs(d[axis] - budget) / budget <= 0.15]
        under = [d for d in grid.values() if d["model"] == "isla" and d.get(axis) and d[axis] <= budget * 1.15]
        entry = {"budget": budget, "gt_default_pressure_l2": ref,
                 "isla_within_15pct": sorted(({"config": f"w{d['width']} x {d['cells'] // 1000}k", axis: d[axis],
                                               "pressure_l2": d["pressure_l2_mean"], "gt_over_isla": ref / d["pressure_l2_mean"]}
                                              for d in within), key=lambda r: r["pressure_l2"]),
                 "isla_at_or_under_budget_best": None}
        if under:
            b = min(under, key=lambda d: d["pressure_l2_mean"])
            entry["isla_at_or_under_budget_best"] = {"config": f"w{b['width']} x {b['cells'] // 1000}k", axis: b[axis],
                                                     "pressure_l2": b["pressure_l2_mean"], "gt_over_isla": ref / b["pressure_l2_mean"]}
        out["isomatch"][axis] = entry
    splits = [d for d in grid.values() if d["model"] == "isla" and (d["width"], d["cells"]) in
              {(192, 600000), (288, 400000), (384, 400000), (512, 200000), (768, 100000), (512, 300000), (768, 200000)}]
    out["isomatch"]["splits_ranked"] = sorted(({"config": f"w{d['width']} x {d['cells'] // 1000}k", "pressure_l2": d["pressure_l2_mean"],
                                                "peak_mem_gb": d.get("peak_mem_gb"), "step_s_median": d.get("step_s_median")}
                                               for d in splits), key=lambda r: r["pressure_l2"])

# budget controls
for model in ("isla", "gt_unit"):
    c500 = grid.get(f"{model}|w{192 if model == 'isla' else 256}|c200k")
    c1000 = [d for d in out["configurations"].values() if d["model"] == model and d["epochs"] == 1000 and d["status"] == "complete"]
    if c500 and c1000:
        out["budget_controls"][model] = {"ep500": c500["pressure_l2_mean"], "ep1000": c1000[0]["pressure_l2_mean"],
                                         "ep1000_over_ep500": c1000[0]["pressure_l2_mean"] / c500["pressure_l2_mean"]}

# fits: error = a N^-alpha + b P^-beta + c per model on seed means (needs >= 5 distinct (N, P) points)
try:
    from scipy.optimize import curve_fit

    def f(X, a, al, b, be, c):
        N, P = X
        return a * (N / 1e5) ** (-al) + b * (P / 1e7) ** (-be) + c

    for model in ("isla", "gt_unit"):
        pts = [d for d in grid.values() if d["model"] == model]
        if len(pts) >= 5:
            N = np.array([d["cells"] for d in pts], float); P = np.array([d["params"] for d in pts], float)
            y = np.array([d["pressure_l2_mean"] for d in pts])
            try:
                p, cov = curve_fit(f, (N, P), y, p0=(0.02, 0.3, 0.005, 0.3, 0.03), bounds=([0, 0, 0, 0, 0], [1, 2, 1, 2, 1]), maxfev=20000)
                resid = y - f((N, P), *p)
                out["fits"][model] = {"a": p[0], "alpha": p[1], "b": p[2], "beta": p[3], "c": p[4], "n_points": len(pts),
                                      "rms_resid": float(np.sqrt(np.mean(resid ** 2))), "alpha_se": float(np.sqrt(cov[1, 1])) if np.isfinite(cov[1, 1]) else None,
                                      "note": "N in units of 1e5 cells, P in units of 1e7 parameters"}
            except Exception as ex:  # noqa: BLE001
                out["fits"][model] = {"error": str(ex), "n_points": len(pts)}
        else:
            out["fits"][model] = {"status": f"pending: {len(pts)} complete points, 5 needed"}
except ImportError:
    out["fits"]["error"] = "scipy unavailable"

# bars (evaluated only where the inputs are complete)
fr = out["frontiers"].get("iso_memory", {}).get("rows", [])
both = [r for r in fr if "gt_over_isla" in r]
if both:
    worst = min(both, key=lambda r: r["gt_over_isla"])
    out["bars"]["iso_memory_frontier"] = {"min_gt_over_isla": worst["gt_over_isla"], "at_budget_gb": worst["budget"],
                                          "verdict": "ISLA at or below at every budget (within 2%)" if worst["gt_over_isla"] >= 0.98 else "GeoTransolver lower by more than 2% at some budget"}
for lab in ("iso_time_median", "iso_time_p10"):
    fr = out["frontiers"].get(lab, {}).get("rows", [])
    both = [r for r in fr if "gt_over_isla" in r]
    if both:
        worst = min(both, key=lambda r: r["gt_over_isla"])
        out["bars"][f"{lab}_frontier"] = {"min_gt_over_isla": worst["gt_over_isla"], "at_budget_s": worst["budget"],
                                          "verdict": "within 3% at every budget" if worst["gt_over_isla"] >= 0.97 else "GeoTransolver lower by more than 3% at some budget"}
if "isla" in out["fits"] and "gt_unit" in out["fits"] and "alpha" in out["fits"]["isla"] and "alpha" in out["fits"]["gt_unit"]:
    da = out["fits"]["isla"]["alpha"] - out["fits"]["gt_unit"]["alpha"]
    out["bars"]["cells_exponent"] = {"alpha_isla": out["fits"]["isla"]["alpha"], "alpha_gt": out["fits"]["gt_unit"]["alpha"], "difference": da,
                                     "verdict": "ISLA converts cells faster (>= 0.1)" if da >= 0.1 else "same cells response (< 0.1)"}
w192, w384 = grid.get("isla|w192|c200k"), grid.get("isla|w384|c200k")
if w192 and w384:
    out["bars"]["width_384_vs_192_at_200k"] = {"w192": w192["pressure_l2_mean"], "w384": w384["pressure_l2_mean"], "ratio_384_over_192": w384["pressure_l2_mean"] / w192["pressure_l2_mean"],
                                              "verdict": "384 improves by >= 3%" if w384["pressure_l2_mean"] <= 0.97 * w192["pressure_l2_mean"] else "no >= 3% gain from width at 200k"}
for model, bc in out["budget_controls"].items():
    bc["verdict"] = "1000 epochs improve by > 2% (budget binds)" if bc["ep1000_over_ep500"] < 0.98 else "within 2% (500-epoch protocol converged)"
if "peak_mem_gb" in out["isomatch"]:
    for axis in ("peak_mem_gb", "step_s_median", "step_s_p10"):
        e = out["isomatch"].get(axis)
        if e and e["isla_within_15pct"]:
            b = e["isla_within_15pct"][0]
            out["bars"][f"isomatch_{axis}"] = {"best_isla_within_15pct": b, "verdict": "lead >= 3%" if b["gt_over_isla"] >= 1.03 else ("deficit > 3%" if b["gt_over_isla"] <= 0.97 else "parity within 3%")}

out_path.write_text(json.dumps(out, indent=2))
print(f"{sum(d['status'] == 'complete' for d in out['configurations'].values())}/{len(out['configurations'])} configurations complete")
for k, d in sorted(out["configurations"].items()):
    tag = f"{d['pressure_l2_mean']:.4f}" if d["status"] == "complete" else f"pending {d['n_evaluated']}/{d['n_runs']}"
    cost_s = f"mem {d['peak_mem_gb']:.1f} GB  step med {d['step_s_median']:.3f} p10 {d['step_s_p10']:.3f} s" if d.get("step_s_median") and d.get("peak_mem_gb") else ""
    print(f"  {k:42s} {tag:16s} {cost_s}")
for lab, fr in out["frontiers"].items():
    print(lab, [(r["budget"], round(r["gt_over_isla"], 3)) for r in fr["rows"] if "gt_over_isla" in r])
print("isomatch:", json.dumps({k: v for k, v in out["isomatch"].items() if k != "splits_ranked"}, indent=None)[:600])
print("splits:", [(s["config"], round(s["pressure_l2"], 4)) for s in out["isomatch"].get("splits_ranked", [])])
print("budget:", out["budget_controls"]); print("fits:", {m: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items()} for m, f in out["fits"].items()})
print("bars:", json.dumps(out["bars"], indent=1)[:1500])
