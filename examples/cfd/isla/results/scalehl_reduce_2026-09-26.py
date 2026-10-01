"""SCALE-HL reduction (prereg #sec-nb-scalehl-prereg): the DrivAerML scaling picture on the 510-wing HiLift rung.

Inputs: results/campaigns_2026-09-20/shl/metrics/<run>/metrics.jsonl (float32 fleet scores at 200,000 cells on the 180
validation wings) for the shl lanes (100,000 training cells: ISLA linear-routed 192/384/768 at the width rule,
GeoTransolver 256/512 at 3e-3, Transolver 256 at 3e-3), the fleet's 200,000-cell medium-rung lanes (ISLA 192 at 1e-3,
GeoTransolver 256 at 1e-3 and 3e-3, Transolver 256), and the ROUTE-NORM 510-wing lanes (cosine 192 and 384 at 1e-3,
linear 384 at 1e-3, 100,000 cells) read beside, and the campaign sc cosine-192 twin at 200,000 cells (2026-10-01) that
decides the ISLA cells clause after the linear 200,000-cell arm was voided by a plateaued seed; results/campaigns_2026-09-20/shl/cost_2026-09-26.tsv (median step
time and peak memory from each train.log). Bars are the preregistration's, on seed-mean pressure relative L2.
usage: python scalehl_reduce_2026-09-26.py [out.json]
"""
import csv, glob, json, os, statistics as st, sys
R = os.path.dirname(os.path.abspath(__file__)); M = f"{R}/campaigns_2026-09-20/shl/metrics"
FIELDS = ["pressure_l2", "temperature_l2", "density_l2", "velocity_l2", "tau_wall_l2"]
def lane(run):
    p = f"{M}/{run}/metrics.jsonl"
    if not os.path.exists(p): return None
    rows = [json.loads(l)["metrics"] for l in open(p) if '"infer_step"' in l]
    keys = [k for k in FIELDS if k in rows[0]]
    return {k: st.mean(r[k] for r in rows) for k in keys} | {"n_cases": len(rows)}
cost = {r["run_id"]: r for r in csv.DictReader(open(f"{R}/campaigns_2026-09-20/shl/cost_2026-09-26.tsv"), delimiter="\t")}
ARMS = {
 "isla_192_c100k": "shl_med_isla_w192_c100k_seed{}", "isla_384_c100k": "shl_med_isla_w384_c100k_seed{}", "isla_768_c100k": "shl_med_isla_w768_c100k_seed{}",
 "gt_256_c100k": "shl_med_gt_w256_c100k_seed{}", "gt_512_c100k": "shl_med_gt_w512_c100k_seed{}", "transolver_256_c100k": "shl_med_transolver_w256_c100k_seed{}",
 "isla_192_c200k": "p200k_hl_isla_medium_seed{}", "gt_256_c200k_lr1e3": "p200k_hl_gt_unit_medium_seed{}", "gt_256_c200k_lr3e3": "p200k_hl_gt_unit_medium_lr3e3_seed{}", "transolver_256_c200k": "p200k_hl_transolver_unit_medium_seed{}",
 "isla_cos_192_c100k": "rn_cos_w192_lr1e_3_med_c100k_seed{}", "isla_cos_192_c200k": "sc_hl_cos_w192_med_c200k_seed{}", "isla_cos_384_lr1e3_c100k": "rn_cos_w384_lr1e_3_med_c100k_seed{}", "isla_lin_384_lr1e3_c100k": "rn_lin_w384_lr1e_3_med_c100k_seed{}",
}
out = {"protocol": __doc__.strip(), "arms": {}}
for a, pat in ARMS.items():
    per = {s: lane(pat.format(s)) for s in (42, 43)}
    per = {s: v for s, v in per.items() if v}
    if not per: continue
    keys = [k for k in FIELDS if k in next(iter(per.values()))]
    c = [cost.get(pat.format(s)) for s in (42, 43)]; c = [x for x in c if x and x["step_s_median"]]
    out["arms"][a] = {"runs": {pat.format(s): per[s] for s in per}, "n_seeds": len(per),
        "seed_means": {k: st.mean(per[s][k] for s in per) for k in keys},
        "pressure_l2_seeds": [per[s]["pressure_l2"] for s in sorted(per)],
        "step_s_median": st.mean(float(x["step_s_median"]) for x in c) if c else None,
        "mem_peak_gb": max(float(x["mem_peak_gb"]) for x in c) if c else None}
A = out["arms"]; P = lambda a: A[a]["seed_means"]["pressure_l2"]
bars = {}
def ratio(a, b): return P(a) / P(b) if a in A and b in A else None
bars["cells_flat"] = {"isla_192_c100k_over_c200k": ratio("isla_192_c100k", "isla_192_c200k"), "isla_192_c100k_over_c200k_void": "the linear 200k arm averages a plateaued seed (p200k_hl_isla_medium_seed42, 0.594); void, see #sec-nb-p200k-hl-ladder-verdict",
                      "isla_cos_192_c100k_over_c200k": ratio("isla_cos_192_c100k", "isla_cos_192_c200k"), "gt_256_c100k_over_c200k_lr3e3": ratio("gt_256_c100k", "gt_256_c200k_lr3e3"),
                      "gt_256_c100k_over_c200k_lr1e3": ratio("gt_256_c100k", "gt_256_c200k_lr1e3"), "transolver_256_c100k_over_c200k": ratio("transolver_256_c100k", "transolver_256_c200k"), "bar": "within 2% (0.98 to 1.02) for ISLA-192 and GeoTransolver-256"}
bars["width_carries"] = {"isla_384_over_192": ratio("isla_384_c100k", "isla_192_c100k"), "gt_512_over_256": ratio("gt_512_c100k", "gt_256_c100k"), "bar": "at most 0.97 for each model (3% gain); within 3% = not parameter-bound"}
bars["turnover"] = {"isla_768_over_384": ratio("isla_768_c100k", "isla_384_c100k"), "bar": "gain under 1% (ratio above 0.99) or a loss reproduces the DrivAerML turnover"}
d_default = (P("gt_256_c100k") / P("isla_192_c100k")) if "gt_256_c100k" in A else None
d_matched = (P("gt_512_c100k") / P("isla_384_c100k")) if "gt_512_c100k" in A else None
bars["deficit_under_width"] = {"gt256_over_isla192": d_default, "gt512_over_isla384": d_matched, "change": (d_matched - d_default) if d_default and d_matched else None, "bar": "+0.05 or more: width closes part of the deficit; -0.05 or less: widens; within 0.05: width-independent"}
bars["iso_memory"] = {"gt_512_mem_gb": A["gt_512_c100k"]["mem_peak_gb"] if "gt_512_c100k" in A else None,
    "isla_best_under_gt512_memory": min(((a, P(a), A[a]["mem_peak_gb"]) for a in A if a.startswith("isla_") and A[a]["mem_peak_gb"] and "gt_512_c100k" in A and A[a]["mem_peak_gb"] <= A["gt_512_c100k"]["mem_peak_gb"]), key=lambda t: t[1], default=None),
    "gt_best_under_gt512_memory": min(((a, P(a), A[a]["mem_peak_gb"]) for a in A if a.startswith("gt_") and A[a]["mem_peak_gb"] and A[a]["mem_peak_gb"] <= (A["gt_512_c100k"]["mem_peak_gb"] if "gt_512_c100k" in A else 1e9)), key=lambda t: t[1], default=None)}
bars["routing_beside"] = {"cos_192_over_lin_192": ratio("isla_cos_192_c100k", "isla_192_c100k"), "cos_384_lr1e3_over_lin_384_lr5e4": ratio("isla_cos_384_lr1e3_c100k", "isla_384_c100k"), "lin_384_lr1e3_over_lin_384_lr5e4": ratio("isla_lin_384_lr1e3_c100k", "isla_384_c100k")}
out["bars"] = bars
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else f"{R}/scalehl_2026-09-26.json", "w"), indent=1)
for a in A: print(f"{a:28s} p={P(a):.4f} seeds={[round(x,4) for x in A[a]['pressure_l2_seeds']]} step={A[a]['step_s_median']} mem={A[a]['mem_peak_gb']}")
print(json.dumps(bars, indent=0))
