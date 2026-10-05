"""LUG-PREC (2026-10-05, #sec-nb-char-lug-final): the lug campaign lanes scored by the same scorer in bfloat16 (the training
precision) and float32 (the reporting protocol), per validation case, to separate a precision sensitivity of the
nearest-boundary-sample geometry from a configuration effect. Reads campaigns_2026-10-02/char/lug_prec/bfloat16/<run>/<run>/metrics.jsonl
and campaigns_2026-10-02/char/cl/metrics/<run>/<run>/metrics.jsonl (float32). Writes results/char/lug_prec_2026-10-05.json:
lanes[run] = {bf16, fp32, n_cases, median_abs_diff, frac_fp32_worse_1p5x, frac_fp32_better_1p5x, worst_case}.
usage: python lug_prec_2026-10-05.py"""
import json, os, statistics as st

R = os.path.dirname(os.path.abspath(__file__)); RES = os.path.dirname(R)
RUNS = {"cl_lug_isla_seed42": "ISLA 768 (first pass)", "cl_lug_isla_seed43": "ISLA 768 (first pass)",
        "cl_r_lug_isla_seed42": "ISLA 512 with 512 slices", "cl_r_lug_isla_seed43": "ISLA 512 with 512 slices",
        "cl_lug_gt_seed42": "GeoTransolver 1,024 (first pass)", "cl_r_lug_gt_seed42": "GeoTransolver 256",
        "cl_lug_t_seed42": "Transolver 1,024 (first pass)", "cl_r_lug_t_seed42": "Transolver 512 x 16"}


def per_case(p):
    out = {}; summ = None
    for l in open(p):
        r = json.loads(l)
        if r.get("phase") == "infer_step":
            out[r["sample_id"]] = r["metrics"]["von_mises_l2"]
        elif r.get("phase") == "infer_summary":
            summ = r["metrics"]["von_mises_l2"]
    return out, summ


res = {"protocol": __doc__, "lanes": {}}
for run, label in RUNS.items():
    pb, sb = per_case(f"{RES}/campaigns_2026-10-02/char/lug_prec/bfloat16/{run}/{run}/metrics.jsonl")
    pf, sf = per_case(f"{RES}/campaigns_2026-10-02/char/cl/metrics/{run}/{run}/metrics.jsonl")
    common = sorted(k for k in pb if k in pf)
    worst = max(common, key=lambda k: pf[k] - pb[k])
    res["lanes"][run] = {"label": label, "bf16": sb, "fp32": sf, "n_cases": len(common),
                         "median_abs_diff": st.median(abs(pf[k] - pb[k]) for k in common),
                         "frac_fp32_worse_1p5x": sum(pf[k] > 1.5 * pb[k] for k in common) / len(common),
                         "frac_fp32_better_1p5x": sum(pb[k] > 1.5 * pf[k] for k in common) / len(common),
                         "worst_case": {"id": worst, "fp32": pf[worst], "bf16": pb[worst]}}
    d = res["lanes"][run]
    print(f"{run:22s} {label:34s} bf16 {sb:.4f} fp32 {sf:.4f}  fp32 worse >1.5x: {d['frac_fp32_worse_1p5x']:.1%}  better >1.5x: {d['frac_fp32_better_1p5x']:.1%}  worst {worst[-4:]}: {d['worst_case']['fp32']:.4f} vs {d['worst_case']['bf16']:.4f}")
json.dump(res, open(f"{R}/lug_prec_2026-10-05.json", "w"), indent=1)
