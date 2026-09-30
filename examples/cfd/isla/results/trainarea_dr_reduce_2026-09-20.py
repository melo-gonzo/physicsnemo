"""TRAIN-AREA-DR reduction (prereg #sec-nb-trainarea-dr-prereg): the DrivAerML twin of the training-law test.

Six runs trained on 435 DrivAerML cars under the area-proportional Poisson draw (drivaer_ml_surface_area, expected
100,000 cells, exact Horvitz-Thompson weights) and their six uniform-trained ISO-COST twins at 100,000 cells, each scored
by the consistency benchmark on the 48 validation cars under the independent uniform (unif), stratified (strat) and
10:1-biased draws at 100,000 cells, float32, two seeds.

Bars (independent uniform draw, two-seed means, area-trained over uniform-trained per model): dependence reproduced =
GeoTransolver and Transolver >= 1.2x worse, ISLA within 3%; wing property = both baselines within 1.1x; ISLA moves =
ISLA more than 5% either way. Consistency: ISLA's density factor (biased / unif) within 0.95 to 1.3 under both laws.

usage: python trainarea_dr_reduce_2026-09-20.py <out.json> <fetched bench_evals dir>
"""
import glob, json, statistics as st, sys
from pathlib import Path

out_path, root = Path(sys.argv[1]), Path(sys.argv[2])
RUNS = {("isla", "uniform"): "iso_dr_isla_w192_c100k_seed{}", ("isla", "area"): "tadr_dr_isla_w192_area_c100k_seed{}",
        ("gt_unit", "uniform"): "iso_dr_gt_unit_w256_c100k_seed{}", ("gt_unit", "area"): "tadr_dr_gt_w256_area_c100k_seed{}",
        ("transolver_unit", "uniform"): "iso_dr_transolver_unit_w256_c100k_seed{}", ("transolver_unit", "area"): "tadr_dr_transolver_w256_area_c100k_seed{}"}
MODELS = ("isla", "gt_unit", "transolver_unit"); LAWS = ("uniform", "area"); DRAWS = ("unif", "strat", "biased")


def val(run, d):
    ps = glob.glob(str(root / run / f"{d}_100000" / "**" / "metrics.jsonl"), recursive=True)
    if not ps:
        return None
    rows = [json.loads(l) for l in open(ps[0]) if l.strip()]
    s = [r for r in rows if r.get("phase") == "infer_summary"]
    return s[0]["metrics"]["pressure_l2"] if s else None


res = {"study": "TRAIN-AREA-DR: 435 DrivAerML cars, runs trained under the uniform-over-cells law and under the area-proportional "
                "Poisson law, each scored under three benchmark draws at 100,000 cells, float32, two seeds", "arms": {}}
for (m, law), pat in RUNS.items():
    a = {"per_seed": {}, "mean": {}}
    for s in (42, 43):
        a["per_seed"][f"seed{s}"] = {d: val(pat.format(s), d) for d in DRAWS}
    for d in DRAWS:
        v = [a["per_seed"][k][d] for k in a["per_seed"] if a["per_seed"][k][d] is not None]
        a["mean"][d] = st.mean(v) if len(v) == 2 else None
        a[f"n_{d}"] = len(v)
    if a["mean"]["unif"] and a["mean"]["biased"]:
        a["density_factor"] = a["mean"]["biased"] / a["mean"]["unif"]
    res["arms"][f"{m}_{law}"] = a

v = {}
for d in DRAWS:
    for m in MODELS:
        u, ar = res["arms"][f"{m}_uniform"]["mean"][d], res["arms"][f"{m}_area"]["mean"][d]
        if u and ar:
            v[f"{m}_area_over_uniform_{d}"] = ar / u
    for law in LAWS:
        I = res["arms"][f"isla_{law}"]["mean"][d]
        for m in ("gt_unit", "transolver_unit"):
            M = res["arms"][f"{m}_{law}"]["mean"][d]
            if I and M:
                v[f"{m}_over_isla_{law}_{d}"] = M / I
k = ("isla_area_over_uniform_unif", "gt_unit_area_over_uniform_unif", "transolver_unit_area_over_uniform_unif")
if all(x in v for x in k):
    isla_move = abs(1 - v[k[0]]); base = [v[k[1]], v[k[2]]]
    if isla_move > 0.05:
        v["verdict"] = "ISLA moves (indifference not general)"
    elif min(base) >= 1.2 and isla_move <= 0.03:
        v["verdict"] = "dependence reproduced (baselines >= 1.2x, ISLA within 3%)"
    elif max(base) <= 1.1:
        v["verdict"] = "dependence is a wing property (baselines within 1.1x)"
    else:
        v["verdict"] = "between (baselines 1.1x to 1.2x, or one of the two above 1.2x, or ISLA 3% to 5%)"
    v["isla_change_unif"] = 1 - v[k[0]]
    v["baseline_area_over_uniform_unif"] = {"gt_unit": v[k[1]], "transolver_unit": v[k[2]]}
for law in LAWS:
    df = res["arms"][f"isla_{law}"].get("density_factor")
    if df:
        v[f"isla_{law}_density_factor"] = df
        v[f"consistency_{law}"] = "within 0.95-1.3" if 0.95 <= df <= 1.3 else "outside 0.95-1.3"
res["verdict"] = v
out_path.write_text(json.dumps(res, indent=2))
for arm, a in res["arms"].items():
    print(f"{arm:26s}", {d: (round(x, 4) if x else None) for d, x in a["mean"].items()}, "density", round(a["density_factor"], 2) if a.get("density_factor") else None)
print(json.dumps({k2: (round(x, 4) if isinstance(x, float) else x) for k2, x in v.items()}, indent=1))
