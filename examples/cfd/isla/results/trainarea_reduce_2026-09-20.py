"""TRAIN-AREA reduction (prereg #sec-nb-trainarea-prereg): the six 35-wing runs trained on the area-proportional
Poisson draw, scored under the consistency benchmark's independent uniform (unif), stratified (strat) and 10:1-biased
draws at 200,000 cells, float32, 180 HiLift validation wings; compared model by model with the uniform-draw-trained
runs of WING-BENCH-200K under the same three draws.

Bars (independent uniform draw, two-seed means): mechanism supported = ISLA improves >= 5% while GeoTransolver and
Transolver move < 2%; sampler effect = all three improve >= 5%; null = ISLA moves < 5%. Consistency: ISLA's density
factor (biased / unif) within 1.05 to 1.3; above 1.5 rejects the area draw.

usage: python reduce_trainarea.py <out.json> <fetched bench_evals dir>
"""
import glob, json, statistics as st, sys
from pathlib import Path

out_path, root = Path(sys.argv[1]), Path(sys.argv[2])
MODELS = {"isla": "p200k_hl_isla_super_scarce{}_seed{}", "gt_unit": "p200k_hl_gt_unit_super_scarce{}_seed{}",
          "transolver_unit": "p200k_hl_transolver_unit_super_scarce{}_seed{}"}
LAWS = {"uniform": "", "area": "_area"}
DRAWS = ("unif", "strat", "biased")


def val(run, d):
    ps = glob.glob(str(root / run / f"{d}_200000" / "**" / "metrics.jsonl"), recursive=True)
    if not ps:
        return None
    rows = [json.loads(l) for l in open(ps[0]) if l.strip()]
    s = [r for r in rows if r.get("phase") == "infer_summary"]
    return s[0]["metrics"]["pressure_l2"] if s else None


res = {"study": "TRAIN-AREA: 35-wing rung, runs trained under the uniform-over-cells law and under the area-proportional "
                "Poisson law, each scored under three benchmark draws at 200,000 cells, float32, two seeds", "arms": {}}
for m, pat in MODELS.items():
    for law, infix in LAWS.items():
        a = {"per_seed": {}, "mean": {}}
        for s in (42, 43):
            a["per_seed"][f"seed{s}"] = {d: val(pat.format(infix, s), d) for d in DRAWS}
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
    isla_gain = 1 - v[k[0]]
    base_moves = [abs(1 - v[k[1]]), abs(1 - v[k[2]])]
    base_gains = [1 - v[k[1]], 1 - v[k[2]]]
    if isla_gain >= 0.05 and max(base_moves) < 0.02:
        v["verdict"] = "mechanism supported"
    elif isla_gain >= 0.05 and min(base_gains) >= 0.05:
        v["verdict"] = "sampler effect, not mechanism"
    elif isla_gain < 0.05:
        v["verdict"] = "null"
    else:
        v["verdict"] = "between (ISLA >= 5%, baselines moved >= 2% but not all improved >= 5%)"
    v["isla_gain_unif"] = isla_gain
    v["baseline_changes_unif"] = {"gt_unit": 1 - v[k[1]], "transolver_unit": 1 - v[k[2]]}
df = res["arms"]["isla_area"].get("density_factor")
if df:
    v["isla_area_density_factor"] = df
    v["consistency"] = "within 1.05-1.3" if 1.05 <= df <= 1.3 else ("rejected (> 1.5)" if df > 1.5 else "outside 1.05-1.3 but below 1.5")
res["verdict"] = v
out_path.write_text(json.dumps(res, indent=2))
for arm, a in res["arms"].items():
    print(f"{arm:26s}", {d: (round(x, 4) if x else None) for d, x in a["mean"].items()}, "density", round(a["density_factor"], 2) if a.get("density_factor") else None)
print(json.dumps({k2: (round(x, 4) if isinstance(x, float) else x) for k2, x in v.items()}, indent=1))
