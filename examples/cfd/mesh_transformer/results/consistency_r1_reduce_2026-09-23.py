"""Consistency factors of the reference configuration R1 (2026-09-23, REF-BOARD inference half,
#sec-nb-refboard-prereg): the four R1 checkpoints (ref_isla_r1_dr435_seed{42,43}, ref_isla_r1_hl35_seed{42,43}) scored
by the consistency benchmark's three draws (independent uniform `unif`, stratified Morton-systematic `strat`, 10:1
front/back Poisson-biased `biased`; every draw from the 320,000-cell pool with exact Horvitz-Thompson weights; float32;
the validation split) at 200,000 cells, and the cars also at 100,000 cells (the count of the 192-wide originals and of
the GeoTransolver-256 rows of results/trainarea_dr_reduction_2026-09-20.json). Factors: biased / unif and strat / unif
per seed and on the seed means, as in the TRAIN-AREA-DR and WING-BENCH-200K reductions.

Comparators are read from the existing artifacts (not re-run): the 192-wide ISLA and GeoTransolver-256 rows of
trainarea_dr_reduction_2026-09-20.json (cars, 100,000 cells) and wingbench_200k_reduction_2026-09-19.json (35 wings,
200,000 cells). Uniform references: the REF fleet scores (results/campaigns_2026-09-22/ref/metrics; the recipe's own
uniform draw of the full mesh at 200,000 cells, not the bench pool).

Bar (stated in the REF-BOARD entry): R1's factors within 0.03 of the 192-wide lanes' on the same dataset; GeoTransolver's
factors are quoted unchanged (released baseline, not re-run).

usage: python consistency_r1_reduce_2026-09-23.py <out.json> [<metrics mirror root>]
       (mirror layout: <root>/<run>/<sampler>_<count>/<run>/metrics.jsonl, as written by kernel/rp_bench.sbatch)
"""
from __future__ import annotations

import json, statistics as st, sys
from pathlib import Path

RES = Path(__file__).resolve().parent
out_path = Path(sys.argv[1])
root = Path(sys.argv[2]) if len(sys.argv) > 2 else RES / "campaigns_2026-09-23" / "rp" / "metrics"
DRAWS = ("unif", "strat", "biased")
BAR = 0.03


def summary(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    s = [r for r in rows if r.get("phase") == "infer_summary"]
    return (s[0]["metrics"]["pressure_l2"], s[0]["num_samples"]) if s else (None, None)


def val(run, draw, count):
    p = root / run / f"{draw}_{count}" / run / "metrics.jsonl"
    return summary(p) if p.exists() else (None, None)


def ref_uniform(run):
    return summary(RES / "campaigns_2026-09-22" / "ref" / "metrics" / run / run / "metrics.jsonl")[0]


def factors(mean):
    return {"biased_over_unif": mean["biased"] / mean["unif"], "strat_over_unif": mean["strat"] / mean["unif"]}


ARMS = {
    "isla_r1_dr435_200k": dict(dataset="DrivAerML Surface, 435 training cars, 48 validation cars", runs="ref_isla_r1_dr435_seed{}", count=200000,
                               comparator_192="isla_192_dr435_100k", comparator_gt="gt256_dr435_100k"),
    "isla_r1_dr435_100k": dict(dataset="DrivAerML Surface, 435 training cars, 48 validation cars", runs="ref_isla_r1_dr435_seed{}", count=100000,
                               comparator_192="isla_192_dr435_100k", comparator_gt="gt256_dr435_100k"),
    "isla_r1_hl35_200k": dict(dataset="HiLiftAeroML, 35 training cases, 180 validation cases", runs="ref_isla_r1_hl35_seed{}", count=200000,
                              comparator_192="isla_192_hl35_200k", comparator_gt="gt256_hl35_200k"),
}

res = {"study": "REF-BOARD inference half: consistency factors of the reference configuration R1 (384 x 12, cosine routing, fused kernel, 5e-4) "
                "under the consistency benchmark's independent uniform, stratified and 10:1-biased draws, float32, two seeds",
       "config": "R1",
       "protocol": {"snapshot": "code_ref", "recipe": "recipe_globin + the six bench dataset yamls and StratifiedSubsampleMesh (copy $HOME/rp_probes/recipe_rp)",
                    "pool_cells": 320000, "draw_seed": "recipe default 42 for every draw (as the original bench lanes)", "precision": "float32",
                    "biased": "PoissonBiasedSubsampleMesh bias=10 (10:1 front/back on centroid x), exact HT weights",
                    "strat": "StratifiedSubsampleMesh (Morton-ordered systematic sample, inclusion n/N, weights N/n)"},
       "arms": {}, "comparators": {}, "verdict": {}}

for name, a in ARMS.items():
    arm = {"dataset": a["dataset"], "cells": a["count"], "per_seed": {}, "mean": {}, "n_cases": None, "ref_uniform_reference": {}}
    for s in (42, 43):
        run = a["runs"].format(s)
        vals = {}
        for d in DRAWS:
            v, n = val(run, d, a["count"])
            vals[d] = v
            if n:
                arm["n_cases"] = n
        arm["per_seed"][f"seed{s}"] = vals
        arm["ref_uniform_reference"][f"seed{s}"] = ref_uniform(run)
    for d in DRAWS:
        v = [arm["per_seed"][k][d] for k in arm["per_seed"] if arm["per_seed"][k][d] is not None]
        arm["mean"][d] = st.mean(v) if len(v) == 2 else None
        arm[f"n_{d}"] = len(v)
    if all(arm["mean"][d] for d in DRAWS):
        arm.update(factors(arm["mean"]))
        arm["density_factor"] = arm["biased_over_unif"]
        arm["per_seed_factors"] = {k: factors(v) for k, v in arm["per_seed"].items() if all(v[d] for d in DRAWS)}
        arm["ref_uniform_reference"]["mean"] = st.mean(arm["ref_uniform_reference"].values())
        arm["bench_unif_over_ref_uniform"] = arm["mean"]["unif"] / arm["ref_uniform_reference"]["mean"]
    res["arms"][name] = arm

TC = json.load(open(RES / "trainarea_dr_reduction_2026-09-20.json"))["arms"]
WB = json.load(open(RES / "wingbench_200k_reduction_2026-09-19.json"))["arms"]
for name, src, arm, label in (("isla_192_dr435_100k", "trainarea_dr_reduction_2026-09-20.json:arms.isla_uniform", TC["isla_uniform"], "ISLA, 192 wide, linear routing, 1e-3 (iso_dr_isla_w192_c100k)"),
                              ("gt256_dr435_100k", "trainarea_dr_reduction_2026-09-20.json:arms.gt_unit_uniform", TC["gt_unit_uniform"], "GeoTransolver-256, 3e-3 (iso_dr_gt_unit_w256_c100k)"),
                              ("transolver256_dr435_100k", "trainarea_dr_reduction_2026-09-20.json:arms.transolver_unit_uniform", TC["transolver_unit_uniform"], "Transolver-256, 3e-3 (iso_dr_transolver_unit_w256_c100k)"),
                              ("isla_192_hl35_200k", "wingbench_200k_reduction_2026-09-19.json:arms.isla", WB["isla"], "ISLA, 192 wide, linear routing, 1e-3 (p200k_hl_isla_super_scarce)"),
                              ("gt256_hl35_200k", "wingbench_200k_reduction_2026-09-19.json:arms.gt_unit_lr3e3", WB["gt_unit_lr3e3"], "GeoTransolver-256, 3e-3 (p200k_hl_gt_unit_super_scarce)"),
                              ("transolver256_hl35_200k", "wingbench_200k_reduction_2026-09-19.json:arms.transolver_unit", WB["transolver_unit"], "Transolver-256, 3e-3 (p200k_hl_transolver_unit_super_scarce)")):
    c = {"source": src, "label": label, "cells": 100000 if "dr435" in name else 200000, "mean": arm["mean"], "per_seed": arm["per_seed"]}
    c.update(factors(arm["mean"]))
    c["per_seed_factors"] = {k: factors(v) for k, v in arm["per_seed"].items()}
    res["comparators"][name] = c

v = {"bar": f"R1's biased/unif and strat/unif factors within {BAR} of the 192-wide lane's on the same dataset (GeoTransolver's quoted unchanged)"}
ok = []
for name, a in ARMS.items():
    arm = res["arms"][name]
    if "biased_over_unif" not in arm:
        v[name] = "pending"
        continue
    c = res["comparators"][a["comparator_192"]]
    g = res["comparators"][a["comparator_gt"]]
    row = {"biased_over_unif": arm["biased_over_unif"], "strat_over_unif": arm["strat_over_unif"],
           "delta_biased_vs_192": arm["biased_over_unif"] - c["biased_over_unif"], "delta_strat_vs_192": arm["strat_over_unif"] - c["strat_over_unif"],
           "gt256_biased_over_unif": g["biased_over_unif"], "gt256_strat_over_unif": g["strat_over_unif"],
           "gt256_factor_over_r1_factor_biased": g["biased_over_unif"] / arm["biased_over_unif"],
           "same_count_as_comparators": arm["cells"] == c["cells"]}
    row["within_bar"] = abs(row["delta_biased_vs_192"]) <= BAR and abs(row["delta_strat_vs_192"]) <= BAR
    ok.append(row["within_bar"])
    v[name] = row
v["verdict"] = ("bar met on every arm" if ok and all(ok) else "bar missed on at least one arm" if ok else "pending")
res["verdict"] = v
out_path.write_text(json.dumps(res, indent=2))
for name, arm in res["arms"].items():
    print(f"{name:22s} n={arm['n_cases']} ", {d: (round(x, 5) if x else None) for d, x in arm["mean"].items()},
          "biased/unif", round(arm["biased_over_unif"], 4) if "biased_over_unif" in arm else None,
          "strat/unif", round(arm["strat_over_unif"], 4) if "strat_over_unif" in arm else None,
          "unif/REF", round(arm["bench_unif_over_ref_uniform"], 4) if "bench_unif_over_ref_uniform" in arm else None)
    for k, f in arm.get("per_seed_factors", {}).items():
        print(f"   {k}: biased/unif {f['biased_over_unif']:.4f}  strat/unif {f['strat_over_unif']:.4f}  unif {arm['per_seed'][k]['unif']:.5f} (REF {arm['ref_uniform_reference'][k]:.5f})")
for name, c in res["comparators"].items():
    print(f"{name:26s} biased/unif {c['biased_over_unif']:.4f}  strat/unif {c['strat_over_unif']:.4f}  unif {c['mean']['unif']:.5f}")
print(json.dumps(v, indent=1, default=str))
