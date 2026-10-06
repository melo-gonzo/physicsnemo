"""QUERY-POP (2026-10-06): does the Volume model's prediction at a point depend on the population of prediction
points it is scored with? Reduces kernel/char/qpop/<run>/summary.json (copied to results/campaigns_2026-10-02/char/qpop/)
for the DrivAerML Volume references cl_r_vol_isla_seed42/43 (48 validation cars, float32, 200,000 native points):
every metric on the SAME scored set (the first 100,000 native points); conditions differ only in the query population:
native_h (the native 200,000 tokens), half (the 100,000 alone), box (+200,000 companions uniform in the interior's
bounding box, far-field heavy), wall (+200,000 companions resampled from the native points with probability 1/|sdf|,
near-wall heavy); full = native 200,000 scored on all (reproduces evals_char native). Also the learned query routing
mass exp(qt.logw) as a fraction of the boundary measure. Writes qpop_2026-10-06.json next to this file."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "campaigns_2026-10-02" / "char" / "qpop"
RUNS = {"42": "cl_r_vol_isla_seed42", "43": "cl_r_vol_isla_seed43"}
FIELDS = ("pressure_l2", "velocity_l2", "nut_l2")
CONDITIONS = ("native_h", "half", "box", "wall", "full")


def main() -> None:
    out = {
        "protocol": (
            "QUERY-POP (2026-10-06, #sec-nb-query-pop): the DrivAerML Volume ISLA references scored on the first 100,000 "
            "of their 200,000 native interior points under four query populations (evaluation-only; the built-in geometry "
            "of every scored point is unchanged by construction), plus the learned query routing mass."
        ),
        "conditions": {
            "native_h": "the native 200,000 tokens; metric on the first 100,000 (control)",
            "half": "tokens = the first 100,000 only",
            "box": "tokens = native 200,000 + 200,000 companions uniform in the interior's bounding box",
            "wall": "tokens = native 200,000 + 200,000 companions resampled from the native points with probability 1/|sdf|",
            "full": "native 200,000 tokens, metric on all (reproduces evals_char native)",
        },
        "seeds": {},
    }
    for seed, run in RUNS.items():
        d = json.load(open(SRC / run / "summary.json"))
        means = {c: {f: d["means"][c][f] for f in FIELDS} for c in CONDITIONS}
        out["seeds"][seed] = {
            "run": run,
            "epoch": d["epoch"],
            "query_routing_mass_fraction_of_boundary": d["query_routing_mass_fraction_of_boundary"],
            "n_companions": d["n_companions"],
            "means": means,
            "ratio_to_native_h": {c: {f: means[c][f] / means["native_h"][f] for f in FIELDS} for c in ("half", "box", "wall")},
            "sanity_full": d["sanity_full"],
        }
    seeds = list(out["seeds"].values())
    out["mean_ratio_to_native_h"] = {
        c: {f: sum(s["ratio_to_native_h"][c][f] for s in seeds) / len(seeds) for f in FIELDS} for c in ("half", "box", "wall")
    }
    out["query_routing_mass_fraction_of_boundary"] = {s["run"]: s["query_routing_mass_fraction_of_boundary"] for s in seeds}
    with open(HERE / "qpop_2026-10-06.json", "w") as f:
        json.dump(out, f, indent=1)
    for c in ("half", "box", "wall"):
        print(f"{c:6s} " + "  ".join(f"{fld[:-3]}={out['mean_ratio_to_native_h'][c][fld]:.3f}" for fld in FIELDS))
    print("query mass:", out["query_routing_mass_fraction_of_boundary"])


if __name__ == "__main__":
    main()
