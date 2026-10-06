"""AF-GEO refinement (2026-10-06): the AirfRANS ISLA references under the count-preserving RE-PANELIZATION protocol.

Reduces kernel/char/af_geo_ref/<run>/summary.json (copied to results/campaigns_2026-10-02/char/af_geo_ref/) for
cl_r_af_isla_seed42/43 (200 full_test cases, float32, 100,000 native interior points) with the airfoil's 1,007
segments split into collinear equal pieces by SplitLineCells: refu (2 / 2, uniform; 2,014 tokens), reff (3 ahead of
mid-chord / 1 behind; 2,206 tokens); geometry and boundary measure unchanged, only the panel pattern; and the swap
controls <k>_natgeo (the variant's tokens, the whole airfoil's built-in geometry fed to the queries). The excess that
the swap removes is the built-in channel's; what it leaves is the slice states' (none expected: no information is lost).
Writes af_geo_refine_2026-10-06.json next to this file."""

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "campaigns_2026-10-02" / "char" / "af_geo_ref"
RUNS = {"42": "cl_r_af_isla_seed42", "43": "cl_r_af_isla_seed43"}
FIELDS = ("pressure_l2", "velocity_l2", "nut_l2")
VARIANTS = ("refu", "reff")
CONDITIONS = ("native",) + VARIANTS + tuple(f"{k}_natgeo" for k in VARIANTS)
SHIFT_KEYS = tuple(
    f"{k}_{m}" for k in VARIANTS for m in ("n_tokens", "frac_nearest_token_changed", "frac_nearest_token_changed_nearwall",
                                          "rms_log_absd_over_absdNat_nearwall", "cos_u_uNat_nearwall", "sign_agree_d_dNat")
) + ("n_tokens_native",)


def main() -> None:
    out = {
        "protocol": (
            "AF-GEO refinement (2026-10-06, #sec-nb-af-geo-refine): the AirfRANS ISLA references with the airfoil "
            "re-panelized by SplitLineCells (uniform 2/2; front-loaded 3/1), geometry and measure unchanged; with and "
            "without the native built-in geometry."
        ),
        "conditions": {
            "native": "whole airfoil, 1,007 segments",
            "refu": "every segment split in 2 (2,014 tokens)",
            "reff": "segments ahead of mid-chord split in 3, behind in 1 (2,206 tokens)",
            "refu_natgeo": "refu tokens, built-in geometry from the native airfoil",
            "reff_natgeo": "reff tokens, built-in geometry from the native airfoil",
        },
        "seeds": {},
    }
    for seed, run in RUNS.items():
        d = json.load(open(SRC / run / "summary.json"))
        means = {c: {f: d["means"][c][f] for f in FIELDS} for c in CONDITIONS}
        ratio = {c: {f: means[c][f] / means["native"][f] for f in FIELDS} for c in CONDITIONS}
        removed = {
            k: {f: (math.log(means[k][f]) - math.log(means[f"{k}_natgeo"][f])) / math.log(means[k][f] / means["native"][f]) for f in FIELDS}
            for k in VARIANTS
        }
        out["seeds"][seed] = {
            "run": run, "epoch": d["epoch"], "num_samples": d["num_samples"],
            "same_interior_points_all_cases": d["same_interior_points_all_cases"],
            "means": means, "ratio_to_native": ratio, "fraction_of_log_excess_removed": removed,
            "sanity_native": d["sanity_native"],
            "shift": {k: d["shift_means_all"][k] for k in SHIFT_KEYS},
        }
    seeds = list(out["seeds"].values())
    out["mean_ratio_to_native"] = {c: {f: sum(s["ratio_to_native"][c][f] for s in seeds) / len(seeds) for f in FIELDS} for c in CONDITIONS}
    out["mean_fraction_of_log_excess_removed"] = {k: {f: sum(s["fraction_of_log_excess_removed"][k][f] for s in seeds) / len(seeds) for f in FIELDS} for k in VARIANTS}
    out["shift"] = {k: sum(s["shift"][k] for s in seeds) / len(seeds) for k in SHIFT_KEYS}
    with open(HERE / "af_geo_refine_2026-10-06.json", "w") as f:
        json.dump(out, f, indent=1)
    for c in CONDITIONS:
        print(f"{c:12s} " + "  ".join(f"{fld[:-3]}={out['mean_ratio_to_native'][c][fld]:.3f}" for fld in FIELDS))
    for k in VARIANTS:
        print(f"removed {k}: " + "  ".join(f"{fld[:-3]}={out['mean_fraction_of_log_excess_removed'][k][fld]:.3f}" for fld in FIELDS))


if __name__ == "__main__":
    main()
