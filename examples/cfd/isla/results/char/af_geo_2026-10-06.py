"""AF-GEO step one (2026-10-06): the AirfRANS ISLA references under the boundary-RESAMPLING protocol.

Reduces kernel/char/af_geo/<run>/summary.json (copied to results/campaigns_2026-10-02/char/af_geo/) for the two
AirfRANS reference checkpoints cl_r_af_isla_seed42/43 (nearest built-in geometry + stored sdf / sdf_normals query
inputs; whole airfoil, ~1,007 segments) scored on the 200 full_test cases, float32, 100,000 native interior points:

  native       the whole airfoil (reproduces evals_char/<run>/native per case to 1e-6)
  subu         an expected 400 segments under the uniform-over-cells law, exact HT measures
  suba         an expected 400 segments under the length-proportional law
  subu_natgeo  subu tokens, the queries' built-in geometry from the whole airfoil (swap control)
  suba_natgeo  suba tokens, native built-in geometry

fraction_of_log_excess_removed = (log e_sub - log e_sub_natgeo) / (log e_sub - log e_native): the share of the
resampling excess carried by the built-in channel; the remainder is carried by the slice states over 400 instead of
1,007 tokens (the information lost to the coarsening, a confound the Volume protocol did not have: there the count
stayed at 200,000 and only the law changed). Writes af_geo_2026-10-06.json next to this file.
"""

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "campaigns_2026-10-02" / "char" / "af_geo"
RUNS = {"42": "cl_r_af_isla_seed42", "43": "cl_r_af_isla_seed43"}
FIELDS = ("pressure_l2", "velocity_l2", "nut_l2")
CONDITIONS = ("native", "subu", "suba", "subu_natgeo", "suba_natgeo")
SHIFT_KEYS = (
    "n_tokens_native", "n_nearwall", "median_abs_dNat_nearwall", "median_abs_sdf_nearwall",
    "rms_log_absdNat_over_abssdf_nearwall", "sign_agree_dNat_sdf",
    "subu_n_tokens", "subu_total_measure", "subu_frac_nearest_token_changed", "subu_frac_nearest_token_changed_nearwall",
    "subu_rms_log_absd_over_absdNat_nearwall", "subu_cos_u_uNat_nearwall", "subu_sign_agree_d_dNat",
    "suba_n_tokens", "suba_total_measure", "suba_frac_nearest_token_changed", "suba_frac_nearest_token_changed_nearwall",
    "suba_rms_log_absd_over_absdNat_nearwall", "suba_cos_u_uNat_nearwall", "suba_sign_agree_d_dNat",
)


def main() -> None:
    out = {
        "protocol": (
            "AF-GEO step one (2026-10-06, #sec-nb-soft-geo-prereg / #sec-nb-af-geo-step-one): the AirfRANS ISLA "
            "references (cl_r_af_isla_seed42/43: nearest built-in geometry + stored sdf/sdf_normals query inputs) "
            "scored on the 200 full_test cases, float32, 100,000 native interior points, with the airfoil resampled "
            "to an expected 400 of its ~1,007 segments under two laws, with and without the native built-in geometry."
        ),
        "conditions": {
            "native": "whole airfoil (~1,007 segments)",
            "subu": "expected 400 segments, uniform-over-cells law, exact HT measures",
            "suba": "expected 400 segments, length-proportional law, exact HT measures",
            "subu_natgeo": "subu tokens, built-in geometry from the whole airfoil",
            "suba_natgeo": "suba tokens, built-in geometry from the whole airfoil",
        },
        "seeds": {},
    }
    for seed, run in RUNS.items():
        d = json.load(open(SRC / run / "summary.json"))
        means = {c: {f: d["means"][c][f] for f in FIELDS} for c in CONDITIONS}
        ratio = {c: {f: means[c][f] / means["native"][f] for f in FIELDS} for c in CONDITIONS}
        removed = {}
        for k in ("subu", "suba"):
            removed[k] = {}
            for f in FIELDS:
                excess = math.log(means[k][f] / means["native"][f])
                removed[k][f] = (math.log(means[k][f]) - math.log(means[f"{k}_natgeo"][f])) / excess
        out["seeds"][seed] = {
            "run": run,
            "epoch": d["epoch"],
            "num_samples": d["num_samples"],
            "same_interior_points_all_cases": d["same_interior_points_all_cases"],
            "means": means,
            "ratio_to_native": ratio,
            "fraction_of_log_excess_removed": removed,
            "sanity_native": d["sanity_native"],
            "shift": {k: d["shift_means_all"][k] for k in SHIFT_KEYS},
        }
    seeds = list(out["seeds"].values())
    out["mean"] = {c: {f: sum(s["means"][c][f] for s in seeds) / len(seeds) for f in FIELDS} for c in CONDITIONS}
    out["mean_ratio_to_native"] = {c: {f: sum(s["ratio_to_native"][c][f] for s in seeds) / len(seeds) for f in FIELDS} for c in CONDITIONS}
    out["mean_fraction_of_log_excess_removed"] = {
        k: {f: sum(s["fraction_of_log_excess_removed"][k][f] for s in seeds) / len(seeds) for f in FIELDS} for k in ("subu", "suba")
    }
    out["shift"] = {k: sum(s["shift"][k] for s in seeds) / len(seeds) for k in SHIFT_KEYS}
    with open(HERE / "af_geo_2026-10-06.json", "w") as f:
        json.dump(out, f, indent=1)
    for c in CONDITIONS:
        print(f"{c:12s} " + "  ".join(f"{fld[:-3]}={out['mean_ratio_to_native'][c][fld]:.3f}" for fld in FIELDS))
    for k in ("subu", "suba"):
        print(f"removed {k}: " + "  ".join(f"{fld[:-3]}={out['mean_fraction_of_log_excess_removed'][k][fld]:.3f}" for fld in FIELDS))


if __name__ == "__main__":
    main()
