"""AF-PLANAR reduction: seed values and seed means of velocity_l2 (all components) and velocity_xy_l2 (in-plane).

Reads results/<run>/<protocol>/planar_summary.json (means over the 200 per-case values, as infer_summary averages
velocity_l2). usage: python reduce_planar.py > planar_table.md ; also writes planar_scores.json.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
ARMS = {
    "GeoTransolver afu_ud_gt (unit drive)": "afu_ud_gt",
    "Transolver afu_ud_t (unit drive)": "afu_ud_t",
    "GeoTransolver afu_r_gt (raw drive)": "afu_r_gt",
    "Transolver afu_r_t (raw drive)": "afu_r_t",
}
ISLA = {"ISLA A2 afu_a2s (airfrans_a2.json)": (0.009697543978691102, 0.009214075803756714),
        "ISLA afu_r_isla (evals_afu)": (0.011607264280319213, 0.012358256578445435)}


def load(run, proto):
    return json.loads((HERE / "results" / run / proto / "planar_summary.json").read_text())


out = {}
for proto in ("native", "posed"):
    print(f"\n### {proto}\n")
    print("| arm | velocity_l2 (3-D) s42 / s43 / mean | velocity_xy_l2 s42 / s43 / mean | xy / 3-D | z share of squared error (max seed) | check (max per-case rel. diff vs shared rows) |")
    print("|---|---|---|---|---|---|")
    for label, stem in ARMS.items():
        s = [load(f"{stem}_seed{k}", proto) for k in (42, 43)]
        v3 = [x["mean"]["velocity_l2"] for x in s]
        vxy = [x["mean"]["velocity_xy_l2"] for x in s]
        m3, mxy = sum(v3) / 2, sum(vxy) / 2
        zs = max(x["mean"]["err2_share_z"] for x in s)
        chk = max(x["max_reldiff_vs_shared_rows"] for x in s)
        assert all(x["check_ok"] for x in s)
        out[f"{stem}/{proto}"] = {"velocity_l2": v3, "velocity_xy_l2": vxy, "mean_velocity_l2": m3, "mean_velocity_xy_l2": mxy,
                                  "err2_share_z_max": zs, "max_reldiff_vs_shared": chk}
        print(f"| {label} | {v3[0]:.5f} / {v3[1]:.5f} / **{m3:.5f}** | {vxy[0]:.5f} / {vxy[1]:.5f} / **{mxy:.5f}** | "
              f"{mxy / m3:.7f} | {zs:.1e} | {chk:.1e} |")

print("\n### native ratios, baseline / ISLA (seed means)\n")
print("| arm | vs A2 0.00946: 3-D | vs A2: in-plane | vs afu_r_isla 0.01198: 3-D | vs afu_r_isla: in-plane |")
print("|---|---|---|---|---|")
a2 = sum(ISLA["ISLA A2 afu_a2s (airfrans_a2.json)"]) / 2
ri = sum(ISLA["ISLA afu_r_isla (evals_afu)"]) / 2
for label, stem in ARMS.items():
    d = out[f"{stem}/native"]
    print(f"| {label} | {d['mean_velocity_l2'] / a2:.3f} | {d['mean_velocity_xy_l2'] / a2:.3f} | "
          f"{d['mean_velocity_l2'] / ri:.3f} | {d['mean_velocity_xy_l2'] / ri:.3f} |")
isla = load("afu_r_isla_seed42", "native")
out["afu_r_isla_seed42/native"] = isla["mean"] | {"max_reldiff_vs_shared": isla["max_reldiff_vs_shared_rows"]}
out["isla_reference"] = {"afu_a2s_mean": a2, "afu_r_isla_mean": ri}
print(f"\nISLA check (afu_r_isla_seed42 native, 2-D): velocity_l2 {isla['mean']['velocity_l2']:.6f}, "
      f"velocity_xy_l2 {isla['mean']['velocity_xy_l2']:.6f} (identical: 2 components), "
      f"max per-case rel. diff vs shared rows {isla['max_reldiff_vs_shared_rows']:.1e}")
(HERE / "planar_scores.json").write_text(json.dumps(out, indent=1) + "\n")
