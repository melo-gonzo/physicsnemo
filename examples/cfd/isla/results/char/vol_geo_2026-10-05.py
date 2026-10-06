"""VOL-GEO (2026-10-05, #sec-nb-char-vol-geo-prereg / #sec-nb-char-vol-geo-verdict): the DrivAerML Volume reference checkpoints
(cl_r_vol_isla_seed42/43, ISLA width 512 with 512 slices) scored in float32 on the 48 validation cars under: native draw; the
area-proportional boundary draw with exact Horvitz-Thompson measures (the campaign's point-cloud protocol); the area-draw tokens with
the queries' built-in nearest-boundary geometry (signed distance and direction) computed from the NATIVE draw's boundary; the area-draw
tokens with the built-in geometry replaced by the stored signed-distance field and its gradient; and the native tokens with the stored
field. Scoring-time swap of `nearest_boundary_geometry` only (kernel/char/vol_geo/geo_swap.py, copied under
campaigns_2026-10-02/char/vol_geo/). Writes results/char/vol_geo_2026-10-05.json: per seed and seed mean of pressure / velocity / eddy
viscosity relative L2 per condition, ratios to native, the fraction of the area-draw log-excess removed by each swap, and the input
shift statistics (fraction of queries whose nearest token changes, RMS log ratio of the built-in distance between draws and against
the stored field, near-wall medians). usage: python vol_geo_2026-10-05.py"""
import json, math, os, statistics as st

R = os.path.dirname(os.path.abspath(__file__)); RES = os.path.dirname(R)
SRC = f"{RES}/campaigns_2026-10-02/char/vol_geo"
F = ["pressure_l2", "velocity_l2", "nut_l2"]
COND = {"native": "native draw", "area": "area draw (campaign point-cloud protocol)",
        "area_natgeo": "area tokens, built-in geometry from the native draw",
        "area_sdfgeo": "area tokens, built-in geometry from the stored signed-distance field",
        "native_sdfgeo": "native tokens, built-in geometry from the stored signed-distance field"}
out = {"protocol": __doc__, "conditions": COND, "seeds": {}, "mean": {}, "shift": {}}
for s in (42, 43):
    d = json.load(open(f"{SRC}/cl_r_vol_isla_seed{s}/summary.json"))
    out["seeds"][s] = {"means": {c: d["means"][c] for c in COND if c in d["means"]},
                       "ratio_to_native": {c: d["ratio_to_native"][c] for c in COND if c in d.get("ratio_to_native", {})},
                       "fraction_of_log_excess_removed": d.get("fraction_of_log_excess_removed_vs_area"),
                       "sanity": d.get("sanity")}
    out["shift"][s] = d["shift_means_all"]
for c in COND:
    out["mean"][c] = {f: st.mean(out["seeds"][s]["means"][c][f] for s in (42, 43)) for f in F}
nat = out["mean"]["native"]
out["mean_ratio_to_native"] = {c: {f: out["mean"][c][f] / nat[f] for f in F} for c in COND}
ex = {f: math.log(out["mean"]["area"][f] / nat[f]) for f in F}
out["mean_fraction_of_log_excess_removed"] = {c: {f: 1 - math.log(out["mean"][c][f] / nat[f]) / ex[f] for f in F} for c in ("area_natgeo", "area_sdfgeo")}
json.dump(out, open(f"{R}/vol_geo_2026-10-05.json", "w"), indent=1)
for c, label in COND.items():
    m = out["mean"][c]; r = out["mean_ratio_to_native"][c]
    print(f"{c:14s} {label:62s} " + " / ".join(f"{m[f]:.4f}" for f in F) + "   x native " + " / ".join(f"{r[f]:.2f}" for f in F))
print("fraction of log-excess removed:", {c: [round(v, 3) for v in d.values()] for c, d in out["mean_fraction_of_log_excess_removed"].items()})
sh = out["shift"][42]
print(f"nearest token changed: {sh['frac_nearest_token_changed']:.3f} (near wall {sh['frac_nearest_token_changed_nearwall']:.3f}); RMS log|d_area/d_nat| near wall {sh['rms_log_absdArea_over_absdNat_nearwall_sdf']:.3f}; RMS log|d_nat/sdf| {sh['rms_log_absdNat_over_abssdf_nearwall_sdf']:.3f}; medians |d|/L near wall: d_nat {sh['median_abs_dNat_over_L_nearwall_sdf']:.2e} d_area {sh['median_abs_dArea_over_L_nearwall_sdf']:.2e} sdf {sh['median_abs_sdf_over_L_nearwall_sdf']:.2e}")
