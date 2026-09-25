"""FRAME-AF-X reduction (2026-09-24, #sec-nb-frame-af-x-prereg): three arms on the 40 local AirfRANS test airfoils, seeds 42/43,
float32 per-case relative L2 from infer.py metrics.jsonl: base (inlet direction only, K = 1), framex (+ x axis, K = 2), framey (+ y axis, K = 2; 2026-09-24),
frame (+ x and y axes, K = 3). Reports two-seed means, ratios to base, the log-scale fraction of the two-axis gain that one
axis recovers f = ln(e_base/e_x)/ln(e_base/e_xy), by-angle bins, and the preregistered verdict.
usage: python frame_af_x_reduce_2026-09-24.py <infer_root>; writes frame_af_xy_2026-09-24.json beside this script."""
from __future__ import annotations
import json, math, re, statistics as st, sys
from collections import defaultdict
from pathlib import Path
RES = Path(__file__).resolve().parent
root = Path(sys.argv[1])
FIELDS = ("pressure_l2", "velocity_l2", "nut_l2")
SEEDS = (42, 43); ARMS = ("base", "framex", "framey", "framee", "framef", "frameg", "frameh", "framek4", "frame")  # frameg added (G control)  # framee/framef added 2026-09-24 (E/F controls)  # framey added 2026-09-24 (FRAME-AF-Y)
def load(run):
    per = {}
    for p in root.glob(f"{run}/*/metrics.jsonl"):
        for l in p.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("phase") == "infer_step": per[r["sample_id"]] = {f: r["metrics"][f] for f in FIELDS}
    return per
runs = {(a, s): load(f"frameaf_{a}_seed{s}") for a in ARMS for s in SEEDS}
ids = sorted(set.intersection(*(set(v) for v in runs.values())))
aoa = lambda s: float(re.search(r"airFoil2D_SST_[-\d.]+_([-\d.]+)_", s).group(1))
binname = lambda a: "aoa<0" if a < 0 else "0-5" if a < 5 else "5-10" if a < 10 else ">=10"
BINS = ("aoa<0", "0-5", "5-10", ">=10")
mean = lambda d, grp, f: st.fmean(d[s][f] for s in grp)
out = {"protocol": __doc__, "n_cases": len(ids), "seeds": list(SEEDS), "per_run": {}, "seed_means": {}, "ratio_to_base": {},
       "fraction_recovered": {}, "by_aoa": defaultdict(dict), "by_aoa_ratio_to_base": defaultdict(dict), "by_aoa_x_over_xy": {}}
for (a, s), d in runs.items():
    out["per_run"][f"{a}_seed{s}"] = {f: mean(d, ids, f) for f in FIELDS}
for a in ARMS:
    out["seed_means"][a] = {f: st.fmean(out["per_run"][f"{a}_seed{s}"][f] for s in SEEDS) for f in FIELDS}
    for b in BINS:
        grp = [s for s in ids if binname(aoa(s)) == b]
        out["by_aoa"][b][a] = {"n": len(grp), **{f: st.fmean(mean(runs[(a, s)], grp, f) for s in SEEDS) for f in FIELDS}}
sm = out["seed_means"]
for a in ("framex", "framey", "framee", "framef", "frameg", "frameh", "framek4", "frame"):
    out["ratio_to_base"][a] = {f: sm[a][f] / sm["base"][f] for f in FIELDS}
    for b in BINS: out["by_aoa_ratio_to_base"][b][a] = {f: out["by_aoa"][b][a][f] / out["by_aoa"][b]["base"][f] for f in FIELDS}
out["fraction_recovered"] = {f: math.log(sm["base"][f] / sm["framex"][f]) / math.log(sm["base"][f] / sm["frame"][f]) for f in FIELDS}
out["fraction_recovered_y"] = {f: math.log(sm["base"][f] / sm["framey"][f]) / math.log(sm["base"][f] / sm["frame"][f]) for f in FIELDS}
out["by_aoa_y_over_xy"] = {b: {f: out["by_aoa"][b]["framey"][f] / out["by_aoa"][b]["frame"][f] for f in FIELDS} for b in BINS}
out["fraction_recovered_ef"] = {a: {f: math.log(sm["base"][f] / sm[a][f]) / math.log(sm["base"][f] / sm["frame"][f]) for f in FIELDS} for a in ("framee", "framef", "frameg", "frameh", "framek4")}
out["ef_over_xy"] = {a: {f: sm[a][f] / sm["frame"][f] for f in FIELDS} for a in ("framee", "framef", "frameg", "frameh", "framek4")}
fy = out["fraction_recovered_y"]; ymax = max(abs(out["by_aoa_y_over_xy"][b]["pressure_l2"] - 1) for b in BINS)
out["verdict_y"] = "reading supported" if (fy["pressure_l2"] >= 0.8 and ymax <= 0.10) else "reading refuted" if fy["pressure_l2"] <= 0.5 else "between (reported as measured)"
out["by_aoa_x_over_xy"] = {b: {f: out["by_aoa"][b]["framex"][f] / out["by_aoa"][b]["frame"][f] for f in FIELDS} for b in BINS}
fr = out["fraction_recovered"]; neg = out["by_aoa_x_over_xy"]["aoa<0"]["pressure_l2"]
pos = st.fmean(out["by_aoa_x_over_xy"][b]["pressure_l2"] for b in ("0-5", "5-10", ">=10"))
if fr["pressure_l2"] >= 0.8 and fr["velocity_l2"] >= 0.8 and abs(neg - 1) <= 0.10: verdict = "one axis suffices"
elif fr["pressure_l2"] <= 0.5 or (neg > 1.15 and abs(pos - 1) <= 0.05): verdict = "both axes needed" + (" (sign-loss signature)" if neg > 1.15 and abs(pos - 1) <= 0.05 else "")
else: verdict = "between (reported as measured)"
out["verdict"] = verdict; out["x_over_xy_negative_bin_pressure"] = neg; out["x_over_xy_positive_bins_pressure_mean"] = pos
(RES / "frame_af_xy_2026-09-24.json").write_text(json.dumps(out, indent=1))
print(f"cases {len(ids)}")
for a in ARMS: print(f"  {a:7s}", {f: round(sm[a][f], 4) for f in FIELDS}, "| per seed p:", [round(out['per_run'][f'{a}_seed{s}']['pressure_l2'], 4) for s in SEEDS])
print("ratio to base:", {a: {f: round(v[f], 3) for f in FIELDS} for a, v in out["ratio_to_base"].items()})
print("fraction recovered by one axis:", {f: round(v, 2) for f, v in fr.items()})
for b in BINS: print(f"  {b:6s} n={out['by_aoa'][b]['base']['n']:2d} x/base {out['by_aoa_ratio_to_base'][b]['framex']['pressure_l2']:.2f} xy/base {out['by_aoa_ratio_to_base'][b]['frame']['pressure_l2']:.2f} x/xy {out['by_aoa_x_over_xy'][b]['pressure_l2']:.2f} (pressure) | velocity x/xy {out['by_aoa_x_over_xy'][b]['velocity_l2']:.2f}")
print("verdict:", verdict)
for a in ("framee", "framef", "frameg", "frameh", "framek4"): print(a, "fraction recovered", {f: round(v, 2) for f, v in out["fraction_recovered_ef"][a].items()}, "| ÷ two-axis", {f: round(v, 3) for f, v in out["ef_over_xy"][a].items()}, "| per seed p:", [round(out["per_run"][f"{a}_seed{s}"]["pressure_l2"], 4) for s in SEEDS])
print("Y: fraction recovered", {f: round(v, 2) for f, v in fy.items()}, "| y/xy pressure by bin", {b: round(out["by_aoa_y_over_xy"][b]["pressure_l2"], 2) for b in BINS}, "| per seed p:", [round(out["per_run"][f"framey_seed{s}"]["pressure_l2"], 4) for s in SEEDS], "| verdict_y:", out["verdict_y"])
