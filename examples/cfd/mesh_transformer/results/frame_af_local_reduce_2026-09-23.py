"""FRAME-AF local mechanism rung reduction (2026-09-23, #sec-nb-frame-af-local-prereg): float32 per-case relative L2 of the
two arms (baseline: freestream direction as the one global vector; frame: plus the laboratory axes, K = 3) on the 40 local
test airfoils, seeds 42 and 43, from the recipe's infer.py metrics.jsonl (one row per case); per-seed and seed-mean field
means, frame ÷ baseline (verdict on the two-seed means), per-angle bins, and the preregistered verdict (pays / neutral / hurts). usage: python frame_af_local_reduce_2026-09-23.py <infer_root>
where <infer_root>/<arm>/<run>/metrics.jsonl holds the rows; writes frame_af_local_2026-09-23.json beside this script."""
from __future__ import annotations
import json, re, statistics as st, sys
from collections import defaultdict
from pathlib import Path
RES = Path(__file__).resolve().parent
root = Path(sys.argv[1])
FIELDS = ("pressure_l2", "velocity_l2", "nut_l2")
def load(arm):
    per = {}
    for p in root.glob(f"{arm}/*/metrics.jsonl"):
        for l in p.read_text().splitlines():
            if not l.strip(): continue
            r = json.loads(l)
            if r.get("phase") == "infer_step": per[r["sample_id"]] = {f: r["metrics"][f] for f in FIELDS}
    return per
SEEDS = (42, 43)
arms = {f"frameaf_{arm}_seed{sd}": load(f"frameaf_{arm}_seed{sd}") for arm in ("base", "frame") for sd in SEEDS}
ids = sorted(set.intersection(*(set(v) for v in arms.values())))
aoa = lambda s: float(re.search(r"airFoil2D_SST_[-\d.]+_([-\d.]+)_", s).group(1))
binname = lambda a: "aoa<0" if a < 0 else "0-5" if a < 5 else "5-10" if a < 10 else ">=10"
BINS = ("aoa<0", "0-5", "5-10", ">=10")
out = {"protocol": __doc__, "n_cases": len(ids), "seeds": list(SEEDS), "means": {}, "seed_means": {}, "frame_over_base": {},
       "frame_over_base_per_seed": {}, "by_aoa": defaultdict(dict), "by_aoa_seed_mean": defaultdict(dict)}
for a, d in arms.items():
    out["means"][a] = {f: st.fmean(d[s][f] for s in ids) for f in FIELDS}
    for b in BINS:
        grp = [s for s in ids if binname(aoa(s)) == b]
        if grp: out["by_aoa"][b][a] = {"n": len(grp), **{f: st.fmean(d[s][f] for s in grp) for f in FIELDS}}
for arm in ("base", "frame"):
    out["seed_means"][arm] = {f: st.fmean(out["means"][f"frameaf_{arm}_seed{sd}"][f] for sd in SEEDS) for f in FIELDS}
    for b in BINS:
        rows = [out["by_aoa"][b][f"frameaf_{arm}_seed{sd}"] for sd in SEEDS if f"frameaf_{arm}_seed{sd}" in out["by_aoa"][b]]
        if rows: out["by_aoa_seed_mean"][b][arm] = {"n": rows[0]["n"], **{f: st.fmean(r[f] for r in rows) for f in FIELDS}}
for sd in SEEDS:
    b_, f_ = out["means"][f"frameaf_base_seed{sd}"], out["means"][f"frameaf_frame_seed{sd}"]
    out["frame_over_base_per_seed"][str(sd)] = {f: f_[f] / b_[f] for f in FIELDS}
base, frame = out["seed_means"]["base"], out["seed_means"]["frame"]
out["frame_over_base"] = {f: frame[f] / base[f] for f in FIELDS}
r = out["frame_over_base"]
if r["pressure_l2"] <= 0.90 and r["velocity_l2"] <= 1.05: verdict = "frame pays"
elif all(abs(v - 1) <= 0.05 for v in r.values()): verdict = "frame neutral"
elif any(v > 1.05 for v in r.values()): verdict = "frame hurts"
else: verdict = "between (reported as measured)"
out["verdict"] = verdict
(RES / "frame_af_local_2026-09-23.json").write_text(json.dumps(out, indent=1))
print(f"cases {len(ids)}; two-seed means: base {', '.join(f'{f} {base[f]:.4f}' for f in FIELDS)}; frame {', '.join(f'{f} {frame[f]:.4f}' for f in FIELDS)}")
print("ratio (frame/base):", {f: round(r[f], 3) for f in FIELDS}, "| per seed:", {k: {f: round(v[f], 3) for f in FIELDS} for k, v in out["frame_over_base_per_seed"].items()}, "| verdict:", verdict)
for b in BINS:
    if b in out["by_aoa_seed_mean"] and "frame" in out["by_aoa_seed_mean"][b]:
        bb, ff = out["by_aoa_seed_mean"][b]["base"], out["by_aoa_seed_mean"][b]["frame"]
        print(f"  {b:6s} n={bb['n']:2d} " + " ".join(f"{f} {ff[f]/bb[f]:.3f}" for f in FIELDS))
print("frame ÷ base", {f: round(v, 3) for f, v in r.items()}, "->", verdict)
for b, d in out["by_aoa"].items(): print(b, {a[8:-7]: round(v["pressure_l2"], 4) for a, v in d.items()})
