"""FRAME-AF local mechanism rung reduction (2026-09-23, #sec-nb-frame-af-local-prereg): float32 per-case relative L2 of the
two arms (baseline: freestream direction as the one global vector; frame: plus the laboratory axes, K = 3) on the 40 local
test airfoils, from the recipe's infer.py metrics.jsonl (one row per case); field means, frame ÷ baseline, per-angle bins,
and the preregistered verdict (pays / neutral / hurts). usage: python frame_af_local_reduce_2026-09-23.py <infer_root>
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
arms = {a: load(a) for a in ("frameaf_base_seed42", "frameaf_frame_seed42")}
ids = sorted(set.intersection(*(set(v) for v in arms.values())))
aoa = lambda s: float(re.search(r"airFoil2D_SST_[-\d.]+_([-\d.]+)_", s).group(1))
binname = lambda a: "aoa<0" if a < 0 else "0-5" if a < 5 else "5-10" if a < 10 else ">=10"
out = {"protocol": __doc__, "n_cases": len(ids), "means": {}, "frame_over_base": {}, "by_aoa": defaultdict(dict)}
for a, d in arms.items():
    out["means"][a] = {f: st.fmean(d[s][f] for s in ids) for f in FIELDS}
    for b in ("aoa<0", "0-5", "5-10", ">=10"):
        grp = [s for s in ids if binname(aoa(s)) == b]
        if grp: out["by_aoa"][b][a] = {"n": len(grp), **{f: st.fmean(d[s][f] for s in grp) for f in FIELDS}}
base, frame = out["means"]["frameaf_base_seed42"], out["means"]["frameaf_frame_seed42"]
out["frame_over_base"] = {f: frame[f] / base[f] for f in FIELDS}
r = out["frame_over_base"]
if r["pressure_l2"] <= 0.90 and r["velocity_l2"] <= 1.05: verdict = "frame pays"
elif all(abs(v - 1) <= 0.05 for v in r.values()): verdict = "frame neutral"
elif any(v > 1.05 for v in r.values()): verdict = "frame hurts"
else: verdict = "between (reported as measured)"
out["verdict"] = verdict
(RES / "frame_af_local_2026-09-23.json").write_text(json.dumps(out, indent=1))
print(f"cases {len(ids)}; base {', '.join(f'{f} {base[f]:.4f}' for f in FIELDS)}; frame {', '.join(f'{f} {frame[f]:.4f}' for f in FIELDS)}")
print("frame ÷ base", {f: round(v, 3) for f, v in r.items()}, "->", verdict)
for b, d in out["by_aoa"].items(): print(b, {a[8:-7]: round(v["pressure_l2"], 4) for a, v in d.items()})
