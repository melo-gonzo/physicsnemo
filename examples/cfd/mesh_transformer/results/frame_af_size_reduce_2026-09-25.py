"""FRAME-AF-SIZE reduction (2026-09-25, #sec-nb-frame-af-size-prereg): the direction-count step (inlet + x + y over inlet
only, K = 3 / K = 1) against training-set size on the local AirfRANS rung. Float32 per-case relative L2 on the 40 test
airfoils from infer.py metrics.jsonl; per size the two-seed mean of each arm and the ratio; the 120 rung is the ladder's
base / frame pair. usage: python frame_af_size_reduce_2026-09-25.py <infer_root>; writes frame_af_size_2026-09-25.json."""
from __future__ import annotations
import json, statistics as st, sys
from pathlib import Path
RES = Path(__file__).resolve().parent; root = Path(sys.argv[1])
F = ("pressure_l2", "velocity_l2", "nut_l2"); SEEDS = (42, 43)
def load(run):
    per = {}
    for p in root.glob(f"{run}/*/metrics.jsonl"):
        for l in p.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("phase") == "infer_step": per[r["sample_id"]] = {f: r["metrics"][f] for f in F}
    return per
name = lambda arm, n, s: f"frameaf_{arm}_seed{s}" if n == 120 else f"frameaf_{arm}_n{n}_seed{s}"
out = {"protocol": __doc__, "sizes": {}, "ratio_k3_over_k1": {}}
for n in (30, 60, 120, 240):
    runs = {(a, s): load(name(a, n, s)) for a in ("base", "frame") for s in SEEDS}
    if any(len(v) == 0 for v in runs.values()): continue
    ids = sorted(set.intersection(*(set(v) for v in runs.values())))
    per = {f"{a}_seed{s}": {f: st.fmean(d[i][f] for i in ids) for f in F} for (a, s), d in runs.items()}
    mean = {a: {f: st.fmean(per[f"{a}_seed{s}"][f] for s in SEEDS) for f in F} for a in ("base", "frame")}
    out["sizes"][str(n)] = {"n_cases": len(ids), "per_run": per, "mean": mean}
    out["ratio_k3_over_k1"][str(n)] = {f: mean["frame"][f] / mean["base"][f] for f in F}
r = {int(k): v["pressure_l2"] for k, v in out["ratio_k3_over_k1"].items()}
if all(k in r for k in (30, 60, 120)):
    mono_dec = r[30] <= r[60] <= r[120]; mono_inc = r[30] >= r[60] >= r[120]
    if r[30] <= r[120] - 0.10 and mono_dec: v = "shrinks with data"
    elif all(abs(r[k] - 0.75) <= 0.07 for k in r): v = "holds"
    elif r[30] >= r[120] + 0.10 and mono_inc: v = "grows with data"
    else: v = "between (reported as measured)"
    out["verdict_30_60_120"] = v
(RES / "frame_af_size_2026-09-25.json").write_text(json.dumps(out, indent=1))
for k, v in sorted(out["sizes"].items(), key=lambda kv: int(kv[0])):
    m = v["mean"]; print(f"n={k:>3}: base {m['base']['pressure_l2']:.4f} frame {m['frame']['pressure_l2']:.4f} ratio p {out['ratio_k3_over_k1'][k]['pressure_l2']:.3f} v {out['ratio_k3_over_k1'][k]['velocity_l2']:.3f} nut {out['ratio_k3_over_k1'][k]['nut_l2']:.3f} | base seeds {[round(v['per_run'][f'base_seed{s}']['pressure_l2'],3) for s in SEEDS]} frame seeds {[round(v['per_run'][f'frame_seed{s}']['pressure_l2'],3) for s in SEEDS]}")
print("verdict (30/60/120):", out.get("verdict_30_60_120"))
