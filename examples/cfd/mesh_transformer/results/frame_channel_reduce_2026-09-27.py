"""FRAME-CHANNEL reduction (2026-09-27, #sec-nb-frame-channel-prereg): which channel carries the laboratory frame's gain on the
local AirfRANS rung. Arms (frameaf_<arm>_seed{42,43}, 192 x 12 x 256, cosine, fused, 20,000 queries, 300 epochs, lr 1e-3, 40
local test airfoils): base (K = 1) and frame (K = 3, all channels) from the FRAME-AF rung; seedonly (geometry_directions 1: the
three directions enter the token seed only) and geoonly (seed_directions 1: the geometry channels only). Readout: float32 per-case
relative L2 (pressure, velocity, eddy viscosity) from the recipe's infer.py metrics.jsonl; two-seed means; fraction of the frame's
log-gain ln(base/arm)/ln(base/frame) per field. Bars on pressure: seed carries it (seedonly >= 0.75 and geoonly <= 0.35); geometry
carries it (reverse); joint (both <= 0.5); either (both >= 0.75); between otherwise. usage: python frame_channel_reduce_2026-09-27.py
<infer_root>  where <infer_root>/<run>/<run or *>/metrics.jsonl holds the rows; arms with one scored seed are reported as interim;
writes frame_channel_2026-09-27.json beside this script."""
from __future__ import annotations
import json, math, re, statistics as st, sys
from pathlib import Path

RES = Path(__file__).resolve().parent
root = Path(sys.argv[1])
FIELDS = ("pressure_l2", "velocity_l2", "nut_l2")
SEEDS = (42, 43)
ARMS = ("base", "frame", "seedonly", "geoonly")


def load(run):
    per = {}
    for p in root.glob(f"{run}/**/metrics.jsonl"):
        for l in p.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("phase") == "infer_step":
                    per[r["sample_id"]] = {f: r["metrics"][f] for f in FIELDS}
    return per


runs = {(a, s): load(f"frameaf_{a}_seed{s}") for a in ARMS for s in SEEDS}
runs = {k: v for k, v in runs.items() if v}
ids = sorted(set.intersection(*(set(v) for v in runs.values())))
out = {"protocol": __doc__, "n_cases": len(ids), "means": {}, "seed_means": {}, "n_seeds": {}, "ratio_to_base": {},
       "ratio_to_frame": {}, "fraction_of_frame_gain": {}, "fraction_per_seed": {}, "by_aoa": {}}
for (a, s), d in runs.items():
    out["means"][f"{a}_seed{s}"] = {f: st.fmean(d[i][f] for i in ids) for f in FIELDS}
for a in ARMS:
    ss = [s for s in SEEDS if (a, s) in runs]
    if not ss:
        continue
    out["n_seeds"][a] = len(ss)
    out["seed_means"][a] = {f: st.fmean(out["means"][f"{a}_seed{s}"][f] for s in ss) for f in FIELDS}
base, frame = out["seed_means"]["base"], out["seed_means"]["frame"]
frac = lambda b, f, u: math.log(b / u) / math.log(b / f)
for a in ("seedonly", "geoonly"):
    if a not in out["seed_means"]:
        continue
    m = out["seed_means"][a]
    out["ratio_to_base"][a] = {f: m[f] / base[f] for f in FIELDS}
    out["ratio_to_frame"][a] = {f: m[f] / frame[f] for f in FIELDS}
    out["fraction_of_frame_gain"][a] = {f: frac(base[f], frame[f], m[f]) for f in FIELDS}
    out["fraction_per_seed"][a] = {str(s): {f: frac(out["means"][f"base_seed{s}"][f], out["means"][f"frame_seed{s}"][f],
                                                    out["means"][f"{a}_seed{s}"][f]) for f in FIELDS}
                                   for s in SEEDS if (a, s) in runs}
aoa = lambda i: float(re.search(r"airFoil2D_SST_[-\d.]+_([-\d.]+)_", i).group(1))
binname = lambda x: "aoa<0" if x < 0 else "0-5" if x < 5 else "5-10" if x < 10 else ">=10"
for b in ("aoa<0", "0-5", "5-10", ">=10"):
    grp = [i for i in ids if binname(aoa(i)) == b]
    if grp:
        out["by_aoa"][b] = {"n": len(grp), **{a: {f: st.fmean(st.fmean(runs[(a, s)][i][f] for i in grp) for s in SEEDS if (a, s) in runs)
                                                    for f in FIELDS} for a in ARMS if a in out["seed_means"]}}
if all(a in out["fraction_of_frame_gain"] for a in ("seedonly", "geoonly")):
    fs, fg = out["fraction_of_frame_gain"]["seedonly"]["pressure_l2"], out["fraction_of_frame_gain"]["geoonly"]["pressure_l2"]
    v = ("seed carries it" if fs >= 0.75 and fg <= 0.35 else "geometry carries it" if fg >= 0.75 and fs <= 0.35
         else "joint" if fs <= 0.5 and fg <= 0.5 else "either suffices" if fs >= 0.75 and fg >= 0.75 else "between")
    out["verdict"] = v
    out["interim"] = any(out["n_seeds"][a] < 2 for a in ("seedonly", "geoonly"))
(RES / "frame_channel_2026-09-27.json").write_text(json.dumps(out, indent=1))
print(f"cases {len(ids)}; seeds {out['n_seeds']}")
for a in ARMS:
    if a in out["seed_means"]:
        print(f"  {a:9s} " + " ".join(f"{f} {out['seed_means'][a][f]:.4f}" for f in FIELDS))
for a in ("seedonly", "geoonly"):
    if a in out["fraction_of_frame_gain"]:
        print(f"  {a:9s} fraction of frame gain " + " ".join(f"{f} {out['fraction_of_frame_gain'][a][f]:.2f}" for f in FIELDS)
              + " | per seed " + str({s: round(d["pressure_l2"], 2) for s, d in out["fraction_per_seed"][a].items()}))
print("verdict:", out.get("verdict"), "(interim)" if out.get("interim") else "")
