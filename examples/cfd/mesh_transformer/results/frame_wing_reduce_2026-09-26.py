"""FRAME-WING reduction (prereg @sec-nb-frame-wing-launch, bar from @sec-nb-hl35-percase): the body frame (span y and up z
beside the far-field direction, K = 3) on the 35-wing HiLiftAeroML rung at 200,000 cells, ISLA R1 (fa rows 8-9, code_v4)
against the reference pair (ref_isla_r1_hl35, code_ref) and GeoTransolver-512 (ref_gt_w512_hl35). Float32 per-case
pressure relative L2 on the 180 validation wings, seed means, grouped by angle of attack (parsed from the sample id). The
bar: the log-gap to GeoTransolver-512 narrows by at least 10% in either angle group (4-14 or 16-22 degrees), both seeds.
usage: python frame_wing_reduce_2026-09-26.py [out.json]"""
from __future__ import annotations
import glob, json, math, re, statistics as st, sys
from collections import defaultdict
from pathlib import Path
RES = Path(__file__).resolve().parent
SRC = {"isla_r1": ("campaigns_2026-09-22/ref/metrics", "ref_isla_r1_hl35_seed{}"), "gt512": ("campaigns_2026-09-22/ref/metrics", "ref_gt_w512_hl35_seed{}"),
       "isla_frame": ("campaigns_2026-09-23/fa/metrics", "fa_hl35_frame_seed{}")}
FIELDS = ["pressure_l2", "temperature_l2", "density_l2", "velocity_l2", "tau_wall_l2"]
def load(rel, pat, seed):
    ps = glob.glob(str(RES / rel / pat.format(seed) / "**" / "metrics.jsonl"), recursive=True) + glob.glob(str(RES / rel / pat.format(seed) / "metrics.jsonl"))
    per = {}
    for l in open(sorted(set(ps))[0]):
        r = json.loads(l)
        if r.get("phase") == "infer_step":
            per[re.sub(r"^\d+_", "", r["sample_id"])] = r["metrics"]  # strip the manifest index prefix; match by case name
    return per
data = {a: {s: load(rel, pat, s) for s in (42, 43)} for a, (rel, pat) in SRC.items()}
ids = sorted(set.intersection(*(set(d) for arm in data.values() for d in arm.values())))
aoa = lambda s: int(re.search(r"AoA_(\d+)", s).group(1))
def mean_over(arm, seed, cases, f="pressure_l2"): return st.fmean(data[arm][seed][c][f] for c in cases)
out = {"protocol": __doc__, "n_cases": len(ids), "fields": {}, "by_aoa": {}, "bar": {}}
for f in FIELDS:
    if f not in next(iter(data["isla_frame"][42].values())): continue
    row = {a: {"seeds": {s: mean_over(a, s, ids, f) for s in (42, 43)}} for a in data}
    for a in row: row[a]["mean"] = st.fmean(row[a]["seeds"].values())
    row["frame_over_r1"] = row["isla_frame"]["mean"] / row["isla_r1"]["mean"]
    row["gt512_over_r1"] = row["gt512"]["mean"] / row["isla_r1"]["mean"]; row["gt512_over_frame"] = row["gt512"]["mean"] / row["isla_frame"]["mean"]
    out["fields"][f] = row
groups = {"4_to_14": [c for c in ids if 4 <= aoa(c) <= 14], "16_to_22": [c for c in ids if 16 <= aoa(c) <= 22]}
for a in sorted({aoa(c) for c in ids}):
    cs = [c for c in ids if aoa(c) == a]
    r = {arm: st.fmean(mean_over(arm, s, cs) for s in (42, 43)) for arm in data}
    out["by_aoa"][str(a)] = {"n": len(cs), **r, "gt512_over_r1": r["gt512"] / r["isla_r1"], "gt512_over_frame": r["gt512"] / r["isla_frame"], "frame_over_r1": r["isla_frame"] / r["isla_r1"]}
for g, cs in groups.items():
    per_seed = {}
    for s in (42, 43):
        r1, fr, gt = mean_over("isla_r1", s, cs), mean_over("isla_frame", s, cs), st.fmean(mean_over("gt512", t, cs) for t in (42, 43))
        gap_r1, gap_fr = math.log(r1 / gt), math.log(fr / gt)
        per_seed[s] = {"isla_r1": r1, "isla_frame": fr, "gt512": gt, "log_gap_r1": gap_r1, "log_gap_frame": gap_fr, "fraction_narrowed": (gap_r1 - gap_fr) / gap_r1 if gap_r1 > 0 else None, "frame_over_r1": fr / r1}
    out["bar"][g] = {"n": len(cs), "per_seed": per_seed, "fraction_narrowed_mean": st.fmean(v["fraction_narrowed"] for v in per_seed.values() if v["fraction_narrowed"] is not None),
                     "both_seeds_at_least_0.10": all((v["fraction_narrowed"] or 0) >= 0.10 for v in per_seed.values())}
out["verdict"] = "frame pays (bar met)" if any(g["both_seeds_at_least_0.10"] for g in out["bar"].values()) else "frame does not pay (bar missed)"
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else RES / "frame_wing_2026-09-26.json", "w"), indent=1)
p = out["fields"]["pressure_l2"]
print(f"cases {len(ids)}; pressure: R1 {p['isla_r1']['mean']:.4f} frame {p['isla_frame']['mean']:.4f} (seeds {[round(v,4) for v in p['isla_frame']['seeds'].values()]}) GT {p['gt512']['mean']:.4f}; frame/R1 {p['frame_over_r1']:.3f}; GT/R1 {p['gt512_over_r1']:.2f} GT/frame {p['gt512_over_frame']:.2f}")
for g, v in out["bar"].items(): print(g, {s: round(x["fraction_narrowed"], 3) if x["fraction_narrowed"] is not None else None for s, x in v["per_seed"].items()}, v["both_seeds_at_least_0.10"])
print({f: round(r["frame_over_r1"], 3) for f, r in out["fields"].items()}); print(out["verdict"])
