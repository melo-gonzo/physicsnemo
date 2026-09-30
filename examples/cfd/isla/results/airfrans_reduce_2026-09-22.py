"""AIRFRANS-INT reduction (prereg #sec-nb-airfrans-prereg): ISLA interior (linear 5e-4, cosine 1e-3) and
GeoTransolver-volume (1e-3) on the AirfRANS full split (800 train / 200 test), 100,000 interior queries per case,
float32, two seeds. Per field (pressure, velocity, nut): seed values, means, GeoTransolver-volume ÷ ISLA, and the
preregistered verdict: reproduced if the ratio is on the same side of one as on DrivAerML (pressure < 1, velocity > 1)
with both seed-paired ratios agreeing in sign; refuted for a field if it crosses one with seeds agreeing; nut has no prior.

usage: python airfrans_reduce.py <fetched fp32 dir> <out.json>
"""
from __future__ import annotations

import glob, json, statistics as st, sys
from pathlib import Path

root, out_path = Path(sys.argv[1]), Path(sys.argv[2])
FIELDS = ("pressure_l2", "velocity_l2", "nut_l2")
ARMS = {"isla_linear_5e-4": "af_isla_lin_lr5e4_seed{}", "isla_cosine_1e-3": "af_isla_cos_lr1e3_seed{}", "gt_volume_1e-3": "af_gt_vol_lr1e3_seed{}"}
PRIOR = {"pressure_l2": "<1", "velocity_l2": ">1", "nut_l2": None}   # DrivAerML: GT-vol / ISLA = 0.87 / 1.42 / 0.98


def summary(run):
    ps = glob.glob(str(root / run / "**" / "metrics.jsonl"), recursive=True)
    if not ps:
        return None
    rows = [json.loads(l) for l in open(ps[0]) if l.strip()]
    s = [r for r in rows if r.get("phase") == "infer_summary"]
    return {f: s[0]["metrics"][f] for f in FIELDS if f in s[0]["metrics"]} if s else None


arms = {}
for name, pat in ARMS.items():
    runs = {pat.format(s): summary(pat.format(s)) for s in (42, 43)}
    a = {"runs": runs, "status": "complete" if all(runs.values()) else "pending"}
    if a["status"] == "complete":
        for f in FIELDS:
            vals = [r[f] for r in runs.values()]
            a[f] = {"seeds": vals, "mean": st.mean(vals), "seed_spread": abs(vals[0] - vals[1]) / st.mean(vals)}
    arms[name] = a

verdicts = {}
gt = arms["gt_volume_1e-3"]
for isla in ("isla_linear_5e-4", "isla_cosine_1e-3"):
    v = {}
    if arms[isla]["status"] == "complete" and gt["status"] == "complete":
        for f in FIELDS:
            ratio = gt[f]["mean"] / arms[isla][f]["mean"]
            paired = [gt[f]["seeds"][i] / arms[isla][f]["seeds"][i] for i in range(2)]
            agree = (paired[0] > 1) == (paired[1] > 1)
            prior = PRIOR[f]
            if prior is None:
                verdict = "no prior; reported"
            elif not agree:
                verdict = "seeds disagree on the side of one; undecided"
            else:
                same = (ratio < 1) if prior == "<1" else (ratio > 1)
                verdict = "reproduced" if same else "refuted"
            v[f] = {"gt_over_isla": ratio, "paired_ratios": paired, "seeds_agree": agree, "drivaerml_prior": prior, "verdict": verdict}
    verdicts[isla] = v

out = {"study": "AIRFRANS-INT: ISLA interior (linear 5e-4, cosine 1e-3) and GeoTransolver-volume (1e-3) on the AirfRANS full split, 800 train / 200 test cases, 100,000 interior queries per case, 500 epochs, float32, two seeds",
       "drivaerml_reference_ratios": {"pressure_l2": 0.87, "velocity_l2": 1.42, "nut_l2": 0.98, "source": "results/ratevol_reduction_2026-09-20.json (200,000 cells, ISLA at 5e-4)"},
       "arms": arms, "verdicts": verdicts}
out_path.write_text(json.dumps(out, indent=2))
for name, a in arms.items():
    print(name, a["status"], {f: (round(a[f]["mean"], 4), [round(x, 4) for x in a[f]["seeds"]]) for f in FIELDS if f in a})
for isla, v in verdicts.items():
    for f, r in v.items():
        print(f"  GT-vol / {isla} {f}: {r['gt_over_isla']:.3f} (paired {r['paired_ratios'][0]:.3f}, {r['paired_ratios'][1]:.3f}) -> {r['verdict']}")
