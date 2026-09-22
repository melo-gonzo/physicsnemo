"""AIRFRANS-INT extrapolation splits (notebook #sec-nb-airfrans-extrap-prereg).

Lanes (af rows 6-13): ISLA cosine 192 at 1e-3 and GeoTransolver-volume at 1e-3 on the `reynolds` split (504 train / 496
test, Reynolds number outside the training band) and the `aoa` split (804 / 196, angle of attack outside the band), seeds
42 / 43, 500 epochs, 100,000 interior queries, float32 scoring on each split's test cases.

Readout per field: growth factor g = extrapolation-split test error / the same architecture's full-split test error
(seed means; paired by seed for the sign test), and the ratio of growth factors g_GT / g_ISLA. Bars: ISLA extrapolates
better on a field if g_GT / g_ISLA > 1.10 with both seed pairs above one; GeoTransolver-volume if < 0.90 with both pairs
below one; otherwise no difference. The absolute GeoTransolver-volume / ISLA ratio on each split is reported without a bar.

usage: python airfrans_extrap_reduce_2026-09-22.py <fetched p200k_fp32 dir> <airfrans_reduction (full split).json> <out.json>
"""
from __future__ import annotations

import glob, json, statistics as st, sys
from pathlib import Path

root, full_path, out_path = (Path(a) for a in sys.argv[1:4])
FULL = json.load(open(full_path))["arms"]
FIELDS = ("pressure_l2", "velocity_l2", "nut_l2")
ARMS = {"isla_cosine_1e-3": ("af_isla_cos_lr1e3_{split}_seed{seed}", "isla_cosine_1e-3"), "gt_volume_1e-3": ("af_gt_vol_lr1e3_{split}_seed{seed}", "gt_volume_1e-3")}
SEEDS = (42, 43)


def summary(run):
    """Float32 infer_summary metrics of one run, or None if unscored."""
    ps = glob.glob(str(root / run / "**" / "metrics.jsonl"), recursive=True)
    if not ps:
        return None
    rows = [json.loads(l) for l in open(ps[0]) if l.strip()]
    s = [r for r in rows if r.get("phase") == "infer_summary"]
    return {f: s[0]["metrics"][f] for f in FIELDS if f in s[0]["metrics"]} if s else None


out = {"study": "AIRFRANS-INT extrapolation splits: growth of the interior error from the full split to the Reynolds and angle-of-attack splits, ISLA cosine 192 (1e-3) and GeoTransolver-volume (1e-3), two seeds, 100,000 queries, float32",
       "splits": {}}
for split in ("reynolds", "aoa"):
    S = {"arms": {}, "growth": {}, "gt_over_isla_on_split": {}, "verdict": {}}
    for arm, (pat, full_key) in ARMS.items():
        runs = {pat.format(split=split, seed=s): summary(pat.format(split=split, seed=s)) for s in SEEDS}
        got = [v for v in runs.values() if v]
        a = {"runs": runs, "status": "complete" if len(got) == 2 else ("partial" if got else "pending")}
        if got:
            for f in FIELDS:
                vals = [v[f] for v in got]; full_seeds = FULL[full_key][f]["seeds"]
                a[f] = {"seeds": vals, "mean": st.mean(vals), "full_split_mean": FULL[full_key][f]["mean"],
                        "growth_mean": st.mean(vals) / FULL[full_key][f]["mean"],
                        "growth_paired": [vals[i] / full_seeds[i] for i in range(len(vals))]}
        S["arms"][arm] = a
    gi, gg = S["arms"]["isla_cosine_1e-3"], S["arms"]["gt_volume_1e-3"]
    if gi["status"] == "complete" and gg["status"] == "complete":
        for f in FIELDS:
            r = gg[f]["growth_mean"] / gi[f]["growth_mean"]
            paired = [gg[f]["growth_paired"][i] / gi[f]["growth_paired"][i] for i in range(2)]
            if r > 1.10 and all(p > 1 for p in paired):
                v = "ISLA extrapolates better"
            elif r < 0.90 and all(p < 1 for p in paired):
                v = "GeoTransolver-volume extrapolates better"
            else:
                v = "no difference (inside 0.90 to 1.10 or seeds disagree)"
            S["growth"][f] = {"g_isla": gi[f]["growth_mean"], "g_gt": gg[f]["growth_mean"], "g_gt_over_g_isla": r, "paired": paired}
            S["gt_over_isla_on_split"][f] = {"mean": gg[f]["mean"] / gi[f]["mean"], "paired": [gg[f]["seeds"][i] / gi[f]["seeds"][i] for i in range(2)]}
            S["verdict"][f] = v
    out["splits"][split] = S
out_path.write_text(json.dumps(out, indent=2))
for split, S in out["splits"].items():
    print(f"== {split}")
    for arm, a in S["arms"].items():
        print(f"  {arm:18s} {a['status']:8s}", {f: (round(a[f]["mean"], 4), round(a[f]["growth_mean"], 2)) for f in FIELDS if f in a})
    for f, g in S["growth"].items():
        print(f"  {f:12s} g_ISLA {g['g_isla']:.2f} g_GT {g['g_gt']:.2f} ratio {g['g_gt_over_g_isla']:.2f} paired {[round(p, 2) for p in g['paired']]} | GT/ISLA on split {S['gt_over_isla_on_split'][f]['mean']:.2f} -> {S['verdict'][f]}")
