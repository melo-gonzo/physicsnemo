"""Per-case anatomy of the 35-wing deficit (2026-09-27): float32 surface-pressure relative L2 per validation case on the
HiLiftAeroML 210-case rung at 200,000 cells, ISLA (reference configuration, ref_isla_r1_hl210_seed{42,43}) against
GeoTransolver-512 (the parameter match, ref_gt_w512_hl210_seed{42,43}), seed-mean per case, grouped by the case's angle of
attack (parsed from the sample id, geo_LHCnnn_AoA_a). Also the Spearman rank correlation of the two models' per-case errors
(do they find the same cases hard?) and the distribution of the per-case ratio. Input: campaigns_2026-09-22/ref/metrics.
Output: hl210_percase_2026-09-27.json. The 210-case twin of hl35_percase_2026-09-23.py (WING-ANATOMY; bar: a widening of at least 0.08 between the 4-14 and 16-22 degree ratios reproduces the 35-wing location)."""
from __future__ import annotations
import glob, json, re, statistics as st
from collections import defaultdict
from pathlib import Path
RES = Path(__file__).resolve().parent
ARMS = {"isla_r1": "ref_isla_r1_hl210_seed4*", "gt512": "ref_gt_w512_hl210_seed4*", "isla_r2": "ref_isla_r2_hl210_seed4*"}
def load(pattern):
    per, lanes = defaultdict(list), []
    for p in sorted((RES / "campaigns_2026-09-22" / "ref" / "metrics").glob(f"{pattern}/*/metrics.jsonl")):
        lanes.append(p.parent.name)
        for l in p.read_text().splitlines():
            if not l.strip(): continue
            r = json.loads(l)
            if r.get("phase") == "infer_step": per[r["sample_id"]].append(r["metrics"]["pressure_l2"])
    return {k: st.fmean(v) for k, v in per.items()}, lanes
data = {a: load(pat) for a, pat in ARMS.items()}
ids = sorted(set.intersection(*(set(d) for d, _ in data.values())))
aoa = lambda s: int(re.search(r"AoA_(\d+)", s).group(1)); geo = lambda s: re.search(r"geo_(LHC\d+)", s).group(1)
out = {"protocol": __doc__, "n_cases": len(ids), "n_geometries": len({geo(s) for s in ids}), "lanes": {a: l for a, (_, l) in data.items()}, "by_aoa": {}, "per_case": {}}
for a in sorted({aoa(s) for s in ids}):
    grp = [s for s in ids if aoa(s) == a]
    row = {"n": len(grp)}
    for arm, (d, _) in data.items(): row[arm] = st.fmean(d[s] for s in grp)
    row["gt512_over_isla_r1"] = row["gt512"] / row["isla_r1"]
    out["by_aoa"][str(a)] = row
isla, gt = data["isla_r1"][0], data["gt512"][0]
def rank(d): return {s: i for i, s in enumerate(sorted(ids, key=lambda s: d[s]))}
ri, rg = rank(isla), rank(gt); n = len(ids)
out["spearman_isla_r1_gt512"] = 1 - 6 * sum((ri[s] - rg[s]) ** 2 for s in ids) / (n * (n * n - 1))
ratios = sorted(gt[s] / isla[s] for s in ids)
out["per_case_ratio_gt512_over_isla_r1"] = {"p10": ratios[n // 10], "median": ratios[n // 2], "p90": ratios[9 * n // 10], "isla_better_count": sum(r > 1 for r in ratios)}
out["per_case"] = {s: {"aoa": aoa(s), "geometry": geo(s), **{arm: d[s] for arm, (d, _) in data.items()}} for s in ids}
lo = [out["by_aoa"][str(a)]["gt512_over_isla_r1"] for a in (4, 6, 8, 10, 12, 14) if str(a) in out["by_aoa"]]
hi = [out["by_aoa"][str(a)]["gt512_over_isla_r1"] for a in (16, 18, 20, 22) if str(a) in out["by_aoa"]]
out["summary"] = {"ratio_aoa_4_to_14_mean": st.fmean(lo), "ratio_aoa_16_to_22_mean": st.fmean(hi)}
(RES / "hl210_percase_2026-09-27.json").write_text(json.dumps(out, indent=1))
print(f"cases {n}, geometries {out['n_geometries']}; GT-512 ÷ ISLA by AoA:", {a: round(r['gt512_over_isla_r1'], 2) for a, r in out['by_aoa'].items()})
print(f"AoA 4-14 mean {out['summary']['ratio_aoa_4_to_14_mean']:.2f}, AoA 16-22 mean {out['summary']['ratio_aoa_16_to_22_mean']:.2f}; Spearman {out['spearman_isla_r1_gt512']:.2f}; ISLA better on {out['per_case_ratio_gt512_over_isla_r1']['isla_better_count']}/{n}")
print("R2 by AoA:", {a: round(r['isla_r2'] / r['isla_r1'], 2) for a, r in out['by_aoa'].items()})
