"""Per-car anatomy of the 435-car standing (2026-09-23): float32 surface-pressure relative L2 per validation car on the
DrivAerML Surface task at 200,000 cells, ISLA (reference configuration, ref_isla_r1_dr435_seed{42,43}) against the
parameter-matched GeoTransolver-512 (ref_gt_w512_dr435_seed{42,43}), seed-mean per car: Spearman rank correlation of the
two models' per-car errors, distribution of the per-car ratio, count of cars on which ISLA is the more accurate model.
Input: campaigns_2026-09-22/ref/metrics. Output: drivaer435_percase_2026-09-23.json."""
from __future__ import annotations
import json, statistics as st
from collections import defaultdict
from pathlib import Path
RES = Path(__file__).resolve().parent
def load(pattern):
    per = defaultdict(list)
    for p in sorted((RES / "campaigns_2026-09-22" / "ref" / "metrics").glob(f"{pattern}/*/metrics.jsonl")):
        for l in p.read_text().splitlines():
            if not l.strip(): continue
            r = json.loads(l)
            if r.get("phase") == "infer_step": per[r["sample_id"]].append(r["metrics"]["pressure_l2"])
    return {k: st.fmean(v) for k, v in per.items()}
isla, gt = load("ref_isla_r1_dr435_seed4*"), load("ref_gt_w512_dr435_seed4*")
ids = sorted(set(isla) & set(gt)); n = len(ids)
rank = lambda d: {s: i for i, s in enumerate(sorted(ids, key=lambda s: d[s]))}
ri, rg = rank(isla), rank(gt)
ratios = sorted(gt[s] / isla[s] for s in ids); errs = sorted(isla[s] for s in ids)
out = {"protocol": __doc__, "n_cars": n,
       "spearman_isla_r1_gt512": 1 - 6 * sum((ri[s] - rg[s]) ** 2 for s in ids) / (n * (n * n - 1)),
       "per_car_ratio_gt512_over_isla_r1": {"p10": ratios[n // 10], "median": ratios[n // 2], "p90": ratios[9 * n // 10], "isla_better_count": sum(r > 1 for r in ratios)},
       "isla_r1_per_car_error": {"p10": errs[n // 10], "median": errs[n // 2], "p90": errs[9 * n // 10], "max": errs[-1]},
       "per_car": {s: {"isla_r1": isla[s], "gt512": gt[s]} for s in ids}}
(RES / "drivaer435_percase_2026-09-23.json").write_text(json.dumps(out, indent=1))
print(f"cars {n}; Spearman {out['spearman_isla_r1_gt512']:.2f}; ratio p10/median/p90 {ratios[n//10]:.2f}/{ratios[n//2]:.2f}/{ratios[9*n//10]:.2f}; ISLA better {out['per_car_ratio_gt512_over_isla_r1']['isla_better_count']}/{n}")
