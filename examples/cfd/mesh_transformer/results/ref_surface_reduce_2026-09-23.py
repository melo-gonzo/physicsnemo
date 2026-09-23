"""REF standings reduction (2026-09-23): per-lane means of every float32 metric written by the fleet scorer for the
reference-configuration lanes (group `ref`: ISLA R1 = 384 x 12, R2 = 384 x 24, the R1 rate control at 1e-3, and
GeoTransolver-512 on DrivAerML Surface 435 cars, the 35-wing HiLiftAeroML rung, DrivAerML Volume and AirfRANS) and
for the REF-BOARD DrivAerML Volume rungs (group `rb`, 54 and 109 training cars).

The mean over cases is the arithmetic mean of the per-case `infer_step` metrics, as every other reduction in
`results/` computes it; a lane whose second seed has not been scored is kept with one seed and the chapter says so.

Input: results/campaigns_2026-09-22/ref/metrics and results/campaigns_2026-09-23/rb/metrics (fetched from AGA).
Output: results/ref_surface_2026-09-23.json.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIRS = [ROOT / "campaigns_2026-09-22" / "ref" / "metrics", ROOT / "campaigns_2026-09-23" / "rb" / "metrics"]
PAT_REF = re.compile(r"^ref_(isla_r1_lr1e3|isla_r1|isla_r2|gt_w512)_(dr435|hl35|drvol|af)_seed(\d+)$")
PAT_RB = re.compile(r"^rb_dr_isla_vol_n(\d+)_seed(\d+)$")
DATASET = {"dr435": "DrivAerML Surface, 435 cars, 200,000 cells", "hl35": "HiLiftAeroML, 35 wings, 200,000 cells",
           "drvol": "DrivAerML Volume, 435 cars, 200,000 cells and 200,000 queries",
           "af": "AirfRANS interior, 800 airfoils, 100,000 queries"}
ARM = {"isla_r1": "ISLA R1 (384 x 12, cosine, fused, 5e-4)", "isla_r2": "ISLA R2 (384 x 24, 5e-4)",
       "isla_r1_lr1e3": "ISLA R1 at 1e-3 (rate control)", "gt_w512": "GeoTransolver-512 (3e-3)"}


def lane_means(path: Path) -> tuple[dict[str, float], int]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    steps = [r["metrics"] for r in rows if r.get("phase") == "infer_step"]
    fields = sorted({k for m in steps for k in m})
    return {f: statistics.fmean(m[f] for m in steps if f in m) for f in fields}, len(steps)


def main() -> None:
    lanes: dict[str, dict] = {}
    for d in DIRS:
        for p in sorted(d.glob("*/*/metrics.jsonl")):
            name = p.parent.name
            if m := PAT_REF.match(name):
                arm, ds, seed = m.group(1), m.group(2), int(m.group(3))
            elif m := PAT_RB.match(name):
                arm, ds, seed = "isla_r1", f"drvol_n{m.group(1)}", int(m.group(2))
            else:
                continue
            means, n = lane_means(p)
            if not n:
                continue
            lanes[name] = {"arm": arm, "dataset": ds, "seed": seed, "n_cases": n, "fields": means}
    summary: dict[str, dict] = {}
    for name, l in sorted(lanes.items()):
        g = summary.setdefault(f"{l['arm']}|{l['dataset']}", {"lanes": [], "seeds": [], "fields": {}})
        g["lanes"].append(name); g["seeds"].append(l["seed"])
        for f, v in l["fields"].items():
            g["fields"].setdefault(f, {"seeds": []})["seeds"].append(v)
    for g in summary.values():
        for f in g["fields"].values():
            f["mean"] = statistics.fmean(f["seeds"])
        g["n_seeds"] = len(g["lanes"])
    out = {"protocol": "float32 scoring of the final-epoch (500) checkpoints on the validation split by the fleet scorer; "
                       "per-case relative L2 (area-weighted on the surface tasks), arithmetic mean over cases, then over seeds; "
                       "bf16 training; every ISLA lane under the `code_ref` snapshot with the class defaults (cosine routing, fused "
                       "geometry kernel, RMS-pairwise-distance unit, `source_total` query mass on the interior tasks)",
           "arms": ARM, "datasets": DATASET, "lanes": lanes, "summary": summary}
    (ROOT / "ref_surface_2026-09-23.json").write_text(json.dumps(out, indent=1))
    for k, g in summary.items():
        head = {f: round(g["fields"][f]["mean"], 5) for f in ("pressure_l2", "velocity_l2", "nut_l2") if f in g["fields"]}
        print(f"{k:<24s} seeds={g['seeds']} {head}")


if __name__ == "__main__":
    main()
