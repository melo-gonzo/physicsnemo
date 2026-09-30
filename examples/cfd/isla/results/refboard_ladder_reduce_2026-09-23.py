"""REF-BOARD DrivAerML ladder reduction (2026-09-23): float32 pressure relative L2 at 200,000 cells on the 48 validation
cars, per training-case rung, for ISLA R1 (rb lanes), the earlier 192-wide ISLA (p200k_dr_isla_n*), GeoTransolver-256 at
1e-3 and 3e-3 (p200k_dr_gt_unit_n*), Transolver-256 (p200k_dr_transolver_unit_n*), plus the 435-car standings from REF.

Input: results/campaigns_2026-09-23/rb/metrics and results/campaigns_2026-09-22/lad/metrics (fetched from AGA p200k_fp32/).
Output: results/refboard_ladder_2026-09-23.json.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIRS = [ROOT / "campaigns_2026-09-23" / "rb" / "metrics", ROOT / "campaigns_2026-09-22" / "lad" / "metrics"]
PAT = re.compile(r"^(rb_dr_isla|p200k_dr_isla|p200k_dr_gt_unit|p200k_dr_transolver_unit)_n(\d+)(_lr3e3)?_seed(\d+)$")
ARM = {"rb_dr_isla": "isla_r1", "p200k_dr_isla": "isla_192", "p200k_dr_gt_unit": "gt256", "p200k_dr_transolver_unit": "transolver256"}


def mean_field(path: Path, field: str) -> float | None:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    vals = [r["metrics"][field] for r in rows if r.get("phase") == "infer_step" and field in r["metrics"]]
    return statistics.fmean(vals) if vals else None


def main() -> None:
    summary: dict[str, dict] = {}
    for d in DIRS:
        for p in sorted(d.glob("*/*/metrics.jsonl")):
            m = PAT.match(p.parent.name)
            if not m:
                continue
            arm = ARM[m.group(1)] + ("_lr3e3" if m.group(3) else "")
            n = int(m.group(2))
            v = mean_field(p, "pressure_l2")
            if v is None:
                continue
            g = summary.setdefault(f"{arm}|n{n}", {"lanes": [], "seeds": []})
            g["lanes"].append(p.parent.name); g["seeds"].append(v)
    for g in summary.values():
        g["mean"] = statistics.fmean(g["seeds"])
    rungs = sorted({int(k.split("|n")[1]) for k in summary})
    fits = {}
    for arm in sorted({k.split("|")[0] for k in summary}):
        pts = [(n, summary[f"{arm}|n{n}"]["mean"]) for n in rungs if f"{arm}|n{n}" in summary]
        if len(pts) >= 3:
            xs = [math.log(n) for n, _ in pts]; ys = [math.log(e) for _, e in pts]
            mx, my = statistics.fmean(xs), statistics.fmean(ys)
            fits[arm] = {"exponent": -sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs), "rungs": [n for n, _ in pts]}
    out = {"protocol": "float32 pressure relative L2 on the 48 DrivAerML validation cars at 200,000 cells, 500 epochs, seed means; R1 = 384x12 cosine fused 5e-4 (rb lanes, 2026-09-23); 192-wide = isla_surface_reference at 1e-3 linear routing (p200k fleet); GeoTransolver-256 at 1e-3 and 3e-3; Transolver-256 at 3e-3",
           "summary": summary, "learning_curve_exponents": fits}
    (ROOT / "refboard_ladder_2026-09-23.json").write_text(json.dumps(out, indent=1))
    for n in rungs:
        row = {k.split("|")[0]: v["mean"] for k, v in summary.items() if k.endswith(f"|n{n}")}
        print(f"n{n:<4d}", "  ".join(f"{a}={row[a]:.4f}" for a in sorted(row)))
    print("exponents", {k: round(v["exponent"], 2) for k, v in fits.items()})


if __name__ == "__main__":
    main()
