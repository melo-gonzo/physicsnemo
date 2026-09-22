"""SOLIDS lug reduction (2026-09-22): fp32 per-sample relative L2 of von Mises stress on the 600 GINOT Bracket Lug test
cases for every scored lug lane (ISLA R1 / R2, GeoTransolver-512 / 256) along the 300 / 600 / 1,200 / 2,400 ladder,
reduced to per-rung means, GeoTransolver ÷ ISLA ratios and learning-curve exponents.

Input: results/campaigns_2026-09-22/so/metrics/<lane>/<lane>/metrics.jsonl (fetched from AGA p200k_fp32/).
Output: results/solids_lug_2026-09-22.json.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MET = ROOT / "campaigns_2026-09-22" / "so" / "metrics"


def lane_mean(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    cfg = next(r for r in rows if r.get("phase") == "config")
    steps = [r for r in rows if r.get("phase") == "infer_step"]
    vals = [r["metrics"]["von_mises_l2"] for r in steps if "von_mises_l2" in r["metrics"]]
    return {"n_cases": len(vals), "epoch": cfg.get("epoch"), "mean": statistics.fmean(vals), "median": statistics.median(vals)}


def main() -> None:
    lanes = {}
    for p in sorted(MET.glob("so_lug_*/*/metrics.jsonl")):
        name = p.parent.name
        m = re.match(r"so_lug_(isla_r[12]|gt512|gt256)_n(\d+)_seed(\d+)", name)
        lanes[name] = {"arm": m.group(1), "n": int(m.group(2)), "seed": int(m.group(3)), **lane_mean(p)}
    summary: dict[str, dict] = {}
    for name, d in lanes.items():
        g = summary.setdefault(f"{d['arm']}|n{d['n']}", {"lanes": [], "seeds": []})
        g["lanes"].append(name); g["seeds"].append(d["mean"])
    for g in summary.values():
        g["mean"] = statistics.fmean(g["seeds"])
    ratios = {k: summary[f"gt512|n{k.split('|n')[1]}"]["mean"] / v["mean"] for k, v in summary.items()
              if k.startswith("isla") and f"gt512|n{k.split('|n')[1]}" in summary}
    fits = {}
    for arm in ("isla_r1", "gt512"):
        pts = sorted((int(k.split("|n")[1]), v["mean"]) for k, v in summary.items() if k.startswith(arm + "|"))
        if len(pts) >= 3:
            xs = [math.log(n) for n, _ in pts]; ys = [math.log(e) for _, e in pts]
            mx, my = statistics.fmean(xs), statistics.fmean(ys)
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
            fits[arm] = {"exponent": -slope, "rungs": [n for n, _ in pts]}
    out = {"protocol": "fp32 inference on the 600 GINOT Bracket Lug test cases, every node queried, per-sample relative L2 of von Mises stress averaged over cases; bf16 training, 500 epochs; ISLA R1 = 384x12 cosine (K=0, eager), lr 5e-4; GeoTransolver-512 lr 1e-3",
           "lanes": lanes, "summary": summary, "gt512_over_isla": ratios, "learning_curve_exponents": fits}
    (ROOT / "solids_lug_2026-09-22.json").write_text(json.dumps(out, indent=1))
    for k in sorted(summary):
        print(f"{k:16s} mean={summary[k]['mean']:.4f} seeds={[round(s, 4) for s in summary[k]['seeds']]}")
    print("ratios", {k: round(v, 2) for k, v in ratios.items()})
    print("exponents", {k: round(v["exponent"], 2) for k, v in fits.items()})


if __name__ == "__main__":
    main()
