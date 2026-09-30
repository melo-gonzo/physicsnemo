"""DEEPJEB reduction (2026-09-22): fp32 per-sample relative L2 of von Mises stress and displacement on the 5,540 clean
DeepJEB++ test files for every scored lane, reduced per rung and per load case (the sample id ends in the load tag), with
GeoTransolver-512 ÷ ISLA ratios and learning-curve exponents over the equal-step ladder.

Input: results/campaigns_2026-09-22/jb/metrics/<lane>/<lane>/metrics.jsonl. Output: results/deepjeb_2026-09-22.json.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MET = ROOT / "campaigns_2026-09-22" / "jb" / "metrics"
FIELDS = ["von_mises", "displacement"]
LOADS = ["ver", "hor", "dia", "tor"]
RUNG_CASES = {"n256": 256, "n1024": 1024, "n4096": 4096, "full": 12451, "full_e40": 12451}


def lane_means(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    cfg = next(r for r in rows if r.get("phase") == "config")
    steps = [r for r in rows if r.get("phase") == "infer_step"]
    out = {"n_cases": len(steps), "epoch": cfg.get("epoch"), "fields": {}, "by_load": {}}
    for f in FIELDS:
        vals = [r["metrics"][f"{f}_l2"] for r in steps if f"{f}_l2" in r["metrics"]]
        out["fields"][f] = statistics.fmean(vals)
        for ld in LOADS:
            v = [r["metrics"][f"{f}_l2"] for r in steps if f"{f}_l2" in r["metrics"] and r["sample_id"].endswith(f"_{ld}")]
            if v:
                out["by_load"].setdefault(ld, {})[f] = statistics.fmean(v)
    return out


def main() -> None:
    lanes = {}
    for p in sorted(MET.glob("jb_*/*/metrics.jsonl")):
        name = p.parent.name
        m = re.match(r"jb_(isla_r[12]|gt512_lr3e3|gt512|gt256)_(n256|n1024|n4096|full_e40|full)_seed(\d+)", name)
        lanes[name] = {"arm": m.group(1), "rung": m.group(2), "seed": int(m.group(3)), **lane_means(p)}
    summary: dict[str, dict] = {}
    for name, d in lanes.items():
        g = summary.setdefault(f"{d['arm']}|{d['rung']}", {"lanes": [], "fields": {}, "by_load": {}})
        g["lanes"].append(name)
        for f, v in d["fields"].items():
            g["fields"].setdefault(f, []).append(v)
        for ld, fv in d["by_load"].items():
            for f, v in fv.items():
                g["by_load"].setdefault(ld, {}).setdefault(f, []).append(v)
    for g in summary.values():
        g["fields"] = {f: {"mean": statistics.fmean(v), "seeds": v} for f, v in g["fields"].items()}
        g["by_load"] = {ld: {f: statistics.fmean(v) for f, v in fv.items()} for ld, fv in g["by_load"].items()}
    ratios = {}
    for k, g in summary.items():
        arm, rung = k.split("|")
        gt = summary.get(f"gt512|{rung}")
        if arm.startswith("isla") and gt:
            ratios[k] = {f: gt["fields"][f]["mean"] / g["fields"][f]["mean"] for f in FIELDS if f in g["fields"] and f in gt["fields"]}
    fits = {}
    for arm in ("isla_r1", "gt512"):
        pts = sorted((RUNG_CASES[k.split("|")[1]], k) for k in summary if k.startswith(arm + "|"))
        if len(pts) >= 3:
            for f in FIELDS:
                xs = [math.log(n) for n, _ in pts]; ys = [math.log(summary[k]["fields"][f]["mean"]) for _, k in pts]
                mx, my = statistics.fmean(xs), statistics.fmean(ys)
                slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
                fits[f"{arm}|{f}"] = {"exponent": -slope, "rungs": [n for n, _ in pts]}
    out = {"protocol": "fp32 inference on the 5,540 clean DeepJEB++ test files (1,385 brackets x 4 load cases), every surface cell, per-sample relative L2 averaged over files; bf16 training on the clean ladder 256 / 1,024 / 4,096 / 12,451 brackets at equal optimizer steps (500 / 125 / 32 / 10 epochs); ISLA R1 = 384x12 cosine fused, K=1 load direction, lr 5e-4; GeoTransolver-512 lr 1e-3 (3e-3 control)",
           "lanes": lanes, "summary": summary, "gt512_over_isla": ratios, "learning_curve_exponents": fits}
    (ROOT / "deepjeb_2026-09-22.json").write_text(json.dumps(out, indent=1))
    for k in sorted(summary):
        g = summary[k]
        print(f"{k:22s}", "  ".join(f"{f}={v['mean']:.4f}" for f, v in g["fields"].items()), f"(seeds {len(g['lanes'])})",
              " loads:", " ".join(f"{ld}:{fv['von_mises']:.3f}" for ld, fv in g["by_load"].items()))
    print("--- GeoTransolver-512 ÷ ISLA:", {k: {f: round(v, 2) for f, v in r.items()} for k, r in ratios.items()})
    print("--- exponents:", {k: round(v["exponent"], 2) for k, v in fits.items()})


if __name__ == "__main__":
    main()
