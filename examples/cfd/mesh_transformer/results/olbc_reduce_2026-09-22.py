"""OLBC reduction (2026-09-22): fp32 per-sample relative L2 on the 256 test cases of poisson-square-bc5 and
elasticity-circlehollow-m1 for every scored OLBC lane, reduced to per-rung means, GeoTransolver-512 ÷ ISLA ratios and
learning-curve exponents (error ~ cases^-alpha over the rungs an arm has).

Input: results/campaigns_2026-09-22/ol/metrics/<lane>/<lane>/metrics.jsonl (fetched from AGA p200k_fp32/).
Output: results/olbc_2026-09-22.json.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MET = ROOT / "campaigns_2026-09-22" / "ol" / "metrics"
SKIP = re.compile(r"_(x|y|z)_l2$")


def lane_means(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    cfg = next(r for r in rows if r.get("phase") == "config")
    steps = [r for r in rows if r.get("phase") == "infer_step"]
    keys = sorted({k for r in steps for k in r["metrics"] if k.endswith("_l2") and not SKIP.search(k)})
    out = {"n_cases": len(steps), "epoch": cfg.get("epoch"), "fields": {}}
    for k in keys:
        vals = [r["metrics"][k] for r in steps if k in r["metrics"]]
        out["fields"][k[:-3]] = {"mean": statistics.fmean(vals), "median": statistics.median(vals)}
    return out


def main() -> None:
    lanes = {}
    for p in sorted(MET.glob("ol_*/*/metrics.jsonl")):
        name = p.parent.name
        m = re.match(r"ol_(poisson|elasticity)_(isla_r[12](?:_lr25e4)?|gt512|gt256)_n(\d+)_seed(\d+)", name)
        lanes[name] = {"set": m.group(1), "arm": m.group(2), "n": int(m.group(3)), "seed": int(m.group(4)), **lane_means(p)}
    summary: dict[str, dict] = {}
    for name, d in lanes.items():
        g = summary.setdefault(f"{d['set']}|{d['arm']}|n{d['n']}", {"lanes": [], "fields": {}})
        g["lanes"].append(name)
        for f, v in d["fields"].items():
            g["fields"].setdefault(f, []).append(v["mean"])
    for g in summary.values():
        g["fields"] = {f: {"mean": statistics.fmean(v), "seeds": v} for f, v in g["fields"].items()}
    ratios = {}
    for k, g in summary.items():
        s, arm, n = k.split("|")
        gt = summary.get(f"{s}|gt512|{n}")
        if arm.startswith("isla") and gt:
            ratios[k] = {f: gt["fields"][f]["mean"] / g["fields"][f]["mean"] for f in g["fields"] if f in gt["fields"]}
    fits = {}
    for s in ("poisson", "elasticity"):
        for arm in ("isla_r1", "gt512"):
            rungs = sorted(int(k.split("|n")[1]) for k in summary if k.startswith(f"{s}|{arm}|"))
            if len(rungs) < 3:
                continue
            for f in summary[f"{s}|{arm}|n{rungs[0]}"]["fields"]:
                pts = [(math.log(n), math.log(summary[f"{s}|{arm}|n{n}"]["fields"][f]["mean"])) for n in rungs]
                mx = statistics.fmean(x for x, _ in pts); my = statistics.fmean(y for _, y in pts)
                slope = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)
                fits[f"{s}|{arm}|{f}"] = {"exponent": -slope, "rungs": rungs}
    out = {"protocol": "fp32 inference on the 256 OLBC test cases per set, every node queried, per-sample relative L2 averaged over cases; bf16 training; 512- and 2,048-case rungs 500 epochs, 8,448-case rung 120 epochs (equal optimizer steps to 2,048 x 500); ISLA R1 = 384x12 cosine, lr 5e-4 (K=0 Poisson, K=2 frame on elasticity, eager); GeoTransolver-512 lr 1e-3",
           "lanes": lanes, "summary": summary, "gt512_over_isla": ratios, "learning_curve_exponents": fits}
    (ROOT / "olbc_2026-09-22.json").write_text(json.dumps(out, indent=1))
    for k in sorted(summary):
        print(f"{k:28s}", "  ".join(f"{f}={v['mean']:.4f}" for f, v in summary[k]["fields"].items()), f"(seeds {len(summary[k]['lanes'])})")
    print("--- GeoTransolver-512 ÷ ISLA")
    for k in sorted(ratios):
        print(f"{k:28s}", "  ".join(f"{f}={v:.2f}" for f, v in ratios[k].items()))
    for k, v in sorted(fits.items()):
        print(f"{k:34s} alpha={v['exponent']:.2f}")


if __name__ == "__main__":
    main()
