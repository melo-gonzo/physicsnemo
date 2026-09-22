"""SOLIDS-FRAME reduction (2026-09-22): fp32 per-sample relative L2 on PLAID Tensile2d val_100 for every scored
Tensile2d lane (frameless ISLA, frame1/frame2/leak/nosig12 variants, frame-configured ISLA ladder, GeoTransolver-512),
reduced to per-field means and GeoTransolver ÷ ISLA ratios.

Input: results/campaigns_2026-09-22/sf/metrics/<lane>/<lane>/metrics.jsonl (fetched from AGA p200k_fp32/).
Output: results/solids_frame_2026-09-22.json.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MET = ROOT / "campaigns_2026-09-22" / "sf" / "metrics"
FIELDS = ["sig11", "sig22", "sig12", "displacement", "q"]


def lane_means(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    cfg = next(r for r in rows if r.get("phase") == "config")
    steps = [r for r in rows if r.get("phase") == "infer_step"]
    out = {"n_cases": len(steps), "epoch": cfg.get("epoch"), "model": cfg.get("model")}
    for f in FIELDS:
        vals = [r["metrics"][f"{f}_l2"] for r in steps if f"{f}_l2" in r["metrics"]]
        if vals:
            out[f] = {"mean": statistics.fmean(vals), "median": statistics.median(vals)}
    return out


def parse(name: str) -> dict:
    m = re.match(r"(so|sf)_t2d_(isla_r[12]|gt512|gt256)(?:_(frame1|frame2|leak|nosig12|frame))?_n(\d+)_seed(\d+)", name)
    arch, variant, n, seed = m.group(2), m.group(3) or "noframe", int(m.group(4)), int(m.group(5))
    if arch.startswith("gt"):
        variant = "gt"
    if variant == "frame":
        variant = "frame2"  # so_*_frame_* lanes use isla_tensile2d_frame2.yaml
    return {"arch": arch, "variant": variant, "n": n, "seed": seed}


def main() -> None:
    lanes = {}
    for p in sorted(MET.glob("*/*/metrics.jsonl")):
        name = p.parent.name
        lanes[name] = {**parse(name), **lane_means(p)}

    # group: (arch-or-gt, variant, n) -> seeds
    groups: dict[tuple, dict] = {}
    for name, d in lanes.items():
        key = (d["arch"], d["variant"], d["n"])
        g = groups.setdefault(key, {"lanes": [], "seeds": {}})
        g["lanes"].append(name)
        g["seeds"][d["seed"]] = {f: d[f]["mean"] for f in FIELDS if f in d}
    summary = {}
    for (arch, variant, n), g in groups.items():
        per_field = {}
        for f in FIELDS:
            vals = [s[f] for s in g["seeds"].values() if f in s]
            if vals:
                per_field[f] = {"mean": statistics.fmean(vals), "seeds": vals}
        summary[f"{arch}|{variant}|n{n}"] = {"lanes": g["lanes"], "fields": per_field}

    # GeoTransolver-512 ÷ ISLA per rung and variant (mean over seeds of each)
    ratios = {}
    for key, s in summary.items():
        arch, variant, n = key.split("|")
        if arch.startswith("gt"):
            continue
        gt = summary.get(f"gt512|gt|{n}")
        if not gt:
            continue
        ratios[key] = {
            f: gt["fields"][f]["mean"] / s["fields"][f]["mean"]
            for f in FIELDS
            if f in s["fields"] and f in gt["fields"]
        }
    out = {
        "protocol": "fp32 inference on val_100 (100 held-out PLAID Tensile2d cases), every node queried, per-sample relative L2 averaged over cases; training bf16, 500 epochs; ISLA R1 = 384x12 cosine, lr 5e-4; GeoTransolver-512 lr 1e-3",
        "variants": {
            "noframe": "ISLA with no global vector (K=0, eager)",
            "frame1": "ISLA with +y (load axis) as one global vector (K=1, fused)",
            "frame2": "ISLA with (+x,+y) laboratory frame as two global vectors (K=2, eager); the canonical Tensile2d configuration",
            "leak": "ISLA K=0 with absolute coordinates as extra query scalars (diagnostic; breaks the invariance contract)",
            "nosig12": "ISLA K=0 without the sigma_12 head",
        },
        "lanes": lanes,
        "summary": summary,
        "gt512_over_isla": ratios,
    }
    (ROOT / "solids_frame_2026-09-22.json").write_text(json.dumps(out, indent=1))
    for key in sorted(summary):
        f = summary[key]["fields"]
        print(f"{key:28s}", "  ".join(f"{k}={v['mean']:.4f}" for k, v in f.items() if k != "q"), f"(seeds {len(summary[key]['lanes'])})")
    print("--- GeoTransolver-512 ÷ ISLA")
    for key in sorted(ratios):
        print(f"{key:28s}", "  ".join(f"{k}={v:.2f}" for k, v in ratios[key].items() if k != "q"))


if __name__ == "__main__":
    main()
