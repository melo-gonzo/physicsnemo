"""Generic per-lane reduction (2026-09-23): mean fp32 per-sample relative L2 per field for every scored lane of the given
campaign metric directories (tn, om, v2 and any group without a dedicated reducer), printed and written to
results/lane_means_2026-09-23.json. Component metrics (*_x/_y/_z) are skipped; the aggregate field metric is kept.

usage: python lane_means_2026-09-23.py [group ...]   (default: tn om v2 rb)
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GROUPS = {"tn": "campaigns_2026-09-22/tn/metrics", "om": "campaigns_2026-09-22/om/metrics",
          "v2": "campaigns_2026-09-22/v2/metrics", "rb": "campaigns_2026-09-23/rb/metrics"}
SKIP = re.compile(r"_(x|y|z)_l2$")


def lane_means(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    cfg = next((r for r in rows if r.get("phase") == "config"), {})
    steps = [r for r in rows if r.get("phase") == "infer_step"]
    keys = sorted({k for r in steps for k in r["metrics"] if k.endswith("_l2") and not SKIP.search(k)})
    return {"n_cases": len(steps), "epoch": cfg.get("epoch"),
            "fields": {k[:-3]: statistics.fmean(r["metrics"][k] for r in steps if k in r["metrics"]) for k in keys}}


def main() -> None:
    groups = sys.argv[1:] or list(GROUPS)
    out = json.load(open(ROOT / "lane_means_2026-09-23.json")) if (ROOT / "lane_means_2026-09-23.json").exists() else {}
    for g in groups:
        for p in sorted((ROOT / GROUPS[g]).glob("*/*/metrics.jsonl")):
            d = lane_means(p)
            if d["n_cases"] == 0:
                continue
            out[p.parent.name] = {"group": g, **d}
            print(f"{p.parent.name:44s} n={d['n_cases']:4d}", "  ".join(f"{k}={v:.4f}" for k, v in d["fields"].items()))
    (ROOT / "lane_means_2026-09-23.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
