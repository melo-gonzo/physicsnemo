"""E3b ANCHOR readout: AF-UNIFORM native / posed / area rel-L2 per run and seed means per arm (PREREG "Metrics").

Per run and protocol: the mean over the 200 full_test cases of the `infer_step` rows of
$T/evals_anc/<run>/<protocol>/<run>/metrics.jsonl (pressure_l2, velocity_l2, nut_l2), checked against the file's
`infer_summary`. Usage: python scores.py <evals_anc dir>  -> JSON on stdout (runs that are not scored are null).
"""
import json, sys
from pathlib import Path

ROOT = Path(sys.argv[1])
ARMS = ["isla", "passive", "anchor", "dtt", "mesh"]
FIELDS = ["pressure_l2", "velocity_l2", "nut_l2"]
out = {}
for arm in ARMS:
    for seed in (42, 43):
        run = f"anc_{arm}_seed{seed}"
        for proto in ("native", "posed", "area"):
            f = ROOT / run / proto / run / "metrics.jsonl"
            done = (ROOT / run / proto / ".done").exists()
            if not f.exists():
                out[f"{run}/{proto}"] = None
                continue
            rows, summ, epoch = [], None, None
            for line in f.open():
                r = json.loads(line)
                if r["phase"] == "infer_step":
                    rows.append(r["metrics"])
                elif r["phase"] == "infer_summary":
                    summ = r["metrics"]
                elif r["phase"] == "config":
                    epoch = r.get("epoch")
            m = {k: sum(x[k] for x in rows) / len(rows) for k in FIELDS} if rows else None
            dev = max(abs(m[k] / summ[k] - 1) for k in FIELDS) if (m and summ) else None
            out[f"{run}/{proto}"] = dict(n=len(rows), epoch=epoch, done=done, summary_dev=dev, **(m or {}))
print(json.dumps(out, indent=1))
