"""Reduce DRIVE-FIX scorer outputs (2026-10-19): per run x protocol, the infer_summary row if present, else the mean of the
last infer_step row per sample id. Prints per-seed values and seed means for R1 (AirfRANS), R2 (DrivAerML Surface
Transolver) and R3 (lug GeoTransolver)."""
import json, pathlib, collections, statistics, sys

root = pathlib.Path(__file__).parent / "scores"

def reduce_file(p):
    summ, steps, cfg = None, {}, {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ph = r.get("phase")
        if ph == "config":
            cfg = r
        elif ph == "infer_summary":
            summ = r.get("metrics", r)
        elif ph == "infer_step":
            steps[r["sample_id"]] = r["metrics"]
    if summ is not None and isinstance(summ, dict) and any(k.endswith("_l2") for k in summ):
        return {k: v for k, v in summ.items() if isinstance(v, (int, float))}, len(steps), cfg, "summary"
    keys = collections.defaultdict(list)
    for m in steps.values():
        for k, v in m.items():
            if isinstance(v, (int, float)):
                keys[k].append(v)
    return {k: statistics.fmean(v) for k, v in keys.items()}, len(steps), cfg, "steps"

rows = {}
for p in sorted(root.rglob("metrics.jsonl")):
    rel = p.relative_to(root).parts  # evals_x/<run>/<protocol>/<run>/metrics.jsonl
    ev, run, proto = rel[0], rel[1], rel[2]
    m, n, cfg, src = reduce_file(p)
    rows[(ev, run, proto)] = (m, n, cfg.get("dataset"), cfg.get("epoch"), src)
for (ev, run, proto), (m, n, ds, ep, src) in rows.items():
    keys = [k for k in sorted(m) if k.endswith("_l2")]
    print(f"{run:40s} {proto:7s} n={n:3d} ep={ep} {src:7s} ds={ds}  " + " ".join(f"{k}={m[k]:.5f}" for k in keys))
json.dump({f"{ev}/{run}/{proto}": {"metrics": m, "n": n, "dataset": ds, "epoch": ep, "source": src}
           for (ev, run, proto), (m, n, ds, ep, src) in rows.items()},
          open(root.parent / "drivefix_scores_2026-10-19.json", "w"), indent=1)
