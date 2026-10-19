"""T-SCORESET (2026-10-19): Transolver 512x16 DrivAerML Surface at 100k vs 200k scored cells. Per run x protocol the
infer_summary row if present, else the mean of the last infer_step row per sample id (as reduce_drivefix.py). Sources:
scores/evals_char (AGA, 100k), scores/evals_r80ud (LHR, 100k), scores/evals_tset200k (LHR, 200k). Writes tset200k_scores.json."""
import json, pathlib, collections, statistics
root = pathlib.Path(__file__).parent / "scores"
def reduce_file(p):
    summ, steps = None, {}
    for line in p.read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line); ph = r.get("phase")
        if ph == "infer_summary": summ = r.get("metrics", r)
        elif ph == "infer_step": steps[r["sample_id"]] = r["metrics"]
    if isinstance(summ, dict) and any(k.endswith("_l2") for k in summ):
        return {k: v for k, v in summ.items() if isinstance(v, (int, float))}, len(steps), "summary"
    keys = collections.defaultdict(list)
    for m in steps.values():
        for k, v in m.items():
            if isinstance(v, (int, float)): keys[k].append(v)
    return {k: statistics.fmean(v) for k, v in keys.items()}, len(steps), "steps"
SRC100 = {"r80_dr_t_h512_l16_c100k": "evals_char", "r80ud_dr_t_h512_l16_c100k": "evals_r80ud"}
out = {"rows": [], "files": {}}
print(f"| lane | protocol | field | 100k s42 | 100k s43 | 100k mean | 200k s42 | 200k s43 | 200k mean | 200k/100k (mean) | s42 ratio | s43 ratio |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for lane, ev100 in SRC100.items():
    for proto in ("native", "area", "posed"):
        vals = {}
        for cells, ev in ((100, ev100), (200, "evals_tset200k")):
            for s in (42, 43):
                run = f"{lane}_seed{s}"
                p = root / ev / run / proto / run / "metrics.jsonl"
                m, n, src = reduce_file(p)
                vals[(cells, s)] = m
                out["files"][f"{ev}/{run}/{proto}"] = {"metrics": m, "n": n, "source": src}
                assert n == 48, (p, n)
        for fld in ("pressure_l2", "wss_l2"):
            v = {k: vals[k][fld] for k in vals}
            m100 = (v[(100, 42)] + v[(100, 43)]) / 2; m200 = (v[(200, 42)] + v[(200, 43)]) / 2
            row = {"lane": lane, "protocol": proto, "field": fld, "c100k": [v[(100, 42)], v[(100, 43)], m100],
                   "c200k": [v[(200, 42)], v[(200, 43)], m200], "ratio_mean": m200 / m100,
                   "ratio_seed": [v[(200, 42)] / v[(100, 42)], v[(200, 43)] / v[(100, 43)]]}
            out["rows"].append(row)
            print(f"| {lane} | {proto} | {fld} | {v[(100,42)]:.5f} | {v[(100,43)]:.5f} | {m100:.5f} | {v[(200,42)]:.5f} | {v[(200,43)]:.5f} | {m200:.5f} | {m200/m100:.4f} | {row['ratio_seed'][0]:.4f} | {row['ratio_seed'][1]:.4f} |")
json.dump(out, open(root.parent / "tset200k_scores.json", "w"), indent=1)
