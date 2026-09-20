"""Per-lane training cost for the ISO-COST / ISO-MATCH grid, read from each lane's own train.log.

For every manifest row in group "iso": median step time over all logged steps after excluding warm-up steps
(the first two steps of every epoch, which carry compile and data-loader start-up, and any step above 5 s),
the peak of the logged per-step memory, the number of steps used, and the epochs completed. Writes a TSV.
usage: python isocost_cost.py <manifest.tsv> <runs dir> <out.tsv>
"""
import csv, re, statistics as st, sys
from pathlib import Path

manifest, runs, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
pat = re.compile(r"Epoch (\d+) \[(\d+)/(\d+)\] Loss: ([0-9.naN]+) Step: ([0-9.]+)s Mem: ([0-9.]+)GB")
rows = []
for r in csv.DictReader(open(manifest), delimiter="\t"):
    if r["group"] != "iso":
        continue
    log = runs / r["run_id"] / "train.log"
    steps, mems, epochs, completed = [], [], set(), False
    if log.exists():
        for line in open(log, errors="replace"):
            m = pat.search(line)
            if m:
                ep, i, n, loss, s, g = m.groups(); ep, i, s, g = int(ep), int(i), float(s), float(g)
                epochs.add(ep); mems.append(g)
                if i > 2 and s < 5.0:
                    steps.append(s)
            elif "Training completed" in line:
                completed = True
    rows.append({"run_id": r["run_id"], "step_s_median": round(st.median(steps), 4) if steps else "",
                 "step_s_p10": round(sorted(steps)[len(steps) // 10], 4) if steps else "",
                 "step_s_p90": round(sorted(steps)[len(steps) * 9 // 10], 4) if steps else "",
                 "peak_mem_gb": max(mems) if mems else "", "n_steps": len(steps),
                 "epochs_logged": len(epochs), "completed": int(completed)})
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t"); w.writeheader(); w.writerows(rows)
for r in rows:
    print("\t".join(str(v) for v in r.values()))
