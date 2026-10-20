"""E3b ANCHOR readout: step time and peak memory per run from runs/<run>/train.log (PREREG "Cost").

Median step time over epoch-0 steps 10-200 and over a mid-run epoch (250; the first occurrence, so a resumed
link's repeat of an epoch is not double counted), the median over all logged steps, and the peak `Mem:` (rank 0).
Usage: python cost.py <runs dir> <run>...   -> JSON on stdout.
"""
import json, re, statistics, sys
from pathlib import Path

PAT = re.compile(r"Epoch (\d+) \[(\d+)/(\d+)\] Loss: \S+ Step: ([\d.]+)s Mem: ([\d.]+)GB")
out = {}
for run in sys.argv[2:]:
    seen, e0, mid, allsteps, mem = set(), [], [], [], 0.0
    for line in (Path(sys.argv[1]) / run / "train.log").open(errors="replace"):
        m = PAT.search(line)
        if not m:
            continue
        ep, st, step, gb = int(m[1]), int(m[2]), float(m[4]), float(m[5])
        mem = max(mem, gb)
        if (ep, st) in seen:
            continue
        seen.add((ep, st))
        if st >= 10:
            allsteps.append(step)
        if ep == 0 and 10 <= st <= 200:
            e0.append(step)
        if ep == 250:
            mid.append(step)
    out[run] = dict(epoch0_median_s=statistics.median(e0), epoch0_n=len(e0),
                    epoch250_median_s=statistics.median(mid), epoch250_n=len(mid),
                    all_median_s=statistics.median(allsteps), peak_mem_gb=mem, epochs_logged=len({e for e, _ in seen}))
print(json.dumps(out, indent=1))
