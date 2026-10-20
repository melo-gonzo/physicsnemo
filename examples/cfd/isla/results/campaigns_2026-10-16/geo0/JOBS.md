# GEO0 jobs (all submitted after the PREREG.md amendment, 2026-10-16T21:16-04:00)
| cluster | job | name | rows (kernel/geo0/) | units | state |
|---|---|---|---|---|---|
| LHR | 241664 (array 0) | isla.geo0.af | rows_af.tsv, GEO0_U=4 | 16 | COMPLETED 7:39, all 16 units |
| LHR | 241665 (array 0) | isla.geo0.dv | rows_dv.tsv, GEO0_U=4 | 16 | COMPLETED 5:13: 4 native units fine; 12 g0 redraw units stopped in the guard (722-cell non-vehicle boundary), no rows |
Command (from $T/kernel/geo0): `sbatch -q short --time=01:59:00 -J <name> --array=0 --export=ALL,GEO0_ROWS=$G/<rows>,GEO0_U=4 score.sbatch`
(short QOS: 11 of my jobs in short at submission, at most 15 allowed).
Outputs: $T/kernel/geo0/evals/<run>/<protocol>/<run>/metrics.jsonl; logs evals/<run>/<protocol>.log; sbatch_logs/.
| LHR | 241802 (array 0) | isla.geo0.dv1 | rows_dv1.tsv, GEO0_U=3 (DrivAerML g1a/g1u/g1v, guard off for the small non-vehicle boundaries; see PREREG.md note) | 12 | COMPLETED 6:59, all 12 units, 48/48 vehicle draws each |
