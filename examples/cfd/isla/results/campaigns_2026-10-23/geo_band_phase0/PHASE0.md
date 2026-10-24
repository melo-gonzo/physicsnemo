# GEO-BAND phase 0: cost of the merged head 921472488 (2026-10-24)

Job 368333 (GB300 short QoS, one node nvl72d032-T08, 4 GPUs, 24:02; the five bad nodes excluded), launcher
`lhr/timing_gb.sbatch` (md5 48c6dc6e): FAST-MERGE's `timing_gb.sbatch` (job 365898) unchanged except the head snapshot
`$T/code_geothin3` (geometry.py 7472ecf0, geometry_fast.py 37d1bbf7, model.py 21a4d086, each equal to 921472488's; its
ISLA model and test tree equals 6bac57cbd's) and the launcher dir `$T/kernel/geoband`. Protocol (GEO-FAST's final2):
`recipe_r090` (`recipe_r090_fd` for the whole boundary) train.py under `timed_train2.py` (md5 845fe1b4), REF-090
settings, bfloat16, eager, medians after 5 warm-up calls or steps, the same five cases and inputs; arms base (0.9.0,
`code_cmpaf_fix`), head forced to `kernel="reference"` and head under auto (fast on GB300), passes with the GPUs rotated,
4 runs at once. Reduced by FAST-MERGE's `book_table.py` and `reduce_timing.py` (local copies in `lhr/`); raw output
`lhr/out_timing/`, `lhr/timing_368333.log`.

One run failed: the forced-reference AeroZoo arm of pass 1 (a torch.distributed rendezvous port collision between the
four concurrent runs, EADDRINUSE, before training started). The book row for it uses passes 2 and 3 (`book_table.py`
notes the substitution).

## The book's 06 cost table (mean over the two rotated passes of the per-pass medians)

| case (Q / F) | geometry call: base / reference / fast [ms] | step: base / reference / fast [ms] | step ratio: reference / fast |
|---|--:|--:|--:|
| AirfRANS 2-D (100000 / 1,018) | 12.7 / 31.1 / 33.5 | 374 / 357 / 366 | 0.95 / 0.98 |
| lug (9808 / 6,488) | 8.2 / 77.0 / 77.7 | 146 / 205 / 220 | 1.41 / 1.51 |
| AeroZoo (200000 / 501,802) | 19.1 / 121.4 / 110.1 | 511 / 600 / 602 | 1.17 / 1.18 |
| DrivAerML Volume, vehicle (200000 / 16,953,702) | 97.3 / 424.7 / 369.7 | 573 / 909 / 855 | 1.59 / 1.49 |
| DrivAerML Volume, whole boundary (200000 / 16,958,756) | 2,450.6 / 499.8 / 432.5 | 2,916 / 989 / 935 | 0.34 / 0.32 |

Per pass, with peak memory: `lhr/timing_table.md`.

## Against FAST-MERGE's head (7213c70c5-era geometry, job 365898, another node; its book_table)

| case | geometry call, reference [ms] | geometry call, fast [ms] | step ratio to base, reference / fast |
|---|---|---|---|
| AirfRANS 2-D | 29.9 -> 31.1 | 30.1 -> 33.5 | 1.06 / 1.16 -> 0.95 / 0.98 |
| lug | 44.0 -> 77.0 (+75 %) | 36.7 -> 77.7 (+112 %) | 1.27 / 1.21 -> 1.41 / 1.51 |
| AeroZoo | 103.0 -> 121.4 (+18 %) | 83.0 -> 110.1 (+33 %) | 1.17 / 1.15 -> 1.17 / 1.18 |
| DrivAerML vehicle | 369.5 -> 424.7 (+15 %) | 307.7 -> 369.7 (+20 %) | 1.52 / 1.39 -> 1.59 / 1.49 |
| DrivAerML whole boundary | 431.4 -> 499.8 (+16 %) | 353.7 -> 432.5 (+22 %) | 0.32 / 0.29 -> 0.34 / 0.32 |

- The base column agrees between the two jobs within 1-3 % on every case, so the comparison holds across the nodes.
- The thin-cell fix (GEO-THIN's F2 and F3) costs most on the lug: 33.5 % of its candidate pairs are guarded (GEO-THIN,
  probe_guard), so F3's accurate point runs on a third of the pairs, and the fast path is no faster than the reference
  there any more (77.7 against 77.0 ms; pass 2 of the fast arm, 88.6 ms and a 249 ms step, is an outlier). On DrivAerML
  the call rose 16-22 %, as GEO-THIN's standalone 15-18 % (412-474 and 343-405 ms); AeroZoo 18-33 %; AirfRANS (2-D, no F2
  or F3) unchanged within noise.
- This is GEO-BAND's cost baseline: DrivAerML whole boundary 499.8 ms (reference), 432.5 ms (fast) per call in training.
