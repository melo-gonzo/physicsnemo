# DTT-FAST: the table on the final heads (GB300, the recipe's training step)

Written 2026-10-24 17:50 EDT by DTT-FAST FINAL-TABLE-2 (pose column added 18:15 EDT, after ../pose/STATUS.md reported 38a7915cc pushed and its GB300 suite passed at 17:56). Built with `python reduce2.py
out/out_371869,out/out_372139,out/out_372260 out/out_372548 > out/tables.md` and `python surface2.py out/out_371869 >
out/surface_tables.md`; both are pasted below the summary.

Geometry cost in the columns (supervisor, 09:57 EDT 10-24): every column at b4b10d5a9, 4d281a702 and the pose head
contains psharpe/isla 921472488 (GEO-THIN's thin-cell fix, about +60 ms per DrivAerML whole-boundary step for both
forms) and its later fused thin-cell kernels (GEO-THINFAST, 2a521b20b). The carried-over before columns f13eb5ce0 and
7213c70c5, and the pass-2 column, contain neither.

## For FREEZE-PLAN

The step the final rerun will run, at each interior dataset's registered count (FREEZE-PLAN `FINAL_RERUN_PLAN.md`
section 1), measured by this harness (one GB300, bf16, per-block compile; median of 15 steps), in the fixed-count
state that training at that count runs (the varied-count state beside it). The pose head is the step if DTT-POSE-EXP
adopts the forest, else 4d281a702. Percentages compare with the pass-2 table FREEZE-PLAN read (`../final/TABLE.md`,
job 367435).
- **DrivAerML Volume, whole boundary, 200k:** pose head **1,174 ms** fixed (1,177 ms varied), **139.9 GiB**,
  **-19.6%** against pass 2's 1,460 ms and 107.3 GiB. 4d281a702: 1,209-1,231 ms fixed (1,212-1,219 varied), 143.0 GiB,
  -15.7% to -17.2%. The peak rises 1.30-1.33x over pass 2 because the size rule now runs the batch in one chunk.
- **AeroZoo, 200k:** pose head **691 ms** (both states), **124.8 GiB**, **-14.7%** against 810 ms and 125.5 GiB.
  4d281a702: 699-718 ms, 126.0 GiB, -11.4% to -13.7%.
- **AirfRANS with its far field, 100k:** pose head **521 ms** fixed (458 ms varied), **27.4 GiB**, **-5.3%** against
  550 ms (525 ms in pass 2's same-code repeat, job 368334) and 27.4 GiB. 4d281a702: 512-530 ms fixed, -3.6% to -6.9%.
  The varied-count state on AirfRANS moves by up to 17-20% between processes of the same code (452 to 529 ms), so read
  the fixed-count numbers, which repeat within 3.5%.
- **lug, all nodes (this dump: 9,048):** pose head **240 ms**, **5.5 GiB**; 4d281a702 229-261 ms in two processes
  (spread 14%). There is no pass-2 harness row; FREEZE-PLAN's S2 used 280 ms (PROFILE-2's 242 ms + 41 ms thin-cell, x
  0.99) and its S2-pass2 an EST of 418 ms, against which the pose head is -14% and -43%.
- **Tensile2d:** not measured (no dump).
- Same node, the pose head takes 0.97x the step of 4d281a702 on DrivAerML (0.95x at 400k), 0.98-0.99x on AeroZoo and
  1.00-1.04x on AirfRANS, with 0-3% less peak. Interacting ISLA at b4b10d5a9 (per-block compile, fixed counts):
  875 ms on DrivAerML, 594 ms on AeroZoo, 473 ms on AirfRANS at 100k; the pose head costs 1.34x, 1.16x and 1.10x of
  it there.

## Summary

On the final heads, ISLA-DTT's training step at 4d281a702 is 1.62-1.75x faster than the reference ISLA-DTT at
b4b10d5a9 on DrivAerML, 1.79-1.81x on AeroZoo, 1.40x on the lug and 1.00-1.37x on AirfRANS (1.07-1.09x at 100k), with
0.69-0.82x of its peak memory; the reference still runs out of memory at 400k DrivAerML points, 4d281a702 runs them
at 205-212 GiB. Against interacting ISLA at b4b10d5a9 (per-block compile) it costs 1.27-1.48x the step on DrivAerML,
1.15x on AeroZoo, 1.10-1.33x on AirfRANS and 1.69x on the lug, with 1.08-1.13x its peak on DrivAerML, 0.95-1.00x on
AeroZoo and 0.77-0.79x on AirfRANS. Against the pass-2 table it is 9.5-21.5% faster on DrivAerML and 12.7-16.2% on
AeroZoo, at higher peak memory (one chunk instead of chunks of 100k). The pose head (38a7915cc, 0.13.0) is another
3-5% faster on DrivAerML on the same node and 1-2% on AeroZoo; on AirfRANS and the lug it is level to 5% slower,
inside those cases' noise. The surface task's ISLA-DTT step
on DrivAerML whole boundary (SURFACE's harness, no chunking) falls from pass 2's 1,492 to 1,188 ms at 200k and from
2,493 to 1,914 ms at 400k, with 8-10% more peak memory (179.5 and 283.4 GB).

## What ran, and how

- **Snapshots** (read-only on LHR, each checked here by md5 against `git archive` of its commit: every file of
  `physicsnemo/` and of the recipe's `src/`, and no extra file; lists in `lhr/<commit>_*.md5`):
  - `code_dttfast_integ_refb4b` + `recipe_dttfast_integ_refb4b`: **b4b10d5a9** (psharpe/isla; ISLA 0.12.0, DEFAULT-DTT,
    GEO-THINFAST). model.py 8716db38, tree_read.py 57891d30, train.py 22311613. Its recipe has no chunked step and its
    model has no `tree_read_kernel` attribute: its ISLA-DTT is the reference read, unchunked, by construction.
  - `code_dttfast_integ_p5c` + `recipe_dttfast_integ_p5c`: **4d281a702** (DTT-FAST pass 5 on b4b10d5a9). model.py
    fa031026, tree_read.py 9cdc1c62, train.py 24f9b499. In every run's log of it and of the pose head: geometry 'fast',
    tree_read_kernel 'fused', tree_build 'fast', tree walk 'fast'; geo_kernel 'fused' on DrivAerML and AeroZoo and
    'eager' on AirfRANS and the lug, as in every column (b4b10d5a9's included).
  - `code_dttfast_pose_r` (built by the pose lane, `pose/remerge/lhr/snap/mk_r.sh`; checked here) + `recipe_dttfast_integ_p5c`:
    **38a7915cc** (psharpe/isla-dttfast-pose: 4d281a702 merged into DTT-POSE's forest, then ISLA 0.13.0). model.py
    b1871185, tree_read.py afcde57e; its recipe `src/` equals 4d281a702's file for file. The same fast paths resolved
    (its walk reads float32 boxes).
  - Both snapshot sets were built by the integration lane (pass 5, `integration/lhr/p5/mk2.sh`); I did not rebuild them.
- **Harness** `lhr/step.py` (md5 7e6d9b40): `../final/lhr/step.py` plus `--pre`. The model is built from PROFILE's
  first-batch dump's constructor arguments (seed 42) with `query_independent` set per column (True for ISLA-DTT, False
  for interacting ISLA) and `compile_blocks=True`; the timed function is the recipe's own `train._run_epoch`, with the
  optimizer, scheduler, loss and metrics built as `train.py main()` builds them. The configs at b4b10d5a9 and
  4d281a702 (`isla_volume.yaml`, `isla_volume_aerozoo.yaml`, `isla_volume_airfrans.yaml`, `isla_lug.yaml`) set
  `compile: false`, `model.compile_blocks: true` and `model.query_independent: true`; the dumps' resolved configs
  (PROFILE, `compile=false`) give the same `compile: false`, so the optimizer is not compiled, as in the configs. The
  interacting column keeps per-block compile too, as the first table ran it; its configs' own interacting form would be
  the whole-model compile (`compile=true model.query_independent=false model.compile_blocks=false`), not measured here.
  - **Varied counts** (the main table): each process first runs 3 untimed training batches of 0.90, 0.95 and 0.85 N
    points (random subsets of the dump's points, seeded), then 3 warm-up and 15 timed batches at N. So the compiled
    block reads have left their first, static-shape graph before timing, as FLAG and pass 5 found a process does once
    its counts vary. The step's chunking line is logged at the first batch (0.9 N).
  - **Fixed counts** (one table below): no untimed batches, so the point count never changes in the process. This is
    what training runs at the registered counts on DrivAerML (200k), AeroZoo (200k) and AirfRANS (100k): SubsampleMesh
    draws exactly `sampling_resolution` points from cases that all have more (AirfRANS 156k-210k), so the brief's
    "counts varying" holds only for the lug (all of each case's 5k-15k nodes). Both states are given. On DrivAerML and
    AeroZoo they differ by at most 3% in every column; on AirfRANS the varied-count state is unstable (see the caveats).
- **Launchers** `lhr/run.sbatch` (md5 efdb427d; jobs 371869, 372139) and `lhr/run2.sbatch` (md5 c70d30ff, adds the
  fixed-count columns; 372260, 372548): four workers on one node, one process and one GPU per run, the arms of
  a case queued next to each other; a 400k run that runs out of memory falls back to 300k. Each worker keeps its own
  Inductor/Triton caches across its runs, as in the first table. Queues `lhr/queue_{a,b,c}.txt` and `lhr/queue_d.txt` (pose and a 4d281a702 repeat, queued case by case on one node).
- **Surface rows**: SURFACE's harnesses unchanged (copies in `lhr/`: `dtt_surface_cost.py` md5 6721e9ab, the model's
  own forward, loss and backward with no optimizer and no chunking; `mem_trace.py` md5 6a47d7b6, the first final
  table's copy that wraps `ISLA._tree_read_pairs`), on 4d281a702, DrivAerML run_1 whole boundary at 200k and 400k, in
  job 371869.
- **Carried over** (cited from `../final/TABLE.md`, not rerun): f13eb5ce0 (job 366717), 7213c70c5 and pass 2
  757a5a877 (job 367435, the pass-2 table FREEZE-PLAN read), each run there without the untimed batches.

## Caveats

- Ratios inside one job are same-node; the carried-over columns and the pass-2 comparison are across nodes. In the
  first table the same column moved 0.9-2.9% between nodes on DrivAerML, 2.6-6.4% on AeroZoo and 4.7-9.6% on AirfRANS.
  Here 4d281a702 ran in two jobs on every setting (table "Repeats"): DrivAerML and AeroZoo repeat within 0.9%
  (varied) and 2.7% (fixed), so their same-node ratios are good to about 1%. AirfRANS does not: in the varied-count
  state the same code gives 529 and 452 ms (whole boundary, 100k), 473 and 394 ms (airfoil, 100k) and 740 and 866 ms
  (whole boundary, all points) in two processes, a 17-20% spread, while the fixed-count state repeats within 3.5% (530
  and 512 ms). This fits FLAG's finding that a compiled step depends on the process's compile history (here, which
  counts the dynamic graph was first tuned at), so AirfRANS ratios between jobs mean little; same-job ratios (pose
  against its 4d281a702 repeat) stay usable. The lug moved 14% (261 and 229 ms). The interacting and reference columns
  at b4b10d5a9 ran once per state, so on AirfRANS they carry the same uncertainty (interacting at 100k whole boundary:
  473 ms fixed, 398 ms varied; reference 626 and 574 ms).
- The "pass 2" percentages compare against the first table's pass-2 runs, which had no untimed batches; at the
  registered counts that matches the fixed-count state, which is given too.
- 4d281a702's size rule (pass 4, RULE) puts every point of these batches in one chunk (the logged rule: largest chunk
  that fits 569k-1,135k points), where pass 2 cut chunks of 100,000. So its peaks are higher than pass 2's on DrivAerML
  from 200k up (1.32-1.93x: 139.5-212.1 GiB, against about 107 GiB; 1.02-1.03x at 100k) and on AirfRANS at all points (47 against 27 GiB), and the step
  is faster. It still fits a GB300 (275.7 GiB as the rule reads it) at 400k with 64-70 GiB to spare, and its peak stays
  below the unchunked reference's (0.69-0.82x). Against interacting ISLA its peak is now 1.08-1.13x on DrivAerML,
  0.95-1.00x on AeroZoo and 0.77-0.79x on AirfRANS.
- AirfRANS airfoil-only at all 180,593 points is the one setting where 4d281a702 is slower than pass 2 (782 and 775 ms
  in two jobs, and 783 ms in a third, against 693 and 653 ms for pass 2 in two earlier jobs), and level with the
  reference (779-790 ms). It is the one-chunk setting the pass-5 lane studied (`integration/REPORT_PASS5.md`, decision
  1); here the untimed batches did not bring it to pass 2's speed. I did not isolate the cause. The final rerun trains
  neither that boundary nor that count (AirfRANS with its far field at 100k: 512-530 ms fixed, 4-7% faster than pass 2).
- The lug's single dump (9,048 nodes) is one case; its step is host-bound and its spread is wide (IQR 250-278 ms).
- Mean losses over all 21 steps (table "Every run") agree between 4d281a702 or the pose head and the reference at
  b4b10d5a9 within 1.8e-3 relative in every setting both ran (9.5e-4 within job 371869), the size of bf16 run-to-run
  spread in the first table (8.4e-4). This table tests no numerics beyond that.
- Tensile2d has no dump, so it is not measured.

## Jobs and GPU-h (LHR GB300; GPU-h = 4 GPUs x elapsed)

| job | name | what | node | elapsed | GPU-h |
|---|---|---|---|---|---|
| 371869 | isla.dttfast.final2.a | run.sbatch, queue_a: islab4b, refb4b, after x 11 settings (+2 OOM fallbacks), surface cost and memory | nvl72d018-T11 | 15:29 | 1.03 |
| 372139 | isla.dttfast.final2.b | run.sbatch, queue_b: AirfRANS airfoil and whole boundary at 100k x 3 columns, AirfRANS-all repeat x 2 | nvl72d015-T10 | 3:48 | 0.25 |
| 372260 | isla.dttfast.final2.c | run2.sbatch, queue_c: fixed-count islab4bS, refb4bS, afterS at the 3 registered counts | nvl72d025-T13 | 6:02 | 0.40 |
| 372548 | isla.dttfast.final2.d | run2.sbatch, queue_d: pose and a 4d281a702 repeat x 13 settings varied + 3 fixed (32 runs) | nvl72d042-T11 | 12:50 | 0.86 |

GPU-h total: 2.54 of the 3 cap.

## Tables

### Main table: step time and peak allocated memory (median of 15 timed steps)

| case | points | f13eb5ce0 (366717) | 7213c70c5 (367435) | pass 2 757a5a877 (367435) | interacting ISLA b4b10d5a9 | ISLA-DTT ref b4b10d5a9 | ISLA-DTT 4d281a702 | ISLA-DTT pose head | pose / 4d281a702 (same node) | ref / 4d281a702 (speedup) | 4d281a702 / interacting | 4d281a702 vs pass 2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DrivAerML vehicle | 100,000 | 1,379 ms, 129.7 GiB | 1,339 ms, 129.7 GiB | 903 ms, 104.1 GiB | 643 ms, 95.7 GiB | 1,351 ms, 129.8 GiB | 817 ms, 106.6 GiB | 801 ms, 105.0 GiB | 0.97x | 1.65x | 1.27x | -9.5% |
| DrivAerML vehicle | 200,000 | 1,954 ms, 183.7 GiB | 1,888 ms, 183.7 GiB | 1,291 ms, 106.0 GiB | 821 ms, 126.8 GiB | 1,896 ms, 183.9 GiB | 1,094 ms, 139.5 GiB | 1,051 ms, 136.3 GiB | 0.97x | 1.73x | 1.33x | -15.3% |
| DrivAerML vehicle | 400,000 | OOM (300k: 2,486 ms, 237.7 GiB) | OOM (300k: 2,426 ms, 237.7 GiB) | 2,021 ms, 108.0 GiB | 1,085 ms, 189.4 GiB | OOM (300k: 2,450 ms, 238.0 GiB) | 1,586 ms, 205.2 GiB | 1,497 ms, 198.9 GiB | 0.95x | - | 1.46x | -21.5% |
| DrivAerML whole boundary | 100,000 | 2,806 ms, 133.1 GiB | 1,475 ms, 133.1 GiB | 990 ms, 105.2 GiB | 687 ms, 95.8 GiB | 1,450 ms, 133.2 GiB | 895 ms, 108.3 GiB | 871 ms, 106.8 GiB | 0.97x | 1.62x | 1.30x | -9.6% |
| DrivAerML whole boundary | 200,000 | 4,112 ms, 190.6 GiB | 2,105 ms, 190.6 GiB | 1,460 ms, 107.3 GiB | 893 ms, 126.8 GiB | 2,126 ms, 190.7 GiB | 1,212 ms, 142.9 GiB | 1,177 ms, 139.9 GiB | 0.97x | 1.75x | 1.36x | -17.0% |
| DrivAerML whole boundary | 400,000 | OOM (300k: 5,440 ms, 248.0 GiB) | OOM (300k: 2,707 ms, 248.0 GiB) | 2,272 ms, 109.7 GiB | 1,214 ms, 189.4 GiB | OOM (300k: 2,713 ms, 248.2 GiB) | 1,792 ms, 212.1 GiB | 1,705 ms, 206.1 GiB | 0.95x | - | 1.48x | -21.1% |
| AeroZoo | 200,000 | 1,274 ms, 158.0 GiB | 1,250 ms, 158.0 GiB | 810 ms, 125.5 GiB | 616 ms, 125.9 GiB | 1,264 ms, 158.2 GiB | 707 ms, 126.0 GiB | 691 ms, 124.8 GiB | 0.99x | 1.79x | 1.15x | -12.7% |
| AeroZoo | 400,000 | 1,883 ms, 242.3 GiB | 1,831 ms, 242.3 GiB | 1,207 ms, 178.6 GiB | 882 ms, 188.8 GiB | 1,831 ms, 242.6 GiB | 1,011 ms, 179.4 GiB | 982 ms, 177.0 GiB | 0.98x | 1.81x | 1.15x | -16.2% |
| AirfRANS airfoil | 100,000 | 634 ms, 38.4 GiB | 661 ms, 38.4 GiB | 517 ms, 26.6 GiB | 425 ms, 34.1 GiB | 506 ms, 38.4 GiB | 473 ms, 26.6 GiB | 409 ms, 26.6 GiB | 1.04x | 1.07x | 1.11x | -8.5% |
| AirfRANS airfoil | 180,593 | 959 ms, 68.4 GiB | 974 ms, 68.4 GiB | 693 ms, 26.6 GiB | 593 ms, 60.7 GiB | 779 ms, 68.4 GiB | 782 ms, 47.0 GiB | 790 ms, 47.0 GiB | 1.01x | 1.00x | 1.32x | +12.9% |
| AirfRANS whole boundary | 100,000 | 629 ms, 39.5 GiB | 672 ms, 39.5 GiB | 550 ms, 27.4 GiB | 398 ms, 34.7 GiB | 574 ms, 39.5 GiB | 529 ms, 27.4 GiB | 458 ms, 27.4 GiB | 1.01x | 1.09x | 1.33x | -3.8% |
| AirfRANS whole boundary | 180,593 | 1,107 ms, 69.9 GiB | 1,124 ms, 69.9 GiB | 790 ms, 27.4 GiB | 672 ms, 61.3 GiB | 1,013 ms, 69.9 GiB | 740 ms, 47.9 GiB | 867 ms, 48.0 GiB | 1.00x | 1.37x | 1.10x | -6.4% |
| lug | 9,048 | - | - | - | 154 ms, 5.9 GiB | 366 ms, 6.8 GiB | 261 ms, 5.5 GiB | 240 ms, 5.5 GiB | 1.05x | 1.40x | 1.69x | - |

### Peak ratios (max_memory_allocated over the timed steps)

| case | points | 4d281a702 / ref b4b10d5a9 | 4d281a702 / interacting b4b10d5a9 | 4d281a702 / pass 2 | ref b4b10d5a9 / 7213c70c5 | interacting b4b10d5a9 / 7213c70c5 | pose / 4d281a702 |
|---|---|---|---|---|---|---|---|
| DrivAerML vehicle | 100,000 | 0.82x | 1.11x | 1.02x | 1.00x | 1.00x | 0.99x |
| DrivAerML vehicle | 200,000 | 0.76x | 1.10x | 1.32x | 1.00x | 1.00x | 0.98x |
| DrivAerML vehicle | 400,000 | - | 1.08x | 1.90x | - | 1.00x | 0.97x |
| DrivAerML whole boundary | 100,000 | 0.81x | 1.13x | 1.03x | 1.00x | 1.00x | 0.99x |
| DrivAerML whole boundary | 200,000 | 0.75x | 1.13x | 1.33x | 1.00x | 1.00x | 0.98x |
| DrivAerML whole boundary | 400,000 | - | 1.12x | 1.93x | - | 1.00x | 0.97x |
| AeroZoo | 200,000 | 0.80x | 1.00x | 1.00x | 1.00x | 1.00x | 0.99x |
| AeroZoo | 400,000 | 0.74x | 0.95x | 1.00x | 1.00x | 1.00x | 0.99x |
| AirfRANS airfoil | 100,000 | 0.69x | 0.78x | 1.00x | 1.00x | 1.00x | 1.00x |
| AirfRANS airfoil | 180,593 | 0.69x | 0.77x | 1.77x | 1.00x | 1.00x | 1.00x |
| AirfRANS whole boundary | 100,000 | 0.69x | 0.79x | 1.00x | 1.00x | 1.00x | 1.00x |
| AirfRANS whole boundary | 180,593 | 0.69x | 0.78x | 1.75x | 1.00x | 1.00x | 1.00x |
| lug | 9,048 | 0.80x | 0.93x | - | - | - | 1.00x |

### Fixed-count state at the registered training counts (no untimed varied-count batches: the point count never varies in the process, as in training at a fixed sampling_resolution) beside the varied-count state

| case | points | interacting b4b10d5a9 fixed / varied | ref b4b10d5a9 fixed / varied | 4d281a702 fixed / varied | pose fixed / varied | 4d281a702 repeat fixed / varied (pose job) | pose / 4d281a702 fixed (same node) | ref / 4d281a702 fixed | 4d281a702 / interacting fixed |
|---|---|---|---|---|---|---|---|---|---|
| DrivAerML whole boundary | 200,000 | 875 ms / 893 ms | 2,152 ms / 2,126 ms | 1,231 ms / 1,212 ms | 1,174 ms / 1,177 ms | 1,209 ms / 1,219 ms | 0.97x | 1.75x | 1.41x |
| AeroZoo | 200,000 | 594 ms / 616 ms | 1,239 ms / 1,264 ms | 718 ms / 707 ms | 691 ms / 691 ms | 699 ms / 701 ms | 0.99x | 1.73x | 1.21x |
| AirfRANS whole boundary | 100,000 | 473 ms / 398 ms | 626 ms / 574 ms | 530 ms / 529 ms | 521 ms / 458 ms | 512 ms / 452 ms | 1.02x | 1.18x | 1.12x |

### Every run: median (IQR), max timed step, peak allocated / reserved GiB, the 3 untimed varied-count steps and the 3 warm-up steps (ms), mean loss of all 21 steps, the chunking line

| run | points | median (IQR) ms | max ms | peak GiB | pre-steps ms (counts) | warm-up ms | mean loss | chunking |
|---|---|---|---|---|---|---|---|---|
| out_371869/after_aerozoo_200000 | 200,000 | 707 (704-717) | 2,069 | 126.0 / 126.6 | 50,963, 4,562, 677 (180,000, 190,000, 170,000) | 724, 714, 714 | 0.862488 | 180000 prediction points in 1 chunk (a chunk's predicted pea |
| out_371869/after_aerozoo_400000 | 400,000 | 1,011 (1,003-1,015) | 2,178 | 179.4 / 181.5 | 32,843, 1,917, 923 (360,000, 380,000, 340,000) | 1,040, 1,025, 1,009 | 0.862488 | 360000 prediction points in 1 chunk (a chunk's predicted pea |
| out_371869/after_airfrans_all | 180,593 | 782 (779-787) | 1,107 | 47.0 / 47.4 | 46,748, 10,752, 676 (162,533, 171,563, 153,504) | 791, 778, 775 | 0.268369 | 162533 prediction points in 1 chunk (a chunk's predicted pea |
| out_371869/after_airfrans_whole_all | 180,593 | 740 (735-744) | 751 | 47.9 / 49.6 | 45,281, 11,107, 639 (162,533, 171,563, 153,504) | 752, 742, 740 | 0.241760 | 162533 prediction points in 1 chunk (a chunk's predicted pea |
| out_371869/after_drv_body_100000 | 100,000 | 817 (813-823) | 915 | 106.6 / 106.7 | 19,528, 4,198, 798 (90,000, 95,000, 85,000) | 826, 813, 820 | 0.083435 | 90000 prediction points in 1 chunk (a chunk's predicted peak |
| out_371869/after_drv_body_200000 | 200,000 | 1,094 (1,084-1,101) | 2,223 | 139.5 / 140.0 | 35,148, 6,070, 1,072 (180,000, 190,000, 170,000) | 1,107, 1,089, 1,089 | 0.082906 | 180000 prediction points in 1 chunk (a chunk's predicted pea |
| out_371869/after_drv_body_400000 | 400,000 | 1,586 (1,579-1,602) | 1,698 | 205.2 / 206.9 | 34,228, 2,379, 1,443 (360,000, 380,000, 340,000) | 1,629, 1,602, 1,615 | 0.083256 | 360000 prediction points in 1 chunk (a chunk's predicted pea |
| out_371869/after_drv_whole_100000 | 100,000 | 895 (890-941) | 2,126 | 108.3 / 108.6 | 67,458, 6,429, 866 (90,000, 95,000, 85,000) | 908, 942, 888 | 0.091234 | 90000 prediction points in 1 chunk (a chunk's predicted peak |
| out_371869/after_drv_whole_200000 | 200,000 | 1,212 (1,204-1,226) | 2,356 | 142.9 / 143.7 | 67,585, 7,455, 1,112 (180,000, 190,000, 170,000) | 1,231, 1,259, 1,211 | 0.090665 | 180000 prediction points in 1 chunk (a chunk's predicted pea |
| out_371869/after_drv_whole_400000 | 400,000 | 1,792 (1,785-1,798) | 1,854 | 212.1 / 213.7 | 40,631, 7,630, 1,620 (360,000, 380,000, 340,000) | 1,843, 1,835, 1,782 | 0.091119 | 360000 prediction points in 1 chunk (a chunk's predicted pea |
| out_371869/after_lug_all | 9,048 | 261 (250-278) | 386 | 5.5 / 5.6 | 50,028, 6,962, 372 (8,143, 8,595, 7,690) | 269, 255, 253 | 0.337964 | 8143 prediction points in 1 chunk (a chunk's predicted peak  |
| out_371869/islab4b_aerozoo_200000 | 200,000 | 616 (612-620) | 624 | 125.9 / 127.4 | 39,589, 12,402, 572 (180,000, 190,000, 170,000) | 620, 612, 615 | 0.877604 | unchunked (no chunked step) |
| out_371869/islab4b_aerozoo_400000 | 400,000 | 882 (880-888) | 892 | 188.8 / 190.6 | 43,507, 14,841, 812 (360,000, 380,000, 340,000) | 901, 890, 884 | 0.877659 | unchunked (no chunked step) |
| out_371869/islab4b_airfrans_all | 180,593 | 593 (592-595) | 598 | 60.7 / 63.1 | 40,541, 14,567, 513 (162,533, 171,563, 153,504) | 593, 592, 592 | 0.239613 | unchunked (no chunked step) |
| out_371869/islab4b_airfrans_whole_all | 180,593 | 672 (670-673) | 675 | 61.3 / 64.0 | 31,685, 13,077, 589 (162,533, 171,563, 153,504) | 671, 670, 672 | 0.243885 | unchunked (no chunked step) |
| out_371869/islab4b_drv_body_100000 | 100,000 | 643 (641-648) | 654 | 95.7 / 97.7 | 34,009, 8,510, 627 (90,000, 95,000, 85,000) | 652, 646, 647 | 0.079053 | unchunked (no chunked step) |
| out_371869/islab4b_drv_body_200000 | 200,000 | 821 (816-826) | 832 | 126.8 / 129.6 | 16,350, 1,985, 771 (180,000, 190,000, 170,000) | 820, 818, 826 | 0.078755 | unchunked (no chunked step) |
| out_371869/islab4b_drv_body_400000 | 400,000 | 1,085 (1,078-1,094) | 1,104 | 189.4 / 192.2 | 47,476, 2,127, 1,030 (360,000, 380,000, 340,000) | 1,106, 1,106, 1,093 | 0.079253 | unchunked (no chunked step) |
| out_371869/islab4b_drv_whole_100000 | 100,000 | 687 (686-689) | 690 | 95.8 / 97.8 | 58,895, 11,766, 677 (90,000, 95,000, 85,000) | 701, 692, 689 | 0.084346 | unchunked (no chunked step) |
| out_371869/islab4b_drv_whole_200000 | 200,000 | 893 (891-895) | 897 | 126.8 / 129.2 | 40,562, 12,852, 831 (180,000, 190,000, 170,000) | 896, 892, 897 | 0.083863 | unchunked (no chunked step) |
| out_371869/islab4b_drv_whole_400000 | 400,000 | 1,214 (1,207-1,217) | 1,226 | 189.4 / 193.0 | 47,396, 1,901, 1,105 (360,000, 380,000, 340,000) | 1,207, 1,204, 1,213 | 0.084082 | unchunked (no chunked step) |
| out_371869/islab4b_lug_all | 9,048 | 154 (150-157) | 161 | 5.9 / 6.4 | 37,580, 11,271, 172 (8,143, 8,595, 7,690) | 169, 160, 158 | 0.329599 | unchunked (no chunked step) |
| out_371869/refb4b_aerozoo_200000 | 200,000 | 1,264 (1,257-1,279) | 1,291 | 158.2 / 160.3 | 29,486, 12,653, 1,188 (180,000, 190,000, 170,000) | 1,264, 1,278, 1,279 | 0.862419 | unchunked (no chunked step) |
| out_371869/refb4b_aerozoo_400000 | 400,000 | 1,831 (1,818-1,839) | 1,858 | 242.6 / 246.6 | 55,047, 2,495, 1,671 (360,000, 380,000, 340,000) | 1,834, 1,830, 1,864 | 0.862536 | unchunked (no chunked step) |
| out_371869/refb4b_airfrans_all | 180,593 | 779 (775-781) | 789 | 68.4 / 70.2 | 37,197, 14,174, 676 (162,533, 171,563, 153,504) | 786, 774, 767 | 0.268393 | unchunked (no chunked step) |
| out_371869/refb4b_airfrans_whole_all | 180,593 | 1,013 (1,011-1,022) | 1,027 | 69.9 / 73.4 | 36,016, 13,723, 894 (162,533, 171,563, 153,504) | 1,027, 1,010, 1,010 | 0.241734 | unchunked (no chunked step) |
| out_371869/refb4b_drv_body_100000 | 100,000 | 1,351 (1,342-1,362) | 1,365 | 129.8 / 131.5 | 21,088, 2,025, 1,292 (90,000, 95,000, 85,000) | 1,350, 1,360, 1,377 | 0.083455 | unchunked (no chunked step) |
| out_371869/refb4b_drv_body_200000 | 200,000 | 1,896 (1,883-1,903) | 1,928 | 183.9 / 187.8 | 26,484, 2,513, 1,750 (180,000, 190,000, 170,000) | 1,884, 1,873, 1,901 | 0.082951 | unchunked (no chunked step) |
| out_371869/refb4b_drv_body_300000 | 300,000 | 2,450 (2,433-2,463) | 2,482 | 238.0 / 246.6 | 46,815, 3,086, 2,222 (270,000, 285,000, 255,000) | 2,450, 2,444, 2,456 | 0.082997 | unchunked (no chunked step) |
| out_371869/refb4b_drv_body_400000 | 400000 | oom | | | | | | |
| out_371869/refb4b_drv_whole_100000 | 100,000 | 1,450 (1,441-1,457) | 1,469 | 133.2 / 135.5 | 59,196, 12,827, 1,384 (90,000, 95,000, 85,000) | 1,468, 1,448, 1,446 | 0.091227 | unchunked (no chunked step) |
| out_371869/refb4b_drv_whole_200000 | 200,000 | 2,126 (2,117-2,139) | 2,198 | 190.7 / 196.4 | 44,656, 11,011, 1,944 (180,000, 190,000, 170,000) | 2,124, 2,112, 2,125 | 0.090751 | unchunked (no chunked step) |
| out_371869/refb4b_drv_whole_300000 | 300,000 | 2,713 (2,709-2,725) | 2,741 | 248.2 / 265.9 | 44,553, 14,636, 2,471 (270,000, 285,000, 255,000) | 2,779, 2,740, 2,709 | 0.090906 | unchunked (no chunked step) |
| out_371869/refb4b_drv_whole_400000 | 400000 | oom | | | | | | |
| out_371869/refb4b_lug_all | 9,048 | 366 (359-389) | 394 | 6.8 / 7.3 | 40,026, 11,438, 369 (8,143, 8,595, 7,690) | 371, 347, 350 | 0.337882 | unchunked (no chunked step) |
| out_372139/after_airfrans_100000 | 100,000 | 473 (471-476) | 478 | 26.6 / 26.7 | 50,112, 9,712, 437 (90,000, 95,000, 85,000) | 481, 472, 472 | 0.268099 | 90000 prediction points in 1 chunk (a chunk's predicted peak |
| out_372139/after_airfrans_all | 180,593 | 775 (769-776) | 782 | 47.0 / 47.3 | 48,977, 1,986, 665 (162,533, 171,563, 153,504) | 795, 778, 776 | 0.268426 | 162533 prediction points in 1 chunk (a chunk's predicted pea |
| out_372139/after_airfrans_whole_100000 | 100,000 | 529 (525-538) | 545 | 27.4 / 28.1 | 50,487, 10,188, 485 (90,000, 95,000, 85,000) | 541, 530, 527 | 0.241294 | 90000 prediction points in 1 chunk (a chunk's predicted peak |
| out_372139/islab4b_airfrans_100000 | 100,000 | 425 (425-430) | 434 | 34.1 / 35.7 | 50,094, 12,615, 368 (90,000, 95,000, 85,000) | 437, 428, 429 | 0.239271 | unchunked (no chunked step) |
| out_372139/islab4b_airfrans_whole_100000 | 100,000 | 398 (396-398) | 399 | 34.7 / 35.9 | 47,330, 12,966, 365 (90,000, 95,000, 85,000) | 410, 398, 398 | 0.243585 | unchunked (no chunked step) |
| out_372139/refb4b_airfrans_100000 | 100,000 | 506 (501-527) | 530 | 38.4 / 39.5 | 50,105, 11,587, 481 (90,000, 95,000, 85,000) | 526, 504, 508 | 0.268101 | unchunked (no chunked step) |
| out_372139/refb4b_airfrans_all | 180,593 | 790 (783-794) | 795 | 68.4 / 70.4 | 46,651, 2,266, 711 (162,533, 171,563, 153,504) | 812, 790, 790 | 0.268510 | unchunked (no chunked step) |
| out_372139/refb4b_airfrans_whole_100000 | 100,000 | 574 (569-579) | 593 | 39.5 / 41.3 | 47,363, 13,669, 544 (90,000, 95,000, 85,000) | 587, 566, 565 | 0.241406 | unchunked (no chunked step) |
| out_372260/afterS_aerozoo_200000 | 200,000 | 718 (715-721) | 2,152 | 126.0 / 126.8 |  () | 73,447, 715, 718 | 0.883480 | 200000 prediction points in 1 chunk (a chunk's predicted pea |
| out_372260/afterS_airfrans_whole_100000 | 100,000 | 530 (529-535) | 539 | 27.4 / 28.1 |  () | 48,848, 531, 530 | 0.250656 | 100000 prediction points in 1 chunk (a chunk's predicted pea |
| out_372260/afterS_drv_whole_200000 | 200,000 | 1,231 (1,219-1,286) | 2,374 | 143.0 / 143.7 |  () | 72,104, 1,225, 1,216 | 0.094546 | 200000 prediction points in 1 chunk (a chunk's predicted pea |
| out_372260/islab4bS_aerozoo_200000 | 200,000 | 594 (592-594) | 595 | 125.9 / 127.4 |  () | 56,083, 597, 593 | 0.897959 | unchunked (no chunked step) |
| out_372260/islab4bS_airfrans_whole_100000 | 100,000 | 473 (471-476) | 495 | 34.7 / 36.4 |  () | 84,818, 470, 473 | 0.254352 | unchunked (no chunked step) |
| out_372260/islab4bS_drv_whole_200000 | 200,000 | 875 (873-877) | 884 | 126.8 / 129.4 |  () | 69,094, 872, 878 | 0.086720 | unchunked (no chunked step) |
| out_372260/refb4bS_aerozoo_200000 | 200,000 | 1,239 (1,234-1,247) | 1,258 | 158.2 / 160.3 |  () | 56,371, 1,223, 1,241 | 0.883423 | unchunked (no chunked step) |
| out_372260/refb4bS_airfrans_whole_100000 | 100,000 | 626 (622-628) | 633 | 39.5 / 41.2 |  () | 37,407, 619, 626 | 0.250582 | unchunked (no chunked step) |
| out_372260/refb4bS_drv_whole_200000 | 200,000 | 2,152 (2,142-2,162) | 2,229 | 190.8 / 196.1 |  () | 69,030, 2,131, 2,149 | 0.094670 | unchunked (no chunked step) |
| out_372548/afterS_aerozoo_200000 | 200,000 | 699 (696-707) | 718 | 126.0 / 126.8 |  () | 33,439, 707, 703 | 0.883354 | 200000 prediction points in 1 chunk (a chunk's predicted pea |
| out_372548/afterS_airfrans_whole_100000 | 100,000 | 512 (508-513) | 517 | 27.4 / 28.1 |  () | 27,491, 525, 520 | 0.250696 | 100000 prediction points in 1 chunk (a chunk's predicted pea |
| out_372548/afterS_drv_whole_200000 | 200,000 | 1,209 (1,204-1,211) | 1,220 | 143.0 / 143.7 |  () | 39,065, 1,211, 1,198 | 0.094504 | 200000 prediction points in 1 chunk (a chunk's predicted pea |
| out_372548/after_aerozoo_200000 | 200,000 | 701 (698-703) | 2,090 | 126.0 / 127.2 | 54,606, 4,252, 676 (180,000, 190,000, 170,000) | 723, 710, 711 | 0.862521 | 180000 prediction points in 1 chunk (a chunk's predicted pea |
| out_372548/after_aerozoo_400000 | 400,000 | 1,007 (1,001-1,011) | 2,282 | 179.4 / 181.7 | 47,301, 5,287, 926 (360,000, 380,000, 340,000) | 1,027, 1,006, 1,012 | 0.862417 | 360000 prediction points in 1 chunk (a chunk's predicted pea |
| out_372548/after_airfrans_100000 | 100,000 | 394 (393-396) | 399 | 26.6 / 26.7 | 58,386, 8,876, 372 (90,000, 95,000, 85,000) | 408, 399, 397 | 0.268270 | 90000 prediction points in 1 chunk (a chunk's predicted peak |
| out_372548/after_airfrans_all | 180,593 | 783 (779-793) | 847 | 47.0 / 47.3 | 51,427, 10,767, 671 (162,533, 171,563, 153,504) | 791, 772, 774 | 0.268528 | 162533 prediction points in 1 chunk (a chunk's predicted pea |
| out_372548/after_airfrans_whole_100000 | 100,000 | 452 (450-454) | 458 | 27.4 / 28.1 | 48,484, 2,372, 423 (90,000, 95,000, 85,000) | 474, 464, 461 | 0.241435 | 90000 prediction points in 1 chunk (a chunk's predicted peak |
| out_372548/after_airfrans_whole_all | 180,593 | 866 (861-873) | 975 | 47.9 / 49.6 | 40,018, 3,371, 736 (162,533, 171,563, 153,504) | 879, 855, 860 | 0.241991 | 162533 prediction points in 1 chunk (a chunk's predicted pea |
| out_372548/after_drv_body_100000 | 100,000 | 822 (819-827) | 916 | 106.6 / 106.7 | 43,980, 2,939, 794 (90,000, 95,000, 85,000) | 832, 825, 822 | 0.083427 | 90000 prediction points in 1 chunk (a chunk's predicted peak |
| out_372548/after_drv_body_200000 | 200,000 | 1,086 (1,080-1,089) | 1,103 | 139.5 / 140.0 | 47,034, 1,914, 1,039 (180,000, 190,000, 170,000) | 1,114, 1,096, 1,097 | 0.082890 | 180000 prediction points in 1 chunk (a chunk's predicted pea |
| out_372548/after_drv_body_400000 | 400,000 | 1,576 (1,568-1,593) | 1,746 | 205.2 / 206.6 | 27,534, 2,480, 1,429 (360,000, 380,000, 340,000) | 1,598, 1,573, 1,593 | 0.083210 | 360000 prediction points in 1 chunk (a chunk's predicted pea |
| out_372548/after_drv_whole_100000 | 100,000 | 893 (892-898) | 2,135 | 108.3 / 108.6 | 70,759, 7,074, 860 (90,000, 95,000, 85,000) | 926, 948, 894 | 0.091239 | 90000 prediction points in 1 chunk (a chunk's predicted peak |
| out_372548/after_drv_whole_200000 | 200,000 | 1,219 (1,213-1,222) | 2,383 | 142.9 / 143.7 | 70,994, 6,562, 1,128 (180,000, 190,000, 170,000) | 1,240, 1,266, 1,215 | 0.090761 | 180000 prediction points in 1 chunk (a chunk's predicted pea |
| out_372548/after_drv_whole_400000 | 400,000 | 1,790 (1,784-1,795) | 1,801 | 212.1 / 213.5 | 51,646, 3,332, 1,603 (360,000, 380,000, 340,000) | 1,834, 1,789, 1,775 | 0.090992 | 360000 prediction points in 1 chunk (a chunk's predicted pea |
| out_372548/after_lug_all | 9,048 | 229 (227-234) | 238 | 5.5 / 5.6 | 51,917, 6,956, 248 (8,143, 8,595, 7,690) | 256, 226, 221 | 0.337800 | 8143 prediction points in 1 chunk (a chunk's predicted peak  |
| out_372548/poseS_aerozoo_200000 | 200,000 | 691 (688-696) | 706 | 124.8 / 125.7 |  () | 33,787, 698, 693 | 0.883328 | 200000 prediction points in 1 chunk (a chunk's predicted pea |
| out_372548/poseS_airfrans_whole_100000 | 100,000 | 521 (518-529) | 530 | 27.4 / 28.0 |  () | 28,923, 533, 524 | 0.250514 | 100000 prediction points in 1 chunk (a chunk's predicted pea |
| out_372548/poseS_drv_whole_200000 | 200,000 | 1,174 (1,169-1,182) | 1,189 | 139.9 / 140.6 |  () | 40,236, 1,183, 1,166 | 0.094662 | 200000 prediction points in 1 chunk (a chunk's predicted pea |
| out_372548/pose_aerozoo_200000 | 200,000 | 691 (689-698) | 2,148 | 124.8 / 125.1 | 56,997, 4,254, 2,105 (180,000, 190,000, 170,000) | 702, 690, 696 | 0.862423 | 180000 prediction points in 1 chunk (a chunk's predicted pea |
| out_372548/pose_aerozoo_400000 | 400,000 | 982 (977-988) | 1,079 | 177.0 / 179.1 | 47,126, 7,442, 2,172 (360,000, 380,000, 340,000) | 998, 979, 978 | 0.862531 | 360000 prediction points in 1 chunk (a chunk's predicted pea |
| out_372548/pose_airfrans_100000 | 100,000 | 409 (406-411) | 418 | 26.6 / 26.7 | 60,376, 8,885, 384 (90,000, 95,000, 85,000) | 419, 422, 410 | 0.268062 | 90000 prediction points in 1 chunk (a chunk's predicted peak |
| out_372548/pose_airfrans_all | 180,593 | 790 (786-798) | 804 | 47.0 / 47.9 | 53,310, 10,756, 680 (162,533, 171,563, 153,504) | 794, 779, 782 | 0.268566 | 162533 prediction points in 1 chunk (a chunk's predicted pea |
| out_372548/pose_airfrans_whole_100000 | 100,000 | 458 (456-462) | 471 | 27.4 / 27.8 | 48,450, 2,298, 437 (90,000, 95,000, 85,000) | 1,794, 461, 457 | 0.241389 | 90000 prediction points in 1 chunk (a chunk's predicted peak |
| out_372548/pose_airfrans_whole_all | 180,593 | 867 (865-875) | 879 | 48.0 / 49.2 | 41,812, 3,349, 748 (162,533, 171,563, 153,504) | 878, 857, 864 | 0.241829 | 162533 prediction points in 1 chunk (a chunk's predicted pea |
| out_372548/pose_drv_body_100000 | 100,000 | 801 (799-807) | 895 | 105.0 / 105.2 | 45,506, 2,972, 784 (90,000, 95,000, 85,000) | 813, 800, 806 | 0.083342 | 90000 prediction points in 1 chunk (a chunk's predicted peak |
| out_372548/pose_drv_body_200000 | 200,000 | 1,051 (1,043-1,056) | 1,063 | 136.3 / 137.0 | 46,704, 1,925, 982 (180,000, 190,000, 170,000) | 1,058, 1,061, 1,054 | 0.082912 | 180000 prediction points in 1 chunk (a chunk's predicted pea |
| out_372548/pose_drv_body_400000 | 400,000 | 1,497 (1,489-1,506) | 1,545 | 198.9 / 200.6 | 27,541, 2,312, 1,352 (360,000, 380,000, 340,000) | 1,512, 1,501, 1,496 | 0.083233 | 360000 prediction points in 1 chunk (a chunk's predicted pea |
| out_372548/pose_drv_whole_100000 | 100,000 | 871 (869-875) | 962 | 106.8 / 107.0 | 70,722, 7,116, 843 (90,000, 95,000, 85,000) | 892, 874, 877 | 0.091184 | 90000 prediction points in 1 chunk (a chunk's predicted peak |
| out_372548/pose_drv_whole_200000 | 200,000 | 1,177 (1,171-1,183) | 1,284 | 139.9 / 140.6 | 70,719, 6,562, 1,087 (180,000, 190,000, 170,000) | 1,191, 2,299, 1,172 | 0.090789 | 180000 prediction points in 1 chunk (a chunk's predicted pea |
| out_372548/pose_drv_whole_400000 | 400,000 | 1,705 (1,695-1,711) | 2,625 | 206.1 / 207.7 | 51,787, 2,444, 1,542 (360,000, 380,000, 340,000) | 1,724, 1,688, 1,702 | 0.091106 | 360000 prediction points in 1 chunk (a chunk's predicted pea |
| out_372548/pose_lug_all | 9,048 | 240 (236-245) | 248 | 5.5 / 5.6 | 52,069, 6,875, 247 (8,143, 8,595, 7,690) | 249, 233, 227 | 0.337999 | 8143 prediction points in 1 chunk (a chunk's predicted peak  |

### Repeats of a column on two nodes (median ms)

| run | jobs | medians | spread |
|---|---|---|---|
| after_aerozoo_200000 | out_371869, out_372548 | 707, 701 | 0.9% |
| after_aerozoo_400000 | out_371869, out_372548 | 1,011, 1,007 | 0.5% |
| after_airfrans_100000 | out_372139, out_372548 | 473, 394 | 20.0% |
| after_airfrans_all | out_371869, out_372139, out_372548 | 782, 775, 783 | 1.1% |
| after_airfrans_whole_100000 | out_372139, out_372548 | 529, 452 | 17.1% |
| after_airfrans_whole_all | out_371869, out_372548 | 740, 866 | 17.1% |
| after_drv_body_100000 | out_371869, out_372548 | 817, 822 | 0.6% |
| after_drv_body_200000 | out_371869, out_372548 | 1,094, 1,086 | 0.7% |
| after_drv_body_400000 | out_371869, out_372548 | 1,586, 1,576 | 0.6% |
| after_drv_whole_100000 | out_371869, out_372548 | 895, 893 | 0.2% |
| after_drv_whole_200000 | out_371869, out_372548 | 1,212, 1,219 | 0.5% |
| after_drv_whole_400000 | out_371869, out_372548 | 1,792, 1,790 | 0.1% |
| after_lug_all | out_371869, out_372548 | 261, 229 | 13.9% |
| afterS_aerozoo_200000 | out_372260, out_372548 | 718, 699 | 2.7% |
| afterS_airfrans_whole_100000 | out_372260, out_372548 | 530, 512 | 3.5% |
| afterS_drv_whole_200000 | out_372260, out_372548 | 1,231, 1,209 | 1.9% |
| refb4b_airfrans_all | out_371869, out_372139 | 779, 790 | 1.4% |


### Surface task, DrivAerML run_1 whole boundary (SURFACE's harness: the model's own forward, loss and backward, no optimizer; GB = 1e9 B, peak allocated)

| count | ISLA-DTT f13eb5ce0 (363161) | pass 2 757a5a877 (367436) | 4d281a702 (371869) | 4d281a702 / pass 2 (time, peak) | 4d281a702 / f13eb5ce0 (time, peak) | eval: pass 2 / 4d281a702 | option A (367436) | 4d281a702 / option A (time) |
|---|---|---|---|---|---|---|---|---|
| 100,000 | 3,033 ms, 146.1 GB | 984 ms, 122.6 GB | - | - | - | 481 ms, 23.0 GB / - | 147 ms, 36.7 GB | - |
| 200,000 | 3,996 ms, 211.3 GB | 1,492 ms, 166.8 GB | 1,188 ms, 179.5 GB | 0.80x, 1.08x | 0.30x, 0.85x | 698 ms, 24.6 GB / 587 ms, 24.5 GB | 276 ms, 70.7 GB | 4.3x |
| 300,000 | - | 1,995 ms, 212.3 GB | - | - | - | 869 ms, 35.9 GB / - | - | - |
| 400,000 | - | 2,493 ms, 257.6 GB | 1,914 ms, 283.4 GB | 0.77x, 1.10x | - | 1,063 ms, 47.2 GB / 861 ms, 47.2 GB | 533 ms, 138.6 GB | 3.6x |

### Surface memory by stage (peak allocated GB, max over each stage's calls)

| case | phase | code | cell_measures peak | _gauge peak | exact_boundary_geometry peak | tree_read_pairs peak | run peak | status |
|---|---|---|---|---|---|---|---|---|
| drv_200k | eval | f13eb5ce0 (363410) | 1.9 | 3.5 | 23.3 | 26.1 | 26.1 | ok |
| drv_200k | train | f13eb5ce0 (363410) | 2.0 | 3.5 | 23.3 | 33.2 | 207.9 | ok |
| drv_200k | eval | 757a5a877 (367583) | 1.9 | 3.5 | 22.7 | 24.2 | 24.2 | ok |
| drv_200k | train | 757a5a877 (367583) | 2.0 | 3.5 | 22.7 | 25.2 | 163.8 | ok |
| drv_200k | eval | 4d281a702 (371869) | 1.9 | 3.5 | 22.9 | 24.2 | 24.2 | ok |
| drv_200k | train | 4d281a702 (371869) | 2.0 | 3.5 | 23.0 | 25.2 | 175.6 | ok |
| drv_400k | eval | 4d281a702 (371869) | 2.0 | 3.5 | 23.0 | 46.9 | 46.9 | ok |
| drv_400k | train | 4d281a702 (371869) | 2.0 | 3.5 | 23.0 | 48.4 | 275.8 | ok |

Surface caveat: the rise in the surface step's peak (163.8 to 175.6 GB at 200k in the memory-by-stage run) sits outside
the pair stage (25.2 GB both), so it comes from the blocks' part of the step between pass 2 and 4d281a702; I did not
locate it. At 400k the step peaks at 283.4 GB (264 GiB) of the GB300's 297 GB.

## Records

- `out/out_371869/`, `out_372139/`, `out_372260/`, `out_372548/`: one JSON and log per run (`<column>_<dump>_<n>`,
  with every step's time and peak, the RAN line with the files' md5s, the resolved kernels and the chunking line);
  `surfcost_*.jsonl` and `surfmem_*.jsonl` in out_371869. Launcher logs `out/run_<job>.log`.
- LHR copies: `$T/kernel/dttfast_final2/out_<job>/`.
