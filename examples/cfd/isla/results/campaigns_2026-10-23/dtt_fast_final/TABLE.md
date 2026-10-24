# DTT-FAST: the before/after table (GB300, the recipe's training step)

Written 2026-10-24 08:12 EDT by DTT-FAST FINAL-TABLE. Built from the records in `out/` with `reduce.py`
(`python reduce.py out/out_366717 out/out_367435`), `surface_rows.py`, `surface_mem.py` and `check_after.py`.

## Summary

ISLA-DTT after (pass 2, 757a5a877: fused read, GPU forest build and walk, the fast geometry, and chunked training)
against ISLA-DTT before (f13eb5ce0, the campaign's base) and against 7213c70c5 (ISLA 0.11.0 with the fast geometry):

| case, points | before f13eb5ce0 | before 7213c70c5 | after | speedup vs f13 / vs 7213 | peak after / before | interacting ISLA (7213) |
|---|---|---|---|---|---|---|
| DrivAerML vehicle, 200k | 1,954 ms, 183.7 GiB | 1,888 ms, 183.7 GiB | 1,291 ms, 106.0 GiB | 1.51x / 1.46x | 0.58x | 768 ms, 126.8 GiB |
| DrivAerML whole boundary, 200k | 4,112 ms, 190.6 GiB | 2,105 ms, 190.6 GiB | 1,460 ms, 107.3 GiB | 2.82x / 1.44x | 0.56x | 832 ms, 126.8 GiB |
| DrivAerML vehicle, 400k | out of memory (300k: 2,486 ms, 237.7 GiB) | out of memory | 2,021 ms, 108.0 GiB | - | - | 1,049 ms, 189.4 GiB |
| DrivAerML whole boundary, 400k | out of memory (300k: 5,440 ms, 248.0 GiB) | out of memory | 2,272 ms, 109.7 GiB | - | - | 1,151 ms, 189.4 GiB |
| AeroZoo, 200k | 1,274 ms, 158.0 GiB | 1,250 ms, 158.0 GiB | 810 ms, 125.5 GiB | 1.57x / 1.54x | 0.79x | 608 ms, 125.9 GiB |
| AirfRANS, all 180,593 | 959 ms, 68.4 GiB | 974 ms, 68.4 GiB | 693 ms, 26.6 GiB | 1.38x / 1.41x | 0.39x | 761 ms, 60.7 GiB |
| Surface: DrivAerML run_1 whole boundary, 200k | 3,996 ms, 211.3 GB | - | 1,492 ms, 166.8 GB | 2.68x / - | 0.79x | (option A: 276 ms, 70.7 GB) |

So ISLA-DTT's training step is 1.44-1.56x faster than at 7213c70c5 on the 3-D cases (1.22-1.42x on AirfRANS), and
1.51-2.83x faster than at f13eb5ce0 (1.14-1.40x on AirfRANS). Its peak memory is 0.39-0.85x of either before column. It
now trains 400k DrivAerML points at about 110 GiB, where both before columns run out of memory above 300k. Against
interacting ISLA on the same commit it still costs 1.33-1.97x the time on the 3-D cases, and 0.88-0.91x on AirfRANS at
its natural count, where DTT is faster. On DrivAerML from 200k points up it needs less memory than interacting ISLA
(0.84x at 200k, 0.57x at 400k); on AeroZoo about the same (1.00x at 200k, 0.95x at 400k); at 100k 1.05-1.10x. Peaks are `max_memory_allocated` in GiB (2^30 B).
The surface rows use SURFACE's units (GB = 1e9 B).

## Setup

- **Harness** `lhr/step.py` (md5 3ae3fdc6): PROFILE's first-batch dumps (`$T/kernel/dttfast_profile/dumps/`, job
  363153: the recipe's first training batch at REF-090 settings, sampling 400k, augment off), prediction points
  subset by PROFILE's draw (seed 0). The model is built from the dump's constructor arguments (seed 42) with
  `query_independent` set per column and `compile_blocks=True`. The optimizer (`build_muon_optimizer`), scheduler,
  loss (each dataset's `loss_type`: huber on DrivAerML and AirfRANS, case_relative_l2 on AeroZoo) and metrics are
  built as `train.py main()` builds them. The timed function is **the recipe's own `train._run_epoch`** over 3 warm-up
  and 15 timed batches, so the after column takes the recipe's chunked step (`chunked_forward_backward`: chunks of
  100,000 points under huber; one chunk under case_relative_l2, chosen by the size rule) and the before columns take
  the unchunked `forward_pass` plus backward. Every batch is a fresh device copy of the dump (the loader's
  `recursive_to_device`), so no Mesh cache carries over; `torch.manual_seed(1000 + s)` is set per batch, so every
  column sees the same draws. Step time is the interval between the ends of consecutive optimizer steps
  (cuda-synchronized), so it includes the device copy of the batch, the loss, the metrics and the step's logging.
  Peaks are the allocator's maximum over each such interval. Every row has the median of 15 steps and the
  interquartile range.
- **Launcher** `lhr/run.sbatch` (md5 9d7293ce): one process per run, one GB300 each, four workers on a queue
  (`lhr/queue_before.txt`, `lhr/queue_after.txt`); at 400k a run that runs out of memory falls back to 300k.
- **Columns.** before-f13 = `code_dvcost2a` (md5-checked: `physicsnemo/` equals f13eb5ce0 except `segment_softmax.py`,
  one commit older, which only changes how Triton is imported; model.py 33542324, tree_read.py 57891d30);
  before-ref = `code_dttpose_base` (`physicsnemo/` bitwise 7213c70c5; model.py d89d3cd3); both with recipe
  `recipe_dttfast_final_ref` (git archive of f13eb5ce0's recipe; its `src/` equals 7213c70c5's and pass 1's; train.py
  22311613). after = `code_dttfast_integ_p2b` (the integration lane's snapshot; I checked that its `physicsnemo/`
  equals 757a5a877 by md5; model.py 6374a775, tree_read.py 074c39a9) with `recipe_dttfast_integ_p2d` (`src/` equal to
  757a5a877's; train.py 23e137ef). ISLA = interacting ISLA (`query_independent=False`) on each before snapshot. Every
  log prints the files and md5s it ran (`RAN` lines). In the after runs every fast path resolved: geometry 'fast',
  tree_build 'fast', tree walk 'fast', tree_read_kernel 'fused'.
- **Surface rows:** SURFACE's harness rerun unchanged on the after snapshot (`v2/cost_gb.sbatch`,
  `dtt_surface_cost.py` md5 6721e9ab). This is the model's own forward, loss and backward with no optimizer, and
  not the recipe's chunked step, because the surface task has no recipe step in ISLA-DTT form. Its memory by stage
  ran on a copy of SURFACE's `mem_trace.py` (`lhr/mem/`, md5 6a47d7b6). The copy wraps `ISLA._tree_read_pairs`,
  because 757a5a877 has no module-level `tree_read_pairs`; the first try (367470) failed on that name.

## Caveats

- Noise: the 7213c70c5 DTT column ran in both jobs on different nodes. The medians differ by 0.0-4.7% (DrivAerML at
  most 1.9%; AirfRANS and AeroZoo 100k at most 4.7%, with wider ranges in 367435). So ratios within about 5% are
  noise. The before-f13 and ISLA-f13 columns ran on one node (366717), and the after and ISLA-ref columns on another
  (367435).
- AirfRANS's natural count, 180,593, is not a multiple of 8, and under chunking its second chunk is 80,593 points.
  The after peak at AirfRANS 180,593 (26.6 GiB) equals its 100k peak, since one 100k chunk is live at a time.
- The after column's step is the recipe's chunked step, which the before columns do not have. So "after" measures
  the whole of pass 2 against the base, including chunking's memory saving; QCHUNK measured chunking alone (no time
  cost at 200k; -60 GiB).
- Losses agree: over the 18 steps, the after mean loss is within 8.4e-4 relative of the 7213c70c5 DTT run's in every
  setting, the same spread as f13eb5ce0 against 7213c70c5 (table below). The integration lane reported a bf16
  gradient excess of the chunked step on AirfRANS and the DrivAerML whole boundary (integration `STATUS.md` 07:44).
  This table does not test that.
- Pass 2 does not include BASIS (its fix landed after the checks started; pass 3).
- Surface: option A ran in both surface jobs on the same code (`code_surfbc`). Its HiLift peaks are 10.2 GB lower in
  the rerun at every count, with the same times; the cause is unexplained, and DrivAerML's option A rows agree within
  0.4 GB. HiLift ISLA-DTT still runs out of memory inside `exact_boundary_geometry` at 10k and at 100k (that stage
  peaks at 292 GB from 35.6 GB before it), even for evaluation: the brief expected this. The after harness cannot
  time the pair stage (`tree_read_pairs_ms` reads "not a name in this model.py"), but the geometry stage fell from
  2,162 to 263 ms at 200k.

## Jobs and GPU-h (LHR GB300; GPU-h = 4 GPUs x elapsed)

| job | name | what | node | elapsed | GPU-h |
|---|---|---|---|---|---|
| 366717 | isla.dttfast.final.before | columns f13, ref, f13isla (13 settings + 4 fallbacks) | nvl72d007-T14 | 19:44 | 1.32 |
| 367435 | isla.dttfast.final.after | columns after, ref (repeat), refisla | nvl72d049-T15 | 15:15 | 1.02 |
| 367436 | isla.dttfast.final.surfcost | SURFACE v2 cost_gb.sbatch, CODE_DTT=code_dttfast_integ_p2b, OUT=after_final | nvl72d048-T07 | 5:40 | 0.38 |
| 367470 | isla.dttfast.final.surfmem | SURFACE v3 mem_gb.sbatch as is: failed on the renamed tree_read_pairs | nvl72d024-T10 | 1:33 | 0.10 |
| 367583 | isla.dttfast.final.surfmem | the same with the mem_trace.py copy (lhr/mem/) | nvl72d003-T15 | 2:26 | 0.16 |

Total 2.98 GPU-h of the 10 cap.

## Full tables (one row per dataset and count)

Columns: DTT before-f13 (f13eb5ce0), DTT before-ref (7213c70c5; the 367435 repeat, the last job given), DTT after (757a5a877, chunked step), interacting ISLA on each before commit. OOM: out of memory at that count (the next count down is the row below it).

### Step time (ms, median of 15 synchronized steps after 3 warm-up steps; interquartile range in brackets)

| dataset | points | DTT before-f13 | DTT before-ref | DTT after | ISLA f13 | ISLA ref | f13 / after | ref / after | after / ISLA ref | after / ISLA f13 |
|---|---|---|---|---|---|---|---|---|---|---|
| DrivAerML vehicle | 100,000 | 1,379 (1,376-1,400) | 1,339 (1,335-1,358) | 903 (891-917) | 655 (653-657) | 613 (611-615) | 1.53x | 1.48x | 1.47x | 1.38x |
| DrivAerML vehicle | 200,000 | 1,954 (1,938-1,970) | 1,888 (1,865-1,893) | 1,291 (1,280-1,312) | 827 (822-833) | 768 (763-771) | 1.51x | 1.46x | 1.68x | 1.56x |
| DrivAerML vehicle | 300,000 | 2,486 (2,475-2,497) | 2,426 (2,410-2,441) | - | - | - | - | - | - | - |
| DrivAerML vehicle | 400,000 | OOM | OOM | 2,021 (2,014-2,024) | 1,155 (1,150-1,157) | 1,049 (1,045-1,056) | - | - | 1.93x | 1.75x |
| DrivAerML whole boundary | 100,000 | 2,806 (2,792-2,861) | 1,475 (1,462-1,492) | 990 (982-1,002) | 2,025 (2,015-2,059) | 647 (643-649) | 2.83x | 1.49x | 1.53x | 0.49x |
| DrivAerML whole boundary | 200,000 | 4,112 (4,075-4,290) | 2,105 (2,089-2,113) | 1,460 (1,448-1,505) | 2,854 (2,845-2,908) | 832 (828-852) | 2.82x | 1.44x | 1.75x | 0.51x |
| DrivAerML whole boundary | 300,000 | 5,440 (5,426-5,466) | 2,707 (2,690-2,736) | - | - | - | - | - | - | - |
| DrivAerML whole boundary | 400,000 | OOM | OOM | 2,272 (2,262-2,319) | 4,512 (4,498-4,519) | 1,151 (1,145-1,162) | - | - | 1.97x | 0.50x |
| AeroZoo | 100,000 | 961 (956-963) | 969 (941-1,083) | 621 (599-656) | 479 (476-486) | 459 (453-499) | 1.55x | 1.56x | 1.35x | 1.29x |
| AeroZoo | 200,000 | 1,274 (1,268-1,295) | 1,250 (1,242-1,256) | 810 (806-822) | 619 (618-621) | 608 (606-639) | 1.57x | 1.54x | 1.33x | 1.31x |
| AeroZoo | 400,000 | 1,883 (1,877-1,891) | 1,831 (1,822-1,851) | 1,207 (1,193-1,267) | 933 (925-962) | 878 (877-884) | 1.56x | 1.52x | 1.38x | 1.29x |
| AirfRANS | 100,000 | 634 (629-644) | 661 (644-692) | 517 (507-524) | 475 (473-476) | 476 (475-478) | 1.23x | 1.28x | 1.09x | 1.09x |
| AirfRANS | 180,593 | 959 (953-962) | 974 (960-1,006) | 693 (677-710) | 763 (761-764) | 761 (758-770) | 1.38x | 1.41x | 0.91x | 0.91x |
| AirfRANS whole boundary | 100,000 | 629 (626-635) | 672 (643-716) | 550 (547-566) | 462 (461-463) | 469 (466-472) | 1.14x | 1.22x | 1.17x | 1.19x |
| AirfRANS whole boundary | 180,593 | 1,107 (1,098-1,110) | 1,124 (1,113-1,176) | 790 (769-795) | 886 (884-887) | 897 (890-913) | 1.40x | 1.42x | 0.88x | 0.89x |

### Peak memory (GiB, max over the timed steps: max_memory_allocated / max_memory_reserved)

| dataset | points | DTT before-f13 | DTT before-ref | DTT after | ISLA f13 | ISLA ref | after / f13 | after / ref | after / ISLA ref |
|---|---|---|---|---|---|---|---|---|---|
| DrivAerML vehicle | 100,000 | 129.7 / 132.2 | 129.7 / 132.4 | 104.1 / 105.0 | 95.7 / 97.4 | 95.7 / 97.4 | 0.80x | 0.80x | 1.09x |
| DrivAerML vehicle | 200,000 | 183.7 / 187.0 | 183.7 / 187.4 | 106.0 / 106.9 | 126.8 / 129.4 | 126.8 / 129.6 | 0.58x | 0.58x | 0.84x |
| DrivAerML vehicle | 300,000 | 237.7 / 246.4 | 237.7 / 246.2 | - | - | - | - | - | - |
| DrivAerML vehicle | 400,000 | OOM | OOM | 108.0 / 110.0 | 189.4 / 193.1 | 189.4 / 192.8 | - | - | 0.57x |
| DrivAerML whole boundary | 100,000 | 133.1 / 136.7 | 133.1 / 136.4 | 105.2 / 105.7 | 95.8 / 97.7 | 95.8 / 97.5 | 0.79x | 0.79x | 1.10x |
| DrivAerML whole boundary | 200,000 | 190.6 / 195.5 | 190.6 / 195.6 | 107.3 / 108.4 | 126.8 / 129.3 | 126.8 / 129.0 | 0.56x | 0.56x | 0.85x |
| DrivAerML whole boundary | 300,000 | 248.0 / 261.7 | 248.0 / 259.6 | - | - | - | - | - | - |
| DrivAerML whole boundary | 400,000 | OOM | OOM | 109.7 / 112.5 | 189.4 / 193.6 | 189.4 / 193.4 | - | - | 0.58x |
| AeroZoo | 100,000 | 115.9 / 118.1 | 115.9 / 117.6 | 99.0 / 99.3 | 94.4 / 95.8 | 94.4 / 95.6 | 0.85x | 0.85x | 1.05x |
| AeroZoo | 200,000 | 158.0 / 160.2 | 158.0 / 160.2 | 125.5 / 126.4 | 125.9 / 127.5 | 125.9 / 127.6 | 0.79x | 0.79x | 1.00x |
| AeroZoo | 400,000 | 242.3 / 245.8 | 242.3 / 246.1 | 178.6 / 180.3 | 188.8 / 191.0 | 188.8 / 191.1 | 0.74x | 0.74x | 0.95x |
| AirfRANS | 100,000 | 38.4 / 39.5 | 38.4 / 39.5 | 26.6 / 27.1 | 34.1 / 35.6 | 34.1 / 35.6 | 0.69x | 0.69x | 0.78x |
| AirfRANS | 180,593 | 68.4 / 70.4 | 68.4 / 70.4 | 26.6 / 27.7 | 60.7 / 63.3 | 60.7 / 64.1 | 0.39x | 0.39x | 0.44x |
| AirfRANS whole boundary | 100,000 | 39.5 / 41.0 | 39.5 / 41.0 | 27.4 / 28.3 | 34.7 / 36.2 | 34.7 / 36.2 | 0.69x | 0.69x | 0.79x |
| AirfRANS whole boundary | 180,593 | 69.9 / 72.7 | 69.9 / 72.7 | 27.4 / 28.0 | 61.3 / 64.0 | 61.3 / 64.0 | 0.39x | 0.39x | 0.45x |

### Repeats (the same column in two jobs: run-to-run and node-to-node noise)

| column | dataset | points | job out_366717 | job out_367435 | spread |
|---|---|---|---|---|---|
| DTT before-ref | AeroZoo | 100000 | 929 (921-943) | 969 (941-1,083) | 4.3% |
| DTT before-ref | AeroZoo | 200000 | 1,222 (1,209-1,230) | 1,250 (1,242-1,256) | 2.3% |
| DTT before-ref | AeroZoo | 400000 | 1,831 (1,823-1,838) | 1,831 (1,822-1,851) | 0.0% |
| DTT before-ref | AirfRANS | 100000 | 646 (644-651) | 661 (644-692) | 2.4% |
| DTT before-ref | AirfRANS | all | 945 (943-953) | 974 (960-1,006) | 3.0% |
| DTT before-ref | AirfRANS whole boundary | 100000 | 642 (639-646) | 672 (643-716) | 4.7% |
| DTT before-ref | AirfRANS whole boundary | all | 1,092 (1,086-1,096) | 1,124 (1,113-1,176) | 2.9% |
| DTT before-ref | DrivAerML vehicle | 100000 | 1,356 (1,338-1,363) | 1,339 (1,335-1,358) | 1.2% |
| DTT before-ref | DrivAerML vehicle | 200000 | 1,866 (1,859-1,878) | 1,888 (1,865-1,893) | 1.2% |
| DTT before-ref | DrivAerML vehicle | 300000 | 2,415 (2,401-2,423) | 2,426 (2,410-2,441) | 0.4% |
| DTT before-ref | DrivAerML vehicle | 400000 | OOM | OOM | - |
| DTT before-ref | DrivAerML whole boundary | 100000 | 1,448 (1,434-1,453) | 1,475 (1,462-1,492) | 1.9% |
| DTT before-ref | DrivAerML whole boundary | 200000 | 2,073 (2,062-2,081) | 2,105 (2,089-2,113) | 1.6% |
| DTT before-ref | DrivAerML whole boundary | 300000 | 2,687 (2,682-2,695) | 2,707 (2,690-2,736) | 0.8% |
| DTT before-ref | DrivAerML whole boundary | 400000 | OOM | OOM | - |

### Chunking decision and mean loss of the 18 steps (`check_after.py`)

| setting | points | chunking (after) | mean loss f13 | ref | after | after / ref - 1 |
|---|---|---|---|---|---|---|
| aerozoo | 100,000 | every prediction point in one chunk (predicted peak minus capacity -176.5 GiB) | 0.882946 | 0.883012 | 0.882858 | -1.74e-04 |
| aerozoo | 200,000 | every prediction point in one chunk (predicted peak minus capacity -149.4 GiB) | 0.883322 | 0.883343 | 0.883405 | +7.06e-05 |
| aerozoo | 400,000 | every prediction point in one chunk (predicted peak minus capacity -95.3 GiB) | 0.883183 | 0.883334 | 0.883093 | -2.73e-04 |
| airfrans | 100,000 | chunks of 100000 (pointwise loss) | 0.278720 | 0.278855 | 0.278844 | -4.22e-05 |
| airfrans | 180,593 | chunks of 100000 (pointwise loss) | 0.279138 | 0.279328 | 0.279230 | -3.49e-04 |
| airfrans_whole | 100,000 | chunks of 100000 (pointwise loss) | 0.250549 | 0.250533 | 0.250591 | +2.30e-04 |
| airfrans_whole | 180,593 | chunks of 100000 (pointwise loss) | 0.250795 | 0.250746 | 0.250957 | +8.42e-04 |
| drv_body | 100,000 | chunks of 100000 (pointwise loss) | 0.086066 | 0.086131 | 0.086062 | -7.98e-04 |
| drv_body | 200,000 | chunks of 100000 (pointwise loss) | 0.085469 | 0.085412 | 0.085380 | -3.86e-04 |
| drv_body | 400,000 | chunks of 100000 (pointwise loss) | - | - | 0.085856 | - |
| drv_whole | 100,000 | chunks of 100000 (pointwise loss) | 0.095193 | 0.095196 | 0.095154 | -4.36e-04 |
| drv_whole | 200,000 | chunks of 100000 (pointwise loss) | 0.094515 | 0.094574 | 0.094576 | +2.72e-05 |
| drv_whole | 400,000 | chunks of 100000 (pointwise loss) | - | - | 0.094951 | - |

## Surface task (SURFACE harness; before = SURFACE jobs 363161 and 363323 at f13eb5ce0, after = 367436 at 757a5a877; GB = 1e9 B; peak allocated)

| case | count | ISLA-DTT before | ISLA-DTT after | after / before (time, peak) | DTT eval before / after | option A before | option A rerun | after / option A (time) |
|---|---|---|---|---|---|---|---|---|
| DrivAerML run_1, whole boundary | 100,000 | 3,033 ms, 146.1 GB | 984 ms, 122.6 GB | 0.32x, 0.84x | 2,367 ms / 481 ms | 146 ms, 37.0 GB | 147 ms, 36.7 GB | 6.7x |
| DrivAerML run_1, whole boundary | 200,000 | 3,996 ms, 211.3 GB | 1,492 ms, 166.8 GB | 0.37x, 0.79x | 2,888 ms / 698 ms | 274 ms, 71.0 GB | 276 ms, 70.7 GB | 5.4x |
| DrivAerML run_1, whole boundary | 250,000 | 4,832 ms, 243.8 GB | - | - | 3,483 ms / - | - | - | - |
| DrivAerML run_1, whole boundary | 300,000 | - | 1,995 ms, 212.3 GB | - | - / 869 ms | - | - | - |
| DrivAerML run_1, whole boundary | 400,000 | - | 2,493 ms, 257.6 GB | - | - / 1,063 ms | 526 ms, 138.9 GB | 533 ms, 138.6 GB | 4.7x |
| DrivAerML run_1, whole boundary | 600,000 | - | - | - | - / - | 803 ms, 206.8 GB | 811 ms, 206.6 GB | - |
| DrivAerML run_1, whole boundary | 800,000 | - | - | - | - / - | 1,081 ms, 274.8 GB | 1,089 ms, 274.5 GB | - |
| HiLift case 0 | 100,000 | - | - | - | - / - | 119 ms, 53.3 GB | 125 ms, 43.1 GB | - |
| HiLift case 0 | 200,000 | - | - | - | - / - | 223 ms, 87.3 GB | 224 ms, 77.1 GB | - |
| HiLift case 0 | 400,000 | - | - | - | - / - | 437 ms, 155.3 GB | 439 ms, 145.0 GB | - |
| HiLift case 0 | 600,000 | - | - | - | - / - | 660 ms, 223.2 GB | 663 ms, 213.0 GB | - |
| HiLift case 0 | 800,000 | - | - | - | - / - | 890 ms, 291.2 GB | 894 ms, 280.9 GB | - |

### Surface memory by stage (peak allocated GB; before = 363410, after = 367583)

| case | phase | code | cell_measures peak | _gauge peak | exact_boundary_geometry peak | tree_read_pairs peak | run peak | status |
|---|---|---|---|---|---|---|---|---|
| drv_100k | eval | before | 1.9 | 3.5 | 23.3 | 14.4 | 23.3 | ok |
| drv_100k | train | before | 2.0 | 3.5 | 23.3 | 18.2 | 144.2 | ok |
| drv_100k | eval | after | 1.9 | 3.5 | 22.7 | 12.9 | 22.7 | ok |
| drv_100k | train | after | 2.0 | 3.5 | 22.7 | 13.6 | 121.1 | ok |
| drv_200k | eval | before | 1.9 | 3.5 | 23.3 | 26.1 | 26.1 | ok |
| drv_200k | train | before | 2.0 | 3.5 | 23.3 | 33.2 | 207.9 | ok |
| drv_200k | eval | after | 1.9 | 3.5 | 22.7 | 24.2 | 24.2 | ok |
| drv_200k | train | after | 2.0 | 3.5 | 22.7 | 25.2 | 163.8 | ok |
| hl_10k | eval | before | 27.0 | 52.1 | 277.5 | - | 277.5 | failed: CUDA out of memory. Tried to allocate 19.08 GiB. GPU 0 has a t |
| hl_10k | train | before | 27.0 | 52.1 | 277.5 | - | 277.5 | failed: CUDA out of memory. Tried to allocate 19.08 GiB. GPU 0 has a t |
| hl_10k | eval | after | 27.0 | 52.1 | 292.0 | - | 292.0 | failed: CUDA out of memory. Tried to allocate 6.36 GiB. GPU 0 has a to |
| hl_10k | train | after | 27.0 | 52.1 | 292.0 | - | 292.0 | failed: CUDA out of memory. Tried to allocate 6.36 GiB. GPU 0 has a to |
| hl_100k | eval | before | 27.0 | 52.1 | 277.5 | - | 277.5 | failed: CUDA out of memory. Tried to allocate 19.08 GiB. GPU 0 has a t |
| hl_100k | train | before | 27.0 | 52.1 | 277.5 | - | 277.5 | failed: CUDA out of memory. Tried to allocate 19.08 GiB. GPU 0 has a t |
| hl_100k | eval | after | 27.0 | 52.1 | 292.0 | - | 292.0 | failed: CUDA out of memory. Tried to allocate 6.36 GiB. GPU 0 has a to |
| hl_100k | train | after | 27.0 | 52.1 | 292.0 | - | 292.0 | failed: CUDA out of memory. Tried to allocate 6.36 GiB. GPU 0 has a to |

## Records

- `out/out_366717/`, `out/out_367435/`: one JSON and log per run (`<column>_<dataset>_<count>`), with per-step times and peaks; launcher logs `out/run_366717.log`, `out/run_367435.log`.
- `out/surface/after_final/` (cost), `out/surface/mem_after_final/` (memory by stage), `out/surface/mem_after_final_failed_367470/` (the failed first try); before records stay in `../surface/out/`.
- LHR copies: `$T/kernel/dttfast_final/out_<job>/`, `$T/kernel/dttfast_surface/v2/after_final/`, `$T/kernel/dttfast_final/mem/after_final/`.
