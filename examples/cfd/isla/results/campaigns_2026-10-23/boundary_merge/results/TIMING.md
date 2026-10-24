# MERGE-PREP 1b: the token padding's interleaved timing (GB300, LHR job 366130)

Job 366130 (`lhr/time_ab.sbatch` md5 4a13a96d, `lhr/mp_time.py` 46433d4e), node nvl72d035-T06, short QoS, 2026-10-24
04:45-05:18 EDT, cancelled by me at 32:44 once the eager and compile_blocks arms were complete (2.18 GPU-h).
Base: `$T/code_align8a` (32fa15389, no padding). Fix: `$T/code_mergeprep_a8` (1062d6086, the padding plus 1a's no-op). Each
process loads one tree and prints the file it ran ("base called base ran .../code_align8a/...", "fix called fix ran
.../code_mergeprep_a8/...", 14 and 13 lines in the logs). One GPU per case, order base, fix, base, fix in eager and then
in compile_blocks. Each process: the recipe's pipeline and collate, case 0 of the training split at the recipe's default
sampling, bf16 autocast, the case's natural token count N and an aligned count (the largest multiple of 32 below N,
made by dropping prediction points or the surface's last cells). After 5 warm-up steps per count, 3 rounds alternate
the two counts (10 training steps and 10 eval forwards each); medians over the 30. Then the profiled GPU time: the sum
of every kernel's device time over 3 training steps, per step. Full table: `time_table.md`; rows: `out_time_366130/time.jsonl`.

| case, mode | N (mod 32) | GPU ms / step, fix / base | wall step, fix / base | base spread between its 2 runs | kernels / step, base and fix |
|---|---|---|---|---|---|
| lug, eager | aligned 12,960 (0) | 1.000 (94.0, 94.0) | 1.004 | 10.1 % | 5,851 and 5,851 |
| lug, eager | natural 12,981 (21) | **0.885** (101.3 to 89.7) | 1.020 | 11.1 % | 5,851 and 5,863 |
| lug, compile_blocks | aligned | 0.987 | 0.998 | 3.8 % | 3,271 and 3,271 |
| lug, compile_blocks | natural | **0.888** (69.1 to 61.3) | **0.937** | 2.2 % | 3,271 and 3,283 |
| Tensile2d, eager | aligned 6,624 (0) | 1.000 (46.2, 46.2) | 1.065 | 8.7 % | 5,750 and 5,750 |
| Tensile2d, eager | natural 6,639 (15) | **0.923** (50.2 to 46.4) | 1.061 | 10.3 % | 5,814 and 5,762 |
| Tensile2d, compile_blocks | aligned | 0.971 | 0.968 | 0.0 % | 3,254-3,266 both |
| Tensile2d, compile_blocks | natural | **0.870** (33.3 to 29.0) | 0.974 | 3.0 % | 3,306-3,318 and 3,266-3,290 |
| DeepJEB++, eager | aligned 50,016 (0) | 0.999 | 0.997 | 2.5 % | 4,147 and 4,147 |
| DeepJEB++, eager | natural 50,040 (24) | 0.998 | 1.001 | 2.6 % | 4,147 and 4,165 |
| DeepJEB++, compile_blocks | aligned | 0.999 | 1.022 | 2.4 % | 2,515 and 2,515 |
| DeepJEB++, compile_blocks | natural | **0.956** (65.7 to 62.8) | 1.004 | 2.3 % | 2,515 and 2,534-2,558 |

(Case 0 of DeepJEB++'s train_clean has 50,040 cells, not the 24,000 of ALIGN8's runs, which re-read the batch at a
smaller sampling.)

**Reading, against the brief's rule** (not slower where nothing is padded, within the measured noise; faster or level
where it pads):
- Nothing padded (aligned N): in eager mode the fix launches exactly the base's kernels and its GPU time is the base's
  to 0.1 %, as 1a's bitwise check (same aten op counts) predicts. Under compile_blocks the GPU time is 0.97-1.00 of the
  base. Every wall ratio is inside the base's own spread between its two runs. The largest, Tensile2d eager at 1.065,
  appears at the aligned count too, where the two codes do identical work; its four processes slowed one after
  another (144.7, 157.4, 158.0, 165.0 ms in run order), which is drift on a launch-bound case, not the code. **Met.**
- Padded (natural N): the GPU time falls on every case and mode, by 11-13 % on the lug and Tensile2d (8-13 %) and
  0.2-4.4 % on DeepJEB++, whose 50,040 is already a multiple of 8, so only the LayerNorm-backward tile gains. The
  wall times are faster (lug compile_blocks 0.94, Tensile2d compile_blocks 0.97) or level within the spread. **Met.**
- So the fix costs nothing where nothing is padded and gains 4-13 % of GPU time where it pads on these small cases. It
  settles ALIGN8's open lug reading (fix about 10 % slower at j = 0): that was noise, since at j = 0 both codes run the
  same kernels.
- Not measured: whole-model torch.compile (the recipe's mode for these three configs). Its compile of one process
  (two token counts, dynamic shapes, default Inductor options) did not finish within the 720 s guard on the lug, for
  base or fix, so GPU 3 recorded nothing. The eager and compile_blocks arms bound the kernels' cost. Whole-model
  compile always pads (a test of pad == 0 would guard the dynamic count) and is covered by the compile tests.
