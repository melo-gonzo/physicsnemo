# HL-AREA-COST: training cost of the six HL-DUP stage-2 lanes beside the stored-wing HiLift lanes (2026-10-24, written 21:05 EDT)

Rule: the cost_090.json "protocol" rule, unchanged. step = median of train_step rows with step_time_s < 10 s, seed mean;
peak = max over epochs of train_summary peak_memory_gib, seed mean; n_parameters from the config row (asserted equal to the
scorer's row). GPU-h = sum over a chain's links of sacct ElapsedRaw x AllocTRES gres/gpu, failed and requeued attempts included.
Wall = the same sum of ElapsedRaw. Stored = book `cost.hilift` (h090_hl_isla, udrv_hl_gt, udrv_hl_t). Its GPU-h and wall were
not in the book; I computed them the same way from the stored runs' own link job ids. Ratio = dedup / stored.

| arch | metric | dedup s42 / s43 | dedup mean | stored s42 / s43 | stored mean | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| ISLA | step_seconds | 0.3220 / 0.3240 | 0.3230 | 0.3409 / 0.3418 | 0.3414 | 0.946 |
| ISLA | peak_gib | 64.61 / 64.61 | 64.61 | 66.48 / 64.65 | 65.57 | 0.985 |
| ISLA | n_parameters | 68,817,006 | | 68,817,006 | | 1 |
| ISLA | GPU-h | 32.84 / 32.70 | 32.77 | 32.43 / 32.63 | 32.53 | 1.007 |
| ISLA | wall h | 8.21 / 8.17 | | 8.11 / 8.16 | | |
| GT 256 | step_seconds | 0.4336 / 0.4277 | 0.4307 | 0.4566 / 0.4579 | 0.4572 | 0.942 |
| GT 256 | peak_gib | 77.86 / 77.86 | 77.86 | 77.91 / 77.91 | 77.91 | 0.999 |
| GT 256 | n_parameters | 9,047,301 | | 9,047,301 | | 1 |
| GT 256 | GPU-h | 38.49 / 38.14 | 38.32 | 38.86 / 39.47 | 39.17 | 0.978 |
| GT 256 | wall h | 9.62 / 9.54 | | 9.72 / 9.87 | | |
| T 512x16 | step_seconds | 0.3218 / 0.3237 | 0.3228 | 0.3408 / 0.3431 | 0.3420 | 0.944 |
| T 512x16 | peak_gib | 62.26 / 62.26 | 62.26 | 62.31 / 62.31 | 62.31 | 0.999 |
| T 512x16 | n_parameters | 47,239,305 | | 47,239,305 | | 1 |
| T 512x16 | GPU-h | 32.47 / 32.18 | 32.32 | 31.57 / 32.32 | 31.94 | 1.012 |
| T 512x16 | wall h | 8.12 / 8.04 | | 7.89 / 8.08 | | |

GPU-h totals: the six dedup lanes 206.8 (ISLA 65.5, GT 76.6, T 64.6); the six stored lanes 207.3. Not counted in the lane
totals: smokes 367139 and 367146, 1.49 GPU-h; scoring 371516_0, 371516_2 and 372745_1, 2.79 GPU-h.

Every lane on both sides ran bfloat16 on 4 GPUs with 4 ranks: one lane per 1-node, 4-GPU array job, `torchrun --nproc-per-node 4`
(lanes.sbatch 8192a90b and 8eecec8f). No job carried more than one lane, so no GPU-h is shared between lanes. All runs completed
500 epochs. Data: `cost/cost_hldupx.json` (cost_090 schema under `cost.hilift_dedup`, plus per-seed gpu_hours, wall_hours,
links and metrics md5; `gpu_hours_total` per arch; the stored lanes under `stored_gpu_hours`).

## Jobs (from each train.log's `=== link start ... job=` lines, cross-checked with sacct; LHR times -07:00)
- ISLA s42 367678, 367684, 369503, 371318. ISLA s43 367679 (FAILED after 293 s, 0.33 GPU-h, included), 367683, 367733, 369595, 371387.
- GT s42 367681, 367709, 369504, 371317. GT s43 367682, 367710, 369538, 371356.
- T s42 367686, 367711, 369539, 371357. T s43 367687, 367723, 369540, **371355** (a 9 s closing link the brief's list omits).
- Stored: h090 ISLA 300297, 300400, 302845, 304731 / 300298, 300401 (FAILED after 132 s), 302846, 302865, 304768. udrv GT
  288792, 289797, 294236, 297542 / 288793, 289824, 294262, 297539. udrv T 288795, 289827, 294260 / 288796 (a NODE_FAIL attempt
  of 705 s, 0.78 GPU-h, seen only with `sacct -D`, then requeued), 289825, 294590.

## Commands (in `cost/`)
- `ssh lhr2 "nice -n 19 <td14 venv python> - <6 runs>" < extract_cost.py > extract.jsonl`. extract_cost.py is copied unchanged
  (md5 9b87878e). The same command on the six stored runs gives `extract_stored.jsonl`. Its metrics md5s equal the book's
  extract.jsonl, and the reducer asserts that it reproduces the book's per-seed step and peak values exactly.
- `grep -o '^=== link start [^ ]* job=[0-9]*' runs/<run>/train.log` > link_jobs.txt; `sacct -D -X -P -j <ids>` > sacct_links_D.psv
  (also sacct_stage2.psv, sacct_all_hldupx.psv, sacct_extra.psv); `python reduce_cost.py` > cost_hldupx.json.

## Caveats
- The like-for-like check passes. ISLA: code_h090, compile=False, model.compile_blocks=true, the same venv (.venv-recipe) as h090,
  so both sides are "before the compile fix". GT and T: code_a2r, uncompiled, as udrv. GPUs, ranks and precision are equal.
  The recipe snapshots differ only by the dedup files (recipe_hldupx_* = recipe_h090 / recipe_a2r_hl + dedup) and the dataset.
- The step is 5-6 % faster, but it does less work. The dedup reader drops duplicates from the sampled cells: 185.4-186.1 k of
  200 k (ISLA, GT) and 92.7-93.1 k of 100 k (T) are kept, 7 % fewer, by the JOBS.md smoke counts. So step_seconds is per
  step, not per cell, and the ratio is about what the cell count predicts.
- GPU-h does not follow the step (0.98-1.01x). The summed epoch time per lane is 7.7-9.2 h, against about 64 k steps x 0.32-0.43 s
  = 5.7-7.7 h of steps, so loading and validation take up the rest. Six dedup lanes read the data at once (JOBS.md: walls of
  59-80 s in the early epochs). GPU-h are therefore a cost of these runs on a shared cluster, not of the method; treat
  differences of ±3 % as noise.
- The failed links are counted on both sides: dedup ISLA s43 0.33 GPU-h, stored ISLA s43 0.15, stored T s43 0.78. Without
  them, the T ratio would be about 1.02 and the ISLA ratio about 1.004.
