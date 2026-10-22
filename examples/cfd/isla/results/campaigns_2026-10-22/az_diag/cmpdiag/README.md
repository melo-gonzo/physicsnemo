# CMPDIAG: is AeroZoo exposed to the compiled-block bug (pytorch#195845)? (AZ-DECIDE sub-lane, 2026-10-22)

Written 10:40 EDT, 2026-10-22. Reading: **not exposed.** On every case, compiled and eager agree far more closely than
float32 eager agrees with float64 eager. N % 4 is 0 on every case, as the config makes it, and the geometry runs in the
fused kernel. That gives two independent reasons why the #195845 fusion cannot fire at the scorer's inputs.
Side finding (not part of the reading rule): the float32 floor on AeroZoo is large. On 4 of 20 ISLA cases and 6 of 20
ISLA-DTT cases, float32 eager and float64 eager differ by more than 1e-2 of RMS(target), up to 0.28 (§4).

## 1. Method

- Scripts (scripts/): `cmpaf_diag.py` and `cmpaf_diag_f64.py` are the reviewer's copies in `$T/review_2026-10-22/`
  (md5 234e44ef / 84100a1f; the reviewer's cmpaf_diag.py = COMPILE-AF's plus a pre-0.9.0 branch that is unused here).
  `patch_readouts.py` applies the same additions to both and leaves the logic alone: per field `max_abs` and
  `rel_l2_vs_eager` = ||pred_x − pred_e|| / ||pred_e||, an input hash per case (`input_sha1`, interior and boundary
  points), and `--redact-abs`, which drops absolute per-case metrics so that no test-side score is written or read.
  Final md5: cmpaf_diag.py e356f46a, cmpaf_diag_f64.py 3bc692b8. Launcher `cmpdiag.sbatch` (2b357638); rows `rows.tsv`
  (78c11221); reducer `reduce_cmpdiag.py` → `tables.md`, `cases_<run>.tsv` (per field).
- Inputs are the scorer's (kernel/azdecide/eval_azi/eval_test.sbatch): code_r090 and recipe_r090, .venv-recipe
  (torch 2.12.0+cu130), `model=isla_volume_aerozoo`, `train_split=full.train val_split=full.test infer_split=test`,
  200,000 interior points, float32, `compile=false`. The script instantiates the model with
  `model_output_stats` + `model_dataset_constants` (the isla_constants injection) and loads the persisted norm_stats.pt
  (it matched the yaml; no "differs" message). Cases: the first 20 of full.test in loader order (00002 … 00093). No
  reference metrics file was read (`ref = -`).
- Per case, in one process: eager float32 (`model.compile_blocks=false`), then compiled, i.e. the model's own
  `_block_callables` with `compile_blocks` on, at default Inductor options, so mix-order reduction is on, as in the
  code_r090 lanes. `--blockwise` feeds each compiled block the eager block's inputs; for ISLA-DTT, `--qi` runs the
  query-independence probe. In a second process (`--f64`), the other path is a deep copy of the model in float64,
  eager, run on the same float32 inputs cast to float64: this is the floor.
  The input hashes match between the two processes on 40 of 40 cases, so both saw the same draw.
- d = RMS(pred_x − pred_e) / RMS(target), per field; velocity uses the vector norm. N = the token count at block 0
  (`blocks[0].n_tokens`), the count that COMPILE-AF's N % 4 rule uses.
- Checkpoints: azc_full_isla_seed42 at epoch 250 (final). azc_full_dtt_seed42 at **epoch 155**, its latest; it is still
  training. That checkpoint was read from a pinned copy, `$T/kernel/azdecide/cmpdiag/ckpt/` (md5-verified against runs/:
  ISLA.0.155.mdlus c3fdc652, norm_stats.pt ac2d456d).

## 2. Job

LHR job **333632** (`isla.azdecide.cmpdiag`, short QoS, 1 node × 4 GPUs, 07:17-07:36 PDT; about 1.3 GPU-h allocated,
about 0.7 GPU-h used: the compiled units took 4 min, the f64 units 17-19 min). All four units rc=0. Cluster dir
`$T/kernel/azdecide/cmpdiag/` (log cmpdiag-333632.log, out/<run>.<tag>.log, out/<run>/<tag>/cases.jsonl). Local
copies are in out/. Registry: the existing row `^isla\.(azdecide\.|azc\.test-eval)` covers the job name.

## 3. Results (full per-case tables: tables.md; per field: cases_*.tsv)

| run | N (all cases, all blocks) | N % 4 | geometry | d compiled vs eager: median / max | d f64 vs f32 eager: min / median / max | worst per-field ratio d_cmp / d_f64 |
|---|---|---|---|---|---|---|
| azc_full_isla_seed42 e250 | 400,000 | 0 (20/20) | fused | 1.8e-6 / 4.8e-6 | 3.0e-5 / 5.0e-4 / 2.8e-1 | 0.22 |
| azc_full_dtt_seed42 e155 | 400,000 | 0 (20/20) | fused | 3.4e-6 / 6.9e-4 (1.1e-5 without case 00078) | 1.8e-4 / 4.2e-3 / 4.4e-2 | 0.22 |

- Compiled against eager: the relative L2 against the eager prediction is at most 5.7e-6 for ISLA, and for ISLA-DTT at
  most 5.7e-6 apart from case 00078. The case-metric difference is at most 6e-6 for ISLA and 5e-5 for ISLA-DTT.
  The one-block errors are 1.0e-7 to 8.4e-6 in every block of every case. Dynamo captured 1 graph for ISLA and 2 for
  ISLA-DTT, and nothing recompiled.
- ISLA-DTT case 00078: d = 6.9e-4 (tke) and the qi probe gives compiled 1.2e-3 against eager 1.8e-4. Its block errors
  are only ≤ 6e-6, so the network amplifies roundoff on this case, as COMPILE-AF found for lug seed 43. It is still
  below that case's own floor: d_f64 = 3.4e-3, per-field ratio ≤ 0.22.
- Why N is constant: isla_volume_aerozoo draws `boundary_tokens = 200,000` inside the model, water-filled over the four
  roles. Every block therefore sees 200,000 + 200,000 = 400,000 tokens whenever a case has more than 200,000 boundary
  faces, and these 20 cases have 0.29-1.39 M. So **the N % 4 rule is degenerate here**: no case with N % 4 ≠ 0 exists at
  the scorer's settings. It also needs no such case, since the geometry is `fused` (COMPILE-AF condition 1 fails).
  Only a case with fewer than 200,000 boundary faces could give N % 4 ≠ 0, and the fused geometry would still exclude
  the fusion.
- PROBES-090's 4.9e-4 compared eager with the infer.py scorer at the metric level: azc_full_isla at e155, 40 cases, two
  separate processes. The compiled path does not reproduce it here (same process, same draw: metric difference ≤ 6e-6).
  A 4.9e-4 gap sits inside this checkpoint family's float32 floor (§4). I did not check the source, since that would mean
  reading the test-side score file.

## 4. Side finding: the float32 floor on AeroZoo is percent-level on some cases

Float32 eager against float64 eager, same inputs: ISLA d is above 1e-2 on 4 of 20 cases (00019 0.12 on tke, 00022 0.056,
00045 0.28 on tke, 00083 0.013), and ISLA-DTT on 6 of 20 (up to 0.044). The case metrics differ by up to 8 % (ISLA 00019)
and 4 % (00045). AirfRANS had ≤ 2.3e-4 and the lug s42 about 1e-6. The scorer runs in float32, so some AeroZoo per-case
ISLA numbers carry percent-level float32 sensitivity. Averages over many cases and ratios of arms should mostly absorb
it (the variation looks like noise), but I did not measure that.
I have not established the cause. The two candidates are ill-conditioning of the trained network in float32 (the lug
s43 pattern, here larger), or a float32-sensitive step in the geometry. The coordinates are raw metres, and a
cancellation-prone form such as |x|² + |y|² − 2x·y near walls would do it. The float64 copy may also take a different
geometry kernel; the script does not log which one. A short follow-up would settle it: log the geometry kernel of
model64, and run float64 geometry with float32 blocks.

## 5. What stays on disk

LHR: `$T/kernel/azdecide/cmpdiag/` (scripts, out/ about 1 MB, hydra_cwd/ and .cache/ empty). The pinned DTT checkpoint copy (560 MB)
was deleted after this report was written. Local: this folder (under 1 MB).
