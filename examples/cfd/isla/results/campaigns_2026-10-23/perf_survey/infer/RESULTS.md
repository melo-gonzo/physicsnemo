# PERF-SURVEY/INFER: measured details (written 2026-10-25 09:03 EDT)

Code: ISLA 0.13.0, `$T/code_testbuild_v-cpu-base` (b86c7391d); recipe `$T/recipe_perfsurvey_b86c`; GB300 (LHR);
float32 (the scorers' precision), random weights (seed 42, the reads' output projections redrawn N(0, 0.02^2) as
QCHUNK's equiv.py does, so the reads contribute). Raw JSON and tables: `out/main_381962/`, `out/main2_382065/`,
`out/load_382145/`; `out/TABLES_381962.md`, `out/TABLES_382065.md` (`lhr/reduce.py`). Probes: `lhr/infer_probe.py`,
`lhr/geo_overflow.py`, `lhr/load_probe.py`. Jobs: `JOBS.md`.

"Today's forward" = `ISLA.forward` (`model._forward_tokens`): the prediction points ride through every slice block as
N + Q passive tokens. "Read path" = the same arguments through `ISLA.query_chunks` under `no_grad`, `chunks(a, b)` over
point ranges: the boundary side once, the points through `_SliceBlock.read` and the kernel reads per chunk. 0.13.0's
`forward` does NOT route the passive points through `block.read`; only `query_chunks` (the training step) does.

## 1. What the scorer does today

- Scorers run `src/infer.py` per lane: float32, `sampling_resolution` 200,000 (the training draw, not full
  resolution), one unchunked `model(**forward_kwargs)` per case under `no_grad`, each case loaded synchronously with
  `dataset[idx]` in the main loop (the DataLoader's prefetch is bypassed). AeroZoo's scorer
  (`kernel/azdecide/eval_azi`) keeps `compile_blocks=true`; the REF-090-FIX rows and SLICES-DRV's second scorer run fully
  eager (`compile=false model.compile_blocks=false`).
- Logged per-case wall at 0.9.0-era code: AeroZoo ~5 s, DrivAerML (interacting, 128 slices) 2.1 s compiled / 3.6 s
  eager, AirfRANS 1.1 s, lug 0.4 s. Scoring is 0.7-2.5 GPU-h per lane against 80-300 GPU-h of training
  (FINAL_RERUN_PLAN.md), under 2 % of the program.

## 2. The scorer loop at 0.13.0 (s per case, medians of 5 cases, compiled blocks, GB300)

| case set | load (sync) | forward | metric + .cpu() | total | prefetch loop total (warm cache, see note) |
|---|---|---|---|---|---|
| DrivAerML Volume whole boundary, 200k | 1.50 | 1.71 | 0.002 | 3.23 | 1.75 |
| AeroZoo full.val, 200k | 2.80 | 1.06 | 0.003 | 3.84 | 1.08 |

- First case: compile 33 s (DrivAerML) / 38 s (AeroZoo) on cold private caches; then no recompiles over the 5 cases.
- Note: the prefetch loop re-read the 5 cases the synchronous loop had just read; a warm re-read takes 0.08-0.24 s
  against 1.3-3.2 s cold, so its totals are warm-cache numbers. Cold-cache check on a CPU node (job 382145, AeroZoo,
  8 cases per arm, stand-in forward 1.06 s): synchronous 4.17 s per case (load 2.81 s median), prefetch depth 1
  2.75 s (-34 %), depth 4 3.22 s (loads do not overlap with each other). Expected on GB300 with depth 1:
  DrivAerML ~1.75 s per case (forward-bound, load 1.5 < forward 1.7; -46 %), AeroZoo ~2.8 s (load-bound; -27 %).
- Exactness of prefetch: the loaded targets are bitwise equal on 10/10 cases (per-index generators); the metrics are
  bitwise equal on 9/10. The tenth (AeroZoo case 0, run right after the compile) differs by 6e-8 relative, and the same
  case loaded synchronously later reproduced the prefetch loop's value, so the difference is the forward's own
  run-to-run nondeterminism (DTT-KNOBS: the read's atomics are not bitwise deterministic), not the data.

## 3. Forward variants on one case at the scorer protocol (200k points; ms; peak GiB)

Bench: median of 10 after 2 warm-ups (IQR 10-260 ms: four processes shared the node). Phase pass: one synchronized
pass with timers (cleaner totals).

| variant | DrivAerML bench ms | peak | phase total | AeroZoo bench ms | peak | phase total |
|---|---|---|---|---|---|---|
| today, compiled, no_grad | 1860 (rep 1894) | 31.4 | 1709 | 1147 (rep 1251) | 12.9 | 1063 |
| today, compiled, inference_mode | 1916 | 31.4 | | 1144 | 12.9 | |
| read path, compiled, 1 chunk | 1744 | 32.5 | 1577 | 1042 | 9.4 | 930 |
| read path, compiled, 100k chunks | 1791 | 21.1 | | 996 | 5.6 | |
| today, eager, no_grad | 2061 (rep 1898) | 31.4 | 1836 | 1354 (rep 1353) | 19.8 | 1193 |
| today, eager, inference_mode | 1937 | 31.4 | | 1361 | 19.8 | |
| read path, eager, 1 chunk | 1735 | 32.5 | 1674 | 1201 | 12.2 | 1032 |
| read path, eager, 100k chunks | 1742 | 21.1 | | 1070 | 8.8 | |

Phase split (compiled; ms): DrivAerML today: geometry 418, blocks 786 (400k tokens), kernel reads 159, pairs + plan +
basis 254, forest + walk 24, head 26. Read path: blocks 402 (boundary) + block reads 267, head 3.4, the rest equal.
AeroZoo today: geometry 77, blocks 788, kernel reads 61, pairs 64, head 26; read path: 402 + 268, head 3.4.
So the read path saves 117-118 ms of blocks and 22 ms of head frame per forward at 200k points (float32), -8 % (DrivAerML)
and -12.5 % (AeroZoo). PROFILE-2's bf16-training estimate was ~15 ms.

Exactness at 200k points (float32), output l2 relative difference (max over fields) / largest scorer-metric change:

| comparison | DrivAerML | AeroZoo |
|---|---|---|
| inference_mode vs no_grad (eager and compiled) | bitwise | bitwise |
| today compiled vs today eager | 3.6e-6 / 4.4e-7 | 1.3e-6 / 1.3e-7 |
| read path (any chunking, eager or compiled) vs today eager | 2.1e-4 / 2.2e-5 | 4.5e-6 / 7.6e-7 |

float64 check on 20k points of the same case (distance to today's float64 eager forward, l2 relative; ratio to
today's float32 eager forward's distance):

| | DrivAerML velocity / pressure / nut | ratios | AeroZoo velocity / pressure / nut / tke / omega | ratios |
|---|---|---|---|---|
| today f32 eager | 1.48e-3 / 4.70e-4 / 3.97e-4 | 1 | 8.6e-5 / 3.9e-5 / 2.9e-5 / 3.0e-5 / 7.0e-5 | 1 |
| read path f32 eager | same to 3 digits | 1.00, 1.00, 0.98 | same | 1.00 x 4, 1.00 |
| read path f32 compiled | same | 1.00, 1.00, 0.99 | same | 1.00 x 4, 0.99 |
| read path f64 vs today f64 | 2.8e-15 / 5.9e-15 / 3.5e-15 | 0 | 2.4e-6 / 3.3e-6 / 2.8e-6 / 3.1e-6 / 9.5e-6 | 0.03-0.13 |

- DrivAerML: the two paths are the same computation (float64 agreement at roundoff); the float32 change is summation
  order, inside float32's own error (ratio <= 1.00). Certified within rounding.
- AeroZoo: in float64 the paths differ by up to 1e-5, about the size of their float32 difference, so it is a discrete
  difference, not rounding; it is 10x below float32's own error (8.6e-5) and the ratio stays <= 1.00. The float32
  forward is also not repeatable run to run on this subset (repeat ratio up to 1.012). Likely cause, not isolated: the
  AeroZoo box faces' axis-aligned normals put forest decisions (octant ties, DTT-KNOBS / DTT-POSE) on rounding residues
  that summation order and atomics flip. Class: numerics change with a measured bound (<= 1e-5 relative in float64).
- Random weights: DrivAerML's float32 floor (1.5e-3 from float64) may differ on trained checkpoints.

## 4. Memory and time against the number of points (eager; median of 3 after 1 warm-up)

| points | DrivAerML today | read path, 100k chunks | AeroZoo today | read path, 100k chunks |
|---|---|---|---|---|
| 200k | 2.14 s / 31.5 GiB | 1.88 s / 21.2 GiB | 1.47 s / 20.0 GiB | 1.15 s / 9.0 GiB |
| 500k | 4.19 s / 75.4 GiB | 3.06 s / 21.2 GiB | 2.48 s / 37.9 GiB | 1.90 s / 9.1 GiB |
| 1M | 6.44 s / 148.7 GiB | 5.19 s / 21.4 GiB | 4.18 s / 67.6 GiB | 3.06 s / 9.1 GiB |
| 2M | OOM (a 100 GiB allocation: the materialized pair basis) | 9.30 s / 21.5 GiB | - | 5.29 s / 9.3 GiB |
| 5.87M (AeroZoo full resolution) | - | - | crash: illegal memory access | 14.3 s / 10.7 GiB |

- Today's peak grows 0.15 GiB (DrivAerML) and 0.06 GiB (AeroZoo) per 1,000 points (the all-point walk lists and the (pairs, 16) float32 basis, plus
  N + Q tokens through the blocks); the read path stays at the boundary side's peak.
- The read path with one chunk of all points has today's memory (OOM at 2M); the gain is in chunking.
- Today's forward also walks all points at once, so it meets the walk's 2^31-pair limit (reference fallback) from
  ~2.6-3.1M points on DrivAerML (PROFILE-2); the read path walks per chunk.

## 5. The int32 overflow in the fused geometry kernel (job 382065, `lhr/geo_overflow.py`)

`geo_kernel.py` indexes logits, bias and mix with `offs_n[:, None] * l_sn + offs_s[None, :]` in int32 (lines 242, 252,
257 forward; 419 backward). With tokens x slices >= 2^31 (4,194,304 tokens at 512 slices) the offsets wrap:

| tokens N (S = 512) | N*S >= 2^31 | result |
|---|---|---|
| 4,194,240 | no | matches the eager region to 9.5e-7 |
| 4,194,368 | yes | CUDA illegal memory access |
| 6,074,816 (AeroZoo full case + 200k boundary tokens) | yes | CUDA illegal memory access |

The fused kernel is the default on CUDA (`resolve_geo_kernel` has no size guard), so any forward with that many tokens
in one block call crashes (or, if the wrapped offset lands inside the allocation, could read wrong rows silently). The
read path keeps each block call at the chunk's points or the boundary's tokens, far below the limit. Fix: int64
offsets (`offs_n.to(tl.int64)`) in both kernels, a few lines, with a test just over the limit (memory ~27 GB).

## 6. Chunk size of the read path (eager; median of 3)

| chunk | DrivAerML 1M | DrivAerML 200k | AeroZoo 1M | AeroZoo 200k |
|---|---|---|---|---|
| 25k | 5.45 s / 21.2 | 1.79 s / 21.1 | 3.26 s / 9.1 | 1.15 s / 8.8 |
| 50k | 5.18 / 21.2 | 1.75 / 21.1 | 3.06 / 9.1 | 1.06 / 8.8 |
| 100k | 5.11 / 21.2 | 1.73 / 21.1 | 2.97 / 9.1 | 1.05 / 8.8 |
| 200k | 5.07 / 32.5 | 1.72 / 32.5 | 2.94 / 12.5 | 1.04 / 12.2 |
| 400k | 5.09 / 61.7 | - | 2.91 / 22.9 | - |

Time is flat from 50k to 400k (within 1-3 %); memory steps up above 100k on DrivAerML. So the rule only sets memory:
chunks of `n_boundary_tokens` points (the point side never holds more tokens than the boundary side; 200k here) cost
+11 GiB on DrivAerML and +3.4 GiB on AeroZoo over 100k chunks for 1 % less time; either is far below today's peak.

## 7. Not worth it (measured)

- `inference_mode` instead of `no_grad`: bitwise identical outputs, no measurable time or memory change (table 3).
- Compiled blocks for scoring: -7 % (DrivAerML) and -11 % (AeroZoo) per forward at 200k (phase totals), against a
  33-38 s compile on the first case (cold caches): a net loss on a 48-case DrivAerML split, break-even on a 333-case
  AeroZoo test side, and another compiled path to certify. Keep the eager scorer rows.
- "Skipping the boundary side per chunk": query_chunks already runs the boundary side once per case; nothing to skip.
