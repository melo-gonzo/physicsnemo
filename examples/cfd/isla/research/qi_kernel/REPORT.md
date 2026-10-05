# QI-KERNEL: a GLOBE-style boundary read for read-only ISLA

Lane `isla-qi-kernel`, branch `psharpe/isla-qi-kernel` (base `psharpe/isla-independent`
@ 84b0a4190). AGA task dir:
`/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-10-03-isla-qi-kernel/`
(job prefix `isla.kernel.`). Brief: `BRIEF.md` (this folder).

## Status (2026-10-05 16:05 PDT): complete

- **Verdict: falsified** by the preregistered bars. Kernel / interacting-256 at
  epoch 500 (`infer.py`, float32, seed means): DrivAerML Volume 1.162 / 1.317 / 1.202
  (pressure / velocity / eddy viscosity), AirfRANS 0.795 / 0.995 / 1.239. Against
  read-only-256: DrivAerML 1.049 / 1.075 / 1.024, AirfRANS 0.766 / 0.915 / 0.874.
- **Eddy viscosity fails on both datasets for a structural reason:** its excess sits
  in the wake beyond 1 L (2% of DrivAerML points, 61% of the squared error; swapping
  that band for the interacting model's leaves 1.02-1.03), where a read of radius
  0.01 L reads nothing. Pressure and velocity are helped where the read reaches: on
  AirfRANS pressure ends 20% below interacting.
- **Caveat on DrivAerML pressure and velocity: arithmetic dependence.** The
  bf16-trained DrivAerML kernel models score much better in bf16 than in the
  protocol's float32 (velocity 0.0635 against 0.0888 on seed 42); the references do
  not (they get 1-18% worse in bf16). Scored in bf16 throughout, kernel /
  interacting is DrivAerML 1.051 / 0.954 / 1.301 and AirfRANS 0.844 / 0.980 / 1.221:
  still falsified (eddy viscosity). The read's own arithmetic is not the cause: the
  dependence sits in the token embeddings (rounding only their inputs to bf16
  recovers the bf16 score in an otherwise float32 model); see Results.
- Contracts hold on all four final checkpoints (float64 independence 3e-15 to 3e-12
  of the RMS target; posed residual at or below the read-only class's level).
- Nothing is running. Commits: a7be6350c (the read), be04b6584 (Triton segment
  kernels, shared basis), 9c564e99a (stored pair tensors; the full lanes' code),
  report and diagnostic commits after.

## Question

Read-only ISLA (`query_independent=True`) gives each prediction point 256 slice
summaries per block and its distances to 256 slice anchors. It is 1.04x to 1.42x less
accurate than interacting ISLA (QI campaign, arm A). GLOBE instead sums a learned,
area-weighted kernel over every boundary face. Does a GLOBE-style read of the
individual boundary cells close the gap to interacting ISLA while keeping exact query
independence and every ISLA invariant?

## Design (`ISLA(kernel_read="mean" | "sum", kernel_read_radius=R, kernel_read_blocks=m)`)

Code: `physicsnemo/experimental/models/isla/kernel_read.py` (the read),
`pair_kernels.py` (Triton segment sums), `model.py` (wiring). Off by default; the
read-only path, the interacting path and every existing golden are unchanged.

After each of the last `m` slice blocks (default every block, matched depth), every
prediction point `q` gets a residual update read from the boundary cells `j` within
radius `R L` (L the model's length unit):

- weights: `omega_j = mu_j / sum_i mu_i` (the cell's share of the HT-corrected
  boundary measure, `log_w`), falloff `b = (1 - s^2)^3`, `s = |x_q - y_j| / (R L)`
  (zero with its first two derivatives at the radius);
- kernel: invariants of the pair `iota = [s, log s, rho.n_j, rho.n_q, n_q.n_j, rho.g_k,
  n_j.g_k, rho.v, n_j.v]` (rho the unit separation, n_j the cell normal, n_q the point's
  own direction to the wall, g_k the global directions, v the point's rank-1 inputs,
  i.e. the stored SDF gradient; boundary rank-1 inputs enter the same way) through one
  shared MLP to 16 basis functions per pair, combined linearly per block and head
  (8 heads);
- values: a linear map of the cell's LayerNormed token state at that depth
  (64 channels, 8 per head); output: a linear map back to the token width, zero at
  initialization (the model starts as the plain read-only model), no bias (a point
  with no cell within R reads exactly zero);
- `"sum"` (GLOBE-like): `read_h = exp(gamma_h) sum_j omega_j b K_h v_jh`, an
  unnormalized quadrature with a signed learned kernel;
- `"mean"` (ISLA-like): a softmax over the point's cells of
  `K_h + <q_h, k_jh>/2 + log omega_j + log b`, against a null of learned mass
  `exp(beta_h)` (a fraction of the whole boundary measure) carrying value zero, so the
  read fades smoothly as the neighbourhood empties; q, k are linear maps of the
  point's and the cell's states (4 channels per head).

How each requirement is met:

| requirement | how |
|---|---|
| query independence | the read uses boundary-token states (which in read-only mode never see a point) and the point's own state; points still carry log-weight -inf in every slice softmax |
| covariance | kernel inputs are dot products of unit vectors and lengths in units of L; values are invariant states; the equivariant heads assemble the outputs |
| discretization consistency | weights are measure fractions times a smooth falloff in units of L: a quadrature of a surface integral; the neighbourhood is the exact radius set, no neighbour count anywhere |
| similarity gauge | positions and R in units of L; the measure fraction is unchanged by areas x k^2 |
| measure scale | only measure fractions enter (and the null is a fraction of the total) |
| cost | exact radius search (Warp hash grid on CUDA in float32, exact chunked cdist otherwise) once per forward; pair sums in Triton segment kernels without atomics; per-block pair tensors recomputed in backward |

## Tests (float64 unless noted; `test/experimental/models/isla/test_isla_kernel_read.py`)

32 tests, both modes; the 170 ISLA tests of the base pass unchanged (202 in all, CUDA
included). With `kernel_read=None` the model owns no new parameter and every existing
golden passes.

- exact query independence: alone, prefix, permuted, chunked (1e-12);
- rotation and reflection covariance with translation, 3-D and 2-D (1e-10);
- similarity gauge (x -> 2.7 x + shift, areas x 2.7^2, the sdf input x 2.7), boundary
  refinement (every cell as two coincident half-measure copies: the pair count doubles
  exactly, nothing is dropped, outputs equal), measure scale (x 137);
- locality: a point with no cell within R gets exactly the plain read-only value; the
  falloff vanishes smoothly at R (difference < 1e-8 just inside it);
- every parameter receives a gradient (every block, last block only; DDP);
- bf16 autocast finite (values and gradients); a rank-1 boundary input enters the
  pair invariants covariantly;
- CUDA: the hash-grid radius search returns the exact set (3-D and 2-D, rim pairs
  aside); the Triton segment sum and pair dot equal the eager reference in float64 and
  float32 (forward and both gradients); the protocol's training path (fused geometry
  kernel, compiled blocks, bf16) is finite, matches the eager path to bf16 roundoff
  and stays query-independent; checkpointed and stored pair tensors give identical
  gradients.

## Boundary statistics that fixed R (held-out cases, protocol sample size)

`boundary_stats.py`; AGA `stats/drvol.json`, `stats/af.json`; three cases each.

| | DrivAerML Volume (200k cells, 200k points) | AirfRANS (~1000 segments, 100k points) |
|---|---|---|
| L | 0.41-0.43 | 0.42-0.43 |
| boundary measure / L^(d-1) | 7.6-8.4 | 4.8-4.9 |
| cell measure / L^(d-1) (1%, 50%, 99%) | 3e-7, 7e-6, 5e-4 | 3e-5, 3e-3, 2e-2 |
| \|sdf\|/L median; 72nd pct | 0.0023-0.0026; 0.011-0.014 | 0.025-0.068; 0.22-0.52 |
| cells within R: R = 0.005 L, mean / 99.9% / max | 37 / 257 / 284; 56% of points have one | 12-15 / 150 / 151; 25-30% |
| R = 0.01 L | 141 / 825 / 875; 67-69% | 24-28 / 200 / 200; 33-39% |
| R = 0.02 L | 445 / 2500 / 2650; 74-76% | 43-50 / 253 / 253; 39-48% |
| R = 0.05 L | 1870 / 9700 / 10100; 82-84% | 84-97 / 326 / 326; 47-57% |

The DrivAerML boundary sample is very non-uniform (cell measures span three decades),
so a fixed neighbour count k would shrink the neighbourhood by 10x in the finely meshed
regions; no fixed k could serve R = 0.01 L without binding (875 cells). The read
therefore takes the exact radius set (ragged pairs), with no k at all.

## Cost (one GB300, protocol request, bf16, compiled blocks, fused geometry kernel)

`bench_step.py`, forward + backward + AdamW, median of steps 4-11; AGA `stats/bench*.jsonl`.

| configuration | pairs | step (ms) | peak (GiB) |
|---|---|---|---|
| DrivAerML read-only (no kernel read) | 0 | 262 | 87 |
| v1 (eager gathers / index_add): mean R=0.01 | 28.4M | 3878 | 126 |
| v1: sum R=0.01 | 28.4M | 1788 | 122 |
| v2 (Triton segment sums): mean R=0.005 | 7.4M | 643 | 91 |
| v2: mean R=0.01 | 28.4M | 1479 | 98 |
| v2: sum R=0.01 | 28.4M | 823 | 97 |
| v2: mean R=0.01, last 4 blocks | 28.4M | 683 | 95 |
| AirfRANS read-only | 0 | 237 | 23 |
| v2: AirfRANS mean R=0.02 | 4.3-5.0M | 461 | 24 |
| v2: AirfRANS sum R=0.02 | 4.3-5.0M | 347 | 24 |
| v2: AirfRANS mean R=0.05 | 8.4-9.7M | 636 | 26 |

| v3 (shared kernel basis, `code_kq2`): mean R=0.005 | 7.4M | 608 | 95 |
| v3: mean R=0.01 | 28.4M | 1354 | 108 |
| v3: sum R=0.01 | 28.4M | 692 | 107 |
| v3: sum R=0.01, last 4 blocks | 28.4M | 437 | |
| v4 (pair tensors stored, `code_kq3`): sum R=0.01 | 28.4M | 630 | 140 |
| v4: mean R=0.01, last 4 blocks | 28.4M | 561 | 112 |
| v4: mean R=0.01 | 28.4M | 1070 | 127 |
| v4: AirfRANS sum R=0.01 | 2.4-2.8M | 285 | 28 |
| v4: AirfRANS mean R=0.01, last 4 | 2.4-2.8M | 274 | 26 |
| v4: AirfRANS sum R=0.01, last 4 | 2.4-2.8M | 255 | 26 |

The cost is per-pair work (about 1.3 ns per pair and block for `"sum"`, 3 ns for
`"mean"`, forward and backward) plus a fixed ~7 ms per block for the dense maps of the
200,000 cell and point states.

## Preregistered bars (from the brief; written before any full lane)

- Equal or better: (read-only + kernel) / interacting-256 <= 1.03 on every field of both
  datasets (seed means), no paired seed above 1.06.
- Falsified: >= 1.10 on any field.
- Between: otherwise; fields above 1.03 named.
- Also reported: against read-only-256, and the DrivAerML band anatomy (error by
  \|sdf\|/L band).

## Pilot (label: pilot; no claim rests on it)

DrivAerML Volume, reporting protocol except 100 epochs (the full schedule's first 100
epochs: StepLR steps at 100, so the learning rate is 5e-4 throughout), seed 42, one
4xGB300 node each, code `code_kq2` (be04b6584). Twin: read-only-256 at the same budget.
Val loss (normalized MSE, 48 validation cars) by epoch; the QI campaign's seed-42
references (interacting `qi_drvol_int`, read-only `qi_drvol_pas`, 500-epoch runs) for
context. Step = training step on 4 ranks (s).

| epoch | interacting (ref) | read-only (ref) | read-only twin | sum, last 4 | mean, last 4 | sum, every block |
|---|---|---|---|---|---|---|
| 4 | 0.01376 | 0.02276 | 0.02267 | 0.02041 | 0.01835 | 0.01849 |
| 8 | 0.00635 | 0.01352 | 0.01373 | 0.01121 | 0.01093 | 0.01095 |
| 12 | 0.00429 | 0.01001 | 0.01014 | 0.00829 | 0.00868 | 0.00810 |
| 16 | 0.00330 | 0.00835 | 0.00793 | 0.00689 | 0.00705 | 0.00699 |
| 20 | 0.00293 | 0.00739 | 0.00719 | 0.00599 | 0.00622 | 0.00607 |
| 30 | 0.00226 | 0.00616 | 0.00536 | 0.00466 | - | - |
| step (s) | 0.30 | 0.30 | 0.29 | 0.48 | 0.76 | 0.80 |

All three read variants cut the twin's val loss by 13-20% at matched epochs up to 30;
they are within the val noise of each other; `"mean"` costs 1.6x `"sum"` at the same
placement and every-block placement 1.7x last-4. Chosen (2026-10-05 02:30, at pilot
epochs 14-30): **`"sum"`, R = 0.01 L, last 4 blocks** (equal accuracy, cheapest).
Later the val-loss advantage shrank (sum, last 4: -13% at epochs 30-50, -7% at 70,
-8% at 99). The val loss is dominated by eddy viscosity, which the read does not help
(below).

Pilot scores at epoch 100 (`infer.py`, float32, 48 held-out cars, relative L2; ratio
to the twin in parentheses):

| pilot | pressure | velocity | eddy viscosity |
|---|---|---|---|
| read-only twin | 0.0676 | 0.1158 | 0.2510 |
| sum, R=0.01 L, last 4 (chosen) | 0.0601 (0.889) | 0.1061 (0.916) | 0.2638 (1.051) |
| mean, R=0.01 L, last 4 | 0.0605 (0.895) | 0.1003 (0.866) | 0.3140 (1.251) |
| sum, R=0.01 L, every block | 0.0619 (0.916) | 0.0991 (0.856) | 0.3149 (1.255) |

Band anatomy (pooled relative L2 by \|stored sdf\|/L band; share of points in
parentheses):

| band | pressure: twin, sum4, mean4, sum12 | velocity: same order | eddy viscosity: same order |
|---|---|---|---|
| < 0.01 (72%) | 0.068, 0.060, 0.060, 0.062 | 0.142, 0.127, 0.118, 0.118 | 0.121, 0.119, 0.117, 0.116 |
| 0.01-0.1 (17%) | 0.061, 0.059, 0.065, 0.063 | 0.063, 0.063, 0.062, 0.060 | 0.119, 0.123, 0.114, 0.117 |
| 0.1-1 (9%) | 0.111, 0.089, 0.107, 0.096 | 0.072, 0.074, 0.077, 0.073 | 0.132, 0.147, 0.140, 0.148 |
| > 1 (2%) | 0.357, 0.299, 0.305, 0.274 | 0.086, 0.091, 0.099, 0.096 | 0.408, 0.430, 0.531, 0.533 |

The read helps where it reads: within 0.01 L of the wall velocity drops 10-17% and
pressure 9-12%; it also lowers pressure further out. Eddy viscosity is a little better
near the wall but 5% to 30% worse beyond 1 L, in the wake, where no point reads
anything (R = 0.01 L), and the per-car eddy-viscosity error is dominated by the wake:
the more reading (every block, or the mean read), the larger that loss. Sum / last 4
is the best compromise of the three and remains the full-lane design.

Contracts on the trained pilot checkpoint (sum, last 4, epoch 100; `qi_probe.py`,
first held-out car in float64 with the exact search): 2,000 sentinels alone against the
native request, max difference / RMS target 7e-15 / 4e-14 / 6e-14 (p / U / nut);
the native request in 5 chunks against one call 2e-14 / 2e-13 / 3e-13: exactly
query-independent. Posed residual (random rotation and translation, float64)
5.9e-4 / 1.7e-4 / 1.6e-4 relative L2, below the twin's 1.0e-3 / 3.8e-4 / 2.0e-4 (the
class's on-wall sign flips of the built-in distance, QI amendment 1; not the read).
Float32: sentinels 4e-5 / 6e-5 / 7e-4, chunked 1e-4 / 1.3e-3 / 8e-4 of the RMS target
(float32 floor; the twin's 8e-5 / 7e-5 / 2e-4 and 6e-4 / 2.7e-3 / 8e-4).

## Lanes

Task dir `agents/2026-10-03-isla-qi-kernel`, `kernel/lanes.tsv`, chained 3h59 links
(`kernel/lanes.sbatch`), first job ids in `kernel/submitted.tsv`.

| idx | run | dataset | design | epochs | code | first job | state (final) |
|---|---|---|---|---|---|---|---|
| 0 | kp_drvol_ro_seed42 | DrivAerML | read-only twin | 100 | kq2 | 1041303 | done (epoch 100) |
| 1 | kp_drvol_sum4_seed42 | DrivAerML | sum, last 4 | 100 | kq2 | 1041304 | done (epoch 100) |
| 2 | kp_drvol_mean4_seed42 | DrivAerML | mean, last 4 | 100 | kq2 | 1041305 | done (epoch 100) |
| 3 | kp_drvol_sum12_seed42 | DrivAerML | sum, every block | 100 | kq2 | 1041306 | done (epoch 100) |
| 4 | kq_drvol_sum4_seed42 | DrivAerML | sum, R=0.01, last 4 | 500 | kq3 | 1041666 | done (links 1041666, 1041676, 1045413) |
| 5 | kq_drvol_sum4_seed43 | DrivAerML | same | 500 | kq3 | 1041667 | done (1041667, 1041837, 1045731) |
| 6 | kq_af_sum4_seed42 | AirfRANS | same | 500 | kq3 | 1041668 | done (1041668, 1041860, 1045835) |
| 7 | kq_af_sum4_seed43 | AirfRANS | same | 500 | kq3 | 1041669 | done (1041669, 1041861, 1045834) |

`code_kq3` (9c564e99a) differs from `code_kq2` only in storing the pair tensors
instead of recomputing them in backward (identical outputs and gradients, tested).
Full lanes: DrivAerML Volume = `drivaer_ml_volume_reference_sdf` / `isla_volume_sdf`,
200,000 cells; AirfRANS = `airfrans_volume_2d` / `isla_volume_airfrans`, 100,000,
`train_split=full_train val_split=full_test +training.val_interval=10`; lr 5e-4, bf16,
no augmentation, compiled blocks, seeds 42 and 43; extra `++model.query_independent=true
++model.kernel_read=sum ++model.kernel_read_radius=0.01 ++model.kernel_read_blocks=4`.
Expected duration: DrivAerML ~8.5 h (step 1.6x read-only), AirfRANS ~8.5 h (1.08x).
The first job ids 1041649-1041652 were cancelled while pending (their job names
clashed with the pilot chains' singleton names); `submit.sh` now prefixes the group.
The third links (a few to 50 epochs each) were moved to QoS `short` with 1.5-2 h
limits while pending (they sat at the bottom of a 1,000-job queue under `normal`);
their no-op successors 1050222-1050227 were cancelled once each lane said "Training
completed". Scoring: watcher 1043128 submitted eval 1051777 (moved to `short`) and
reduction 1051778 at 12:26 PDT.

## Full lanes: progress (training-log validation; no verdict rests on it)

The trainer's per-epoch validation relative L2 (DrivAerML: the validation cars of the
training split; AirfRANS: the 200 test airfoils every 10 epochs, chunk-shaped
requests) against the QI references' logs at the same epoch and seed. Ratios kernel /
interacting and kernel / read-only, pressure / velocity / eddy viscosity. These are
noisy single-epoch values, not the scorer's numbers.

| lane | epoch | kernel / interacting | kernel / read-only |
|---|---|---|---|
| DrivAerML s42 | 140 | 1.00 / 0.97 / 1.60 | 0.92 / 0.81 / 0.97 |
| DrivAerML s42 | 240 | 1.02 / 0.93 / 1.29 | 0.94 / 0.79 / 0.95 |
| DrivAerML s43 | 120 | 1.08 / 1.03 / 1.85 | 0.92 / 0.82 / 1.10 |
| DrivAerML s43 | 230 | 1.09 / 0.99 / 1.50 | 0.95 / 0.80 / 0.98 |
| AirfRANS s42 | 110 | 0.79 / 0.97 / 1.08 | 0.80 / 0.95 / 1.00 |
| AirfRANS s42 | 200 | 0.89 / 0.98 / 1.13 | 0.94 / 0.99 / 1.04 |
| AirfRANS s43 | 110 | 0.84 / 0.99 / 1.21 | 0.85 / 0.98 / 1.10 |
| AirfRANS s43 | 200 | 0.85 / 0.99 / 1.19 | 0.91 / 0.96 / 1.01 |
| DrivAerML s42 | 390 | 1.02 / 0.93 / 1.29 | 0.94 / 0.79 / 0.99 |
| DrivAerML s43 | 370 | 1.09 / 0.99 / 1.38 | 0.95 / 0.80 / 1.03 |
| AirfRANS s42 | 350 | 0.84 / 0.96 / 1.07 | 0.91 / 0.97 / 1.00 |
| AirfRANS s43 | 350 | 0.90 / 1.01 / 1.28 | 0.92 / 0.97 / 1.06 |
| DrivAerML s42 | 499 | 1.02 / 0.93 / 1.32 | 0.94 / 0.79 / 0.97 |
| DrivAerML s43 | 499 | 1.08 / 0.99 / 1.36 | 0.95 / 0.80 / 0.98 |
| AirfRANS s42 | 499 | 0.82 / 0.95 / 1.13 | 0.89 / 0.98 / 1.04 |
| AirfRANS s43 | 499 | 0.88 / 1.01 / 1.28 | 0.94 / 0.97 / 1.06 |

So far the read brings pressure and velocity to the interacting level (DrivAerML) or
below it (AirfRANS pressure), and leaves eddy viscosity at the read-only level. The
reference's interacting / read-only eddy-viscosity ratio closed from about 2x during
training to 1.17x (DrivAerML) and 1.42x (AirfRANS) at epoch 500; whether the kernel
lanes follow decides the verdict.

AirfRANS contracts on a trained checkpoint (`kq_af_sum4_seed42` at epoch 91,
`qi_probe.py`, all 200 test airfoils; first one in float64 with the exact search):
sentinels alone 5e-15 / 4e-15 / 5e-14 and 5 chunks 7e-15 / 4e-15 / 5e-14 of the RMS
target (exactly query-independent); posed residual 8.5e-4 / 4.7e-4 / 2.1e-4 relative L2
in float64 (the 2-D class's known on-wall sign effect; the QI read-only reference
measured 1e-4 to 9.5e-4); float32 sentinels and chunks at 2e-6 to 4e-5.

## Results (epoch 500, reporting protocol)

`infer.py` exactly as the QI scorer (float32, held-out split, the training request;
48 DrivAerML cars, 200 AirfRANS airfoils), eval job 1051777, reduction 1051778
(`eval/kq_reduction.{json,txt}`; copied with the scorer rows and probes to
`examples/cfd/isla/results/qi_kernel_2026-10-05/`). Relative L2, seed mean
(seed 42 / seed 43); the references are the QI campaign's rows.

| DrivAerML Volume | pressure | velocity | eddy viscosity |
|---|---|---|---|
| read-only + kernel | 0.0470 (0.0480/0.0460) | 0.0804 (0.0888/0.0720) | 0.1044 (0.1094/0.0994) |
| interacting-256 | 0.0404 (0.0413/0.0396) | 0.0611 (0.0625/0.0596) | 0.0868 (0.0887/0.0849) |
| read-only-256 | 0.0448 (0.0449/0.0447) | 0.0748 (0.0750/0.0747) | 0.1020 (0.1046/0.0993) |
| kernel / interacting | **1.162** (1.161/1.164) | **1.317** (1.422/1.207) | **1.202** (1.233/1.170) |
| kernel / read-only | 1.049 (1.069/1.030) | 1.075 (1.185/0.965) | 1.024 (1.045/1.001) |

| AirfRANS | pressure | velocity | eddy viscosity |
|---|---|---|---|
| read-only + kernel | 0.0264 (0.0258/0.0270) | 0.0163 (0.0164/0.0163) | 0.0368 (0.0346/0.0391) |
| interacting-256 | 0.0332 (0.0334/0.0331) | 0.0164 (0.0165/0.0163) | 0.0297 (0.0303/0.0292) |
| read-only-256 | 0.0345 (0.0357/0.0332) | 0.0179 (0.0179/0.0178) | 0.0422 (0.0450/0.0393) |
| kernel / interacting | 0.795 (0.774/0.816) | 0.995 (0.992/0.999) | **1.239** (1.143/1.337) |
| kernel / read-only | 0.766 (0.723/0.813) | 0.915 (0.916/0.915) | 0.874 (0.770/0.993) |

**Bars: falsified.** DrivAerML is at or above 1.10 on every field; AirfRANS passes
pressure (0.80) and velocity (0.995, no paired seed above 1.06) and fails eddy
viscosity (1.24).

### The DrivAerML scores depend on the arithmetic (not the protocol; diagnostic)

The DrivAerML kernel lanes score far worse in float32 than their own bf16
training-log validation said (velocity 0.0888 against 0.0640 on seed 42; the
references' velocity scores sit 8-9% below their training-log values, the kernel
lanes' 11% and 39% above). Re-scoring every arm with `infer.py` at `precision=bfloat16`
(`eval/prec2.sbatch`, jobs 1054254 and 1054693; nothing else changed) gives:

| arm, bf16 / float32 score | DrivAerML p / U / nut | AirfRANS p / U / nut |
|---|---|---|
| kernel seed 42 | 0.915 / **0.715** / 1.043 | 1.067 / 1.000 / 1.048 |
| kernel seed 43 | 0.972 / **0.897** / 1.177 | 1.063 / 1.015 / 1.140 |
| interacting seed 42 / 43 | 1.044 / 1.097 / 1.012; 1.042 / 1.100 / 1.034 | 1.015 / 1.030 / 1.093; 0.991 / 1.016 / 1.132 |
| read-only seed 42 / 43 | 1.043 / 1.077 / 1.121; 1.049 / 1.083 / 1.176 | 0.895 / 0.946 / 0.828; 0.967 / 0.970 / 1.082 |

Every arm moves by several percent with the arithmetic (bf16 inference is worse for
the references on DrivAerML), but only the DrivAerML kernel lanes get much *better*
in bf16, which is the arithmetic they were trained in. Scored in bf16 throughout
(every arm alike):

| bf16 scoring | kernel / interacting p / U / nut | kernel / read-only p / U / nut |
|---|---|---|
| DrivAerML | 1.051 / 0.954 / **1.301** | 0.946 / 0.792 / 0.987 |
| AirfRANS | 0.844 / 0.980 / **1.221** | 0.877 / 0.963 / 1.013 |

So in its training arithmetic the read does what the pilot and the training logs
showed (DrivAerML velocity 21% below read-only and 5% below interacting), and the
verdict is the same either way: eddy viscosity is at or above 1.22 on both datasets.
`prec_probe.py` (first 6 held-out cars, eager, AGA `eval_prec/probe/`) switches parts
of the model between the two arithmetics: running the kernel basis MLP or the four
kernel reads in the other arithmetic changes nothing (velocity 0.0945 float32 and
0.0730 bf16 on seed 42 whichever way those two parts run; the reads' updates differ
by 3-15% between arithmetics). Switching larger groups (`--groups 1`, job 1055182,
`eval_prec/probe_groups/`) locates it in the token embeddings (`embed` and the point
tokens' `qt`, which take the points' built-in distance features and stored inputs):

| seed 42, first 6 cars, velocity | float32 | bf16 |
|---|---|---|
| whole model | 0.0945 | 0.0730 |
| embeddings switched to the other arithmetic | 0.0720 | 0.0956 |
| early blocks (0-7) switched | 0.0952 | 0.0720 |
| late blocks (8-11), output head, kernel basis or reads switched | 0.0944-0.0951 | 0.0730-0.0733 |

(seed 43: 0.0803 / 0.0738 whole model, 0.0723 / 0.0817 with the embeddings switched;
the 100-epoch pilot shows the same pattern at a smaller size, 0.1130 / 0.1067.)
Rounding the embeddings' inputs to bf16 alone recovers the bf16 score in an
otherwise float32 model. The trained kernel models have learned to depend on the
bf16 rounding of the token input features; the AirfRANS kernel models do not (bf16
makes them worse), and on DrivAerML the references get 1-18% worse in bf16.

### Band anatomy (DrivAerML, protocol arithmetic)

Pooled relative L2 by \|stored sdf\|/L band from the probes (same 48 cars and points
for every arm), kernel / interacting / read-only; the last column is the case-mean
ratio kernel / interacting after swapping that band's error for the interacting
model's on the same points (`band_swap.py`):

| band (points) | pressure | velocity | eddy viscosity | swap: p / U / nut |
|---|---|---|---|---|
| < 0.01 (72%) | 0.0480 / 0.0410 / 0.0459 | 0.1027 / 0.0760 / 0.0947 | 0.0995 / 0.0955 / 0.0999 | 1.006 / 1.009 / 1.183 |
| 0.01-0.1 (17%) | 0.0391 / 0.0367 / 0.0384 | 0.0367 / 0.0341 / 0.0362 | 0.0709 / 0.0751 / 0.0764 | 1.159 / 1.312 / 1.205 |
| 0.1-1 (9%) | 0.0611 / 0.0573 / 0.0616 | 0.0342 / 0.0331 / 0.0345 | 0.0484 / 0.0459 / 0.0478 | 1.162 / 1.316 / 1.199 |
| > 1 (2%) | 0.2502 / 0.1990 / 0.2392 | 0.0298 / 0.0219 / 0.0312 | 0.1462 / 0.1066 / 0.1386 | 1.162 / 1.316 / **1.024** |

All of the pressure and velocity excess over interacting is within 0.01 L of the wall
(where the read reaches, and where float32 undoes its gain); 61% of the kernel model's
squared eddy-viscosity error and nearly all of its eddy-viscosity excess is in the
wake beyond 1 L, 2% of the points, which an R = 0.01 L read never reaches. Against
read-only the kernel model is a little better in the 0.01-1 L band on eddy viscosity
and worse beyond 1 L (as in the pilot). AirfRANS (probe request, not the scorer's
chunk): the pressure gain is near the wall (0.0163 against interacting 0.0233
within 0.01 L) and the eddy-viscosity excess is in the wake (> 1 L: 0.1150 against
0.0638; swapping that band leaves 1.03).

### Contracts on the final checkpoints (epoch 500, `qi_probe.py`)

Max \|difference\| / RMS target, pressure / velocity / eddy viscosity; float64 on the
first held-out case with the exact search; posed = relative L2 under a random
rotation and translation.

| run | float64: sentinels alone | float64: 5 chunks | float64: posed | float32: 2,000 sentinels, chunks |
|---|---|---|---|---|
| DrivAerML s42 | 7e-15 / 5e-14 / 3e-14 | 2e-13 / 9e-13 / 3e-12 | 4.2e-4 / 1.1e-4 / 8.6e-5 | 3e-5 / 2e-4 / 2e-4; 5e-4 / 2e-3 / 5e-3 |
| DrivAerML s43 | 9e-15 / 1e-13 / 6e-14 | 8e-14 / 4e-13 / 3e-13 | 5.7e-4 / 1.4e-4 / 9.2e-5 | 4e-5 / 2e-4 / 8e-4; 1e-3 / 6e-3 / 3e-3 |
| AirfRANS s42 | 6e-15 / 3e-15 / 9e-14 | 7e-15 / 5e-15 / 9e-14 | 1.3e-4 / 1.8e-4 / 7.2e-5 | 4e-6 / 4e-6 / 3e-5; 5e-6 / 3e-5 / 1e-4 |
| AirfRANS s43 | 6e-15 / 3e-15 / 4e-14 | 7e-15 / 4e-15 / 7e-14 | 2.9e-4 / 2.4e-4 / 1.6e-4 | 4e-6 / 3e-6 / 5e-5; 6e-6 / 4e-6 / 7e-5 |
| read-only ref s42 (QI) | 3e-14 / 7e-14 / 3e-14 | 1e-13 / 4e-13 / 2e-12 | 8.8e-4 / 1.4e-4 / 8.7e-5 | 8e-5 / 6e-4 / 3e-4; 6e-4 / 3e-3 / 1e-2 |

Exactly query-independent in float64 (roundoff), the posed residual at or below the
read-only class's known level (on-wall sign flips of the built-in distance), and the
float32 floors those of the reference.

### Cost of the full lanes

Training wall time including three link start-ups: DrivAerML 8 h 17 min and
8 h 28 min, AirfRANS 8 h 47 min and 8 h 43 min (one 4xGB300 node each; the
read-only references took about 5.2 h and 7.5-8 h). Step 1.6x read-only on DrivAerML,
1.08x on AirfRANS (Cost table above).

## Diagnostic: where the excess error lives (band swap, no training)

`band_swap.py` reads the probes' per-case, per-band sums of squared error and squared
target. Every arm is probed on the same points (checked: same cases, same counts per
band), so swapping one \|sdf\|/L band's error for the reference's error on the same
points is an exact counterfactual. "Swap" is the seed-mean ratio arm / reference of
the case-mean relative L2 (the scorer's statistic) after that swap; "share" is the
arm's share of the summed squared error.

QI references at epoch 500 (`results/qi_2026-10-03/probe`), read-only / interacting,
seeds 42 and 43; ratio before any swap, then share and swap for the near-wall band
(< 0.01 L) and the wake band (> 1 L):

| dataset, field | ratio | < 0.01 L: share, swap | > 1 L: share, swap |
|---|---|---|---|
| DrivAerML pressure | 1.108 | 93%, 1.004 | 0.3%, 1.107 |
| DrivAerML velocity | 1.225 | 91%, 1.009 | 0.6%, 1.223 |
| DrivAerML eddy viscosity | 1.174 | 29%, 1.155 | 58%, 1.029 |
| AirfRANS pressure | 1.017 | 90%, 0.982 | 0.9%, 1.026 |
| AirfRANS velocity | 1.039 | 68%, 1.004 | 4.5%, 1.040 |
| AirfRANS eddy viscosity | 1.250 | 2%, 1.234 | 62%, 1.078 |

(The AirfRANS probe's native request is not the scorer's chunk, so its ratios are
smaller than the scorer's 1.037 / 1.088 / 1.417; the anatomy is what matters here.)

Pilots against the read-only twin (seed 42, epoch 100):

| pilot, field | ratio | < 0.01 L: share, swap | > 1 L: share, swap |
|---|---|---|---|
| sum4 pressure / velocity | 0.889 / 0.917 | 91%, 0.995 / 82%, 1.003 | 0.3%, 0.889 / 2.5%, 0.916 |
| sum4 eddy viscosity | 1.051 | 6%, 1.052 | 84%, 1.007 |
| mean4 eddy viscosity | 1.251 | 4%, 1.252 | 90%, 1.000 |
| sum12 eddy viscosity | 1.255 | 4%, 1.257 | 90%, 1.003 |

So read-only's pressure and velocity excess sits entirely within 0.01 L of the wall
(where the read reaches: the pilots' gain is entirely there), and its eddy-viscosity
excess sits mostly in the wake beyond 1 L (2% of DrivAerML points, 58% of the
squared error), where an R = 0.01 L read reads nothing. Every pilot's eddy-viscosity
loss is entirely in that wake band. A read of radius 0.01 L can close the pressure and
velocity gap but cannot by itself close the eddy-viscosity gap.

## Open issues and next steps

- **Eddy viscosity in the wake** (the reason for the verdict). Read-only's and the
  kernel model's eddy-viscosity excess lives beyond 1 L, out of an R = 0.01 L read's
  reach, and every read variant made it slightly worse there. A wider radius is
  unaffordable at full boundary resolution on DrivAerML (1,900 cells per point at
  0.05 L). Candidates: a second read at a larger radius over a measure-preserving
  coarsening of the boundary (an HT-weighted subsample is a quadrature, but not
  exactly refinement-invariant), or the support-at-inference route (QI arm C), which
  already matches the interacting model on every field. Not launched.
- **Arithmetic dependence of the DrivAerML kernel models.** They score 11-39% worse in
  float32 than in their bf16 training arithmetic on velocity; the references move
  1-18% the other way. The read's own operations are not the cause; the token
  embeddings are (`prec_probe.py`, Results). Any follow-up should score in both
  arithmetics; a fix (the embeddings, or the whole point-token input path, in
  float32 under autocast) would need retrained lanes to test, and would change only
  the pressure and velocity story, not the eddy-viscosity verdict.
- **Cost.** 1.6x the read-only step on DrivAerML (sum, last 4 blocks), 1.08x on
  AirfRANS; the remaining per-pair work is torch elementwise ops around the Triton
  segment sums (a fused per-pair kernel would remove most of it).
- The read's cost and the pair count grow with the boundary sample near the points
  (a quadrature at full boundary resolution): hundreds of cells per point at
  R = 0.01 L in DrivAerML's finely meshed regions.
- Re-running the reduction by hand: `python
  examples/cfd/isla/research/qi_kernel/kq_reduce.py --kernel-scores <task>/eval/score
  --ref-scores <qi task>/eval/score --kernel-probes <task>/eval/probe --ref-probes
  <qi task>/eval/probe --out kq_reduction.json`; the bf16 table with
  `--kernel-scores` and `--ref-scores` both at
  `results/qi_kernel_2026-10-05/score_bf16`.
