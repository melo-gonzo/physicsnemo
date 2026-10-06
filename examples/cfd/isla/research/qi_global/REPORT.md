# QI-GLOBAL: an untruncated, exact boundary read for read-only ISLA

Lane `isla-qi-global`, branch `psharpe/isla-qi-global` (base
`psharpe/isla-qi-kernel` @ bb877be65). AGA task dir:
`/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-10-05-isla-qi-global/`
(job prefix `isla.global.`). Brief: `BRIEF.md` (this folder).

## Status (2026-10-06 06:10 PDT): references done and scored, G at epoch ~280

- Global read implemented and tested (commits b10170371 eager, c5d7408e9 fused
  Triton; 224 ISLA tests pass). Cost: G (every block) 508 ms per step, 2.1x
  read-only, peak 34.6 GiB.
- **The lanes run on LHR, not AGA** (AGA's queue had about 1,260 higher-priority
  jobs ahead; LHR has the same GB300 hardware, venv and AirfRANS store; details
  under Lanes). LHR jobs: lanes 99985-99994 (started 2026-10-05 20:45 PDT),
  watcher 100013.
- I, P (seeds 44, 45) and T seed 44 finished at 04:20-05:30 and are scored
  (float32 and bf16, probes). T seed 45 lost 55 min to a hung link (all four
  ranks in uninterruptible I/O wait on node nvl72d019-T18 right after resuming;
  cancelled at 05:40, its successor excludes that node and resumed from epoch
  451); it finishes about 06:30.
- G (seeds 42-45) at epoch ~280 at 06:00; at 111 s per epoch it finishes about
  13:00 PDT; the watcher then scores and probes it and runs the reduction
  (`eval/qg_reduction*.txt` in the LHR task dir).
- References over four seeds (float32): the new seeds move the read-only
  baseline: P / I = 0.958 / 1.050 / 1.251 (two QI seeds: 1.037 / 1.088 / 1.417);
  see Results.
- Training log (no verdict rests on it): through epoch 160 G's validation error is
  0.6-0.8x the interacting model's on pressure and velocity and 0.76-0.99x on eddy
  viscosity (Training-log progress).

## Question

An untruncated boundary read has every prediction point read every boundary cell,
with no length scale anywhere. Does it close read-only ISLA's wake eddy-viscosity
gap on AirfRANS, while keeping exact query independence and every ISLA invariant?

## Design (`ISLA(kernel_read="mean", kernel_read_radius=None, kernel_read_blocks=m)`)

Code: `physicsnemo/experimental/models/isla/kernel_read.py` (the global mode beside
the truncated read: `global_pair_invariants`, `global_kernel_basis`,
`_GlobalSoftmaxRead`, `_GlobalKernelRead`), `global_read_kernels.py` (fused Triton
kernels), `model.py` (wiring). Off by default; with `kernel_read_radius=0.01` the
truncated read is unchanged.

After each of the last `m` slice blocks (default every block, matched depth), every
prediction point `q` reads **every** boundary cell `j` (no radius, falloff or
neighbour count):

- logits per head `h`: `K_h(iota_qj) + <q_qh, k_jh> / 2 + log omega_j`, softmax over
  all `N` cells of the case. `omega_j = mu_j / sum_i mu_i` is the cell's share of the
  HT-corrected boundary measure, so the read is a measure-weighted mean over the
  boundary (ISLA's own quadrature form; the slices use the same log-measure bias).
  No null: the read always averages the whole boundary; the zero-initialized output
  projection decides how much of it enters;
- kernel: `K_h = w_h . phi(iota_qj)`, `phi` one shared MLP (10 -> 64 -> 64 -> 16 on
  AirfRANS) of the pair invariants, evaluated once per forward pass and combined
  linearly per block and head (no bias: a per-head constant cancels in the
  softmax);
- invariants `iota_qj`: `r/L`, `log(r/L)`, the self-similar near-wall invariant
  `log(r/|d_q|)` (`d_q` the point's built-in distance to the nearest boundary cell,
  so the invariant is zero at that cell and positive elsewhere), `rho.n_j`,
  `rho.n_q`, `n_q.n_j`, `rho.g_k`, `n_j.g_k`, `rho.v`, `n_j.v` (and `rho.u`, `n_q.u`
  for boundary rank-1 inputs). Logits linear in `log r` or `log(r/|d_q|)` are power
  laws, so any locality is learnable and is measured in units of L or of the
  point's own wall distance;
- values, keys, queries: linear maps of the LayerNormed boundary-cell and point
  states at that depth (8 heads, 8 value and 4 key channels per head); output: a
  linear map back to the token width, zero at initialization (the model starts as
  the plain read-only model), no bias.

Evaluation is exact, all pairs (about 1e8 per block at the protocol's 100,000
points and about 1,000 cells):

- **eager path** (CPU, float64, the tests' reference): dense tiles of query rows;
  the kernel basis tile by tile under checkpoint; the read as a custom autograd
  function that recomputes each tile's softmax in backward (flash-attention style),
  so no `(pairs, heads)` tensor is stored;
- **fused path** (CUDA below float64, used by the lanes): one Triton kernel computes
  the invariants and the MLP per tile of pairs and stores only the basis (16
  numbers per pair; backward recomputes both and reduces the parameter gradients
  per program, then sums: no atomics); the per-block kernel `z_h = w_h . phi` is
  one cuBLAS product, and the softmax read is an online-softmax Triton kernel over
  cell tiles (forward), with two backward kernels (over query tiles: the per-pair
  logit gradient and `dq`; over cell tiles: `dk`, `dv` summed over every query).
  Accumulation is float32; tile products take bfloat16 operands when the states are
  bfloat16 (autocast training, as flash attention does) and are exact float32 in
  float32 (scoring).

### Every constant, classified

| constant | value | kind | why |
|---|---|---|---|
| `eps` (model's) | 1e-12 | numerical guard | floor on lengths inside `log` and the normalization of `rho`; `r >= \|d_q\| > 0` off the sample, so it only acts on a point exactly on a cell centroid |
| heads, value / key channels | 8, 8 / 4 | capacity | as the truncated read |
| basis MLP width, basis size | 64, 16 | capacity | as the truncated read |
| key scale | 1/sqrt(4) | normalization convention | absorbed by the learned q, k maps |
| `kernel_read_blocks` | 12 (every block) | capacity / placement | matched depth, as the slices (see Cost) |
| `GLOBAL_READ_TILE_PAIRS`, Triton tile sizes | 2^24; 64, 64, 32, 64 | implementation | memory and speed only; every query row is computed alone, so tiling changes roundoff only (tested) |

No radius, falloff, neighbour count, or any length fitted to the data.

## Invariants (how each is met)

| requirement | how |
|---|---|
| query independence | the read uses the boundary-token states (which in read-only mode never see a point) and the point's own state; points keep log-weight -inf in every slice softmax; every query row's softmax is over the boundary cells only |
| covariance | lengths in units of L or of `\|d_q\|`, dot products of unit vectors; values are invariant states; the equivariant heads assemble the outputs |
| similarity gauge | `r/L`, `log(r/L)`, `log(r/\|d_q\|)` and the measure fractions are unchanged by `x -> k x + t`, areas x `k^2` |
| discretization consistency | the softmax weights are `omega_j exp(.)`: a quadrature of a boundary integral; a cell split into coincident halves gives two terms of half the weight each, exactly the same read |
| measure scale | only measure fractions enter |

## Tests (`test/experimental/models/isla/test_isla_global_read.py`, float64 unless noted)

22 tests; the 202 ISLA tests of the base pass unchanged (224 in all, CUDA included).
With the option off a model owns no new parameter and every golden passes.

- exact query independence: alone, prefix, permuted, chunked, and any tiling of
  the pairs (1e-12);
- rotation and reflection covariance with translation, 3-D and 2-D (1e-10);
- similarity gauge (x -> 2.7 x + shift, areas x 2.7^2, the sdf input x 2.7);
  boundary refinement (every cell as two coincident half-measure copies: the pair
  count doubles, outputs equal to 1e-10); measure scale (x 137);
- zero-initialized output: the model equals the plain read-only model; with
  random read weights every point changes, the far ones included (no radius);
- the invariants: `r/L`, `log(r/L)`, `log(r/|d_q|)` (zero at the nearest cell,
  positive elsewhere), scale-free under a similarity, finite for a point on a
  centroid;
- the tiled softmax read against the dense reference (gradcheck and every
  gradient, two tilings); the tiled, checkpointed basis against one tile;
- every parameter receives a gradient (every block, last block only; DDP);
- bf16 autocast finite (values and gradients), points on centroids included; a
  rank-1 boundary input enters the invariants covariantly;
- CUDA: the fused basis and the fused read against the float64 references
  (float32 to 1e-5, bf16 to bf16 roundoff, ragged tiles, forward and every
  gradient); the whole model fused against eager in float32 (outputs 1e-4, the
  gradient 1e-4); CUDA against CPU in float64; the protocol's training path
  (fused geometry kernel, compiled blocks, bf16) finite, matching eager to bf16
  roundoff and query-independent.

## Cost (one GB300, protocol request: 100,000 points, about 1,000 cells; bf16, compiled blocks)

`bench_step.py` (the kernel lane's, real held-out cases, forward + backward + AdamW,
median of steps 4-11; AGA `stats/bench_code_g*.jsonl`) and a kernel profile on a
synthetic case of the same size (`stats/prof_global.py`, `stats/model_bench.py`).

| configuration | pairs per block | step (ms) | peak (GiB) |
|---|---|---|---|
| read-only (P) | 0 | 238 | 23.1 |
| interacting (I) | 0 | 238 | 23.1 |
| truncated read, sum, R = 0.01 L, last 4 (T) | 2.4-2.8M | 258 | 25.6 |
| global, eager tiles (`code_g1`), every block | 96-101M | 1390 | 35.6 |
| global, eager tiles, last 4 | 96-101M | 731 | 33.5 |
| global, fused (`code_g2`), every block (**G**) | 96-101M | 508 | 34.6 |
| global, fused, last 8 | 96-101M | 445 | 33.6 |
| global, fused, last 4 | 96-101M | 431 | 32.5 |

Profile of the fused G step (synthetic, 495 ms against 220 ms read-only): the kernel
basis once per step (invariants and MLP fused) 19 ms forward and 65 ms backward;
per block the softmax read 1.3 ms forward and 6.7 ms backward (Triton), plus about
7 ms of cuBLAS projections and gradient accumulation of the basis. The fixed basis
cost dominates, so reading at every block (matched depth) costs only 18% more than
the last 4 blocks: **G reads at every block**. The G step is 2.1x the read-only
step, so a G lane takes about 16-17 h (five 3h59 links) against 7.5-8 h.

## Lanes

LHR task dir `agents/2026-10-05-isla-qi-global` (same path as on AGA),
`kernel/lanes.tsv`, chained 3h59 singleton links (`kernel/lanes.sbatch`, the
kernel lane's launcher with this task dir), protocol: `airfrans_volume_2d` /
`isla_volume_airfrans`, 100,000 cells,
`train_split=full_train val_split=full_test +training.val_interval=10`, lr 5e-4,
500 epochs, bf16, no augmentation, compiled blocks. Model overrides: P and G
`++model.query_independent=true`; T adds `++model.kernel_read=sum
++model.kernel_read_radius=0.01 ++model.kernel_read_blocks=4`; G adds
`++model.kernel_read=mean ++model.kernel_read_radius=null` (every block).

| idx | run | arm | seed | code | first job (LHR) | state |
|---|---|---|---|---|---|---|
| 0 | gl_af_int_seed44 | I | 44 | g1 | 99985 | training |
| 1 | gl_af_int_seed45 | I | 45 | g1 | 99986 | training |
| 2 | gl_af_ro_seed44 | P | 44 | g1 | 99987 | training |
| 3 | gl_af_ro_seed45 | P | 45 | g1 | 99988 | training |
| 4 | gl_af_trunc_seed44 | T | 44 | g1 | 99989 | training |
| 5 | gl_af_trunc_seed45 | T | 45 | g1 | 99990 | training |
| 6 | gl_af_global_seed42 | G | 42 | g2 | 99991 | training |
| 7 | gl_af_global_seed43 | G | 43 | g2 | 99992 | training |
| 8 | gl_af_global_seed44 | G | 44 | g2 | 99993 | training |
| 9 | gl_af_global_seed45 | G | 45 | g2 | 99994 | training |
| - | qi_af_int_seed42/43 | I | 42, 43 | qi2 | QI campaign | done |
| - | qi_af_pas_seed42/43 | P | 42, 43 | qi2 | QI campaign | done |
| - | kq_af_sum4_seed42/43 | T | 42, 43 | kq3 | kernel lane | done |

Snapshots: `code_g1` = b10170371, `code_g2` = c5d7408e9 (fused kernels; the G
lanes). The default (interacting, read-only) paths of `code_g1`/`code_g2` are
identical to `code_qi2` (2082d37f5), the QI references' snapshot: `diff` shows
only added lines in `model.py` (the kernel-read options, off by default) and the
added `kernel_read.py`, `pair_kernels.py`, `global_read_kernels.py`; every golden
of the base passes. The truncated path is identical to `code_kq3` (9c564e99a)
apart from added lines. `recipe_g` is the kernel lane's `recipe_kq` (its `src/*.py`
md5s equal this worktree's).

## Results (reporting protocol; G pending)

`infer.py` exactly as the QI scorer (float32, the 200 held-out airfoils, the
training request, case mean of the per-case relative L2), `eval/qg_reduce.py`.
Seed values 42 / 43 / 44 / 45; mean +- standard error over seeds.

| arm | pressure | velocity | eddy viscosity |
|---|---|---|---|
| I | 0.0339 +- 0.0004 (0.0334 / 0.0331 / 0.0349 / 0.0343) | 0.0167 +- 0.0003 (0.0165 / 0.0163 / 0.0176 / 0.0166) | 0.0304 +- 0.0004 (0.0303 / 0.0292 / 0.0312 / 0.0310) |
| P | 0.0325 +- 0.0015 (0.0357 / 0.0332 / 0.0326 / 0.0283) | 0.0176 +- 0.0004 (0.0179 / 0.0178 / 0.0182 / 0.0164) | 0.0381 +- 0.0028 (0.0450 / 0.0393 / 0.0363 / 0.0316) |
| T | 0.0259 (0.0258 / 0.0270 / 0.0249 / -) | 0.0167 (0.0164 / 0.0163 / 0.0173 / -) | 0.0362 (0.0346 / 0.0391 / 0.0350 / -) |
| G | pending | pending | pending |

Ratios of the seed means (paired seed ratios in parentheses): P / I 0.958
(1.070 / 1.004 / 0.936 / 0.826), 1.050 (1.083 / 1.092 / 1.037 / 0.990), 1.251
(1.486 / 1.346 / 1.165 / 1.018); T / I (three seeds) 0.764, 0.995, 1.190. The
read-only model's eddy-viscosity excess is smaller over four seeds (1.25) than
over the QI campaign's two (1.42): seed 45 of the read-only model is within 2% of
the interacting model.

bf16 scoring (like with like; `eval_prec/bfloat16`): I 0.0343 / 0.0172 / 0.0341,
P 0.0317 / 0.0173 / 0.0381, T (three seeds) 0.0275 / 0.0163 / 0.0397.

## Training-log progress (no verdict rests on it)

The trainer's validation relative L2 every 10 epochs (the 200 test airfoils,
chunk-shaped requests, bf16), ratios at matched epoch and seed, pressure /
velocity / eddy viscosity; seeds 42, 43 against the QI and kernel lanes' logs,
44, 45 against this lane's. Single-epoch values are noisy (10-30%).

| epoch | G / I, seeds 42 / 43 / 44 / 45 (nut) | G / P (nut) | T / I (nut) |
|---|---|---|---|
| 30 | 0.86 / 0.61 / 0.87 / 0.90 | 0.51 / 0.56 / 0.68 / 0.89 | 1.57 / 1.06 / 1.30 / 1.01 |
| 50 | 0.71 / 1.02 / 0.87 / 0.77 | 0.60 / 0.67 / 0.75 / 0.45 | 0.86 / 1.31 / 0.94 / 1.18 |
| 60 | 0.91 / 0.78 / 0.73 / 0.87 | 0.62 / 0.54 / 0.63 / 0.57 | 1.23 / 1.17 / 1.00 / 1.12 |
| 80 | 0.93 / 1.07 / 0.52 / 1.23 | 0.83 / 1.08 / 0.49 / 0.96 | 1.29 / 0.93 / 1.14 / 1.56 |
| 90 | 1.12 / 0.90 / 0.65 / 1.26 | 0.78 / 0.61 / 0.61 / 0.88 | 1.14 / 1.47 / 1.05 / 1.32 |
| 100 | 0.73 / 0.78 / 0.91 / 0.77 | 0.66 / 0.59 / 0.74 / 0.63 | 1.14 / 1.56 / 0.97 / 1.42 |
| 150 | 0.98 / 0.86 / 0.79 / 0.77 | 0.68 / 0.59 / 0.75 / 0.72 | - |
| 160 | 0.82 / 0.99 / 0.80 / 0.76 | 0.66 / 0.78 / 0.77 / 0.74 | 1.06 / 1.38 / 1.11 / 1.20 |

G / I pressure at epoch 160: 0.59 / 0.76 / 0.76 / 0.75; velocity 0.77 / 0.81 /
0.77 / 0.81 (epoch 100 was the first learning-rate step).

## Contracts on a trained G checkpoint (early; `qi_probe.py`, LHR job 100135)

`gl_af_global_seed42` at epoch 5 (scoring smoke, `eval_mid/`), first held-out
airfoil in float64 with the exact search; max \|difference\| / RMS target,
pressure / velocity / eddy viscosity: 2,000 sentinels alone against the native
request 1.7e-14 / 6.7e-15 / 9.7e-14; the native request in 5 chunks 7.6e-15 /
4.8e-15 / 4.4e-14: exactly query-independent. Posed residual (random rotation and
translation, float64) 2.1e-3 / 1.5e-3 / 1.0e-3 relative L2, against the read-only
twin's 4.9e-3 / 1.1e-3 at its epoch 5 (the class's on-wall sign effect, QI
amendment 1; it shrinks with training); float32 against float64 9.4e-4 / 6.0e-4 /
4.7e-4. The scorer (`infer.py`) wrote all 200 per-case rows (its 2-D summary
check fails as known; the reducer averages the rows).

## Preregistered bars (written 2026-10-05 17:30 PDT, before any full lane)

AirfRANS only. Arms: **G** read-only + global read (seeds 42-45); **I** interacting-256
(42, 43 from the QI campaign, 44, 45 new); **P** read-only-256 (same seeds split);
**T** truncated read, `"sum"`, R = 0.01 L, last 4 blocks (42, 43 from the kernel
lane, 44, 45 new). Scores: `infer.py` exactly as the QI scorer (float32, the 200
held-out airfoils, the training request, per-case relative L2 averaged over cases),
four-seed means; standard error over seeds; paired ratios per seed.

- **Closes the gap:** all of
  - G / I <= 1.03 on pressure and on velocity;
  - G / I <= 1.05 on eddy viscosity;
  - in the wake band (\|sdf\| > 1 L, probe request), G's eddy-viscosity error
    within 10% of I's (G / I <= 1.10 there).
- **Helps the wake:** G's eddy-viscosity excess over I (ratio - 1) is at most half
  of T's, and G's wake-band eddy-viscosity error is below T's.
- **No wake effect:** G is within one standard error of T on eddy viscosity
  (four-seed means, the standard errors of the two means combined in quadrature).
- Otherwise: between; reported by field.

Also reported: G / P, G / T, the band anatomy (error by \|sdf\|/L band, probe
request), cost per step and peak memory, and every arm scored in bf16 beside the
float32 verdict (like with like).
