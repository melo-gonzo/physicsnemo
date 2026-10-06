# QI-GLOBAL: an untruncated, exact boundary read for read-only ISLA

Lane `isla-qi-global`, branch `psharpe/isla-qi-global` (base
`psharpe/isla-qi-kernel` @ bb877be65). AGA task dir:
`/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-10-05-isla-qi-global/`
(job prefix `isla.global.`). Brief: `BRIEF.md` (this folder).

## Status (2026-10-06 14:00 PDT): complete

- **Verdict: closes the gap** by the preregistered bars (float32 protocol scores,
  four-seed means). G / I = **0.721 / 0.825 / 0.888** (pressure / velocity / eddy
  viscosity; bars <= 1.03 / 1.03 / 1.05); wake band (\|sdf\| > 1 L) eddy viscosity
  G / I = **0.968** (bar <= 1.10). It also "helps the wake": G's eddy-viscosity
  ratio to I is 0.888 against T's 1.178, and its wake-band error is 0.58x T's.
  G is 3.5 standard errors below T on eddy viscosity ("no wake effect" does not
  hold). Read-only ISLA with the global read is more accurate than interacting ISLA
  on every field, exactly query-independent.
- Against read-only P: 0.753 / 0.785 / 0.710; against the truncated read T: 0.956 /
  0.837 / 0.754. Scored in bf16 throughout (like with like): G / I 0.711 / 0.819 /
  0.798, the same story; G does not depend on the arithmetic.
- **Caveats.** (1) One seed of four (44) is above I on eddy viscosity (paired 1.048;
  the others 0.72-0.90). (2) In the far field G is *worse* than I on pressure and
  velocity: the > 1 L band (17% of points) has pooled pressure error 0.113 against
  0.095 and velocity 0.0106 against 0.0097; under volume weighting (the L2 norm over
  the CFD domain, dominated by the far field) G / I is 1.28 / 1.10 / 0.89 (count
  0.70 / 0.81 / 0.87, decade 0.77 / 0.81 / 0.89; see the added readout). The global
  read moves error from the near wall to the far field for pressure. (3) The
  four-seed references move the read-only baseline: P / I is 0.958 / 1.050 / 1.251
  (the QI campaign's two seeds: 1.037 / 1.088 / 1.417).
- Contracts on all four final G checkpoints: float64 sentinels alone and 5 chunks
  3e-15 to 1.2e-14 of the RMS target (exactly query-independent); posed residual
  1.3e-4 to 3.0e-4 relative L2 in float64 (at or below the read-only class's level).
- Cost: G step 508 ms, 2.1x read-only (every block); a G lane took 16.2-16.7 h on
  one 4xGB300 node (five links), the references 7.5-9 h.
- Lanes ran on LHR (same GB300 hardware, venv and AirfRANS store; AGA's queue had
  ~1,260 jobs ahead). Nothing is running. Results:
  `examples/cfd/isla/results/qi_global_2026-10-06/`.

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
| 0 | gl_af_int_seed44 | I | 44 | g1 | 99985 | done |
| 1 | gl_af_int_seed45 | I | 45 | g1 | 99986 | done |
| 2 | gl_af_ro_seed44 | P | 44 | g1 | 99987 | done |
| 3 | gl_af_ro_seed45 | P | 45 | g1 | 99988 | done |
| 4 | gl_af_trunc_seed44 | T | 44 | g1 | 99989 | done |
| 5 | gl_af_trunc_seed45 | T | 45 | g1 | 99990 | done |
| 6 | gl_af_global_seed42 | G | 42 | g2 | 99991 | done |
| 7 | gl_af_global_seed43 | G | 43 | g2 | 99992 | done |
| 8 | gl_af_global_seed44 | G | 44 | g2 | 99993 | done |
| 9 | gl_af_global_seed45 | G | 45 | g2 | 99994 | done |
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

## Results (epoch 500, reporting protocol)

`infer.py` exactly as the QI scorer (float32, the 200 held-out airfoils, the
training request, case mean of the per-case relative L2), `eval/qg_reduce.py`
(`results/qi_global_2026-10-06/qg_reduction.{json,txt}`). Seed values 42 / 43 / 44
/ 45; mean +- standard error over seeds.

| arm | pressure | velocity | eddy viscosity |
|---|---|---|---|
| **G** | **0.0245** +- 0.0017 (0.0212 / 0.0246 / 0.0292 / 0.0228) | **0.0138** +- 0.0004 (0.0128 / 0.0136 / 0.0147 / 0.0141) | **0.0270** +- 0.0023 (0.0217 / 0.0263 / 0.0327 / 0.0274) |
| I | 0.0339 +- 0.0004 (0.0334 / 0.0331 / 0.0349 / 0.0343) | 0.0167 +- 0.0003 (0.0165 / 0.0163 / 0.0176 / 0.0166) | 0.0304 +- 0.0004 (0.0303 / 0.0292 / 0.0312 / 0.0310) |
| P | 0.0325 +- 0.0015 (0.0357 / 0.0332 / 0.0326 / 0.0283) | 0.0176 +- 0.0004 (0.0179 / 0.0178 / 0.0182 / 0.0164) | 0.0381 +- 0.0028 (0.0450 / 0.0393 / 0.0363 / 0.0316) |
| T | 0.0256 +- 0.0006 (0.0258 / 0.0270 / 0.0249 / 0.0246) | 0.0165 +- 0.0003 (0.0164 / 0.0163 / 0.0173 / 0.0160) | 0.0358 +- 0.0011 (0.0346 / 0.0391 / 0.0350 / 0.0347) |

| ratio of seed means (paired seeds) | pressure | velocity | eddy viscosity |
|---|---|---|---|
| **G / I** | **0.721** (0.64 / 0.74 / 0.84 / 0.67) | **0.825** (0.78 / 0.83 / 0.84 / 0.85) | **0.888** (0.72 / 0.90 / 1.05 / 0.88) |
| G / P | 0.753 (0.60 / 0.74 / 0.89 / 0.81) | 0.785 (0.72 / 0.76 / 0.81 / 0.86) | 0.710 (0.48 / 0.67 / 0.90 / 0.87) |
| G / T | 0.956 (0.82 / 0.91 / 1.17 / 0.93) | 0.837 (0.79 / 0.84 / 0.85 / 0.88) | 0.754 (0.63 / 0.67 / 0.94 / 0.79) |
| T / I | 0.754 (0.77 / 0.82 / 0.71 / 0.72) | 0.986 (0.99 / 1.00 / 0.99 / 0.96) | 1.178 (1.14 / 1.34 / 1.12 / 1.12) |
| P / I | 0.958 (1.07 / 1.00 / 0.94 / 0.83) | 1.050 (1.08 / 1.09 / 1.04 / 0.99) | 1.251 (1.49 / 1.35 / 1.17 / 1.02) |

### Bars (float32, as preregistered)

| bar | measured | holds |
|---|---|---|
| closes: G / I <= 1.03 pressure, velocity | 0.721, 0.825 | yes |
| closes: G / I <= 1.05 eddy viscosity | 0.888 | yes |
| closes: wake band (> 1 L) eddy viscosity G / I <= 1.10 | 0.968 | yes |
| helps the wake: (G / I - 1) <= (T / I - 1) / 2 | -0.112 <= 0.089 | yes |
| helps the wake: wake-band eddy viscosity G < T | 0.0633 < 0.1083 (0.58x) | yes |
| no wake effect: \|G - T\| <= one SE (eddy viscosity) | 3.5 SE apart | no |

**Verdict: closes the gap** (all three "closes" bars hold; "helps the wake" holds
too, "no wake effect" does not).

### bf16 scoring (like with like; `eval_prec/bfloat16`, `qg_reduction_bf16.txt`)

| arm | pressure | velocity | eddy viscosity |
|---|---|---|---|
| G | 0.0244 | 0.0141 | 0.0272 |
| I | 0.0343 | 0.0172 | 0.0341 |
| P | 0.0317 | 0.0173 | 0.0381 |
| T | 0.0273 | 0.0163 | 0.0392 |
| G / I | 0.711 (0.67 / 0.77 / 0.73 / 0.67) | 0.819 (0.80 / 0.83 / 0.81 / 0.85) | 0.798 (0.73 / 0.84 / 0.85 / 0.77) |
| G / T | 0.894 | 0.860 | 0.694 |

G scores the same in both arithmetics (bf16 / float32: 0.99 / 1.02 / 1.01); the
interacting reference is 12% worse on eddy viscosity in bf16, so the bf16 ratios
favour G slightly more. No arithmetic dependence of the kind the kernel lane found
on DrivAerML.

### Band anatomy

Probe request, every held-out airfoil; pooled relative L2 by \|stored sdf\|/L,
seed means.

| band (points) | pressure G / I / P / T | velocity G / I / P / T | eddy viscosity G / I / P / T |
|---|---|---|---|
| < 0.01 (41%) | 0.0160 / 0.0253 / 0.0230 / 0.0162 | 0.0195 / 0.0245 / 0.0266 / 0.0236 | 0.0334 / 0.0383 / 0.0420 / 0.0376 |
| 0.01-0.1 (20%) | 0.0158 / 0.0220 / 0.0210 / 0.0209 | 0.0130 / 0.0162 / 0.0165 / 0.0163 | 0.0309 / 0.0368 / 0.0377 / 0.0355 |
| 0.1-1 (23%) | 0.0319 / 0.0363 / 0.0324 / 0.0283 | 0.0121 / 0.0129 / 0.0139 / 0.0134 | 0.0367 / 0.0416 / 0.0395 / 0.0575 |
| > 1 (17%) | **0.1127** / 0.0947 / 0.0840 / 0.0679 | 0.0106 / 0.0097 / 0.0097 / 0.0095 | **0.0633** / 0.0654 / 0.0865 / 0.1083 |

Band swap (`band_swap.py`, G against I, exact counterfactual on the same points):
swapping G's near-wall band (< 0.01 L) for I's error takes pressure from 0.716 to
0.969 and velocity from 0.814 to 0.959: most of G's pressure and velocity gain is
within 0.01 L of the wall, as with the truncated read. Eddy viscosity gains in every
band (0.88-0.93 after any one swap). The wake band (> 1 L), where T is 1.66x I on
eddy viscosity, is at 0.97x I for G: the untruncated read reaches the wake. Against
T, G's eddy-viscosity gain is in 0.1-1 L (0.0367 against 0.0575) and beyond 1 L
(0.0633 against 0.1083). The cost is far-field pressure: beyond 1 L G is 1.19x I
and 1.66x T on pressure (4% of G's squared pressure error, so the count-weighted
score barely sees it).

### Contracts on the final G checkpoints (epoch 500, `qi_probe.py`)

Max \|difference\| / RMS target, pressure / velocity / eddy viscosity, first held-out
airfoil; float64 with the exact search; posed = relative L2 under a random rotation
and translation.

| run | float64: sentinels alone | float64: 5 chunks | float64: posed | float32: sentinels; chunks |
|---|---|---|---|---|
| G s42 | 4e-15 / 3e-15 / 6e-15 | 5e-15 / 5e-15 / 6e-15 | 1.3e-4 / 1.8e-4 / 3.3e-5 | 2e-6 / 2e-6 / 3e-6; 3e-6 / 3e-6 / 4e-6 |
| G s43 | 4e-15 / 3e-15 / 1e-14 | 5e-15 / 3e-15 / 9e-15 | 1.4e-4 / 9.7e-5 / 2.8e-5 | 3e-6 / 2e-6 / 4e-6; 3e-6 / 2e-6 / 6e-6 |
| G s44 | 6e-15 / 3e-15 / 6e-15 | 7e-15 / 4e-15 / 9e-15 | 2.8e-4 / 1.4e-4 / 1.1e-4 | 3e-6 / 2e-6 / 3e-6; 4e-6 / 2e-6 / 5e-6 |
| G s45 | 4e-15 / 3e-15 / 7e-15 | 5e-15 / 4e-15 / 1e-14 | 3.0e-4 / 1.9e-4 / 8.2e-5 | 2e-6 / 2e-6 / 5e-6; 3e-6 / 2e-6 / 7e-6 |
| P s44 | 9e-15 / 3e-15 / 4e-14 | 9e-15 / 4e-15 / 6e-14 | 2.1e-4 / 1.6e-4 / 7.7e-5 | 4e-6 / 2e-6 / 2e-5; 6e-6 / 2e-6 / 2e-5 |
| I s44 (interacting) | 8.5e-2 / 3.9e-2 / 1.1 | 0.36 / 0.15 / 1.1 | 2.7e-4 / 2.3e-4 / 8.1e-5 | 8.5e-2 / 3.9e-2 / 1.1; 0.51 / 0.54 / 4.9 |

G is exactly query-independent (float64 roundoff) and as covariant as the read-only
class; the interacting reference changes its predictions by up to 1.1x the RMS
target when the request changes.

## Added readout: principled error weighting (requested 2026-10-06; not a bar)

Peter asked for every accuracy comparison also under principled error weighting,
not only the protocol's count weighting (every mesh node weighted equally). The
weighted-error lane's probe (`we_probe.py`, `we_reduce.py`, branch
`psharpe/isla-volume-quadrature`, `examples/cfd/isla/research/volume_quadrature/weighted_error/`)
predicts at **every mesh point** of the 200 held-out airfoils and scores the same
predictions three ways: **count** (every point equal), **decade** (every \|sdf\|/L
decade band equal, volume-weighted inside), **volume** (exact dual-area weighting:
the L2 norm over the CFD domain). The chunk-trained interacting model I is read on
its own tiles (law `tiles`, its training law); the read-only arms need no writers.
`qg_we_reduce.py` (this folder) reduces G, I, P, T over the four seeds.

Provenance: seeds 42, 43 of I, P, T reuse that lane's rows (AGA
`agents/2026-10-06-isla-weighted-error/probe/`, the same checkpoints and probe,
copied to LHR `we/probe_refs/`); seeds 44, 45 and G run here on LHR (`we/we.sbatch`,
`we/we_runs.tsv`, the lane's recipe copy as `recipe_we`, its dual-area store
`airfrans_pdmsh_dual` copied to LHR `data/`, file list and sizes identical).
The verdict stays on the protocol scores.

Results (`qg_we_reduce.py`, `results/qi_global_2026-10-06/we_reduction.{json,txt}`;
relative L2 in training space, case means then four-seed means; paired seeds in
brackets):

| arm | count p / U / nut | decade p / U / nut | volume p / U / nut |
|---|---|---|---|
| G | 0.0205 / 0.0138 / 0.0248 | 0.0279 / 0.0130 / 0.0309 | 0.1453 / 0.0104 / 0.0279 |
| I (tiles) | 0.0292 / 0.0171 / 0.0287 | 0.0364 / 0.0161 / 0.0348 | 0.1136 / 0.0095 / 0.0313 |
| P | 0.0277 / 0.0172 / 0.0351 | 0.0363 / 0.0155 / 0.0404 | 0.1219 / 0.0097 / 0.0432 |
| T | 0.0221 / 0.0162 / 0.0351 | 0.0340 / 0.0154 / 0.0399 | 0.0957 / 0.0091 / 0.0440 |

| ratio | protocol (float32) | count | decade | volume |
|---|---|---|---|---|
| G / I | 0.721 / 0.825 / 0.888 | 0.702 / 0.808 / 0.865 | 0.766 / 0.808 / 0.890 | **1.279 / 1.100** / 0.893 |
| G / P | 0.753 / 0.785 / 0.710 | 0.738 / 0.800 / 0.707 | 0.768 / 0.838 / 0.766 | 1.192 / 1.070 / 0.646 |
| G / T | 0.956 / 0.837 / 0.754 | 0.927 / 0.852 / 0.706 | 0.819 / 0.844 / 0.776 | 1.518 / 1.140 / 0.635 |
| T / I | 0.754 / 0.986 / 1.178 | 0.757 / 0.947 / 1.225 | 0.935 / 0.957 / 1.146 | 0.842 / 0.964 / 1.407 |
| P / I | 0.958 / 1.050 / 1.251 | 0.951 / 1.009 / 1.224 | 0.997 / 0.964 / 1.162 | 1.073 / 1.028 / 1.382 |

Paired G / I: count [0.61/0.78/0.81, 0.78/0.81/0.95, 0.75/0.80/0.87,
0.67/0.84/0.83]; decade [0.64/0.76/0.78, 0.82/0.79/0.92, 0.85/0.83/0.93,
0.75/0.85/0.92]; volume [0.91/1.07/0.81, 1.36/0.95/0.92, 1.53/1.17/0.94,
1.28/1.20/0.89]. Per-band volume-weighted RMS, G / I: < 0.001 L 0.62 / 0.77 / 1.10;
0.001-0.01 0.68 / 0.74 / 0.80; 0.01-0.1 0.75 / 0.76 / 0.84; 0.1-1 0.99 / 0.88 /
1.01; > 1 L **1.30 / 1.13** / 0.88.

Reading: on the full mesh the count and decade weightings agree with the protocol
(G better than I on every field, by 11-30%). Under volume weighting, which is the
far-field L2 norm (93% of the AirfRANS volume lies beyond 1 L), G keeps its
eddy-viscosity gain (0.89 against I, 0.64 against T) but is 28% worse than I on
pressure and 10% worse on velocity, and 1.5x T on pressure. The global read trades
far-field pressure for near-wall pressure; the truncated read, which reads nothing
in the far field, does not (T / I volume pressure 0.84).

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
| 200 | 0.70 / 0.76 / 0.73 / 0.76 | 0.64 / 0.64 / 0.73 / 0.75 | |
| 250 | 0.78 / 0.80 / 0.80 / 0.70 | 0.74 / 0.65 / 0.79 / 0.73 | |
| 300 | 0.73 / 0.76 / 0.75 / 0.73 | 0.70 / 0.65 / 0.82 / 0.70 | |

G / I pressure at epoch 300: 0.65 / 0.72 / 0.66 / 0.65; velocity 0.78 / 0.83 /
0.79 / 0.82. From epoch 200 on the four seeds agree to a few percent.

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

## Open issues and next steps

- **Far-field pressure.** G is 1.19x I on pressure beyond 1 L (count-weighted) and
  1.28x under volume weighting, where T is better than I. A softmax over every cell
  is a convex average of the cell values; far from the body it tends to a smooth
  average whose pressure update the output map must cancel. Candidates (not run):
  a learned null (a fraction of the whole boundary measure, as in the truncated
  read) so far points can read nothing; or the `"sum"` form (a signed quadrature,
  which can decay as a power law) beside the mean; either needs new lanes.
- **Seed spread.** One G seed (44) is 1.05x I on eddy viscosity; the four-seed
  standard error of G's eddy viscosity (0.0023) is 5x I's. Two more seeds would
  tighten it (about 17 h per lane on one node).
- **DrivAerML (out of scope, not launched): exact all-pairs is not affordable at
  the protocol size.** 200,000 points x 200,000 cells = 4e10 pairs per block, 400x
  AirfRANS. At the measured rates (kernel basis 85 ms forward + backward per 1e8
  pairs, the softmax read about 12 ms per block per 1e8 pairs) one step would take
  about 34 s (basis) + 12 x 4.8 s (reads), about 90 s on one GB300 against 0.26 s
  for read-only: a 500-epoch lane would take about two months on one node. Even one
  read at the last block is about 40 s per step. Kernel work (fusing the per-block
  projection, a faster basis backward) is worth 2-4x, not 350x. An exact read on
  DrivAerML therefore needs a smaller boundary set for the read: for example a
  Horvitz-Thompson subsample of the cells (a quadrature of the same surface
  integral, refinement-consistent in expectation but not exactly) at about 2,000
  cells (4e8 pairs, about 1 s per step, 4x read-only at every block). That is a
  design decision for Peter.
- **Cost on AirfRANS.** 2.1x the read-only step. The basis backward (65 ms per
  step) and cuBLAS projection glue (about 7 ms per block) are the largest items;
  a fused projection and a tuned basis backward would bring it to about 1.6-1.8x.
- **Branch state.** Peter merged `psharpe/isla` into this branch at 13:41 EDT
  (101fdba6d: SOFT-GEO, the exact nearest-boundary search, new class defaults).
  Every result here comes from the snapshots `code_g1` (b10170371) and `code_g2`
  (c5d7408e9), before the merge. The global-read tests were not re-run on the
  merged branch: at 14:00 PDT the shared local venv failed on `import torch`
  (an `ld.so` assertion), so that check is still to do.
- Re-running the reductions by hand (LHR task dir): `cd eval && python
  qg_reduce.py --scores score ../refs/qi_score ../refs/kq_score --probes probe
  ../refs/qi_probe ../refs/kq_probe --out qg_reduction.json`; bf16 with `--scores
  ../eval_prec/bfloat16/score ../refs/kq_score_bf16`; weighted: `cd we && python
  qg_we_reduce.py --probes probe probe_refs --we-reduce . --out we_reduction.json`.
