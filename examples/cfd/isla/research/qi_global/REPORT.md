# QI-GLOBAL: an untruncated, exact boundary read for read-only ISLA

Lane `isla-qi-global`, branch `psharpe/isla-qi-global` (base
`psharpe/isla-qi-kernel` @ bb877be65). AGA task dir:
`/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-10-05-isla-qi-global/`
(job prefix `isla.global.`). Brief: `BRIEF.md` (this folder).

## Status (2026-10-05 18:00 PDT): references launched, G cost probe pending

- Global read implemented and tested (commits b10170371, c5d7408e9; 224 ISLA tests
  pass). Fused Triton path for the lanes (`code_g2`).
- Reference lanes I, P, T (seeds 44, 45; `code_g1`, default paths identical to the
  QI and kernel snapshots) submitted 17:35 (jobs 1061607-1061612), pending.
- G lanes (seeds 42-45) wait for the GB300 cost probe of the fused path (job
  1062212); the eager path cost 1390 ms per step at every block (5.8x read-only).

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
