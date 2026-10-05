# QI-KERNEL: a GLOBE-style boundary read for read-only ISLA

Lane `isla-qi-kernel`, branch `psharpe/isla-qi-kernel` (base `psharpe/isla-independent`
@ 84b0a4190). AGA task dir:
`/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-10-03-isla-qi-kernel/`
(job prefix `isla.kernel.`). Brief: `BRIEF.md` (this folder).

## Status (2026-10-04 23:20 PDT)

- Implementation and float64 contract tests done (commit a7be6350c and the next).
- Cost measured on one GB300 at the protocol request; the first version was 6x to 15x
  the read-only step, so the pair sums were moved into Triton segment kernels and the
  kernel MLP was made one shared basis per forward pass (see Cost).
- Next: pilots on DrivAerML Volume (100 epochs, one seed), then full lanes.

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
| v2: AirfRANS mean R=0.02 / sum R=0.02 / mean R=0.05 | 4.3-5.0M / same / 8.4-9.7M | 461 / 347 / 636 | 24-26 |

v3 (shared kernel basis): pending.

## Preregistered bars (from the brief; written before any full lane)

- Equal or better: (read-only + kernel) / interacting-256 <= 1.03 on every field of both
  datasets (seed means), no paired seed above 1.06.
- Falsified: >= 1.10 on any field.
- Between: otherwise; fields above 1.03 named.
- Also reported: against read-only-256, and the DrivAerML band anatomy (error by
  \|sdf\|/L band).

## Lanes

Pending (pilots first).

## Open issues

- The read's cost grows with the number of boundary cells near the points (a
  quadrature at full boundary resolution); on DrivAerML's finely meshed regions that is
  hundreds of cells per point at R = 0.01 L.
