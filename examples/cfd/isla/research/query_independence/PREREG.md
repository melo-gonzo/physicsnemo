# Preregistration: QI, query independence at matched depth (2026-10-02, before any lane)

## Question

ISLA's interior task admits the prediction points as interacting tokens, so
a prediction depends on which other points are requested with it: on the
DrivAerML Volume reference a fixed sentinel's pressure error was 0.114
alone, 0.080 with 2,000 companions and 0.055 at the training request of
10,000 (`@sec-nb-qmass-verdict`). The book keeps the dependence because
the request-independent decoder it built was 1.5 to 3.9 times less accurate
(`@sec-arch-passive-decoder`; D1, PASSIVE-12 1.63 / 1.51 / 2.07 on pressure
/ velocity / eddy viscosity). Every passive variant it tested read the
*final* slice states only: `research/transfer_program/studies/computational_support/PREREG.md`
("K read blocks that see the final slice states") and the removed
`_ReadBlock`. SMART (Hagnberger and Niepert, arXiv:2601.18707, ablation
K.5) measures that exact choice at 2.4x against queries that read the
encoder state *at every depth*.

Does ISLA with passive prediction points that read every block's slices,
anchors and prototypes at matched depth, and never write, match the
interacting reference while keeping every symmetry and invariance contract?

## Mechanism

`ISLA(interior=True, query_independent=True)` gives the prediction-point
tokens a routing log-weight of -inf (commit 4bfbd1352 on
`psharpe/isla-independent`). They are left out of every softmax over points
(routing prototypes, slice anchors, slice states, head-frame anchors), so
the slices are formed by the boundary alone; the points still pass through
all twelve blocks with the same weights, read the slices back at every
depth, and carry their own relational invariants against the anchors. The
prediction at a point is then a function of that point and the boundary
only. Nothing geometric changes, so the covariance contracts hold as for
the interacting tokens; the unit tests check exact independence (alone,
prefix, permuted, chunked), rotation, translation, the similarity gauge,
boundary refinement, gradients and the CUDA fused-kernel and
compiled-block paths. Parameters: the interacting model's minus the one
routing-mass scalar. Step cost: identical (the passive tokens still run
through every block).

Prior evidence pointing each way: the learned query routing mass of the
current reference is exp(0.11) and exp(0.10) of the boundary measure on
DrivAerML Volume (v18_vol_ref, seeds 42 and 43) and exp(-0.42) on AirfRANS
(v18_af_ref seed 42 at epoch 473), so the queries carry about half of every
slice's weight and the slices are partly volumetric;
the passive arm loses that. SMART's latents are surface-only and its
matched-depth decoder is its best variant.

## Arms (group `qi`, task dir `agents/2026-10-02-isla-independent` on AGA)

Code `code_qi` (worktree commit 4bfbd1352: the adopted class, identical to
code_v18 on every interacting path) with `recipe_qi` (recipe_v8). Both arms
of a pair differ only in `++model.query_independent=true`.

| lane | dataset (config) | model config | request | arm |
|---|---|---|---|---|
| `qi_drvol_int_seed{42,43}` | DrivAerML Volume (`drivaer_ml_volume_reference`) | `isla_volume_sdf` | 200,000 cells, 200,000 interior points | interacting (class default) |
| `qi_drvol_pas_seed{42,43}` | same | same + `query_independent` | same | passive at matched depth |
| `qi_af_int_seed{42,43}` | AirfRANS interior (`airfrans_volume_2d`, `train_split=full_train val_split=full_test`) | `isla_volume_airfrans` | 100,000 interior points, whole airfoil | interacting |
| `qi_af_pas_seed{42,43}` | same | same + `query_independent` | same | passive at matched depth |

Reporting protocol for all eight: reference configuration (384 x 12, 256
slices, sample-conditioned cosine routing, fused geometry kernel,
`compile_blocks`), lr 5e-4, 500 epochs, bf16 training, batch one, no
augmentation, seeds 42 and 43; the stored signed distance and its gradient
as query inputs on both datasets (the CHAR fairness rule and the measured
best practice). 3h59 singleton chain links on `batch` / `normal`, one node
(4 GB300) per lane.

## Readouts

1. **Accuracy (primary).** Float32 inference on the held-out split (48
   DrivAerML validation cars; 200 AirfRANS test airfoils) at the training
   request (200,000 / 100,000), per-case relative L2 of pressure, velocity
   and eddy viscosity in training space, mean over cases then seeds; the
   recipe's `infer.py` as the fleet scorer runs it. Ratio passive ÷
   interacting per field, with the paired per-seed ratios.
2. **Query independence on the trained checkpoints.** Four validation cases
   per dataset and checkpoint, float32: 2,000 sentinel points (a seeded
   random subset of the native request) predicted alone and with 8,000,
   48,000 and all remaining companions of the native request. Readouts: the
   sentinels' relative L2 per request size, and max |prediction - prediction
   at the native request| over sentinels, in units of the RMS target.
3. **A shifted request (SMART's RQ4).** The rear region: interior points
   whose coordinate along the freestream direction lies in the rear quarter
   of the boundary's extent or beyond (s >= s_max - 0.25 (s_max - s_min),
   s = x . U_inf_dir over the boundary cell centroids). Each held-out case's
   rear points requested alone against the same points inside the native
   request; relative L2 on the rear points both ways, every held-out case.
4. **Covariance on the trained checkpoints.** Four held-out cases rotated and
   translated (one random proper rotation and shift per case; boundary,
   prediction points, freestream direction and the stored SDF gradient
   transformed together), float32: relative L2 of (posed prediction mapped
   back - native prediction) over the native prediction, both arms.
5. **Chunked inference (passive arm).** The native request in 20 chunks
   against one call: max |difference| in units of the RMS target.

## Bars, in readout units

- **Equal or better (supported):** passive ÷ interacting <= 1.03 on every
  field of both datasets (seed means), with no paired per-seed ratio above
  1.06. A field is **better** where the seed-mean ratio is <= 0.97 and both
  paired ratios are below 1.
- **Falsified:** passive ÷ interacting >= 1.10 on any field of either
  dataset: the write path carries information that matched-depth reads do
  not recover.
- **Between:** otherwise; the fields above 1.03 are named.
- **Contracts (required for "supported"):** readout 2 and readout 5 at
  float32 roundoff for the passive arm (max |difference| <= 1e-4 of the RMS
  target on every case); readout 4 <= 1e-3 for both arms.
- Readouts 2 and 3 are reported for the interacting arm; they carry no bar
  (they measure the property the passive arm removes).

## Prediction (written before any lane)

Between: within 1.10 on every field, above 1.03 on at least one, most
likely velocity or eddy viscosity on DrivAerML Volume, where the slices of
the interacting arm are half volumetric; pressure within 3% on both
datasets. Readout 3 moves the interacting arm by more than 5% on at least
one field and the passive arm not at all.

## What each outcome changes

- **Supported:** query independence is free; `query_independent=True`
  becomes the candidate interior default (Peter's call), the book's
  request-dependence limit and the passive-decoder ledger row are restated,
  and the deployment section gains chunked, evaluate-anywhere inference.
- **Between:** the excess is located by wall-distance band (the D1 band
  decomposition) and the next mechanism is chosen from it: a
  request-independent interior support (writers drawn by a fixed law from the
  boundary, covariant, with quadrature weights) if the excess is volumetric,
  query-side capacity if it is near the wall.
- **Falsified:** writes carry information reads cannot; the support-token
  route is the next node, with this pair as its reference.

## Amendment 1 (2026-10-02 13:40 PDT, before any QI lane passed epoch 120)

**A third arm, found during the campaign: an independent support at inference.**
`ISLA.forward(support_points=...)` (commit 2082d37f5) lets an *interacting*
model take a support that writes with its learned routing mass while the
prediction points only read. On the two finished v18 references (the same
class, `isla_volume` / `isla_volume_airfrans`, no retraining), six held-out
cases per checkpoint, a support drawn as a seeded random half of a twice
larger interior sample (training law and count) gave support ÷ interacting
0.97 to 1.01 on every field of both datasets, exact independence of the
request (sentinel max difference 1e-5 to 2e-4 of the RMS target, float32
noise), and 11x to 23x worse with nothing writing. That result is
exploratory. Its confirmation is preregistered here on the QI interacting
lanes, which are still training:

- **Arm C:** `qi_drvol_int_seed{42,43}` and `qi_af_int_seed{42,43}` at epoch
  500, every held-out case, read with a twice-larger interior sample
  (DrivAerML: 400,000 points, `drivaer_ml_volume_reference_sdf_400k`;
  AirfRANS: every point, `airfrans_volume_2d_all`). Support A: a seeded
  random half (200,000; 100,000 on AirfRANS); evaluation points B: the
  rest. Readout: per-case relative L2 on B for (i) B as interacting tokens
  at the training request count (padded with A's head where B is smaller)
  and (ii) B as passive readers behind A; also (iii) B with nothing
  writing, and the sensitivity to the support draw (half of A swapped).
  Script `support_probe.py`.
- **Bars for arm C:** support ÷ interacting <= 1.03 on every field of both
  datasets (seed means of the per-case means), no seed above 1.06:
  supported; >= 1.10 on any field: falsified.
- **Prediction:** supported on both datasets, by the v18 result.

**The float32 floor of readouts 2, 4 and 5.** The 1e-4 and 1e-3 float32
thresholds were set before the floor of the interior task was measured.
Smoke runs on epoch 7 to 43 checkpoints measured: native float32 against
float64 differs by 2e-4 to 1.5e-3 relative L2 and up to 0.1 to 0.18 of the
RMS target at single points; the passive arm's sentinel and chunked
differences in float64 are 3e-15 to 3e-13 (structural independence) and in
float32 up to 2.6e-4 of the RMS target (DrivAerML) and 4e-5 (AirfRANS). The
independence contract (readouts 2, 5) is therefore graded in float64 (bar
1e-9 of the RMS target) and the float32 numbers are reported against the
measured float32 floor, not the guessed 1e-4.

**Readout 4 fails for both arms, for a reason unrelated to the arms.** The
posed residual in float64 is 2e-3 (DrivAerML) and 1e-2 to 2.4e-2
(AirfRANS) relative L2 in both arms. Diagnosis (`posed_diag.py`):
(a) on DrivAerML every moving point (about 240 of 200,000) is a sign flip
of the built-in signed distance at a point on or within 4e-5 of the wall,
where the direction to the nearest sampled centroid is tangent to that
centroid's cell (u . n ~ 0) and the sign is decided by rounding;
(b) on AirfRANS 4,435 of 100,000 points change their nearest boundary
sample under a translation with the library kNN (cuML on CUDA), carrying
all of the residual; an exact float64 search has no flips. Both are
properties of the current class's interior geometry, present in the
interacting reference; neither involves the write path. Readout 4 is
reported for both arms against each other (passive ÷ interacting residual)
instead of against the 1e-3 bar, and the two causes go to Peter as
findings about the class.
