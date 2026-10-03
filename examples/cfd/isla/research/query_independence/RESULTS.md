# QI results: query independence in ISLA (2026-10-03)

Preregistration and amendments: `PREREG.md`. Artifacts:
`examples/cfd/isla/results/qi_2026-10-03/` (scorer rows, probe and support
JSONs, `qi_reduction_2026-10-03.json`) and
`examples/cfd/isla/results/qi_support_v18_2026-10-02/`. Cluster task
directory: AGA `agents/2026-10-02-isla-independent`.

## Answer

**Query independence at equal accuracy is reached on DrivAerML Volume, with no
retraining, by giving the interacting model a fixed support.** A seeded sample
of interior points of the training law and count is passed as
`support_points`; it writes into the slices, and every requested point only
reads. Accuracy is 0.996 to 1.005 of the interacting protocol on every field,
both seeds, under random and contiguous splits. Predictions are independent of
the request to 1e-13 in float64, and nothing geometric changes.

**Training without interior writers does not reach it.** That is the
SMART-style passive decode at matched depth (`query_independent=True`).
It is exactly independent, but 4% to 42% less accurate on both datasets. The
preregistered falsification bar (1.10) is crossed on DrivAerML pressure,
velocity and eddy viscosity, and on AirfRANS eddy viscosity.

**AirfRANS gives no clean verdict yet.** Its stored point order is spatially
coherent, so every training and evaluation request is a spatial chunk. A random
support matches the interacting model on random requests (0.998 to 1.007, the
preregistered arm C), but on the full field it is 1.4x to 1.8x worse than the
interacting model on chunk-shaped requests. The trained passive model is 1.05x
to 1.59x worse. Retraining AirfRANS with random requests would separate the
data artifact from the architecture.

## Arm A: passive prediction points at matched depth (`query_independent=True`)

Reporting protocol, float32, held-out split, two seeds. Ratios are passive ÷
interacting (same code snapshot), with the paired seed ratios in parentheses.

| dataset | pressure | velocity | eddy viscosity |
|---|---|---|---|
| DrivAerML Volume, interacting | 0.0404 | 0.0611 | 0.0868 |
| DrivAerML Volume, passive | 0.0448 | 0.0748 | 0.1020 |
| ratio | **1.108** (1.086 / 1.130) | **1.225** (1.199 / 1.252) | **1.174** (1.179 / 1.169) |
| AirfRANS, interacting | 0.0332 | 0.0164 | 0.0297 |
| AirfRANS, passive | 0.0345 | 0.0179 | 0.0422 |
| ratio | 1.037 (1.070 / 1.004) | 1.088 (1.083 / 1.092) | **1.417** (1.486 / 1.346) |

Verdict: **falsified** (>= 1.10 on four fields). Prediction was "between".
Matched depth recovers much of the earlier final-state passive decoder's
loss (PASSIVE-12 1.63 / 1.51 / 2.07 at the exploration budget) but not all
of it. On DrivAerML the excess sits near the wall (|sdf| < 0.01 L, 72% of
points: pressure 1.12, velocity 1.25) and in the far field (2% of points:
1.20 / 1.42 / 1.30).

## Arm C: an independent support at inference (amendment 1, no retraining)

The interacting checkpoints at epoch 500. Support A is a seeded random half
of a twice-larger interior sample (the training count). Evaluation points B
are the other half. Ratios are support ÷ interacting on B.

| | pressure | velocity | eddy viscosity |
|---|---|---|---|
| DrivAerML, random split | 0.998 (0.998 / 0.999) | 0.999 (0.999 / 1.000) | 0.996 (0.994 / 0.998) |
| DrivAerML, block split | 1.002 / 1.005 | 1.002 / 1.003 | 0.997 / 1.003 |
| AirfRANS, random split | 0.998 (0.999 / 0.998) | 0.999 (1.000 / 0.999) | 1.007 (1.007 / 1.006) |
| v18 references (exploratory, same class) | 0.997 to 1.000 | 0.998 to 1.000 | 0.992 to 1.006 |
| nothing writing (routing mass removed) | 12x to 22x | 14x to 17x | 12x to 18x |

Verdict: **supported** on both datasets as preregistered. On DrivAerML the
interacting score on B equals the protocol score (0.0413 / 0.0625 / 0.0870
against 0.0413 / 0.0625 / 0.0887, seed 42). There, arm C equals the
protocol itself.

Support count (DrivAerML, ratios against the full 200,000-point support):
100,000 points 1.01 / 1.01 / 1.02 to 1.03; 50,000 points 1.04 to 1.05 / 1.04
to 1.05 / 1.09 to 1.19. AirfRANS: 50,000 points 1.01; 25,000 points 1.01 to
1.05.

## The AirfRANS confound (amendment 3, post hoc)

The AirfRANS interior is stored in a spatially coherent order: adjacent
stored points are 58x closer than random pairs. The first 50,000 points of
one case span x in [0.33, 4.23] and the last 77,000 span [-2.16, 0.32].
DrivAerML's order is shuffled (adjacent equals random-pair distance). The
reader draws one contiguous cyclic block, so every AirfRANS request, in
training and in the protocol, is a spatial chunk of about 56% of the
domain. Consequences:

| AirfRANS, float32, 200 test airfoils, seeds 42 / 43 | pressure | velocity | eddy viscosity |
|---|---|---|---|
| interacting on a contiguous request, its training law (block split) | 0.0334 / 0.0320 | 0.0164 / 0.0161 | 0.0288 / 0.0285 |
| interacting on a random request | 0.0487 / 0.0367 | 0.0181 / 0.0169 | 0.0430 / 0.0454 |
| support: a contiguous block, B elsewhere | 0.171 / 0.157 | 0.072 / 0.085 | 0.229 / 0.221 |
| region-matched: random support ÷ native, on the protocol's block | 1.55 / 1.30 | 1.09 / 1.04 | 1.27 / 1.32 |
| full field, ÷ interacting on block tiles: random requests | 1.78 / 1.42 | 1.13 / 1.07 | 1.64 / 1.80 |
| full field: random support | 1.78 / 1.42 | 1.13 / 1.07 | 1.65 / 1.81 |
| full field: passive checkpoint (arm A) | 1.11 / 1.05 | 1.08 / 1.07 | 1.42 / 1.59 |

The interacting AirfRANS model is best on chunk-shaped requests, because it
was trained only on them. A support must cover the region it serves: a
block support fails for points outside it, and a whole-domain random
support works at the random-request level. The open prediction, untested:
with random requests in training (`SubsampleMesh(n_points=100000)` after
reading every point, or a shuffled store) the support arm equals the
protocol on AirfRANS as it does on DrivAerML.

## What query independence buys, measured on the interacting model

| readout (float32, every held-out case unless noted) | DrivAerML | AirfRANS |
|---|---|---|
| rear region requested alone ÷ inside the native request (p / U / nut), interacting | 13.6 / 13.7 / 10.8 and 13.4 / 13.0 / 12.0 | 3.8 / 2.4 / 4.4 and 3.3 / 2.3 / 5.2 |
| the same, passive or support | 1.000 exactly | 1.000 exactly |
| 2,000 sentinels alone ÷ inside the native request, interacting (4 cases) | 2.9x to 3.5x pressure, 7.4x eddy viscosity | 1.2x to 1.4x pressure |
| near-body uniform-like set U: support ÷ U inside a padded interacting request | 0.73 / 0.86 / 0.82 (seed 42) | 0.93 / 0.93 / 0.79 (seed 42) |
| U requested alone ÷ U inside a padded request, interacting | 17x / 14x / 16x | 2.7x / 2.8x / 4.1x |
| sentinels alone against the native request, float64, max over sentinels / RMS target | passive 2e-14 to 7e-14, support 4e-14 to 3.5e-13, interacting 1.7 to 10 | passive 3e-15 to 4e-14, support 6e-15 to 1.7e-13, interacting 0.03 to 1.3 |
| native request in chunks against one call, float32 max / RMS target | passive 5e-4 to 1.1e-2 (float32 noise); interacting 2 to 52 | passive 5e-6 to 6e-5; interacting 0.8 to 5 |

## Symmetries and invariances

- **Contract tests (float64):** both new modes are exactly query-independent
  (alone, prefix, permuted, chunked) and exactly rotation-, reflection- and
  translation-covariant. They also hold the similarity gauge and are
  invariant to boundary refinement. With the request as its own support, the
  support mode reproduces the interacting prediction. 170 ISLA tests pass,
  CUDA included.
- **Trained checkpoints:** the posed residual, measured in float64 with an
  exact nearest-neighbour search, is of the same order in all three arms:
  pressure 5e-4 to 1.3e-3 relative L2 on DrivAerML and 1e-4 to 9.5e-4 on
  AirfRANS (velocity and eddy viscosity 5e-5 to 3e-4). The diagnostic
  (`posed_diag.py`, one checkpoint per dataset) found every moving point on
  or within 4e-5 of the wall. There the sign of the model's built-in
  distance is decided by rounding, because the direction to the nearest
  sampled centroid is tangent to that centroid's cell. On AirfRANS a pure
  translation is exact to 2e-14. This is a property of the current class,
  not of the query-independent change.
- **The library nearest-neighbour search (cuML on CUDA)** is not
  translation-robust near the wall. Under a translation, 4,435 of 100,000
  AirfRANS points change neighbour, which moves the output by 1e-2 relative
  L2. It also aborted the CUDA context with out-of-range indices on posed
  2-D input. The probes' posed checks use an exact chunked float64 search.

## Other findings

1. The recipe's `infer.py` refuses to write the summary of a 2-D dataset.
   `MetricCalculator.expected_keys` lists a `velocity_z` that a planar field
   never produces, so its range check fails after every per-case row is
   written. The fleet scorer fails the same way on `v18_af_ref`. The reducer
   here averages the per-case rows, which is what the summary is.
2. One AGA node (nvl72d104-T02) could not see `/home/psharpe/coreai_modulus_cae`.
   `recipe_qi` now addresses DrivAerML through `/scratch/fsw`.

## Cost and next steps

The support mode adds the training count of interior tokens to every forward
pass: 200,000 on DrivAerML, 1.5x the protocol's tokens at the training
request. A chunked request currently re-runs the boundary and support per
chunk. An encode-once API that caches each block's prototypes, anchors and
slice states would make the per-chunk cost the queries alone.

Next experiments, in order of value:

1. AirfRANS with random training requests, both arms, two seeds (about 8 h on
   4 nodes), then the support readouts.
2. A support law drawn from the geometry alone (no CFD volume mesh), trained
   with, for deployment on new shapes.
3. The class fixes: a continuous signed wall-distance channel on the wall,
   and a translation-robust exact k=1 search.
