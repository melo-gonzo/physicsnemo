# AF-FIX task 3: where the read and attention mass goes (W-ISLA, W-DTT against their airfoil-only standings)

Written 2026-10-25 about 08:20 EDT. Probe `probes/readmass.py` (md5 a1e6899b), LHR GB300 short job 381714 (9:26, 0.63 GPU-h);
code `$T/code_cmpaf_fix` (0.9.0, the runs' own code), recipes `recipe_afwhole` (W) and `recipe_r090` (B), as AF-WHOLE's
scorer and near-wall rows; float32 eager, eval; the epoch-500 checkpoints of all eight runs (W = afwhole_af_{isla,dtt},
B = r090eag_af_{isla,dtt}, seeds 42 and 43); 16 full_test cases per run, picked evenly from the loader's 200 (the same 16
cases in every run, checked by source path). Full tables: `out/readmass_tables.md`; raw: `out/rm_<run>.json`.
Definitions in the probe's docstring. Tokens are labelled by position (boundary tokens within 10 chords of the airfoil
are airfoil). "Measured" = read from the trained models' forward; "inferred" = my reading.

## 1. Measured: the far field takes the slices and the read

Means over the 12 blocks, the 16 cases and both seeds (per-block rows in `out/readmass_tables.md`):

| | W-ISLA | B-ISLA | W-DTT | B-DTT |
|---|---|---|---|---|
| slice mass on airfoil / far field / points (mean over the 512 slices) | 0.0009 / 0.366 / 0.633 | 0.515 / 0 / 0.485 | **0.0025 / 0.9975** / 0 | 1.0 / 0 / 0 |
| slices with more than half their mass on the airfoil | **0** of 512 | 285 | **0.4** | 512 |
| slice anchors within 2 chords of the airfoil | 80 of 512 | 512 | **3** | 512 |
| points at 0.01-0.1 chord: airfoil share of the slices they read (E_A) | 0.0007 | 0.416 | 0.029 | 1.0 |
| same points: mix-weighted distance to the anchors they read (chords) | 9.4 | 0.40 | **60.4** | 0.39 |
| same points: mix on anchors within 1 chord | 0.32 | 0.97 | 0.002 | 1.00 |
| ISLA-DTT tree read: attention mass on airfoil sources, points 1e-6 to 1e-5 chord off the wall / 0.01-0.1 / 1+ chord | | | **0.12 / 0.10 / 0.05** | 1.0 / 1.0 / 1.0 |
| tree read: mass on sources within 1 chord, points at 0.01-0.1 chord | | | 0.10 | 1.00 |
| tree read: pairs per point | | | 41.5 | 21.6 |

- The airfoil holds 0.141-0.146 % of the boundary measure on these cases (the far field the rest), and the trained models
  do **not** undo it: no W slice in either form gives the airfoil a majority (W-DTT: 0.4 of 512 on average), though the
  routing logits allow it (at least 20 nats: the cosine part spans 2T = 20 and the geometric bias added to it is
  unbounded, against the measure's ln(0.9986 / 0.0014) = 6.6).
- In W-ISLA the prediction points' own routing mass fills the slices (63 %; `qt.logw` 0.054-0.063, so the points' total is
  about e^0.06 of the whole boundary's measure). The airfoil's own tokens are 0.07 % of what a mid-field point reads.
- In W-DTT the points are passive, so the slices are the boundary's: 99.75 % far field, anchors about 60 chords from the
  points. Its tree read gives the airfoil 3-21 % of the attention (per-block means over all points; 2-25 % for
  points 1e-6 to 1e-5 chords off the wall), **even next to the wall**: about 88 % of a near-wall point's read goes to far-field cells and nodes 200 chords away. B-DTT reads its
  airfoil with 84-100 % of the mass within 1 chord.
- The errors on these 16 cases reproduce AF-WHOLE's pattern (pressure W / B at 0.01-0.1 chord, case means of band e:
  ISLA 0.0169 / 0.0123 = 1.37, DTT 0.0207 / 0.0126 = 1.64).

## 2. Does the 99.86 % explain the mid-field pressure error?

- **Mechanism: measured.** The far field's measure share decides the slices, the slice anchors and (for ISLA-DTT) the
  tree read, in both trained forms; the airfoil's information reaches the mid-field points only through their own tokens
  (W-ISLA) or a read that is 79-97 % far field (W-DTT). This is what the hypothesis needs.
- **Across cases: not supported, weakly.** Within each W run, the per-case pressure ratio W / B at 0.01-0.1 chord does
  not fall with the airfoil share of the read (Spearman rho over 16 cases, W and B paired by seed: ISLA -0.29 / 0.18, DTT
  0.24 / 0.25; with the slice mix on anchors within 1 chord: ISLA 0.33 / 0.56, DTT 0.48 / -0.18, mostly the wrong sign
  for the hypothesis; `probes/correlate.py` md5 5cc7e036, `out/correlate.txt`). The share hardly varies across cases (the
  measure share is 0.141-0.146 % on every case), so this test has little power. Inferred.
- **Counterfactual at inference: uninformative.** Multiplying the airfoil tokens' measure by e^3.3 or e^6.6 (equal
  totals) with the trained weights raises the pressure error 2.3-18x in W-ISLA (0.018 -> 0.041 / 0.33) and 23-55x in
  W-DTT (0.020 -> 0.47 / 1.12): the trained weights are fitted to the tiny share. Off-distribution; no causal reading.
- **So:** the mechanism is measured, the causal link to the 0.01-0.1 chord pressure harm is inferred, and only a retrain
  with a rebalanced quadrature can test it (the campaign's arm C, `PREREG_DRAFT.md`). Two further contributors to W-DTT's
  harm are confounded with it here: the octant tie (16-45 % of far-field cells change tree under pose, and the native
  split follows a case-to-case 1e-6 tilt) and the near-wall softening.
- Also measured, for the near-wall question: the B-DTT tree read is 100 % airfoil; with the far field it is mostly far
  field even in the first decade off the wall, so W-DTT's velocity and nut harm there has a second suspect besides the
  softening (the read). Inferred.

## 3. Principled fixes (none tested)

**C: each declared condition its own quadrature, combined with learned shares.** Every log-weight a boundary token
carries becomes `log a_i - log A_k + beta_k`, with A_k the full-resolution measure of the token's condition k (float64,
before the draw, like L) and beta_k one learned scalar per declared condition (init 0, so the conditions start with
equal shares). It enters every place log w enters in the blocks: the slice routing and states, the routing
prototypes, the head frame's anchors, and ISLA-DTT's read through each pair's log m. **The forest is built from the raw
cell measures**, not from C's weights: in 0.9.0 and in DTT-POSE's port `BoundaryForest.build` takes `omega =
exp(log_frac)` and computes the canonical frame, origin and rotation bound from it, so C must pass the raw fractions there
(else the frame, its eigengap and the tie bands would change with C). With rule 1 (one forest per condition) every node
holds one condition, so C shifts a node's log m by the constant beta_k - log A_k and leaves its virtual cell unchanged.
- Why it is principled: the boundary of a truncated domain is drawn at a chosen radius, and a measure-weighted mean over
  it tends to the far field alone as that radius grows (the airfoil's share is 0.14 % at 200 chords and goes to zero).
  With one quadrature per condition, each condition's share no longer goes to zero as the radius grows. What C does not
  fix: the length unit L (the whole boundary's RMS pairwise distance, 305 chords here) still grows with the radius, so
  every relational invariant in units of L (the slice anchors, the read's r and log r) still depends on it; the
  near-wall softening's dependence on L is ii's to remove. It keeps refinement invariance (A_k is the condition's own exact measure),
  similarity (a uniform rescale shifts every log a_i and log A_k alike), pose and renumbering. It is the device the
  points already have (`qt.logw`, a learned fraction of the boundary measure). No constant, no radius;
  application-agnostic (names come from `boundary_schemas`).
- Costs: K parameters (K = number of conditions); a per-token condition index after the draw.
- Weaknesses: the function depends on how the boundary is split into declared conditions (splitting one condition in two
  changes the weights; conditions are declared physics, so I think that is acceptable); a single-cell condition (Tensile2d's
  `pinned`) starts with a full 1/K share of the slices' mass (beta can learn it down).
- What it changes: every dataset with two or more conditions, in both tasks (AirfRANS with its far field, AeroZoo,
  DrivAerML Volume and Surface with their domain pieces, the lug, Tensile2d, HiLift, SHIFT-SUV, DeepJEB++). No dataset is
  bitwise unchanged after training; an old checkpoint loads with beta = 0 but now computes a different function (equal
  shares), so every standing would need a retrain. Adoption would need checks beyond AirfRANS (AeroZoo at least;
  DrivAerML, the lug and the surface tasks for a full adoption).

**Omega: the tree read's reference weight is an unprojected solid angle, not the measure** (ISLA-DTT only). Each pair's log-measure
bias gains a fixed `-(d - 1) log r` (r the point-source distance), so with zero kernel weights the read is the mean
weighted by `a / r^(d-1)`, the angle the source would subtend at the point if it faced it (dimensionless; the true solid
angle needs |cos| between the source normal and the line of sight; the far field then subtends about 2 pi whatever its
radius, the airfoil about pi near its surface). The learned kernel is a linear combination of an MLP's outputs over
invariants that include log r, so it can approximate the term: the function class is approximately unchanged, and mainly
the read's prior moves. It changes ISLA-DTT's read on every interior dataset
(AirfRANS, AeroZoo, DrivAerML Volume, the lug, Tensile2d) and nothing else (the surface task, the slices). It does not fix
the slices, which in W-DTT hold no airfoil information at all; I therefore rank it below C for the measured mechanism.
Fast paths: the term is per pair, so it belongs with the kernel term `ker`, not with the per-source `src_log_m` of
DTT-FAST's walk.

**Not proposed:** a fixed per-condition reweighting (equal shares with no learned beta; it imposes a choice beta would
learn), a larger routing temperature (the bound is not binding: 20 nats available against 6.6 needed), or a
per-dataset weight (a magic constant).
