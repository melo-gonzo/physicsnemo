# FOREST-NEARTIE task 2: the options compared (2026-10-27)

## Where the forest's covariance comes from

From the drawn tokens, not from the full-resolution boundary.
- `ISLA._case_inputs` (model.py, 8097ed00f lines 1915-1925) draws the tokens: `sample_boundary_cells`, with the start
  from the CUDA RNG in training.
- `_tree_read_walk` passes them as `boundary_r` (the kept cells' centroids over the length unit, no centring) and
  `log_frac` (their Horvitz-Thompson measures, normalized).
- `BoundaryForest._build` (tree_read.py) then computes three things from those tokens:
  - `canonical_frame(points.double(), measure)`;
  - the origin, their weighted centroid;
  - `eta = frame_rotation_bound(x, w64, u)`, from the same tokens' covariance.
- The full-resolution boundary's canonical frame is computed only inside `sample_boundary_cells`, for the Morton order
  of the draw.
- So the gap g, and with it eta, the frame and the trees, change from draw to draw. Where the full-resolution gap is
  small, the draw's noise sets the token gap.
  - That gap follows a Rice law about the full-resolution value: its spread is 15-40% of that value, and noise
    dominates below about g/E = 100.
  - Example: AF-CONFIRM's case 0, full-resolution g/E 167, token g/E 20-450 over 60 draws.

## The candidates

Each candidate was measured on the same saved inputs as the face forest of 8097ed00f: 1,600 AeroZoo census draws,
3,600 near-tie Monte Carlo draws and 32 DrivAerML draws (variants.py, probe_variants.py, probe_risk.py --variants).
The P check is DTT-POSE-EXP's: a small random-weight model in float32, self-posed with the native draw held
(probe_pose.py, job 430228).

| option | what changes | AeroZoo pairs / 0.14.0, census median (min-max) | / octant, census max | / octant, near-tie MC max | pairs at a forced eta 0.05 / 0.2 / 1 (x eta-0) | P: posed pairs unmatched, lug / AirfRANS / T2D median | change on the native pose, vs 0.14.0 | cost | diff |
|---|---|---|---|---|---|---|---|---|---|
| 0.14.0 face forest (8097ed00f) | - | 1 | 2.23 | 8.40 | 8.7 / 162 / 545 (every pair) | 1.0e-6 / 7.0% / 8.7% | - | - | - |
| octant forest (6f1e527f1, 0.13.0) | reference | 1.05 (0.45-1.52) | 1 | 1 | no eta | 53% / 70% / 95% | - | - | - |
| (a) frame and eta from the full-resolution boundary | the frame's inputs | 1.000 (0.77-3.88) | 4.32 | 4.38 | as 0.14.0 at the case's own eta | 1.1e-6 / 13% / 8.6% | AeroZoo 83-90% of pairs re-grouped (new frame), AirfRANS 4.8% | one covariance over the full-resolution cells | ~60-80 lines (model, tree_read; not written) |
| (b) bounding sphere about each node's centroid (eta only in the faces) | admission test | 1.29 (0.52-1.46) | 1.54 | 1.39 | 1.0 / 1.0 / 0.6-0.9 | 2.8e-7 / 7.0% / 8.0% | lug pairs +28%, DrivAerML +37%, T2D +10%, AirfRANS +1% | +29% pairs on AeroZoo | ~25 lines + new tests of the walk's box semantics |
| (c) the rotation bounded as a common rotation of points and cells | separation diameter | 1.000 (0.45-1.00) | 1.18 | 1.27 | 1.38 / 3.4 / no certificate (sqrt(d) eta >= 1: every pair) | 1.2e-6 / 8.2% / 8.7% | AirfRANS pairs -17%, outputs <= 4.2e-7; lug and T2D 3e-5 of pairs | none | +44 / -22 in tree_read.py |
| **(c') = (c) capped by the frame-free ball (the fix, 981ca5aaf)** | separation diameter | **1.000 (0.45-1.00)** | **1.18** | **1.27** | **1.38 / 3.4 / 2.4 (bounded at any eta, at most 3.65)** | **1.2e-6 / 8.2% / 8.7%** | as (c) | none | as (c) |
| (a) + (c') | both | 0.999 (0.48-1.07) | 1.17 | 1.19 | as (c') at the case's eta | 1.2e-6 / 16% / 8.6% | as (a) | as (a) | (a) + (c') |
| eta 0 in the separation test (not a candidate) | drops the exact-frame guarantee | 1.000 (0.40-1.00) | 1.17 | 1.21 | 1.0 | - | - | none | 1 line |

Notes on the table:
- "P outputs", the posed against native RMS difference over RMS, is identical across every face-forest variant to 3
  digits: lug 2.4e-7, AirfRANS 2.8e-5, T2D 7.9e-7, AeroZoo near-tie test cases 3.0e-3. At random weights the forest
  does not show in the outputs. POSE_TABLE.md has every row.
- On AeroZoo's near-tie cases, the posed pair lists differ for every forest. Median / max unmatched:
  - 0.14.0: 17% / 43%;
  - (c'): 18% / 64%;
  - octant: 20% / 61%.
  - The reason: at a near tie, the computed frame rotates inside the nearly degenerate plane when the case is posed,
    and the Morton trees follow it. 0.14.0's over-wide test read more cells one by one, so its lists matched more
    often, at up to 13 times the pairs.
- "Exact pose invariance" in DTT-POSE-EXP's sense holds only where the frame is well conditioned. On those cases, the
  lug and Tensile2d posed lists of (c') equal 0.14.0's to 1e-6 and 0.1% respectively.

## Answers

- **(a) does not remove near ties; it makes them per-case.** The smallest full-resolution gap/E is 30.1 (an
  opposed-hemisphere case) on the census.
  - Spherical and hemispherical far fields are nearly isotropic by construction.
  - Under 0.14.0's formula, (a) still reads up to 4.4 times the octant forest's pairs, on every draw of such a case,
    not on one draw in fifty.
  - An exactly symmetric mesh (an exact cube or sphere, g < E) gets eta = 0 under the existing tie rule. It has no
    blow-up and no pose protection, as today.
  - (a) re-groups most AeroZoo pairs, because its frame is new. It brings no measured pair benefit once (c') holds.
  - Its remaining merit is that the frame would be the same in training and in evaluation. Not adopted.
- **(b) is immune to eta but costs pairs everywhere:** +29% on AeroZoo, +37% on DrivAerML, +28% on the lug. A sphere
  about a flat wall patch's centroid opens the patch for points above it at 1.5 times the box test's distance.
- **(c') is the fix.** The certificate is the same as 0.14.0's: an admitted node is separated in the exact canonical
  frame. Only the bound changes:
  - The frame's rotation moves the point and the cells together. So it changes the box's diagonal and distance only in
    proportion to the node's own size: factor f = (1 + sqrt(d) eta (1 + theta)) / (1 - sqrt(d) eta).
  - 0.14.0 instead used the node's distance from the centroid, which is too conservative by R / D.
  - The frame-free ball certificate caps the diameter at (theta + 2) sqrt(d) D / 2, so the widening is bounded and
    continuous at any eta.
  - Measured: never more pairs than 0.14.0 (1,632 of 1,632 saved draws); at most 1.27 times the octant forest's pairs
    on the near-tie Monte Carlo.
