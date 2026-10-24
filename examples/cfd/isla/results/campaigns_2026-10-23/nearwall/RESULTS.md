# NEARWALL phase A: exposure, response and precision of ISLA's near-wall channel (probes only)

Lane NEARWALL of ISLA (research-lane, launched 2026-10-24 about 07:40 EDT by the supervisor 01a13c2c); written about
08:30 EDT 2026-10-24. Probes only: no training, nothing scored. Notation: s = d / L (d the signed wall distance, L =
ISLA._gauge of the boundaries given), t = |s| / 1e-6, so asinh(s / 1e-6) is linear for t < 1 and logarithmic above.
"Body" = the vehicle or body alone (AirfRANS airfoil, DrivAerML vehicle, AeroZoo no_slip); "whole" = every boundary
the 2b15490b2 config reads. Jobs and snapshots: `JOBS.md`. Detail tables: `exposure_tables.md`, `response_tables.md`,
`precision_tables.md`; raw JSON in `out/`.

## A1. Exposure (`harness/nwprobe.py --mode exposure`, code and recipe 2b15490b2, the recipe's drawn points, val side)

| dataset (cases, points) | L body | L whole | first off-wall t, body / whole (median over cases) | off-wall linear (0 < t < 1), body / whole | t < 10 (incl. wall nodes), body / whole | on-wall nodes (t = 0) |
|---|---|---|---|---|---|---|
| AirfRANS + far field (40, 4.0 M) | 0.411-0.434 chords | 305.3 | 3.77 / **0.0052** | 0 / **19.5 %** | 1.4 % / 37.5 % | 0.56 % |
| DrivAerML Volume, five pieces (6, 1.2 M) | 2.09-2.17 m | 59.50 m | 33.4 / **0.80** | 0 / **0.57 %** | 5.5 % / 15.0 % | 6.5 % |
| AeroZoo, standing config (30, 6.0 M) | 0.011-31.8 m | 0.068-73.2 m (ratio median 1.31) | 69.6 / 27.9 (5 of 30 cases below 1) | 0.00 % / 0.01 % | 0.07 % / 0.13 % | 0 |
| lug (40, 0.38 M) | - | 0.052-0.069 m | - / 2.6e4 | - / 0 | - / 33.4 % (wall only) | 33.4 % |
| Tensile2d (40, 0.38 M) | - | 1.42-1.60 | - / 7.0e3 | - / 0 | - / 4.9 % (wall only) | 4.9 % |

Checks: max |s L - d| / L below 3.7e-8 (s as the model reads it, hooked on qt.seed, against exact_boundary_geometry);
the asinh column equals soft_log_length(s) exactly; the version printed is code_nearwall in every exposure unit.
DrivAerML wall-type L (vehicle + no_slip, `out/e_drvwall.json`): 7.656 units = 38.3 m; first off-wall t 1.23.

In physical and viscous units (U = ISLA's own U, nu = the viscosity role; declared on AirfRANS and AeroZoo only):
- **AirfRANS + far field.** 1e-6 L = 3.05e-4 chords, U (1e-6 L) / nu = 1e-6 Re_L = 610-1,830. The first nodes sit at U
  d / nu about 5 (1st percentile 9.3 of all points). By decade of U d / nu: [1, 10) 0.5 %, median u / U 0.014; [10, 100)
  4.7 %, u / U 0.08; [100, 1e3) 13.2 %, u / U 0.40, nut / nu 0.6; [1e3, 1e4) 17.4 %, u / U 0.65, nut / nu 14. So the
  linear part holds the viscous sublayer and buffer layer and reaches into the turbulent part of the boundary layer
  (nut / nu about 1 at t about 0.3-1); with u_tau / U about 0.04-0.05 (an estimate, not measured), the softening sits at
  y+ about 25-90. The wake and outer flow (u / U above 0.85) are at t above 10, in the logarithmic part.
- **DrivAerML Volume.** 1e-6 L = 59.5 um; first nodes off the car at 45-60 um (t 0.79-0.82). The case files store nu =
  1.0 (a placeholder) and the config declares no viscosity, so no viscous units are given here.
- **AeroZoo.** U d / nu at the 1st percentile 12.8, median 7.4e4; nu / U spans 4.9e-9 to 1.0e-4 L across cases (Re_L
  1e4 to 2e8). Its few linear-part points are nodes 1e-9 to 1e-8 m from a wall on 5 of 30 cases, there under the body L
  too.
- **Solids.** No viscosity or velocity scale; no boundary layer; every off-wall node is above t = 6e3.

## A2. Does the linear part lose information the model needs? (`--mode response`, trained 0.9.0 checkpoints, each on
its own snapshot, one version per process)

Runs: r090eag_af_isla_seed42/43 (code_cmpaf_fix, epoch 500), r090_vol_isla_seed42 (code_r090, epoch 500),
azc_full_isla_seed42 (code_r090, epoch 250). a = qt.seed(X) (the query input layer), h = embed(a). Measured per decade
of t, float32 eval: |da| and |dh| when d is multiplied by 10, the sensitivity |dh / d ln d| (delta 1e-2, as CAP-AUDIT
probe b), the share of a's components whose change exceeds one bfloat16 ulp of the component. "Counterfactual": the
same weights and points with s recomputed in the whole-boundary L (a different input, not a retrained model).

| run | asinh / s weight column norm | median abs(da) for d x10, log part (t 1-1e5) | abs(dh / d ln d), log part | counterfactual whole L: abs(da) for d x10 at t [1e-3, 1e-2) / [1e-2, 0.1) / [0.1, 1) | the same, abs(dh / d ln d) | bf16-resolved share for d x10, t [1e-3, 1e-2) |
|---|---|---|---|---|---|---|
| AirfRANS s42 | 6.41 / 6.44 | 14.7-15.0 | 5.4-6.8 | 0.39 / 2.75 / 10.5 | 0.032 / 0.23 / 1.65 | 0.92 |
| AirfRANS s43 | 6.46 / 6.66 | 14.8-15.0 | 5.6-6.7 | 0.40 / 2.77 / 10.6 | 0.027 / 0.21 / 1.50 | 0.92 |
| DrivAerML Vol s42 | 7.37 / 6.92 | 17.0-17.2 | 3.6-3.7 | (none below t = 1) [1, 10): 16.9 | [1, 10): 3.24 | - |
| AeroZoo s42 (trained, whole) | 6.19 / 17.0 | 14.3-16.1 | 11-12 | trained scale: 0.18 / 1.6 / 11.2 | 0.022 / 0.20 / 3.0 | 0.79 |

Reading:
- The trained input layers use the log channel: |da| per decade of d is flat (about 15 on AirfRANS, 17 on DrivAerML)
  over five decades, so they code log d linearly; the s column adds only beyond s of about 0.1.
- Two near-wall nodes a factor 10 apart differ by ln 10 = 2.30 in the log channel, but by 9 t in the linear part: 0.047
  at the AirfRANS first node under the whole L. With these weights the input layer separates them by |da| 0.39 against
  14.8 in the log part (38x less), and the embedding's sensitivity per unit ln d falls 200x (0.03 against 6.6).
- Against the input layer's resolution at training precision (bfloat16, |a| about 10-20 near the wall, one ulp of a
  component about 2^-8 of it): a factor 10 in d is still resolved (92 % of components), a factor 2 at the first node
  (|da| about 0.03) is at about one ulp, while in the log part a few percent in d is resolved. So, at fixed weights,
  the first decade and a half of AirfRANS nodes (t below 0.1, 5.8 % of points, u / U below 0.09) lose their
  distance resolution from a few percent to about a factor 2. DrivAerML Volume loses almost nothing (0.6 % of points,
  first nodes at t 0.8, |da| 16.9 against 17.0).
- AeroZoo's trained standing shows the same collapse on its own few linear-part points (|da| 0.18 at t 1e-3 to 1e-2):
  a model trained with them does not learn to resolve them, but they are 1e-4 of its points, so this says little about
  a model where they are 20 %.
- What the probes cannot show: whether a retrained model compensates (larger asinh weights, GELU kinks placed at small
  t, or the other channels: positions, and in ISLA-DTT the tree read's unsoftened log(r / |d_q|)); and whether the lost
  resolution costs field accuracy. In the viscous sublayer u is nearly linear in d, so a linear code is the natural one
  there; the buffer layer is where a log code helps. Only a retrain (AF-WHOLE, FLOOR-DRV) prices it.
- CAP-AUDIT's AirfRANS pressure excess sits 1e-3 to 0.1 chords from the wall: t 3.3-330 under the whole L, the
  logarithmic part. The exposed range does not overlap it.

## A3. Precision: how much float32 rounding moves the channel (`harness/nwprec2.py`, code_nearwall, geometry only)

The case is translated by T in float64 and stored in float32: the input a model would get with the case placed T from
the frame's origin. Two roundings, separated (v2, job 367749, `precision_tables_v2.md`):
- "compute": d from exact_boundary_geometry (reference kernel) in float32 against float64 on that same float32 input.
  This is the program's precision property (float32 against float64 on one input);
- "input": float64 arithmetic on the stored-elsewhere input against the stored native case: what storing the case
  elsewhere in float32 costs, which no encoding or kernel can undo.
abs(dc) = the change of asinh(d / eps). Wall nodes (d = 0 exactly, mesh vertices) stay exactly 0 in every row.
(v1, job 367615, `precision_tables.md`, compared float32 at T with float64 native, so it mixed the two; superseded.)

AirfRANS with its far field (12 cases, 1.19 M off-wall points), p99.9 / max of abs(dc):

| eps | rounding | native | T = L body | T = 0.1 L whole (30 chords) | T = L whole | share abs(dc) > 1 at 0.1 L |
|---|---|---|---|---|---|---|
| (i) 1e-6 L whole | compute | 4e-5 / 1e-4 | 1e-4 / 2e-4 | 0.008 / 0.012 | 0.11 / 0.12 | 0 |
| | input | 0 | 1e-4 / 2e-4 | 0.006 / 0.008 | 0.046 / 0.065 | 0 |
| (iii) 2^-24 L whole | compute | 4e-4 / 0.001 | 0.002 / 0.003 | 0.14 / 0.20 | 1.4 / 1.4 | 0 |
| (iv) 1e-6 L body | compute | 0.002 / 0.009 | 0.010 / 0.028 | 2.5 / 2.9 | 5.1 / 5.2 | 0.60 % |
| | input | 0 | 0.008 / 0.022 | 0.43 / 2.6 | 4.0 / 6.7 | 0.03 % |
| (ii) nu / U | compute | 0.002 / 0.009 | 0.010 / 0.029 | 3.1 / 3.8 | 5.9 / 6.1 | 0.61 % |
| | input | 0 | 0.009 / 0.023 | 0.44 / 3.1 | 4.6 / 8.1 | 0.03 % |
| (x) 1e-9 L whole | compute | 0.002 / 0.009 | 0.010 / 0.029 | 2.8 / 3.2 | 5.4 / 5.5 | 0.61 % |

DrivAerML Volume (3 cases): every eps run ((i), (iii), (iv) vehicle, (x)) at most 0.051 (compute) and 0.036 (input) at
any T. AeroZoo (16 cases): (i) at most 0.12; (iv) at most 0.71; (ii) up to 3.3 already in the data frame (compute, on
1.6e-6 of points, its highest-Re cases), (x) up to 4.9.

Reading:
- In the data frame and at the posed protocols' offset (about L body) every candidate keeps the precision property on
  AirfRANS and DrivAerML (at most 0.03). The differences appear only when the body sits far from the frame's origin.
- Under the whole boundary AirfRANS's first nodes sit at about 5.2e-9 L, 11.5x below 2^-24 L (6e-8 L), and nu / U
  at 5.5e-10 to 1.6e-9 L (2^-24 Re_L = 36-109). A float32 coordinate T away from the origin has an ulp of about
  2^-24 T, so any eps that gives those nodes a log code ((ii), (iv), (x)) is sensitive to it: at T = 0.1 L (30 chords,
  inside the domain) rounding moves the channel by up to 3-4, more than 1 on 0.6 % of the points.
- Most of that is the geometry kernel's own float32 arithmetic (compute: p99.9 2.5-3.1), not the input's storage
  (p99.9 0.43-0.44). The kernel forms d = |q - sum w_i v_i| from absolute coordinates. The design reviewer reports, from
  its own numpy probe (not run on ISLA's code), that subtracting in the closest cell's local frame keeps d to 6e-5
  relative at every T. So the compute part looks fixable in the kernel; the input part is not.
- The current eps (1e-6 L = 16.8 x 2^-24 L) keeps both roundings below 0.12 at every T, close to the docstring's
  "at most about 0.1".

## Cost

GB300 short QoS: 367460 (3:33), 367491 (1:45), 367523 (2:29), 367615 (2:30), 367649 (1:42), one 4-GPU node each, 
0.8 GPU-h; with 367749 (2:49, precision v2), 1.0 GPU-h of the 6 GPU-h cap. No cpu-short job.
