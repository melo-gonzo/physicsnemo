# NEARWALL design: ISLA's near-wall distance encoding under the whole boundary

Lane NEARWALL (research-lane, supervisor 01a13c2c). Written 2026-10-24 about 08:25 EDT, revised about 08:37 EDT after
a lane-reviewer pass (`REVIEW_DESIGN.md`: "sound with fixes"; each fix is applied below, and the precision probe was
rerun as v2). Written before any AF-WHOLE or FLOOR-DRV score: I have read neither lane's results, only their PREREG
and JOBS files and the BOUNDARY-AUDIT report. Evidence: `RESULTS.md` (A1 exposure, A2 trained-layer response, A3
float32 precision), from probes only.

## 1. The question and the constraint

The interior query reads `[s, asinh(s / eps)]`, with `s = d / L` and `eps = LENGTH_SOFTENING = 1e-6`. Under the whole
boundary (Peter, 2026-10-23 17:08: canonical, and no accuracy result reverses it) L becomes the size of the domain. A
harm is fixed in the model, never by dropping boundary. The encoding must keep:
- pose;
- units and scale (the channel dimensionless);
- precision (float32 against float64 on the same input, bounded);
- renumbering;
- discretization (re-meshing the boundary changes it only by quadrature).

It must be a function of the boundary meshes and the declared global inputs (Peter's boundary-only input ruling of
2026-10-17). It may bring no magic constant, derived radius or dataset-frame shortcut. The result is one canonical
encoding, with no flag.

## 2. What phase A found

**Exposure is large on one dataset only.**
- AirfRANS with its far field: L goes from 0.42 to 305.3 chords. The first nodes sit at t = |s| / 1e-6 = 0.0052
  (5.2e-9 L), and 19.5 % of drawn points are off-wall in the linear part (t < 1), against none for the standing.
- In viscous units the softening sits at U d / nu = 1e-6 Re_L = 610-1,830. With u_tau / U of about 0.04-0.05
  (estimated, not measured), that is y+ of about 25-90. So the linear part covers the viscous sublayer, the buffer
  layer and the start of the log layer. It reaches the turbulent part of the boundary layer (nut / nu about 1 at t of
  0.3-1). u is nearly linear in d only in the sublayer (t below about 0.1, 5.8 % of points).
- DrivAerML Volume with its five pieces: 0.57 % of points, first nodes at t = 0.80.
- AeroZoo (already whole): 0.01 %, from nodes 1e-9 to 1e-8 m off a wall, which are linear under the body's L too.
- Lug and Tensile2d: none (first off-wall t of 2.6e4 and 7e3).

**The trained layers code log d.**
- On both AirfRANS seeds and on DrivAerML the input layer's change per decade of d is flat over five decades (|da| of
  about 15 and 17).
- With the same weights and the whole-boundary L, a factor 10 in d at the AirfRANS first node changes |da| by only 0.39
  (38x less), and the embedding's sensitivity per unit ln d falls 200x.
- At bfloat16 that node's distance is resolved to about a factor 2, against a few percent in the log part.
- A retrain may compensate; only AF-WHOLE's retrain prices it.

**Precision decides between candidates only away from the frame's origin.**
- float32 resolves a coordinate T from the origin to about 2^-24 T. AirfRANS's first nodes sit 11.5x below 2^-24 L,
  and nu / U sits 40-100x below.
- In the data frame and at the posed protocols' offset (about L body), every candidate keeps the precision property
  (at most 0.03, A3).
- At T = 0.1 L (30 chords, still inside the domain), any eps that gives those nodes a log code moves the channel by up
  to 3-4 (more than 1 on 0.6 % of points). Most of that is the geometry kernel's float32 arithmetic (p99.9 2.5-3.1),
  which the reviewer's numpy probe suggests a local-frame subtraction would remove. Less of it is the input's own
  storage rounding (p99.9 0.43-0.44), which nothing removes.
- The current eps stays below 0.12 for both roundings at every T.

## 3. Candidates

Each candidate is judged on the invariances P (pose), U (units and scale), F (precision), R (renumbering) and D
(discretization); on boundary-only input (B); and on forbidden constructs (C). Then its effect with the phase-A
numbers, its change to the outputs, and its cost. (v) and (vi) were not run; their F is inferred from the measured
small-eps arms.

**(i) Current: eps = 1e-6 L.**
- P, U, R and D are exact.
- F: rounding moves the channel by at most 0.12 (compute) and 0.065 (input) wherever the case sits in its domain.
- B: yes.
- C: 1e-6 = 16.8 x 2^-24 is a precision margin over float32, not a model length; the docstring states it as such.
- Body alone / whole boundary:
  - AirfRANS first nodes at t 3.8 / 0.0052 (19.5 % linear);
  - DrivAerML at 33 / 0.80 (0.57 %);
  - AeroZoo (whole) at 27.9 (0.01 %);
  - lug and Tensile2d: no off-wall node in the linear part.
- Outputs: unchanged. Cost: none.
- Its docstring's second justification, "below the finest wall spacing of the reference datasets ... about 5e-6 of the
  frame's length", is false under the whole boundary.

**(ii) Viscous softening: asinh(d U / nu), in which L cancels.**
- P, R and D are exact.
- U: dimensionless, a local Reynolds number. It is not invariant to a geometric rescale at fixed U and nu, and neither
  is the physics; ISLA's Reynolds input z already reads the same U and nu.
- B: yes, through the declared globals. `_case_scales` takes U from the velocity_scale source and nu from the
  viscosity role, and both exist only where `reynolds` is declared, since ISLA requires the two together.
- Its own merit: the near-wall code does not depend on how far out the domain's outer boundary is drawn, which is a
  choice of truncation of the computational domain.
- F: passes in the data frame and at the posed offset (at most 0.03 on AirfRANS). It fails at T of 0.1 L or more (up to
  3.8 from the kernel, 3.1 from the input). On AeroZoo it reaches 3.3 already in the data frame, on 1.6e-6 of points in
  its highest-Re cases.
- Where it is declared:
  - AirfRANS: nu = 1.56e-5 m^2/s is a constant that the model config sets with SetGlobalField; the case files do not
    store it.
  - AeroZoo: physics.kinematic_viscosity, stored per case.
- Where it is not declared:
  - DrivAerML: its files store nu = 1.0, a placeholder. Supplying air's value is Peter's decision, not taken here.
  - Lug and Tensile2d: elasticity has no viscosity or velocity.
- Where it is undefined, the rule must fall back to (i). That makes two codes, chosen by what the model declares: not a
  user flag, but two encodings.
- Effect:
  - AirfRANS: first nodes at U d / nu of about 5 (asinh 2.3) whatever the far field, as in the body-only standing.
  - AeroZoo: eps moves from 1e-6 L to between 5e-9 and 1e-4 L across cases, so every AeroZoo case's channel changes.
  - DrivAerML and the solids: unchanged (the fallback).
- Outputs: the AirfRANS and AeroZoo models must be retrained. The input width is the same, so checkpoint shapes are
  unchanged but their meaning is not. Cost: none at run time.

**(iii) Arithmetic softening: the float32 resolution at the query.**
- eps_q = 2^-24 |x_q| depends on the frame's origin, so P fails exactly, not only in rounding.
- The frame-free version, eps = 2^-24 L fixed to the float32 bound for any dtype, is (i) with a margin of 1 instead of
  16.8.
- AirfRANS's first nodes then sit at t' = 0.087, still linear, and rounding moves the channel by up to 1.4 at T = L.
  It does not fix the exposure, and it loosens F.

**(iv) A length from the wall-type boundaries only.**
- P, U, R and D are exact.
- B: ISLA would need to know which boundaries are walls. The schemas do not say: every DrivAerML piece has an empty
  schema (inlet, outlet and slip as well as vehicle and no_slip), as does AirfRANS's freestream. So it needs a new
  per-boundary declaration; inferring walls from names would be a dataset shortcut.
- C: a second length unit for one channel, still multiplied by 1e-6.
- F: as (ii) on AirfRANS (up to 2.9 compute and 2.6 input at T = 0.1 L).
- Effect:
  - AirfRANS: L(airfoil) = 0.42 restores the standing's exposure.
  - DrivAerML: the ground is a no-slip wall, and L(vehicle, no_slip) = 38.3 m against 59.5 m, so the first nodes move
    only from t 0.80 to 1.23. Restoring the car's 2.1 m would mean declaring the ground not a wall.
  - AeroZoo: L(no_slip) is 1.31x below L whole.
  - Solids: no wall, so undefined.
- Outputs: retrain AirfRANS, AeroZoo and DrivAerML. Cost: one more gauge per forward.

**(v) Several log channels at fixed decades, asinh(s / 10^-k).**
- Each behaves as (i) with its own constant.
- The set resolves log d down to the smallest eps, so it has that eps's precision: F as the measured 1e-9 L arm, up to
  3.2 at T = 0.1 L.
- It adds a grid of decade constants and +K input width (checkpoints incompatible), and nothing that one channel at
  the smallest eps does not, beyond extra kinks for the first layer.

**(vi) No softening, log|s|, with the on-wall nodes coded separately.**
- On-wall nodes are exact zeros here (mesh vertices; A3 shows no change in any frame), so a separate code for them is
  well defined.
- But log|s| is unbounded as d goes to 0+. The code is discontinuous between the wall and its first nodes, its slope is
  unbounded, and float32 rounding at the first nodes moves it at least as much as the smallest-eps arms.
- It fails F and the continuity the channel was designed for.

**(vii) Floored viscous softening: eps = max(nu / U, 16.8 x 2^-24 L).**
- It keeps F frame-free and uses the physics where float32 can resolve it.
- On AirfRANS with its far field, and on most of AeroZoo, it equals (i), since nu / U lies below the floor; it changes
  only low-Re cases. It shows the tension: an encoding that is precise in every frame cannot give those first nodes a
  log code.

**(viii) The gauge of the nearest condition: asinh(d / (1e-6 L_k)), with L_k the gauge of the named boundary that holds
the closest point** (raised by the reviewer).
- P, U and R hold, and B holds without any wall declaration. It restores AirfRANS to t of about 3.8 and DrivAerML's car
  nodes to about 33.
- D fails: a condition can be a single cell, as Tensile2d's `pinned` is ("the left side's cell touching the pinned
  bottom-left node", isla_tensile2d.yaml). Its L_k is then that cell's size, which re-meshing changes.
- The channel also jumps by ln(L_j / L_k) where two conditions are equidistant (ln 725 = 6.6 between AirfRANS's airfoil
  and far field, about 150 chords out), and at every seam between conditions on one surface (lug, Tensile2d,
  DrivAerML's ground and walls). Rejected.

## 4. Recommendation

**Keep (i), the current encoding, as canonical, and correct its docstring.** The grounds are narrower than in the first
draft: precision alone does not separate the candidates in the frames that are scored.
1. (i) is the one candidate defined identically on every dataset, with no fallback, no new declaration and no recipe
   constant. (ii) is undefined on three of the five datasets. On AirfRANS it rests on a nu that the recipe sets, and on
   AeroZoo it changes every case. (iv) needs a wall declaration and barely moves DrivAerML. (viii) breaks
   discretization invariance.
2. (i) is robust to coordinate rounding wherever the body sits in its domain, at most 0.12 against 3-4 for the
   alternatives at 30 chords. Those alternatives give a log code to nodes 11.5x below float32's resolution at the
   domain's scale, which they keep only while the body sits near the frame's origin. That is a weaker objection than
   a failure of F as defined, since in the data frame and at the posed offset they pass.
3. The exposure is material on AirfRANS with its far field only (19.5 % of points), and there the trained-layer
   evidence shows lost resolution at fixed weights, not a lost accuracy. Changing the canonical encoding before a
   retrain shows a harm would retrain AeroZoo and AirfRANS on a probe argument.

The strongest case against this recommendation is (ii)'s own merit, an L-free near-wall code for flows. It is the
candidate to test if a harm appears.

The docstring change (no change in behaviour): drop the claim that 1e-6 L lies below the reference datasets' finest wall
spacing. State the precision rationale and its consequence: under the whole boundary the channel is linear for d below
1e-6 L, which on AirfRANS with its far field covers the viscous sublayer, the buffer layer and the start of the log
layer (y+ about 25-90, estimated). After approval I would make it on psharpe/isla-nearwall, run the ISLA test directory
on LHR cpu-short, and leave the merge to you.

**Contingency (not proposed now).** If AF-WHOLE reads a harm on velocity or nut with its registered near-wall signature
(R2: W / B larger below 3.05e-4 chords than above, on both seeds), the encoding becomes the leading suspect. The fix to
test is (ii). Before its retrain, the geometry kernel's float32 closest-point arithmetic would be measured, and if
confirmed fixed with the local-frame subtraction, so that the compute part of (ii)'s rounding goes. That fix is a
geometry change for the supervisor.
- The retrain check: AirfRANS with its far field, AF-WHOLE's configuration and snapshot with only eps changed, two
  seeds, against AF-WHOLE's W-ISLA runs.
- A DrivAerML Volume arm would be degenerate: (ii) falls back to (i) there, so the function is identical (protocol
  non-degeneracy).
- AeroZoo changes on every case, so adoption would also need an AeroZoo check.
- Cost: about 180-225 GPU-h for the AirfRANS pair (AF-WHOLE's ISLA lanes run at 1.0-1.25x the standing's 89 GPU-h).
- Time: submitted the night of 10-25, it ends about 10-27 morning, at the scope freeze.
- The trade, the frame-free rounding guarantee for a log code in the boundary layer where a viscosity is declared, is
  Peter's.

My predictions, before any AF-WHOLE result:
- AF-WHOLE's W-ISLA reads a harm on velocity or nut: 0.40.
- Given such a harm, its R2 places it mainly below 3.05e-4 chords: 0.45.
- FLOOR-DRV shows a near-wall-band harm attributable to the encoding: 0.10.

## 5. What I need

Approval of the recommendation: keep (i), and correct the docstring on psharpe/isla-nearwall. Separately: whether to
draft the contingency PREREG for (ii) now, so that it is ready before the scope freeze, to be submitted only if AF-WHOLE
reads the near-wall signature and you or Peter accept the trade. I recommend drafting it now.
