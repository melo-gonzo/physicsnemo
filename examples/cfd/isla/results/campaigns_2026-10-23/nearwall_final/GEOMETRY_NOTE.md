# NEARWALL: float32 rounding of ISLA's wall distance far from the frame's origin (note for the geometry lane)

Written 2026-10-24 about 09:15 EDT by lane NEARWALL, at the supervisor's request (08:43 EDT). This is a geometry
precision question that NEARWALL found and does not pursue. Nothing here changes the current encoding, which keeps the
channel within 0.12 everywhere.

## What was measured

Probe `harness/nwprec2.py` (md5 00cdd5c1), LHR GB300 short QoS job 367749, code `$T/code_nearwall` (2b15490b2,
ISLA's `exact_boundary_geometry`, `kernel="reference"`), recipe `$T/recipe_nearwall`. Inputs are the recipe's drawn
query points (val side, augment off) and the whole boundary each config reads: AirfRANS with its far field (12 cases),
DrivAerML Volume with its five pieces (3 cases), and AeroZoo (16 cases).

For a translation T (applied in float64, then the query points and vertices cast to float32, which is the input a model
would get with the case placed T from the origin):
- **compute** = `exact_boundary_geometry` in float32 against float64 on that same float32 input;
- **input** = float64 arithmetic on the translated float32 input against float64 on the stored native case (the cost
  of storing the case elsewhere in float32, which no kernel can undo).

The change is reported as the change in asinh(d / eps). Raw values are in `out/p_*_v2.json` and the tables in
`precision_tables_v2.md`. Wall nodes (d = 0 exactly) stay 0 in every frame.

## The numbers that matter (AirfRANS with its far field, L = 305.3 chords, off-wall points, p99.9 / max)

| eps | rounding | native | T = L body (0.42 chords) | T = 0.1 L (30.5 chords) | T = L (305 chords) |
|---|---|---|---|---|---|
| 1e-6 L (current) | compute | 4e-5 / 1e-4 | 1.2e-4 / 2.3e-4 | 0.008 / 0.012 | 0.11 / 0.12 |
| 1e-6 L (current) | input | 0 | 1e-4 / 1.6e-4 | 0.006 / 0.008 | 0.046 / 0.065 |
| nu / U (5.5e-10 to 1.6e-9 L) | compute | 0.002 / 0.009 | 0.010 / 0.029 | **3.1 / 3.8** | 5.9 / 6.1 |
| nu / U | input | 0 | 0.009 / 0.023 | 0.44 / 3.1 | 4.6 / 8.1 |
| 1e-9 L | compute | 0.002 / 0.009 | 0.010 / 0.029 | 2.8 / 3.2 | 5.4 / 5.5 |

The first nodes off the airfoil are at about 5.2e-9 L (2e-6 chords), 11.5x below 2^-24 L. In d itself, the compute
error over L reached 1.9e-8 at T = 0.1 L and 1.7e-7 at T = L (v1 job 367615, which mixed the two roundings; the v2
split is above). On DrivAerML every arm stays at or below 0.051 (compute) at any T; on AeroZoo the current eps stays at
or below 0.12.

## Where it comes from (the reviewer's reading, not verified by me)

The design reviewer (`REVIEW_DESIGN.md`, finding 1) located the compute part in the closest-point arithmetic on
absolute coordinates: `geometry.py:1722` `closest = (weights[..., None] * simplex).sum(1)` and `:1817`
`diff = query_points - closest` (line numbers on psharpe/isla 1b25eb899). Both sides are about T in size, so their
difference carries an error of about ulp(T), which is far more than a 2e-6 chord distance.

Its numpy point-to-segment probe (float32, T = 0, 30 and 305 chords; a script in its /tmp, not kept) reported:
- absolute closest point (as the kernel does it): up to 0.74 relative error in d at 30 chords and 5.7 at 305;
- the same float32 arithmetic in the closest cell's local frame, `(q - v0) - sum_i w_i (v_i - v0)`: below 6e-5 relative
  at every T;
- the input's own rounding for a node 1.6e-6 chords off the wall stored at T = 30 chords: up to 0.98 of its distance
  (on the wall at 305), which no kernel can undo.

## What it would mean

- With the current eps it does not matter: the channel moves by at most 0.12 under both roundings.
- For any eps that gives the first AirfRANS nodes a logarithmic code (the viscous nu / U, a wall-only gauge, a smaller
  constant), the compute part dominates at 30 chords (p99.9 2.5-3.1 against 0.43-0.44 for the input). A local-frame
  subtraction, if it holds in ISLA's kernel (2-D brute force and 3-D Warp BVH, with the fast path too), would leave
  only the input part. The input part is about 0.44 p99.9 and 3 max at 30 chords, still far above 0.12.
- It matters beyond the encoding: d enters every interior task. A local-frame subtraction would make d's rounding
  independent of the body's position in the frame, which is what the position-free design intends.

## Suggested checks for the geometry lane

1. Reproduce the reviewer's numpy result on ISLA's reference kernel: patch the closest point to the local-frame form,
   then rerun `nwprec2.py` ("compute" rows) at T = 0.1 L and T = L on AirfRANS with its far field.
2. The same for the fast kernel (`geometry_fast.py`, the Triton path) and the 3-D Warp search on DrivAerML Volume.
3. Confirm that outputs are bitwise or near-bitwise unchanged at native (the native compute error is already 1e-4 in the
   channel), and run the whole ISLA test directory.
