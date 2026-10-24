# FOLD-SIGN pre-registration: the f32floor rerun with the winding-number sign (GEO-ROBUST 2a, item 2)

Written 2026-10-22 16:07 EDT, before any job of this rerun was submitted and before any number from it existed.
Known when writing: F32FLOOR's README, tables and per-case geometry diagnostics (out/geom/geom_3349{62..65}.json);
CMPDIAG's per-case d(float32, float64) for azc_full_isla_seed42 (cmpdiag/tables.md); this lane's CPU sign-change
counts on Tensile2d val_100 (0 changes) and the first lug cases (0 changes). No AeroZoo number from the new geometry.
No test-side score is read: RMS(target) only normalizes d, as in F32FLOOR.

## What runs

- Code: a new read-only snapshot `$T/code_foldsign` = `$T/code_tie` (code_cmpaf_fix + d3d9da49d's geometry.py and
  tree_read.py) with geometry.py replaced by psharpe/isla-fold-sign's, built only after TIE-FIX is merged into that
  branch (supervisor 16:00). Recipe `$T/recipe_r090`, unchanged. The snapshot's SNAPSHOT_HEAD names the commit.
- Checkpoint, inputs and settings as F32FLOOR: azc_full_isla_seed42 epoch 250, `model=isla_volume_aerozoo`,
  `train_split=full.train val_split=full.test infer_split=test`, 200,000 interior points, eval mode, eager, no compile,
  fused geometry kernel, the float64 model a `copy.deepcopy(model).double()`.
- Cases: CMPDIAG's 20 ISLA cases (00002 00017 00019 00022 00026 00031 00036 00037 00039 00042 00045 00067 00068 00078
  00079 00082 00083 00086 00089 00093), which hold F32FLOOR's four high-d ISLA cases (00019, 00022, 00045, 00083), its
  two ISLA controls (00037, 00039), and the cases with negative-d fluid points (00002, 00037, 00079).
- Arms: A32 (float32 throughout) and A64 (float64 throughout), as F32FLOOR, with fresh meshes per arm.
- Geometry readings in the same job, on the delivered points: exact_boundary_geometry of the snapshot in float32 and
  float64 (the d of A32 and A64); the base geometry (code_r090's geometry.py, loaded under its own op name) in float32
  and float64; the winding-number margin |w - sigma/2| (the branch's `_winding_sign` internals, float64).

## Readings (per case; each "before" from the source named)

1. **Sign flips** N_flip = #{points with d32 * d64 < 0}, both nonzero with opposite signs. Reported apart: N_zero32 =
   #{d32 == 0 (either sign of zero) and d64 != 0}, the float32 magnitude rounding to 0 (TIE-REVIEW finding 3).
   Before: F32FLOOR's CPU counts 00019 40, 00022 427, 00045 69, 00083 12, 00002 40, 00037 0, 00039 0, 00079 0, 00089 0,
   00031 0, 00042 0 (sign(s32) != sign(s64); the other 9 cases were not measured); in this job also the base
   geometry's N_flip on all 20.
2. **The floor** gap = d(A32, A64), max over the five fields (d as F32FLOOR). Before: CMPDIAG's d f64-vs-f32 max
   (00019 1.2e-1, 00022 5.6e-2, 00045 2.8e-1, 00083 1.3e-2; the other 16 at 3.0e-5 to 8.7e-3).
3. **Negative-d fluid points** N_neg32, N_neg64 = #{d < 0} (every interior point is fluid), for the snapshot and the
   base geometry; and for the points with base d64 < 0: how many have snapshot d64 > 0 (turned positive), and how
   many of those and of the rest lie in the band. Before: F32FLOOR 00002 53/83, 00019 0/40, 00022 903/740, 00045
   34/103, 00079 51/51, 00083 18/12, 00037 118/118, 00031 1/1, others 0 (float32/float64).
4. **The band** (supervisor condition 5): min over the off-wall points of |w - sigma/2|, and the count below 3e-2 (the
   measured far-field error of the winding number at beta = 2), in float64.

## Reading rules

- R1. "The sign flips go to 0" when N_flip = 0 on all 20 cases outside the band. A flip inside the band is the
  design's known limit (DESIGN k) and is reported with its location; a flip outside the band is a defect, reported
  with its location, and the rule reads "not 0".
- R2 (same 0.2 threshold as F32FLOOR). The new geometry removes most of a case's floor when gap_after <= 0.2 x
  gap_before. Verdict: "the fold sign removes ISLA's float32 floor on AeroZoo" when this holds on at least 3 of the 4
  high-d cases and gap_after <= 1e-2 on all 20; "it removes the fold part, a residual floor remains on <cases>" when
  3 of 4 hold but some case stays above 1e-2; "it does not remove the floor" when at most 2 of 4 hold. Confound,
  stated now: the snapshot differs from code_r090 also by GEO-ROBUST phase 1 (the tie rule, the on-wall rule) and
  TIE-FIX, so the after floor is that of the whole new geometry; F32FLOOR implicates only the sign (W32 removed the
  gap to 0.007-0.044 on 00019, 00022, 00045, but only to 0.52 on 00083).
- R3. Reported, no verdict: N_neg before and after, the turned-positive count, and the band counts. A point that stays
  negative outside the band is a fluid point that the closed boundary puts outside the domain; it is reported with its
  location for a data check.

## Predictions (my probabilities)

- R1: N_flip = 0 on all 20 cases: 0.85.
- R2: at least 3 of 4 high-d cases at <= 0.2: 0.6; gap_after <= 1e-2 on all 20: 0.45. 00083 reaches 0.2: 0.4.
- R3: every base-negative point turns positive: 0.5. The band count is 0 on every case: 0.7.

## Cost

One LHR short-QoS job, 1 node x 4 GB300, 5 cases per GPU, one float64 network pass (about 50 s) plus a float32 pass
and four geometry calls per case: about 15 min wall, about 1 GPU-h allocated. Job name `isla.foldsign.f32floor`
(registry row `^isla\.foldsign\.`).
