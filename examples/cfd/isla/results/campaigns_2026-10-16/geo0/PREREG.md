# GEO0 pre-registration (2026-10-16; written before any scoring job was submitted; approved by the supervisor 2026-10-16T21:16-04:00 with the amendment at the end)

Scoring-only Step 0 of the ISLA exact-geometry design (scratch/isla-geo-exact-design-2026-10-16/REPORT.md): are ISLA
checkpoints trained with the dataset's stored full-mesh SDF as the query geometry (`sg_*_provided`,
`interior_geometry=provided`) insensitive to how the boundary is sampled? CHAR-AREA2 protocol (reduced-budget Poisson
redraws, exact HT measures, guard, uniform-B noise floor), on the code and recipe the checkpoints were trained and scored
with.

## Step 1 findings
* Checkpoints: AGA only (`$TA/runs/sg_{af,drvol}_{provided,internal}_seed{42,43}`), both seeds, all 8 "Training
  completed", epoch 500. Copied with code_v3 and recipe_v3 to LHR `$T/kernel/geo0/{runs_from_aga,snap_from_aga}`
  (md5 identical: md5_runs_{aga,lhr}.txt 40 files, md5_snap_{aga,lhr}.txt 1,097 files).
* Class: ISLA R1 (hidden 384, 12 layers, 256 slices, cosine routing), code_v3 + recipe_v3, 500 epochs, bf16 training;
  scored float32 by the p200k fleet scorer with code_v3 + recipe_v3 (same infer.py call as CHAR-AREA2). NOT the
  512/512 class of the CHAR references, and AirfRANS is the 3-D `airfrans_volume` set (100k-point contiguous
  solver-order interior window, SAMPLING-AUDIT), not `airfrans_uvolume_2d`.
* Geometry at scoring time (code_v3 model.py 2079-2118): under `provided`, a query's frame normal is the stored
  `interior.point_data.sdf_normals` and its rank-0 input the stored `sdf` divided by the gauge L; no
  nearest-token search runs. L (measure-weighted RMS pairwise distance of the boundary tokens) is the only
  boundary-dependent factor in the query geometry, an HT quadrature. Under `nearest_boundary` (`sg_*_internal`) d and
  u come from the nearest boundary token. AirfRANS dry run (case 0, `dryrun_af_case0.log`): interior points, `sdf`
  and `sdf_normals` bitwise identical across native / area / uniform / uniform-B; DrivAerML computes `sdf` from the
  full STL on the interior points before the redraw step, so it is unchanged by construction.
* recipe_v3 difference handled: the AirfRANS model configs read the converter's stored
  `boundaries.airfoil.cell_data.quadrature_measure` (segment lengths), which a `_measure_weights` fold alone would
  not change; the ported guard class multiplies that field by 1/pi too. Dry run: kept 407 / 440 / 414 of 994,
  measure 2.053 -> 2.097 / 2.267 / 2.131, Jaccard area-uniform 0.450, uniform-uniformB 0.269 (CHAR-AREA2: 0.429 / 0.225).

## Protocols (recipe copy `$T/kernel/geo0/recipe_v3g0` = recipe_v3 + appended GuardedPoissonSubsampleMesh + 6 yamls)
* AirfRANS (`airfrans_volume_g0{a,u,v}`): native, then an expected 400 airfoil segments: g0a area law, g0u uniform
  law (lockstep with g0a), g0v uniform with draw_seed 1 (independent boundary draw, interior in lockstep).
* DrivAerML Volume (`drivaer_ml_volume_reference_g0{a,u,v}`): the CHAR area draw in recipe_v3 form (1,000,000-cell
  uniform reader pool, Poisson to an expected 200,000 after the SDF is computed from the STL); g0u the uniform law
  from the same pool (same path), g0v g0u with draw_seed 1.
* native rescored on LHR for every run (environment check against the committed 2026-09-23 records).

## Measures (seed means over cases of the per-case relative L2, per field, per seed)
r = g0a / g0u; d = |g0v / g0u - 1|; count c = g0u / native; DV law-vs-native R = g0a / native.

## Predictions
* (ENV) rescored native within 1% of the committed 2026-09-23 records on every field and run; otherwise stop.
* (G0-AF) sg_af_provided: r <= 1.5 on pressure and velocity, both seeds (reference afu ISLA 2.24/2.01, 1.91/1.85),
  and c <= 4 on every field (reference x13-18).
* (G0-DV) sg_drvol_provided: R within [0.95, 1.05] on every field, both seeds (reference ISLA x6.6-7.9).
* (M-AF, matched control) sg_af_internal: r > 1 + max(0.05, d) on pressure and velocity, both seeds, and r - 1 for
  provided smaller than for internal on pressure and velocity, both seeds.
* (M-DV, matched control) sg_drvol_internal: R >= 2 on every field, both seeds.
* Recorded: nut r; g0u / native and g0a / g0u on DV; d everywhere; the design study's sharper DV forecast
  (provided R within 1 + max(2d, 0.01)).

## Amendment (2026-10-16T21:16-04:00, before any scoring job was submitted)
* Approved by the supervisor with the matched sg_*_internal controls (32 units: 8 runs x native / g0a / g0u / g0v).
* PRIMARY READ: the provided-vs-internal comparison (same snapshot, class, data and training; only the query
  geometry differs), i.e. (M-AF) and (M-DV) together with (G0-AF) and (G0-DV) on the same scoring. Comparisons with the
  CHAR-AREA2 / CHAR references (512/512 class, other snapshots, airfrans_uvolume_2d) are context only: these are ISLA R1
  (384 wide, 256 slices, code_v3) on the 3-D airfrans_volume set with the windowed interior.
* Every bar is kept as written, including the near-edge v <= 1.5 and c <= 4; no retuning after results.
* Reported for every unit and seed: r, c, R (DV), d.

## Implementation note (2026-10-16T21:24-04:00, after job 241665 and before any DrivAerML redraw result existed)
The 12 DrivAerML g0a / g0u / g0v units of job 241665 all stopped at their first case in the guard: a 722-cell in-file
boundary besides the vehicle reached the step (`RuntimeError: GEO0 guard [vehicle]: 722 cells ...`); no infer row was
written. The model reads `boundaries.vehicle` only, and the CHAR area protocol passes such small boundaries through
unchanged. Fix: new twins `drivaer_ml_volume_reference_g1{a,u,v}.yaml` = g0 with `require_resample: false` (small meshes
pass through unchanged, before the shared random draw, so lockstep is unaffected). The no-op guard is replaced on
DrivAerML by a log check: every g1 unit must have one `GEO0 tag=vehicle` line per case with n_before = 1,000,000 and
n_kept near 200,000. Protocol names g1a / g1u / g1v stand for g0a / g0u / g0v in every DrivAerML prediction; no bar
changes. The AirfRANS units (job 241664) completed with the guard on (200 guard lines per redraw unit).
