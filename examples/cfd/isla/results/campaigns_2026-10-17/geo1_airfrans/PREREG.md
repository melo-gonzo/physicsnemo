# GEO1-AF pre-registration (draft 2026-10-17, written before any lane was submitted)

**Approved by the supervisor 2026-10-16T23:44-04:00 (2026-10-17T03:44Z), as written, before any lane was submitted:**
G1-N; G1-C as the absolute bound c <= 4 with a2s's c recorded; G1-P; configuration choices 1-4 of the approval request
(compile_blocks=true; `.venv-recipe` with the snapshots' tensordict 0.15.0; submit.sh's sbatch line with RECIPE=recipe_v11;
geo1 rescored at 400 segments with the whole airfoil as `boundary_geometry`). The DrivAerML sign finding below changes
nothing in these lanes.

Step 1 of the ISLA GEO-EXACT test (design study: `scratch/isla-geo-exact-design-2026-10-16/REPORT.md`), on AirfRANS.
A training comparison that differs in one variable, the query geometry of the prediction points:

| arm | code snapshot (LHR `$T/`) | branch head | query geometry |
|---|---|---|---|
| a2s | `code_a2s` | psharpe/isla-a2 ebf841578 (A2 + merge of psharpe/isla e6cec857e) | today's: signed distance and direction to the exact nearest boundary token centroid |
| geo1 | `code_geo1` | psharpe/isla-geo-exact c7adf4d7a (A2 + GEO-EXACT + the same merge) | exact closest point on the boundary surface's segments; on AirfRANS the airfoil is read whole, so the surface is the token mesh itself (no `boundary_geometry`) |

Everything else is shared: recipe `$T/recipe_v11` (git archive of the geo-exact head's recipe directory, unmodified; AirfRANS
reads the airfoil whole, so both arms use the same recipe), venv, data, seeds, schedule and scorer.

## Lanes (rows 10-13 of `$T/kernel/airfrans_u/lanes.tsv`)
| idx | run_id | code | seed |
|---|---|---|---|
| 10 | afu_a2s_isla_seed42 | code_a2s | 42 |
| 11 | afu_a2s_isla_seed43 | code_a2s | 43 |
| 12 | afu_geo1_isla_seed42 | code_geo1 | 42 |
| 13 | afu_geo1_isla_seed43 | code_geo1 | 43 |

Common columns, the reference configuration of rows 8/9 (afu_bn2, A2 on code_v20a2): dataset `airfrans_uvolume_2d` (uniform
interior draw, airfoil whole), model `isla_volume_airfrans`, extra `train_split=full_train val_split=full_test
+training.val_interval=10 model.hidden=512 model.n_slices=512 model.compile_blocks=true`, lr 3.75e-4, cells 100000, 500
epochs. `lanes.sbatch` adds `precision=bfloat16 augment=false compile=false`, 4 GPUs, save every 2 epochs.

Two configuration notes (facts, checked in STEP C):
* `model.compile_blocks=true` restores the reference lanes' compute configuration. recipe_v8's `isla_volume_airfrans.yaml`
  set it; recipe_v11's sets it false because the template now compiles the whole model, which `lanes.sbatch` disables
  (`compile=false`). It changes speed, not the model.
* Environment: `lanes.sbatch` sets `PYTHONPATH=<code>:<recipe>/src` and picks `.venv-recipe` for these snapshot names
  (torch 2.12.0+cu130, warp 1.14.0). Both snapshots carry tensordict 0.15.0 (`td015`, copied in, recorded in SNAPSHOT_HEAD and
  MD5SUMS), which shadows the venv's. STEP C ran geo1 on `.venv-recipe` and on `.venv-recipe-td14`: identical loss to the
  last digit (0.38506099581718445) and identical geometry; the two venvs differ only in tensordict and the zarr stack.

Submission: `lanes.sbatch` exactly as `submit.sh lanes` does it (one singleton chain per lane, `isla.afu.<slug>`, QOS normal,
03:59:00), with `RECIPE=$T/recipe_v11` in the export (submit.sh hard-codes recipe_v8). Acceptance: each lane >= 100 steps,
no traceback, step time recorded against afu_bn2's 0.386 s/step.

## Scoring
* Native, area, posed: `$T/kernel/airfrans_u/score.sbatch` (float32 infer.py, full_test, 200 cases, 100,000 interior points,
  final epoch-500 checkpoint), its code mapping extended with afu_a2s_* -> code_a2s and afu_geo1_* -> code_geo1, both on
  recipe_v11 and `.venv-recipe` (the training environment).
* 400 boundary segments: the CHAR-AREA2 protocol (`scratch/isla-char-area2-2026-10-16/PREREG.md` with its amendment) after
  training: `GuardedPoissonSubsampleMesh` at 400 airfoil segments, twins ca2a (area law), ca2u (uniform law) and ca2v
  (uniform, independent boundary seed, interior in lockstep), on a recipe_v11 copy whose only changes are that transform and
  the twins. **geo1 is scored with `boundary_geometry: {airfoil: <the whole airfoil>}`**, a copy of the airfoil taken
  before the Poisson step; that is the design (the tokens are the sample, the surface stays whole), and geo1 refuses
  HT-scaled tokens without it. a2s is scored on the same files; its forward does not read the copy. If the copy cannot be
  made without changing anything else in the pipeline, the scoring lane reports before scoring.

## Definitions
Per field (pressure p, velocity v, nut) and per seed: E = mean over the 200 cases of the per-case relative L2 (infer.py).
c = E(ca2u) / E(native); r = E(ca2a) / E(ca2u); d = |E(ca2v) / E(ca2u) - 1| (noise floor).

## Predictions
* **(G1-N) native accuracy.** On p and on v: mean over seeds of E_native(geo1) <= 1.05 x mean over seeds of
  E_native(a2s). Expected 0.84-0.93 (7-16 % better, PATCH-SDF native).
* **(G1-C) count effect at 400.** geo1's c <= 4 on p and on v, each seed. a2s's c is recorded; expected near the
  x13-18 of today's geometry (CHAR-AREA2 afu_r_isla).
* **(G1-P) law effect at 400, pressure.** geo1's r on p <= 1.5, each seed.

Recorded, not predicted: r on velocity (GEO0: the 400-segment velocity law effect, 1.8-2.0, is not geometry) and on nut;
c and r on nut; every d; both arms' posed / native; a2s native against afu_bn2 (same A2 model math on recipe_v8 /
code_v20a2: p 0.0182 / 0.0176, v 0.0095 / 0.0094, nut 0.0212 / 0.0214); step times.

## STEP C evidence (LHR jobs 243376, 243405; logs in `check_logs/`)
* Eager forward + backward on one airfrans_uvolume_2d case (100,000 queries, 1,007 airfoil segments), bf16, both arms:
  finite loss, outputs and gradients; 306 parameters, none without a gradient, none all-zero.
* geo1 geometry against a float64 brute force over the airfoil's own segments: max |d| error 3.1e-7 (median relative
  6e-8), 100 % of queries within 1 %, d >= 0 everywhere (0.6 % are wall nodes at d = 0), |u| = 1 to 1.2e-7.
  a2s (nearest token) on the same case: 0.27 % of queries get d <= 0 (min -0.0041), 65 % within 1 % of the exact distance.
* `model.compile_blocks=true`, two cases: finite, both arms.
* Whole-model `torch.compile(fullgraph=True, dynamic=True)` (mode default), forward + backward on two cases with different
  airfoil segment counts: one graph, no recompile, no graph break, both arms.
* geo1 on DrivAerML Volume (`model=isla_volume dataset=drivaer_ml_volume_reference`, full STL 753,234 triangles,
  200,000 vehicle tokens, 200,000 queries): finite outputs. The orientation vote flipped the STL to the tokens' side and
  then agrees with 99.995 % of the token measure (weighted cos 0.9989). |d| equals the stored full-mesh SDF to a median of
  1.9e-9. **But d is negative in 98.5 % of the fluid queries**: d = -sdf (sign agreement 0.017 %). The vehicle tokens'
  normals point into the car (the inward `.pdmsh` triangles), and the model's convention is "positive on the side the
  tokens' normals point to", as today's nearest form is. Not relevant to these AirfRANS lanes (d >= 0 in the fluid
  there); relevant to the DrivAerML Volume half of Step 1.
