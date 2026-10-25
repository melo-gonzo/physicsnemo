# AZ-SPEC-AUDIT: our AeroZoo runs against the AeroZoo task specification

Written 2026-10-24 21:38 EDT by AZ-SPEC-AUDIT (read-only; no jobs, nothing changed); revised at 21:50 EDT after the
lane-reviewer's REVIEW.md ("AUDIT stands with corrections", no P0; its corrections are applied, see the end).

**Spec.** `/home/psharpe/gh/aerozoo` main 96bd2b62, `docs/book/06-splits.qmd` (md5 52118947), cited `06:L`; commits
18e004c2, 9d500c69, ea74fb29, d5ab6ded, b7c66b9b and their notebook entries. Checklist S1-S28: `SPEC_DIGEST.md`.
**Runs.** Every AeroZoo number that `examples/cfd/isla/book/13-aerozoo.qmd` @ 5b70c8686 prints: AZ-DECIDE's 44 lanes
(`$T/runs/azc_*`), E1-AZ HEAD-NORM (2 lanes `hn090az_isla_*`, nonbox), LRE-AZ (2 lanes `lreaz_reynolds_isla_*`), and
the diagnostics 13 cites (gtopt, pdiag, preddiag, scales, cmpdiag, f32floor, qpop_az). Evidence files:
`INPUTS_STATS.md` (inputs and fitted statistics) and `SPLITS_METRICS.md` (splits, validation use, targets, metric, frame).

## Answer to Peter's question

**No model gets unfair extra data.** Every tensor that reaches every model's forward is the posed problem, something
computed from it, or the query points; every fitted constant comes from the training part of the run's own split; every
run trains, validates and is scored on exactly the AeroZoo v1.2 effective split lists, with no case missing. No model
reads `global_data/reference`, `annotation/body_id`, the cell volume `_target_quadrature_measure`, a pipeline-supplied
length, a per-case centring or any file beside `domain.pdmsh` (other than the split manifest).

**But the runs are not exactly consistent with the specification on the metric.** The book reports the per-case relative
L2 of each physical field, every cell counted equally, on 200,000 random cells, in float32, averaged over cases. The
spec's interior metric is ε_U and ε_p: the RMS error weighted by cell volume, in units of the case's U_ref and q_ref,
over every cell, in float64, with the near-field ε^N beside them and a second score over the `mean_stable` cases; and
the spec says other scores go "beside these, but not instead of them" (06:398-401). Every model is scored by the same code
on the same cells, so the comparison is internally fair, but it is not AeroZoo's metric, and two close verdicts could
change under it (M1 below). The other deviations give a model less data, never more, or are reporting gaps.

## Verdict table

C = consistent, D = deviation (ID below), n.a. = not applicable. Snapshots: ISLA and ISLA-DTT `code_r090` (git 92d9d7800)
+ `recipe_r090`; E1-AZ `code_hn090` + `recipe_r090`; LRE-AZ `code_r090` + `recipe_lreaz`; GeoTransolver and Transolver
`code_azv2` (git 105f62d9f) + `recipe_az_c` (git caac441d7). Each lane's `train.log` records its pair; the resolved
configs hash to one value per family (INPUTS_STATS §1).

| item | ISLA (interacting), 14 + E1-AZ 2 + LRE-AZ 2 | ISLA-DTT, 14 | GeoTransolver, 14 | Transolver, 2 (`full` only) |
|---|---|---|---|---|
| 1a. Geometry and condition types | C: the 4 full-resolution boundary meshes by name (isla yaml:82; SubsampleMesh, SDF, Harmonize, Merge dropped for ISLA, yaml:48) | C: same | C: 50k-face uniform draw per boundary, one-hot condition type from Harmonize (gt yaml:61-83, dataset yaml:65,151); less input than ISLA, allowed (I2) | **D (I1)**: no boundary condition at all: no prescribed values, no condition type; only `interior.points`, `sdf_wall` and normals, `nu_std` |
| 1b. Prescribed values | C: only the declared fields, `m.cell_data.select(declared)` (model.py:1613): inlet velocity, k, ω, γ, Re_θt; outlet gauge pressure and reverse-flow values | C | C: each face's prescribed values, zero-filled for the fields its boundary lacks (Harmonize, dataset yaml:63-82 @ caac441d7), standardized by train-split constants | D (I1) |
| 1c. Global inputs | C: `td.select(global_schema)` (model.py:1525) keeps only `physics.density` (pressure units) and `physics.kinematic_viscosity` | C | C: `nu_std` = (log10 ν − a)/b only (dataset yaml:106) | C: `nu_std` broadcast to every point |
| 1d. `_target_quadrature_measure` as input | C: not read; ISLA's token measure is the full-resolution `cell_areas` (ComposeQuadratureMeasure, yaml:142; model.py:1591); interacting ISLA's query routing mass is a learned share of the total boundary measure over nq (model.py:1874-1877), not the cell volume | C (DTT: no query mass, model.py:1868) | C: not a feature | C |
| 1e. `global_data/reference` | C: dropped by the schema select | C | C: no L_ref/U_inf in the chain; recipe paths that read L_ref are conditional and never fire (forces.py:334, infer.py:316-318 @ caac441d7) | C |
| 1f. Body annotation | C: not read; the geometry is the union of all four boundaries | C | C: SDF over all `no_slip` faces, body and walls alike | C |
| 1g. Per-problem length from the pipeline | C: none; ISLA computes L itself, sqrt(2 Var x) over all boundary cells at full resolution (geometry.py:71-104), the same quantity as `derive_globals`' length | C | C: none; raw metres | C |
| 1h. Inlet velocity and Re | C: U = area-weighted RMS speed of the prescribed inlet velocity, g = direction of its area-weighted mean (scales.py:200-209, model.py:1347); Re = U L / ν, log10 standardized by train-split constants (model.py:1360-1361). The spec's `derive_globals` uses the area-weighted *mean* speed; both are computed from the posed problem inside the model (06:137, 06:180-183) | C | n.a. (not given; S9) | n.a. |
| 1i. Centring | C: no CenterMesh or any centring in any chain (removed 2026-10-19); ISLA has no origin | C | C: raw metres in the stored random pose | C |
| 1j. Files beside `domain.pdmsh` | C: only the split manifest (`aerozoo_v1.2_shuf_pdmsh/manifest.json`, md5 2a7989ad); reader pattern `shard-*/case-*/domain.pdmsh` never reads `result.yaml`, `status.json` or quality records | C | C | C |
| 2. Statistics | C: target norm stats, `isla_constants` (reynolds, input and output scales) fitted on `<split>.train` (isla_constants_fit.py uses `az_man[f"{s}.train"]`; n_cases = n_train for all 7 splits); checkpoint args carry the split's values; LRE-AZ uses fixed {6, 1} | C | C: `norm_stats.pt` one hash per split, identical across architectures; `nu_std` a, b and boundary-value standardization per `<split>.train` (aerozoo_pstats.py:134-191 refuses a missing case) | C |
| 3. Splits and data | C: AeroZoo v1.2 (index sha256 c8e9508e…), row-permuted copy; all 21 train/val/test lists of the 7 splits equal splits.json ∩ v1.2 complete, by md5; scored test counts 716 / 333 / 1367 / 1074 / 578 / 516 / 1385, 0 missing, 0 extra, 0 duplicates; no training filter | C | C | C (`full`) |
| 3b. Validation use | C: final checkpoint (epoch 250), validation only monitored (PREREG_v2 §11(c)); **but D (S1)**: §12(f) chose the default interior form from test sides; note S1b | D (S1), same read; note S1b | C (note S1b) | C (note S1b) |
| 3c. Augmentation | C: none (`augment: false` in the resolved configs; no augmentation block in either recipe's dataset files) | C | C | C |
| 4a. Targets | **D (T1)**: 5 of the 7 interior fields (γ and Re_θt dropped, dataset yaml:125-126); ε_U, ε_p unaffected | D (T1) | D (T1) | D (T1) |
| 4b. Metric | **D (M1)** | D (M1) | D (M1) | D (M1) |
| 4c. Loss weight | D (L1, allowed): per-case relative L2 per field (velocity ×3), each sampled cell counted equally, not volume-weighted (`extract_target_measure` returns None, forward_kwargs.py:301-309 @ 92d9d7800; tests/test_aerozoo_configs.py:675 @ 92d9d7800, :664 @ caac441d7) | D (L1) | D (L1) | D (L1) |
| 4d. Reporting | D (R1): 13 does not name the release version or the `index.json` hash, and gives no `mean_stable` score | D (R1) | D (R1) | D (R1) |
| 5a. Body annotation now a label (ea74fb29) | C: never read | C | C | C |
| 5b. Rotation-invariant length "given to every model" (d5ab6ded) | n.a.: superseded 52 minutes later by b7c66b9b; the current spec gives no model a length (06:180-183; paper/baseline-campaign.md:25-31). ISLA divides coordinates by its own L and takes L through Re, as 06:189-191 requires | n.a., as ISLA | C: lacks nothing the spec gives; coordinates not divided by any per-problem length, so 06:189-191 does not apply | C, as GT |
| 5c. No per-problem length from the pipeline (b7c66b9b) | C | C | C | C |
| 5d. Reference scales are scoring conventions (18e004c2); only the equations' data are inputs (9d500c69) | C | C | C | C (but sees less than the equations' data: I1) |

Every C and D above is backed, with file:line at the named snapshot or a resolved-config key, in INPUTS_STATS.md and
SPLITS_METRICS.md.

## Deviations: direction, size, fix

**M1, different metric (all 48 lanes, every model the same way).** Ours (13:393, 13:419; `infer.py` → `src/metrics.py:105-120`
→ `physicsnemo.metrics.general.relative_l2`, md5s identical in both snapshots): per case and field sqrt(Σ(ŷ−y)²/Σy²) over
200,000 seeded random cells, float32, then the arithmetic mean over cases (median beside it), then over seeds. Spec (06:282-298,
06:429-472): volume-weighted RMS in U_ref and q_ref over every cell, float64; ε^N within L_ref of the body; mean and median over
all test cases and over `mean_stable`. Four parts:
- normalization by the case's own field norm instead of U_ref, q_ref: a per-case factor the same for every model, so it can
  reorder models only through the aggregate over cases;
- no volume weight: ours weights the small near-wall cells, the spec the large far-field cells; this can reorder models;
- 200k random cells instead of all: a consistent estimate, with sampling noise, of the full-mesh unweighted relative L2,
  on the same cells for every model at a seed;
- no ε^N, no `mean_stable` score, float32.

Size, estimated from the per-case score records (SPLITS_METRICS §4b; the normalization part only, the volume weight cannot be
estimated without rescoring): GeoTransolver ÷ ISLA keeps its order on every split and field except `family_non_box` velocity,
where the mean ratio is 1.108 but GT is better on 65% of cases, so a reweighting with a spread of 1.54 could flip it. On pressure
the estimated ε_p ratio GT ÷ ISLA grows on the box splits (for example `full` 7.87 against relL2 4.12). ISLA-DTT ÷ ISLA on pressure
flips on nonbox (est. 0.96 against relL2 1.27) and reynolds (1.05 against 0.93); §12(f)'s consequence (ISLA-DTT the default) would
not change. E1-AZ's pressure ratio, HEADNORM ÷ CONTROL, falls from 1.287 (relL2) to an estimated 1.074, under the 1.10 bar that
fired "reject branch 1": **that registered verdict rests on a metric the spec does not recommend and could reverse.** LRE-AZ's
consequence (not adopted) does not weaken. Under this estimate ISLA's ε_p on crossflow (4.95) is above the zero-pressure
predictor's (2.59). Direction: a different metric, not extra data. Fix: rescore the 48 existing checkpoints with the spec's
reference functions (`rms`, `near_field`, 06:429-472) over every interior cell, weighted by `_target_quadrature_measure`, in
float64, by chunked full-mesh inference; report ε_U, ε_p, ε^N_U, ε^N_p, mean and median, all cases and `mean_stable`, with
relL2 beside them. No retraining. Needs a scorer change and GPU time (cases up to 24 M cells), so it is the main session's
and Peter's call.

**T1, narrower task (all lanes, same for every model).** We predict velocity, pressure, ν_t, k and ω; the spec's interior task
asks for all seven fields of `interior/point_data/solution/*` (06:59, 06:107); γ and Re_θt are dropped (dataset yaml:125-126).
13:155 says "the model predicts all five". Direction: less asked of every model; ε_U and ε_p do not depend on it. Fix: add the
two targets in a future protocol, or name the task "the interior task, five of the seven fields".

**I1, less data (Transolver, 2 lanes).** Transolver receives no boundary value and no condition type, so it is not given the
posed problem of tbl-tasks: it cannot tell cases that differ only in their inlet apart. Pre-registered as a floor
(PREREG_v2 §11(d)), and stated in 13 (tbl-aerozoo-inputs, the limits list). Per 06:132-133, a result that does not take the
posed problem "answers a different question; report it as such". Fix: label the Transolver row in 13 as a floor without the
boundary conditions (it largely is), or give it per-point features computed from the posed problem.

**I2, less data within the spec (GeoTransolver 14, Transolver 2).** GT reads a 50,000-face uniform draw per boundary and both
baselines' SDF comes from the 50k-face `no_slip` draw; ISLA reads every face and draws 200k area-weighted tokens inside the model.
Allowed; it is a difference in how much of the input each model reads, already listed in 13's limits ("Not resource-matched").
Fix: none needed; say it beside tbl-aerozoo-inputs.

**L1, loss not volume-weighted (all lanes, same for every model).** tbl-tasks names the cell volume "Weight, for the loss and the
metrics" (06:61); our loss is the per-case relative L2 over the sampled cells, unweighted. A loss is a modelling choice and the
spec allows it to use the volume, not requires it; no data is added. Fix: none required; it matters only together with M1.

**S1, a choice made on test sides (ISLA and ISLA-DTT, program level).** PREREG_v2 §12(f) (lines 236-246) picks the default interior
form, interacting ISLA or ISLA-DTT, from the held-out test scores. The spec reserves the validation part "for choosing models and
settings" and forbids any statistic of a test sample (06:139). The book reports both forms as separate rows, so nothing is hidden,
and Peter's "on principle" ruling for the close-call band was made before the read; but a later AeroZoo claim for "ISLA" meaning
"the form §12(f) picked" carries a small selection on test between two close variants. Fix: repeat the §12(f) read on the
validation sides (scoring only, no retraining), or keep both rows and say the default was chosen by principle on a test-side close
call.

**S1b, design-time looks at test cases (every model, found by the reviewer).** PREREG_v2.md:159-161 (§10 Q3) adopted the
per-case relative loss for every arm because the 3 largest cases dominate an absolute one, a figure from DESIGN_v2.md:70-74
computed on 22 cases, "one per AeroZoo test split" (`v2/checks_v2.md:1`, per-case target mean squares at :20-23): a setting
chosen with a statistic of test-sample solutions (06:139). The same loss went to every model, and the fact it rests on
(targets spanning decades across cases) also holds on full.train (aerozoo_uvolume.yaml:129-132 @ caac441d7), so its effect is nil
and no model gains. Two more such looks inform no model: the LRE-AZ dataset comment reports ISLA's z range on `reynolds.test`
(its constants are the fixed {6, 1}, not fitted), and the DESIGN_v2 scan 285444 took input-only statistics over 1,156 cases
including test cases, to check that ISLA's g is well defined, fitting nothing. Fix: disclose in the record; optionally rerun the
22-case magnitude check on 22 training cases (CPU only) and record the same conclusion. Not for Peter.

**R1, reporting (13).** The spec's "What to report" (06:723-731) asks for the release version and the `index.json` hash, the
effective counts, and anything the model used that tbl-allowed does not allow; 06:374-377 asks for the `mean_stable` score.
13 gives the effective counts but not "v1.2" or the hash (c8e9508e…), and no `mean_stable` score. Fix: one sentence and, with
M1's rescore, the second score.

**Hardening, not a deviation (H1, the 32 ISLA-family lanes).** The ISLA forward receives whole Mesh objects: boundaries that still
carry `solution/*`, `derived/*` and `annotation/body_id`, and an interior carrying the normalized targets and the cell volume. At
these snapshots the model reads only the declared fields and the points (model.py:1525, 1613, 1662-1666), so nothing leaks, but
the guarantee lives in the model. Fix for future snapshots: pass `interior.points` only, or a recipe `DropMeshFields` of
`solution`, `derived` and `annotation` before the forward, or a collate-time assertion. **H2, provenance nit:** the
`isla_constants` header cites LHR jobs 295774-295776; `results/aerozoo_dataset/README.md` cites 295396-295399; the values agree with
`isla_constants_fits.json`.

## Diagnostics 13 cites

gtopt (GT head re-fit, scored on `full.val` only, 386 cases), cmpdiag and f32floor (compiled vs eager, float32 vs float64 on
`full` test cases: precision checks, not scores), pdiag, preddiag and scales (post hoc analyses of test predictions, using the
true speed and the test cases' scales to explain errors; labelled post hoc in 13), qpop_az (query-population probe on every fourth
`full` test case). None of them trains a model on test data or feeds a label to a model; they are analyses of already-scored
predictions, which the spec does not restrict. They share M1's metric.

## What needs Peter

1. Whether to rescore the AeroZoo standing (48 checkpoints) with the spec's ε_U, ε_p, ε^N and `mean_stable` score (M1), and
   whether E1-AZ's "reject branch 1" verdict is re-read under it (it stands as registered; a spec-metric reading would go beside it).
2. Whether §12(f)'s form choice should be re-read on the validation sides (S1).

The main session can apply T1, I1, I2, R1, S1b, H1 and H2 (book or record text, future snapshots) without Peter.

## IDs across the evidence files, and the review

I1 = INPUTS_STATS D1; I2 = IS D2; L1 = IS D3; H1 = IS D4; H2 = IS D5; T1 = SPLITS_METRICS D1; M1 = SM D2-D5; S1 = SM C1;
S1b = REVIEW P1. REVIEW.md (AZSPEC-REVIEW, 21:47 EDT) re-checked (a) the cell volume never reaching a model, (b) ISLA's
field selects, (c) GeoTransolver's 15 merged fields, (d) train-only statistics, (e) the metric path, three split lists
recomputed from AeroZoo 96bd2b62, and snapshot md5s; all held. Its P1 (S1b) and P2 corrections are applied above.
