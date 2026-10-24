# AZ-COMPLIANT v2 pre-registration: physical inputs, each model brings its own scales (DRAFT, 2026-10-19 15:05 EDT; not submitted)

This replaces `PREREG.md` (v1). Nothing from v1 was launched. The protocol and its checks are in `DESIGN_v2.md`.
Approval by the main session, plus Peter's answers to Q1-Q3 in §9, are needed before anything is launched.

## 1. Question

The recipe gives every model only the PDE data, in physical units, and every model brings its own scales or none. Under
that rule, how do ISLA, ISLA-DTT, GeoTransolver and Transolver stand on AeroZoo, and how far does each move from its
label-era lanes?

## 2. Protocol (DESIGN_v2 §§1-3)

- **Pipeline: `aerozoo_puvolume*.yaml`.**
  - Inputs: the four boundaries with their physical `bc_*` values, ν (the baselines read it as `nu_std`, log10 ν
    standardized by per-dataset constants), the SDF and normals to `no_slip` in metres, and raw coordinates.
  - Targets: physical, with one per-dataset standardization (vector means 0).
  - No label is read, and nothing is derived per case.
- **Inputs per model.**
  - GeoTransolver: Option B (Q1), geometry = every boundary face's [xyz, condition one-hot, prescribed values], with
    `include_local_features: false`; `global_embedding` = `nu_std`.
  - Transolver: xyz plus the `no_slip` SDF and normal, and `fx` = `nu_std`. It has **no inlet drive**.
  - ISLA and ISLA-DTT: boundary meshes by condition, plus ν and ρ. Their scales are internal (DESIGN_v2 §4).
- **Score.** Predictions are de-standardized to physical units and mapped into the label-era space with the labels
  (scoring only). Then the per-case relative L2 is taken, test side, in float32.
  - This is the same quantity the label-era lanes report: the map is exact to ≤ 4e-6.
  - Checkpoints are selected on the same mapped metric on the validation part.
  - Reported beside it: the book's ε_U and ε_p.
  - Velocity and pressure are the headline fields.
- **Unchanged.** Each model's reference width, depth and rate, 250 epochs, 200k interior points, bf16 (ISLA's head
  and redimensionalization in fp32), seeds 42 and 43, and the splits: `full`, `family_non_box`, `reynolds`,
  `bank_blobs`, `bank_zero_to_cad`, `profile_crossflow`, `profile_dual_inlet`.

## 3. Implementation before launch

1. **Recipe** (one lane-writer, about 1 day):
   - the 7 `aerozoo_puvolume*` YAMLs;
   - `StandardizeGlobalLeaf` (from `v2/derive_v2.py`);
   - `MergeBoundaries` (about 20 lines, for Option B);
   - `score_space: label_reference` in `infer.py` and the mapped selection metric, with test D;
   - `geotransolver_volume_aerozoo.yaml` with Option B, and a new `transolver_volume_aerozoo.yaml`;
   - refit of the physical target and `nu_std` statistics per split, on the train part (CPU);
   - a pytest: every label perturbed or deleted leaves every model input bitwise equal (`v2/check_v2.py` B1/B2 on
     the synthetic case).
2. **ISLA scales on bc-names**, after its fix round lands (lane-writer, about 2-3 days): DESIGN_v2 §4, about 250
   lines of model code and 250 of tests, with the new `test_isla_scales.py`.
   - The AeroZoo and AirfRANS ISLA configs.
   - `reynolds` constants refit on the train parts. The scan of 1,156 cases gave log10 Re mean 5.94 and std 1.43,
     and κ ≥ 0.671 everywhere (`v2/scan_summary.md`).
3. **Smokes.** Each arm's exact training command, with `smoke_wrap.py` printing the model inputs, and the runtime
   label check (0 differing inputs). Rescoring 4 label-era lanes through the mapped scorer must reproduce their
   recorded numbers.

## 4. Lanes (`azp_<split>_<arm>_seed<42|43>`)

| arm | lanes | code | depends on |
|---|---|---|---|
| GeoTransolver (Option B) | 14 | code_a2r (model unchanged) | recipe step 1 |
| Transolver | 14, or 2 on `full` (Q2) | code_a2r (model unchanged) | recipe step 1 |
| ISLA | 14 | bc-names + scales (new snapshot `code_bcs`) | steps 1 and 2 |
| ISLA-DTT (`query_independent=true`) | 14 | the same as ISLA | steps 1 and 2 |
| GLOBE | `scratch/isla-globe-2026-10-19/PREREG.md` §8, on this protocol: coordinates in metres, a constant reference length, δ = √(ν·L0/U) with U only if GLOBE computes it itself | – | its own trigger and pre-registration |

## 5. Cost and sequencing (GPU-h per lane from the label-era partners; measured again at the smokes)

| arm | GPU-h per lane | lanes | GPU-h |
|---|---|---|---|
| GeoTransolver | ≤ 225 (Option B drops the ball queries; expect less) | 14 | about 2,800 |
| Transolver | assumed equal to GeoTransolver | 14 (or 2) | about 2,800 (or 400) |
| ISLA | about 135 | 14 | about 1,900 |
| ISLA-DTT | about 285 | 14 | about 4,000 |
| smokes, scoring, statistics | – | – | about 200 |
| **total** | | 56 (or 44) | **about 11,700 (or 9,300)** |

Sequencing:

1. W0, the recipe: day 1.
2. W1, GeoTransolver and Transolver (28 or 16 nodes, about 2 days), from day 2. The ISLA scale change is built at the
   same time (days 1-3).
3. W2, ISLA (14 nodes, about 1.3 days), from about day 4.
4. W3, ISLA-DTT (14 nodes, about 3.5 days), from day 4 if capacity allows.
5. Everything is read about day 8.

LHR is near full (723 nodes allocated, 1 idle at 12:35 EDT; I hold about 95 running jobs). W1 fits as the label-era
AZ-UNIFORM backlog drains or is cut (Q5).

## 6. Labels for the results in flight

The labels are unchanged from v1 §6. DTT-DECIDE, GEO2-AZ, E1 hn_az, AZ-UNIFORM, A2-RERUN azfull and AZ-STANDING are
all marked **pre-ruling (label inputs)**.

- The within-protocol verdicts stay provisional.
- AZ-UNIFORM and AZ-STANDING cross-architecture ratios are not a standing.
- The DTT-DECIDE rule is applied again to the v2 ISLA and ISLA-DTT lanes, and the v2 read governs.
- v1 (`DESIGN.md`, `PREREG.md`) was never launched; it is superseded.

## 7. Readouts

- **R1, the standing.** For each split and field, the seed-mean ratios GeoTransolver ÷ ISLA, Transolver ÷ ISLA and
  ISLA-DTT ÷ ISLA in the mapped metric. Summaries: the six-split means, and the count of splits above 1.
- **R2, the move from the label era.** C = v2 ÷ label-era partner, by split, field and arm.
  - GeoTransolver against `azu_*_gt`: the same model code, so this is clean.
  - ISLA against `dtd_*_isla` (or `geo2_*`) and ISLA-DTT against `dtd_*_dtt`: these are confounded with the
    bc-names, scales and full-mesh-gauge code change. They are read as standing moves, not as a cost.
- **R3, ISLA versus ISLA-DTT.** The DTT-DECIDE v2 rule, verbatim.
- **R4, by inlet speed.** The mapped error per tercile of U_ref (a label used for scoring only), to show the effect of
  the loss weighting in DESIGN_v2 §3.
- **R5, similarity on the trained models.** On 20 test cases, the prediction for the case scaled by (s, c) =
  (3.7, 0.29), against the scaled prediction.

## 8. Predictions (seed means, held-out splits unless noted; the probabilities are mine)

| id | prediction | p |
|---|---|---|
| P1 | GeoTransolver (B): C_GT on velocity ≥ 1.5 on at least 5 of 6. It sees raw coordinates spanning 0.1-125 m through linear projections, without its local features, and is trained mostly on the fastest cases. | 0.70 |
| P2 | GeoTransolver (B): C_GT on pressure ≥ 2 on at least 4 of 6 | 0.60 |
| P3 | Under Option A instead (no drive), GeoTransolver's mapped velocity relative L2 is ≥ 0.5 on every held-out split | 0.85 |
| P4 | Transolver: mapped velocity relative L2 ≥ 0.5 on all 7 splits, `full` included (the inlet speed spans 200× and is unobservable to it) | 0.85 |
| P5 | ISLA leads GeoTransolver (GT ÷ ISLA > 1) on velocity on at least 5 of 6, and on pressure on at least 5 of 6 | 0.80, 0.75 |
| P6 | Mapped error in the lowest U_ref tercile is ≥ 3× that in the highest, for GeoTransolver; ISLA's ratio is smaller than GeoTransolver's | 0.75, 0.70 |
| P7 | C_ISLA on velocity lies in [0.90, 1.25] on at least 4 of 6 (with Q3 = per-case loss: [0.90, 1.10]) | 0.55 (0.60) |
| P8 | The DTT-DECIDE rule gives the same verdict as in the label era | 0.70 |
| P9 | R5: ISLA's scaled prediction matches the scaled case to ≤ 1e-3 relative; GeoTransolver's changes by ≥ 10% | 0.90, 0.90 |
| P10 | Mechanics: 0 differing model inputs in every lane's runtime label check; the mapped scorer reproduces 4 old lanes within 1e-5 | 0.97, 0.95 |

## 9. Decisions needed

- **Q1 (Peter): GeoTransolver's channel for the boundary conditions.**
  - Option B, recommended: geometry = every boundary face with its condition type and prescribed values. This is
    GeoTransolver's documented geometry-features channel. It requires `include_local_features: false`, as in the
    published surface configs, and with radii in metres those features are uninformative on AeroZoo anyway
    (DESIGN_v2 §5).
  - Option A: keep the reference config with ball-query local features. GeoTransolver then receives **no inlet
    drive**.
- **Q2 (Peter): Transolver cannot receive the inlet drive through its own mechanism.**
  - Options: run all 14 lanes as a documented floor; run 2 lanes on `full`; or leave it out of the AeroZoo standing.
  - Recommended: 2 lanes on `full`, about 400 GPU-h.
- **Q3 (Peter): the loss under one per-dataset standardization of physical targets.**
  - The 3 largest of 22 cases carry about 100% of the pressure loss.
  - Recommended: a per-case relative-L2 loss. It is computed from the case's own targets, so it is never an input; it
    is the same for every arm; and it matches the scored metric.
  - Alternative: accept the weighting as part of the rule.
- **Q4 (main session): read log10 ν, standardized by per-dataset constants, as "per-dataset constant
  standardization".** Recommended: yes. Raw ν spans 5 decades, and a linear standardization makes it constant.
- **Q5 (main session): cut the label-era AZ-UNIFORM rows that feed no registered read, to free LHR.** Recommended:
  yes.

## 10. Approval record (supervisor, 2026-10-19 15:05 EDT; added before any v2 lane was submitted)

Approved for implementation (section 3) under Peter's rulings of 2026-10-15 (AeroZoo inputs) and 2026-10-19 (no per-sample
reference length; "If a model wants normalized coordinates, it needs to bring it to the table. GeoTransolver does not.
ISLA does."). Decisions on section 9:
- **Q1: GeoTransolver Option B.** Every boundary face becomes a token carrying its position, condition type and prescribed
  values, through GeoTransolver's own documented geometry-features channel. global_embedding carries nu only, and
  ball-query local features are off. Option A leaves it blind to the inlet velocity, which is not a fair comparison.
  This is a choice of inputs and configuration; the released code stays untouched. Disclosed in the book.
- **Q2: Transolver on `full` only, 2 lanes, as a documented floor.** Its published inputs have no channel for boundary
  values, so it cannot see the inlet velocity at all.
- **Q3: a per-case relative loss for every arm.** Each case's loss is normalized by its own targets' norm, per field.
  This matches the relative-L2 scoring metric and removes the domination by the 3 largest cases. It is implemented in
  recipe step 1 as one training option that every arm's config sets the same way, with a unit test.
- **Q4: yes.** log10 nu standardized by fixed per-dataset constants is constant standardization of a PDE input.
- **Q5: no cuts.** The label-era AeroZoo lanes still answer within-protocol questions (BUDGET-2X, TARGETS-VP). Their
  nice classes (2000-6000) already let nice-0 v2 lanes start first. I revisit this if the v2 lanes wait more than a day.
Sequencing: recipe step 1 (lane-writer, worktree ~/gh/physicsnemo-isla-az-v2, branch psharpe/isla-az-v2 from
psharpe/isla-bc-names f2d52c153), then the GeoTransolver and Transolver smokes and lanes. ISLA scales (step 2) follow
the GEO-PREMERGE fixes on psharpe/isla-bc-names, then the ISLA and ISLA-DTT lanes.

## 11. Amendment before submission (2026-10-19, 18:20 EDT; research-lane AZP-LAUNCH, approved by the supervisor before any baseline lane was submitted)

Peter ruled that the corrected protocol replaces the label-input one in place (recipe caac441d7: the physical-input
files carry the original `aerozoo_uvolume*` names, 41 label-era configs are deleted, and the label-reference scorer is
removed). The baseline lanes are named `azc_<split>_gt_seed<42|43>` and `azc_full_t_seed<42|43>`. Evidence for every
number below: `scratch/isla-azp-launch-2026-10-19/PRELAUNCH.md`.

- **(a) Precision: float32 for GeoTransolver and Transolver** (no autocast; LHR torch 2.12.0+cu130 reports
  matmul.allow_tf32=False and float32_matmul_precision=highest; cudnn.allow_tf32=True is the default, and neither model
  has convolutions). Reason, A/B job 288363 (GT Option B, `full`, 4 GPUs, one epoch, same seed and data order):
  - The head's bf16 − fp32 output difference on the same batch is 2.1-2.8e-3 target std in velocity (about 0.025 m/s)
    and 1.9e-4 std in pressure (about 0.22 Pa), the same in every U_ref tercile. That is about 10 % of a 0.25 m/s case's
    velocity and about 5× the slowest cases' dynamic pressure (0.04 Pa): bf16 sets an error floor on the slow third.
  - Cost: wall step 0.79 s (fp32) against 0.70 s median / 0.86 s mean (bf16); both are data-bound. Peak reserved memory
    111 GB against 87 GB, within the GB300's 277 GB.
  - After one epoch neither arm resolves the slow cases yet (label-space velocity L2 by tercile low/mid/high: bf16
    1.22/1.00/1.00, fp32 1.68/1.04/1.00), so the A/B does not show the final effect, only the floor.
  - **Disclosed asymmetry:** ISLA and ISLA-DTT keep bf16, with their head and redimensionalization in fp32.
- **(b) Dropped: every readout and prediction that compares with the label-era lanes.** Peter's instruction deleted
  those results, so they cannot be evaluated. Dropped: R2 (C = v2 ÷ label-era partner), P1 and P2 (C_GT), P7 (C_ISLA),
  P8 (the DTT-DECIDE verdict compared with the label era's), and the second half of P10 (the mapped scorer reproduces 4
  old lanes). That half was evaluated before the scorer was retired and passed: ≤ 3.0e-7 per case, ≤ 5.4e-8 on the
  summary, over 1,698 cases (`scratch/isla-azp-launch-2026-10-19/rescore/REPORT.md`). Section 6's labels for the
  in-flight results are moot for the same reason. R1, R3 (the DTT-DECIDE rule applied to the v2 lanes alone), R4, R5,
  P4-P6, P9 and the first half of P10 stand. P3 is not evaluated, because Option A is not run.
- **(c) Scoring:** the recipe's default relative L2 per field and case, on the physical targets (scale-only, mean-0
  standardization, so it is the physical relative L2), on the test side, at the final checkpoint (epoch 250), averaged
  over cases. Checkpoint selection on a validation metric (section 2) is replaced by the final checkpoint; validation
  metrics are for monitoring only. The label-reference map is retired. Note: for pressure, nut, tke and omega this is a
  different quantity from the label-era numbers (on one checkpoint and 386 val cases: pressure 92.3 physical against 9.12
  in the label space; velocity 1.0713 against 1.0711).
- **(d) Transolver floor:** `transolver_volume_aerozoo` as written (width 256, 12 layers, 256 slices), lr 1e-3 (every
  REF-80 Transolver row, `kernel/char/r80_lanes.tsv` md5 a9f5c6f9, uses 1e-3 at widths 256-1024 and depths 12-16;
  CHAR cl_vol_t too), 2 lanes on `full`. The REF-80 surface reference was 512 × 16 at 100k cells; it is not used here.
- **(e) Launcher deviation:** the lanes run under the a2r-family `lanes.sbatch` (kernel/geo2's, md5 8192a90b) with
  `site.env`, changed in one line so that the precision comes from `LANE_PRECISION` (float32 in site.env). Its train
  command matches the AZ-UNIFORM launcher's (`kernel/aerozoo/lanes.sbatch`, under which the label-era GT lanes ran)
  apart from the paths (output and checkpoint directories) and the precision.

## 12. Amendment before submission (2026-10-20; research-lane REF-090, written before any ISLA or ISLA-DTT lane was submitted; approved by the supervisor at 02:58 EDT with the edits in (f), before any of these lanes was submitted)

The ISLA and ISLA-DTT lanes of §4, 14 each. Launch record and evidence: `scratch/isla-ref090-2026-10-20/` (PREREG.md §4,
smokes, JOBS.md).

- **(a) Code: ISLA 0.9.0**, psharpe/isla 92d9d7800 (the bc-names + scales snapshot §4 calls `code_bcs`): the exact
  full-mesh gauge (L is the full-resolution boundary's RMS pairwise distance, independent of the token draw), boundary
  conditions by name, exact query geometry with the CONT encoding, ISLA's own scales, upstream/main merged. Snapshots
  `$T/code_r090` (+ tensordict 0.15.2) and `$T/recipe_r090`; launcher `$T/kernel/azi/` (the azp launcher, LANE_PRECISION
  bfloat16). Lanes `azc_<split>_isla_seed<42|43>` (rows 0-13) and `azc_<split>_dtt_seed<42|43>` (rows 14-27), chains
  `isla.azc.<slug>`.
- **(b) The fitted constants.** Each split's `isla_constants` in its dataset file (reynolds mean and std of log10(U L / nu),
  output_scale per output, input_scale per boundary input), fitted on that split's train part with ISLA._gauge and
  ISLA._case_scales, LHR cpu-short jobs 295774-295776 (code 6c3aac5a8); `full`: reynolds 5.909 / 1.415, output_scale
  velocity 0.5749, pressure 0.09715, nut 4.651e-4, tke 9.358e-3, omega 614.4. The model config leaves them null; train.py
  and infer.py inject them. Nothing is derived per case from labels.
- **(c) Precision: ISLA and ISLA-DTT train in bf16, with the head and the redimensionalization in fp32** (the disclosed
  asymmetry of §11a; the baselines train in float32). Scoring is float32 for every arm.
- **(d) ISLA-DTT is `query_independent=true` alone.** The DTT-DECIDE knobs `kernel_read`, `kernel_read_radius` and
  `kernel_read_theta` do not exist at 0.9.0 (its tree read is built in). Rates: 3.75e-4 for both, 250 epochs, 200,000
  cells, the extra of the azc baseline rows.
- **(e) Scoring as §11(c):** test side, the final checkpoint (epoch 250), the recipe's default physical relative L2 per
  field and case, averaged over cases, float32, 200,000 interior points: `cd $T/kernel/azi && sbatch --array=0-6
  eval_test.sbatch` into `$T/p200k_fp32_test_azc/azc_<split>_{isla,dtt}_seed4x/`, the baselines' tree.
- **(f) The ISLA-against-ISLA-DTT rule (R3).** §7 registers it by reference only ("the DTT-DECIDE v2 rule, verbatim"), and
  §11(b) keeps R3 on the v2 lanes alone. That rule (`scratch/isla-dtt-decide-2026-10-17/PREREG.md`, D1) is: R = ISLA-DTT ÷
  ISLA, seed means, velocity and pressure, over the six held-out splits (`full` reported, not in the rule); ISLA-DTT becomes
  the only interior form if R ≤ 1.05 on both fields on every split; ISLA stays the default if the arithmetic mean of R over
  the six splits is ≥ 1.10 on velocity or on pressure; **otherwise, Peter decides**; plus the reference-rate event guard. So
  a rule is registered, but its middle band defers to Peter rather than applying his close-call rule.

  **Approved (supervisor, 2026-10-20 02:58 EDT, with edits i and ii; this replaces D1's "otherwise, Peter decides"):**
  Peter's close-call rule. When the two are close, prefer the principled choice, here ISLA-DTT's exact query independence:
  - **ISLA-DTT clearly:** R ≤ 1.05 on velocity and pressure on every held-out split → ISLA-DTT becomes the reference
    (default) interior form, with `query_independent=true` as the default. Deleting the interacting form is then proposed
    to Peter; the rule does not do it (edit i).
  - **ISLA clearly:** the arithmetic mean of R over the six held-out splits ≥ 1.10 on velocity or on pressure → ISLA stays
    the default and ISLA-DTT stays a switch.
  - **Close call (everything else) → ISLA-DTT, on principle** (its predictions do not depend on which other points are
    queried): it becomes the reference (default) interior form, with `query_independent=true` as the default; deleting the
    interacting form is proposed to Peter, not done by the rule (edit i). The table goes to Peter as a close-call
    application, which he can overrule.
  - **Cost beside the table (edit ii):** each arm's step time and peak memory, since cost is part of what Peter weighs if he
    overrules. From the smokes (LHR 297255, 4 GB300, bf16, 200,000 cells): ISLA mean step 0.71-0.87 s (median 0.64-0.77),
    peak 126-127 GiB; ISLA-DTT mean 1.21-1.38 s (median 1.21-1.37), peak 163-171 GiB; ISLA-DTT ÷ ISLA step time 1.5-2.0 per
    split (`full` 1.55 mean, 1.73 median), GPU-h per lane 1.5-1.9×. Re-measured from the lanes' own logs when the rule is read.
  - Unchanged: seed means decide; a split whose seeds straddle a bar is marked "within seed spread"; the geometric mean is
    reported beside the arithmetic one; the event guard (a lane whose validation loss passes twice its running minimum
    stops the read).
  - My probabilities (0.9.0, compliant inputs): ISLA-DTT clearly 0.30, ISLA clearly 0.20, close call 0.50.
- **(g) Cost, from the REF-090 smokes** (LHR 297255, the lanes' exact command through kernel/azi's launcher, 50 steps, 4
  GB300; `scratch/isla-ref090-2026-10-20/PREREG.md` §§8a, 9a): mean step 0.71-0.87 s (ISLA, peak 127 GiB) and 1.21-1.38 s
  (ISLA-DTT, peak 171 GiB); GPU-h per lane 131-181 (ISLA) and 220-306 (ISLA-DTT); about 2,140 + 3,680 = 5,820 GPU-h for the
  28 lanes, against the earlier estimate of about 4,760 (110 and 230 per lane). Longest lane: ISLA-DTT bank_zero_to_cad,
  about 76 h of wall time.

## 13. Amendment before results (2026-10-20 21:42 EDT, supervisor; no AZ-ISLA or ISLA-DTT test-side score exists yet, and the lanes are still training)

**Why.** §12(f) was approved at 02:58 EDT on 2026-10-20, when the A2-era AirfRANS evidence favoured ISLA-DTT (E3b: ISLA-DTT
÷ interacting ISLA 0.742 / 0.887 / 0.988). Since then, REF-090 scored AirfRANS at 0.9.0 with seeds separated. ISLA-DTT ÷ ISLA
is now 1.163 / 1.147 / 1.396 (p / v / nut), and §12(f)'s own bands applied to AirfRANS would read "ISLA clearly". ISLA-DTT on
DrivAerML Volume was 1.12 / 1.11 / 1.12 times ISLA before A2 and has not been run at 0.9.0. The reversal is not yet
attributed (ATTRIB-090, R5). Peter's ruling "AeroZoo is the real way to decide between ISLA and ISLA-DTT" was given when
the two datasets agreed. A program-wide switch of the default form on AeroZoo alone, against a clear opposite result on a
second dataset, is a decision Peter should take explicitly. This amendment was raised by the reviewer session 01a15c0d
(scratch/isla-review-2026-10-21/REVIEW.md, item 2). It is written with the AirfRANS 0.9.0 result known and before any
AeroZoo ISLA or ISLA-DTT result.

**What changes.**
- (a) **The bands are computed exactly as registered** in §12(f): "ISLA-DTT clearly", "ISLA clearly" or "close call", with
  the same R, splits, seed means, straddle marking and event guard. The verdict label is unchanged and stands as written.
- (b) **Consequence.** If the verdict is "ISLA-DTT clearly" or "close call", the switch of the program-wide default interior
  form goes to Peter as an explicit question (the `Needs you:` line), not as an overrulable table. Until he answers, interacting
  ISLA stays the default, and both forms are reported as AeroZoo rows. The question carries one cross-dataset table:
  - AeroZoo R per split (velocity and pressure, seed means and per seed), with the arithmetic and geometric means;
  - AirfRANS R at 0.9.0 (1.163 / 1.147 / 1.396);
  - the lug ISLA-DTT against ISLA at 0.9.0 (REF-090);
  - the query-population probe at 0.9.0 for both forms (QUERY-POP-090), if it is read by then;
  - the pre-A2 DrivAerML Volume ratio (1.12 / 1.11 / 1.12), labelled as a different code version;
  - ATTRIB-090's R5 (where the AirfRANS order reversed), if read by then;
  - cost: step time, peak memory and GPU-h per lane, from the lanes' own logs.
  It also states the principled case on both sides. For ISLA-DTT: exact query independence. Against it: the canonical
  frame's axis choice on near-symmetric bodies, its constants, and θ. θ's error vanishes as θ → 0, which the knob rule credits.
  If the verdict is "ISLA clearly", ISLA stays the default and ISLA-DTT stays a switch, as registered.
- (c) **Closeness from the measured noise is reported beside the verdict.** Per split, show whether the mean R lies inside
  that split's two-seed spread. This is descriptive and does not change the band.
- (d) Deleting the interacting form remains a separate proposal to Peter, as in §12(f) edit i.
- (e) Note added at 21:52 EDT, still before any result: the reviewer's numbers audit found that the A2-era ISLA-DTT lead on
  AirfRANS held in E3b's code (code_anc) only. The earlier A2-era afu_dtt lanes (code_dtt16) read 0.01996 / 0.0119 /
  0.02706, which is worse than A2 ISLA (0.01799 / 0.00946 / 0.02230). The cross-dataset table in (b) includes this row.
- (f) Note added at 02:27 EDT on 2026-10-21, still before any AZ-ISLA or ISLA-DTT test-side result: the numbers in (e) are
  compiled-path scores, made under the per-block compile fault that COMPILE-AF later found (scratch/isla-compile-af-2026-10-21/;
  notebook #sec-nb-compiled-path-defect). Eager re-scores of the same checkpoints: the code_dtt16 afu_dtt lanes read 0.01752 /
  0.01150 / 0.02560, so ÷ A2 ISLA (afu_a2s, eager) 0.971 / 1.210 / 1.146. The AirfRANS 0.9.0 row in (b) reads 1.181 / 1.154 /
  1.433 eager. The table in (b) shows the eager numbers with the compiled ones beside them, and the AirfRANS row retrained under
  the fix (r090fix) when it exists.
