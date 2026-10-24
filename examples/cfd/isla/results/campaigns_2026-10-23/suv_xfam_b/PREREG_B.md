# SUV-XFAM part B, stage 1: pre-registration (written 2026-10-22 14:54 UTC, before any fastback-trained run exists)

Lane SUV-XFAM-B (continues part A: PREREG.md c0d0caef, RESULTS.md, JOBS.md in this folder). Rules:
scratch/isla-main-2026-10-14/LANE_RULES.md. Job prefix `isla.suvxb.`, run prefix `sxb_`, launcher `$T/kernel/suvxb/`
(LAUNCH_MD5S.txt there), scores `$T/evals_suvxb/`. Stage 1 is approved (about 140 GPU-h); stage 2 (GeoTransolver) needs
separate approval and is only proposed under the trigger in section 5.

**Already known when this was written.** All of part A (RESULTS.md, reduce_out.json): every estate-trained score on
fastback test, fastback val and estate test, and the standing estate-val scores. Values used below (seed 42 · 43):

| arm, field | E(est→fb), fastback test | E(est→est), estate test | E(est→est), estate val |
|---|---|---|---|
| ISLA pressure | 0.1100 · 0.1135 | 0.0445 · 0.0430 | 0.0456 · 0.0434 |
| ISLA wss | 0.1182 · 0.1154 | 0.0659 · 0.0655 | 0.0671 · 0.0660 |
| T pressure | 0.1067 · 0.1060 | 0.0418 · 0.0414 | 0.0419 · 0.0420 |
| T wss | 0.1250 · 0.1233 | 0.0631 · 0.0628 | 0.0639 · 0.0640 |

Part A's penalty P = E(est→fb) ÷ E(est→est): pressure ISLA 2.557, T 2.557; wss ISLA 1.778, T 1.972
(P_T ÷ P_ISLA = 1.110, D = 0.104 at 11 SE). No fastback-trained model exists; nobody writing this has seen any number
from one. The smoke (section 6) will show 2-epoch validation losses of these arms on fastback val; an amendment after
the smoke must say so.

## 1. Arms (trained on the fastback train split, 796 cases; validation on fastback val, 99)

Each lane is its estate reference lane with only the dataset changed (`shift_suv_surface` → `shift_suv_surface_fastback`,
the part-A file: recipe_r090 b09692b7, recipe_v8 c671222b). Rows copied by `kernel/suvxb/src/build.sh` from the
reference rows; lr, precision, epochs, cells, extras, forward_kwargs, code and recipe snapshots, launcher and venv
unchanged. No snapshot file is added.

| arm | runs (seed 42, 43) | reference lanes | code / recipe | model, extra | lr | cells (train) | precision |
|---|---|---|---|---|---|---|---|
| ISLA 0.9.0 | sxb_isla_seed42/43 | r090_suv_isla_seed42/43 (kernel/r090 rows 0-1) | code_r090 / recipe_r090 | isla_surface, `model.hidden=512 model.n_slices=512 model.compile_blocks=true dataloader.prefetch_factor=12` | 3.75e-4 | 200000 | bf16 (site.env) |
| Transolver (fair drive) | sxb_t_seed42/43 | clud_suv_t_seed42/43 (kernel/cl_ud rows 6-7) | code_v20 / recipe_v8 | transolver_surface, `model.n_hidden=512 model.n_layers=16 forward_kwargs.fx.source=global_data.U_inf_dir` | 1e-3 | 100000 | bf16 |

500 epochs, batch 1 on 4 GPUs, StepLR (100, 0.1), augment off, compile off, final checkpoint. Launcher changes, none
in the training command: job names `isla.suvxb.<arm>-seed4x`, one `#SBATCH --exclude` line (the three known bad
nodes), and the Transolver copy's own SELF / LANES paths.

**Normalization (a fact that corrects the brief's wording).** The recipe fits no statistic on any split for SHIFT-SUV:
`norm_stats.pt` holds the dataset file's declared constants (wss mean 0, std 0.00313, the DrivAerML constant carried
by both families' files; pressure is Cp from each case's own q_inf), and ISLA takes no `isla_constants`. The estate
reference runs' norm_stats.pt are exactly these constants. So "norm stats from fastback train" is checked as "norm_stats.pt
equals the declared constants": no statistic of either family reaches any arm in either direction.

## 2. Scoring

Every fastback-trained checkpoint on **fastback test** (99, the lead for G and H), **estate test** (100, the lead for R),
**fastback val** (99, replicate for G and H) and **estate val** (99, replicate for R). Native protocol, `precision=float32`,
`sampling_resolution=200000` for both arms, final checkpoint (epoch 500), part A's scorer (`kernel/suvx/score.sbatch`
copied with only the job name and roots changed: `kernel/suvxb/score/`, 16 rows). Fields `pressure_l2` and `wss_l2`
from `infer_summary`. The estate-trained values are part A's (estate val: the standing scorer, as part A's P10 used).

**Validity, before any number is read (a failure stops the lane and is reported):**
- each of the 4 lanes passes `smoke_check.py`: resolved config differs from its reference run's only in `dataset` and
  `run_id`; Transolver shows `forward_kwargs.fx.source: global_data.U_inf_dir`; logged split sizes 199 train / 25 val
  per rank, dataset_size 996 [corrected to 995 by the amendment in section 8]; all logged losses and metrics finite; norm_stats.pt = the declared constants;
  "Training completed" after 500 epochs;
- `splitcheck.py` (cpu-short): each lane's training loaders read exactly the fastback manifest's train entries (796)
  and val entries (99) under the fastback root, disjoint; the estate reference loaders read estate train / val;
- scored-set identity as in part A: (a) loader digests (scoredset_check.py unchanged) for recipe_r090 + isla_surface
  and recipe_v8 + transolver_surface equal part A's per split (fastback test 4241f134, fastback val 1bb14edd,
  estate test 24460c3d, estate val 983a2c64); (b) per-case reference force coefficients of the 16 new scorings agree
  with part A's (and, on estate val, the standing scorer's) to ≤ 1e-4 relative, same case ids in the same order;
- every scoring: `"epoch": 500`, `num_samples` 99 / 100 / 99 / 99, `precision: float32` in its log, one START and no
  FAILED, the requested split and dataset file.

## 3. Quantities

For arm a ∈ {ISLA, T}, field f ∈ {pressure, wss}, training family x and scoring set y, L_a(x→y) = mean over seeds of
ln E_a,s(x→y) (geometric seed mean, natural logs), with SE_a(x→y) = |ln E_a,42 − ln E_a,43| / 2.

- In-family fastback error: E_a(fb→fb) = exp L_a(fb→fb), on fastback test.
- Forward transfer gap, fastback test: ln G_a = L_a(est→fb) − L_a(fb→fb).
- Reverse transfer gap, estate test: ln R_a = L_a(fb→est) − L_a(est→est).
- Intrinsic hardness: ln H_a = L_a(fb→fb, fastback test) − L_a(est→est, estate test).
- Identity: part A's ln P_a = ln G_a + ln H_a exactly (P_a was the geometric seed mean of per-checkpoint ratios, which
  equals the ratio of geometric seed means).

Comparisons, D_X = ln(X_T ÷ X_ISLA) for X ∈ {G, R, H}. Each ln X_a involves four runs (two seeds in each training
family); seeds of different runs are treated as independent (a shared seed fixes only the initial weights and the
data order, and treating them as unpaired can only overstate the SE):

    SE(D_G) = sqrt( SE_T(est→fb)² + SE_T(fb→fb)² + SE_I(est→fb)² + SE_I(fb→fb)² )
    SE(D_R) = sqrt( SE_T(fb→est)² + SE_T(est→est)² + SE_I(fb→est)² + SE_I(est→est)² )
    SE(D_H) = sqrt( SE_T(fb→fb)² + SE_T(est→est)² + SE_I(fb→fb)² + SE_I(est→est)² )

Each SE term rests on one degree of freedom, so 2 SE is a rough screen, not a 95-percent test (as in part A).

## 4. Rules (2 arms × 2 fields; headline field wss, the field that triggered B; pressure beside it)

**B1, in-family fastback (descriptive lead).** E_a(fb→fb) for both arms, both fields, with seeds; the ratio
E_T(fb→fb) ÷ E_ISLA(fb→fb) under part A's R1 rule (b1 = max(0.03, 2σ), σ = max over the two arms of
|ln(E_42 ÷ E_43)| ÷ √2; leads / trails need the seed ranges to separate; otherwise level).

**B2, the forward gap D_G (the decision cell is wss).** ISLA **transfers better** if D_G > 2·SE(D_G) and D_G ≥ ln 1.03;
**worse** if D_G < −2·SE(D_G) and D_G ≤ −ln 1.03; otherwise **no difference**.

**B3, the reverse gap D_R.** Same rule as B2, on estate test.

**B4, the hardness ratio D_H.** "Fastback relatively **harder for Transolver**" if D_H > 2·SE(D_H) and D_H ≥ ln 1.03;
"harder for ISLA" symmetrically; otherwise "no difference". Reported with the share D_G ÷ D_P and D_H ÷ D_P of part A's
D_P (they sum to 1 by the identity).

Replicates (reported beside, do not change a verdict): G and H on fastback val (part A's E(est→fb val) and the
standing E(est→est val)); R on estate val.

**Reading of the wss cells (registered now):**
- B2 **better**: part A's wall-shear advantage is (at least in part) a **transfer advantage**: ISLA's estate-trained
  model is closer to the in-family fastback reference than Transolver's is. If B3 is also better: "in both
  directions"; otherwise "estate → fastback only".
- B2 **no difference** and B4 **harder for T**: fastback wall shear is relatively harder for Transolver in family; the
  book says only that ISLA **degrades less**, with no transfer claim.
- B2 **no difference** and B4 **no difference**: the split of part A's ratio is unresolved at two seeds; the book says
  only "degrades less".
- B2 **worse**: ISLA's forward gap is larger, and part A's advantage is in-family hardness; the book says only
  "degrades less", and this is flagged to the supervisor.
Pressure cells are read the same way, beside; part A found no pressure difference to explain.

## 5. Stage-2 trigger

Stage 2 (GeoTransolver, seeds 42-43 on fastback train, copies of clud_suv_gt_seed42/43, about 106 GPU-h) is
**proposed** to the supervisor only if B2 on wss (fastback test) reads "ISLA transfers better". Otherwise it is not
proposed. The lane never launches it without approval.

## 6. Execution and cost

- After approval: one cpu-short job (`isla.suvxb.setcheck`: splitcheck.py for both lanes and the estate anchors,
  scoredset_check.py for 2 loaders × 4 splits) and one short-QoS 4-GPU smoke (`isla.suvxb.smoke`: each arm's lane
  launcher, 2 epochs, run ids sxb_smoke_<arm>_seed42, then smoke_check.py). User-wide short QoS kept ≤ 12.
- Then the 4 lanes on normal QoS, 3:59 singleton chain links, staggered (ISLA, then Transolver).
- Cost (scout's measured walls; fastback 796 vs estate 794 train cases): ISLA 2 × 29, Transolver 2 × 41 = 140 GPU-h;
  smoke ≈ 2 GPU-h; 16 scorings ≈ 1.5 GPU-h. About 144 GPU-h.
- A lane that diverges, or fails twice the same way, is stopped and reported; no setting is changed. If a lane cannot
  finish, the rules that need it are reported "incomplete as registered".
- Outputs: RESULTS_B.md, NOTEBOOK_DRAFT_SUVXB.md, JOBS.md (section B), reduce_b.py → reduce_b_out.json.

## 7. Predictions (before any fastback-trained number)

| # | prediction | p |
|---|---|--:|
| Q1 | B2 wss reads "ISLA transfers better" (stage-2 trigger fires) | 0.65 |
| Q2 | B2 pressure reads "no difference" | 0.65 |
| Q3 | B4 wss reads "no difference" | 0.55 |
| Q4 | B3 wss reads "ISLA transfers better" | 0.45 |
| Q5 | B3 pressure reads "no difference" | 0.60 |
| Q6 | B1: Transolver leads ISLA on in-family fastback pressure (as on estate test, ratio 0.951) | 0.65 |
| Q7 | B1: E_T(fb→fb) ÷ E_ISLA(fb→fb) < 1 on wss (as on estate test, 0.958) | 0.60 |
| Q8 | G_a > 1.5 for both arms on both fields (most of part A's penalty is the transfer gap) | 0.80 |
| Q9 | 0.8 < H_a < 1.25 for both arms on both fields | 0.60 |
| Q10 | R_a > 1.5 for both arms on both fields | 0.70 |
| Q11 | The replicates give the same B2 and B4 verdicts as the leads, on both fields | 0.80 |

## 8. Approval and amendment before submission

- md5 as sent: 6e77d876d1ad7d9b51516d456baa6a2a (14:55 UTC). **Approved by the supervisor 2026-10-22 15:00 UTC, before
  any fastback-trained run exists**, as written, with notes that change no rule (keep code_r090 with
  compile_blocks=true for ISLA, as in its estate reference, so both families train alike).
- Amendment, 2026-10-22 15:01 UTC, before any job of this lane was submitted (on the supervisor's arithmetic note):
  the expected `dataset_size` is **995**, not 996. It counts the reader's paths under the root, not manifest cases:
  fastback has 995 paths for 796 + 99 + 99 = 994 manifest cases, and estate has 996 for 993 (the estate reference logs
  996). The extra directories are in no manifest split, so no loader reads them. Nothing else changes.
- Amendment 2, 2026-10-23 07:04 UTC, approved by the supervisor, **before results**: neither the lane nor the supervisor
  had read any error metric of an sxb run. md5 before a1104d7c7cf28d78d3b426e32cd0ae33. reduce_b_out.json (reduce_b.py
  25a1a87c, as registered) already holds the reading; only its `identity`, `checks` and `checks_all_pass` fields were
  opened.
  - Registered failure, reported as is: check (b) fails on est_val, worst relative difference 1.282e-4 > 1e-4. It is
    one case, 00888_run_00892_domain, component CMR: reference -4.360452e-06, Transolver -4.361011e-06, absolute
    difference 5.59e-10. It is identical in all four Transolver est_val scorings (fastback-trained s42 / s43 and the
    standing evals_ud2 s42 / s43), so 4 of 4,752 rows exceed 1e-4, all of them this case. The other splits pass as
    registered (fb_test 1.1e-5, fb_val 4.2e-5, est_test 1.2e-5). The largest absolute difference over all rows,
    5.96e-8, is one float32 ulp on CD and CL values of 0.45-0.84.
  - Cause: a component that crosses zero, plus float32 differences between recipe_v8's and recipe_r090's force
    integration. The failing difference is 1.8e-7 of CMR's median magnitude (3.17e-3). Loader check (a) gives one
    digest per split for both loaders (est_val 983a2c64), so the scored points and targets are identical.
  - Amended check (b), all four splits: pass when the case ids and order match and every
    |diff| <= 1e-4 * max(|ref|, m_c), where m_c is the median |ref| of component c over the split's cases in the
    reference scoring (part A's ISLA seed 42 on that split; the standing evals_r090 scoring on est_val). The 1e-4 stays;
    only the fixed 1e-6 denominator floor becomes the component's own scale. reduce_b.py stays as registered; the
    amended check is computed by identity_b.py beside it, and (b) is reported both ways.

## 9. Stage 2: GeoTransolver (written 2026-10-23 07:12 UTC, before any GeoTransolver smoke, lane or score)

Approved by the supervisor 2026-10-23 (~07:10 UTC) after stage 1 read "ISLA transfers better" on D_G wss. Cap 125 GPU-h
for the whole stage (smoke, training, scoring); if the projection passes it, the lane stops and asks.

**Already known.** All of stage 1 (RESULTS_B.md) and part A, including GeoTransolver's estate-trained scores: fastback
test pressure 0.1234 · 0.1076, wss 0.1299 · 0.1235; fastback val pressure 0.1232 · 0.1068, wss 0.1311 · 0.1241; estate
test pressure 0.0450 · 0.0450, wss 0.0655 · 0.0656; estate val pressure 0.0458 · 0.0452, wss 0.0667 · 0.0663 (seed
42 · 43). Part A's P_GT ÷ P_ISLA: pressure 1.002, wss 1.087 (3.0 SE). No fastback-trained GeoTransolver exists.

- **Arms.** sxb_gt_seed42/43: kernel/cl_ud rows 4-5 (clud_suv_gt_seed42/43) copied by the same script with idx, group,
  run id and dataset (-> shift_suv_surface_fastback) the only row changes: geotransolver_surface,
  `forward_kwargs.global_embedding=global_data.U_inf_dir`, lr 1e-3 (the launcher's `training.optimizer.lr`, as in the
  reference lane; resolved 0.001), 200,000 cells, code_v20 / recipe_v8, bf16, 500 epochs, seeds 42 / 43. Launcher
  $T/kernel/suvxb/gt/: kernel/cl_ud/lanes.sbatch with its own SELF / LANES paths and an `#SBATCH --exclude` of the four
  bad nodes (nvl72d015-T12, nvl72d058-T08, nvl72d052-T01, nvl72d057-T13). Job names isla.suvxb.gt-seed4x.
- **Lane check** (smoke and final runs): the saved config differs from clud_suv_gt_seed4x's only in `dataset` and
  `run_id` (smoke: also `training.num_epochs`) and shows `forward_kwargs.global_embedding: global_data.U_inf_dir`;
  199 / 25 per rank, dataset_size 995; losses finite; norm_stats.pt = the declared constants; Training completed.
  splitcheck.py is not rerun: the GeoTransolver loader is recipe_v8's, whose fastback train / val selection stage 1
  checked (the split does not depend on the model).
- **Scoring.** As section 2: fastback test, estate test, fastback val, estate val; native, float32, 200,000 cells,
  final checkpoint; 8 new rows (16-23) in kernel/suvxb/score/score_rows.tsv with part A's GeoTransolver model, extras,
  snapshots and venv. Same per-scoring checks. Identity (b) **as amended in amendment 2** is the check (reference: part
  A's ISLA seed 42 per split; the standing evals_r090 scoring on est_val), with the registered statistic reported
  beside it. Loader digests (a): part A's job 333701 already hashed the GeoTransolver loader on fastback test, fastback
  val and estate test (equal to ISLA's); est_val is covered by (b).
- **Rules.** B1-B4 of section 4 with B = GeoTransolver in place of Transolver, the same b1 rule, SE formulas and floors
  (2 SE and ln 1.03); D_X = ln(X_GT ÷ X_ISLA) against stage 1's ISLA runs. Section 4's reading rules apply to
  GeoTransolver. Reduced by reduce_b2.py (reduce_b.py with the arm set {isla, gt}; nothing else changes).
- **Predictions** (before any GeoTransolver number of stage 2):

| # | prediction | p |
|---|---|--:|
| S1 | D_G wss (GT) reads "ISLA transfers better" | 0.65 |
| S2 | D_G pressure (GT) reads "no difference" | 0.75 |
| S3 | D_H wss (GT) reads "no difference" | 0.65 |
| S4 | D_R wss (GT) reads "no difference" | 0.65 |
| S5 | D_R pressure (GT) reads "no difference" | 0.70 |
| S6 | B1 (fastback test): both GT cells level | 0.55 |
| S7 | H_GT within 0.95-1.05 on both fields | 0.70 |
| S8 | the GT seeds differ by more than 5 percent on estate test, on at least one field | 0.60 |
| S9 | the replicates give the same D_G and D_H verdicts as the leads | 0.75 |

- **Cost.** 2 × 53 GPU-h (the scout's measured GeoTransolver wall), smoke ≈ 1, scoring ≈ 0.7: about 108 GPU-h; cap 125.
  4 links per lane at 3:59.
- **Deadline.** If GeoTransolver is unread by 2026-10-24 08:45 UTC, the lane writes HANDOFF_B.md with the exact
  scoring and reduction commands.
