# HL-DUP stage 2: ISLA and both baselines retrained on the deduplicated HiLift wing

Written 2026-10-24 at about 18:35 EDT. Pre-registration: `PREREG.md` (md5 072b6b1c3ac1b59df729eadda9f04e40, with the
supervisor's amendment of 06:44 EDT and the registered smoke checks). Jobs and md5s: `JOBS.md`, "Stage 2".
Reduction: `stage2_reduce.py` (a86656c4), giving `stage2_table.md` (d620b6a0) and `stage2_summary.json` (f08ac2ef).
Scores: `scores_stage2/`, from LHR `$T/evals_hldupx/`.

**The registered verdict is mixed, so no verdict is claimed.** Every baseline / ISLA ratio moved toward 1 by
0.019-0.045, and the wall-shear gap narrowed for both baselines as registered. On pressure the rule does not resolve:
for GeoTransolver the move is within the 0.03 band, and for Transolver the two seeds disagree. Both baselines stay
ahead of ISLA on every field in the seed mean. Under the amendment the deduplication is canonical on principle, so
these rows set the book's HiLift standing.

## What ran

- Six lanes, 500 epochs each, all at epoch 500 with `checkpoint.0.500.pt`:
  - ISLA 0.9.0 (code_h090 + `recipe_hldupx_isla`);
  - GeoTransolver 256 and Transolver 512 x 16, both with the unit drive U_inf_dir (code_a2r + `recipe_hldupx_base`);
  - seeds 42 and 43 each; dataset `highlift_surface_medium_dedup` (the standing's pipeline with
    `drop_duplicate_cells`).
- The six chains (all on 2026-10-24) were: ISLA s42 367678 then 367684, 369503, 371318; ISLA s43 367679 (failed
  after 4:53, the Errno 521 Triton-cache race), then 367683, 367733, 369595, 371387; GT s42 367681, 367709, 369504,
  371317; GT s43 367682, 367710, 369538, 371356; T s42 367686, 367711, 369539, 371357; T s43 367687, 367723, 369540.
  Every other link ended at its 03:59 limit, and the chain resumed from checkpoints.
- Monitoring found no non-finite loss, no validation alarm, no BLOCKED marker, no hang and no duplicate link.
- Scoring used the registered scorer (`kernel/hldupx_score`): jobs 371516 (blocks 0 and 2) and 372745 (block 1).
  It ran float32 on the 180 val cases, with 200,000 cells drawn per case and the extra copies then dropped (about
  186,000 kept). Native and posed.
- Cost: 209.6 GPU-h for stage 2 (smokes, lanes and scoring; cap 320), and about 45 core-h of CPU for the masks and
  the checks.

## Checks before reading

- All 12 rows: the `config` row says `"epoch": 500`, with 180 `infer_step` rows per protocol and dataset
  `highlift_surface_medium_dedup`.
- Every score log shows `drop_duplicate_cells ... fingerprint checked`.
- The baselines' saved resolved configs show the intended forward_kwargs: GT `global_embedding: global_data.U_inf_dir`
  and T `fx.source: global_data.U_inf_dir`.
- Posed is not degenerate: it changes every arm's per-case pressure on all 180 cases (0 identical).
- The scored cells are the same as in stage 1's (b). The scorer seeds the reader from training.seed 42, and the
  deduplicating reader draws the same cyclic block over all stored cells and then drops the copies. (b) scored that
  same block without its copies, under the same N/k measure factor. This holds by construction; it was not checked
  sample by sample.

## Results (native unless stated; seed mean, seeds in parentheses; `stage2_table.md` has all five fields)

| arm | pressure | wall shear | velocity | density | temperature |
|---|---|---|---|---|---|
| ISLA 0.9.0 | 0.05458 (0.05304, 0.05613) | 0.09633 (0.09453, 0.09812) | 0.07113 | 0.05401 | 0.06366 |
| GeoTransolver, unit drive | 0.05109 (0.05056, 0.05161) | 0.09215 (0.09211, 0.09218) | 0.06823 | 0.05060 | 0.06123 |
| Transolver, unit drive | 0.05274 (0.05420, 0.05129) | 0.09189 (0.09339, 0.09039) | 0.06917 | 0.05212 | 0.06278 |

Posed / native: ISLA 1.009 on pressure for both seeds, and 1.005-1.006 on wall shear. GeoTransolver 19.5-20.3 on
pressure, Transolver 18.6-19.1, and both baselines 13.5-14.1 on wall shear.

## The rule as registered (PREREG "Rule"; G = seed-mean baseline / seed-mean ISLA)

| baseline, field | stage-1 (b) reference | G (dedup) | change | seed ratios | that cell reads as |
|---|---|---|---|---|---|
| GeoTransolver, pressure | 0.917 | 0.936 | +0.019 | 0.953, 0.920 | within 0.03 (unchanged band) |
| GeoTransolver, wall shear | 0.922 | 0.957 | +0.035 | 0.974, 0.939 | narrows (both seeds above) |
| Transolver, pressure | 0.924 | 0.966 | +0.042 | 1.022, 0.914 | seeds disagree (s43 below) |
| Transolver, wall shear | 0.909 | 0.954 | +0.045 | 0.988, 0.921 | narrows (both seeds above) |

- *Narrows* needs all four cells to narrow. *Widens* needs all four to widen. *Unchanged* needs every cell within
  0.03, which fails on three. So the verdict is **mixed**, reported per cell above, with no verdict claimed.
- The independent numbers are 4 ratios x 2 seeds. Pressure and wall shear share each seed's model, so the 8 seed
  ratios come from 6 trainings.

## Predictions

| # | prediction | probability | outcome |
|---|---|---|---|
| P1 | ISLA dedup native pressure is below its (b), 0.0589, by at least 3 % | 0.55 | **held**: 0.0546, ratio 0.927 (seeds 0.900, 0.955) |
| P2 | each baseline's dedup native pressure is within 5 % of its (b) | 0.70 | **failed**: GT 0.0511 against 0.0540 (0.946, 5.4 % better); T 0.0527 against 0.0544 (0.969, within) |
| P3 | the rule verdict is "unchanged" (narrows 0.30, widens 0.10, mixed 0.10) | 0.50 | **failed**: mixed |
| P4 | ISLA dedup is below 0.065 (its (c) was 0.0832) | 0.85 | **held**: 0.0546 |
| P5 | ISLA posed / native stays within 1.005-1.015 | 0.65 | **undetermined**: the registration names no field. In band on pressure (1.0093 / 1.0091), wall shear (1.0057 / 1.0052) and density (1.0078 / 1.0076); below it on velocity (1.0031 / 1.0029) and temperature (1.0041 / 1.0041). The reducer tests all five fields and records it as not held. The PREREG's reference value, 1.009, is the pressure ratio. |

## Beside the rule

- **ISLA dedup against stage 1** (the standing's checkpoints, on the same scored cells):
  - pressure is 0.900 / 0.955 of its (b) per seed, and wall shear 0.922 / 0.953;
  - against its (c), the stored-data model fed deduplicated input, it is 0.62 / 0.69 on pressure;
  - the baselines gain less from the retrain: pressure 0.935 / 0.957 (GT) and 0.966 / 0.972 (T) of their (b).
  The duplicates cost ISLA more than the baselines in training as well, but less than the input shift (c) suggested.
- **AoA bands** (pressure, seed means):

| band | ISLA | GeoTransolver | Transolver | GT / ISLA | T / ISLA |
|---|---|---|---|---|---|
| 4-12 deg (84 cases) | 0.0262 | 0.0271 | 0.0243 | 1.033 | 0.925 |
| 14-22 deg (96 cases) | 0.0794 | 0.0721 | 0.0777 | 0.908 | 0.978 |

  - Against GeoTransolver, ISLA is now ahead before stall (both seeds: GT / ISLA 1.050 / 1.016) and behind after it
    (0.925 / 0.892). On seed-mean pressure it is lower on 116 of 180 cases, against 54 under (b): 66 of 84 before stall
    and 50 of 96 after.
  - Transolver's lead is pre-stall (T / ISLA 0.925). Post-stall the seeds disagree: 1.047 on seed 42, 0.913 on seed
    43. ISLA is lower than Transolver on 35 of 180 cases, against 20.
  - ISLA's dedup gain over (b) is larger before stall: 0.898 of (b) at 4-12 (0.0262 against 0.0292), and 0.936 at
    14-22 (0.0794 against 0.0849).
- **ISLA's seed spread widened.** Pressure is 0.0530 / 0.0561 (5.8 %), against 0.0427 / 0.0427 for the standing
  (0.07 %). Transolver's spread is 5.7 % (0.0542 / 0.0513) and GeoTransolver's 2.1 %. So the ISLA and Transolver ratios
  carry seed noise of a few percent, which is why the Transolver pressure cell does not resolve.
- **ISLA's length unit on the training cases.** Job 373091 ran the stage-1 exposure probe (`hl_expose.py`, flag mode)
  on all 510 training cases, on the epoch-0 draw, as the model receives it (`out/expose_train/expose.jsonl`); its draw
  matched MeshReader's on 3 of 3 checks. ISLA._gauge without the extra copies, divided by the gauge with them:
  - training cases: 0.773-1.868, median 1.136; val cases: 0.772-1.186;
  - the copies carried 0.01-84 % of the drawn measure on the training cases (median 46 %);
  - across cases, the length unit with the copies has a coefficient of variation of 14.6 % (train) and 12.6 % (val);
    without them, 1.3 % and 0.6 %.
  So the duplicates made ISLA's case scale vary about tenfold more from case to case than the wing geometry does.
  The deduplicated training gives ISLA a nearly constant length unit. Training ran against the stored data before,
  and the val cases stay in the same range, so the training set shows no extra shift.

## What it means for the program

- The new HiLift standing is on the deduplicated wing, with the metric over distinct cells:
  - ISLA: pressure 0.0546, wall shear 0.0963;
  - GeoTransolver: 0.0511 / 0.0921 (baseline / ISLA 0.936 / 0.957);
  - Transolver: 0.0527 / 0.0919 (0.966 / 0.954).
  These are not comparable with the old standing's 0.0427: that metric put 42 % of its weight on copies of about
  25,700 root cells, which are easy cells. Stage 1's (b), about 0.059 for ISLA, is the like-for-like old number.
- What the duplicates did to ISLA's HiLift deficit. As registered, the gap narrowed on wall shear for both
  baselines, and on pressure the rule does not resolve. Beside the rule:
  - every baseline / ISLA ratio moved toward 1, and the baselines' lead fell from about 8 % to about 3-6 %;
  - ISLA is ahead of GeoTransolver before stall and lower than it on most cases;
  - both baselines still lead in the mean: GeoTransolver through post-stall cases, Transolver through pre-stall ones.
- Confidence: firm that ISLA gained more from the correction than the baselines on seed 42 (every field); on seed 43
  the margin is under 1 % on pressure, density and velocity (ISLA 0.955 against GT 0.957 of its (b) on pressure). Not firm
  on the size of the remaining pressure gap to Transolver, where the seed ratios are 1.02 and 0.91.

## Not done here

- ISLA-DTT and the whole-boundary (far-field) HiLift arms. BC-SUPPLY brings the far field, and the deduplication,
  into canonical ISLA.
- An area protocol (degenerate for ISLA, as before).
