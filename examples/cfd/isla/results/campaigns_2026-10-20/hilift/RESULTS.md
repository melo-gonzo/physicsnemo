# UDRV-HL results: HiLift baselines with the unit freestream direction against A2 ISLA (2026-10-20)

Readout of `PREREG.md`. Every arm scored on LHR, float32, epoch-500 checkpoints, `infer_split=val` (180 cases), on ISLA's
200,000-cell set for all three protocols. Relative L2, seed mean (seed 42, seed 43). Full tables (all five fields,
all protocols, ratios, score metadata): `udrv_hl_table.md`; machine-readable: `udrv_hl_scores.json`; reducer
`udrv_table.py` (the 2026-10-18 version kept as `udrv_table_2026-10-18.py`).

## Headline (native)

| arm | drive | pressure | wall shear | velocity | density | temperature | p ÷ ISLA | wss ÷ ISLA |
|---|---|---|---|---|---|---|---|---|
| ISLA A2 (a2r_hl_isla) | U_inf_dir | 0.0432 (0.0435, 0.0428) | 0.0943 (0.0944, 0.0942) | 0.0605 | 0.0577 | 0.0494 | 1 | 1 |
| GeoTransolver 256 (udrv_hl_gt) | U_inf_dir | **0.0390** (0.0391, 0.0390) | **0.0890** (0.0890, 0.0890) | 0.0579 | 0.0527 | 0.0469 | 0.904 | 0.945 |
| GeoTransolver 256 (r80_hl_gt_h256) | U_inf | 0.0525 (0.0554, 0.0496) | 0.1237 (0.1304, 0.1169) | 0.0771 | 0.0710 | 0.0564 | 1.215 | 1.312 |
| Transolver 512 x 16 (udrv_hl_t) | U_inf_dir | **0.0394** (0.0408, 0.0380) | **0.0878** (0.0888, 0.0868) | 0.0574 | 0.0530 | 0.0475 | 0.913 | 0.932 |
| Transolver 512 x 16 (r80_hl_t_h512_l16_c100k) | U_inf | 0.0434 (0.0435, 0.0433) | 0.1005 (0.1008, 0.1002) | 0.0630 | 0.0575 | 0.0506 | 1.005 | 1.066 |

Unit ÷ physical drive (native, seed means): GeoTransolver p 0.744, wss 0.720, velocity 0.750, density 0.742, temperature
0.833; Transolver p 0.908, wss 0.874, velocity 0.910, density 0.922, temperature 0.938.

Area and posed (pressure, seed means; other fields in `udrv_hl_table.md`):

| arm | area p | posed p | area ÷ native | posed ÷ native |
|---|---|---|---|---|
| ISLA A2 | 0.0430 (0.0433, 0.0427) | 0.0436 (0.0439, 0.0432) | 0.995 | 1.009 |
| GeoTransolver, U_inf_dir | 0.1680 (0.1345, 0.2016) | 0.9519 (0.9432, 0.9606) | 4.31 | 24.4 |
| GeoTransolver, U_inf | 0.2215 (0.2360, 0.2070) | 1.7070 (1.6914, 1.7225) | 4.22 | 32.5 |
| Transolver, U_inf_dir | 0.1368 (0.1404, 0.1332) | 0.8747 (0.8836, 0.8658) | 3.47 | 22.2 |
| Transolver, U_inf @200k | 0.2890 (0.2940, 0.2839) | 1.0133 (1.0048, 1.0218) | 6.66 | 23.3 |

## Predictions against outcomes (as registered, like-for-like at 200,000 cells)

| prediction | registered | outcome | verdict |
|---|---|---|---|
| P1 GeoTransolver | native p below the physical drive by >= 10 %, expected 0.035-0.045; better on most cases | 0.0390 vs 0.0525: ratio 0.744; lower on 163/180 cases; median per-case factor 0.606 (UDRV ladder 0.62) | **holds** |
| P1 Transolver | native p below the physical drive, expected 0.033-0.041; better on most cases | 0.0394 vs 0.0434: ratio 0.908; lower on 156/180; median per-case 0.761 (ladder 0.67) | **holds** |
| P2 GeoTransolver ÷ A2 ISLA | <= 1.00; band rule: ahead if <= 0.95 and lower on >= 2/3 of cases | 0.904, but lower on only 99/180 (55 %); median per-case 0.993 | ratio part **holds**; band: none of the three (ratio says ahead, the case count does not meet 2/3; not parity) |
| P2 Transolver ÷ A2 ISLA | same | 0.913, lower on 153/180 (85 %); median per-case 0.889 | **holds; "baseline ahead"** |
| P2 wall shear follows pressure | — | GT 0.945 (lower on 139/180), T 0.932 (157/180) | **holds** |
| P3 baselines | area and posed stay several times native | area 3.5-4.3x, posed 22-24x (unit drive); physical 4.2-6.7x and 23-33x | **holds** |
| P3 ISLA | posed ÷ native within 0.002 of 1 (R-POSE) | 1.003-1.009 per field and seed (pressure 0.0090 / 0.0089, wss 0.0055 / 0.0061) | **fails** as registered |

Physical drive against A2 ISLA, for the record: GeoTransolver 1.215 ("ISLA ahead", ISLA lower on 157/180); Transolver
1.005 ("parity", but ISLA lower on 140/180).

## What it means

- The physical drive did cost both baselines, as the UDRV ladder said: GeoTransolver 26 % of its native pressure error
  (per-case median 0.61, as at 510 cases), Transolver 9 %. With the unit drive both baselines lead A2 ISLA on HiLift
  native by about 9-10 % on pressure and 5-7 % on wall shear. This reverses the CHAR record's HiLift standing (ISLA ahead
  of GeoTransolver by 1.28x and level with Transolver).
- Transolver's lead is broad (153/180 cases). GeoTransolver's is not: it is level with ISLA on the typical case (median
  per-case 0.993), and 95 % of its mean advantage comes from ISLA's 18 worst cases (10 %), which sit at AoA 16-22 (ISLA
  mean p 0.089 at AoA 16 against GT 0.068). At AoA 6-12 GT and ISLA are within 2 % of each other; Transolver beats both
  there by about 10 %.
- Area and posed: the unit drive lowers the baselines' area and posed errors (pressure x0.47-0.86) but leaves them
  3-4x and 22-24x their native error. ISLA is still the only arm whose area and posed errors equal its native error to 1 %.
- Confidence: the native seed spreads are small (GT 0.0391 / 0.0390; T 0.0408 / 0.0380; ISLA 0.0435 / 0.0428), and each
  baseline ÷ ISLA ratio holds on both seeds, so the native verdicts are firm. GeoTransolver's area scores vary a lot
  between seeds (0.1345 / 0.2016).

## Surprise: ISLA's posed error is not equal to its native error to R-POSE precision on HiLift

A2 ISLA's posed ÷ native is 1.003-1.009 on every field and both seeds, against <= 5.4e-4 on DrivAerML and SHIFT-SUV
(A2-RERUN). Area ÷ native is 0.994-1.002. The PREREG took R-POSE from A2-RERUN without a HiLift reading, and
`RESULTS_ISLA_2026-10-19.md` already noted "within 1 %". This lane did not find the cause. Candidates are float32 on
inch-scale coordinates after the translation, and how `U_inf_dir` follows the pose. It is a question for the ISLA
program; it does not change the ranking.

## Caveats

- Unit ÷ physical drive is not the drive alone. The physical-drive lanes trained on AGA with code_v18 and recipe_v8 (REF-80,
  2026-10-02). The unit-drive lanes trained on LHR with code_a2r and recipe_a2r_hl (2026-10-19/20). A2 ISLA trained on JHB to
  epoch 285 and then on LHR to 500. PREREG accepted these recipe and cluster changes (GT/T model sources md5-identical).
  Every lane uses GB300, 4 GPUs, bfloat16 and the same hyperparameters.
- Cell counts: every number above uses the same 200,000-cell set. Transolver trains at 100,000 cells (REF-80) and is scored
  at 200,000 (T-SCORESET). The physical-drive Transolver gives the same native error at either count (0.0434 at 100k in
  the CHAR record, 0.0434 at 200k here), but its area error drops from 0.3956 at 100k to 0.2890 at 200k. P1 against the
  CHAR 100k value gives the same 0.908.
- The CHAR record's GeoTransolver (0.0522) is *not* on ISLA's set, although it also used 200,000 cells. CHAR scored on
  AGA's full HiLift dataset, where each case has a different dataset index (e.g. `00003_geo_LHC001_AoA_16` on AGA,
  `00002_` on the LHR subset). The per-sample subsample generator is seeded by that index. So the table uses the LHR
  re-score, 0.0525 (0.0554, 0.0496); the CHAR record is within 0.0004 of it on every native field.
- Set identity (job 299154, `phys/scoreset_phys_identity.md`): with the CHAR composition (code_v20 + recipe_v8_hl at 200k),
  the physical-drive rows get the same sample order, geometry and targets as ISLA's set on 180/180 cases in each
  protocol. The whole-sample hash differs. Job 299279 (`phys/leafdiff.md`) traces that to four measure leaves, which the
  two code versions compute differently: the vehicle `_cache.cell.areas`, `_cache.cell.normals`, and the two
  `_effective_measure` leaves. The models read the dataset normals, which are equal. The metric weights by the effective
  measure, so the same checkpoints were also scored with the A2 scorer (job 299326, `phys/scorer_check.md`). The two
  scorers agree to <= 2.4e-7 per case on every row.
- Normalization: all four physical-drive `norm_stats.pt` match the recipe_v8_hl dataset YAML (job 299154). The six
  unit-drive and ISLA runs match too (normcheck 299023).
- The recipe snapshot is recipe_v8 with one line changed: `$T/recipe_v8_hl` sets the HiLift root to the LHR copy
  `PhysicsNeMo-HighLiftAeroML-medium` (`diff -r` shows that line only). The LHR copy was verified against AGA on
  2026-10-19. code_v20 and recipe_v8 are md5-identical between AGA and LHR (`phys/md5_code_recipe.*.txt`). The venv is
  the same path with the same torch 2.12.0+cu130; physicsnemo is imported from code_v20 (`phys/out/imports.txt`).

## Provenance

| step | where | job / command | output |
|---|---|---|---|
| unit-drive scoring (blocks 0-3) | LHR | 299041 (`$K/score.sbatch`, rows md5 04fcc4be) | `$T/evals_udrv/udrv_hl_*` -> `scores/` |
| A2 ISLA scores (identical rows, same scorer md5 8e238045) | LHR | 293808 / 294162 | `$T/evals_a2r/a2r_hl_isla_seed4{2,3}` -> `scores/` |
| checkpoint copy AGA -> LHR | aga2dc rsync to lhr dc-01 | 20 files (epoch-500 `checkpoint.0.500.pt` + `<Model>.0.500.mdlus`, norm_stats, resolved_config, train.log) | `$T/runs_from_aga/r80_hl_*`; md5 equal both ends (`phys/md5_copy.*.txt`) |
| set identity + norm stats | LHR short | 299154 (`$K/phys_s200k/hash_phys.sbatch`) | `phys/out/`, `phys/scoreset_phys_identity.md` |
| leaf diff | LHR short | 299279 (`leafdiff.sbatch`) | `phys/leafdiff.md` |
| physical-drive scoring, CHAR scorer copy | LHR short | 299159 (`$K/phys_s200k/protocol_eval_s200k.sbatch`, md5 f66d26fb; AGA original 35e01fc8) | `$T/evals_udrv/r80_hl_*_s200k` -> `scores/` |
| physical-drive scoring, A2 scorer (check) | LHR short | 299326 (`$K/score.sbatch`, rows `phys_s200k/score_rows_phys_a2r.tsv`) | `$T/evals_udrv_phys_a2r` -> `scores_phys_a2r/` |
| reduction | local | `python udrv_table.py` | `udrv_hl_table.md`, `udrv_hl_scores.json` |

`K=$T/kernel/udrv_hl`. Local copies of every new cluster file are in `phys/`. After scoring, the copied `.pt` and
`.mdlus` checkpoints in `$T/runs_from_aga/r80_hl_*` were deleted (the AGA originals remain); the small files stay.

Cost: this readout used 4.1 GPU-h (jobs 299154, 299159, 299279, 299326; 1 node x 4 GB300 each; 3,724 node-s). Job 299041
used 2.2 GPU-h. Total 6.3 GPU-h.
