# R-HL090 readout: ISLA 0.9.0 on HiLift against A2 (2026-10-20 15:38 EDT)

Pre-registration: PREREG.md (md5 a057d9987d5d2ca93a38df71d563dd64, written before submission). Lanes h090_hl_isla_seed42 and
seed43 (jobs 300297/300298 and their links), code_h090 / recipe_h090 (75ac723d8 + td0152), 510 training cases, 500
epochs. Scorer job 304962: infer.py, float32, 180 validation cases, ISLA's 200,000-cell set, epoch-500 checkpoint, native,
area and posed. Checks: every `config` row says `"epoch": 500`, 180 `infer_step` rows per protocol and seed. Scores:
scores/evals_h090/. Partner: a2r_hl (scratch/isla-hilift-lhr-2026-10-19/scores/evals_a2r/;
scratch/isla-udrv-hl-2026-10-18/udrv_hl_scores.json). Reduction: the mean over cases of the per-case relative L2 (last row
per sample id), then seed means.

## Table (native unless stated; seeds 42, 43 in parentheses)

| arm | pressure | wall shear | arm ÷ ISLA 0.9.0 |
|---|---|---|---|
| ISLA 0.9.0 | 0.04272 (0.04271, 0.04274) | 0.09634 (0.09612, 0.09657) | 1 |
| A2 ISLA | 0.04318 (0.04352, 0.04284) | 0.09426 (0.09437, 0.09415) | 1.011 / 0.978 |
| GeoTransolver, unit drive | 0.03902 | 0.08903 | 0.913 / 0.924 |
| Transolver, unit drive | 0.03941 | 0.08783 | 0.922 / 0.912 |

| protocol | ISLA 0.9.0 p / wss | A2 p / wss | GeoTransolver unit p | Transolver unit p | baselines ÷ 0.9.0 on p |
|---|---|---|---|---|---|
| area | 0.04281 / 0.09681 | 0.04298 / 0.09443 | 0.16801 | 0.13683 | 3.92 / 3.20 |
| posed | 0.04312 / 0.09684 | 0.04357 / 0.09480 | 0.95193 | 0.87470 | 22.1 / 20.3 |

Posed ÷ native, ISLA 0.9.0: pressure 1.0092 / 1.0094, wall shear 1.0049 / 1.0055 (A2: 1.0090 / 1.0089 and 1.0055 / 1.0061).
Area ÷ native, ISLA 0.9.0: pressure 0.9978 / 1.0061, wall shear 1.0006 / 1.0090.

## Rule R-HL090

- Pressure: G = 0.9894; r_42 = 0.9812, r_43 = 0.9977.
- Wall shear: G = 1.0221; r_42 = 1.0186, r_43 = 1.0257.
- Checked in the order worse (G ≥ 1.05 with both seeds ≥ 1.05 on a field: no), better (G ≤ 0.95 on both fields: no),
  level (G within 0.95-1.05 on both fields: yes).
- **Verdict: LEVEL.** Under the registered R2-style clause, the h090 rows become HiLift's ISLA standing. The A2 rows stay in
  the record as the partner.

## Predictions

| # | prediction | probability | outcome |
|---|---|---|---|
| P1 | R-HL090 = level | 0.60 | **held** |
| P2 | all three AoA-16 cases stay wrong | 0.55 | **held**: LHC153_16 0.219 (0.174, 0.264; A2 0.278), LHC162_16 0.205 (0.230, 0.181; A2 0.248), LHC119_16 0.189 (0.173, 0.205; A2 0.208) |
| P2' | each one stays wrong | 0.75 each | **held** for all three; each improved, none below 0.15 |
| P3 | LHC002_AoA_18 and LHC132_AoA_22 both stay wrong | 0.75 | **failed**: LHC132_22 stays wrong at 0.198 (0.250, 0.146; A2 0.190), but LHC002_18 is between at 0.137 (0.081, 0.193; A2 0.219). Seed 42 is the first ISLA run right on that case |
| P4 | posed ÷ native persists in the bands on both seeds | 0.70 | **held** (pressure 1.0092 / 1.0094; wall shear 1.0049 / 1.0055) |
| P5 | the unit-drive baselines stay ahead on native pressure | 0.80 | **held** (0.04272 > 0.03941 and 0.03902) |
| P6 | the unit-drive baselines stay ahead on native wall shear | 0.80 | **held** (0.09634 > 0.08903 and 0.08783) |
| P7 | under area and pose, 0.9.0 ahead of both baselines by ≥ 3× on pressure | 0.95 | **held** (3.20× at the closest, Transolver area) |

## Beside the rules

- Per-case changes against A2 (seed-mean pressure). The largest improvements are LHC002_AoA_18 −0.082, LHC148_AoA_20
  −0.065, LHC153_AoA_16 −0.059, LHC162_AoA_16 −0.042 and LHC131_AoA_18 −0.030. The largest regressions are LHC035_AoA_22
  +0.030, LHC110_AoA_22 +0.029, LHC079_AoA_22 +0.023, LHC130_AoA_18 +0.021 and LHC053_AoA_22 +0.011. The post-stall
  cases move both ways and largely cancel in the mean.
- The seed means agree to 0.07% on pressure. The post-stall cases flip between seeds: LHC002_18 0.081 / 0.193, LHC132_22
  0.250 / 0.146, LHC153_16 0.174 / 0.264.
- Section 5 of the PREREG applies: level with P2 holding, so 0.9.0 replaces A2 as the standing. The baselines' errors are
  0.91-0.92 of 0.9.0's on both fields. Corrected 2026-10-20 21:50 EDT after the reviewer's audit: "stays a stall-onset
  problem" holds only against GeoTransolver on pressure. Per case at 0.9.0, with seed means of h090 against udrv_hl_t/gt:
  - against GeoTransolver on pressure, the gap sits mostly in post-stall cases. ISLA is worse on 127 of 180 cases, with a
    median per-case ratio of 1.03 at 4-12 degrees and 1.06 at 14-22 degrees. Without the 10 largest-gap cases, ISLA's
    mean error is 1.016 times GeoTransolver's;
  - against Transolver on pressure, the deficit is broad. ISLA is worse on 158 of 180 cases, with medians of 1.18 at 4-12
    degrees and 1.13 at 14-22 degrees, and 1.030 without the 10 largest-gap cases;
  - on wall shear the deficit is broad against both: worse on 159 (GeoTransolver) and 165 (Transolver) of 180 cases, and
    1.049 and 1.067 without the 10 largest-gap cases.
  The CAP-AUDIT phase-1 case geo_LHC002_AoA_18 asks whether ISLA can fit a post-stall case at all.
- The wall-shear G of 1.022 is inside the level band, but both 0.9.0 seeds (1.019, 1.026) lie above both A2 seeds. That
  is about 10 times A2's 0.2 percent wall-shear seed gap: a small but consistent shift.
