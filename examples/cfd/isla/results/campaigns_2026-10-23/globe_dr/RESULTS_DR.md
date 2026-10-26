# GLOBE-SCORE: DrivAerML Surface readout (2026-10-25, 10:25 EDT)

**Verdict.** Under the CHAR protocol, GLOBE at its authors' DrivAerML configuration trails ISLA A2, the registered
partner, by 2.20× on pressure and 1.73× on wall shear (native, 200k cells). Both seeds are above 1 on both fields, and
GLOBE is worse than ISLA 0.9.0 on every one of the 48 cars on both fields. It also trails both fair-drive baselines by
about the same factor (1.74× to 2.31×).

- **§8 AeroZoo trigger: does not fire.** The rule asks for GLOBE within 1.10× of ISLA A2 on at least one headline field on
  at least one dataset. The closest field is wall shear here, at 1.73×, and AirfRANS pressure, at 1.32× (RESULTS_AF.md).
  Peter has not asked for the stage. GLOBE-SCORE therefore ends DONE.
- **§1 fallback: not triggered** on DrivAerML. The worst validation was 1.016× its running minimum (seed 42) and 1.015×
  (seed 43), over 51 validations each.
- Unlike AirfRANS, **GLOBE's posed score does not equal native on DrivAerML**: posed ÷ native is 1.04-1.08. The
  likely cause is the Barnes-Hut tree (θ = 1.5 here, θ = 0 on AirfRANS), which is built from axis-aligned boxes. This is
  read from the code, not isolated by an experiment.

Pre-registration: `PREREG.md` (approved 2026-10-19 11:04 EDT). Jobs, commands and the intervention log: `JOBS.md`.
Reduction: `reduce_dr.py`, which writes `scores_dr/globe_dr_scores.json` (full precision, per seed, every ratio). The raw
records are in `scores_dr/evals/` (200k) and `scores_dr/evals_c100k/` (100k).

## What was scored

- **Runs:** `globe_dr_seed42` and `globe_dr_seed43` on LHR. First links 283158 / 283159; about 33 links each; both logged
  "Training completed!" at 07:03 PDT on 10-25, with no traceback and no BLOCKED marker. `GLOBE.0.500.mdlus` was loaded.
- **Scoring:** LHR jobs 383023_[0-1] (200k: native, area and posed; rows 0-5 of `kernel/globe_dr/score_rows.tsv`) and
  383024_[0] (100k native; `score_rows_c100k.tsv`, `EVAL_ROOT=evals_globe_dr_c100k`), submitted 10:07 EDT with the
  JOBS.md commands. All 8 rows have a `.done`, one START line each.
- **Checks, asserted by `reduce_dr.py` for every row:** `"epoch": 500` in the config row; 48 cars in infer_summary and 48
  per-car rows, with the summary equal to the row mean within 1e-4; the scorer's START line says `cells=200000` (or
  `100000` for the secondary). Precision is float32, fixed by the registered scorer.

## GLOBE, the protocols (relative L2; seed mean, then seed 42 and seed 43)

| protocol | pressure | wall shear |
|---|---|---|
| native, 200k | 0.09782 (0.09701, 0.09863) | 0.12145 (0.12037, 0.12254) |
| area (uniform point cloud), 200k | 0.09881 (0.09797, 0.09965) | 0.12402 (0.12280, 0.12524) |
| posed (rotated and translated), 200k | 0.10391 (0.10480, 0.10302) | 0.12714 (0.12775, 0.12653) |
| native, 100k (secondary) | 0.10383 (0.10293, 0.10473) | 0.12924 (0.12802, 0.13046) |

- Posed ÷ native (s42 / s43): pressure 1.080 / 1.045, wall shear 1.061 / 1.033. The largest change on one car is 14.6%.
- Area ÷ native: pressure 1.010 / 1.010, wall shear 1.020 / 1.022.
- 100k ÷ 200k: pressure 1.061 / 1.062, wall shear 1.064 / 1.065. GLOBE's boundary input is the scored subsample itself, so
  at 100k it integrates over a coarser surface. Its scores therefore depend on the cell count, as the PREREG did not expect
  (G6).

## Comparison (GLOBE ÷ arm, seed mean, then per seed; same 200k scored set unless marked)

| arm | arm's native p / wss | native p | native wss | posed p / wss |
|---|---|---|---|---|
| **ISLA A2, `a2r_dr` (registered partner)** | 0.04452 / 0.07004 | **2.197** (2.226, 2.170) | **1.734** (1.761, 1.708) | 2.334 / 1.815 |
| ISLA 0.9.0, `r090_dr_isla` (beside, not registered) | 0.04113 / 0.06622 | 2.378 (2.357, 2.399) | 1.834 (1.818, 1.850) | 2.526 / 1.920 |
| GeoTransolver 256, unit drive, `r80ud_dr_gt_h256` | 0.04552 / 0.06962 | 2.149 (2.137, 2.162) | 1.744 (1.730, 1.759) | 0.082 / 0.100 |
| Transolver 512×16, unit drive, at 200k (`r80ud_dr_t`) | 0.04234 / 0.06782 | 2.310 (2.293, 2.328) | 1.791 (1.781, 1.801) | 0.080 / 0.094 |
| GeoTransolver 256, raw drive (record) | 0.04551 / 0.06953 | 2.149 | 1.747 | 0.085 / 0.093 |
| Transolver 512×16, raw drive, at 200k (record) | 0.04352 / 0.06961 | 2.248 | 1.745 | 0.106 / 0.109 |
| Transolver 512×16, unit drive, **at 100k** (GLOBE at 100k) | 0.04226 / 0.06760 | 2.457 (2.436, 2.478) | 1.912 (1.899, 1.925) | — |
| Transolver 512×16, raw drive, **at 100k** (CHAR record) | 0.04368 / 0.06961 | 2.377 | 1.857 | — |

- Every per-seed ratio lies on the same side of 1 as its seed mean, in every row.
- Area ratios (GLOBE ÷ arm): ISLA A2 2.220 / 1.753; ISLA 0.9.0 2.392 / 1.847; the baselines 0.23-0.44 (their area scores
  are 0.23-0.48).
- Per car, GLOBE ÷ ISLA 0.9.0 (native) runs from 1.50 to 3.19 on pressure (median 2.44) and 1.24 to 2.29 on wall shear
  (median 1.91), on both seeds: 48 of 48 cars worse.
- Sources: `results/char/drivaerml_surface_a2.json` (A2, the unit-drive arms, the raw-drive arms and the CHAR 100k arm);
  `scratch/isla-ref090-2026-10-20/scores/evals_r090` (0.9.0; the script asserts 0.04113 / 0.06622, as in
  RESULTS_partial.md); `scratch/isla-drive-fix2-2026-10-19/partB/tset200k_scores.json` (unit-drive Transolver at 100k).

## Predictions against outcomes (PREREG §7, the DrivAerML parts)

| # | prediction (probability) | outcome |
|---|---|---|
| G1 | DrivAerML posed ÷ native within 1 ± 1e-3 on every field and seed (0.9) | **fails**: 1.033-1.080 |
| G2 | area ÷ native ≤ 1.05 on both fields (0.75) | **holds**: 1.010 / 1.021 |
| G5 | GLOBE ÷ ISLA A2 below 1: pressure (0.3) | **fails**: 2.197 |
| G5 | wall shear (0.25) | **fails**: 1.734 |
| G5 | point guesses p 0.050, wss 0.080 | actual 0.0978 / 0.1215 |
| G6 | 100k within 3% of 200k on both fields (0.8) | **fails**: +6.1% / +6.4% |
| G7 | no DrivAerML lane triggers the fallback (0.7) | **holds** |

## What this means, and how sure I am

- **Direction: firm.** Two seeds 1.7% apart, every car on the same side, and a factor of about 2. No plausible seed or
  scoring noise closes it.
- **It is a fit limit, not a scoring fault.** GLOBE's final training loss (0.00175, seed 42) is about its validation loss
  (0.00195), and 6.7× ISLA 0.9.0's training loss (0.00026) under the same loss and normalization. So GLOBE underfits the
  training set too; it does not overfit. Its validation loss was 0.00255 at epoch 100, 0.00198 at 200 and 0.00195 at
  300, then flat.
- **What the configuration does and does not test.** This is GLOBE's own DrivAerML example settings (1.0M parameters,
  80k faces per step, lr 1e-2) under the shared recipe protocol: StepLR, per-component Huber, Muon + AdamW, no gradient
  clipping, no ground-plane boundaries. None of these departures was isolated, so none is a measured cause. ISLA 0.9.0
  has 69M parameters, about 70 times GLOBE's 1.0M, which is the size its authors set.
- **Posed.** GLOBE's architecture is rotation-equivariant, but its accelerated summation (θ = 1.5) is not exactly so: the
  posed scores are 3-8% worse than native. Even so, GLOBE posed (0.104 / 0.127) is about 10× better than the posed
  baselines (1.26-1.36), and 2.3× worse than ISLA posed.

## Interventions during training (details in JOBS.md)

- None by this lane. On 10-22 20:38 EDT the LHR share hit its quota (another session's copy) and both running links
  hung in NCCL; the supervisor cancelled 336652_0 / 336751_1 at epoch 281 and the chain resumed from the checkpoint.
  LHR maintenance (10-21) delayed links by about 9 h; nothing was resubmitted.
