# AF-WHOLE results: ISLA and ISLA-DTT trained with the AirfRANS far field (2026-10-25)

Read as registered (PREREG.md md5 2f67af124c696b3b68273ccbaa6abc6b: 19a1274b as approved plus section 10, an amendment before
submission). Generated tables: `read.md` / `read.json` (`harness/read.py`, all checks and the far-field guard passed); the
near-wall reductions `scores/nearwall_{isla,dtt}.json`; per-seed bands `r2_isla_perseed.txt`. Jobs and md5s: JOBS.md.
W = trained with the far field (`recipe_afwhole`, commit 8e93f2307); B = the eager-trained standings r090eag_af_{isla,dtt}.
Float32, full_test (200 cases), epoch 500, 100,000 points per case.

## 1. Verdicts

| pair | pressure | velocity (in-plane) | nut | pose (R3) |
|---|---|---|---|---|
| W-ISLA / B-ISLA | **harms** 1.349 (\|D\|/2SE 2.48) | **harms** 1.204 (2.62) | **harms** 1.282 (4.21) | **holds** (1.0001-1.0014) |
| W-DTT / B-DTT | **harms** 1.943 (1.77) | **harms** 1.531 (1.89) | **harms** 1.893 (1.63) | **broken** (1.53-2.52) |

CMs (s42 / s43): W-ISLA pressure 0.0164151 / 0.0172603, velocity 0.0073997 / 0.0079004, nut 0.0199853 / 0.0193769;
W-DTT pressure 0.0285735 / 0.0196431, velocity 0.0121810 / 0.0097862, nut 0.0465597 / 0.0314415. B's: PREREG section 6.
Posed labels equal the native ones on every field (the same runs; no independent number). Area equals native to 6e-8 on
every row: degenerate, no claim. Beside: W beats B on 15 / 8 / 16 % of cases (ISLA) and 10 / 2 / 11 % (DTT); case-median
ratios 1.46 / 1.22 / 1.38 (ISLA), 1.69 / 1.43 / 1.58 (DTT); seed-matched ISLA 1.25-1.46 / 1.15-1.26 / 1.27-1.30.

R3 posed / native, seeds 42 / 43:
- W-ISLA: pressure 1.0011 / 1.0011, velocity 1.0014 / 1.0012, nut 1.0006 / 1.0001, closer to one than B-ISLA's 1.0022 / 1.0025 on pressure.
- W-DTT: pressure 1.8536 / 1.9483, velocity 1.6331 / 1.5334, nut 2.5205 / 2.4788, against B-DTT's 1.0017-1.0022. **Pose broken.**

## 2. R2: where the harm sits (pooled band e_R, W / B, seed means; labels by the R1 rule on the band)

| band (chords) | share | ISLA p | ISLA v | ISLA nut | DTT p | DTT v | DTT nut |
|---|---|---|---|---|---|---|---|
| [1e-6, 1e-5) | 0.023 | 1.01 | **1.43** | **1.12** | 1.33 | **1.41** | 1.16 |
| [1e-5, 1e-4) | 0.096 | 1.04 | **1.26** | **1.17** | 1.32 | **1.54** | 1.27 |
| [1e-4, 1e-3) | 0.164 | 1.07 | **1.15** | 1.13 | 1.38 | **1.87** | **1.31** |
| [1e-3, 1e-2) | 0.185 | 1.17 | **1.14** | **1.18** | **1.75** | **1.87** | **1.65** |
| [1e-2, 0.1) | 0.209 | **1.28** | **1.11** | 1.01 | **1.77** | **1.52** | **1.88** |
| [0.1, 1) | 0.237 | 1.16 | 1.01 | 1.00 | **1.56** | 1.12 | **1.53** |
| [1, inf) | 0.081 | 1.21 | 1.01 | 1.28 | **1.39** | 1.10 | 2.31 |
| below 3.05e-4 (the softening's linear part) | 0.201 | 1.04 | **1.24** | **1.15** | 1.33 | **1.64** | **1.27** |
| above 3.05e-4 | 0.799 | 1.13 | **1.12** | 1.02 | **1.58** | **1.58** | **1.70** |

Bold: "band harms". The band (-inf, 1e-6) holds only the wall nodes (target zero velocity): empty for velocity and nut.
Checks: the same points in all eight runs (hash ab498a7b9722); the dumps reproduce the scorer's pressure and velocity l2
exactly (nut differs by the scorer's standardization, as in SLICES-AF). Per seed (W-ISLA, `r2_isla_perseed.txt`): velocity
below / above 3.05e-4 is 1.146 / 1.068 (s42) and 1.337 / 1.166 (s43); nut 1.045 / 0.984 and 1.258 / 1.061: larger below on
both seeds for both fields. Pressure has no such ordering (s42 0.873 / 0.934, s43 1.283 / 1.398).

## 3. Cause analysis (PREREG section 7)

- **W-ISLA, velocity and nut: the near-wall softening.** The harm is largest in the first decades off the wall
  (velocity 1.43 at 1e-6 to 1e-5 chords, falling to 1.01 beyond 0.1) and larger below 3.05e-4 chords than above it on
  both seeds. That is section 7's criterion for the softening 1e-6 L as the leading cause. It also meets NEARWALL's
  contingency trigger (a harm near the wall). The held viscous softening asinh(d U / nu) is the supervisor's and Peter's
  to test; nothing was launched.
- **W-ISLA, pressure: not near the wall.** The harm is largest at 0.01-0.1 chords (1.28) and in the outer bands, and the
  wall bands read 1.01-1.07. My untested hypothesis is the measure weighting. ISLA's slice routing and states are
  measure-weighted, and the far field holds 99.86 % of the boundary measure (the airfoil 0.14 %). So the airfoil's 35-39 %
  of the boundary tokens carry almost no slice mass. Not tested here.
- **W-DTT: a tie in the tree read's octant split, exposed by the far field.** This was measured after the read, so it is
  analysis beside the verdicts, not a rule.
  - `tree_read.normal_octants` assigns each cell to one of 2^d trees by the sign (`> 0`) of its normal's components in
    the canonical frame.
  - The far field's straight edges (x = 200, y = +-200) have normals exactly on the lab axes. The frame's axes are the
    lab axes too, tilted by about 1e-6 by the airfoil. So one component of every edge normal is about +-2e-6
    (as low as 1.7e-8), a sign tie.
  - On all 200 test cases (LHR cpu-short 381148, `harness/octant_pose.py`, `out/octant_pose.jsonl`), the posed pipeline
    moves **16-45 % (median 29 %) of the far-field cells to a different octant tree**, against a median of 2 airfoil
    cells (max 9).
  - The read weights cells by their share of the measure, which is almost all far field. So the far-field clusters each
    point reads change with the pose, and the trained W-DTT responds by a factor 1.5-2.5.
  - The same tie makes the native split of whole far-field edges depend on a 1e-6 tilt that varies from case to case.
    That is a plausible share of W-DTT's native harm, which is also spread over every band (1.33-1.88), unlike W-ISLA's.
  - Not the frame sign: condition 3 counted 0 flips of either axis on the 200 cases (`out/frame_pose_summary.txt`).
  - Not the far field's placement: the posed far field follows the airfoil's rigid map to 4.2e-5 chords.
  - B-DTT is unaffected: the airfoil's normals are not axis-aligned.
  - A fix (an octant rule with no exact ties, such as octants in a frame rotated off the cells' normals, or one tree per
    coherent piece) is a model change for the supervisor and Peter.

## 4. Predictions (PREREG section 8)

| id | outcome | held? |
|---|---|---|
| P1 smoke passes first time (0.85) | passed | yes |
| P2-P4 W-ISLA p / v / nut: harms at 0.25 / 0.35 / 0.40 | harms, harms, harms | the less likely outcome on all three |
| P5-P7 W-DTT p / v / nut: harms at 0.30 / 0.25 / 0.30 | harms, harms, harms | the less likely outcome on all three |
| P8 W-ISLA nut e_R in [1e-6, 1e-4) above B's on both seeds (0.60) | 1.06 / 1.09 (s42), 1.19 / 1.25 (s43) | yes |
| P9 W-ISLA W / B larger below 3.05e-4 than above, velocity and nut (0.60) | both, both seeds | yes |
| P10 W-ISLA pose holds (0.85) | holds | yes |
| P11 W-DTT pose holds (0.75) | broken | no |
| P12 / P13 smoke step W / B in [1.00, 1.25] (0.65 / 0.50) | ISLA 0.984, DTT 1.005 | no / yes |
| P14 four lanes with no guard stop (0.85) | no stop | yes |

## 5. Cost (R4)

- Smoke, same node (361876): W / B median step 0.984 (ISLA, 0.746 / 0.758 s) and 1.005 (DTT, 0.956 / 0.951 s); peak allocated
  1.018 (43.8 / 43.0 GiB) and 1.029 (48.7 / 47.3 GiB).
- Whole runs: median train_step 0.746 / 0.747 s (W-ISLA), 0.943 / 0.940 s (W-DTT); the standings' were 0.754 / 0.756 s
  (ISLA). Peak trainer mem_gb 46.7 / 51.5.
- **GPU-h (sacct elapsed x 4): 397.5 of the 520 cap.** W-ISLA 86.5 + 86.5 (7 links each), W-DTT 110.4 (8) + 110.3 (10),
  scoring 1.5, near-wall inference 1.65, smoke 0.65. CPU: 8 cpu-short jobs.
- The far field costs nothing per step on AirfRANS: its 1,700 cells are 2-3 % of the tokens. The two-dimensional exact
  search is not the DrivAerML Warp seed that set BOUNDARY-AUDIT's 5x.

## 6. Side finding (recorded at the supervisor's request; no action)

On 168 of the 1,000 case files (36 of them in full_test), the stored airfoil `normals` field disagrees with the polyline's
geometric normals on 0.8-38.6 % of the segments. The case list with the fraction per case is in
`out/airfoil_normals_cases.tsv`. The converter (kernel/airfrans_convert.py) orients each segment away from the polyline's
mean point, a test that fails on cambered aft sections. No ISLA config reads that field (ISLA takes the geometric normals,
which follow the winding, into the fluid on all 1,000 cases). A baseline that reads `boundaries.airfoil.cell_data.normals`
would see the flipped segments.

## 7. Data

- Scores: `$T/evals_afwhole/afwhole_af_{isla,dtt}_seed4x/{native,area,posed}/` (local copy `scores/evals_afwhole/`).
- Near-wall dumps: `$T/evals_afwhole_nw/` (W four runs, B-DTT and B-ISLA re-dumped, 1.0 GB per run, 8 runs).
- Checkpoints: `$T/runs/afwhole_af_{isla,dtt}_seed4x/`.
