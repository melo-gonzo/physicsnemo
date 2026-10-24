# SLICES-DRV results (written 2026-10-24 12:15 EDT)

ISLA 0.9.0 with `n_slices=128` (S) against the DrivAerML Volume standing at 512 slices (B, r090_vol_isla), body-only
configuration (`drivaer_ml_volume_reference`), the 48 val cases, seeds 42 / 43, 500 epochs, lr 3.75e-4, bf16 training on
code_r090 / recipe_r090 with the per-block compile (as B), scored native / area / posed in float32 at epoch 500 with B's
scorer. Pre-registration PREREG.md (md5 88e36cd7: 9d8c65e1 plus the supervisor's amendment of 00:20 EDT to the decision
use, approved before any job). Jobs and commands: JOBS.md.

## Registered reading (R1, native decides): no change on every field, no sub-bar flag, so 128 slices is a scope-freeze candidate on two datasets

From `read_slicesdrv.py` (md5 1588c4b5) -> `reading.md`, `reading.json`. CM = the scorer's case-mean relative L2.

| field | B (s42, s43) | S (s42, s43) | S ÷ B | D | s_B, s_S | 2 SE | seed-matched S ÷ B | label | sub-bar flag |
|---|---|---|---|---|---|---|---|---|---|
| **pressure** | 0.038377 (0.037954, 0.038801) | 0.038466 (0.038438, 0.038494) | **1.002** | +0.0023 | 0.0156, 0.0010 | 0.022 | 1.013, 0.992 | **no change** | none |
| **velocity** | 0.057705 (0.057326, 0.058084) | 0.057408 (0.057401, 0.057415) | **0.995** | -0.0052 | 0.0093, 0.0002 | 0.013 | 1.001, 0.988 | **no change** | none |
| **nu_t** | 0.084655 (0.084012, 0.085298) | 0.084549 (0.083887, 0.085210) | **0.999** | -0.0013 | 0.0107, 0.0111 | 0.022 | 0.999, 0.999 | **no change** | none |

- Harm needed S ÷ B above 1.10 (the ln 1.10 bar binds: 2 SE is 0.013-0.022). The upward flag needed S ÷ B above 1.03
  with both S seeds above both B seeds; on every field S's seed mean is within 0.5 % of B's and S's seeds lie inside or
  below B's range. **Decision (PREREG section 6 with the 00:20 amendment): no harm and no upward flag on any field, so
  128 slices is a candidate for the scope freeze on two datasets (AirfRANS and DrivAerML Volume).** No downward flag either.
- Posed (beside; the same four runs, no independent number): the same labels, S ÷ B 1.001 / 0.995 / 0.998; posed ÷ native
  for S 1.0025 / 1.0019 (p), 1.0034 / 1.0033 (v), 0.9988 / 0.9989 (nu_t), for B 1.0027 / 1.0035, 1.0040 / 1.0033, 1.0001 /
  0.9998: the pose costs S what it costs B.
- Area: per case within 1.5e-5 of native in both arms (B) and 6.6e-6 (S), float roundoff between two passes over the same
  input (ISLA draws by area inside the model): **degenerate**, no claim.
- Beside, per case (seed means): pressure median S ÷ B 0.996, S lower on 52 % of the 48 cases; velocity 0.989, 58 %;
  nu_t 1.001, 50 %.
- S's two seeds sit much closer than B's on pressure (0.15 % against 2.2 %) and velocity (0.02 % against 1.3 %); on nu_t the
  spreads match (1.6 % and 1.5 %). The rule's SE is then set by B's spread.
- **Guard 3, eager re-score** (369191): S's compiled and eager scores agree to at most 5.9e-6 (s42) and 4.5e-6 (s43)
  relative per case and field (bar 1e-3), so the read uses the compiled scores of both arms, as registered.
- Checks passed: every row epoch 500; 48 infer_step rows with B's sample ids in every protocol; summary = case mean;
  SNAPSHOT_USED code_r090; DATASET_USED as B's; the scorer's extra holds `model.n_slices=128`; S's resolved configs differ
  from B's only in `run_id` and `n_slices` (both seeds); every training link resolved the geometry kernel to 'fused' on all
  four ranks, never 'eager'.
- Slots: 2 arms x 2 seeds = 4 independent training runs; three fields and two protocols are readings of those 4.

## Certification before the lanes (PREREG section 4; smoke 363505)

- Guard 1: `geo_kernel=auto resolved to 'fused'` on cuda:0-3 at 128 slices (smoke, certification run, both diags).
- Guard 2: block-wise compiled-against-eager float32 on 20 val cases of a 2-epoch S checkpoint: prediction deviation
  d = 4.8e-7 to 5.2e-7 of RMS(target) on every case (bar 1e-5), block error at most 4.3e-7. Beside, B's final checkpoint on
  the same cases: d 3.3e-6 to 4.5e-6, block at most 6.2e-7.

## R3, cost (descriptive)

| measure | B (512) | S (128) | S ÷ B |
|---|---|---|---|
| smoke 363505, same node nvl72d039-T18, 50 steps, 4 ranks: median step | 0.645 s | 0.403 s | **0.625** |
| smoke: peak allocated per rank | 128.6 GiB | 107.1 GiB | **0.832** |
| whole run, trainer `train_step` median (s42 / s43) | 0.593 / 0.593 s | 0.396 / 0.400 s | 0.67 |
| whole run, peak `mem_gb` | 155.1 | 133.5 | 0.86 |
| training GPU-h per lane (sacct elapsed x 4) | 40.4 / 40.9 (4-5 links; one failed on the NFS cache race) | 28.7 / 28.7 (2 links + a 9-s exit) | 0.70 |

On DrivAerML Volume, which runs the fused geometry kernel, quartering the slices costs 0.63-0.67 of the step and
0.83-0.86 of the memory. AirfRANS, on the unfused eager region, saved more (0.42 and 0.78). The fused kernel never
materializes the per-token, per-slice tensors, so the slice count weighs less in the 3-D step.

## R2, the far field and the wall-distance decades (descriptive; `farfield/farfield_table.md`, `farfield/farfield_{b512,s128}.json`)

Scorer space, count weights, CAP-AUDIT's decades of the datapipe's sdf in units of L_ref = 5 m, pooled over all points of
the 48 cases (each case's own draw whole, the 15.8 rule). The four runs see the same points (hash 5605832cdfbd); the dumps
reproduce the scorer's per-case pressure / velocity / nu_t to at most 2.0e-7.

| band | points % | nu_t e_R B (s42, s43) | nu_t e_R S | S ÷ B | nu_t share of e^2 B / S | velocity e_R B | velocity e_R S | S ÷ B |
|---|---|---|---|---|---|---|---|---|
| < 0.5 mm | 8.9 | 0.0230, 0.0249 | 0.0238, 0.0233 | 1.032, 0.933 | 0.2 / 0.2 % | 0.0829, 0.0837 | 0.0823, 0.0826 | 0.992, 0.987 |
| 0.5-5 mm | 41.3 | 0.1173, 0.1212 | 0.1132, 0.1197 | 0.965, 0.987 | 15.2-15.7 / 14.2-15.4 % | 0.0747, 0.0757 | 0.0744, 0.0748 | 0.995, 0.987 |
| 5 mm-5 cm | 28.2 | 0.0786, 0.0814 | 0.0787, 0.0812 | 1.002, 0.997 | 26.6-27.7 / 26.8-27.6 % | 0.0578, 0.0589 | 0.0587, 0.0582 | 1.016, 0.988 |
| 5-50 cm | 14.3 | 0.0585, 0.0591 | 0.0598, 0.0602 | 1.022, 1.018 | 6.1-6.2 / 6.4-6.5 % | 0.0269, 0.0272 | 0.0278, 0.0271 | 1.030, 0.998 |
| 0.5-5 m | 6.6 | 0.0433, 0.0438 | 0.0434, 0.0431 | 1.000, 0.986 | 4.6 / 4.5-4.7 % | 0.0340, 0.0332 | 0.0342, 0.0345 | 1.004, 1.041 |
| **> 5 m (far field)** | 0.7 | **0.1141, 0.1140** | **0.1144, 0.1140** | **1.003, 1.001** | **45.7-47.3 / 46.0-47.7 %** | 0.0209, 0.0207 | 0.0214, 0.0202 | 1.023, 0.974 |

- **The far-field eddy viscosity does not move with the slice count.** Beyond 5 m from the car S's nu_t e_R is 1.003 /
  1.001 times B's, seed-matched (case mean 1.005). The concern behind the brief (far points own slices from block 4 on, so
  fewer slices could starve them) does not show at the scorer's resolution.
- **Velocity per decade:** seed-matched S ÷ B 0.97-1.04 in every decade, case mean 0.99-1.02; no decade moves beyond the
  two-seed scatter. Pressure (in the table file) likewise, except the far field where pressure e_R is about 0.85-0.94 in
  both arms on 0.2-0.3 % of the e^2.
- Beside, about the standing itself: on full data at 512 slices, the 0.7 % of points beyond 5 m hold 46-47 % of ISLA's
  nu_t e^2. Their relative error (e_R 0.114) is about that of the 0.5-5 mm band (0.117-0.121); the share is large because
  the standardized nu_t is large out there: those 0.7 % of points hold about 27 % of the squared target
  (share of e^2 x e^2 / e_R^2). CAP-AUDIT found
  the same localization on its single case (49-66 %; e_R 0.035-0.044 there, a training-set floor, not comparable in
  level). Slice count does not move it; FLOOR-DRV's input lead (the floor and domain walls) remains the candidate.

## Predictions (PREREG section 7)

| id | prediction | p | outcome |
|---|---|---|---|
| P1 | smoke and certification pass first time | 0.85 | held |
| P2 | pressure: no change | 0.90 | held (1.002) |
| P3 | velocity: no change | 0.90 | held (0.995) |
| P4 | nu_t: no change | 0.87 | held (0.999) |
| P5 | no harm on all three fields | 0.82 | held |
| P6 | sub-bar flag fires on some field | 0.30 | did not fire |
| P7 | point estimates within [0.95, 1.05] on all three | 0.60 | held (0.995-1.002) |
| P8 | far-field nu_t e_R within 10 % of B, both seeds | 0.55 | held (1.003, 1.001) |
| P9 | smoke step ratio in [0.55, 0.80] | 0.60 | held (0.625) |
| P10 | smoke memory ratio in [0.80, 0.95] | 0.55 | held (0.832) |
| P11 | no event-guard stop | 0.88 | held (worst validation ratio 1.17, s43 epoch 88; B's 1.13) |

## What it means

- On DrivAerML Volume, 128 slices predicts as well as 512: every field within 0.5 % of the standing, the far-field nu_t and
  every velocity decade unchanged. With SLICES-AF (AirfRANS 0.998 / 0.997 / 0.969), the slice count now reads no harm on
  the two interior datasets, the 2-D one on the eager geometry path and the 3-D one on the fused path.
- Confidence: high that 128 slices costs no more than about 2-3 % on any field here. On pressure S's seeds lie inside B's
  seed range; on velocity both S seeds are below B's mean; on nu_t S matches B seed for seed (0.999, 0.999). The rule cannot
  resolve effects below about 2 %, and the check covers interior tasks only: no surface dataset (DrivAerML Surface, SHIFT-SUV,
  HiLift, DeepJEB++) and no solid-mechanics dataset was run at 128 slices.
- The cost argument for the scope freeze: 0.63-0.67 of the step, 0.83-0.86 of the memory and 0.70 of the GPU-h on 3-D
  DrivAerML Volume (0.42 / 0.78 on AirfRANS). Whether to change the reference configuration is the supervisor's and Peter's
  call; by the registered decision use, this check clears it on two datasets.

## Cost of this lane

Training 57.4 GPU-h; smoke and certification 0.80; scoring 0.54; eager re-score 0.31; far-field inference 0.86; two
cpu-short reductions (12 min). **About 60 GPU-h against the cap of 120.**
