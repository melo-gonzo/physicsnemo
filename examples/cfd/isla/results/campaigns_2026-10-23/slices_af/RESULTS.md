# SLICES-AF results (written 2026-10-23 23:57 EDT)

ISLA 0.9.0 with `n_slices=128` (S) against the eager-trained AirfRANS standing at 512 slices (B, r090eag_af_isla), full
AirfRANS (full_train -> the 200 full_test cases), seeds 42 / 43, 500 epochs, lr 3.75e-4, scored native / area / posed,
float32, epoch 500. Pre-registration PREREG.md (md5 0f5c94ff, approved before any job). Jobs: JOBS.md.

## Registered reading (R1, native decides): no change on every field

From `read_slices.py` (md5 f1097cba) -> `reading.md`, `reading.json`. CM = the scorer's case-mean relative L2.

| field | B (s42, s43) | S (s42, s43) | S ÷ B | D | s_B, s_S | 2 SE | seed-matched S ÷ B | label |
|---|---|---|---|---|---|---|---|---|
| **pressure** | 0.012479 (0.013163, 0.011794) | 0.012450 (0.012106, 0.012794) | **0.998** | -0.0023 | 0.078, 0.039 | 0.123 | 0.920, 1.085 | **no change** |
| velocity (in-plane) | 0.0063554 (0.0064407, 0.0062701) | 0.0063333 (0.0063734, 0.0062932) | 0.997 | -0.0035 | 0.019, 0.009 | 0.030 | 0.990, 1.004 | **no change** |
| nu_t | 0.015353 (0.015738, 0.014968) | 0.014885 (0.015241, 0.014528) | 0.969 | -0.0310 | 0.036, 0.034 | 0.069 | 0.968, 0.971 | **no change** |

- Helps on pressure would have needed D < ln 0.90 = -0.105 and |D| > 0.123, so S ÷ B below 0.88. S sits at 0.998.
- Harm is ruled out on all three fields; the largest point estimate is the nu_t gain of 3 % (both seeds 0.97), inside the
  bar and inside 2 SE.
- Posed (beside; the same four runs, no independent number): the same labels and ratios (0.998 / 0.996 / 0.970);
  posed ÷ native 1.0022-1.0025 on pressure for both arms, so pose costs S what it costs B.
- Area: equals native exactly for both arms on every field and seed: **degenerate**, no claim.
- Beside, per case (seed means): pressure median S ÷ B 1.006, S lower on 47 % of the 200 cases; velocity 1.002, 52 %;
  nu_t 1.033, 52 %.
- S's two pressure seeds sit 5.5 % apart (B's 11.6 %); S's s42 is below B's s42 and S's s43 above B's s43, so S lands
  at B's mean.
- Checks passed: every row epoch 500, 200 infer_step rows with B's sample ids, summary = case mean, SNAPSHOT_USED
  code_cmpaf_fix, DATASET_USED as B's, the scorer's extra holds `model.n_slices=128`; S's resolved configs differ from B's
  only in `run_id` and `n_slices` (both seeds).
- Slots: 2 arms x 2 seeds = 4 independent training runs; three fields and two protocols are readings of those 4.

**Consequence (PREREG section 6): no stage 2.** 128 slices is not adopted for accuracy. S harms no field, so its cost is
reported for the scope freeze as a cost result on one dataset only; adopting it would also need a no-harm check on a second
dataset, which the supervisor decides and which is not registered here.

## R3, cost (descriptive)

| measure | B (512) | S (128) | S ÷ B |
|---|---|---|---|
| smoke 354977, same node nvl72d037-T14, 50 steps, 4 ranks: median step | 0.761 s | 0.323 s | **0.424** |
| smoke: peak allocated per rank | 43.0 GiB | 33.4 GiB | **0.777** |
| whole run, trainer `train_step` median (s42 / s43) | 0.754 / 0.756 s | 0.312 / 0.312 s | 0.41 |
| whole run, peak `mem_gb` | 45.4 | 34.8 | 0.77 |
| training GPU-h per lane (sacct elapsed x 4) | 88.9 / 89.4 (7 links; one cancelled by another user) | 37.0 / 37.1 (3 links, last two in short QoS) | 0.42 |

Quartering the slices cuts the AirfRANS step to 0.42 and memory to 0.78 of the reference. AirfRANS runs the eager (unfused)
geometry region, so the N x M routing and geometry tensors dominate its step; on datasets that run the fused geometry
kernel (3-D) the saving may be smaller, which this lane did not measure.

## R2, near-wall band (descriptive; `nearwall/nearwall_table.md`, `nearwall_{b512,s128}.json`)

Pressure e_R by wall-distance band (chords) over the 200 cases, CAP-AUDIT's definitions (count weights, the stored sdf).
The four runs see the same points (hash ab498a7b9722); per-case pressure and velocity e reproduce the scorer to 2.2e-7.

| band | B pooled (s42, s43) | S pooled (s42, s43) | S ÷ B | B case-mean | S case-mean | S ÷ B |
|---|---|---|---|---|---|---|
| < 1e-4 | 0.0213, 0.0156 | 0.0151, 0.0154 | 0.83 | 0.0134, 0.0120 | 0.0123, 0.0129 | 0.99 |
| 1e-4 to 1e-3 | 0.0190, 0.0130 | 0.0131, 0.0139 | 0.85 | 0.0126, 0.0112 | 0.0115, 0.0122 | 1.00 |
| **1e-3 to 0.1** | **0.0137, 0.0106** | **0.0110, 0.0110** | **0.90** | 0.0122, 0.0112 | 0.0115, 0.0121 | **1.01** |
| 0.1 to 1 | 0.0221, 0.0168 | 0.0175, 0.0193 | 0.95 | 0.0257, 0.0213 | 0.0217, 0.0241 | 0.98 |
| > 1 | 0.0591, 0.0502 | 0.0497, 0.0555 | 0.96 | 0.115, 0.101 | 0.100, 0.105 | 0.95 |

- The pooled near-wall gain (0.90) comes from B's seed 42 alone: on seed 43, S's band error is above B's (0.0110 against
  0.0106). The case mean, which weighs each airfoil equally, is unchanged (1.01). P6 fails.
- So the single-case lead (band 0.0017-0.0018 against 0.0024-0.0031) does not carry to the cross-case test: on full data
  the band moves with the seed, not with the slice count. As in CAP-AUDIT, the band from 1e-3 to 0.1 chords holds 39 % of
  the points; the innermost two bands hold 77-85 % of the pooled pressure e^2 in both arms.

## Predictions (PREREG section 7)

| id | prediction | p | outcome |
|---|---|---|---|
| P1 | smoke passes first time | 0.90 | held |
| P2 | pressure: no change | 0.70 | held (helps 0.20, harms 0.10: neither) |
| P3 | velocity: no change | 0.85 | held |
| P4 | nu_t: no change | 0.83 | held |
| P5 | pressure point estimate below 1 | 0.55 | held, barely (0.998) |
| P6 | S's pooled band below B's on both seeds | 0.40 | failed (s43 above) |
| P7 | smoke step ratio in [0.60, 0.85] | 0.65 | failed: 0.424, cheaper than predicted |
| P8 | memory ratio below 0.80 | 0.60 | held (0.777) |
| P9 | no event-guard stop | 0.85 | held (worst validation ratio 3.23 at epoch 80 on s42, B's 3.05 at epoch 80; s43 1.56) |
| P10 | stage 2 proposed | 0.17 | not proposed |

## What it means

- 128 slices predicts full AirfRANS as well as 512, not better: every field within 3 %, every label no change. The
  single-case s128 lead was seed spread, as CAP-AUDIT warned. Confidence: high that there is no gain of 12 % or more on
  pressure (S's seeds bracket B's mean); the two-seed rule cannot resolve effects below about 10 %.
- 128 slices costs 0.42 of the step and 0.78 of the memory on AirfRANS, harming nothing at the rule's resolution. For the
  scope freeze this is a speed argument on one dataset only (Peter's third priority), which needs a second-dataset no-harm
  check before any change to the reference configuration. CAP-AUDIT's single-case DrivAerML Volume ladder also found
  quartering the slices costs nothing (1.03 / 0.99 / 1.04 per halving), one seed, single-case.

## Cost of this lane

Training 74.1 GPU-h; smoke 0.37; scoring 0.63; near-wall inference 0.60 (B) + 0.49 (S); CPU reductions (cpu-short, 33
min including one 20-min timeout). **About 76 GPU-h against the cap of 200.**
