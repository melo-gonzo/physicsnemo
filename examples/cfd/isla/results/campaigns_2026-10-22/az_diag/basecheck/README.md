# BASECHECK: are the AeroZoo baselines' ~1.0 scores a pipeline bug or their floor? (2026-10-22)

Sub-lane of AZ-DECIDE, answering the reviewer's M2_methods item 1. Read-only on runs, snapshots and datasets: no
launcher, recipe, dataset or score file was edited, and no ISLA-DTT file was read.

## Answer

**No pipeline bug.** Every field reaches the loss through its own output channel, the normalization matches both the
dataset YAML and the checkpoint's `norm_stats.pt`, and no head is frozen: the pressure row of the output layer gets the
largest gradient of any row. My driver reproduces the scorer's per-case L2s: exact for Transolver, within 1.6 % for
GeoTransolver, where the gap comes from the SubsampleMesh boundary draw.

**The baselines are zero predictors, literally.** At epoch 250 the output-layer weight rows for pressure, ν_t, k and
ω are about 0. Norm ratios below are the medians over the cases in `cases_table.md`.

| run | ‖W‖ rows u_x u_y u_z / p ν_t k ω | ‖p̂‖/‖p‖ (range) | ‖û‖/‖u‖ (range) |
|---|---|---|---|
| azc_full_gt_seed42 | 1.46 1.37 1.37 / 0.00 0.03 0.02 0.00 | 0.04 (0.002–0.5) | 0.63 (0.35–0.84) |
| azc_nonbox_gt_seed42 | 4.70 4.69 4.68 / 0.00 0.03 0.03 0.00 | 0.01 | 1.04 (0.84–1.66) |
| azc_full_t_seed42 | 0.62 0.59 0.55 / 0.00 0.00 0.00 0.00 | 0.018 | 0.011 (0.0004–0.08) |

- **Transolver's predictions are constants.** Its pressure prediction is the output bias, 9e-6 × 1129 Pa = **0.0100
  Pa** in every case, whatever the case's RMS pressure (0.01 Pa to 17 kPa).
- **The mean above 1 comes from the slowest cases.** That constant exceeds their whole field: case a48a547f, with
  an RMS pressure of 0.0126 Pa, reads 1.34 for Transolver and 3.33 for GeoTransolver seed 42.
- **Transolver's velocity is also essentially 0** (0.02 m/s).
- **GeoTransolver learns velocity, partly.** ‖û‖/‖u‖ tracks a 200× range of inlet speed, and the cosine with the
  truth is about 0.9.

## Reading per baseline

- **Transolver: (a), at its expected floor.** It is blind to the inlet speed. Even with a perfect flow shape, the
  best scale-blind velocity (one constant c = 0.28 m/s over the training cases' RMS speeds) scores a mean of 0.77 and
  a median of 0.89. Transolver collapses a little further, to the zero predictor (median 0.9975). Its pressure,
  ν_t, k and ω are the same collapse.
- **GeoTransolver: neither (a) nor (b). It hits a floor set by the protocol and the optimizer, not by its inputs.**
  - **The scale is in its inputs.** Its tokens carry the inlet values, and its velocity amplitude follows them.
    Pressure is not fixed by any information limit either.
  - **One dataset-wide scale cannot cover the pressure range.** Under that scale, the training cases' RMS pressure
    spans **8 decades** (2.6e-4 Pa to 2.8e4 Pa; 6.2 decades from the 1st to the 99th percentile). The median is
    0.0027 standardized units, and 10 % of the cases lie below 2e-5 (0.024 Pa).
  - **The loss leaves only the zero predictor.** A case-relative loss scores any absolute output error δ as δ/RMS,
    so the gradient weights cases by 1/RMS: the slowest 100 of the 3,332 cases carry 51 % of it. To beat zero on
    those cases, the head would have to output values of order 1e-5 with relative precision.
  - **What the training curves show** (`curves_table.md`):
    - **Epochs 0-99 (lr 1e-3):** the mean per-case pressure loss is 20-50 and its median 4-10, far worse than zero.
    - **After each LR decay** (×0.1 at epochs 100 and 200), it falls back to about 1.
    - **Final epoch:** the step median is 1.001-1.016 and the 10th percentile 0.990-0.997 on all 16 lanes. No lane
      ever has a median below 0.99.
    - **The output weights are a likely contributor.** The recipe's Muon step updates 2-D weights by orth(G), whose
      7 rows all have norm 1.00 (measured). So the pressure row moves by a fixed step set by the learning rate, not
      by the precision it needs. This is a hypothesis; no ablation was run.
  - **Velocity is optimization-limited too.** Velocity's per-case scale spans 3 decades.
    - **Plateau:** every lane sits at the zero-predictor plateau (step median ≥ 0.99) until an epoch that varies by
      lane: nonbox 40-42, crossflow and dual-inlet 100-121, bank_zero_to_cad 118-120, full 172-174, blobs 152/181,
      reynolds 115/197.
    - **Still falling at the end:** on `full`, velocity falls from 0.88 at epoch 200 to 0.59 at epoch 249, still
      dropping about 0.004 per epoch.
    - **Seed spread:** final validation velocity is 0.48 against 0.94 on reynolds (seeds 42/43) and 0.93 against
      0.52 on blobs.
    - So GeoTransolver's velocity numbers are a seed lottery around an under-converged run, not a converged
      capability.
- **Minor, not material.** `CASE_RELATIVE_FLOOR` = 1e-6 standardized units, which is 1.1e-3 Pa of pressure. It floors
  the loss denominator of fewer than 1 % of the training cases (minimum 2.3e-7 units), so on those cases the training
  loss and the scored metric differ. It does not drive the collapse.

**What it means for the program.** The AeroZoo baseline columns should be labelled as zero predictors on pressure (and
on ν_t, k and ω), not as competitors' accuracy. For Transolver this is the documented floor of PREREG_v2 Q2. For
GeoTransolver, the honest wording is: "under the compliant protocol (no per-case scale, one dataset-wide target
standardization, case-relative loss), GeoTransolver as released cannot represent pressure spanning 8 decades and
converges to zero. Its velocity is under-trained, with a large seed spread."

**Confidence.**
- High: no code bug, and the baselines are zero predictors.
- Medium: the mechanism (dynamic range × case-relative loss × fixed-step optimizer).
- Not tested: whether a different schedule would let GeoTransolver learn pressure. A cheap test, not run here, would
  be one GeoTransolver lane on `full` at a lower constant learning rate, or with AdamW on the head.

## Method and jobs (LHR)

1. **Training curves.** CPU job **333585** (`isla.azdecide.basecheck.curves`, cpu-short, about 3 min) ran
   `scripts/curves.py`, launched by `scripts/curves.sbatch`, over all 16 lanes' `metrics.jsonl`.
   - It recorded every train_summary and val_summary row, and per epoch the p10/p50/p90 of the per-step (per-case)
     `loss/<field>`.
   - Reduction: `scripts/reduce_curves.py` → `curves_table.md`. Velocity's weight of 3 is divided out.
2. **Outputs, gradients and norm stats at epoch 250.** GPU job **333690** (`isla.azdecide.basecheck`, short QoS, 4
   GPUs, 4 min 43 s, about 0.31 GPU-h, nvl72d046-T02) ran `scripts/basecheck.py`, launched by
   `scripts/basecheck.sbatch`.
   - **Runs:** azc_full_gt_seed42, azc_full_gt_seed43, azc_nonbox_gt_seed42 and azc_full_t_seed42.
   - **Environment:** the lanes' own (`.venv-recipe`, code_azv2, recipe_az_c, the `kernel/azp/lanes.tsv` overrides),
     float32, 200,000 points, the train config.
   - **Cases:** 12 val and 12 test cases, evenly spaced in the sampler, and 6 random train cases with gradients
     (train mode, the training loss).
   - **Reduction:** `scripts/reduce_cases.py` → `cases_table.md`. It asserts L2² = r² + 1 − 2rC for every case and
     field.
   - **Scorer check:** `scripts/check_vs_scorer.py` compares the test-case L2s with the scorer's rows.
3. **Pressure scale distribution.** Computed from `../scales/scales_train_full.tsv`, one row per train case (3,332
   unique cases; per-case RMS = `out_factor` × `head_rms`).

## Files

- **Here:** `curves_table.md`, `cases_table.md`, `scripts/`, and `raw/` (curves JSONs and per-case JSONL, a few MB).
- **Cluster:** `$T/kernel/azdecide/basecheck/` (scripts, `curves/`, `out/<run>/`, `logs/`).
