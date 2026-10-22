# GTOPT: is GeoTransolver's AeroZoo pressure collapse optimizer-limited? (2026-10-22)

Sub-lane of AZ-DECIDE. A diagnostic only: no launcher, snapshot, dataset or run directory was changed;
`$T/runs/azc_full_gt_seed42` was only read (its final checkpoint copied, md5-checked). Pre-registration: `PREREG.md`
(md5 acb8f68f…, written 11:13 EDT before the first submission). Only full.val was scored.

## Answer

**No.** Retraining the output head with AdamW from the epoch-250 checkpoint, with everything else frozen, does
not pull pressure off the zero predictor. At lr 1e-3 it gets much worse; at lr 1e-4 it gets slowly worse. No scored
epoch at either lr comes near the 0.9 threshold: the lowest median is 1.0039 (A4 epoch 2), against E0's 1.0000.

**Reading against the registered rule** (the supervisor's): the per-case median pressure L2 did not fall below 0.9 at
either lr. So "the baseline columns stand as 'about a zero predictor', with the dynamic-range mechanism". The
fairness question does not go to Peter on this evidence.

## Results (full.val, 386 cases, infer.py, float32, 200k cells; `RESULTS_table.md`, `raw/summary.json`)

| arm | epoch | p L2 median | p L2 mean | share p < 0.9 | ‖p̂‖/‖p‖ median | u L2 median | u L2 mean |
|---|---|---|---|---|---|---|---|
| E0 (no fine-tuning, epoch 250) | 0 | **1.0000** | 1.311 | 0.052 | 0.018 | 0.5905 | 0.6099 |
| A3 AdamW lr 1e-3 | 1 | 1.2202 | 16.09 | 0.005 | 0.424 | 0.7033 | 0.7298 |
| A4 AdamW lr 1e-4 | 1 | 1.0054 | 1.534 | 0.008 | 0.048 | 0.5862 | 0.6132 |
| A4 AdamW lr 1e-4 | 2 | 1.0039 | 2.426 | 0.080 | 0.125 | 0.5808 | 0.6102 |
| A4 AdamW lr 1e-4 | 3 | 1.0256 | 3.474 | 0.013 | 0.151 | 0.5709 | 0.5991 |

Median pressure L2 by tercile of the case's pressure RMS (low 0.002-0.39 Pa, mid 0.42-16 Pa, high 16 Pa-28 kPa):

| arm | low | mid | high | ‖p̂‖/‖p‖ median, low / mid / high |
|---|---|---|---|---|
| E0 | 1.092 | 0.999 | 1.000 | 0.56 / 0.018 / 0.001 |
| A3 e1 | 16.97 | 1.223 | 1.003 | 16.7 / 0.40 / 0.025 |
| A4 e1 | 1.557 | 1.007 | 1.000 | 0.99 / 0.045 / 0.004 |
| A4 e3 | 3.148 | 1.035 | 1.000 | 2.67 / 0.16 / 0.010 |

- **The head can move.** AdamW grows the pressure row (norm 0.0015 → 0.016 at lr 1e-3, and → 0.0055 at lr 1e-4).
  The median ‖p̂‖/‖p‖ rises 8× to 23×.
- **But the amplitude does not follow the case's scale.** The low-RMS cases overshoot (ratio 2.7-17), and the
  high-RMS cases stay at ratio 0.001-0.025, L2 1.000. A larger pressure output only moves cases from L2 ≈ 1 to
  L2 > 1. That is the dynamic-range mechanism: the frozen trunk's LayerNorm-ed features give the linear head no
  amplitude that spans 4-7 decades.
- **The training loss agrees.** Train-step pressure loss (all-rank mean over 4 cases per step) has a p50 of 4.9-7.7
  at lr 1e-3 and 1.23-1.41 at lr 1e-4, never below 1 (`RESULTS_table.md`).
- **Velocity barely moves.** The lr 1e-4 head improves the velocity median from 0.5905 to 0.5709 in 3 epochs, still
  falling. lr 1e-3 makes velocity worse.
- **The scorer checks out.** The wrapper's physical-unit relative L2 matches infer.py's per-case metric to < 1e-5
  on every case, and every arm has the same 386 cases.

**Predictions** (PREREG): (1) median < 0.9, p 0.15: no. (2) median < 0.99, p 0.35: no. (3) A3 mean below E0's,
p 0.6: no (16.1). (4) A3 ‖p̂‖/‖p‖ median > 0.1, p 0.35: yes (0.42). (5) A4 velocity within ±0.05, p 0.7: yes (−0.020).

**Confidence and limits.** High that head-only re-optimization cannot fix pressure: this holds at both lrs and on
every epoch, and the per-tercile pattern shows why. The test does **not** cover two cases:
- **A trunk trained with another optimizer from the start.** Basecheck's Muon hypothesis is about the output row
  during training. Here the trunk's features were already shaped by a run with a dead pressure row.
- **Adam's own step.** AdamW is also scale-free per coordinate (step ≈ lr·m/√v). Strictly, the test shows that "a
  per-coordinate adaptive step on the head" does not help either, not that no optimizer could.

So the result supports the dynamic-range reading. It does not prove that a full retrain with another optimizer would
also fail; that test would be a full GeoTransolver run, about 250 epochs.

## Method

- **Driver** (`scripts/gtopt_train.py`). The recipe's train.py cannot switch optimizer or freeze parameters by
  config: train.py:953 always calls `build_muon_optimizer`. So a minimal driver imports `train` and replaces
  `initialize_from_checkpoint`, `build_muon_optimizer` and `val_epoch` in its namespace, then calls `train.main`
  with the run's overrides.
  - `initialize_from_checkpoint` is the recipe's own loader, then a freeze of everything outside the head, before
    DDP wraps the model.
  - `build_muon_optimizer` returns AdamW over the head only.
  - `val_epoch` is skipped. In job 334505 this replacement was undone by `Profiler.initialize()`, which re-binds
    every `@profile` name, so A3 ran the in-loop validation. That changes logging only.
- **Head**: `ln_mlp_out[0]` = LayerNorm(256) + Linear(256, 7), built at
  `code_azv2/physicsnemo/models/geotransolver/geotransolver.py:612-621` and applied at :902-905. It has 2,311
  trainable parameters; the other 9,053,692 are frozen (asserted). The original run's pressure row norm was 0.0015.
- **AdamW**: betas (0.9, 0.999), eps 1e-8, weight decay 1e-4 (the recipe's AdamW settings). The StepLR is constant
  over these epochs.
- **Unchanged from the lane**: case_relative_l2 with velocity weight 3, aerozoo_uvolume full.train, float32, 200k
  points, seed 42, 4 GPUs × batch 1 (678 steps per epoch).
  - One loader change in job 2: 20 workers per rank instead of 12. The data are the same; epochs took 4.5 min
    instead of 9.
- **Scoring** (`scripts/gtopt_infer.py`): the recipe's infer.py under torchrun with 4 ranks, shards merged and
  deduplicated (388 → 386). The wrapper changes two things only:
  - `attach_and_save` writes nothing and records the physical ‖p̂‖/‖p‖ and the relative L2 per case.
  - The norm-stats comparison runs on CPU. infer.py's own comparison crashes on ranks above 0 because
    norm_stats.pt loads onto cuda:0; a mismatch now raises instead.
- **Reduction**: `scripts/reduce.py raw` → `RESULTS_table.md`. Output-row norms come from `scripts/headnorms.py`.

## Amendment after partial results (deviations from PREREG)

- **A3 got 1 epoch, not 3.** Job 334505 (12.7 min) trained A3's first epoch. It was then cancelled for three
  reasons:
  - infer.py failed under torchrun (fixed above).
  - Epochs took 9 min, so 3+3 epochs and the scoring would not fit the 3 GPU-h cap.
  - The in-loop validation and train-step losses had already shown A3 diverging (pressure 4-60 per step).
  
  A3's epochs 2-3 were dropped with that loss curve already seen. The rule reads A3 on its one scored epoch;
  completing it would not plausibly change the reading, given A3 e1's mean of 16 and its rising train-step losses.
- **The PREREG baseline is not per case.** Its "in-loop baseline" (median 1.032) is a median of per-step
  **all-rank means over 4 cases**, not of per-case values (train.py:461-470 reduces each step across ranks). The
  official baseline is E0 (per-case median 1.0000). The same caveat applies to basecheck's "per-step (per-case)"
  curve quantiles: each step there is a 4-case mean.

## Jobs (LHR, short QoS, 4 × GB300)

| job | what | elapsed | GPU-h |
|---|---|---|---|
| 334505 (`isla.azdecide.gtopt`, nvl72d037-T13) | A3 epoch 1. E0 scoring failed (torchrun device bug). Cancelled. | 12:44 | 0.85 |
| 334714 (`isla.azdecide.gtopt2`, nvl72d008-T10) | A4 3 epochs; infer.py on E0, A3 e1, A4 e1-e3; head norms | 17:56 | 1.20 |

Total 2.05 GPU-h.

## Files

- **Here**: `PREREG.md`, `RESULTS_table.md`, `scripts/` (both sbatch files, the drivers, `headnorms.py`,
  `reduce.py`), and `raw/` (3.8 MB: infer metrics and ratio files, train metrics and logs, job logs,
  `summary.json`).
- **Cluster**: `$T/kernel/azdecide/gtopt/` (7 MB) holds the scripts, `logs/`, `runs/gtopt_A{3,4}/` (logs and
  metrics), `scores/`, and `job334505/` (the failed E0 attempt).
  - The fine-tuned checkpoints and the init copy were deleted after scoring.
  - `ACTIVE_EXPERIMENTS.tsv`: the row was added at launch and removed at the end.
