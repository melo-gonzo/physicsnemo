# PREDDIAG: is ISLA's ≈√2 velocity error on profile_crossflow "right speed, wrong direction"? (2026-10-22)

Sub-lane of AZ-DECIDE. Written 2026-10-22 09:18 EDT.

## Conclusion

1. **No.** ISLA predicts roughly the right *direction* and about **twice the speed**. On profile_crossflow its RMS speed
   ratio |u_pred|/|u_true| has a median of 2.26 (seed 42) and 2.04 (seed 43) over 30 cases (median pointwise ratio 2.2 / 2.0).
   Its global cosine is 0.79 / 0.71, the same as GeoTransolver's 0.71, which gets the speed right (ratio 0.99).
   √2 is a coincidence: L² = r² + 1 − 2rC with r ≈ 2 and C ≈ 0.75 gives about 2. At the true speed, ISLA's directions
   would score 0.65 / 0.77, against 0.77 for GeoTransolver (column l2_speed_matched).
2. **The direction comes from the inlet, for both models.** The mean predicted velocity lies within 3–9° (ISLA) and
   1.4° (GT) of g, ISLA's own drive direction: g = Σ a·v / |Σ a·v| over the inlet's prescribed bc_velocity, which is the
   inlet normal in every case. The true mean flow turns about 35° toward the adjacent outlet (median). Neither model turns it: GT
   predicts no velocity normal to g at all (RMS 0.01 against a true 0.61, in units of the true RMS speed). ISLA's
   excess error lies along g (err_g 1.2–1.3 against err_perp 0.53–0.59). The predicted mean component along g is
   1.8–1.9, against a true 0.71. Relative to the inlet's RMS speed U (ISLA's velocity scale), the true RMS speed is
   0.89 on both splits. ISLA predicts 1.9–2.1 U on crossflow, 1.0–1.1 U on nonbox; GT predicts 0.9 U on both.
3. **family_non_box's worst cases are magnitude errors too.** The direction is right there: angle between mean
   vectors < 1°, cos_of_means 1.00. The worst 20 have speed ratios of 1.9 (s42) and 1.5 (s43), and err_g dominates. The random 10 have ratios of
   1.3 / 1.1 with L2 0.59 / 0.49 (GT: 1.11, 0.58). ISLA's failure on these OOD splits is the velocity *amplitude*, not its frame.
   Why the amplitude extrapolates badly (for example, invariants like g·n_outlet = 0, never seen in training with an opposed outlet) is
   a hypothesis this lane did not test.

**What is new in profile_crossflow** (AeroZoo `splits/README.md`, book `01-tour.qmd` / `07-posing.qmd`): the test side is
every *crossflow box*: the outlet is on a face **adjacent** to the inlet, with one to three other faces no-slip, so the stream turns
through a right angle around the body. Training and validation use the other four box layouts (open, ground-effect,
confined, dual-inlet), all with the outlet **opposite** the inlet, plus the spheres, hemispheres and ducts. The inlet
itself is not new: its area-mean velocity is normal to the inlet face in every case (angle_g_inlet_normal ≈ 0°), with
angle_g_exit = 90°.

## Method

- Cases: `cluster/cases_<split>.tsv` (`make_cases.py`): the 20 worst per split from `../worst/cases.tsv`, plus 10 random
  test cases per split (numpy `default_rng(20261022)`, drawn from the rest of the test side). Global indices come from
  the scorer's sample ids.
- Driver `cluster/diag.py`, launched by `cluster/preddiag.sbatch`: the environment, snapshots and infer.py overrides of
  the scorer (`kernel/azdecide/eval_az{i,p}/eval_test.sbatch`). ISLA: code_r090/recipe_r090; GT: code_azv2/recipe_az_c.
  Settings: the lane's model / dataset / split overrides with `val_split=<split>.test`, float32, 200,000 interior points,
  a writable Hydra cwd. The setup is infer.py's `main()` line by line: build_dataloaders, the norm_stats.pt override, the
  model with the injected output stats and isla_constants (r090), load_checkpoint (epoch 250), and the same
  MetricCalculator. It walks the scorer's sampler in order.
  - **Draw reproduction.** GeoTransolver's pipeline starts with SubsampleMesh (the per-boundary 50,000-face draw), whose
    generator advances from sample to sample. Exact scores therefore need every earlier sample's draw. The driver
    replays each unselected case before the last selected one through that transform only (reader + SubsampleMesh,
    about 4 s per case). ISLA's model config drops SubsampleMesh, and its reader draw depends only on
    (seed, epoch, index). Its eval-mode token draw has a fixed start, so it skips unselected cases.
  - Diagnostics use the redimensionalized (SI) velocity on the scored points. Unweighted: the target measure is None,
    and the driver's own L2 equals the metric's to ≤1e-7.
  - Inlet: area-weighted mean of `bc_velocity` on the case's inlet mesh, the same formula as ISLA's g. GT's inlet is the
    50k-face draw, ISLA's the full mesh.
  - Frame for the per-axis split: e1 = g; e2 = the true mean velocity's part normal to g (on crossflow, the exit side);
    e3 = e1 × e2. err_k = ‖p_k − t_k‖/‖t‖, so err_g² + err_perp² + err_third² = L².
- Reduction: `fetch.sh` (rows into `raw/`), `reduce_preddiag.py` (→ `predictions_diag.tsv`, `medians.md`),
  `percase_table.py` (→ `percase.md`).

## Equivalence check

180 (case, model) rows, the 6 runs × 30 cases. For each, the driver's velocity_l2 and pressure_l2 were compared with the
scorer's infer_step row for the same sample_id in `../scores/p200k_fp32_test_azc/<run>/<run>/metrics.jsonl` (the block
after the last config row, epoch 250). **Maximum relative difference: 3.4e-5** (velocity and pressure; ≤1e-4 required).
The GT rows match to 0.0, which confirms that the SubsampleMesh replay reproduced the scorer's draws. ISLA nonbox
differs by ≤3.4e-5, and ISLA profile_crossflow matches to 0.0. Columns eq_rel_vel / eq_rel_p are in the tsv.

## Jobs (LHR)

- 331798 (`isla.azdecide.preddiag`, short QoS, 4 GPUs, 55 min, nvl72d025-T11): all four ISLA runs (about 4 min each),
  GT profile_crossflow (35 min), and GT nonbox up to sampler position 920, where the unit's 3,300 s timeout stopped it.
- 331933 (same script, `-t 00:40:00`, dependency afterany:331798, 8.7 min, nvl72d036-T14): resumed GT nonbox from
  the saved generator state at position 921 and finished it. The completed units were skipped.
- Real work: about 1.5 GPU-h (the GT replays). Allocated: about 4.2 GPU-h, because the ISLA GPUs idled after 8 min.
- Cluster files: `$T/kernel/azdecide/preddiag/` (diag.py, preddiag.sbatch, cases_*.tsv, out/<run>/diag.{jsonl,log},
  logs/). No prediction arrays were written (`+save_predictions=false`), and the resume states were deleted.

## Medians per split and model (all = 30 cases; worst = the 20 from ../worst; random = 10)

Speed ratio = RMS |u_pred| / RMS |u_true|. cos pt = mean pointwise cosine. cos_global = Σp·t/(‖p‖‖t‖).
cos_of_means = cosine of the point-averaged vectors. angle_g_* in degrees. err_* as in Method.

| split | run | cases | sel | vel_l2 | l2_speed_matched | speed_ratio_rms | speed_ratio_median | cos_pointwise_mean | cos_global | cos_of_means | angle_g_true_mean | angle_g_pred_mean | angle_true_pred_means | err_g | err_perp | err_third |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| profile_crossflow | isla_seed42 | 30 | all | 1.63 | 0.65 | 2.26 | 2.22 | 0.757 | 0.788 | 0.916 | 35 | 8.72 | 23.6 | 1.34 | 0.525 | 0.582 |
| profile_crossflow | isla_seed42 | 10 | random | 1.28 | 0.641 | 1.88 | 1.8 | 0.775 | 0.792 | 0.922 | 51 | 31.4 | 22.7 | 0.884 | 0.525 | 0.582 |
| profile_crossflow | isla_seed42 | 20 | worst | 1.72 | 0.65 | 2.35 | 2.31 | 0.721 | 0.788 | 0.911 | 30.3 | 7.07 | 24.4 | 1.41 | 0.559 | 0.595 |
| profile_crossflow | isla_seed43 | 30 | all | 1.44 | 0.767 | 2.04 | 2 | 0.712 | 0.706 | 0.855 | 35 | 2.93 | 31.3 | 1.21 | 0.591 | 0.538 |
| profile_crossflow | isla_seed43 | 10 | random | 1.29 | 0.808 | 1.8 | 1.63 | 0.689 | 0.674 | 0.758 | 51 | 12.3 | 40.7 | 1.05 | 0.604 | 0.529 |
| profile_crossflow | isla_seed43 | 20 | worst | 1.62 | 0.7 | 2.15 | 2.12 | 0.73 | 0.755 | 0.873 | 30.3 | 1.97 | 29.2 | 1.32 | 0.567 | 0.541 |
| profile_crossflow | gt_seed42 | 30 | all | 0.807 | 0.766 | 0.985 | 0.966 | 0.707 | 0.707 | 0.815 | 35 | 1.41 | 35.4 | 0.434 | 0.614 | 0.0937 |
| profile_crossflow | gt_seed42 | 10 | random | 0.86 | 0.932 | 0.761 | 0.843 | 0.615 | 0.566 | 0.659 | 51 | 1.84 | 48.7 | 0.343 | 0.781 | 0.097 |
| profile_crossflow | gt_seed42 | 20 | worst | 0.765 | 0.7 | 1.07 | 1.02 | 0.719 | 0.755 | 0.882 | 30.3 | 1.39 | 28 | 0.461 | 0.511 | 0.0925 |
| nonbox | isla_seed42 | 30 | all | 0.932 | 0.514 | 1.68 | 1.58 | 0.95 | 0.868 | 1 | 0.457 | 0.497 | 0.553 | 0.874 | 0.139 | 0.134 |
| nonbox | isla_seed42 | 10 | random | 0.59 | 0.47 | 1.33 | 1.29 | 0.966 | 0.89 | 1 | 0.669 | 0.583 | 0.619 | 0.558 | 0.0745 | 0.0726 |
| nonbox | isla_seed42 | 20 | worst | 1.81 | 0.761 | 1.88 | 1.81 | 0.795 | 0.71 | 1 | 0.401 | 0.431 | 0.468 | 1.3 | 0.148 | 0.146 |
| nonbox | isla_seed43 | 30 | all | 0.626 | 0.499 | 1.3 | 1.14 | 0.955 | 0.875 | 1 | 0.457 | 0.454 | 0.761 | 0.596 | 0.0981 | 0.0938 |
| nonbox | isla_seed43 | 10 | random | 0.487 | 0.442 | 1.11 | 1.05 | 0.965 | 0.902 | 1 | 0.669 | 0.434 | 0.886 | 0.48 | 0.0652 | 0.06 |
| nonbox | isla_seed43 | 20 | worst | 1.67 | 0.756 | 1.5 | 1.33 | 0.795 | 0.714 | 1 | 0.401 | 0.524 | 0.629 | 1.05 | 0.104 | 0.103 |
| nonbox | gt_seed42 | 30 | all | 0.652 | 0.555 | 1.11 | 1.06 | 0.956 | 0.846 | 0.998 | 0.457 | 3.83 | 3.64 | 0.541 | 0.0806 | 0.0628 |
| nonbox | gt_seed42 | 10 | random | 0.575 | 0.461 | 1.11 | 1.03 | 0.975 | 0.892 | 0.998 | 0.669 | 2.83 | 3.64 | 0.541 | 0.0748 | 0.0564 |
| nonbox | gt_seed42 | 20 | worst | 0.778 | 0.679 | 1.11 | 1.39 | 0.825 | 0.769 | 0.998 | 0.401 | 4.74 | 3.81 | 0.533 | 0.0904 | 0.0751 |

## Per-case table

Speed ratio is the RMS ratio, cos pt the mean pointwise cosine, angle g-pred the angle between g and the mean predicted
velocity. Every column (component L2s, median ratio, cos_of_means, exit angles, frame components) is in `predictions_diag.tsv`.

| split | idx | case | selection | angle g-true mean | isla42 vel L2 | isla42 speed ratio | isla42 cos pt | isla42 angle g-pred | isla43 vel L2 | isla43 speed ratio | isla43 cos pt | isla43 angle g-pred | gt42 vel L2 | gt42 speed ratio | gt42 cos pt | gt42 angle g-pred |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nonbox | 3 | 4 | random | 0.6 | 0.173 | 1.09 | 1.00 | 0.2 | 0.171 | 1.05 | 1.00 | 0.3 | 0.278 | 1.05 | 1.00 | 2.0 |
| nonbox | 74 | 82 | worst_velocity_l2 | 0.4 | 1.939 | 2.48 | 0.72 | 0.4 | 1.823 | 2.36 | 0.72 | 0.1 | 0.889 | 1.05 | 0.70 | 11.0 |
| nonbox | 124 | 137 | worst_pressure_l2 | 4.0 | 0.755 | 1.59 | 0.95 | 1.4 | 0.339 | 1.13 | 0.96 | 1.6 | 0.299 | 1.07 | 0.96 | 3.3 |
| nonbox | 495 | 568 | worst_velocity_l2 | 91.8 | 2.013 | 1.65 | -0.07 | 8.9 | 1.837 | 1.44 | -0.07 | 8.2 | 9.892 | 10.63 | 0.79 | 89.4 |
| nonbox | 496 | 569 | random | 0.0 | 0.339 | 1.07 | 0.99 | 0.2 | 0.123 | 1.01 | 1.00 | 0.1 | 0.124 | 0.97 | 0.99 | 4.1 |
| nonbox | 518 | 596 | random | 0.0 | 0.322 | 1.25 | 0.99 | 0.5 | 0.097 | 1.01 | 0.99 | 0.2 | 0.221 | 1.18 | 0.99 | 3.5 |
| nonbox | 519 | 599 | random | 0.4 | 0.484 | 0.71 | 0.95 | 0.6 | 0.528 | 0.63 | 0.96 | 0.5 | 0.643 | 1.18 | 0.94 | 12.9 |
| nonbox | 595 | 690 | random | 2.2 | 0.859 | 1.53 | 0.95 | 6.9 | 0.717 | 1.32 | 0.95 | 3.8 | 0.661 | 1.22 | 0.95 | 7.6 |
| nonbox | 692 | 801 | worst_pressure_l2 | 1.3 | 0.793 | 1.54 | 0.97 | 3.8 | 0.464 | 1.25 | 0.98 | 0.8 | 0.305 | 0.90 | 0.99 | 5.1 |
| nonbox | 929 | 1071 | random | 2.0 | 0.716 | 1.41 | 0.97 | 2.9 | 0.656 | 1.28 | 0.96 | 2.9 | 0.708 | 1.33 | 0.98 | 2.1 |
| nonbox | 1039 | 1194 | worst_velocity_l2 | 0.1 | 2.573 | 3.20 | 0.80 | 0.3 | 2.403 | 3.03 | 0.80 | 0.2 | 0.947 | 1.31 | 0.80 | 7.4 |
| nonbox | 1521 | 1735 | random | 0.1 | 0.488 | 0.71 | 0.97 | 0.2 | 0.447 | 0.72 | 0.97 | 0.2 | 0.756 | 0.27 | 0.97 | 4.3 |
| nonbox | 1816 | 2069 | worst_velocity_l2 | 0.4 | 2.529 | 2.98 | 0.66 | 0.1 | 2.301 | 2.76 | 0.67 | 0.2 | 3.731 | 4.12 | 0.66 | 3.2 |
| nonbox | 1913 | 2185 | worst_velocity_l2 | 0.3 | 2.558 | 3.03 | 0.62 | 0.3 | 2.400 | 2.86 | 0.62 | 0.0 | 2.265 | 2.65 | 0.59 | 14.7 |
| nonbox | 1990 | 2272 | worst_pressure_l2 | 1.1 | 0.952 | 1.71 | 0.97 | 4.7 | 0.544 | 1.25 | 0.98 | 3.3 | 0.409 | 0.99 | 0.99 | 1.6 |
| nonbox | 2086 | 2382 | worst_pressure_l2 | 0.0 | 0.341 | 1.12 | 1.00 | 0.4 | 0.116 | 1.05 | 1.00 | 0.4 | 0.156 | 0.91 | 0.99 | 7.3 |
| nonbox | 2164 | 2466 | worst_velocity_l2 | 0.2 | 2.123 | 2.77 | 0.84 | 0.4 | 1.958 | 2.60 | 0.84 | 0.2 | 2.212 | 2.80 | 0.84 | 0.8 |
| nonbox | 2271 | 2590 | worst_pressure_l2 | 0.5 | 1.058 | 1.81 | 0.97 | 8.1 | 0.479 | 1.25 | 0.98 | 2.0 | 0.399 | 1.15 | 1.00 | 0.9 |
| nonbox | 2516 | 2869 | random | 2.1 | 0.877 | 1.53 | 0.97 | 5.3 | 0.592 | 1.26 | 0.96 | 1.5 | 0.507 | 0.76 | 0.99 | 1.6 |
| nonbox | 2569 | 2927 | worst_velocity_l2 | 0.1 | 2.659 | 3.22 | 0.75 | 0.4 | 2.573 | 3.14 | 0.75 | 0.2 | 2.630 | 3.14 | 0.73 | 14.7 |
| nonbox | 2724 | 3096 | worst_pressure_l2 | 0.5 | 1.011 | 1.81 | 0.96 | 4.4 | 0.596 | 1.32 | 0.95 | 3.9 | 0.415 | 0.98 | 0.96 | 14.6 |
| nonbox | 2958 | 3363 | worst_pressure_l2 | 0.5 | 0.912 | 1.72 | 0.98 | 2.1 | 0.479 | 1.25 | 0.97 | 1.7 | 0.365 | 0.81 | 1.00 | 1.9 |
| nonbox | 2966 | 3373 | worst_pressure_l2 | 55.1 | 1.689 | 1.94 | 0.52 | 9.9 | 1.378 | 1.44 | 0.34 | 1.0 | 0.607 | 0.82 | 0.86 | 58.0 |
| nonbox | 3002 | 3417 | random | 0.7 | 0.805 | 1.49 | 0.79 | 0.1 | 0.741 | 1.39 | 0.79 | 0.1 | 1.151 | 1.83 | 0.79 | 1.6 |
| nonbox | 3072 | 3502 | worst_velocity_l2 | 0.0 | 2.471 | 3.02 | 0.77 | 0.3 | 2.305 | 2.85 | 0.77 | 0.2 | 2.416 | 2.92 | 0.76 | 4.3 |
| nonbox | 3098 | 3534 | random | 3.0 | 0.692 | 1.41 | 0.95 | 1.5 | 0.362 | 1.16 | 0.97 | 0.7 | 0.269 | 0.97 | 0.97 | 1.6 |
| nonbox | 3187 | 3643 | worst_pressure_l2 | 0.0 | 0.345 | 1.12 | 1.00 | 0.2 | 0.101 | 1.02 | 1.00 | 0.7 | 0.066 | 0.98 | 1.00 | 1.8 |
| nonbox | 3412 | 3909 | worst_velocity_l2 | 0.1 | 2.633 | 3.24 | 0.79 | 0.3 | 2.526 | 3.13 | 0.79 | 0.1 | 2.552 | 3.10 | 0.79 | 2.3 |
| nonbox | 3672 | 4199 | worst_pressure_l2 | 39.8 | 1.642 | 1.82 | 0.48 | 4.4 | 1.519 | 1.55 | 0.31 | 16.0 | 0.668 | 1.00 | 0.81 | 53.1 |
| nonbox | 3686 | 4218 | worst_velocity_l2 | 0.1 | 2.692 | 3.26 | 0.77 | 0.2 | 2.528 | 3.10 | 0.77 | 0.1 | 1.508 | 1.95 | 0.77 | 2.1 |
| profile_crossflow | 24 | 28 | random | 47.7 | 1.813 | 2.25 | 0.56 | 9.0 | 1.529 | 1.90 | 0.57 | 0.9 | 0.908 | 0.97 | 0.61 | 1.1 |
| profile_crossflow | 82 | 91 | worst_velocity_l2 | 37.9 | 1.864 | 2.48 | 0.70 | 8.4 | 1.755 | 2.31 | 0.68 | 1.8 | 0.811 | 1.07 | 0.70 | 1.2 |
| profile_crossflow | 240 | 265 | worst_pressure_l2 | 18.2 | 1.353 | 2.12 | 0.80 | 3.5 | 1.118 | 1.80 | 0.79 | 0.8 | 0.529 | 0.96 | 0.78 | 1.9 |
| profile_crossflow | 253 | 284 | worst_velocity_l2 | 37.9 | 1.736 | 2.28 | 0.70 | 9.1 | 1.911 | 2.43 | 0.70 | 4.7 | 0.812 | 1.06 | 0.71 | 0.8 |
| profile_crossflow | 492 | 565 | random | 58.9 | 1.368 | 1.86 | 0.54 | 27.9 | 1.657 | 2.06 | 0.56 | 12.5 | 0.982 | 0.82 | 0.51 | 5.2 |
| profile_crossflow | 525 | 606 | random | 33.4 | 1.589 | 2.20 | 0.76 | 9.5 | 1.409 | 1.92 | 0.73 | 3.8 | 0.742 | 0.97 | 0.70 | 0.6 |
| profile_crossflow | 533 | 616 | random | 25.4 | 1.871 | 2.49 | 0.75 | 6.8 | 1.326 | 2.01 | 0.83 | 2.8 | 0.612 | 0.71 | 0.78 | 2.8 |
| profile_crossflow | 618 | 716 | random | 68.0 | 1.186 | 1.63 | 0.65 | 34.9 | 1.165 | 1.41 | 0.58 | 20.9 | 0.959 | 0.58 | 0.43 | 2.6 |
| profile_crossflow | 658 | 764 | worst_pressure_l2 | 14.4 | 0.989 | 1.78 | 0.85 | 2.8 | 1.213 | 1.97 | 0.85 | 0.6 | 0.487 | 1.09 | 0.86 | 1.6 |
| profile_crossflow | 709 | 820 | worst_pressure_l2 | 26.1 | 1.503 | 2.18 | 0.66 | 5.1 | 1.353 | 2.00 | 0.66 | 3.2 | 0.733 | 1.09 | 0.66 | 1.9 |
| profile_crossflow | 962 | 1109 | random | 46.8 | 0.574 | 1.44 | 0.94 | 49.3 | 1.195 | 1.65 | 0.75 | 13.6 | 0.843 | 0.81 | 0.66 | 0.2 |
| profile_crossflow | 1006 | 1160 | worst_pressure_l2 | 21.6 | 1.453 | 2.18 | 0.81 | 1.7 | 1.394 | 2.10 | 0.82 | 0.4 | 0.584 | 1.13 | 0.83 | 2.8 |
| profile_crossflow | 1203 | 1370 | worst_pressure_l2 | 21.3 | 1.267 | 2.00 | 0.73 | 1.6 | 1.467 | 2.19 | 0.74 | 1.5 | 0.615 | 0.98 | 0.72 | 1.2 |
| profile_crossflow | 1248 | 1422 | worst_pressure_l2 | 16.9 | 1.372 | 2.16 | 0.82 | 8.2 | 1.065 | 1.81 | 0.82 | 1.9 | 0.538 | 1.08 | 0.81 | 4.0 |
| profile_crossflow | 1463 | 1671 | worst_pressure_l2 | 30.0 | 1.679 | 2.28 | 0.69 | 6.6 | 1.632 | 2.23 | 0.70 | 4.6 | 0.763 | 1.14 | 0.72 | 0.9 |
| profile_crossflow | 1567 | 1791 | worst_velocity_l2 | 40.6 | 1.943 | 2.48 | 0.64 | 8.4 | 1.728 | 2.23 | 0.63 | 7.7 | 0.795 | 0.75 | 0.60 | 3.1 |
| profile_crossflow | 1576 | 1802 | worst_pressure_l2 | 22.4 | 1.684 | 2.44 | 0.77 | 4.4 | 1.195 | 1.89 | 0.76 | 1.5 | 0.924 | 0.10 | 0.72 | 4.5 |
| profile_crossflow | 1603 | 1830 | random | 29.0 | 1.550 | 2.28 | 0.79 | 11.2 | 1.483 | 2.16 | 0.76 | 1.3 | 0.825 | 1.34 | 0.77 | 1.3 |
| profile_crossflow | 1933 | 2208 | worst_velocity_l2 | 45.5 | 1.978 | 2.52 | 0.66 | 10.4 | 1.672 | 2.14 | 0.63 | 2.7 | 0.845 | 0.92 | 0.64 | 0.8 |
| profile_crossflow | 2129 | 2429 | worst_pressure_l2 | 18.7 | 1.703 | 2.41 | 0.83 | 1.9 | 1.313 | 2.03 | 0.86 | 1.7 | 0.563 | 1.12 | 0.87 | 2.0 |
| profile_crossflow | 2144 | 2445 | worst_velocity_l2 | 53.1 | 1.927 | 2.47 | 0.66 | 15.9 | 1.765 | 2.17 | 0.57 | 5.0 | 0.942 | 0.92 | 0.56 | 1.0 |
| profile_crossflow | 2487 | 2836 | worst_velocity_l2 | 43.2 | 1.965 | 2.54 | 0.71 | 11.8 | 1.638 | 2.09 | 0.67 | 2.0 | 0.887 | 1.08 | 0.66 | 0.8 |
| profile_crossflow | 2658 | 3024 | random | 62.1 | 1.067 | 1.75 | 0.80 | 41.2 | 1.220 | 1.48 | 0.54 | 13.3 | 0.958 | 0.67 | 0.50 | 0.8 |
| profile_crossflow | 2720 | 3092 | worst_velocity_l2 | 45.5 | 1.822 | 2.30 | 0.62 | 13.2 | 1.773 | 2.27 | 0.66 | 12.1 | 0.864 | 1.01 | 0.65 | 5.4 |
| profile_crossflow | 3095 | 3530 | random | 54.3 | 0.882 | 1.65 | 0.88 | 47.0 | 1.048 | 1.40 | 0.69 | 12.1 | 0.878 | 0.60 | 0.55 | 2.4 |
| profile_crossflow | 3189 | 3645 | random | 55.3 | 1.173 | 1.89 | 0.83 | 35.6 | 1.255 | 1.71 | 0.69 | 16.6 | 0.836 | 0.65 | 0.62 | 6.4 |
| profile_crossflow | 3270 | 3739 | worst_velocity_l2 | 33.8 | 2.007 | 2.68 | 0.78 | 7.1 | 1.605 | 2.21 | 0.76 | 3.0 | 0.767 | 1.11 | 0.76 | 0.2 |
| profile_crossflow | 3287 | 3760 | worst_pressure_l2 | 15.5 | 1.245 | 2.05 | 0.82 | 6.8 | 1.181 | 1.96 | 0.81 | 0.9 | 0.519 | 0.98 | 0.81 | 0.8 |
| profile_crossflow | 3539 | 4051 | worst_velocity_l2 | 36.3 | 1.921 | 2.43 | 0.68 | 7.0 | 1.696 | 2.23 | 0.72 | 6.0 | 0.804 | 1.06 | 0.71 | 0.6 |
| profile_crossflow | 3786 | 4331 | worst_velocity_l2 | 30.6 | 1.966 | 2.71 | 0.85 | 11.9 | 1.761 | 2.40 | 0.77 | 1.2 | 0.674 | 1.10 | 0.81 | 7.7 |
