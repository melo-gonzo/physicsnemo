# PDIAG: why ISLA's AeroZoo pressure relative L2 exceeds 1 (2026-10-22)

Sub-lane of AZ-DECIDE. Written 2026-10-22 10:35 EDT. Extends `../preddiag/` (its `cluster/diag.py`, equivalence-checked
against the scorer). ISLA 0.9.0, epoch 250, float32, the scorer's inputs (200,000 interior points, `val_split=<split>.test`).

## Conclusion

1. **profile_crossflow: the pressure amplitude comes from the velocity overshoot.** The best affine fit p ≈ α p̂ + β has
   β ≈ 0 (0.03 / −0.07 RMS p) and α·(speed ratio)² = 1.06 / 1.34 (seed 42 / 43; per-case IQR 0.66–1.38 / 0.87–1.62), with
   residual 0.22 / 0.30 and corr 0.84 / 0.69. Dividing p̂ by (speed ratio)² alone takes the median L2 from 3.34 / 2.00
   to 0.33 / 0.36. The large offset share (0.85 / 0.90) is not a free gauge offset: the true pressure is mostly a level
   (mean 0.91 RMS p, the inlet-to-outlet rise), and scaling p by ~sr² ≈ 4 puts most of the error into the mean.
   Secondary: ISLA does not hold p ≈ 0 at the adjacent outlet: on the 1 % of scored points nearest the outlet it
   predicts 1.2 / 1.8 RMS p (true 0.017).
2. **family_non_box: a shape error, not explained by the velocity.** Affine residual 0.79 / 0.75, corr 0.48 / 0.49,
   α·sr² 0.29 / 0.52 (the pressure RMS overshoot 2.3× / 1.2× exceeds sr² = 1.6 / 1.2). By family (`family.md`): opposed
   spheres / hemispheres overpredict the pressure 1.3–7.5× with poor shape (corr 0.4–0.6); ducts predict almost no
   pressure (RMS ratio 0.01–0.75, L2 ≈ 1); bends overshoot 6–9×. No general gauge error (near-outlet prediction −0.001 /
   −0.06 RMS p), except seed 42's opposed hemispheres (4.9 RMS p, seed 43 −0.04): seed-dependent.
3. **Elsewhere the fit is clean:** full, reynolds, blobs, bank_zero_to_cad have α = 0.97–0.98, offset share ≤ 4 %,
   speed ratio 0.99–1.00, and an affine residual of 0.25–0.49 against L2 0.30–0.55: what is left is shape error. profile_dual_inlet is the mirror of crossflow
   (speed ratio 0.49, pressure RMS ratio 0.37, offset share 0.70, residual 0.26), an amplitude error that stays below 1.

The outlet's prescribed gauge pressure is exactly 0 Pa on every case (max |bc_pressure| = 0 over 270 rows), and the true
pressure on the scored points nearest the outlet is 0.0002–0.017 RMS p (median per split).

## Medians per split (30 seeded-random test cases each; ratios of pressure in units of the case's RMS p)

| split | seed | n | pressure_l2 | velocity_l2 | speed_ratio_rms | offset_share | meanremoved_l2 | alpha | inv_sr2 | alpha_x_sr2 | beta_rel | affine_res_l2 | l2_rescaled_sr2 | corr | rms_ratio | mean_p_rel | mean_ph_rel | outlet_pred_mean_rel | outlet_pred_rms_rel | near_outlet_true_mean_rel | near_outlet_pred_mean_rel | aug_p_change_rel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 42 | 30 | 0.302 | 0.104 | 1 | 0.0363 | 0.257 | 0.983 | 1 | 0.979 | 0.00387 | 0.25 | 0.268 | 0.95 | 0.994 | 0.528 | 0.539 | -0.0317 | 0.0662 | 0.00206 | 0.00739 | 0.0369 |
| reynolds | 42 | 30 | 0.554 | 0.138 | 0.994 | 0.0291 | 0.507 | 0.976 | 1.01 | 0.963 | 0.00251 | 0.491 | 0.512 | 0.838 | 0.868 | 0.316 | 0.0683 | -0.0362 | 0.0606 | 0.00167 | 0.0106 | 0.0372 |
| blobs | 42 | 30 | 0.369 | 0.151 | 0.996 | 0.00309 | 0.365 | 0.98 | 1.01 | 0.966 | -0.00575 | 0.361 | 0.369 | 0.919 | 0.942 | 0.534 | 0.446 | -0.0191 | 0.0998 | 0.00185 | 0.0108 | 0.0258 |
| bank_zero_to_cad | 42 | 30 | 0.498 | 0.216 | 0.996 | 0.0157 | 0.41 | 0.969 | 1.01 | 0.974 | -0.0043 | 0.384 | 0.478 | 0.889 | 0.97 | 0.0118 | -0.0387 | -0.0211 | 0.0749 | 0.00119 | 0.00946 | 0.0552 |
| profile_dual_inlet | 42 | 30 | 0.649 | 0.65 | 0.487 | 0.7 | 0.363 | 2.55 | 4.21 | 0.64 | 0.0035 | 0.26 | 0.672 | 0.828 | 0.374 | 0.876 | 0.34 | 0.0481 | 0.0551 | 0.012 | 0.037 | 0.0108 |
| nonbox | 42 | 30 | 2.22 | 0.648 | 1.26 | 0.412 | 1.98 | 0.165 | 0.629 | 0.289 | -0.187 | 0.787 | 1.61 | 0.484 | 2.26 | 0.327 | 1 | -0.0181 | 2.35 | 0.000213 | -0.00145 | 0.243 |
| nonbox | 43 | 30 | 1.1 | 0.466 | 1.1 | 0.322 | 0.881 | 0.425 | 0.833 | 0.52 | -0.0768 | 0.749 | 1.02 | 0.493 | 1.2 | 0.327 | 0.492 | -0.599 | 0.66 | 0.000213 | -0.0561 | 0.0841 |
| profile_crossflow | 42 | 30 | 3.34 | 1.49 | 2.07 | 0.845 | 1.21 | 0.225 | 0.234 | 1.06 | 0.0315 | 0.222 | 0.333 | 0.843 | 4.32 | 0.908 | 4.01 | 0.534 | 0.858 | 0.0172 | 1.22 | 0.08 |
| profile_crossflow | 43 | 30 | 2 | 1.4 | 1.86 | 0.898 | 0.591 | 0.366 | 0.29 | 1.34 | -0.0687 | 0.296 | 0.362 | 0.693 | 2.91 | 0.908 | 2.84 | 0.999 | 1.59 | 0.0172 | 1.83 | 0.107 |

All columns (and α0 / scale-only residual, std_ratio, mean_err_rel) per case in `pdiag.tsv`; every median in `medians.md`;
nonbox and crossflow by scenario / profile in `family.md` (labels from `examples/cfd/isla/results/aerozoo_dataset/case_labels.json`,
scoring side only).

Per-case test of α ≈ 1/sr²: corr(log α, log 1/sr²) = 0.64 / 0.62 on crossflow, 0.81 / 0.76 on nonbox (but α·sr²
centred at 0.29 / 0.52 there; 4 / 6 of 30 nonbox cases in [0.67, 1.5], against 20 / 15 on crossflow).

## Definitions (per case, physical gauge pressure on the 200,000 scored points, unweighted like the metric)

e = p̂ − p. offset_share = mean(e)² / mean(e²). meanremoved_l2 = ‖(p̂ − mean p̂) − (p − mean p)‖ / ‖p‖
(= pressure_l2 · √(1 − offset_share)). Affine fit by least squares: α = cov(p̂, p) / var(p̂), β = mean p − α mean p̂,
beta_rel = β / RMS p, affine_res_l2 = ‖α p̂ + β − p‖ / ‖p‖. corr = Pearson. rms_ratio = RMS p̂ / RMS p. speed_ratio_rms
= RMS |u_pred| / RMS |u_true| (preddiag's). inv_sr2 = 1 / speed_ratio², alpha_x_sr2 = α · speed_ratio²,
l2_rescaled_sr2 = ‖p̂ / sr² − p‖ / ‖p‖. mean_p_rel, mean_ph_rel = mean p, mean p̂ over RMS p.
Outlet: near_outlet_* = mean p and p̂ over the 1 % of scored points nearest the outlet surface (exact distance,
`exact_boundary_geometry`), from the scorer's own forward pass. outlet_pred_mean_rel / _rms_rel = area-weighted mean /
RMS of p̂ at 4,096 outlet face centroids (uniform draw seeded by the case index), over RMS p of the interior.

**Caveat on the outlet-face query.** ISLA's interior points interact (not query-set independent), so the face
centroids were appended to the 200,000 scored points in a second forward pass; the scored metrics come from the first
pass only. Adding 2 % more points moved the interior pressure by a median 1–11 % of ‖p‖ (aug_p_change_rel), 24 % on
nonbox seed 42. The face numbers therefore describe a slightly different state; the near-outlet band (exact pass) is
the cleaner gauge test, and the two agree in sign and order of magnitude (crossflow ≫ 0, other splits ≈ 0 in median).

## Equivalence with the scorer

270 rows (9 runs × 30). Driver's MetricCalculator against the scorer's infer_step row for the same sample_id
(`../scores/p200k_fp32_test_azc/<run>/<run>/metrics.jsonl`, after the last config row, epoch 250): pressure_l2 max
relative difference 7.4e-5; velocity_l2 ≤ 2.6e-5 except one case (bank_zero_to_cad idx 2739, vel L2 0.517: 4.7e-4,
run-to-run float32 nondeterminism; its pressure matches to < 1e-4). The driver's own pressure L2 equals the metric's to
1.2e-7, and no target measure is used (unweighted).

## Cases

`scripts/make_cases.py`: 30 per split, numpy `default_rng(20261022)` per split, from the test part's sample ids in the
ISLA seed-42 scorer rows (seed 43 has the same list on nonbox and crossflow): `scripts/cases/cases_<split>.tsv`.
Splits: full, nonbox (family_non_box), reynolds, blobs (bank_blobs), bank_zero_to_cad, profile_crossflow,
profile_dual_inlet. Runs: azc_<split>_isla_seed42, plus seed 43 on nonbox and profile_crossflow.

## Job (LHR)

- 333637 `isla.azdecide.pdiag` (short QoS, 4 GPUs, nvl72d041-T18, 11.5 min, 2026-10-22 07:17–07:29 PDT): 9 units,
  about 6 s per case. Allocated 0.77 GPU-h.
- Scripts: `scripts/diag.py` (md5 184051a3), `scripts/pdiag.sbatch` (md5 9cb9e804), copied to
  `$T/kernel/azdecide/pdiag/`; environment, snapshots (code_r090 / recipe_r090) and overrides as the scorer's
  (`kernel/azdecide/eval_azi/eval_test.sbatch`) via `kernel/azi/lanes.tsv`. No prediction arrays were written
  (`+save_predictions=false`); resume states, Hydra cwd and caches deleted after the run.
- Reduction: `raw/` (rows and logs, copied from `$T/kernel/azdecide/pdiag/out/`), `scripts/reduce.py`
  (→ `pdiag.tsv`, `medians.md`, equivalence), `scripts/family.py` (→ `family.md`).
