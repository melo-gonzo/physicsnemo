# DK-GAP: ISLA-DTT against interacting ISLA, per case and per region (reduce_gap.py)

## af / test (200 cases)

Gate: r090eag_af_dtt_seed42: mean 0.0144081, 0.00805624, 0.0261139 (vs standing max rel 2.6e-01), scorer per-case max rel 2.0e+00, recompute 2.2e-07; r090eag_af_dtt_seed43: mean 0.0148086, 0.00777977, 0.0283674 (vs standing max rel 3.8e-01), scorer per-case max rel 2.3e+00, recompute 2.3e-07; r090eag_af_isla_seed42: mean 0.0159403, 0.0074524, 0.0217707 (vs standing max rel 3.8e-01), scorer per-case max rel 2.0e+00, recompute 2.1e-07; r090eag_af_isla_seed43: mean 0.0146753, 0.00728855, 0.0203866 (vs standing max rel 3.6e-01), scorer per-case max rel 2.5e+00, recompute 2.2e-07; targets identical across the four runs: True.

| field | ISLA-DTT mean | ISLA mean | ratio of means | per-case ratio q10/q25/q50/q75/q90 | DTT worse in | top-10% share of the gap | seed noise: DTT s43/s42, ISLA s43/s42 (ratio of means; per-case ratio q10-q90) |
|---|---|---|---|---|---|---|---|
| pressure | 0.0146083 | 0.0153078 | 0.954 (s42 0.904, s43 1.009) | 0.74 / 0.86 / 0.94 / 1.03 / 1.16 | 32% | -1.10 | 1.028 (0.80-1.28), 0.921 (0.70-1.13) |
| velocity | 0.00791801 | 0.00737047 | 1.074 (s42 1.081, s43 1.067) | 0.85 / 0.94 / 0.99 / 1.10 / 1.32 | 48% | 1.19 | 0.966 (0.85-1.07), 0.978 (0.86-1.10) |
| nut | 0.0272407 | 0.0210786 | 1.292 (s42 1.200, s43 1.391) | 1.00 / 1.11 / 1.29 / 1.52 / 1.74 | 90% | 0.44 | 1.086 (0.85-1.23), 0.936 (0.76-1.10) |

By covariate quartile (ratio of means ISLA-DTT / ISLA; [s42, s43]):

| aoa_deg quartile | pressure | velocity | nut |
|---|---|---|---|
| -4.94..-0.517 (n 50) | 0.939 [0.87, 1.01] | 1.123 [1.07, 1.17] | 1.496 [1.24, 1.78] |
| -0.327..4.776 (n 50) | 0.987 [0.97, 1.01] | 1.226 [1.27, 1.18] | 1.523 [1.42, 1.64] |
| 4.923..8.84 (n 50) | 0.998 [0.96, 1.04] | 1.004 [1.03, 0.97] | 1.255 [1.17, 1.35] |
| 8.883..14.69 (n 50) | 0.886 [0.82, 0.96] | 0.955 [0.97, 0.94] | 0.988 [1.02, 0.96] |

| Re quartile | pressure | velocity | nut |
|---|---|---|---|
| 2.039e+06..2.959e+06 (n 50) | 0.947 [0.89, 1.02] | 1.075 [1.08, 1.07] | 1.160 [1.10, 1.23] |
| 2.973e+06..3.957e+06 (n 50) | 0.968 [0.87, 1.08] | 1.054 [1.04, 1.07] | 1.375 [1.22, 1.54] |
| 4.01e+06..5.007e+06 (n 50) | 0.954 [0.88, 1.03] | 1.061 [1.05, 1.07] | 1.259 [1.15, 1.37] |
| 5.011e+06..5.975e+06 (n 50) | 0.948 [0.98, 0.92] | 1.108 [1.15, 1.07] | 1.404 [1.36, 1.45] |

Spearman of the per-case log ratio with: pressure: aoa_deg -0.14, Re +0.03, ISLA's own error -0.02; velocity: aoa_deg -0.29, Re +0.03, ISLA's own error +0.05; nut: aoa_deg -0.34, Re +0.08, ISLA's own error +0.05.

Regions by wall_band (share of points; per field: share of squared error ISLA / ISLA-DTT, pooled RMS ratio DTT/ISLA [seed noise DTT s43/s42, ISLA s43/s42], share of the squared-error gap):

| region | points | pressure | velocity | nut |
|---|---|---|---|---|
| d<1e-4 | 0.125 | 0.292 / 0.306; 1.004 [1.16, 0.97]; -0.07 | 0.185 / 0.157; 1.024 [0.95, 0.97]; +0.04 | 0.005 / 0.004; 1.088 [1.11, 0.97]; +0.00 |
| 1e-4..1e-3 | 0.163 | 0.309 / 0.294; 0.955 [1.11, 0.95]; +0.71 | 0.266 / 0.181; 0.919 [0.95, 0.97]; -0.17 | 0.009 / 0.007; 1.082 [1.05, 0.98]; +0.00 |
| 1e-3..1e-2 | 0.184 | 0.210 / 0.174; 0.892 [1.09, 0.97]; +1.12 | 0.251 / 0.181; 0.946 [1.00, 1.01]; -0.11 | 0.057 / 0.044; 1.053 [1.06, 1.08]; +0.01 |
| 1e-2..0.1 | 0.209 | 0.109 / 0.110; 0.984 [1.01, 0.94]; +0.09 | 0.177 / 0.204; 1.193 [0.97, 0.99]; +0.31 | 0.145 / 0.086; 0.918 [1.06, 0.93]; -0.06 |
| 0.1..1 | 0.237 | 0.062 / 0.088; 1.173 [0.96, 0.90]; -0.61 | 0.098 / 0.232; 1.713 [0.96, 0.90]; +0.79 | 0.660 / 0.511; 1.047 [1.11, 0.93]; +0.15 |
| >=1 | 0.081 | 0.018 / 0.029; 1.238 [0.98, 0.92]; -0.25 | 0.023 / 0.045; 1.566 [0.99, 0.90]; +0.14 | 0.125 / 0.347; 1.987 [1.35, 0.91]; +0.88 |

Regions by zone (share of points; per field: share of squared error ISLA / ISLA-DTT, pooled RMS ratio DTT/ISLA [seed noise DTT s43/s42, ISLA s43/s42], share of the squared-error gap):

| region | points | pressure | velocity | nut |
|---|---|---|---|---|
| near_wall d<0.01 | 0.473 | 0.812 / 0.774; 0.958 [1.13, 0.96]; +1.77 | 0.702 / 0.520; 0.957 [0.97, 0.98]; -0.25 | 0.070 / 0.056; 1.060 [1.06, 1.06]; +0.02 |
| wake | 0.096 | 0.025 / 0.044; 1.311 [0.97, 0.96]; -0.47 | 0.070 / 0.201; 1.882 [0.93, 0.93]; +0.75 | 0.854 / 0.895; 1.218 [1.20, 0.93]; +0.99 |
| outer | 0.431 | 0.164 / 0.182; 1.034 [0.99, 0.92]; -0.30 | 0.228 / 0.280; 1.234 [0.99, 0.96]; +0.50 | 0.075 / 0.049; 0.958 [1.00, 0.94]; -0.01 |

Regions by target_nut (share of points; per field: share of squared error ISLA / ISLA-DTT, pooled RMS ratio DTT/ISLA [seed noise DTT s43/s42, ISLA s43/s42], share of the squared-error gap):

| region | points | pressure | velocity | nut |
|---|---|---|---|---|
| nut below the case's 90th pct | 0.900 | 0.968 / 0.951; 0.972 [1.10, 0.95]; +1.39 | 0.898 / 0.827; 1.069 [0.98, 0.98]; +0.53 | 0.123 / 0.237; 1.652 [1.49, 0.97]; +0.51 |
| nut top decile | 0.100 | 0.032 / 0.049; 1.209 [1.00, 1.03]; -0.39 | 0.102 / 0.173; 1.445 [0.90, 0.94]; +0.47 | 0.877 / 0.763; 1.110 [1.10, 0.94]; +0.49 |

## lug / test (600 cases)

Gate: r090fix_lug_dtt_seed42: mean 0.00366516 (vs standing max rel 6.5e-01), scorer per-case max rel 1.9e+00, recompute 1.3e-07; r090fix_lug_dtt_seed43: mean 0.00332328 (vs standing max rel 8.1e-01), scorer per-case max rel 2.2e+00, recompute 1.3e-07; r090fix_lug_isla_seed42: mean 0.00334296, scorer per-case max rel 4.4e+00, recompute 1.4e-07; r090fix_lug_isla_seed43: mean 0.00335967, scorer per-case max rel 3.6e+00, recompute 1.3e-07; targets identical across the four runs: True.

| field | ISLA-DTT mean | ISLA mean | ratio of means | per-case ratio q10/q25/q50/q75/q90 | DTT worse in | top-10% share of the gap | seed noise: DTT s43/s42, ISLA s43/s42 (ratio of means; per-case ratio q10-q90) |
|---|---|---|---|---|---|---|---|
| von_mises | 0.00349422 | 0.00335131 | 1.043 (s42 1.096, s43 0.989) | 0.94 / 0.99 / 1.06 / 1.12 / 1.17 | 73% | 0.51 | 0.907 (0.80-1.00), 1.005 (0.89-1.11) |

By covariate quartile (ratio of means ISLA-DTT / ISLA; [s42, s43]):

| pressure_mpa quartile | von_mises |
|---|---|
| 50.14..62.03 (n 150) | 1.080 [1.12, 1.03] |
| 62.08..73.89 (n 150) | 1.065 [1.11, 1.02] |
| 74.02..87.03 (n 150) | 1.045 [1.11, 0.98] |
| 87.61..99.96 (n 150) | 0.994 [1.05, 0.94] |

| hole_radius quartile | von_mises |
|---|---|
| 0.01..0.0113 (n 150) | 1.060 [1.10, 1.02] |
| 0.01133..0.01256 (n 150) | 1.060 [1.12, 1.00] |
| 0.01257..0.01377 (n 150) | 1.019 [1.08, 0.96] |
| 0.01377..0.015 (n 150) | 1.037 [1.10, 0.98] |

| length quartile | von_mises |
|---|---|
| 0.08005..0.09056 (n 150) | 1.075 [1.12, 1.03] |
| 0.09056..0.1001 (n 150) | 1.067 [1.12, 1.02] |
| 0.1001..0.1095 (n 150) | 1.035 [1.11, 0.96] |
| 0.1097..0.12 (n 150) | 1.001 [1.05, 0.96] |

| thickness quartile | von_mises |
|---|---|
| 0.01501..0.0192 (n 150) | 1.025 [1.08, 0.97] |
| 0.01926..0.02261 (n 150) | 1.038 [1.09, 0.99] |
| 0.02263..0.02584 (n 150) | 1.061 [1.11, 1.01] |
| 0.02585..0.02999 (n 150) | 1.048 [1.10, 0.99] |

| plastic_frac quartile | von_mises |
|---|---|
| 0.000334..0.00427 (n 150) | 1.081 [1.11, 1.05] |
| 0.004319..0.0132 (n 150) | 1.075 [1.13, 1.02] |
| 0.01335..0.05051 (n 150) | 1.056 [1.12, 1.00] |
| 0.05102..0.3607 (n 150) | 0.980 [1.04, 0.92] |

| vm_max_mpa quartile | von_mises |
|---|---|
| 382.6..453.2 (n 150) | 1.077 [1.12, 1.03] |
| 453.2..466.7 (n 150) | 1.077 [1.13, 1.03] |
| 466.8..483.7 (n 150) | 1.048 [1.10, 0.99] |
| 483.9..754.7 (n 150) | 0.986 [1.05, 0.93] |

Spearman of the per-case log ratio with: von_mises: pressure_mpa -0.19, hole_radius -0.07, length -0.18, thickness +0.09, plastic_frac -0.26, vm_max_mpa -0.23, ISLA's own error -0.55.

Regions by dist_band (share of points; per field: share of squared error ISLA / ISLA-DTT, pooled RMS ratio DTT/ISLA [seed noise DTT s43/s42, ISLA s43/s42], share of the squared-error gap):

| region | points | von_mises |
|---|---|---|
| on surface d<1e-6 L | 0.336 | 0.424 / 0.493; 1.110 [0.88, 1.02]; +1.63 |
| 1e-6..0.01 L | 0.000 | 0.000 / 0.000; 0.000 [0.00, 0.00]; +0.00 |
| 0.01..0.03 L | 0.008 | 0.006 / 0.006; 0.963 [0.89, 0.95]; -0.01 |
| 0.03..0.1 L | 0.460 | 0.436 / 0.376; 0.957 [0.94, 1.02]; -0.61 |
| >=0.1 L | 0.196 | 0.133 / 0.125; 0.997 [0.94, 1.00]; -0.01 |

Regions by nearest_part (share of points; per field: share of squared error ISLA / ISLA-DTT, pooled RMS ratio DTT/ISLA [seed noise DTT s43/s42, ISLA s43/s42], share of the squared-error gap):

| region | points | von_mises |
|---|---|---|
| fixed | 0.046 | 0.151 / 0.164; 1.074 [0.89, 1.07]; +0.38 |
| hole | 0.054 | 0.043 / 0.044; 1.037 [0.90, 0.96]; +0.05 |
| loaded | 0.053 | 0.057 / 0.056; 1.025 [0.93, 0.99]; +0.05 |
| free | 0.847 | 0.749 / 0.736; 1.021 [0.92, 1.01]; +0.52 |

Regions by yield (share of points; per field: share of squared error ISLA / ISLA-DTT, pooled RMS ratio DTT/ISLA [seed noise DTT s43/s42, ISLA s43/s42], share of the squared-error gap):

| region | points | von_mises |
|---|---|---|
| elastic (target vm <= 380 MPa) | 0.958 | 0.824 / 0.817; 1.026 [0.92, 0.98]; +0.71 |
| above yield (target vm > 380 MPa) | 0.042 | 0.176 / 0.183; 1.049 [0.88, 1.18]; +0.29 |

## Fit against generalization: ratio of means ISLA-DTT / ISLA on the train subset and on the test set

| family | field | test | train (subset) | train [s42, s43] |
|---|---|---|---|---|
