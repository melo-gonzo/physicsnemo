# TIE-EXP registered read (read_tie.py)

## lug

| field | B native 42 / 43 (mean) | TIE native 42 / 43 (mean) | r = TIE / B | log r | sigma_reg | s_pool | sigma | 2 sigma | N verdict |
|---|---|---|---|---|---|---|---|---|---|
| von_mises_l2 | 0.00138 / 0.00177 (0.00157) | 0.00139 / 0.00177 (0.00158) | 1.004 | +0.004 | 0.036 | 0.176 | 0.176 | 0.353 | no cost |

| field | B posed / native (e_B) | TIE posed / native (e_TIE) | e_TIE / e_B | P |
|---|---|---|---|---|
| von_mises_l2 | 1.4561 (+0.4561) | 1.0019 (+0.0019) | 0.004 | the medial ties carry most of the lug's pose sensitivity |

posed (seeds 42 / 43): b von_mises_l2 0.00172 / 0.00287; tie von_mises_l2 0.00139 / 0.00178

## t2d

| field | B native 42 / 43 (mean) | TIE native 42 / 43 (mean) | r = TIE / B | log r | sigma_reg | s_pool | sigma | 2 sigma | N verdict |
|---|---|---|---|---|---|---|---|---|---|
| displacement_l2 | 0.00783 / 0.00629 (0.00706) | 0.00628 / 0.00806 (0.00717) | 1.016 | +0.016 | 0.232 | 0.166 | 0.232 | 0.464 | no cost |
| stress_l2 | 0.00964 / 0.00794 (0.00879) | 0.00844 / 0.00973 (0.00908) | 1.034 | +0.033 | 0.330 | 0.120 | 0.330 | 0.660 | no cost |
| q_l2 | 0.41040 / 0.25005 (0.33022) | 0.36701 / 0.28775 (0.32738) | 0.991 | -0.009 | 0.303 | 0.276 | 0.303 | 0.606 | no cost |

| field | B posed / native (e_B) | TIE posed / native (e_TIE) | e_TIE / e_B | P |
|---|---|---|---|---|
| displacement_l2 | 0.9999 (-0.0001) | 0.9999 (-0.0001) | 1.274 | control: |log((1+e_TIE)/(1+e_B))| = 0.0000 <= 2 sigma 0.464 (not a verdict) |
| stress_l2 | 1.0000 (-0.0000) | 0.9999 (-0.0001) | 3.641 | control: |log((1+e_TIE)/(1+e_B))| = 0.0001 <= 2 sigma 0.660 (beside) |
| q_l2 | 1.0000 (+0.0000) | 1.0000 (-0.0000) | -0.411 | control: |log((1+e_TIE)/(1+e_B))| = 0.0000 <= 2 sigma 0.606 (beside) |

posed (seeds 42 / 43): b displacement_l2 0.00783 / 0.00629; b stress_l2 0.00964 / 0.00794; b q_l2 0.41040 / 0.25005; tie displacement_l2 0.00628 / 0.00806; tie stress_l2 0.00844 / 0.00972; tie q_l2 0.36701 / 0.28775

## F (flip_probe_tie2.py; median over cases of d_F per target field)

| probe | cases | field | d_F median | d_F max | d_U median |
|---|---|---|---|---|---|
| F_code_cmpaf_fix_r090eag_lug_isla_seed42 | 60 | von_mises | 1.40e-04 | 3.67e-04 | 1.59e+00 |
| F_code_cmpaf_fix_r090eag_lug_isla_seed43 | 60 | von_mises | 1.56e-04 | 4.46e-04 | 1.94e+00 |
| F_code_cmpaf_fix_r090eag_t2dz_isla_seed42 | 100 | displacement | 8.22e-07 | 8.84e-06 | 3.73e+00 |
| F_code_cmpaf_fix_r090eag_t2dz_isla_seed42 | 100 | q | 2.74e-05 | 3.49e-05 | 9.31e+01 |
| F_code_cmpaf_fix_r090eag_t2dz_isla_seed42 | 100 | stress | 1.46e-06 | 6.82e-06 | 2.91e+00 |
| F_code_cmpaf_fix_r090eag_t2dz_isla_seed43 | 100 | displacement | 8.04e-07 | 8.89e-06 | 4.12e+00 |
| F_code_cmpaf_fix_r090eag_t2dz_isla_seed43 | 100 | q | 2.99e-05 | 5.45e-05 | 1.33e+02 |
| F_code_cmpaf_fix_r090eag_t2dz_isla_seed43 | 100 | stress | 1.43e-06 | 5.76e-06 | 3.11e+00 |
| F_code_tie2_tie_lug_isla_seed42 | 60 | von_mises | 1.38e-07 | 2.05e-07 | 1.55e+00 |
| F_code_tie2_tie_lug_isla_seed43 | 60 | von_mises | 1.41e-07 | 4.08e-07 | 1.40e+00 |
| F_code_tie2_tie_t2dz_isla_seed42 | 100 | displacement | 0.00e+00 | 0.00e+00 | 3.30e+00 |
| F_code_tie2_tie_t2dz_isla_seed42 | 100 | q | 0.00e+00 | 0.00e+00 | 1.14e+02 |
| F_code_tie2_tie_t2dz_isla_seed42 | 100 | stress | 0.00e+00 | 0.00e+00 | 2.50e+00 |
| F_code_tie2_tie_t2dz_isla_seed43 | 100 | displacement | 0.00e+00 | 0.00e+00 | 3.82e+00 |
| F_code_tie2_tie_t2dz_isla_seed43 | 100 | q | 0.00e+00 | 0.00e+00 | 1.75e+02 |
| F_code_tie2_tie_t2dz_isla_seed43 | 100 | stress | 0.00e+00 | 0.00e+00 | 2.88e+00 |

F verdict: exact flip symmetry restored
