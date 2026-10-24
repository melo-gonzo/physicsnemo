> Phase-1 note (PREREG 15.8): the ISLA held-out values of drv in this table are **invalid (frame mismatch)**; those of az are valid (15.8 erratum and check); training-point floors and verdicts are valid.

## Floors (training-point relative L2, median of the last 5 logs; held-out in brackets; S2 = steps to 2x floor)

| case | field | seed | ISLA F [held-out] (S2) | GeoTransolver F [held-out] (S2) | Transolver F [held-out] (S2) | R = ISLA / best | complete |
|---|---|---|---|---|---|---|---|
| af | pressure_l2 | 42 | 0.000923 [0.0104] (14900) | 0.00019 [0.00385] (17400) | 0.000206 [0.00354] (17100) | 4.86 | True |
| af | pressure_l2 | 43 | 0.000779 [0.00804] (15900) | 0.000193 [0.00364] (17200) | 0.000201 [0.0047] (17500) | 4.04 | True |
| af | velocity_l2 | 42 | 0.000803 [0.0116] (16300) | 0.000198 [0.00717] (17400) | 0.000208 [0.00741] (18000) | 4.06 | True |
| af | velocity_l2 | 43 | 0.000812 [0.00712] (15900) | 0.000201 [0.00621] (17500) | 0.00022 [0.00762] (18200) | 4.04 | True |
| af | nut_l2 | 42 | 0.0006 [0.147] (16300) | 0.00024 [0.00786] (17300) | 0.000222 [0.00936] (17200) | 2.7 | True |
| af | nut_l2 | 43 | 0.000552 [0.0765] (16600) | 0.000254 [0.00826] (17300) | 0.000227 [0.00916] (17300) | 2.43 | True |
| az | velocity_l2 | 42 | 0.00491 [0.0163] (10400) | 0.86 [0.86] (15100) | 0.0849 [0.0843] (7900) | 0.0578 | True |
| az | velocity_l2 | 43 | 0.00474 [0.0196] (10900) | 1.81 [1.8] (13200) | 0.342 [0.339] (15100) | 0.0139 | True |
| az | pressure_l2 | 42 | 0.00316 [0.104] (14600) | 1.3 [1.31] (18100) | 0.688 [1.35] (12100) | 0.0046 | True |
| az | pressure_l2 | 43 | 0.00279 [0.106] (14900) | 4.3 [4.32] (19100) | 0.932 [0.934] (17400) | 0.003 | True |
| az | nut_l2 | 42 | 0.0145 [0.0463] (14000) | 1.29 [1.45] (17600) | 0.375 [0.479] (10100) | 0.0387 | True |
| az | nut_l2 | 43 | 0.0148 [0.0519] (9400) | 1.69 [1.81] (18100) | 0.713 [0.764] (17000) | 0.0208 | True |
| az | tke_l2 | 42 | 0.00429 [0.155] (13700) | 2.32 [2.39] (19200) | 0.572 [0.763] (11700) | 0.0075 | True |
| az | tke_l2 | 43 | 0.00422 [0.162] (14100) | 7.01 [7.14] (18400) | 0.949 [0.952] (17600) | 0.00445 | True |
| az | omega_l2 | 42 | 0.00145 [0.351] (15800) | 1.17 [1.22] (16500) | 0.886 [0.904] (7800) | 0.00164 | True |
| az | omega_l2 | 43 | 0.00143 [0.346] (15700) | 1.17 [1.26] (17900) | 0.957 [1.46] (14200) | 0.00149 | True |
| drs | pressure_l2 | 42 | 0.00188 [0.394] (13600) | 0.00135 [0.137] (15900) | 0.000776 [0.127] (16800) | 2.43 | True |
| drs | pressure_l2 | 43 | 0.00201 [0.405] (13200) | 0.00133 [0.136] (16000) | 0.000712 [0.13] (17000) | 2.82 | True |
| drs | wss_l2 | 42 | 0.00531 [0.56] (12700) | 0.00294 [0.195] (15000) | 0.00131 [0.186] (16300) | 4.07 | True |
| drs | wss_l2 | 43 | 0.0129 [0.548] (7900) | 0.00294 [0.195] (14800) | 0.00118 [0.186] (16300) | 10.9 | True |
| drv | pressure_l2 | 42 | 0.000671 [0.453] (16500) | 6.32e-05 [0.525] (18700) | 0.00016 [0.105] (18300) | 10.6 | True |
| drv | pressure_l2 | 43 | 0.000699 [0.419] (16600) | 5.72e-05 [0.545] (19000) | 0.000168 [0.102] (18200) | 12.2 | True |
| drv | velocity_l2 | 42 | 0.00226 [0.517] (14600) | 0.000139 [0.517] (18700) | 0.000359 [0.166] (17900) | 16.2 | True |
| drv | velocity_l2 | 43 | 0.00231 [0.507] (14500) | 0.000129 [0.511] (18700) | 0.000376 [0.168] (17800) | 18 | True |
| drv | nut_l2 | 42 | 0.00101 [0.542] (16300) | 4.53e-05 [0.664] (19100) | 0.000183 [0.244] (18300) | 22.2 | True |
| drv | nut_l2 | 43 | 0.00106 [0.504] (16200) | 4.43e-05 [0.672] (19100) | 0.000191 [0.242] (18000) | 23.9 | True |
| hl18 | pressure_l2 | 42 | 0.00217 [0.087] (13900) | 0.00136 [0.0325] (15100) | 0.000767 [0.0391] (16400) | 2.83 | True |
| hl18 | pressure_l2 | 43 | 0.00203 [0.0799] (13700) | 0.00139 [0.0325] (14600) | 0.000807 [0.0294] (16400) | 2.51 | True |
| hl18 | tau_wall_l2 | 42 | 0.0119 [0.218] (12000) | 0.00676 [0.103] (12000) | 0.00346 [0.105] (14300) | 3.45 | True |
| hl18 | tau_wall_l2 | 43 | 0.0111 [0.202] (11500) | 0.00675 [0.102] (12000) | 0.00366 [0.0954] (14000) | 3.05 | True |
| hl18 | temperature_l2 | 42 | 0.00694 [0.105] (10500) | 0.00412 [0.0512] (11900) | 0.00213 [0.0579] (13600) | 3.26 | True |
| hl18 | temperature_l2 | 43 | 0.00654 [0.0976] (9700) | 0.00405 [0.0525] (12400) | 0.00234 [0.0466] (13600) | 2.8 | True |
| hl18 | density_l2 | 42 | 0.00313 [0.095] (13200) | 0.00192 [0.035] (13700) | 0.00106 [0.0419] (15500) | 2.95 | True |
| hl18 | density_l2 | 43 | 0.00291 [0.0878] (12900) | 0.00197 [0.0347] (13900) | 0.00113 [0.032] (15000) | 2.56 | True |
| hl18 | velocity_l2 | 42 | 0.0109 [0.131] (9400) | 0.00708 [0.0594] (9600) | 0.00387 [0.0608] (13500) | 2.82 | True |
| hl18 | velocity_l2 | 43 | 0.0105 [0.119] (8900) | 0.00703 [0.0595] (9500) | 0.004 [0.0567] (13300) | 2.62 | True |
| hl6 | pressure_l2 | 42 | 0.00157 [0.0419] (13500) | 0.00164 [0.0321] (13800) | 0.000699 [0.0304] (16700) | 2.25 | True |
| hl6 | pressure_l2 | 43 | 0.00166 [0.0416] (13300) | 0.00169 [0.031] (14300) | 0.000587 [0.0324] (16900) | 2.83 | True |
| hl6 | tau_wall_l2 | 42 | 0.00681 [0.126] (12800) | 0.00613 [0.105] (13400) | 0.0017 [0.0993] (16700) | 4 | True |
| hl6 | tau_wall_l2 | 43 | 0.00799 [0.126] (12600) | 0.00657 [0.103] (12800) | 0.00142 [0.1] (16600) | 5.63 | True |
| hl6 | temperature_l2 | 42 | 0.00532 [0.0675] (11500) | 0.00471 [0.0565] (12100) | 0.00114 [0.0532] (16800) | 4.66 | True |
| hl6 | temperature_l2 | 43 | 0.00624 [0.0658] (10700) | 0.00508 [0.056] (11800) | 0.000892 [0.054] (17000) | 6.99 | True |
| hl6 | density_l2 | 42 | 0.00246 [0.0464] (12600) | 0.00237 [0.0358] (13500) | 0.000894 [0.0334] (16300) | 2.75 | True |
| hl6 | density_l2 | 43 | 0.00276 [0.0462] (12500) | 0.00249 [0.0349] (13300) | 0.000729 [0.0351] (16500) | 3.79 | True |
| hl6 | velocity_l2 | 42 | 0.00689 [0.0849] (11500) | 0.00594 [0.0671] (12100) | 0.00179 [0.0632] (16300) | 3.85 | True |
| hl6 | velocity_l2 | 43 | 0.00789 [0.0866] (10800) | 0.00641 [0.0663] (11600) | 0.00144 [0.0642] (16500) | 5.49 | True |
| lug | von_mises_l2 | 42 | 0.000198 [0.0172] (17600) | 3.79e-05 [0.0411] (19000) | 3.14e-05 [0.0189] (19200) | 6.31 | True |
| lug | von_mises_l2 | 43 | 0.00018 [0.015] (17700) | 3.74e-05 [0.0267] (19000) | 3.02e-05 [0.035] (19200) | 5.96 | True |
| t2d | stress_l2 | 42 | 0.00133 [0.00392] (14200) | 0.000473 [0.00677] (16200) | 0.000276 [0.0451] (16500) | 4.83 | True |
| t2d | stress_l2 | 43 | 0.00119 [0.00756] (14300) | 0.000522 [0.00613] (15800) | 0.000256 [0.0252] (17400) | 4.65 | True |
| t2d | displacement_l2 | 42 | 0.000338 [0.00584] (16400) | 0.000176 [0.0109] (16900) | 0.000156 [0.0265] (17500) | 2.16 | True |
| t2d | displacement_l2 | 43 | 0.000271 [0.00743] (16200) | 0.000169 [0.00893] (16600) | 0.000155 [0.0168] (17100) | 1.75 | True |
| t2d | q_l2 | 42 | 0.000653 [0.0326] (15200) | 0.000155 [0.0827] (18400) | 0.000103 [0.0874] (18300) | 6.35 | True |
| t2d | q_l2 | 43 | 0.000548 [0.0826] (15700) | 0.000177 [0.1] (18100) | 9.64e-05 [0.15] (18500) | 5.68 | True |

## Verdicts (headline fields; deficit: R >= 2 on both seeds; advantage: R <= 0.5 on both)

- af:pressure_l2: deficit (R42 4.86, R43 4.04)
- af:velocity_l2: deficit (R42 4.06, R43 4.04)
- af:nut_l2: deficit (R42 2.7, R43 2.43)
- az:velocity_l2: advantage (R42 0.0578, R43 0.0139)
- az:pressure_l2: advantage (R42 0.0046, R43 0.003)
- drs:pressure_l2: deficit (R42 2.43, R43 2.82)
- drs:wss_l2: deficit (R42 4.07, R43 10.9)
- drv:pressure_l2: deficit (R42 10.6, R43 12.2)
- drv:velocity_l2: deficit (R42 16.2, R43 18)
- drv:nut_l2: deficit (R42 22.2, R43 23.9)
- hl18:pressure_l2: deficit (R42 2.83, R43 2.51)
- hl18:tau_wall_l2: deficit (R42 3.45, R43 3.05)
- hl6:pressure_l2: deficit (R42 2.25, R43 2.83)
- hl6:tau_wall_l2: deficit (R42 4, R43 5.63)
- lug:von_mises_l2: deficit (R42 6.31, R43 5.96)
- t2d:stress_l2: deficit (R42 4.83, R43 4.65)
- t2d:displacement_l2: level (R42 2.16, R43 1.75)
- t2d:q_l2: deficit (R42 6.35, R43 5.68)

## Budget check (40k / 20k floor, seed 42) and bf16 / fp32 (DrivAerML Surface, seed 42)

- af gt 40k: pressure_l2 0.632, velocity_l2 0.644, nut_l2 0.549 (complete True)
- af isla 40k: pressure_l2 0.316, velocity_l2 0.611, nut_l2 0.606 (complete True)
- af t 40k: pressure_l2 0.516, velocity_l2 0.553, nut_l2 0.469 (complete True)
- drs gt bf16: pressure_l2 4.45, wss_l2 3.41 (complete True)
- drs isla bf16: pressure_l2 2.84, wss_l2 1.8 (complete True)
- drs t bf16: pressure_l2 5.48, wss_l2 5.03 (complete True)
- lug gt 40k: von_mises_l2 0.437 (complete True)
- lug isla 40k: von_mises_l2 0.487 (complete True)
- lug t 40k: von_mises_l2 0.425 (complete True)
- t2d gt 40k: stress_l2 0.346, displacement_l2 0.504, q_l2 0.416 (complete True)
- t2d isla 40k: stress_l2 0.805, displacement_l2 0.73, q_l2 0.48 (complete True)
- t2d t 40k: stress_l2 0.337, displacement_l2 0.422, q_l2 0.379 (complete True)

## Cost (wall h, median s/step, peak GiB)

- cap_af_gt_40k_s42: 3.94 h, 0.353 s/step, 50.6 GiB, complete True
- cap_af_gt_s42: 1.97 h, 0.353 s/step, 50.6 GiB, complete True
- cap_af_gt_s43: 1.97 h, 0.353 s/step, 50.6 GiB, complete True
- cap_af_isla_40k_s42: 5.08 h, 1.136 s/step, 59.7 GiB, complete True
- cap_af_isla_s42: 6.30 h, 1.128 s/step, 59.5 GiB, complete True
- cap_af_isla_s43: 6.30 h, 1.128 s/step, 59.5 GiB, complete True
- cap_af_t_40k_s42: 2.80 h, 0.837 s/step, 87.4 GiB, complete True
- cap_af_t_s42: 4.63 h, 0.830 s/step, 87.3 GiB, complete True
- cap_af_t_s43: 4.63 h, 0.831 s/step, 87.3 GiB, complete True
- cap_az_gt_s42: 3.76 h, 0.673 s/step, 98.9 GiB, complete True
- cap_az_gt_s43: 3.76 h, 0.673 s/step, 98.9 GiB, complete True
- cap_az_isla_s42: 9.34 h, 2.789 s/step, 229.8 GiB, complete True
- cap_az_isla_s43: 9.34 h, 2.789 s/step, 229.8 GiB, complete True
- cap_az_t_s42: 3.64 h, 0.653 s/step, 96.1 GiB, complete True
- cap_az_t_s43: 3.65 h, 0.652 s/step, 96.1 GiB, complete True
- cap_drs_gt_bf16_s42: 2.22 h, 0.398 s/step, 77.6 GiB, complete True
- cap_drs_gt_s42: 3.83 h, 0.687 s/step, 101.0 GiB, complete True
- cap_drs_gt_s43: 3.84 h, 0.687 s/step, 101.0 GiB, complete True
- cap_drs_isla_bf16_s42: 1.97 h, 0.353 s/step, 85.8 GiB, complete True
- cap_drs_isla_s42: 1.15 h, 1.367 s/step, 115.7 GiB, complete True
- cap_drs_isla_s43: 7.62 h, 1.365 s/step, 115.5 GiB, complete True
- cap_drs_t_bf16_s42: 3.06 h, 0.547 s/step, 123.7 GiB, complete True
- cap_drs_t_s42: 1.38 h, 1.642 s/step, 174.1 GiB, complete True
- cap_drs_t_s43: 1.38 h, 1.645 s/step, 174.1 GiB, complete True
- cap_drv_gt_s42: 1.87 h, 1.674 s/step, 160.2 GiB, complete True
- cap_drv_gt_s43: 2.81 h, 1.674 s/step, 160.2 GiB, complete True
- cap_drv_isla_s42: 9.44 h, 2.818 s/step, 231.8 GiB, complete True
- cap_drv_isla_s43: 9.43 h, 2.815 s/step, 231.8 GiB, complete True
- cap_drv_t_s42: 1.38 h, 1.644 s/step, 174.1 GiB, complete True
- cap_drv_t_s43: 1.38 h, 1.645 s/step, 174.1 GiB, complete True
- cap_hl18_gt_s42: 3.85 h, 0.690 s/step, 101.0 GiB, complete True
- cap_hl18_gt_s43: 3.86 h, 0.691 s/step, 101.0 GiB, complete True
- cap_hl18_isla_s42: 1.15 h, 1.369 s/step, 115.7 GiB, complete True
- cap_hl18_isla_s43: 7.62 h, 1.365 s/step, 115.5 GiB, complete True
- cap_hl18_t_s42: 1.38 h, 1.651 s/step, 174.1 GiB, complete True
- cap_hl18_t_s43: 1.38 h, 1.649 s/step, 174.1 GiB, complete True
- cap_hl6_gt_s42: 3.92 h, 0.702 s/step, 101.0 GiB, complete True
- cap_hl6_gt_s43: 3.76 h, 0.709 s/step, 101.0 GiB, complete True
- cap_hl6_isla_s42: 7.73 h, 1.386 s/step, 115.4 GiB, complete True
- cap_hl6_isla_s43: 7.73 h, 1.386 s/step, 115.4 GiB, complete True
- cap_hl6_t_s42: 1.86 h, 1.671 s/step, 174.0 GiB, complete True
- cap_hl6_t_s43: 1.86 h, 1.669 s/step, 174.0 GiB, complete True
- cap_lug_gt_40k_s42: 0.85 h, 0.074 s/step, 6.2 GiB, complete True
- cap_lug_gt_s42: 0.44 h, 0.078 s/step, 6.2 GiB, complete True
- cap_lug_gt_s43: 0.44 h, 0.078 s/step, 6.2 GiB, complete True
- cap_lug_isla_40k_s42: 2.85 h, 0.255 s/step, 12.1 GiB, complete True
- cap_lug_isla_s42: 1.43 h, 0.254 s/step, 12.1 GiB, complete True
- cap_lug_isla_s43: 1.43 h, 0.257 s/step, 12.1 GiB, complete True
- cap_lug_t_40k_s42: 1.36 h, 0.121 s/step, 11.2 GiB, complete True
- cap_lug_t_s42: 0.68 h, 0.120 s/step, 11.2 GiB, complete True
- cap_lug_t_s43: 0.68 h, 0.120 s/step, 11.2 GiB, complete True
- cap_t2d_gt_40k_s42: 0.91 h, 0.080 s/step, 4.0 GiB, complete True
- cap_t2d_gt_s42: 0.50 h, 0.076 s/step, 4.0 GiB, complete True
- cap_t2d_gt_s43: 0.51 h, 0.082 s/step, 4.0 GiB, complete True
- cap_t2d_isla_40k_s42: 1.73 h, 0.151 s/step, 5.5 GiB, complete True
- cap_t2d_isla_s42: 0.90 h, 0.157 s/step, 5.5 GiB, complete True
- cap_t2d_isla_s43: 0.83 h, 0.145 s/step, 5.5 GiB, complete True
- cap_t2d_t_40k_s42: 0.91 h, 0.080 s/step, 7.0 GiB, complete True
- cap_t2d_t_s42: 0.45 h, 0.080 s/step, 7.0 GiB, complete True
- cap_t2d_t_s43: 0.45 h, 0.079 s/step, 7.0 GiB, complete True
