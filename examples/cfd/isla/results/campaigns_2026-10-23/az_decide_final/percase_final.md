# Per-case structure of the test-side errors (seed mean per case; beside the rules)

## velocity_l2: per-case distribution

| split | arm | cases | mean | median | p90 | max | share of the mean from the 5 worst |
|---|---|---|---|---|---|---|---|
| full | ISLA | 716 | 0.1181 | 0.1071 | 0.2047 | 0.4717 | 0.024 |
| full | ISLA-DTT | 716 | 0.1105 | 0.1005 | 0.1898 | 0.5394 | 0.026 |
| full | GeoTransolver | 716 | 0.609 | 0.5938 | 0.9044 | 2.685 | 0.019 |
| full | Transolver | 716 | 0.9975 | 0.9999 | 1.006 | 1.069 | 0.007 |
| nonbox | ISLA | 1074 | 0.5332 | 0.4796 | 0.9422 | 2.616 | 0.022 |
| nonbox | ISLA-DTT | 1074 | 0.4913 | 0.4366 | 0.8132 | 2.686 | 0.024 |
| nonbox | GeoTransolver | 1074 | 0.5906 | 0.4502 | 0.874 | 28.58 | 0.120 |
| reynolds | ISLA | 1385 | 0.1642 | 0.1527 | 0.2716 | 0.9389 | 0.016 |
| reynolds | ISLA-DTT | 1385 | 0.1522 | 0.1403 | 0.2558 | 0.6043 | 0.013 |
| reynolds | GeoTransolver | 1385 | 0.7095 | 0.6636 | 0.9269 | 3.652 | 0.012 |
| blobs | ISLA | 1367 | 0.1452 | 0.141 | 0.2113 | 0.5754 | 0.013 |
| blobs | ISLA-DTT | 1367 | 0.1297 | 0.1241 | 0.1954 | 0.602 | 0.015 |
| blobs | GeoTransolver | 1367 | 0.7179 | 0.6981 | 0.9303 | 5.315 | 0.013 |
| bank_zero_to_cad | ISLA | 333 | 0.175 | 0.164 | 0.2944 | 0.6238 | 0.040 |
| bank_zero_to_cad | ISLA-DTT | 333 | 0.1666 | 0.1484 | 0.2952 | 0.637 | 0.044 |
| bank_zero_to_cad | GeoTransolver | 333 | 0.5057 | 0.4458 | 0.8882 | 1.771 | 0.038 |
| profile_crossflow | ISLA | 578 | 1.419 | 1.444 | 1.68 | 1.863 | 0.011 |
| profile_crossflow | ISLA-DTT | 578 | 1.337 | 1.378 | 1.674 | 1.883 | 0.012 |
| profile_crossflow | GeoTransolver | 578 | 0.8312 | 0.8565 | 0.9746 | 1.128 | 0.011 |
| profile_dual_inlet | ISLA | 516 | 0.6525 | 0.6529 | 0.712 | 0.7721 | 0.011 |
| profile_dual_inlet | ISLA-DTT | 516 | 0.6065 | 0.6131 | 0.6703 | 0.7462 | 0.012 |
| profile_dual_inlet | GeoTransolver | 516 | 0.7885 | 0.7909 | 0.8712 | 0.961 | 0.012 |

## pressure_l2: per-case distribution

| split | arm | cases | mean | median | p90 | max | share of the mean from the 5 worst |
|---|---|---|---|---|---|---|---|
| full | ISLA | 716 | 0.3037 | 0.2874 | 0.5815 | 1.158 | 0.023 |
| full | ISLA-DTT | 716 | 0.2561 | 0.246 | 0.4854 | 1.113 | 0.026 |
| full | GeoTransolver | 716 | 1.25 | 1 | 1.369 | 26.75 | 0.088 |
| full | Transolver | 716 | 1.167 | 1 | 1.172 | 24.45 | 0.073 |
| nonbox | ISLA | 1074 | 2.774 | 1.507 | 6.283 | 30.22 | 0.041 |
| nonbox | ISLA-DTT | 1074 | 3.534 | 2.382 | 7.414 | 61 | 0.050 |
| nonbox | GeoTransolver | 1074 | 1.168 | 1 | 1.353 | 9.875 | 0.027 |
| reynolds | ISLA | 1385 | 0.5092 | 0.5358 | 0.8078 | 3.579 | 0.020 |
| reynolds | ISLA-DTT | 1385 | 0.4727 | 0.456 | 0.7876 | 3.918 | 0.023 |
| reynolds | GeoTransolver | 1385 | 1.036 | 1 | 1.031 | 12.14 | 0.018 |
| blobs | ISLA | 1367 | 0.3881 | 0.4186 | 0.6855 | 1.879 | 0.012 |
| blobs | ISLA-DTT | 1367 | 0.3069 | 0.3286 | 0.5306 | 1.894 | 0.015 |
| blobs | GeoTransolver | 1367 | 1.114 | 1 | 1.182 | 6.781 | 0.020 |
| bank_zero_to_cad | ISLA | 333 | 0.3774 | 0.369 | 0.6565 | 1.078 | 0.041 |
| bank_zero_to_cad | ISLA-DTT | 333 | 0.3607 | 0.3514 | 0.6553 | 1.09 | 0.043 |
| bank_zero_to_cad | GeoTransolver | 333 | 1.123 | 1.001 | 1.255 | 3.949 | 0.048 |
| profile_crossflow | ISLA | 578 | 2.661 | 2.492 | 4.339 | 6.522 | 0.021 |
| profile_crossflow | ISLA-DTT | 578 | 2.193 | 2.249 | 3.068 | 4.143 | 0.016 |
| profile_crossflow | GeoTransolver | 578 | 1.003 | 1 | 1.01 | 1.115 | 0.009 |
| profile_dual_inlet | ISLA | 516 | 0.799 | 0.7946 | 0.9533 | 0.9971 | 0.012 |
| profile_dual_inlet | ISLA-DTT | 516 | 0.8262 | 0.8529 | 0.9635 | 0.9977 | 0.012 |
| profile_dual_inlet | GeoTransolver | 516 | 1 | 1 | 1.001 | 1.012 | 0.010 |

## nonbox: the 10 worst ISLA cases by pressure_l2 (seed mean; seeds 42 / 43), GeoTransolver on the same cases

| case | ISLA | ISLA seeds | GT | ISLA vel | GT vel | ISLA p | GT p | family | profile | U_m | L_m | nu_m | log10_re |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9c74c47c8862 | 30.22 | 52.7 / 7.739 | 0.9986 | 0.6953 | 0.3607 | 30.22 | 0.9986 | external_opposed_hemisphere | opposed_hemisphere_symmetry | 14.1 | 11.12 | 1.988e-06 | 7.897 |
| e8dbcd90d346 | 24.06 | 38.9 / 9.211 | 0.9854 | 0.8035 | 0.4198 | 24.06 | 0.9854 | external_opposed_hemisphere | opposed_hemisphere_symmetry | 1.868 | 8.519 | 1.49e-06 | 7.029 |
| 2f0436ec5be6 | 23.66 | 40.57 / 6.743 | 0.9961 | 0.7688 | 0.378 | 23.66 | 0.9961 | external_opposed_hemisphere | opposed_hemisphere_symmetry | 15.02 | 0.09843 | 2.544e-08 | 7.764 |
| 5741908096fa | 22.8 | 28.66 / 16.95 | 6.004 | 0.2227 | 0.1454 | 22.8 | 6.004 | external_opposed_sphere | opposed_hemispheres | 0.722 | 0.7988 | 0.0001631 | 3.548 |
| 941ee9a1f234 | 21.56 | 29.76 / 13.35 | 0.9783 | 1.534 | 0.6482 | 21.56 | 0.9783 | internal_duct | bend | 1.103 | 0.05636 | 3.278e-06 | 4.278 |
| 15203d5819a0 | 20.23 | 36.3 / 4.154 | 1.029 | 0.5472 | 0.281 | 20.23 | 1.029 | external_opposed_hemisphere | opposed_hemisphere_symmetry | 3.343 | 0.02085 | 1.067e-09 | 7.815 |
| 1f5c36724a1e | 19.86 | 25.22 / 14.51 | 0.9982 | 1.581 | 0.7144 | 19.86 | 0.9982 | internal_duct | bend | 2.868 | 6.207 | 2.847e-05 | 5.796 |
| 08bfe70b6f73 | 17.63 | 20.9 / 14.36 | 2.856 | 0.2281 | 0.166 | 17.63 | 2.856 | external_opposed_sphere | opposed_hemispheres | 1.108 | 4.065 | 0.0009207 | 3.69 |
| 0bf0a69bd121 | 17.22 | 29.04 / 5.411 | 1.003 | 0.6283 | 0.3002 | 17.22 | 1.003 | external_opposed_hemisphere | opposed_hemisphere_symmetry | 34.71 | 1.312 | 3.016e-05 | 6.179 |
| 9e53c2d9ad81 | 16.44 | 27.51 / 5.374 | 1.004 | 0.7482 | 0.4317 | 16.44 | 1.004 | external_opposed_hemisphere | opposed_hemisphere_symmetry | 22.3 | 0.1285 | 8.853e-08 | 7.51 |

Mean without these 10: ISLA 2.599; GT 1.164 (with them: ISLA 2.774, GT 1.168)

## nonbox: the 10 worst ISLA cases by velocity_l2 (seed mean; seeds 42 / 43), GeoTransolver on the same cases

| case | ISLA | ISLA seeds | GT | ISLA vel | GT vel | ISLA p | GT p | family | profile | U_m | L_m | nu_m | log10_re |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 742c63329dd1 | 2.616 | 2.659 / 2.573 | 1.789 | 2.616 | 1.789 | 1.395 | 1.484 | internal_duct | sudden_expansion | 0.4227 | 0.5393 | 2.46e-06 | 4.967 |
| 8373a18555f7 | 2.61 | 2.692 / 2.528 | 1.196 | 2.61 | 1.196 | 1.431 | 0.9995 | internal_duct | sudden_expansion | 9.747 | 0.02636 | 3.549e-06 | 4.86 |
| 10fa871112d7 | 2.579 | 2.633 / 2.526 | 1.958 | 2.579 | 1.958 | 1.45 | 1.003 | internal_duct | sudden_expansion | 6.76 | 0.1334 | 1.093e-05 | 4.916 |
| e0abc95b3231 | 2.488 | 2.573 / 2.403 | 0.874 | 2.488 | 0.874 | 1.513 | 1.005 | internal_duct | sudden_expansion | 2.466 | 0.1603 | 1.913e-06 | 5.315 |
| 2fa54ac390d4 | 2.479 | 2.558 / 2.4 | 1.961 | 2.479 | 1.961 | 1.355 | 1.229 | internal_duct | sudden_expansion | 0.5583 | 0.5163 | 3.823e-06 | 4.877 |
| 217fe7dd0d0e | 2.415 | 2.529 / 2.301 | 3.356 | 2.415 | 3.356 | 1.879 | 0.9997 | internal_duct | sudden_expansion | 26.52 | 0.2317 | 0.003232 | 3.279 |
| 2f82d828c6cf | 2.388 | 2.471 / 2.305 | 1.691 | 2.388 | 1.691 | 1.489 | 0.9996 | internal_duct | sudden_expansion | 23.43 | 10.81 | 0.0005711 | 5.647 |
| e61a915541a7 | 2.041 | 2.123 / 1.958 | 1.594 | 2.041 | 1.594 | 1.502 | 1.003 | internal_duct | sudden_expansion | 6.662 | 0.5852 | 0.0001143 | 4.533 |
| 71e501898b52 | 1.925 | 2.013 / 1.837 | 7.452 | 1.925 | 7.452 | 9.089 | 1.18 | internal_duct | bend | 0.2792 | 16.44 | 5.538e-08 | 7.919 |
| fd0d4f8f3275 | 1.881 | 1.939 / 1.823 | 0.8875 | 1.881 | 0.8875 | 1.373 | 1.23 | internal_duct | sudden_expansion | 0.5587 | 0.02344 | 3.519e-07 | 4.571 |

Mean without these 10: ISLA 0.5162; GT 0.5747 (with them: ISLA 0.5332, GT 0.5906)

## profile_crossflow: the 10 worst ISLA cases by pressure_l2 (seed mean; seeds 42 / 43), GeoTransolver on the same cases

| case | ISLA | ISLA seeds | GT | ISLA vel | GT vel | ISLA p | GT p | family | profile | U_m | L_m | nu_m | log10_re |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ed08a530274e | 6.522 | 9.19 / 3.855 | 1 | 1.439 | 0.7932 | 6.522 | 1 | external_box | crossflow | 1.57 | 10.45 | 0.002677 | 3.787 |
| a6062b4b9dd7 | 6.432 | 8.828 / 4.035 | 1 | 1.213 | 0.5202 | 6.432 | 1 | external_box | crossflow | 48.39 | 5.121 | 0.02549 | 3.988 |
| ba1d10516afb | 6.406 | 7.721 / 5.091 | 0.999 | 1.656 | 0.7712 | 6.406 | 0.999 | external_box | crossflow | 10.48 | 0.888 | 8.44e-07 | 7.043 |
| b4cebc66695a | 6.347 | 8.59 / 4.104 | 0.9987 | 1.508 | 0.5535 | 6.347 | 0.9987 | external_box | crossflow | 15.89 | 0.03195 | 2.723e-07 | 6.271 |
| f840c36493af | 6.341 | 8.893 / 3.789 | 0.9981 | 1.235 | 0.4999 | 6.341 | 0.9981 | external_box | crossflow | 2.172 | 7.654 | 0.008695 | 3.282 |
| add997a6b305 | 6.284 | 6.868 / 5.699 | 0.9991 | 1.367 | 0.6118 | 6.284 | 0.9991 | external_box | crossflow | 19.07 | 0.07687 | 0.001218 | 3.08 |
| 226a3208bac7 | 6.13 | 9.261 / 2.998 | 0.9774 | 1.218 | 0.5456 | 6.13 | 0.9774 | external_box | crossflow | 0.4569 | 0.05332 | 9.842e-08 | 5.394 |
| 02c458d9e782 | 6.019 | 6.744 / 5.295 | 0.9982 | 1.101 | 0.4801 | 6.019 | 0.9982 | external_box | crossflow | 10.81 | 0.1025 | 1.165e-05 | 4.978 |
| 37e91c9c4e71 | 6.007 | 7.152 / 4.862 | 1.098 | 1.428 | 0.7367 | 6.007 | 1.098 | external_box | crossflow | 0.401 | 0.04566 | 3.311e-06 | 3.743 |
| c9bb7a9b07ee | 5.881 | 7.697 / 4.065 | 1.075 | 1.424 | 0.5562 | 5.881 | 1.075 | external_box | crossflow | 0.4497 | 1.808 | 4.576e-06 | 5.25 |

Mean without these 10: ISLA 2.598; GT 1.003 (with them: ISLA 2.661, GT 1.003)

## profile_crossflow: the 10 worst ISLA cases by velocity_l2 (seed mean; seeds 42 / 43), GeoTransolver on the same cases

| case | ISLA | ISLA seeds | GT | ISLA vel | GT vel | ISLA p | GT p | family | profile | U_m | L_m | nu_m | log10_re |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 03429dae50a5 | 1.863 | 1.966 / 1.761 | 0.7147 | 1.863 | 0.7147 | 4.525 | 1.004 | external_box | crossflow | 0.2649 | 1.424 | 5.473e-05 | 3.838 |
| d17988768276 | 1.846 | 1.926 / 1.765 | 0.9099 | 1.846 | 0.9099 | 2.69 | 1 | external_box | crossflow | 22.03 | 7.673 | 0.0004862 | 5.541 |
| ef7e85685559 | 1.836 | 1.943 / 1.728 | 0.834 | 1.836 | 0.834 | 4.684 | 0.9992 | external_box | crossflow | 3.209 | 3.224 | 2.647e-06 | 6.592 |
| 0c81c7ae55f8 | 1.825 | 1.978 / 1.672 | 0.8406 | 1.825 | 0.8406 | 3.171 | 1.001 | external_box | crossflow | 2.866 | 1.044 | 6.049e-06 | 5.694 |
| d3e5635b37cf | 1.823 | 1.736 / 1.91 | 0.7901 | 1.823 | 0.7901 | 4.3 | 1.002 | external_box | crossflow | 3.318 | 0.06952 | 1.495e-06 | 5.188 |
| a238c6dde3ac | 1.809 | 1.864 / 1.755 | 0.822 | 1.809 | 0.822 | 3.551 | 1.003 | external_box | crossflow | 3.711 | 0.0209 | 6.043e-07 | 5.108 |
| 13d255b285fc | 1.808 | 1.921 / 1.696 | 0.7732 | 1.808 | 0.7732 | 4.667 | 1 | external_box | crossflow | 20.56 | 0.5859 | 5.084e-06 | 6.375 |
| a6ce4bbb283c | 1.806 | 2.007 / 1.605 | 0.7541 | 1.806 | 0.7541 | 3.475 | 1 | external_box | crossflow | 31.57 | 0.06861 | 1.752e-06 | 6.092 |
| 790ecea3bf8c | 1.802 | 1.965 / 1.638 | 0.833 | 1.802 | 0.833 | 2.908 | 1 | external_box | crossflow | 49.55 | 1.278 | 0.0001531 | 5.617 |
| 16650c5a229f | 1.798 | 1.822 / 1.773 | 0.8592 | 1.798 | 0.8592 | 3.918 | 0.9905 | external_box | crossflow | 0.4029 | 0.8374 | 1.509e-06 | 5.349 |

Mean without these 10: ISLA 1.412; GT 0.8315 (with them: ISLA 1.419, GT 0.8312)

