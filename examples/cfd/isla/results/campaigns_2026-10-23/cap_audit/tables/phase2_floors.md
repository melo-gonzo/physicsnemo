## Part A: the ladder (AirfRANS, random draw; F_ho = held-out floor, seed values and geometric mean)

| rung | params | s/step | pressure_l2 F_ho | velocity_l2 F_ho | nut_l2 F_ho | complete | path check (compiled vs eager, held-out pressure) |
|---|---|---|---|---|---|---|---|
| w256 | 18.1 M | 0.604 | 0.00209 | 0.00122 | 0.00172 | True | 5.8e-07 |
| ref | 68.8 M | 0.991 | 0.00198 / 0.00199 (gm 0.00198) | 0.00121 / 0.00122 (gm 0.00121) | 0.00174 / 0.00129 (gm 0.0015) | True | 6.9e-07, 1.3e-06 |
| w1024 | 268.2 M | 2.285 | 0.00168 / 0.00221 (gm 0.00193) | 0.00117 / 0.00133 (gm 0.00124) | 0.00163 / 0.00165 (gm 0.00164) | True | 1.5e-06, 1.2e-06 |
| d6 | 34.9 M | 0.520 | 0.00209 | 0.00134 | 0.00178 | True | 1.1e-06 |
| d24 | 136.6 M | 1.926 | 0.00192 / 0.00169 (gm 0.0018) | 0.00119 / 0.00117 (gm 0.00118) | 0.00127 / 0.00123 (gm 0.00125) | True | 2.2e-06, 1.0e-06 |
| s128 | 66.3 M | 0.640 | 0.0014 | 0.00108 | 0.00143 | True | 7.9e-07 |
| s2048 | 79.0 M | 1.946 | 0.00187 / 0.00216 (gm 0.00201) | 0.0012 / 0.00137 (gm 0.00128) | 0.00153 / 0.00202 (gm 0.00176) | True | 9.3e-07, 1.4e-06 |
| w1024_40k | 268.2 M | 2.286 | 0.000802 | 0.000752 | 0.00115 | True | - |
| d24_40k | 136.6 M | 1.931 | 0.000637 | 0.000647 | 0.000835 | True | 2.5e-06 |
| s2048_40k | 79.0 M | 1.949 | 0.000708 | 0.00061 | 0.000929 | True | 2.4e-06 |

Phase-1b reference beside it (seeds 42 / 43, eager, code_r090; not in the fit): pressure_l2 0.00165 / 0.00168, velocity_l2 0.00112 / 0.00119, nut_l2 0.00165 / 0.00157

## Axis verdicts (pressure decides; all fields shown)

- pressure_l2 width: b -0.058, ref/top 1.029, per doubling 0.972, seeds separated False -> **plateau**; 40k/20k 0.476 (budget-limited True); b with the 40k top point -0.691
- pressure_l2 depth: b -0.106, ref/top 1.101, per doubling 0.909, seeds separated True -> **weak gain**; 40k/20k 0.332 (budget-limited True); b with the 40k top point -0.856
- pressure_l2 slices: b 0.132, ref/top 0.986, per doubling 1.007, seeds separated False -> **plateau**; 40k/20k 0.379 (budget-limited True); b with the 40k top point -0.245
- velocity_l2 width: b 0.016, ref/top 0.974, per doubling 1.027, seeds separated False -> **plateau**; 40k/20k 0.645 (budget-limited True); b with the 40k top point -0.347
- velocity_l2 depth: b -0.088, ref/top 1.024, per doubling 0.977, seeds separated True -> **plateau**; 40k/20k 0.542 (budget-limited True); b with the 40k top point -0.524
- velocity_l2 slices: b 0.061, ref/top 0.947, per doubling 1.027, seeds separated False -> **plateau**; 40k/20k 0.51 (budget-limited True); b with the 40k top point -0.206
- nut_l2 width: b -0.034, ref/top 0.913, per doubling 1.096, seeds separated False -> **worse**; 40k/20k 0.703 (budget-limited False); b with the 40k top point -0.292
- nut_l2 depth: b -0.255, ref/top 1.199, per doubling 0.834, seeds separated True -> **weak gain**; 40k/20k 0.66 (budget-limited True); b with the 40k top point -0.545
- nut_l2 slices: b 0.074, ref/top 0.851, per doubling 1.084, seeds separated False -> **worse**; 40k/20k 0.607 (budget-limited True); b with the 40k top point -0.156

**Reading for Peter's question (pressure, 15.7 condition 1):** mixed: width plateau, depth weak gain, slices plateau
Best top-rung pressure F_ho 0.00168 (40k: 0.000637) against GeoTransolver's 1b F_ho 0.000626 (ratio 2.69, 40k 1.02)

## Part B: DrivAerML Volume random-draw control

| field | seed | ISLA F_ho [F_tr] | GT F_ho [F_tr] | T F_ho [F_tr] | R_drv | Q isla / gt / t | complete |
|---|---|---|---|---|---|---|---|
| pressure_l2 | 42 | 0.0101 [0.0108] | 0.00648 [0.00658] | 0.00589 [0.00594] | 1.718 | 0.0223 / 0.0124 / 0.0561 | True |
| pressure_l2 | 43 | 0.0115 [0.0109] | 0.00653 [0.00659] | 0.00576 [0.00565] | 1.995 | 0.0274 / 0.012 / 0.0563 | True |
| velocity_l2 | 42 | 0.0192 [0.0191] | 0.0101 [0.0112] | 0.00878 [0.00894] | 2.183 | 0.0371 / 0.0196 / 0.0528 | True |
| velocity_l2 | 43 | 0.02 [0.0194] | 0.00968 [0.00981] | 0.00864 [0.00853] | 2.316 | 0.0395 / 0.0189 / 0.0514 | True |
| nut_l2 | 42 | 0.0239 [0.0245] | 0.0109 [0.0101] | 0.00978 [0.00933] | 2.442 | 0.0441 / 0.0163 / 0.04 | True |
| nut_l2 | 43 | 0.0219 [0.026] | 0.0106 [0.01] | 0.00996 [0.0092] | 2.203 | 0.0436 / 0.0157 / 0.0412 | True |
- pressure_l2: R_drv 1.72 / 1.99 -> no deficit
- velocity_l2: R_drv 2.18 / 2.32 -> DEFICIT
- nut_l2: R_drv 2.44 / 2.2 -> DEFICIT

Cost: cap_af_isla_p2d24_40k_s44: 1.9308909177780151 s/step, 102.60858106613159 GiB, draw wait 7.867813110351562e-06; cap_af_isla_p2d24_s44: 1.9257391691207886 s/step, 102.60858106613159 GiB, draw wait 7.62939453125e-06; cap_af_isla_p2d24_s45: 1.930673360824585 s/step, 102.60858106613159 GiB, draw wait 9.059906005859375e-06; cap_af_isla_p2d6_s44: 0.5196199417114258 s/step, 29.601529598236084 GiB, draw wait 6.4373016357421875e-06; cap_af_isla_p2ref_s44: 0.9909055233001709 s/step, 54.02401685714722 GiB, draw wait 6.67572021484375e-06; cap_af_isla_p2ref_s45: 0.9909813404083252 s/step, 54.02401685714722 GiB, draw wait 6.4373016357421875e-06; cap_af_isla_p2s128_s44: 0.639551043510437 s/step, 46.831241607666016 GiB, draw wait 7.152557373046875e-06; cap_af_isla_p2s2048_40k_s44: 1.9490725994110107 s/step, 82.96439123153687 GiB, draw wait 7.62939453125e-06; cap_af_isla_p2s2048_s44: 1.9463493824005127 s/step, 82.96439123153687 GiB, draw wait 1.049041748046875e-05; cap_af_isla_p2s2048_s45: 1.9460458755493164 s/step, 82.96439123153687 GiB, draw wait 1.049041748046875e-05; cap_af_isla_p2w1024_40k_s44: 2.2857402563095093 s/step, 100.40389680862427 GiB, draw wait 1.33514404296875e-05; cap_af_isla_p2w1024_s44: 2.2851654291152954 s/step, 100.40389680862427 GiB, draw wait 8.821487426757812e-06; cap_af_isla_p2w1024_s45: 2.2852863073349 s/step, 100.40389680862427 GiB, draw wait 7.152557373046875e-06; cap_af_isla_p2w256_s44: 0.6042038202285767 s/step, 31.497053146362305 GiB, draw wait 7.152557373046875e-06; cap_drv_gt_rd_s42: 1.7345235347747803 s/step, 160.23197507858276 GiB, draw wait 0.06479716300964355; cap_drv_gt_rd_s43: 1.7339366674423218 s/step, 160.23197507858276 GiB, draw wait 0.06435179710388184; cap_drv_isla_rd_s42: 2.9983837604522705 s/step, 234.38781118392944 GiB, draw wait 0.16691303253173828; cap_drv_isla_rd_s43: 2.9983394145965576 s/step, 234.38781118392944 GiB, draw wait 0.16562294960021973; cap_drv_t_rd_s42: 1.7232112884521484 s/step, 174.09832859039307 GiB, draw wait 0.06698775291442871; cap_drv_t_rd_s43: 1.7232102155685425 s/step, 174.09832859039307 GiB, draw wait 0.06754350662231445
