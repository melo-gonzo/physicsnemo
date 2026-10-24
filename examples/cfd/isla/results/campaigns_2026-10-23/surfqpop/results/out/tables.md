## r090_dr_isla_seed42 (48 cases)

Validity: V1 rebuilt native block identical on 48/48 cases (prediction max change over RMS 0.0e+00); V2 eager vs scorer max 3.9e-06, median 2.6e-07; targets of shared cells agree to 0.0e+00.

### pressure

| N | spread s | delta = s/sqrt2 | delta / e(200k) | s x sqrt(N/200k) | error, uniform | error, block | error / error(200k) [95%] |
|---|---|---|---|---|---|---|---|
| 50k | 0.02484 | 0.01757 | 0.424 | 0.01242 | 0.04449 | 0.04415 | 1.0751 [1.0646, 1.0863] |
| 100k | 0.01738 | 0.01229 | 0.297 | 0.01229 | 0.04229 | 0.04232 | 1.0218 [1.0153, 1.0279] |
| 200k | 0.01233 | 0.00872 | 0.211 | 0.01233 | 0.04138 | 0.04122 | 1.0000 [1.0000, 1.0000] |
| 400k | 0.00857 | 0.00606 | 0.146 | 0.01212 | 0.04080 | 0.04081 | 0.9860 [0.9826, 0.9895] |
| 800k | 0.00606 | 0.00429 | 0.104 | 0.01212 | 0.04058 | 0.04067 | 0.9807 [0.9771, 0.9842] |

Slope of log s on log N: -0.509 (95% case bootstrap [-0.517, -0.501]; unweighted norm -0.514). Plateau (first N within 1% of 800k): 400k. Excess e^2 50k-800k 3.32e-04 vs delta^2 2.90e-04.

Q2: R 0.04136, 4 x 50k chunks 0.04446, ratio 1.0749 [1.0669, 1.0833], median per case 1.0809; median deviation from R 0.0166.

Q3 low_x: half alone 0.66925, same cells in R 0.03646, ratio 18.353 [17.391, 19.352], median per case 19.095; median deviation from R 0.6678.

Q3 high_x: half alone 0.86643, same cells in R 0.05029, ratio 17.228 [15.639, 19.006], median per case 19.615; median deviation from R 0.8763.

Q4: R repeated: d max 0.0e+00, median 0.0e+00, max change/RMS 0.0e+00; native repeated: d max 0.0e+00, median 0.0e+00, max change/RMS 0.0e+00.

### wss

| N | spread s | delta = s/sqrt2 | delta / e(200k) | s x sqrt(N/200k) | error, uniform | error, block | error / error(200k) [95%] |
|---|---|---|---|---|---|---|---|
| 50k | 0.03596 | 0.02543 | 0.382 | 0.01798 | 0.07094 | 0.07008 | 1.0657 [1.0568, 1.0743] |
| 100k | 0.02508 | 0.01773 | 0.266 | 0.01773 | 0.06785 | 0.06783 | 1.0193 [1.0149, 1.0236] |
| 200k | 0.01799 | 0.01272 | 0.191 | 0.01799 | 0.06656 | 0.06635 | 1.0000 [1.0000, 1.0000] |
| 400k | 0.01271 | 0.00899 | 0.135 | 0.01797 | 0.06581 | 0.06571 | 0.9887 [0.9860, 0.9912] |
| 800k | 0.00932 | 0.00659 | 0.099 | 0.01864 | 0.06557 | 0.06553 | 0.9851 [0.9822, 0.9880] |

Slope of log s on log N: -0.488 (95% case bootstrap [-0.500, -0.475]; unweighted norm -0.501). Plateau (first N within 1% of 800k): 400k. Excess e^2 50k-800k 7.33e-04 vs delta^2 6.03e-04.

Q2: R 0.06653, 4 x 50k chunks 0.07074, ratio 1.0632 [1.0568, 1.0700], median per case 1.0667; median deviation from R 0.0248.

Q3 low_x: half alone 0.65793, same cells in R 0.06350, ratio 10.360 [9.833, 10.897], median per case 10.938; median deviation from R 0.6538.

Q3 high_x: half alone 0.76988, same cells in R 0.06935, ratio 11.102 [10.154, 12.115], median per case 12.655; median deviation from R 0.7703.

Q4: R repeated: d max 0.0e+00, median 0.0e+00, max change/RMS 0.0e+00; native repeated: d max 0.0e+00, median 0.0e+00, max change/RMS 0.0e+00.

## r090_dr_isla_seed43 (48 cases)

Validity: V1 rebuilt native block identical on 48/48 cases (prediction max change over RMS 0.0e+00); V2 eager vs scorer max 8.6e-07, median 2.7e-07; targets of shared cells agree to 0.0e+00.

### pressure

| N | spread s | delta = s/sqrt2 | delta / e(200k) | s x sqrt(N/200k) | error, uniform | error, block | error / error(200k) [95%] |
|---|---|---|---|---|---|---|---|
| 50k | 0.02444 | 0.01728 | 0.417 | 0.01222 | 0.04450 | 0.04442 | 1.0742 [1.0632, 1.0857] |
| 100k | 0.01727 | 0.01221 | 0.295 | 0.01221 | 0.04241 | 0.04248 | 1.0238 [1.0191, 1.0288] |
| 200k | 0.01214 | 0.00858 | 0.207 | 0.01214 | 0.04142 | 0.04142 | 1.0000 [1.0000, 1.0000] |
| 400k | 0.00861 | 0.00609 | 0.147 | 0.01218 | 0.04095 | 0.04096 | 0.9886 [0.9853, 0.9917] |
| 800k | 0.00598 | 0.00422 | 0.102 | 0.01195 | 0.04071 | 0.04078 | 0.9827 [0.9791, 0.9864] |

Slope of log s on log N: -0.507 (95% case bootstrap [-0.514, -0.501]; unweighted norm -0.510). Plateau (first N within 1% of 800k): 400k. Excess e^2 50k-800k 3.23e-04 vs delta^2 2.81e-04.

Q2: R 0.04152, 4 x 50k chunks 0.04457, ratio 1.0735 [1.0661, 1.0816], median per case 1.0745; median deviation from R 0.0164.

Q3 low_x: half alone 0.63993, same cells in R 0.03667, ratio 17.449 [16.511, 18.422], median per case 18.375; median deviation from R 0.6340.

Q3 high_x: half alone 0.76966, same cells in R 0.05034, ratio 15.288 [13.918, 16.775], median per case 17.385; median deviation from R 0.7621.

Q4: R repeated: d max 0.0e+00, median 0.0e+00, max change/RMS 0.0e+00; native repeated: d max 0.0e+00, median 0.0e+00, max change/RMS 0.0e+00.

### wss

| N | spread s | delta = s/sqrt2 | delta / e(200k) | s x sqrt(N/200k) | error, uniform | error, block | error / error(200k) [95%] |
|---|---|---|---|---|---|---|---|
| 50k | 0.03545 | 0.02506 | 0.376 | 0.01772 | 0.07099 | 0.07040 | 1.0639 [1.0539, 1.0735] |
| 100k | 0.02469 | 0.01746 | 0.262 | 0.01746 | 0.06813 | 0.06812 | 1.0210 [1.0178, 1.0242] |
| 200k | 0.01798 | 0.01271 | 0.190 | 0.01798 | 0.06673 | 0.06667 | 1.0000 [1.0000, 1.0000] |
| 400k | 0.01293 | 0.00914 | 0.137 | 0.01829 | 0.06607 | 0.06594 | 0.9901 [0.9878, 0.9924] |
| 800k | 0.00913 | 0.00645 | 0.097 | 0.01826 | 0.06579 | 0.06573 | 0.9860 [0.9832, 0.9889] |

Slope of log s on log N: -0.485 (95% case bootstrap [-0.495, -0.474]; unweighted norm -0.509). Plateau (first N within 1% of 800k): 400k. Excess e^2 50k-800k 7.11e-04 vs delta^2 5.87e-04.

Q2: R 0.06678, 4 x 50k chunks 0.07098, ratio 1.0628 [1.0568, 1.0695], median per case 1.0639; median deviation from R 0.0247.

Q3 low_x: half alone 0.66577, same cells in R 0.06423, ratio 10.365 [9.852, 10.889], median per case 10.758; median deviation from R 0.6662.

Q3 high_x: half alone 0.71861, same cells in R 0.06895, ratio 10.422 [9.539, 11.355], median per case 11.978; median deviation from R 0.7150.

Q4: R repeated: d max 0.0e+00, median 0.0e+00, max change/RMS 0.0e+00; native repeated: d max 0.0e+00, median 0.0e+00, max change/RMS 0.0e+00.

