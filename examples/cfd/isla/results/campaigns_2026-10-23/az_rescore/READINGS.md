## E1-AZ (HEADNORM ÷ CONTROL, family_non_box test, 1,074 cases)

Registered rule, branch 1 (PREREG_AZ §4): reject if a headline ratio is ≥ 1.10 with the seed-mean ratio and both seed-matched ratios ≥ 1.10. G14 = (0.7814 · r_v · r_p)^(1/14), reject if ≥ 1.05. Ratios above 1 mean HEADNORM is worse.

| metric | statistic | velocity: seed-mean ratio (s42, s43) | pressure: seed-mean ratio (s42, s43) | branch 1 fires? | G14 |
|---|---|---|---|---|---|
| registered relL2 (200k, float32) | all mean (the rule’s form) | 1.047 (1.022, 1.081) | 1.287 (1.339, 1.178) | yes | 1.0037 |
| registered relL2 (200k, float32) | all median | 1.031 (1.009, 1.058) | 1.488 (1.762, 1.072) | no | 1.0131 |
| registered relL2 (200k, float32) | mean_stable mean | 1.080 (1.079, 1.082) | 1.432 (1.689, 1.023) | no | 1.0136 |
| registered relL2 (200k, float32) | mean_stable median | 1.197 (1.301, 0.988) | 1.534 (1.993, 0.937) | no | 1.0261 |
| full-mesh relL2 (float64) | all mean (the rule’s form) | 1.047 (1.022, 1.081) | 1.289 (1.342, 1.177) | yes | 1.0038 |
| full-mesh relL2 (float64) | all median | 1.030 (1.010, 1.053) | 1.486 (1.763, 1.069) | no | 1.0128 |
| full-mesh relL2 (float64) | mean_stable mean | 1.080 (1.079, 1.081) | 1.435 (1.695, 1.022) | no | 1.0138 |
| full-mesh relL2 (float64) | mean_stable median | 1.197 (1.301, 0.988) | 1.537 (2.001, 0.938) | no | 1.0263 |
| spec eps_U / eps_p | all mean (the rule’s form) | 1.075 (1.080, 1.066) | 1.064 (1.002, 1.157) | no | 0.9920 |
| spec eps_U / eps_p | all median | 1.192 (1.276, 1.046) | 1.298 (1.412, 1.103) | yes | 1.0137 |
| spec eps_U / eps_p | mean_stable mean | 1.131 (1.180, 1.053) | 1.073 (1.065, 1.083) | no | 0.9962 |
| spec eps_U / eps_p | mean_stable median | 1.304 (1.491, 0.812) | 1.634 (2.393, 0.801) | no | 1.0370 |
| spec eps^N_U / eps^N_p | all mean (the rule’s form) | 1.066 (1.127, 0.942) | 1.007 (0.971, 1.065) | no | 0.9875 |
| spec eps^N_U / eps^N_p | all median | 1.225 (1.362, 0.939) | 1.255 (1.608, 0.853) | no | 1.0132 |
| spec eps^N_U / eps^N_p | mean_stable mean | 1.168 (1.267, 0.965) | 1.000 (1.010, 0.988) | no | 0.9935 |
| spec eps^N_U / eps^N_p | mean_stable median | 1.247 (1.397, 0.921) | 1.101 (1.611, 0.659) | no | 1.0051 |

## LRE-AZ (B = fixed Reynolds scale ÷ A = per-split constants, reynolds test, 1,385 cases)

Registered R1: D = ln(seed-mean CM(B) / seed-mean CM(A)), SE = sqrt((s_A² + s_B²)/2), s the two-seed sample std of ln CM; helps if D < ln 0.90 and |D| > 2 SE, harms if D > ln 1.10 and |D| > 2 SE. Consequence: harms on velocity or pressure → not adopted without Peter.

| metric | field | B ÷ A = exp(D) | D | 2 SE | per seed (42, 43) | label |
|---|---|---|---|---|---|---|
| registered relL2 | velocity | 1.126 | +0.118 | 0.046 | 1.139, 1.113 | harms |
| registered relL2 | pressure | 1.067 | +0.065 | 0.100 | 1.020, 1.114 | no change |
| full-mesh relL2 | velocity | 1.125 | +0.118 | 0.046 | 1.138, 1.113 | harms |
| full-mesh relL2 | pressure | 1.067 | +0.065 | 0.100 | 1.021, 1.114 | no change |
| spec eps (case means) | velocity | 1.218 | +0.197 | 0.154 | 1.249, 1.190 | harms |
| spec eps (case means) | pressure | 1.433 | +0.360 | 0.404 | 1.419, 1.444 | no change |
| spec eps (case medians, beside) | velocity | 1.083 | +0.080 | 0.013 | 1.073, 1.093 | no change |
| spec eps (case medians, beside) | pressure | 1.116 | +0.110 | 0.256 | 0.970, 1.264 | no change |
| spec eps^N (case means) | velocity | 1.155 | +0.144 | 0.111 | 1.155, 1.156 | harms |
| spec eps^N (case means) | pressure | 1.488 | +0.397 | 0.425 | 1.484, 1.490 | no change |

## §12(f)'s bands on the test sides under the spec metric (descriptive, beside the registered read)

R = ISLA-DTT ÷ ISLA, seed means of case means, per held-out split (full reported, not in the rule). Bands as registered: ISLA-DTT clearly if R ≤ 1.05 on both fields on every held-out split; ISLA clearly if the arithmetic mean of R over the six is ≥ 1.10 on velocity or pressure; otherwise a close call → ISLA-DTT. This is a test-side reading like the registered one; under Peter's rulings (10-23, 10-24) it decides nothing.


**registered relL2**

| split | R velocity (s42, s43) | R pressure (s42, s43) |
|---|---|---|
| bank_zero_to_cad | 0.952 (0.955, 0.949) | 0.956 (0.949, 0.963) |
| blobs | 0.893 (0.896, 0.890) | 0.791 (0.774, 0.808) |
| nonbox | 0.921 (0.869, 0.993) | 1.274 (1.207, 1.414) |
| profile_crossflow | 0.943 (0.826, 1.064) | 0.824 (0.443, 1.442) |
| profile_dual_inlet | 0.929 (0.939, 0.920) | 1.034 (1.175, 0.920) |
| reynolds | 0.927 (0.986, 0.870) | 0.928 (0.997, 0.860) |
| full (reported) | 0.935 (0.924, 0.948) | 0.843 (0.831, 0.856) |

Arithmetic mean of R over the six: velocity 0.928, pressure 0.968 (geometric 0.927, 0.956); band: **close call → ISLA-DTT**.

**spec eps_U / eps_p**

| split | R velocity (s42, s43) | R pressure (s42, s43) |
|---|---|---|
| bank_zero_to_cad | 0.989 (1.000, 0.978) | 0.967 (0.984, 0.950) |
| blobs | 0.912 (0.936, 0.890) | 0.940 (0.880, 1.003) |
| nonbox | 0.966 (0.852, 1.147) | 0.981 (0.882, 1.131) |
| profile_crossflow | 0.971 (0.897, 1.042) | 0.945 (0.575, 1.518) |
| profile_dual_inlet | 0.922 (0.918, 0.925) | 1.025 (1.214, 0.879) |
| reynolds | 0.990 (1.158, 0.842) | 1.082 (1.516, 0.753) |
| full (reported) | 0.978 (0.959, 0.998) | 0.959 (0.957, 0.960) |

Arithmetic mean of R over the six: velocity 0.958, pressure 0.990 (geometric 0.958, 0.989); band: **close call → ISLA-DTT**.

**spec eps^N**

| split | R velocity (s42, s43) | R pressure (s42, s43) |
|---|---|---|
| bank_zero_to_cad | 0.974 (0.987, 0.960) | 0.960 (0.968, 0.951) |
| blobs | 0.904 (0.913, 0.896) | 0.884 (0.827, 0.945) |
| nonbox | 0.912 (0.784, 1.171) | 0.959 (0.908, 1.043) |
| profile_crossflow | 0.979 (0.896, 1.059) | 0.981 (0.596, 1.627) |
| profile_dual_inlet | 0.911 (0.878, 0.944) | 1.032 (1.238, 0.875) |
| reynolds | 0.985 (1.101, 0.877) | 1.121 (1.637, 0.737) |
| full (reported) | 0.957 (0.936, 0.978) | 0.924 (0.918, 0.930) |

Arithmetic mean of R over the six: velocity 0.944, pressure 0.989 (geometric 0.943, 0.987); band: **close call → ISLA-DTT**.
