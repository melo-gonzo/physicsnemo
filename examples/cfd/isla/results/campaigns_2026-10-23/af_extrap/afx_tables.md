*ISLA family fix-trained (code_cmpaf_fix); its in-distribution reference is the fix-trained r090fix_af_* (like for like).*

## reynolds (496 test cases; σ pooled per field: pressure 0.0689, velocity 0.0822, eddy viscosity 0.0710)

| arm | pressure test (s42, s43) | velocity test (s42, s43) | eddy viscosity test (s42, s43) | penalty g = test ÷ full_test (p / v / nut) |
|---|---|---|---|---|
| ISLA 0.9.0 | 0.03890 (0.04200, 0.03580) | 0.01947 (0.02245, 0.01650) | 0.04569 (0.04238, 0.04900) | 3.18 / 3.08 / 2.94 |
| ISLA-DTT 0.9.0 | 0.03349 (0.03281, 0.03417) | 0.01752 (0.01823, 0.01682) | 0.05904 (0.06252, 0.05556) | 2.65 / 2.46 / 2.75 |
| GeoTransolver 256 | 0.02459 (0.02670, 0.02248) | 0.04662 (0.04632, 0.04692) | 0.04674 (0.04358, 0.04991) | 2.26 / 1.17 / 2.74 |
| Transolver 512 x 16 | 0.02736 (0.02698, 0.02773) | 0.03810 (0.03670, 0.03950) | 0.04664 (0.04583, 0.04744) | 2.51 / 1.28 / 2.62 |

| cell | R1 ratio B ÷ ISLA [band] verdict (book rule) | R2 ρ = g_B ÷ g_ISLA [band] (pairs) verdict |
|---|---|---|
| GeoTransolver 256 ÷ ISLA, pressure | 0.632 [±0.138] ISLA trails (ISLA trails) | 0.709 [×1.215] (0.746, 0.671) ISLA extrapolates worse |
| GeoTransolver 256 ÷ ISLA, velocity | 2.394 [±0.164] ISLA leads (ISLA leads) | 0.379 [×1.262] (0.334, 0.440) ISLA extrapolates worse |
| GeoTransolver 256 ÷ ISLA, eddy viscosity | 1.023 [±0.142] level (level) | 0.932 [×1.222] (0.922, 0.943) no difference |
| Transolver 512 x 16 ÷ ISLA, pressure | 0.703 [±0.138] ISLA trails (ISLA trails) | 0.789 [×1.215] (0.781, 0.798) ISLA extrapolates worse |
| Transolver 512 x 16 ÷ ISLA, velocity | 1.956 [±0.164] ISLA leads (ISLA leads) | 0.417 [×1.262] (0.353, 0.504) ISLA extrapolates worse |
| Transolver 512 x 16 ÷ ISLA, eddy viscosity | 1.021 [±0.142] level (level) | 0.892 [×1.222] (0.963, 0.830) no difference |
| ISLA-DTT 0.9.0 ÷ ISLA, pressure | 0.861 [±0.138] ISLA trails (ISLA trails) | 0.832 [×1.215] (0.796, 0.871) no difference |
| ISLA-DTT 0.9.0 ÷ ISLA, velocity | 0.900 [±0.164] level (level) | 0.799 [×1.262] (0.735, 0.887) no difference |
| ISLA-DTT 0.9.0 ÷ ISLA, eddy viscosity | 1.292 [±0.142] ISLA leads (ISLA leads) | 0.936 [×1.222] (1.167, 0.756) no difference |

Beside the rules: ISLA-DTT in ISLA's place against each baseline (R1 / R2 as registered):

| cell | R1 ratio B ÷ ISLA-DTT [band] verdict | R2 ρ = g_B ÷ g_DTT [band] (pairs) verdict |
|---|---|---|
| GeoTransolver 256 ÷ ISLA-DTT, pressure | 0.734 [±0.138] ISLA-DTT trails | 0.852 [×1.215] (0.937, 0.770) no difference |
| GeoTransolver 256 ÷ ISLA-DTT, velocity | 2.661 [±0.164] ISLA-DTT leads | 0.474 [×1.262] (0.454, 0.496) ISLA-DTT extrapolates worse |
| GeoTransolver 256 ÷ ISLA-DTT, eddy viscosity | 0.792 [±0.142] ISLA-DTT trails | 0.996 [×1.222] (0.790, 1.249) no difference |
| Transolver 512 x 16 ÷ ISLA-DTT, pressure | 0.817 [±0.138] ISLA-DTT trails | 0.948 [×1.215] (0.981, 0.917) no difference |
| Transolver 512 x 16 ÷ ISLA-DTT, velocity | 2.174 [±0.164] ISLA-DTT leads | 0.522 [×1.262] (0.480, 0.568) ISLA-DTT extrapolates worse |
| Transolver 512 x 16 ÷ ISLA-DTT, eddy viscosity | 0.790 [±0.142] ISLA-DTT trails | 0.953 [×1.222] (0.826, 1.098) no difference |

S-RE (PREREG §11): 504-case in-distribution error I_504 (full_test, seed means p / v / nut) and I_800 ÷ I_504: ISLA 0.9.0 0.01681 / 0.00820 / 0.02032 (0.727 / 0.771 / 0.766); GeoTransolver 256 0.01484 / 0.04459 / 0.02062 (0.735 / 0.897 / 0.828); Transolver 512 x 16 0.01317 / 0.03257 / 0.02031 (0.828 / 0.911 / 0.877)

| cell | registered ρ (I_800) verdict | ρ_504 (pairs) verdict |
|---|---|---|
| GeoTransolver 256 ÷ ISLA, pressure | 0.709 ISLA extrapolates worse | 0.716 (0.821, 0.624) ISLA extrapolates worse |
| GeoTransolver 256 ÷ ISLA, velocity | 0.379 ISLA extrapolates worse | 0.440 (0.391, 0.507) ISLA extrapolates worse |
| GeoTransolver 256 ÷ ISLA, eddy viscosity | 0.932 no difference | 1.008 (1.017, 1.000) no difference |
| Transolver 512 x 16 ÷ ISLA, pressure | 0.789 ISLA extrapolates worse | 0.898 (0.886, 0.912) no difference |
| Transolver 512 x 16 ÷ ISLA, velocity | 0.417 ISLA extrapolates worse | 0.493 (0.411, 0.604) ISLA extrapolates worse |
| Transolver 512 x 16 ÷ ISLA, eddy viscosity | 0.892 no difference | 1.022 (1.088, 0.964) no difference |

Task summary (R2 over the 6 baseline cells): **ISLA extrapolates worse than the baselines** {'better': 0, 'worse': 4, 'no difference': 2}
S-RE trigger: {'fires': True, 'architectures': ['gt', 'isla', 't'], 'cells': [('gt', 'pressure_l2', 'ISLA extrapolates worse'), ('gt', 'velocity_l2', 'ISLA extrapolates worse'), ('t', 'pressure_l2', 'ISLA extrapolates worse'), ('t', 'velocity_l2', 'ISLA extrapolates worse')]}
nut space factor over the test cases: 0.9997-1.0053 (mean 1.0007); nut in the 7.9e-4 space: ISLA 0.9.0 0.04573, ISLA-DTT 0.9.0 0.05909, GeoTransolver 256 0.04679, Transolver 512 x 16 0.04668
posed ÷ native (s42 / s43, p, v, nut): ISLA 0.9.0 1.0003/1.0003, 1.0002/1.0003, 1.0000/1.0000; ISLA-DTT 0.9.0 1.0003/1.0004, 1.0001/1.0003, 1.0000/1.0000; GeoTransolver 256 74.2533/90.7845, 25.5730/26.5419, 34.7241/27.8797; Transolver 512 x 16 47.6391/48.2744, 33.3837/30.1255, 36.9799/30.0890

Per-case error against distance outside the training range, below the band (254 cases; quartile bins; seed-mean per case):

| bin (distance) | ISLA 0.9.0 p / v / nut | ISLA-DTT 0.9.0 p / v / nut | GeoTransolver 256 p / v / nut | Transolver 512 x 16 p / v / nut |
|---|---|---|---|---|
| 0.000994-0.0351 (n=63) | 0.0186 / 0.0091 / 0.0196 | 0.0196 / 0.0102 / 0.0387 | 0.0193 / 0.0423 / 0.0230 | 0.0166 / 0.0310 / 0.0213 |
| 0.0352-0.0814 (n=64) | 0.0293 / 0.0146 / 0.0381 | 0.0226 / 0.0125 / 0.0406 | 0.0248 / 0.0441 / 0.0558 | 0.0224 / 0.0329 / 0.0432 |
| 0.0814-0.127 (n=63) | 0.0770 / 0.0343 / 0.0783 | 0.0709 / 0.0299 / 0.0946 | 0.0403 / 0.0473 / 0.0744 | 0.0626 / 0.0412 / 0.0720 |
| 0.128-0.175 (n=64) | 0.0924 / 0.0493 / 0.1183 | 0.0648 / 0.0422 / 0.1489 | 0.0367 / 0.0445 / 0.0836 | 0.0502 / 0.0377 / 0.0795 |
Spearman (distance, per-case error): ISLA 0.9.0 0.84 / 0.90 / 0.88; ISLA-DTT 0.9.0 0.79 / 0.85 / 0.79; GeoTransolver 256 0.42 / 0.00 / 0.82; Transolver 512 x 16 0.62 / 0.28 / 0.83

Per-case error against distance outside the training range, above the band (242 cases; quartile bins; seed-mean per case):

| bin (distance) | ISLA 0.9.0 p / v / nut | ISLA-DTT 0.9.0 p / v / nut | GeoTransolver 256 p / v / nut | Transolver 512 x 16 p / v / nut |
|---|---|---|---|---|
| 0.000378-0.0225 (n=60) | 0.0198 / 0.0088 / 0.0202 | 0.0235 / 0.0108 / 0.0408 | 0.0179 / 0.0473 / 0.0254 | 0.0175 / 0.0394 / 0.0277 |
| 0.0227-0.0421 (n=61) | 0.0178 / 0.0092 / 0.0223 | 0.0161 / 0.0088 / 0.0274 | 0.0143 / 0.0485 / 0.0277 | 0.0129 / 0.0399 / 0.0301 |
| 0.0425-0.0589 (n=60) | 0.0217 / 0.0120 / 0.0281 | 0.0203 / 0.0105 / 0.0319 | 0.0181 / 0.0498 / 0.0338 | 0.0146 / 0.0411 / 0.0418 |
| 0.0589-0.0791 (n=61) | 0.0313 / 0.0168 / 0.0365 | 0.0282 / 0.0137 / 0.0447 | 0.0242 / 0.0496 / 0.0472 | 0.0200 / 0.0421 / 0.0557 |
Spearman (distance, per-case error): ISLA 0.9.0 0.36 / 0.68 / 0.49; ISLA-DTT 0.9.0 0.26 / 0.47 / 0.24; GeoTransolver 256 0.33 / 0.16 / 0.43; Transolver 512 x 16 0.19 / 0.21 / 0.66

*ISLA family fix-trained (code_cmpaf_fix); its in-distribution reference is the fix-trained r090fix_af_* (like for like).*

## aoa (196 test cases; σ pooled per field: pressure 0.0990, velocity 0.0669, eddy viscosity 0.0986)

| arm | pressure test (s42, s43) | velocity test (s42, s43) | eddy viscosity test (s42, s43) | penalty g = test ÷ full_test (p / v / nut) |
|---|---|---|---|---|
| ISLA 0.9.0 | 0.04849 (0.04999, 0.04698) | 0.01627 (0.01663, 0.01591) | 0.05378 (0.05607, 0.05148) | 3.97 / 2.57 / 3.46 |
| ISLA-DTT 0.9.0 | 0.05275 (0.04864, 0.05687) | 0.01684 (0.01640, 0.01728) | 0.07454 (0.07085, 0.07822) | 4.17 / 2.36 / 3.47 |
| GeoTransolver 256 | 0.18093 (0.15132, 0.21054) | 0.08796 (0.07675, 0.09918) | 0.23132 (0.19833, 0.26431) | 16.60 / 2.20 / 13.54 |
| Transolver 512 x 16 | 0.08557 (0.08699, 0.08415) | 0.04932 (0.04921, 0.04943) | 0.09379 (0.10191, 0.08567) | 7.85 / 1.66 / 5.27 |

| cell | R1 ratio B ÷ ISLA [band] verdict (book rule) | R2 ρ = g_B ÷ g_ISLA [band] (pairs) verdict |
|---|---|---|
| GeoTransolver 256 ÷ ISLA, pressure | 3.731 [±0.198] ISLA leads (ISLA leads) | 4.185 [×1.323] (3.551, 4.786) ISLA extrapolates better |
| GeoTransolver 256 ÷ ISLA, velocity | 5.407 [±0.134] ISLA leads (ISLA leads) | 0.856 [×1.208] (0.747, 0.965) no difference |
| GeoTransolver 256 ÷ ISLA, eddy viscosity | 4.301 [±0.197] ISLA leads (ISLA leads) | 3.920 [×1.322] (3.173, 4.756) ISLA extrapolates better |
| Transolver 512 x 16 ÷ ISLA, pressure | 1.765 [±0.198] ISLA leads (ISLA leads) | 1.979 [×1.323] (2.116, 1.845) ISLA extrapolates better |
| Transolver 512 x 16 ÷ ISLA, velocity | 3.031 [±0.134] ISLA leads (ISLA leads) | 0.646 [×1.208] (0.639, 0.654) ISLA extrapolates worse |
| Transolver 512 x 16 ÷ ISLA, eddy viscosity | 1.744 [±0.197] ISLA leads (ISLA leads) | 1.524 [×1.322] (1.619, 1.426) ISLA extrapolates better |
| ISLA-DTT 0.9.0 ÷ ISLA, pressure | 1.088 [±0.198] level (level) | 1.051 [×1.323] (0.992, 1.104) no difference |
| ISLA-DTT 0.9.0 ÷ ISLA, velocity | 1.035 [±0.134] level (level) | 0.919 [×1.208] (0.893, 0.946) no difference |
| ISLA-DTT 0.9.0 ÷ ISLA, eddy viscosity | 1.386 [±0.197] ISLA leads (ISLA leads) | 1.004 [×1.322] (1.000, 1.013) no difference |

Beside the rules: ISLA-DTT in ISLA's place against each baseline (R1 / R2 as registered):

| cell | R1 ratio B ÷ ISLA-DTT [band] verdict | R2 ρ = g_B ÷ g_DTT [band] (pairs) verdict |
|---|---|---|
| GeoTransolver 256 ÷ ISLA-DTT, pressure | 3.430 [±0.198] ISLA-DTT leads | 3.982 [×1.323] (3.581, 4.336) ISLA-DTT extrapolates better |
| GeoTransolver 256 ÷ ISLA-DTT, velocity | 5.223 [±0.134] ISLA-DTT leads | 0.931 [×1.208] (0.837, 1.020) no difference |
| GeoTransolver 256 ÷ ISLA-DTT, eddy viscosity | 3.103 [±0.197] ISLA-DTT leads | 3.905 [×1.322] (3.173, 4.697) ISLA-DTT extrapolates better |
| Transolver 512 x 16 ÷ ISLA-DTT, pressure | 1.622 [±0.198] ISLA-DTT leads | 1.883 [×1.323] (2.134, 1.672) ISLA-DTT extrapolates better |
| Transolver 512 x 16 ÷ ISLA-DTT, velocity | 2.928 [±0.134] ISLA-DTT leads | 0.703 [×1.208] (0.716, 0.691) ISLA-DTT extrapolates worse |
| Transolver 512 x 16 ÷ ISLA-DTT, eddy viscosity | 1.258 [±0.197] ISLA-DTT leads | 1.519 [×1.322] (1.620, 1.409) ISLA-DTT extrapolates better |

Task summary (R2 over the 6 baseline cells): **mixed / no difference** {'better': 4, 'worse': 1, 'no difference': 1}
nut space factor over the test cases: 0.9573-1.0033 (mean 0.9961); nut in the 7.9e-4 space: ISLA 0.9.0 0.05376, ISLA-DTT 0.9.0 0.07441, GeoTransolver 256 0.23092, Transolver 512 x 16 0.09369
posed ÷ native (s42 / s43, p, v, nut): ISLA 0.9.0 1.0004/1.0005, 1.0004/1.0005, 1.0000/1.0000; ISLA-DTT 0.9.0 1.0002/1.0002, 1.0004/1.0004, 1.0000/0.9992; GeoTransolver 256 9.6743/7.7349, 14.9562/11.5789, 5.6726/4.5282; Transolver 512 x 16 14.1644/14.4294, 23.1403/23.1679, 11.8857/13.3336

Per-case error against distance outside the training range, below the band (111 cases; quartile bins; seed-mean per case):

| bin (distance) | ISLA 0.9.0 p / v / nut | ISLA-DTT 0.9.0 p / v / nut | GeoTransolver 256 p / v / nut | Transolver 512 x 16 p / v / nut |
|---|---|---|---|---|
| 0.011-0.614 (n=27) | 0.0248 / 0.0083 / 0.0165 | 0.0469 / 0.0109 / 0.0392 | 0.0295 / 0.0356 / 0.0196 | 0.0302 / 0.0291 / 0.0178 |
| 0.627-1.12 (n=28) | 0.0193 / 0.0072 / 0.0162 | 0.0260 / 0.0077 / 0.0334 | 0.0735 / 0.0401 / 0.0384 | 0.0554 / 0.0325 / 0.0283 |
| 1.13-1.86 (n=28) | 0.0291 / 0.0102 / 0.0299 | 0.0369 / 0.0106 / 0.0382 | 0.1692 / 0.0627 / 0.1321 | 0.0998 / 0.0442 / 0.0425 |
| 1.87-2.44 (n=28) | 0.2189 / 0.0533 / 0.1005 | 0.2175 / 0.0531 / 0.1275 | 0.5172 / 0.1683 / 0.4305 | 0.3217 / 0.1031 / 0.1659 |
Spearman (distance, per-case error): ISLA 0.9.0 0.19 / 0.20 / 0.30; ISLA-DTT 0.9.0 0.02 / 0.07 / 0.25; GeoTransolver 256 0.89 / 0.86 / 0.89; Transolver 512 x 16 0.85 / 0.82 / 0.82

Per-case error against distance outside the training range, above the band (85 cases; quartile bins; seed-mean per case):

| bin (distance) | ISLA 0.9.0 p / v / nut | ISLA-DTT 0.9.0 p / v / nut | GeoTransolver 256 p / v / nut | Transolver 512 x 16 p / v / nut |
|---|---|---|---|---|
| 0.006-0.626 (n=21) | 0.0124 / 0.0080 / 0.0288 | 0.0124 / 0.0080 / 0.0412 | 0.0141 / 0.0570 / 0.0590 | 0.0111 / 0.0376 / 0.0454 |
| 0.722-1.24 (n=21) | 0.0107 / 0.0084 / 0.0430 | 0.0094 / 0.0085 / 0.0642 | 0.0454 / 0.0614 / 0.1366 | 0.0187 / 0.0396 / 0.0790 |
| 1.27-1.79 (n=21) | 0.0243 / 0.0148 / 0.0878 | 0.0223 / 0.0166 / 0.1160 | 0.1887 / 0.1052 / 0.4228 | 0.0429 / 0.0499 / 0.1759 |
| 1.81-2.43 (n=22) | 0.0162 / 0.0150 / 0.1198 | 0.0135 / 0.0142 / 0.1511 | 0.3716 / 0.1815 / 0.6816 | 0.0488 / 0.0535 / 0.2259 |
Spearman (distance, per-case error): ISLA 0.9.0 0.49 / 0.58 / 0.66; ISLA-DTT 0.9.0 0.52 / 0.56 / 0.61; GeoTransolver 256 0.97 / 0.84 / 0.94; Transolver 512 x 16 0.86 / 0.68 / 0.88

