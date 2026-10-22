# AF-EXTRAP post hoc readouts (beside the registered verdicts, which stand)

ISLA family fix-trained (code_cmpaf_fix); its in-distribution reference is the fix-trained r090fix_af_* (like for like).

## reynolds

### 1. Absolute test error, in-distribution error, additive and quadrature excess (seed means)

| arm | test E (p / v / nut) | full_test I | E − I | sqrt(E² − I²) |
|---|---|---|---|---|
| ISLA | 0.0389 / 0.0195 / 0.0457 | 0.0122 / 0.0063 / 0.0156 | 0.0267 / 0.0131 / 0.0301 | 0.0369 / 0.0184 / 0.0430 |
| ISLA-DTT | 0.0335 / 0.0175 / 0.0590 | 0.0127 / 0.0071 / 0.0215 | 0.0208 / 0.0104 / 0.0375 | 0.0310 / 0.0160 / 0.0550 |
| GeoTransolver | 0.0246 / 0.0466 / 0.0467 | 0.0109 / 0.0400 / 0.0171 | 0.0137 / 0.0066 / 0.0297 | 0.0220 / 0.0240 / 0.0435 |
| Transolver | 0.0274 / 0.0381 / 0.0466 | 0.0109 / 0.0297 / 0.0178 | 0.0165 / 0.0084 / 0.0288 | 0.0251 / 0.0239 / 0.0431 |

| ratio ISLA-family ÷ baseline (above 1: the ISLA-family arm's excess is larger) | E − I (p / v / nut) | sqrt(E² − I²) | E (test error) |
|---|---|---|---|
| ISLA ÷ GeoTransolver | 1.95 / 1.98 / 1.02 | 1.68 / 0.77 / 0.99 | 1.58 / 0.42 / 0.98 |
| ISLA ÷ Transolver | 1.62 / 1.56 / 1.05 | 1.47 / 0.77 / 1.00 | 1.42 / 0.51 / 0.98 |
| ISLA-DTT ÷ GeoTransolver | 1.52 / 1.56 / 1.27 | 1.41 / 0.67 / 1.26 | 1.36 / 0.38 / 1.26 |
| ISLA-DTT ÷ Transolver | 1.27 / 1.23 / 1.30 | 1.24 / 0.67 / 1.28 | 1.22 / 0.46 / 1.27 |

### 2. Per-case error against distance outside the band: least-squares slope (error per decade of Re) and Spearman

| side (cases) | ISLA slope p / v / nut (ρ_s) | ISLA-DTT slope p / v / nut (ρ_s) | GeoTransolver slope p / v / nut (ρ_s) | Transolver slope p / v / nut (ρ_s) |
|---|---|---|---|---|
| below (254) | 0.615 / 0.317 / 0.763 (0.84 / 0.90 / 0.88) | 0.417 / 0.259 / 0.894 (0.79 / 0.85 / 0.79) | 0.161 / 0.020 / 0.444 (0.42 / 0.00 / 0.82) | 0.316 / 0.062 / 0.453 (0.62 / 0.28 / 0.83) |
| above (242) | 0.209 / 0.143 / 0.301 (0.36 / 0.68 / 0.49) | 0.091 / 0.050 / 0.043 (0.26 / 0.47 / 0.24) | 0.114 / 0.065 / 0.381 (0.33 / 0.16 / 0.43) | 0.042 / 0.062 / 0.503 (0.19 / 0.21 / 0.66) |

### 3. Matched-case penalty: the 103 full_test cases outside 3e6-5e6 (all in reynolds_test), reynolds-trained ÷ full-trained on the same cases

| arm | E on these cases (p / v / nut) | I (full-trained) on the same cases | g_matched |
|---|---|---|---|
| ISLA | 0.0320 / 0.0171 / 0.0408 | 0.0124 / 0.0065 / 0.0149 | 2.59 / 2.63 / 2.75 |
| ISLA-DTT | 0.0272 / 0.0157 / 0.0559 | 0.0132 / 0.0074 / 0.0214 | 2.05 / 2.12 / 2.61 |
| GeoTransolver | 0.0191 / 0.0447 / 0.0379 | 0.0108 / 0.0382 / 0.0160 | 1.78 / 1.17 / 2.37 |
| Transolver | 0.0203 / 0.0361 / 0.0393 | 0.0112 / 0.0288 / 0.0167 | 1.81 / 1.26 / 2.35 |
ρ_matched = g_B ÷ g_ISLA-family (p / v / nut): GeoTransolver ÷ ISLA 0.69 / 0.44 / 0.86; Transolver ÷ ISLA 0.70 / 0.48 / 0.86; GeoTransolver ÷ ISLA-DTT 0.87 / 0.55 / 0.91; Transolver ÷ ISLA-DTT 0.88 / 0.59 / 0.90

### 4. Input-space reach (ISLA's in-model z of log10 Re; the baselines' standardized speed u)

- ISLA z: train -2.05 to 1.80; test -4.85 to 3.04.
- Baselines u: train -1.80 to 1.75; test -3.57 to 3.54.
- Reach beyond the training edge, in each model's own input units (test extreme minus train extreme): ISLA below -2.81, above 1.24; baselines below -1.78, above 1.79.

| side | ISLA: Spearman(error, own reach) p / v / nut | ISLA-DTT: Spearman(error, own reach) p / v / nut | GeoTransolver: Spearman(error, own reach) p / v / nut | Transolver: Spearman(error, own reach) p / v / nut |
|---|---|---|---|---|
| below | 0.86 / 0.92 / 0.88 | 0.81 / 0.88 / 0.79 | 0.42 / 0.00 / 0.82 | 0.62 / 0.28 / 0.83 |
| above | 0.34 / 0.67 / 0.45 | 0.24 / 0.45 / 0.21 | 0.33 / 0.16 / 0.43 | 0.19 / 0.21 / 0.66 |

Below the band, by quartile of distance (same cases for every arm): each model's own input at the bin's edges, and the error

| bin | ISLA z | baselines u | ISLA p / v / nut | GeoTransolver p / v / nut | Transolver p / v / nut |
|---|---|---|---|---|---|
| 1 (n=63) | -1.75 to -2.51 | -1.81 to -2.32 | 0.0179 / 0.0090 / 0.0198 | 0.0188 / 0.0416 / 0.0234 | 0.0161 / 0.0305 / 0.0218 |
| 2 (n=64) | -2.51 to -3.22 | -2.17 to -2.76 | 0.0300 / 0.0147 / 0.0380 | 0.0251 / 0.0447 / 0.0558 | 0.0226 / 0.0332 / 0.0429 |
| 3 (n=63) | -3.22 to -4.01 | -2.67 to -3.25 | 0.0694 / 0.0322 / 0.0788 | 0.0324 / 0.0451 / 0.0708 | 0.0591 / 0.0392 / 0.0674 |
| 4 (n=64) | -4.03 to -4.85 | -3.10 to -3.57 | 0.0999 / 0.0514 / 0.1176 | 0.0446 / 0.0469 / 0.0869 | 0.0538 / 0.0398 / 0.0838 |

## aoa

### 1. Absolute test error, in-distribution error, additive and quadrature excess (seed means)

| arm | test E (p / v / nut) | full_test I | E − I | sqrt(E² − I²) |
|---|---|---|---|---|
| ISLA | 0.0485 / 0.0163 / 0.0538 | 0.0122 / 0.0063 / 0.0156 | 0.0363 / 0.0099 / 0.0382 | 0.0469 / 0.0150 / 0.0515 |
| ISLA-DTT | 0.0528 / 0.0168 / 0.0745 | 0.0127 / 0.0071 / 0.0215 | 0.0401 / 0.0097 / 0.0530 | 0.0512 / 0.0153 / 0.0714 |
| GeoTransolver | 0.1809 / 0.0880 / 0.2313 | 0.0109 / 0.0400 / 0.0171 | 0.1700 / 0.0480 / 0.2142 | 0.1806 / 0.0784 / 0.2307 |
| Transolver | 0.0856 / 0.0493 / 0.0938 | 0.0109 / 0.0297 / 0.0178 | 0.0747 / 0.0196 / 0.0760 | 0.0849 / 0.0394 / 0.0921 |

| ratio ISLA-family ÷ baseline (above 1: the ISLA-family arm's excess is larger) | E − I (p / v / nut) | sqrt(E² − I²) | E (test error) |
|---|---|---|---|
| ISLA ÷ GeoTransolver | 0.21 / 0.21 / 0.18 | 0.26 / 0.19 / 0.22 | 0.27 / 0.18 / 0.23 |
| ISLA ÷ Transolver | 0.49 / 0.51 / 0.50 | 0.55 / 0.38 / 0.56 | 0.57 / 0.33 / 0.57 |
| ISLA-DTT ÷ GeoTransolver | 0.24 / 0.20 / 0.25 | 0.28 / 0.19 / 0.31 | 0.29 / 0.19 / 0.32 |
| ISLA-DTT ÷ Transolver | 0.54 / 0.49 / 0.70 | 0.60 / 0.39 / 0.78 | 0.62 / 0.34 / 0.79 |

### 2. Per-case error against distance outside the band: least-squares slope (error per degree) and Spearman

| side (cases) | ISLA slope p / v / nut (ρ_s) | ISLA-DTT slope p / v / nut (ρ_s) | GeoTransolver slope p / v / nut (ρ_s) | Transolver slope p / v / nut (ρ_s) |
|---|---|---|---|---|
| below (111) | 0.067 / 0.016 / 0.031 (0.19 / 0.20 / 0.30) | 0.057 / 0.014 / 0.029 (0.02 / 0.07 / 0.25) | 0.236 / 0.063 / 0.205 (0.89 / 0.86 / 0.89) | 0.126 / 0.031 / 0.064 (0.85 / 0.82 / 0.82) |
| above (85) | 0.004 / 0.005 / 0.054 (0.49 / 0.58 / 0.66) | 0.003 / 0.005 / 0.065 (0.52 / 0.56 / 0.61) | 0.200 / 0.068 / 0.359 (0.97 / 0.84 / 0.94) | 0.023 / 0.010 / 0.110 (0.86 / 0.68 / 0.88) |

