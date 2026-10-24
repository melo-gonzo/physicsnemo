# Reynolds split: per-case test error against distance above the train Re band (seed mean per case)

Train band log10 Re [3.0028, 5.9994]; the 1,385 test cases lie 0.002-2.0 decades above it, none below.

## velocity_l2: mean (median) per-case relative L2 per distance bin, decades above the train max

| bin | cases | ISLA | ISLA-DTT | GeoTransolver |
|---|---|---|---|---|
| [0.0, 0.5) | 411 | 0.1435 (0.1356) | 0.1301 (0.1232) | 0.7187 (0.6854) |
| [0.5, 1.0) | 389 | 0.155 (0.1449) | 0.1411 (0.1336) | 0.7107 (0.6528) |
| [1.0, 1.5) | 312 | 0.1723 (0.1656) | 0.1634 (0.155) | 0.7148 (0.6704) |
| [1.5, 2.0] | 273 | 0.1994 (0.1774) | 0.1886 (0.1663) | 0.6879 (0.6329) |

- ISLA: Spearman(distance, error) +0.207; slope of log10 error per decade +0.090 (×1.23 per decade)
- ISLA-DTT: Spearman(distance, error) +0.245; slope of log10 error per decade +0.108 (×1.28 per decade)
- GeoTransolver: Spearman(distance, error) -0.065; slope of log10 error per decade -0.012 (×0.97 per decade)
- ISLA-DTT ÷ ISLA per case: slope of log10 ratio per decade +0.018; near the band (< 0.5) 0.896, far (≥ 1.5) 0.949 (geometric means)
- GeoTransolver ÷ ISLA per case: slope of log10 ratio per decade -0.101; near the band (< 0.5) 5.686, far (≥ 1.5) 4.031 (geometric means)

## pressure_l2: mean (median) per-case relative L2 per distance bin, decades above the train max

| bin | cases | ISLA | ISLA-DTT | GeoTransolver |
|---|---|---|---|---|
| [0.0, 0.5) | 411 | 0.3765 (0.3994) | 0.3101 (0.3267) | 1.027 (1) |
| [0.5, 1.0) | 389 | 0.4587 (0.4895) | 0.3893 (0.4239) | 1.049 (1) |
| [1.0, 1.5) | 312 | 0.5661 (0.6105) | 0.5559 (0.5422) | 1.035 (1) |
| [1.5, 2.0] | 273 | 0.7159 (0.708) | 0.7411 (0.6521) | 1.032 (1) |

- ISLA: Spearman(distance, error) +0.434; slope of log10 error per decade +0.223 (×1.67 per decade)
- ISLA-DTT: Spearman(distance, error) +0.535; slope of log10 error per decade +0.299 (×1.99 per decade)
- GeoTransolver: Spearman(distance, error) +0.032; slope of log10 error per decade +0.002 (×1.01 per decade)
- ISLA-DTT ÷ ISLA per case: slope of log10 ratio per decade +0.075; near the band (< 0.5) 0.827, far (≥ 1.5) 1.039 (geometric means)
- GeoTransolver ÷ ISLA per case: slope of log10 ratio per decade -0.221; near the band (< 0.5) 3.555, far (≥ 1.5) 1.662 (geometric means)

figure: reynolds/reynolds_extrap.png
