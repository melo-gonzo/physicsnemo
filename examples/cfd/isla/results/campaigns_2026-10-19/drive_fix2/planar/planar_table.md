
### native

| arm | velocity_l2 (3-D) s42 / s43 / mean | velocity_xy_l2 s42 / s43 / mean | xy / 3-D | z share of squared error (max seed) | check (max per-case rel. diff vs shared rows) |
|---|---|---|---|---|---|
| GeoTransolver afu_ud_gt (unit drive) | 0.03973 / 0.04022 / **0.03998** | 0.03973 / 0.04022 / **0.03998** | 0.9999978 | 6.3e-06 | 1.7e-07 |
| Transolver afu_ud_t (unit drive) | 0.02976 / 0.02958 / **0.02967** | 0.02976 / 0.02958 / **0.02967** | 0.9999999 | 3.4e-07 | 2.2e-07 |
| GeoTransolver afu_r_gt (raw drive) | 0.03801 / 0.03499 / **0.03650** | 0.03801 / 0.03499 / **0.03650** | 0.9999959 | 1.2e-05 | 1.6e-07 |
| Transolver afu_r_t (raw drive) | 0.06517 / 0.06871 / **0.06694** | 0.06517 / 0.06871 / **0.06694** | 1.0000000 | 8.2e-08 | 2.4e-07 |

### posed

| arm | velocity_l2 (3-D) s42 / s43 / mean | velocity_xy_l2 s42 / s43 / mean | xy / 3-D | z share of squared error (max seed) | check (max per-case rel. diff vs shared rows) |
|---|---|---|---|---|---|
| GeoTransolver afu_ud_gt (unit drive) | 1.18477 / 1.14921 / **1.16699** | 1.18476 / 1.14921 / **1.16699** | 0.9999981 | 6.5e-06 | 2.0e-07 |
| Transolver afu_ud_t (unit drive) | 1.25506 / 1.20603 / **1.23054** | 1.25506 / 1.20603 / **1.23054** | 1.0000000 | 3.5e-09 | 2.0e-07 |
| GeoTransolver afu_r_gt (raw drive) | 1.06369 / 1.12515 / **1.09442** | 1.06342 / 1.12509 / **1.09426** | 0.9998495 | 6.0e-04 | 2.2e-07 |
| Transolver afu_r_t (raw drive) | 1.10187 / 1.14605 / **1.12396** | 1.10187 / 1.14605 / **1.12396** | 1.0000000 | 4.3e-09 | 2.3e-07 |

### native ratios, baseline / ISLA (seed means)

| arm | vs A2 0.00946: 3-D | vs A2: in-plane | vs afu_r_isla 0.01198: 3-D | vs afu_r_isla: in-plane |
|---|---|---|---|---|
| GeoTransolver afu_ud_gt (unit drive) | 4.228 | 4.228 | 3.336 | 3.336 |
| Transolver afu_ud_t (unit drive) | 3.138 | 3.138 | 2.476 | 2.476 |
| GeoTransolver afu_r_gt (raw drive) | 3.860 | 3.860 | 3.046 | 3.046 |
| Transolver afu_r_t (raw drive) | 7.079 | 7.079 | 5.587 | 5.587 |

ISLA check (afu_r_isla_seed42 native, 2-D): velocity_l2 0.011607, velocity_xy_l2 0.011607 (identical: 2 components), max per-case rel. diff vs shared rows 2.0e-07
