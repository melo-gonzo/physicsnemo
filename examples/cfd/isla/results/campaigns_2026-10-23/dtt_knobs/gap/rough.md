# Roughness of the error (roughness.py): 0 = the same at neighbouring points, 1 = uncorrelated

## af test (200 cases)

| region | points | pressure: target / ISLA err / DTT err / DTT-ISLA diff | velocity: target / ISLA err / DTT err / DTT-ISLA diff | nut: target / ISLA err / DTT err / DTT-ISLA diff |
|---|---|---|---|---|
| d<1e-3 | 0.288 | 0.000 / 0.004 / 0.004 / 0.007 | 0.001 / 0.032 / 0.032 / 0.028 | 0.001 / 0.027 / 0.029 / 0.027 |
| 1e-3..1e-2 | 0.184 | 0.000 / 0.021 / 0.018 / 0.003 | 0.000 / 0.034 / 0.026 / 0.010 | 0.003 / 0.027 / 0.028 / 0.024 |
| 1e-2..0.1 | 0.171 | 0.000 / 0.025 / 0.025 / 0.007 | 0.000 / 0.019 / 0.013 / 0.008 | 0.003 / 0.013 / 0.021 / 0.027 |
| 0.1..1 (not wake) | 0.198 | 0.000 / 0.006 / 0.009 / 0.006 | 0.000 / 0.003 / 0.005 / 0.004 | 0.002 / 0.094 / 0.088 / 0.076 |
| >=1 (not wake) | 0.062 | 0.000 / 0.008 / 0.012 / 0.011 | 0.000 / 0.010 / 0.009 / 0.011 | 0.018 / 0.045 / 0.095 / 0.112 |
| wake (d>=1e-2) | 0.096 | 0.000 / 0.011 / 0.014 / 0.015 | 0.000 / 0.010 / 0.006 / 0.007 | 0.001 / 0.004 / 0.007 / 0.009 |

## lug test (600 cases)

| region | points | von_mises: target / ISLA err / DTT err / DTT-ISLA diff |
|---|---|---|
| on surface d<1e-6 L | 0.336 | 0.046 / 0.880 / 0.777 / 0.711 |
| interior | 0.664 | 0.032 / 0.736 / 0.706 / 0.658 |

