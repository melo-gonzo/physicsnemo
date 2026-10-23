## SHIFT-SUV (r090_suv_isla, epoch 500, 99 cases)

| field | R1 | 0.9.0 native (s42, s43) | A2 native (s42, s43) | r = 0.9.0 ÷ A2 | q42, q43 | posed ÷ native s42, s43 | area ÷ native s42, s43 | posed 0.9.0 ÷ A2 |
|---|---|---|---|---|---|---|---|---|
| pressure | low-noise | 0.044501 (0.045559, 0.043444) | 0.045624 (0.045798, 0.04545) | 0.975 | 0.999, 0.952 | 1.0000, 1.0000 | 0.9984, 1.0005 | 0.975 |
| wall shear | low-noise | 0.066532 (0.06709, 0.065973) | 0.067659 (0.068023, 0.067294) | 0.983 | 0.992, 0.975 | 1.0001, 1.0001 | 1.0022, 1.0018 | 0.983 |

G_d = 0.979

## Tensile2d (r090_t2dz_isla, epoch 500, 100 cases)

| field | R1 | 0.9.0 native (s42, s43) | A2 native (s42, s43) | r = 0.9.0 ÷ A2 | q42, q43 | posed ÷ native s42, s43 | area ÷ native s42, s43 | posed 0.9.0 ÷ A2 |
|---|---|---|---|---|---|---|---|---|
| displacement | low-noise | 0.0071644 (0.0072464, 0.0070825) | 0.0068792 (0.0067943, 0.0069641) | 1.041 | 1.053, 1.030 | 1.0000, 1.0000 | 1.0000, 1.0000 | 0.951 |
| stress (Frobenius) | noisy | 0.0088602 (0.0089978, 0.0087226) | 0.0078084 (0.0070233, 0.0085934) | 1.135 | 1.152, 1.117 | 1.0000, 1.0000 | 1.0000, 1.0000 | 1.041 |
| plastic strain | noisy | 0.28644 (0.33535, 0.23753) | 0.41451 (0.36868, 0.46035) | 0.691 | 0.809, 0.573 | 1.0000, 1.0000 | 1.0000, 1.0000 | 0.689 |

G_d = 1.041

## DrivAerML Surface (r090_dr_isla, epoch 500, 48 cases)

| field | R1 | 0.9.0 native (s42, s43) | A2 native (s42, s43) | r = 0.9.0 ÷ A2 | q42, q43 | posed ÷ native s42, s43 | area ÷ native s42, s43 | posed 0.9.0 ÷ A2 |
|---|---|---|---|---|---|---|---|---|
| pressure | low-noise | 0.041134 (0.04116, 0.041108) | 0.044516 (0.043578, 0.045454) | 0.924 | 0.925, 0.923 | 1.0000, 1.0000 | 1.0016, 1.0070 | 0.924 |
| wall shear | low-noise | 0.066221 (0.066194, 0.066248) | 0.070038 (0.068352, 0.071723) | 0.946 | 0.945, 0.946 | 1.0000, 1.0001 | 1.0124, 1.0157 | 0.946 |

G_d = 0.935

## DrivAerML Volume (r090_vol_isla, epoch 500, 48 cases)

| field | R1 | 0.9.0 native (s42, s43) | A2 native (s42, s43) | r = 0.9.0 ÷ A2 | q42, q43 | posed ÷ native s42, s43 | area ÷ native s42, s43 | posed 0.9.0 ÷ A2 |
|---|---|---|---|---|---|---|---|---|
| pressure | low-noise | 0.038377 (0.037954, 0.038801) | 0.040859 (0.041315, 0.040402) | 0.939 | 0.929, 0.950 | 1.0027, 1.0035 | 1.0000, 1.0000 | 0.942 |
| velocity | low-noise | 0.057705 (0.057326, 0.058084) | 0.065956 (0.066216, 0.065696) | 0.875 | 0.869, 0.881 | 1.0040, 1.0033 | 1.0000, 1.0000 | 0.878 |
| eddy viscosity | low-noise | 0.084655 (0.084012, 0.085298) | 0.088064 (0.089467, 0.086661) | 0.961 | 0.954, 0.969 | 1.0001, 0.9998 | 1.0000, 1.0000 | 0.961 |

G_d = 0.924

## AirfRANS uniform (afu_a2s_isla) (r090_af_isla, epoch 500, 200 cases)

| field | R1 | 0.9.0 native (s42, s43) | A2 native (s42, s43) | r = 0.9.0 ÷ A2 | q42, q43 | posed ÷ native s42, s43 | area ÷ native s42, s43 | posed 0.9.0 ÷ A2 |
|---|---|---|---|---|---|---|---|---|
| pressure | low-noise | 0.011857 (0.012507, 0.011207) | 0.017994 (0.018541, 0.017447) | 0.659 | 0.695, 0.623 | 1.0025, 1.0028 | 1.0000, 1.0000 | 0.661 |
| velocity | low-noise | 0.0064361 (0.0065379, 0.0063343) | 0.0094558 (0.0096975, 0.0092141) | 0.681 | 0.691, 0.670 | 1.0015, 1.0015 | 1.0000, 1.0000 | 0.682 |
| eddy viscosity | low-noise | 0.01581 (0.016227, 0.015392) | 0.022298 (0.023255, 0.021342) | 0.709 | 0.728, 0.690 | 1.0000, 1.0000 | 1.0000, 1.0000 | 0.709 |

G_d = 0.683

## DeepJEB++ (jbn2_isla) (r090_jb_isla, epoch 40, 5540 cases)

| field | R1 | 0.9.0 native (s42, s43) | A2 native (s42, s43) | r = 0.9.0 ÷ A2 | q42, q43 | posed ÷ native s42, s43 | area ÷ native s42, s43 | posed 0.9.0 ÷ A2 |
|---|---|---|---|---|---|---|---|---|
| von Mises | low-noise | 0.39747 (0.39238, 0.40257) | 0.39185 (0.39416, 0.38955) | 1.014 | 1.001, 1.027 | 1.0000, 1.0000 | 1.0000, 1.0000 | 1.014 |
| displacement | low-noise | 0.16549 (0.15831, 0.17266) | 0.16532 (0.16879, 0.16186) | 1.001 | 0.958, 1.044 | 1.0000, 1.0000 | 1.0000, 1.0000 | 1.001 |

G_d = 1.008

## GINOT lug (r090_lug_isla, epoch 500, 600 cases)

| field | R1 | 0.9.0 native (s42, s43) | A2 native (s42, s43) | r = 0.9.0 ÷ A2 | q42, q43 | posed ÷ native s42, s43 | area ÷ native s42, s43 | posed 0.9.0 ÷ A2 |
|---|---|---|---|---|---|---|---|---|
| von Mises | noisy | 0.0019044 (0.0018163, 0.0019925) | 0.0017256 (0.0016015, 0.0018497) | 1.104 | 1.053, 1.155 | 1.1413, 1.1071 | 1.0000, 0.9998 | 0.358 |

**R1 (pooled, 13 fields): G13 = 0.891; 1.10-with-both-seeds fields: none; verdict: confirmed**

## Standing baselines ÷ ISLA (native | posed, seed means): against 0.9.0, and [against A2]

- SHIFT-SUV, GeoTransolver 256: pressure 1.023 | 23.724 [0.998] level; wall shear 0.999 | 17.596 [0.983] level
- SHIFT-SUV, Transolver 512 x 16 at 100k cells: pressure 0.942 | 24.188 [0.919] ISLA trails; wall shear 0.961 | 17.768 [0.945] ISLA trails
- Tensile2d, GeoTransolver 256: stress (Frobenius) 0.601 | 121.533 [0.682] ISLA trails; displacement 0.561 | 175.616 [0.584] ISLA trails; plastic strain 1.607 | 11.118 [1.111] ISLA leads
- Tensile2d, Transolver 512 x 16: stress (Frobenius) 0.778 | 113.547 [0.883] ISLA trails; displacement 0.769 | 172.588 [0.800] ISLA trails; plastic strain 0.804 | 16.162 [0.556] level
- DrivAerML Surface, GeoTransolver 256: pressure 1.107 | 30.720 [1.022] ISLA leads; wall shear 1.051 | 19.250 [0.994] ISLA leads
- DrivAerML Surface, Transolver 512 x 16, trained at 100k cells: pressure 1.029 | 31.428 [0.951] level; wall shear 1.024 | 20.453 [0.968] level
- DrivAerML Volume, GeoTransolver 256: pressure 1.253 | 21.228 [1.176] ISLA leads; velocity 1.732 | 23.160 [1.515] ISLA leads; eddy viscosity 0.998 | 12.465 [0.960] level
- DrivAerML Volume, Transolver 512 x 16 at 100k cells: pressure 1.289 | 20.009 [1.211] ISLA leads; velocity 1.946 | 21.975 [1.702] ISLA leads; eddy viscosity 1.041 | 12.788 [1.000] ISLA leads
- AirfRANS uniform (afu_a2s_isla), GeoTransolver 256: pressure 0.919 | 151.745 [0.606] ISLA trails; velocity 6.211 | 181.046 [4.228] ISLA leads; eddy viscosity 1.080 | 100.499 [0.766] ISLA leads
- AirfRANS uniform (afu_a2s_isla), Transolver 512 x 16: pressure 0.919 | 106.919 [0.606] ISLA trails; velocity 4.610 | 190.906 [3.138] ISLA leads; eddy viscosity 1.127 | 117.022 [0.799] ISLA leads
- DeepJEB++ (jbn2_isla), GeoTransolver 256: von Mises 1.251 | 3.115 [1.269] ISLA leads; displacement 1.394 | 9.218 [1.395] ISLA leads
- DeepJEB++ (jbn2_isla), Transolver 512 x 16: von Mises 1.098 | 2.932 [1.114] ISLA leads; displacement 1.175 | 8.187 [1.176] ISLA leads
- GINOT lug, GeoTransolver 256: von Mises 1.176 | 445.640 [1.297] ISLA leads
- GINOT lug, Transolver 512 x 16: von Mises 1.138 | 478.657 [1.256] ISLA leads

## Fields with 0.9.0 ÷ A2 > 1.03 or both 0.9.0 seeds above both A2 seeds (native, scorer numbers); flagged = both

| dataset | field | R1 class | r | 0.9.0 seeds | A2 seeds | r > 1.03 | both above | flagged |
|---|---|---|---|---|---|---|---|---|
| Tensile2d | displacement | low-noise | 1.041 | 0.0072464, 0.0070825 | 0.0067943, 0.0069641 | yes | yes | **yes** |
| Tensile2d | stress (Frobenius) | noisy | 1.135 | 0.0089978, 0.0087226 | 0.0070233, 0.0085934 | yes | yes | **yes** |
| GINOT lug | von Mises | noisy | 1.104 | 0.0018163, 0.0019925 | 0.0016015, 0.0018497 | yes | no | no |

## R1 beside the registered one: eager re-scores on AirfRANS, Tensile2d, the lug (both sides); scorer numbers elsewhere

| dataset | field | 0.9.0 eager (s42, s43) | A2 eager (s42, s43) | r eager | q42, q43 | r scorer |
|---|---|---|---|---|---|---|
| Tensile2d | displacement | 0.0071291 (0.0072856, 0.0069726) | 0.0068209 (0.0067913, 0.0068504) | 1.045 | 1.068, 1.022 | 1.041 |
| Tensile2d | stress (Frobenius) | 0.0088554 (0.0089959, 0.0087149) | 0.0078022 (0.0070944, 0.00851) | 1.135 | 1.153, 1.117 | 1.135 |
| Tensile2d | plastic strain | 0.29116 (0.34283, 0.23949) | 0.40657 (0.36762, 0.44551) | 0.716 | 0.843, 0.589 | 0.691 |
| AirfRANS uniform (afu_a2s_isla) | pressure | 0.01194 (0.012644, 0.011236) | 0.018042 (0.018595, 0.017489) | 0.662 | 0.701, 0.623 | 0.659 |
| AirfRANS uniform (afu_a2s_isla) | velocity | 0.0064229 (0.006525, 0.0063208) | 0.0095102 (0.0097534, 0.0092671) | 0.675 | 0.686, 0.665 | 0.681 |
| AirfRANS uniform (afu_a2s_isla) | eddy viscosity | 0.015589 (0.016011, 0.015166) | 0.022332 (0.023137, 0.021528) | 0.698 | 0.717, 0.679 | 0.709 |
| GINOT lug | von Mises | 0.0018933 (0.0017804, 0.0020062) | 0.0017235 (0.001591, 0.0018559) | 1.099 | 1.033, 1.164 | 1.104 |

Eager-side G_d: shift_suv 0.979, tensile2d 1.045, drivaerml_surface 0.935, drivaerml_volume 0.924, airfrans 0.678, deepjeb 1.008
Eager-side pooled G13 = 0.890; 1.10 clause: none

## REF-090-FIX AirfRANS uniform (afu_a2s_isla): retrained (r090fix_af_isla, eager-scored) against the original r090 rows (eager) and A2 (eager)

| field | retrained (s42, s43) | original eager (s42, s43) | T = retrained ÷ original (s42, s43 paired) | |T−1| vs A2 two-seed gap | retrained ÷ A2 eager (q42, q43) | posed ÷ native s42, s43 |
|---|---|---|---|---|---|---|
| pressure | 0.012224 (0.012987, 0.01146) | 0.01194 (0.012644, 0.011236) | 1.024 (1.027, 1.020) | 0.024 vs 0.061 | 0.678 (0.720, 0.635) | 1.0025, 1.0029 |
| velocity | 0.0063269 (0.0064288, 0.0062249) | 0.0064229 (0.006525, 0.0063208) | 0.985 (0.985, 0.985) | 0.015 vs 0.051 | 0.665 (0.676, 0.655) | 1.0015, 1.0016 |
| eddy viscosity | 0.015565 (0.01574, 0.01539) | 0.015589 (0.016011, 0.015166) | 0.998 (0.983, 1.015) | 0.002 vs 0.086 | 0.697 (0.705, 0.689) | 1.0000, 1.0001 |

## REF-090-FIX Tensile2d: retrained (r090fix_t2dz_isla, eager-scored) against the original r090 rows (eager) and A2 (eager)

| field | retrained (s42, s43) | original eager (s42, s43) | T = retrained ÷ original (s42, s43 paired) | |T−1| vs A2 two-seed gap | retrained ÷ A2 eager (q42, q43) | posed ÷ native s42, s43 |
|---|---|---|---|---|---|---|
| displacement | 0.0068885 (0.0073367, 0.0064402) | 0.0071291 (0.0072856, 0.0069726) | 0.966 (1.007, 0.924) | 0.034 vs 0.025 EXCEEDS | 1.010 (1.076, 0.944) | 1.0005, 1.0000 |
| stress (Frobenius) | 0.0087955 (0.0093799, 0.0082111) | 0.0088554 (0.0089959, 0.0087149) | 0.993 (1.043, 0.942) | 0.007 vs 0.202 | 1.127 (1.202, 1.052) | 1.0001, 1.0000 |
| plastic strain | 0.18313 (0.20261, 0.16365) | 0.29116 (0.34283, 0.23949) | 0.629 (0.591, 0.683) | 0.371 vs 0.222 EXCEEDS | 0.450 (0.498, 0.403) | 1.0000, 1.0000 |

## REF-090-FIX GINOT lug: retrained (r090fix_lug_isla, eager-scored) against the original r090 rows (eager) and A2 (eager)

| field | retrained (s42, s43) | original eager (s42, s43) | T = retrained ÷ original (s42, s43 paired) | |T−1| vs A2 two-seed gap | retrained ÷ A2 eager (q42, q43) | posed ÷ native s42, s43 |
|---|---|---|---|---|---|---|
| von Mises | 0.0014537 (0.0014625, 0.0014449) | 0.0018933 (0.0017804, 0.0020062) | 0.768 (0.821, 0.720) | 0.232 vs 0.144 EXCEEDS | 0.843 (0.849, 0.838) | 1.3461, 1.3829 |

r090fix_af_dtt ÷ r090fix_af_isla (native | posed): pressure 1.035 | 1.035 (DTT 0.012654); velocity 1.126 | 1.126 (DTT 0.0071261); eddy viscosity 1.381 | 1.381 (DTT 0.021494)

r090fix_lug_dtt ÷ r090fix_lug_isla (native | posed): von Mises 1.394 | 1.769 (DTT 0.0020268)
