# HL-AREA: area protocol on the six deduplicated-wing checkpoints (seed mean; seeds 42, 43)

| arm | protocol | pressure | wall shear | velocity | density | temperature |
|---|---|---|---|---|---|---|
| ISLA | native | 0.05458 (0.05304, 0.05613) | 0.09633 (0.09453, 0.09812) | 0.07113 | 0.05401 | 0.06366 |
| ISLA | posed | 0.05509 (0.05353, 0.05664) | 0.09685 (0.09506, 0.09864) | 0.07134 | 0.05442 | 0.06392 |
| ISLA | area | 0.05426 (0.05245, 0.05606) | 0.09612 (0.09417, 0.09807) | 0.07091 | 0.05370 | 0.06335 |
| GeoTransolver | native | 0.05109 (0.05056, 0.05161) | 0.09214 (0.09211, 0.09218) | 0.06823 | 0.05060 | 0.06123 |
| GeoTransolver | posed | 1.01703 (1.02585, 1.00820) | 1.26181 (1.28339, 1.24023) | 1.20033 | 0.99966 | 0.62772 |
| GeoTransolver | area | 0.26088 (0.23886, 0.28289) | 0.35275 (0.33094, 0.37455) | 0.23895 | 0.24314 | 0.23238 |
| Transolver | native | 0.05274 (0.05420, 0.05129) | 0.09189 (0.09339, 0.09039) | 0.06917 | 0.05211 | 0.06278 |
| Transolver | posed | 0.99325 (1.00942, 0.97709) | 1.28170 (1.31991, 1.24349) | 1.25255 | 0.96762 | 0.63411 |
| Transolver | area | 0.18938 (0.19495, 0.18381) | 0.29541 (0.31049, 0.28032) | 0.20052 | 0.17595 | 0.18805 |

Baseline / ISLA (seed-mean ratio; per-seed ratios s42, s43)

| baseline | protocol | pressure | wall shear |
|---|---|---|---|
| GeoTransolver | native | 0.936 (0.953, 0.920) | 0.957 (0.974, 0.939) |
| GeoTransolver | posed | 18.462 (19.164, 17.799) | 13.029 (13.501, 12.574) |
| GeoTransolver | area | 4.808 (4.554, 5.046) | 3.670 (3.514, 3.819) |
| Transolver | native | 0.966 (1.022, 0.914) | 0.954 (0.988, 0.921) |
| Transolver | posed | 18.030 (18.857, 17.250) | 13.234 (13.885, 12.607) |
| Transolver | area | 3.491 (3.717, 3.279) | 3.073 (3.297, 2.858) |

Area / native per seed (s42, s43)

| arm | pressure_l2 | tau_wall_l2 | velocity_l2 | density_l2 | temperature_l2 |
|---|---|---|---|---|---|
| ISLA | 0.989, 0.999 | 0.996, 0.999 | 0.997, 0.997 | 0.990, 0.999 | 0.992, 0.998 |
| GeoTransolver | 4.724, 5.481 | 3.593, 4.063 | 3.275, 3.730 | 4.456, 5.149 | 3.560, 4.028 |
| Transolver | 3.597, 3.584 | 3.325, 3.101 | 2.990, 2.805 | 3.372, 3.381 | 3.038, 2.951 |

Checks

- hldupx_hl_isla_seed42/native: {'epoch': 500, 'cases': 180, 'dataset': 'highlift_surface_medium_dedup', 'precision': None}
- hldupx_hl_isla_seed42/posed: {'epoch': 500, 'cases': 180, 'dataset': 'highlift_surface_medium_dedup_posed', 'precision': None}
- hldupx_hl_isla_seed42/area: {'epoch': 500, 'cases': 180, 'dataset': 'highlift_surface_medium_dedup_area', 'precision': None}
- hldupx_hl_isla_seed42/area_cases_equal_native: True
- hldupx_hl_isla_seed42/area_identical_cases_pressure: 0
- hldupx_hl_isla_seed43/native: {'epoch': 500, 'cases': 180, 'dataset': 'highlift_surface_medium_dedup', 'precision': None}
- hldupx_hl_isla_seed43/posed: {'epoch': 500, 'cases': 180, 'dataset': 'highlift_surface_medium_dedup_posed', 'precision': None}
- hldupx_hl_isla_seed43/area: {'epoch': 500, 'cases': 180, 'dataset': 'highlift_surface_medium_dedup_area', 'precision': None}
- hldupx_hl_isla_seed43/area_cases_equal_native: True
- hldupx_hl_isla_seed43/area_identical_cases_pressure: 0
- hldupx_hl_gt_seed42/native: {'epoch': 500, 'cases': 180, 'dataset': 'highlift_surface_medium_dedup', 'precision': None}
- hldupx_hl_gt_seed42/posed: {'epoch': 500, 'cases': 180, 'dataset': 'highlift_surface_medium_dedup_posed', 'precision': None}
- hldupx_hl_gt_seed42/area: {'epoch': 500, 'cases': 180, 'dataset': 'highlift_surface_medium_dedup_area', 'precision': None}
- hldupx_hl_gt_seed42/area_cases_equal_native: True
- hldupx_hl_gt_seed42/area_identical_cases_pressure: 0
- hldupx_hl_gt_seed43/native: {'epoch': 500, 'cases': 180, 'dataset': 'highlift_surface_medium_dedup', 'precision': None}
- hldupx_hl_gt_seed43/posed: {'epoch': 500, 'cases': 180, 'dataset': 'highlift_surface_medium_dedup_posed', 'precision': None}
- hldupx_hl_gt_seed43/area: {'epoch': 500, 'cases': 180, 'dataset': 'highlift_surface_medium_dedup_area', 'precision': None}
- hldupx_hl_gt_seed43/area_cases_equal_native: True
- hldupx_hl_gt_seed43/area_identical_cases_pressure: 0
- hldupx_hl_t_seed42/native: {'epoch': 500, 'cases': 180, 'dataset': 'highlift_surface_medium_dedup', 'precision': None}
- hldupx_hl_t_seed42/posed: {'epoch': 500, 'cases': 180, 'dataset': 'highlift_surface_medium_dedup_posed', 'precision': None}
- hldupx_hl_t_seed42/area: {'epoch': 500, 'cases': 180, 'dataset': 'highlift_surface_medium_dedup_area', 'precision': None}
- hldupx_hl_t_seed42/area_cases_equal_native: True
- hldupx_hl_t_seed42/area_identical_cases_pressure: 0
- hldupx_hl_t_seed43/native: {'epoch': 500, 'cases': 180, 'dataset': 'highlift_surface_medium_dedup', 'precision': None}
- hldupx_hl_t_seed43/posed: {'epoch': 500, 'cases': 180, 'dataset': 'highlift_surface_medium_dedup_posed', 'precision': None}
- hldupx_hl_t_seed43/area: {'epoch': 500, 'cases': 180, 'dataset': 'highlift_surface_medium_dedup_area', 'precision': None}
- hldupx_hl_t_seed43/area_cases_equal_native: True
- hldupx_hl_t_seed43/area_identical_cases_pressure: 0
