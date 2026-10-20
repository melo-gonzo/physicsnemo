## UDRV-HL readout: HiLift (highlift_surface_medium, 180 val cases, float32, epoch-500 checkpoints, ISLA's 200,000-cell set)

Physical-drive GeoTransolver source: lhr. Seed mean (seed 42, seed 43); relative L2.

| arm | protocol | pressure | wall shear | velocity | density | temperature |
|---|---|---|---|---|---|---|
| ISLA A2 (a2r_hl_isla) | native | 0.0432 (0.0435, 0.0428) | 0.0943 (0.0944, 0.0942) | 0.0605 (0.0609, 0.0602) | 0.0577 (0.0582, 0.0573) | 0.0494 (0.0497, 0.0490) |
| ISLA A2 (a2r_hl_isla) | area | 0.0430 (0.0433, 0.0427) | 0.0944 (0.0945, 0.0943) | 0.0604 (0.0608, 0.0600) | 0.0574 (0.0578, 0.0570) | 0.0493 (0.0496, 0.0490) |
| ISLA A2 (a2r_hl_isla) | posed | 0.0436 (0.0439, 0.0432) | 0.0948 (0.0949, 0.0947) | 0.0607 (0.0610, 0.0604) | 0.0582 (0.0586, 0.0577) | 0.0496 (0.0499, 0.0492) |
| GeoTransolver 256, U_inf_dir (udrv_hl_gt) | native | 0.0390 (0.0391, 0.0390) | 0.0890 (0.0890, 0.0890) | 0.0579 (0.0578, 0.0580) | 0.0527 (0.0527, 0.0527) | 0.0469 (0.0469, 0.0470) |
| GeoTransolver 256, U_inf_dir (udrv_hl_gt) | area | 0.1680 (0.1345, 0.2016) | 0.2777 (0.2336, 0.3219) | 0.1688 (0.1437, 0.1939) | 0.2195 (0.1792, 0.2598) | 0.1437 (0.1179, 0.1695) |
| GeoTransolver 256, U_inf_dir (udrv_hl_gt) | posed | 0.9519 (0.9432, 0.9606) | 1.4668 (1.4448, 1.4888) | 1.2605 (1.2392, 1.2819) | 1.2716 (1.2575, 1.2857) | 0.5840 (0.5899, 0.5780) |
| GeoTransolver 256, U_inf (r80_hl_gt_h256) | native | 0.0525 (0.0554, 0.0496) | 0.1237 (0.1304, 0.1169) | 0.0771 (0.0816, 0.0726) | 0.0710 (0.0751, 0.0670) | 0.0564 (0.0583, 0.0544) |
| GeoTransolver 256, U_inf (r80_hl_gt_h256) | area | 0.2215 (0.2360, 0.2070) | 0.3786 (0.3891, 0.3681) | 0.2662 (0.3024, 0.2300) | 0.2793 (0.2922, 0.2664) | 0.1981 (0.2090, 0.1873) |
| GeoTransolver 256, U_inf (r80_hl_gt_h256) | posed | 1.7070 (1.6914, 1.7225) | 1.6064 (1.5041, 1.7086) | 1.1102 (1.0245, 1.1958) | 1.8499 (1.8844, 1.8154) | 1.2780 (1.1899, 1.3661) |
| Transolver 512 x 16, U_inf_dir (udrv_hl_t) | native | 0.0394 (0.0408, 0.0380) | 0.0878 (0.0888, 0.0868) | 0.0574 (0.0576, 0.0572) | 0.0530 (0.0547, 0.0513) | 0.0475 (0.0482, 0.0467) |
| Transolver 512 x 16, U_inf_dir (udrv_hl_t) | area | 0.1368 (0.1404, 0.1332) | 0.2556 (0.2593, 0.2518) | 0.1552 (0.1577, 0.1527) | 0.1760 (0.1782, 0.1739) | 0.1289 (0.1324, 0.1253) |
| Transolver 512 x 16, U_inf_dir (udrv_hl_t) | posed | 0.8747 (0.8836, 0.8658) | 1.4620 (1.4993, 1.4247) | 1.2720 (1.2794, 1.2647) | 1.1439 (1.1496, 1.1383) | 0.5521 (0.5610, 0.5432) |
| Transolver 512 x 16, U_inf (r80_hl_t_h512_l16_c100k) @200k | native | 0.0434 (0.0435, 0.0433) | 0.1005 (0.1008, 0.1002) | 0.0630 (0.0630, 0.0631) | 0.0575 (0.0577, 0.0573) | 0.0506 (0.0507, 0.0506) |
| Transolver 512 x 16, U_inf (r80_hl_t_h512_l16_c100k) @200k | area | 0.2890 (0.2940, 0.2839) | 0.4639 (0.4738, 0.4540) | 0.2750 (0.2794, 0.2706) | 0.3776 (0.3889, 0.3662) | 0.2238 (0.2224, 0.2252) |
| Transolver 512 x 16, U_inf (r80_hl_t_h512_l16_c100k) @200k | posed | 1.0133 (1.0048, 1.0218) | 1.4372 (1.2995, 1.5750) | 1.0786 (1.0519, 1.1054) | 1.1752 (1.2078, 1.1426) | 0.7535 (0.7175, 0.7895) |
| GeoTransolver 256, U_inf, LHR re-score @200k [check] | native | 0.0525 (0.0554, 0.0496) | 0.1237 (0.1304, 0.1169) | 0.0771 (0.0816, 0.0726) | 0.0710 (0.0751, 0.0670) | 0.0564 (0.0583, 0.0544) |
| GeoTransolver 256, U_inf, LHR re-score @200k [check] | area | 0.2215 (0.2360, 0.2070) | 0.3786 (0.3891, 0.3681) | 0.2662 (0.3024, 0.2300) | 0.2793 (0.2922, 0.2664) | 0.1981 (0.2090, 0.1873) |
| GeoTransolver 256, U_inf, LHR re-score @200k [check] | posed | 1.7070 (1.6914, 1.7225) | 1.6064 (1.5041, 1.7086) | 1.1102 (1.0245, 1.1958) | 1.8499 (1.8844, 1.8154) | 1.2780 (1.1899, 1.3661) |
| GeoTransolver 256, U_inf, CHAR record (AGA) @200k | native | 0.0522 (0.0552, 0.0492) | 0.1233 (0.1301, 0.1165) | 0.0771 (0.0816, 0.0726) | 0.0708 (0.0749, 0.0666) | 0.0561 (0.0581, 0.0541) |
| GeoTransolver 256, U_inf, CHAR record (AGA) @200k | area | 0.2215 (0.2363, 0.2067) | 0.3784 (0.3891, 0.3677) | 0.2663 (0.3026, 0.2299) | 0.2795 (0.2927, 0.2662) | 0.1980 (0.2091, 0.1870) |
| GeoTransolver 256, U_inf, CHAR record (AGA) @200k | posed | 1.7095 (1.6946, 1.7244) | 1.6039 (1.5042, 1.7036) | 1.1076 (1.0256, 1.1897) | 1.8543 (1.8877, 1.8209) | 1.2757 (1.1899, 1.3615) |
| Transolver 512 x 16, U_inf, CHAR record (AGA) @100k [not like-for-like] | native | 0.0434 (0.0435, 0.0433) | 0.1009 (0.1012, 0.1006) | 0.0632 (0.0631, 0.0633) | 0.0576 (0.0577, 0.0575) | 0.0505 (0.0506, 0.0505) |
| Transolver 512 x 16, U_inf, CHAR record (AGA) @100k [not like-for-like] | area | 0.3956 (0.4232, 0.3679) | 0.5562 (0.5733, 0.5392) | 0.3626 (0.3735, 0.3517) | 0.5391 (0.5882, 0.4900) | 0.2612 (0.2601, 0.2624) |
| Transolver 512 x 16, U_inf, CHAR record (AGA) @100k [not like-for-like] | posed | 1.0144 (1.0052, 1.0237) | 1.4390 (1.3011, 1.5769) | 1.0783 (1.0517, 1.1049) | 1.1769 (1.2093, 1.1444) | 0.7548 (0.7176, 0.7921) |

### Unit drive ÷ physical drive, per arm, like-for-like at 200k (seed means)

| ratio | protocol | pressure | wall shear | velocity | density | temperature |
|---|---|---|---|---|---|---|
| GeoTransolver | native | 0.744 | 0.720 | 0.750 | 0.742 | 0.833 |
| GeoTransolver | area | 0.759 | 0.734 | 0.634 | 0.786 | 0.726 |
| GeoTransolver | posed | 0.558 | 0.913 | 1.135 | 0.687 | 0.457 |
| Transolver | native | 0.908 | 0.874 | 0.910 | 0.922 | 0.938 |
| Transolver | area | 0.474 | 0.551 | 0.564 | 0.466 | 0.576 |
| Transolver | posed | 0.863 | 1.017 | 1.179 | 0.973 | 0.733 |

### Baseline ÷ ISLA A2 (seed means)

| ratio | protocol | pressure | wall shear | velocity | density | temperature |
|---|---|---|---|---|---|---|
| GeoTransolver 256, U_inf_dir (udrv_hl_gt) | native | 0.904 | 0.945 | 0.956 | 0.913 | 0.951 |
| GeoTransolver 256, U_inf_dir (udrv_hl_gt) | area | 3.909 | 2.941 | 2.795 | 3.824 | 2.915 |
| GeoTransolver 256, U_inf_dir (udrv_hl_gt) | posed | 21.849 | 15.472 | 20.756 | 21.866 | 11.778 |
| Transolver 512 x 16, U_inf_dir (udrv_hl_t) | native | 0.913 | 0.932 | 0.948 | 0.918 | 0.961 |
| Transolver 512 x 16, U_inf_dir (udrv_hl_t) | area | 3.184 | 2.706 | 2.570 | 3.067 | 2.614 |
| Transolver 512 x 16, U_inf_dir (udrv_hl_t) | posed | 20.076 | 15.421 | 20.945 | 19.671 | 11.136 |
| GeoTransolver 256, U_inf (r80_hl_gt_h256) | native | 1.215 | 1.312 | 1.273 | 1.231 | 1.141 |
| GeoTransolver 256, U_inf (r80_hl_gt_h256) | area | 5.153 | 4.009 | 4.409 | 4.866 | 4.018 |
| GeoTransolver 256, U_inf (r80_hl_gt_h256) | posed | 39.178 | 16.944 | 18.280 | 31.810 | 25.777 |
| Transolver 512 x 16, U_inf (r80_hl_t_h512_l16_c100k) @200k | native | 1.005 | 1.066 | 1.041 | 0.996 | 1.025 |
| Transolver 512 x 16, U_inf (r80_hl_t_h512_l16_c100k) @200k | area | 6.723 | 4.913 | 4.554 | 6.578 | 4.539 |
| Transolver 512 x 16, U_inf (r80_hl_t_h512_l16_c100k) @200k | posed | 23.257 | 15.160 | 17.760 | 20.209 | 15.198 |

### P1: unit drive against the physical-drive self, native pressure, like-for-like (200k)

- GeoTransolver: unit 0.0390 vs physical 0.0525; ratio 0.744 (rule <= 0.90); unit lower on 163 of 180 cases (seed-mean pairs), median per-case factor 0.606 -> HOLDS
  - wall shear: ratio 0.720, lower on 172 of 180, median per-case 0.707
- Transolver: unit 0.0394 vs physical 0.0434; ratio 0.908 (rule < 1.00); unit lower on 156 of 180 cases (seed-mean pairs), median per-case factor 0.761 -> HOLDS
  - wall shear: ratio 0.874, lower on 166 of 180, median per-case 0.820
- Record only (not like-for-like): Transolver unit @200k ÷ CHAR physical @100k, native pressure 0.908

### P2: baseline ÷ ISLA A2, native, seed means; cases lower paired on seed-mean values

- GeoTransolver 256, U_inf_dir (udrv_hl_gt): pressure 0.904, wall shear 0.945, velocity 0.956, density 0.913, temperature 0.951; pressure lower on 99 of 180 cases (median per-case 0.993); wall shear lower on 139 of 180; band: between (ratio <= 0.95 but lower on < 2/3 of cases)
- Transolver 512 x 16, U_inf_dir (udrv_hl_t): pressure 0.913, wall shear 0.932, velocity 0.948, density 0.918, temperature 0.961; pressure lower on 153 of 180 cases (median per-case 0.889); wall shear lower on 157 of 180; band: baseline ahead
- GeoTransolver 256, U_inf (r80_hl_gt_h256): pressure 1.215, wall shear 1.312, velocity 1.273, density 1.231, temperature 1.141; pressure lower on 23 of 180 cases (median per-case 1.612); wall shear lower on 13 of 180; band: ISLA ahead
- Transolver 512 x 16, U_inf (r80_hl_t_h512_l16_c100k) @200k: pressure 1.005, wall shear 1.066, velocity 1.041, density 0.996, temperature 1.025; pressure lower on 40 of 180 cases (median per-case 1.206); wall shear lower on 22 of 180; band: parity

### P3: area ÷ native and posed ÷ native (seed means; ISLA also per seed, R-POSE |posed/native - 1| <= 0.002)

- ISLA A2 (a2r_hl_isla): pressure area 0.995, posed 1.0090; wall shear area 1.002, posed 1.0058; velocity area 0.997, posed 1.0031; density area 0.994, posed 1.0074; temperature area 0.998, posed 1.0040
- GeoTransolver 256, U_inf_dir (udrv_hl_gt): pressure area 4.306, posed 24.3947; wall shear area 3.119, posed 16.4744; velocity area 2.917, posed 21.7856; density area 4.166, posed 24.1373; temperature area 3.062, posed 12.4389
- GeoTransolver 256, U_inf (r80_hl_gt_h256): pressure area 4.222, posed 32.5359; wall shear area 3.062, posed 12.9908; velocity area 3.453, posed 14.3981; density area 3.931, posed 26.0375; temperature area 3.516, posed 22.6785
- Transolver 512 x 16, U_inf_dir (udrv_hl_t): pressure area 3.472, posed 22.1949; wall shear area 2.910, posed 16.6450; velocity area 2.704, posed 22.1695; density area 3.321, posed 21.5812; temperature area 2.715, posed 11.6298
- Transolver 512 x 16, U_inf (r80_hl_t_h512_l16_c100k) @200k: pressure area 6.658, posed 23.3477; wall shear area 4.616, posed 14.3032; velocity area 4.363, posed 17.1142; density area 6.568, posed 20.4450; temperature area 4.421, posed 14.8866
- ISLA |posed/native - 1| per field and seed: pressure 42 0.0090, pressure 43 0.0089, wall shear 42 0.0055, wall shear 43 0.0061, velocity 42 0.0030, velocity 43 0.0031, density 42 0.0075, density 43 0.0074, temperature 42 0.0040, temperature 43 0.0039

### Score metadata

| arm | protocol | seed | epoch | num_samples | file |
|---|---|---|---|---|---|
| isla | native | 42 | 500 | 180 | scores/a2r_hl_isla_seed42/native/a2r_hl_isla_seed42/metrics.jsonl |
| isla | native | 43 | 500 | 180 | scores/a2r_hl_isla_seed43/native/a2r_hl_isla_seed43/metrics.jsonl |
| isla | area | 42 | 500 | 180 | scores/a2r_hl_isla_seed42/area/a2r_hl_isla_seed42/metrics.jsonl |
| isla | area | 43 | 500 | 180 | scores/a2r_hl_isla_seed43/area/a2r_hl_isla_seed43/metrics.jsonl |
| isla | posed | 42 | 500 | 180 | scores/a2r_hl_isla_seed42/posed/a2r_hl_isla_seed42/metrics.jsonl |
| isla | posed | 43 | 500 | 180 | scores/a2r_hl_isla_seed43/posed/a2r_hl_isla_seed43/metrics.jsonl |
| gt_unit | native | 42 | 500 | 180 | scores/udrv_hl_gt_seed42/native/udrv_hl_gt_seed42/metrics.jsonl |
| gt_unit | native | 43 | 500 | 180 | scores/udrv_hl_gt_seed43/native/udrv_hl_gt_seed43/metrics.jsonl |
| gt_unit | area | 42 | 500 | 180 | scores/udrv_hl_gt_seed42/area/udrv_hl_gt_seed42/metrics.jsonl |
| gt_unit | area | 43 | 500 | 180 | scores/udrv_hl_gt_seed43/area/udrv_hl_gt_seed43/metrics.jsonl |
| gt_unit | posed | 42 | 500 | 180 | scores/udrv_hl_gt_seed42/posed/udrv_hl_gt_seed42/metrics.jsonl |
| gt_unit | posed | 43 | 500 | 180 | scores/udrv_hl_gt_seed43/posed/udrv_hl_gt_seed43/metrics.jsonl |
| gt_phys | native | 42 | 500 | 180 | scores/r80_hl_gt_h256_seed42_s200k/native/r80_hl_gt_h256_seed42/metrics.jsonl |
| gt_phys | native | 43 | 500 | 180 | scores/r80_hl_gt_h256_seed43_s200k/native/r80_hl_gt_h256_seed43/metrics.jsonl |
| gt_phys | area | 42 | 500 | 180 | scores/r80_hl_gt_h256_seed42_s200k/area/r80_hl_gt_h256_seed42/metrics.jsonl |
| gt_phys | area | 43 | 500 | 180 | scores/r80_hl_gt_h256_seed43_s200k/area/r80_hl_gt_h256_seed43/metrics.jsonl |
| gt_phys | posed | 42 | 500 | 180 | scores/r80_hl_gt_h256_seed42_s200k/posed/r80_hl_gt_h256_seed42/metrics.jsonl |
| gt_phys | posed | 43 | 500 | 180 | scores/r80_hl_gt_h256_seed43_s200k/posed/r80_hl_gt_h256_seed43/metrics.jsonl |
| t_unit | native | 42 | 500 | 180 | scores/udrv_hl_t_seed42/native/udrv_hl_t_seed42/metrics.jsonl |
| t_unit | native | 43 | 500 | 180 | scores/udrv_hl_t_seed43/native/udrv_hl_t_seed43/metrics.jsonl |
| t_unit | area | 42 | 500 | 180 | scores/udrv_hl_t_seed42/area/udrv_hl_t_seed42/metrics.jsonl |
| t_unit | area | 43 | 500 | 180 | scores/udrv_hl_t_seed43/area/udrv_hl_t_seed43/metrics.jsonl |
| t_unit | posed | 42 | 500 | 180 | scores/udrv_hl_t_seed42/posed/udrv_hl_t_seed42/metrics.jsonl |
| t_unit | posed | 43 | 500 | 180 | scores/udrv_hl_t_seed43/posed/udrv_hl_t_seed43/metrics.jsonl |
| t_phys | native | 42 | 500 | 180 | scores/r80_hl_t_h512_l16_c100k_seed42_s200k/native/r80_hl_t_h512_l16_c100k_seed42/metrics.jsonl |
| t_phys | native | 43 | 500 | 180 | scores/r80_hl_t_h512_l16_c100k_seed43_s200k/native/r80_hl_t_h512_l16_c100k_seed43/metrics.jsonl |
| t_phys | area | 42 | 500 | 180 | scores/r80_hl_t_h512_l16_c100k_seed42_s200k/area/r80_hl_t_h512_l16_c100k_seed42/metrics.jsonl |
| t_phys | area | 43 | 500 | 180 | scores/r80_hl_t_h512_l16_c100k_seed43_s200k/area/r80_hl_t_h512_l16_c100k_seed43/metrics.jsonl |
| t_phys | posed | 42 | 500 | 180 | scores/r80_hl_t_h512_l16_c100k_seed42_s200k/posed/r80_hl_t_h512_l16_c100k_seed42/metrics.jsonl |
| t_phys | posed | 43 | 500 | 180 | scores/r80_hl_t_h512_l16_c100k_seed43_s200k/posed/r80_hl_t_h512_l16_c100k_seed43/metrics.jsonl |
| gt_phys_lhr | native | 42 | 500 | 180 | scores/r80_hl_gt_h256_seed42_s200k/native/r80_hl_gt_h256_seed42/metrics.jsonl |
| gt_phys_lhr | native | 43 | 500 | 180 | scores/r80_hl_gt_h256_seed43_s200k/native/r80_hl_gt_h256_seed43/metrics.jsonl |
| gt_phys_lhr | area | 42 | 500 | 180 | scores/r80_hl_gt_h256_seed42_s200k/area/r80_hl_gt_h256_seed42/metrics.jsonl |
| gt_phys_lhr | area | 43 | 500 | 180 | scores/r80_hl_gt_h256_seed43_s200k/area/r80_hl_gt_h256_seed43/metrics.jsonl |
| gt_phys_lhr | posed | 42 | 500 | 180 | scores/r80_hl_gt_h256_seed42_s200k/posed/r80_hl_gt_h256_seed42/metrics.jsonl |
| gt_phys_lhr | posed | 43 | 500 | 180 | scores/r80_hl_gt_h256_seed43_s200k/posed/r80_hl_gt_h256_seed43/metrics.jsonl |
| gt_phys_char | native | 42 | 500 | 180 | ../../examples/cfd/isla/results/campaigns_2026-10-02/char/protocols/r80_hl_gt_h256_seed42/native/metrics.jsonl |
| gt_phys_char | native | 43 | 500 | 180 | ../../examples/cfd/isla/results/campaigns_2026-10-02/char/protocols/r80_hl_gt_h256_seed43/native/metrics.jsonl |
| gt_phys_char | area | 42 | 500 | 180 | ../../examples/cfd/isla/results/campaigns_2026-10-02/char/protocols/r80_hl_gt_h256_seed42/area/metrics.jsonl |
| gt_phys_char | area | 43 | 500 | 180 | ../../examples/cfd/isla/results/campaigns_2026-10-02/char/protocols/r80_hl_gt_h256_seed43/area/metrics.jsonl |
| gt_phys_char | posed | 42 | 500 | 180 | ../../examples/cfd/isla/results/campaigns_2026-10-02/char/protocols/r80_hl_gt_h256_seed42/posed/metrics.jsonl |
| gt_phys_char | posed | 43 | 500 | 180 | ../../examples/cfd/isla/results/campaigns_2026-10-02/char/protocols/r80_hl_gt_h256_seed43/posed/metrics.jsonl |
| t_phys_c100k | native | 42 | 500 | 180 | ../../examples/cfd/isla/results/campaigns_2026-10-02/char/protocols/r80_hl_t_h512_l16_c100k_seed42/native/metrics.jsonl |
| t_phys_c100k | native | 43 | 500 | 180 | ../../examples/cfd/isla/results/campaigns_2026-10-02/char/protocols/r80_hl_t_h512_l16_c100k_seed43/native/metrics.jsonl |
| t_phys_c100k | area | 42 | 500 | 180 | ../../examples/cfd/isla/results/campaigns_2026-10-02/char/protocols/r80_hl_t_h512_l16_c100k_seed42/area/metrics.jsonl |
| t_phys_c100k | area | 43 | 500 | 180 | ../../examples/cfd/isla/results/campaigns_2026-10-02/char/protocols/r80_hl_t_h512_l16_c100k_seed43/area/metrics.jsonl |
| t_phys_c100k | posed | 42 | 500 | 180 | ../../examples/cfd/isla/results/campaigns_2026-10-02/char/protocols/r80_hl_t_h512_l16_c100k_seed42/posed/metrics.jsonl |
| t_phys_c100k | posed | 43 | 500 | 180 | ../../examples/cfd/isla/results/campaigns_2026-10-02/char/protocols/r80_hl_t_h512_l16_c100k_seed43/posed/metrics.jsonl |
