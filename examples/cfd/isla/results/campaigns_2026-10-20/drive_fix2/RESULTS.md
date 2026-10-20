# DRIVE-FIX2 results (unit-direction baseline reruns; reduced 2026-10-19 19:56 EDT)

Float32, final checkpoint (epoch 500), scored on ISLA's 200k-cell sets with the CHAR protocol. Values are seed means,
with seeds 42 and 43 in parentheses. The physical-drive references are the book's REF-80 / CHAR rows.

| lane | native | unit ÷ physical drive | ÷ ISLA A2 | posed |
|---|---|---|---|---|
| GeoTransolver 256, DrivAerML Surface (r80ud_dr_gt_h256) | p 0.0455 (0.0454, 0.0456), wss 0.0696 (0.0696, 0.0697) | 1.000 / 1.002 | 1.023 / 0.995 | 1.26 / 1.27 |
| Transolver, DrivAerML Volume (clud_vol_t) | p 0.0495 (0.0495, 0.0495), v 0.1123 (0.1138, 0.1108), nut 0.0881 (0.0881, 0.0881) | 0.942 / 0.898 / 0.986 | 1.210 / 1.701 / 1.000 | 0.77 / 1.27 / 1.08 |

- GeoTransolver does not respond to the drive's scale. Transolver gains 6% on pressure and 10% on velocity, as on
  DrivAerML Surface (R2, 3%).
- Area: GeoTransolver on Surface reads 0.377 / 0.432. For Transolver on Volume the area protocol equals native exactly,
  which suggests this scorer's recipe has no separate area set for DrivAerML Volume. Check before quoting a Volume area
  ratio.
- Still training: GeoTransolver on DrivAerML Volume and SHIFT-SUV, and Transolver on SHIFT-SUV (finish overnight).

## SHIFT-SUV (reduced 2026-10-20 02:00 EDT; scorer 296934)

Unit-direction reruns against their constant-drive CHAR predecessors (cl_r_suv_gt, cl_r_suv_t). Native seed means
(seeds 42, 43), against ISLA A2's 0.0456 / 0.0677.

| lane | native p / wss | unit ÷ constant drive | ÷ ISLA A2 | area | posed |
|---|---|---|---|---|---|
| GeoTransolver 256 (clud_suv_gt) | 0.0455 (0.0458, 0.0452) / 0.0665 (0.0667, 0.0663) | 0.990 / 0.991 | 0.999 / 0.982 | 0.136 / 0.180 | 1.06 / 1.17 |
| Transolver 512×16 (clud_suv_t) | 0.0419 (0.0419, 0.0420) / 0.0640 (0.0639, 0.0640) | 0.988 / 0.988 | 0.919 / 0.945 | 0.131 / 0.170 | 1.08 / 1.18 |

- The unit direction changes both baselines' native error by about 1 percent on SHIFT-SUV.
- Transolver's new rows are scored on ISLA's 200k set (T-SCORESET). The CHAR predecessor used Transolver's own cell
  count, so its area and posed values (0.120 / 0.145 and 0.98 / 1.10) are not comparable with the new ones. Compare
  area and posed only between arms on the same 200k set.
- Remaining: GeoTransolver on DrivAerML Volume (clud_vol_gt), still training.

## DrivAerML Volume, GeoTransolver (reduced 2026-10-20 05:38 EDT; scorer 299613)

| lane | native p / v / nut | unit ÷ constant drive | ÷ ISLA A2 (0.0409 / 0.0660 / 0.0881) | posed |
|---|---|---|---|---|
| GeoTransolver 256 (clud_vol_gt) | 0.0481 (0.0483, 0.0479) / 0.1000 (0.0989, 0.1010) / 0.0845 (0.0840, 0.0850) | 0.988 / 0.997 / 1.009 | 1.175 / 1.514 / 0.959 | 0.82 / 1.34 / 1.06 |

Area equals native for this scorer on DrivAerML Volume, as for Transolver.

## Summary of the ten reruns (native, unit ÷ constant drive; then ÷ ISLA A2)

| dataset | GeoTransolver | Transolver |
|---|---|---|
| DrivAerML Surface (p / wss) | 1.000 / 1.002; 1.023 / 0.995 | R2 at 200k: 0.973 / 0.974; 0.951 / 0.969 |
| DrivAerML Volume (p / v / nut) | 0.988 / 0.997 / 1.009; 1.175 / 1.514 / 0.959 | 0.942 / 0.898 / 0.986; 1.210 / 1.701 / 1.000 |
| SHIFT-SUV (p / wss) | 0.990 / 0.991; 0.999 / 0.982 | 0.988 / 0.988; 0.919 / 0.945 |

The unit direction moves GeoTransolver by at most about 1 percent on every constant-drive dataset. It lowers Transolver's
error by 1 to 10 percent, the most on DrivAerML Volume.
