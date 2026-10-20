# v2 checks: 22 cases, one per AeroZoo test split (LHR cpu-short job 285163, final YAML; first run 285027/285028 in run1/)

## Leak tests (bitwise, every non-label leaf the pipeline hands over)

- v2 pipeline: perturbed labels 0 differing leaves, deleted labels 0 differing leaves, of 2550 leaves.
- v2_isla pipeline: perturbed labels 0 differing leaves, deleted labels 0 differing leaves, of 1450 leaves.

## ISLA-internal scales (prototype of the design, float64)

- Pose (2 rigid motions per case): max rel change L 2.2e-16, U 4.4e-16, Re 4.4e-16; |g - R g0| 3.4e-16; nondimensional scalar inputs 3.6e-15 (abs).
- Similarity s3.7_c0.29: max |L'/(sL) - 1| 2.2e-16, |U'/(cU) - 1| 2.2e-16, |dlog Re| 2.4e-16, nondimensional inputs max rel 6.9e-16, nondimensional targets max rel velocity 1.2e-07, pressure 9.8e-05, nut 2.3e-06, tke 4.5e-05, omega 7.2e-07
- Similarity s0.05_c12.0: max |L'/(sL) - 1| 2.2e-16, |U'/(cU) - 1| 3.3e-16, |dlog Re| 1.9e-16, nondimensional inputs max rel 5.2e-16, nondimensional targets max rel velocity 1.1e-07, pressure 7.8e-06, nut 4.7e-06, tke 3.5e-06, omega 1.2e-07
- U (area-RMS inlet speed) / U_ref: one-inlet cases (20) max |.-1| 5.7e-08; layout: U/U_ref 1.210, |mean v|/U 0.779; profile_dual_inlet: U/U_ref 0.764, |mean v|/U 0.682
- log10 Re (ISLA, gauge x RMS inlet speed / nu) vs log10 Re_ref: corr 0.993 (R^2 0.986); offset +0.58 +- 0.19; ISLA log Re mean 6.09, std 1.56.

## Score map (D): v2 physical targets -> label-era space vs the label-era pipeline, max rel error

- max over cases: velocity 2.1e-07, pressure 4.2e-06, nut 4.1e-06, tke 5.0e-06, omega 2.2e-07

## Physical targets: how one per-dataset standardization weights the cases

- velocity: per-case mean square spans 7.80e-02 to 6.51e+02 (x8.3e+03); the 3 largest of 22 cases carry 67% of the summed square (bank_blender_streamlined, bank_blender_bluff, bank_cars); the 11 smallest carry 5.2e-03.
- pressure: per-case mean square spans 1.19e-03 to 4.54e+06 (x3.8e+09); the 3 largest of 22 cases carry 100% of the summed square (reynolds, bank_streamlined, bank_cars); the 11 smallest carry 2.8e-06.

## Raw coordinates and GeoTransolver's fixed radii [0.01, 0.05, 0.25, 1, 2.5, 5] m, caps [8, 16, 32, 32, 64, 128]

- Interior extent (max axis): 0.11 to 125.5 m (median 6.2); max |x| up to 62.8 m; |mean x| median 0.96 m.

| split | extent m | geometry | r=0.01 m: median n, empty, at cap | r=0.25 m: median n, empty, at cap | r=1.0 m: median n, empty, at cap | r=5.0 m: median n, empty, at cap |
|---|---|---|---|---|---|---|
| reynolds_interior | 0.1 | interior | 5710, 0.00, 1.00 | 199990, 0.00, 1.00 | 199990, 0.00, 1.00 | 199990, 0.00, 1.00 |
| reynolds_interior | 0.1 | boundary | 1319, 0.23, 0.77 | 163545, 0.00, 1.00 | 163545, 0.00, 1.00 | 163545, 0.00, 1.00 |
| bank_streamlined | 0.4 | interior | 80, 0.00, 1.00 | 199160, 0.00, 1.00 | 199990, 0.00, 1.00 | 199990, 0.00, 1.00 |
| bank_streamlined | 0.4 | boundary | 0, 0.72, 0.28 | 178240, 0.00, 1.00 | 184476, 0.00, 1.00 | 184476, 0.00, 1.00 |
| profile_ground_effect | 0.4 | interior | 210, 0.01, 0.99 | 198470, 0.00, 1.00 | 199990, 0.00, 1.00 | 199990, 0.00, 1.00 |
| profile_ground_effect | 0.4 | boundary | 0, 0.67, 0.33 | 167938, 0.00, 1.00 | 178471, 0.00, 1.00 | 178471, 0.00, 1.00 |
| profile_open_opposed | 0.5 | interior | 30, 0.06, 0.94 | 148140, 0.00, 1.00 | 199990, 0.00, 1.00 | 199990, 0.00, 1.00 |
| profile_open_opposed | 0.5 | boundary | 0, 0.79, 0.19 | 65181, 0.00, 1.00 | 115113, 0.00, 1.00 | 115113, 0.00, 1.00 |
| bank_blobs | 0.8 | interior | 20, 0.20, 0.80 | 28860, 0.00, 1.00 | 199990, 0.00, 1.00 | 199990, 0.00, 1.00 |
| bank_blobs | 0.8 | boundary | 23, 0.22, 0.76 | 22205, 0.00, 1.00 | 175808, 0.00, 1.00 | 175808, 0.00, 1.00 |
| bank_cars | 1.2 | interior | 0, 0.71, 0.29 | 31190, 0.00, 1.00 | 199710, 0.00, 1.00 | 199990, 0.00, 1.00 |
| bank_cars | 1.2 | boundary | 0, 0.89, 0.07 | 17003, 0.01, 0.99 | 174825, 0.00, 1.00 | 177998, 0.00, 1.00 |
| layout | 1.9 | interior | 0, 0.89, 0.11 | 12680, 0.00, 1.00 | 172220, 0.00, 1.00 | 199990, 0.00, 1.00 |
| layout | 1.9 | boundary | 0, 0.93, 0.04 | 4637, 0.12, 0.88 | 137186, 0.00, 1.00 | 200000, 0.00, 1.00 |
| family_duct | 2.4 | interior | 30, 0.16, 0.84 | 30310, 0.00, 1.00 | 131760, 0.00, 1.00 | 199990, 0.00, 1.00 |
| family_duct | 2.4 | boundary | 17, 0.15, 0.81 | 16071, 0.00, 1.00 | 90154, 0.00, 1.00 | 151508, 0.00, 1.00 |
| scarce | 4.0 | interior | 0, 0.85, 0.15 | 3670, 0.00, 1.00 | 69170, 0.00, 1.00 | 199990, 0.00, 1.00 |
| scarce | 4.0 | boundary | 0, 0.77, 0.02 | 948, 0.00, 1.00 | 20627, 0.00, 1.00 | 96766, 0.00, 1.00 |
| bank_blender_bluff | 5.5 | interior | 0, 1.00, 0.00 | 340, 0.00, 0.94 | 6680, 0.00, 1.00 | 193120, 0.00, 1.00 |
| bank_blender_bluff | 5.5 | boundary | 0, 0.95, 0.00 | 216, 0.17, 0.83 | 3647, 0.04, 0.96 | 110832, 0.00, 1.00 |
| bank_zero_to_cad | 6.2 | interior | 0, 0.98, 0.02 | 250, 0.00, 0.98 | 15750, 0.00, 1.00 | 198300, 0.00, 1.00 |
| bank_zero_to_cad | 6.2 | boundary | 0, 0.88, 0.00 | 60, 0.46, 0.52 | 6229, 0.02, 0.98 | 161702, 0.00, 1.00 |
| super_scarce | 6.2 | interior | 0, 1.00, 0.00 | 400, 0.00, 0.97 | 19230, 0.00, 1.00 | 199990, 0.00, 1.00 |
| super_scarce | 6.2 | boundary | 0, 0.97, 0.00 | 0, 0.71, 0.28 | 8686, 0.02, 0.97 | 188556, 0.00, 1.00 |
| bank_blender_streamlined | 6.7 | interior | 0, 1.00, 0.00 | 270, 0.00, 0.99 | 19450, 0.00, 1.00 | 199980, 0.00, 1.00 |
| bank_blender_streamlined | 6.7 | boundary | 0, 1.00, 0.00 | 0, 0.75, 0.23 | 2604, 0.16, 0.84 | 117531, 0.00, 1.00 |
| profile_crossflow | 7.6 | interior | 0, 1.00, 0.00 | 210, 0.01, 0.90 | 10430, 0.00, 1.00 | 192170, 0.00, 1.00 |
| profile_crossflow | 7.6 | boundary | 0, 0.96, 0.00 | 0, 0.61, 0.37 | 3363, 0.14, 0.86 | 138594, 0.00, 1.00 |
| family_opposed_hemisphere | 14.6 | interior | 0, 1.00, 0.00 | 50, 0.13, 0.68 | 1110, 0.00, 1.00 | 37340, 0.00, 1.00 |
| family_opposed_hemisphere | 14.6 | boundary | 0, 0.99, 0.00 | 51, 0.18, 0.75 | 930, 0.09, 0.91 | 26669, 0.00, 1.00 |
| bank_airframes | 14.8 | interior | 0, 1.00, 0.00 | 40, 0.11, 0.62 | 940, 0.00, 1.00 | 31000, 0.00, 1.00 |
| bank_airframes | 14.8 | boundary | 0, 1.00, 0.00 | 43, 0.16, 0.75 | 789, 0.07, 0.93 | 22456, 0.00, 1.00 |
| profile_dual_inlet | 16.1 | interior | 0, 1.00, 0.00 | 20, 0.15, 0.21 | 1380, 0.00, 1.00 | 81080, 0.00, 1.00 |
| profile_dual_inlet | 16.1 | boundary | 0, 1.00, 0.00 | 0, 0.85, 0.11 | 3, 0.50, 0.50 | 60928, 0.00, 1.00 |
| profile_confined_opposed | 30.2 | interior | 0, 1.00, 0.00 | 0, 0.67, 0.00 | 230, 0.00, 0.97 | 19450, 0.00, 1.00 |
| profile_confined_opposed | 30.2 | boundary | 0, 1.00, 0.00 | 0, 0.89, 0.06 | 0, 0.69, 0.26 | 8761, 0.00, 0.99 |
| full | 31.2 | interior | 0, 1.00, 0.00 | 0, 0.69, 0.03 | 90, 0.00, 0.91 | 10160, 0.00, 1.00 |
| full | 31.2 | boundary | 0, 1.00, 0.00 | 0, 0.86, 0.04 | 0, 0.71, 0.28 | 6445, 0.04, 0.95 |
| reynolds | 64.1 | interior | 0, 1.00, 0.00 | 0, 0.98, 0.00 | 10, 0.42, 0.22 | 680, 0.00, 1.00 |
| reynolds | 64.1 | boundary | 0, 1.00, 0.00 | 0, 0.96, 0.01 | 0, 0.88, 0.05 | 0, 0.52, 0.43 |
| family_opposed_sphere | 95.8 | interior | 0, 1.00, 0.00 | 0, 0.98, 0.00 | 10, 0.40, 0.05 | 430, 0.00, 0.85 |
| family_opposed_sphere | 95.8 | boundary | 0, 1.00, 0.00 | 0, 0.88, 0.00 | 8, 0.29, 0.02 | 273, 0.16, 0.83 |
| family_non_box | 125.5 | interior | 0, 1.00, 0.00 | 0, 0.99, 0.00 | 0, 0.61, 0.01 | 230, 0.02, 0.79 |
| family_non_box | 125.5 | boundary | 0, 1.00, 0.00 | 0, 0.93, 0.00 | 3, 0.35, 0.01 | 149, 0.20, 0.71 |

Interior counts are from a 20,000-point draw scaled by 10 (the lanes draw 200,000); boundary counts use the 4 x 50,000-face draw.
