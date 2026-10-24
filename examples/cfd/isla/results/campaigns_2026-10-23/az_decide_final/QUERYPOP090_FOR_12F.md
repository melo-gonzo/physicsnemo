# QUERY-POP-090 for PREREG_v2 §12(f) / §13(b) (from research-lane PROBES-090, scratch/isla-probes090-2026-10-21/)

Written 2026-10-21 02:05 EDT. Registration: scratch/isla-probes090-2026-10-21/PREREG.md (md5 4f188984, approved by the
supervisor 2026-10-20 23:59 EDT). Measurement only; no adoption rule. AeroZoo ISLA is added below (2026-10-21 23:00 EDT); ISLA-DTT follows ~10-23 (handoff, JOBS.md).

**What the numbers are.** For each case, the error on a fixed scored set (a seeded random half of the native points)
when the model is queried with a different population of prediction points, divided by the error with the native
population (r), per field, seed mean [seed 42, seed 43]. d = RMS(pred_mode − pred_native) / RMS(target), median over
cases. Modes: half = the scored set alone; chunks4 = the native points in 4 forwards; subK = 1,000 scored points queried
in groups of K (sub1 = one point per forward); same = native + 50% more points at the native law; far = native + as many
points uniform in the bounding box (outside the body); wall = native + as many resampled ∝ 1/|sdf|.
Labels (registered): roundoff |r−1| ≤ 1e-3 and d ≤ 1e-3 (ISLA-DTT check); material r > 1.10 on p or v, both seeds, in
half/chunks4/same; severe r > 2 on p or v, both seeds, in far/wall (and the subsets).

**Provenance.** Fully eager float32. AirfRANS rows (both forms, and GEO1) are the eager function of weights trained
under the compiled-path defect (scratch/isla-probes090-2026-10-21/qpop/COMPILED_PATH_EVIDENCE.md; fix b62a93f2c); a clean
re-read on the retrained REF-090 AirfRANS checkpoints follows as a registered follow-up. DrivAerML Volume is unaffected
(eager = its scorer to 1e-5). No 0.9.0 ISLA-DTT exists on Volume. AirfRANS: full_test 200 cases, 100,000 points;
Volume: 48 val cars, 200,000 points. LHR job 310635.

## Summary (seed means; pressure / velocity / nut)

AirfRANS rows: defect-trained weights; fix-trained (PREREG §8) and eager-trained (§9, the AirfRANS standing's checkpoints) re-reads pending.


| arm | half | chunks4 | sub1000 | sub1 | same (+50%) | far | wall |
|---|---|---|---|---|---|---|---|
| AirfRANS ISLA 0.9.0 (defect-trained weights) | 1.11 / 1.12 / 1.14 material | 1.33 / 1.36 / 1.40 material | 4.7 / 5.0 / 5.7 severe | 85 / 91 / 118 severe | 0.95 / 0.95 / 0.95 | 136 / 62 / 133 severe | 20 / 30 / 22 severe |
| AirfRANS ISLA-DTT 0.9.0 (defect-trained weights) | roundoff | roundoff | roundoff | roundoff | roundoff | roundoff | roundoff |
| AirfRANS GEO1 (pre-0.9.0, defect-trained weights) | 1.14 / 1.09 / 1.15 material | 1.44 / 1.30 / 1.40 material | 5.6 / 4.6 / 5.4 severe | 237 / 74 / 87 severe | 0.93 / 0.96 / 0.95 | 30 / 30 / 50 severe | 52 / 21 / 59 severe |
| DrivAerML Volume ISLA 0.9.0 | 1.01 / 1.01 / 1.02 | 1.03 / 1.03 / 1.15 | 4.5 / 4.0 / 5.5 severe | 21 / 13 / 18 severe | 1.00 / 1.00 / 1.00 | 20 / 16 / 14 severe | 6.2 / 4.5 / 4.5 severe |

ISLA-DTT: every |r − 1| ≤ 4.3e-6 (all keys, incl. velocity components) and median d 2.6e-7 to 8.2e-7 in all ten modes, both seeds.
The A2 rung of the ladder is "incomplete as registered" (its validity gate failed: an ill-posed sign of the wall distance at on-wall points, diag_a2/A2_FLIPS.md); no A2 row is reported.

## Full tables (per seed)

### r090_af_isla (2 seeds; query_independent=False; cases [200, 200])

| mode | pressure r [s42, s43] | velocity r [s42, s43] | nut r [s42, s43] | d pressure med | d velocity med | d nut med | label |
|---|---|---|---|---|---|---|---|
| repeat | 1 [1, 1] | 1 [1, 1] | 1 [1, 1] | 0, 0 | 0, 0 | 0, 0 | floor |
| half | 1.114 [1.122, 1.106] | 1.12 [1.124, 1.115] | 1.135 [1.139, 1.132] | 0.0051, 0.0041 | 0.0028, 0.0027 | 0.007, 0.0066 | MATERIAL |
| chunks4 | 1.333 [1.363, 1.303] | 1.361 [1.368, 1.354] | 1.401 [1.422, 1.38] | 0.01, 0.0081 | 0.0055, 0.0054 | 0.014, 0.012 | MATERIAL |
| sub1000 | 4.673 [4.915, 4.432] | 5.036 [5.144, 4.928] | 5.675 [5.845, 5.505] | 0.053, 0.043 | 0.029, 0.027 | 0.076, 0.069 | SEVERE |
| sub100 | 27.13 [20.67, 33.58] | 19.21 [19.21, 19.22] | 38.91 [33.39, 44.42] | 0.22, 0.24 | 0.12, 0.11 | 0.44, 0.44 | SEVERE |
| sub10 | 73.64 [58.47, 88.8] | 65.16 [64.99, 65.33] | 105 [84, 125.9] | 0.67, 0.88 | 0.41, 0.4 | 1.3, 1.8 | SEVERE |
| sub1 | 85.44 [74.79, 96.08] | 91.27 [86.38, 96.15] | 118.3 [91.72, 144.8] | 0.94, 1 | 0.55, 0.6 | 1.4, 2.1 | SEVERE |
| same | 0.9515 [0.9379, 0.9651] | 0.9474 [0.9479, 0.9469] | 0.9496 [0.946, 0.9532] | 0.0032, 0.0025 | 0.0016, 0.0015 | 0.0041, 0.0036 | not material |
| far | 135.7 [90.56, 180.9] | 61.88 [61.25, 62.51] | 132.9 [126.1, 139.7] | 0.84, 1.6 | 0.38, 0.39 | 2, 2.1 | SEVERE |
| wall | 19.67 [17.73, 21.61] | 29.77 [30.42, 29.12] | 22.07 [23.91, 20.22] | 0.2, 0.24 | 0.2, 0.18 | 0.37, 0.3 | SEVERE |

### r090_af_dtt (2 seeds; query_independent=True; cases [200, 200])

| mode | pressure r [s42, s43] | velocity r [s42, s43] | nut r [s42, s43] | d pressure med | d velocity med | d nut med | label |
|---|---|---|---|---|---|---|---|
| repeat | 1 [1, 1] | 1 [1, 1] | 1 [1, 1] | 5.1e-07, 4.3e-07 | 2.6e-07, 2.6e-07 | 2.9e-07, 3e-07 | roundoff |
| half | 1 [1, 1] | 1 [1, 1] | 1 [1, 1] | 5.1e-07, 4.3e-07 | 2.6e-07, 2.6e-07 | 2.9e-07, 3e-07 | roundoff |
| chunks4 | 1 [1, 1] | 1 [1, 1] | 1 [1, 1] | 5.9e-07, 4.9e-07 | 3e-07, 3e-07 | 3.6e-07, 4.3e-07 | roundoff |
| sub1000 | 1 [1, 1] | 1 [1, 1] | 1 [1, 1] | 8.1e-07, 6.2e-07 | 4.5e-07, 4.6e-07 | 5.5e-07, 5.7e-07 | roundoff |
| sub100 | 1 [1, 1] | 1 [1, 1] | 1 [1, 1] | 8e-07, 6.3e-07 | 4.5e-07, 4.6e-07 | 6.6e-07, 7.4e-07 | roundoff |
| sub10 | 1 [1, 1] | 1 [1, 1] | 1 [1, 1] | 8.2e-07, 6.3e-07 | 4.5e-07, 4.6e-07 | 6.9e-07, 8.2e-07 | roundoff |
| sub1 | 1 [1, 1] | 1 [1, 1] | 1 [1, 1] | 7.7e-07, 6.2e-07 | 4.5e-07, 4.6e-07 | 6.1e-07, 6.8e-07 | roundoff |
| same | 1 [1, 1] | 1 [1, 1] | 1 [1, 1] | 5.1e-07, 4.3e-07 | 2.6e-07, 2.6e-07 | 2.9e-07, 3e-07 | roundoff |
| far | 1 [1, 1] | 1 [1, 1] | 1 [1, 1] | 5.1e-07, 4.3e-07 | 2.6e-07, 2.6e-07 | 2.9e-07, 3e-07 | roundoff |
| wall | 1 [1, 1] | 1 [1, 1] | 1 [1, 1] | 5.1e-07, 4.3e-07 | 2.6e-07, 2.6e-07 | 2.9e-07, 3e-07 | roundoff |

### afu_geo1_isla (2 seeds; query_independent=False; cases [200, 200])

| mode | pressure r [s42, s43] | velocity r [s42, s43] | nut r [s42, s43] | d pressure med | d velocity med | d nut med | label |
|---|---|---|---|---|---|---|---|
| repeat | 1 [1, 1] | 1 [1, 1] | 1 [1, 1] | 0, 0 | 0, 0 | 0, 0 | floor |
| half | 1.141 [1.158, 1.124] | 1.09 [1.089, 1.09] | 1.154 [1.135, 1.174] | 0.0064, 0.0069 | 0.003, 0.0034 | 0.0088, 0.01 | MATERIAL |
| chunks4 | 1.44 [1.447, 1.432] | 1.303 [1.299, 1.308] | 1.4 [1.387, 1.413] | 0.013, 0.014 | 0.0061, 0.0072 | 0.016, 0.019 | MATERIAL |
| sub1000 | 5.603 [5.685, 5.521] | 4.568 [4.578, 4.558] | 5.407 [5.258, 5.555] | 0.07, 0.074 | 0.035, 0.036 | 0.093, 0.1 | SEVERE |
| sub100 | 25.68 [24.36, 27] | 15.52 [15.43, 15.61] | 22.74 [23, 22.47] | 0.33, 0.36 | 0.13, 0.13 | 0.44, 0.46 | SEVERE |
| sub10 | 156.3 [148.4, 164.1] | 47.05 [47.73, 46.36] | 81.39 [79.52, 83.26] | 2.2, 2.7 | 0.4, 0.42 | 1.5, 1.7 | SEVERE |
| sub1 | 236.9 [257.6, 216.2] | 74.04 [69.62, 78.47] | 86.75 [74.96, 98.55] | 4.4, 3.9 | 0.59, 0.71 | 1.4, 2.1 | SEVERE |
| same | 0.9311 [0.9283, 0.934] | 0.9625 [0.966, 0.9589] | 0.9528 [0.9533, 0.9522] | 0.0041, 0.0038 | 0.0018, 0.002 | 0.005, 0.0059 | not material |
| far | 29.93 [21.15, 38.71] | 30.13 [31.92, 28.34] | 50.24 [40.45, 60.03] | 0.31, 0.57 | 0.27, 0.25 | 0.8, 1.1 | SEVERE |
| wall | 51.65 [58.46, 44.84] | 20.9 [21.93, 19.88] | 59.4 [79.05, 39.74] | 0.85, 0.58 | 0.18, 0.18 | 1.5, 0.86 | SEVERE |

### r090_vol_isla (2 seeds; query_independent=False; cases [48, 48])

| mode | pressure r [s42, s43] | velocity r [s42, s43] | nut r [s42, s43] | d pressure med | d velocity med | d nut med | label |
|---|---|---|---|---|---|---|---|
| repeat | 1 [1, 1] | 1 [1, 1] | 1 [1, 1] | 9.3e-07, 9.7e-07 | 3.3e-06, 3.4e-06 | 1.5e-06, 1.6e-06 | floor |
| half | 1.011 [1.011, 1.011] | 1.01 [1.01, 1.01] | 1.02 [1.015, 1.025] | 0.0057, 0.0059 | 0.0087, 0.0089 | 0.011, 0.013 | not material |
| chunks4 | 1.034 [1.034, 1.034] | 1.034 [1.034, 1.033] | 1.154 [1.12, 1.188] | 0.01, 0.01 | 0.015, 0.015 | 0.025, 0.025 | not material |
| sub1000 | 4.462 [3.962, 4.962] | 4.012 [3.615, 4.408] | 5.452 [5.621, 5.283] | 0.13, 0.16 | 0.18, 0.23 | 0.33, 0.33 | SEVERE |
| sub100 | 13.18 [12.84, 13.52] | 9.79 [9.384, 10.2] | 11.69 [11.74, 11.64] | 0.49, 0.52 | 0.54, 0.59 | 0.76, 0.77 | SEVERE |
| sub10 | 17.84 [17.68, 18.01] | 13 [12.63, 13.37] | 14.57 [14.58, 14.57] | 0.67, 0.69 | 0.72, 0.77 | 0.95, 0.98 | SEVERE |
| sub1 | 20.6 [20.07, 21.12] | 13.17 [13.44, 12.91] | 17.68 [16.8, 18.57] | 0.73, 0.77 | 0.73, 0.72 | 1.2, 1.3 | SEVERE |
| same | 0.9963 [0.9965, 0.996] | 0.9957 [0.9956, 0.9958] | 1.003 [1.004, 1.001] | 0.0033, 0.0034 | 0.0051, 0.005 | 0.0064, 0.0072 | not material |
| far | 20.46 [19.07, 21.86] | 16.35 [16.12, 16.57] | 13.62 [14.66, 12.59] | 0.72, 0.85 | 0.93, 0.97 | 1.2, 1.1 | SEVERE |
| wall | 6.193 [3.826, 8.559] | 4.528 [2.668, 6.388] | 4.454 [2.251, 6.656] | 0.14, 0.33 | 0.14, 0.36 | 0.17, 0.56 | SEVERE |


## AeroZoo `full`, ISLA (added 2026-10-21 23:00 EDT; ISLA-DTT follows ~10-23 as a handoff)

LHR job 322430, qpop090.py v2 (CAP-AUDIT frame assertion: AeroZoo raw rows reproduce the native points exactly,
frame_err 0 on every case). **Subsample:** every 4th test case (179 of 716, deterministic), F = 25 single-point and
10-point groups. Validity: `full` against the §12(e) test-side record (epoch 250) max 3.1e-4 / 1.7e-4 (median ~1e-7),
179 / 179 cases, both seeds. Eager float32 (AeroZoo is not affected by the compiled-path defect).
Native error on the scored half (s42 / s43): pressure 0.312 / 0.289, velocity 0.123 / 0.118, nut 0.362 / 0.345, tke 0.555
/ 0.529, omega 0.355 / 0.361. These are large, so error ratios are compressed; read d beside r.

Summary (seed means, pressure / velocity / nut / tke / omega):
| mode | r | median d (p / v) | label | caveat |
|---|---|---|---|---|
| half | 1.00 / 1.01 / 1.00 / 1.00 / 1.00 | 0.012 / 0.003 | not material | 179 of 716 test cases; native p 0.30, v 0.12 (ratios compressed) |
| chunks4 | 1.00 / 1.01 / 1.00 / 1.00 / 1.00 | 0.021 / 0.006 | not material | 179 of 716 test cases; native p 0.30, v 0.12 (ratios compressed) |
| sub1000 | 1.09 / 1.05 / 1.04 / 1.06 / 1.04 | 0.13 / 0.04 | - (below 1.10) | 179 of 716 test cases; native p 0.30, v 0.12 (ratios compressed) |
| sub100 | 1.40 / 1.18 / 1.14 / 1.17 / 1.24 | 0.31 / 0.09 | material | 179 of 716 test cases; native p 0.30, v 0.12 (ratios compressed) |
| sub10 | 1.96 / 1.54 / 1.41 / 1.27 / 1.63 | 0.50 / 0.15 | material | 179 of 716 test cases; native p 0.30, v 0.12 (ratios compressed) |
| sub1 | 3.99 / 2.92 / 2.30 / 1.45 / 4.37 | 0.82 / 0.28 | severe | 179 of 716 test cases; native p 0.30, v 0.12 (ratios compressed) |
| same (+50%) | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 | 0.007 / 0.002 | not material | 179 of 716 test cases; native p 0.30, v 0.12 (ratios compressed) |
| far | 1.04 / 1.05 / 1.03 / 1.01 / 1.02 | 0.10 / 0.045 | not severe (below 1.10) | 179 of 716 test cases; native p 0.30, v 0.12 (ratios compressed) |
| wall | 1.44 / 1.24 / 1.30 / 1.21 / 1.62 | 0.30 / 0.10 | material, not severe | 179 of 716 test cases; native p 0.30, v 0.12 (ratios compressed) |

Reading: on AeroZoo, interacting ISLA's error barely moves under chunking, halving, same-law additions or far-field
companions (r ≤ 1.06). Its predictions still move: median d is 0.10 on pressure with far-field companions and 0.30 with
wall-heavy ones. Small queries raise the error 1.4 to 4.4 times (single points), and wall-heavy companions 1.2 to 1.6
times. Against AirfRANS (far 62-136×) and DrivAerML Volume (far 14-20×), AeroZoo's ratios are far smaller, in part
its native errors are much larger (pressure 0.30, against 0.013 on AirfRANS and 0.038 on DrivAerML
Volume).

Full table (per seed):

### azc_full_isla (2 seeds; query_independent=False; cases [179, 179])

| mode | pressure r [s42, s43] | velocity r [s42, s43] | nut r [s42, s43] | tke r [s42, s43] | omega r [s42, s43] | d pressure med | d velocity med | d nut med | d tke med | d omega med | label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| repeat | 1 [1, 1] | 1 [1, 1] | 1 [1, 1] | 1 [1, 1] | 1 [1, 1] | 0, 0 | 0, 0 | 0, 0 | 0, 0 | 0, 0 | floor |
| half | 1 [1, 1.001] | 1.009 [1.016, 1.001] | 1.001 [1.001, 1.001] | 1.001 [1.002, 1.001] | 1 [1, 1] | 0.013, 0.011 | 0.0037, 0.0031 | 0.0078, 0.0064 | 0.016, 0.013 | 0.009, 0.0082 | not material |
| chunks4 | 1.002 [1.002, 1.002] | 1.005 [1.009, 1.001] | 1 [1, 1] | 1 [1, 1] | 1.001 [1.001, 1] | 0.023, 0.019 | 0.0065, 0.0057 | 0.013, 0.012 | 0.031, 0.025 | 0.015, 0.014 | not material |
| sub1000 | 1.088 [1.086, 1.09] | 1.05 [1.062, 1.038] | 1.039 [1.035, 1.042] | 1.059 [1.051, 1.067] | 1.039 [1.031, 1.047] | 0.14, 0.12 | 0.04, 0.037 | 0.087, 0.083 | 0.18, 0.15 | 0.083, 0.086 | - |
| sub100 | 1.397 [1.382, 1.412] | 1.178 [1.179, 1.177] | 1.14 [1.148, 1.131] | 1.169 [1.188, 1.15] | 1.237 [1.234, 1.239] | 0.33, 0.29 | 0.09, 0.087 | 0.22, 0.2 | 0.41, 0.35 | 0.23, 0.23 | MATERIAL |
| sub10 | 1.957 [1.878, 2.035] | 1.537 [1.52, 1.553] | 1.41 [1.429, 1.392] | 1.267 [1.266, 1.269] | 1.633 [1.673, 1.592] | 0.49, 0.5 | 0.15, 0.14 | 0.38, 0.37 | 0.55, 0.54 | 0.41, 0.44 | MATERIAL |
| sub1 | 3.987 [3.585, 4.39] | 2.916 [2.696, 3.135] | 2.302 [2.164, 2.44] | 1.445 [1.368, 1.522] | 4.365 [3.86, 4.869] | 0.8, 0.83 | 0.27, 0.28 | 0.72, 0.75 | 0.8, 0.75 | 0.91, 0.9 | SEVERE |
| same | 0.9998 [0.9997, 0.9999] | 0.9997 [0.9992, 1] | 1 [1, 0.9999] | 0.9999 [0.9999, 1] | 1 [1, 1] | 0.008, 0.0058 | 0.0022, 0.0019 | 0.0043, 0.0035 | 0.0092, 0.0085 | 0.005, 0.0045 | not material |
| far | 1.037 [1.028, 1.046] | 1.053 [1.046, 1.06] | 1.029 [1.018, 1.04] | 1.014 [1.005, 1.023] | 1.019 [1.022, 1.016] | 0.1, 0.099 | 0.047, 0.043 | 0.11, 0.11 | 0.19, 0.18 | 0.079, 0.081 | - |
| wall | 1.44 [1.419, 1.461] | 1.24 [1.239, 1.241] | 1.3 [1.356, 1.244] | 1.212 [1.249, 1.175] | 1.619 [1.543, 1.696] | 0.28, 0.33 | 0.1, 0.099 | 0.29, 0.26 | 0.44, 0.4 | 0.38, 0.45 | material (r > 1.10) |

## AeroZoo ISLA-DTT (PROBES-090 item (d), run by AZ-DECIDE; 2026-10-23 22:20 EDT)

Job 361222 (short QoS, 2 tasks of about 1 h). Units `prod_az_dtt_eager.tsv` (md5 ac31c125), qpop090.py d282a4f3, eager,
every 4th full.test case (179 of 716), F = 25. Both seeds were scored first at epoch 250, 716 cases (scorer 360414).
Reduced with the ISLA rows already present in out_az (`reduce.py`, merged; local copy in `qpop_az/`). Labelled with
`isla-probes090-2026-10-21/analysis/label_qpop.py`.

**Validity gate.** The registered gate is "full vs scorer, max rel (head) ≤ 1e-3".
- It **fails for ISLA-DTT**: the maximum is 1.09e-1 (seed 42) and 1.40e-1 (seed 43). The median is 1.9e-7 / 2.3e-7.
- ISLA passes at 3.1e-4 / 1.7e-4.
- The failure comes from a few cases. The probe's own `repeat` mode, an identical second forward, also moves single
  ISLA-DTT cases by up to 2.6e-2 (velocity) and 9.6e-2 (nut). So ISLA-DTT's float32 forward is not reproducible run to
  run on a few AeroZoo cases. This matches the float32 floor found by F32FLOOR (`f32floor/README.md`): discrete geometry
  choices, namely the wall-distance sign at folds and the in-model Morton token draw, flip under roundoff. For ISLA-DTT
  the token draw dominates.
- **So the table below is descriptive, not a gated reading.**

**Reading, descriptive.**
- Case-mean ratios stay within 1.5e-3 of 1 in every mode on both seeds.
- The median per-case d is 2e-7 to 3e-5 in every mode, the same size as `half`/`chunks4` and `repeat`.
- `label_qpop.py` gives "roundoff" in 7 of 10 modes. It gives "NOT roundoff" in sub1000, same and wall, on nut ratios
  of 0.9985-0.9989. Those sit inside the `repeat` mode's own spread: nut r 1.001 / 0.999 for an identical re-forward.
- So ISLA-DTT's query independence holds on AeroZoo up to its own run-to-run nondeterminism. Interacting ISLA, on the
  same cases: sub100 1.18-1.40×, sub10 1.54-1.96×, sub1 2.9-4.0×, wall-heavy 1.24-1.44× (table above).

### azc_full_dtt (2 seeds; query_independent=True; cases [179, 179])

| mode | pressure r [s42, s43] | velocity r [s42, s43] | nut r [s42, s43] | tke r [s42, s43] | omega r [s42, s43] | d pressure med | d velocity med | d nut med | d tke med | d omega med | label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| repeat | 1 [1, 1] | 1 [0.9997, 1] | 0.9998 [1.001, 0.999] | 0.9999 [0.9999, 0.9998] | 0.9999 [0.9998, 1] | 4e-07, 3.6e-07 | 2.3e-07, 2.2e-07 | 2.9e-07, 2.6e-07 | 4.2e-07, 3.8e-07 | 3.5e-07, 3.4e-07 | roundoff |
| half | 1 [1, 1] | 1 [0.9997, 1] | 1 [1, 1] | 1 [0.9999, 1] | 1 [0.9998, 1] | 5.4e-06, 7.2e-06 | 4.5e-06, 5.4e-06 | 6e-06, 9.4e-06 | 1.2e-05, 1.6e-05 | 3.7e-06, 5.3e-06 | roundoff |
| chunks4 | 1 [1, 1] | 0.9998 [0.9997, 0.9999] | 1 [0.9998, 1] | 0.9999 [0.9998, 1] | 0.9999 [0.9998, 1] | 6.5e-06, 7.6e-06 | 5.3e-06, 6.6e-06 | 6.9e-06, 1.1e-05 | 1.3e-05, 1.8e-05 | 4.3e-06, 5.9e-06 | roundoff |
| sub1000 | 1 [1, 0.9999] | 0.9999 [0.9997, 1] | 0.9994 [0.9988, 1] | 0.9997 [0.9994, 1] | 0.9999 [0.9998, 1] | 6.1e-06, 8e-06 | 4.9e-06, 6.6e-06 | 6.7e-06, 9.9e-06 | 1.2e-05, 1.8e-05 | 4.3e-06, 5.9e-06 | NOT roundoff |
| sub100 | 1 [1, 0.9999] | 0.9999 [0.9998, 1] | 1 [1, 1] | 1 [0.9999, 1] | 0.9999 [0.9998, 1] | 6.4e-06, 8.3e-06 | 4.8e-06, 7.4e-06 | 7e-06, 1.2e-05 | 1.1e-05, 1.9e-05 | 4.7e-06, 6.4e-06 | roundoff |
| sub10 | 1 [1, 0.9998] | 0.9999 [0.9997, 1] | 0.9999 [0.9998, 1] | 0.9999 [0.9998, 1] | 1 [0.9999, 1] | 5.5e-06, 7.6e-06 | 4.6e-06, 6.4e-06 | 6.4e-06, 1.1e-05 | 1e-05, 1.8e-05 | 4.9e-06, 6.9e-06 | roundoff |
| sub1 | 1 [1, 0.9999] | 0.9999 [0.9997, 1] | 1 [1, 1] | 0.9999 [0.9998, 1] | 0.9999 [0.9999, 0.9999] | 6.6e-06, 7.7e-06 | 4.1e-06, 6.1e-06 | 5.1e-06, 8.6e-06 | 9.3e-06, 1.7e-05 | 8.4e-06, 1.2e-05 | roundoff |
| same | 1 [1, 1] | 1 [1, 1] | 0.9994 [0.9985, 1] | 0.9998 [0.9996, 1] | 1 [1, 1] | 6.1e-06, 6.5e-06 | 5.1e-06, 8.3e-06 | 6.2e-06, 1.2e-05 | 1.3e-05, 1.7e-05 | 4.3e-06, 6.4e-06 | NOT roundoff |
| far | 1 [0.9999, 1] | 0.9998 [1, 0.9995] | 1 [1, 1] | 1 [1, 1] | 1 [1, 1] | 1.2e-05, 1.2e-05 | 7.5e-06, 9.7e-06 | 1e-05, 1.9e-05 | 2.1e-05, 3e-05 | 7.1e-06, 1e-05 | roundoff |
| wall | 1 [1, 0.9999] | 0.9999 [0.9997, 1] | 0.9997 [1, 0.9989] | 0.9998 [0.9999, 0.9997] | 0.9999 [0.9998, 1] | 1.2e-05, 1.3e-05 | 7.5e-06, 9.4e-06 | 1.1e-05, 1.7e-05 | 2.1e-05, 2.9e-05 | 7.2e-06, 9.9e-06 | NOT roundoff |
