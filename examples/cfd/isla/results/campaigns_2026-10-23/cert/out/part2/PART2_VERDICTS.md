# CERT Part 2 verdicts (summarize_part2.py; null-calibrated, one-sided)

Files: af_p2.jsonl, az_p2.jsonl, djb_p2.jsonl, drs_p2.jsonl, hl_p2.jsonl, lug_p2.jsonl, suv_p2.jsonl, t2d_p2.jsonl, vol_p2.jsonl, af_p3.jsonl, az_p3.jsonl, djb_p3.jsonl, drs_p3.jsonl, hl_p3.jsonl, lug_p3.jsonl, suv_p3.jsonl, t2d_p3.jsonl, vol_p3.jsonl

Null: 2000 resamples, gate at the 95th percentile. Pooled groups: 4 roles, 2 cases, l2.

| family | G1' fp32: compiled max over processes vs joint thresholds | G2' bf16: N (cN, null median) / R (cR, null median) | G3' float64 failing groups | verdict |
|---|---|---|---|---|
| af | max over processes N1 3 (cN 5) / R1 2.81 (cR 3.04); level 95.00 | 3.0 (1.0, 0.0) / 1.030 (1.003, 1.000) | 0 (max d(comp64) 7.2e-13) | **not certified** |
| az | max over processes N1 2 (cN 4) / R1 1.45 (cR 2.16); level 95.00 | 1.0 (1.0, 0.0) / 1.016 (1.004, 1.000) | 0 (max d(comp64) 3.8e-14) | **not certified** |
| djb | max over processes N1 1 (cN 4) / R1 1.49 (cR 2.24); level 95.00 | 0.0 (1.0, 0.0) / 0.987 (1.000, 1.000) | 0 (max d(comp64) 8.0e-15) | **certified** |
| drs | max over processes N1 3 (cN 2) / R1 1.70 (cR 1.51); level 95.00 | 2.0 (1.0, 0.0) / 1.165 (1.000, 1.000) | 0 (max d(comp64) 5.8e-14) | **not certified** |
| hl | max over processes N1 2 (cN 3) / R1 2.64 (cR 2.32); level 95.00 | 2.0 (1.0, 0.0) / 1.331 (1.000, 0.998) | 0 (max d(comp64) 4.4e-14) | **not certified** |
| lug | max over processes N1 0 (cN 19) / R1 0.55 (cR 2.89); level 95.00 | 1.0 (1.0, 0.0) / 2.005 (1.005, 1.000) | 0 (max d(comp64) 7.5e-15) | **not certified** |
| suv | max over processes N1 1 (cN 4) / R1 1.56 (cR 2.87); level 95.00 | 8.0 (1.0, 0.0) / 1.257 (1.000, 0.999) | 3 (max d(comp64) 3.3e-12) | **not certified** |
| t2d | max over processes N1 3 (cN 8) / R1 1.53 (cR 2.61); level 95.00 | 7.0 (1.0, 0.0) / 11.442 (1.002, 1.000) | 2 (max d(comp64) 1.5e-12) | **not certified** |
| vol | max over processes N1 4 (cN 3) / R1 6.45 (cR 3.45); level 95.00 | 12.0 (1.0, 0.0) / 1.603 (1.009, 1.000) | 0 (max d(comp64) 1.3e-13) | **not certified** |

## af

- checkpoint /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/runs/afwhole_af_dtt_seed42/checkpoints/ISLA.0.500.mdlus (epoch 500), cases ['00009_airFoil2D_SST_31.812_1.334_0.371_3.287_0.0_19.548_airFoil2D_SST_31.812_1.334_0.371_3.287_0.0_19.548', '00010_airFoil2D_SST_31.863_-1.58_1.567_6.956_10.028_airFoil2D_SST_31.863_-1.58_1.567_6.956_10.028'], draws c/e 8/16, processes ['p2', 'p3'], torch 2.13.0+cu130; samples (compiled, eager) bf16 ((192, 16), (192, 32))
- G1' fp32 (gate): pass; null over 256 eager stand-in pairs, joint level 95.00 (null pass 1.000): cN1 5, cR1 3.04; p2 N1 3 R1 2.81; p3 N1 3 R1 2.54
- G2' bf16 (gate): FAIL; groups 192; compiled N 3.0, R 1.030; cN 1.0 (null median 0.0), cR 1.003 (median 1.000); joint level 95.10 (null pass 0.951); largest compiled ratios: fwd11.routing c1 1.027, fwd5.routing c0 1.015, fwd5.routing c1 1.014, fwd10.in c0 0.999, fwd11.in c1 0.999, fwd11.in c0 0.999
- G3' float64 (gate): 0 failing; max d(comp64, ref) 7.21e-13
- beside: G1' bf16 pass (cN1 7, cR1 5.04; p2 0 / 0.55; p3 0 / 0.77); G2' fp32 pass (N 1.0 vs q95 1.0, R 1.000 vs q95 1.002)
- beside, per tensor bf16 (max-relative, 1152 groups): N 5.0 (null q95 3.0), R 1.057 (null q95 1.100); outside the null

## az

- checkpoint /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/runs/azc_full_dtt_seed42/checkpoints/ISLA.0.250.mdlus (epoch 250), cases ['00000_case-0000000000000000000-a48a547f8d4c_domain', '00005_case-0000000000000000007-c38807c2d829_domain'], draws c/e 8/16, processes ['p2', 'p3'], torch 2.13.0+cu130; samples (compiled, eager) bf16 ((192, 16), (192, 32))
- G1' fp32 (gate): pass; null over 256 eager stand-in pairs, joint level 95.00 (null pass 1.000): cN1 4, cR1 2.16; p2 N1 2 R1 1.45; p3 N1 2 R1 1.42
- G2' bf16 (gate): FAIL; groups 192; compiled N 1.0, R 1.016; cN 1.0 (null median 0.0), cR 1.004 (median 1.000); joint level 95.25 (null pass 0.950); largest compiled ratios: fwd9.params c1 1.016, fwd10.in c0 1.000, fwd11.in c0 1.000, fwd9.in c0 1.000, fwd8.in c0 0.999, fwd11.in c1 0.999
- G3' float64 (gate): 0 failing; max d(comp64, ref) 3.76e-14
- beside: G1' bf16 pass (cN1 11, cR1 5.63; p2 0 / 0.30; p3 0 / 0.27); G2' fp32 fail (N 12.0 vs q95 1.0, R 1.022 vs q95 1.002)
- beside, per tensor bf16 (max-relative, 1152 groups): N 19.0 (null q95 3.0), R 2.857 (null q95 1.132); outside the null

## djb

- checkpoint /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/runs/r090_jb_isla_seed42/checkpoints/ISLA.0.40.mdlus (epoch 40), cases ['00076_012-079-IS02_dia_012-079-IS02_dia', '00077_012-079-IS02_hor_012-079-IS02_hor'], draws c/e 8/16, processes ['p2', 'p3'], torch 2.13.0+cu130; samples (compiled, eager) bf16 ((96, 16), (96, 32))
- G1' fp32 (gate): pass; null over 256 eager stand-in pairs, joint level 95.00 (null pass 1.000): cN1 4, cR1 2.24; p2 N1 1 R1 1.49; p3 N1 1 R1 1.34
- G2' bf16 (gate): pass; groups 96; compiled N 0.0, R 0.987; cN 1.0 (null median 0.0), cR 1.000 (median 1.000); joint level 95.00 (null pass 0.950); largest compiled ratios: sfwd11.in c0 0.986, sfwd11.in c1 0.984, sfwd6.in c0 0.982, sfwd6.in c1 0.982, sfwd7.in c1 0.982, sfwd4.in c1 0.982
- G3' float64 (gate): 0 failing; max d(comp64, ref) 8.02e-15
- beside: G1' bf16 pass (cN1 2, cR1 2.47; p2 0 / 0.27; p3 0 / 0.29); G2' fp32 fail (N 1.0 vs q95 1.0, R 1.077 vs q95 1.000)
- beside, per tensor bf16 (max-relative, 600 groups): N 1.0 (null q95 2.0), R 1.917 (null q95 1.077); outside the null

## drs

- checkpoint /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/runs/slicenormalx_dra_ctl_seed42/checkpoints/ISLA.0.500.mdlus (epoch 500), cases ['00015_run_112_domain_run_112', '00018_run_115_domain_run_115'], draws c/e 8/16, processes ['p2', 'p3'], torch 2.13.0+cu130; samples (compiled, eager) bf16 ((96, 16), (96, 32))
- G1' fp32 (gate): FAIL; null over 256 eager stand-in pairs, joint level 95.00 (null pass 1.000): cN1 2, cR1 1.51; p2 N1 3 R1 1.70; p3 N1 3 R1 1.69
- G2' bf16 (gate): FAIL; groups 96; compiled N 2.0, R 1.165; cN 1.0 (null median 0.0), cR 1.000 (median 1.000); joint level 95.00 (null pass 0.951); largest compiled ratios: sfwd0.routing c1 1.165, sfwd0.routing c0 1.098, sfwd2.in c0 0.996, sfwd2.in c1 0.996, sfwd3.in c1 0.995, sfwd3.in c0 0.994
- G3' float64 (gate): 0 failing; max d(comp64, ref) 5.82e-14
- beside: G1' bf16 pass (cN1 1, cR1 1.57; p2 0 / 0.29; p3 0 / 0.28); G2' fp32 fail (N 44.0 vs q95 1.0, R 1.086 vs q95 1.000)
- beside, per tensor bf16 (max-relative, 600 groups): N 19.0 (null q95 2.0), R 2.560 (null q95 1.056); outside the null

## hl

- checkpoint /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/runs/hldupx_hl_isla_seed42/checkpoints/ISLA.0.500.mdlus (epoch 500), cases ['00002_geo_LHC001_AoA_16_domain_LHC001_AoA_16', '00008_geo_LHC002_AoA_18_domain_LHC002_AoA_18'], draws c/e 8/16, processes ['p2', 'p3'], torch 2.13.0+cu130; samples (compiled, eager) bf16 ((96, 16), (96, 32))
- G1' fp32 (gate): FAIL; null over 256 eager stand-in pairs, joint level 95.00 (null pass 1.000): cN1 3, cR1 2.32; p2 N1 2 R1 2.29; p3 N1 2 R1 2.64
- G2' bf16 (gate): FAIL; groups 96; compiled N 2.0, R 1.331; cN 1.0 (null median 0.0), cR 1.000 (median 0.998); joint level 95.00 (null pass 0.952); largest compiled ratios: sfwd0.routing c0 1.331, sfwd0.routing c1 1.250, sfwd3.in c1 0.992, sfwd4.in c0 0.992, sfwd3.in c0 0.991, sfwd4.in c1 0.990
- G3' float64 (gate): 0 failing; max d(comp64, ref) 4.37e-14
- beside: G1' bf16 pass (cN1 2, cR1 3.26; p2 1 / 1.02; p3 0 / 0.99); G2' fp32 fail (N 16.0 vs q95 1.0, R 1.087 vs q95 1.000)
- beside, per tensor bf16 (max-relative, 600 groups): N 16.0 (null q95 2.0), R 2.607 (null q95 1.058); outside the null

## lug

- checkpoint /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/runs/r090fix_lug_dtt_seed42/checkpoints/ISLA.0.500.mdlus (epoch 500), cases ['00004_lug_0004_lug_0004', '00011_lug_0011_lug_0011'], draws c/e 8/16, processes ['p2', 'p3'], torch 2.13.0+cu130; samples (compiled, eager) bf16 ((192, 16), (192, 32))
- G1' fp32 (gate): pass; null over 256 eager stand-in pairs, joint level 95.00 (null pass 1.000): cN1 19, cR1 2.89; p2 N1 0 R1 0.55; p3 N1 0 R1 0.54
- G2' bf16 (gate): FAIL; groups 192; compiled N 1.0, R 2.005; cN 1.0 (null median 0.0), cR 1.005 (median 1.000); joint level 95.40 (null pass 0.950); largest compiled ratios: fwd2.routing c1 2.005, fwd2.in c0 0.998, fwd3.in c1 0.998, fwd2.in c1 0.998, fwd3.in c0 0.996, fwd1.in c0 0.996
- G3' float64 (gate): 0 failing; max d(comp64, ref) 7.53e-15
- beside: G1' bf16 pass (cN1 23, cR1 5.14; p2 1 / 2.00; p3 1 / 2.09); G2' fp32 fail (N 3.0 vs q95 1.0, R 1.023 vs q95 1.002)
- beside, per tensor bf16 (max-relative, 1152 groups): N 11.0 (null q95 3.0), R 2.977 (null q95 1.088); outside the null

## suv

- checkpoint /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/runs/r090_suv_isla_seed42/checkpoints/ISLA.0.500.mdlus (epoch 500), cases ['00021_run_00023_domain', '00039_run_00041_domain'], draws c/e 8/16, processes ['p2', 'p3'], torch 2.13.0+cu130; samples (compiled, eager) bf16 ((96, 16), (96, 32))
- G1' fp32 (gate): pass; null over 256 eager stand-in pairs, joint level 95.00 (null pass 1.000): cN1 4, cR1 2.87; p2 N1 1 R1 1.56; p3 N1 1 R1 1.50
- G2' bf16 (gate): FAIL; groups 96; compiled N 8.0, R 1.257; cN 1.0 (null median 0.0), cR 1.000 (median 0.999); joint level 95.00 (null pass 0.951); largest compiled ratios: sfwd2.routing c0 1.257, sfwd0.routing c1 1.197, sfwd2.routing c1 1.120, sfwd0.in c1 1.077, sfwd8.routing c0 1.052, sfwd0.in c0 1.049
- G3' float64 (gate): 3 failing; p2 sfwd9.routing c0 1.85e-12 > 1.00e-12; p3 sfwd9.routing c0 1.20e-12 > 1.00e-12; p3 sfwd0.routing c1 3.33e-12 > 2.09e-12; max d(comp64, ref) 3.33e-12
- beside: G1' bf16 pass (cN1 4, cR1 7.98; p2 1 / 1.24; p3 1 / 1.07); G2' fp32 fail (N 12.0 vs q95 1.0, R 1.028 vs q95 1.000)
- beside, per tensor bf16 (max-relative, 600 groups): N 27.0 (null q95 2.0), R 3.020 (null q95 1.061); outside the null

## t2d

- checkpoint /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/runs/r090fix_t2dz_isla_seed42/checkpoints/ISLA.0.500.mdlus (epoch 500), cases ['00001_sample_000001_sample_000001', '00010_sample_000010_sample_000010'], draws c/e 8/16, processes ['p2', 'p3'], torch 2.13.0+cu130; samples (compiled, eager) bf16 ((192, 16), (192, 32))
- G1' fp32 (gate): pass; null over 256 eager stand-in pairs, joint level 95.00 (null pass 1.000): cN1 8, cR1 2.61; p2 N1 3 R1 1.40; p3 N1 2 R1 1.53
- G2' bf16 (gate): FAIL; groups 192; compiled N 7.0, R 11.442; cN 1.0 (null median 0.0), cR 1.002 (median 1.000); joint level 95.00 (null pass 0.950); largest compiled ratios: fwd2.routing c0 11.442, fwd0.routing c0 2.948, fwd2.in c0 2.473, fwd0.in c0 1.092, fwd3.routing c0 1.039, fwd0.params c0 1.016
- G3' float64 (gate): 2 failing; p2 fwd0.routing c1 1.53e-12 > 1.00e-12; p3 fwd0.routing c1 1.53e-12 > 1.00e-12; max d(comp64, ref) 1.53e-12
- beside: G1' bf16 fail (cN1 19, cR1 14.21; p2 6 / 30.71; p3 6 / 30.15); G2' fp32 fail (N 7.0 vs q95 1.0, R 1.033 vs q95 1.001)
- beside, per tensor bf16 (max-relative, 1152 groups): N 31.0 (null q95 3.0), R 22.719 (null q95 1.085); outside the null

## vol

- checkpoint /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/runs/r090_vol_isla_seed42/checkpoints/ISLA.0.500.mdlus (epoch 500), cases ['00015_run_112_domain_run_112', '00018_run_115_domain_run_115'], draws c/e 8/16, processes ['p2', 'p3'], torch 2.13.0+cu130; samples (compiled, eager) bf16 ((192, 16), (192, 32))
- G1' fp32 (gate): FAIL; null over 256 eager stand-in pairs, joint level 95.00 (null pass 1.000): cN1 3, cR1 3.45; p2 N1 4 R1 6.45; p3 N1 4 R1 6.37
- G2' bf16 (gate): FAIL; groups 192; compiled N 12.0, R 1.603; cN 1.0 (null median 0.0), cR 1.009 (median 1.000); joint level 96.10 (null pass 0.950); largest compiled ratios: fwd3.routing c1 1.603, fwd4.routing c1 1.500, fwd4.routing c0 1.233, fwd0.routing c1 1.219, fwd3.routing c0 1.207, fwd3.in c0 1.081
- G3' float64 (gate): 0 failing; max d(comp64, ref) 1.29e-13
- beside: G1' bf16 pass (cN1 3, cR1 2.01; p2 2 / 1.72; p3 2 / 1.79); G2' fp32 fail (N 23.0 vs q95 1.0, R 1.098 vs q95 1.001)
- beside, per tensor bf16 (max-relative, 1152 groups): N 42.0 (null q95 3.0), R 2.146 (null q95 1.082); outside the null

