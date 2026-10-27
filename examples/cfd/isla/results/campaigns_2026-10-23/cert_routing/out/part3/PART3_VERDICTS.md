# CERT Part 3 verdicts (summarize_part3.py; CERT-ROUTING CERT_PLAN.md section 10)

Files: af_p4.jsonl, af_p5.jsonl, az_p4.jsonl, az_p5.jsonl, vol_p4.jsonl, vol_p5.jsonl, lug_p4.jsonl, lug_p5.jsonl, t2d_p4.jsonl, t2d_p5.jsonl, drs_p4.jsonl, drs_p5.jsonl, hl_p4.jsonl, hl_p5.jsonl, suv_p4.jsonl, suv_p5.jsonl

Null: 2000 resamples, joint 95% calibration. Pooled groups: 4 roles, 2 cases, l2. G3'' scale q = 1.608 (99th percentile of 17472 fp64 leave-one-out ratios).

Bitwise checks (replacements with no step changed): CHK_I bfloat16 fwd: ok, CHK_I bfloat16 read: ok, CHK_I bfloat16 sfwd: ok, CHK_I float32 fwd: ok, CHK_I float32 read: ok, CHK_I float32 sfwd: ok, CHK_L bfloat16 fwd: ok, CHK_L bfloat16 read: ok, CHK_L bfloat16 sfwd: ok, CHK_L float32 fwd: ok, CHK_L float32 read: ok, CHK_L float32 sfwd: ok, CHK_P bfloat16 fwd: ok, CHK_P bfloat16 read: ok, CHK_P bfloat16 sfwd: ok, CHK_P float32 fwd: ok, CHK_P float32 read: ok, CHK_P float32 sfwd: ok

| family | G1'' fp32: N1 / R1 (cN1 / cR1) | G2'' bf16: N / R (cN / cR) | G3'' fp64 failing | positive control G2'' | verdict | prediction |
|---|---|---|---|---|---|---|
| af | 1 / 1.15 (5 / 2.19) | 1.0 / 1.007 (0.0 / 1.000) | 0 (max 7.2e-13) | fail (N 1.0 / R 1.020) | **not certified** | 0.75 |
| az | 0 / 0.43 (3 / 1.75) | 0.0 / 1.000 (0.0 / 1.000) | 0 (max 3.8e-14) | fail (N 1.0 / R 1.022) | **not certified (gate without power)** | 0.75 |
| drs | 2 / 1.57 (1 / 3.14) | 1.0 / 1.017 (0.0 / 1.000) | 0 (max 8.7e-14) | fail (N 6.0 / R 1.033) | **not certified** | 0.30 |
| hl | 1 / 1.48 (2 / 2.34) | 0.0 / 0.992 (0.0 / 0.998) | 0 (max 4.4e-14) | pass (N 0.0 / R 0.990) | **not certified (gate without power)** | 0.30 |
| lug | 0 / 0.60 (17 / 2.62) | 0.0 / 0.998 (0.0 / 1.000) | 0 (max 7.5e-15) | pass (N 0.0 / R 0.998) | **not certified (gate without power)** | 0.60 |
| suv | 1 / 1.45 (4 / 2.78) | 0.0 / 0.990 (0.0 / 0.999) | 4 (max 3.3e-12) | pass (N 0.0 / R 0.992) | **not certified** | 0.50 |
| t2d | 1 / 1.18 (3 / 2.40) | 0.0 / 0.995 (0.0 / 1.000) | 0 (max 1.5e-12) | fail (N 2.0 / R 1.781) | **not certified (gate without power)** | 0.75 |
| vol | 3 / 5.46 (2 / 1.79) | 0.0 / 0.994 (0.0 / 1.000) | 0 (max 1.3e-13) | pass (N 0.0 / R 0.993) | **not certified** | 0.75 |

Power rule: the positive control passes G2'' in 4 of 8 families; MORE THAN 2: gate without power, no family certifies.

## af

- checkpoint /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/runs/afwhole_af_dtt_seed42/checkpoints/ISLA.0.500.mdlus (epoch 500), cases ['00009_airFoil2D_SST_31.812_1.334_0.371_3.287_0.0_19.548_airFoil2D_SST_31.812_1.334_0.371_3.287_0.0_19.548', '00010_airFoil2D_SST_31.863_-1.58_1.567_6.956_10.028_airFoil2D_SST_31.863_-1.58_1.567_6.956_10.028'], processes ['p4', 'p5'], torch 2.13.0+cu130, patches md5 666659ee5b21be93659067d37eb827f8, cert4 md5 9a52a61053761ad57cbab61114379447; bf16 sizes (compiled, reference, subset) (16, 68, 52)
- G1'' fp32: pass; null over 400 stand-in pairs, joint level 95.00 (null pass 1.000): cN1 5, cR1 2.19; p4 N1 1 R1 1.15; p5 N1 1 R1 1.14
- G2'' bf16: FAIL; groups 192; N 1.0, R 1.007; cN 0.0 (null median 0.0), cR 1.000 (median 1.000); level 95.00 (null pass 0.951); largest: fwd11.routing c1 1.007, fwd11.in c0 0.998, fwd11.in c1 0.998, fwd10.in c1 0.998, fwd10.in c0 0.998, fwd8.in c0 0.998
- G3'' fp64: 0 failing; max d(c64, ref) 7.21e-13
- positive control (compiled, bf16 softmax-over-points inputs): fail (N 1.0 / R 1.020); thresholds cN 0.0 cR 0.999; groups 96
- beside, Part 2's G2' (eager-only reference) on this data: N 2.0 / R 1.023 (cN 1.0 / cR 1.003): fail
- beside, each member against eager alone (N / R vs cN / cR): O1 1 / 1.01 (2 / 1.05); O2 0 / 1.00 (2 / 1.05); P1 0 / 1.00 (2 / 1.05); P2 1 / 1.00 (2 / 1.05); P3 1 / 1.00 (2 / 1.05); P4 0 / 1.00 (2 / 1.05); P5 2 / 1.00 (2 / 1.05); P6 0 / 1.00 (2 / 1.05); P7 0 / 1.00 (2 / 1.05)

## az

- checkpoint /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/runs/azc_full_dtt_seed42/checkpoints/ISLA.0.250.mdlus (epoch 250), cases ['00000_case-0000000000000000000-a48a547f8d4c_domain', '00005_case-0000000000000000007-c38807c2d829_domain'], processes ['p4', 'p5'], torch 2.13.0+cu130, patches md5 666659ee5b21be93659067d37eb827f8, cert4 md5 9a52a61053761ad57cbab61114379447; bf16 sizes (compiled, reference, subset) (16, 68, 52)
- G1'' fp32: pass; null over 400 stand-in pairs, joint level 95.00 (null pass 1.000): cN1 3, cR1 1.75; p4 N1 0 R1 0.42; p5 N1 0 R1 0.43
- G2'' bf16: pass; groups 192; N 0.0, R 1.000; cN 0.0 (null median 0.0), cR 1.000 (median 1.000); level 95.00 (null pass 0.951); largest: fwd10.in c0 1.000, fwd11.in c0 1.000, fwd9.in c0 1.000, fwd11.in c1 0.999, fwd8.in c0 0.998, fwd10.in c1 0.998
- G3'' fp64: 0 failing; max d(c64, ref) 3.76e-14
- positive control (compiled, bf16 softmax-over-points inputs): fail (N 1.0 / R 1.022); thresholds cN 0.0 cR 1.000; groups 96
- beside, Part 2's G2' (eager-only reference) on this data: N 1.0 / R 1.021 (cN 1.0 / cR 1.002): fail
- beside, each member against eager alone (N / R vs cN / cR): O1 3 / 2.97 (2 / 1.08); O2 12 / 1.05 (2 / 1.08); P1 9 / 1.05 (2 / 1.08); P2 14 / 1.04 (2 / 1.08); P3 3 / 1.00 (2 / 1.08); P4 0 / 1.00 (2 / 1.08); P5 2 / 1.07 (2 / 1.08); P6 0 / 1.00 (2 / 1.08); P7 0 / 1.00 (2 / 1.08)

## drs

- checkpoint /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/runs/slicenormalx_dra_ctl_seed42/checkpoints/ISLA.0.500.mdlus (epoch 500), cases ['00015_run_112_domain_run_112', '00018_run_115_domain_run_115'], processes ['p4', 'p5'], torch 2.13.0+cu130, patches md5 666659ee5b21be93659067d37eb827f8, cert4 md5 9a52a61053761ad57cbab61114379447; bf16 sizes (compiled, reference, subset) (16, 68, 52)
- G1'' fp32: FAIL; null over 400 stand-in pairs, joint level 95.00 (null pass 1.000): cN1 1, cR1 3.14; p4 N1 2 R1 1.57; p5 N1 2 R1 1.57
- G2'' bf16: FAIL; groups 96; N 1.0, R 1.017; cN 0.0 (null median 0.0), cR 1.000 (median 1.000); level 95.00 (null pass 0.951); largest: sfwd0.routing c0 1.017, sfwd2.in c0 0.996, sfwd2.in c1 0.996, sfwd3.in c0 0.994, sfwd3.in c1 0.994, sfwd4.in c1 0.983
- G3'' fp64: 0 failing; max d(c64, ref) 8.69e-14
- positive control (compiled, bf16 softmax-over-points inputs): fail (N 6.0 / R 1.033); thresholds cN 0.0 cR 1.000; groups 96
- beside, Part 2's G2' (eager-only reference) on this data: N 3.0 / R 1.153 (cN 1.0 / cR 1.000): fail
- beside, each member against eager alone (N / R vs cN / cR): O1 3 / 1.01 (2 / 1.01); O2 0 / 1.00 (2 / 1.01); P1 6 / 1.01 (2 / 1.01); P2 8 / 1.01 (2 / 1.01); P3 2 / 1.04 (2 / 1.01); P4 1 / 1.00 (2 / 1.01); P5 1 / 1.01 (2 / 1.01); P6 4 / 1.27 (2 / 1.01); P7 0 / 1.00 (2 / 1.01)

## hl

- checkpoint /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/runs/hldupx_hl_isla_seed42/checkpoints/ISLA.0.500.mdlus (epoch 500), cases ['00002_geo_LHC001_AoA_16_domain_LHC001_AoA_16', '00008_geo_LHC002_AoA_18_domain_LHC002_AoA_18'], processes ['p4', 'p5'], torch 2.13.0+cu130, patches md5 666659ee5b21be93659067d37eb827f8, cert4 md5 9a52a61053761ad57cbab61114379447; bf16 sizes (compiled, reference, subset) (16, 68, 52)
- G1'' fp32: pass; null over 400 stand-in pairs, joint level 95.00 (null pass 1.000): cN1 2, cR1 2.34; p4 N1 1 R1 1.48; p5 N1 1 R1 1.41
- G2'' bf16: pass; groups 96; N 0.0, R 0.992; cN 0.0 (null median 0.0), cR 0.998 (median 0.997); level 95.00 (null pass 0.951); largest: sfwd4.in c0 0.992, sfwd3.in c0 0.990, sfwd3.in c1 0.989, sfwd11.in c1 0.989, sfwd4.in c1 0.985, sfwd11.in c0 0.983
- G3'' fp64: 0 failing; max d(c64, ref) 4.37e-14
- positive control (compiled, bf16 softmax-over-points inputs): pass (N 0.0 / R 0.990); thresholds cN 0.0 cR 0.998; groups 96
- beside, Part 2's G2' (eager-only reference) on this data: N 2.0 / R 1.353 (cN 1.0 / cR 1.000): fail
- beside, each member against eager alone (N / R vs cN / cR): O1 0 / 1.00 (2 / 1.05); O2 0 / 1.00 (2 / 1.05); P1 1 / 1.01 (2 / 1.05); P2 0 / 1.00 (2 / 1.05); P3 0 / 1.00 (2 / 1.05); P4 0 / 1.00 (2 / 1.05); P5 0 / 1.00 (2 / 1.05); P6 2 / 1.29 (2 / 1.05); P7 0 / 1.00 (2 / 1.05)

## lug

- checkpoint /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/runs/r090fix_lug_dtt_seed42/checkpoints/ISLA.0.500.mdlus (epoch 500), cases ['00004_lug_0004_lug_0004', '00011_lug_0011_lug_0011'], processes ['p4', 'p5'], torch 2.13.0+cu130, patches md5 666659ee5b21be93659067d37eb827f8, cert4 md5 9a52a61053761ad57cbab61114379447; bf16 sizes (compiled, reference, subset) (16, 68, 52)
- G1'' fp32: pass; null over 400 stand-in pairs, joint level 95.00 (null pass 1.000): cN1 17, cR1 2.62; p4 N1 0 R1 0.56; p5 N1 0 R1 0.60
- G2'' bf16: pass; groups 192; N 0.0, R 0.998; cN 0.0 (null median 0.0), cR 1.000 (median 0.999); level 95.00 (null pass 0.951); largest: fwd2.in c1 0.998, fwd2.in c0 0.998, fwd3.in c1 0.997, fwd3.in c0 0.997, fwd4.in c0 0.996, fwd1.in c0 0.995
- G3'' fp64: 0 failing; max d(c64, ref) 7.52e-15
- positive control (compiled, bf16 softmax-over-points inputs): pass (N 0.0 / R 0.998); thresholds cN 0.0 cR 1.000; groups 96
- beside, Part 2's G2' (eager-only reference) on this data: N 1.0 / R 2.027 (cN 1.0 / cR 1.002): fail
- beside, each member against eager alone (N / R vs cN / cR): O1 0 / 1.00 (2 / 1.02); O2 2 / 1.00 (2 / 1.02); P1 1 / 1.90 (2 / 1.02); P2 1 / 1.87 (2 / 1.02); P3 2 / 1.01 (2 / 1.02); P4 0 / 1.00 (2 / 1.02); P5 0 / 1.00 (2 / 1.02); P6 1 / 1.00 (2 / 1.02); P7 0 / 1.00 (2 / 1.02)

## suv

- checkpoint /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/runs/r090_suv_isla_seed42/checkpoints/ISLA.0.500.mdlus (epoch 500), cases ['00021_run_00023_domain', '00039_run_00041_domain'], processes ['p4', 'p5'], torch 2.13.0+cu130, patches md5 666659ee5b21be93659067d37eb827f8, cert4 md5 9a52a61053761ad57cbab61114379447; bf16 sizes (compiled, reference, subset) (16, 68, 52)
- G1'' fp32: pass; null over 400 stand-in pairs, joint level 95.00 (null pass 1.000): cN1 4, cR1 2.78; p4 N1 1 R1 1.45; p5 N1 1 R1 1.45
- G2'' bf16: pass; groups 96; N 0.0, R 0.990; cN 0.0 (null median 0.0), cR 0.999 (median 0.999); level 95.00 (null pass 0.951); largest: sfwd2.in c1 0.990, sfwd2.in c0 0.986, sfwd1.in c1 0.983, sfwd6.in c1 0.962, sfwd1.params c1 0.958, sfwd6.in c0 0.955
- G3'' fp64: 4 failing; p4 sfwd9.routing c0 1.20e-12 > 1.18e-12; p4 sfwd0.routing c1 3.33e-12 > 3.17e-12; p5 sfwd9.routing c0 1.20e-12 > 1.15e-12; p5 sfwd0.routing c1 3.33e-12 > 2.93e-12; max d(c64, ref) 3.33e-12
- positive control (compiled, bf16 softmax-over-points inputs): pass (N 0.0 / R 0.992); thresholds cN 0.0 cR 0.999; groups 96
- beside, Part 2's G2' (eager-only reference) on this data: N 9.0 / R 1.256 (cN 1.0 / cR 1.001): fail
- beside, each member against eager alone (N / R vs cN / cR): O1 0 / 1.00 (2 / 1.04); O2 0 / 1.00 (2 / 1.04); P1 10 / 1.15 (2 / 1.04); P2 5 / 1.19 (2 / 1.04); P3 4 / 2.31 (2 / 1.04); P4 0 / 1.00 (2 / 1.04); P5 0 / 1.00 (2 / 1.04); P6 2 / 1.46 (2 / 1.04); P7 0 / 1.00 (2 / 1.04)

## t2d

- checkpoint /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/runs/r090fix_t2dz_isla_seed42/checkpoints/ISLA.0.500.mdlus (epoch 500), cases ['00001_sample_000001_sample_000001', '00010_sample_000010_sample_000010'], processes ['p4', 'p5'], torch 2.13.0+cu130, patches md5 666659ee5b21be93659067d37eb827f8, cert4 md5 9a52a61053761ad57cbab61114379447; bf16 sizes (compiled, reference, subset) (16, 68, 52)
- G1'' fp32: pass; null over 400 stand-in pairs, joint level 95.00 (null pass 1.000): cN1 3, cR1 2.40; p4 N1 0 R1 1.00; p5 N1 1 R1 1.18
- G2'' bf16: pass; groups 192; N 0.0, R 0.995; cN 0.0 (null median 0.0), cR 1.000 (median 0.999); level 95.00 (null pass 0.951); largest: fwd9.in c1 0.995, fwd11.in c1 0.994, fwd11.in c0 0.994, fwd9.in c0 0.994, fwd5.in c1 0.993, read9.in c0 0.992
- G3'' fp64: 0 failing; max d(c64, ref) 1.53e-12
- positive control (compiled, bf16 softmax-over-points inputs): fail (N 2.0 / R 1.781); thresholds cN 0.0 cR 0.998; groups 96
- beside, Part 2's G2' (eager-only reference) on this data: N 5.0 / R 9.772 (cN 1.0 / cR 1.003): fail
- beside, each member against eager alone (N / R vs cN / cR): O1 0 / 1.00 (2 / 1.03); O2 0 / 1.00 (2 / 1.03); P1 6 / 11.45 (2 / 1.03); P2 4 / 7.10 (2 / 1.03); P3 1 / 1.01 (2 / 1.03); P4 0 / 1.00 (2 / 1.03); P5 0 / 1.00 (2 / 1.03); P6 0 / 1.00 (2 / 1.03); P7 0 / 1.00 (2 / 1.03)

## vol

- checkpoint /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/runs/r090_vol_isla_seed42/checkpoints/ISLA.0.500.mdlus (epoch 500), cases ['00015_run_112_domain_run_112', '00018_run_115_domain_run_115'], processes ['p4', 'p5'], torch 2.13.0+cu130, patches md5 666659ee5b21be93659067d37eb827f8, cert4 md5 9a52a61053761ad57cbab61114379447; bf16 sizes (compiled, reference, subset) (16, 68, 52)
- G1'' fp32: FAIL; null over 400 stand-in pairs, joint level 95.00 (null pass 1.000): cN1 2, cR1 1.79; p4 N1 3 R1 5.46; p5 N1 3 R1 4.68
- G2'' bf16: pass; groups 192; N 0.0, R 0.994; cN 0.0 (null median 0.0), cR 1.000 (median 1.000); level 95.00 (null pass 0.952); largest: fwd4.in c0 0.994, fwd5.in c1 0.992, fwd11.in c0 0.991, fwd4.in c1 0.991, fwd7.in c1 0.990, fwd5.in c0 0.990
- G3'' fp64: 0 failing; max d(c64, ref) 1.29e-13
- positive control (compiled, bf16 softmax-over-points inputs): pass (N 0.0 / R 0.993); thresholds cN 0.0 cR 0.999; groups 96
- beside, Part 2's G2' (eager-only reference) on this data: N 13.0 / R 1.692 (cN 1.0 / cR 1.008): fail
- beside, each member against eager alone (N / R vs cN / cR): O1 0 / 1.00 (2 / 1.06); O2 0 / 1.00 (2 / 1.06); P1 24 / 1.47 (2 / 1.06); P2 23 / 1.47 (2 / 1.06); P3 1 / 1.01 (2 / 1.06); P4 0 / 1.00 (2 / 1.06); P5 0 / 1.00 (2 / 1.06); P6 2 / 1.05 (2 / 1.06); P7 0 / 1.00 (2 / 1.06)

