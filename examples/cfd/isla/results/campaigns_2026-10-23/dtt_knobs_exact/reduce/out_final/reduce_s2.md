# DTT-KNOBS stage 2: registered reading (reduce_s2.py)

PREREG.md md5 39d8272fc0ec3bc17d1ecac1bad8151b (sections 4, 5, 9). Inputs: {"scores": "/home/psharpe/gh/physicsnemo/scratch/isla-dtt-knobs-2026-10-23/reduce/s2data/scores", "ctrl": "/home/psharpe/gh/physicsnemo/scratch/isla-dtt-knobs-2026-10-23/reduce/ctrl", "regions": "/home/psharpe/gh/physicsnemo/scratch/isla-dtt-knobs-2026-10-23/reduce/s2data/regions", "ctrl_regions": "/home/psharpe/gh/physicsnemo/scratch/isla-dtt-knobs-2026-10-23/reduce/ctrl/regions", "gap_md5": "44b84474dc61f4d4e0f1486a9edea389", "attrib_md5": {"native": "1c58aa8828c6210a71b4df7836af782b", "posed": "5b55f04bb3089f81fbf89f461cbbb8a5"}, "lanes_md5": {"s2": "78899ea33ad2f4ba16235225b464e2db", "s2n": "b5796f096bcaffd17981d9c91071c8d5", "r090eagaf": "81937645b8a4b6df64575f764267e282"}}.

Conventions: R1-R3 D = ln of the ratio of arithmetic seed means (SLICES-AF's read_slices.py); R5 Delta = difference of mean ln CM (ATTRIB reduce.py); the other form is computed beside and any label it would change is flagged. R4 statistic: **registered** (see the module docstring).

## Checks before reading (PREREG section 4)

| run | protocol | valid | failed checks |
|---|---|---|---|
| r090eag_af_dtt_seed42 | native | yes | - |
| r090eag_af_dtt_seed43 | native | yes | - |
| r090eag_af_dtt_seed42 | posed | yes | - |
| r090eag_af_dtt_seed43 | posed | yes | - |
| r090eag_af_isla_seed42 | native | yes | - |
| r090eag_af_isla_seed43 | native | yes | - |
| r090eag_af_isla_seed42 | posed | yes | - |
| r090eag_af_isla_seed43 | posed | yes | - |
| r090fix_af_dtt_seed42 | native | yes | - |
| r090fix_af_dtt_seed43 | native | yes | - |
| r090fix_af_dtt_seed42 | posed | yes | - |
| r090fix_af_dtt_seed43 | posed | yes | - |
| att2_af_dttnosc_seed42 | native | yes | - |
| att2_af_dttnosc_seed43 | native | yes | - |
| att2_af_dttnosc_seed42 | posed | yes | - |
| att2_af_dttnosc_seed43 | posed | yes | - |
| dttknobs_s2_theta2_seed42 | native | yes | - |
| dttknobs_s2_theta2_seed43 | native | yes | - |
| dttknobs_s2_theta2_seed42 | posed | yes | - |
| dttknobs_s2_theta2_seed43 | posed | yes | - |
| dttknobs_s2_theta05_seed42 | native | yes | - |
| dttknobs_s2_theta05_seed43 | native | yes | - |
| dttknobs_s2_theta05_seed42 | posed | yes | - |
| dttknobs_s2_theta05_seed43 | posed | yes | - |
| dttknobs_s2_b1_seed42 | native | yes | - |
| dttknobs_s2_b1_seed43 | native | yes | - |
| dttknobs_s2_b1_seed42 | posed | yes | - |
| dttknobs_s2_b1_seed43 | posed | yes | - |
| dttknobs_s2_basis8_seed42 | native | yes | - |
| dttknobs_s2_basis8_seed43 | native | yes | - |
| dttknobs_s2_basis8_seed42 | posed | yes | - |
| dttknobs_s2_basis8_seed43 | posed | yes | - |
| dttknobs_s2_exact_seed42 | native | yes | - |
| dttknobs_s2_exact_seed43 | native | yes | - |
| dttknobs_s2_exact_seed42 | posed | yes | - |
| dttknobs_s2_exact_seed43 | posed | yes | - |
| dttknobs_s2_noscstd_seed42 | native | yes | - |
| dttknobs_s2_noscstd_seed43 | native | yes | - |
| dttknobs_s2_noscstd_seed42 | posed | yes | - |
| dttknobs_s2_noscstd_seed43 | posed | yes | - |

Control against the registered numbers: pressure seeds 0.012428 / 0.0123887 (max rel 1.8e-06), s_c 0.0022 (registered 0.0022); velocity seeds 0.00737341 / 0.00697252 (max rel 4.7e-07), s_c 0.0395 (registered 0.0395); nut seeds 0.0207112 / 0.0204842 (max rel 1.2e-06), s_c 0.0078 (registered 0.0078).

## R1, per field, arms 1-5 against r090eag_af_dtt (native decides)

| arm | field | arm (s42, s43) | control (s42, s43) | arm ÷ ctrl | D | s_arm, s_c | 2 SE | label | flag | seed ratios | posed: D, label | case median ratio | cases won | arm ÷ ISLA (ctrl ÷ ISLA) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| K-THETA2 | pressure | 0.0132885 (0.013215, 0.013362) | 0.0124084 (0.012428, 0.0123887) | 1.071 | +0.0685 | 0.0078, 0.0022 | 0.0115 | **no change** | up | 1.063, 1.079 | 0.0686, no change | 1.056 | 36 % | 1.065 (0.994) |
| K-THETA2 | velocity | 0.00745764 (0.00777823, 0.00713705) | 0.00717297 (0.00737341, 0.00697252) | 1.040 | +0.0389 | 0.0608, 0.0395 | 0.1026 | **no change** | - | 1.055, 1.024 | 0.0389, no change | 1.036 | 28 % | 1.173 (1.129) |
| K-THETA2 | nut | 0.0201911 (0.0192515, 0.0211307) | 0.0205977 (0.0207112, 0.0204842) | 0.980 | -0.0199 | 0.0659, 0.0078 | 0.0938 | **no change** | - | 0.930, 1.032 | -0.0200, no change | 0.908 | 62 % | 1.315 (1.342) |
| K-THETA05 | pressure | 0.0116173 (0.0116669, 0.0115677) | 0.0124084 (0.012428, 0.0123887) | 0.936 | -0.0659 | 0.0060, 0.0022 | 0.0091 | **no change** | down | 0.939, 0.934 | -0.0659, no change | 0.851 | 75 % | 0.931 (0.994) |
| K-THETA05 | velocity | 0.00681243 (0.00694353, 0.00668132) | 0.00717297 (0.00737341, 0.00697252) | 0.950 | -0.0516 | 0.0272, 0.0395 | 0.0679 | **no change** | down | 0.942, 0.958 | -0.0515, no change | 0.973 | 75 % | 1.072 (1.129) |
| K-THETA05 | nut | 0.0192027 (0.0182896, 0.0201158) | 0.0205977 (0.0207112, 0.0204842) | 0.932 | -0.0701 | 0.0673, 0.0078 | 0.0958 | **no change** | down | 0.883, 0.982 | -0.0706, no change | 0.895 | 68 % | 1.251 (1.342) |
| K-B1 | pressure | 0.0125639 (0.0120369, 0.013091) | 0.0124084 (0.012428, 0.0123887) | 1.013 | +0.0125 | 0.0594, 0.0022 | 0.0840 | **no change** | - | 0.969, 1.057 | 0.0124, no change | 0.999 | 54 % | 1.007 (0.994) |
| K-B1 | velocity | 0.00708343 (0.00705666, 0.00711021) | 0.00717297 (0.00737341, 0.00697252) | 0.988 | -0.0126 | 0.0053, 0.0395 | 0.0564 | **no change** | - | 0.957, 1.020 | -0.0124, no change | 0.962 | 72 % | 1.115 (1.129) |
| K-B1 | nut | 0.0204219 (0.0197052, 0.0211385) | 0.0205977 (0.0207112, 0.0204842) | 0.991 | -0.0086 | 0.0496, 0.0078 | 0.0711 | **no change** | - | 0.951, 1.032 | -0.0086, no change | 0.898 | 61 % | 1.330 (1.342) |
| K-BASIS8 | pressure | 0.0130748 (0.0139169, 0.0122327) | 0.0124084 (0.012428, 0.0123887) | 1.054 | +0.0523 | 0.0912, 0.0022 | 0.1290 | **no change** | - | 1.120, 0.987 | 0.0522, no change | 1.038 | 34 % | 1.048 (0.994) |
| K-BASIS8 | velocity | 0.00736643 (0.00757042, 0.00716244) | 0.00717297 (0.00737341, 0.00697252) | 1.027 | +0.0266 | 0.0392, 0.0395 | 0.0787 | **no change** | - | 1.027, 1.027 | 0.0266, no change | 1.007 | 42 % | 1.159 (1.129) |
| K-BASIS8 | nut | 0.026518 (0.0268603, 0.0261757) | 0.0205977 (0.0207112, 0.0204842) | 1.287 | +0.2526 | 0.0183, 0.0078 | 0.0281 | **harms** | up | 1.297, 1.278 | 0.2526, harms | 1.224 | 22 % | 1.727 (1.342) |
| K-EXACT | pressure | 0.0118696 (0.0118839, 0.0118553) | 0.0124084 (0.012428, 0.0123887) | 0.957 | -0.0444 | 0.0017, 0.0022 | 0.0040 | **no change** | down | 0.956, 0.957 | -0.0442, no change | 0.951 | 56 % | 0.951 (0.994) |
| K-EXACT | velocity | 0.00700205 (0.00688113, 0.00712297) | 0.00717297 (0.00737341, 0.00697252) | 0.976 | -0.0241 | 0.0244, 0.0395 | 0.0657 | **no change** | - | 0.933, 1.022 | -0.0239, no change | 0.955 | 70 % | 1.102 (1.129) |
| K-EXACT | nut | 0.0180975 (0.0156516, 0.0205434) | 0.0205977 (0.0207112, 0.0204842) | 0.879 | -0.1294 | 0.1923, 0.0078 | 0.2722 | **no change** | - | 0.756, 1.003 | -0.1295, no change | 0.814 | 65 % | 1.179 (1.342) |

## R2, the theta ladder (2, 1, 0.5, 0)

| field | L_2, L_0.5, L_0 | D_2, D_0.5, D_0 | steps (D_tight - D_loose vs 2 SE) | label |
|---|---|---|---|---|
| pressure | no change, no change, no change | +0.0685, -0.0659, -0.0444 | (2->1) -0.0685 vs 0.0115; (1->0.5) -0.0659 vs 0.0091; (0.5->0) +0.0215 vs 0.0089 **rises** | **pure cost knob** |
| velocity | no change, no change, no change | +0.0389, -0.0516, -0.0241 | (2->1) -0.0389 vs 0.1026; (1->0.5) -0.0516 vs 0.0679; (0.5->0) +0.0275 vs 0.0517 | **pure cost knob** (free to theta 2) |
| nut | no change, no change, no change | -0.0199, -0.0701, -0.1294 | (2->1) +0.0199 vs 0.0938; (1->0.5) -0.0701 vs 0.0958; (0.5->0) -0.0593 vs 0.2881 | **pure cost knob** (free to theta 2) |

**Overall: pure cost knob** (free to theta 2); fields carrying 'free to theta 2': 2.

## R3, decision use

| arm | labels (p, v, nut) | flags | reading |
|---|---|---|---|
| K-THETA2 | no change, no change, no change | up, -, - | not a free cost cut (up flag on pressure): to the scope freeze as a cost-against-accuracy trade, or dropped |
| K-B1 | no change, no change, no change | -, -, - | candidate for the scope freeze (free cost cut); stage 3 proposed; K-B1 is an architecture change: adoption also needs Peter's word |
| K-BASIS8 | no change, no change, harms | -, -, up | not a free cost cut (harms on nut, up flag on nut): to the scope freeze as a cost-against-accuracy trade, or dropped |
| K-THETA05 | no change, no change, no change | down, down, down | not a candidate (needs helps on at least one field and no harm) |
| K-EXACT | no change, no change, no change | down, -, - | not a candidate (needs helps on at least one field and no harm) |

## R4, the far-field gap (statistic: registered; native, 200 cases, regional probe at the arm's own read)

Control's regional output reproduces DK-GAP's seed noise (noise_D) to 7.0e-08.

| arm | cell | D_R | s_arm, s_c | 2 SE_R | label | other statistic: D_R, label | seed ratios |
|---|---|---|---|---|---|---|---|
| K-THETA2 | velocity wake | +0.0293 | 0.1231, 0.0427 | 0.1842 | **level** | +0.0139, level | 1.086, 0.970 |
| K-THETA2 | nut >=1 chord | +0.0480 | 0.3517, 0.1320 | 0.5312 | **level** | +0.1842, level | 0.875, 1.194 |
| K-THETA05 | velocity wake | -0.1224 | 0.0664, 0.0427 | 0.1117 | **narrows** | -0.1457, narrows | 0.899, 0.870 |
| K-THETA05 | nut >=1 chord | -0.0767 | 0.0372, 0.1320 | 0.1939 | **level** | -0.2072, level | 0.994, 0.870 |
| K-B1 | velocity wake | -0.0033 | 0.0722, 0.0427 | 0.1187 | **level** | -0.0092, level | 1.017, 0.975 |
| K-B1 | nut >=1 chord | +0.1669 | 0.0698, 0.1320 | 0.2112 | **level** | +0.3210, level | 1.239, 1.134 |
| K-BASIS8 | velocity wake | +0.0917 | 0.0629, 0.0427 | 0.1075 | **level** | +0.0694, level | 1.017, 1.180 |
| K-BASIS8 | nut >=1 chord | +0.5156 | 0.5342, 0.1320 | 0.7782 | **level** | +0.4946, level | 1.180, 2.085 |
| K-EXACT | velocity wake | +0.0008 | 0.0433, 0.0427 | 0.0860 | **level** | -0.0038, level | 1.001, 1.000 |
| K-EXACT | nut >=1 chord | -0.2477 | 0.0731, 0.1320 | 0.2134 | **narrows** | -0.5666, narrows | 0.816, 0.751 |

**Q3: a finer read narrows the far-field gap**

Beside (per seed [s42, s43]): arm ÷ control, registered form | arm ÷ control, DK-GAP form | arm ÷ ISLA (DK-GAP form) with the control's ÷ ISLA in brackets (stage 1: {'velocity wake': (1.95, 1.99), 'nut >=1 chord': (1.48, 2.47)}).

### K-THETA2: zone

| field | near_wall d<0.01 | wake | outer |
|---|---|---|---|
| pressure | 1.06, 1.17; 1.06, 1.21; ISLA 0.98, 1.26 [0.92, 1.04] | 0.91, 1.00; 1.01, 1.02; ISLA 1.46, 1.47 [1.44, 1.44] | 1.05, 1.05; 1.06, 1.06; ISLA 1.15, 1.24 [1.08, 1.16] |
| velocity | 1.06, 1.12; 1.06, 1.09; ISLA 1.08, 1.07 [1.02, 0.98] | 1.09, 0.97; 1.08, 0.94; ISLA 2.11, 1.88 [1.95, 1.99] | 1.07, 1.03; 1.06, 1.03; ISLA 1.40, 1.42 [1.32, 1.38] |
| nut | 1.03, 0.95; 1.04, 0.97; ISLA 1.24, 1.17 [1.19, 1.21] | 0.98, 1.09; 0.99, 1.17; ISLA 1.01, 1.58 [1.02, 1.35] | 1.21, 1.10; 1.13, 1.11; ISLA 1.14, 1.19 [1.01, 1.07] |

### K-THETA2: wall_band

| field | d<1e-4 | 1e-4..1e-3 | 1e-3..1e-2 | 1e-2..0.1 | 0.1..1 | >=1 |
|---|---|---|---|---|---|---|
| pressure | 1.08, 1.16; 1.09, 1.22; ISLA 1.04, 1.34 [0.95, 1.10] | 1.05, 1.19; 1.06, 1.24; ISLA 0.98, 1.28 [0.92, 1.03] | 0.93, 1.14; 1.01, 1.13; ISLA 0.89, 1.08 [0.88, 0.95] | 1.03, 1.02; 1.07, 1.05; ISLA 1.11, 1.16 [1.03, 1.10] | 1.00, 1.09; 1.03, 1.07; ISLA 1.32, 1.48 [1.28, 1.38] | 0.98, 1.00; 1.03, 1.02; ISLA 1.39, 1.52 [1.35, 1.49] |
| velocity | 1.10, 1.09; 1.10, 1.08; ISLA 1.22, 1.13 [1.11, 1.04] | 1.05, 1.21; 1.05, 1.15; ISLA 1.04, 1.07 [0.98, 0.93] | 1.03, 1.06; 1.03, 1.04; ISLA 1.02, 1.02 [0.99, 0.98] | 1.09, 0.98; 1.08, 0.97; ISLA 1.43, 1.28 [1.32, 1.32] | 1.08, 1.03; 1.07, 1.00; ISLA 1.97, 1.99 [1.84, 1.98] | 1.05, 1.00; 1.05, 1.00; ISLA 1.74, 1.78 [1.66, 1.79] |
| nut | 1.12, 1.04; 1.13, 1.00; ISLA 1.04, 1.14 [0.92, 1.14] | 1.06, 1.11; 1.08, 1.10; ISLA 1.08, 1.25 [1.00, 1.13] | 1.03, 0.94; 1.03, 0.95; ISLA 1.28, 1.16 [1.25, 1.22] | 1.13, 1.07; 1.10, 0.98; ISLA 1.05, 1.15 [0.96, 1.18] | 0.98, 1.08; 1.00, 1.09; ISLA 0.99, 1.29 [0.98, 1.19] | 0.87, 1.19; 0.89, 1.38; ISLA 1.32, 3.42 [1.48, 2.47] |

### K-THETA05: zone

| field | near_wall d<0.01 | wake | outer |
|---|---|---|---|
| pressure | 0.83, 0.90; 0.95, 1.03; ISLA 0.88, 1.07 [0.92, 1.04] | 1.02, 0.99; 0.99, 0.95; ISLA 1.43, 1.38 [1.44, 1.44] | 0.92, 0.97; 0.99, 0.95; ISLA 1.07, 1.10 [1.08, 1.16] |
| velocity | 0.95, 0.98; 0.98, 0.98; ISLA 1.00, 0.96 [1.02, 0.98] | 0.90, 0.87; 0.88, 0.84; ISLA 1.73, 1.68 [1.95, 1.99] | 0.93, 0.98; 0.95, 0.98; ISLA 1.25, 1.35 [1.32, 1.38] |
| nut | 1.00, 0.97; 0.95, 0.91; ISLA 1.14, 1.10 [1.19, 1.21] | 1.02, 1.01; 1.04, 0.83; ISLA 1.07, 1.12 [1.02, 1.35] | 1.10, 1.13; 1.07, 1.07; ISLA 1.08, 1.14 [1.01, 1.07] |

### K-THETA05: wall_band

| field | d<1e-4 | 1e-4..1e-3 | 1e-3..1e-2 | 1e-2..0.1 | 0.1..1 | >=1 |
|---|---|---|---|---|---|---|
| pressure | 0.84, 0.90; 0.93, 0.99; ISLA 0.89, 1.09 [0.95, 1.10] | 0.80, 0.87; 0.94, 1.06; ISLA 0.87, 1.08 [0.92, 1.03] | 0.94, 0.97; 1.00, 1.06; ISLA 0.87, 1.01 [0.88, 0.95] | 0.93, 0.98; 1.02, 0.95; ISLA 1.05, 1.05 [1.03, 1.10] | 0.97, 0.99; 0.94, 0.95; ISLA 1.20, 1.30 [1.28, 1.38] | 0.97, 0.94; 1.03, 0.95; ISLA 1.40, 1.41 [1.35, 1.49] |
| velocity | 0.90, 1.00; 0.95, 1.00; ISLA 1.05, 1.05 [1.11, 1.04] | 0.97, 1.01; 1.00, 1.00; ISLA 0.98, 0.94 [0.98, 0.93] | 0.98, 0.94; 0.99, 0.94; ISLA 0.98, 0.92 [0.99, 0.98] | 0.93, 0.94; 0.95, 0.92; ISLA 1.25, 1.22 [1.32, 1.32] | 0.90, 0.92; 0.88, 0.91; ISLA 1.63, 1.80 [1.84, 1.98] | 0.96, 0.99; 0.96, 0.98; ISLA 1.59, 1.76 [1.66, 1.79] |
| nut | 1.10, 1.02; 1.05, 1.01; ISLA 0.97, 1.15 [0.92, 1.14] | 1.01, 0.99; 1.01, 1.00; ISLA 1.01, 1.13 [1.00, 1.13] | 0.99, 0.96; 0.94, 0.89; ISLA 1.17, 1.09 [1.25, 1.22] | 1.12, 1.07; 1.11, 0.94; ISLA 1.07, 1.11 [0.96, 1.18] | 1.01, 1.03; 1.03, 0.89; ISLA 1.01, 1.06 [0.98, 1.19] | 0.99, 0.87; 1.09, 0.65; ISLA 1.61, 1.62 [1.48, 2.47] |

### K-B1: zone

| field | near_wall d<0.01 | wake | outer |
|---|---|---|---|
| pressure | 0.83, 1.29; 0.96, 1.07; ISLA 0.89, 1.11 [0.92, 1.04] | 0.87, 1.03; 0.88, 0.97; ISLA 1.27, 1.40 [1.44, 1.44] | 0.94, 1.02; 0.96, 1.02; ISLA 1.04, 1.19 [1.08, 1.16] |
| velocity | 0.95, 1.15; 0.98, 1.16; ISLA 1.01, 1.13 [1.02, 0.98] | 1.02, 0.98; 1.02, 0.96; ISLA 1.99, 1.92 [1.95, 1.99] | 0.98, 1.00; 0.97, 1.01; ISLA 1.29, 1.39 [1.32, 1.38] |
| nut | 0.90, 1.05; 0.96, 1.05; ISLA 1.15, 1.27 [1.19, 1.21] | 0.93, 1.13; 1.27, 1.10; ISLA 1.30, 1.49 [1.02, 1.35] | 1.27, 1.05; 1.71, 1.07; ISLA 1.73, 1.14 [1.01, 1.07] |

### K-B1: wall_band

| field | d<1e-4 | 1e-4..1e-3 | 1e-3..1e-2 | 1e-2..0.1 | 0.1..1 | >=1 |
|---|---|---|---|---|---|---|
| pressure | 0.80, 1.29; 0.94, 1.04; ISLA 0.89, 1.15 [0.95, 1.10] | 0.84, 1.32; 0.99, 1.10; ISLA 0.91, 1.13 [0.92, 1.03] | 0.92, 1.20; 0.97, 1.07; ISLA 0.85, 1.02 [0.88, 0.95] | 0.96, 1.02; 1.01, 1.03; ISLA 1.04, 1.14 [1.03, 1.10] | 0.88, 1.04; 0.88, 0.99; ISLA 1.12, 1.36 [1.28, 1.38] | 0.85, 0.96; 0.82, 0.98; ISLA 1.11, 1.45 [1.35, 1.49] |
| velocity | 0.91, 1.10; 0.94, 1.10; ISLA 1.04, 1.15 [1.11, 1.04] | 0.96, 1.24; 0.99, 1.23; ISLA 0.97, 1.15 [0.98, 0.93] | 1.01, 1.11; 1.02, 1.12; ISLA 1.01, 1.10 [0.99, 0.98] | 1.03, 0.99; 1.03, 0.99; ISLA 1.36, 1.31 [1.32, 1.32] | 0.98, 0.99; 0.98, 0.98; ISLA 1.81, 1.95 [1.84, 1.98] | 0.92, 1.00; 0.93, 1.00; ISLA 1.54, 1.79 [1.66, 1.79] |
| nut | 1.07, 1.12; 1.08, 1.04; ISLA 1.00, 1.18 [0.92, 1.14] | 1.04, 1.12; 1.08, 1.12; ISLA 1.08, 1.26 [1.00, 1.13] | 0.88, 1.04; 0.94, 1.05; ISLA 1.17, 1.28 [1.25, 1.22] | 1.02, 1.02; 1.17, 0.96; ISLA 1.13, 1.14 [0.96, 1.18] | 0.91, 1.14; 1.20, 1.10; ISLA 1.17, 1.31 [0.98, 1.19] | 1.24, 1.13; 1.79, 1.14; ISLA 2.65, 2.81 [1.48, 2.47] |

### K-BASIS8: zone

| field | near_wall d<0.01 | wake | outer |
|---|---|---|---|
| pressure | 1.09, 0.99; 1.11, 0.96; ISLA 1.02, 0.99 [0.92, 1.04] | 1.30, 1.11; 1.16, 1.03; ISLA 1.67, 1.49 [1.44, 1.44] | 1.04, 1.07; 1.16, 1.02; ISLA 1.25, 1.18 [1.08, 1.16] |
| velocity | 1.05, 1.03; 1.08, 1.01; ISLA 1.11, 0.99 [1.02, 0.98] | 1.02, 1.18; 0.99, 1.16; ISLA 1.93, 2.31 [1.95, 1.99] | 1.08, 1.02; 1.09, 1.02; ISLA 1.44, 1.40 [1.32, 1.38] |
| nut | 1.37, 0.99; 1.19, 0.87; ISLA 1.42, 1.05 [1.19, 1.21] | 1.34, 1.47; 1.40, 1.50; ISLA 1.43, 2.03 [1.02, 1.35] | 1.41, 1.74; 1.59, 2.51; ISLA 1.61, 2.68 [1.01, 1.07] |

### K-BASIS8: wall_band

| field | d<1e-4 | 1e-4..1e-3 | 1e-3..1e-2 | 1e-2..0.1 | 0.1..1 | >=1 |
|---|---|---|---|---|---|---|
| pressure | 1.06, 0.99; 1.08, 0.91; ISLA 1.03, 1.00 [0.95, 1.10] | 1.11, 0.99; 1.10, 0.99; ISLA 1.02, 1.01 [0.92, 1.03] | 1.14, 1.03; 1.17, 1.01; ISLA 1.03, 0.96 [0.88, 0.95] | 1.06, 1.06; 1.18, 1.00; ISLA 1.22, 1.10 [1.03, 1.10] | 1.20, 1.11; 1.11, 1.02; ISLA 1.41, 1.41 [1.28, 1.38] | 1.08, 1.11; 1.21, 1.12; ISLA 1.64, 1.66 [1.35, 1.49] |
| velocity | 0.98, 1.00; 1.01, 1.00; ISLA 1.12, 1.05 [1.11, 1.04] | 1.06, 1.07; 1.10, 1.04; ISLA 1.08, 0.97 [0.98, 0.93] | 1.10, 1.00; 1.13, 0.98; ISLA 1.12, 0.96 [0.99, 0.98] | 1.12, 1.01; 1.14, 1.00; ISLA 1.51, 1.32 [1.32, 1.32] | 0.99, 1.15; 0.96, 1.14; ISLA 1.77, 2.26 [1.84, 1.98] | 0.95, 1.19; 0.96, 1.23; ISLA 1.59, 2.19 [1.66, 1.79] |
| nut | 1.05, 1.16; 1.07, 1.06; ISLA 0.99, 1.21 [0.92, 1.14] | 1.03, 1.05; 1.05, 1.02; ISLA 1.05, 1.15 [1.00, 1.13] | 1.40, 0.97; 1.21, 0.84; ISLA 1.51, 1.03 [1.25, 1.22] | 1.56, 1.25; 1.69, 1.22; ISLA 1.63, 1.43 [0.96, 1.18] | 1.32, 1.42; 1.36, 1.48; ISLA 1.34, 1.76 [0.98, 1.19] | 1.18, 2.08; 1.37, 1.79; ISLA 2.03, 4.44 [1.48, 2.47] |

### K-EXACT: zone

| field | near_wall d<0.01 | wake | outer |
|---|---|---|---|
| pressure | 0.86, 0.96; 0.94, 0.83; ISLA 0.87, 0.86 [0.92, 1.04] | 0.86, 1.08; 0.94, 1.07; ISLA 1.35, 1.55 [1.44, 1.44] | 0.91, 1.04; 0.92, 1.02; ISLA 1.00, 1.18 [1.08, 1.16] |
| velocity | 0.90, 1.04; 0.94, 1.04; ISLA 0.96, 1.02 [1.02, 0.98] | 1.00, 1.00; 1.01, 0.98; ISLA 1.97, 1.96 [1.95, 1.99] | 0.92, 1.01; 0.92, 1.01; ISLA 1.22, 1.39 [1.32, 1.38] |
| nut | 1.06, 1.12; 1.00, 0.99; ISLA 1.19, 1.20 [1.19, 1.21] | 0.92, 1.11; 0.95, 0.85; ISLA 0.97, 1.14 [1.02, 1.35] | 1.01, 1.16; 0.95, 1.08; ISLA 0.96, 1.16 [1.01, 1.07] |

### K-EXACT: wall_band

| field | d<1e-4 | 1e-4..1e-3 | 1e-3..1e-2 | 1e-2..0.1 | 0.1..1 | >=1 |
|---|---|---|---|---|---|---|
| pressure | 0.82, 1.01; 0.94, 0.76; ISLA 0.89, 0.84 [0.95, 1.10] | 0.87, 0.93; 0.94, 0.82; ISLA 0.87, 0.85 [0.92, 1.03] | 0.97, 0.89; 0.97, 0.96; ISLA 0.85, 0.92 [0.88, 0.95] | 0.95, 1.05; 0.98, 1.04; ISLA 1.01, 1.15 [1.03, 1.10] | 0.80, 1.06; 0.84, 1.02; ISLA 1.07, 1.41 [1.28, 1.38] | 0.90, 1.00; 0.96, 0.99; ISLA 1.30, 1.47 [1.35, 1.49] |
| velocity | 0.86, 1.06; 0.89, 1.04; ISLA 0.99, 1.09 [1.11, 1.04] | 0.90, 1.04; 0.93, 1.03; ISLA 0.92, 0.96 [0.98, 0.93] | 0.96, 1.03; 0.98, 1.06; ISLA 0.97, 1.04 [0.99, 0.98] | 0.96, 1.04; 0.97, 1.05; ISLA 1.28, 1.39 [1.32, 1.32] | 0.95, 0.96; 0.95, 0.95; ISLA 1.76, 1.89 [1.84, 1.98] | 1.00, 0.99; 1.01, 0.98; ISLA 1.67, 1.75 [1.66, 1.79] |
| nut | 1.05, 1.01; 1.06, 0.97; ISLA 0.98, 1.10 [0.92, 1.14] | 0.98, 1.03; 1.00, 0.99; ISLA 1.00, 1.12 [1.00, 1.13] | 1.07, 1.14; 1.00, 0.99; ISLA 1.24, 1.21 [1.25, 1.22] | 1.04, 1.23; 1.07, 1.12; ISLA 1.03, 1.33 [0.96, 1.18] | 0.92, 1.14; 0.96, 0.94; ISLA 0.94, 1.12 [0.98, 1.19] | 0.82, 0.75; 0.78, 0.45; ISLA 1.15, 1.10 [1.48, 2.47] |

## R5, DTT-NOSC-STD against B = r090fix_af_dtt and DTT-NOSC = att2_af_dttnosc (fix-trained)

### native (registered)

| field | NOSC-STD (s42, s43) | σ_f 6 dof -> 7 dof | SE | Δ (÷ B) | Δ_N (÷ DTT-NOSC) | A | N | reading | φ = Δ / G_f | Δ / D_DTT | bundle thread (C_f) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pressure | 0.0143767, 0.0161851 | 0.0422 -> 0.0503 | 0.0503 | +0.1869 (1.205) | -0.0476 (0.953) | yes | no | **standardization leaves it (the scales carry it)** | +0.80 (G 0.234) | +1.81 | the cost sits in the scales, as far as two seeds can show |
| velocity | 0.00726362, 0.00988128 | 0.0291 -> 0.0866 | 0.0866 | +0.1730 (1.189) | -0.0298 (0.971) | no | no | **unresolved** (arithmetic form: standardization leaves it (the scales carry it)) | +0.85 (G 0.203) | +1.11 | not read (C_f 0.047 < ln 1.05) |
| nut | 0.0231498, 0.0274351 | 0.0637 -> 0.0744 | 0.0744 | +0.1619 (1.176) | +0.0151 (1.015) | yes | no | **standardization leaves it (the scales carry it)** | +1.10 (G 0.147) | +6.23 | the cost sits in the scales, as far as two seeds can show |

Per seed: pressure ÷ B 1.128, 1.288; ÷ DTT-NOSC 0.916, 0.992; ÷ DTT-CONT 1.057, 1.118; velocity ÷ B 1.022, 1.382; ÷ DTT-NOSC 0.813, 1.159; ÷ DTT-CONT 0.887, 1.165; nut ÷ B 1.164, 1.188; ÷ DTT-NOSC 0.909, 1.134; ÷ DTT-CONT 1.049, 1.253.
Checks: pressure trees vs ATTRIB pool 0.0e+00, σ6 matches True, G matches True; velocity trees vs ATTRIB pool 0.0e+00, σ6 matches True, G matches True; nut trees vs ATTRIB pool 0.0e+00, σ6 matches True, G matches True.

### posed (beside, not a verdict)

| field | NOSC-STD (s42, s43) | σ_f 6 dof -> 7 dof | SE | Δ (÷ B) | Δ_N (÷ DTT-NOSC) | A | N | reading | φ = Δ / G_f | Δ / D_DTT | bundle thread (C_f) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pressure | 0.0143951, 0.0162045 | 0.0421 -> 0.0502 | 0.0502 | +0.1862 (1.205) | -0.0475 (0.954) | yes | no | **standardization leaves it (the scales carry it)** | +0.80 (G 0.234) | +1.81 | the cost sits in the scales, as far as two seeds can show |
| velocity | 0.00727047, 0.00988621 | 0.0291 -> 0.0864 | 0.0864 | +0.1726 (1.188) | -0.0298 (0.971) | no | no | **unresolved** (arithmetic form: standardization leaves it (the scales carry it)) | +0.85 (G 0.202) | +1.11 | not read (C_f 0.046 < ln 1.05) |
| nut | 0.0231508, 0.0274354 | 0.0637 -> 0.0744 | 0.0744 | +0.1617 (1.176) | +0.0151 (1.015) | yes | no | **standardization leaves it (the scales carry it)** | +1.10 (G 0.147) | +6.22 | the cost sits in the scales, as far as two seeds can show |

Per seed: pressure ÷ B 1.128, 1.287; ÷ DTT-NOSC 0.916, 0.993; ÷ DTT-CONT 1.056, 1.118; velocity ÷ B 1.022, 1.381; ÷ DTT-NOSC 0.813, 1.159; ÷ DTT-CONT 0.888, 1.165; nut ÷ B 1.163, 1.188; ÷ DTT-NOSC 0.909, 1.134; ÷ DTT-CONT 1.049, 1.252.
Checks: pressure trees vs ATTRIB pool 0.0e+00, σ6 matches None, G matches None; velocity trees vs ATTRIB pool 0.0e+00, σ6 matches None, G matches None; nut trees vs ATTRIB pool 0.0e+00, σ6 matches None, G matches None.

Interacting-ISLA mirror (ATTRIB RESULTS.md 2c, beside): NOSC-STD ÷ B = 1.003 / 1.029 / 1.043, no measurable effect; 'input standardization carries the step; the scales do not'.

## Cost per arm (train.log 'Step:'; sacct elapsed x nodes; GPU-h = 4 x node-h)

| run | median step (p10-p90) s | steps | peak mem GB | links | node-h | GPU-h | control beside |
|---|---|---|---|---|---|---|---|
| dttknobs_s2_theta2_seed42 | 0.935 (0.920-0.947) | 101264 | 49.9 | 8 | 27.12 | 108.5 | r090eag_af_dtt_seed42: 0.931 s, 27.69 node-h, 110.8 GPU-h |
| dttknobs_s2_theta2_seed43 | 0.935 (0.920-0.948) | 100758 | 49.6 | 8 | 26.98 | 107.9 | r090eag_af_dtt_seed43: 0.937 s, 27.51 node-h, 110.1 GPU-h |
| dttknobs_s2_theta05_seed42 | 0.939 (0.926-0.951) | 101217 | 50.4 | 8 | 27.20 | 108.8 | r090eag_af_dtt_seed42: 0.931 s, 27.69 node-h, 110.8 GPU-h |
| dttknobs_s2_theta05_seed43 | 0.939 (0.924-0.952) | 100966 | 50.3 | 8 | 27.18 | 108.7 | r090eag_af_dtt_seed43: 0.937 s, 27.51 node-h, 110.1 GPU-h |
| dttknobs_s2_b1_seed42 | 0.885 (0.874-0.895) | 100431 | 47.6 | 8 | 25.51 | 102.0 | r090eag_af_dtt_seed42: 0.931 s, 27.69 node-h, 110.8 GPU-h |
| dttknobs_s2_b1_seed43 | 0.889 (0.875-0.900) | 100417 | 47.7 | 8 | 25.53 | 102.1 | r090eag_af_dtt_seed43: 0.937 s, 27.51 node-h, 110.1 GPU-h |
| dttknobs_s2_basis8_seed42 | 0.936 (0.921-0.950) | 101101 | 49.7 | 8 | 27.08 | 108.3 | r090eag_af_dtt_seed42: 0.931 s, 27.69 node-h, 110.8 GPU-h |
| dttknobs_s2_basis8_seed43 | 0.931 (0.917-0.944) | 101122 | 49.9 | 8 | 27.15 | 108.6 | r090eag_af_dtt_seed43: 0.937 s, 27.51 node-h, 110.1 GPU-h |
| dttknobs_s2_exact_seed42 | 1.354 (1.331-1.392) | 102451 | 85.3 | 10 | 39.69 | 158.7 | r090eag_af_dtt_seed42: 0.931 s, 27.69 node-h, 110.8 GPU-h |
| dttknobs_s2_exact_seed43 | 1.353 (1.331-1.393) | 102467 | 85.3 | 10 | 39.71 | 158.8 | r090eag_af_dtt_seed43: 0.937 s, 27.51 node-h, 110.1 GPU-h |
| dttknobs_s2_noscstd_seed42 | 0.574 (0.560-0.587) | 100981 | 41.3 | 5 | 16.77 | 67.1 | r090fix_af_dtt_seed42: 0.566 s, 16.5 node-h, 66.0 GPU-h |
| dttknobs_s2_noscstd_seed43 | 0.571 (0.557-0.584) | 100543 | 41.0 | 6 | 16.55 | 66.2 | r090fix_af_dtt_seed43: 0.575 s, 16.75 node-h, 67.0 GPU-h |

Per arm (two seeds): K-THETA2 216.4 GPU-h; K-THETA05 217.5 GPU-h; K-B1 204.2 GPU-h; K-BASIS8 216.9 GPU-h; K-EXACT 317.6 GPU-h; DTT-NOSC-STD 133.3 GPU-h.
