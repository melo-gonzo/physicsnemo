# SLICES-ALL reading, arm LUG: slicesall_lug_dtt_s128 against r090fix_lug_dtt (600 cases; native decides)

| protocol | field | control (s42, s43) | arm (s42, s43) | arm ÷ control | D | s_c, s_a | 2 SE | seed-matched | label | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| native | von_mises_l2 | 0.00202682 (0.00221933, 0.00183431) | 0.0019582 (0.00206216, 0.00185425) | 0.966 | -0.0344 | 0.1347, 0.0751 | 0.2182 | 0.929, 1.011 | **no change** | none |
| posed | von_mises_l2 | 0.00350931 (0.00339135, 0.00362727) | 0.00340146 (0.00374039, 0.00306253) | 0.969 | -0.0312 | 0.0476, 0.1414 | 0.2110 | 1.103, 0.844 | **no change** | none |

Arm clean (every native field 'no change', no up flag): **True**

Beside (native, seed mean per case): von_mises_l2: median arm ÷ control 0.952, arm better on 56 % of cases
posed ÷ native: ctl von_mises_l2 1.5281 / 1.9775; arm von_mises_l2 1.8138 / 1.6516
area == native exactly: {'ctl': False, 'arm': False} (degenerate, no claim)
