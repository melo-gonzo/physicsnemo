# SLICES-ALL reading, arm DR: slicesall_dr_isla_s128 against r090_dr_isla (48 cases; native decides)

| protocol | field | control (s42, s43) | arm (s42, s43) | arm ÷ control | D | s_c, s_a | 2 SE | seed-matched | label | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| native | pressure_l2 | 0.0411344 (0.0411605, 0.0411084) | 0.041383 (0.041297, 0.0414691) | 1.006 | +0.0060 | 0.0009, 0.0029 | 0.0043 | 1.003, 1.009 | **no change** | none |
| native | wss_l2 | 0.0662212 (0.0661942, 0.0662482) | 0.0665341 (0.0664878, 0.0665805) | 1.005 | +0.0047 | 0.0006, 0.0010 | 0.0016 | 1.004, 1.005 | **no change** | none |
| posed | pressure_l2 | 0.0411353 (0.0411611, 0.0411095) | 0.0413843 (0.041298, 0.0414706) | 1.006 | +0.0060 | 0.0009, 0.0029 | 0.0044 | 1.003, 1.009 | **no change** | none |
| posed | wss_l2 | 0.066225 (0.0661972, 0.0662528) | 0.0665392 (0.0664929, 0.0665856) | 1.005 | +0.0047 | 0.0006, 0.0010 | 0.0016 | 1.004, 1.005 | **no change** | none |
| area | pressure_l2 | 0.0413107 (0.0412269, 0.0413945) | 0.0415257 (0.0414334, 0.0416179) | 1.005 | +0.0052 | 0.0029, 0.0031 | 0.0060 | 1.005, 1.005 | **no change** | none |
| area | wss_l2 | 0.0671495 (0.0670125, 0.0672866) | 0.0674296 (0.0672736, 0.0675855) | 1.004 | +0.0042 | 0.0029, 0.0033 | 0.0062 | 1.004, 1.004 | **no change** | none |

Arm clean (every native field 'no change', no up flag): **True**

Beside (native, seed mean per case): pressure_l2: median arm ÷ control 1.022, arm better on 42 % of cases; wss_l2: median arm ÷ control 1.014, arm better on 50 % of cases
posed ÷ native: ctl pressure_l2 1.0000 / 1.0000, wss_l2 1.0000 / 1.0001; arm pressure_l2 1.0000 / 1.0000, wss_l2 1.0001 / 1.0001
Eager guard (arm compiled vs eager, max per-case relative difference): {'42': 1.5827998380846253e-06, '43': 2.7729071061521893e-06} -> pass True
