# SLICES-ALL reading, arm AF: slicesall_af_dtt_s128 against r090eag_af_dtt (200 cases; native decides)

| protocol | field | control (s42, s43) | arm (s42, s43) | arm ÷ control | D | s_c, s_a | 2 SE | seed-matched | label | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| native | pressure_l2 | 0.0124084 (0.012428, 0.0123887) | 0.012761 (0.0138019, 0.01172) | 1.028 | +0.0280 | 0.0022, 0.1156 | 0.1635 | 1.111, 0.946 | **no change** | none |
| native | velocity_l2 | 0.00717297 (0.00737341, 0.00697252) | 0.007242 (0.0080583, 0.0064257) | 1.010 | +0.0096 | 0.0395, 0.1601 | 0.2332 | 1.093, 0.922 | **no change** | none |
| native | nut_l2 | 0.0205977 (0.0207112, 0.0204842) | 0.0210885 (0.020981, 0.0211959) | 1.024 | +0.0235 | 0.0078, 0.0072 | 0.0150 | 1.013, 1.035 | **no change** | none |
| posed | pressure_l2 | 0.0124324 (0.012449, 0.0124158) | 0.0127825 (0.0138194, 0.0117457) | 1.028 | +0.0278 | 0.0019, 0.1150 | 0.1626 | 1.110, 0.946 | **no change** | none |
| posed | velocity_l2 | 0.00718049 (0.00738044, 0.00698055) | 0.00724953 (0.00806515, 0.0064339) | 1.010 | +0.0096 | 0.0394, 0.1598 | 0.2327 | 1.093, 0.922 | **no change** | none |
| posed | nut_l2 | 0.0206001 (0.0207131, 0.0204872) | 0.0210915 (0.0209849, 0.0211981) | 1.024 | +0.0236 | 0.0078, 0.0071 | 0.0149 | 1.013, 1.035 | **no change** | none |

Arm clean (every native field 'no change', no up flag): **True**

Beside (native, seed mean per case): pressure_l2: median arm ÷ control 0.925, arm better on 59 % of cases; velocity_l2: median arm ÷ control 1.014, arm better on 38 % of cases; nut_l2: median arm ÷ control 0.922, arm better on 60 % of cases
posed ÷ native: ctl pressure_l2 1.0017 / 1.0022, velocity_l2 1.0010 / 1.0012, nut_l2 1.0001 / 1.0001; arm pressure_l2 1.0013 / 1.0022, velocity_l2 1.0009 / 1.0013, nut_l2 1.0002 / 1.0001
area == native exactly: {'ctl': False, 'arm': False} (degenerate, no claim)
