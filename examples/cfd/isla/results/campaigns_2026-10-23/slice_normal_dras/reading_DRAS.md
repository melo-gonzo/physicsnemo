# SLICE-NORMAL-EXP reading DRAS (registered rule): slicenormalx_dras against slicenormalx_dra_ctl (48 cases; native decides)

| protocol | field | control (s42, s43) | arm (s42, s43) | arm ÷ control | D | s_c, s_a | 2 SE | seed-matched | label | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| native | pressure_l2 | 0.0523637 (0.0542982, 0.0504292) | 0.074468 (0.0758453, 0.0730908) | 1.422 | +0.3522 | 0.0523, 0.0262 | 0.0827 | 1.397, 1.449 | **harms** | up |
| native | wss_l2 | 0.0802529 (0.0822529, 0.0782528) | 0.104094 (0.110592, 0.0975964) | 1.297 | +0.2601 | 0.0353, 0.0884 | 0.1346 | 1.345, 1.247 | **harms** | up |
| posed | pressure_l2 | 0.0523661 (0.0543001, 0.050432) | 0.0744708 (0.0758478, 0.0730937) | 1.422 | +0.3521 | 0.0523, 0.0262 | 0.0826 | 1.397, 1.449 | **harms** | up |
| posed | wss_l2 | 0.0802577 (0.0822549, 0.0782604) | 0.104098 (0.110594, 0.0976018) | 1.297 | +0.2601 | 0.0352, 0.0884 | 0.1345 | 1.345, 1.247 | **harms** | up |

Native 'harms' on: ['pressure_l2', 'wss_l2']

Beside (native, seed mean per case): pressure_l2: median arm ÷ control 1.396, arm better on 0 % of cases; wss_l2: median arm ÷ control 1.293, arm better on 0 % of cases
posed ÷ native: ctl pressure_l2 1.0000 / 1.0001, wss_l2 1.0000 / 1.0001; arm pressure_l2 1.0000 / 1.0000, wss_l2 1.0000 / 1.0001
