# SLICE-NORMAL-EXP reading DRA (registered rule): slicenormalx_dra_m against slicenormalx_dra_ctl (48 cases; native decides)

| protocol | field | control (s42, s43) | arm (s42, s43) | arm ÷ control | D | s_c, s_a | 2 SE | seed-matched | label | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| native | pressure_l2 | 0.0523637 (0.0542982, 0.0504292) | 0.0610951 (0.0641129, 0.0580773) | 1.167 | +0.1542 | 0.0523, 0.0699 | 0.1234 | 1.181, 1.152 | **harms** | up |
| native | wss_l2 | 0.0802529 (0.0822529, 0.0782528) | 0.0899414 (0.0928311, 0.0870517) | 1.121 | +0.1140 | 0.0353, 0.0455 | 0.0813 | 1.129, 1.112 | **harms** | up |
| posed | pressure_l2 | 0.0523661 (0.0543001, 0.050432) | 0.0610977 (0.0641157, 0.0580797) | 1.167 | +0.1542 | 0.0523, 0.0699 | 0.1234 | 1.181, 1.152 | **harms** | up |
| posed | wss_l2 | 0.0802577 (0.0822549, 0.0782604) | 0.0899466 (0.0928362, 0.087057) | 1.121 | +0.1140 | 0.0352, 0.0454 | 0.0813 | 1.129, 1.112 | **harms** | up |

Native 'harms' on: ['pressure_l2', 'wss_l2']

Beside (native, seed mean per case): pressure_l2: median arm ÷ control 1.149, arm better on 0 % of cases; wss_l2: median arm ÷ control 1.121, arm better on 0 % of cases
posed ÷ native: ctl pressure_l2 1.0000 / 1.0001, wss_l2 1.0000 / 1.0001; arm pressure_l2 1.0000 / 1.0000, wss_l2 1.0001 / 1.0001
