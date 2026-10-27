# SLICE-NORMAL-EXP reading MDRA (marginal effect of (s), beside (not a verdict)): slicenormalx_dras against slicenormalx_dra_m (48 cases; native decides)

| protocol | field | control (s42, s43) | arm (s42, s43) | arm ÷ control | D | s_c, s_a | 2 SE | seed-matched | label | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| native | pressure_l2 | 0.0610951 (0.0641129, 0.0580773) | 0.074468 (0.0758453, 0.0730908) | 1.219 | +0.1979 | 0.0699, 0.0262 | 0.1056 | 1.183, 1.259 | **harms** | up |
| native | wss_l2 | 0.0899414 (0.0928311, 0.0870517) | 0.104094 (0.110592, 0.0975964) | 1.157 | +0.1461 | 0.0455, 0.0884 | 0.1406 | 1.191, 1.121 | **harms** | up |
| posed | pressure_l2 | 0.0610977 (0.0641157, 0.0580797) | 0.0744708 (0.0758478, 0.0730937) | 1.219 | +0.1979 | 0.0699, 0.0262 | 0.1056 | 1.183, 1.259 | **harms** | up |
| posed | wss_l2 | 0.0899466 (0.0928362, 0.087057) | 0.104098 (0.110594, 0.0976018) | 1.157 | +0.1461 | 0.0454, 0.0884 | 0.1405 | 1.191, 1.121 | **harms** | up |

Native 'harms' on: ['pressure_l2', 'wss_l2']

Beside (native, seed mean per case): pressure_l2: median arm ÷ control 1.216, arm better on 0 % of cases; wss_l2: median arm ÷ control 1.154, arm better on 0 % of cases
posed ÷ native: ctl pressure_l2 1.0000 / 1.0000, wss_l2 1.0001 / 1.0001; arm pressure_l2 1.0000 / 1.0000, wss_l2 1.0000 / 1.0001
