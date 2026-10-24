# DK-TREEPREC: the training forest (bf16) against the standing's (float32)

Arms (all float32 but the forest, theta 1, leaf 4): R the standing; T the forest built and walked under
torch.autocast('cuda', bfloat16), as training builds it (lists equal to a bf16 forward's, bit for bit, on the check
cases); TL T's lists with float32 node measures. d = RMS(pred_T - pred_R) / RMS(target), per case. Changed = share
of points whose pair list (as sets of cells and of node cell-sets) differs between T and R.

## Per run

| run | protocol | field | R | T | T / R | TL / R | per-case T/R median [p10, p90] | T worse | d(T,R) mean / median / max | d / err median | d(R2,R) mean | changed (mean, cases at 100%) | pairs/pt R -> T |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r090eag_af_dtt_seed42 | native | pressure | 0.012428 | 0.0124081 | 0.9984 | 0.9982 | 1.000 [0.963, 1.033] | 0.49 | 0.0017 / 0.0014 / 0.0223 | 0.137 | 4.3e-07 | 0.824, 0.10 | 21.6 -> 20.9 |
| r090eag_af_dtt_seed42 | native | velocity | 0.00737341 | 0.00738569 | 1.0017 | 1.0012 | 1.002 [0.993, 1.015] | 0.66 | 0.0008 / 0.0006 / 0.0042 | 0.109 | 2.7e-07 | 0.824, 0.10 | 21.6 -> 20.9 |
| r090eag_af_dtt_seed42 | native | nut | 0.0207112 | 0.0208523 | 1.0068 | 1.0038 | 1.008 [0.975, 1.043] | 0.67 | 0.0028 / 0.0025 / 0.0107 | 0.169 | 2.7e-07 | 0.824, 0.10 | 21.6 -> 20.9 |
| r090eag_af_dtt_seed42 | posed | pressure | 0.012449 | 0.012964 | 1.0414 | 1.0393 | 1.037 [0.966, 1.179] | 0.75 | 0.0040 / 0.0035 / 0.0133 | 0.354 | 4.3e-07 | 0.963, 0.79 | 21.6 -> 24.4 |
| r090eag_af_dtt_seed42 | posed | velocity | 0.00738044 | 0.00765261 | 1.0369 | 1.0364 | 1.030 [1.004, 1.102] | 0.94 | 0.0022 / 0.0018 / 0.0084 | 0.317 | 5.8e-07 | 0.963, 0.79 | 21.6 -> 24.4 |
| r090eag_af_dtt_seed42 | posed | nut | 0.0207131 | 0.0210596 | 1.0167 | 1.0134 | 1.012 [0.970, 1.083] | 0.66 | 0.0035 / 0.0029 / 0.0310 | 0.198 | 2.7e-07 | 0.963, 0.79 | 21.6 -> 24.4 |
| r090eag_af_dtt_seed43 | native | pressure | 0.0123887 | 0.0123898 | 1.0001 | 1.0001 | 1.003 [0.968, 1.044] | 0.56 | 0.0017 / 0.0015 / 0.0205 | 0.138 | 3.9e-07 | 0.824, 0.10 | 21.6 -> 20.9 |
| r090eag_af_dtt_seed43 | native | velocity | 0.00697252 | 0.00699321 | 1.0030 | 1.0024 | 1.002 [0.995, 1.013] | 0.66 | 0.0008 / 0.0007 / 0.0050 | 0.114 | 2.7e-07 | 0.824, 0.10 | 21.6 -> 20.9 |
| r090eag_af_dtt_seed43 | native | nut | 0.0204842 | 0.0206241 | 1.0068 | 1.0033 | 1.006 [0.985, 1.042] | 0.66 | 0.0025 / 0.0023 / 0.0078 | 0.179 | 2.7e-07 | 0.824, 0.10 | 21.6 -> 20.9 |
| r090eag_af_dtt_seed43 | posed | pressure | 0.0124158 | 0.0127105 | 1.0237 | 1.0225 | 1.026 [0.956, 1.126] | 0.67 | 0.0036 / 0.0032 / 0.0179 | 0.337 | 3.9e-07 | 0.963, 0.79 | 21.6 -> 24.4 |
| r090eag_af_dtt_seed43 | posed | velocity | 0.00698055 | 0.00707348 | 1.0133 | 1.0131 | 1.013 [0.994, 1.043] | 0.83 | 0.0016 / 0.0014 / 0.0046 | 0.261 | 5.7e-07 | 0.963, 0.79 | 21.6 -> 24.4 |
| r090eag_af_dtt_seed43 | posed | nut | 0.0204872 | 0.0206497 | 1.0079 | 1.0046 | 1.010 [0.967, 1.073] | 0.67 | 0.0032 / 0.0027 / 0.0198 | 0.220 | 2.7e-07 | 0.963, 0.79 | 21.6 -> 24.4 |
| r090fix_lug_dtt_seed42 | native | von_mises | 0.00221933 | 0.00222676 | 1.0033 | 1.0026 | 1.003 [0.988, 1.023] | 0.61 | 0.0005 / 0.0005 / 0.0018 | 0.233 | 1.1e-07 | 0.992, 0.78 | 118.4 -> 118.1 |
| r090fix_lug_dtt_seed42 | posed | von_mises | 0.00339135 | 0.00661543 | 1.9507 | 1.9506 | 1.766 [1.045, 3.428] | 0.92 | 0.0061 / 0.0053 / 0.0154 | 1.599 | 1.2e-07 | 1.000, 0.89 | 151.3 -> 128.9 |
| r090fix_lug_dtt_seed43 | native | von_mises | 0.00183431 | 0.00184146 | 1.0039 | 1.0036 | 1.003 [0.983, 1.027] | 0.57 | 0.0004 / 0.0004 / 0.0021 | 0.249 | 1.1e-07 | 0.992, 0.78 | 118.4 -> 118.1 |
| r090fix_lug_dtt_seed43 | posed | von_mises | 0.00362727 | 0.00672412 | 1.8538 | 1.8538 | 1.662 [0.860, 3.363] | 0.83 | 0.0065 / 0.0056 / 0.0138 | 1.619 | 1.1e-07 | 1.000, 0.89 | 151.3 -> 128.9 |

## Two-seed means against interacting ISLA (ISLA: DK-GAP's float32 native two-seed mean)

| family | protocol | field | DTT R | DTT T | T / R [s42, s43] | TL / R | two-seed spread R / T | DTT R / ISLA | DTT T / ISLA |
|---|---|---|---|---|---|---|---|---|---|
| af | native | pressure | 0.0124084 | 0.0123989 | 0.9992 [0.9984, 1.0001] | 0.9991 | 0.003 / 0.001 | 0.994 | 0.994 |
| af | native | velocity | 0.00717297 | 0.00718945 | 1.0023 [1.0017, 1.0030] | 1.0018 | 0.056 / 0.055 | 1.129 | 1.131 |
| af | native | nut | 0.0205977 | 0.0207382 | 1.0068 [1.0068, 1.0068] | 1.0035 | 0.011 / 0.011 | 1.342 | 1.351 |
| af | posed | pressure | 0.0124324 | 0.0128372 | 1.0326 [1.0414, 1.0237] | 1.0309 | 0.003 / 0.020 | 0.996 | 1.029 |
| af | posed | velocity | 0.00718049 | 0.00736305 | 1.0254 [1.0369, 1.0133] | 1.0251 | 0.056 / 0.079 | 1.130 | 1.159 |
| af | posed | nut | 0.0206001 | 0.0208546 | 1.0124 [1.0167, 1.0079] | 1.0090 | 0.011 / 0.020 | 1.342 | 1.358 |
| lug | native | von_mises | 0.00202682 | 0.00203411 | 1.0036 [1.0033, 1.0039] | 1.0030 | 0.190 / 0.189 | 1.394 | 1.399 |
| lug | posed | von_mises | 0.00350931 | 0.00666978 | 1.9006 [1.9507, 1.8538] | 1.9006 | 0.067 / 0.016 | 2.414 | 4.588 |

## Gate and checks

- r090eag_af_dtt_seed42/native: per-case scorer max rel 4.8e-06; pressure rel vs standing 1.8e-06; velocity rel vs standing 4.4e-07; nut rel vs standing 1.1e-06; TL lists == T all cases True; R2 changed lists max 0.0e+00; check cases 3: BF == T lists [True, True, True] (same inputs [True, True, True], boxes/diameters/measures equal [True, True, True]); TMM == T [False, False, False] (same inputs [True, True, True], boxes equal [True, True, True], diameter_sq equal [False, False, False], changed vs T [0.0189, 0.0396, 0.0367]); fp32 measure recompute == R [True, True, True]
- r090eag_af_dtt_seed42/posed: per-case scorer max rel 2.4e-06; TL lists == T all cases True; R2 changed lists max 0.0e+00; check cases 3: BF == T lists [True, True, True] (same inputs [True, True, True], boxes/diameters/measures equal [True, True, True]); TMM == T [False, False, False] (same inputs [True, True, True], boxes equal [True, True, True], diameter_sq equal [False, False, False], changed vs T [0.0154, 0.0034, 0.0068]); fp32 measure recompute == R [True, True, True]
- r090eag_af_dtt_seed43/native: per-case scorer max rel 1.5e-06; pressure rel vs standing 5.6e-10; velocity rel vs standing 3.5e-07; nut rel vs standing 3.1e-07; TL lists == T all cases True; R2 changed lists max 0.0e+00; check cases 3: BF == T lists [True, True, True] (same inputs [True, True, True], boxes/diameters/measures equal [True, True, True]); TMM == T [False, False, False] (same inputs [True, True, True], boxes equal [True, True, True], diameter_sq equal [False, False, False], changed vs T [0.0189, 0.0396, 0.0367]); fp32 measure recompute == R [True, True, True]
- r090eag_af_dtt_seed43/posed: per-case scorer max rel 2.2e-06; TL lists == T all cases True; R2 changed lists max 0.0e+00; check cases 3: BF == T lists [True, True, True] (same inputs [True, True, True], boxes/diameters/measures equal [True, True, True]); TMM == T [False, False, False] (same inputs [True, True, True], boxes equal [True, True, True], diameter_sq equal [False, False, False], changed vs T [0.0154, 0.0034, 0.0068]); fp32 measure recompute == R [True, True, True]
- r090fix_lug_dtt_seed42/native: per-case scorer max rel 4.5e-06; von_mises rel vs standing 1.5e-04; TL lists == T all cases True; R2 changed lists max 0.0e+00; check cases 3: BF == T lists [True, True, True] (same inputs [True, True, True], boxes/diameters/measures equal [True, True, True]); TMM == T [False, False, False] (same inputs [True, True, True], boxes equal [True, True, True], diameter_sq equal [False, False, False], changed vs T [0.2267, 0.1775, 0.1803]); fp32 measure recompute == R [True, True, True]
- r090fix_lug_dtt_seed42/posed: per-case scorer max rel 6.4e-06; TL lists == T all cases True; R2 changed lists max 0.0e+00; check cases 3: BF == T lists [True, True, True] (same inputs [True, True, True], boxes/diameters/measures equal [True, True, True]); TMM == T [False, False, False] (same inputs [True, True, True], boxes equal [True, True, True], diameter_sq equal [False, False, False], changed vs T [0.2449, 0.2109, 0.2009]); fp32 measure recompute == R [True, True, True]
- r090fix_lug_dtt_seed43/native: per-case scorer max rel 5.0e-06; von_mises rel vs standing 1.7e-04; TL lists == T all cases True; R2 changed lists max 0.0e+00; check cases 3: BF == T lists [True, True, True] (same inputs [True, True, True], boxes/diameters/measures equal [True, True, True]); TMM == T [False, False, False] (same inputs [True, True, True], boxes equal [True, True, True], diameter_sq equal [False, False, False], changed vs T [0.2267, 0.1775, 0.1803]); fp32 measure recompute == R [True, True, True]
- r090fix_lug_dtt_seed43/posed: per-case scorer max rel 2.7e-06; TL lists == T all cases True; R2 changed lists max 0.0e+00; check cases 3: BF == T lists [True, True, True] (same inputs [True, True, True], boxes/diameters/measures equal [True, True, True]); TMM == T [False, False, False] (same inputs [True, True, True], boxes equal [True, True, True], diameter_sq equal [False, False, False], changed vs T [0.2449, 0.2109, 0.2009]); fp32 measure recompute == R [True, True, True]

## What the downstream code sees (first check case, AirfRANS seed 42 native)

| arm | tree_points | aabb | diameter_sq | walk_box | node_total_area | forest_measure_as_built | forest_measure_used | center | normal | vecs | autocast_at_build_call |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | None | [False, 'float16'] |
| T | bfloat16 | bfloat16 | float32 | float32 | bfloat16 | bfloat16 | bfloat16 | float32 | float32 | None | [False, 'float16'] |
| TMM | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | float32 | float32 | None | [False, 'float16'] |
| BF | bfloat16 | bfloat16 | float32 | float32 | bfloat16 | bfloat16 | bfloat16 | float32 | float32 | None | [True, 'bfloat16'] |
