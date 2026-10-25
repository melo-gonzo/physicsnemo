Pairs whose better model differs between the registered relL2 and the spec metric (each cell names the better model under registered relL2 → under ε, under ε^N). 21 of the (split, pair, case set, statistic) rows change on at least one field.

| split | pair | case set, statistic | velocity: registered relL2 → ε_U, ε^N_U | pressure: registered relL2 → ε_p, ε^N_p |
|---|---|---|---|---|
| full | ISLA vs ISLA-DTT | stable, median | ISLA-DTT → ISLA, ISLA-DTT **changed** | ISLA-DTT → ISLA-DTT, ISLA-DTT |
| full | GeoTransolver vs Transolver | all, median | GeoTransolver → GeoTransolver, GeoTransolver | Transolver → GeoTransolver, Transolver **changed** |
| full | GeoTransolver vs Transolver | stable, median | GeoTransolver → GeoTransolver, GeoTransolver | Transolver → GeoTransolver, Transolver **changed** |
| bank_zero_to_cad | ISLA vs ISLA-DTT | all, median | ISLA-DTT → ISLA-DTT, ISLA-DTT | ISLA-DTT → ISLA, ISLA **changed** |
| bank_zero_to_cad | ISLA vs ISLA-DTT | stable, median | ISLA-DTT → ISLA, ISLA-DTT **changed** | ISLA-DTT → ISLA-DTT, ISLA **changed** |
| family_non_box | ISLA vs ISLA-DTT | all, mean | ISLA-DTT → ISLA-DTT, ISLA-DTT | ISLA → ISLA-DTT, ISLA-DTT **changed** |
| family_non_box | ISLA vs ISLA-DTT | stable, median | ISLA → ISLA-DTT, ISLA-DTT **changed** | ISLA → ISLA, ISLA |
| family_non_box | ISLA vs GeoTransolver | all, mean | ISLA → GeoTransolver, GeoTransolver **changed** | GeoTransolver → GeoTransolver, GeoTransolver |
| family_non_box | ISLA vs GeoTransolver | stable, mean | ISLA → GeoTransolver, GeoTransolver **changed** | GeoTransolver → GeoTransolver, GeoTransolver |
| family_non_box | ISLA vs GeoTransolver | stable, median | ISLA → GeoTransolver, GeoTransolver **changed** | GeoTransolver → GeoTransolver, GeoTransolver |
| family_non_box | ISLA-DTT vs GeoTransolver | all, mean | ISLA-DTT → GeoTransolver, GeoTransolver **changed** | GeoTransolver → GeoTransolver, GeoTransolver |
| family_non_box | ISLA-DTT vs GeoTransolver | all, median | ISLA-DTT → GeoTransolver, GeoTransolver **changed** | GeoTransolver → GeoTransolver, GeoTransolver |
| family_non_box | ISLA-DTT vs GeoTransolver | stable, mean | ISLA-DTT → GeoTransolver, GeoTransolver **changed** | GeoTransolver → GeoTransolver, GeoTransolver |
| family_non_box | ISLA-DTT vs GeoTransolver | stable, median | ISLA-DTT → GeoTransolver, GeoTransolver **changed** | GeoTransolver → GeoTransolver, GeoTransolver |
| profile_crossflow | ISLA vs ISLA-DTT | all, median | ISLA-DTT → ISLA, ISLA **changed** | ISLA-DTT → ISLA-DTT, ISLA-DTT |
| profile_crossflow | ISLA vs ISLA-DTT | stable, mean | ISLA-DTT → ISLA-DTT, ISLA-DTT | ISLA-DTT → ISLA-DTT, ISLA **changed** |
| profile_crossflow | ISLA vs ISLA-DTT | stable, median | ISLA-DTT → ISLA, ISLA-DTT **changed** | ISLA-DTT → ISLA, ISLA **changed** |
| reynolds | ISLA vs ISLA-DTT | all, mean | ISLA-DTT → ISLA-DTT, ISLA-DTT | ISLA-DTT → ISLA, ISLA **changed** |
| reynolds | ISLA vs ISLA-DTT | all, median | ISLA-DTT → ISLA-DTT, ISLA-DTT | ISLA-DTT → ISLA, ISLA **changed** |
| reynolds | ISLA vs ISLA-DTT | stable, mean | ISLA-DTT → ISLA-DTT, ISLA-DTT | ISLA-DTT → ISLA, ISLA **changed** |
| reynolds | ISLA vs ISLA-DTT | stable, median | ISLA-DTT → ISLA, ISLA-DTT **changed** | ISLA-DTT → ISLA, ISLA **changed** |

**Above the zero predictor (mean over all cases, seed means).**

| split | metric | zero | models above it |
|---|---|---|---|
| full | eps_p | 1.68 (free stream 1.68) | GeoTransolver (1.69), Transolver (1.69) |
| full | eps_p_near | 1.88 (free stream 1.88) | GeoTransolver (1.89), Transolver (1.88) |
| bank_zero_to_cad | eps_p | 1.78 (free stream 1.78) | GeoTransolver (1.81) |
| bank_zero_to_cad | eps_p_near | 1.92 (free stream 1.92) | GeoTransolver (1.93) |
| bank_blobs | eps_p | 1.76 (free stream 1.76) | GeoTransolver (1.8) |
| bank_blobs | eps_p_near | 1.98 (free stream 1.98) | GeoTransolver (2.01) |
| family_non_box | eps_p | 0.467 (free stream 0.467) | ISLA (interacting) (0.72), ISLA-DTT (0.706), GeoTransolver (0.479) |
| family_non_box | eps_p_near | 0.347 (free stream 0.347) | ISLA (interacting) (0.585), ISLA-DTT (0.561), GeoTransolver (0.357) |
| profile_crossflow | eps_U | 1.04 (free stream 0.855) | ISLA (interacting) (1.29), ISLA-DTT (1.26) |
| profile_crossflow | eps_p | 2.59 (free stream 2.59) | ISLA (interacting) (5.01), ISLA-DTT (4.74), GeoTransolver (2.59) |
| profile_crossflow | eps_U_near | 0.929 (free stream 0.799) | ISLA (interacting) (1.55), ISLA-DTT (1.52) |
| profile_crossflow | eps_p_near | 2.92 (free stream 2.92) | ISLA (interacting) (5.15), ISLA-DTT (5.05), GeoTransolver (2.92) |
| profile_dual_inlet | eps_p | 9.34 (free stream 9.34) | GeoTransolver (9.34) |
| profile_dual_inlet | eps_p_near | 10.1 (free stream 10.1) | GeoTransolver (10.1) |
| reynolds | eps_p | 1.75 (free stream 1.75) | GeoTransolver (1.75) |
| reynolds | eps_p_near | 1.9 (free stream 1.9) | GeoTransolver (1.9) |
