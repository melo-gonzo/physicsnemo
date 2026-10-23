| quantity (chapters that print it) | fix-trained (e) | eager-trained (f) |
|---|---|---|
| qpop ISLA chunks4 p/v/nut (03, 04, 12, 20) | 1.329 / 1.37 / 1.405 | 1.309 / 1.364 / 1.404 |
| qpop ISLA half p/v/nut (12, 20) | 1.109 / 1.13 / 1.141 | 1.098 / 1.123 / 1.145 |
| qpop ISLA small (<=1,000 points) min / max (03, 04, 12, 20) | 4.648 / 159.5 | 4.456 / 125.6 |
| qpop ISLA far p/v/nut (12) | 97.72 / 54.32 / 131.2 | 120.3 / 74.76 / 105.3 |
| qpop ISLA wall p/v/nut | 18.88 / 29.11 / 23.69 | 19.34 / 30.03 / 22.29 |
| qpop ISLA same p/v/nut | 0.9518 / 0.9453 / 0.9485 | 0.9542 / 0.9518 / 0.9424 |
| qpop ISLA-DTT largest abs(r - 1), all modes (04, 05, 09, 12, 20) | 4.45e-06 | 4.08e-06 |
| remesh ISLA largest abs(r - 1) on refine2/4/31 (12 remesh_dev; 20 dev(AF, refine)) | 1.12e-04 | 1.15e-04 |
| remesh exact_dev, AirfRANS refine + Volume tok100k/400k/subdiv4 (04) | 0.005191 | 0.005191 |
| remesh ISLA tok800/600/400/200 pressure (04, 12, 20) | 1.023 / 1.317 / 2.408 / 5.316 | 1.022 / 1.319 / 2.297 / 5.222 |
| remesh ISLA redraw400 / tok400 pressure (12, 20) | 2.454 / 2.408 | 2.432 / 2.297 |
| remesh ISLA d(redraw400, tok400) pressure median, s42 / s43 (20) | 0.02926 / 0.03174 | 0.03158 / 0.03261 |
| remesh ISLA d(tok400, native) pressure median, s42 / s43 (20) | 0.01577 / 0.01876 | 0.01693 / 0.01764 |
| remesh ISLA decim400 / decim200 velocity | 3.449 / 7.627 | 3.479 / 7.677 |
| remesh ISLA-DTT refine2/4/31 pressure (04 t1, 05 DTT_REFINE) | 1.01 / 1.039 / 1.07 | 1.017 / 1.057 / 1.059 |
| remesh ISLA-DTT largest r on refine2/4/31, any field (12 dtt_ref) | 1.07 | 1.059 |
| remesh ISLA-DTT tok800/600/400/200 pressure | 1.081 / 1.619 / 3.062 / 6.864 | 1.06 / 1.566 / 2.72 / 6.234 |
| remesh ISLA-DTT decim400 / decim200 velocity | 3.196 / 7.037 | 3.103 / 6.925 |
| remesh n_cells median, seed 42 (04 n_af) | 1015 | 1015 |

### AirfRANS rows of tbl-query-population / tbl-airfrans-qpop (ISLA p / v / nut seed means; ISLA-DTT largest abs(r - 1))

| mode | ISLA fix | ISLA eag | DTT fix | DTT eag |
|---|---|---|---|---|
| half | 1.11 / 1.13 / 1.14 | 1.1 / 1.12 / 1.15 | 1.0e-07 | 3.6e-08 |
| chunks4 | 1.33 / 1.37 / 1.41 | 1.31 / 1.36 / 1.4 | 3.3e-06 | 1.4e-06 |
| sub1000 | 4.65 / 5.08 / 5.5 | 4.46 / 5 / 5.7 | 4.5e-06 | 1.5e-06 |
| sub100 | 23 / 18.9 / 33.3 | 21.4 / 18.5 / 32.5 | 3.6e-06 | 1.9e-06 |
| sub10 | 64.6 / 64.9 / 104 | 61.2 / 64.8 / 88 | 4.1e-06 | 2.7e-06 |
| sub1 | 94 / 87.6 / 159 | 90.6 / 87.8 / 126 | 3.2e-06 | 4.1e-06 |
| same | 0.952 / 0.945 / 0.949 | 0.954 / 0.952 / 0.942 | 8.0e-08 | 2.5e-08 |
| far | 97.7 / 54.3 / 131 | 120 / 74.8 / 105 | 6.7e-08 | 3.4e-08 |
| wall | 18.9 / 29.1 / 23.7 | 19.3 / 30 / 22.3 | 6.8e-08 | 3.1e-08 |
