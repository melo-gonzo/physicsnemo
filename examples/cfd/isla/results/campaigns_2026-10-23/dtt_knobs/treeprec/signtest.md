# DK-TREEPREC: paired sign test of T against R (per case; `python3 signtest_treeprec.py raw`)

z = (k - n/2) / sqrt(n/4), k = cases where T's error > R's, n = cases without a tie; p exact two-sided binomial.

| run | protocol | field | T worse / cases (ties) | z | p (two-sided) | per-case T/R median | mean T / mean R |
|---|---|---|---|---|---|---|---|
| r090eag_af_dtt_seed42 | native | pressure | 99 / 200 (0) | -0.14 | 9.4e-01 | 0.9997 | 0.9984 |
| r090eag_af_dtt_seed42 | native | velocity | 132 / 200 (0) | 4.53 | 7.1e-06 | 1.0022 | 1.0017 |
| r090eag_af_dtt_seed42 | native | nut | 134 / 200 (0) | 4.81 | 1.7e-06 | 1.0080 | 1.0068 |
| r090eag_af_dtt_seed42 | posed | pressure | 150 / 200 (0) | 7.07 | 8.4e-13 | 1.0373 | 1.0414 |
| r090eag_af_dtt_seed42 | posed | velocity | 189 / 200 (0) | 12.59 | 5.1e-43 | 1.0296 | 1.0369 |
| r090eag_af_dtt_seed42 | posed | nut | 132 / 200 (0) | 4.53 | 7.1e-06 | 1.0121 | 1.0167 |
| r090eag_af_dtt_seed43 | native | pressure | 112 / 200 (0) | 1.70 | 1.0e-01 | 1.0030 | 1.0001 |
| r090eag_af_dtt_seed43 | native | velocity | 132 / 200 (0) | 4.53 | 7.1e-06 | 1.0020 | 1.0030 |
| r090eag_af_dtt_seed43 | native | nut | 132 / 200 (0) | 4.53 | 7.1e-06 | 1.0057 | 1.0068 |
| r090eag_af_dtt_seed43 | posed | pressure | 134 / 200 (0) | 4.81 | 1.7e-06 | 1.0249 | 1.0237 |
| r090eag_af_dtt_seed43 | posed | velocity | 166 / 200 (0) | 9.33 | 4.6e-22 | 1.0134 | 1.0133 |
| r090eag_af_dtt_seed43 | posed | nut | 134 / 200 (0) | 4.81 | 1.7e-06 | 1.0099 | 1.0079 |
| r090fix_lug_dtt_seed42 | native | von_mises | 365 / 600 (0) | 5.31 | 1.2e-07 | 1.0033 | 1.0033 |
| r090fix_lug_dtt_seed42 | posed | von_mises | 552 / 600 (0) | 20.58 | 1.4e-109 | 1.7570 | 1.9507 |
| r090fix_lug_dtt_seed43 | native | von_mises | 343 / 600 (0) | 3.51 | 5.1e-04 | 1.0025 | 1.0039 |
| r090fix_lug_dtt_seed43 | posed | von_mises | 496 / 600 (0) | 16.00 | 3.8e-62 | 1.6598 | 1.8538 |
