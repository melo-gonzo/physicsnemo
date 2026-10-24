| code | model | boundary | pass | Q / F | geometry call | train step (n) | peak mem GB |
|---|---|---|---|---|---|---|---|
| base | isla | body | 1 | 200000 / 17244760 | 97.2 ms | 566 ms (14) | 140.5 |
| base | isla | body | 2 | 200000 / 17244760 | 99.0 ms | 574 ms (14) | 140.5 |
| base | isla | whole | 1 | 200000 / 17249814 | 2428.1 ms | 2900 ms (14) | 140.6 |
| base | isla | whole | 2 | 200000 / 17249814 | 2413.8 ms | 2894 ms (14) | 140.6 |
| base | dtt | body | 1 | - | - | - | - |
| base | dtt | body | 2 | - | - | - | - |
| base | dtt | whole | 1 | - | - | - | - |
| base | dtt | whole | 2 | - | - | - | - |
| geo | isla | body | 1 | 200000 / 17244760 | 364.5 ms | 849 ms (14) | 140.8 |
| geo | isla | body | 2 | 200000 / 17244760 | 363.1 ms | 851 ms (14) | 140.8 |
| geo | isla | whole | 1 | 200000 / 17249814 | 2758.4 ms | 3308 ms (14) | 141.1 |
| geo | isla | whole | 2 | 200000 / 17249814 | 2778.3 ms | 3309 ms (14) | 141.1 |
| geo | dtt | body | 1 | - | - | - | - |
| geo | dtt | body | 2 | - | - | - | - |
| geo | dtt | whole | 1 | - | - | - | - |
| geo | dtt | whole | 2 | - | - | - | - |

| code | model | geometry body | geometry whole | ratio | step body | step whole | ratio | added step |
|---|---|---|---|---|---|---|---|---|
| base | isla | 98.1 ms | 2421.0 ms | 24.68x | 570 ms | 2897 ms | 5.08x | +2327 ms |
| geo | isla | 363.8 ms | 2768.3 ms | 7.61x | 850 ms | 3309 ms | 3.89x | +2459 ms |

branch_isla_volume (c1e07b7fd): {'geo_ms': 2417.6979059993755, 'step_ms': 2895.5272215098375, 'n': 14, 'F': 17249814, 'Q': 200000, 'mem': 140.6328125}
