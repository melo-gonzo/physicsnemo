| code | model | boundary | pass | Q / F | geometry call | train step (n) | peak mem GB |
|---|---|---|---|---|---|---|---|
| base | isla | body | 1 | - | - | - | - |
| base | isla | body | 2 | - | - | - | - |
| base | isla | whole | 1 | - | - | - | - |
| base | isla | whole | 2 | - | - | - | - |
| base | dtt | body | 1 | 200000 / 17244760 | 97.0 ms | 1724 ms (14) | 200.5 |
| base | dtt | body | 2 | 200000 / 17244760 | 97.3 ms | 1732 ms (14) | 200.3 |
| base | dtt | whole | 1 | 200000 / 17249814 | 2419.5 ms | 4174 ms (14) | 209.9 |
| base | dtt | whole | 2 | 200000 / 17249814 | 2484.2 ms | 4309 ms (14) | 209.9 |
| geo | isla | body | 1 | - | - | - | - |
| geo | isla | body | 2 | - | - | - | - |
| geo | isla | whole | 1 | - | - | - | - |
| geo | isla | whole | 2 | - | - | - | - |
| geo | dtt | body | 1 | 200000 / 17244760 | 364.3 ms | 1981 ms (14) | 201.1 |
| geo | dtt | body | 2 | 200000 / 17244760 | 365.0 ms | 1991 ms (14) | 200.7 |
| geo | dtt | whole | 1 | 200000 / 17249814 | 2800.3 ms | 4582 ms (14) | 210.8 |
| geo | dtt | whole | 2 | 200000 / 17249814 | 2828.0 ms | 4657 ms (14) | 211.5 |

| code | model | geometry body | geometry whole | ratio | step body | step whole | ratio | added step |
|---|---|---|---|---|---|---|---|---|
| base | dtt | 97.2 ms | 2451.8 ms | 25.23x | 1728 ms | 4242 ms | 2.45x | +2514 ms |
| geo | dtt | 364.7 ms | 2814.2 ms | 7.72x | 1986 ms | 4619 ms | 2.33x | +2634 ms |

branch_isla_volume (c1e07b7fd): None
