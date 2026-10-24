| code | boundary | pass | F | call | seed (Warp) | call without seed | step | step without seed |
|---|---|---|---|---|---|---|---|---|
| base | body | 1 | 17244760 | 100.3 ms | 55.3 ms | 45.9 ms | 573 ms (14) | 518 ms |
| base | body | 2 | 17244760 | 100.1 ms | 54.7 ms | 44.7 ms | 573 ms (14) | 519 ms |
| base | whole | 1 | 17249814 | 2428.7 ms | 2385.3 ms | 45.2 ms | 2894 ms (14) | 509 ms |
| base | whole | 2 | 17249814 | 2432.1 ms | 2389.2 ms | 44.7 ms | 2906 ms (14) | 516 ms |
| geo | body | 1 | 17244760 | 370.9 ms | 55.0 ms | 316.3 ms | 858 ms (14) | 803 ms |
| geo | body | 2 | 17244760 | 376.6 ms | 54.6 ms | 326.2 ms | 874 ms (14) | 819 ms |
| geo | whole | 1 | 17249814 | 2804.9 ms | 2402.4 ms | 382.6 ms | 3309 ms (14) | 907 ms |
| geo | whole | 2 | 17249814 | 2801.5 ms | 2417.8 ms | 385.4 ms | 3304 ms (14) | 886 ms |

| code | step body | step whole | ratio | added | step without seed, body | whole | ratio | added |
|---|---|---|---|---|---|---|---|---|
| base | 573 ms | 2900 ms | 5.06x | +2327 ms | 518 ms | 513 ms | 0.99x | -6 ms |
| geo | 866 ms | 3307 ms | 3.82x | +2441 ms | 811 ms | 896 ms | 1.11x | +85 ms |
