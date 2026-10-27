| lane (PRICE) | key | seeds | steps x epochs | compiled step [s] | eager step [s] | overhead | GPU-h all seeds compiled | eager | eager - compiled | LANES now | proposed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DrivAerML Volume | vol | 2 | 108 x 500 | 1.1547 | 1.3585 | 1.119 | 155.1 | 182.4 | +27.4 | compiled | eager |
| AeroZoo full | az | 2 | 678 x 250 | 0.8461 | 1.0717 | 1.142 | 364.0 | 461.0 | +97.0 | compiled | eager |
| AeroZoo nonbox | az | 2 | 597 x 250 | 0.7810 | 0.9893 | 1.101 | 285.2 | 361.3 | +76.0 | compiled | eager |
| AeroZoo reynolds | az | 2 | 529 x 250 | 0.8337 | 1.0560 | 1.118 | 273.9 | 347.0 | +73.0 | compiled | eager |
| AeroZoo blobs | az | 2 | 539 x 250 | 0.8417 | 1.0661 | 1.123 | 283.0 | 358.5 | +75.5 | compiled | eager |
| AeroZoo bank_zero_to_cad | az | 2 | 760 x 250 | 0.8495 | 1.0761 | 1.176 | 421.8 | 534.3 | +112.5 | compiled | eager |
| AeroZoo profile_crossflow | az | 2 | 705 x 250 | 0.8649 | 1.0956 | 1.096 | 371.3 | 470.3 | +99.0 | compiled | eager |
| AeroZoo profile_dual_inlet | az | 2 | 718 x 250 | 0.8664 | 1.0974 | 1.109 | 383.3 | 485.5 | +102.2 | compiled | eager |
| AirfRANS | af | 2 | 200 x 500 | 0.3317 | 0.4276 | 1.038 | 76.5 | 98.6 | +22.1 | compiled | eager |
| lug | lug | 2 | 600 x 500 | 0.2848 | 0.3626 | 1.130 | 214.5 | 273.2 | +58.6 | compiled | eager |
| Tensile2d (eager) | t2d | 4 | 100 x 500 | 0.2769 | 0.3689 | 1.060 | 65.2 | 86.9 | +21.7 | eager | eager |
| DrivAerML Surface | drs | 2 | 108 x 500 | 0.2379 | 0.3712 | 1.102 | 31.5 | 49.1 | +17.6 | compiled | eager |
| SHIFT-SUV | suv | 2 | 198 x 500 | 0.2421 | 0.3746 | 1.073 | 57.2 | 88.4 | +31.3 | compiled | eager |
| HiLift | hl | 2 | 127 x 500 | 0.2397 | 0.3653 | 1.266 | 42.8 | 65.3 | +22.4 | compiled | eager |
| DeepJEB++ | djb | 2 | 12451 x 40 | 0.1183 | 0.1784 | 1.292 | 169.2 | 255.1 | +85.9 | compiled | compiled |

Training GPU-h of the 32 ISLA lanes priced on these smokes: LANES as now 3216, proposed 4031, change +815.
