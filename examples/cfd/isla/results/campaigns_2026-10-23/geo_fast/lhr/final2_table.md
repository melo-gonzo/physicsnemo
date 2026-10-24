| case | Q / F | version | geometry call (ms) | training step (ms) | step / base | peak GiB | kernel |
|---|---|---|---|---|---|---|---|
| AirfRANS 2-D | 100000 / 1,018 | base | 13.1 / 10.2 | 367 / 361 | 1.00x | 34.4 | none |
| AirfRANS 2-D | 100000 / 1,018 | f13eb5ce0 | 30.5 / 30.5 | 463 / 382 | 1.16x | 34.4 | none |
| AirfRANS 2-D | 100000 / 1,018 | head, reference | 30.5 / 30.8 | 389 / 385 | 1.06x | 34.4 | reference |
| AirfRANS 2-D | 100000 / 1,018 | head, auto | 30.5 / 29.1 | 389 / 382 | 1.06x | 34.4 | fast |
| lug | 9808 / 6,488 | base | 7.5 / 7.3 | 141 / 136 | 1.00x | 8.1 | none |
| lug | 9808 / 6,488 | f13eb5ce0 | 46.3 / 42.7 | 183 / 172 | 1.28x | 8.1 | none |
| lug | 9808 / 6,488 | head, reference | 43.5 / 43.7 | 177 / 177 | 1.28x | 8.1 | reference |
| lug | 9808 / 6,488 | head, auto | 35.4 / 36.1 | 172 / 175 | 1.25x | 8.1 | fast |
| AeroZoo | 200000 / 501,802 | base | 18.2 / 17.1 | 505 / 503 | 1.00x | 126.3 | none |
| AeroZoo | 200000 / 501,802 | f13eb5ce0 | 96.2 / 94.2 | 587 / 581 | 1.16x | 126.3 | none |
| AeroZoo | 200000 / 501,802 | head, reference | 104.4 / 98.2 | 603 / 583 | 1.18x | 126.3 | reference |
| AeroZoo | 200000 / 501,802 | head, auto | 81.6 / 82.8 | 577 / 577 | 1.15x | 126.3 | fast |
| DrivAerML Volume, vehicle | 200000 / 16,953,702 | base | 97.5 / 98.5 | 570 / 577 | 1.00x | 128.4 | none |
| DrivAerML Volume, vehicle | 200000 / 16,953,702 | f13eb5ce0 | 373.6 / 372.0 | 851 / 858 | 1.49x | 128.4 | none |
| DrivAerML Volume, vehicle | 200000 / 16,953,702 | head, reference | 366.0 / 360.6 | 862 / 843 | 1.49x | 128.4 | reference |
| DrivAerML Volume, vehicle | 200000 / 16,953,702 | head, auto | 296.0 / 295.4 | 786 / 776 | 1.36x | 128.4 | fast |
| DrivAerML Volume, whole boundary | 200000 / 16,958,756 | base | 2423.4 / 2440.5 | 2936 / 2935 | 1.00x | 128.4 | none |
| DrivAerML Volume, whole boundary | 200000 / 16,958,756 | f13eb5ce0 | 2777.8 / 2758.2 | 3270 / 3247 | 1.11x | 128.4 | none |
| DrivAerML Volume, whole boundary | 200000 / 16,958,756 | head, reference | 421.7 / 423.9 | 928 / 920 | 0.31x | 128.4 | reference |
| DrivAerML Volume, whole boundary | 200000 / 16,958,756 | head, auto | 351.5 / 349.0 | 847 / 839 | 0.29x | 128.4 | fast |
