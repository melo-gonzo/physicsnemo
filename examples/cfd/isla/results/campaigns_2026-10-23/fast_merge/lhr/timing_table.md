| case | Q / F | version | geometry call (ms) | training step (ms) | step / base | step / GEO-FAST base | peak GiB | kernel |
|---|---|---|---|---|---|---|---|---|
| AirfRANS 2-D | 100000 / 1,018 | base (0.9.0) | 11.0 / 12.4 | 362 / 368 | 1.00x | 1.00x | 35.0 | none |
| AirfRANS 2-D | 100000 / 1,018 | head, reference | 29.7 / 30.2 / 32.2 | 386 / 386 / 394 | 1.07x | 1.07x | 34.4 | reference |
| AirfRANS 2-D | 100000 / 1,018 | head, auto | 30.8 / 29.3 / 29.9 | 392 / 458 / 386 | 1.13x | 1.13x | 34.4 | fast |
| AirfRANS 2-D | | GEO-FAST base (363375) | 13.1 / 10.2 | 367 / 361 | | 1.00x | | |
| lug | 9808 / 6,488 | base (0.9.0) | 7.4 / 8.4 | 133 / 149 | 1.00x | 1.02x | 8.1 | none |
| lug | 9808 / 6,488 | head, reference | 44.6 / 43.5 / 46.6 | 182 / 177 / 183 | 1.28x | 1.30x | 8.1 | reference |
| lug | 9808 / 6,488 | head, auto | 38.3 / 35.2 / 36.7 | 177 / 165 / 173 | 1.22x | 1.24x | 8.1 | fast |
| lug | | GEO-FAST base (363375) | 7.5 / 7.3 | 141 / 136 | | 1.00x | | |
| AeroZoo | 200000 / 501,802 | base (0.9.0) | 18.5 / 17.7 | 506 / 505 | 1.00x | 1.00x | 126.3 | none |
| AeroZoo | 200000 / 501,802 | head, reference | 100.9 / 105.0 / 116.2 | 588 / 595 / 616 | 1.19x | 1.19x | 126.3 | reference |
| AeroZoo | 200000 / 501,802 | head, auto | 86.2 / 79.7 / 83.0 | 593 / 565 / 578 | 1.15x | 1.15x | 126.3 | fast |
| AeroZoo | | GEO-FAST base (363375) | 18.2 / 17.1 | 505 / 503 | | 1.00x | | |
| DrivAerML Volume, vehicle | 200000 / 16,953,702 | base (0.9.0) | 97.4 / 98.6 | 567 / 569 | 1.00x | 0.99x | 128.4 | none |
| DrivAerML Volume, vehicle | 200000 / 16,953,702 | head, reference | 367.3 / 371.6 / 381.3 | 854 / 868 / 877 | 1.53x | 1.51x | 128.4 | reference |
| DrivAerML Volume, vehicle | 200000 / 16,953,702 | head, auto | 312.5 / 302.9 / 303.9 | 796 / 778 / 796 | 1.39x | 1.38x | 128.4 | fast |
| DrivAerML Volume, vehicle | | GEO-FAST base (363375) | 97.5 / 98.5 | 570 / 577 | | 1.00x | | |
| DrivAerML Volume, whole boundary | 200000 / 16,958,756 | base (0.9.0) | 2426.5 / 2414.8 | 2901 / 2894 | 1.00x | 0.99x | 128.4 | none |
| DrivAerML Volume, whole boundary | 200000 / 16,958,756 | head, reference | 425.1 / 437.7 / 436.3 | 923 / 932 / 937 | 0.32x | 0.32x | 128.4 | reference |
| DrivAerML Volume, whole boundary | 200000 / 16,958,756 | head, auto | 359.7 / 347.8 / 355.9 | 857 / 828 / 849 | 0.29x | 0.29x | 128.4 | fast |
| DrivAerML Volume, whole boundary | | GEO-FAST base (363375) | 2423.4 / 2440.5 | 2936 / 2935 | | 1.00x | | |
