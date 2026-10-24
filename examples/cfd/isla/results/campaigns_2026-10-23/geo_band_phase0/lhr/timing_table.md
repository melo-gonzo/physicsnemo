| case | Q / F | version | geometry call (ms) | training step (ms) | step / base | step / FAST-MERGE base | peak GiB | kernel |
|---|---|---|---|---|---|---|---|---|
| AirfRANS 2-D | 100000 / 1,018 | base (0.9.0) | 10.4 / 15.0 | 373 / 375 | 1.00x | 1.03x | 34.4 | none |
| AirfRANS 2-D | 100000 / 1,018 | head, reference | 31.4 / 30.7 / 29.5 | 358 / 356 / 353 | 0.95x | 0.97x | 34.4 | reference |
| AirfRANS 2-D | 100000 / 1,018 | head, auto | 29.6 / 37.4 / 30.0 | 355 / 377 / 355 | 0.97x | 0.99x | 34.4 | fast |
| AirfRANS 2-D | | FAST-MERGE base (365898) | 11.0 / 12.4 | 362 / 368 | | 1.00x | | |
| lug | 9808 / 6,488 | base (0.9.0) | 8.3 / 8.1 | 144 / 148 | 1.00x | 1.03x | 8.1 | none |
| lug | 9808 / 6,488 | head, reference | 76.5 / 77.4 / 73.7 | 207 / 204 / 194 | 1.38x | 1.43x | 8.1 | reference |
| lug | 9808 / 6,488 | head, auto | 66.8 / 88.6 / 73.7 | 190 / 249 / 202 | 1.47x | 1.52x | 8.1 | fast |
| lug | | FAST-MERGE base (365898) | 7.4 / 8.4 | 133 / 149 | | 1.00x | | |
| AeroZoo | 200000 / 501,802 | base (0.9.0) | 19.7 / 18.6 | 521 / 502 | 1.00x | 1.01x | 126.3 | none |
| AeroZoo | 200000 / 501,802 | head, reference | - / 118.7 / 124.1 | - / 595 / 604 | 1.17x | 1.19x | 126.3 | reference |
| AeroZoo | 200000 / 501,802 | head, auto | 101.5 / 118.7 / 103.3 | 578 / 626 / 587 | 1.17x | 1.18x | 126.3 | fast |
| AeroZoo | | FAST-MERGE base (365898) | 18.5 / 17.7 | 506 / 505 | | 1.00x | | |
| DrivAerML Volume, vehicle | 200000 / 16,953,702 | base (0.9.0) | 97.2 / 97.3 | 569 / 577 | 1.00x | 1.01x | 128.4 | none |
| DrivAerML Volume, vehicle | 200000 / 16,953,702 | head, reference | 433.8 / 415.6 / 425.4 | 915 / 902 / 912 | 1.59x | 1.60x | 128.4 | reference |
| DrivAerML Volume, vehicle | 200000 / 16,953,702 | head, auto | 354.6 / 384.9 / 357.6 | 837 / 873 / 842 | 1.49x | 1.50x | 128.4 | fast |
| DrivAerML Volume, vehicle | | FAST-MERGE base (365898) | 97.4 / 98.6 | 567 / 569 | | 1.00x | | |
| DrivAerML Volume, whole boundary | 200000 / 16,958,756 | base (0.9.0) | 2435.5 / 2465.7 | 2908 / 2924 | 1.00x | 1.01x | 128.4 | none |
| DrivAerML Volume, whole boundary | 200000 / 16,958,756 | head, reference | 504.7 / 494.9 / 487.1 | 988 / 990 / 973 | 0.34x | 0.34x | 128.4 | reference |
| DrivAerML Volume, whole boundary | 200000 / 16,958,756 | head, auto | 422.1 / 443.0 / 412.5 | 911 / 959 / 900 | 0.32x | 0.32x | 128.4 | fast |
| DrivAerML Volume, whole boundary | | FAST-MERGE base (365898) | 2426.5 / 2414.8 | 2901 / 2894 | | 1.00x | | |
