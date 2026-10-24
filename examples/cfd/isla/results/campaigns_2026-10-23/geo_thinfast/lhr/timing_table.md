## Per pass

| case | Q / F | arm | geometry call per pass (ms) | step per pass (ms) | peak GiB | kernel |
|---|---|---|---|---|---|---|
| lug | 9808 / 6,488 | base, fast (2e9cd7fda) | 71.1 / 72.1 / 79.2 | 201 / 199 / 212 | 8.1 | fast |
| lug | 9808 / 6,488 | head, fast (b959a6735) | 35.5 / 37.1 / 40.2 | 162 / 165 / 173 | 8.1 | fast |
| lug | 9808 / 6,488 | reference | 89.1 / 77.8 | 221 / 205 | 8.1 | reference |
| AirfRANS 2-D | 100000 / 1,018 | base, fast (2e9cd7fda) | 30.7 / 31.7 / 32.7 | 354 / 357 / 359 | 34.4 | fast |
| AirfRANS 2-D | 100000 / 1,018 | head, fast (b959a6735) | 31.5 / 31.1 / 32.1 | 356 / 356 / 358 | 34.4 | fast |
| AirfRANS 2-D | 100000 / 1,018 | reference | 33.5 / 32.9 | 362 / 360 | 34.4 | reference |
| AeroZoo | 200000 / 501,802 | base, fast (2e9cd7fda) | 114.4 / 111.6 / 115.0 | 600 / 591 / 603 | 126.3 | fast |
| AeroZoo | 200000 / 501,802 | head, fast (b959a6735) | 96.3 / 91.1 / 90.9 | 582 / 579 / 582 | 126.3 | fast |
| AeroZoo | 200000 / 501,802 | reference | 134.0 / 137.7 | 626 / 628 | 126.3 | reference |
| DrivAerML vehicle | 200000 / 16,953,702 | base, fast (2e9cd7fda) | 381.8 / 387.1 / 384.3 | 862 / 871 / 870 | 128.4 | fast |
| DrivAerML vehicle | 200000 / 16,953,702 | head, fast (b959a6735) | 342.7 / 349.3 / 343.6 | 823 / 839 / 824 | 128.4 | fast |
| DrivAerML vehicle | 200000 / 16,953,702 | reference | 446.6 / 454.2 | 940 / 944 | 128.4 | reference |
| DrivAerML whole boundary | 200000 / 16,958,756 | base, fast (2e9cd7fda) | 432.4 / 433.5 / 442.3 | 929 / 927 / 935 | 128.4 | fast |
| DrivAerML whole boundary | 200000 / 16,958,756 | head, fast (b959a6735) | 404.8 / 404.1 / 400.9 | 895 / 896 / 902 | 128.4 | fast |
| DrivAerML whole boundary | 200000 / 16,958,756 | reference | 525.2 / 503.4 | 1018 / 1000 | 128.4 | reference |

## Mean over passes

| case (Q / F) | geometry call: base fast / head fast / reference [ms] | head / base (call) | step: base / head / reference [ms] | head / base (step) |
|---|--:|--:|--:|--:|
| lug (9808 / 6,488) | 74.1 / 37.6 / 83.5 | 0.51 | 204 / 166 / 213 | 0.82 |
| AirfRANS 2-D (100000 / 1,018) | 31.7 / 31.6 / 33.2 | 1.00 | 356 / 357 / 361 | 1.00 |
| AeroZoo (200000 / 501,802) | 113.7 / 92.8 / 135.9 | 0.82 | 598 / 581 / 627 | 0.97 |
| DrivAerML vehicle (200000 / 16,953,702) | 384.4 / 345.2 / 450.4 | 0.90 | 867 / 829 / 942 | 0.96 |
| DrivAerML whole boundary (200000 / 16,958,756) | 436.1 / 403.3 / 514.3 | 0.92 | 930 / 897 / 1,009 | 0.96 |
