# AZ-RESCORE tables (all lanes)

Values are two-seed means of per-lane case statistics (mean or median over the split side's cases). eps_* are AeroZoo's interior metrics (volume-weighted RMS over every cell, in U_ref / q_ref, float64); rel_* the full-mesh unweighted relative L2 (float64); reg_rel_* the registered 200k-cell float32 relative L2.


## full (full.test, 716 cases, 716 effective)


**All cases** (n = 716)

| model | eps_U mean / median | eps_p mean / median | eps_U_near mean / median | eps_p_near mean / median | rel_velocity mean / median | rel_pressure mean / median | reg_rel_velocity mean / median | reg_rel_pressure mean / median |
|---|---|---|---|---|---|---|---|---|
| GeoTransolver | 0.6946 / 0.5679 | 1.694 / 0.1475 | 0.6772 / 0.5528 | 1.889 / 0.1611 | 0.609 / 0.6065 | 1.25 / 1 | 0.609 / 0.6066 | 1.25 / 1 |
| ISLA (interacting) | 0.07909 / 0.05861 | 0.1693 / 0.02787 | 0.1142 / 0.09686 | 0.1849 / 0.04909 | 0.1182 / 0.1066 | 0.3045 / 0.2856 | 0.1181 / 0.106 | 0.3037 / 0.2853 |
| ISLA-DTT | 0.07734 / 0.05703 | 0.1623 / 0.02564 | 0.1092 / 0.09326 | 0.1708 / 0.0432 | 0.1105 / 0.1006 | 0.2568 / 0.244 | 0.1105 / 0.1004 | 0.2561 / 0.2431 |
| Transolver | 1.153 / 1.006 | 1.687 / 0.1494 | 1.145 / 1.01 | 1.884 / 0.1535 | 0.9975 / 0.9999 | 1.166 / 1 | 0.9975 / 0.9999 | 1.167 / 1 |
| zero predictor | 1.155 / 1.005 | 1.684 / 0.11 | 1.145 / 1.007 | 1.884 / 0.1358 | 1 / 1 | 1 / 1 | - | - |
| free stream (AeroZoo) | 0.4702 / 0.1839 | 1.684 / 0.11 | 0.4588 / 0.1829 | 1.884 / 0.1358 | 0.4688 / 0.3617 | 1 / 1 | - | - |

**mean_stable cases** (n = 515)

| model | eps_U mean / median | eps_p mean / median | eps_U_near mean / median | eps_p_near mean / median | rel_velocity mean / median | rel_pressure mean / median | reg_rel_velocity mean / median | reg_rel_pressure mean / median |
|---|---|---|---|---|---|---|---|---|
| GeoTransolver | 0.7039 / 0.5379 | 1.825 / 0.09222 | 0.6925 / 0.5182 | 2.026 / 0.1293 | 0.5706 / 0.5537 | 1.332 / 1 | 0.5707 / 0.5542 | 1.333 / 1 |
| ISLA (interacting) | 0.0761 / 0.04918 | 0.2072 / 0.02173 | 0.1088 / 0.09077 | 0.2202 / 0.04139 | 0.1098 / 0.1018 | 0.3371 / 0.3279 | 0.1097 / 0.1019 | 0.3362 / 0.3272 |
| ISLA-DTT | 0.07323 / 0.04924 | 0.1977 / 0.01926 | 0.1035 / 0.08449 | 0.2026 / 0.03721 | 0.1017 / 0.09441 | 0.2808 / 0.2744 | 0.1017 / 0.09436 | 0.2801 / 0.2756 |
| Transolver | 1.201 / 1.007 | 1.817 / 0.0928 | 1.193 / 1.01 | 2.019 / 0.1244 | 0.9978 / 0.9999 | 1.228 / 1 | 0.9978 / 0.9999 | 1.228 / 1 |
| zero predictor | 1.203 / 1.005 | 1.809 / 0.05868 | 1.193 / 1.007 | 2.014 / 0.1077 | 1 / 1 | 1 / 1 | - | - |
| free stream (AeroZoo) | 0.4401 / 0.1098 | 1.809 / 0.05868 | 0.442 / 0.1598 | 2.014 / 0.1077 | 0.389 / 0.2586 | 1 / 1 | - | - |

Order (best first), all cases:
- eps_U mean: ISLA-DTT (0.07734) < ISLA (interacting) (0.07909) < GeoTransolver (0.6946) < Transolver (1.153)
- eps_U median: ISLA-DTT (0.05703) < ISLA (interacting) (0.05861) < GeoTransolver (0.5679) < Transolver (1.006)
- eps_p mean: ISLA-DTT (0.1623) < ISLA (interacting) (0.1693) < Transolver (1.687) < GeoTransolver (1.694)
- eps_p median: ISLA-DTT (0.02564) < ISLA (interacting) (0.02787) < GeoTransolver (0.1475) < Transolver (0.1494)
- eps_U_near mean: ISLA-DTT (0.1092) < ISLA (interacting) (0.1142) < GeoTransolver (0.6772) < Transolver (1.145)
- eps_U_near median: ISLA-DTT (0.09326) < ISLA (interacting) (0.09686) < GeoTransolver (0.5528) < Transolver (1.01)
- eps_p_near mean: ISLA-DTT (0.1708) < ISLA (interacting) (0.1849) < Transolver (1.884) < GeoTransolver (1.889)
- eps_p_near median: ISLA-DTT (0.0432) < ISLA (interacting) (0.04909) < Transolver (0.1535) < GeoTransolver (0.1611)
- rel_velocity mean: ISLA-DTT (0.1105) < ISLA (interacting) (0.1182) < GeoTransolver (0.609) < Transolver (0.9975)
- rel_velocity median: ISLA-DTT (0.1006) < ISLA (interacting) (0.1066) < GeoTransolver (0.6065) < Transolver (0.9999)
- rel_pressure mean: ISLA-DTT (0.2568) < ISLA (interacting) (0.3045) < Transolver (1.166) < GeoTransolver (1.25)
- rel_pressure median: ISLA-DTT (0.244) < ISLA (interacting) (0.2856) < Transolver (1) < GeoTransolver (1)
- reg_rel_velocity mean: ISLA-DTT (0.1105) < ISLA (interacting) (0.1181) < GeoTransolver (0.609) < Transolver (0.9975)
- reg_rel_velocity median: ISLA-DTT (0.1004) < ISLA (interacting) (0.106) < GeoTransolver (0.6066) < Transolver (0.9999)
- reg_rel_pressure mean: ISLA-DTT (0.2561) < ISLA (interacting) (0.3037) < Transolver (1.167) < GeoTransolver (1.25)
- reg_rel_pressure median: ISLA-DTT (0.2431) < ISLA (interacting) (0.2853) < Transolver (1) < GeoTransolver (1)

## bank_zero_to_cad (bank_zero_to_cad.test, 333 cases, 333 effective)


**All cases** (n = 333)

| model | eps_U mean / median | eps_p mean / median | eps_U_near mean / median | eps_p_near mean / median | rel_velocity mean / median | rel_pressure mean / median | reg_rel_velocity mean / median | reg_rel_pressure mean / median |
|---|---|---|---|---|---|---|---|---|
| GeoTransolver | 0.5251 / 0.2958 | 1.806 / 0.1757 | 0.5413 / 0.3468 | 1.935 / 0.2409 | 0.5057 / 0.444 | 1.123 / 1 | 0.5057 / 0.4438 | 1.123 / 1 |
| ISLA (interacting) | 0.1152 / 0.08689 | 0.2238 / 0.04374 | 0.1777 / 0.1494 | 0.2647 / 0.08269 | 0.175 / 0.1596 | 0.3779 / 0.3668 | 0.175 / 0.1594 | 0.3774 / 0.3652 |
| ISLA-DTT | 0.114 / 0.08553 | 0.2165 / 0.04453 | 0.173 / 0.1456 | 0.2541 / 0.08405 | 0.1666 / 0.1508 | 0.361 / 0.3469 | 0.1666 / 0.1506 | 0.3607 / 0.3495 |
| zero predictor | 1.163 / 1.009 | 1.783 / 0.145 | 1.149 / 1.009 | 1.916 / 0.2028 | 1 / 1 | 1 / 1 | - | - |
| free stream (AeroZoo) | 0.4858 / 0.2063 | 1.783 / 0.145 | 0.5118 / 0.2955 | 1.916 / 0.2028 | 0.5042 / 0.4263 | 1 / 1 | - | - |

**mean_stable cases** (n = 234)

| model | eps_U mean / median | eps_p mean / median | eps_U_near mean / median | eps_p_near mean / median | rel_velocity mean / median | rel_pressure mean / median | reg_rel_velocity mean / median | reg_rel_pressure mean / median |
|---|---|---|---|---|---|---|---|---|
| GeoTransolver | 0.5308 / 0.248 | 2.117 / 0.1219 | 0.5436 / 0.2962 | 2.253 / 0.1761 | 0.4514 / 0.3658 | 1.142 / 1 | 0.4514 / 0.3657 | 1.142 / 1 |
| ISLA (interacting) | 0.1118 / 0.07855 | 0.2797 / 0.03912 | 0.1703 / 0.1365 | 0.3155 / 0.07079 | 0.1599 / 0.1515 | 0.4001 / 0.4008 | 0.1599 / 0.1518 | 0.3995 / 0.3993 |
| ISLA-DTT | 0.1105 / 0.07859 | 0.2678 / 0.03809 | 0.1654 / 0.1322 | 0.2983 / 0.07259 | 0.152 / 0.1376 | 0.3793 / 0.3678 | 0.1519 / 0.1375 | 0.3786 / 0.3672 |
| zero predictor | 1.226 / 1.01 | 2.093 / 0.09349 | 1.203 / 1.009 | 2.234 / 0.1624 | 1 / 1 | 1 / 1 | - | - |
| free stream (AeroZoo) | 0.4817 / 0.1499 | 2.093 / 0.09349 | 0.5067 / 0.2459 | 2.234 / 0.1624 | 0.435 / 0.3458 | 1 / 1 | - | - |

Order (best first), all cases:
- eps_U mean: ISLA-DTT (0.114) < ISLA (interacting) (0.1152) < GeoTransolver (0.5251)
- eps_U median: ISLA-DTT (0.08553) < ISLA (interacting) (0.08689) < GeoTransolver (0.2958)
- eps_p mean: ISLA-DTT (0.2165) < ISLA (interacting) (0.2238) < GeoTransolver (1.806)
- eps_p median: ISLA (interacting) (0.04374) < ISLA-DTT (0.04453) < GeoTransolver (0.1757)
- eps_U_near mean: ISLA-DTT (0.173) < ISLA (interacting) (0.1777) < GeoTransolver (0.5413)
- eps_U_near median: ISLA-DTT (0.1456) < ISLA (interacting) (0.1494) < GeoTransolver (0.3468)
- eps_p_near mean: ISLA-DTT (0.2541) < ISLA (interacting) (0.2647) < GeoTransolver (1.935)
- eps_p_near median: ISLA (interacting) (0.08269) < ISLA-DTT (0.08405) < GeoTransolver (0.2409)
- rel_velocity mean: ISLA-DTT (0.1666) < ISLA (interacting) (0.175) < GeoTransolver (0.5057)
- rel_velocity median: ISLA-DTT (0.1508) < ISLA (interacting) (0.1596) < GeoTransolver (0.444)
- rel_pressure mean: ISLA-DTT (0.361) < ISLA (interacting) (0.3779) < GeoTransolver (1.123)
- rel_pressure median: ISLA-DTT (0.3469) < ISLA (interacting) (0.3668) < GeoTransolver (1)
- reg_rel_velocity mean: ISLA-DTT (0.1666) < ISLA (interacting) (0.175) < GeoTransolver (0.5057)
- reg_rel_velocity median: ISLA-DTT (0.1506) < ISLA (interacting) (0.1594) < GeoTransolver (0.4438)
- reg_rel_pressure mean: ISLA-DTT (0.3607) < ISLA (interacting) (0.3774) < GeoTransolver (1.123)
- reg_rel_pressure median: ISLA-DTT (0.3495) < ISLA (interacting) (0.3652) < GeoTransolver (1)

## blobs (bank_blobs.test, 1367 cases, 1367 effective)


**All cases** (n = 1367)

| model | eps_U mean / median | eps_p mean / median | eps_U_near mean / median | eps_p_near mean / median | rel_velocity mean / median | rel_pressure mean / median | reg_rel_velocity mean / median | reg_rel_pressure mean / median |
|---|---|---|---|---|---|---|---|---|
| GeoTransolver | 0.8359 / 0.6721 | 1.798 / 0.1659 | 0.8221 / 0.6627 | 2.012 / 0.1638 | 0.7179 / 0.7074 | 1.114 / 1 | 0.7179 / 0.7073 | 1.114 / 1 |
| ISLA (interacting) | 0.09626 / 0.07276 | 0.1952 / 0.0343 | 0.1387 / 0.1166 | 0.2072 / 0.0578 | 0.1452 / 0.1393 | 0.3884 / 0.4173 | 0.1452 / 0.1398 | 0.3881 / 0.4173 |
| ISLA-DTT | 0.0878 / 0.06647 | 0.1835 / 0.03052 | 0.1255 / 0.1061 | 0.1832 / 0.05109 | 0.1297 / 0.1238 | 0.3071 / 0.3268 | 0.1297 / 0.1241 | 0.3069 / 0.3266 |
| zero predictor | 1.171 / 1.005 | 1.765 / 0.1094 | 1.164 / 1.012 | 1.981 / 0.1364 | 1 / 1 | 1 / 1 | - | - |
| free stream (AeroZoo) | 0.4764 / 0.1702 | 1.765 / 0.1094 | 0.4645 / 0.1836 | 1.981 / 0.1364 | 0.4681 / 0.3174 | 1 / 1 | - | - |

**mean_stable cases** (n = 917)

| model | eps_U mean / median | eps_p mean / median | eps_U_near mean / median | eps_p_near mean / median | rel_velocity mean / median | rel_pressure mean / median | reg_rel_velocity mean / median | reg_rel_pressure mean / median |
|---|---|---|---|---|---|---|---|---|
| GeoTransolver | 0.8315 / 0.6378 | 1.696 / 0.1023 | 0.8223 / 0.6334 | 1.865 / 0.1336 | 0.6855 / 0.6583 | 1.136 / 1 | 0.6856 / 0.6582 | 1.136 / 1 |
| ISLA (interacting) | 0.09365 / 0.06515 | 0.1936 / 0.02841 | 0.1372 / 0.1113 | 0.1971 / 0.05226 | 0.1388 / 0.1339 | 0.4392 / 0.4693 | 0.1388 / 0.134 | 0.4388 / 0.469 |
| ISLA-DTT | 0.08469 / 0.05925 | 0.1816 / 0.02401 | 0.1227 / 0.09954 | 0.1692 / 0.0445 | 0.1221 / 0.1172 | 0.3407 / 0.3617 | 0.1221 / 0.1174 | 0.3402 / 0.3612 |
| zero predictor | 1.198 / 1.006 | 1.665 / 0.06559 | 1.191 / 1.011 | 1.838 / 0.1129 | 1 / 1 | 1 / 1 | - | - |
| free stream (AeroZoo) | 0.4206 / 0.1052 | 1.665 / 0.06559 | 0.4263 / 0.1636 | 1.838 / 0.1129 | 0.3815 / 0.2474 | 1 / 1 | - | - |

Order (best first), all cases:
- eps_U mean: ISLA-DTT (0.0878) < ISLA (interacting) (0.09626) < GeoTransolver (0.8359)
- eps_U median: ISLA-DTT (0.06647) < ISLA (interacting) (0.07276) < GeoTransolver (0.6721)
- eps_p mean: ISLA-DTT (0.1835) < ISLA (interacting) (0.1952) < GeoTransolver (1.798)
- eps_p median: ISLA-DTT (0.03052) < ISLA (interacting) (0.0343) < GeoTransolver (0.1659)
- eps_U_near mean: ISLA-DTT (0.1255) < ISLA (interacting) (0.1387) < GeoTransolver (0.8221)
- eps_U_near median: ISLA-DTT (0.1061) < ISLA (interacting) (0.1166) < GeoTransolver (0.6627)
- eps_p_near mean: ISLA-DTT (0.1832) < ISLA (interacting) (0.2072) < GeoTransolver (2.012)
- eps_p_near median: ISLA-DTT (0.05109) < ISLA (interacting) (0.0578) < GeoTransolver (0.1638)
- rel_velocity mean: ISLA-DTT (0.1297) < ISLA (interacting) (0.1452) < GeoTransolver (0.7179)
- rel_velocity median: ISLA-DTT (0.1238) < ISLA (interacting) (0.1393) < GeoTransolver (0.7074)
- rel_pressure mean: ISLA-DTT (0.3071) < ISLA (interacting) (0.3884) < GeoTransolver (1.114)
- rel_pressure median: ISLA-DTT (0.3268) < ISLA (interacting) (0.4173) < GeoTransolver (1)
- reg_rel_velocity mean: ISLA-DTT (0.1297) < ISLA (interacting) (0.1452) < GeoTransolver (0.7179)
- reg_rel_velocity median: ISLA-DTT (0.1241) < ISLA (interacting) (0.1398) < GeoTransolver (0.7073)
- reg_rel_pressure mean: ISLA-DTT (0.3069) < ISLA (interacting) (0.3881) < GeoTransolver (1.114)
- reg_rel_pressure median: ISLA-DTT (0.3266) < ISLA (interacting) (0.4173) < GeoTransolver (1)

## nonbox (family_non_box.test, 1074 cases, 1074 effective)


**All cases** (n = 1074)

| model | eps_U mean / median | eps_p mean / median | eps_U_near mean / median | eps_p_near mean / median | rel_velocity mean / median | rel_pressure mean / median | reg_rel_velocity mean / median | reg_rel_pressure mean / median |
|---|---|---|---|---|---|---|---|---|
| E1-AZ HEADNORM | 0.4496 / 0.3988 | 0.7656 / 0.2282 | 0.3874 / 0.3821 | 0.5893 / 0.2163 | 0.5584 / 0.4859 | 3.575 / 2.055 | 0.5584 / 0.4855 | 3.57 / 2.071 |
| GeoTransolver | 0.396 / 0.2635 | 0.4788 / 0.1141 | 0.3117 / 0.217 | 0.3572 / 0.1171 | 0.5904 / 0.4425 | 1.167 / 1.001 | 0.5906 / 0.4426 | 1.168 / 1.001 |
| ISLA (interacting) | 0.4184 / 0.3345 | 0.7198 / 0.1758 | 0.3633 / 0.3118 | 0.5851 / 0.1723 | 0.5332 / 0.4719 | 2.774 / 1.383 | 0.5332 / 0.4708 | 2.774 / 1.391 |
| ISLA-DTT | 0.4041 / 0.3238 | 0.7061 / 0.2933 | 0.3313 / 0.2728 | 0.5612 / 0.287 | 0.4913 / 0.4324 | 3.534 / 2.271 | 0.4913 / 0.4321 | 3.534 / 2.279 |
| zero predictor | 1.014 / 1.001 | 0.467 / 0.09241 | 1.034 / 1.013 | 0.3468 / 0.1089 | 1 / 1 | 1 / 1 | - | - |
| free stream (AeroZoo) | 0.2696 / 0.1793 | 0.467 / 0.09241 | 0.1629 / 0.1504 | 0.3468 / 0.1089 | 0.4609 / 0.4188 | 1 / 1 | - | - |

**mean_stable cases** (n = 692)

| model | eps_U mean / median | eps_p mean / median | eps_U_near mean / median | eps_p_near mean / median | rel_velocity mean / median | rel_pressure mean / median | reg_rel_velocity mean / median | reg_rel_pressure mean / median |
|---|---|---|---|---|---|---|---|---|
| E1-AZ HEADNORM | 0.42 / 0.3094 | 0.8501 / 0.1776 | 0.3516 / 0.3532 | 0.5947 / 0.1559 | 0.4401 / 0.2899 | 3.82 / 2.038 | 0.4401 / 0.2892 | 3.812 / 2.037 |
| GeoTransolver | 0.3425 / 0.2025 | 0.609 / 0.05743 | 0.2704 / 0.1908 | 0.442 / 0.09577 | 0.4309 / 0.2933 | 1.222 / 1.001 | 0.4309 / 0.2938 | 1.224 / 1.001 |
| ISLA (interacting) | 0.3715 / 0.2374 | 0.792 / 0.1087 | 0.3011 / 0.2832 | 0.5946 / 0.1416 | 0.4075 / 0.2421 | 2.662 / 1.326 | 0.4074 / 0.2416 | 2.662 / 1.328 |
| ISLA-DTT | 0.3621 / 0.2184 | 0.836 / 0.2727 | 0.2826 / 0.2417 | 0.6403 / 0.2685 | 0.388 / 0.2431 | 4.116 / 2.62 | 0.388 / 0.2432 | 4.115 / 2.614 |
| zero predictor | 1.057 / 1.001 | 0.5954 / 0.03909 | 1.016 / 0.9998 | 0.4305 / 0.08704 | 1 / 1 | 1 / 1 | - | - |
| free stream (AeroZoo) | 0.2378 / 0.07498 | 0.5954 / 0.03909 | 0.1469 / 0.1363 | 0.4305 / 0.08704 | 0.339 / 0.1765 | 1 / 1 | - | - |

Order (best first), all cases:
- eps_U mean: GeoTransolver (0.396) < ISLA-DTT (0.4041) < ISLA (interacting) (0.4184) < E1-AZ HEADNORM (0.4496)
- eps_U median: GeoTransolver (0.2635) < ISLA-DTT (0.3238) < ISLA (interacting) (0.3345) < E1-AZ HEADNORM (0.3988)
- eps_p mean: GeoTransolver (0.4788) < ISLA-DTT (0.7061) < ISLA (interacting) (0.7198) < E1-AZ HEADNORM (0.7656)
- eps_p median: GeoTransolver (0.1141) < ISLA (interacting) (0.1758) < E1-AZ HEADNORM (0.2282) < ISLA-DTT (0.2933)
- eps_U_near mean: GeoTransolver (0.3117) < ISLA-DTT (0.3313) < ISLA (interacting) (0.3633) < E1-AZ HEADNORM (0.3874)
- eps_U_near median: GeoTransolver (0.217) < ISLA-DTT (0.2728) < ISLA (interacting) (0.3118) < E1-AZ HEADNORM (0.3821)
- eps_p_near mean: GeoTransolver (0.3572) < ISLA-DTT (0.5612) < ISLA (interacting) (0.5851) < E1-AZ HEADNORM (0.5893)
- eps_p_near median: GeoTransolver (0.1171) < ISLA (interacting) (0.1723) < E1-AZ HEADNORM (0.2163) < ISLA-DTT (0.287)
- rel_velocity mean: ISLA-DTT (0.4913) < ISLA (interacting) (0.5332) < E1-AZ HEADNORM (0.5584) < GeoTransolver (0.5904)
- rel_velocity median: ISLA-DTT (0.4324) < GeoTransolver (0.4425) < ISLA (interacting) (0.4719) < E1-AZ HEADNORM (0.4859)
- rel_pressure mean: GeoTransolver (1.167) < ISLA (interacting) (2.774) < ISLA-DTT (3.534) < E1-AZ HEADNORM (3.575)
- rel_pressure median: GeoTransolver (1.001) < ISLA (interacting) (1.383) < E1-AZ HEADNORM (2.055) < ISLA-DTT (2.271)
- reg_rel_velocity mean: ISLA-DTT (0.4913) < ISLA (interacting) (0.5332) < E1-AZ HEADNORM (0.5584) < GeoTransolver (0.5906)
- reg_rel_velocity median: ISLA-DTT (0.4321) < GeoTransolver (0.4426) < ISLA (interacting) (0.4708) < E1-AZ HEADNORM (0.4855)
- reg_rel_pressure mean: GeoTransolver (1.168) < ISLA (interacting) (2.774) < ISLA-DTT (3.534) < E1-AZ HEADNORM (3.57)
- reg_rel_pressure median: GeoTransolver (1.001) < ISLA (interacting) (1.391) < E1-AZ HEADNORM (2.071) < ISLA-DTT (2.279)

## profile_crossflow (profile_crossflow.test, 578 cases, 578 effective)


**All cases** (n = 578)

| model | eps_U mean / median | eps_p mean / median | eps_U_near mean / median | eps_p_near mean / median | rel_velocity mean / median | rel_pressure mean / median | reg_rel_velocity mean / median | reg_rel_pressure mean / median |
|---|---|---|---|---|---|---|---|---|
| GeoTransolver | 0.8265 / 0.7768 | 2.594 / 1.973 | 0.7545 / 0.725 | 2.922 / 2.199 | 0.8311 / 0.8566 | 1.003 / 1 | 0.8312 / 0.857 | 1.003 / 1 |
| ISLA (interacting) | 1.295 / 1.313 | 5.014 / 5.1 | 1.552 / 1.564 | 5.149 / 5.218 | 1.419 / 1.459 | 2.661 / 2.457 | 1.419 / 1.459 | 2.661 / 2.456 |
| ISLA-DTT | 1.257 / 1.354 | 4.738 / 4.79 | 1.519 / 1.591 | 5.051 / 5.088 | 1.337 / 1.382 | 2.193 / 2.194 | 1.337 / 1.382 | 2.193 / 2.195 |
| zero predictor | 1.043 / 0.9899 | 2.589 / 1.966 | 0.9295 / 0.8966 | 2.917 / 2.193 | 1 / 1 | 1 / 1 | - | - |
| free stream (AeroZoo) | 0.8553 / 0.8041 | 2.589 / 1.966 | 0.7994 / 0.7787 | 2.917 / 2.193 | 0.8671 / 0.9002 | 1 / 1 | - | - |

**mean_stable cases** (n = 173)

| model | eps_U mean / median | eps_p mean / median | eps_U_near mean / median | eps_p_near mean / median | rel_velocity mean / median | rel_pressure mean / median | reg_rel_velocity mean / median | reg_rel_pressure mean / median |
|---|---|---|---|---|---|---|---|---|
| GeoTransolver | 0.9222 / 0.8598 | 3.494 / 2.621 | 0.8652 / 0.8448 | 3.861 / 2.786 | 0.8652 / 0.8913 | 1.002 / 1 | 0.8651 / 0.8912 | 1.002 / 1 |
| ISLA (interacting) | 1.356 / 1.382 | 5.401 / 5.553 | 1.65 / 1.681 | 5.527 / 5.637 | 1.385 / 1.428 | 2.301 / 2.146 | 1.385 / 1.426 | 2.3 / 2.141 |
| ISLA-DTT | 1.339 / 1.424 | 5.337 / 5.577 | 1.637 / 1.677 | 5.619 / 5.76 | 1.323 / 1.364 | 2.011 / 2.037 | 1.322 / 1.364 | 2.011 / 2.038 |
| zero predictor | 1.116 / 1.05 | 3.49 / 2.624 | 1.012 / 0.9683 | 3.856 / 2.783 | 1 / 1 | 1 / 1 | - | - |
| free stream (AeroZoo) | 0.9473 / 0.8859 | 3.49 / 2.624 | 0.907 / 0.8826 | 3.856 / 2.783 | 0.8949 / 0.9337 | 1 / 1 | - | - |

Order (best first), all cases:
- eps_U mean: GeoTransolver (0.8265) < ISLA-DTT (1.257) < ISLA (interacting) (1.295)
- eps_U median: GeoTransolver (0.7768) < ISLA (interacting) (1.313) < ISLA-DTT (1.354)
- eps_p mean: GeoTransolver (2.594) < ISLA-DTT (4.738) < ISLA (interacting) (5.014)
- eps_p median: GeoTransolver (1.973) < ISLA-DTT (4.79) < ISLA (interacting) (5.1)
- eps_U_near mean: GeoTransolver (0.7545) < ISLA-DTT (1.519) < ISLA (interacting) (1.552)
- eps_U_near median: GeoTransolver (0.725) < ISLA (interacting) (1.564) < ISLA-DTT (1.591)
- eps_p_near mean: GeoTransolver (2.922) < ISLA-DTT (5.051) < ISLA (interacting) (5.149)
- eps_p_near median: GeoTransolver (2.199) < ISLA-DTT (5.088) < ISLA (interacting) (5.218)
- rel_velocity mean: GeoTransolver (0.8311) < ISLA-DTT (1.337) < ISLA (interacting) (1.419)
- rel_velocity median: GeoTransolver (0.8566) < ISLA-DTT (1.382) < ISLA (interacting) (1.459)
- rel_pressure mean: GeoTransolver (1.003) < ISLA-DTT (2.193) < ISLA (interacting) (2.661)
- rel_pressure median: GeoTransolver (1) < ISLA-DTT (2.194) < ISLA (interacting) (2.457)
- reg_rel_velocity mean: GeoTransolver (0.8312) < ISLA-DTT (1.337) < ISLA (interacting) (1.419)
- reg_rel_velocity median: GeoTransolver (0.857) < ISLA-DTT (1.382) < ISLA (interacting) (1.459)
- reg_rel_pressure mean: GeoTransolver (1.003) < ISLA-DTT (2.193) < ISLA (interacting) (2.661)
- reg_rel_pressure median: GeoTransolver (1) < ISLA-DTT (2.195) < ISLA (interacting) (2.456)

## profile_dual_inlet (profile_dual_inlet.test, 516 cases, 516 effective)


**All cases** (n = 516)

| model | eps_U mean / median | eps_p mean / median | eps_U_near mean / median | eps_p_near mean / median | rel_velocity mean / median | rel_pressure mean / median | reg_rel_velocity mean / median | reg_rel_pressure mean / median |
|---|---|---|---|---|---|---|---|---|
| GeoTransolver | 1.672 / 1.597 | 9.345 / 7.374 | 1.618 / 1.59 | 10.11 / 7.798 | 0.7885 / 0.7909 | 1 / 1 | 0.7885 / 0.7909 | 1 / 1 |
| ISLA (interacting) | 1.394 / 1.373 | 7.044 / 5.884 | 1.49 / 1.445 | 7.44 / 6.116 | 0.6525 / 0.6568 | 0.7991 / 0.7976 | 0.6525 / 0.657 | 0.799 / 0.7967 |
| ISLA-DTT | 1.285 / 1.285 | 7.219 / 6.171 | 1.357 / 1.355 | 7.676 / 6.477 | 0.6065 / 0.6139 | 0.8263 / 0.8598 | 0.6065 / 0.6141 | 0.8262 / 0.8598 |
| zero predictor | 2.143 / 2.065 | 9.344 / 7.336 | 2.123 / 2.101 | 10.11 / 7.778 | 1 / 1 | 1 / 1 | - | - |
| free stream (AeroZoo) | 1.677 / 1.59 | 9.344 / 7.336 | 1.626 / 1.594 | 10.11 / 7.778 | 0.7933 / 0.7935 | 1 / 1 | - | - |

**mean_stable cases** (n = 459)

| model | eps_U mean / median | eps_p mean / median | eps_U_near mean / median | eps_p_near mean / median | rel_velocity mean / median | rel_pressure mean / median | reg_rel_velocity mean / median | reg_rel_pressure mean / median |
|---|---|---|---|---|---|---|---|---|
| GeoTransolver | 1.653 / 1.571 | 9.135 / 7.14 | 1.602 / 1.568 | 9.887 / 7.621 | 0.7869 / 0.7897 | 1 / 1 | 0.7869 / 0.7892 | 1 / 1 |
| ISLA (interacting) | 1.379 / 1.357 | 6.91 / 5.776 | 1.474 / 1.434 | 7.309 / 5.955 | 0.6518 / 0.6562 | 0.8032 / 0.8024 | 0.6518 / 0.6566 | 0.8032 / 0.8025 |
| ISLA-DTT | 1.272 / 1.274 | 7.089 / 6.059 | 1.343 / 1.341 | 7.547 / 6.317 | 0.6066 / 0.614 | 0.8312 / 0.8698 | 0.6066 / 0.6141 | 0.8312 / 0.8699 |
| zero predictor | 2.121 / 2.047 | 9.134 / 7.15 | 2.105 / 2.069 | 9.886 / 7.619 | 1 / 1 | 1 / 1 | - | - |
| free stream (AeroZoo) | 1.659 / 1.563 | 9.134 / 7.15 | 1.612 / 1.57 | 9.886 / 7.619 | 0.7927 / 0.7924 | 1 / 1 | - | - |

Order (best first), all cases:
- eps_U mean: ISLA-DTT (1.285) < ISLA (interacting) (1.394) < GeoTransolver (1.672)
- eps_U median: ISLA-DTT (1.285) < ISLA (interacting) (1.373) < GeoTransolver (1.597)
- eps_p mean: ISLA (interacting) (7.044) < ISLA-DTT (7.219) < GeoTransolver (9.345)
- eps_p median: ISLA (interacting) (5.884) < ISLA-DTT (6.171) < GeoTransolver (7.374)
- eps_U_near mean: ISLA-DTT (1.357) < ISLA (interacting) (1.49) < GeoTransolver (1.618)
- eps_U_near median: ISLA-DTT (1.355) < ISLA (interacting) (1.445) < GeoTransolver (1.59)
- eps_p_near mean: ISLA (interacting) (7.44) < ISLA-DTT (7.676) < GeoTransolver (10.11)
- eps_p_near median: ISLA (interacting) (6.116) < ISLA-DTT (6.477) < GeoTransolver (7.798)
- rel_velocity mean: ISLA-DTT (0.6065) < ISLA (interacting) (0.6525) < GeoTransolver (0.7885)
- rel_velocity median: ISLA-DTT (0.6139) < ISLA (interacting) (0.6568) < GeoTransolver (0.7909)
- rel_pressure mean: ISLA (interacting) (0.7991) < ISLA-DTT (0.8263) < GeoTransolver (1)
- rel_pressure median: ISLA (interacting) (0.7976) < ISLA-DTT (0.8598) < GeoTransolver (1)
- reg_rel_velocity mean: ISLA-DTT (0.6065) < ISLA (interacting) (0.6525) < GeoTransolver (0.7885)
- reg_rel_velocity median: ISLA-DTT (0.6141) < ISLA (interacting) (0.657) < GeoTransolver (0.7909)
- reg_rel_pressure mean: ISLA (interacting) (0.799) < ISLA-DTT (0.8262) < GeoTransolver (1)
- reg_rel_pressure median: ISLA (interacting) (0.7967) < ISLA-DTT (0.8598) < GeoTransolver (1)

## reynolds (reynolds.test, 1385 cases, 1385 effective)


**All cases** (n = 1385)

| model | eps_U mean / median | eps_p mean / median | eps_U_near mean / median | eps_p_near mean / median | rel_velocity mean / median | rel_pressure mean / median | reg_rel_velocity mean / median | reg_rel_pressure mean / median |
|---|---|---|---|---|---|---|---|---|
| GeoTransolver | 0.8199 / 0.6244 | 1.747 / 0.07534 | 0.8053 / 0.6283 | 1.901 / 0.1153 | 0.7095 / 0.6782 | 1.039 / 1 | 0.7095 / 0.6781 | 1.036 / 1 |
| ISLA (interacting) | 0.1151 / 0.07089 | 0.3708 / 0.0375 | 0.1447 / 0.1074 | 0.3657 / 0.06294 | 0.1642 / 0.1524 | 0.5096 / 0.53 | 0.1642 / 0.152 | 0.5092 / 0.5302 |
| ISLA-DTT | 0.1139 / 0.06985 | 0.4012 / 0.04625 | 0.1424 / 0.1041 | 0.4098 / 0.06316 | 0.1522 / 0.14 | 0.473 / 0.4487 | 0.1522 / 0.1401 | 0.4727 / 0.4482 |
| LRE-AZ B | 0.1401 / 0.07677 | 0.5315 / 0.04186 | 0.1672 / 0.1102 | 0.544 / 0.06842 | 0.1848 / 0.1628 | 0.544 / 0.5748 | 0.1849 / 0.1624 | 0.5433 / 0.5728 |
| zero predictor | 1.171 / 1.004 | 1.746 / 0.0688 | 1.155 / 1.006 | 1.901 / 0.1125 | 1 / 1 | 1 / 1 | - | - |
| free stream (AeroZoo) | 0.4314 / 0.1189 | 1.746 / 0.0688 | 0.4272 / 0.1465 | 1.901 / 0.1125 | 0.4295 / 0.3029 | 1 / 1 | - | - |

**mean_stable cases** (n = 1001)

| model | eps_U mean / median | eps_p mean / median | eps_U_near mean / median | eps_p_near mean / median | rel_velocity mean / median | rel_pressure mean / median | reg_rel_velocity mean / median | reg_rel_pressure mean / median |
|---|---|---|---|---|---|---|---|---|
| GeoTransolver | 0.8073 / 0.6009 | 1.642 / 0.05177 | 0.8 / 0.6071 | 1.775 / 0.09278 | 0.6787 / 0.6448 | 1.051 / 1 | 0.6787 / 0.645 | 1.047 / 1 |
| ISLA (interacting) | 0.1061 / 0.05759 | 0.3701 / 0.02948 | 0.1342 / 0.09678 | 0.3572 / 0.05288 | 0.151 / 0.1403 | 0.555 / 0.5703 | 0.151 / 0.1404 | 0.5544 / 0.5694 |
| ISLA-DTT | 0.1024 / 0.05924 | 0.3873 / 0.03785 | 0.1297 / 0.09445 | 0.3896 / 0.05491 | 0.1378 / 0.1271 | 0.512 / 0.4686 | 0.1378 / 0.1273 | 0.5115 / 0.4696 |
| LRE-AZ B | 0.1276 / 0.06166 | 0.5159 / 0.03335 | 0.1528 / 0.09895 | 0.5202 / 0.05812 | 0.169 / 0.1495 | 0.5845 / 0.6025 | 0.1691 / 0.1494 | 0.5834 / 0.6019 |
| zero predictor | 1.186 / 1.003 | 1.64 / 0.04574 | 1.174 / 1.006 | 1.773 / 0.08938 | 1 / 1 | 1 / 1 | - | - |
| free stream (AeroZoo) | 0.3786 / 0.076 | 1.64 / 0.04574 | 0.382 / 0.1282 | 1.773 / 0.08938 | 0.3587 / 0.246 | 1 / 1 | - | - |

Order (best first), all cases:
- eps_U mean: ISLA-DTT (0.1139) < ISLA (interacting) (0.1151) < LRE-AZ B (0.1401) < GeoTransolver (0.8199)
- eps_U median: ISLA-DTT (0.06985) < ISLA (interacting) (0.07089) < LRE-AZ B (0.07677) < GeoTransolver (0.6244)
- eps_p mean: ISLA (interacting) (0.3708) < ISLA-DTT (0.4012) < LRE-AZ B (0.5315) < GeoTransolver (1.747)
- eps_p median: ISLA (interacting) (0.0375) < LRE-AZ B (0.04186) < ISLA-DTT (0.04625) < GeoTransolver (0.07534)
- eps_U_near mean: ISLA-DTT (0.1424) < ISLA (interacting) (0.1447) < LRE-AZ B (0.1672) < GeoTransolver (0.8053)
- eps_U_near median: ISLA-DTT (0.1041) < ISLA (interacting) (0.1074) < LRE-AZ B (0.1102) < GeoTransolver (0.6283)
- eps_p_near mean: ISLA (interacting) (0.3657) < ISLA-DTT (0.4098) < LRE-AZ B (0.544) < GeoTransolver (1.901)
- eps_p_near median: ISLA (interacting) (0.06294) < ISLA-DTT (0.06316) < LRE-AZ B (0.06842) < GeoTransolver (0.1153)
- rel_velocity mean: ISLA-DTT (0.1522) < ISLA (interacting) (0.1642) < LRE-AZ B (0.1848) < GeoTransolver (0.7095)
- rel_velocity median: ISLA-DTT (0.14) < ISLA (interacting) (0.1524) < LRE-AZ B (0.1628) < GeoTransolver (0.6782)
- rel_pressure mean: ISLA-DTT (0.473) < ISLA (interacting) (0.5096) < LRE-AZ B (0.544) < GeoTransolver (1.039)
- rel_pressure median: ISLA-DTT (0.4487) < ISLA (interacting) (0.53) < LRE-AZ B (0.5748) < GeoTransolver (1)
- reg_rel_velocity mean: ISLA-DTT (0.1522) < ISLA (interacting) (0.1642) < LRE-AZ B (0.1849) < GeoTransolver (0.7095)
- reg_rel_velocity median: ISLA-DTT (0.1401) < ISLA (interacting) (0.152) < LRE-AZ B (0.1624) < GeoTransolver (0.6781)
- reg_rel_pressure mean: ISLA-DTT (0.4727) < ISLA (interacting) (0.5092) < LRE-AZ B (0.5433) < GeoTransolver (1.036)
- reg_rel_pressure median: ISLA-DTT (0.4482) < ISLA (interacting) (0.5302) < LRE-AZ B (0.5728) < GeoTransolver (1)

## Lanes

| lane | side | cases | complete | eps_U mean | eps_p mean | GPU-s infer | cells |
|---|---|---|---|---|---|---|---|
| azc_bank_zero_to_cad_dtt_seed42 | bank_zero_to_cad.test | 333/333 | True | 0.1158 | 0.2261 | 9345 | 2.31 G |
| azc_bank_zero_to_cad_dtt_seed43 | bank_zero_to_cad.test | 333/333 | True | 0.1122 | 0.2068 | 9224 | 2.31 G |
| azc_bank_zero_to_cad_gt_seed42 | bank_zero_to_cad.test | 333/333 | True | 0.5172 | 1.828 | 2560 | 2.31 G |
| azc_bank_zero_to_cad_gt_seed43 | bank_zero_to_cad.test | 333/333 | True | 0.533 | 1.784 | 2556 | 2.31 G |
| azc_bank_zero_to_cad_isla_seed42 | bank_zero_to_cad.test | 333/333 | True | 0.1158 | 0.2298 | 10469 | 2.31 G |
| azc_bank_zero_to_cad_isla_seed43 | bank_zero_to_cad.test | 333/333 | True | 0.1147 | 0.2178 | 10130 | 2.31 G |
| azc_blobs_dtt_seed42 | bank_blobs.test | 1367/1367 | True | 0.08793 | 0.1765 | 37244 | 9.43 G |
| azc_blobs_dtt_seed43 | bank_blobs.test | 1367/1367 | True | 0.08767 | 0.1904 | 37220 | 9.43 G |
| azc_blobs_gt_seed42 | bank_blobs.test | 1367/1367 | True | 1.054 | 1.803 | 10464 | 9.43 G |
| azc_blobs_gt_seed43 | bank_blobs.test | 1367/1367 | True | 0.6174 | 1.793 | 10446 | 9.43 G |
| azc_blobs_isla_seed42 | bank_blobs.test | 1367/1367 | True | 0.09398 | 0.2006 | 41365 | 9.43 G |
| azc_blobs_isla_seed43 | bank_blobs.test | 1367/1367 | True | 0.09855 | 0.1899 | 41341 | 9.43 G |
| azc_full_dtt_seed42 | full.test | 716/716 | True | 0.07751 | 0.1657 | 17135 | 4.30 G |
| azc_full_dtt_seed43 | full.test | 716/716 | True | 0.07717 | 0.1588 | 17124 | 4.30 G |
| azc_full_gt_seed42 | full.test | 716/716 | True | 0.6931 | 1.7 | 4796 | 4.30 G |
| azc_full_gt_seed43 | full.test | 716/716 | True | 0.6961 | 1.687 | 4756 | 4.30 G |
| azc_full_isla_seed42 | full.test | 716/716 | True | 0.08086 | 0.1731 | 18926 | 4.30 G |
| azc_full_isla_seed43 | full.test | 716/716 | True | 0.07732 | 0.1654 | 18914 | 4.30 G |
| azc_full_t_seed42 | full.test | 716/716 | True | 1.152 | 1.685 | 4700 | 4.30 G |
| azc_full_t_seed43 | full.test | 716/716 | True | 1.154 | 1.689 | 4600 | 4.30 G |
| azc_nonbox_dtt_seed42 | family_non_box.test | 1074/1074 | True | 0.4369 | 0.7646 | 17527 | 3.72 G |
| azc_nonbox_dtt_seed43 | family_non_box.test | 1074/1074 | True | 0.3714 | 0.6476 | 17195 | 3.72 G |
| azc_nonbox_gt_seed42 | family_non_box.test | 1074/1074 | True | 0.3763 | 0.4687 | 4153 | 3.72 G |
| azc_nonbox_gt_seed43 | family_non_box.test | 1074/1074 | True | 0.4157 | 0.4889 | 4136 | 3.72 G |
| azc_nonbox_isla_seed42 | family_non_box.test | 1074/1074 | True | 0.513 | 0.8673 | 17881 | 3.72 G |
| azc_nonbox_isla_seed43 | family_non_box.test | 1074/1074 | True | 0.3239 | 0.5724 | 16824 | 3.72 G |
| azc_profile_crossflow_dtt_seed42 | profile_crossflow.test | 578/578 | True | 1.146 | 3.505 | 17058 | 4.44 G |
| azc_profile_crossflow_dtt_seed43 | profile_crossflow.test | 578/578 | True | 1.367 | 5.97 | 17053 | 4.44 G |
| azc_profile_crossflow_gt_seed42 | profile_crossflow.test | 578/578 | True | 0.8325 | 2.574 | 4994 | 4.44 G |
| azc_profile_crossflow_gt_seed43 | profile_crossflow.test | 578/578 | True | 0.8205 | 2.614 | 4947 | 4.44 G |
| azc_profile_crossflow_isla_seed42 | profile_crossflow.test | 578/578 | True | 1.278 | 6.095 | 19440 | 4.44 G |
| azc_profile_crossflow_isla_seed43 | profile_crossflow.test | 578/578 | True | 1.312 | 3.933 | 19423 | 4.44 G |
| azc_profile_dual_inlet_dtt_seed42 | profile_dual_inlet.test | 516/516 | True | 1.279 | 7.457 | 13767 | 3.62 G |
| azc_profile_dual_inlet_dtt_seed43 | profile_dual_inlet.test | 516/516 | True | 1.291 | 6.982 | 13761 | 3.62 G |
| azc_profile_dual_inlet_gt_seed42 | profile_dual_inlet.test | 516/516 | True | 1.674 | 9.313 | 4060 | 3.62 G |
| azc_profile_dual_inlet_gt_seed43 | profile_dual_inlet.test | 516/516 | True | 1.671 | 9.376 | 4127 | 3.62 G |
| azc_profile_dual_inlet_isla_seed42 | profile_dual_inlet.test | 516/516 | True | 1.392 | 6.14 | 15842 | 3.62 G |
| azc_profile_dual_inlet_isla_seed43 | profile_dual_inlet.test | 516/516 | True | 1.395 | 7.948 | 15827 | 3.62 G |
| azc_reynolds_dtt_seed42 | reynolds.test | 1385/1385 | True | 0.1245 | 0.4848 | 35295 | 8.92 G |
| azc_reynolds_dtt_seed43 | reynolds.test | 1385/1385 | True | 0.1032 | 0.3175 | 35272 | 8.92 G |
| azc_reynolds_gt_seed42 | reynolds.test | 1385/1385 | True | 0.5348 | 1.746 | 9868 | 8.92 G |
| azc_reynolds_gt_seed43 | reynolds.test | 1385/1385 | True | 1.105 | 1.748 | 9868 | 8.92 G |
| azc_reynolds_isla_seed42 | reynolds.test | 1385/1385 | True | 0.1076 | 0.3199 | 39159 | 8.92 G |
| azc_reynolds_isla_seed43 | reynolds.test | 1385/1385 | True | 0.1225 | 0.4217 | 39132 | 8.92 G |
| hn090az_isla_seed42 | family_non_box.test | 1074/1074 | True | 0.554 | 0.869 | 16865 | 3.72 G |
| hn090az_isla_seed43 | family_non_box.test | 1074/1074 | True | 0.3453 | 0.6623 | 16723 | 3.72 G |
| lreaz_reynolds_isla_seed42 | reynolds.test | 1385/1385 | True | 0.1344 | 0.4538 | 39075 | 8.92 G |
| lreaz_reynolds_isla_seed43 | reynolds.test | 1385/1385 | True | 0.1459 | 0.6091 | 38840 | 8.92 G |

## Notes

- duplicate (lane, case) rows: 0, worst eps relative disagreement among them 0.00e+00
