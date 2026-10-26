## Checks
- F s42: complete True (last 20000 of 20000), boundaries ['inlet', 'no_slip', 'outlet', 'slip', 'vehicle'], point hashes as Part B True, heldout_whole True, finite True, step 5.253480792045593 s, peak 234.40046310424805 GiB
- F s43: complete True (last 20000 of 20000), boundaries ['inlet', 'no_slip', 'outlet', 'slip', 'vehicle'], point hashes as Part B True, heldout_whole True, finite True, step 5.247898459434509 s, peak 234.40046310424805 GiB
- C s42: complete True (last 20000 of 20000), boundaries ['vehicle'], point hashes as Part B True, heldout_whole True, finite True, step 3.0133907794952393 s, peak 234.38781118392944 GiB
- C s43: complete True (last 20000 of 20000), boundaries ['vehicle'], point hashes as Part B True, heldout_whole True, finite True, step 3.013375163078308 s, peak 234.38781118392944 GiB
- isla_rd s42: complete True (last 20000 of 20000), boundaries ['vehicle'], point hashes as Part B True, heldout_whole None, finite True, step 2.98952579498291 s, peak 234.38781118392944 GiB
- isla_rd s43: complete True (last 20000 of 20000), boundaries ['vehicle'], point hashes as Part B True, heldout_whole None, finite True, step 2.990046501159668 s, peak 234.38781118392944 GiB
- t s42: complete True (last 20000 of 20000), boundaries [], point hashes as Part B True, heldout_whole True, finite True, step 1.7202928066253662 s, peak 174.09832859039307 GiB
- t s43: complete True (last 20000 of 20000), boundaries [], point hashes as Part B True, heldout_whole True, finite True, step 1.7202036380767822 s, peak 174.09832859039307 GiB
- gt s42: complete True (last 20000 of 20000), boundaries [], point hashes as Part B True, heldout_whole None, finite True, step 1.7153048515319824 s, peak 160.23197507858276 GiB
- gt s43: complete True (last 20000 of 20000), boundaries [], point hashes as Part B True, heldout_whole None, finite True, step 1.7152067422866821 s, peak 160.23197507858276 GiB
- ref44 s44: complete True (last 20000 of 20000), boundaries ['vehicle'], point hashes as Part B True, heldout_whole True, finite True, step 2.998545289039612 s, peak 234.38781118392944 GiB

## F_ho (held-out; [epoch-0 draw])
| arm | velocity s42 | velocity s43 | pressure s42 | pressure s43 | nut s42 | nut s43 |
|---|---|---|---|---|---|---|
| F | 0.01532 [0.01482] | 0.01488 [0.01482] | 0.009239 [0.01002] | 0.007931 [0.008732] | 0.01436 [0.01531] | 0.01324 [0.01245] |
| C | 0.01907 [0.0192] | 0.02067 [0.02009] | 0.009752 [0.01066] | 0.01165 [0.01081] | 0.02077 [0.03433] | 0.02038 [0.03486] |
| isla_rd | 0.01918 [0.01915] | 0.02001 [0.01943] | 0.01011 [0.01079] | 0.01149 [0.01085] | 0.02389 [0.02453] | 0.02194 [0.02603] |
| gt | 0.01011 [0.01125] | 0.00968 [0.009814] | 0.006485 [0.006575] | 0.006532 [0.006586] | 0.01085 [0.01012] | 0.01057 [0.01004] |
| t | 0.008783 [0.008945] | 0.008639 [0.008527] | 0.005888 [0.005936] | 0.005758 [0.005649] | 0.00978 [0.009326] | 0.009957 [0.009205] |
15.10 ref s44: velocity 0.01921, pressure 0.01024, nut 0.02073

## Ratios per seed
- velocity s42: R_F = F / best baseline 1.744; R_C 2.172; F / C 0.803; C / Part B ISLA 0.995
- velocity s43: R_F = F / best baseline 1.723; R_C 2.392; F / C 0.720; C / Part B ISLA 1.033
- pressure s42: R_F = F / best baseline 1.569; R_C 1.656; F / C 0.947; C / Part B ISLA 0.964
- pressure s43: R_F = F / best baseline 1.377; R_C 2.024; F / C 0.681; C / Part B ISLA 1.014
- nut s42: R_F = F / best baseline 1.468; R_C 2.124; F / C 0.691; C / Part B ISLA 0.870
- nut s43: R_F = F / best baseline 1.330; R_C 2.047; F / C 0.650; C / Part B ISLA 0.929

## Far-field ([1, inf) m) e_R, held-out [epoch-0 draw]
- nut: F s42 0.01304 [0.01805]; F s43 0.01192 [0.008384]; C s42 0.03103 [0.06448]; C s43 0.02807 [0.06483]; isla_rd s42 0.04101 [0.04078]; isla_rd s43 0.03452 [0.04401]; gt s42 0.01154 [0.008197]; gt s43 0.01135 [0.008656]; t s42 0.009741 [0.008124]; t s43 0.01065 [0.007659]; ref s44 0.03149
- velocity: F s42 0.007389 [0.008436]; F s43 0.00642 [0.006235]; C s42 0.01119 [0.009595]; C s43 0.01028 [0.01225]; isla_rd s42 0.01281 [0.01017]; isla_rd s43 0.01091 [0.01448]; gt s42 0.005041 [0.004587]; gt s43 0.004538 [0.00523]; t s42 0.004152 [0.004946]; t s43 0.00446 [0.004724]; ref s44 0.009446
- pressure: F s42 0.2184 [0.2912]; F s43 0.1802 [0.1935]; C s42 0.2385 [0.374]; C s43 0.2761 [0.2365]; isla_rd s42 0.233 [0.225]; isla_rd s43 0.2366 [0.2405]; gt s42 0.1808 [0.184]; gt s43 0.1884 [0.1581]; t s42 0.1332 [0.2501]; t s43 0.1428 [0.1755]; ref s44 0.2357

## velocity: e_R per wall-distance decade (held-out), and e_R / Transolver's e_R in brackets
| band | F s42 | F s43 | C s42 | C s43 | t s42 | t s43 |
|---|---|---|---|---|---|---|
| [-inf,0.0001) | 0.02893 [1.52] | 0.02493 [1.38] | 0.03234 [1.70] | 0.03821 [2.11] | 0.01903 [1.00] | 0.01808 [1.00] |
| [0.0001,0.001) | 0.01964 [1.77] | 0.01889 [1.73] | 0.02404 [2.17] | 0.02613 [2.39] | 0.01108 [1.00] | 0.01093 [1.00] |
| [0.001,0.01) | 0.01412 [1.83] | 0.01464 [1.91] | 0.01872 [2.43] | 0.01964 [2.56] | 0.007711 [1.00] | 0.00768 [1.00] |
| [0.01,0.1) | 0.005839 [1.60] | 0.005769 [1.60] | 0.008056 [2.21] | 0.008144 [2.26] | 0.003651 [1.00] | 0.003607 [1.00] |
| [0.1,1) | 0.006648 [1.58] | 0.006465 [1.53] | 0.008563 [2.03] | 0.0097 [2.30] | 0.00422 [1.00] | 0.004221 [1.00] |
| [1,inf) | 0.007389 [1.78] | 0.00642 [1.44] | 0.01119 [2.70] | 0.01028 [2.30] | 0.004152 [1.00] | 0.00446 [1.00] |

## pressure: e_R per wall-distance decade (held-out), and e_R / Transolver's e_R in brackets
| band | F s42 | F s43 | C s42 | C s43 | t s42 | t s43 |
|---|---|---|---|---|---|---|
| [-inf,0.0001) | 0.01435 [1.31] | 0.01241 [1.15] | 0.01578 [1.44] | 0.01903 [1.77] | 0.01093 [1.00] | 0.01077 [1.00] |
| [0.0001,0.001) | 0.008434 [1.75] | 0.007175 [1.53] | 0.008675 [1.80] | 0.0106 [2.26] | 0.004811 [1.00] | 0.004694 [1.00] |
| [0.001,0.01) | 0.00769 [1.80] | 0.006643 [1.59] | 0.008027 [1.88] | 0.008751 [2.10] | 0.004267 [1.00] | 0.004176 [1.00] |
| [0.01,0.1) | 0.006573 [1.67] | 0.005797 [1.52] | 0.007285 [1.85] | 0.007865 [2.07] | 0.003936 [1.00] | 0.003803 [1.00] |
| [0.1,1) | 0.01573 [1.59] | 0.01503 [1.51] | 0.01647 [1.66] | 0.01947 [1.96] | 0.009914 [1.00] | 0.009934 [1.00] |
| [1,inf) | 0.2184 [1.64] | 0.1802 [1.26] | 0.2385 [1.79] | 0.2761 [1.93] | 0.1332 [1.00] | 0.1428 [1.00] |

## nut: e_R per wall-distance decade (held-out), and e_R / Transolver's e_R in brackets
| band | F s42 | F s43 | C s42 | C s43 | t s42 | t s43 |
|---|---|---|---|---|---|---|
| [-inf,0.0001) | 0.009548 [1.22] | 0.008566 [1.18] | 0.009987 [1.27] | 0.01049 [1.44] | 0.007846 [1.00] | 0.007279 [1.00] |
| [0.0001,0.001) | 0.02231 [1.39] | 0.02059 [1.30] | 0.02482 [1.55] | 0.02612 [1.65] | 0.01601 [1.00] | 0.0158 [1.00] |
| [0.001,0.01) | 0.01469 [1.58] | 0.01347 [1.45] | 0.01687 [1.82] | 0.01708 [1.84] | 0.009279 [1.00] | 0.009302 [1.00] |
| [0.01,0.1) | 0.01183 [1.55] | 0.01098 [1.41] | 0.01527 [2.00] | 0.01578 [2.03] | 0.007638 [1.00] | 0.007779 [1.00] |
| [0.1,1) | 0.01054 [1.46] | 0.01011 [1.38] | 0.01478 [2.05] | 0.01618 [2.20] | 0.007225 [1.00] | 0.007347 [1.00] |
| [1,inf) | 0.01304 [1.34] | 0.01192 [1.12] | 0.03103 [3.19] | 0.02807 [2.64] | 0.009741 [1.00] | 0.01065 [1.00] |

## Registered readings
- **nu_t far-field e_R: closes**  F 0.01304, 0.01192; C 0.03103, 0.02807; T 0.009741, 0.01065; REF 0.04101, 0.03452, 0.03149, 0.03103, 0.02807; F/T 1.339, 1.119; F/C 0.4203, 0.4246
- **velocity F_ho: helps**  F 0.01532, 0.01488; C 0.01907, 0.02067; T 0.008783, 0.008639; REF 0.01918, 0.02001, 0.01921, 0.01907, 0.02067; F/T 1.744, 1.723; F/C 0.803, 0.7201
