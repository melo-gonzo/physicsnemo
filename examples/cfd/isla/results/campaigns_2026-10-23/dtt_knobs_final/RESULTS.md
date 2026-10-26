# DTT-KNOBS stage 2: results (PREREG.md md5 39d8272f)

Read by the coordinator. Every rule is read as registered. Sources: `reduce/out_r5/` (reduce_s2.py md5 e2b605d2) and
`reduce/out_r5/r5_registered_arith.txt` (the registered arithmetic-mean form, recomputed by the coordinator).

## R5: DTT-NOSC-STD against B (read 2026-10-24 22:58 EDT; fix-trained; second-read and corrected 23:30, RESULTS_R5_REVIEW.md)

The runs dttknobs_s2_noscstd_seed42 and seed43 trained 500 epochs. They were scored by job 376686 (array task 6: native and
posed, 200 full_test cases each, float32, eager, code_att090fix + recipe_att090b). Every section-4 check passes: epoch 500,
200 cases with the control's sample ids, the snapshot, the dataset and the extra. B is r090fix_af_dtt 42 / 43. DTT-NOSC is
att2_af_dttnosc. σ_f is ATTRIB 3c's six-arm pool plus DTT-NOSC-STD's two seeds (7 dof): 0.0503 / 0.0866 / 0.0744. The
large velocity value comes from DTT-NOSC-STD's own velocity seeds, which sit 36 % apart (0.00726, 0.00988). The
reducer reproduces ATTRIB's 6-dof σ (0.0422 / 0.0291 / 0.0637) and G_f to the registered three decimals.

| field | DTT-NOSC-STD (s42, s43) | ÷ B, arithmetic (registered) | 2 SE | ÷ DTT-NOSC | A | N | **registered reading** | geometric form (beside) |
|---|---|---|---|---|---|---|---|---|
| pressure | 0.014377, 0.016185 | 1.208 (Δ +0.189) | 0.101 | 0.955 | yes | no | **standardization leaves it (the scales carry it)** | same (Δ +0.187) |
| in-plane velocity | 0.007264, 0.009881 | 1.203 (Δ +0.185) | 0.173 | 0.982 | yes | no | **standardization leaves it (the scales carry it)** | unresolved (Δ +0.173 < 2 SE): **flagged** |
| nut | 0.023150, 0.027435 | 1.177 (Δ +0.163) | 0.149 | 1.018 | yes | no | **standardization leaves it (the scales carry it)** | same (Δ +0.162) |

Per seed, ÷ B: pressure 1.128 / 1.288, velocity 1.022 / 1.382, nut 1.164 / 1.188. Per seed, ÷ DTT-NOSC: 0.916 / 0.992,
0.813 / 1.159, 0.909 / 1.134. In the registered form, φ = Δ / G_f is 0.81 / 0.91 / 1.11 and Δ / D_DTT is
1.83 / 1.18 / 6.26 (geometric form: 0.80 / 0.85 / 1.10 and 1.81 / 1.11 / 6.23). Posed agrees within 0.001 on the ratios
(1.207 / 1.202 / 1.177) and gives the same labels; it is not an independent number.

**Registered readings.**
- R5 per field: standardization leaves DTT-NOSC's loss on pressure, velocity and nut (the scales carry it). On velocity
  the registered arithmetic form clears 2 SE by 7 %; the geometric form would read "unresolved". This is flagged, and
  the registered label stands.
- Bundle thread (Q4), on pressure and nut: C_f, the part of ISLA-DTT's CONT-to-0.9.0 bundle other than the scales,
  measured with the raw speed, **sits in the scales, as far as two seeds can show**. There is no sign that it is an
  interaction with the raw speed: with the speed standardized and no scales, ISLA-DTT is still about 20 % worse than B,
  and not measurably better than DTT-NOSC (pressure ÷ DTT-NOSC 0.955, unresolved).
- Prediction P10 ("removes" on pressure, p 0.60) failed. P12 (nut "unresolved", p 0.45) failed.
- Labels: fix-trained.

**Beside the rule (not a verdict).**
- The two forms read differently. Interacting ISLA's mirror, NOSC-STD ÷ B (ATTRIB 2c, geometric seed means), read
  1.003 / 1.029 / 1.043: no measurable effect, "the input standardization carries the step; the scales do not". No
  registered rule tests the difference between the forms. A plain 2-SE check of Δ_DTT - Δ_ISLA, with
  SE = sqrt(σ_7^2 + σ_2c^2) and 2c's 8-dof 0.063 / 0.045 / 0.059, resolves it on pressure only (0.186 against 2 SE
  0.161). On velocity (0.156 against 0.195; ISLA-DTT's own A is the flagged one and rests on seed 43) and on nut (0.121
  against 0.190), it is within two seeds' noise.
- For the scope freeze: on pressure, two seeds show that ISLA-DTT uses the scales where interacting ISLA does not.
- A reading of why, not a test: ISLA-DTT's passive points never share state with one another. The physical scales may
  give them a frame that interacting ISLA rebuilds from the coupled points.
- Cost: a median step of 0.574 / 0.571 s and 67.1 / 66.2 GPU-h per run (sacct), against B's 0.566 / 0.575 s and
  66.1 / 67.0.
  The arm used 133.3 GPU-h of its registered estimate of 136.

## Arms 1-4: R1, R3, R4 (read 2026-10-25 09:36 EDT; eager; K-EXACT still training, so R2 and Q3 wait; second-read and corrected 10:08, RESULTS_A14_REVIEW.md)

The arms retrain ISLA-DTT on AirfRANS with one change each against the standing r090eag_af_dtt (eager, seeds 42 / 43):
K-THETA2 a coarser tree read (theta 2), K-THETA05 a finer one (theta 0.5), K-B1 the boundary read after every second
block only, K-BASIS8 a narrower kernel basis (8 functions, not 16). A ratio below 1 is better. R1 labels a field
"helps" or "harms" only past a 10 % bar and 2 SE; the "flag" marks a seed-separated shift of more than 3 % that the bar
cannot label. The "scope freeze" is the program's decision on what the final ISLA keeps; "stage 3" is the DrivAerML
Volume check that PREREG section 7 names for a cheaper arm that reads no harm and raises no flag.

Sources: `reduce/out_r14/reduce_s2.md` (md5 3219ad59) and `.json` (914f6c90), from reduce_s2.py (md5 e2b605d2) on the fetched
trees. Scorer 382281 (array 0-3, score rows 0-15, native and posed, 200 full_test cases each, float32, eager, each arm on its
own snapshot). Regional jobs 382386 / 382387 (probe_s2reg.py: DK-SENS's registered probe with four changes, reduce/REGIONAL.md, diff
43dbafe8; `--regions --ref-only`). Every section-4 check passes on all
16 rows: epoch 500, 200 sample ids equal to the control's, SNAPSHOT_USED, DATASET_USED, the extra. The regional gate
passes on all 8 runs, and each run's SETTING line matches its arm. The control reproduces its registered seeds to
<= 1.8e-6, and its s_c is 0.0022 / 0.0395 / 0.0078 as registered.

### R1 (native decides; arm ÷ control as ratios of arithmetic seed means; [s42, s43] seed-matched)

| arm | pressure | in-plane velocity | nut | R3 reading |
|---|---|---|---|---|
| K-THETA2 | 1.071 [1.063, 1.079], **no change**, flag **up** | 1.040, no change | 0.980, no change | **not a free cost cut** (up flag on pressure): to the scope freeze as a cost-against-accuracy trade, or dropped |
| K-THETA05 | 0.936 [0.939, 0.934], **no change**, flag **down** | 0.950 [0.942, 0.958], **no change**, flag **down** | 0.932 [0.883, 0.982], **no change**, flag **down** | **not a candidate** under R3's tighter-arm rule (it needs "helps" on a field; each gain is below the 10 % bar) |
| K-B1 | 1.013, no change | 0.988, no change | 0.991, no change | **candidate for the scope freeze (free cost cut)**; stage 3 proposed; adoption also needs Peter's word (architecture) |
| K-BASIS8 | 1.054, no change | 1.027, no change | **1.287 [1.297, 1.278], harms**, flag up | **not a free cost cut** (harms nut): to the scope freeze as a trade, or dropped |

2 SE per field and arm, the case medians, cases won, posed (same labels everywhere) and arm ÷ ISLA are in the reducer's
table. Arm ÷ interacting ISLA (control 0.994 / 1.129 / 1.342): K-THETA05 0.931 / 1.072 / 1.251, K-B1 1.007 / 1.115 /
1.330, K-THETA2 1.065 / 1.173 / 1.315, K-BASIS8 1.048 / 1.159 / 1.727.

### R4, the far-field gap (registered pooled e_R; D_R and 2 SE_R from each pair's own seeds)

| arm | velocity, wake | nut, beyond one chord |
|---|---|---|
| K-THETA2 | +0.029 vs 0.184, level | +0.048 vs 0.531, level |
| K-THETA05 | **-0.122 vs 0.112, narrows** (seeds 0.899, 0.870) | -0.077 vs 0.194, level |
| K-B1 | -0.003 vs 0.119, level | +0.167 vs 0.211, level |
| K-BASIS8 | +0.092 vs 0.108, level | +0.516 vs 0.778, level |

DK-GAP's case-normalized statistic, beside, gives the same eight labels. K-THETA05's velocity-wake error ÷ ISLA's (DK-GAP's
case-normalized form) is 1.73 / 1.68 against the control's 1.95 / 1.99. **Q3 is incomplete** until K-EXACT's regional read; K-THETA05 alone already
narrows a cell, and no arm read so far widens one.

### Predictions (arms 1-4)

P1 (K-THETA2 no harm, p 0.45): **held**. P2 (K-THETA2 up flag, p 0.60): **held** (pressure). P3 (K-THETA05 helps on a
field, p 0.20): **failed** (gains of 5-7 %, below the 10 % bar, flagged down on all three fields). P4 (K-B1 no harm, p 0.40):
**held**. P5 (K-BASIS8 a scope-freeze candidate, p 0.45): **failed** (harms nut). P9 (a finer read narrows the gap,
p 0.25): pending Q3.

### Beside the rules (not verdicts)

- **K-THETA05.** Its case means are below the control's on all three fields, in both seeds (seed-separated). In the
  wake it improves in-plane velocity; pressure (1.02 / 0.99) and nut (1.02 / 1.01) there are level. It is not
  uniformly better: it wins 68-75 % of cases, and nut is worse in the outer zone (1.10 / 1.13) and in the 0.01-0.1 chord
  band (1.12 / 1.07). Its gains are below the registered 10 % bar, so R3 cannot make it a candidate, and without
  K-EXACT R2 cannot say whether the ladder converges. This is the first seed-separated measurement in which a tighter
  trained read is more accurate on AirfRANS. E5's earlier ladder on older code (`code_dtt12`; research/dtt/REPORT.md)
  saw theta 0.5 ÷ theta 1 at 0.95 / 0.99 / 1.00, unresolved (pressure seeds 0.85 / 1.04).
- **Car-scale cost of a tighter or looser read.** Stage 1's cost model measured the code before DTT-FAST (code_cmpaf_fix
  and code_geobundle2). There, on DrivAerML Volume with the whole boundary at 200k tokens, theta 0.5 reads 5,830 pairs
  per point: about 7.0x theta 1's 833, and 3.2x theta 0.5's own box-only read. On that code 400k points fit a GB300
  only at theta >= 1.5. DTT-FAST (4d281a702, now on psharpe/isla) runs 400k whole-boundary points at theta 1 in 1,792 ms
  / 212 GiB (scratch/isla-dtt-fast-2026-10-23/final2/TABLE.md). On the new code, therefore, theta 2's case rests on
  step time, not on the memory fit, and theta 0.5's fit is not measured. These arms trained on code_cmpaf_fix, as the
  control did.
- **K-B1.** No field moves under R1. Beside R1, nut beyond one chord is worse in both seeds (1.24 / 1.13; D_R +0.167
  against 2 SE 0.211, so level). It saves 5.0 % of the step (0.887 s against 0.934 s). In GPU-h it saves 7.6 % against
  the control pair (204.2 against 220.8), but about 2 % of that every arm shows against the control (K-THETA2 216.4 at
  an equal step); against the arms trained alongside it the saving is 5.7-6.1 %, in line with the step. Stage 1
  measured the per-block reads at 9 % of the AirfRANS step and 26 % of the lug's (eager, a lower bound); it gives no
  car-scale share, so the car-scale saving of K-B1 is not known.
- **Basis 8.** The basis is not slack for nut in training. At inference stage 1 put basis 8 at 1.07 / 1.34 on nut (and
  1.11 / 1.28 averaged over the fields; cap/cap_table.md, basis_kw_8). Retraining recovered pressure and velocity but not
  nut (1.287). It saves no time (0.934 s, the control's step).
- **Cost per arm** (two seeds; sacct GPU-h; median step; peak memory per run from the lanes' `Mem:` field), beside the
  control r090eag_af_dtt (110.8 / 110.0 GPU-h, 220.8 per pair; 0.931 / 0.937 s):

| arm | GPU-h (2 seeds) | median step | peak memory per run |
|---|---|---|---|
| control | 220.8 | 0.934 s | - |
| K-THETA2 | 216.4 | 0.935 s | 49.9 / 49.6 GB |
| K-THETA05 | 217.5 | 0.939 s | 50.4 / 50.3 GB |
| K-B1 | 204.2 | 0.887 s | 47.6 / 47.7 GB |
| K-BASIS8 | 216.9 | 0.934 s | 49.7 / 49.9 GB |
| DTT-NOSC-STD (fix-trained; against B's 133.1, 0.57 s) | 133.3 | 0.572 s | 41.3 / 41.0 GB |

## K-EXACT, R2 (the ladder) and Q3 (read 2026-10-26 06:27 EDT; eager; second-read and corrected 06:34, RESULTS_EXACT_REVIEW.md)

Sources: `reduce/out_final/reduce_s2.md` (md5 adfd6f8a) and `.json` (e289de96), from reduce_s2.py (md5 e2b605d2) on the
full fetched trees. Arms 1-4's and arm 6's rows reproduce out_r14 / out_r5 unchanged. K-EXACT (the exact read: every point
reads every boundary cell, theta 0) trained 500 epochs on code_dttk_exact, eager, in 10 links per seed. Scorer 400809
(array 4: score rows 16-19). Regional job 400920. Every section-4 check passes on all 4 rows, the regional gate passes, and
the SETTING lines read theta 0.

### R1 and R3 for K-EXACT

| field | K-EXACT ÷ control [s42, s43] | D vs 2 SE | label | flag |
|---|---|---|---|---|
| pressure | 0.957 [0.956, 0.957] | -0.044 vs 0.004 | no change | **down** |
| in-plane velocity | 0.976 [0.933, 1.022] | -0.024 vs 0.066 | no change | - |
| nut | 0.879 [0.756, 1.003] | -0.129 vs 0.272 | no change | - |

R3 reads **not a candidate**: no field helps. Arm ÷ interacting ISLA is 0.951 / 1.102 / 1.179, against the control's
0.994 / 1.129 / 1.342. Posed gives the same labels.

### R2, the ladder (theta 2, 1 = the standing, 0.5, 0), registered labels

| field | D_2, D_0.5, D_0 | adjacent steps (D_tight - D_loose against 2 SE) | label |
|---|---|---|---|
| pressure | +0.069, -0.066, -0.044 | (2 to 1) -0.069 vs 0.012; (1 to 0.5) -0.066 vs 0.009; (0.5 to 0) **+0.022 vs 0.009, rises** | **pure cost knob** |
| in-plane velocity | +0.039, -0.052, -0.024 | none rises | **pure cost knob, free to theta 2** |
| nut | -0.020, -0.070, -0.129 | none rises | **pure cost knob, free to theta 2** |

**Overall: pure cost knob, free to theta 2.** Pure cost knob is carried by all three fields, the sub-label by two. Pressure's
rising (0.5, 0) step enters only the "converges" test, so it does not affect "pure cost knob". K-THETA05's and K-EXACT's
seed-separated pressure gains of 4-6 % (their down flags) do not enter the label either: a flag changes no label, while
K-THETA2's up flag is part of the sub-label's registered test. In the knob rule's terms, R2 reads
pure cost knob (registered gloss: the approximation costs nothing at theta <= 1); its "converges" label does not apply,
because retrained at theta 0.5 and at the exact read no field gains past the registered 10 % bar, and pressure rises from
theta 0.5 to 0. R2 is a retrained-ladder test: it does not test the book's knob-rule credit for pose invariance, which
rests on the read's error at fixed weights vanishing as theta -> 0 (stage 1: 3.9-27 % per block at theta 1, falling
with theta). Whether that credit stands is the supervisor's call. On velocity and nut, theta is a pure cost knob from 0 up to 2. On pressure
it is a pure cost knob up to 1. At theta 2 pressure pays a seed-separated 7 % (the up flag of R1), so pressure does not
carry "free to theta 2".

### Q3 and R4 for K-EXACT

K-EXACT: velocity in the wake reads level (D_R +0.001 vs 0.086). Nut beyond one chord **narrows** (-0.248 vs 0.213; seeds
0.816 / 0.751), and ISLA-DTT ÷ ISLA there falls from 1.48 / 2.47 to 1.15 / 1.10 (DK-GAP's form). With K-THETA05's narrowing
of wake velocity, and no arm widening either cell: **Q3, a finer read narrows the far-field gap** (from K-THETA05 and
K-EXACT).

### Predictions (the rest)

- P6 (K-EXACT helps on a field, p 0.25): failed.
- P7 (K-EXACT harms on a field, p 0.20): did not occur.
- P8 (R2 overall, if K-EXACT was trained and read): pure cost knob (p 0.45) **held**.
- P9 (a finer read narrows the gap, p 0.25): **held**.
- P11 (the eager smoke projects above 300 GPU-h): **held** at 310. Under the amended gate (400) the arm ran; it used
  317.6 GPU-h.

### Beside the rules (not verdicts)

- **The ladder in plain terms.** No rung of the theta ladder moves any field past the registered 10 % bar. Within it, pressure
  falls a resolved 6-7 % per rung from theta 2 to 0.5 (+0.069, 0, -0.066 in D), then rises a resolved 2 % to the exact
  read (the two arms' pressure seeds agree within 1 %). So on pressure the most accurate rung is theta 0.5, not the
  exact read. Velocity and nut move by less than 2 SE at every rung. Apart from that, K-THETA05's gains on them are
  seed-separated (down flags), and on nut the exact read has the lowest mean (0.879) with its seeds at 0.756 and 1.003.
  E5 (research/dtt/REPORT.md, older code) read its ladder as "inside the seed spread".
- **Where the finer reads help is the far field,** the part of the gap that stage 1 located, and each finer arm narrows
  a different cell. Theta 0.5 narrows wake velocity (-0.122 against 2 SE 0.112) and leaves nut beyond one chord level.
  The exact read narrows nut beyond one chord: registered ratios 0.82 / 0.75, D_R -0.248 against 2 SE 0.213, a 16 %
  margin. It leaves wake velocity exactly level (1.001 / 1.000; ÷ ISLA 1.97 / 1.96). In DK-GAP's case-normalized form,
  ISLA-DTT ÷ ISLA beyond one chord falls from 1.48 / 2.47 to 1.15 / 1.10, still 10-15 % above interacting ISLA, and
  seed 43's 0.45 of the control in that form rests on a few cases. The whole-domain errors (R1) do not resolve these
  regional gains.
- **Cost.** K-EXACT steps 1.354 s against the control's 0.934 s on AirfRANS (1.45x; 1,007 pairs per point against
  21.6). It used 317.6 GPU-h for two seeds against the control's 220.8, at 85.3 GB peak against the other eager arms'
  47.6-50.4 GB. On the car the exact read is out of reach: the model reads about 200k boundary tokens there.
- **For the scope freeze, on AirfRANS.** No rung of the theta ladder moves any field past the registered bar, and on AirfRANS
  none is cheaper to train, apart from K-B1. Theta 2 saves no step time (0.935 against 0.934 s) and costs a
  seed-separated 7 % on pressure; its case is the car's step time (see the arms 1-4 section). Theta 0.5 gains a
  seed-separated 5-7 % on every field at +0.5 % step time (0.939 s; its pair used 217.5 GPU-h against the control's
  220.8). On the car, stage 1's cost model (code before DTT-FAST; the pair count carries over, its time is unmeasured)
  put it at about 7x theta 1's pairs.

## Stage-2 compute (sacct, with requeued attempts)

| item | GPU-h | cap |
|---|---|---|
| arms 1-4 and 6, training | 988.3 (216.4 / 217.5 / 204.2 / 216.9 / 133.3) | |
| smokes 366116 / 366117, scorers 376686 / 382281, regional 382386 / 382387 | 2.19 + 1.80 + 0.72 = 4.71 | |
| **arms 1-4 and 6 in all** | **993.0** | 1,050 |
| K-EXACT training | 317.6 | |
| K-EXACT scorer 400809, regional 400920 | 0.40 + 0.42 = 0.82 | |
| **K-EXACT in all** | **318.4** (its smoke units are counted above) | 400 |
| **stage 2** | **1,311.4** | 1,450 |

Stage 1 used 21.76 GPU-h (cap 30), and DK-COHCOUNT used cpu-short only. The campaign used about 1,333 GPU-h.
