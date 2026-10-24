# REF-090 results: the final record (2026-10-23 06:15 EDT; the last item, the lug ISLA-DTT seed 42, read 11:12 EDT by REF-090-CLOSE2: lug_dtt_read.md)

Pre-registration: PREREG.md (§§1-11 before submission; §12-§15 amendments, §13 a deviation for Tensile2d; md5 at this writing
6ebab6ab). Jobs: JOBS.md. Interim record: RESULTS_partial.md. Reductions: `reduce_r090.py` → r090_final.json /
r090_final_tables.md; `t2d_fourseed.py`, `t2d_control.py`, `eag_reads.py`. All scores: float32, final checkpoint (epoch 500;
DeepJEB++ 40), the partner's scored set and case counts (checked by assertion in every reducer).

"Compiled path" marks numbers from lanes trained and scored in ISLA's per-block compile with pytorch#195845 live (AirfRANS,
Tensile2d, the lug; COMPILE-AF). SHIFT-SUV, DrivAerML Surface and Volume, and DeepJEB++ are unaffected.

## For the book (standings that change, baseline ÷ ISLA 0.9.0, native)

- **AirfRANS** (r090eag_af_isla, eager-trained): GeoTransolver 256 0.874 / 6.290 / 1.113 and Transolver 512 × 16 0.874 /
  4.669 / 1.160 (pressure / velocity / eddy viscosity; ISLA trails on pressure, its error 1.14 times theirs, and leads on the other two). A2
  stood at 0.606 / 4.228 / 0.766 and 0.606 / 3.138 / 0.799.
- **DrivAerML Surface** (r090_dr_isla): GeoTransolver 1.107 / 1.051 (ISLA leads), Transolver 1.029 / 1.024 (level); A2 1.022 /
  0.994 and 0.951 / 0.968.
- **DrivAerML Volume** (r090_vol_isla): GeoTransolver 1.253 / 1.732 / 0.998, Transolver 1.289 / 1.946 / 1.041; A2 1.176 / 1.515
  / 0.960 and 1.211 / 1.702 / 1.000.
- **SHIFT-SUV** (r090_suv_isla): GeoTransolver 1.023 / 0.999 (level), Transolver 0.942 / 0.961 (ISLA trails); A2 0.998 / 0.983
  and 0.919 / 0.945.
- **DeepJEB++** (r090_jb_isla): GeoTransolver 1.251 / 1.394, Transolver 1.098 / 1.175 (ISLA leads); A2 1.269 / 1.395 and
  1.114 / 1.176.
- **Tensile2d** (r090eag_t2dz_isla 42-45, eager-trained): GeoTransolver 0.473 / 0.478 / 1.684, Transolver 0.649 / 0.619 / 0.843
  (displacement / stress / plastic strain); A2 0.584 / 0.682 / 1.111 and 0.800 / 0.883 / 0.556.
- **lug** (r090eag_lug_isla, eager-trained): GeoTransolver 1.423, Transolver 1.378 (ISLA leads every seed); A2 1.297 / 1.256.
- **HiLift** (h090_hl_isla, R-HL090): GeoTransolver 0.913 / 0.924, Transolver 0.922 / 0.912.
- **lug ISLA-DTT** (secondary arm, not a standing; r090fix_lug_dtt, fix-trained): von Mises 0.002027 (0.002219, 0.001834),
  1.394 × ISLA 0.9.0 fix-trained and 1.288 × the eager-trained ISLA standing; both ISLA-DTT seeds lie above every fix- and
  eager-trained ISLA seed. Posed ÷ native 1.53 / 1.98 (ISLA eager-trained 1.25 / 1.62). New text only (no chapter shows a lug
  ISLA-DTT number): 16 and 20 sec-oq-dtt.

## 1. R1, as registered (original rows, scorer numbers): **confirmed**

Pooled G over the 13 low-noise fields **0.891**; no field ≥ 1.10 on both seeds (largest q 1.053, Tensile2d displacement).
R1 cannot un-adopt 0.9.0 (registered), so "confirmed" reports that the standings move; it is not a test of the adoption.

| dataset | fields | 0.9.0 ÷ A2 | q42, q43 | G_d |
|---|---|---|---|---|
| SHIFT-SUV | pressure / wall shear | 0.975 / 0.983 | 0.999, 0.952 / 0.992, 0.975 | 0.979 |
| DrivAerML Surface | pressure / wall shear | 0.924 / 0.946 | 0.925, 0.923 / 0.945, 0.946 | 0.935 |
| DrivAerML Volume | p / v / nut | 0.939 / 0.875 / 0.961 | 0.929, 0.950 / 0.869, 0.881 / 0.954, 0.969 | 0.924 |
| DeepJEB++ | von Mises / displacement | 1.014 / 1.001 | 1.001, 1.027 / 0.958, 1.044 | 1.008 |
| Tensile2d (compiled path) | displacement (contested field) | 1.041 | 1.053, 1.030 | 1.041 |
| AirfRANS (compiled path) | p / v / nut | 0.659 / 0.681 / 0.709 | 0.695, 0.623 / 0.691, 0.670 / 0.728, 0.690 | 0.683 |
| lug (compiled path; noisy, no verdict) | von Mises | 1.104 | 1.052, 1.155 (seed-matched 1.134, 1.077) | – |

No G_d > 1.05 (none listed for Peter; Tensile2d's 1.041 has a registered false-alarm rate of 0.045 at G_d > 1.03).
DeepJEB++ (scorers 312784, 314657): von Mises 0.39747 (0.39238, 0.40257) against 0.39185; displacement 0.16549 (0.15831,
0.17266) against 0.16532.

**Fields above 1.03 or with both 0.9.0 seeds above both A2 seeds** (flag = both): Tensile2d displacement 1.041 (flag) and stress
1.135 (noisy; flag); the lug von Mises 1.104 (seeds overlap: 0.001816 < 0.001850; no flag); HiLift wall shear 1.022 under
R-HL090 (both seeds above, under 1.03; about 10× A2's 0.2 % seed gap).

**Beside R1.**
- Eager re-scores of both sides on the three affected datasets (COMPILE-AF): pooled G 0.890; Tensile2d 1.045, AirfRANS 0.678,
  lug 1.099. No verdict, flag or band changes.
- Under one training function: Tensile2d eager-trained ÷ eager-trained (r090eag 42-45 ÷ a2reag 42/43) displacement 1.200
  (0.998 on seeds 42/43), stress 1.366 (1.079), plastic strain 0.885 (1.069); Tensile2d fix-trained ÷ fix-trained (seeds
  42/43) displacement 0.831, stress 0.944; AirfRANS ISLA fix-trained ÷ bug-trained A2 (eager) 0.678 / 0.665 / 0.697 ("mixed
  training paths"); lug fix-trained ÷ bug-trained A2 0.843 and eager-trained ÷ bug-trained A2 0.913 ("mixed training paths").
- **Comparability check** (job 312826, setcheck/SETCHECK.md): the scored points and targets of the 0.9.0 and A2 loaders are
  identical per case on DrivAerML Volume (48), DrivAerML Surface (48) and SHIFT-SUV (99); on Volume only the vehicle boundary
  input differs (0.9.0 reads it whole: the 0.9.0 change itself). LHR's data copy; the DrivAerML partners were scored on JHB's,
  whose manifest md5 matches. No R1 reading changes.

## 2. R3, posed ÷ native (seeds 42, 43)

SHIFT-SUV, DrivAerML Surface, DeepJEB++ 1.0000-1.0001. Tensile2d 1.0000 (A2 1.06-1.12 on stress and displacement, 1.01 / 1.00
on plastic strain). AirfRANS 1.0015-1.0028 (p, v), 1.0000 (nut). DrivAerML Volume 1.0027-1.0040 (p, v), 0.9998-1.0001 (nut):
the Morton draw of the vehicle tokens, ordered along the dataset's axes, changes under the pose. Lug 1.141 / 1.107 (A2 4.14 /
2.87). Area: DrivAerML Surface 1.0016-1.0157; elsewhere area = native. **DrivAerML Volume's area protocol is degenerate at
0.9.0** (ISLA draws by area in the model, so area equals native by construction): no robustness claim; REMESH-090 measures it.

## 3. Predictions P1-P14

P1-P4 held (SUV 0.979, DrivAerML S 0.935, V 0.924 and ≤ 1.00, DeepJEB++ 1.008). P5 failed (Tensile2d displacement 1.041; r ≤
0.97 failed too; contested field). P6 held (AirfRANS G 0.683; confirmed). P7 held (lug 1.104 in [0.85, 1.20]). P8 held
(0.891). P9 held. P10 held. P11 held at ±0.01, failed at ±0.002 (Volume velocity 1.0040). P12 held (1.141 / 1.107). P13:
AirfRANS part failed as registered (ISLA-DTT ÷ ISLA on the r090 rows 1.163 / 1.147 / 1.396, G 1.230, against ≤ 1.05); beside
it, fix-trained 1.035 / 1.126 / 1.381 and eager-trained 0.994 / 1.129 / 1.342 (G 1.146) also exceed 1.05. **Lug part failed**
(r090fix_lug_dtt ÷ r090fix_lug_isla 1.394 against [0.8, 1.25]; seed-matched 1.517 (s42), 1.270 (s43); both ISLA-DTT seeds
above both ISLA seeds). Beside it: ÷ r090eag_lug_isla 1.288, ÷ r090_lug_isla (bug-trained) 1.065. So P13 failed on both parts. P14 failed (the r090 lug ISLA-DTT lanes were
cancelled before epoch 500).

## 4. R2, the standings (ISLA rows; baselines are the book's standing arms, results/char/<dataset>_a2.json)

| dataset | ISLA standing rows | status | baseline ÷ ISLA (native; GeoTransolver 256 / Transolver 512 × 16) |
|---|---|---|---|
| SHIFT-SUV | r090_suv_isla | final | p 1.023 / 0.942, wss 0.999 / 0.961 |
| DrivAerML Surface | r090_dr_isla | final | p 1.107 / 1.029, wss 1.051 / 1.024 |
| DrivAerML Volume | r090_vol_isla | final | p 1.253 / 1.289, v 1.732 / 1.946, nut 0.998 / 1.041 |
| DeepJEB++ | r090_jb_isla | final | vM 1.251 / 1.098, disp 1.394 / 1.175 (ISLA leads every seed) |
| HiLift | h090_hl_isla (R-HL090) | final | p 0.913 / 0.922, wss 0.924 / 0.912 |
| Tensile2d | r090eag_t2dz_isla 42-45 (eager-trained; §13 (b)) | final | disp 0.473 / 0.649, stress 0.478 / 0.619, q 1.684 / 0.843 |
| lug | r090eag_lug_isla 42/43 (eager-trained) | final | vM 1.423 / 1.378 (ISLA leads every seed); posed 416 / 447 |
| AirfRANS | r090eag_af_isla 42/43 (eager-trained) | final | p 0.874 / 0.874 (ISLA trails), v 6.290 / 4.669, nut 1.113 / 1.160 (ISLA leads); compiled-path rows were 0.919 / 0.919, 6.212 / 4.610, 1.080 / 1.127 |

AirfRANS r090eag_af_isla: pressure 0.012479 (0.013163, 0.011794), velocity 0.006355 (0.006441, 0.006270), eddy viscosity
0.015353 (0.015738, 0.014968); posed ÷ native 1.0022-1.0025 / 1.0016 / 1.0000; ÷ A2 (bug-trained, eager-scored; mixed training
paths) 0.692 / 0.668 / 0.687 (eag_af_isla.md). Values: Tensile2d r090eag displacement 0.00849 (0.00783, 0.00629, 0.00891, 0.01092), stress 0.01113, plastic strain 0.273
(t2d_standing.md); lug r090eag von Mises 0.001574 (0.001376, 0.001771), posed 0.002292 (eag_lug.md). Secondary arms (not
standings): AirfRANS ISLA-DTT → r090eag_af_dtt, eager-trained: pressure 0.012408 (0.012428, 0.012389), velocity 0.007173
(0.007373, 0.006973), eddy viscosity 0.020598 (0.020711, 0.020484); ÷ r090eag_af_isla 0.994 / 1.129 / 1.342; posed ÷ native
1.0017-1.0022 / 1.0010-1.0012 / 1.0001 (eag_af_dtt.md). Lug ISLA-DTT → r090fix_lug_dtt, labelled fix-trained (§15 (c)):
von Mises 0.002027 (0.002219, 0.001834), posed 0.003509 (0.003391, 0.003627), posed ÷ native 1.53 / 1.98; ÷ r090fix_lug_isla
1.394 (posed 1.770), ÷ r090eag_lug_isla 1.288 (posed 1.531), ÷ r090_lug_isla 1.065 (posed 1.641) (lug_dtt_read.md).

## 5. The compiled-path defect: retrain (§12) and training-path reads (§§13-15)

**§12(c), T = fix-trained ÷ bug-trained (both eager-scored), against the A2-pair gap:**

| dataset / model | field | T (seeds) | gap | fires? |
|---|---|---|---|---|
| Tensile2d ISLA (seeds 42/43) | displacement / stress / q | 0.966 (1.007, 0.924) / 0.993 / 0.629 | 0.025 / 0.20 / 0.22 | **yes** (displacement) |
| AirfRANS ISLA | p / v / nut | 1.024 (1.027, 1.020) / 0.985 / 0.998 | 0.061 / 0.051 / 0.086 | no |
| AirfRANS ISLA-DTT (secondary) | p / v / nut | 0.898 (0.885, 0.911) / 0.961 / 0.962 | – | (no rule) |
| lug ISLA | von Mises (noisy) | 0.768 (0.821, 0.720) | 0.144 | not formally (noisy field); real (PREREG §14a-b) |

Tensile2d on four seeds against four (t2d_fourseed.md): fix ÷ bug 1.099 / 1.075 / 0.778; seed ranges overlap and no ratio
exceeds 2 SE, so under §13's rule fix-trained and bug-trained **match**. A2: a2rfix ÷ a2r 1.216 / 1.194 / 0.525.

**§13, Tensile2d control: overall (b), as registered, on weak evidence** (t2d_control.md). A2 reads (b) on displacement
(eager ÷ fix 0.853 at 1.06 × 2SE; ÷ bug 1.037) and stress; ISLA 0.9.0 reads (c) on every field (displacement and stress lean
(a), plastic strain leans (b)); the registered weighing lets A2 decide. Consequence: eager-trained standings (§4).

**§15, the lug read (decisive for the fixed path): (c)** (eag_lug.md; PREREG §15a). Eager ÷ fix 1.083 (0.31 × 2SE), eager ÷
bug 0.831 (0.66 × 2SE); the eager pair's 25 % seed gap prevents separation; leaning (a). Predictions L1 failed, L2-L4 held, L5
failed. Lug ISLA-DTT eager pair stays held.

**§14, AirfRANS read: ISLA (c)** (eag_af_isla.md, scorer 344920): eager ÷ fix 1.021 / 1.005 / 0.986 (|d|/2SE 0.12 / 0.11 /
0.25), eager ÷ bug 1.045 / 0.989 / 0.985 (0.27 / 0.25 / 0.21); all three training paths within 5 %, seed ranges overlap; no
(b), so AF-EXTRAP's and ATTRIB-090's fix-path lanes stand, labelled fix-trained. **ISLA-DTT: (c)** (eag_af_dtt.md, scorers
350103 / 350296; PREREG §14c): pressure (c) (eager ÷ fix 0.981 at 1.37 × 2SE, ÷ bug 0.880 at 2.97 × 2SE: differs from both),
velocity (c) (1.007, 0.968), eddy viscosity (a) (0.958 at 0.28, 0.922 at 1.26 × 2SE); one field of three, so (c).

**Mechanism checks beside the rules (never in them).**
- Lockstep v1 (317142, the registered bf16 check): inconclusive. Eager is bitwise deterministic (zero floor), and the fixed and
  bug paths both leave eager by bf16 roundoff (300-step parameter distance 0.94 and 1.13 of eager's displacement).
- Lockstep v2/v3 (317294, 323041; reference eager float32 gradients): the fixed path's fp32 gradient equals eager's (medians
  4e-6 to 3e-5; 2.7e-4 on the A2 fix-trained checkpoint); the bug's is off by 7-106 %; in bf16 all three paths sit at the
  eager-bf16 floor (0.12 at epoch 51, 1.0-1.4 late).
- **MECH-090** (scratch/isla-mech090-2026-10-22/RESULTS.md): the fixed path's fp32 gradients sit at a one-ulp eager floor on
  eight Tensile2d checkpoints of both families (0.55-1.19× the floor; the bug 1,574-32,531×), and its 300-step trajectory
  drifts from eager no faster than a one-ulp copy; so §13's A2 (b) is best read as two-seed chance (the verdict stands as
  registered). The bug was an attention sink in the slice-state softmax: k = (−N) mod 4 phantom tokens of raw measure 1 with
  the block's geo_logit bias, about 37× a slice's real mass on about half the lug's training cases, consistent with T = 0.768.
- Certifications: a2rfix Tensile2d (315498, d ≤ 5.9e-5 against 4.5e-2 without the variable); a2r_lug under the variable
  (343190: MIXOFF d ≤ 1.7e-7 on 100 cases; its float64 floor unit is invalid on code_a2r, the float64 model scoring 2-15× worse).
- Tensile2d eager seed 45: best in-training validation 0.01012 at epoch 180, a spike to 0.0188 at epoch 200, then a plateau
  near 0.012 to 0.01196 (not a late spike). σ(log) of displacement: eager 0.23, fix 0.16, bug 0.025; on the lug the opposite
  (fix seeds 1.2 % apart, bug 12 %).

## 6. Secondary arm: lug ISLA-DTT, fix-trained (read 2026-10-23 11:12 EDT; lug_dtt_read.md)

Scorers 354186 (seed 42) and 350102 (seed 43); float32 eager, 600 test cases, epoch 500 (asserted by reduce_r090.py); area
equals native by construction (degenerate). Seed 42 trained through link 345709_0 (07:06-10:41 EDT 10-23) after its link
342582_0 hung on the second quota event.

| r090fix_lug_dtt | value (s42, s43) | ÷ r090fix_lug_isla 0.0014537 | ÷ r090eag_lug_isla 0.001574 (standing) | ÷ r090_lug_isla 0.001904 |
|---|---|---|---|---|
| von Mises, native | 0.002027 (0.002219, 0.001834) | **1.394** (1.517, 1.270) | 1.288 (1.613, 1.036) | 1.065 |
| von Mises, posed | 0.003509 (0.003391, 0.003627) | 1.770 | 1.531 | 1.641 |

Posed ÷ native 1.528 / 1.977 (ISLA fix-trained 1.346 / 1.383, eager-trained 1.246 / 1.619). The ISLA-DTT seeds lie 21 %
apart, and both lie above all four ISLA seeds of the fix- and eager-trained pairs (lowest ISLA-DTT 0.001834 against highest
ISLA 0.001771), so on the lug ISLA-DTT trails interacting ISLA beyond seed noise, by 27-52 % seed-matched against the same
training path. **P13's lug part ([0.8, 1.25], p 0.6) failed.** No registered rule escalates it; the arm is secondary and
changes no standing. Beside it (means only; mixed training paths, the baselines' standing arms): GeoTransolver 256 ÷ lug
ISLA-DTT 1.105, Transolver 512 × 16 ÷ lug ISLA-DTT 1.070. r090_lug_dtt (rows 16-17) was cancelled under the defect
("incomplete as registered"); r090fix_lug_dtt fills it, labelled fix-trained. No A2-era ISLA-DTT lug value exists in the
record.
r090_lug_dtt (rows 16-17) was cancelled under the defect ("incomplete as registered"). r090fix_lug_dtt (kernel/r090fix rows
0-1) fills it, labelled fix-trained. Report against lug ISLA 0.9.0 (r090 0.001904, posed 0.002139; r090fix 0.001454; r090eag
0.001574) and P13's [0.8, 1.25]. No A2-era ISLA-DTT lug value exists in the record (searched the book, results/, scratch and
LHR runs/).

## 7. What ran (LHR)

Training: r090 wave 298076-298093 (and their chains); r090fix 312864-312865, 312884-312885, 313191-313192, 313357-313358,
313540-313541; r090fix2 315442-315443; a2rfix 315538-315539; r090eag Tensile2d 317148-317151; a2reag 317152-317153;
r090eagaf 323068-323069, 323076-323077, 323083-323084. Smokes 312815, 323028. Scorers: 312784, 314657 (DeepJEB++), 315170,
317020, 317050, 321760, 322902, 325815, 326815, 330338, 343170. Checks: 312826 (setcheck), 315431/315466/315498 (a2rfix cert),
317142, 317294, 323041 (lockstep), 343190 (certlug). Later scorers: 343170 (lug eager), 344920, 350103, 350296 (AirfRANS
eager), 350102 (lug ISLA-DTT s43), 354186 (lug ISLA-DTT s42). PROBES-090 (f): 350362, 350363, 350364 (read in
scratch/isla-probes090-2026-10-21/RESULTS.md §7b; NOTEBOOK_DRAFT_PROBES_F.md). Lug ISLA-DTT s42 links: 342582_0 (hung,
quota), 345709_0, 351296_0 (exited at once).
