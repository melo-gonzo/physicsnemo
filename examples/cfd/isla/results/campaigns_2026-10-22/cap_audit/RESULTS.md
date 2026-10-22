# CAP-AUDIT results (written 2026-10-21 from 02:36 EDT)

> Corrected 2026-10-22 11:15 EDT after the reviewer's second read (scratch/isla-review-2026-10-21/SECOND_READ_2026-10-22.md): AirfRANS pressure ratio 2.70 / 2.61; AirfRANS nu_t ratio 1.57 / 1.36 (Transolver is the best baseline at seed 43); the 9-15,000x gap range is without AeroZoo (81 pairs); the phase-1 median G on matched runs is 36; the fixed-draw fit range excludes the AeroZoo baselines (0.085-4.3); ISLA-DTT's AirfRANS error is the most concentrated (1.6 % of points against ISLA's 2.6 %); the headline reads 'on the three fresh-draw cases (drs, hl18, af)'.

These are Peter's representation-capacity audit results. One model is trained on one case until it is overfit, and its
floor is compared with GeoTransolver 256 and Transolver 512 x 16 on the same case, points and budget.
- Pre-registration: `PREREG.md`. Sections 1-12: phase 1. Section 13: phase 1b. Section 14: the amendment before the
  phase-1 results, with its disclosure. Section 15: phase 2.
- Jobs: `JOBS.md`.
- Full tables, reduced on LHR after every run had ended: `tables/phase1_floors.md` (md5 4792c802 on LHR, from `reduce.py`; the local copy adds a 15.8 note at the top) and
  `tables/phase1b_floors.md` (md5 9e1484de, from `reduce_rd.py`), with their JSON beside them.
- Residual maps: `tables/resid_*.md`, from `tables/residuals_p1_p1b.csv` (`harness/residuals.py`,
  `harness/resid_tables.py`).

Compiled path: no registered run used ISLA's per-block compile. Every ISLA row of `rows_full.tsv` and
`rows_rd_full.tsv` has compile_blocks false (resolved configs checked; COMPILE-AF's inventory agrees). Only the four
phase-1 smoke rows `cap_*_isla_cb_smk_s42` compiled, and they carry no number. All floors below are eager.

## Phase 1: fixed point-set fitting (60 of 60 runs complete, all losses finite)

Phase 1 fixes one draw of each case and trains on it. **Its floors measure how well a model fits a fixed point set**
(section 14, rule 1). On every separate-draw case (DrivAerML, HiLift, AirfRANS), every arm's held-out error is 9 to
15,000 times its training-point floor (G, median 40; 81 pairs; AeroZoo is not in this range). The phase-1 deficits below are therefore deficits
in fitting a fixed point set, not in representing the field. The exception is AeroZoo's baselines, which fit nothing
(G about 1).

**Invalid (frame mismatch; PREREG 15.8): ISLA's phase-1 held-out values on DrivAerML Volume.** There the held-out set
paired epoch 1's points with epoch 0's boundary. CenterMesh centers each draw on its own mean, so ISLA, which computes
its wall geometry from the boundary, read geometry shifted by 1.16 mm (1.4 times the median wall distance; 15.8 check,
job 318829). These values and their G are excluded everywhere below. ISLA's phase-1 AeroZoo held-out values are valid
(15.8 erratum and check, 2026-10-21 21:01 EDT). aerozoo_uvolume runs no CenterMesh, and on the az case the epoch-0 and
epoch-1 draws read identical boundaries (all four: max point difference 0, cells equal) and identical global data, in
raw coordinates (interior means about -0.03 m, not 0). On AeroZoo ISLA's held-out error is 3.3-4.1 times its training
floor on velocity and 33-38 times on pressure: fixed point-set fitting, as elsewhere. The registered training-point floors and verdicts are not affected. The dumps are
deleted, so phase 1 is not re-scored. The Part B control measures drv with a corrected held-out set.

Training-point floor F (median of the last five logs, steps 19,600-20,000), seeds 42 / 43. Held-out floors are in
`tables/phase1_floors.md`. R = ISLA ÷ the better baseline.

| case | field | ISLA | GeoTransolver | Transolver | R_42 / R_43 | verdict (section 7) |
|---|---|---|---|---|---|---|
| drs | pressure | 0.00188 / 0.00201 | 0.00135 / 0.00133 | 0.00078 / 0.00071 | 2.43 / 2.82 | deficit |
| drs | wss | 0.0053 / 0.0129 | 0.0029 / 0.0029 | 0.0013 / 0.0012 | 4.07 / 10.9 | deficit |
| drv | pressure | 0.00067 / 0.00070 | 6.3e-5 / 5.7e-5 | 0.00016 / 0.00017 | 10.6 / 12.2 | deficit (S2 > 15k: possibly budget-limited) |
| drv | velocity | 0.0023 / 0.0023 | 1.4e-4 / 1.3e-4 | 3.6e-4 / 3.8e-4 | 16.2 / 18.0 | deficit |
| drv | nu_t | 0.0010 / 0.0011 | 4.5e-5 / 4.4e-5 | 1.8e-4 / 1.9e-4 | 22.2 / 23.9 | deficit (possibly budget-limited) |
| hl18 | pressure | 0.00217 / 0.00203 | 0.00136 / 0.00139 | 0.00077 / 0.00081 | 2.83 / 2.51 | deficit |
| hl18 | tau_wall | 0.0119 / 0.0111 | 0.0068 / 0.0068 | 0.0035 / 0.0037 | 3.45 / 3.05 | deficit |
| hl6 | pressure | 0.00157 / 0.00166 | 0.00164 / 0.00169 | 0.00070 / 0.00059 | 2.25 / 2.83 | deficit |
| hl6 | tau_wall | 0.0068 / 0.0080 | 0.0061 / 0.0066 | 0.0017 / 0.0014 | 4.00 / 5.63 | deficit |
| af | pressure | 0.00092 / 0.00078 | 0.00019 / 0.00019 | 0.00021 / 0.00020 | 4.86 / 4.04 | deficit, **budget-limited** (40k: 0.32x) |
| af | velocity | 0.00080 / 0.00081 | 0.00020 / 0.00020 | 0.00021 / 0.00022 | 4.06 / 4.04 | deficit, budget-limited (0.61x) |
| af | nu_t | 0.00060 / 0.00055 | 0.00024 / 0.00025 | 0.00022 / 0.00023 | 2.70 / 2.43 | deficit, budget-limited (0.61x) |
| lug | von Mises | 1.98e-4 / 1.80e-4 | 3.8e-5 / 3.7e-5 | 3.1e-5 / 3.0e-5 | 6.31 / 5.96 | deficit, budget-limited (0.49x) |
| t2d | stress | 0.00133 / 0.00119 | 0.00047 / 0.00052 | 0.00028 / 0.00026 | 4.83 / 4.65 | deficit (40k: 0.81x, not budget-limited) |
| t2d | displacement | 3.4e-4 / 2.7e-4 | 1.8e-4 / 1.7e-4 | 1.6e-4 / 1.6e-4 | 2.16 / 1.75 | level |
| t2d | q | 6.5e-4 / 5.5e-4 | 1.6e-4 / 1.8e-4 | 1.0e-4 / 9.6e-5 | 6.35 / 5.68 | deficit, budget-limited (0.48x) |
| az | velocity | 0.0049 / 0.0047 | 0.86 / 1.81 | 0.085 / 0.34 | 0.058 / 0.014 | ISLA advantage (scale handling, section 10) |
| az | pressure | 0.0032 / 0.0028 | 1.3 / 4.3 | 0.69 / 0.93 | 0.005 / 0.003 | ISLA advantage (scale handling) |

- **The baselines' floors fall at 2x budget too**, so the deficits are budget-limited by the registered rule but do not
  close. At 40k (seed 42) ISLA ÷ best is: af pressure 2.8, velocity 4.3, nu_t 3.5; lug 7.2; t2d stress 11.5, q 8.0.
- **Steps to 2x floor**, ISLA S2: 7,900-17,700. Transolver's floors are reached later (S2 13,300-19,200); its floors
  are still falling at 20k.
- **bf16** (drs, seed 42), bf16 floor ÷ fp32 floor: ISLA 2.84 (pressure) and 1.80 (wss); GeoTransolver 4.45 / 3.41;
  Transolver 5.48 / 5.03. Pressure is precision-limited for every arm, and ISLA the least (it stays the fastest:
  0.353 s/step, against 0.398 and 0.547).
- **Cost** (fp32, median s/step; wall time = s/step x steps, because reduce.py's wall_s counts only the last segment of
  a requeued run): ISLA 0.15 (t2d) to 2.82 (drv) s/step, GeoTransolver 0.074-1.67, Transolver 0.080-1.65. ISLA is the
  slowest arm on every case except drs, hl18 and hl6, where Transolver is slower (1.64-1.67 against 1.37-1.39). Peak
  memory: ISLA 5.5-232 GiB, GeoTransolver 4-160, Transolver 7-174. Per-run table: `tables/phase1_floors.md`, "Cost".
- **Jobs**: 302616_0 and 302617_1-5 were requeued twice by node failures (restarts 1-2). Runs resumed from their
  checkpoints, and completed runs were skipped. 302617 and 303992 were cancelled at 02:2x EDT, requeued with nothing
  left to run.

### Phase-1 predictions (section 8)

| id | result |
|---|---|
| H1 (ISLA drs pressure F <= 1e-3) | false (0.00188 / 0.00201) |
| H1b (<= 1e-2) | true |
| H2 (>= 40 % of ISLA's area-weighted hl18 pressure error on the 2 % most-suction area) | **false**: 0.119 / 0.124. Transolver 0.007 / 0.008, GeoTransolver 0.036 / 0.030. Under random drawing, ISLA reaches 0.272 / 0.262 |
| H3 (no deficit anywhere) | false (15 of 18 headline fields) |
| H4 (a deficit falls on AirfRANS or Tensile2d) | true as worded; deficits fall there, but on every other non-AeroZoo case too, so they are not specific to those two |
| H5 (in-case interpolation, G <= 1.5 on >= 80 % of the separate-draw pairs) | **false**: 7 of 89 valid pairs (ISLA's drv held-out invalid, 15.8); with ISLA's AeroZoo pairs restored, 7 of 93, median G 36 |
| H6 (2x budget lowers some arm's floor >= 1.5x on each of lug, t2d, af) | true |
| H7 (ISLA bf16 floor >= 2x fp32 on drs pressure) | true (2.84x) |
| H8 (ISLA hl18 pressure F within 1.5x of hl6) | true (1.38x / 1.22x) |
| H9 (all 60 finite) | true |

H2 measures: cap_hl18_isla_s43's dump (overfit.py 935245dd) has no cell measures. They were taken from
cap_hl18_isla_s42's dump, which has the same training-point hash (4780dddaa212), instead of a `--steps 0` rerun; the
draw is deterministic, so this is the same data (`harness/h2_borrow.py`). The recomputed rel L2 equals the logged
floors (0.002169, 0.002028).

**Withdrawn** (section 14, rule 5): the interim explanation that Transolver interpolates best "because it does not read
point-set context". Its physics-attention slices pool all N points too. The observation holds only in part in phase 1. Transolver has the smallest held-out error on drs and drv. It splits with GeoTransolver on hl18 and af. ISLA has the smallest on the lug (0.017 / 0.015) and on Tensile2d q (and stress on seed 42). The explanation does not stand.

## Phase 1b: random-draw control (20 of 20 runs complete, eval sets hash as phase 1)

The held-out floor F_ho is on phase 1's held-out draw after training on a fresh draw every step. Seeds 42 / 43. The
best arm is in bold.

| case | field | ISLA | ISLA-DTT | GeoTransolver | Transolver | ISLA ÷ best baseline |
|---|---|---|---|---|---|---|
| drs | pressure | 0.0166 / 0.0174 | - | 0.0149 / 0.0150 | **0.0119 / 0.0118** | 1.40 / 1.48 |
| drs | wss | 0.0353 / 0.0366 | - | 0.0293 / 0.0293 | **0.0229 / 0.0228** | 1.54 / 1.61 |
| hl18 | pressure | 0.0082 / 0.0076 | - | 0.0055 / 0.0055 | **0.0049 / 0.0048** | 1.69 / 1.60 |
| hl18 | tau_wall | 0.0409 / 0.0408 | - | 0.0345 / 0.0349 | **0.0332 / 0.0323** | 1.23 / 1.26 |
| af | pressure | 0.00165 / 0.00168 | 0.00148 / 0.00219 | **0.00061 / 0.00064** | 0.00097 / 0.00077 | **2.70 / 2.61** |
| af | velocity | 0.00112 / 0.00119 | 0.00125 / 0.00137 | **0.00072 / 0.00086** | 0.00088 / 0.00087 | 1.55 / 1.39 |
| af | nu_t | 0.00165 / 0.00157 | 0.00123 / 0.00121 | **0.00104 / 0.00121** | 0.00129 / 0.00115 | 1.57 / 1.36 |

- **Representation-limited** (section 13 rule): only AirfRANS pressure, for ISLA (2.70 / 2.61 x the best arm) and
  ISLA-DTT (2.43 / 3.40). Every other arm, case and field is under 2x.
- **Against phase 1.** Every arm's held-out floor falls by 3-90x (Q 0.011-0.34). The gap closes, from a phase-1 median
  G1 of 36 (on the matched runs) to G 0.91-2.19. The training-draw floor rises for every arm (P4).
- **The single-case field floors are well below ISLA's cross-case test errors** on the same cases: drs pressure 0.017
  against 0.040, hl18 pressure 0.008 against 0.219, af pressure 0.0017 against 0.015. Fitting one case's field is not
  what limits ISLA's generalization error on these datasets.
- Predictions: P1 true (21/21), P2 false (AirfRANS pressure 2.70 / 2.61), P3 true (DTT G 0.98 against ISLA 1.05),
  P4 true (21/21), P5 true (42/48), P6 true (Q 0.042 / 0.043), P7 false (Transolver's Q is 0.09-0.34: random drawing
  also removes most of its fixed-draw held-out error).

## Residual maps by region (Part C; phase 1 and 1b dumps)

- **AirfRANS pressure** (phase 1b, held-out; `tables/resid_af_rd.md`). By wall distance in chords:
  - ISLA's error relative to GeoTransolver's, per band (e_R ratio): 3.7x at 1e-3 to 1e-2, 3.0x at 1e-2 to 0.1, 2.0x
    below 1e-4, and 1.7x beyond 0.1.
  - The 1e-3 to 0.1 chord band holds 39 % of the points, 67 % of ISLA's squared error and **71 % of ISLA's excess**
    over GeoTransolver.
  - ISLA's error is more localized than the baselines': half of its e^2 sits on 2.6 % of the points (ISLA-DTT 1.6 %,
    the most concentrated; GeoTransolver 9.5 %, Transolver 6.9 %).
  - The same band carries ISLA's excess on velocity (1e-4 to 0.1 chords: 129 % of the excess, while ISLA is better
    than the baselines in the innermost 1e-4) and on nu_t (1e-3 to 1e-2: 35 %).
  - The fixed-draw maps (phase 1) put ISLA's excess in the same 1e-4 to 1e-2 band (81 %).
  - L7 (prediction: >= 50 % of the excess below 1e-2 chords) is not scored yet; it is read on the phase-2 maps.
- **HiLift 18° pressure and tau_wall** (element labels): ISLA's excess spreads over the elements in proportion to their
  error.
  - Random draw, ISLA's e_R over Transolver's: slat 1.4x, main 1.6x, flap 1.6x, other 1.8x.
  - Fixed draw: 2.5-3.6x on every element.
  - No element or region is singled out. The deficit is a uniform scale-up.
- **DrivAerML Surface**: the random-draw excess follows the error, mostly on the lower third of the car (z0: 80 % of
  the excess on 53 % of the area). Every arm's error sits there.
- **DrivAerML Volume, fixed draw** (`tables/resid_drv_p1.md`): ISLA's training-point error is about 10x
  GeoTransolver's in every wall-distance decade. On the held-out set, GeoTransolver's and Transolver's errors sit
  within 1e-3 m of the wall (82 % and 86 % of e^2): point-set memorization of the near-wall points. ISLA's held-out
  map in that table is invalid (frame mismatch, 15.8). The 15.8 check confirms whether the baselines' held-out values
  are unaffected (it requires GeoTransolver to move by under 5 %), and the Part B control tells whether there is a
  field-fitting deficit.
- **Lug and Tensile2d, fixed draw**: ISLA's excess on the lug sits on the 10-50 % deviation band (60 %) and on the top
  2 % (11 %, where Transolver has 2.7 %). On Tensile2d stress the yielded nodes hold 17 % of ISLA's error on 6 % of the
  nodes.
- Checks:
  - The labels' point hashes equal phase 1's (af 09e7f114aa37 / 78a89c8b3164, hl18 4780dddaa212 / 8e740def4117,
    drv f26c3708e117).
  - HiLift elements match through a constant offset (deviation 2.5e-4 in).
  - The decomposition reproduces the logged rel L2 within 2 %, except AirfRANS baseline velocity (z component dropped:
    in-plane only, as the rules require) and Tensile2d stress (the recipe's Frobenius weights; 4-17 %; shares are of
    our own e^2).

## Phase 2 (PREREG section 15 with 15.7-15.11)

Status at 11:45 EDT 10-22: Part B is complete and read. Part A has 10 of 14 runs complete; still running are d24 s45 and
the 40k checks of w1024, d24 and s2048. The 15.9-15.11 runs are queued. Readouts: `harness/reduce_p2.py` (registered),
`reduce_ctl.py` (15.9), `reduce_drvl.py` (15.10), `reduce_p2x.py` (15.11).

### Part B: DrivAerML Volume, random draw (6 of 6 runs complete; corrected held-out set, 15.8)

F_ho on the registered (corrected) held-out set; in brackets, F on the epoch-0 whole draw. Seeds 42 / 43.

| field | ISLA | GeoTransolver | Transolver | R_drv (registered) | R on the epoch-0 draw | verdict |
|---|---|---|---|---|---|---|
| pressure | 0.0101 [0.0108] / 0.0115 [0.0109] | 0.0065 [0.0066] / 0.0065 [0.0066] | 0.0059 [0.0059] / 0.0058 [0.0057] | 1.72 / 1.99 | 1.82 / 1.93 | no deficit |
| velocity | 0.0192 [0.0191] / 0.0200 [0.0194] | 0.0101 [0.0112] / 0.0097 [0.0098] | 0.0088 [0.0089] / 0.0086 [0.0085] | 2.18 / 2.32 | 2.14 / 2.27 | **deficit** |
| nu_t | 0.0239 [0.0245] / 0.0219 [0.0260] | 0.0109 [0.0101] / 0.0106 [0.0100] | 0.0098 [0.0093] / 0.0100 [0.0092] | 2.44 / 2.20 | 2.63 / 2.83 | **deficit** |

- The two draws give the same verdicts, so no field is draw-sensitive.
- Q (random-draw F_ho ÷ phase 1's held-out value) is 0.01-0.06 for every arm, so phase 1's drv gap was point-set
  memorization.
- With fresh draws, ISLA trails Transolver by about 2.2x on velocity and nu_t, and by 1.7-2.0x on pressure (seed 43
  sits at the bar).
- Predictions (15.5):
  - D1 (ISLA not representation-limited on any field, 0.55): failed (velocity and nu_t).
  - D2 (G <= 1.5 everywhere, 0.7): held (G 0.84-1.08).
  - D3 (Transolver lowest on pressure, 0.5): held.
  - D4 (Q < 0.5 for ISLA and GeoTransolver, 0.8): held.
- By the 15.7 rule, the deficit opens a drv ladder: 15.10 runs the down-rungs (the up-rungs do not fit in memory).

**Where the drv deficit sits** (`tables/resid_drv_partB.md`, by wall distance in metres):
- **Velocity:** a uniform multiple. ISLA's e_R is 1.9-2.8x Transolver's in every decade, from below 1e-4 m to beyond
  1 m; 61 % of the excess is at 1e-4 to 1e-3 m only because 45 % of the points are there.
- **nu_t:** regional. The 0.7 % of points more than 1 m from the car hold 49-66 % of ISLA's nu_t error and 61 %
  (held-out) to 72 % (epoch-0 draw) of its excess. There ISLA's e_R is 0.035-0.044, against Transolver's 0.010 and
  GeoTransolver's 0.011; nearer the wall ISLA is 1.4-2.1x.
- The draw-to-draw nu_t difference seen at 10k sits in that same band, which holds only about 1,300 points per draw (s43
  far-field e_R 0.0345 against 0.044; every other band agrees to 0.001). At 20k the totals agree (0.0239 against 0.0245,
  0.0219 against 0.0260).

**Candidate causes of the far-field nu_t miss** (for a decision on a probe; none run). Transolver fits the beyond-1-m
level (e_R 0.010); ISLA does not (0.035-0.044). On drv, ISLA reads only the vehicle boundary and U_inf_dir
(isla_volume.yaml): no inlet, outlet or domain walls, while the baselines read raw coordinates and the stored sdf.

| # | candidate | test that tells it apart | cost |
|---|---|---|---|
| 1 | The far-field level is set by the inlet condition and decays along the domain. ISLA sees no domain boundary and only body-relative geometry, so far points at equal distance from the car but different distance from the inlet look alike. One case's inlet value is a constant a bias could absorb; a decay with x could not. | From the Part B dumps: far-field true nu_t against x and against distance from the car, and ISLA's far-field residual against x. If nu_t varies mainly with x at fixed car distance and ISLA's error tracks x, this is it. | data only, minutes on CPU |
| 2 | ISLA's distance channels saturate far from the body: the built-in signed-distance encoding loses resolution beyond about a body length, so beyond 1 m every point looks alike. | Read the encoding's sensitivity d(features)/d(distance) against distance (code plus a forward on synthetic far points along rays at 0.5-15 m), and whether ISLA's prediction goes flat beyond some distance while the target does not. | code reading plus about 0.5 GPU-h of inference |
| 3 | Routing: far points get little slice mass (45 % of the points lie within 1e-3 m of the wall), so the far field shares slices dominated by near-wall tokens. | Hooks on the final model: each point's top slice weight and the slices' far-point mass, against Transolver's slices on the same points. | about 1 GPU-h of inference plus a script |
| 4 | Slice resolution against boundary-layer mixing (from af, see Part A): at 512 slices the slices may mix layers that need separate representation; on af, s128 lowered the near-wall error 1.4x. | The 15.10 s128 rung on drv answers it for nu_t's far field and the velocity multiple; the af s128 second seed is in 15.11. | already queued (15.10, 15.11) |
| 5 | Sampling weight: about 1,300 of 200,000 points per draw lie beyond 1 m. Transolver gets the same draw and loss, so weight alone does not explain the gap, but ISLA's interacting queries may pool those few points into near-wall context. | Train ISLA with the far field oversampled (say 10 %, reweighted to the same quadrature). | about 18 GPU-h, one run |

### Part A: the ladder on AirfRANS (interim; the registered verdicts are read when d24 s45 and the 40k checks finish)

- **20k floors**, af pressure F_ho, geometric mean of seeds 44 and 45 where both ran:

| rung | ref | w256 | w1024 | d6 | d24 | s128 | s2048 |
|---|---|---|---|---|---|---|---|
| F_ho | 0.00198 | 0.00209 | 0.00193 | 0.00209 | (s45 running) | 0.00140 | 0.00201 |

  Phase 1b's eager reference stands beside them at 0.00165 / 0.00168. That gap is the question of 15.11.
- **Path check** (15.7.3, job 334500; `runs_p2/*/pathcheck.json`). On all 10 finished rungs, compiled and eager agree:
  relative prediction differences of 0.5-1.5e-6 on pressure, up to 1.1e-5 on velocity and up to 2.6e-4 on nu_t, and
  the floors agree to 4 digits.
- **Residual maps by rung** (`tables/resid_af_ladder.md`). In the 1e-3 to 0.1 chord bands, ISLA's pressure e_R is
  0.0024-0.0031 on every rung except s128 (0.0017-0.0018), against GeoTransolver's 0.0005-0.0007. Width and depth do
  not move the near-wall band at 20k.
