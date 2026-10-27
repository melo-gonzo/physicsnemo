# SLICES-ALL results (arms read as they land)

Pre-registration PREREG.md (md5 e0221626, written before any job). Jobs: JOBS.md. Reading: `read_slicesall.py` -> `reading_<arm>.md` /
`.json`. CM = the scorer's case-mean relative L2, float32, epoch 500, the control's scored set.

## DR: the interacting surface form on DrivAerML Surface (vehicle only, 0.9.0), against r090_dr_isla (read 2026-10-24 16:12 EDT)

Registered reading (R1, native decides): **no change on both fields, no flag. The arm is clean.** (`reading_DR.md`, md5 be4d5062;
`read_slicesall.py` md5 afde8721.)

| protocol | field | control (s42, s43) | arm, 128 slices (s42, s43) | arm ÷ control | D | s_c, s_a | 2 SE | seed-matched | label | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| **native** | pressure | 0.0411344 (0.0411605, 0.0411084) | 0.041383 (0.041297, 0.0414691) | **1.006** | +0.0060 | 0.0009, 0.0029 | 0.0043 | 1.003, 1.009 | **no change** | none |
| **native** | wall shear | 0.0662212 (0.0661942, 0.0662482) | 0.0665341 (0.0664878, 0.0665805) | **1.005** | +0.0047 | 0.0006, 0.0010 | 0.0016 | 1.004, 1.005 | **no change** | none |
| posed | pressure | 0.0411353 | 0.0413843 | 1.006 | +0.0060 | | 0.0044 | 1.003, 1.009 | no change | none |
| posed | wall shear | 0.066225 | 0.0665392 | 1.005 | +0.0047 | | 0.0016 | 1.004, 1.005 | no change | none |
| area | pressure | 0.0413107 | 0.0415257 | 1.005 | +0.0052 | 0.0029, 0.0031 | 0.0060 | 1.005, 1.005 | no change | none |
| area | wall shear | 0.0671495 | 0.0674296 | 1.004 | +0.0042 | 0.0029, 0.0033 | 0.0062 | 1.004, 1.004 | no change | none |

- 128 slices costs the surface form 0.5-0.6 % on both fields. Both arm seeds lie above both control seeds on both fields, so
  the shift is seed-separated, but it is a fifth of the 3 % flag threshold and a sixteenth of the ln 1.10 bar. D exceeds 2 SE on
  both fields, because the control's seeds agree to 0.13 % and 0.08 %, but D is far below the bar, so the label is "no change".
- Beside, per case (seed means): pressure median arm ÷ control 1.022, the arm better on 42 % of the 48 cases; wall shear 1.014,
  50 %.
- Posed: on DrivAerML Surface the pose changes the input (a rotated and translated vehicle), and both arms score posed equal to
  native to 1e-4. The labels match native; these are the same four runs, so posed is not an independent number. Area is
  non-degenerate here: area ÷ native is 1.002-1.016 for both arms. It reads the same labels.
- Eager guard: the arm's compiled and eager native scores differ by at most 2.8e-6 relative per case (bar 1e-3), so the read uses
  the compiled scores, as the control's.
- Checks passed: config row epoch 500 and 48 infer_step rows with the control's sample ids, every protocol; scorer START line
  code=code_r090, `model.n_slices=128` (arm) / 512 (control); dataset per protocol as the control's; the final resolved configs differ
  from the control's only in `run_id` and `n_slices` (both seeds); geo_kernel 'fused' slices=128 on 4 of 4 ranks.
- Slots: 2 arms x 2 seeds = 4 runs; 2 fields and 3 protocols are readings of them.

Cost (R3): smoke on one node, median step 0.160 s against 0.327 s (**0.49**); 0.47 after the first 5 steps), peak allocated 53.4
against 64.5 GiB (0.83). Over the whole run, the trainer's median step was 0.154 / 0.151 s against the control's 0.241 / 0.245 s (0.63),
and peak `mem_gb` was 55.0 against 66.8 (0.82). Training took 10.9 + 10.9 GPU-h against the control's 16.2 + 16.5 (0.67).
Worst validation loss over the running minimum after epoch 10 was 1.14 / 1.16 (guard 2).

## AF: ISLA-DTT on AirfRANS, eager-trained, against r090eag_af_dtt (PREREG §9 A1; read 2026-10-25 15:30 EDT)

Registered reading (R1, native decides): **no change on all three fields, no flag. The arm is clean.** (`reading_AF.md`, md5 5a8d1785;
`AF_CTL=r090eag_af_dtt python3 read_slicesall.py AF`.)

| protocol | field | control (s42, s43) | arm, 128 slices (s42, s43) | arm ÷ control | D | s_c, s_a | 2 SE | seed-matched | label | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| **native** | pressure | 0.0124084 (0.012428, 0.0123887) | 0.012761 (0.0138019, 0.0117200) | **1.028** | +0.0280 | 0.0022, 0.1156 | 0.1635 | 1.111, 0.946 | **no change** | none |
| **native** | velocity (in-plane) | 0.00717297 (0.00737341, 0.00697252) | 0.007242 (0.0080583, 0.0064257) | **1.010** | +0.0096 | 0.0395, 0.1601 | 0.2332 | 1.093, 0.922 | **no change** | none |
| **native** | nu_t | 0.0205977 (0.0207112, 0.0204842) | 0.0210885 (0.0209810, 0.0211959) | **1.024** | +0.0235 | 0.0078, 0.0072 | 0.0150 | 1.013, 1.035 | **no change** | none |
| posed | pressure / velocity / nu_t | | | 1.028 / 1.010 / 1.024 | | | 0.163 / 0.233 / 0.015 | | no change (all) | none |

- **Resolution is weaker than the control promised.** The control's seeds agree, but the arm's do not: on pressure and velocity
  its seed 42 is the worst of all four runs (11 % and 9 % above the control's seed 42), and its seed 43 is the best (5 % and 8 %
  below). Its sd of ln is 0.116 / 0.160 against the control's 0.002 / 0.040, so 2 SE is 0.16 / 0.23 and a harm label on those fields
  would have needed an arm ÷ control above about 1.18 / 1.26. On nu_t both pairs agree (0.008 / 0.007): the ln 1.10 bar binds,
  and the read resolves.
- **nu_t: a seed-separated 2.4 % shift, under the flag.** Both arm seeds (0.02098, 0.02120) lie above both control seeds (0.02071,
  0.02048), and D = 0.0235 exceeds 2 SE = 0.0150. But the ratio 1.024 is below the flag's 1.03 and the ln 1.10 bar, so the label is
  "no change" with no flag, as registered.
- Beside, per case (seed means): pressure median arm ÷ control 0.925, the arm better on 59 % of the 200 cases; velocity 1.014, 38 %;
  nu_t 0.922, 60 %. The arm's case medians sit below the control's on pressure and nu_t, while its case means sit above: the arm's
  excess is in a minority of hard cases.
- Beside: the arm's final in-training validation loss is 0.000248 / 0.000262 against the control's 0.000197 / 0.000215, about 1.23
  times. That loss is the training loss form at training resolution in bf16, not a registered metric.
- Posed changes the input (a pose of the airfoil and its domain) and reads like native: posed ÷ native is 1.0013-1.0022 on
  pressure for both arms. These are the same four runs.
- Area: the same points as native, so the protocol is **degenerate** and supports no claim. Area equals native to roundoff, not
  bitwise: per case within 3.6e-6, case means within 2.4e-8, for both arms. The registered check asked for exact equality; run-to-run
  float roundoff of the eager scorer explains the difference.
- Checks passed: config row epoch 500 and 200 infer_step rows with the control's sample ids, every protocol; scorer START line
  code=code_cmpaf_fix and `model.n_slices=128` / 512; dataset per protocol as the control's; the final resolved configs differ only in
  `run_id` and `n_slices`; geo_kernel 'eager' slices=128 on every rank of every link (16 of 16 lines per lane).

Cost (R3): smoke median step 0.510 s against 0.959 s (**0.53**), peak allocated 38.0 against 47.3 GiB (0.80). Over the whole run the
trainer's median step was 0.498 / 0.492 s against 0.931 / 0.937 s (0.53), and peak `mem_gb` 39.3 against 50.1 (0.78). Training took
58.2 + 57.6 GPU-h against the control's 110.8 + 110.0 (0.52). Worst validation loss over the running minimum after epoch 10 was
1.99 / 1.60 (guard 4).

## LUG: ISLA-DTT on the GINOT lug (solid mechanics), compiled on the fixed path, against r090fix_lug_dtt (read 2026-10-26 10:52 EDT)

Registered reading (R1, native decides): **no change, no flag. The arm is clean.** (`reading_LUG.md`, md5 1879e2c0.)

| protocol | field | control (s42, s43) | arm, 128 slices (s42, s43) | arm ÷ control | D | s_c, s_a | 2 SE | seed-matched (beside) | label | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| **native** | von Mises | 0.00202682 (0.00221933, 0.00183431) | 0.0019582 (0.00206216, 0.00185425) | **0.966** | -0.0344 | 0.1347, 0.0751 | 0.2182 | 0.929, 1.011 | **no change** | none |
| posed | von Mises | 0.00350931 (0.00339135, 0.00362727) | 0.00340146 (0.00374039, 0.00306253) | 0.969 | -0.0312 | 0.0476, 0.1414 | 0.2110 | 1.103, 0.844 | no change | none |

- On the native mesh the arm lies inside the control's seed range: seed 42 is 7 % below the control's seed 42, and seed 43 is 1 %
  above the control's seed 43. The arm's seeds sit 11 % apart, the control's 21 %. As registered, the read resolves only large
  effects: 2 SE = 0.22, so a harm label needed a ratio above about 1.24. The seed-matched ratios (0.929, 1.011) are beside and
  change no label. The arms do not share an initialization, since `n_slices` changes the parameter shapes.
- Beside, per case (seed means): median arm ÷ control 0.952, the arm better on 56 % of the 600 test lugs.
- Posed changes the input (an SO(3) pose plus a translation; no centering in this pipeline). It reads 0.969 with no change, but it
  is no robustness claim: 0.9.0's ISLA-DTT forest is pose-dependent (DTT-POSE), and posed ÷ native is 1.81 / 1.65 for the arm
  against 1.53 / 1.98 for the control. P14 held.
- Area: the same nodes as native, so the protocol is **degenerate** and supports no claim. Area equals native to roundoff: per case
  within 4.5e-6, case means within 8e-8, for both arms.
- Checks passed: config row epoch 500 and 600 infer_step rows with the control's sample ids, every protocol; scorer START line
  code=code_cmpaf_fix, `model.n_slices=128` / 512, eager rows as the control's; dataset per protocol as the control's; the final resolved
  configs differ only in `run_id` and `n_slices`; geo_kernel 'eager' slices=128 on every rank of every link.

Cost (R3): smoke median step 0.418 s against 0.403 s (**1.04**), peak allocated 7.6-8.3 against 9.2-10.1 GiB (0.82). Over the whole
run the trainer's median step was 0.405 / 0.396 s against the control's 0.425 / 0.420 s (**0.95**), and peak `mem_gb` 8.8 against 10.9
(0.81). Training took 145.3 + 141.3 GPU-h against the control's 171.3 + 153.0 (with its lost links; about 0.88). On the lug, ISLA-DTT's
step is dominated by work that does not scale with the slice count, so 128 slices saves about 5 %, not the 30-60 % it saves
elsewhere. Worst validation loss over the running minimum after epoch 10 was 2.12 / 2.00 (guard 4; the control's 1.43 / 2.00).

## The registered verdict (PREREG §6): **candidate for every form**

Every field of every arm reads **no change**, and no up flag fires: DR 1.006 / 1.005, AF 1.028 / 1.010 / 1.024, LUG 0.966
(`AF_CTL=r090eag_af_dtt python3 read_slicesall.py VERDICT`). By the rule, 128 slices is a scope-freeze candidate for interacting ISLA
(AirfRANS and DrivAerML Volume, earlier lanes), ISLA-DTT (AirfRANS and the lug) and the interacting surface form (DrivAerML Surface,
vehicle only). The adoption is the supervisor's decision at the freeze, reported to Peter.

What weighs on that decision:
- **Seed-separated shifts under the flag.** DR moves +0.5-0.6 % on both fields and AF nu_t +2.4 %, each with both arm seeds above
  both control seeds. All are below the registered 3 % flag, so they keep no 512, but they are measured costs, not zero.
- **Resolution.** AF pressure and velocity (2 SE 0.16 / 0.23, from the arm's own 12-17 % seed spread) and the lug (2 SE 0.22) could
  label only large harms. Only DR and AF nu_t resolve near the ln 1.10 bar.
- **The saving is form-dependent.** The step ratio is 0.53 for AirfRANS ISLA-DTT (eager) and 0.63 whole-run for DrivAerML Surface
  (0.49 in the smoke), but 0.95 for the lug's ISLA-DTT. FREEZE-PLAN's scenario S4 assumed 0.85-0.95 for the lug, which holds;
  its other factors can now use these measured ones.
- **Code.** Every arm ran 0.9.0 code (code_r090, code_cmpaf_fix), not the rerun's code. The surface arm read the vehicle only, not
  option A's whole boundary.
- Slots: 3 comparisons x 2 arms x 2 seeds = 12 training runs (6 new). DR has 2 fields, AF 3 and LUG 1; posed and area are readings
  of the same runs.

## Predictions (PREREG §7)

| id | prediction | p | outcome |
|---|---|---|---|
| P1 | DR and LUG smokes pass first time | 0.80 | held (AF's also passed) |
| P2 | DR no change on both fields | 0.93 | held |
| P3 | DR up flag | 0.20 | did not fire (seed-separated, 0.5 %) |
| P4 | LUG no change | 0.92 | held (0.966) |
| P5 | LUG up flag | 0.15 | did not fire |
| P6 | AF no change on all three | 0.85 | held |
| P7 | AF up flag | 0.20 | did not fire (nu_t 1.024, seed-separated) |
| P8 | verdict: candidate for every form | 0.50 | held |
| P9 | DR point estimates in [0.97, 1.03] | 0.70 | held (1.006 / 1.005) |
| P10 | DR smoke step ratio in [0.50, 0.75] | 0.60 | failed narrowly (0.49; whole run 0.63) |
| P11 | LUG smoke step ratio in [0.80, 0.97] | 0.60 | failed (1.04; whole run 0.95) |
| P12 | AF smoke step ratio in [0.40, 0.70] | 0.60 | held (0.53) |
| P13 | no event-guard stop | 0.85 | held |
| P14 | LUG arm posed ÷ native in [1.3, 2.2] | 0.80 | held (1.81 / 1.65) |

## GPU-h (sacct elapsed x 4; `sacct_all.txt`)

| item | GPU-h |
|---|---|
| smokes (369628, 369629, 377354) | 2.93 |
| DR training (2 lanes) | 21.81 |
| LUG training (2 lanes) | 286.56 |
| AF training (2 lanes) | 115.81 |
| scorers (371108, 371109, 388478, 404080, 404692) | 1.98 |
| **total** | **429.1** (cap 500 after A1; under the original 450) |
