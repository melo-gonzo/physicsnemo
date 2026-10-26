# FLOOR-DRV results (ISLA on DrivAerML Volume with the domain's whole boundary; first written 2026-10-24 about 07:34 EDT, updated 15:01 EDT; read and filled 2026-10-26 about 05:33 EDT)

**Status: DONE.** All four runs of job 355565 completed (rc=0, `.done`): C at 01:52 EDT 10-25, F at 05:23 EDT 10-26,
after seven requeues (Deviations). Read on 10-26 by the reader lane with `HANDOFF_FLOOR.md`'s commands: residual job
399764, `reduce_fd.py`, `resid_tables.py`; outputs in `tables/` (reduce_fd.md eea5db7c, resid_fd.md 560eba98,
residuals_fd.csv 6eaf5138).

**Readings (registered, PREREG section 4): nu_t far-field e_R "closes"; velocity F_ho "helps".**

- Pre-registration: `PREREG.md` (approved 12:18 EDT 10-23, md5 507582d0; amendment A1, section 7, md5 e24b2891;
  amendment A2, section 8, md5 d04462aa). Jobs, md5s and events: `JOBS.md`.
- **Adopted by Peter's ruling of 2026-10-23; FLOOR-DRV measures its effect.** Peter, 17:08 EDT 10-23 (LANE_RULES.md,
  "Peter's rulings on canonical ISLA"): "not only CAN it read the whole boundary - it SHOULD read the whole boundary!
  ... Seeing where the floor is through raw coordinates is a bug, not a feature. Even if there were an accuracy *cost*
  to reading the whole boundary (not that I expect it), I would recommend this. I don't care that they make each step
  1.58x as long." The registered readings, written while adoption was open, stand as written.

## Question and design

On one DrivAerML Volume car (run_171), trained with a fresh draw of its points every step (CAP-AUDIT's Part B
protocol: 20,000 steps, eager, float32, 200,000 interior points, corrected held-out set), does ISLA fit the far-field
eddy viscosity and the velocity as well as Transolver and GeoTransolver once it reads the domain's four stored boundary
meshes besides the vehicle?

| arm | config | boundary meshes | seeds | runs |
|---|---|---|---|---|
| F | isla_volume_floor (psharpe/isla-floor-drv 564f803cc) | vehicle, no_slip, slip, inlet, outlet, each `{}` | 42, 43 | cap_drv_isla_fdF_s4x |
| C | isla_volume (Part B's ISLA row rerun in the same job) | vehicle | 42, 43 | cap_drv_isla_fdC_s4x |
| comparators | CAP Part B: Transolver 512 x 16, GeoTransolver 256, ISLA; 15.10 ISLA reference s44 | | | read, not rerun |

The baselines read raw coordinates in the dataset's frame, so a point's z carries its height above the ground
implicitly; ISLA is pose-invariant and knows where the ground is only if it is given the ground's mesh.

## Checks (all passed)

- **Data** (raw files, all 484 cases; `harness/walls_all_cases.py`, `$T/floordrv/walls_all_cases.json`): the four
  domain meshes exist and are non-empty on every case, every face lies on the domain box (x -40 to 80 m, y -22 to
  22 m, 20 m high), and every face points into the fluid. None carries cell data, so each enters as a condition with
  no fields (`{}`), by its file name, as on AeroZoo.
- **Frame** (job 355437, `$T/floordrv/frame_check_355437.json`): on the draws of epochs 0, 1, 1000, 1001 and 1002, all
  five meshes map from their stored rows by one scale (1 / L_ref, L_ref = 5 m; equal within 3.4e-8) and one
  translation (equal within 0.004 mm), the interior mean is 0, so the ground and walls are centered with the car in
  each draw's own frame, and the held-out set uses epoch 1's draw whole. 87-214 interior points per draw lie at a
  union distance of -4e-8 m: CFD nodes on the ground, float32 rounding.
- **What the walls change inside ISLA** (`harness/domain_geom.py`, frame check): the length unit L rises from 2.04 m
  (vehicle) to 59.5 m (all five), so every s = d / L is 29 times smaller; the vehicle holds 0.20 % of the boundary
  measure; the domain meshes (5,054 faces) are kept whole and the vehicle gets 194,946 of the 200,000 tokens; 18 % of
  the interior points are nearer a domain wall than the car, and so are 99.9-100 % of the points more than 5 m from the
  car.
- **Acceptance** (07:31 EDT 10-24): all four rows log finite losses; F's setup lists the five meshes, C's the vehicle;
  point hashes f26c3708e117 / 8f4c7299c04c as Part B; distinct draws; peak memory 234.1 GiB each.
- run_171 is one of the 187 files whose ground already meets the body (DRV-GROUND), so the recipe's later
  PlaceGroundUnderBody correction does not change this lane's input.

## Cost finding (final)

**The four domain meshes make ISLA 0.9.0's DrivAerML Volume step 1.58 to 1.61 times as long**: 5.27 s against 3.33 s
in the smoke (355126), 5.29 s against 3.29 s in the full job's first logs; peak memory is unchanged (234.4 GiB).
Over the full runs the ratio was larger, **1.74-1.76x** (median step F 5.25 s, C 3.01 s, on the same nodes; reduce_fd.md
Checks and log.jsonl): C's step settled at 3.00 s after its first 1,000 steps, while F's stayed at 5.24-5.30 s. The
1.58x of the stop rule (A2) was read from 200-step medians, before C settled. By BOUNDARY-AUDIT
and GEO-FAST (book notebook), almost all of the added time is the Warp query that seeds the closest-cell search, and
ISLA's own cell-tree seed removes most of it. Peter's ruling stands whatever the cost; this is the cost of the 0.9.0
geometry path that the standings trained with.

## Registered readings (PREREG section 4), from `tables/reduce_fd.md`

| reading | rule | F s42 / s43 | C s42 / s43 | Transolver s42 / s43 | label |
|---|---|---|---|---|---|
| nu_t far-field e_R (beyond 5 m) | closes <= 1.5 T; helps <= C / 1.5 and outside REF; hurts >= 1.5 C and outside REF | 0.01304 / 0.01192 | 0.03103 / 0.02807 | 0.00974 / 0.01065 | **closes** (F/T 1.339 / 1.119; F/C 0.420 / 0.425) |
| velocity F_ho | closes <= 1.5 T; helps <= C / 1.2 and outside REF; hurts >= 1.2 C and outside REF | 0.01532 / 0.01488 | 0.01907 / 0.02067 | 0.00878 / 0.00864 | **helps** (F/T 1.744 / 1.723; F/C 0.803 / 0.720; max F 0.01532 < min REF 0.01907) |

As `reduce_fd.py` prints them:

```
- **nu_t far-field e_R: closes**  F 0.01304, 0.01192; C 0.03103, 0.02807; T 0.009741, 0.01065; REF 0.04101, 0.03452, 0.03149, 0.03103, 0.02807; F/T 1.339, 1.119; F/C 0.4203, 0.4246
- **velocity F_ho: helps**  F 0.01532, 0.01488; C 0.01907, 0.02067; T 0.008783, 0.008639; REF 0.01918, 0.02001, 0.01921, 0.01907, 0.02067; F/T 1.744, 1.723; F/C 0.803, 0.7201
```

REF (both readings): Part B ISLA s42 / s43, 15.10 reference s44, C s42 / s43. Far-field e_R of the three existing
references: 0.0410, 0.0345, 0.0315; velocity F_ho 0.0192, 0.0200, 0.0192. Checks block: complete True for all four F
and C rows (20,000 of 20,000), F's boundaries inlet, no_slip, outlet, slip, vehicle, C's the vehicle, point hashes as
Part B, heldout_whole True, finite; residual `e` = logged `e` within 4e-5 relative on every row.

What the labels mean (PREREG section 4, descriptive): "closes" on nu_t says the missing ground and walls were a cause
of ISLA's far-field eddy-viscosity miss on this car: given them, F's error beyond 5 m from the car falls 2.4x below C's
and lands within 1.12-1.34x of Transolver's (GeoTransolver 0.0115 / 0.0114, so F is 1.13 / 1.05x GeoTransolver).
"helps" on velocity says the uniform velocity factor is in part a missing-input effect: F's floor is 20-28 % below C's
but still 1.72-1.74x Transolver's, so most of the factor remains and has another cause (15.10: depth).

## Readouts beside the readings (descriptive; no rule reads them)

**Held-out floors** F_ho (median of steps 19,600-20,000; R = arm / best of Transolver, GeoTransolver; per seed s42 / s43):

| field | F | C | Part B ISLA | Transolver | GeoTransolver | R_F | R_C | F / C |
|---|---|---|---|---|---|---|---|---|
| velocity | 0.01532 / 0.01488 | 0.01907 / 0.02067 | 0.01918 / 0.02001 | 0.00878 / 0.00864 | 0.01011 / 0.00968 | 1.74 / 1.72 | 2.17 / 2.39 | 0.80 / 0.72 |
| pressure | 0.00924 / 0.00793 | 0.00975 / 0.01165 | 0.01011 / 0.01149 | 0.00589 / 0.00576 | 0.00649 / 0.00653 | 1.57 / 1.38 | 1.66 / 2.02 | 0.95 / 0.68 |
| nu_t | 0.01436 / 0.01324 | 0.02077 / 0.02038 | 0.02389 / 0.02194 | 0.00978 / 0.00996 | 0.01085 / 0.01057 | 1.47 / 1.33 | 2.12 / 2.05 | 0.69 / 0.65 |

15.10 reference s44: velocity 0.01921, pressure 0.01024, nu_t 0.02073. F's floor is below C's on every field and seed.

**Far field (beyond 5 m from the car), held-out e_R** (s42 / s43): velocity F 0.00739 / 0.00642, C 0.01119 / 0.01028,
Transolver 0.00415 / 0.00446 (F / C 0.66 / 0.62); pressure F 0.218 / 0.180, C 0.239 / 0.276, Transolver 0.133 /
0.143. **Share of nu_t e^2 beyond 5 m** (0.7 % of the points; resid_fd.md, mean of the two seeds): F 16.3 %, C 41.1 %.

**e_R per wall-distance band, held-out** (tables/reduce_fd.md; bands in metres: within 0.5 mm, 0.5-5 mm, 5-50 mm,
5-50 cm, 0.5-5 m, beyond 5 m): **F's e_R is below C's in every band, on every field and both seeds (36 of 36)**, so
the 29 times smaller s costs nothing near the wall here. F / Transolver by band, nearest first:

| field | F s42 | F s43 | C s42 | C s43 |
|---|---|---|---|---|
| velocity | 1.52, 1.77, 1.83, 1.60, 1.58, 1.78 | 1.38, 1.73, 1.91, 1.60, 1.53, 1.44 | 1.70, 2.17, 2.43, 2.21, 2.03, 2.70 | 2.11, 2.39, 2.56, 2.26, 2.30, 2.30 |
| pressure | 1.31, 1.75, 1.80, 1.67, 1.59, 1.64 | 1.15, 1.53, 1.59, 1.52, 1.51, 1.26 | 1.44, 1.80, 1.88, 1.85, 1.66, 1.79 | 1.77, 2.26, 2.10, 2.07, 1.96, 1.93 |
| nu_t | 1.22, 1.39, 1.58, 1.55, 1.46, 1.34 | 1.18, 1.30, 1.45, 1.41, 1.38, 1.12 | 1.27, 1.55, 1.82, 2.00, 2.05, 3.19 | 1.44, 1.65, 1.84, 2.03, 2.20, 2.64 |

The gain is not confined to the far field: velocity improves in every band (F / C 0.63-0.90),
and nu_t most beyond 5 m (0.42) and least within 0.5 mm (0.82-0.96).

**C against Part B ISLA** (run-to-run spread, same code and protocol): velocity 0.995 / 1.033, pressure 0.964 / 1.014,
nu_t 0.870 / 0.929; far-field nu_t e_R 0.757 / 0.813 (C 0.0310 / 0.0281 against 0.0410 / 0.0345).

**Epoch-0 whole-draw floors** (the brackets in reduce_fd.md) agree with the held-out ones in direction; C's nu_t is
higher there (0.0343 / 0.0349, far field 0.064 / 0.065) than on the held-out draw, F's is not (0.0153 / 0.0125, far
field 0.018 / 0.008).

**Predictions** (PREREG section 5, written before any run):

| id | prediction | p | outcome |
|---|---|---|---|
| W1 | nu_t far field reads "closes" | 0.15 | **true** |
| W2 | nu_t far field reads "helps" | 0.30 | false |
| W3 | nu_t far field reads "hurts" | 0.10 | false |
| W4 | nu_t far field reads "no effect" | 0.45 | false |
| W5 | velocity reads "closes" | 0.03 | false |
| W6 | velocity reads "helps" | 0.12 | **true** |
| W7 | velocity reads "hurts" | 0.20 | false |
| W8 | velocity reads "no effect" | 0.65 | false |
| W9 | C / Part B ISLA velocity F_ho within [0.9, 1.1] on both seeds | 0.8 | **true** (0.995 / 1.033) |
| W10 | F's held-out nu_t F_ho below C's on both seeds | 0.55 | **true** (0.0144 / 0.0132 against 0.0208 / 0.0204) |
| W11 | F's far-field velocity e_R below C's on both seeds | 0.5 | **true** (0.0074 / 0.0064 against 0.0112 / 0.0103) |

Both labels came out on the side I gave low probability (W1 0.15, W6 0.12): I underrated the missing boundary.

**How sure.** One car, two seeds per arm. The effect is large against the noise: far-field nu_t F / C is 0.42 on both
seeds while the five ISLA references span 1.46x (0.0281-0.0410), and both F seeds lie below every reference. On
velocity the "helps" margin is thin on s42 (F / C 0.803 against the bar 0.833). Whether it holds across cars is not
tested here.

## Deviations from the registered protocol (disclosed; no registered rule changes)

- **A2 (before any full-length job):** the smoke's step-time stop rule tripped (F 1.58x C, bar 1.3x); the supervisor
  approved running as registered at 20,000 steps, with the rows' wall guard raised from 1,140 to 1,860 min and the cap
  raised to 125 GPU-h (PREREG section 8).
- **Preemption and a non-bitwise resume (11:34:35 EDT 10-24).** Job 355565 was preempted after 4:29:41 on
  nvl72d046-T09, requeued by the cluster and restarted at 11:35:51 EDT on nvl72d056-T12. Every row resumed from its
  last ckpt.pt: **F s42 and s43 resumed from step 2,000 after reaching 2,900** (900 steps retrained); **C s42 and s43
  from step 5,000 after reaching 5,100**. ISLA on DrivAerML Volume does not resume bit-for-bit (CAP-AUDIT), so steps
  after the resume differ from what an uninterrupted run would give, for F and C alike; the readings use the final
  floors (steps 19,600-20,000) as registered. Part B's ISLA runs also resumed once (at 10,000), so the comparators carry
  the same kind of break. The supervisor approved letting the job run with the **cap raised to 140 GPU-h** (11:59 EDT
  10-24); the projected total is about 131 GPU-h (15:01 EDT 10-24).
- **Six further requeues (final record, JOBS.md "355565: segments, resumes and cost").** The job ran in eight
  segments; segments 1-7 each ended "CANCELLED ... DUE TO PREEMPTION" in the Slurm log (sacct PREEMPTED), segment 8
  (11 s) only skipped the finished rows; `scontrol` shows Restarts=7. Every resume came from ckpt.pt and is not
  bit-for-bit, for F and C alike. Resume steps (last logged step -> resumed step):
  - **F s42 and s43** (identical): 2,900 -> 2,000 (11:35 EDT 10-24); 5,200 -> 5,000 (16:48 EDT 10-24); 9,800 -> 9,000
    (00:04 EDT 10-25); 13,900 -> 13,000 (07:33 EDT 10-25); 16,300 -> 16,000 (12:34 EDT 10-25); 18,900 -> 18,000
    (02:19 EDT 10-26). Six resumes, 4,000 steps retrained; done 05:23 EDT 10-26.
  - **C s42 and s43** (identical): 5,100 -> 5,000 (11:35 EDT 10-24); 10,700 -> 10,000 (16:48 EDT 10-24); 18,400 ->
    18,000 (00:04 EDT 10-25). Three resumes, 1,200 steps retrained; done 01:52 EDT 10-25.
  - Both arms last resumed at step 18,000, so the floors read (steps 19,600-20,000) and the final dumps lie inside one
    segment for every row. F carries twice as many breaks as C; the break is not a protocol change, but it is a
    difference between the arms. The last requeue came at the second the batch script ended (05:23:45 EDT 10-26) and
    cost 11 s.
  - The supervisor raised the cap to **150 GPU-h** (14:15 EDT 10-25). **Final cost 148.2 GPU-h** (Cost).

## What was seen before any floor

All of CAP-AUDIT (PREREG section 0); the smoke's 200-step losses; and, in a dry run of `reduce_fd.py` at 07:33 EDT
10-24, F's and C's held-out values at steps 200 and 400 (0.60 to 0.85 on every field: early training, not floors). No
rule was changed after the PREREG's amendments.

## Cost

Smoke 355126 and frame check 355437: 1.55 GPU-h. Job 355565, by `sacct -j 355565 -D -X -o
JobID,State,Start,End,Elapsed,NodeList`: eight segments, 131,969 s = 36.66 node-h x 4 GPUs = **146.6 GPU-h**. **Total
148.2 GPU-h**, under the final cap of 150 GPU-h (registered 80; A2 125; supervisor 140 at 11:59 EDT 10-24 and 150 at
14:15 EDT 10-25). Against the 120 GPU-h planned for one uninterrupted segment, the requeues cost about 25 GPU-h (F's
4,000 retrained steps, about 6 node-h, and seven start-ups). The residual job 399764 ran on CPU (cpu-short, about 1 min).

## Data kept

`tables/` here: reduce_fd.md, resid_fd.md, residuals_fd.csv (md5 above), and the smoke rows' logs (`tables/smoke/`).
On LHR, `$T/floordrv/runs/cap_drv_isla_fd[FC]_s4[23]/`: final_model.pt (for probes), log.jsonl, stdout.log,
resolved_config.yaml, .exit, .done. The final_predictions.pt files and the smoke dirs were deleted after the residual
maps existed (JOBS.md, "Registry and cleanup"); the residual maps can be rebuilt only by rerunning the final dump from
final_model.pt on the held-out draw.
