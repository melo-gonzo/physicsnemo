# FLOOR-DRV pre-registration: ISLA on DrivAerML Volume with the ground and slip walls as named boundary meshes

Written 2026-10-23 12:15-12:40 EDT by the research lane FLOOR-DRV, before any job of this lane (smoke included).
Supervisor: ISLA Autoresearch (01a13c2c). Lead: CAP-AUDIT phase 2, `scratch/isla-cap-audit-2026-10-20/RESULTS.md`
(Part B, the far-field nu_t probes) and HANDOFF.md decision 2.1.

## 0. Disclosure: what I have seen

- **All of CAP-AUDIT**: RESULTS.md, HANDOFF.md, JOBS.md, PREREG.md sections 15.3-15.11, the residual tables
  (`tables/resid_drv_partB.md`, `residuals_p2.csv`, `residuals_p2b.csv`) and the harness. In particular Part B's floors
  (ISLA velocity F_ho 0.0192 / 0.0200, Transolver 0.0088 / 0.0086; nu_t 0.0239 / 0.0219 against 0.0098 / 0.0100), the
  far-field (beyond 1 m) held-out nu_t e_R (ISLA 0.0410 / 0.0345, 15.10 reference s44 0.0315; Transolver 0.0097 /
  0.0107; GeoTransolver 0.0115 / 0.0113), the per-decade velocity e_R, the 15.10 down-rung ladder and probes (a)-(c).
- **No number of any FLOOR-DRV arm exists.** Nothing of this lane has run on a GPU.
- **Data facts read before writing this** (no model involved; `harness/domain_geom.py`, `harness/walls_all_cases.py`, on
  LHR login node, raw files):
  - run_171 stores five boundaries: `vehicle` (18.8 M triangles), `no_slip` (722 triangles, 400 points: the ground
    z = -0.3176 m from x = -2.339 m to the outlet at x = 80 m, the full span y = +-22 m), `slip` (2,888 triangles,
    1,600 points: the two sides y = +-22 m, the roof z = 19.68 m, and the ground upstream of no_slip, x = -40 to
    -2.34 m), `inlet` and `outlet` (722 each). No boundary but the vehicle carries cell data.
  - On all 484 cases the no_slip and slip faces point into the fluid (0 misoriented meshes), and the domain box is the
    same on every case up to the ground's height (-0.318 or -0.320 m; 4 distinct boxes). The vehicle's stored cells are
    wound into the body (signed volume -5.58 m^3 on run_171), which the dataset file already reverses.
  - ISLA's length unit L (the measure-weighted RMS pairwise distance of the given boundaries, with each cell's second
    moment; the vehicle estimated from every 100th cell): **2.04 m with the vehicle alone, 55.5 m with vehicle + no_slip +
    slip** (38.3 m with no_slip only). The domain walls hold **99.8 %** of the boundary measure (vehicle 33.8 m^2 against
    15,360 m^2).

## 1. Question

On DrivAerML Volume (run_171, the CAP single-case protocol with a fresh draw every step), if ISLA also gets the domain's
ground (`no_slip`) and slip walls (`slip`) as named boundary meshes, does it fit the case's far-field eddy viscosity and
its velocity as well as Transolver and GeoTransolver do?

This is a diagnostic of a missing input, not a change of DrivAerML Volume's input contract (that needs Peter's word at
the scope freeze). The baselines read raw coordinates in the dataset frame, which carry the ground's position
implicitly (z of a point is its height above a fixed floor, up to CenterMesh's per-draw shift of about a millimetre);
ISLA is pose-invariant and cannot know where the ground is unless it is given the ground's mesh.

**Input ruling** (Peter; checked by the supervisor): boundary conditions enter ISLA as boundary meshes with their
condition types; the model takes the full-resolution boundary surfaces and only those; nothing precomputed is passed.
no_slip and slip are boundary surfaces of the PDE stored in the case files, passed whole by name; their condition
type is the file's own name, as on AeroZoo (isla_volume_aerozoo.yaml: `no_slip: {}`, `slip: {}`). Neither carries a
prescribed value in the files, so neither declares a field. The inlet and outlet are not added (the brief's design; on
drv they carry no prescribed values either, and the far-field level does not follow the distance from the inlet,
probe (a): R^2 0.08).

## 2. Arms (one 4-GPU node, one job)

All rows are CAP Part B's ISLA row except as stated: ISLA 0.9.0 code snapshot `code_r090` (92d9d7800 + td0152), eager
(compile=false, compile_blocks=false), float32, hidden 512, 12 layers, 512 slices, lr 3.75e-4 (Muon, cosine to 0 over
20,000 steps, weight decay 0), 200,000 interior points, run_171, 20,000 steps, a fresh datapipe draw every step
(epoch 1000 + k, `--draw_sync`, 1 draw worker), the corrected held-out set (`--heldout_whole`: epoch 1's draw whole,
its interior with its own boundaries and global data), harness `overfit_rdp.py` md5 14a34367 (CAP's, byte-identical).

| run | GPU | recipe snapshot | model config | seed |
|---|---|---|---|---|
| cap_drv_isla_fdF_s42 | 0 | recipe_r090_fd | isla_volume_floor | 42 |
| cap_drv_isla_fdF_s43 | 1 | recipe_r090_fd | isla_volume_floor | 43 |
| cap_drv_isla_fdC_s42 | 2 | recipe_r090 | isla_volume | 42 |
| cap_drv_isla_fdC_s43 | 3 | recipe_r090 | isla_volume | 43 |

- **Arm F** (`conf/model/isla_volume_floor.yaml`, branch psharpe/isla-floor-drv, commit cba4197f3): `defaults:
  [isla_volume, _self_]` plus `forward_kwargs.boundary_meshes.{no_slip: boundaries.no_slip, slip: boundaries.slip}`
  and `model.boundary_schemas.{no_slip: {}, slip: {}}`. The composed config differs from isla_volume in exactly those
  four keys (checked by a Hydra diff and by the new test `test_isla_volume_floor_adds_the_ground_and_slip_walls`; the
  recipe's ISLA config tests pass, 57, and test_utils / test_forward_kwargs / test_synthetic_configs, 158). No model
  code and no dataset file changes: drivaer_ml_volume_reference already keeps the in-file boundaries
  (`drop_in_file_boundaries: false`, only `stl_geometry` dropped) and applies CenterMesh's translation to every
  boundary (`CenterMesh.apply_to_domain`). The snapshot `recipe_r090_fd` is `recipe_r090` plus that one file (md5
  ffc47632; the other 135 files md5-identical; SNAPSHOT_HEAD line).
- What F changes inside the model, as ISLA's design dictates (no knob of this lane):
  - each prediction point's built-in distance and direction come from the union of vehicle, ground and slip walls, so
    a point nearer the ground than the car reads the ground;
  - the boundary tokens: 200,000 drawn by water filling; no_slip (722) and slip (2,888) are kept whole, the vehicle
    gets 196,390 (Part B: 200,000);
  - **the length unit L rises from about 2.0 m to about 55 m**, so every s = d / L is about 27 times smaller and the
    channel asinh(s / 1e-6) turns logarithmic only beyond d of about 55 um instead of 2 um; and the walls hold 99.8 %
    of the routing measure. Both are consequences of ISLA's frame (L from all given boundaries), as on AeroZoo. They
    can hurt the near-wall fit, which is why the readings include "hurts".
- **Controls C**: Part B's ISLA row rerun at seeds 42 and 43 in the same job (ISLA on drv does not resume or rerun
  bit-for-bit), to measure the run-to-run spread on the same node and code. C is the comparator of the "helps" and
  "hurts" readings; Part B's two ISLA runs and the 15.10 reference s44 stand beside it.
- **Baselines**: Part B's Transolver 512 x 16 (T) and GeoTransolver 256 (GT) runs at seeds 42 and 43 (complete, read,
  their logs and residual rows on LHR and in `tables/residuals_p2.csv`). Not rerun.

### Frame and validity checks (smoke, before the full job; `harness/frame_check.py`)

On the datapipe's own draws of run_171 under arm F (epochs 0, 1, 1000, 1001, 1002):
- each boundary's processed points minus its raw stored rows is one constant translation (per-point deviation <= 0.02 mm,
  float32 at |x| <= 80 m), **the same for vehicle, no_slip and slip** (means within 0.01 mm), and the draw's interior
  mean is 0: the ground and walls are centered with the car in each draw's own frame (CAP PREREG 15.8);
- the held-out set of F uses epoch 1's draw whole, so all three meshes come from that draw (`--heldout_whole`; never
  another draw's boundary);
- L for vehicle alone and for all three (ISLA._gauge on the full-resolution meshes), the fraction of interior points
  whose nearest surface is a domain wall (all, and beyond 1 m from the car), and the fraction with a negative union
  distance (must be about 0: walls oriented into the fluid);
- smoke training rows: F and C, 200 steps each, short QoS; their setup records must show F's three boundary meshes and
  C's one, the point hashes of Part B (f26c3708e117 / 8f4c7299c04c), finite losses, step times and peak memory.
- **Stop rule**: if the frame check fails, or F's smoke step time exceeds 1.3 x C's, or peak memory exceeds 270 GiB,
  I stop and report before the full job.

## 3. Readouts (`harness/reduce_fd.py`, written before any run; residual maps by CAP's `residuals.py` dddb2f0e and
`resid_tables.py` 2468aff7, on CAP's drv labels, `$T/cap/labels/labels_drv.npz`)

Per seed (42, 43):
1. **Held-out floor** F_ho = the median of the last five held-out logs (steps 19,600-20,000) of velocity, pressure and
   nu_t for F and C, beside Part B's T, GT and ISLA; also the epoch-0 whole-draw floor in brackets. R = arm / min(T, GT).
2. **Far-field nu_t e_R**: residuals.py's region error on the final dump's held-out set, wall-distance band [1, inf) m
   (distance to the car, labels.py), for F, C, Part B's ISLA, T and GT (and the 15.10 reference s44).
3. **Velocity e_R / Transolver's e_R per wall-distance decade** (the six bands), for F and C.
4. Beside them: the far-field e_R of velocity and pressure; the share of nu_t e^2 beyond 1 m; step time and peak memory.

## 4. Registered readings

REF = the five ISLA reference values of the readout (Part B s42, s43; 15.10 ref s44; C s42, s43). Both seeds must agree
for any label other than "no effect".

**nu_t far field** (readout 2, held-out):
- **closes**: e_R(F_s) <= 1.5 x e_R(T_s) on both seeds;
- **helps**: not closes, e_R(F_s) <= e_R(C_s) / 1.5 on both seeds, and max_s e_R(F_s) < min over REF;
- **hurts**: e_R(F_s) >= 1.5 x e_R(C_s) on both seeds, and min_s e_R(F_s) > max over REF;
- **no effect**: otherwise.

**Velocity** (readout 1, held-out F_ho):
- **closes**: F_ho(F_s) <= 1.5 x F_ho(T_s) on both seeds;
- **helps**: not closes, F_ho(F_s) <= F_ho(C_s) / 1.2 on both seeds, and max_s F_ho(F_s) < min over REF;
- **hurts**: F_ho(F_s) >= 1.2 x F_ho(C_s) on both seeds, and min_s F_ho(F_s) > max over REF;
- **no effect**: otherwise.

Why these bars, against the supervisor's example (1.5 against C for both fields): the far-field e_R of the three
existing ISLA references spans 1.30x (0.0315-0.0410), so 1.5 is just above its noise; ISLA's velocity F_ho spans 1.04x
over three seeds, so 1.2 is well clear of its noise while 1.5 against C (0.013) would be almost the "closes" bar
(0.013). The REF condition makes "helps" and "hurts" noise-aware: F must also leave the whole reference range. "hurts"
is added because F changes ISLA's length unit 27x (section 2), which can cost the near-wall fit.

If an F or C run does not complete, the reading is "incomplete as registered"; I report what the finished runs show and
escalate. Pressure (no Part B deficit) is reported beside them, with no reading. Interpretation (descriptive, for the
write-up): "closes" or "helps" on nu_t means the missing ground was a cause of the far-field miss; on velocity it means
the uniform 2x factor is in part a missing-input effect; "no effect" on velocity with "closes"/"helps" on nu_t means the
velocity factor has another cause (15.10: depth).

## 5. Predictions (mine)

| id | prediction | p |
|---|---|---|
| W1 | nu_t far field reads "closes" | 0.15 |
| W2 | nu_t far field reads "helps" | 0.30 |
| W3 | nu_t far field reads "hurts" | 0.10 |
| W4 | nu_t far field reads "no effect" | 0.45 |
| W5 | velocity reads "closes" | 0.03 |
| W6 | velocity reads "helps" | 0.12 |
| W7 | velocity reads "hurts" | 0.20 |
| W8 | velocity reads "no effect" | 0.65 |
| W9 | C reproduces Part B: C / Part B ISLA velocity F_ho within [0.9, 1.1] on both seeds | 0.8 |
| W10 | F's held-out total nu_t F_ho is below C's on both seeds | 0.55 |
| W11 | F's far-field velocity e_R is below C's on both seeds | 0.5 |

## 6. Cost, cap, launch

- **Smoke** (short QoS, one 4-GPU node, `-t 01:00:00`): `rows_fd_smoke.tsv` (F and C, 200 steps, GPUs 0-1) and
  frame_check.py on GPU 3; about 0.75 h x 4 GPUs = 3 GPU-h allocated.
- **Full job** (normal QoS, batch_long, `-t 20:00:00`): `rows_fd.tsv`, four runs at about 3.0 s/step (Part B: 2.99-3.00),
  about 17-18 h; `--max_minutes 1140` per run. About 72 GPU-h allocated (4 GPUs x 18 h).
- **Residual maps** (cpu-short or cpu-normal, 8 cores): minutes.
- **Total about 75 GPU-h; cap 80 GPU-h.** If a requeue or a node failure would push the total over 80 GPU-h, I stop and
  report instead of resubmitting (and I never cancel or resubmit myself; the supervisor decides).
- Commands, from `$T/floordrv/harness`, `T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0`:
  - smoke: `sbatch -q short -t 01:00:00 -J isla.floordrv.smoke --exclude=nvl72d015-T12,nvl72d058-T08,nvl72d052-T01 --array=0 --export=ALL,ROWS=$T/floordrv/harness/rows_fd_smoke.tsv,DRAW_SYNC=1,DRAW_WORKERS=1,HELDOUT_WHOLE=1,FRAME_CHECK=1 overfit_fd.sbatch`
  - full: `sbatch -q normal -p batch_long -t 20:00:00 -J isla.floordrv.train --exclude=nvl72d015-T12,nvl72d058-T08,nvl72d052-T01 --array=0 --export=ALL,ROWS=$T/floordrv/harness/rows_fd.tsv,DRAW_SYNC=1,DRAW_WORKERS=1,HELDOUT_WHOLE=1 overfit_fd.sbatch`
- Outputs `$T/floordrv/runs/<run>/` (log.jsonl, stdout.log, resolved_config.yaml, ckpt.pt, final_predictions.pt,
  final_model.pt, .exit, .done); Slurm logs `$T/floordrv/sbatch_logs/`. Registry row `^isla\.floordrv\.`.
- **Deadline.** The harness deadline is about 16:00 EDT 2026-10-24. If the full job starts later than about 20:00 EDT
  today it cannot finish in time; HANDOFF_FLOOR.md is written by 14:00 EDT 10-24 if the runs are still unread, with the
  reduce and residual commands.
- md5 (local = LHR `$T/floordrv/harness/`): overfit_rdp.py 14a34367, overfit_fd.sbatch 84159d2b, frame_check.py
  c1564b2c, reduce_fd.py 0aea72df, rows_fd.tsv 06d8f772, rows_fd_smoke.tsv 87a0d9b9, domain_geom.py 99b23395,
  walls_all_cases.py ed49054a; isla_volume_floor.yaml ffc47632.

## 7. Amendment A1 before submission (2026-10-23 12:22 EDT; the supervisor's condition of approval, before any job)

Approved by the supervisor (reply to the approval request, about 12:18 EDT) with one change to arm F. No job of this
lane has run (smoke included), so no number of any FLOOR-DRV arm existed when this was written. Seen since section 0:
only data facts (below), no model output.

- **Arm F reads all four domain meshes** the case files store: `no_slip`, `slip`, `inlet` and `outlet`, each by its
  file name with `{}` (no prescribed values in the files), beside `vehicle`: the complete PDE boundary, as on AeroZoo.
  Reasons (supervisor): the tested configuration should be the one that would be adopted at the scope freeze; and a
  generalized-winding-number sign (the candidate geometry bundle) needs a closed domain surface, which floor and walls
  alone do not give. ISLA 0.9.0's pseudo-normal sign, used here, does not need it.
  - Config: `conf/model/isla_volume_floor.yaml` at psharpe/isla-floor-drv **564f803cc** (md5 041657b6; was cba4197f3,
    md5 ffc47632). Test `test_isla_volume_floor_adds_the_domain_boundaries`; recipe tests test_isla_configs,
    test_utils, test_forward_kwargs, test_synthetic_configs: 215 passed. Hooks on, `-s`, pushed.
  - Snapshot `recipe_r090_fd`: isla_volume_floor.yaml replaced (md5 before ffc47632, after 041657b6); the other 135
    files md5-identical to recipe_r090; SNAPSHOT_HEAD line "amendment A1".
- **Data checks extended** (`harness/walls_all_cases.py` 0ff7ece1, raw files, all 484 cases): all four domain meshes
  exist and are non-empty on every case (0 missing), every face lies on the domain box (0 off-box), and every face of
  every mesh points into the fluid (0 misoriented). Eight distinct bounding boxes: the four meshes at the two ground
  heights (-0.318 m on 187 cases, -0.320 m on 297).
- **Length unit and measure** (`harness/domain_geom.py` b93d4505, run_171, vehicle from every 100th cell): L = 2.04 m
  (vehicle), 55.5 m (+ no_slip + slip), **59.5 m with all five**; the vehicle holds **0.197 %** of the boundary
  measure (0.219 % without inlet and outlet). Tokens: inlet and outlet (722 each) are kept whole too, so the vehicle
  gets 200,000 - 722 x 3 - 2,888 = 194,946 of the 200,000 boundary tokens. Every s = d / L is about 29x smaller than
  in C. The section-4 "hurts" reading stands for this reason.
- **Frame check** (`harness/frame_check.py` 5fb72108) checks every boundary the config passes, so all four domain
  meshes, against the vehicle's translation in each draw.
- **Descriptive readout widened** (`harness/reduce_fd.py` c0504e6c): the e_R per wall-distance decade, with its ratio
  to Transolver's, for velocity, pressure and nu_t (was velocity only), for F, C and Transolver, so a near-wall cost of
  the length unit shows on every field. Descriptive only; no rule reads it.
- **Unchanged**: arms C, seeds, protocol, rows (`rows_fd.tsv` 06d8f772, `rows_fd_smoke.tsv` 87a0d9b9), readings and
  bars (section 4), predictions (section 5; written for the floor-and-walls arm, kept as written for the four-mesh
  arm), stop rule, cost and the 80 GPU-h cap.

## 8. Amendment A2 before submission (2026-10-23 12:56 EDT; the smoke's stop rule; approved by the supervisor, option A)

Written before any full-length job. Seen: only the smoke's 200-step losses (no score of any arm at any floor step).

- **The stop rule tripped** (section 2). Smoke 355126 (nvl72d009-T06): F 5.27 s/step, C 3.33 s/step, **1.58x** (bar
  1.3x); peak memory 234.1 GiB each (bar 270); both rows rc 0, finite, five meshes in F's setup and the vehicle
  alone in C's, point hashes as Part B. The extra 1.9 s/step is the four domain meshes in ISLA 0.9.0's in-model
  geometry (18 % of the interior points are nearest a domain wall). It is reported as a cost finding beside the
  readings: if the input were adopted, ISLA's DrivAerML Volume step would cost 1.58x.
- **Frame check: PASS after an instrument fix** (job 355437, frame_check.py 0505c3bc). The first version (in 355126)
  required processed = raw + t and failed by construction: drivaer_ml_volume_reference also divides every
  coordinate by L_ref = 5 m (NonDimensionalizeByMetadata, scale_geometry true by default). The fixed check fits
  processed = a raw + t per mesh: on epochs 0, 1, 1000, 1001, 1002, a = 1 / L_ref within 2e-8, the same for all five
  meshes within 3.4e-8 relative, t the same within 0.004 mm, per-point residual at most 0.0035 mm; interior means 0.
  87-214 interior points per draw have a union distance of -4e-8 m, all nearest a domain wall: CFD nodes on the
  ground, float32 rounding (vehicle alone: none negative). L_all = 11.90 L_ref = 59.5 m, L_vehicle = 0.408 L_ref.
- **Approved deviation**: run as registered at 20,000 steps, all four rows, one 4-GPU node. **Cap raised to 125
  GPU-h** (smoke and check about 1.5 GPU-h; the full job about 4 x 30 = 120 GPU-h allocated).
- **Schedule**: one segment. LHR normal QoS has no MaxWall and batch_long allows 7 days, so `-t 32:00:00` fits; no
  resume is planned. (If a node failure requeues it, the runs resume from ckpt.pt, and ISLA on DrivAerML Volume does
  not resume bit-for-bit: that affects F and C alike and is recorded in JOBS.md.) Expected end about 21:00 EDT 10-24,
  after this lane's deadline (16:00 EDT 10-24): HANDOFF_FLOOR.md by 14:00 EDT 10-24; a successor reads.
- **Rows**: the wall-time guard `max_minutes` of all four rows raised from 1140 to 1860 (rows_fd.tsv 06d8f772 ->
  61e65032; every other column unchanged). 1140 would stop F at about step 13,000 and leave C at the smoke's 3.33
  s/step (18.5 h = 1,110 min) with no margin. A guard only; it changes no training.
- **Units of the distance bands.** The labels' sdf (labels.py: interior.point_data.sdf, computed after the scaling) is
  in units of L_ref = 5 m, so every CAP-AUDIT and FLOOR-DRV wall-distance band is in L_ref units. The registered
  far-field band [1, inf) is unchanged in those units; in prose it reads **beyond 5 m from the car**, and CAP's
  "1e-4 to 1e-3 m" is 0.5-5 mm. Readouts keep the band labels; the write-up converts to metres.
