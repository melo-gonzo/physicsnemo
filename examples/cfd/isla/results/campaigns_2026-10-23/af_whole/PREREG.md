# AF-WHOLE: ISLA and ISLA-DTT reading the AirfRANS far field (pre-registration, 2026-10-23)

Lane AF-WHOLE of ISLA (research-lane, launched 2026-10-23 about 20:25 EDT by the supervisor 01a13c2c). Written about
21:00 EDT 2026-10-23, **before any GPU job of this lane: no smoke, lane or score job has been submitted, and no W
checkpoint, log or score exists.** The only jobs so far are three read-only cpu-short jobs (facts and the constants
fit, section 3).

## 1. Question and rulings

Peter ruled (2026-10-23 17:08 EDT) that canonical ISLA reads every stored piece of the PDE domain's boundary, on every
dataset, and that no accuracy or cost result reverses this. The supervisor decided (the supervisor's decisions, not
Peter's): on AirfRANS the PDE's boundary is the airfoil plus the far field; the crop's edges are not a boundary of the
PDE; the far field enters as a named boundary `freestream`, read in the recipe from the raw release beside the pdmsh,
with no per-cell fields (its U, p and nut are solution values), its prescribed value being the global U_inf ISLA
already reads; no dataset is modified; normals into the fluid.

This experiment **measures the effect** of reading the far field on ISLA 0.9.0 and ISLA-DTT on full AirfRANS
(full_train to full_test, 200 cases). Adoption is already ruled; a "harms" reading starts a cause search (section 7),
not a rollback.

## 2. What I have seen before registering (disclosure)

- The standings' scores, all of them (`../isla-ref090-2026-10-20/eag_af_isla.md`, `eag_af_dtt.md`), epoch 500, 200
  cases, float32 (table in section 5), and their posed / native ratios.
- The standings' cost from sacct: r090eag_af_isla 7 links, 22.2 / 22.4 h, 88.9 / 89.4 GPU-h; r090eag_af_dtt 10 / 9 links,
  27.7 / 27.5 h, 110.8 / 110.0 GPU-h. B's smoke 323028: 0.766 s median step, 43.0 GiB.
- SLICES-AF's PREREG and JOBS.md (its smoke and launcher), not its results. BOUNDARY-AUDIT's REPORT.md.
- The facts of section 3 (geometry and data only; no model was trained or scored).
- Not opened: scratch/isla-tie-2026-10-22/ (TIE-EXP blinding).

## 3. Facts before training (all 1,000 cases; `FACTS.md`, `out/facts_summary.txt`)

LHR cpu-short jobs 361286 / 361287 (`harness/facts.py`, code `$T/code_cmpaf_fix`, recipe src of the branch) and 361288
(the constants fit), 55-59 s each. Lengths in chords.

| quantity | airfoil only (standing) | airfoil + far field (W) |
|---|---|---|
| far field | - | 1,563-1,923 segments (median 1,727), one closed chain, counterclockwise, normals into the fluid on every cell of every case; half circle r = 200 upstream, lines x = 200 and y = +-200 downstream; perimeter 1,428 |
| boundary tokens (all kept, budget 200,000) | 843-1,217 | 2,406-3,140 (airfoil 35-39 % of tokens, 0.14 % of the measure) |
| ISLA's L (`ISLA._gauge`, float64) | 0.411-0.437 (median 0.421) | **305.3 on every case** (725x) |
| 1e-6 L (LENGTH_SOFTENING) | 4.1-4.4e-7 | **3.05e-4** |
| first-node height (per wall node, the sdf of its nearest off-wall node) | median 2.10e-6, 1st percentile 1.59e-6 (per-case medians 1.91-2.11e-6) | same |
| first node in units of the softening, s / 1e-6 | 4.4-5.1 (asinh 2.3: start of the log part) | **0.0069** (asinh 0.0069: deep in the linear part) |
| interior nodes with \|s\| < 1e-6 (linear part) | 0.56 % (the wall nodes, d = 0, only) | **20.2 %** (19.7 % of the off-wall nodes) |
| interior nodes with \|s\| < 1e-5 | 1.4 % | 37.6 % |
| log10 Re = log10(U L / nu) | 5.92-6.41 | 8.79-9.26: shift **+2.86** (2.84-2.87) |
| z_LRE = log10(Re / 1e6) | -0.08 to 0.41 | 2.79 to 3.26 |
| reference constants (full_train fit) | reynolds {6.205, 0.1347}, nut 7.511e-05 | **reynolds {9.065, 0.1338}, nut 1.045e-07** |
| canonical_frame: eigenvalue gap (l1 - l0) / l1 | 0.94-0.996 | **0.0330-0.0331** |
| frame axis 0 | normal to the chord (87-92 deg) | the lab x axis (the far field's axis), sign by a skew of 0.184 |
| frame axis 1 sign | skew 7e-6 to 0.017 | **decided by the airfoil**: skew 5e-8 to 8e-5, + on 724 cases, - on 276; 15 cases below 1e-6, 1 below 1e-7 (a train case), none below the tie 1e-9 |

Meaning:
- The far field is not isotropic: a half disc plus a rectangle gives a 3.3 % gap, so the frame's axes are the far
  field's, decided in float64 with a large margin; only the sign of the second axis follows the airfoil (its third
  moment is 1e-5 of the far field's scale). A rigid pose moves it with the case; the smallest skew (5e-8) is far above
  float64 rounding, though float32 inputs could flip it on a near-tie case (none in full_test below 1e-7).
- With L = 305 chords the near-wall channel asinh(s / 1e-6) is linear for the inner 3e-4 chords: the boundary layer's
  first decades lose their logarithmic code (the docstring of `soft_log_length` assumes the first nodes at about
  5e-6 L; here they are at 7e-9 L). This is the leading suspect for any near-wall harm.
- The constants must be refit (with the airfoil-only constants z would be about 21 standard deviations off): done for
  every ISLA AirfRANS split with the implemented scale functions (`examples/cfd/isla/results/aerozoo_dataset/
  airfrans_whole_constants_fits.json`).
- Side finding, not acted on: on 168 of 1,000 cases the converter's stored airfoil `normals` disagree with the
  geometric normals (the polyline's winding) on 5-39 % of the segments. ISLA reads the geometric normals, which follow
  the winding; the stored field looks like the converter's per-segment centroid heuristic failing on cambered aft
  sections. Not investigated further; no config reads the stored field.

Implementation (task 1): commit **8e93f2307** on psharpe/isla-whole-boundary (pushed; then merged with
psharpe/isla-drv-ground as 4b772ec59 on the supervisor's steer, which changes no AirfRANS file). Recipe tests 540 passed /
12 skipped (545 / 12 after the merge); the ISLA test directory 273 passed / 47 skipped. The reference-width model's
forward on CPU on 2 real cases through the recipe's own pipeline, ISLA and ISLA-DTT, native and posed: finite, airfoil +
freestream as 2-D line meshes, L 305.32, 41-59 s per 20,000-point forward (JOBS.md).

## 4. Arms

`T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0` (LHR, GB300).

- **B-ISLA (reused):** `r090eag_af_isla_seed42/43`, rows 0-1 of `$T/kernel/r090eagaf/lanes.tsv`: code
  `$T/code_cmpaf_fix`, recipe `$T/recipe_r090`, `model=isla_volume_airfrans`, `dataset=airfrans_uvolume_2d`, extra
  `train_split=full_train val_split=full_test +training.val_interval=10 model.hidden=512 model.n_slices=512
  model.compile_blocks=true compile=false model.compile_blocks=false` (eager), lr 3.75e-4, 500 epochs, 100,000 points,
  bfloat16, seeds 42 / 43.
- **B-DTT (reused):** `r090eag_af_dtt_seed42/43`, rows 2-3: the same plus `++model.query_independent=true`.
- **W-ISLA, W-DTT:** B-ISLA / B-DTT with the recipe snapshot `$T/recipe_afwhole` in place of `recipe_r090`. Nothing else
  differs: same code snapshot, lanes.tsv extra, seeds, precision, epochs, cells, launcher scripts.
  - `recipe_afwhole` = a copy of `recipe_r090` with only these files replaced, verified by md5 against the branch
    commit 8e93f2307 (SNAPSHOT_HEAD): `src/domain_transforms.py` (the r090 file is byte-equal to the branch's base; the branch
    adds `VtkBoundaryDomainMeshReader`, its case-key twin and `read_vtk_boundary`), `conf/model/isla_volume_airfrans.yaml`
    (r090's file plus the branch's two entries: `forward_kwargs.boundary_meshes.freestream`,
    `model.boundary_schemas.freestream: {}`, and comments), `datasets/airfrans_{u,}volume_2d{,_area,_posed}.yaml` (r090's
    are byte-equal to the branch's base; the branch changes the reader to the VTK-boundary reader with
    `vtk_boundaries.freestream: ${raw_datadir}/{case}/{case}_freestream.vtp`, `vtk_plane_axis: 2`, and the refit
    `isla_constants`), `requirements.txt` (pyvista, already in the venv).
  - Launcher `$T/kernel/afwhole/`: `lanes.sbatch`, `ckpt_preflight.py`, `score.sbatch` byte copies of kernel/r090eagaf's
    (cmp); `submit.sh` r090eagaf's with the prefix `isla.afwhole.`; `site.env` r090eagaf's with `RECIPE=$T/recipe_afwhole`
    and `LANE_SBATCH_EXTRA="--segment=1 --exclude=$T/kernel/afwhole/exclude_nodes.txt"`, the file listing the five bad
    nodes (nvl72d015-T12, nvl72d058-T08, nvl72d052-T01, nvl72d057-T13, nvl72d043-T18); `lanes.tsv` r090eagaf rows 0-3
    with group `afwhole` and run ids `afwhole_af_{isla,dtt}_seed4x`, nothing else. Job names
    `isla.afwhole.af-{isla,dtt}-seed4x`; normal QoS, 03:59:00 chained singleton links.

No new control runs. The two-seed spread of each standing is the floor.

## 5. Smoke, training, scoring, checks

- **Smoke** (one short-QoS 4-GPU job, 00:40:00, the five nodes excluded; `isla.afwhole.smoke`): r090eagaf's
  smoke.sbatch and smoke_wrap.py, paths and job name changed, four units one after another on one node, 50 steps each
  through the lane launcher with NO_CHAIN: B-ISLA and B-DTT (`kernel/afwholeb`: byte copies of r090eagaf's lanes.sbatch,
  site.env, lanes.tsv, ckpt_preflight.py, so B runs B's files without writing into r090eagaf), W-ISLA and W-DTT
  (`kernel/afwhole` rows 0 and 2). Passes if every unit has rc 0, no traceback, finite losses; each W unit's
  resolved_config.yaml equals its B unit's except `run_id`, the dataset reader (`_target_`, `vtk_boundaries`,
  `vtk_plane_axis`), `raw_datadir`, `isla_constants` and the model's injected `reynolds` and nut output scale, `forward_kwargs.boundary_meshes.
  freestream` and `model.boundary_schemas.freestream`; and each B unit's equals `runs/r090eag_af_{isla,dtt}_seed42/
  resolved_config.yaml` except the smoke paths. Lanes are submitted only if it passes. Cost: W / B median step and
  peak allocated memory per rank, same node, same settings.
- **Lanes:** `bash submit.sh lanes 0 3`. At each lane's first link its resolved_config.yaml must equal its B run's
  except the keys listed above; otherwise I stop and report.
- **Event guard:** a W lane's validation loss above 4x its running minimum after epoch 10 (B's worst: 3.05), a
  non-finite loss, a BLOCKED marker, a hung or failed link: I stop and tell the supervisor. I never resubmit; I cancel
  only a finished run's trailing pending links, by explicit id.
- **Scoring exactly as B:** kernel/r090eagaf/score.sbatch unchanged, submitted from kernel/afwhole as
  `isla.afwhole.score`, nodes excluded, `SCORE_ROWS=$T/kernel/afwhole/score_rows.tsv` (r090eagaf's rows 0-2, 4-6, 8-10,
  12-14 with the run ids and the recipe column changed to recipe_afwhole), `EVAL_ROOT=$T/evals_afwhole`: native, area,
  posed; float32, `infer_split=val` (full_test), 100,000 points, the epoch-500 checkpoint.
- **Near-wall inference:** SLICES-AF's nearwall.sbatch (native, `+save_predictions=true`) for the four W runs and the two
  B-DTT runs (B-ISLA's dumps exist: `$T/evals_slices_nw/r090eag_af_isla_seed4x/native/`); reduction by SLICES-AF's
  nearwall.py with the decades of section 6 R2.
- **Checks before reading** each W row: `"epoch": 500`, 200 `infer_step` rows with the same sample ids as B's,
  SNAPSHOT_USED `code_cmpaf_fix`, DATASET_USED as B's, the same scored points as B (the near-wall dumps' coordinate
  hash per case; the far field never draws, so the interior draw is unchanged), the recipe column recipe_afwhole.
  A failed check makes the row invalid and the read incomplete as registered.

## 6. Rules (registered)

**R1, per field, native protocol, the 200 full_test cases; one reading per arm pair (W-ISLA against B-ISLA, W-DTT
against B-DTT).** Fields: pressure, velocity (in-plane; never velocity_z), nut. CM = the scorer's case-mean relative L2.
- `D = ln( mean_seeds CM(W) / mean_seeds CM(B) )`, `SE = sqrt((s_W^2 + s_B^2) / 2)`, s = an arm's two-seed sample sd
  (n - 1) of ln CM.
- **Helps** if D < ln 0.90 and |D| > 2 SE; **harms** if D > ln 1.10 and |D| > 2 SE; otherwise **no effect**.

The standings (B) and their floors:

| B | field | s42 | s43 | mean | s_B | W / B bar for helps / harms if s_W = s_B |
|---|---|---|---|---|---|---|
| ISLA | pressure | 0.0131631 | 0.0117940 | 0.0124786 | 0.0777 | 0.856 / 1.168 |
| ISLA | velocity | 0.0064407 | 0.0062701 | 0.0063554 | 0.0190 | 0.909 / 1.100 |
| ISLA | nut | 0.0157382 | 0.0149675 | 0.0153529 | 0.0355 | 0.909 / 1.100 |
| DTT | pressure | 0.0124280 | 0.0123887 | 0.0124084 | 0.0022 | 0.909 / 1.100 |
| DTT | velocity | 0.0073734 | 0.0069725 | 0.0071730 | 0.0395 | 0.909 / 1.100 |
| DTT | nut | 0.0207112 | 0.0204842 | 0.0205977 | 0.0078 | 0.909 / 1.100 |

The same D, SE and label on the **posed** protocol beside (the same runs: no independent number). The **area** protocol
is checked to equal native and then labelled degenerate (both boundaries are below the draw's count). Beside, not
verdicts: seed-matched ratios, case medians, the fraction of the 200 cases where W beats B. Slots: 2 arm pairs x 2
seeds x 2 arms = 8 runs, 4 of them new; three fields and two protocols are readings of those runs.

**R2, per field and wall-distance decade (descriptive labels, not verdicts).** From the native dumps, the stored signed
wall distance in chords, decades (-inf, 1e-6), [1e-6, 1e-5), [1e-5, 1e-4), [1e-4, 1e-3), [1e-3, 1e-2), [1e-2, 0.1),
[0.1, 1), [1, inf); count weights; pooled band e_R = sqrt(sum_R |p - t|^2 / sum_R |t|^2) over the 200 cases (pressure /
(0.5 U^2), velocity / U, nut in m^2/s), and beside the case mean of per-case band e_R. Plus the split at the softening,
d < 3.05e-4 (the linear part with W's L) against d >= 3.05e-4. Per band and field: the R1 rule on the pooled band e_R
(each run's band value as its CM), labelled "band helps / harms / no effect". Checks: the same points per case in all
eight runs (coordinate hash) and the dumps' per-case e equal to the scorer's pressure and velocity l2 to 1e-4.

**R3, the pose reading, per arm (W-ISLA, W-DTT), per field and seed.** posed / native of CM. **Pose holds** if every
field and seed lies within 0.002 of one. Beside it, the standing's own values (B-ISLA pressure 1.0022 / 1.0025, B-DTT
pressure 1.0017 / 1.0022), and the label **"holds at the standing's level"** when the 0.002 bar fails only on a field
where W's worst seed is no farther from one than its B's worst seed. A failure beyond that is "pose broken", which I
report as a defect to trace (the frame's second-axis sign is the first suspect for W-DTT).

**R4, cost, descriptive.** W / B smoke median step and peak allocated memory (same node), each lane's median
`train_step` time and GPU-h (sacct elapsed x 4).

## 7. What a reading leads to

- No harm and pose holds: the far field is ISLA's reference AirfRANS input at no accuracy cost; the W runs become the
  standings' successors for the book (the supervisor decides the standing).
- Harms on a field: I look for the cause, starting with the evidence R2 gives. If the harm sits in the decades below
  3.05e-4 chords (W / B band ratio there above the outer bands' ratio on both seeds), the leading cause is the
  softening 1e-6 L, which the far field moves outside the boundary layer; a fix would make the softening a property
  of the geometry rather than a fixed fraction of L, a model change for the supervisor and Peter, not tested here.
  Otherwise, for W-DTT the frame (its second axis) and the far field's 99.86 % share of the measure in the
  measure-weighted reads are the next suspects. I report the cause analysis; I launch nothing beyond this registration.
- Pose broken: reported as a defect with the cases that break it.

## 8. Predictions (mine)

| id | prediction | p |
|---|---|---|
| P1 | The smoke passes on its first run | 0.85 |
| P2 | W-ISLA pressure: helps / no effect / harms | 0.10 / 0.65 / 0.25 |
| P3 | W-ISLA velocity: helps / no effect / harms | 0.05 / 0.60 / 0.35 |
| P4 | W-ISLA nut: helps / no effect / harms | 0.05 / 0.55 / 0.40 |
| P5 | W-DTT pressure: helps / no effect / harms | 0.10 / 0.60 / 0.30 |
| P6 | W-DTT velocity: helps / no effect / harms | 0.05 / 0.70 / 0.25 |
| P7 | W-DTT nut: helps / no effect / harms | 0.05 / 0.65 / 0.30 |
| P8 | R2: W-ISLA's pooled nut e_R in [1e-6, 1e-4) above B-ISLA's on both seeds | 0.60 |
| P9 | R2: W / B is larger below 3.05e-4 chords than above it, for velocity and nut, on W-ISLA | 0.60 |
| P10 | R3: pose holds or holds at the standing's level, W-ISLA | 0.85 |
| P11 | R3: the same, W-DTT | 0.75 |
| P12 | R4: W / B smoke median step in [1.00, 1.25] for ISLA | 0.65 |
| P13 | R4: the same for ISLA-DTT | 0.50 |
| P14 | All four W lanes finish with no event-guard stop | 0.85 |

Why: the far field is 200 chords from every prediction point and carries no field, so its information is the
geometry's frame and scale; what it changes is the length unit, which moves 20 % of the interior into the linear part
of the near-wall channel and shifts every log-distance input by log 725. Retraining absorbs the shift, not the lost
log resolution; ISLA-DTT keeps log(r / |d_q|) in its boundary read, so it should lose less near the wall. The added
1,700 tokens are 2-3 % of the 100,000 prediction points, but the two-dimensional exact geometry search grows with the
cells (about 2.7x).

## 9. Cost, cap and schedule

- Lanes: B used 89 (ISLA) and 110 (DTT) GPU-h per lane; at 1.0-1.25x the step, **about 400-500 GPU-h for four**.
  Smoke at most 2.7 GPU-h (00:40:00 x 4). Scoring about 2 GPU-h (12 rows). Near-wall inference about 2 GPU-h, plus
  cpu-short. **Total about 410-510 GPU-h, cap 520.** If the smoke projects past 520, I ask before the lanes.
- Wall: ISLA lanes about 22-28 h, DTT lanes about 28-35 h, plus queue gaps. If they start by about 01:00 EDT 10-24, ISLA
  ends about 10-25 03:00-08:00 and DTT about 10-25 08:00-15:00 EDT; scoring an hour after each. Harness deadline 10-25
  20:25 EDT; HANDOFF_AFWHOLE.md at 17:25 EDT if any arm is unread. The DTT pair is at risk of the deadline.
- Registry row `^isla\.afwhole\.` (expected_end 2026-10-25) was added at the first cpu-short job (20:44 EDT) and is
  removed after the last job.

## 10. Amendment before submission (2026-10-23 21:45 EDT, before any GPU job)

Approved by the supervisor at 21:38 EDT as written (md5 19a1274b9d13682124639936742a75b0), cap 520 GPU-h, with four
conditions, recorded here. Nothing of any W run existed when they were added.

1. **Scoring guard.** Every W scoring row's saved resolved config must list `freestream` in
   `forward_kwargs.boundary_meshes` and in `model.boundary_schemas`, and the dataset file it read (in recipe_afwhole)
   must have the md5 of the branch's far-field file (8e93f2307). The reducer asserts both before any number is read; the
   smoke asserts them for the W units. A row that fails is invalid (section 5's rule).
2. **Posed far field** (done on CPU before the smoke, all 200 full_test cases, the recipe's own native and posed
   pipelines, LHR cpu-short 361835, `harness/frame_pose.py`, `out/frame_pose.jsonl`): the rigid map fitted on the
   airfoil maps the native far field onto the posed one to at most 4.2e-5 chords (median 2.3e-5; float32 at 200 chords
   is about 2.4e-5 per coordinate); airfoil residual at most 1.3e-7; the same cells in both.
3. **Frame sign, a fact beside R3** (same job, the frame as ISLA-DTT's tree read builds it: float32 r = centroid / L,
   measure fractions): over the 200 cases, posed axis against the rotated native axis, **axis 0 flipped on 0 cases, axis
   1 on 0 cases**. The second axis's skew in float32 is 2.85e-8 at the least (5 % below 3.3e-6, median 2.8e-5); the
   pose changes it by 0.6 % at the median and 56 % at the worst, never through zero. So a W-DTT pose deviation in R3, if
   any, does not come from a frame-sign flip on the scored cases.
4. The fixed-scale file (airfrans_uvolume_2d_reynolds_lre.yaml) keeps reynolds {6.0, 1.0} and refits only the nut scale;
   with the far field its z runs 2.79 to 3.26.

Also, at the supervisor's request: RESULTS.md records the stored-airfoil-normals finding (section 3) with its case list,
beside the rules. No action on it.
