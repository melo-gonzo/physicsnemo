# SURFACE-BC: an ISLA surface task that reads the whole boundary (option A), built and measured

Research lane of the ISLA program, launched 2026-10-23 about 20:53 EDT by the supervisor (01a13c2c); written 2026-10-23
about 22:45 EDT. Peter has not yet chosen between (A) and (B); this is the supervisor's recommendation (A), on its own
branch, not merged. Scratch: this folder. LHR: `$T/kernel/surfbc`, snapshots `$T/code_surfbc`, `$T/recipe_surfbc`,
`$T/recipe_surfbc2` (`T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0`).
Details: `DESIGN.md` (design, both API shapes, what changes, the cost of (B)), `cost_table.md` (GB300 numbers),
`JOBS.md` (every job and commit, with md5s).

## 1. Design

One behaviour for every surface task: every given boundary's cells are tokens (as for the interior task), and the
outputs are predicted at the cells of the boundaries named as carrying them. There is no default that predicts on every
boundary and no flag for the old path.

- **Shape 1 (built, recommended): `forward(..., prediction_boundaries=["vehicle"])`**, keyword-only. It is the
  surface analogue of `prediction_points`: required for a surface task and refused for an interior task. It must be a
  non-empty list or tuple of distinct given names; a bare string is refused. It owns no parameters, so by the model's
  own rule (a constructor argument is what changes the parameter set) it is data of the call. The constructor, the state
  dict and every 0.9.0 checkpoint (the release test data included) are unchanged. A call that names every given
  boundary computes what 0.9.0 computed: DeepJEB++ names `[fixed, free, loaded]`, HiLift names `[vehicle]`. The recipe
  spells it `prediction_boundaries: {literal: [vehicle]}` (a new `{literal: value}` form of the forward_kwargs
  resolver). No version bump; `_refuse_earlier_state_dict` untouched.
- **Shape 2 (not built): the constructor argument `ISLA(prediction_boundaries=[...])`** beside `boundary_schemas`. It
  is validated once and recorded in the checkpoint, so inference cannot predict on the floor by a config slip. Against
  it: it owns no parameters and has no meaningful class default, so every stored 0.9.0 surface checkpoint stops
  constructing from its recorded arguments unless a default keeps the old path (a shim).
- **What changes.** The token draw is none: the surface task still reads every cell; the vehicle is the dataset's 200,000-cell
  subsample and the tunnel is read whole, 5,054 cells. The output cells' geometry is unchanged in kind: tokens with
  centroid, normal and measure, and no wall-distance channel. The floor and walls reach the vehicle cells through the
  slice anchors they form. L is the RMS pairwise distance over all five boundaries: 2.17 m becomes 59.50 m on the
  measured case, and the vehicle spans about 0.08 L. No surface config declares `reynolds`; z would move by +1.45 if one
  did. The winding-number sign does not enter (A), which computes no signed distance. Pose equivariance is exact with
  every boundary moved alike, which is tested. The tunnel holds 99.8 % of the boundary measure, as on DrivAerML Volume
  since c1e07b7fd.
- **(B) would cost** a different model (the interior task's parameters and query coupling) with 200,000 duplicated
  tokens. Every prediction point would sit on the wall, so its distance channels are degenerate. It would also need a
  full-resolution vehicle in every step, and a winding sign at the open vehicle-floor seam is ill-defined. Its step is
  the Volume step: 2,897 ms and 140.6 GB on 0.9.0 (BOUNDARY-AUDIT), against 271-275 ms and 70.7 GB for (A).

## 2. Branch and commits

`psharpe/isla-surface-bc` (worktree `/home/psharpe/gh/physicsnemo-isla-surface-bc`, from `origin/psharpe/isla-whole-boundary`
c1e07b7fd), pushed with plain pushes, head e636f29bd. All commits `-s`, hooks on; ruff-format reformatted touched files
that were stored unformatted, and each commit was redone after the hook.

| commit | what |
|---|---|
| fd0b7bf31 | ISLA: `prediction_boundaries` (model.py, docs/api isla.rst, CHANGELOG); every surface call in the ISLA tests passes it; new `test_isla_surface_conditions.py` |
| 129c66294 | Recipe: `MeshReaderWithGlobalData(domain_boundaries=...)` returns a DomainMesh (vehicle as interior until `SurfaceMeshToDomainMesh`), the four DrivAerML Surface dataset files read inlet, no_slip, outlet, slip; `isla_surface.yaml` reads all five, predicts at the vehicle, reverses the vehicle's winding for ISLA only (every piece's normals into the fluid); `{literal: ...}` resolver form; HiLift and DeepJEB++ configs declare their prediction boundaries; README |
| 2ea2fad32 | two recipe tests with ISLA surface specs name their prediction boundary |
| 422df2970 | merge of `origin/psharpe/isla-drv-ground` (c4419b299), `--no-ff --signoff`, no conflicts |
| 35a460b34 | `whole_surface_transforms` in the reader: DRV-GROUND's `PlaceGroundUnderBody` on the whole vehicle (memory map, raw metres) before its subsample and every pose, in the four DrivAerML Surface files |
| e636f29bd | README line for it |

Not touched: DrivAerML Volume's config (the merge brings DRV-GROUND's own Volume changes), AirfRANS, the geometry kernels,
the ISLA version.

## 3. Tests

- **ISLA test directory** (`test/experimental/models/isla/`, CPU): 280 passed, 47 skipped, 0 failed, both locally
  (`logs/pytest_isla_2.log`, niced, 45 min) and on LHR cpu-short in three groups (362047-362049,
  `logs/lhr/tests_isla.surfbc.t[ABC]_*.log`). The skips are CUDA-only tests and one test that needs an external sample.
  No failure, so no base comparison was needed. The model code has not changed since.
- **New model tests** (`test_isla_surface_conditions.py`, 7): outputs at the prediction boundary's cells only, in cell
  order; more names return more rows of the same prediction; a condition-only boundary is read (dropping it, moving it,
  or changing the inlet's field changes the vehicle's output); pose covariance (proper and improper) with every boundary
  moved alike; gradients reach every condition's parameters; the argument's checks.
- **Recipe tests** (LHR 362167, all of `tests/`, recipe at 35a460b34): 534 passed, 12 skipped, 2 failed. The two failures
  (`test_campaign_c_hooks` initialize_from_checkpoint, `TypeError: int() argument ... 'NoneType'` in
  `DistributedManager.initialize_slurm` inside a Slurm job) fail with the same message on the base c1e07b7fd (362118),
  and pass locally. New: the reader test (the domain pieces whole, the vehicle's draw unchanged, the ground placed from
  a whole vehicle whose subsample misses its lowest vertex); the pipeline test (`drivaer_ml_surface` and `_posed`: the
  one similarity fitted on the placed tunnel maps every output vehicle point onto a stored one); the config tests (five
  boundaries, predictions at the vehicle, the flip; every DrivAerML Surface file reads the tunnel and places the ground
  in the reader); the literal resolver.
- **Ground on real files** (LHR 362186, `logs/lhr/ground.jsonl`): after the chain the ground sits at the whole
  vehicle's lowest z, (z_min - centre_z) / L_ref, within 3e-8 m. This holds on run_3 and run_28 (offset files: moved up
  2.403 mm) and on run_1, run_100 and run_188 (matching files: moved by 0).

## 4. Cost on GB300 (`cost_table.md`; 0.71 GPU-h of the 4 GPU-h cap)

One case (run_1): the training step takes 223 ms body only and 273 ms whole (1.22x); the eval forward 64 and 73 ms
(1.15x); peak training memory is 69.1 and 70.7 GB. The recipe's train.py over 20 steps gives 1.24x and 65.0 against
67.3 GB; the whole run is also the 20-step smoke, with finite losses (0.121 to 0.062). **The 1.2x is the token count's alignment,
not the tunnel.** 205,054 is not a multiple of 8, and every count that is not a multiple of 8 costs about 268 ms, the
body at 199,994 or 205,054 cells included: the bf16 GEMMs fall to slower kernels. At 205,048 tokens the whole boundary
takes 229.5 ms against 228.0 ms for the body at the same count, 1.03x the 200,000-token body. This affects every ISLA run
with an unaligned token count, and was measured in eager mode with compile_blocks only.

## 5. What SHIFT-SUV and HiLift would still need (from SETUP.md; nothing done)

- **Both** have the model change and a reader that loads named sibling boundaries and runs whole-surface transforms. Still
  missing is the domain pieces themselves: SHIFT-SUV stores none, and HiLift stores them on 2 of 1,800 files. Pieces from a
  new versioned sidecar folder would need a reader path option, since the reader now looks only beside the surface.
- **SHIFT-SUV.** A box sidecar from curator's generator (y +-20 m, floor at the interior's z_min, 1.15 mm below the
  vehicle). It is blocked on condition types that no source documents: walls, roof, outlet, and the belt layout. The tyre
  closure is not stored. The vehicle is open at the tyre patches.
- **HiLift.** A hemisphere sidecar (inlet, outlet, symmetry). Peter must rule on the radius: 75 MAC documented, 88.9-90.6
  MAC stored. The symmetry disk needs the full-resolution wing root; at 248-303 M triangles that is a per-case offline
  build, not a per-step transform. `isla_surface_highlift.yaml` already names `prediction_boundaries: [vehicle]`.

## 6. For the book

No standing changes before Peter chooses (A) or (B). If (A) stands, the forward-pass chapter's surface-task sentence
becomes "a surface task reads every given boundary and predicts at the cells of its `prediction_boundaries`". The cost
finding (unaligned token counts, +20 % step) belongs in the performance chapter, as measured here.
