# REYNOLDS-OUT: taking the Reynolds number out of ISLA's constructor (design, 2026-10-24)

Base: psharpe/isla 5b70c8686 (ISLA 0.12.0). Copies of the two model files read: `src/model.py`, `src/scales.py`.
Peter's direction (b) of 2026-10-24 ~21:43 EDT (memory topic isla-api-and-library-rulings-2026-10-24.md). "Measured"
below means a registered or recorded result with its source; "estimated" or "inferred" means reasoning or arithmetic
done in this lane.

## 1. Survey: what in ISLA's model code is application-specific

### 1.1 The pieces

| piece (5b70c8686) | what it computes | used by | what the function depends on |
|---|---|---|---|
| `ISLA.__init__(..., reynolds=None)`, model.py:743; stored in `_args` (854) | `{"mean": m, "std": s}`: the standardization constants of log10(U L / nu). With it, one more global scalar: `n_global_scalars = len(global_scalar_names) + (reynolds is not None)` (873), so it widens every condition's seed layer by one column | conf/model/isla_volume_airfrans.yaml and isla_volume_aerozoo.yaml only, both with `reynolds: null`, filled by `output_normalize.model_dataset_constants` (train.py:1139, infer.py:544) from the dataset file's `isla_constants.reynolds` | m and s are per-split constants fitted on the split's training cases. AirfRANS (whole boundary, AF-WHOLE refit): 9.065 / 0.1338 (full; also the _2d, _area, _posed files), 9.067 / 0.1321 (aoa), 9.084 / 0.06181 (reynolds), 6.0 / 1.0 (reynolds_lre, AF-EXTRAP's fixed scale), 9.066 / 0.1347 (s504). AeroZoo: 5.909 / 1.415 (full), 5.062 / 0.8985 (reynolds), 5.846-5.969 / 1.38-1.41 (blobs, nonbox, bank_zero_to_cad, profile_crossflow, profile_dual_inlet) |
| `_init_scales(out_extras, b_extras, g_extras, reynolds, output_stats)`, model.py:1057-1151 | Checks the ISLA-local keys of every field, builds `_output_units`, `_output_scale`, `_global_units`, `_global_scale`, `_boundary_units`, `_boundary_scale`, `_boundary_sources`, `_global_sources`, `_roles`; enforces "viscosity role iff reynolds", "reynolds needs a velocity_scale source (Re = U L / nu)", "units name rho needs the density role"; sets `_scaled` and `_fp32_head` (the head in fp32 whenever any scale, `scale` or `output_stats` is declared) | the same two configs | the head's precision (`_fp32_head`) depends on whether anything is declared |
| `_case_scales`, model.py:1511-1556 | Per case, float64, before the token draw: L = the gauge (below, computed in the meshes' dtype, then `.double()`); U = measure-weighted RMS speed pooled over the boundary `velocity_scale` sources (AeroZoo: `inlet.bc_velocity`), or a global source's magnitude (AirfRANS: `U_inf`); g = the sources' measure-weighted mean direction (boundary sources only; appended as a global direction, K = 1 on AeroZoo); rho and nu from the `density` and `viscosity` roles; `z = (log10(U * L / nu) - m) / s` | the same two configs | z is the only input through which ISLA sees a case's Reynolds number (section 2.4) |
| `_global_fields`, model.py:1468-1497 | Packs the rank-1 globals into unit directions and the rank-0 globals (role fields excluded) into scalars; divides the rank-0 globals that declare `units`/`scale`; appends g and then z as the last scalar | every config (the z and g branches only with the scales) | |
| `_nondimensional`, model.py:1566 | Divides each declared boundary field by `prod scale^exp` of its units and by its constant `scale` | AeroZoo (inlet and outlet values); AirfRANS declares none | |
| `_redimensionalize`, model.py:1582 | Multiplies each output by its `scale` and its units' factor, then standardizes by `output_stats` | AeroZoo (all five outputs), AirfRANS (nut only; velocity and Cp are dimensionless targets) | the loss sees the dataset's standardized physical targets |
| scales.py: `BASE_SCALES = ("L", "U", "rho")`, `ROLES = {"density": {"rho": 1}, "viscosity": {"U": 1, "L": 1}}`, `SCALE_KEYS = ("units", "velocity_scale", "role", "scale")`, `velocity_moments` | The vocabulary of the scales: a velocity U, a density rho, the roles density and viscosity, the flag `velocity_scale` | the same two configs | names only; the arithmetic is generic (an RMS magnitude, a value, products of powers) |
| model.py docstrings: module *Scales* section (lines 122-146: "a velocity U and a density rho", "({"U": 2, "rho": 1} for a gauge pressure in Pa)", "z = (log10(U L / nu) - m) / s"), the `reynolds` and `output_stats` argument docs (665-681), the version note (709-710), comments at 863-865 and 2300 ("Pa pressure"); line 28 lists "a direction, a diffusivity, a Reynolds number" as examples of global inputs | documentation | | |
| `isla_constants` (dataset files, not model code): `reynolds`, `output_scale`, `input_scale` | Per-split constants fitted on training cases by `examples/cfd/isla/results/aerozoo_dataset/isla_constants_fit.py` with ISLA's own `_gauge` and `_case_scales`; injected by `model_dataset_constants` | 7 AeroZoo and 12 AirfRANS dataset files | `output_scale`/`input_scale` are the `scale` constants c (order-one sizes of the dimensionless fields) |

Not application-specific, though fluids motivated them: the length unit L (the measure-weighted RMS pairwise distance of
the whole boundary, `rms_pairwise_distance` + `cell_second_moments` in geometry.py; every config uses it for the
geometry), `units`, `scale` and `output_stats` (generic dimensional analysis and standardization), the derived direction
g (the mean direction of a declared rank-1 boundary field). The geometry files' `rho` is the token-to-anchor direction,
not a density.

No other ISLA config uses the scales: DrivAerML Surface/Volume, HiLift, SHIFT-SUV (`U_inf_dir` rank-1 only), DeepJEB++
(`load_dir`, `is_force`, `is_torque`), Tensile2d and the lug (recipe-standardized rank-0 globals such as `pressure_std`,
`P_std`, made by `SplitGlobalVector`). Those two structural configs already pass recipe-computed, standardized rank-0
globals as ordinary inputs: the pattern the relocation follows.

Tests that exercise the pieces: test_isla_scales.py (58 hits: similarity, hand values, refusals, `reynolds`),
test_isla_compile.py, test_isla_conditions.py (the `scaled` parameter), test_isla.py (rotation with scales); recipe:
test_aerozoo_configs.py, test_airfrans_configs.py (isla_constants), test_aerozoo_transforms.py (`ComputeLogReynolds`).

### 1.2 The recipe already has half of the relocation

`src/domain_transforms.py:1545` `ComputeLogReynolds(speed_field, length_field, viscosity_field, mean, std, output_field,
standardized_field)` writes log10(U L / nu) and its standardization into `global_data` (domain-aware). It was used by the
label-era AeroZoo chain (reading `reference.*`, which the 2026-10-19 correction removed; no dataset file uses it now). It
computes in float32 (`.float()`), and nothing in the recipe computes ISLA's U and L from the boundary.

## 2. Evidence: does a Reynolds input help?

1. **ISLA-SCALES (0.9.0's opt-in scales).** REF-090 R1 (book 14-program-status.qmd, REF-090 row): 0.9.0 ÷ A2 on AirfRANS
   G_d = 0.683, the largest step of any dataset; ATTRIB-090 §2 (scratch/isla-attrib090-2026-10-21/RESULTS.md): CONT
   (0.6.0) → 0.9.0 is 0.778 / 0.739 / 0.709 (p / v / nut). Measured, 2 seeds.
2. **NOSC-STD and STDU (ATTRIB-090 §2c, §2d).** No scales, inlet speed standardized: ÷ 0.9.0 1.003 / 1.029 / 1.043 (φ
   0.01 / 0.10 / 0.12, no measurable effect). Scales kept, standardized speed in place of z: 1.008 / 1.015 / 0.995 (no
   measurable effect). RAWU (scales, raw speed) and NOSC (no scales, raw speed) both fall back to CONT's level; NUT0 (nut
   without units) has no effect. So on AirfRANS the whole gain is the standardization of the one global scalar; log Re
   and standardized speed are interchangeable, and the scales' physics carries nothing measurable. Reason (ATTRIB): nu is
   constant and L nearly so (305.3 chords on every case once the far field is read), so log10(U L / nu) is a monotone
   reparametrization of U. Measured, 2 seeds.
2b. **DTT-NOSC-STD (DTT-KNOBS R5, scratch/isla-dtt-knobs-2026-10-23/RESULTS.md §R5, read 22:58 EDT 10-24; second read
   RESULTS_R5_REVIEW.md "confirmed with corrections").** ISLA-DTT with the global input standardized and no scales,
   against B (ISLA-DTT with the scales), AirfRANS, 2 seeds each: 1.208 / 1.203 / 1.177 (2 SE 0.101 / 0.173 / 0.149),
   registered reading "standardization leaves it (the scales carry it)" on every field; velocity flagged (it rests on
   seed 43; the geometric form reads unresolved). So the interior reference form, ISLA-DTT, uses the scales: about 20 %
   worse without them, where interacting ISLA lost nothing (2c). The arm removes the Re input, nut's units and the fp32
   head together, so which part carries it is not split. Measured, 2 seeds.
3. **AF-EXTRAP and LRE (scratch/isla-af-extrap-2026-10-21/RESULTS.md §3f, §6; LRE.md).** On AirfRANS's reynolds split
   ISLA extrapolated worse than both baselines on pressure; clamping z to its training range removed the below-band
   pressure gap (clamped ÷ native 0.469). Retrained with the fixed scale z = log10(Re/1e6) (LRE), below-band pressure fell
   to 0.530 / 0.576 of the fitted-scale model per seed ("helps"), test pressure 0.0389 → 0.0246 (= GeoTransolver's), in-band
   fit 0.91-0.94. A linear train-standardized Re (LIN) helped less (0.620 / 0.632). Measured, 2 seeds. So how the scalar
   is scaled decides extrapolation; the scalar itself is not the problem.
4. **LRE-AZ (scratch/isla-lre-az-2026-10-23/RESULTS.md).** On AeroZoo's reynolds split the fixed scale **harms velocity**
   (fixed ÷ fitted 1.126, D = 0.118 > ln 1.10 and > 2 SE 0.046; both seeds 1.139 / 1.113; worst 1-2 decades above the band,
   up to 1.28), pressure no change (1.067), nut helps (0.833). Inside ISLA's training band the two scales agree to about
   1 %. Registered consequence: the fixed scale is not adopted without Peter, and the per-split fitted constants stay.
   Measured, 2 seeds. The fitted-vs-fixed choice is Peter's and is independent of where z is computed.
5. **No arm has removed z on AeroZoo, and none has removed every global scalar on AirfRANS.** Inferred (not measured):
   with the scales on, ISLA divides every boundary value by its units (U, L, rho), positions by L, consumes nu, and makes
   the measures scale-free, so a similarity of the problem (lengths × s, velocities × c, nu × s c) leaves every internal
   state unchanged (test_isla_scales.py::test_similarity_leaves_the_internals_and_scales_the_outputs). A change of nu alone
   at fixed geometry and boundary values reaches the network only through z. Without z, ISLA cannot represent any Reynolds
   dependence of its dimensionless outputs; on AeroZoo's full split ISLA's log10 Re has mean 5.909 and std 1.415 over the
   training cases (isla_constants), so the cases span several decades, and the equations carry a transition closure. On
   AirfRANS (Re 2-6 × 10^6) the loss would likely be smaller, but nut / (U L) depends on Re. The baselines differ: they read raw coordinates (metres),
   bc_velocity in m/s and log10 nu, so the Reynolds number is implicit in their inputs.
6. **AeroZoo's task specification (/home/psharpe/gh/aerozoo/docs/book/06-splits.qmd #sec-allowed, at 96bd2b62).** "A
   model may use anything computed from a sample's inputs and from the training part of its split, and nothing computed
   from a test sample's solution"; allowed from the sample: "an area-weighted inlet velocity"; and "A model that wants a
   free-stream velocity, a length scale or a Reynolds number computes it from the inputs as part of the model, in its
   architecture or its own preprocessing; a benchmark that compares models does not compute it for them." Its reference
   `derive_globals` computes an area-weighted mean inlet speed, the RMS distance between two points of the boundaries (the
   quantity of ISLA's L) and their Reynolds number. It also says a model that divides coordinates by such a length "must
   also take the length as an input". The `reference.*` labels (L_ref, U_ref, a Re built on them) are forbidden.
7. **Peter's ruling of 2026-10-19 (memory isla-program-context.md):** "GeoTransolver (or other non-ISLA models) should NOT
   be able to use the area-weighted RMS distance between boundary points. That's an ISLA-specific thing ... If a model
   wants normalized coordinates, it needs to bring it to the table. GeoTransolver does not. ISLA does."

**Reading.** On AirfRANS, for interacting ISLA, the Reynolds input is one standardized scalar that a standardized speed
replaces at no measured cost; for ISLA-DTT the scales together (Re input, units, fp32 head) are worth about 20 %; the
extrapolation depends on how z is scaled. On AeroZoo it is ISLA's only channel for the Reynolds number, which its
scale-free design strips from every other input, so removing it is expected to harm (inferred, not measured). Keep it,
but as data (the recipe's), not as a model argument.

## 3. Options

| | (i) relocate exactly | (ii) remove the feature | (iii) generic recipe transform | (iv) generic group in the model |
|---|---|---|---|---|
| model.py | loses `reynolds`, the viscosity role, z in `_case_scales`/`_global_fields`, its `n_global_scalars` term | same as (i) | same as (i) | `reynolds` and roles go; a rank-0 global with units enters as its dimensionless value (already so today), plus a new generic `log: true` key and fitted input standardization (`input_stats`, the mirror of `output_stats`) |
| recipe | computes the same z (ISLA's U and L, float64) and passes it as an ordinary rank-0 global `re_z: {rank: 0}` | nothing (nu dropped or kept as a plain global) | (i) built from two reusable transforms: `ComputeBoundaryScales` (L and a source's RMS speed from the boundary) + `ComputeLogReynolds` in float64; constants from `isla_constants.reynolds` | nu declared `{rank: 0, units: {U: 1, L: 1}, log: true}` with constants |
| function of trained models | unchanged, bitwise (section 4: child check) | changed: Re-blind on AirfRANS and AeroZoo | unchanged (it is (i)) | changed at float32 rounding: z' = -z, log10(nu/(UL)) rounds differently from log10(UL/nu); a checkpoint would need its z column negated |
| retrain check | none | needed, and expected to fail on AeroZoo | none | none needed for function, but a new model knob to test |
| cost | small | no-harm check about 950-1,300 GPU-h (estimated: AirfRANS ISLA-DTT 2 seeds × ~66 GPU-h, AeroZoo full and reynolds splits 2 seeds × 200-293 GPU-h each, plus scoring) | small | medium |

Notes per option:
- (ii) keeping nu as a plain global does not restore the information: ISLA never sees U or L in physical units, so nu
  alone cannot make a Reynolds number. On AirfRANS NOSC-STD (no scales at all, standardized speed) is measured no-harm,
  but that drops the scales too; on AeroZoo the same move needs L as an input (AeroZoo's spec says so), which is ISLA's
  length supplied by the recipe: (i) in another form.
- (iv) keeps similarity enforced by the model and computes the group from the model's own L, but it puts a log and fitted
  constants back into the constructor, and the model still forms a Reynolds number in all but name. Peter's words point to
  the recipe.
- Since R5, any design must keep the scales' function exactly (ISLA-DTT loses about 20 % without them). The recipe
  could also compute U and rho and pass them as rank-0 globals that `define` a scale (step 2 allows it), but U pooled
  over the inlet cells and its direction g are bitwise only when computed where they are today (g is normalized in
  float64 before the cast; a recipe-supplied direction would be renormalized in float32 by `_global_fields`), and the
  model would still need L. So keep the source computations in the model under generic names.
- Moving the output scaling (units, `scale`, `output_stats`) to the recipe as well: not recommended. It is dimensional
  analysis, not fluid physics (a stress in units of a modulus, a displacement in L); it needs L per case, which only the
  model has without duplication; the fp32-head decision and the output standardization live in the head; and the model
  alone would then return dimensionless outputs. Only its vocabulary is fluid-specific, so rename it (section 5, step 2).

**Baselines.** No recipe-level Re for them. Peter's 2026-10-19 ruling forbids giving non-ISLA models ISLA's RMS length,
and "Baselines get no ISLA-specific features" (LANE_RULES). AeroZoo's spec leaves derived globals to each model's own
preprocessing ("a benchmark that compares models does not compute it for them"). The baselines also need it less: their
dimensional inputs (raw coordinates, bc_velocity in m/s, log10 nu on AeroZoo; standardized speed on AirfRANS) already
determine Re. A "baselines + Re" steel-man would be a separate, registered experiment, not part of this change.

**AeroZoo's input rule.** The relocated z reads only the posed problem: the boundary meshes (geometry and measures), the
inlet's `prescribed.dirichlet.velocity` and `physics.kinematic_viscosity`, plus training-split constants. It never reads
`reference.*` or any `solution/*`. It must run in ISLA's own model config (`dataset_append_transforms`), never in a dataset
file's chain shared by every model (the 2026-10-19 correction: "nothing here ... derives a per-case scale").

## 4. Can (i) reproduce today's function bitwise?

Argument (from the code): the model computes z in float64 and rounds it once to the meshes' dtype
(`scales["z"].to(like.dtype)`); a rank-0 global leaf is rounded once the same way (`_global_rows`: `.to(like.dtype)`).
If the recipe computes the same float64 z, with the same operations on the same tensors on the same device, the seed
input is bitwise equal, the parameter shapes are equal (one scalar either way, last column), and everything downstream is
identical. Conditions: (a) L from `rms_pairwise_distance` on the declared boundaries in sorted order, in the meshes' dtype
(float32), on the same device, then `.double()` (the datapipe runs its transforms on the GPU: datasets.py `device`
default "auto"); (b) U from `velocity_moments` over the same sources and `cell_measures`; (c) nu and a global source rounded
to the meshes' dtype before `.double()`, as `_global_rows` does; (d) `log10(U * L / nu)` in that order; (e) the transform runs
last in the chain, on the meshes the model receives. The Reynolds path is byte-identical between code_r090 (92d9d7800,
the standing checkpoints' code) and 5b70c8686 (`_case_scales`, `_global_fields`, `_gauge`, `_global_rows`, `_init_scales`,
`rms_pairwise_distance`, `cell_second_moments`: md5 of each function equal).

Measured (`bitwise/RESULTS_BITWISE.md`, child REYNOLDS-OUT-BITWISE; LHR GB300 short job 376561, 0.245 GPU-h; CPU
debug jobs 376218/376219; code_r090, recipe_r090, float32 eager, eval mode): model A = checkpoint as trained (built as
infer.py builds it); model B = `type(A)(**edited args)` with `reynolds=None`, the viscosity field removed and
`re_z: {rank: 0}` added, `load_state_dict(A.state_dict(), strict=True)`; z from `proto/recipe_z.py`.

| run | z cases: float64 and float32 bitwise | forward cases: network input (g_scalars) bitwise | outputs bitwise | A-vs-A floor |
|---|---|---|---|---|
| r090eag_af_isla_seed42 | 64 / 64 | 4 / 4 | 4 / 4 | bitwise |
| r090eag_af_dtt_seed42 | 64 / 64 | 4 / 4 | 0 / 4, max 7.4e-6 | not bitwise, 2.3e-6 (the DTT read is not run-to-run deterministic on the GPU) |
| azc_reynolds_isla_seed42 | 32 / 32 | 2 / 2 | 1 / 2 (case 0: 2.6e-6) | bitwise in one repeat |

On CPU (one case per run) A vs B and A vs A are bitwise for every output. So the relocated z is bitwise the in-model z
on all 160 cases, and the network's input is bitwise equal in every forward; the output gaps are GPU run-to-run
nondeterminism of the same size as A's own repeat. Repeat job 376855 (0.161 GPU-h; A,B,A,B,A,B with block hooks on
cases 0-1 of each run): AirfRANS ISLA 9/9 A-B pairs bitwise on both cases at every block; AirfRANS DTT A-B gaps
1.8-2.6e-6 inside the A-A and B-B spread 1.9-2.4e-6 (departure at block 1, after the tree read); AeroZoo case 1 all
bitwise; AeroZoo case 0: A1 = A2 = A3 and B3 = A bitwise at all 24 block tensors and outputs, while B1 and B2 differ
from A and from each other (up to 8.8e-6) from block 0's input on, i.e. GPU nondeterminism before block 0, not z.
Side finding (not Reynolds): ISLA 0.9.0's eager AeroZoo forward does not always repeat before block 0 (block-0 input
differs by O(1), outputs by about 1e-4 relative: the child's reading is token order, untested). The checkpoints' AirfRANS constants are {6.205, 0.1347} (airfoil-only L,
REF-090 era); 5b70c8686's dataset files carry {9.065, 0.1338} because AF-WHOLE refit L with the far field. Both are right
for their code.

## 5. Recommendation and code change in outline (ISLA 0.13.0, with ISLA-API's psharpe/isla-api-013)

**Recommend (i), built as (iii), plus step 2**: the recipe computes the standardized log Reynolds number with two generic
transforms in ISLA's model configs; the model keeps its scales' function exactly (R5: ISLA-DTT needs it) under a generic
vocabulary, so model.py names no Reynolds number, viscosity, density or velocity. Both steps change no arithmetic.

Step 1, the relocation (function unchanged):
- physicsnemo/experimental/models/isla/model.py: drop the `reynolds` argument, `self.reynolds`, its `_args` entry and
  docstring; `_init_scales(out_extras, b_extras, g_extras, output_stats)` without the viscosity/reynolds checks;
  `n_global_scalars = len(self.global_scalar_names)`; `_case_scales` without nu and z; `_global_fields` without the z
  append; the module *Scales* docstring and the version note (0.13.0, shared with ISLA-API).
- physicsnemo/experimental/models/isla/scales.py: `ROLES` loses `viscosity`.
- Recipe src/domain_transforms.py: new registered `ComputeBoundaryScales(boundaries, length_field, speed_field,
  sources=None, global_source=None)`: L = `rms_pairwise_distance` with `cell_second_moments` over the named boundaries in
  sorted order, in the meshes' dtype, stored as float64; U = the RMS speed of the sources (`velocity_moments`) or a global
  vector's norm, float64 (globals rounded to the meshes' dtype first). Import the two geometry helpers from ISLA, or from
  physicsnemo.mesh if lane LIB-MOVE moves them there (a boundary's RMS pairwise distance is a generic statistic;
  AeroZoo's `derive_globals` computes it). `ComputeLogReynolds`: compute in float64 (today `.float()`), the viscosity
  rounded to the meshes' dtype first.
- Recipe conf/model/isla_volume_airfrans.yaml, isla_volume_aerozoo.yaml: append the two transforms (AirfRANS: after
  SetGlobalField nu; sources: global `U_inf`; AeroZoo: sources `[[inlet, bc_velocity]]`; boundaries = the
  forward_kwargs names); `global_schema`: drop the viscosity field, add `re_z: {rank: 0}`; drop `reynolds: null`.
- Recipe src/datasets.py `_apply_dataset_transform_appends` (it holds both configs): fill a `ComputeLogReynolds` entry's
  `mean: null, std: null` from the dataset file's `isla_constants.reynolds`, refusing a file without it (the check
  `model_dataset_constants` makes today); src/output_normalize.py `model_dataset_constants` stops injecting `reynolds`.
  Dataset files unchanged (the per-split fitted constants stay, as LRE-AZ registered; reynolds_lre keeps {6.0, 1.0}).
- Baseline configs unchanged.

Step 2, generic scale vocabulary (function unchanged; renames only): scales.py `BASE_SCALES` → the built-in `L` plus any
scale a field declares with a new key `defines: <name>` (a rank-1 field: its measure-weighted RMS magnitude, and a boundary
field's mean direction g as today; a rank-0 global: its value, consumed); `velocity_scale` and `role` go; `ROLES` goes;
`velocity_moments` → `rms_moments`; `units` may name `L` and the declared names. Configs: AeroZoo `bc_velocity: {rank: 1,
units: {U: 1}, defines: U}`, `physics.density: {rank: 0, defines: rho}`; AirfRANS `U_inf: {rank: 1, defines: U}`. Keep
today's limit of one rank-1 defined scale (nothing needs more).

Tests:
- test/experimental/models/isla/test_isla_scales.py: remove the `reynolds` and viscosity-role cases; the similarity test
  feeds `re_z` unchanged under the similarity (it is dimensionless); hand values and refusals for `defines` (step 2: on a
  rank-2 field, `defines: L`, units naming an undeclared scale, two rank-1 defined scales); "declaring nothing is the
  release model" stays. Update test_isla_compile.py, test_isla_conditions.py, test_isla.py configs. Whole ISLA directory on
  CPU, and on one GB300 short job with every CUDA test (LANE_RULES).
- Recipe tests/test_aerozoo_transforms.py: `ComputeBoundaryScales` hand values (unit square, segment polyline), its L and U
  bitwise equal to ISLA's `_gauge` and `_case_scales` on a synthetic case; invariance to every `global_data["reference"]`
  leaf (reads no label); `ComputeLogReynolds` in float64. test_aerozoo_configs.py, test_airfrans_configs.py: the chain writes
  `re_z`, the constants come from `isla_constants.reynolds`, a file without them is refused, the model builds without
  `reynolds`.
- One-time integration check on the 0.13.0 branch (part of ISLA-API's GB300 job): for a few cases per dataset, `re_z` from
  the real datapipe equals the old in-model z (`proto/recipe_z.py` logic on the same meshes) bitwise.

Retrain: none for the relocation if section 4 holds (the inputs to the network are bitwise those of today, in training
and inference alike). 0.13.0's other changes carry their own checks.

## 6. For Peter or the supervisor to decide

1. Keep the Reynolds feature, computed by ISLA's recipe config (recommended), or drop it (option ii, expected to harm
   AeroZoo; its no-harm check would cost about 950-1,300 GPU-h, estimated).
2. Step 2 (generic `defines` in place of U, rho, `velocity_scale`, roles): part of the recommendation, in the same 0.13.0
   bump; the key's name (`defines`) is an API-naming judgment Peter may want to see.
5. Side finding for the supervisor: a bitwise A/B test on AeroZoo needs repeats (pre-block GPU nondeterminism of 0.9.0's
   eager forward on case 0); worth locating with hooks on `sample_boundary_cells` / `_condition_tokens`.
3. Still open and unchanged by this design: fitted per-split vs fixed log10(Re/1e6) standardization (AF-EXTRAP helps on
   AirfRANS, LRE-AZ harms AeroZoo velocity; registered as Peter's choice).
4. Memory: record that the Reynolds feature is ISLA's recipe-side preprocessing (in its model config, never in a dataset
   file's chain or a baseline config), computed only from posed-problem inputs and training-split constants.
