# ISLA brings its own scales (AZ-COMPLIANT v2): design support (read-only, 2026-10-19 13:57–14:20 EDT)

How to read the references. `M:` is `physicsnemo/experimental/models/isla/model.py` at bc-names HEAD bb07cf413 (read with `git show`). `R:` is `examples/cfd/external_aerodynamics/unified_external_aero_recipe/`. The uncommitted diff (model.py +24, test_isla_release.py +48) adds only a `load_state_dict` pre-hook that refuses state dicts from before 0.7.0, plus its test. Nothing in it touches scales.

## 1. The API now, and where units and source flags can live
- **Constructor** (M:540-555), keyword-only: `output_schema`, `boundary_schemas` (Mapping name→schema, sorted, M:590-616), `query_schema`, `global_schema` (ranks 0 and 1, M:619), `interior`, `query_independent`, `boundary_tokens`, `hidden`, `n_layers`, `n_slices`, `compile_blocks`, `dim`.
  - There is no separate "prediction schema". The outputs are `output_schema`, read through one linear `self.head` (M:787) and the equivariant basis (heads.py).
- **Field grammar.** `_declare` (inputs.py:36-73) calls `FieldSchema.parse` (mesh/fields.py:334). `RankSpec` **refuses unknown keys**: `_ATTRIBUTES = {rank, symmetric, parity}` (fields.py:69, 214-226). So `units:` cannot simply be added to a field's mapping.
- **Recommended: ISLA-local inline keys.** `_declare` pops `units` and `velocity_scale` (and optionally `log`) from each field's mapping before `FieldSchema.parse`, and keeps them in `self.field_units[kind][name]`. Source flags are then attached to a field of a condition's own declaration, so no condition names appear in code.
  - Leave the shared `RankSpec` alone: changing it would affect every model, its hash and its `__reduce__`.
  - One catch: `_args` stores the parsed schemas (M:655-657), which would drop the units. It must store the merged declaration instead.
- **Current `isla_volume_aerozoo.yaml`** (R:conf/model, :79-118):
  - Four conditions, each declaring only the fields it prescribes:
    - `inlet`: `bc_velocity` (rank 1), `bc_tke`, `bc_omega`, `bc_intermittency`, `bc_retheta`;
    - `outlet`: `bc_pressure`, `bc_velocity_reverse` (rank 1), and `bc_{tke,omega,intermittency,retheta}_reverse`;
    - `no_slip: {}` and `slip: {}`.
  - Globals: `U_inf_dir` (rank 1), `log_re_std`, and `inlet_{tke,omega,intermittency,retheta}_std`.
  - Every value is nondimensionalized by the pipeline with `reference.velocity` and L_ref (R:datasets/aerozoo_volume.yaml:67-113). The globals are built at :79-86 and :121-171.
  - Targets are standardized in the datapipe (`NormalizeMeshFields`, :223-230). The velocity **mean is a nonzero lab-frame vector**.

## 2. Insertion points in `forward` (M:1220-1347)
- **(a) Scales, at full resolution and before the draw.** Insert between `global_td` (M:1290) and `_global_fields` (M:1291). A new `_case_scales(names, meshes, measures, global_td)` returns per case:
  - U = sqrt(Σa|v|²/Σa), pooled over every field flagged `velocity_scale` (boundary cells, weighted by area). A flagged rank-1 global counts as one vector with U = |v|.
  - g = Σav/|Σav|, appended to `g_unit` after the declared global vectors. K counts it; `n_frame = 1+K` (M:691) is fixed at construction.
  - The full-mesh gauge L (see (c)).
  - ρ and ν, read from flagged globals.
  - z = (log10(UL/ν) − mean)/std, appended to `g_scalars`. This makes the seed one column wider in `_BoundaryCondition` and in the query seed.
  - Everything in float64, as per-case scalars.
  - Because the draw happens at M:1330-1338, U, g and L **do not depend on the draw**. Refuse an absent source or a zero-area source.
- **(b) Nondimensionalization.**
  - Boundary fields: in `_condition_tokens` (M:865-903), multiply each leaf of `fields` by Π scale^(−exp) before `inputs.features` (M:882). The norms and dot products (inputs.py:218-224) and the block vectors carry magnitudes, so they must be scaled. The frame cosines do not need it.
  - Globals: scale the declared rank-0 globals the same way inside `_global_fields` (M:1082-1094). ν and ρ are consumed and **not** appended raw.
- **(c) Gauge.** Today it is computed in `_forward_tokens` (M:1495), on the **drawn** tokens in the interior task and on every cell in the surface task.
  - It can be the full mesh *exactly*: `rms_pairwise_distance(cat(centroids), cat(measures))` before the draw. This is the quantity the HT sample estimates, and it equals today's surface-task value.
  - Pass it as an optional `gauge=` argument to `_forward_tokens`. Its uses there are r (M:1511, 1562), the query wall distance (M:1530) and the rank-0 query fields (M:1539).
  - Tie the full-mesh gauge to the option, so that models without it stay bitwise identical. Making it the default could be a separate commit, because it changes existing interior models by the sampling noise of L.
- **(d) Outputs.** The head emits the coefficients at M:1592 and assembles the rank ≥ 1 outputs at M:1655-1661. Re-dimensionalize after `_forward_tokens` returns (M:1341): `y = ŷ·Π scale^exp`, then `(y − mean)/std` with constant `output_stats`.
  - Do this in fp32 with autocast off, and run `self.head` in fp32 as well (see §6).
- **The recipe has no hook for this.** Targets are normalized in the datapipe, and the loss compares the model's output with them directly. `output_normalize.py` only unpacks fields (:17-30, :156-197). `NormalizeMeshFields` has `__call__(mesh)` and `inverse_td` only (physicsnemo/datapipes/transforms/mesh/transforms.py:756, 815). `infer.py:213-228` undoes the normalizer and then the nondimensionalization; a stage whose transform is absent is skipped.
  - Smallest bridge: the model takes `output_stats` in `normalizer.stats` format (transforms.py:856: {name: {type, mean, std}}). `train.py` already has `normalizer` (:857) before `instantiate(cfg.model)` (:912), so it can inject the stats or assert they are equal (about 10 lines).
- **`_forward_tensors`** (M:1387-1453), the tensor API with one condition and no meshes, must compute the scales from its tensors, or refuse when scales are declared.

## 3. Per-dataset config (opt-in: nothing declared means today's model exactly)
- **AeroZoo.** Keep the inputs dimensional: drop `NonDimensionalizeByMetadata`, `ComputeLogReynolds`, `ComputeFreestreamDirection` and the `BoundaryFieldToGlobal` turbulence globals. The inlet fields already carry those turbulence values.
  ```yaml
  boundary_schemas:
    inlet:  {bc_velocity: {rank: 1, units: {U: 1}, velocity_scale: true}, bc_tke: {rank: 0, units: {U: 2}},
             bc_omega: {rank: 0, units: {U: 1, L: -1}}, bc_intermittency: {rank: 0}, bc_retheta: {rank: 0}}
    outlet: {bc_pressure: {rank: 0, units: {U: 2, rho: 1}}, bc_velocity_reverse: {rank: 1, units: {U: 1}},
             bc_tke_reverse: {rank: 0, units: {U: 2}}, bc_omega_reverse: {rank: 0, units: {U: 1, L: -1}}, ...}
  global_schema:                                   # no U_inf_dir: g is derived (K = 1)
    physics.density: {rank: 0, role: density}      # consumed for the units only
    physics.kinematic_viscosity: {rank: 0, units: {U: 1, L: 1}, role: viscosity}
  reynolds: {mean: <scan>, std: <scan>}            # L is the box-scale gauge: not 5.5/1.5 (those are for L_ref)
  output_schema: {velocity: {rank: 1, units: {U: 1}}, pressure: {rank: 0, units: {U: 2, rho: 1}},
                  nut: {rank: 0, units: {U: 1, L: 1}}, tke: {rank: 0, units: {U: 2}}, omega: {rank: 0, units: {U: 1, L: -1}}}
  output_stats: <= NormalizeMeshFields stats of the physical targets (new scan; vector means 0)>
  ```
- **AirfRANS** (`isla_volume_airfrans.yaml`).
  - **Sources must be able to be globals.** The pipeline gives `global_data.U_inf` as a vector (airfrans_volume_2d.yaml:36). The model currently declares `U_inf_dir` plus a raw `U_inf_mag`, which has units of m/s and already breaks the similarity.
  - Proposal: `global_schema: {U_inf: {rank: 1, units: {U: 1}, velocity_scale: true}, nu: {rank: 0, units: {U: 1, L: 1}, role: viscosity}}`. The global itself is g (K stays 1), and U = |U_inf|.
  - The dataset has no ν: add `SetGlobalField nu: 1.56e-5`, a physical constant.
  - Outputs: `velocity {U: 1}`, kinematic `pressure {U: 2}`, `nut {U: 1, L: 1}`.
  - The queries' `sdf` is already divided by the gauge.
- **DrivAerML and SHIFT** (`isla_surface.yaml`). The freestream is constant and the targets are Cp-like. Declare **nothing**: a constant rescale is absorbed by the target stats, and Re is constant, so z would be useless. Keep `U_inf_dir`.
  - Optional, if the program wants physical targets: `U_inf: {rank: 1, units: {U: 1}, velocity_scale: true}`, `rho_inf` with role density, `pressure`/`wss` with units `{U: 2, rho: 1}`, and no `reynolds`.
- **HiLift** (`isla_surface_highlift.yaml`). Compressible, with temperature and density outputs, so it needs a T and a ρ scale and a Mach input. **Declare nothing for now.** The grammar should refuse base scales other than L, U and ρ until they are needed.
- **Structural.** No velocity, so **declare nothing**:
  - `isla_tensile2d.yaml`: plastic; inputs `loaded.bc_pressure` plus six standardized globals; outputs q (dimensionless), displacement and stress (2×2 symmetric).
  - `isla_lug.yaml`: outputs von_mises (MPa); inputs `bc_pressure` in MPa plus four standardized design globals.
  - `isla_surface_deepjebpp.yaml`: outputs von_mises and displacement; inputs `bc_load` (rank 1) and `load_dir`.
  - Lug and DeepJEB++ are linear elastic, so σ ∝ load *exactly*. A generic named scale (`scale: S` from `loaded.bc_pressure`) would make that exact, but `pressure_std` duplicates P and the standardized lengths duplicate the geometry. Name the flag generically (`scale_source: U`) so S can be added later.
  - Separate issue: DeepJEB++'s torsion `load_dir` is a moment axis, i.e. a pseudovector, declared as a true vector.

## 4. ISLA-DTT (query_independent=True)
- g reaches the tree read only through the `g_unit` argument: `_tree_read_pairs(..., g_unit)` (M:1570, 1362-1385) → `tree_pair_invariants` uses `rho·g` and `n_j·g` (tree_read.py:454). The in-model g flows through unchanged, and the kernel widths depend only on K (`kernel_geo_width`, M:826).
- `canonical_frame` (tree_read.py:143-171) uses points and measures only. Its tie-break uses the lab axis (:167-168), which is not equivariant. An in-model g offers an equivariant tie-break (sign by axis·g). That would be an improvement, not something required.
- Nothing breaks if g is computed inside forward: it is data-derived, carries no gradient, and has a static shape.
- The block vectors read by the tree read (`boundary_vecs`) become dimensionless, which is what we want.
- The forest is still built on the drawn tokens, so only g, U and L are made draw-independent.

## 5. Tests (`test/experimental/models/isla/`)
- **Existing tests to extend with a scaled model:**
  - `test_isla_conditions.py::test_every_condition_s_fields_are_covariant` (:121; rotations, proper and improper; g rotates with cell_data);
  - `test_isla.py::test_rotation_equivariance` (:92), `test_measure_scale_invariance` (:112) and `test_token_permutation_equivariance` (:119);
  - `test_isla_global_inputs.py::test_zero_global_vectors` and `test_two_global_vectors` (:81, :99; K counting with a derived g);
  - `test_isla_inputs.py::test_rank0_query_inputs_are_scaled_by_the_gauge` (:231; the expected value becomes the full-mesh L);
  - `test_isla_sampling.py` stays as it is.
- **Existing tests to scope to unscaled models, or to rewrite with ν scaled:** `test_geometric_scale_equivariance` (:346) and `test_global_vector_magnitude_invariance` (:105). With scales on, a geometry scale with ν fixed changes Re, as it should.
- **New tests** (a `test_isla_scales.py` file, double precision, atol 1e-10):
  1. Similarity: (x, v, ν, ρ, k, ω, p) → (sx, cv, scν, rρ, c²k, (c/s)ω, rc²p) leaves the seeds, r, z and g identical, and scales the physical outputs by Π scale^exp.
  2. Label independence: the same physics with the flag on a condition named `a` or on one named `b` gives equal outputs.
  3. Draw independence: U, g and L are equal for two training starts (`torch.rand`) and under `_meshes.refine` (:127) of the source condition.
  4. Opt-in identity: with no scales declared the outputs are bitwise equal to the stored release outputs (test_isla_release.py, data/*.pth).
  5. Unit checks: U, g and z against hand-computed values; refusal of an absent or zero source, and of mixed boundary and global sources.
  6. `output_stats` round trip against the `NormalizeMeshFields` stats.
  7. Global source (AirfRANS style): K is unchanged and U = |U_inf|.

## 6. Size and risks
- **Size (my estimate):**
  - model.py plus a new `scales.py`: about 200-250 lines of code and documentation.
  - inputs.py `_declare`: about 30.
  - Tests: about 250.
  - Recipe: about 10 in train.py; 2 model YAMLs (AeroZoo, AirfRANS); 2 dataset YAMLs; a cluster scan for the stats (z and the physical targets).
  - No checkpoint version bump: new parameters exist only when the option is declared.
- **Cancellation of g on multi-inlet cases.** In `dual_inlet`, `open_opposed` and `confined_opposed` (`reference.secondary_inlet_velocities`), κ = |Σav|/Σa|v| can approach 0, and then g is noise. There is no equivariant fallback: an eigenvector's sign is arbitrary. **Scan κ per case before adopting** (a cluster CPU job), and refuse or warn below a threshold.
- **K = 0 breaks 3-D velocity.** The even rank-1 basis is the true vectors only (heads.py:150-152); with K = 0 it is {n̂, r̂, polar(r̂, n̂)}, all in span{n̂, r̂}. A degenerate g has the same effect. The cure, a learned vector from the input channels in the head basis, is a larger change.
- **bf16.** The range is fine, since bf16 has fp32's exponent. Precision is the problem: a bf16 head gives a relative error of about 4e-3 before the redimensionalization, which is fatal where |mean|/std ≫ 1 (HiLift temperature 1.0/0.002) and lossy for Pa pressure. Run the head and the redimensionalization in fp32, about 3 lines.
- **Loss weighting.** One constant std over physical targets weights high-U cases by U² (pressure). That is a program ruling, not ISLA's.
- **Wide-range boundary scalars.** Dimensionless scalars such as ωL/U, Re_θ ~ 340 and k/U² ~ 1e-3 enter raw (inputs.py:210-216). Offer a per-field `log` (the [s, sign·log|s|] pair the queries already use) or per-field standardization constants.
- **Nonzero vector means.** A nonzero vector mean in `output_stats` breaks exact rotation equivariance of the standardized output. Use zero vector means, or test the physical outputs.
