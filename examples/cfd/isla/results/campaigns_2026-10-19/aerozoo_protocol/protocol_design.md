# AZ-COMPLIANT v2: AeroZoo with physical inputs, where each model brings its own scales (design, 2026-10-19)

This replaces `DESIGN.md` (v1, kept as the record). The change follows Peter's answer to v1's Q1 (2026-10-19): "GeoTransolver
(or other non-ISLA models) should NOT be able to use the area-weighted RMS distance between boundary points... If a model
wants normalized coordinates, it needs to bring it to the table. GeoTransolver does not. ISLA does."

I designed it only; nothing is implemented in tracked code, and no model has been trained.

- CPU checks ran on LHR `cpu-short`: job 285163 (final YAML, 22 cases), 285007/285027/285028 (an earlier YAML that
  still had sdf_slip, sdf_inlet and sdf_outlet; same results) and the scan 285444.
- The files are in `v2/`:
  - `aerozoo_puvolume.yaml`, the proposed pipeline;
  - `derive_v2.py`, which holds `StandardizeGlobalLeaf` and a *prototype* of ISLA's internal scales;
  - `check_v2.py`, `checks_v2.md`, `scan_scales.py`;
  - `baselines/REPORT.md`, a code reading of GeoTransolver and Transolver;
  - `islascale/REPORT.md`, a code reading of ISLA on bc-names bb07cf413.

## 1. The recipe: PDE inputs in physical units, the same for every arm

`v2/aerozoo_puvolume.yaml` is recipe_geo2's `aerozoo_uvolume.yaml` with every label read removed and nothing derived per
case.

| kept / added | removed |
|---|---|
| the four boundaries, `SubsampleMesh` (50k faces each, HT measure), `HarmonizeBoundaryFields` (one-hot condition type, zero-fill), the prescribed fields renamed to `bc_*`, **in physical units** (m/s, Pa, m²/s², 1/s) | the `reference.velocity → U_inf` and `reference.length → L_ref` renames; `ComputeLogReynolds`; both `NonDimensionalizeByMetadata`; `ComputeFreestreamDirection` |
| the SDF and normals to `no_slip`, in metres: the published body-SDF input of both baselines | `CenterMesh`: the coordinates stay raw. The rule says a model that wants normalized coordinates brings them itself. |
| `nu_std` = (log10 ν − a)/b with per-dataset constants a and b. This is a fixed elementwise map, not a per-case scale. | `BoundaryFieldToGlobal`, `SplitGlobalVector` and `global_vector`: the inlet reductions are derived by the recipe. |
| `FlipBoundaryOrientation` (normals into the fluid) | `sdf_slip`: its fill for an empty slip mesh is the bounding-box diagonal, a per-case length (`sdf.py:108`) |
| targets in **physical units**, one `NormalizeMeshFields` standardization per dataset, the same for every arm (vector means 0) | the label-scaled targets |

Density appears only inside ISLA, where it converts pressure in Pa. ISLA keeps its `dataset_drop_transforms`:
`SubsampleMesh` (it draws its tokens itself), the SDF passes and `HarmonizeBoundaryFields`.

## 2. What each model receives (evidence: `v2/baselines/REPORT.md`, file:line and a toy probe)

| model | input | AeroZoo data there under v2 | can it see the inlet drive? |
|---|---|---|---|
| **GeoTransolver** | `geometry` (B, N, C) | **Option B (recommended):** every boundary face as one token, [centroid xyz m, one-hot condition (4), bc_velocity (3), bc_pressure, bc_tke, bc_omega, bc_intermittency, bc_retheta, the reverse-flow values], each column standardized by a per-dataset constant. This is GeoTransolver's documented geometry-*features* channel (its tokenizer takes any `geometry_dim`; GT:221, CP:1048). It works **only with `include_local_features: false`**, because the ball query refuses non-xyz columns (BQ:86, CP:555). Option A: geometry = xyz only (interior points as today, or faces), with local features on. | **B: yes, per face, with no reduction.** A: **no**. |
| | `local_embedding` / `local_positions` | interior xyz (m), plus the SDF and normal to `no_slip` / interior xyz | – |
| | `global_embedding` (B, 1, 1) | `nu_std` only | – |
| | ball radii [0.01 … 5] m (A only) | fixed, in metres; not compensated (§5) | – |
| **Transolver** | `embedding` | interior xyz (m), plus the SDF and normal to `no_slip` | – |
| | `fx` | `nu_std`, broadcast per point (`expand_like`) | **No.** Transolver has only per-query channels (TS:760-763). AeroZoo has no global freestream, and any reduction of the inlet faces is derived by the recipe. It cannot observe the flow's speed or direction. |
| **ISLA, ISLA-DTT** | `boundary_meshes` by condition (full resolution), with the `bc_*` values in physical units; `global_data` | ν and ρ (declared roles) | **yes**, per face. Its scales are internal (§4). |

Notes on the table:

- The plumbing for Option B is about 20 lines: a recipe transform, `MergeBoundaries`, that writes
  `domain.merge_boundaries(preserve_data=True)` as a fifth boundary. Its spec is
  `geometry: [boundaries.all.cell_centroids, boundaries.all.cell_data.bc_*]`.
- There is a second way for GeoTransolver to see the per-face boundary conditions, which the API allows but no
  published config uses: per-face BC tokens as N_g > 1 `global_embedding` tokens, with ball query on. I do not
  recommend it.

## 3. Targets and the scorer

- **Targets:** u (m/s), p (Pa), ν_t, k and ω in SI units, each standardized by one per-dataset constant, the same
  for every arm.
- **Score: the label-era metric, unchanged.**
  - The scorer de-standardizes each prediction to physical units, which needs only constants.
  - It then maps the result into the label-era space, y_old = (y/s_ref − μ_old)/σ_old. Here s_ref is built from
    `reference.speed` and `.length`, as the label-era nondimensionalization did. The labels are used in scoring
    only, which is allowed.
  - Then it takes the same per-case relative L2.
  - Test D, on 22 cases: mapped targets equal the label-era targets to ≤ 4e-6 relative (velocity 2e-7). So v2
    scores and label-era scores are the same quantity, and old lanes keep their numbers.
  - Implementation: `score_space: label_reference` in `infer.py`, plus that test.
- **What changes: the training objective, not the metric.**
  - With one constant standardization of physical targets, each case weighs in the loss as its squared physical
    magnitude.
  - On 22 cases the per-case mean square of pressure spans a factor of 3.8e9. The 3 largest cases carry ~100% of
    the summed square, and the 11 smallest 3e-6.
  - For velocity the factor is 8.3e3, with 67% in 3 cases.
  - So unless the loss is changed (Q4), every arm trains almost only on the fastest, largest cases.
  - In the label era, nondimensional targets weighted all cases about equally.
- **Checkpoint selection:** use the mapped (label-space) metric on the validation part, the metric that is scored.
  Labels may normalize metrics.
- **Secondary metrics:** the AeroZoo book's ε_U and ε_p (volume-weighted, in U_ref and q_ref), reported beside the
  main score.

## 4. ISLA brings its own scales (design on bc-names; insertion points in `v2/islascale/REPORT.md`)

The mechanism is opt-in and generic. Nothing in code names a condition, and with nothing declared the model is
bitwise today's.

1. **Units per field.** Each field of a condition's schema, of `global_schema` and of `output_schema` may declare
   `units` as exponents of three base scales: L (length), U (velocity) and ρ (density).
   - The keys are ISLA-local: `_declare` pops them before `FieldSchema.parse`, so the shared `RankSpec` stays
     untouched. The `_args` stored for checkpoints keep the merged declaration.
2. **Length L.** ISLA's gauge, the area-weighted RMS pairwise distance of all boundary cells, computed on the
   **full-resolution** meshes before the token draw.
   - It is exact and independent of the draw (today's interior task computes it on the drawn tokens).
   - It is passed into `_forward_tokens` as `gauge=`, so positions, query wall distance and rank-0 query fields use
     one unit.
3. **Velocity U and direction g.** Any field flagged `velocity_scale: true` is a source: a rank-1 boundary field, or
   a rank-1 global.
   - U = √(Σ a\|v\|²/Σ a), the area-weighted RMS speed pooled over the sources. It cannot cancel.
   - g = Σ a v/\|Σ a v\| is appended to the rank-1 global directions, so K = 1 without `U_inf_dir`.
   - Both are computed per case in float64, before the draw.
   - The model refuses an absent or zero source.
4. **ρ and ν.** These are globals with roles `density` and `viscosity`. They are consumed for units and Re, and are
   not appended raw.
5. **Re.** z = (log10(U L/ν) − m)/s with config constants per dataset. z is one extra global scalar column in every
   condition's seed and in the query seed.
6. **Inputs.** Every declared boundary field and rank-0 global is divided by Π scale^exp before `inputs.features`:
   - the velocities by U, k by U², ω by U/L, gauge pressure by ρU²;
   - γ and Re_θt are dimensionless (offer a `log` option for Re_θt ~ 340).
7. **Outputs.** The model re-dimensionalizes after the head, y = ŷ·Π scale^exp, then standardizes with the
   dataset's constant `output_stats`. train.py injects the `NormalizeMeshFields` stats and asserts equality, about
   10 lines.
   - So the loss is computed on the same physical standardized targets as every other arm.
   - The head and this step run in fp32: bf16's relative precision of 4e-3 is lossy for Pa pressure.
   - Vector means are 0, which keeps exact rotation equivariance.
8. **ISLA-DTT.**
   - The in-model g flows into the pair invariants (ρ·g, n_j·g; tree_read.py:454) through the existing `g_unit`
     argument.
   - The scales are computed before the forest.
   - The canonical frame needs nothing. Optionally, g gives it an equivariant tie-break in place of today's lab-axis
     one.
9. **The `_forward_tensors` path** computes the scales from its tensors, or refuses when scales are declared.

**AeroZoo declaration** (in `isla_volume_aerozoo.yaml`; the scale-relevant keys only):

```yaml
boundary_schemas:
  inlet:  {bc_velocity: {rank: 1, units: {U: 1}, velocity_scale: true}, bc_tke: {rank: 0, units: {U: 2}},
           bc_omega: {rank: 0, units: {U: 1, L: -1}}, bc_intermittency: {rank: 0}, bc_retheta: {rank: 0}}
  outlet: {bc_pressure: {rank: 0, units: {U: 2, rho: 1}}, bc_velocity_reverse: {rank: 1, units: {U: 1}}, ...}
  no_slip: {}
  slip: {}
global_schema:
  physics.density: {rank: 0, role: density}
  physics.kinematic_viscosity: {rank: 0, units: {U: 1, L: 1}, role: viscosity}
reynolds: {mean: 6.1, std: 1.55}        # PLACEHOLDER (22 test cases: 6.09, 1.56); refit on train (scan, section 6)
output_schema: {velocity: {rank: 1, units: {U: 1}}, pressure: {rank: 0, units: {U: 2, rho: 1}},
                nut: {rank: 0, units: {U: 1, L: 1}}, tke: {rank: 0, units: {U: 2}}, omega: {rank: 0, units: {U: 1, L: -1}}}
output_stats: ${dataset target stats}
```

**The other datasets use the same mechanism** (from their configs):

- **AirfRANS.**
  - The source is the **global** `U_inf: {rank: 1, units: {U: 1}, velocity_scale: true}`. It is the farfield
    Dirichlet value, a PDE input there. g is that vector's direction, and U = \|U_inf\|.
  - ν is the physical constant 1.56e-5, set with `SetGlobalField`, and Re varies only through U.
  - Outputs: velocity {U: 1}, kinematic pressure {U: 2}, nut {U: 1, L: 1}.
  - This also removes today's raw `U_inf_mag` input (m/s), which breaks similarity.
- **DrivAerML and SHIFT.**
  - The freestream is constant and the targets are already Cp-like with constant scales. **Declare nothing**: Re is
    constant, so z would be a constant, and a constant rescale is absorbed by the target stats. `U_inf_dir` stays,
    since it is a constant PDE input.
  - Optional for physical targets: a global source `U_inf`, ρ, and p and τ in {U: 2, ρ: 1}.
- **HiLift.** It is compressible: temperature and density outputs need T and ρ scales and a Mach input. Declare
  nothing until those base scales exist; the grammar refuses other base scales.
- **Tensile2d, the lug, DeepJEB++.** There is no velocity, so declare nothing; the model is identical to today's.
  The flag name generalizes (`scale_source: S` from a load), which would make the linear-elastic lug and DeepJEB++
  exactly load-proportional. That is future work.

**Tests and size.**

- New tests in `test_isla_scales.py`, in float64:
  1. similarity, (x, v, ν, ρ, k, ω, p) → (sx, cv, scν, rρ, c²k, (c/s)ω, rc²p), gives identical internals and
     outputs scaled by their units;
  2. renaming the flagged condition changes nothing;
  3. draw and refinement independence of U, g and L;
  4. opt-in identity against the release goldens;
  5. hand values and the refusals;
  6. the `output_stats` round trip;
  7. a global source (the AirfRANS case).
- Existing covariance, rotation and global-vector tests are extended. `test_geometric_scale_equivariance` is scoped
  to models without scales.
- Estimated size: about 250 lines of model code, about 250 of tests and about 10 in the recipe. It is a lane-writer
  task on bc-names after its fix round lands. The checkpoint version is unchanged, because the new parameters exist
  only when the option is declared.

## 5. Checks (`v2/checks_v2.md`; 22 cases, one per test split; CPU)

| check | result |
|---|---|
| every label perturbed (reference.* ×7.3 plus noise, identity, diagnostics, solution_definition, closure id, body_id): v2 baseline pipeline / v2 ISLA pipeline | **0 of 2,550 / 0 of 1,450 leaves differ (bitwise)** |
| every label **deleted**: both pipelines run | **0 differing leaves**, so no transform reads a label |
| ISLA prototype scales under 2 rigid motions per case (float64) | L, U and Re change by ≤ 4.4e-16; g is equivariant to 3e-16; nondimensional scalars change by ≤ 4e-15 |
| similarity (s, c) = (3.7, 0.29) and (0.05, 12) applied to the raw case | L/(sL₀) and U/(cU₀) equal 1 to 3e-16, Δlog Re ≤ 2e-16; nondimensional inputs identical to 7e-16; nondimensional targets identical to ≤ 1e-4 (the float32 storage of the targets) |
| U against the label U_ref | one-inlet cases: within 6e-8. Two-inlet cases: 1.21 (layout) and 0.76 (dual_inlet), with \|mean v\|/U 0.78 and 0.68 |
| ISLA's log Re against the label Re | corr 0.993 (R² 0.986), offset +0.58 ± 0.19 decades |
| score map (test D) | ≤ 4.2e-6 relative on every field |
| κ = \|mean v\|/U, the conditioning of g (scan 285444, 1,156 of 3,816 cases in shard order: it hit the 20-min cap) | min 0.671 (146 two-inlet cases, median 0.71); 1 on every one-inlet case. g is well defined everywhere. ISLA log10 Re: mean 5.94, std 1.43 (284 cases) |

**GeoTransolver's fixed metre radii in raw coordinates (table in `checks_v2.md`).**

- Interior extents run from 0.11 m to 125 m (median 6.2 m).
- In the 6 cases with extents up to 1.2 m, every ball at r ≥ 0.25 m is at its cap, whether the geometry is the
  interior draw or the boundary faces. Under Warp, the kept neighbours are the first K in hash order.
- In the 5 cases above 30 m, 0.67 to 0.99 of the queries have **empty** balls at r = 0.25 m, and 0.29 to 0.88 at
  r = 1 m (boundary geometry). An empty ball gives the constant tanh(MLP(0)).
- So the multiscale local features carry little information at either end of the range. The rule says not to
  compensate, so this is recorded.

## 6. Predictions for GeoTransolver with raw coordinates, and decisions: `PREREG_v2.md` §§7-8
