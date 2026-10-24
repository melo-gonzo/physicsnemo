# Baseline view of the domain boundary (read-only survey, 2026-10-23)

Paths: `R/` = `examples/cfd/external_aerodynamics/unified_external_aero_recipe/` in worktree `physicsnemo-isla-whole-boundary`; `D/` = `scratch/isla-boundary-audit-2026-10-23/dsconf/` (the five recipe-resident pairings are byte-identical to `D/` copies, checked with `cmp`). Facts from static reading only; nothing was run.

## Baseline configs per dataset

| Dataset (config) | GeoTransolver | Transolver | GLOBE |
|---|---|---|---|
| DrivAerML Volume (`drivaer_ml_volume_reference`) | `geotransolver_volume` | `transolver_volume` | `globe_volume` |
| AirfRANS (`airfrans_uvolume_2d`) | `geotransolver_volume_airfrans` | `transolver_volume_airfrans` | none for AirfRANS (`globe_volume` keys `vehicle`) |
| DrivAerML Surface / SHIFT-SUV / HiLift surface | `geotransolver_surface` | `transolver_surface` | `globe_surface` |
| DeepJEB++ (`deepjebpp_surface`) | none (`geotransolver_surface_deepjebpp.yaml` is referenced at `R/conf/model/transolver_surface_deepjebpp.yaml:21` but absent) | `transolver_surface_deepjebpp` | none |
| Tensile2d (`plaid_tensile2d_2d`) | none (`geotransolver_tensile2d.yaml` referenced at `transolver_tensile2d.yaml:22`, absent) | `transolver_tensile2d` | none |
| lug (`ginot_lug_volume`) | none | `transolver_lug` | none |
| AeroZoo (`aerozoo_uvolume`) | `geotransolver_volume_aerozoo` | `transolver_volume_aerozoo` | none |

The surface templates name DrivAerML, SHIFT-SUV and HiLift (`geotransolver_surface.yaml:19,33-38`, `transolver_surface.yaml:29-30`). `geotransolver_volume_highlift` is for HiLift *volume*. The AirfRANS baseline templates say "Pair with datasets/airfrans_uvolume_ud.yaml" (`geotransolver_volume_airfrans.yaml:21`, `transolver_volume_airfrans.yaml:21`). They read `global_data.U_inf_mag_std`, which only the `_ud` datasets' `SplitGlobalVector` writes (`:30-31` / `:28-29`). `airfrans_uvolume_2d` has no such step (`D/airfrans_uvolume_2d.yaml:41-60`) and is two-dimensional (RestrictToPlane `:44-47`), while both templates declare 3-D sizes (`functional_dim: 7`, `geometry_dim: 3`).

## 1-2. What each forward receives and what it sees of the non-body boundary

Frame notes:
- `CenterMesh.apply_to_domain` subtracts the interior centre of mass and translates every boundary by the same amount (`physicsnemo/datapipes/transforms/mesh/transforms.py:264-286`).
- `NonDimensionalizeByMetadata` divides points by `global_data.L_ref` when it is present and `scale_geometry` is true (`R/src/nondim.py:207-214`). It applies to the interior and to every boundary (`nondim.py:332-339`).
- On DrivAerML, augmentations rotate about z only and translate in x/y only (`D/drivaer_ml_volume_reference.yaml:59-72`). The z axis therefore stays floor-normal. After CenterMesh, the floor's z is minus the z of the drawn interior points' centroid divided by L_ref, so it is not exactly constant across draws. GLOBE's comment gives `L_ref=5.0` (`globe_surface.yaml:26-28`).

| Dataset / baseline | coords | sdf, sdf_normals | geometry / boundary input | global | non-body boundary pieces |
|---|---|---|---|---|---|
| DrivAerML Vol / GeoTransolver | interior pts, centred, /L_ref | computed: `ComputeSDFFromBoundary(boundary_name: stl_geometry)` (extra `*_single_solid.stl.pmsh`), then stl dropped (`D/drivaer_ml_volume_reference.yaml:55-57,89-98`) | `geometry: interior.points` (`geotransolver_volume.yaml:49`), i.e. volume points, not a boundary | `U_inf_dir` | not explicit; implicit via z in the fixed frame and the domain-shaped interior cloud |
| DrivAerML Vol / Transolver | same (`transolver_volume.yaml:39-42`) | same | none | `fx = U_inf_dir` broadcast | implicit only |
| DrivAerML Vol / GLOBE | interior pts, same frame | not used | `boundary_meshes: {vehicle: boundaries.vehicle}` (`globe_volume.yaml:36-37,49-50`) | none | not at all (no_slip/slip/inlet/outlet are loaded, since `drop_in_file_boundaries` is false, but not passed); implicit via coordinates |
| AirfRANS / GT, Transolver | interior pts (2-D after RestrictToPlane); NonDim fields only velocity/pressure/nut (`:50-55`) | **stored** in file (`sdf`, `sdf_normals`), source not stated in config | GT `geometry: interior.points` (`geotransolver_volume_airfrans.yaml:42`) | `U_inf_dir` + `U_inf_mag_std` | implicit only (coordinates) |
| DrivAerML Surf / SHIFT / HiLift: GT, Transolver | interior = vehicle cell centroids (`MeshToDomainMesh`), centred, /L_ref (`D/drivaer_ml_surface.yaml:67-95`) | none; `boundaries.vehicle.cell_data.normals` from ComputeSurfaceNormals | GT `geometry: interior.points` = body only (`geotransolver_surface.yaml:43-47`) | `U_inf_dir` | only the body is read (surface files hold only `boundaries/vehicle`, reader pattern `:34`); implicit through z |
| same / GLOBE | centroids, nondim | none | `{vehicle: boundaries.vehicle}` (`globe_surface.yaml:46-50`) | none | body only |
| DeepJEB++ / Transolver | cell centroids, centred, mm to 0.1 m (`D/deepjebpp_surface.yaml:33-36`) | none | none; per-cell `normals, bc_fixed, bc_loaded, bc_load` (`transolver_surface_deepjebpp.yaml:26-31`) | `load_dir, load_type` | clamped/loaded faces explicit as per-cell labels on the one `surface` mesh; free faces = cells with zero labels |
| Tensile2d / Transolver | interior nodes, 3-coordinate form (z = 0) | stored (sdf to the outer polyline, per header `D/plaid_tensile2d_2d.yaml:6-12`) | none; nodal `tag_top/bottom/bottomleft` (`transolver_tensile2d.yaml:30-36`) | `params_std` | loaded/fixed/pinned edges through nodal tags; `boundaries.outer` (bc_*, traction) not read. Note: the dataset applies RestrictToPlane (`:31-34`) although the header says the baselines read the 3-coordinate form |
| lug / Transolver | interior nodes, raw (no Center/Scale in pipeline) | stored (sdf to the watertight surface, `D/ginot_lug_volume.yaml:4-5`) | none (`transolver_lug.yaml:30-36`) | `params_std` (includes pressure) | clamped/hole/loaded faces not seen; only the SDF to the whole surface; `bc_*` unused (stated at `:21-22`) |
| AeroZoo / GeoTransolver | interior, raw SI metres, centred | computed: `ComputeSDFFromBoundary(boundary_name: no_slip)` into `sdf_wall` (`D/aerozoo_uvolume.yaml:99-102`) | `geometry` = `boundaries.merged` cell centroids + bc one-hots and values, from `MergeBoundaries([inlet,outlet,no_slip,slip])` (`:151-154`; `geotransolver_volume_aerozoo.yaml:67-83`), `include_local_features: false` (`:105`) | `nu_std` | all four pieces explicit as tokens |
| AeroZoo / Transolver | same | `sdf_wall` (no_slip) | none (`transolver_volume_aerozoo.yaml:45-51`) | `nu_std` | implicit only; the config comment says it does not see the inlet (`:25-28`) |

Static inconsistency (not run): `geotransolver_volume`, `transolver_volume`, `*_airfrans`, `transolver_lug` and `transolver_tensile2d` set `dataset_reader_overrides.drop_in_file_boundaries: true`. `_apply_dataset_reader_overrides` merges that into the reader (`R/src/datasets.py:103-113`), and the reader then skips in-file boundaries (`physicsnemo/datapipes/readers/mesh.py:573-578,685`). Several pipeline steps still name those boundaries:
- DrivAerML Volume's `FlipBoundaryOrientation(boundaries: [vehicle])` (`D/drivaer_ml_volume_reference.yaml:103-104`), which raises if a named boundary is missing (`R/src/domain_transforms.py:1767-1772`).
- AirfRANS' RestrictToPlane `boundaries.airfoil.cell_data.normals`.
- Tensile2d's `boundaries.outer...` and the lug's `bc_pressure` normalization.

Whether those runs fail, or whether the transforms tolerate the missing boundary, was not verified.

## 3. What the released code accepts from config without code change

- **forward_kwargs resolver** (`R/src/forward_kwargs.py:129-192`):
  - A string is a dotted path, so any `boundaries.<name>...` leaf can be named.
  - A list is concatenated along the last dim, which means feature channels, not tokens. It requires equal leading shapes, so a list cannot stack several boundaries' faces into one token set.
  - A dict resolves to a dict of resolved values, which gives GLOBE its `boundary_meshes`.
- **GeoTransolver**:
  - `forward(..., geometry: Tensor[batch, tokens, geometry_dim] | None)` takes one tensor (`physicsnemo/models/geotransolver/geotransolver.py:717-731`); `local_embedding`/`local_positions` may be tuples.
  - Several boundaries as geometry tokens therefore require one merged mesh. That is what AeroZoo does with recipe-local `MergeBoundaries(boundaries, fields, output, standardize)` (`R/src/domain_transforms.py:1976-2060`), which uses `Mesh.merge`.
  - `MergeBoundaries` requires a non-empty `fields` list present on every listed boundary at the same trailing shape (`:2008-2009,2041-2048`). `HarmonizeBoundaryFields` provides that on AeroZoo (`D/aerozoo_uvolume.yaml:65`).
  - `geometry_dim` is free. The ball-query local features take xyz positions (`context_projector.py:534,692`); AeroZoo turns them off.
- **ComputeSDFFromBoundary**:
  - `boundary_name: str`, one key of `domain.boundaries`, which must have cells unless `allow_empty_boundary` is set (`R/src/sdf.py:87-103,139-152`). A union of boundaries is not a parameter.
  - A `MergeBoundaries` output placed before it in the transform list is a boundary like any other, and `use_winding_number` (default true) is the documented option for non-watertight surfaces (`sdf.py:70-72`).
  - `ComputeSDFFromBoundary` can be instantiated several times with different `sdf_field`s, giving per-boundary distances. Transolver/GeoTransolver can read them as extra `embedding`/`local_embedding` channels if `embedding_dim`/`functional_dim` are raised.
- **GLOBE**:
  - `__init__(boundary_source_schemas: dict[str, schema])` sets the BC names (`physicsnemo/experimental/models/globe/model.py:251-306`).
  - `forward(prediction_points, boundary_meshes: dict[str, Mesh], reference_lengths, global_data=None)` (`model.py:902-908`), whose keys must be a subset of those names (`:190-193`). Its docstring example uses `{"no_slip": ..., "freestream": ...}` (`:238-244`).
  - Several named meshes (e.g. `vehicle, no_slip, slip, inlet, outlet`) can therefore be passed by listing them in both `forward_kwargs.boundary_meshes` and `model.boundary_source_schemas`. Per-boundary `cell_data` fields are consumed only if declared in that boundary's schema (`globe_volume.yaml:24-30`).
- **Transolver**: `forward(fx, embedding)` takes per-point tensors only (`physicsnemo/models/transolver/transolver.py:691-694`). It has no boundary-token input, so boundary information can reach it only as per-point channels (SDFs, tags) or broadcast globals.

## Residual risks
- The stored-SDF source surfaces (AirfRANS, lug, Tensile2d) come from config header comments, not converter code.
- Whether `L_ref` exists in each dataset's global_data was not checked against the data files.
- The `drop_in_file_boundaries` failure path was found statically and not executed.
- Batching or collation of variable token counts for multi-boundary geometry was not inspected.
