# AZ-compliant v2: which AeroZoo PDE data GeoTransolver and Transolver can take (read-only, 2026-10-19)

I read the code at psharpe/isla. Toy CPU probe: `toy_probe.py` → `toy_probe.log` (random tensors, `.venv`, `nice -n 19`).
Paths are relative to `physicsnemo/`. CP = `models/geotransolver/context_projector.py`, GT = `models/geotransolver/geotransolver.py`, BQ = `nn/module/ball_query.py`, TS = `models/transolver/transolver.py`, FK = recipe `src/forward_kwargs.py`.

## How GeoTransolver uses its inputs (Q1, Q2, Q5)
- **`geometry` takes two paths.**
  - (a) *Tokenizer.* `ContextProjector(geometry_dim)` (CP:1048) applies a linear projection to all columns and makes slice tokens, which are then used as cross-attention context in every GALE block. This path accepts any `geometry_dim` and any N_geom ≠ N_query (probe: geometry_dim=12, N_geom=40, N_q=64 runs).
  - (b) *Ball query*, only when `include_local_features=True` (CP:1029). Here `geometry` must be exactly xyz. `BQWarp` raises unless both inputs have last dim 3 (BQ:86), and the per-radius MLP has `in_features = geometry_dim*K` but receives neighbour **coordinates** of width 3 (CP:555, CP:598). Probe: geometry_dim=12 with local features fails with `ValueError: The last dimension of p_grid and x must be 3`.
  - The model uses no subset of columns: it never slices `[:3]`. It does not compute feature-space distances either; any extra columns make the ball query raise.
- **Ball-query directions.**
  - Context path: `processor(local_positions, geometry)` (CP:736) calls `radius_search(points=local_positions, queries=geometry)`. For each **geometry point** it collects the absolute xyz of up to K query points within r, and this is tokenized over N_geom.
  - Local path: `processor(geometry, local_positions)` (CP:761). For each **query point** it collects the absolute xyz of up to K geometry points within r. The result is passed through `tanh(MLP)` (CP:604) and concatenated to the hidden state (GT:888).
  - The features are absolute coordinates, not offsets. Radii are in the coordinates' own units.
- **Is `geometry` with extra features supported?** The docstring calls it "Pointwise dimension of the geometry input features" (GT:221), and the tokenizer path supports it. No published config uses it:
  - the transformer_models example passes subsampled STL vertex xyz (`transolver_datapipe.py` `process_geometry`), with `geometry_dim: 3`;
  - the unified recipe configs pass `interior.points` (dim 3).
  - The `pad_input_for_fp8` helper in that example's `train.py:262` zero-pads `geometry` to a multiple of 16, so it does not assume `geometry_dim` is 3.
- **`local_embedding`** is per-query features, through a preprocess MLP (GT:878). Published content: position + SDF to the body + closest-point normal. **`local_positions`** is query xyz (B,N,3), used only by the ball query. The published `train.py:351` uses `embeddings[:, :, :3]`.
- **`global_embedding`** is (B, N_g, C_g) and goes to its own ContextProjector (CP:1056).
  - It can be nu alone: `global_dim=1`, (1,1,1) runs. With N_g=1 the slice softmax over a single token gives essentially an affine function of nu, the same in every slice. So nu should be standardized with a per-dataset constant, since raw 1.5e-5 is effectively zero.
  - Optional only at construction. With `global_dim=None` and `global_embedding=None` it runs. With `global_dim=1` and the embedding left out, it crashes (context width mismatch; see log).
  - Published content: [air_density, stream_velocity], N_g=1.
  - N_g>1 tokens are allowed by the API but no published config uses them.
- **Empty balls and caps (Q5).**
  - Unused slots are filled with 0 (`radius_search.py:47`; probe: an empty ball gives four rows of (0,0,0)). A zero slot cannot be told apart from a neighbour at the origin.
  - Above the cap, Warp keeps the **first K found in hash-grid order** (`kernels.py:219`), not the nearest; torch keeps the nearest K (`_torch_impl.py:112`). The docstring says the two backends may give different neighbour sets above the cap (`radius_search.py:54`).
- **Effect in metres on AeroZoo** (radii [0.01, 0.05, 0.25, 1, 2.5, 5], K=[8, 16, 32, 32, 64, 128]):
  - On a 0.02 m body, every radius from 0.05 up holds the whole body plus far field. Those balls saturate at K, with an arbitrary subset under Warp, so only r=0.01 resolves the shape.
  - On a 20 m body in a 340 m domain, every query point more than 5 m from any geometry face gets all-empty balls, i.e. a constant `tanh(MLP(0))`. That is most of the far field if geometry is the body alone. The inlet, outlet and slip faces are tens to hundreds of metres from most points.
  - Raw coordinates of up to ~170 m enter the MLPs unscaled unless a per-dataset constant scale is applied. The published DrivAerML example divides by the constant `reference_scale` [12, 4.5, 3.25].
  - One set of radii in metres cannot cover a dataset whose scales span three orders of magnitude. This is a limit of the model's mechanism, and the rule says the recipe may not rescale per case to fix it.

## GeoTransolver: input → AeroZoo data under the rule
| Input | Allowed | Not allowed |
|---|---|---|
| `geometry`, tokenizer only (`include_local_features=false`, as in the published surface and HiLift configs) | All boundary faces' centroids [xyz m, one-hot inlet/outlet/no_slip/slip (4), bc_velocity (3), bc_pressure, bc_tke, bc_omega, bc_intermittency, bc_retheta, reverse-flow values]. Each column standardized with a per-dataset constant. **This carries the inlet drive per face.** | per-case nondimensionalization (current `NonDimensionalizeByMetadata`), `U_inf_dir`, `log_re` |
| `geometry` with ball query on | boundary-face xyz only (dim 3) | any prescribed value (the model raises) |
| `local_embedding` | interior xyz (m) + SDF and normals to `no_slip`, which is the published body-SDF analogue | `CenterMesh` offset; the `sdf_slip` fill of the bounding-box diagonal (sdf.py:108, a per-case length) |
| `local_positions` | interior xyz (m) | none |
| `global_embedding` | nu (dim 1, per-dataset constant standardization) | U_inf, Re, inlet averages (`BoundaryFieldToGlobal` is an area-average reduction, i.e. derived by the recipe) |

**GeoTransolver can receive the inlet drive** through its documented `geometry` *features* channel, per face, with no reduction, but only with the ball-query local features off. Ball query and prescribed values together would need a model change (ball query on `geometry[..., :3]`, tokenizer on all columns). A model change is outside the rule. A second option, not published: keep geometry as xyz with the ball query on, and send per-face BC tokens [xyz, one-hot, values] as N_g>1 `global_embedding` tokens. The API allows it (GT:300, "N_g is number of global tokens"); whether it counts as a "published mechanism" is for the user to rule.

## Transolver (Q3)
`forward(fx, embedding)` concatenates `embedding` and `fx` per token (TS:760) and passes the result through one preprocess MLP (TS:763). The two names differ only by convention: embedding = geometry, fx = input function. There is no separate point set, cross-attention or context. Every input has to be a per-query-point channel.

| Input | Allowed | Not allowed |
|---|---|---|
| `embedding` | interior xyz (m); SDF and closest-point normal to `no_slip`. The published mechanism is the SDF and normals to the body STL (`transolver_datapipe.py` volume embeddings) | SDF to inlet/outlet/slip: this applies the same mechanism to non-body surfaces, i.e. new channels; needs a ruling |
| `fx` | nu broadcast with `expand_like` (published fx = broadcast global scalars [ρ, \|U\|]) | inlet velocity or turbulence as a vector: the dataset has no global freestream, and any reduction of the per-face values is derived by the recipe |

**Transolver cannot receive the inlet drive** through its own mechanism on AeroZoo. It gets geometry (xyz, wall SDF and normal) and nu, so the flow direction and speed are unobservable to it. The only per-face route would be to append boundary faces as extra tokens with zero targets and mask them in the loss. Transolver's papers do not do this, so it would be an invented mechanism.

## Recipe plumbing (Q4)
- FK accepts dotted paths through getattr, then getitem: `boundaries.inlet.cell_centroids` (a `Mesh` property, mesh.py:963) and `boundaries.<b>.cell_data.<f>` (already used: `boundaries.vehicle.cell_data.normals`).
  - Lists are concatenated along the **last** dim only (FK `resolve_spec`). `expand_like` broadcasts a global along axis -2 (FK `_apply_modifier`).
  - Methods cannot be called, so `merge_boundaries` cannot be called from a spec.
  - Collate makes 0-d become (1,1,1) and (N,C) become (1,N,C) (collate.py:72), so `global_data.physics.kinematic_viscosity` arrives as (1,1,1).
- **The four boundaries cannot be concatenated along the point axis without new code.** `HarmonizeBoundaryFields` (domain_transforms.py:1517, already in `aerozoo_volume.yaml`) already gives every boundary the same `cell_data` keys and the one-hot labels. `DomainMesh.merge_boundaries(preserve_data=True)` → `Mesh.merge` (domain_mesh.py:1512) can then merge them.
- **Minimal adapter** (~20 lines): a registered recipe transform `MergeBoundaries(output: str)`. It writes `domain.merge_boundaries(preserve_data=True)` as one extra boundary (e.g. `boundaries.all`) and runs last in the pipeline. The spec would then be `geometry: [boundaries.all.cell_centroids, boundaries.all.cell_data.bc_inlet, …, bc_velocity, bc_tke, …]`.
  - Untested assumption: `Mesh.merge` also needs the boundaries' `point_data` to be concatenable.
  - An alternative is a forward-kwargs modifier `concat_points:` that takes a list of per-boundary lists.

## Confidence
- High on the code facts: they come from file:line reads and the probe.
- Medium on what counts as "published": the boundary-feature geometry and the multi-token global embedding are documented by the API but appear in no published config.
