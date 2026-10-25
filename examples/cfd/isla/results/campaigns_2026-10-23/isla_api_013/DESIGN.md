# ISLA-API: interior means ISLA-DTT, Mesh's names in ISLA's signatures, ISLA 0.13.0

Lane ISLA-API (013), 2026-10-24 21:50 EDT. Branch `psharpe/isla-api-013` from origin/psharpe/isla 5b70c8686,
worktree `/home/psharpe/gh/physicsnemo-isla-api-013`. Scope: `physicsnemo/experimental/models/isla/`,
`test/experimental/models/isla/`, the unified recipe, and the ISLA API page `docs/api/models/isla.rst` (it names
`query_independent`; it is the model's reference page, not the book). No book edits. Historical files under
`examples/cfd/isla/{research,results}` that construct or mention the old API are records of past runs: untouched.

Peter's words (2026-10-24, about 21:43 EDT): (a) "in the interior=True case, query_independent=True is kind of a
no-brainer. So basically, interior means "use DTT"." (e) "make the notation of the signatures match Mesh's conventions
where possible? For example I think that `dim ` should be n_spatial_dims, right?"

Three commits, in this order, each `git commit -s` with hooks on: (a), (e), then the version commit last. The
reference checkpoint `data/isla_default_v1.mdlus` stores the constructor's arguments (`query_independent`, `dim`),
so `test_isla_reference_checkpoint_matches_golden` fails on the (a) and (e) commits and passes again at the version
commit, which regenerates the file; every other test passes at every commit. The suites run on the head.

## (a) The `query_independent` argument goes

The interior task (`interior=True`) always takes the query-independent read (ISLA-DTT: passive prediction-point
tokens with `log_w = -inf`, plus the tree read after every block). The surface task keeps the interacting form
(option A, `prediction_boundaries`): it has no prediction-point tokens, so nothing about it changes.

### Code deleted (used only by interacting interior ISLA)

| Where | What | Why it goes |
|---|---|---|
| `model.py` `ISLA.__init__` | the `query_independent` argument, the `self.query_independent` attribute, its resolution and its "needs interior=True" check, and `self._args["__args__"]["query_independent"]` | the form is now fixed by `interior`; a checkpoint names its form through `interior` alone |
| `model.py` `ISLA.__init__` | `routing_mass=not self.query_independent` passed to `_QueryTokens`; `if self.query_independent:` around the tree read's construction becomes `if self.interior:` | the interior task always owns `kernel_basis` and `kernel_reads` |
| `blocks.py` `_QueryTokens` | the `routing_mass` argument and the `logw` parameter (`qt.logw`, the interacting tokens' learned fraction of the boundary measure) | passive tokens carry no routing weight |
| `model.py` `_query_tokens` | the `else` branch that sets the interacting tokens' log weight from `qt.logw` | as above; the passive `-inf` branch becomes the only body |
| `model.py` `_log_count` | the whole helper (`log(n)` of the query count under compile) | its only caller is that branch |
| `model.py` `query_chunks` | the `if not self.query_independent: raise` guard | replaced by `if not self.interior: raise ValueError(...)`: a surface model has no prediction points to chunk |
| `model.py`, `blocks.py`, `tree_read.py`, `docs/api/models/isla.rst` | docstrings and comments that describe the interacting interior form, its routing mass, its query-set dependence, or `query_independent` as a choice (module docstring: *Surface and interior tasks*, *Data layout*; class docstring: `interior`, `query_independent`; `forward`'s `prediction_points`; `_refuse_earlier_state_dict` keeps its history) | the form no longer exists. The whole-model fullgraph compile is stated for the surface task only, and ISLA-DTT's `compile_blocks` for the interior |

Kept, since the surface task or ISLA-DTT needs it: `_QueryTokens.seed` (the prediction points' input layer),
`_softmax_over_points` and every passive-token path in the blocks, `_frame_anchors`, `_head`, the token padding,
the in-model draw (`sample_boundary_cells`), `exact_boundary_geometry`, every tree-read module, `_QueryChunks`, and
`_forward_tensors` (the private batched routine the release tests pin; its `query_points` path now runs ISLA-DTT).

Changed behaviour (not a deletion, but a capability the interior task loses): rank-2 boundary inputs. ISLA-DTT's
boundary read has no pair contractions for them, so `interior=True` with a rank-2 boundary field is refused at
construction, as `query_independent=True` with one is today (message reworded to name the interior task). The
surface task keeps rank-2 boundary inputs. No recipe config declares a rank-2 boundary input on an interior task
(checked at build time by the config tests).

### Recipe

- Interior model configs (`isla_volume.yaml`, `isla_volume_aerozoo.yaml`, `isla_volume_airfrans.yaml`,
  `isla_lug.yaml`, `isla_tensile2d.yaml`) lose the `query_independent: true` line, and their comments that offer
  `model.query_independent=false` (interacting ISLA's whole-model compile) as a choice. Surface configs lose their
  comment lines that mention `model.query_independent=true`.
- `src/train.py`: the chunked step keys on `interior`. `_chunked_step` returns true for a model with `query_chunks`
  and `interior` true, and a batch with prediction points; the training-step log line keys on the same test (one
  helper, so the two cannot disagree). The recipe's README loses `query_independent`.
- `tests/test_isla_configs.py`, `tests/test_train_helpers.py`: assertions on `query_independent` become assertions
  on `interior` (the resolved config of every interior config has no `query_independent` key, and `interior: true`).

### Tests: deleted, and why (one line each)

Rule: a test of a contract that only interacting interior ISLA has (routing mass, query-set dependence, interacting
fullgraph compile, the two forms' separation) is deleted. A test that used interacting interior ISLA only as a
vehicle for a shared contract (geometry, the draw, scales, names, heads, two dimensions, conditions, padding) is
kept and runs on ISLA-DTT instead (the `query_independent=False` keyword dropped; a test whose assertions read the
interacting tokens' routing, such as a parameter count with `qt.logw`, gets ISLA-DTT's numbers, worked out, never a
loosened tolerance). Where ISLA-DTT already has the same test beside it, the interacting copy is deleted rather than
converted. The writer records each deletion in `scratch/isla-api-013-2026-10-24/DELETED_TESTS.md`. Expected:

| Test (file::name or parameter) | Reason |
|---|---|
| `data/isla_interior_v1.pth` and `test_isla_release.py` `CONFIGS["interior"]` (its forward golden, Mesh-wrapper and attribute cases) | the golden of interacting interior ISLA, a form that no longer exists; `CONFIGS["dtt"]` and `isla_dtt_v1.pth` cover the interior task |
| `test_isla_release.py::test_a_state_dict_of_one_interior_form_does_not_load_into_the_other` | there is one interior form |
| `test_isla_release.py::test_new_interior_checkpoints_store_their_form` | the form is not stored; `interior` is, and the round trip is covered by `test_isla_save_from_checkpoint_round_trip` and the attribute test |
| `test_isla.py::test_interior_task_contracts` | its distinctive assertion is the interacting form's query-set dependence; the covariance contracts of the interior task are `test_query_independent_interior_contracts` (kept, renamed `test_interior_task_contracts`) |
| `test_isla.py::test_query_mass_refinement_invariance` | tests the interacting tokens' routing mass (`qt.logw` as a fraction of the boundary measure) |
| `test_isla.py::test_defaults_are_the_reference_configuration`, its interacting-interior part | the form choice is gone; the test keeps the surface and interior defaults and asserts `query_independent` is no constructor argument |
| `test_isla_compile.py`, interior cases of the whole-model fullgraph compile (`_model("interior...")`, `_budget_model`, `test_the_in_model_draw_compiles_under_max_autotune_at_full_resolution`) | ISLA-DTT's boundary read breaks a fullgraph trace, so the interior task compiles with `compile_blocks` (covered in `test_isla_kernels.py`); surface cases stay |
| `test_isla_token_padding.py`, the `"interior"` (interacting) parameter | `"interior_dtt"` covers the interior task's padding |
| `test_isla_conditions.py::test_every_condition_s_parameters_train_even_when_it_is_absent[isla]` if it is interior | `[isla_dtt]` covers it (the surface variant stays if present) |
| `test_isla_tree_read.py::test_query_chunks_need_query_independent_points` | converted, not deleted: a surface model's `query_chunks` is refused |

Kept and converted to ISLA-DTT (expected, the writer confirms each): `test_every_parameter_receives_a_gradient[interior]`,
`test_geometric_scale_equivariance[interior]`, `test_boundary_scalars_reach_the_interior_prediction`,
`test_frame_is_sampling_consistent`, `test_interior_flag_structures_the_forward`,
`test_scalar_only_model_has_no_frame_routing_...`; in global_inputs, heads, inputs, interior_geometry, names, nd,
query_encoding, routing, sampling, scales, surface_conditions: every `interior=True, query_independent=False`.
A converted test that declared rank-2 boundary inputs on an interior model moves those inputs to a surface model,
and the interior refusal gets its own assertion.

Goldens kept bitwise (not regenerated): `isla_default_v1.pth`, `isla_dtt_v1.pth`, `isla_backbone_v1.pth`,
`isla_default_checkpoint_output_v1.pth`. Neither (a) nor (e) may change a number: the forward of the surface task and
of ISLA-DTT is untouched, so these pass unchanged. That is the check that the refactor is exact.

## (e) Mesh's names in ISLA's signatures

Mesh's names, read from `physicsnemo/mesh/mesh.py` and `domain_mesh.py`: `points`, `cells`, `point_data`,
`cell_data`, `global_data`, `n_points`, `n_cells`, `n_spatial_dims`, `n_manifold_dims`, `cell_centroids`,
`cell_normals`, `cell_areas`, `Mesh[n_manifold_dims, n_spatial_dims]` (symbolic `Mesh["n-1", "n"]`); DomainMesh:
`interior`, `boundaries`, `boundary_names`, `n_boundaries`, `global_data`. `physicsnemo.mesh.fields`: `RankSpec`
(`rank`, `symmetric`, `parity`; `numel(n_spatial_dims)`, `shape(n_spatial_dims)`), `FieldSchema`. GLOBE, the other
mesh-native model, uses `n_spatial_dims`, `output_schema`, `global_schema`, `boundary_meshes`, `prediction_points`,
`global_data`.

| ISLA name (where) | Mesh concept | Decision |
|---|---|---|
| `dim` (constructor) | `Mesh.n_spatial_dims`, `RankSpec.shape(n_spatial_dims)`, GLOBE's `n_spatial_dims` | **rename to `n_spatial_dims`**; the attribute `self.dim` becomes `self.n_spatial_dims`; recipe keys `model.dim: 2` (airfrans, tensile2d) become `model.n_spatial_dims: 2`; the error messages and docstrings follow (`Mesh[dim - 1, dim]` becomes `Mesh[n_spatial_dims - 1, n_spatial_dims]`; jaxtyping `"n_points dim"` becomes `"n_points n_spatial_dims"`) |
| `query_independent` (constructor) | none | deleted by (a) |
| `output_schema`, `global_schema` (constructor) | `FieldSchema`; GLOBE's same names | keep (already the shared grammar's names) |
| `boundary_schemas` (constructor) | DomainMesh `boundaries` keyed by condition name, each a `FieldSchema` | keep: no Mesh name for a schema of the boundaries; GLOBE's `boundary_source_schemas` means source strengths, a different concept |
| `interior` (constructor, bool) | DomainMesh `interior` (the interior Mesh) | keep: the same concept (predict on the domain's interior, not on its boundaries) and already the same word |
| `boundary_tokens` (constructor) | none (a token budget); Mesh counts are `n_*` | keep: no Mesh concept. **Case for Peter** below (the `n_` count prefix) |
| `hidden`, `n_layers`, `n_slices`, `compile_blocks` | none | keep: network hyperparameters; `n_layers` and `n_slices` already follow the `n_` count prefix |
| `reynolds` | none | untouched here: REYNOLDS-OUT removes it (direction (b)) |
| `output_stats` | none (the recipe's `NormalizeMeshFields.stats` format) | keep |
| schema keys `rank`, `symmetric`, `parity` | `RankSpec` | keep (they are Mesh's own) |
| ISLA-local schema keys `units`, `velocity_scale`, `role`, `scale` | none | keep: physical scales, no Mesh concept |
| `boundary_meshes` (forward, query_chunks) | DomainMesh `boundaries` (a mapping of name to Mesh); GLOBE's `boundary_meshes` | keep. **Case for Peter** below (two names could fit) |
| `prediction_points` (forward, query_chunks) | a point-cloud Mesh's `points`; GLOBE's `prediction_points` | keep (GLOBE's name; it takes a Mesh or an `(N, n_spatial_dims)` tensor) |
| `prediction_boundaries` (forward; a list of names) | DomainMesh `boundaries` (meshes) or `boundary_names` (their names) | keep. **Case for Peter** below (two names could fit) |
| `global_data` (forward, query_chunks; output Mesh) | `Mesh.global_data`, `DomainMesh.global_data` | keep (already Mesh's) |
| output `Mesh` `points`, `point_data`, `global_data` | Mesh | keep |
| `cell_data["_effective_measure"]` read from boundary meshes | `Mesh.cell_data` | keep |
| `ISLA.boundary_names` (attribute, tuple) | `DomainMesh.boundary_names` | keep (same name, same meaning) |
| `ISLA.n_global_vectors`, `n_global_scalars`, `n_boundary_vecs`, `n_boundary_tens`, `n_block_vecs`, `n_block_tens` (attributes) | none | keep (`n_` counts already) |
| `_QueryChunks.n_points` | `Mesh.n_points` | keep (same name, same meaning) |
| `_forward_tensors(points, normals, measure_weights, global_inputs, query_points, surface_points, surface_cells, cell_second_moments)` | private | keep: a private routine the release tests pin; its `global_inputs` is a flat mapping of dotted names, not a `global_data` TensorDict. Listed so the omission is visible |
| internal helpers' `dim` parameters (`_InputSet(..., dim=)`, `_OutputLayout(..., dim)`, `_pseudovectors(primaries, dim)`, locals `K, dim = ...`) | private | keep, to keep the diff surgical; only the public argument and attribute are renamed. A module-level function whose parameter is spatial dimension and that a caller outside ISLA reads, if the writer finds one, is renamed too and listed |

### Cases for Peter (unchanged until he decides)

1. `boundary_meshes` (GLOBE's forward name, today's) or `boundaries` (DomainMesh's field). Recommend keeping
   `boundary_meshes`: the two mesh-native models then share one forward vocabulary, and `boundaries` is the field of a
   container rather than an argument holding meshes.
2. `prediction_boundaries` (today's) or `prediction_boundary_names` (after DomainMesh's `boundary_names`). Recommend
   keeping `prediction_boundaries`: it pairs with `prediction_points`, and its value is a list of names.
3. `boundary_tokens` or `n_boundary_tokens`: not a Mesh concept, but Mesh writes every count as `n_*` (`n_points`,
   `n_cells`, `n_boundaries`), as ISLA does for `n_layers` and `n_slices`. Recommend `n_boundary_tokens` if Peter wants
   the count convention applied; it costs one more renamed recipe key (`model.boundary_tokens`) in every interior
   config. Not done here without his word, since the brief renames only where Mesh has the same concept.

## The version: ISLA 0.13.0 (last commit)

The constructor's signature changes (`query_independent` gone, `dim` renamed), and interior checkpoints lose the
interacting form, so this is ISLA 0.13.0, in BAND-MERGE's form (3f3fef69f):

- `__model_checkpoint_version__ = "0.13.0"`, `_version = 13`; the version comment block gains a 0.13.0 line (the
  interior task is ISLA-DTT without a switch; `dim` is `n_spatial_dims`; same parameters for the surface task and
  ISLA-DTT, new constructor arguments). `_refuse_earlier_state_dict`'s docstring and message gain "before 0.13.0, the
  constructor's `dim` and `query_independent`, and interacting interior ISLA", and refuse 0.7.0 to 0.12.0 by
  state-dict version (1, 8, 9, 10, 11, 12) for every ISLA, surface and interior alike.
- `test_isla_release.py`: the earlier-version parameter lists gain 0.12.0 (and `("0.12.0", 12)`); the test name
  becomes `test_isla_0_7_0_to_0_12_0_state_dicts_are_refused_on_every_loading_path`; the version asserts read 0.13.0.
- `data/isla_default_v1.mdlus` is regenerated by the procedure in `test_isla_release.py` (deleted, recreated by the
  test, then the test run again to pass). Check, recorded in the commit message: its tensors are bitwise the previous
  file's (all of them, compared by name), and the metadata differs only in `mdlus_file_version`, the root state-dict
  version, and the stored arguments (`dim` now `n_spatial_dims`, `query_independent` gone). Its output golden
  `isla_default_checkpoint_output_v1.pth` is unchanged.
- If the supervisor asks to fold REYNOLDS-OUT's result into this branch, it lands before this commit, so one version
  bump covers (a), (b) and (e).

## Tests and runs

- ISLA directory on LHR cpu-short (`scratch/isla-surface-bc-2026-10-23/lhr/tests_cpu.sbatch` model) and the recipe
  tests, on a read-only code snapshot of the head under `$T`.
- The whole ISLA directory on one GB300 node (short QoS, five bad nodes excluded, about 60 min guard), every CUDA test.
- A GB300 smoke per interior config (`isla_volume`, `isla_volume_aerozoo`, `isla_volume_airfrans`, `isla_lug`,
  `isla_tensile2d`) and one surface config (`isla_surface`): compose the config with its dataset through Hydra, assert
  the resolved model config (`interior`, no `query_independent`, `n_spatial_dims` where 2), instantiate, and run one
  forward and backward on a synthetic case of the config's boundary names (or one real case where cheap).
- Job names `isla.isla-api.*`; registry row `^isla\.isla-api\.` (expected_end 2026-10-25) before the first job; cap
  3 GPU-hours.
- An assertion audit of the diff: no tolerance loosened, no assertion dropped from a kept test, every deleted test in
  `DELETED_TESTS.md` with its reason.

## Book

No edits. `BOOK_CHANGES.md`: the book's cells run against the head with the line-start-fence runner, every cell that
breaks or prints differently listed, and the prose naming `query_independent`, `dim`, or the interacting interior form
as the default or as a choice. The interior standings stay interacting ISLA 0.9.0's until the final rerun.

## Addendum, 2026-10-25 01:50 EDT: what was built (after the supervisor's steers of 23:27 and 23:41 EDT)

- The name cases, as the supervisor picked them (Peter may reverse): `boundary_meshes` and `prediction_boundaries` stay;
  `boundary_tokens` became `n_boundary_tokens` (own commit 7c957ac7e, recipe keys included).
- REYNOLDS-OUT's design was folded in (its DESIGN.md 44f40d13): `reynolds`, the viscosity role and z left the
  constructor; the recipe computes `re_z` (ComputeBoundaryScales, ComputeLogReynolds in float64) in ISLA's two model
  configs only; `defines: <name>` replaced `velocity_scale` and the roles (one constant `scales.DEFINES`).
- Added rows to the mapping table: `boundary_tokens` -> `n_boundary_tokens` (renamed); `reynolds` (deleted);
  schema keys `velocity_scale`, `role` -> `defines` (provisional name, flagged to Peter); `velocity_moments` ->
  `rms_moments` (module function).
- The interior contract tests run a live boundary read (`live(model)` in _meshes.py); two refinement checks run their
  refine assertion at theta 0 (the exact read). The three draw-under-compile tests compile ISLA-DTT's `_case_inputs`.
