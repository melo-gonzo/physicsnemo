# ISLA test coverage, runtime and skips at 5b70c8686 (ISLA 0.12.0)

TEST-AUDIT, measured 2026-10-24 (jobs in JOBS.md). Python 3.13.16, torch 2.12.0+cu130, triton 3.7.0, warp 1.14.0,
coverage.py 7.14.1 (`coverage run --branch`; no pytest-cov in the LHR venv). Source: `$T/code_mainmerge_base`
(git archive of 5b70c8686). Tables are produced by `analyze_cov.py` and `analyze_times.py` from `results/`. To rerun
`analyze_cov.py`, first restore `src/`, which was deleted after the report: `git archive 5b70c8686
physicsnemo/experimental/models/isla | tar -x -C src`. The child reports cite paths "relative to SRC", which is the same
commit: read them with `git show 5b70c8686:<path>`.

## Line and branch coverage per module

"Non-kernel" statements leave out the bodies of `@triton.jit`, `@wp.kernel`, `@wp.func` and `@wp.struct` functions
(1,473 lines). coverage.py cannot see code that runs as a GPU or Warp kernel, so those lines always read as missed.
- CPU: the whole ISLA dir (job 376354).
- GPU: the whole ISLA dir on one GB300 (job 376355), CPU tests included. 19 compile tests were cut by the guard; see
  the caveats below.
- recipe: the unified recipe's tests (job 376354), measured on ISLA's modules.
- union: CPU, GPU and recipe together.

| module | stmts (non-kernel) | kernel-body lines | branches | CPU line | CPU branch | GPU line | GPU branch | recipe line | union line | union branch | CUDA-only lines | never-run lines |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| __init__.py | 2 | 0 | 0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0 | 0 |
| blocks.py | 116 | 0 | 10 | 94.8 | 70.0 | 100.0 | 100.0 | 91.4 | 100.0 | 100.0 | 6 | 0 |
| fused_basis.py | 161 | 163 | 34 | 51.6 | 17.6 | 85.1 | 55.9 | 43.5 | 85.1 | 55.9 | 54 | 24 |
| fused_read.py | 167 | 178 | 34 | 50.9 | 23.5 | 76.6 | 50.0 | 41.3 | 76.6 | 50.0 | 43 | 39 |
| geo_kernel.py | 150 | 324 | 44 | 30.7 | 4.5 | 68.0 | 45.5 | 30.7 | 68.0 | 45.5 | 56 | 48 |
| geometry.py | 714 | 364 | 146 | 98.7 | 95.9 | 98.7 | 95.9 | 78.2 | 98.7 | 95.9 | 0 | 9 |
| geometry_fast.py | 98 | 199 | 10 | 80.6 | 50.0 | 100.0 | 80.0 | 69.4 | 100.0 | 80.0 | 19 | 0 |
| heads.py | 90 | 0 | 24 | 100.0 | 100.0 | 100.0 | 100.0 | 85.6 | 100.0 | 100.0 | 0 | 0 |
| inputs.py | 114 | 0 | 28 | 98.2 | 96.4 | 98.2 | 96.4 | 79.8 | 98.2 | 96.4 | 0 | 2 |
| model.py | 793 | 0 | 344 | 96.2 | 89.0 | 96.3 | 89.0 | 83.4 | 96.5 | 89.2 | 1 | 28 |
| sampling.py | 65 | 0 | 8 | 100.0 | 100.0 | 100.0 | 100.0 | 96.9 | 100.0 | 100.0 | 0 | 0 |
| scales.py | 101 | 0 | 62 | 93.1 | 88.7 | 93.1 | 88.7 | 86.1 | 93.1 | 88.7 | 0 | 7 |
| segment_softmax.py | 97 | 86 | 16 | 66.0 | 25.0 | 81.4 | 37.5 | 66.0 | 81.4 | 37.5 | 15 | 18 |
| tree_build.py | 169 | 69 | 26 | 99.4 | 92.3 | 99.4 | 92.3 | 21.3 | 99.4 | 92.3 | 0 | 1 |
| tree_read.py | 411 | 0 | 82 | 96.6 | 86.6 | 99.0 | 92.7 | 86.4 | 99.0 | 92.7 | 10 | 4 |
| tree_walk.py | 70 | 90 | 10 | 92.9 | 60.0 | 92.9 | 60.0 | 34.3 | 92.9 | 60.0 | 0 | 5 |
| **total** | 3318 | 1473 | 878 | 88.2 | 78.9 | 94.4 | 85.0 | 71.6 | 94.4 | 85.1 | 204 | 185 |

### Caveats (read before using the numbers)
1. **CUDA backward passes are invisible to coverage.py.** Autograd runs CUDA backward on its own device thread,
   which coverage.py does not trace. So the backward bodies of `_FusedBasis`, `_FusedKernelRead`, `_NodeMeans`,
   `_SegmentSoftmaxRead` and `physicsnemo::isla_geo_region_bwd` read as "never run", although tests run them on CUDA
   (for example `test_isla_tree_read.py:1468`, `:1435`, `:1669`, `:917`; `test_isla_kernels.py:110`). They are about
   93 of the 185 never-run lines. Fix for any later GPU coverage run: call
   `torch.autograd.set_multithreading_enabled(False)` in a conftest under coverage.
2. Code that runs only inside a compiled graph is traced symbolically by Dynamo, so coverage.py does not see it.
   Compiled regions count only through their eager runs.
3. Under coverage, the GPU job's compile groups hit their 1560 s guard. Not measured: `test_isla_compile`
   fullgraph `[scales-*]` and `[global_source-*]`, and 15 compile tests in `test_isla_kernels.py` (compile_blocks and
   fullgraph). All of them pass at 5b70c8686 in MAIN-MERGE's GPU job 376126 (709 passed, 1 skipped). They are compile
   tests, so they would add little line coverage (caveat 2). The likely exception is `model.py:1384-1389` (the
   compiled block reads).
4. Branch coverage is lower than line coverage mostly for the same reasons: guard and fallback branches.

### Lines no run reaches, apart from the caveats above (from `uncovered.json`)
| module | where | what it is | risk |
|---|---|---|---|
| inputs.py | `_InputSet._features` 211-212 | pairwise dot products when one condition has **more than one rank-1 boundary vector** (P > 1) | **real feature path never run** |
| geometry.py | `_ball_candidates` 1114-1117 | the second pass when a query's ball holds more candidates than the first pass allowed | **an exactness path of the geometry search never run**; it moves with LIB-MOVE P6 |
| geometry.py | `_nearest_cells` 1258; `rms_pairwise_distance` 123 (no measures); `_min_norm_points` 634; `_centroid` 1769; `_closest_features` 2132 | deep-tree stack guard; unweighted gauge; edge branches | medium (guards) |
| model.py | `_validate_inputs` 1411-1445 (8 raises), `_forward_tensors` 2063 | input validation of the batched routine | medium: untested errors |
| model.py | `_init_scales` 1072/1083/1116/1131; `_case_scales` 1547 | scale-declaration errors; a zero velocity source | medium |
| model.py | `_boundary_tokens` 2199 and 2203-2206 | a gauge computed when none is passed; boundaries without measures (count-weighted quadrature) | medium: an untested input mode |
| model.py | `_hint_token_axis` 418-419; `_log_count` 450 (deleted by ISLA-API); resolver cache returns 1286/1329/1360; `_apply_geo_kernel` 1399; `_case_inputs` 1900 | compile-only or dead after 0.13.0 | low |
| scales.py | `_check_local` 56/66/76/99; `check_output_stats` 157/169/174 | schema errors a recipe config can trigger | medium: untested errors |
| tree_build.py | `forest_tables` 287 | fallback when the depth bins overflow | medium (moves with P1) |
| tree_walk.py | `walk_pairs` 300-305 | empty forest or empty queries; too large for Warp, so None | medium (moves with P2) |
| tree_read.py | 500 (`n_pairs_cell`), 618/654 (chunked-basis edges), 961 (fused read's output dtype under autocast with float32 inputs) | edges | low to medium |
| fused_basis.py | `unsupported_reason` 96/107/109/111; `_sm_count` 457-460; `_computed_gelu_is_pytorchs` 448 | why the fused basis falls back (no Triton; fp16 or no autocast cache; not 2-D or 3-D; widths not a power of two) | medium: fallback choices untested on CUDA |
| fused_read.py | `fused_kernel_read` 622 | dtype-mismatch raise | low |
| geo_kernel.py | `_check` 628-659; `_fwd_block`/`_bwd_block` 686-701; `fused_geo_region` 990-999 | input guards; launch-size branches for S > 256 / 512; non-bf16 autocast raise | medium: S > 256 never run on the kernel |
| segment_softmax.py | `segment_softmax_read` 343 | default `scale=None` | low |

### Recipe tests
Over the recipe's `src`, the recipe tests reach 60.7 % of lines and branches together (`results/cpu_376354/recipe_cov.json`).
`train.py` reaches 45 % and `infer.py` 43 %; the transforms reach 86 %. On ISLA's modules they reach 71.6 % of
non-kernel lines (table above). The recipe's 2 failures (`test_campaign_c_hooks.py`) come from the environment. They
fail the same way at the base in MAIN-MERGE 376127, and pass locally: `DistributedManager.initialize_env` reads
`RANK`, which an sbatch shell without srun does not set.

## Runtime, skips and slow tests (CPU plain run, 8 threads, job 376354)
| file | tests | CPU pass | CPU skip | CPU s | GPU pass (cov run) | GPU s (cov run, inflated) |
|---|---|---|---|---|---|---|
| test_isla_compile | 14 | 8 | 6 | 795.3 | 10 | 2442.3 |
| test_isla_conditions | 35 | 35 | 0 | 79.6 | 35 | 260.2 |
| test_isla_interior_geometry | 82 | 76 | 6 | 17.2 | 81 | 19.3 |
| test_isla_tree_read | 128 | 81 | 47 | 12.8 | 128 | 137.3 |
| test_isla_kernels | 38 | 3 | 35 | 12.4 | 23 | 39.5 |
| test_isla_sampling | 16 | 16 | 0 | 11.5 | 16 | 31.9 |
| test_isla | 52 | 52 | 0 | 7.3 | 52 | 6.9 |
| test_isla_geometry_thin | 41 | 21 | 20 | 4.5 | 41 | 16.0 |
| test_isla_geometry_fast | 120 | 20 | 100 | 1.8 | 120 | 13.7 |
| test_isla_query_encoding | 3 | 3 | 0 | 1.7 | 3 | 1.5 |
| test_isla_heads | 21 | 20 | 1 | 1.2 | 21 | 16.0 |
| test_isla_geometry_degenerate | 34 | 18 | 16 | 1.1 | 34 | 4.5 |
| test_isla_scales | 17 | 17 | 0 | 1.1 | 17 | 1.4 |
| test_isla_release | 30 | 30 | 0 | 1.0 | 38 | 2.1 |
| test_isla_token_padding | 5 | 5 | 0 | 0.5 | 5 | 0.6 |
| test_isla_geometry_band | 23 | 13 | 10 | 0.5 | 23 | 1.7 |
| test_isla_global_inputs | 10 | 10 | 0 | 0.4 | 10 | 0.4 |
| test_isla_nd | 6 | 6 | 0 | 0.2 | 6 | 0.2 |
| test_isla_surface_conditions | 7 | 7 | 0 | 0.2 | 7 | 0.2 |
| test_isla_inputs | 6 | 6 | 0 | 0.2 | 6 | 0.2 |
| test_isla_names | 7 | 7 | 0 | 0.1 | 7 | 0.1 |
| test_isla_routing | 2 | 2 | 0 | 0.1 | 2 | 0.1 |
| test_isla_prototypes | 4 | 4 | 0 | 0.1 | 4 | 0.1 |
| test_isla_perf | 1 | 1 | 0 | 0.0 | 1 | 0.0 |
| **total** | 702 | 461 | 241 | 950.6 | 690 | 2996.1 |

Slowest 25 tests on CPU (plain run):

| s | test |
|---|---|
| 166.7 | test_isla_compile::test_the_in_model_boundary_draw_compiles_once_across_the_budget |
| 133.8 | test_isla_compile::test_a_warm_compile_cache_reuses_the_graph_across_the_budget |
| 118.0 | test_isla_compile::test_whole_model_compiles_fullgraph_with_dynamic_shapes_and_matches_eager[inputs-cpu] |
| 113.8 | test_isla_compile::test_whole_model_compiles_fullgraph_with_dynamic_shapes_and_matches_eager[interior-cpu] |
| 102.1 | test_isla_compile::test_whole_model_compiles_fullgraph_with_dynamic_shapes_and_matches_eager[scales-cpu] |
| 85.5 | test_isla_compile::test_whole_model_compiles_fullgraph_with_dynamic_shapes_and_matches_eager[global_source-cpu] |
| 78.4 | test_isla_conditions::test_a_change_of_condition_subset_compiles_once_per_subset |
| 53.5 | test_isla_compile::test_whole_model_compiles_fullgraph_with_dynamic_shapes_and_matches_eager[surface-cpu] |
| 21.8 | test_isla_compile::test_whole_model_does_not_guard_on_the_key_order_of_its_inputs |
| 9.2 | test_isla_kernels::test_compile_blocks_matches_eager[cpu] |
| 6.4 | test_isla_sampling::test_one_compiled_graph_for_every_cell_count |
| 4.9 | test_isla_sampling::test_horvitz_thompson_measure_sums_are_unbiased |
| 3.3 | test_isla::test_every_parameter_receives_a_gradient[interior] |
| 3.1 | test_isla_interior_geometry::test_closed_surfaces_take_the_sign_of_the_exact_winding_number[torus] |
| 2.9 | test_isla_kernels::test_compile_blocks_see_an_aligned_token_count[surface-cpu] |
| 1.5 | test_isla_query_encoding::test_query_encoding_is_continuous_at_the_surface[3] |
| 1.4 | test_isla::test_frame_is_sampling_consistent |
| 1.3 | test_isla::test_query_independent_interior_contracts |
| 1.1 | test_isla_interior_geometry::test_closed_surfaces_take_the_sign_of_the_exact_winding_number[sphere] |
| 1.1 | test_isla_tree_read::test_tree_read_pairs_basis_from_the_plan_is_exact_for_a_batch[3] |
| 1.0 | test_isla_interior_geometry::test_float32_points_within_the_bound_are_on_the_wall[torus] |
| 0.9 | test_isla_interior_geometry::test_on_wall_direction_in_float32_is_the_float64_one[torus] |
| 0.8 | test_isla_interior_geometry::test_two_dimensional_search_chunks_over_the_queries |
| 0.8 | test_isla_interior_geometry::test_tie_directions_take_the_exact_least_norm_point[torus-dtype0] |
| 0.8 | test_isla_interior_geometry::test_tie_directions_take_the_exact_least_norm_point[raw torus-dtype1] |

CPU skip reasons (count):

- 120: CUDA
- 82: CUDA and Triton
- 35: CUDA kernels
- 2: compares CUDA with CPU
- 1: fused kernel needs CUDA
- 1: set ISLA_AF_POSED_SAMPLE to a forward-kwargs .pt saved by the posed-inference reproduction

CPU total 951 s; tests over 10 s: 9 taking 874 s; the rest 77 s.

On GPU, the whole directory takes 28-30 min on one GB300 (MAIN-MERGE 376126: 1818 s; KERNEL-TESTS 368568:
1714-1761 s). `test_isla_compile.py` alone takes about 1,300 s of that. Times under coverage are inflated 2.5x on CPU
and 3x on GPU for the compile tests (2477 s against 981 s on CPU).

## Flaky signs
None found at the same code. QUALITY's sample of suite logs under `scratch/` found three things: the
environment-dependent recipe pair above (it recurs in 4 lane reports); 5 `test_isla_kernels.py` failures that were a
real regression (token padding, 921472488, fixed in 396459aec); and a conditions compile test that fails only under
a forced `fast_plugin`. Determinism the tests depend on: one CPU thread in two tests (`test_isla_tree_read.py:1276`,
`test_isla_geometry_fast.py:164`), and CUDA `index_add` atomics absorbed by tolerance (`test_isla_release.py:243-258`).
