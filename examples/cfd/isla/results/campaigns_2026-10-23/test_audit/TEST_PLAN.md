# ISLA test plan: phased (TEST-AUDIT, written 23:22 EDT 2026-10-24)

The basis for this plan is in `AUDIT.md` (standards, the property map, quality flags) and `COVERAGE.md`. Peter's
direction (about 22:05, 10-24): "It's high time we start making sure that ISLA has a really good test suite"; tests
come before the library moves and the cleanup.

Rules for every phase:
- New tests use the 0.13.0 API: `n_spatial_dims`, no `query_independent`, interior means ISLA-DTT.
- One writer per worktree, each on its own branch from origin/psharpe/isla; the supervisor merges.
- New tests go in **new files** wherever possible, so that the three writers do not conflict.
- Before a writer calls a phase ready, it runs the whole ISLA directory on CPU (LHR cpu-short) and on a GB300 with
  every CUDA test (LANE_RULES), plus the recipe tests.
- Sizes are ESTIMATED in added test lines.

## Writers, worktrees, order

| writer | worktree / branch | phases | may start | waits for |
|---|---|---|---|---|
| W1 GOLDENS | `physicsnemo-isla-tests-golden` / `psharpe/isla-tests-golden` | P0 module groups, then P0 model groups, then P1 | **now**: the module groups touch only new files, and ISLA-API changes no number there (its DESIGN, "Goldens kept bitwise") | the model groups and P1 wait for ISLA-API's merge (constructor, `_QueryTokens`, the regenerated `.mdlus`) |
| W2 PROPERTIES | `physicsnemo-isla-tests-props` / `psharpe/isla-tests-props` | P2 | after ISLA-API merges (it edits the files ISLA-API converts: `_meshes.py`, conditions, heads, scales, test_isla.py) | ISLA-API |
| W3 KERNELS | `physicsnemo-isla-tests-kernels` / `psharpe/isla-tests-kernels` | P3 | **now**, in new files. The P3-10 compile replacements go on a 0.13.0 base | ISLA-API for P3-10 only |
| W1 (or the supervisor) | W1's worktree, after P0-P3 merge | P4 | last: it touches markers in every file | P0-P3 |

Calendar (ESTIMATED):
- 10-25: W1 and W3 start; ISLA-API merges.
- 10-26: W1's module goldens merge; W2 starts.
- 10-27: P1 and P2 merge; P3 merges 10-27/28.
- 10-28: P4.

Every check that a LIB-MOVE switch needs (the table at the end) is then in place before those switches, which come
after the 10-29 freeze. The cleanup comes after LIB-MOVE.

GPU estimate: W3 about 4 GPU-h of development runs. Each writer's pre-merge whole-directory run costs one 30-minute
node (2 GPU-h). Total about 10-12 GPU-h on short QoS.

---

## P0: characterization goldens (W1). About 1,300 lines; committed data at most 1 MB

Purpose: let every LIB-MOVE switch and every cleanup commit prove itself bitwise in eager, at module level, so that
a break is located at once.

Two layers, because bitwise hashes hold only on the machine that recorded them (BLAS and LAPACK `eigh`/`qr`, the
GPU model; the backbone golden already differs across CPU architectures):
1. **A/B harness: the proof of bitwise equality.** The harness lives in
   `test/experimental/models/isla/goldens/{cases.py, record.py, compare.py}`. `record.py --groups ... --device
   cpu|cuda --out DIR` writes, for each case, the int-view SHA-256 of every input, output and gradient, plus small
   float64 tensors. `compare.py A B` fails on any difference.
   - A LIB-MOVE switch or cleanup commit runs `record.py` at its base and at its head, on the same machine, **each in
     its own process**. This respects the custom-op rule: `isla_closest_candidates`, `isla_cumsum` and
     `isla_geo_region_*` register process-wide.
   - Run it on CPU with 1 thread, and on a GB300 for the CUDA paths.
   - Each record prints which version of each module ran (as `audit_cuts.py` does).
   - **How to read the records** (`DETERMINISM.md`):
     - CPU records are bitwise.
     - GPU records with kernels auto are bitwise, except the draw's `weights`. Compare those at 1e-15 relative
       until the draw is fixed, and compare `kept` first.
     - Where `kept` flips run to run, the A/B result is inconclusive: rerun it, never pass it.
     - Stages on the reference read are judged against a 5-run in-process family.
2. **Committed CI goldens: the regression guard.** `test_isla_goldens.py` with `data/goldens_v1.pt`.
   - Integer outputs exact: forest tables, walk lists, `PairPlan`, kept cells.
   - float64 outputs at 1e-12 relative to the tensor's scale, which holds across BLAS builds.
   - float32 at a stated bound.
   - Regenerated only by `python -m test.experimental.models.isla.goldens.record --regenerate`, which writes a
     manifest (commit, versions, CPU/BLAS, GPU). A missing file **fails**: no create-on-missing.

Case groups (inputs from `_meshes.py` and the existing test builders, seeded through `torch.Generator`; inputs are
stored or hashed, since they pass through library primitives):

| group | functions | paths | record at | lines |
|---|---|---|---|---|
| G-FOREST | `BoundaryForest.build` (reference and fast), `forest_tables`, `node_means` and its gradient | CPU Warp; CUDA | 5b70c8686 or later | 90 |
| G-WALK | `point_pairs`/`frame_pairs` (reference and fast, `count_only`), `walk_pairs`, `walk_range_pairs` | CPU; CUDA | same | 90 |
| G-SAMP + G-FRAME | `water_fill`, `pps_inclusion_probabilities`, `sample_boundary_cells` (kept, weights), `draw_inputs`, `canonical_frame`, `frame_coordinates` (including the skew-tie branch) | CPU f64/f32; CUDA | same | 120 |
| G-SEG | `PairPlan.build`; `segment_softmax_read` eager and Triton. Triton's `dk`/`dv` use atomics, so they get a stored-tensor tolerance, not a hash | CPU; CUDA fp32/bf16 | same | 80 |
| G-GEO | `exact_boundary_geometry` (d, u, dq, dp), the 6 outputs of `_closest_candidates[_fast]`, `w`; `run_start`, `canonical_cells`, `closest_weights`; 2-D and 3-D; the thin and degenerate sets | CPU reference; CUDA reference and fast | same | 200 |
| G-READ | `_TreeKernelRead` (live), `_KernelBasis`, `tree_pair_invariants`, `fused_kernel_read`, `fused_node_means`, `fused_kernel_basis`, `fused_geo_region`, `_geo_region` | CPU; CUDA | same | 160 |
| G-MODEL | `ISLA.forward` for surface, DTT (live), 2-D, a rich surface model (rank-1/2 boundary fields, rank-2 and odd outputs, scales); `query_chunks` over 3 ranges; every parameter gradient | CPU f64/f32; CUDA | **ISLA-API's 0.13.0 head** | 120 |
| G-PARTS | from G-MODEL's forward hooks: `_SliceBlock`, `_RoutingLogits`, `_QueryTokens`, `_assemble_output`, `_pseudovectors`, `scale_factor`/`velocity_moments`, `soft_log_length`, `rms_pairwise_distance`, `cell_second_moments` | as G-MODEL | 0.13.0 | 100 |
| harness | hashing, the manifest, the platform fingerprint, the tolerance fallback, a CPU run-twice self-check | — | — | 250 |

Also in P0, before LIB-MOVE P1/P2 switch (about 40 lines, in `test_isla_tree_read.py`, W1): in each `patch then
compare` test, **assert that the forced path ran**. Without it, a patch of `resolve_walk`, `resolve_tree_build` or
`forest_tables` that no longer reaches the moved code passes vacuously.

## P1: MOD-008a/b/c (W1, after ISLA-API merges). About 450 lines; data about 0.5 MB

Files: `test_isla_release.py` (extend), `data/` (new goldens).
- **008a.** Add a third configuration, `n_spatial_dims=2`. Make `with_custom_args` set every constructor argument to
  a non-default value (`boundary_tokens`, `compile_blocks`, `output_stats`, scales). Assert every public attribute,
  from a list kept beside the test and checked against `vars(model)` (non-underscore, non-module, non-parameter),
  so that a new attribute fails until someone lists it.
- **008b.**
  - Keep `default` and `dtt`, and add a float64 parameter per task at 1e-12.
  - Add `rich_surface` (boundary fields of rank 1 and 2, rank-2 and odd outputs, global scalars, `output_stats`),
    `dtt_2d`, and `dtt_draw` (about 2k cells, `boundary_tokens=512`, default θ, eval mode).
  - Add a `query_chunks` golden (outputs and gradients); the standard requires every public method.
  - On CUDA, use the `device` fixture with kernels `auto` and a documented tolerance.
- **008c.**
  - Commit `isla_dtt_v1.mdlus` with its output golden, and test it through `Module.from_checkpoint` (attributes and
    forward).
  - Round-trip a scaled model and a 2-D model through `.mdlus`.
  - Version refusal on every path: the metadata version through `Module.load` too, and the state-dict `_version`
    through `from_checkpoint` too. Add 0.12.0 to the refused list.
- **Regeneration.** One `--regenerate` script for every `data/` file, the backbone golden included. Remove the
  create-on-missing path.
- **Merge the duplicates**: one parametrized checkpoint round trip over schemas (heads, names, interior geometry),
  and one unknown-argument test.

## P2: properties and invariances (W2, after ISLA-API merges). About 750 lines

New file `test_isla_dtt_contracts.py`, plus edits in `_meshes.py` and the converted files. In order:
1. `_meshes.live(m, seed)`: randomize the read (`normal_(0, 0.3)`, as at `test_isla_tree_read.py:83-86`). Apply it in
   every interior test that ISLA-API converted: the conditions tests, heads :273, interior_geometry :76, nd :157,
   scales :208, padding [interior_dtt], and test_isla.py :317 and :419. Keep each test's tolerance, with rtol=0, at
   θ=0 where the contract is exact. (ISLA-API could do this in its conversion; reported to the supervisor at 23:20.)
2. `same()` gets the default `rtol=0`, and the contract tests' `allclose` calls get `rtol=0`. Measured safe
   (`probes/rtol_probe.jsonl`: 8 calls depend on rtol). Of those 8:
   - set the chunk-equality atol to 1e-11 (measured error 7e-12);
   - leave the on-wall least-norm and fold tolerances as they are; they are relative by design (say so in a comment).
3. Interior boundary-cell permutation, model level: renumber cells and vertices, with the read live. θ=0: atol 1e-12,
   rtol 0. Default θ: forest lists bitwise and outputs at 1e-12. Repeat with `boundary_tokens` below the cell count.
4. Query independence (`tree_read:666`) and chunk outputs (`:1738`) at rtol 0, atol 1e-11.
5. `query_chunks` with the draw active (`boundary_tokens=128`, training mode, seeded): outputs at 1e-11, gradients at
   rtol 1e-9.
6. Odd-parity and rank-2 outputs in interior covariance (heads' `OUTPUTS`), proper and improper rotations, 2-D and
   3-D, with the read live.
7. Model-level pose invariance with the draw subsampling (eval): rotate, reflect and translate case and points
   together; atol 1e-12.
8. 2-D surface contracts in float64 (`nd:113`), at atol 1e-11 and rtol 0.
9. One condition with two rank-1 boundary vectors (`inputs.py:211-212` never runs): covariance and gradient.
10. Model contracts on CUDA in float32, through the `device` fixture, at a stated bound: one rotation and one
    translation test per task.
11. One `@pytest.mark.slow` contract at the default width, slices and layers (about 2k cells, f64).
12. Errors and devices:
    - `pytest.raises` with a match for the untested user-facing raises: `_forward_tensors`' validations,
      `scales.py` schemas, `_init_scales`, the zero velocity source;
    - mixed devices refused, or documented;
    - the whole model run twice on CUDA, eager: bitwise, or a stated tolerance where `index_add` feeds in;
    - an fp16-autocast CUDA forward. It is expected to raise today (inferred from `geo_kernel.py:998`); if it does,
      that is a bug report for ISLA-API's successor, not a test to loosen.

## P3: kernels against references, and gradchecks (W3). About 850 lines; GPU tier

New files `test_isla_gradcheck.py` and `test_isla_kernel_refs.py`; additions to `test_isla_geometry_fast.py` and
`test_isla_kernels.py`.
1. `torch.autograd.gradcheck` in float64 on `exact_boundary_geometry` (d, u with respect to the query points and the
   vertices; generic points off the wall and off ties; 2-D and 3-D; CPU). On `fused_geo_region`'s custom op (CUDA
   fp64; it accepts fp64). On `_BasisSink` (CPU stand-in).
   - `torch.library.opcheck` on `isla_closest_candidates[_fast]`, `isla_cumsum` and `isla_geo_region_fwd/bwd`.
   - The Triton Functions refuse fp64 (`_FusedBasis`, `_FusedKernelRead`, `_NodeMeans`, `_SegmentSoftmaxRead`), so
     they keep the 2×-error rule against fp64 eager autograd, and add the backward with respect to every
     differentiable input.
2. Geo-region bf16 **backward** against the eager region under the same autocast (2×-error rule).
3. Geo region at S ∈ {257, 513, 1024} and with `FWD_BLOCK_N`/`BWD_BLOCK_N` overrides (fp64 1e-11).
4. 2-D cases in `geometry_fast._cases` (NACA, box ties, circle on-wall, fold): fast equals reference bitwise,
   gradients included.
5. Segment softmax:
   - replace fp32 1e-4 by the 2×-error rule;
   - add shapes (4, 8, 16) and (8, 4, 8);
   - a width that is not a power of two on CUDA must equal eager bitwise;
   - `logm.requires_grad` is either refused or documented (the Triton backward returns None for it).
6. fp16 autocast for the fused basis and fused read (the code claims it).
7. CUDA draw against CPU: kept cells bitwise, weights at 1e-12.
8. Paths that coverage shows never run:
   - the ball search's second pass (`geometry.py:1114`);
   - the deep-tree raise (`:1258`);
   - the tree build's depth-bin fallback (`tree_build.py:287`);
   - the walk on an empty forest or empty queries, and past its limit (`tree_walk.py:300-305`);
   - `fused_basis.unsupported_reason`'s branches;
   - `resolve_geo_kernel`'s disqualifiers **on CUDA**.
9. GPU coverage measured properly: `torch.autograd.set_multithreading_enabled(False)` under coverage (a launcher
   option, not a test).
10. **Replace the sampling-under-compile tests that ISLA-API deletes**: a fullgraph draw on a surface model across
    the budget, a warm-cache reuse test, and the GB300 `isla_cumsum` guard at about 1.4M cells. Needed before
    LIB-MOVE P4.
11. **Determinism tests** (`test_isla_determinism.py`, GPU tier; detail in `DETERMINISM.md`). Measured in job
    377000: the eager GPU forward with kernels auto is bitwise run to run, except the draw's float64 weights. The
    draw's CUDA `cumsum` wobbles by ulps, so a kept cell can flip when a hit lands on a line boundary: the likely
    cause of REYNOLDS-OUT's AeroZoo case 0. The reference tree read is also not deterministic on CUDA (about 2e-6).
    The tests:
    - the draw twice on CUDA, bitwise, with planted boundary hits (`xfail(strict=True)` until the draw is fixed);
    - the whole model twice (3-D and 2-D, surface and DTT, fp32 and bf16, forward and backward), bitwise with
      kernels auto;
    - the reference read on CUDA against its 5-run spread;
    - the whole model twice on CPU with one thread, bitwise (this one in the CPU fast tier).

## P4: test tiers (last). About 150 lines plus one launcher

- Mark the compile tests `@pytest.mark.slow`; `test/conftest.py` already registers the marker. They are 9 tests,
  874 of 951 s on CPU, and about 470 s after ISLA-API.
- Use `@pytest.mark.cuda`, or one `requires_cuda_triton` mark, in place of the ad-hoc skipifs.
- **CPU fast tier**: `-m "not slow"`, about 80 s on 8 cores. This is what physicsnemo's CI should run, if Peter
  agrees (decision 1).
- **CPU full tier**: everything on LHR cpu-short (about 16 min).
- **GPU tier**: the whole directory on one GB300, as one shared launcher under `$T/kernel/`. It is
  `lhr/gpu.sbatch` from this lane without coverage: 4 groups with the compile file split, about 8 min wall time
  instead of 30. It is the LANE_RULES "GPU tests" run.
- Hygiene:
  - restore TF32 state in `test_isla_kernels.py:393` (monkeypatch);
  - delete or replace the always-skipped `test_saved_posed_airfrans_case_runs`;
  - fold `test_isla_perf.py` into the blocks tests;
  - extract `_geometry_cases.py` from the geometry files' cross-imports (needed before LIB-MOVE P6);
  - make `test_campaign_c_hooks.py` independent of Slurm (`monkeypatch.delenv` of the SLURM variables, or set
    `RANK`/`WORLD_SIZE`).

---

## Which tests move with each LIB-MOVE phase (detail: `REPORT_P0_INVENTORY.md` §3)

| LIB-MOVE phase | must exist first (from this plan) | tests that move to the library (retargeted) | stay in ISLA (retarget imports or patches) |
|---|---|---|---|
| P1 tree build (`tree_build.py`) | G-FOREST; forced-path assertions; a bitwise test that `node_means` is unchanged when `levels` become ascending ids | `test_isla_tree_read.py::test_fast_forest_build_is_the_reference_forest` (:1150), `…_on_cuda` (:1236) and their helpers (:1023-1124); `…refuses_what_it_cannot_build` (:1206); the resolver half of `…auto_resolves_to_the_reference_off_cuda` (:1158) | `:297, :352, :381, :390, :410, :610`; `test_tree_read_kernel_is_the_tree_reads_one_execution_attribute` (:1177) |
| P2 walk (`tree_walk.py`) | G-WALK; forced-path assertions | `test_fast_walk_gives_the_reference_lists_in_the_reference_order` (:471), `…on_cuda…` (:976), `…opening_test…at_the_threshold` (:503), the library half of `…falls_back…past_its_limits` (:584); a new test of the torch path's order against ISLA's concatenation | `:432, :562, :610, :1007, :1844` |
| P4 sampling and frame | G-SAMP, G-FRAME; P3-10 (compile replacements); new frame tests (O(d) covariance, the skew-tie branch) | `test_isla_sampling.py` :77, :98, :110, :143, copies of :324 and :364, the library half of :166 | :56, :157, :196, :275 (model level), the multi-condition copies |
| P5 segment softmax and `PairPlan` | G-SEG; P3-5 | `test_pair_plan_is_the_histogram_plan…` (:879), `test_fused_segment_softmax_read_matches_eager` (:919) | the fused-read tests :1290, :1362, :1390, :1437, :1476, :1578, :1613, :1671 (retarget the `PairPlan` import); a pin that `_source_items` agrees between the library and `fused_read` |
| P6 exact geometry | G-GEO (stored inputs; per-GPU hashes for the fast path); P3-1, P3-4, P3-8; `_geometry_cases.py` | all of `test_isla_geometry_fast.py` except :193; all of `test_isla_geometry_band.py`, `_thin.py` and `_degenerate.py`; the geometry-only tests of `test_isla_interior_geometry.py` (:315-572, :886-1462) | `test_isla_geometry_fast.py:193` (the resolver); `interior_geometry` :76, :109, :124, :141, :165, :640, :664, :715, :739, :777, :816; `nd:157`; `tree_read:1738`; the `soft_log_length`, `rms_pairwise_distance` and `cell_second_moments` tests |

The cleanup after the moves: each commit runs `record.py` at base and head (CPU 1 thread, plus a GB300 run for
commits touching CUDA paths), `compare.py` bitwise, then the whole directory on CPU and GPU. The compiled path is
recertified separately, under the compiled-regions rule.
