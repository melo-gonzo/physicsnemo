# COMPILE-FLAG: torch 2.13's process-wide `is_compiling()` and the ISLA final rerun

Status: DONE (read-only bug hunt; written 07:03 EDT 10-26). Code: origin/psharpe/isla fa1cab63d. The probes used a git-archive extraction in `src/` (deleted afterwards; recreate with `mkdir src && git archive fa1cab63d | tar -x -C src`).
Torch: 2.13.0+cu130 (`.venv`), with 2.12.0+cu130 from `~/gh/physicsnemo-isla-field-schema/.venv` as the reference.
Probes ran on CPU under the local_heavy lock, one at a time, with PYTHONPATH=td0153:src. Output files sit beside each script.

## 0. Who reads the flag in a thread that does not compile, during the rerun
- In the rerun only the training thread compiles: ISLA lanes run `compile=false model.compile_blocks=true`
  (LANES.tsv), so `torch.compile(block.forward)` / `torch.compile(block.read)` (isla/model.py:1335, :1347) compile there.
  Tensile2d (eager) and the GT/T baselines compile nothing, so 2.13 changes nothing in their processes.
- Nothing else compiles. No datapipe, transform or recipe loader code calls `torch.compile`. The optimizer is compiled
  only when `cfg.compile` is set (train.py:1232), and it is false. Dynamo compiles are serialized by
  `convert_frame.compile_lock`.
- Other threads: MeshDataset's worker threads, the IOPump thread (datapipes/io_pump.py:119) and the DataLoader's prefetch
  pump. Transforms run on CUDA after `data.to(device)` (mesh_dataset.py:201/339).
- So the affected code is whatever those threads call: `physicsnemo.mesh` (33 guards), recipe transforms that call it,
  ISLA helpers called from the loader (`ComputeBoundaryScales`: isla geometry/scales, which have **no** guards), and
  tensordict (td0153, 119 guards; outside the brief, but Mesh is a tensorclass).
- `physicsnemo/datapipes` itself has 0 guards. The recipe's single guard (utils.py:419, `align_target_measure`) runs in
  loss and metrics on the training thread.
- Exposure: while the training thread compiles. RERUN-ENV measured that another thread read True 1,541 of 10,716 times
  (`isla-rerun-env-2026-10-26/logs/iscompiling_probe.txt`). Compiles run through about step 40 (compile_blocks
  recompiles on new shapes), and again on any later recompile.

## 1. Call sites (Q1). Risk is for the 2.13 rerun; "after fix" means #2172 with `is_dynamo_compiling` in `_cached` and `set_cell_measures`

| # | site (origin/psharpe/isla) | what reading True in a loader thread does | shown by | risk now | risk after the #2172 fix |
|---|---|---|---|---|---|
| 1 | mesh/mesh.py:2710 `Mesh._cached` (all cached geometry: cell areas, centroids, normals, point normals, curvatures) | Cache bypass. Geometry is recomputed from the current vertices instead of carried through `translate`/`scale`/`rotate`. Effects: the HiLift crash at measure.py:216; different `DropDegenerateCells` decisions (drops a cell once its float32 area collapses to 0); area-weighted sampling probabilities and normals change at rounding level | RERUN-ENV hl_repro; probe_physicsnemo case_cache (2.13 `cached: False`, 2.12 `True`) | **P0** crash, and different data | fixed |
| 2 | mesh/calculus/measure.py:213 `_transfer_cell_measures` zero-area check | Check skipped. A zero-area cell's nonzero measure is silently multiplied by 0 (its denominator is set to 1) instead of raising | probe_physicsnemo case_measure_check: 2.12 raises ValueError; 2.13 with a compile in flight gives `measure after scale = [0.0]` | **P0 wrong numbers**: the same slivers as #1 lose their quadrature weight silently whenever `scale`/`transform` runs while the flag is up. 2.13 runs that did not crash still trained on this | P1. The check is off in loader threads during compiles, but with the cache kept its precondition never arose on 2.12 data (0 of about 56,000 collapsed cells, HILIFT_213 §2) |
| 3 | mesh/calculus/measure.py:110 `set_cell_measures` alias check | The `.clone()` is skipped, so the stored measure can alias the cached `cell_areas` tensor if the caller read it before the flag rose | code reading; not reproduced in the pipeline | P2 (no in-place write to either tensor found on the loader path) | fixed |
| 4 | mesh/transformations/geometric.py:679 `transform` point-normal propagation | For 2-manifolds without `assume_similarity`, the cached point normals are dropped (recomputed later) instead of rotated | code reading. The only loader caller without `assume_similarity` is the recipe's `FixedRandomPose` (domain_transforms.py:617); RandomRotateMesh, rotate and scale pass it | P2 for the rerun (only `*_posed` datasets use FixedRandomPose; no rerun lane does) | P2. In posed lanes it would be a P1 rounding-level normals change |
| 5 | mesh/mesh.py:1454 `Mesh.merge` | Skips the one-mesh shortcut `return meshes[0].clone()` and stacks `global_data`, so each leaf gains a leading dim of 1 | probe case_merge_single: `re.shape=()` on 2.12, `(1,)` on 2.13 in flight | P2: the rerun's merges have ≥ 2 parts (MergeBoundaries: 4; domain_transforms.py:2208 guards `len(parts)==1`) | P2 |
| 6 | mesh/sampling/random_point_sampling.py:123 | `alpha != 1` raises NotImplementedError in eager code | probe case_dirichlet | P2: no rerun config sets alpha. A latent P0 for anyone sampling with alpha≠1 in a loader | P2 |
| 7 | domain_parallel/shard_utils/halo.py:599 | p2p halo exchange raises "not supported under torch.compile" in eager | code reading | P2 (no domain parallelism in the rerun) | P2 |
| 8 | tensordict utils.py:1262/1291, tensorclass.py:4075 (`_as_context_manager`: `to`, `unsqueeze`, ...) | `_last_op` holds a strong ref instead of a weakref, so an op's output keeps its input alive | probe_tensordict: 2.13 in flight `True`, 2.12 `False` | P2. On the loader path the GPU mesh from `data.to(device)` keeps its host mesh alive only until the first transform replaces the object | same (td issue) |
| 9 | tensordict utils.py:1044 `_cache_while_locked`, :1075 `_copy_cached_result` | Cache bypass on locked tensordicts (performance). A flag flip between the two reads can hand out the cached object uncopied | code reading only | P2 | same |

Harmless, grouped (P2):
- **Validation that is merely skipped** (only bad input would raise; the rerun's data passes them on 2.12):
  - mesh.py:575, :2928, :2945, :3054
  - measure.py:150
  - geometric.py:384, :605, :765
  - random_point_sampling.py:98
  - integration.py (8 sites)
  - neighbors/_adjacency.py:80
  - tessellation/triangulate.py:206
  - domain_mesh.py:190
  - spatial/cluster_tree.py (4 sites)

  Probe case_bad_cells: a Mesh with float cells raises TypeError on 2.12 and is built silently on 2.13 in flight.
- **Branches that do not matter here:**
  - mesh.py:123 `_check_geometry_values`: integer points only.
  - spatial/bvh.py:691: CPU query chunking. Loader transforms run on CUDA; on CPU the traversal would be unchunked,
    with the same result and more memory.
  - torch's own nn/modules/module.py:1880: exception path with always-call hooks.
- **Training-thread-only sites** are unaffected, because their thread is the one compiling and the flag is down outside
  its sessions:
  - isla/model.py (13), including :1331/:1343, which choose the compiled or eager block, and isla/tree_build.py:83.
  - recipe utils.py:419; distributed/fused_all_reduce.py:86.
  - every physicsnemo/models (68), diffusion (28), nn (50) and other experimental (49) site.
  - torch.optim (`_get_value`, `_stack_if_compiling`, the capturable and CPU-step checks, the foreach fallback); DDP
    `_lazy_init` (distributed.py:1748).
  - The optimizers would change path (tensor instead of `.item()` step values, no foreach) only if another thread
    compiled during `step()`. Nothing does in the rerun.

Bottom line for the rerun: #2172's fix removes the only P0 (sites 1, and with it 2's precondition). After it, what is
left in loader threads during compiles is skipped validation (P1 for site 2, P2 elsewhere) and tensordict memory and
performance details (P2). Any ISLA compile_blocks run on 2.13 **before** the fix (the 2.13 smokes and goldens) trained on
data that differs from 2.12's: through site 1 (and silently zeroed sliver measures through site 2, where it did not
crash).

## 2. Is the process-wide behaviour intended? (Q2)
- **The session flag is intended; reaching other threads looks unintended.**
  - PR pytorch/pytorch#184614 "Set is_compiling flag for the whole torch.compile session" (author ssam18, merged
    2026-06-01 by jansel; fixes #184547 "`torch.compiler.is_compiling` doesn't work inside wrapper subclasses").
  - It is in the v2.13.0 release notes: "Set the `is_compiling` flag for the whole `torch.compile` session (#184614)".
  - Its stated goal is code that "runs inside a compile session but is not directly Dynamo-traced (wrapper subclass
    `__torch_dispatch__` invoked by AOTAutograd, custom backends, make_fx called from inside the session)". All of that
    runs in the compiling thread.
  - The PR and the issue never mention threads. It sets the module global with a save/restore context manager
    (`torch/compiler/__init__.py` `_compile_session_context`, entered in `convert_frame.py` `compile_inner`).
- **Docs:** the `is_compiling()` docstring says "Indicates whether a graph is executed/traced as part of torch.compile()
  or torch.export()". It says nothing about threads; its example is the per-call `if not is_compiling():` idiom.
  `is_dynamo_compiling()` is documented as "only set to True when TorchDynamo is used". Neither is called thread-local.
  The global `_is_exporting_flag` (non-strict export) has always been process-wide.
- **Still on main:** torch/compiler/__init__.py:625 and convert_frame.py:1768 on pytorch main, fetched 10-26.
- **Known?** I found no issue about threads. I searched pytorch/pytorch for is_compiling with thread, thread-local,
  "compile session" and "_compile_session_context", and everything created since 2026-06-01. The nearest is RFC
  #186449, Contextvars support in Dynamo, which discusses thread safety in general.
  - tensordict 0.15.3 already works around it: tensorclass.py:3996 says "`is_compiling()` stays true in every thread
    while any thread compiles ... must not take the branch meant for Dynamo tracing", and uses `is_dynamo_compiling()`.
- **Repros:**
  - `repro_torch_threads.py` → `repro_torch_threads.txt`: the other thread reads True on 2.13 and False on 2.12.
  - `repro_stuck_flag.py` → `repro_stuck_flag.txt`, a second bug from the same global save/restore: a compile session
    overlapping a non-strict `torch.export` in another thread leaves `is_compiling()` **True for the rest of the
    process** on 2.13 (False on 2.12). Not reachable in the rerun, which does not export.
- **Draft issue (not filed; Peter approves):** `PYTORCH_ISSUE_DRAFT.md`, in the bug template, with both repros and
  `collect_env_213.txt`.

## 3. Recommended fix pattern (Q3)
- **Use `torch.compiler.is_dynamo_compiling()`** for every guard in code that Dynamo traces and that a loader thread can
  also run eagerly. That is all of physicsnemo/mesh: `_cached`, `set_cell_measures`, measure.py:213 and :150,
  geometric.py:679, `Mesh.merge`, `random_point_sampling`, and the validations.
  - It is thread-correct: it is a constant False in eager, and Dynamo folds it to True in the frame it traces
    (probe_thread_check: `(True, True)` in the traced frame on both torches).
  - On 2.12 it behaves exactly as `is_compiling()` did for Dynamo; RERUN-ENV's fixtests passed 55 tests on both torches.
  - Best as one helper in physicsnemo/core, e.g. `is_tracing()`, rather than 33 hand edits, so that the export question
    below is answered once.
- **Where `is_dynamo_compiling()` alone is wrong:**
  - **Non-strict `torch.export` (and dynamo-based ONNX export):** `is_dynamo_compiling()` reads False there, where
    `is_compiling()` read True through `_is_exporting_flag`.
    - Validation guards would then run `bool(tensor)` / `.item()` on fake tensors (data-dependent errors).
    - `_cached` would read and write the input mesh's `_cache` with traced tensors.
    - Use `is_dynamo_compiling() or is_exporting()` wherever export must work. `is_exporting()` is also process-wide,
      but it is up only during an export, as on 2.12.
    - No physicsnemo test exports mesh or ISLA code today (git grep). Some models carry ONNX tests (legacy TorchScript
      exporter). Their guards run only on the compiling thread, so **leave models/, nn/ and diffusion/ alone**.
  - **Tensor-subclass dispatch and custom-op fake kernels** (ShardTensor `__torch_function__`/`__torch_dispatch__`,
    halo.py inside an autograd.Function under AOT). These are exactly #184614's target: `is_dynamo_compiling()` is
    False during AOTAutograd tracing.
    - Keep shard_tensor.py's existing pattern (`_is_tracing`: is any operand a FakeTensor?).
    - Or use a thread check: `is_compiling() and (torch._guards.CompileContext.try_get() is not None or
      TracingContext.try_get() is not None)`. Both contexts are thread-local. probe_thread_check: True in the backend of
      the compiling thread, False in the other thread on 2.13, False everywhere on 2.12. It uses private API, so keep
      it in that one helper.
- `is_compiling()` with a thread check is only needed for those subclass and AOT sites. For everything the loader
  reaches, `is_dynamo_compiling()` (plus `or is_exporting()` if export matters) is simpler and needs no private API.
- tensordict's remaining guards (sites 8 and 9) are a tensordict follow-up of the same kind (Peter maintains it).

## Checked and not affected / not checked
- Checked:
  - every `torch.compiler.is_compiling()` / `torch._dynamo.is_compiling()` / alias in physicsnemo/ and the recipe
    (no other aliases or imports exist);
  - torch 2.13's eager-path users in optim, nn, autograd, amp, cuda, distributions, `_library`, `_ops`,
    utils/checkpoint and DDP;
  - which threads run during compiles; the rerun's lane configs and their transforms.
- Not checked:
  - tensordict's 119 guards one by one (I probed only the strong-ref one);
  - CUDA-side behaviour of the probes (all ran on CPU);
  - whether recompiles at validation time widen the exposure window in the real run;
  - the 2.13 golden/smoke numbers' sensitivity to sites 1 and 2.
