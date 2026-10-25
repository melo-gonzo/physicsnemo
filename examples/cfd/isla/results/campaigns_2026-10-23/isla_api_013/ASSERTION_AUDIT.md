# ISLA-API 013: assertion audit (fc6ed19d0..7b60bdf86, test code only)

Read-only review. No tests run; every finding comes from the source at 7b60bdf86 (paths are under `test/experimental/models/isla/` or the recipe's `tests/`).

## Findings

| # | Sev | Where | Finding | Evidence | Smallest fix |
|---|-----|-------|---------|----------|--------------|
| 1 | P1 | recipe `tests/test_aerozoo_transforms.py:309` | `test_boundary_scales_and_log_reynolds_are_isla_s_bitwise` builds `ISLA(..., boundary_tokens=16)`. That keyword was renamed `n_boundary_tokens` in 7c957ac7e, so this test raises `TypeError` at HEAD and fails CI. | The `__init__` at model.py:705-719 is keyword-only with no `**kwargs`, and `Module.__new__` calls `sig.bind_partial`, which raises on an unknown keyword. The test came from the RE branch, which branched from 092ee6a0f before the rename. WRITER_RE_REPORT says "`boundary_tokens` is untouched", and its "14 passed" was a run on that old base. `git grep` finds no other stale `boundary_tokens=`, `dim=`, `reynolds=`, `velocity_scale` or `role` in the tests or configs. | Change it to `n_boundary_tokens=16`. |
| 2 | P2 | `test_isla_token_padding.py:127-143` | `test_an_aligned_token_count_runs_no_padding[interior]` is weaker than before. The old check was that no `constant_pad_nd` runs anywhere in the forward and backward. The new one is the token count seen by `blocks[0]` (`[aligned+ALIGN, aligned]`). It does not cover later blocks, the backward or the read's own padding. | The reason is recorded in DELETED_TESTS.md: the forest build runs `constant_pad_nd` of its own. The surface parameter keeps the op-level check. | Acceptable as recorded. To strengthen it, hook every block, not only `blocks[0]`. |
| 3 | P2 | `test_isla_scales.py:609` | `test_a_global_source_adds_no_direction` is an interior similarity contract, but it builds its own ISLA without `live(...)`. The read's zero projection therefore adds nothing there. | C3 says that `live` reaches test_isla_scales.py through `_model` (line 117), but this test does not call `_model`. It is no regression against base, which had no read. | Wrap it in `live(...)`, with `_tree_read_theta = 0.0` if needed. |
| 4 | P2 | recipe `tests/test_isla_configs.py:553` | `assert raw.model.interior is True` sits inside `if raw.model.get("interior", False):`, so it always holds. The real content is the `cfg.model.interior` check and the two new `"query_independent" not in` checks. | Read from the source. | None needed. You could drop the tautology. |
| 5 | P2 | `test_isla_conditions.py:405` | With `[isla]` removed, nothing tests the gradient of a rank-2 tensor map when its condition has no cells (that parameter was the only one with the outlet's `T`). | Noted in DELETED_TESTS.md. The interior task now refuses rank-2 inputs; the surface task still takes them but has no such test. | A follow-up could add a surface parameter. |

## Checklist

1. **Tolerances:** none were loosened. Every changed atol/rtol is unchanged or tighter.
   - `test_log_reynolds_is_written_raw_and_standardized`: the float32 default tolerance became float64 `atol=1e-15, rtol=0`.
   - A refusal `match` was narrowed: `"got None|reynolds"` became `"got None"`, and `"density"` became `"'physics.density'\) must be positive"`.
   - The refine checks keep `atol=1e-10, rtol=0`.
   - The `_assert_close` `rel=3e-2` is unchanged.
   - No xfail or skip markers were added or removed.
   - New tolerances: `re_z` at 1e-5 in the AirfRANS similarity test; the `live` test's threshold of `> 1e-3` and its bitwise restore.
2. **Deletions:** every deleted test, parameter, golden and in-test assertion appears with a reason in DELETED_TESTS.md or DELETED_TESTS_RE.md:
   - `isla_interior_v1.pth`, `CONFIGS["interior"]` and its parameters;
   - the form-load and form-storage tests, `_boundary_read_parameters`;
   - `test_query_mass_refinement_invariance` and the interacting contract test;
   - the `query_independent` parts of the defaults test;
   - the compile and padding `[interior]` parameters, conditions `[isla]`, scales `[isla-*]`;
   - every `z` assertion and the `reynolds`/viscosity refusals;
   - `m.reynolds`, `gs["nu"]["role"]`, `velocity_scale: False`.
   The checkpoint "every argument stored" assertion was moved into the round-trip test, which stays bitwise on fully random weights. The `isla_default_v1.mdlus` golden was regenerated for 0.13.0. Its output golden is unchanged, so the outputs are held fixed. Nothing is missing.
3. **Changed assertions:** all are at least as strong, except the padding check in finding 2, which is recorded.
   - `ValueError("interior=True")` became `TypeError("unexpected keyword argument 'query_independent'")`, now asserted for both tasks.
   - `qt.logw is None` became `not hasattr(qt, "logw")`.
   - The `query_chunks` refusal moved from an interacting model to a surface model.
   - `query_independent is True` became `kernel_reads is not None`.
   - The nd parameter count `+129` became `+128+5776+2*6464`, worked out in a comment.
   - The `z` hand value moved to recipe tests that compare bitwise.
   - The draw tests now compare every drawn token leaf, where they used to compare the model outputs, and gain a token-count assert. This is necessary because ISLA-DTT does not trace fullgraph, and it is recorded in C3.
4. **Strict xfail:** `test_on_wall_direction_at_quad_diagonals` (test_isla_interior_geometry.py:1335-1345) is still `strict=True`, with the same reason and body. The branch diff does not touch it. It tests `exact_boundary_geometry` alone, so the model-form change cannot affect it.
5. **`live(model)`** (`_meshes.py:191`) is used in all of these:
   - test_isla.py: gradient[interior], `test_interior_task_contracts` and its bf16 model, geometric_scale, boundary_scalars, frame_sampling, scalar_only;
   - through `_model` in global_inputs, heads, names, nd, scales and conditions;
   - routing's interior model.
   The dedicated test `test_live_makes_the_boundary_read_contribute` (test_isla.py:362) checks three things: the projection starts at zero; the outputs move by more than 1e-3 under `live`; zeroing it again gives the fresh outputs bitwise.
   - **The two `_tree_read_theta = 0.0` refinement checks are principled, not a weakening** (test_isla.py:326 and test_isla_conditions.py:228).
     - Refinement invariance is a property of the exact read. At θ=1, Barnes-Hut changes its forest when the boundary is refined.
     - Before this branch, the DTT refine check at θ=1 passed only because the read was zero, so it was hollow. It now really tests the read, at the unchanged 1e-10.
     - Both sides of each comparison run at θ=0 (test_isla.py no longer compares against the θ=1 `base`).
     - The other contracts stay at the default θ.
     - θ=0 refinement already has a precedent, `test_tree_read_similarity_refinement_and_measure_scale_at_theta_zero`, and the θ>0 accuracy is tested by `test_tree_read_error_falls_with_theta_on_a_smooth_boundary`.
6. **Other:**
   - No skips were added.
   - Removed CUDA coverage: `test_whole_model_compiles...[interior-cuda]` (recorded). The max-autotune CUDA test is kept but now compiles only `_case_inputs` (recorded in C3).
   - The whole-model compile no longer covers `prediction_points` passed as a point-cloud `Mesh`, because no interior model compiles whole.
   - I found no test that can no longer fail.

Merge verdict: BLOCK (finding 1 only; one-word fix)

{"verdict": "block"}

## Lane's resolution (2026-10-25 01:45 EDT)
- Finding 1 (P1): fixed in b64e01981 (`n_boundary_tokens=16`); recipe suite 378081 then has only the two pre-existing
  campaign_c_hooks failures.
- Finding 3 (P2): fixed in 7c88b2e1f (`live(...)` at theta 0 in test_a_global_source_adds_no_direction; test_isla_scales.py
  15 passed locally).
- Findings 2, 4, 5: left as recorded (2 and 5 are in DELETED_TESTS.md; 4 is a harmless tautology).
