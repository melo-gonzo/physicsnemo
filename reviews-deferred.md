# Reviews — consolidated findings (round-1 GeoPT data-gen + PR 2 recipe)

This document consolidates the findings from three read-only review
subagents (correctness, DRY-ness, maintainability) that ran on the
production + recipe code on the `exp-geopt-datagen-r1` branch. It is
the ledger for what landed, what is deferred, and what is verified-skip.

## 1. Summary of reviews run

Three read-only review subagents ran on this branch:

- **Correctness review** — checked algorithmic correctness of the
  GeoPT data-gen ports (`signed_distance_field`,
  `mesh_ray_intersection`, `constrained_walk`, `build_pretraining_sample`)
  against the released GeoPT reference at
  `external-repos/GeoPT/data_generation/GeoPT_PreTraining_Data.py`,
  plus the PR 2 recipe wiring (`TransolverPretrainBackbone`,
  `WalkSampler`, `geopt_pretrain.yaml`).
- **DRY-ness review** — flagged duplicated logic across the
  `pnm_pretraining/ops/*` files, the builder, and the recipe glue.
- **Maintainability review** — flagged complexity, unclear naming,
  branching, abstraction layers, and ease-of-future-change.

Findings were cross-referenced against the actual source where they
conflicted. Verdicts below — Tier 1 landed, Tier 2 deferred with
triggers, Tier 3 verified-skip.

## 2. Tier 1 — landed in this commit

| ID   | Description                                                                                                                  | Status  |
| ---- | ---------------------------------------------------------------------------------------------------------------------------- | ------- |
| T1.1 | Rename `hit` → `closest` in `_rejection_sample_volume_points` to match the `closest_points` semantics in the docstring.       | landed  |
| T1.2 | Hoist `0.99` sticking factor + `BASE_WALKS = 10` + `PERTURB_SIGMA = 0.05` to module constants with GeoPT line citations.      | landed  |
| T1.3 | Document `n_steps` coupling between `transolver_pretrain.yaml` and `geopt_pretrain.yaml` (Prong A only — Prong B abandoned).  | landed  |
| T1.4 | Factor `_normalize_mesh_indices` / `_build_warp_mesh` / `_check_vec3` into `ops/_warp_helpers.py`.                            | abandoned |
| T1.5 | Replace load-bearing `assert` on the surface-pin schema invariant in builder with explicit `RuntimeError`.                    | landed  |

Commit SHA: see the most recent commit on `exp-geopt-datagen-r1`.

### T1.3 Prong B — abandoned, with reason

The dispatch budgeted 30 minutes to try a Hydra `OmegaConf` resolver
that would auto-generate the `targets:` block from `model.n_steps`.
A `grep -rn "register_new_resolver"` over the recipe found **zero**
existing registration sites, meaning a clean landing requires:

1. Identifying the recipe's `@hydra.main` entrypoint (multiple
   candidates in `src/`).
2. Registering the resolver before any config compose (so it is
   visible to all consumers, not just the train flow).
3. Verifying the resolver does not interact poorly with the
   recipe's other interpolations (`${dp:...}`, `${out_dim}` etc.).
4. Adding test coverage for the resolver path.

That is a multi-touch change that ought to come with its own commit
and review. Per the dispatch rule "if it's painful, abandon Prong B
and ship Prong A only", abandoned. The two YAML warning blocks (Prong
A) document the coupling explicitly. Revisit Prong B as a follow-up
when the recipe acquires its first custom OmegaConf resolver for
another reason — the registration site can be reused.

### T1.4 — abandoned, with reason

Implemented and tested cleanly (all 19 ops tests passed), but the
call-site reduction was below threshold. After replacing
`_normalize_mesh_indices` / `_build_warp_mesh` / `_check_vec3`
inline-bodies with helper calls in both `mesh_ray_intersection.py`
and `constrained_walk.py`, `git diff --stat` reported

```
constrained_walk.py        | 49 ++++++++++++----------
mesh_ray_intersection.py   | 27 ++++--------
2 files changed, 34 insertions(+), 42 deletions(-)
```

— a net call-site reduction of **8 lines**, against the dispatch's
25-line threshold ("if T1.4 produces less than 25 lines of net code
reduction, the helpers aren't pulling their weight — keep the code
inline and skip the file"). Abandoned and reverted. The new helper
file would have been 80 lines (mostly SPDX header + module docstring
boilerplate), so the project-level delta would have been ~+72 lines
to save 8 lines of repetition across two call sites.

The duplication is real but small (one 4-line `wp.from_torch` +
`wp.Mesh` block, one 8-line indices-reshape branch, two 1-line
last-dim-3 checks per ops file). Revisit when a third op joins the
`pnm_pretraining/ops/` family — the case for shared helpers grows
with each new consumer.

## 3. Tier 2 — deferred, planned

Each item lists the trigger event that should cause us to revisit it.

### T2.1 — multi-worker DataLoader RNG (correctness W1)

**File**: `physicsnemo/experimental/pnm_pretraining/data/transforms.py:495-516`.

**Symptom**: `WalkSampler.generator` is a single CPU `torch.Generator`
attached to the transform instance. Under a multi-worker DataLoader
the workers fork a copy; all workers then sample from identical RNG
state, producing identical walk indices for the same `__getitem__`
call across workers. No correctness violation today (training is
single-worker), but a silent bias the moment the first multi-worker
training run is set up.

**Trigger**: first multi-worker training run is set up.

**Suggested fix**: detect via `torch.utils.data.get_worker_info()` in
`apply_to_domain` and re-seed on first call per worker, with a
worker-id offset so each worker draws an independent stream.

### T2.2 — generate_walks emission order diverges from GeoPT (correctness W4)

**File**: `physicsnemo/experimental/pnm_pretraining/ops/constrained_walk.py:497-500,555-605`.

**Symptom**: We emit walks interleaved as `(base[0], jitter[0,0], …,
jitter[0,8], base[1], jitter[1,0], …)`, while GeoPT's reference at
lines 585-608 emits sequential `(base[0..9], then jittered[0..89])`.
Numerically equivalent (each walk's data is identical given the same
seed), but a per-walk parity test against GeoPT would fail without
re-ordering.

**Trigger**: first per-walk parity test against GeoPT is written.

**Suggested fix**: either (a) match GeoPT's sequential layout in
`generate_walks` (one base loop, then one jittered loop), or (b)
expose `is_independent` and let the parity test re-order on read.

### T2.3 — `build_pretraining_sample` is 270 lines / 12 kwargs (maint #3)

**File**: `physicsnemo/experimental/pnm_pretraining/data/builder.py`,
the `build_pretraining_sample` function.

**Symptom**: pure readability. The function is correct and
well-commented but dense; reasoning about a single step requires
scrolling through the whole body.

**Trigger**: when the function gets its next major change (e.g. a new
sampling step, a swap of the alignment policy, or any addition that
forces the reader to understand the full data flow).

**Suggested split**: `_sample_surface_pre_alignment`,
`_assemble_interior_point_data`, `_build_domain_global_data`. Group
walk + alignment kwargs into a `BuildConfig` dataclass to drop the
12-kwarg surface area to ~3.

### T2.4 — `TransolverPretrainBackbone` not extensible to other backbones (maint #2)

**Files**: `physicsnemo/experimental/pnm_pretraining/models/`.

**Symptom**: `TransolverPretrainBackbone.__init__` hard-codes
`self.transolver.blocks[-1].ln_mlp2 = nn.Identity()` to strip the
stock Transolver's output projection. A future
`GeoTransolverPretrainBackbone` (and beyond) will need to repeat the
same head-replacement logic with the appropriate inner-block name.

**Trigger**: starting `GeoTransolverPretrainBackbone` (or any second
pretrain backbone variant).

**Suggested approach**: extract `_PretrainBackboneBase` ABC that
declares `_replace_output_projection(backbone)` as the policy hook
each subclass implements; the trajectory head + `out_dim` arithmetic
moves up to the base.

### T2.5 — supervise_step0 fp32 round-off mismatch (correctness W2)

**File**: `physicsnemo/experimental/pnm_pretraining/data/builder.py`
(builder emits `supervise_step0`); `constrained_walk.py` (M2 walk
arrays).

**Symptom**: builder emits `0.0` exactly on surface rows for
`supervise_step0` (we hard-zero them via `surf_supervise_step0 =
torch.zeros_like(surf_pts_post)`). The first per-step entry of the
M2 walk arrays — which is also a closest-point query at the original
surface position — produces values on the order of `O(1e-7)` because
the closest-point query does not exactly recover the input position
in fp32. Diagnostic only: no test currently asserts they agree, no
downstream consumer compares them.

**Trigger**: when the schema discrepancy bites a test or a
downstream consumer.

**Suggested fix (when triggered)**: either route the builder's
surface zeroing through the same closest-point op (so both sides
share the same round-off), or document the discrepancy in a schema
note and have downstream consumers tolerate ~1e-6 disagreement.

### T2.6 — under-collection warning bias understated (correctness W3)

**File**: `physicsnemo/experimental/pnm_pretraining/data/builder.py`,
`_rejection_sample_volume_points`.

**Symptom**: when rejection sampling falls short of `n_volume_points`
within `max_iter` iterations, the function pads-with-replacement and
issues a `warnings.warn(...)`. The warning text reports the count
ratio but does not quantify the resulting per-point duplication
factor. A user with a tight-bbox geometry (high duplication) gets a
silently-degraded sample.

**Trigger**: first user hits a tight-bbox geometry where the warning
fires loudly enough to investigate.

**Suggested fix**: tighten the warning text to surface the
duplication factor (`(n_volume_points - n_accepted) / n_volume_points`),
or add a `strict=True` kwarg that raises instead of padding.

## 4. Tier 3 — verified-skip

These items were flagged by reviewers but verified against the actual
source as either false positives or not worth the touch.

- **`load_pretrained_backbone` claimed to duplicate `load_model_weights`
  (DRY P2 inferred)**. **Verified false.** Reading
  `physicsnemo/utils/checkpoint.py`, `load_pretrained_backbone`
  already reuses `_unwrap_ddp_compile`, `_unwrap_fsdp`,
  `_get_dtensor_param_placements`, `_redistribute_sd_for_dtensor`,
  `set_model_state_dict`, `_extract_mdlus_state_dict`, and
  `_cache_if_needed`. The new `_read_source_state_dict` is genuinely
  additive — it handles the wrapper-dict-unwrapping that
  `load_model_weights` does not need. Leave as-is.

- **`align_mesh_geopt_general` claimed compositional from `ScaleMesh`
  + flip primitives (DRY P2 inferred)**. **Verified
  partially-true but not worth refactoring.** `ScaleMesh` is
  uniform-scalar only (not per-axis); the X-flip has no existing
  primitive in `physicsnemo.datapipes.transforms.mesh.transforms`.
  The custom function carries GeoPT-specific axis policy correctly
  (improvement I3). Skip.

- **Builder calls `signed_distance_field` directly vs `ComputeSDF`
  (DRY P2)**. Skip — `ComputeSDF` is the TensorDict-based variant;
  the rejection-sampling loop in `_rejection_sample_volume_points`
  runs on raw torch tensors and would pay TensorDict construction
  overhead per inner iteration for no benefit.

- **`MeshRayIntersection.warp_forward` claimed non-staticmethod
  (DRY D-P2)**. **Verified false.** `SignedDistanceField` uses the
  same convention (a regular function decorated by
  `FunctionSpec.register`); the `FunctionSpec.register` decorator
  handles dispatch. No bug.

- **Other nits not worth fixing now**:
  - `rel_l2` scalar/vector branches are already factored via
    `_rel_l2_per_example_mean`.
  - `clamp_min(1e-10)` vs `+1e-10` — numerically equivalent;
    cosmetic.
  - `_OVERSIZE_LIMIT_X = 5.5` mirrors the GeoPT reference verbatim;
    documenting it as a magic number would be louder than the
    reference, no value gained.
  - `step_lengths.unsqueeze(-1)` is a one-liner; cosmetic.
  - `n_source` broadcast wasted on non-rank-0 — negligible cost
    relative to a pretraining epoch.
  - Benchmark comment cosmetic.

## 5. TensorDict reuse opportunity

The user observed during review that PhysicsNeMo's TensorDict-native
components could enable more reuse across the pretraining stack.
Direct reading of the implementation gives:

- **Already TensorDict-compliant**: `WalkSampler` uses
  `TensorDict.clone()`, `.exclude()`, and indexed assignment
  natively. No fix needed.

- **Already TensorDict-compliant via Mesh constructor coercion**:
  the builder hands flat `dict[str, torch.Tensor]` to `Mesh()` /
  `DomainMesh()`; both classes' `__post_init__` auto-coerce to
  `TensorDict` with the right `batch_size`. No fix needed.

- **Future opportunity**: the builder could construct TensorDicts
  directly (rather than flat dicts and relying on coercion). This
  would enable chains like `point_data.update(...)` or applying
  `NormalizeMeshFields` from
  `physicsnemo.datapipes.transforms.mesh.transforms` during
  construction (rather than post-hoc on the loaded `DomainMesh`).
  Defer until the builder gets its T2.3 split — that is the natural
  place to land the TensorDict-direct construction without
  inflating the diff for the current cleanup commit.

---

**Maintenance note.** Update this file when (a) any T2 item lands or
(b) any T3 verdict is invalidated by new evidence.
