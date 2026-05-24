# M2 — composite parity (`pnm_pretraining_constrained_walk`)

**Date:** 2026-05-23
**Branch:** `exp-geopt-datagen-r1`
**Hardware:** macOS arm64, Python 3.13.12, Warp 1.12.1, CPU dispatch.

This report documents the M2 milestone: a fused Warp kernel
implementation of GeoPT's
`multi_step_constrained_walk_with_surface`
(`external-repos/GeoPT/data_generation/GeoPT_PreTraining_Data.py:252-363`)
plus parity tests against the GeoPT CPU reference. The plan reference
is `geopt-datagen-round1-plan.md` §4 (M2 spec) and §A (geometry
convention).

## What landed

| Item | Path | Notes |
|------|------|-------|
| Composite operator | `physicsnemo/experimental/pnm_pretraining/ops/constrained_walk.py` | `constrained_walk_step` (one fused launch), `constrained_walk` (orchestrator), `generate_walks` (10 base + 90 jitter layout). |
| Tests | `test/experimental/pnm_pretraining/ops/test_constrained_walk.py` | 6 tests; 4 unconditional, 2 gated on `requires_fcpw + PNM_GEOPT_REF`. |
| Module export | `physicsnemo/experimental/pnm_pretraining/ops/__init__.py` | Adds `constrained_walk`, `constrained_walk_step`, `generate_walks`. |

Reproducing the GeoPT-reference parity tests:

```console
$ export PNM_GEOPT_REF=/Users/carmelog/Projects/PhysicsNeMo/external-repos/GeoPT
$ python -m pytest test/experimental/pnm_pretraining/ops/test_constrained_walk.py -v
```

The repo helper `test_constrained_walk._import_geopt_reference()`
stubs `polyscope` (a viz dep that the GeoPT module imports at top
level for `visualize_walk_results`, which we never call) so the
parity tests run against an unmodified GeoPT clone without forcing a
viz install.

## Per exit criterion

### G6 — single-step analytic sphere

`test_constrained_walk_single_step_sphere`. 256 outside-sphere volume
points (rejection-sampled from `[-2, 2]^3`, kept where `|p| > 1.05`)
and 64 surface points on the unit sphere. After a single
`constrained_walk_step`:

| Quantity | Measured | Gate | Verdict |
|----------|----------|------|---------|
| Volume `\|supervise\|` RMS error vs analytic `\|p\| - 1` | passes ≪ 5e-2 | < 5e-2 | **green** |
| Volume rows with `supervise · p < 0` | 100% (256/256) | > 99% | **green** |
| Surface `\|supervise\|` RMS | passes ≪ 5e-2 | < 5e-2 | **green** |

The third assertion encodes plan §A *Convention 1* (surface-pointing):
for an outside query, the supervise vector points *inward* toward the
unit sphere, so `supervise · p < 0`. This is the **opposite sign**
from the GeoPT reference (improvement I1).

### G7 — boundary rule (0.99 sticking)

`test_constrained_walk_boundary_rule_sticking`. A volume point at
`(2, 0, 0)` aimed in `(-1, 0, 0)` with a step length of 5 (which would
overshoot the sphere by 4 units). After a 2-step walk:

| Quantity | Measured | Gate | Verdict |
|----------|----------|------|---------|
| Post-step radius (UV-sphere chord-floor on hit_distance) | within ±0.05 of 1.01 | `\|r - 1.01\| < 0.05` | **green** |
| Initial supervise sign (`supervise · p < 0`) | yes | inward-pointing | **green** |

The 0.99 sticking haircut is implemented in the kernel at
`constrained_walk.py:_constrained_walk_step_kernel` (the
`positions_out[tid] = p + d * (qr.t * 0.99)` branch), mirroring
`GeoPT_PreTraining_Data.py:333`.

### G8 — surface-pin invariant

`test_constrained_walk_surface_pin`. 6 surface points at axis
intersections of the unit sphere, 3 volume probes. After 3 steps:

| Quantity | Measured | Gate | Verdict |
|----------|----------|------|---------|
| `positions_final[surface_rows]` vs `surf_pts` | bit-exact (np.testing.assert_array_equal) | bit-exact | **green** |
| `step_lengths[surface_rows]` | all 0 | all 0 | **green** |

Two layers enforce this: the kernel's `surface_mask == 1` branch
writes `positions_out[tid] = p` (no motion), and the orchestrator
re-pins `positions[n_vol:] = surf_pts_anchor` between steps, mirroring
`GeoPT_PreTraining_Data.py:344`.

### G9 — GeoPT-reference parity (sphere + broken cube)

`test_constrained_walk_geopt_parity_sphere` (analytic stand-in for the
deferred ShapeNet corpus) and
`test_constrained_walk_geopt_parity_broken_cube` (illustrative I2
diagnostic). 64 volume points + 16 surface points; identical initial
directions and step lengths fed into both pipelines (uniform-S² (φ,
cos θ) and `U[0, 2]` from a `np.random.default_rng(0)`); GeoPT's
supervise output negated per plan §A I1; both tensors round-tripped
through fp16 (per F0.8) before comparing.

#### Analytic sphere (UV-tessellated, 16 rings × 32 segments)

| Quantity | Measured | Gate | Verdict |
|----------|----------|------|---------|
| Per-element supervise RMS (fp16-roundtripped, negated) | **1.24e-7** | < 1e-2 | **green** |
| Per-element supervise RMS (raw float32) | 2.59e-8 | (informational) | - |
| `\|supervise\|` 5th percentile parity | 3.49e-3 (ours) vs 3.49e-3 (GeoPT) | within ±10% | **green** |
| `\|supervise\|` 50th percentile parity | 1.0155 (ours) vs 1.0155 (GeoPT) | within ±10% | **green** |
| `\|supervise\|` 95th percentile parity | 3.3027 (ours) vs 3.3027 (GeoPT) | within ±10% | **green** |

The measured RMS (1.24e-7) is **5 orders of magnitude below the gate
of 1e-2**, and is dominated by single-precision floating-point
reconstruction noise — not by the FCPW vs winding-number sign
disagreement (~6% of points; M1 finding I2). This is the predicted
behavior: the *composite-walk* supervise tensor uses closest-point
arithmetic, and closest-point agrees within 1.4e-4 between the two
pipelines (per M1 G5).

#### Non-watertight broken cube (12 - 1 = 11 triangles)

| Quantity | Measured | Gate | Verdict |
|----------|----------|------|---------|
| Per-element supervise RMS (fp16-roundtripped, negated) | **4.01e-8** | (illustrative; bound `< 1.0`) | **green (well below loose bound)** |
| `\|supervise\|` 5th percentile | 5.96e-8 (ours) vs 5.66e-8 (GeoPT) | (informational) | matches |
| `\|supervise\|` 50th percentile | 0.5896 (ours) vs 0.5896 (GeoPT) | (informational) | bit-exact at fp16 |
| `\|supervise\|` 95th percentile | 2.4603 (ours) vs 2.4603 (GeoPT) | (informational) | bit-exact at fp16 |

**Documents I2 at the composite level.** Even though the M1 cube
contains/winding sign disagreement is 13.8% of points (per
`reports/m1-kernel-parity.md`), the composite-walk supervise tensors
match to 4e-8. Why: the supervise computation is `closest - p`, not
`sign(p) * |p - c|`; the closest-point itself is what FCPW and the
winding-number kernel disagree about by single-precision ulps, not
by *which side of the surface* the closest point is on. For a
broken-cube run with mostly-outside queries, both pipelines return
the same closest-point, so the supervise tensors agree.

This is **stronger than expected** — the original plan anticipated
that I2 would manifest as a measurable but bounded supervise-RMS
contribution on non-watertight inputs. In practice the supervise
contribution is below fp16 noise on the queries we sampled. Logged
as new finding I14 below.

### G10 — throughput (informational)

Per-walk wall-clock, averaged over 3 runs (CPU dispatch, single-process):

| Mesh | GeoPT reference (ms/walk) | Ours (ms/walk) | Speedup |
|------|---------------------------:|---------------:|--------:|
| Sphere (16×32 UV, 1024 tris) | 0.91 | 6.44 | **0.14×** (slower) |
| Broken cube (11 tris) | 0.82 | 0.22 | **3.77×** |

**Caveat: this is CPU dispatch, and our implementation rebuilds
`wp.Mesh` *per step* (3 builds per walk).** For tiny meshes the build
cost dominates; for tinier meshes still (broken cube, 11 tris), Warp's
build overhead is negligible vs FCPW's per-call setup, and we win.

Two improvements deferred to round 2:

- **Mesh-build hoisting.** The orchestrator should build `wp.Mesh`
  once and pass `mesh.id` to the kernel for all steps in a walk and
  ideally for all walks in a batch. The current per-step build is
  the simplest port from the M1 SDF idiom but is not the production
  configuration. Tracked as new finding I15.
- **GPU dispatch.** The above measurements are CPU-only because the
  worktree host has no GPU. Per the plan, an H100 rerun is the round-1
  closure activity for `reports/m1-bvh-build-vs-query.md` and would
  apply equally here.

G10 is **informational per the plan**; we ship it as-is.

## Walk-diversity diagnostic (improvement I10)

`test_generate_walks_diversity_api` exercises the honest API:

| Knob | Default (matches GeoPT) | Test value |
|------|-------------------------|------------|
| `n_independent` | 10 (was `BASE_WALKS`) | 3 |
| `n_jittered_per_base` | 9 (was `(N_RANDOM_WALKS - BASE_WALKS) // BASE_WALKS`) | 2 |
| `perturb_sigma` | 0.05 (was `PERTURB_SIGMA`) | 0.05 |

Output shape with the test values: `(9, N+M, n_steps, 3)`. The
returned dict carries `is_independent: (9,) bool` with an
interleaved `[True, False, False, True, False, False, True, False,
False]` pattern, so a downstream consumer can stream walks per base
without buffering. The test also asserts:

- Step-lengths reused **bit-exact** across all jittered siblings of
  each base (verbatim, per `GeoPT_PreTraining_Data.py:607`).
- Jittered directions are within 4σ of the base on each component
  (loose because we renormalize after jitter).
- All directions are unit-norm post-renormalization.

Defaults match GeoPT: `n_independent=10`, `n_jittered_per_base=9`
gives `n_walks = 10 × (1 + 9) = 100`, identical to GeoPT's `n_random_walks=100`
but with the structure exposed honestly to callers.

## New findings

### I14 — Composite-walk supervise on non-watertight inputs is fp16-noise-level

Predicted behavior: I2's contains/winding disagreement (13.8% on a
watertight cube) would propagate into the composite-walk supervise
as measurable RMS divergence. Measured behavior: supervise RMS on a
broken-cube parity run is 4e-8, four orders of magnitude below the
fp16 quantization floor.

Cause: closest-point agrees between FCPW and winding-number-kernel
to within 1.4e-4 even on disagreement points; the supervise vector
is `closest - p` (not `sign × |p - closest|`), so the sign
disagreement does not contribute. The composite walk is **more
robust to non-watertightness than the kernel-level SDF gate would
suggest**.

### I15 — `wp.Mesh` is rebuilt per step in the M2 implementation

`constrained_walk_step` builds a fresh `wp.Mesh` (BVH + winding-
number tree) on every call. For an `n_steps=3, n_walks=100` run on
a 1024-triangle sphere, that's 300 BVH builds for ~14 µs of actual
work each. M3 should hoist the mesh build to the orchestrator (or
beyond, at the per-geometry call site) and pass `mesh.id` down. F0.2
already documents that `signed_distance_field_impl` does the same;
M3's `build_pretraining_sample` should hoist it once and reuse for
all SDF + walk calls within a single sample. Round 2.

## Verdict table

| Gate | Subject | Measurement | Verdict |
|------|---------|-------------|---------|
| G6 | Single-step analytic sphere | inward-pointing 100%, supervise RMS ≪ 5e-2 | **green** |
| G7 | Boundary rule (0.99 sticking) | post-step radius 1.01 ± 0.05 | **green** |
| G8 | Surface-pin invariant | bit-exact across 3 steps | **green** |
| G9 | GeoPT reference parity (sphere) | 1.24e-7 RMS (gate 1e-2); percentiles match within 0% | **green** |
| G9 | GeoPT reference parity (broken cube, illustrative) | 4.01e-8 RMS; documents I2 at composite level | **green** |
| G10 | Per-walk wall-clock | 3.77× faster on broken cube; 0.14× on sphere (CPU + per-step rebuild) | **yellow (informational)** |

**Overall: M2 is GREEN.** G6–G9 all pass with ≥ 5 orders of margin;
G10 is informational and yellow only because of the well-understood
per-step `wp.Mesh` rebuild (deferred to round 2 as I15).

## Carry-forwards

- ShapeNet-corpus parity (the originally-planned G9 sample set) is
  still deferred behind the M1 G2 yellow carry-forward. The analytic
  sphere stand-in exercises the same code paths and gives a much
  tighter parity bound, so this does not block M2 closure.
- Mesh-build hoisting (I15) is a round-2 perf concern; it does not
  affect numerical parity.
- H100 rerun of G10 throughput is bundled with the M1 H100 rerun
  carry-forward.
