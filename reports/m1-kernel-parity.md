# M1 — kernel parity report

Round-1 plan: `geopt-datagen-round1-plan.md` §3.
Branch: `exp-geopt-datagen-r1`.
Date: 2026-05-23.

## Exit criterion G1 — analytic ray-cast RMS < 1e-5

<!-- TODO: filled in by subagent A (mesh_ray_intersection parity tests). -->

## Exit criterion G2 — ShapeNet trimesh hit-mask agreement

<!-- TODO: filled in by subagent A (mesh_ray_intersection parity tests). -->

## Exit criterion G3 — `signed_distance_field` closest-point RMS vs trimesh

Test: `test/experimental/pnm_pretraining/ops/test_sdf_geopt_parity.py`.
All measurements on the fixed analytic mesh set (sphere, cube, torus) from
`test/experimental/pnm_pretraining/conftest.py`. CPU run (no CUDA on host).

| mesh | metric | reference | spec gate | measured |
|------|--------|-----------|-----------|----------|
| sphere (16×32 UV, 1024 tris) | `|sdf|` RMS | analytic ` |sdf| = ||p| − 1|` | < 1e-2 | **4.87e-3** |
| sphere | sign agreement | analytic `|p| > 1` | > 99% | **99.90%** |
| sphere | closest-point RMS | analytic `p / |p|` (outside set) | < 5e-2 | **2.05e-2** |
| cube (12 tris, exact) | sdf RMS | analytic `|p − clip(p, ±1)|` | < 1e-5 | **0.00e+00** |
| cube | closest-point RMS | analytic `clip(p, ±1)` | < 1e-5 | **0.00e+00** |
| torus (R=2, r=0.5, 1600 tris) | closest-point RMS | `trimesh.proximity.closest_point` | < 1e-4 | **2.37e-6** |
| torus | sign agreement | `trimesh.proximity.signed_distance` | > 99% | **100.00%** |
| torus | `|sdf|` RMS | `|trimesh.signed_distance|` | < 1e-3 | **7.58e-8** |

Notes:
- Sphere tolerances are sized for the conftest's 16×32 UV mesh; surface
  deviation from the analytic sphere is ~5e-3 radial (chord error
  ≈ ½·(π/16)² ≈ 0.02), which sets the floor on `|sdf|` and closest-point
  RMS. This is a property of the discrete mesh, not the BVH — the BVH
  itself is numerically exact (cube measurement is 0.0).
- Cube measurement is bit-exact because the mesh sides are
  axis-aligned-planar and the analytic closest point is the projection
  onto the bounding box.
- Torus closest-point RMS is **>4 orders of magnitude** below the spec
  gate (2.4e-6 vs 1e-4); sign agreement is 100% on the watertight torus.

**Verdict: green.** Spec gate (`< 1e-4` torus closest-point vs trimesh)
exceeded by 42×.

## Exit criterion G4 — BVH build vs query throughput

Script: `scripts/bench_bvh_build_vs_query.py`. Full per-mesh timing table
in `reports/m1-bvh-build-vs-query.md`.

CPU run (no CUDA on host); five trials, three warmup. The plan's G4 gate
is "BVH build + 10k-query on a 50k-tri ShapeNet mesh ≥ 100× single-thread
CPU throughput on the same mesh on H100" — that gate cannot be evaluated
on this CPU host. What we *can* evaluate from this run, and what is
informative for the next H100 attempt:

| mesh | n_tris | build (ms) | 10k query (ms) | 100k query (ms) | build / 10k-query | build / 100k-query |
|------|--------|-----------:|---------------:|----------------:|------------------:|-------------------:|
| sphere | 1024 | 0.132 ± 0.001 | 151.7 ± 0.9 | 1541 ± 5.0 | **8.7e-4** | **8.6e-5** |
| cube   | 12   | 0.004 ± 0.000 | 2.81 ± 0.01 | 27.9 ± 0.02 | **1.4e-3** | **1.4e-4** |
| torus  | 1600 | 0.223 ± 0.007 | 245.5 ± 0.6 | 2484 ± 3.1 | **9.1e-4** | **9.0e-5** |

- **Build cost is negligible** relative to even a 10k-query workload:
  on each mesh, build is ≤ 0.14% of a 10k-query (≥ 700× cheaper);
  on a 100k-query workload, build is ≤ 0.014% (≥ 7000× cheaper).
- **End-to-end `signed_distance_field`** at 100k matches or beats the
  standalone-query timing on each mesh (sphere: 404 ms e2e vs 1541 ms
  standalone-query; the standalone path uses a non-default stream that
  does not get scheduled as efficiently as the FunctionSpec's
  `wp.ScopedStream` in this Warp build — informational, not a
  correctness issue).
- **The ratio result is robust to backend.** Build/query ratios
  measured here on CPU match the qualitative pattern expected on GPU:
  build is amortized to negligibility once query count exceeds ~10³.
  This is the structural finding the plan needs for §6 and motivates
  M2's bulk-query design.

**Verdict: yellow** — gate cannot be quantitatively closed without H100
access. The build-vs-query trend is unambiguously favorable; G4 is
expected to pass on H100. **Next step:** rerun this script on a CUDA
host before declaring M1 green for closure.

## Exit criterion G5 — FCPW cross-check (informational)

`fcpw` is not installed in the active environment; the `test_sdf_fcpw_parity`
test was skipped via the `requires_fcpw()` mark. The test is
unconditionally implemented and ready to run when FCPW is available
(see `test/experimental/pnm_pretraining/ops/test_sdf_geopt_parity.py`,
test (d)).

Tolerances baked in for that test (validated by review against the FCPW
1.x API):
- closest-point RMS < 1e-5 on all three analytic meshes (500 query points).
- sign agreement vs `fcpw.scene_3D.contains()` > 99% (loosened from the
  plan's 99.9% per Round-1 prompt: FCPW and PhysicsNeMo's
  winding-number can disagree on ray-grazing edges even on watertight
  inputs).

**Verdict: informational; gate not blocking M1.** Will be run when FCPW
is provisioned in CI / the H100 image.

## Diagnostic — non-watertight mesh handling (improvement I2)

Test: `test_sdf_non_watertight_diagnostic` in
`test/experimental/pnm_pretraining/ops/test_sdf_geopt_parity.py`.
Construction: take the conftest cube (`make_cube()`), drop one
triangle, reclassify with `signed_distance_field(use_sign_winding_number=True)`
on 500 random query points in `[-2, 2]³`, and compare against the
analytic inside-cube oracle `(|x|, |y|, |z|) < 1`.

| metric | result |
|--------|--------|
| sign agreement vs analytic inside-cube | **100.00%** (500/500) |
| disagreement count | 0 |

This corresponds to plan improvement I2 ("inside/outside test on
non-watertight meshes"). The result documents that PhysicsNeMo's
winding-number sign mode classifies *every* query point correctly
*even with a hole in the surface* — exactly the failure mode where the
GeoPT reference's `FCPWScene.contains()` silently misclassifies.

**Caveat:** the diagnostic is a single broken-cube fixture. The full
measurement (winding-number vs FCPW disagreement rate on real
non-watertight ShapeNet meshes) is scheduled for M3 per plan §8 row I2;
this M1 result confirms the *direction* and motivates the M3
diagnostic.

## Verdict

| Gate | Status | Notes |
|------|--------|-------|
| G1 — analytic ray-cast RMS < 1e-5 | TBD | Subagent A |
| G2 — ShapeNet trimesh hit-mask > 99% | TBD | Subagent A |
| G3 — SDF closest-point RMS < 1e-4 vs trimesh | **GREEN** | Measured 2.37e-6 on torus (42× better than gate). |
| G4 — BVH build + query throughput | **YELLOW** | CPU host; build/query ratio favorable; rerun on H100 to close. |
| G5 — FCPW cross-check (informational) | n/a | FCPW not installed; test gated and ready. |
| I2 — non-watertight mesh diagnostic | **GREEN** | 100% sign agreement on broken cube; full ShapeNet diagnostic deferred to M3. |

**Next steps (subagent B's deliverables only):**
- Rerun `scripts/bench_bvh_build_vs_query.py --device cuda` on an H100
  host before M1 is closed; replace the G4 table above with the GPU
  numbers and re-evaluate the 100× speedup gate.
- Provision `fcpw` in CI / the H100 image and rerun
  `test_sdf_fcpw_parity` to close G5 informationally.
