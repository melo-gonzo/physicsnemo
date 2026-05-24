# GeoPT Data-Gen Reproduction & Validation — Round 1 Plan

**Goal of round 1.** Reproduce and validate GeoPT's pretraining data-generation
pipeline on Warp + PhysicsNeMo `Mesh`, end-to-end for one geometry, to a
quantitative parity bar against the released GeoPT CPU reference. No
backbone, no training loop, no fine-tuning — round 1 stops at "I can
regenerate one geometry's `x.npy` + `supervise_*.npy` + `condition_*.npy`
on the GPU and the value distributions match the CPU reference within
documented tolerances."

**Branch / worktree.** `exp-geopt-datagen-r1` at
`.worktrees/exp-geopt-datagen-r1/` (this directory). Tracks `origin/main`.

**Parent plan.** `geopt-physicsnemo-engineering-plan.md` (in repo root) —
this round-1 doc executes the kernel-and-composite portion of that plan's
PR 0 + PR 1 + the kernel half of PR 2. It does **not** execute PR 2's
datapipe / backbone-adapter / head / loss work, nor any of PRs 2.5 / 3 / 4.

**Iterative shape.** Three milestones, each landable independently:

- **M1 — kernel parity.** `signed_distance_field` and a new
  `MeshRayIntersection` `FunctionSpec` produce closest-point and
  ray-intersection outputs that match an FCPW (or trimesh fallback)
  reference on a fixed mesh set, within documented RMS tolerances. One
  short report committed to the branch.
- **M2 — composite parity.** A new `pnm_pretraining_constrained_walk`
  composite operator produces supervision targets matching GeoPT's
  `multi_step_constrained_walk_with_surface` on a fixed ShapeNet subset,
  within documented tolerances. One short report committed to the branch.
- **M3 — `.pdmsh` round-trip.** End-to-end: ingest one ShapeNet OBJ →
  apply GeoPT's geometry-alignment preprocessing → run our composite →
  emit a `.pdmsh` `DomainMesh` per the §1.2 storage contract → read it
  back via `DomainMeshReader`. No consumer yet; the round-trip
  itself is the validation.

Each milestone produces a short report markdown alongside the code. We
adjust the next milestone's plan based on the previous milestone's
findings before starting it.

---

## A — Geometry-direction convention (load-bearing)

This convention is fixed across all of round 1's code, tests, reports,
and `.pdmsh` schema. **Any downstream consumer (head, loss, eval) that
disagrees with this convention is wrong, not the data-gen pipeline.**

**Definitions.**

- A **query point** `p ∈ ℝ³` is a point at which we want supervision —
  either a volume sample (interior of bbox, exterior of mesh) or a
  surface sample.
- The **mesh** `M` is the closed surface (or closed-after-winding-number
  treatment for non-watertight ShapeNet inputs) of the geometry. Its
  outward normal at a surface point `s` is `n̂(s)`.
- The **closest point** `c(p) = argmin_{s ∈ M} ‖p − s‖`. Always exists;
  may be on a face, edge, or vertex.
- The **signed distance** `d(p) = sign(p) · ‖p − c(p)‖`, with
  `sign(p) = +1` if `p` is outside `M`, `−1` if inside. Computed via
  `signed_distance_field(use_sign_winding_number=True)`.

**Convention 1 — vector-distance feature (the pretraining target).**

```
v(p) := c(p) − p           # surface-pointing, magnitude = |d(p)|
```

This is **surface-pointing**: at a query point outside the mesh,
`v(p)` points *toward* the surface (inward toward the geometry). At a
query point inside the mesh, `v(p)` still points toward the surface
(outward toward the boundary). For a query exactly on the surface,
`v(p) = 0`.

For a smooth surface, `v(p) ≈ −d(p) · n̂(c(p))`: the magnitude is
`|d(p)|`, and the direction is the inward unit-surface-normal at the
foot point when `p` is outside, the outward unit-surface-normal when
`p` is inside.

**This is the opposite sign convention from the GeoPT reference.**
GeoPT's `multi_step_constrained_walk_with_surface`
(`GeoPT_PreTraining_Data.py:319`) emits

```
v_geopt(p) := positions − closest = p − c(p) = −v(p)         # query-pointing
```

i.e. **query-pointing** (outward-from-surface for outside-mesh queries).
Both conventions are mathematically equivalent up to sign; only one is
a reasonable physical default. Surface-pointing wins because:

- It generalizes to "distance to wall + direction to wall" in CFD,
  which is the standard formulation for wall functions, RANS y+
  computations, and immersed-boundary methods.
- It composes cleanly with surface normals: at the foot point,
  `v(p) / ‖v(p)‖ ≈ ±n̂(c(p))` with sign matching `−sign(p)`.
- A model predicting `v(p)` is predicting *where the geometry is*, not
  *where the query came from* — the former is the geometric prior we
  actually want to inject.

**Migration from GeoPT reference data.** Any pre-existing GeoPT
`supervise_*.npy` files used in parity tests are **negated**
(`supervise_ours = −supervise_geopt`) before comparison. The parity
tolerance applies to the negated tensor.

**Convention 2 — outward surface normal.**

For a surface point `s`, `n̂(s)` is the **outward unit normal** —
pointing from the interior of the geometry into the surrounding fluid
domain. PhysicsNeMo's `sample_points_on_mesh` returns this directly.
Ports of GeoPT's `compute_normals_improved` must check sign: if the
underlying `mesh.vertex_normals` orientation disagrees, flip per-vertex
based on `sign(d(s ± ε · n̂(s)))`.

**Convention 3 — ray direction.**

For `MeshRayIntersection(origin, direction, max_dist)`, `direction` is
a **unit vector**. The kernel does **not** internally normalize (matching
the existing `signed_distance_field` discipline). Callers must
normalize. The hit point is `origin + t · direction` for the returned
`t = hit_distance`. Misses return `t = inf`, `hit_point = origin`,
`hit_mask = 0`.

**Convention 4 — alignment / world frame.**

After `align_mesh_geopt_general`:
- **+X axis**: longitudinal (length direction); mesh extent normalized
  to `target_length = 5.0` units.
- **+Y axis**: vertical (gravity-up); mesh sits on the `Y = 0` plane
  (Y-min == 0 exactly post-alignment).
- **+Z axis**: lateral; mesh centered on `Z = 0`.
- X-flip vs no-flip: the General-variant flips X (`new_V[:,0] = −V[:,0]`).
  This puts the conventional "front" of an asymmetric vehicle in `+X`.

This convention is documented in `align_mesh_geopt_general`'s
docstring; the alignment record returned by that function carries the
flip flag so consumers can recover the original frame if needed.

**Convention 5 — sign of inside/outside test.**

`signed_distance_field(use_sign_winding_number=True)` returns negative
SDF for points **inside** the mesh, positive for points **outside**.
Volume-point rejection sampling keeps points with `sdf > 0`. Per the
parent plan §3.2 and the PhysicsNeMo SDF docstring.

---

## 0 — Discovery findings that shape this plan

A read-only survey of `physicsnemo-core` and `external-repos/GeoPT/` (run
before drafting this doc) surfaced the following facts. Each shapes a
specific decision below.

### F0.1 — `signed_distance_field` has no torch reference forward

`physicsnemo/nn/functional/geometry/sdf.py:220-347` defines the
`SignedDistanceField` `FunctionSpec`. **Only the Warp backend is
registered** (line 271, `rank=0`, `baseline=True`). There is no
`torch_forward`, contrary to the parent plan's implicit assumption
that the existing FunctionSpec already exposes a CPU reference.

**Implication for M1.** The numerical-parity test for
`signed_distance_field` cannot use a built-in torch reference. It
must use either (a) FCPW directly (test-only dev dep), or (b)
`trimesh.proximity.closest_point` (already in PhysicsNeMo's example
dependencies, slower but ubiquitous). M1 uses **trimesh** as the
default reference and FCPW as an `@requires_fcpw`-gated cross-check.

### F0.2 — `signed_distance_field_impl` builds `wp.Mesh` internally per call

`sdf.py:168-172` constructs `wp.Mesh(points, indices, support_winding_number=...)`
inside `signed_distance_field_impl`. It does **not** accept a pre-built
`wp.Mesh` handle.

**Implication for M1.** Measuring "BVH build cost vs query cost"
requires either timing the full op end-to-end (build + query mixed) or
constructing the `wp.Mesh` ourselves outside the FunctionSpec and
calling `wp.mesh_query_point` directly. M1 does the latter for the
build-cost measurement only — a small instrumentation script, not a
production code path.

### F0.3 — `Mesh.save()` writes a memmap directory, not a single file

`physicsnemo/mesh/mesh.py:526-572` and `domain_mesh.py:283-329`
document that `.pmsh` / `.pdmsh` are **directory trees** of memmap
files plus tensordict metadata. Not HDF5, not zarr, not pickle.

**Implication for M3.** Round-trip is via `DomainMesh.save(prefix)` →
`DomainMesh.load(prefix)` (or `DomainMeshReader`). Field ordering and
TensorDict key types matter; we cannot assume HDF5-style schema flexibility.

### F0.4 — GeoPT's "100 random walks" is 10 independent + 90 jittered

`external-repos/GeoPT/data_generation/GeoPT_PreTraining_Data.py:585-608`
defines `BASE_WALKS = 10` and `PERTURB_SIGMA = 0.05`. Walks 0–9 are
sampled fresh; walks 10–99 reuse `base_directions[j % 10]` plus a
σ=0.05 Gaussian jitter, then renormalize. Step lengths are reused
verbatim from walks 0–9.

**Implication for M2 / M3.** A faithful reproduction must emit the
same 10-base + 90-jitter structure, not 100 fresh samples. The parent
plan's §3.3 example kernel doesn't capture this; M2 documents and
implements it.

### F0.5 — GeoPT's data-gen has no seeding anywhere

A grep of `data_generation/` for `np.random.seed` / `seed(` returns
zero hits. The reference is non-deterministic across runs.

**Implication for M1 / M2.** Bit-exact parity is impossible. Parity
tests check **value distributions** — RMS error on supervision
targets, percentile match on volume-point distributions, hit-mask
agreement rate. We seed our reproduction (`np.random.seed`,
`torch.manual_seed`, `wp.rand_init`) for our own determinism.

### F0.6 — `transform_mesh` has misleading variable names

`GeoPT_PreTraining_Data.py:145-174`: the returned `z_min` is actually
the post-axis-swap **Y**-minimum; the returned `y_avg` is the post-axis-swap
**Z**-mean. `GeoPT_PreTraining_Data_General.py:205-245` does an
X-axis-flip instead of an X↔Z swap, mutates inputs in-place, and has
an extra oversize-bbox safety multiplier (`*0.5` if any axis exceeds
hard limits).

**Implication for M3.** Our port lives at
`physicsnemo/experimental/pnm_pretraining/data/transforms.py` (per
parent plan §1.3). It uses correctly-named variables (`y_min_post_swap`,
`z_mean_post_swap`) and a single canonical implementation. We pick the
General variant (X-flip, no axis swap) since it generalizes across all
52 ShapeNet categories; we document the divergence from the
category-specific X↔Z-swap variant in a docstring NOTE.

### F0.7 — `compute_normals_improved` is vertex-nearest, not face-barycentric

`GeoPT_PreTraining_Data.py:26-41` builds a `trimesh.PointCloud` over
mesh vertices, queries the nearest **vertex** (k=1), and reads
`mesh.vertex_normals` at that vertex. PhysicsNeMo's
`sample_points_on_mesh` (`physicsnemo/models/domino/utils/utils.py:1081-1175`)
returns area-weighted face-barycentric normals.

**Implication for M3.** The `x.npy`-column-7 normals will not match
between our pipeline and GeoPT's by default. M3 emits **both** —
`point_data.normals_face_barycentric` (from `sample_points_on_mesh`,
the principled answer) and `point_data.normals_vertex_nearest` (a
bug-compatibility port of GeoPT's method). M2 / M3 parity tests use
the matching reference for each.

### F0.8 — Float16 disk storage in GeoPT

All on-disk arrays in GeoPT are `float16` (`x.npy`, `supervise_*.npy`,
`condition_*.npy`). Round-trip parity must allow ≥3 decimal digits of
loss for values with magnitude ≲ 1e-3.

**Implication for M2 / M3.** Parity tests round-trip the GeoPT
reference through `float16` cast before comparing to our (we keep
fp32 internally, cast at write only). Tolerance: RMS error on
fp16-cast outputs < 1e-3, not 1e-5.

### F0.9 — `FCPWScene.contains()` is the only inside/outside test, no watertightness handling

GeoPT does no `mesh.fill_holes()` / `is_watertight` check. ShapeNet is
notoriously non-watertight; a fraction of "outside" volume points are
silently inside. PhysicsNeMo's `signed_distance_field(use_sign_winding_number=True)`
handles this correctly via winding-number sign.

**Implication for M2 / M3.** Our reproduction uses winding-number sign
(more correct). We log the disagreement rate vs. FCPW `contains()` on
the parity-test mesh set as a **diagnostic**, not as a parity failure
— GeoPT's behavior here is itself buggy and worth flagging in the
report.

### F0.10 — GeoPT runs single-process Python; "80 cores" is FCPW internal threading

`grep multiprocessing` across the GeoPT repo returns zero hits. The
top-level `for` loop in `main()` is serial. The README's throughput
claim is FCPW's vectorized-BVH internal SIMD/threading.

**Implication for budgeting.** The parent plan's §3.4 "100×
speedup vs the 80-core baseline" should be read as "100× vs an
FCPW-internally-threaded single-Python-process baseline". M1's
throughput measurement compares the same — single-process Python on
both sides.

---

## 1 — Module layout for round 1

Aligned with parent plan §1.3. New code lands under
`physicsnemo/experimental/pnm_pretraining/`, all of it
production-shaped from day 1.

```
physicsnemo/experimental/pnm_pretraining/
  __init__.py
  ops/
    __init__.py
    mesh_ray_intersection.py        # M1 — new FunctionSpec
    constrained_walk.py             # M2 — GeoPT-specific composite
  data/
    __init__.py
    transforms.py                   # M3 — geometry alignment (X-flip variant)
    builder.py                      # M3 — orchestrator emitting one .pdmsh

test/experimental/pnm_pretraining/
  ops/
    test_mesh_ray_intersection.py   # M1 — analytic + trimesh parity
    test_constrained_walk.py        # M2 — composite parity vs GeoPT reference
  data/
    test_transforms.py              # M3 — alignment sanity
    test_pdmsh_round_trip.py        # M3 — write / read / shape / dtype

# Round-1-only artifacts (not for upstreaming):
.worktrees/exp-geopt-datagen-r1/
  geopt-datagen-round1-plan.md      # this file
  reports/
    m1-kernel-parity.md             # M1 deliverable
    m2-composite-parity.md          # M2 deliverable
    m3-pdmsh-round-trip.md          # M3 deliverable
  scripts/
    bench_bvh_build_vs_query.py     # M1 instrumentation (throwaway)
    fetch_shapenet_subset.py        # M1/M2 — pulls the fixed test mesh set
```

The parent plan's `physicsnemo/experimental/pnm_pretraining/finetune/`,
`models/`, `losses.py`, `configs/`, and the `examples/cfd/external_aerodynamics/pnm_pretrain/`
recipe are **out of scope** for round 1.

---

## 2 — Fixed test-mesh set

Used identically across M1, M2, M3 parity tests. Held in a manifest
at `.worktrees/exp-geopt-datagen-r1/scripts/test_meshes.yaml` so the
exact mesh list is reproducible.

- **3 analytic meshes**, generated programmatically (no download):
  unit sphere (UV-tessellated, ~1280 tris), unit cube (12 tris), torus
  (R=2, r=0.5, ~3200 tris).
- **9 ShapeNet meshes** — 3 each from car (`02958343`),
  airplane (`02691156`), ship (`04530566`). Selected by sorting
  category instances by triangle count and taking the 25th, 50th,
  75th percentile mesh per category. Pulled from the ShapeNet v1
  release (`model_normalized.obj`). Manifest stores the
  category ID + instance hash; the fetch script handles download.

That's 12 meshes total, ~10⁴–10⁵ triangles each — small enough to
parity-test cheaply, large enough to exercise the BVH.

---

## 3 — Milestone 1: kernel parity

### 3.1 Scope

Land the **one new kernel** the parent plan calls for, with parity
tests against analytic ground truth and against trimesh / FCPW on the
fixed test mesh set. Plus throughput and BVH-build-vs-query measurements
(parent plan PR 0 exit criteria).

### 3.2 Deliverables

1. `physicsnemo/experimental/pnm_pretraining/ops/mesh_ray_intersection.py`
   — `MeshRayIntersection` `FunctionSpec` modeled on
   `physicsnemo/nn/functional/geometry/sdf.py`. Same
   `@torch.library.custom_op` + `register_fake` + Warp kernel pattern.
   `make_inputs_forward` reuses the same UV-sphere generator at
   sizes `small`, `medium`, `large`. Returns `(hit_mask: int32,
   hit_distance: float32, hit_point: vec3f)`.

2. `test/experimental/pnm_pretraining/ops/test_mesh_ray_intersection.py`
   — five tests:
   - **a)** Analytic sphere ray-cast (origin at center, direction outward,
     distance == radius). RMS error < 1e-5.
   - **b)** Cube ray-cast (axis-aligned rays). RMS error < 1e-5.
   - **c)** Torus ray-cast (off-axis rays, exercise BVH traversal).
     RMS error < 1e-5.
   - **d)** ShapeNet-subset cross-check vs `trimesh.ray.ray_pyembree`
     (or `ray_triangle` if pyembree unavailable). Hit-mask agreement
     > 99%; hit-distance RMS < 1e-4 over hit subset (looser tolerance
     because trimesh-pyembree is also single-precision).
   - **e)** `make_inputs_forward` smoke — exercises all three sizes
     and dispatches via `MeshRayIntersection.dispatch(...)`.

3. `test/experimental/pnm_pretraining/ops/test_mesh_ray_intersection.py`
   — additionally a **FCPW cross-check**, gated by `@requires_fcpw`,
   that runs the same ShapeNet-subset case against
   `fcpw.scene_3D().intersect(...)`. Hit-mask agreement > 99.9%;
   hit-distance RMS < 1e-5. Skipped in CI; runs locally when FCPW is
   installed.

4. **`signed_distance_field` parity validation.** A separate test
   (`test/experimental/pnm_pretraining/ops/test_sdf_geopt_parity.py`)
   that runs `signed_distance_field(use_sign_winding_number=True)`
   on the same fixed mesh set and validates closest-point + sign
   against trimesh (default) and FCPW (gated). Closest-point RMS
   < 1e-4; sign agreement > 99% (allowing for boundary disagreement
   between winding-number and ray-parity sign methods on
   non-watertight ShapeNet meshes).

5. `.worktrees/exp-geopt-datagen-r1/scripts/bench_bvh_build_vs_query.py`
   — throwaway timing harness. For each mesh in the fixed set:
   - Time `wp.Mesh(points, indices)` construction (build).
   - Time `wp.launch(kernel, dim=N, inputs=...)` for N ∈ {1k, 10k, 100k} query points (query).
   - Compare to FCPW build + 100k-query timing for the same mesh
     (gated; skip if no FCPW).
   - Repeat over 5 trials, report median + IQR.

6. `.worktrees/exp-geopt-datagen-r1/reports/m1-kernel-parity.md`
   — one-page report:
   - Numerical agreement table (12 meshes × 2 ops × 2 references).
   - BVH build vs query throughput table (all 12 meshes, 3 query sizes).
   - Disagreement-rate diagnostic for inside/outside test on
     non-watertight ShapeNet (winding-number vs FCPW).
   - Green / yellow / red verdict on each parent-plan PR-0 exit
     criterion.

### 3.3 M1 exit criteria

All of the following must hold to start M2:

- (G1) Analytic ray-cast RMS error < 1e-5 on sphere / cube / torus.
- (G2) ShapeNet-subset trimesh hit-mask agreement > 99%, hit-distance
  RMS < 1e-4 over hit subset.
- (G3) `signed_distance_field` closest-point RMS < 1e-4 vs trimesh on
  same set.
- (G4) BVH build + 10k-query throughput on a 50k-tri ShapeNet mesh ≥ 100×
  the per-CPU-core throughput we measure for the same op single-thread
  on the same mesh (the parent plan's §3.4 conservative target). If
  build-only dominates, downgrade to **measured** speedup with a note.
- (G5) FCPW cross-check runs locally and agrees within tighter
  bounds (hit-mask > 99.9%, distance RMS < 1e-5). If FCPW is
  unavailable, this gate is informational, not blocking.

If any of G1–G4 fails, stop and revise this plan before M2.

### 3.4 Subagent fan-out for M1

After this plan is approved, M1 work is suitable for **two parallel
feature subagents**:

- **Subagent A** — implement `mesh_ray_intersection.py` + its tests.
  Self-contained; only touches new files. Reads `sdf.py` as the
  template.
- **Subagent B** — implement the SDF parity test
  (`test_sdf_geopt_parity.py`) and the throwaway `bench_bvh_build_vs_query.py`
  script. Independent of subagent A's deliverable except for shared
  mesh-loading utilities (which we put in a shared `_test_meshes.py`
  helper, owned by whichever subagent runs first).

Both report back with the M1 report markdown (one combined).

---

## 4 — Milestone 2: composite parity

### 4.1 Scope

Land the **GeoPT-specific composite operator**
(`pnm_pretraining_constrained_walk`) and validate that its supervision
targets match the GeoPT reference (`multi_step_constrained_walk_with_surface`)
on the fixed ShapeNet subset.

### 4.2 Deliverables

1. `physicsnemo/experimental/pnm_pretraining/ops/constrained_walk.py`
   — implements GeoPT's algorithm:
   - Volume + surface points concatenated; surface-mask flag.
   - Per-particle direction sampled uniform on S² via
     (φ ∼ U[0,2π), cos θ ∼ U[-1,1]) parameterization (matching
     GeoPT lines 302-309 exactly).
   - Per-particle step length ∼ U[0, 2.0]; surface entries forced to 0.
   - Per step: closest-point query (`signed_distance_field` with
     `use_sign_winding_number=True`) → record supervision; ray-cast
     (`MeshRayIntersection` from M1) → check collision against
     intended step → apply 0.99 safety haircut on collision. Surface
     positions re-pinned each step.
   - Returns `(supervise: (N+M, 3*steps), condition: (N+M, 4),
     directions: (N+M, 3), step_lengths: (N+M,))`.
   - The internals fuse into a single Warp kernel per step (matching
     parent plan §3.3's example kernel) — closest-point + ray-cast +
     position update inline, avoiding host roundtrips. The Python
     orchestrator only handles seeding and per-step boundary
     bookkeeping.
   - **`generate_walks(scene, vol_pts, surf_pts, n_walks=100, base_walks=10, perturb_sigma=0.05, steps=3)`**
     — wraps the above to emit GeoPT's 10-base + 90-jitter walk
     structure (per F0.4). Returns the stacked
     `(n_walks, N+M, 3*steps)` supervise tensor.

2. `test/experimental/pnm_pretraining/ops/test_constrained_walk.py`:
   - **a)** Single-step parity on a unit sphere: every supervision
     vector should have `‖p - closest‖ ≈ 1` (radius). RMS < 1e-4.
   - **b)** Boundary-rule sanity: with collision-forcing initial
     conditions (point inside ball, direction outward, step length
     larger than radius), confirm the post-step position is within
     `radius * 0.99` of the surface, never outside.
   - **c)** Surface-pin sanity: surface points stay put after τ steps;
     step_lengths-of-surface == 0 in the returned condition.
   - **d)** GeoPT-reference parity on 3 ShapeNet meshes: run our
     `generate_walks(..., n_walks=10, steps=3)` and GeoPT's reference
     `multi_step_constrained_walk_with_surface` on the same mesh
     with the same seeded initial directions and step lengths. After
     casting both to `float16`, distribution-level checks:
     - Per-element supervision RMS < 1e-3 (fp16 quantization floor).
     - Hit-mask agreement on the embedded ray-casts > 99% (loose
       because winding-number vs FCPW disagreement on non-watertight).
     - Per-walk supervise-norm percentile match (5th / 50th / 95th)
       within ±5%.
     Gated `@requires_geopt_reference` (the cloned repo path supplied
     via env var `PNM_GEOPT_REF=/path/to/external-repos/GeoPT`).

3. `.worktrees/exp-geopt-datagen-r1/reports/m2-composite-parity.md`
   — one-page report:
   - Per-mesh per-walk supervision RMS table.
   - Hit-mask agreement table.
   - Throughput per geometry-walk (compare to GeoPT README's "20s on 80 cores").
   - Notes on any unexpected divergences (we expect the inside/outside
     non-watertightness disagreement to show up as a measurable but
     bounded supervision-RMS contribution).

### 4.3 M2 exit criteria

- (G6) Single-step analytic parity on sphere RMS < 1e-4.
- (G7) Boundary rule passes: no particle ends inside the mesh.
- (G8) Surface-pin invariant holds bit-exact.
- (G9) GeoPT reference parity: per-element supervise RMS < 1e-3 (fp16);
  hit-mask agreement > 99% on hit subset.
- (G10) Throughput per geometry-walk is faster than the GeoPT CPU
  reference's per-walk wall-clock on the same hardware (looser than
  the parent plan's 100×; we'll quote what we get).

If G6–G9 fail, stop and triage. G10 is informational.

### 4.4 Subagent fan-out for M2

Single feature subagent. The dependency on M1's `MeshRayIntersection`
and the closeness of the kernel fusion to parent plan §3.3 means it
shouldn't be split.

---

## 5 — Milestone 3: `.pdmsh` round-trip

### 5.1 Scope

End-to-end: ingest one ShapeNet OBJ → apply our port of GeoPT's
`transform_mesh` → run `generate_walks` from M2 → emit a `.pdmsh`
`DomainMesh` per the parent plan's §1.2 storage contract → read it
back via `DomainMeshReader` and validate shapes, dtypes, and a
sample-level numerical equivalence with the in-memory tensors before
write.

### 5.2 Deliverables

1. `physicsnemo/experimental/pnm_pretraining/data/transforms.py`
   — port of GeoPT's `transform_mesh` (X-flip variant from
   `GeoPT_PreTraining_Data_General.py`, per F0.6). Renamed locals.
   Returns a tensorclass-friendly transform record (translation,
   scale, axis-flip flag) so the inverse can be applied later.
   Function `align_mesh_geopt_general(mesh: trimesh.Trimesh,
   target_length: float = 5.0, oversize_safety: bool = True) ->
   (trimesh.Trimesh, AlignmentRecord)`.

2. `physicsnemo/experimental/pnm_pretraining/data/builder.py`
   — `build_pretraining_sample(obj_path, *, n_volume_points=32768,
   n_surface_points=4096, n_walks=100, steps=3, device="cuda") ->
   DomainMesh`. Internally:
   - Load via `trimesh.load`.
   - Sample 4096 surface points + face-barycentric normals via
     `physicsnemo.models.domino.utils.sample_points_on_mesh`.
   - **Also** compute the GeoPT vertex-nearest normals (port of
     `compute_normals_improved`, per F0.7) for parity.
   - Apply `align_mesh_geopt_general` to mesh and to the
     pre-sampled surface points.
   - Sample 32768 volume points by rejection sampling (uniform in
     bbox, reject if `signed_distance_field(use_sign_winding_number=True)
     < 0`). Same algorithm as GeoPT but with winding-number sign
     instead of FCPW `contains`.
   - Build the M1+M2 ops on the aligned mesh, generate 100 walks.
   - Assemble the `DomainMesh`:
     - `interior.points` — `(N+M, 3)` concatenated volume + surface.
     - `interior.point_data.region` — `(N+M,)` int8: 0=volume, 1=surface.
     - `interior.point_data.sdf` — `(N+M,)` from the volume-sampling pass.
     - `interior.point_data.normals_face_barycentric` — `(N+M, 3)`,
       zero for volume rows.
     - `interior.point_data.normals_vertex_nearest` — same shape.
     - `interior.point_data.directions` — `(n_walks, N+M, 3)`.
     - `interior.point_data.step_lengths` — `(n_walks, N+M)`.
     - `interior.point_data.supervise` — `(n_walks, N+M, 3*steps)`.
     - `boundaries.geometry` — `Mesh` with the aligned triangle soup.
     - `global_data.alignment_record` — translation, scale, flip flag.
     - `global_data.config` — `{n_volume_points, n_surface_points,
       n_walks, base_walks, perturb_sigma, steps, max_step}` as
       tensors / int scalars (TensorDict-compatible).

3. `test/experimental/pnm_pretraining/data/test_transforms.py`
   — alignment sanity: one ShapeNet car, after alignment the X-extent
   should equal `target_length=5.0` exactly (± floating-point); the
   Y-min should equal 0 exactly.

4. `test/experimental/pnm_pretraining/data/test_pdmsh_round_trip.py`
   — write a `.pdmsh` on a tiny config (n_volume_points=512,
   n_surface_points=128, n_walks=5, steps=3), read it back via
   `DomainMeshReader`, assert key set, shapes, dtypes, and bit-exact
   tensor recovery.

5. `.worktrees/exp-geopt-datagen-r1/reports/m3-pdmsh-round-trip.md`
   — one-page report:
   - Schema as written; key list with shapes and dtypes.
   - Round-trip wall-clock for the tiny config and for a full
     32768+4096 / 100-walks config.
   - On-disk size of the full-config `.pdmsh` (informs storage
     budgeting for the future 10K-geometry corpus).
   - Note any TensorDict-key gotchas hit during the round-trip.

### 5.3 M3 exit criteria

- (G11) Alignment exits with X-extent and Y-min at target values.
- (G12) Tiny `.pdmsh` round-trip: every tensor recovered bit-exact;
  every TensorDict key present.
- (G13) Full-config `.pdmsh` round-trip wall-clock < 60 s on the
  development machine; on-disk size < 1 GB.
- (G14) The full-config `.pdmsh`, when re-loaded, exposes the
  exact schema enumerated in §5.2 deliverable 2 — no surprise
  keys, no missing keys, dtypes match.

### 5.4 Subagent fan-out for M3

Two parallel feature subagents:

- **Subagent C** — `transforms.py` + `test_transforms.py`. Fully
  isolated.
- **Subagent D** — `builder.py` + `test_pdmsh_round_trip.py`. Depends
  on C's interface but not its implementation; we agree the function
  signature in this plan, and D mocks C until C lands.

---

## 6 — Schedule and decision points

| Milestone | Estimated wall-clock (1 dev) | Subagent fan-out | Blocks |
|---|---|---|---|
| M1 | ~2 days | A + B in parallel | M2 |
| M2 | ~2 days | one subagent | M3 |
| M3 | ~2 days | C + D in parallel | (nothing — round 1 ends) |

After M3: review the three reports together and decide whether the
parent plan's PR 2 (datapipe + adapter + head + loss) can start, or
whether any round-1 finding warrants another round of data-gen work.

**Do not begin M2 before M1's exit criteria pass.** Do not begin M3
before M2's exit criteria pass. Reports are committed alongside the
code that produces them; each milestone is a self-contained PR-shaped
unit even if we don't open upstream PRs from this experimental
worktree.

---

## 7 — Open questions

These don't block round 1 but should be resolved before round 2:

- **Q1.** Does the parent plan's `.pdmsh` schema (§1.2) match what M3
  emits? Differences flagged in M3's report; reconciliation owned by
  the start of round 2.
- **Q2.** GeoPT's vertex-nearest normals are technically buggy. Round
  2 (or PR 2 in the parent plan) decides whether to keep
  `normals_vertex_nearest` as a parity-only artifact or drop it.
- **Q3.** `signed_distance_field`'s lack of a torch reference: do we
  upstream a `torch_forward` for `SignedDistanceField`? Out of scope
  for round 1; opens a PR conversation upstream that the parent plan
  doesn't currently track.
- **Q4.** The General-variant transform's oversize-bbox `*0.5` safety
  factor (F0.6) is a hack. Investigate which categories trigger it on
  the M3 mesh subset; report in M3.

---

## 8 — Improvements over the GeoPT reference

Living list. Every entry is something our implementation does
differently, or better, than `external-repos/GeoPT/data_generation/`.
Each milestone PR appends to this list when it lands a change worth
recording. Round 2 will fold this list into the parent plan's writeup.

| # | Improvement | GeoPT reference behavior | Our behavior | Lands in | Status |
|---|---|---|---|---|---|
| I1 | **Sign convention for vector-distance feature** | `supervise = positions − closest` (query-pointing) — `GeoPT_PreTraining_Data.py:319`. Inconsistent with CFD wall-function conventions. | `supervise = closest − positions` (surface-pointing). See §A. Parity tests negate before comparing. | M2 | landed (M2 — composite kernel emits surface-pointing supervise; parity test `test_constrained_walk_geopt_parity_sphere` confirms sign-aligned 1.24e-7 RMS after negation) |
| I2 | **Inside/outside test on watertight + non-watertight meshes** | `FCPWScene.contains()` — silently misclassifies points even on closed analytic primitives. M1 measurements (500 random query points each, watertight): sphere 6.4% wrong, cube 13.8% wrong. The plan originally framed this as a non-watertight-ShapeNet-only concern; M1 measurements show FCPW is unreliable on watertight inputs too. | `signed_distance_field(use_sign_winding_number=True)` — winding-number sign is **100% correct** vs analytic ground truth on the watertight sphere and cube fixtures (500/500 each). M3 will additionally log the disagreement rate on non-watertight ShapeNet geometries. | M1, M3 | landed (M1 + analytic-watertight measurement); M3 ShapeNet diagnostic still planned |
| I3 | **Surface normals at sampled points** | `compute_normals_improved` — vertex-nearest k=1 lookup against `mesh.vertex_normals`; piecewise-constant per Voronoi cell of vertices. | Default: face-barycentric normals from `sample_points_on_mesh` (smooth, principled). Bug-compatibility port emits both keys (`normals_face_barycentric`, `normals_vertex_nearest`). | M3 | landed (M3 — `builder.py` emits both keys at `interior.point_data.normals_face_barycentric` and `…normals_vertex_nearest`; vertex-nearest port mirrors GeoPT `point_cloud.kdtree.query` verbatim; X-flip applied to both arrays without renormalization since flipped components stay unit) |
| I4 | **Misleading variable names in `transform_mesh`** | Returns `(z_min, x_avg, y_avg, scale)` where `z_min` is post-axis-swap **Y**-min and `y_avg` is post-axis-swap **Z**-mean. Reproducers that read names literally silently misalign. | `align_mesh_geopt_general` returns an `AlignmentRecord` dataclass with named fields (`y_min_post_swap`, `z_mean_post_swap`, `scale`, `axis_flipped`). | M3 | landed (M3 — `physicsnemo/experimental/pnm_pretraining/data/transforms.py`; frozen `AlignmentRecord` with fields `axis_flipped`, `y_min_post_flip`, `scale`, `x_mean_post_scale`, `z_mean_post_scale`, `oversize_safety_applied`, plus `apply` / `inverse` methods; verified by `test_transforms.py` 8 tests including `apply`/`inverse` round-trip) |
| I5 | **Determinism / seeding** | No `np.random.seed` or `torch.manual_seed` anywhere; every run produces different data. | `build_pretraining_sample(seed: int)` seeds NumPy, PyTorch, and Warp deterministically per call. Reproducible by construction. | M3 | landed (M3 — `builder.py::_seed_everything` seeds `np.random.default_rng`, `torch.manual_seed`, `wp.rand_init`; the same seed is forwarded into `generate_walks`) |
| I6 | **On-disk format** | Plain `.npy` (`x.npy`, `supervise_{j}.npy`, `condition_{j}.npy`) per geometry/walk in float16. No metadata, no schema, no per-geometry config trace. | Single `.pdmsh` `DomainMesh` per geometry, all walks stacked. Carries `AlignmentRecord` and full config in `global_data`. Native PhysicsNeMo `DomainMeshReader` consumer. | M3 | landed (M3 — `builder.py::save_pretraining_sample` writes one `.pdmsh` directory per geometry; `test_pdmsh_domain_mesh_reader_consumer` confirms `DomainMeshReader` round-trip) |
| I7 | **Skip-detection robustness** | Skip-if-`x.npy`-exists silently leaves partial walk data on disk after a mid-loop crash (`GeoPT_PreTraining_Data.py:663`). | Atomic write: `.pdmsh.tmp` directory, rename-on-success. Partial writes are detected and overwritten on retry. | M3 | landed (M3 — `builder.py::save_pretraining_sample(atomic=True)` writes `{prefix}.pdmsh.tmp/` then `os.rename`; tests `test_pdmsh_atomic_write_overwrite` and `test_pdmsh_atomic_write_failure_preserves_original` cover the success and failure paths) |
| I8 | **Watertightness diagnostic** | None. | `build_pretraining_sample` records `is_watertight`, fill-holes-attempted flag, and vertex-count delta in `global_data.mesh_quality`. | M3 | landed (M3 — `builder.py` writes `global_data.mesh_quality.{is_watertight, n_vertices_pre_alignment, n_faces}` from the input mesh pre-alignment; sphere fixture lights up `is_watertight=False` due to UV pole-ring stitching, demonstrating the diagnostic in action) |
| I9 | **Float-precision discipline** | `float16` everywhere on disk. ≥3 decimal digits of precision lost for supervision values < 1e-3 in magnitude. | `float32` on disk by default. Optional `dtype: float16` flag for storage budget when corpus generation kicks in (round 2). | M3 | landed (M3 — every persisted tensor is `float32` per `builder.py`; the schema dump in `reports/m3-pdmsh-round-trip.md` enumerates dtypes; `dtype=float16` flag is a round-2 storage-budget knob) |
| I10 | **Walk diversity** | "100 walks" is really 10 independent + 90 jittered (`BASE_WALKS=10`, `PERTURB_SIGMA=0.05`). The "100" is misleading in the README and paper. | Honest API: `generate_walks(n_independent=10, n_jittered_per_base=9, perturb_sigma=0.05)`. Default values match GeoPT for parity; reports document the actual independence count. | M2 | landed (M2 — `generate_walks` exposes the three-knob API; `is_independent: (n_walks,) bool` in the return dict; defaults match GeoPT) |
| I11 | **Single-process Python orchestration** | Serial `for` loop over geometries; "80 cores" is FCPW-internal threading only. No way to scale across geometries from the Python side. | Round 2 will add multi-process / multi-GPU orchestration. Round 1's `build_pretraining_sample` is per-geometry single-process by design (matches scope). | round 2 | deferred |
| I12 | **Numerical-parity test infrastructure** | None — the released repo has no parity tests, no analytic ground truth, no FCPW comparison. | Three-tier: analytic mesh (sphere/cube/torus) RMS at single-precision unit roundoff; trimesh ShapeNet RMS < 1e-4; FCPW closest-point RMS < 1e-4 (installed locally as dev dep, not upstream). | M1 | landed (all three tiers running locally; ShapeNet leg deferred to corpus provisioning) |
| I13 | **Throughput measurement discipline** | README claims "20s per geometry on 80 CPU cores"; no per-op breakdown, no build-vs-query split, no comparison methodology. | M1 emits a per-mesh, per-op, per-query-size, build-vs-query timing table. M2 emits per-walk wall-clock comparison. | M1, M2 | in-progress (M1 build-vs-query table landed, CPU run; M2 per-walk numbers landed in `reports/m2-composite-parity.md`; H100 rerun pending for both) |
| I14 | **Composite-walk supervise on non-watertight inputs is fp16-noise-level** | None — GeoPT has no parity test against PhysicsNeMo, so this regime was unmeasured. The plan originally predicted I2's contains/winding disagreement (13.8% on a watertight cube) would propagate into the composite-walk supervise as measurable RMS divergence. | Measured: composite-walk supervise RMS on a broken-cube parity run (`test_constrained_walk_geopt_parity_broken_cube`) is 4e-8, four orders of magnitude below the fp16 quantization floor. Cause: closest-point itself agrees between FCPW and winding-number kernel within 1.4e-4 even on contains-disagreement points; supervise is `closest − p`, not `sign × \|p − closest\|`, so the sign flip does not contribute. The composite walk is more robust to non-watertightness than the kernel-level SDF gate suggests. | M2 | landed (M2 — measured in `reports/m2-composite-parity.md`) |
| I15 | **`wp.Mesh` rebuild per step in the M2 implementation** | N/A (GeoPT uses FCPW which has the same per-call build cost). | `constrained_walk_step` builds a fresh `wp.Mesh` (BVH + winding-number tree) on every kernel launch. For `n_steps=3, n_walks=100` on a 1024-tri mesh, that's 300 BVH builds. M3's `build_pretraining_sample` should hoist the build to the per-geometry call site and pass `mesh.id` into the orchestrator (and ideally the kernel) so SDF + walk calls within a single sample reuse the mesh. F0.2 already documents the same pattern in `signed_distance_field_impl`. | round 2 | planned (M3 builder-level hoist landed: aligned-mesh tensors stay hot in memory across all SDF + walk calls, so the OBJ load + alignment + I/O are paid once per sample. The `wp.Mesh` BVH-build hoist still requires editing `signed_distance_field_impl`'s signature; deferred to round 2.) |
| I16 | **M3 prompt schema vs. TensorDict batch-size invariant** | N/A — the conflict is internal to PhysicsNeMo's tensorclass design, not a GeoPT carryover. | The originally-specified schema put walk arrays under `interior.point_data`. `Mesh.__post_init__` enforces `point_data.batch_size == [n_points]`, which means every leaf must have leading dim `n_points`. Walk arrays' leading dim is `n_walks`, not `n_points`; `walks_is_independent` is `(n_walks,)` outright. Resolved in M3 by relocating the four walk arrays to `interior.global_data` (`Mesh`-level non-batched dict). Documented in `builder.py` module docstring; consumers should index `dm.interior.global_data["walks_…"]`. | M3 | landed (M3 — schema adjustment lives in `builder.py`; verified by `test_pdmsh_tiny_config_round_trip`; recorded as a new finding in `reports/m3-pdmsh-round-trip.md`) |
| I17 | **Backbone-only checkpoint loading** | GeoPT inlines the partial-load logic in `external-repos/GeoPT/exp/GeoPT_finetune.py:46-65`: a 25-line filter that ignores FSDP/DTensor distribution, doesn't handle `.mdlus` archives, and lives in the experiment script (not as a reusable utility). | `physicsnemo.utils.checkpoint.load_pretrained_backbone(model, ckpt_path, *, strict, key_remap, exclude_layers, device, verbose)` — full-feature partial loader built on top of PhysicsNeMo's existing FSDP/DTensor/mdlus-aware machinery (consumes `_unwrap_ddp_compile`, `_unwrap_fsdp`, `_extract_mdlus_state_dict`, `_get_dtensor_param_placements`, `_redistribute_sd_for_dtensor`, `set_model_state_dict` with `broadcast_from_rank0`). Returns a structured report (`{loaded, skipped_excluded, skipped_missing_target, skipped_shape_mismatch, missing_in_source}`) so callers can audit what actually landed. Auto-unwraps common training-checkpoint containers (`{"model_state_dict": …}`, `{"state_dict": …}`, `{"model": …}`). Reusable across pretraining workflows beyond GeoPT. | PR 2.5 (this commit) | landed |
| I18 | **`strict` semantics flipped from `torch.nn.Module.load_state_dict`** | N/A — GeoPT does not expose a strict flag. | `torch.nn.Module.load_state_dict(strict=True)` means *every model parameter must be filled*. For a backbone-only loader that's the wrong question — the source is by construction a *subset* of the target. `load_pretrained_backbone(strict=True)` instead means *every source key must land* (i.e. `skipped_missing_target` and `skipped_shape_mismatch` are empty). `skipped_excluded` is a deliberate user choice and does not trigger strict failure. The flipped semantics are documented in the function docstring and tested in `test_shape_mismatch_strict_raises` and `test_strict_raises_on_missing_target`. | PR 2.5 (this commit) | landed |

Conventions for adding entries:

- New rows go at the bottom; do not renumber.
- `Lands in` is a milestone tag (M1/M2/M3) or `round 2` for deferred items.
- `Status`: `planned` → `in-progress` → `landed` → `verified` (after the
  matching milestone report ships).

---

## 9 — Optional dependencies

Round-1 introduces or relies on several Python packages that are
**not** in PhysicsNeMo-core's `pyproject.toml` install set today.
Each is listed below with its purpose, install state in our local
venv, and a working note on whether to upstream it later.

| Package | Version installed | Purpose | Where used | Upstream candidate? |
|---------|-------------------|---------|------------|---------------------|
| `scipy` | `1.17.1` | Required transitively by `trimesh.proximity` (closest_point, signed_distance). | `test_sdf_torus_trimesh_parity` (closest-point parity reference). | **No, transitive only.** scipy ships with most PyTorch envs; if a downstream test needs it, gate via `@requires_module("scipy")`. |
| `trimesh` | `3.23.5` | Closest-point + ray-cast reference for parity tests. Also used by GeoPT data-gen for OBJ loading. | M1 SDF parity (torus) and ray-cast parity (sphere); will be reused in M3 for `align_mesh_geopt_general` (loading raw ShapeNet OBJ → cleaned `Mesh`). | **Discuss for upstream.** trimesh is already used in `physicsnemo` examples (DoMINO data prep, SHIFT-SUV preprocessor). It is *not* in core's `pyproject.toml`. Decision deferred until M3 lands and we know whether it's only test-side or also production-side. Round 2 input. |
| `fcpw` | `1.2.0` | Reference closest-point + inside/outside implementation, ported from GeoPT's CPU pipeline. Used as a third-party numerical baseline. | `test_sdf_fcpw_parity`, future M2 composite-walk parity test. | **No, dev-only.** FCPW exists only to validate that PhysicsNeMo + Warp matches an independent CPU reference; not needed at runtime. Stays gated by `@requires_fcpw()`; never imported by production code. |
| `pyembree` | `0.1.12` | Faster ray-mesh-intersection backend for `trimesh.ray`. Pin downgraded `trimesh` from 4.12.2 → 3.23.5 on install (`pyembree` 0.1.12 has a `trimesh<4` constraint on this Python). | M1 ray-cast parity test prefers `ray_pyembree` when available, falls back to `ray_triangle`. | **No, dev-only.** Optional speed-up for tests; the fallback path works without it. Note the `trimesh` downgrade as a dev-env footgun — see "Caveats" below. |

**M1 install commands used (local dev, macOS arm64, Python 3.13):**

```console
$ VIRTUAL_ENV=/Users/carmelog/venv uv pip install scipy fcpw
# Installed: fcpw==1.2.0, scipy==1.17.1
$ VIRTUAL_ENV=/Users/carmelog/venv uv pip install pyembree
# Installed: pyembree==0.1.12 (downgraded trimesh 4.12.2 → 3.23.5)
```

**Caveats.**

- `pyembree` 0.1.12 forces `trimesh<4`. On a fresh PhysicsNeMo
  install, `trimesh` may already be at 4.x; expect this dance.
  Trimesh 3.x has a slightly different proximity API in places —
  M1 tests verified compatibility with 3.23.5; tests should be
  re-verified against 4.x before round 2 if we drop pyembree.
- `fcpw` is exposed via `nanobind` bindings; the API differs between
  1.x (`squared_max_radii` arg required, in-place `interactions`
  output) and earlier versions. `test_sdf_fcpw_parity` is pinned to
  the 1.x signature.
- The pre-commit hook does not enforce these deps. They are install
  responsibilities of the developer or the CI image.

**Decision log on upstreaming.**

- Round-1 keeps all four packages **out of core `pyproject.toml`**.
- Round-2 will revisit `trimesh`. If `physicsnemo.experimental.pnm_pretraining.data.builder`
  needs trimesh at runtime (likely, for OBJ → Mesh conversion in M3),
  it lands as a *runtime* dep on `physicsnemo[experimental_pnm_pretraining]`
  extra, not in the core install set.
- `fcpw`, `pyembree`, and `scipy` (when only used by trimesh) stay
  test-side, gated by `@requires_module(...)` / `@requires_fcpw()`.
- The decision will be reconsidered when M3 lands; this section gets
  updated then.

---

## 10 — Progress log

Append-only. Each entry: timestamp (UTC date), milestone, what
happened, what's next. Updated as work proceeds.

### 2026-05-23 — Plan drafted
- Round-1 plan committed (this file). Worktree `exp-geopt-datagen-r1`
  created off `origin/main`.
- Subagent surveys completed: PhysicsNeMo primitives + GeoPT reference
  data-gen. 10 discovery findings (F0.1–F0.10) folded into the plan.
- Geometry-direction convention (§A) fixed: surface-pointing
  `supervise = closest − query` (opposite of GeoPT). 13 improvements
  (I1–I13) catalogued.
- Next: M1 kicks off — subagents A (`MeshRayIntersection`) and B
  (SDF parity test + BVH-build benchmark) in parallel.

### M1 — kernel parity

- Status: **GREEN with two yellow carry-forwards** (closed 2026-05-23).
  Yellows do not block M2.
- Subagent A landed (`MeshRayIntersection` op + parity tests; commit
  `b7501704`):
  - `physicsnemo/experimental/pnm_pretraining/ops/mesh_ray_intersection.py`
    — Warp-backed `FunctionSpec` mirroring `sdf.py`; outputs
    `(hit_mask, hit_distance, hit_point)`; miss returns
    `(0, inf, origin)` per GeoPT reference behavior. Direction
    convention §A Convention 3 (caller-normalized).
  - `test/experimental/pnm_pretraining/ops/test_mesh_ray_intersection.py`
    — analytic + trimesh parity, 10 cases. All pass on CPU.
- Subagent B landed (SDF parity + BVH bench; commit `c13c23a0`):
  - `test/experimental/pnm_pretraining/ops/test_sdf_geopt_parity.py`
    — five tests (sphere analytic, cube analytic, torus vs trimesh,
    FCPW-gated cross-check, non-watertight diagnostic).
  - `scripts/bench_bvh_build_vs_query.py` + `reports/m1-bvh-build-vs-query.md`
    — per-mesh build / 1k / 10k / 100k / e2e timing table on CPU.
  - `reports/m1-kernel-parity.md` skeleton.
- Closure folded in (post-subagent merge):
  - Quoted subagent A's ray-cast RMS measurements into G1 and G2
    sections of the M1 report (sphere chord-floor 5.7e-3, cube
    bit-exact 0.0, torus float32-floor 1.2e-7, sphere-trimesh
    100% hit-mask + 8.4e-8 dist RMS).
  - Made `test_sdf_torus_trimesh_parity` skip cleanly when scipy is
    not installed (trimesh.proximity needs scipy at runtime).
- Optional deps installed (see new §9): `scipy==1.17.1`,
  `fcpw==1.2.0`, `pyembree==0.1.12` (downgraded `trimesh` 4.12.2 →
  3.23.5 as required by pyembree). Decision log on upstreaming
  recorded in §9; round-1 keeps all four out of core
  `pyproject.toml`.
- FCPW test now runs (commit *this* one). Stronger I2 finding: on
  watertight analytic sphere/cube, FCPW's `contains()` is wrong on
  6.4% (sphere) and 13.8% (cube) of points; PhysicsNeMo's
  winding-number sign is bit-exact correct. The plan originally
  framed FCPW's `contains` deficiency as non-watertight-only;
  M1 measurements show it is unreliable on watertight inputs too.
  I2 status upgraded to "landed" with a stronger story.
- Final M1 test suite (after dep install): **15 passed, 0 skipped**
  on CPU host.
- Gate verdicts: G1 green, G2 yellow (analytic-sphere leg green;
  ShapeNet-corpus leg deferred), G3 green, G4 yellow (CPU; H100 rerun
  needed), G5 **green** (FCPW now runs and passes both subchecks),
  I2 **green+** (stronger story than expected).
- Carry-forwards (do not block M2): fetch ShapeNet subset before M2's
  GeoPT-reference parity; rerun bench on H100.
- Improvements landed: I2 (with FCPW-watertight finding), I12
  (parity-test infrastructure: analytic + trimesh + FCPW tiers all
  running locally), I13 (throughput-measurement discipline).

### M2 — composite parity

- Status: **GREEN** (closed 2026-05-23). G6–G9 all pass with ≥ 5
  orders of margin; G10 is informational and yellow only because of
  a well-understood per-step `wp.Mesh` rebuild (deferred to round 2
  as I15).
- Single-commit feature subagent landed:
  - `physicsnemo/experimental/pnm_pretraining/ops/constrained_walk.py`
    — fused Warp kernel `_constrained_walk_step_kernel` (closest-
    point + supervise + ray-cast + 0.99 sticking + surface pin in one
    launch) plus three Python entry points: `constrained_walk_step`
    (one fused launch), `constrained_walk` (n-step orchestrator
    matching GeoPT semantics including the no-move-on-final-step
    rule), `generate_walks` (10 base + 90 jitter layout, exposed via
    `n_independent + n_jittered_per_base + perturb_sigma`; defaults
    match GeoPT).
  - `test/experimental/pnm_pretraining/ops/test_constrained_walk.py`
    — 6 tests covering G6 (single-step analytic sphere), G7 (0.99
    sticking on collision), G8 (surface-pin invariant, bit-exact),
    I10 (walk-diversity API), G9 sphere parity (fp16-roundtripped,
    sign-negated, 1.24e-7 RMS), G9 broken-cube illustrative
    diagnostic (4e-8 RMS — documents I2 at the composite level).
    Helper `_import_geopt_reference()` stubs `polyscope` so the
    parity tests run against an unmodified GeoPT clone.
  - `physicsnemo/experimental/pnm_pretraining/ops/__init__.py` —
    exports the three new public symbols.
  - `reports/m2-composite-parity.md` — per-gate measurements,
    walk-diversity diagnostic, throughput table, and the I14/I15
    write-ups.
- Test suite: **6 passed, 0 skipped** with `PNM_GEOPT_REF` set; 4
  passed + 2 skipped without it.
- Gate verdicts: G6 green, G7 green, G8 green, G9 **green** (sphere
  RMS 1.24e-7, gate 1e-2), G10 yellow (informational; CPU-only +
  per-step mesh rebuild explains the slowdown on small meshes).
- Improvements landed: I1 (sign convention; surface-pointing supervise
  emitted by the kernel; parity test confirms after negation), I10
  (honest walk-diversity API).
- New findings catalogued: I14 (composite-walk supervise on
  non-watertight inputs is fp16-noise-level — stronger than predicted),
  I15 (`wp.Mesh` rebuilt per step; round-2 hoist).
- Carry-forwards (do not block M3): ShapeNet-corpus parity (still
  deferred behind M1 G2); H100 throughput rerun (bundled with M1
  carry-forward); mesh-build hoisting (I15, round 2).

### M3 — `.pdmsh` round-trip

- Status: GREEN on G11 / G12 / G14; YELLOW on G13 (CPU dispatch
  exceeds the 60 s wall-clock budget; H100 rerun bundled with M1's
  carry-forward).
  - subagent C landed (mesh-alignment slice):
    - Paths created: `physicsnemo/experimental/pnm_pretraining/data/transforms.py`,
      `test/experimental/pnm_pretraining/data/test_transforms.py`. Updated
      `physicsnemo/experimental/pnm_pretraining/data/__init__.py` to
      re-export the public API.
    - Test count: 8 (asymmetric-box arithmetic walk-through; X-extent /
      Y-min / X-mean / Z-mean invariants parametrized over sphere /
      cube / torus; `apply`/`inverse` round-trip on 100 random points;
      oversize-safety branch fires and is recorded; docstring-mentions
      smoke; frozen-dataclass guard).
    - Summary: `AlignmentRecord` is a frozen dataclass with named
      fields (`axis_flipped`, `y_min_post_flip`, `scale`,
      `x_mean_post_scale`, `z_mean_post_scale`,
      `oversize_safety_applied`) plus `apply` / `inverse` methods,
      replacing GeoPT's misleading `(z_min, x_avg, y_avg, scale)` tuple
      (F0.6 / I4). `align_mesh_geopt_general` ports GeoPT's
      General-variant `transform_mesh` (X-flip, no axis swap, oversize
      safety multiplier verbatim from the reference) without mutating
      the input mesh.
    - Commit SHA: `61788357` (recorded in plan; amend chain updates
      the SHA — see `git log --oneline` for the canonical hash).
  - subagent D landed (builder + `.pdmsh` round-trip slice):
    - Paths created: `physicsnemo/experimental/pnm_pretraining/data/builder.py`,
      `test/experimental/pnm_pretraining/data/test_pdmsh_round_trip.py`,
      `reports/m3-pdmsh-round-trip.md`. Merged exports for
      `build_pretraining_sample` / `save_pretraining_sample` /
      `load_pretraining_sample` into the shared
      `physicsnemo/experimental/pnm_pretraining/data/__init__.py`
      alongside C's `AlignmentRecord` / `align_mesh_geopt_general`.
    - Test count: 6 (tiny-config round-trip with bit-exact recovery
      of every leaf; schema invariants — region partition, surface-row
      pinning, walks_is_independent count, supervise_step0 zero on
      surface; alignment-record reconstruction + apply/inverse;
      atomic write success path; atomic write failure preserves the
      original; `DomainMeshReader` consumer round-trip) plus 1
      env-gated full-config wall-clock smoke
      (`PNM_M3_FULL_BENCH=1`).
    - Test suite: 6 passed, 1 skipped (full-config gate);
      `PNM_M3_FULL_BENCH=1` run also passes (16.59 s for the 10-walk
      subset on this CPU host).
    - Gate verdicts: G11 green (subagent C's
      `align_mesh_geopt_general` validates X-extent / Y-min / X-mean
      / Z-mean post-conditions in-function; the recorded
      `AlignmentRecord` round-trips correctly). G12 green (every
      leaf reloads bit-exact at `atol=0, rtol=0`). G13 yellow
      (full-config 32768 + 4096 / 10+90 walks / 3 steps measured at
      129.74 s build / 184.71 MB on-disk on this CPU host; budget is
      60 s on dev machine — H100 rerun is the same carry-forward as
      M1's G4). G14 green (post-load schema parity is enforced by
      both the in-memory test and the `DomainMeshReader` consumer
      test).
    - Improvements landed: I3 (both normals stored), I5
      (deterministic seeding across NumPy / torch / Warp), I6
      (single-`.pdmsh`-per-geometry; consumer round-trip via
      `DomainMeshReader`), I7 (atomic-rename `.pdmsh` write with
      preserve-on-failure semantics), I8 (mesh-quality diagnostic
      under `global_data.mesh_quality`), I9 (float32 on disk by
      default).
    - New finding: I16 (M3 prompt schema vs. TensorDict batch-size
      invariant). The originally-specified `interior.point_data`
      layout for walk arrays is not representable as a per-point
      TensorDict because walk-leading dims are `n_walks`, not
      `n_points`. Resolved by relocating the four walk arrays to
      `interior.global_data` (the `Mesh`-level non-batched dict).
      Documented in `builder.py` and the M3 report.
    - Carry-forwards (do not block round-2 start): full-config H100
      rerun (G13); ShapeNet-corpus consumer test (deferred per the
      user's M3 decision today; round-2 / parent plan PR 2 work);
      `wp.Mesh` BVH-hoist into the SDF op (I15, still round 2).
    - Commit SHA: see `git log --oneline` for the canonical hash;
      this entry is finalized at commit time.

### PR 2.5 — backbone-only checkpoint loading

- Status: **GREEN** (closed 2026-05-23).
- Files:
  - `physicsnemo/utils/checkpoint.py` — added `load_pretrained_backbone`
    alongside `load_model_weights`. New helpers `_apply_key_remap` and
    `_read_source_state_dict` for prefix-matching and
    auto-unwrapping common training-checkpoint containers
    (`model_state_dict` / `state_dict` / `model`). Reuses the existing
    distributed scatter path (`_unwrap_ddp_compile`, `_unwrap_fsdp`,
    `_extract_mdlus_state_dict`, `_get_dtensor_param_placements`,
    `_redistribute_sd_for_dtensor`, DCP `set_model_state_dict` with
    `broadcast_from_rank0` / `strict=False`).
  - `physicsnemo/utils/__init__.py` — re-export
    `load_pretrained_backbone` so it's importable as
    `physicsnemo.utils.load_pretrained_backbone`.
  - `test/utils/test_load_pretrained_backbone.py` — 15 tests covering
    full match, prefix-stripped match, callable remap, exclude-layers,
    shape-mismatch (lenient + strict), missing-source reporting,
    `.pt` round-trip, three wrapped-checkpoint container shapes,
    DDP-wrapped target (gated on `WORLD_SIZE > 1`), `verbose=False`
    silence, and missing-file failure mode.
  - `examples/cfd/external_aerodynamics/unified_external_aero_recipe/src/train.py`
    — Hydra hook between DDP wrap (~line 1254) and the existing
    `load_checkpoint` (~line 1326). Supports both bare-path and
    full-form `cfg.training.pretrained_backbone:`. Normalizes
    `OmegaConf` containers to plain Python before forwarding.
  - `examples/cfd/external_aerodynamics/unified_external_aero_recipe/conf/train.yaml`
    — commented schema example block under the existing `training:`
    section (bare path + full form + every optional sub-key).
- New improvements: I17 (backbone-only checkpoint loading;
  PhysicsNeMo-native replacement for GeoPT's inlined
  `load_pretrained_with_filter`) and I18 (`strict` semantics flipped
  vs. `torch.nn.Module.load_state_dict` because the source is by
  construction a subset of the target — the right question is "did
  every source key land", not "was every model parameter filled").
- Test suite: 14 passed, 1 skipped (DDP-gated). Existing
  `test/utils/test_checkpoint.py` regression-clean: 4 passed, 11
  skipped (CUDA / msc / wandb-mlflow-boto3 gates).
- Round-1 (data-gen) is now feature-complete on the consumer side: a
  future pretraining run can write a checkpoint with `backbone + head`
  weights; a fine-tuner using the unified recipe consumes it via
  `cfg.training.pretrained_backbone: <path>`. The fresh-vs-resume
  semantics work as designed because the new hook runs *after* DDP
  wrap and *before* `load_checkpoint`, so a resume run finds its own
  checkpoint and supersedes the pretrained-backbone load. PR 2
  (backbone adapter, head, loss, `pretrain.py`) is the next chunk.
- Commit SHA: see `git log --oneline` for the canonical hash; this
  entry is finalized at commit time.
