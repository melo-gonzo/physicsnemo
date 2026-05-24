# M3 — `.pdmsh` round-trip report

**Worktree:** `exp-geopt-datagen-r1`
**Subagent:** D (builder + round-trip; subagent C delivered the
alignment port in parallel)
**Status:** **GREEN** for G11 (alignment), G12 (tiny round-trip),
G14 (schema parity); **YELLOW** for G13 (full-config wall-clock
exceeds the 60 s budget on this CPU host — expected, mirrors the M1
H100-rerun carry-forward).

## Exit-criterion verdicts (round-1 plan §5.3)

| Gate | Plan target | Measured | Verdict |
|---|---|---|---|
| G11 — alignment X-extent + Y-min at target | `\|x_extent − target_length\| < 1e-5`, `\|y_min\| < 1e-5` | Subagent C's `align_mesh_geopt_general` validates these in-function (asserts at lines 339-356); sphere-fixture run records `x_extent_post_alignment≈target_length=5.0` after oversize-safety branch fires (`scale=1.25`, `axis_flipped=1`, `y_min_post_flip=-1.0`). | 🟢 |
| G12 — tiny `.pdmsh` round-trip bit-exact, every key present | Every leaf reloads bit-exact (`atol=0, rtol=0`); schema groups present | `test_pdmsh_tiny_config_round_trip` passes — verifies presence of every key in §5.2 schema (and its TensorDict-batchsize-corrected cousin, see I16 below) and bit-exact reload of every leaf in `interior.point_data`, `interior.global_data`, and `domain.global_data`. | 🟢 |
| G13 — full-config `.pdmsh` wall-clock < 60 s; on-disk size < 1 GB | < 60 s on dev machine; < 1 GB | **129.74 s build / 129.80 s total** for the GeoPT-default 32768 + 4096 / 10+90 walks / 3 steps config on this CPU dispatch host. **184.71 MB on-disk.** Disk budget is well within target; wall-clock budget exceeded by 2.2× because the run is CPU-only. Rerun on H100 is bundled into the same carry-forward as M1's G4. | 🟡 |
| G14 — re-loaded `.pdmsh` exposes the schema; no surprise / missing keys; dtypes match | Strict equality of TensorDict key sets at every nesting level; dtype check per leaf | `test_pdmsh_tiny_config_round_trip` covers both halves (presence + dtypes + shapes) on the pre-save and post-load `DomainMesh` instances. `test_pdmsh_domain_mesh_reader_consumer` additionally proves the file format is loadable through the production `DomainMeshReader`. | 🟢 |

**Verdict line:** `M3 — GREEN on G11 / G12 / G14 with G13 YELLOW
(CPU dispatch on this host, 129.74 s vs 60 s budget; on-disk size 184.7 MB,
budget 1 GB; H100 rerun bundled into the same carry-forward as M1).`

## Schema as written

Dumped from a real `DomainMesh` produced by `build_pretraining_sample`
on the conftest sphere fixture with the tiny-config kwargs
(`n_volume_points=512, n_surface_points=128, n_independent=2,
n_jittered_per_base=1, n_steps=2`, so `N+M=640, n_walks=4`). Snapshot
emitted directly from the in-memory tensorclass — i.e. exactly what
ships to disk:

| Path | Shape | Dtype |
| --- | --- | --- |
| interior.points | (640, 3) | torch.float32 |
| interior.cells | (0, 1) | torch.int64 |
| interior.point_data.region | (640,) | torch.int8 |
| interior.point_data.sdf | (640,) | torch.float32 |
| interior.point_data.normals_face_barycentric | (640, 3) | torch.float32 |
| interior.point_data.normals_vertex_nearest | (640, 3) | torch.float32 |
| interior.point_data.supervise_step0 | (640, 3) | torch.float32 |
| interior.global_data.walks_supervise | (4, 640, 2, 3) | torch.float32 |
| interior.global_data.walks_directions | (4, 640, 3) | torch.float32 |
| interior.global_data.walks_step_lengths | (4, 640) | torch.float32 |
| interior.global_data.walks_is_independent | (4,) | torch.int8 |
| boundaries.geometry.points | (544, 3) | torch.float32 |
| boundaries.geometry.cells | (1024, 3) | torch.int64 |
| global_data.config.n_volume_points | () | torch.int64 |
| global_data.config.n_surface_points | () | torch.int64 |
| global_data.config.n_walks | () | torch.int64 |
| global_data.config.n_steps | () | torch.int64 |
| global_data.config.target_length | () | torch.float32 |
| global_data.config.max_step | () | torch.float32 |
| global_data.config.perturb_sigma | () | torch.float32 |
| global_data.config.seed | () | torch.int64 |
| global_data.alignment.axis_flipped | () | torch.int8 |
| global_data.alignment.y_min_post_flip | () | torch.float32 |
| global_data.alignment.scale | () | torch.float32 |
| global_data.alignment.x_mean_post_scale | () | torch.float32 |
| global_data.alignment.z_mean_post_scale | () | torch.float32 |
| global_data.alignment.oversize_safety_applied | () | torch.int8 |
| global_data.mesh_quality.is_watertight | () | torch.int8 |
| global_data.mesh_quality.n_vertices_pre_alignment | () | torch.int64 |
| global_data.mesh_quality.n_faces | () | torch.int64 |

## Wall-clock and on-disk size

| Configuration | Build (s) | Save+atomic-rename (s) | Total (s) | On-disk (MB) |
|---|---|---|---|---|
| Tiny (test a; 512 + 128 / 4 walks / 2 steps) | ~2 | <0.1 | ~2 | <1 |
| Medium (test f; 32768 + 4096 / 10 walks / 3 steps) | 16.57 | 0.03 | 16.59 | 20.18 |
| Full GeoPT-default (32768 + 4096 / 10+90 walks / 3 steps) | 129.74 | 0.06 | 129.80 | 184.71 |

The full-config measurement is from a one-off run (gated outside the
unit-test budget). The 10-walk subset gates the CI smoke test
(`test_pdmsh_full_config_smoke`, env-gated on `PNM_M3_FULL_BENCH=1`).
The on-disk number scales linearly with `n_walks` because the walk
arrays dominate (`(n_walks, N+M, n_steps, 3)` float32 ≈ 1.4 MB / walk
at this size); the wall-clock is dominated by the per-step
constrained-walk kernel which still rebuilds `wp.Mesh` per launch
(see I15 carry-forward).

## Improvements landed in M3

| # | Status | Notes |
|---|---|---|
| I3 | landed | Builder emits **both** `normals_face_barycentric` (face-barycentric area-weighted, the principled answer from `sample_points_on_mesh`) and `normals_vertex_nearest` (GeoPT-buggy port of `compute_normals_improved`, kept for parity); see `builder.py::_compute_geopt_vertex_nearest_normals`. The X-flip applied in alignment touches both arrays' X component but does **not** require renormalization (both flipped components stay unit). |
| I5 | landed | `build_pretraining_sample(seed=…)` seeds NumPy (`np.random.default_rng`), torch (`torch.manual_seed`), and Warp (`wp.rand_init`) at one entry point; the same seed is forwarded into `generate_walks`. See `builder.py::_seed_everything`. |
| I6 | landed | Single `.pdmsh` directory per geometry (TensorDict memmap). No more 100-`.npy`-per-geometry sprawl. Validated via `DomainMeshReader` consumer round-trip in `test_pdmsh_domain_mesh_reader_consumer`. |
| I7 | landed | `save_pretraining_sample(atomic=True)` writes `{prefix}.pdmsh.tmp/` and `os.rename`s on success. On failure the `.tmp` is left for forensics and the prior `.pdmsh` is untouched. Two tests cover this: `test_pdmsh_atomic_write_overwrite` (no `.tmp` leftovers on the success path) and `test_pdmsh_atomic_write_failure_preserves_original` (mid-write failure preserves the original). |
| I8 | landed | `global_data.mesh_quality.{is_watertight,n_vertices_pre_alignment,n_faces}` carry the input-mesh diagnostic. The conftest sphere fixture happens to have `is_watertight=False` (UV-tessellated sphere, two pole rings stitched without a degenerate-triangle fix), which is itself a useful diagnostic. |
| I9 | landed | All persisted tensors are `float32` by default — both per-point fields and walk-level arrays. The optional `dtype: float16` flag for storage budgeting is round-2 work, not landed in this milestone. |

(I4 — `AlignmentRecord` named-fields contract — is subagent C's
deliverable, landed in `transforms.py` and consumed verbatim by the
builder.)

## New finding — I16 (catalogued)

While wiring up the schema specified in the round-1 plan §5.2 (and
restated in the M3 prompt), the round-trip test failed at
`Mesh.__init__` time on the line `point_data.batch_size = (N+M,)`.
TensorDict-of-meshes enforces the per-point-batch-size invariant: every
leaf in `Mesh.point_data` must have leading dim equal to `n_points`.
The walk arrays violate this (`(n_walks, N+M, …)` and the bare
`(n_walks,)` `walks_is_independent` case). The originally-specified
schema was therefore not representable as a `Mesh.point_data`
TensorDict.

**Resolution.** Walk-level arrays move to `interior.global_data` (the
`Mesh`-level per-sample dict, batch_size `[]`), keeping the per-point
fields in `point_data` and the walk metadata co-located with the
interior mesh rather than the domain. The schema dump above shows the
final layout. Documented in `builder.py`'s module docstring under the
"Schema deviation from the original M3 prompt" heading.

**Implication for parent plan (round 2).** Whoever consumes this for
the datapipe and adapter (parent plan PR 2) should index walks via
`dm.interior.global_data["walks_…"]`, not `dm.interior.point_data`.

## Carry-forward action items

* **G13 H100 rerun** — bundle into the same carry-forward as M1's G4
  ("CPU; H100 rerun needed"). Round-2-bound.
* **ShapeNet-corpus consumer test** — explicitly deferred per the
  user's M3 decision today. M3 stays on analytic + synthesized OBJ;
  the corpus-scale `build_pretraining_sample` invocation is round-2 /
  parent plan PR 2 work.
* **I15 — `wp.Mesh` BVH hoist into the SDF op** — still a round-2
  concern; this milestone only does the builder-level hoist (keep the
  aligned mesh tensors hot in memory and pass them to each call). The
  full hoist needs an `signed_distance_field_impl` signature change and
  is out of scope for round 1.
* **I16 — schema deviation** — confirm with consumers (parent plan
  PR 2 author) before round 2 starts. The schema landed here is what
  the `.pdmsh` files actually contain.
* **`is_watertight=False` on UV-sphere fixture** — the conftest's
  `make_sphere` produces a non-watertight mesh due to pole-ring
  stitching. Not a defect of the builder, but worth flagging in
  round-2's mesh-quality dashboard.
