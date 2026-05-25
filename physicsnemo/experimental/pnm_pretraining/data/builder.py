# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end orchestrator for one GeoPT-style pretraining sample.

This module assembles a single ``DomainMesh`` per input geometry,
matching the on-disk schema documented in
``geopt-datagen-round1-plan.md`` §5 (Milestone 3) and the
geometry-direction conventions in §A. The orchestrator chains:

1. Mesh load (OBJ → ``trimesh.Trimesh``).
2. Mesh-quality diagnostic (improvement I8 — watertightness, vertex /
   face counts pre-alignment).
3. Surface-point sampling pre-alignment (improvement I3 — both
   face-barycentric and GeoPT-vertex-nearest normals stored).
4. Alignment via subagent C's ``align_mesh_geopt_general`` (improvement
   I4 — named record fields).
5. Volume-point rejection sampling using winding-number-signed SDF
   (improvement I2 — correct on non-watertight meshes).
6. Constrained-walk supervision (M2's ``generate_walks``).
7. Assembly into a ``DomainMesh`` with the schema below; serialization
   via atomic-rename ``.pdmsh`` (improvement I7).

Round-1 scope is strictly per-geometry: the orchestrator runs
single-process for one input mesh. Corpus-scale fan-out and
multi-process orchestration are round-2 concerns (improvement I11).

Schema (every TensorDict key, shape, dtype)
-------------------------------------------
``DomainMesh`` produced by :func:`build_pretraining_sample`:

::

    interior:                            # Mesh
      points: (N+M, 3) float32           # concatenated volume + surface
      cells:  (0, 1)   int64             # empty (interior is a point cloud)
      point_data:                        # batched on N+M (TensorDict invariant)
        region:                  (N+M,)            int8
        sdf:                     (N+M,)            float32
        normals_face_barycentric:(N+M, 3)          float32
        normals_vertex_nearest:  (N+M, 3)          float32
        supervise_step0:         (N+M, 3)          float32
      global_data:                       # per-mesh, non-batched (see note below)
        walks_supervise:         (n_walks, N+M, n_steps, 3) float32
        walks_directions:        (n_walks, N+M, 3) float32
        walks_step_lengths:      (n_walks, N+M)    float32
        walks_is_independent:    (n_walks,)        int8

    boundaries:
      "geometry": Mesh
        points: (n_vertices, 3) float32
        cells:  (n_faces, 3)    int64

    global_data:                         # domain-level
      config:
        n_volume_points:     ()  int64
        n_surface_points:    ()  int64
        n_walks:             ()  int64
        n_steps:             ()  int64
        target_length:       ()  float32
        max_step:            ()  float32
        perturb_sigma:       ()  float32
        seed:                ()  int64
      alignment:
        axis_flipped:           ()  int8
        y_min_post_flip:        ()  float32
        scale:                  ()  float32
        x_mean_post_scale:      ()  float32
        z_mean_post_scale:      ()  float32
        oversize_safety_applied:()  int8
      mesh_quality:
        is_watertight:           ()  int8
        n_vertices_pre_alignment:()  int64
        n_faces:                 ()  int64

Where ``N == n_volume_points``, ``M == n_surface_points``, and
``n_walks == n_independent * (1 + n_jittered_per_base)``. All
persisted tensors are float32 by default per improvement I9.

Schema deviation from the original M3 prompt (finding I16)
----------------------------------------------------------
The round-1 plan §5.2 deliverable-2 schema put ``walks_supervise``,
``walks_directions``, ``walks_step_lengths``, and
``walks_is_independent`` under ``interior.point_data``. That is
**not representable** as a TensorDict-of-meshes ``point_data`` —
``Mesh.__post_init__`` enforces ``point_data.batch_size ==
torch.Size([n_points])`` for the per-point auto-batching contract,
which means every leaf must have leading dim equal to ``n_points``.
The walk tensors do not (their leading dim is ``n_walks``);
``walks_is_independent`` is ``(n_walks,)`` outright. We resolve the
conflict by moving the four walk arrays into
``interior.global_data`` — the ``Mesh``-level per-sample
non-batched dict — keeping the point-level ``point_data`` clean
and the walk-level metadata co-located with the interior mesh
rather than the domain. Cataloged as improvement / finding I16.

Notes
-----
* ``trimesh`` is imported lazily inside functions, never at module
  top, per round-1 plan §9. Importing this module on a host without
  ``trimesh`` succeeds; only the call sites that actually need OBJ
  I/O require it.
* The volume-row entries of surface-only fields and vice-versa are
  zero. ``walks_step_lengths`` for surface rows is exactly 0 (M2 G8
  surface-pin invariant). ``sdf`` and ``supervise_step0`` for surface
  rows are exactly 0 (the closest point of an on-surface query is
  itself).
"""

from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from physicsnemo.experimental.pnm_pretraining.data.transforms import (
    AlignmentRecord,
    align_mesh_geopt_general,
)
from physicsnemo.experimental.pnm_pretraining.ops.constrained_walk import (
    generate_walks,
)
from physicsnemo.mesh.domain_mesh import DomainMesh
from physicsnemo.mesh.mesh import Mesh
from physicsnemo.models.domino.utils.utils import sample_points_on_mesh
from physicsnemo.nn.functional import signed_distance_field

if TYPE_CHECKING:  # pragma: no cover
    import trimesh as _trimesh_t

__all__ = [
    "build_pretraining_sample",
    "load_pretraining_sample",
    "save_pretraining_sample",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_everything(seed: int) -> np.random.Generator:
    """Seed NumPy / torch / Warp RNGs for reproducibility (improvement I5).

    GeoPT's reference data-gen has no seeding (finding F0.5); our
    reproduction is deterministic by construction. Returns the NumPy
    ``Generator`` for callers that prefer the modern RNG API.
    """
    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))
    try:  # warp's RNG init is best-effort; missing on stripped builds.
        import warp as wp

        wp.rand_init(int(seed))
    except (ImportError, AttributeError):
        pass
    return rng


def _ensure_trimesh_input(
    obj_path: "str | Path | _trimesh_t.Trimesh",
) -> "_trimesh_t.Trimesh":
    """Load a single triangle mesh from path or accept a passthrough.

    `process=False` avoids vertex merging on load (round-1 plan §9
    caveat). If ``obj_path`` is already a ``trimesh.Trimesh``, we
    clone it so downstream mutations (alignment in particular) don't
    surprise the caller. Anything else (e.g. ``trimesh.Scene`` with
    multiple objects) is rejected with a clear error.
    """
    import trimesh

    if isinstance(obj_path, trimesh.Trimesh):
        return obj_path.copy()

    path = Path(obj_path)
    loaded = trimesh.load(path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(
            f"build_pretraining_sample expected a single triangle mesh from "
            f"{path!r}; got {type(loaded).__name__}. Convert to a single "
            f"Trimesh (e.g. via trimesh.util.concatenate(scene.geometry.values()))."
        )
    return loaded


def _compute_geopt_vertex_nearest_normals(
    tm: "_trimesh_t.Trimesh",
    surface_points: np.ndarray,
) -> np.ndarray:
    """Port of GeoPT's ``compute_normals_improved`` (vertex-nearest).

    Reproduces ``GeoPT_PreTraining_Data.py:26-41`` literally — build a
    ``trimesh.PointCloud`` over **mesh vertices**, query the nearest
    vertex for each surface point, and look up
    ``mesh.vertex_normals[nearest_vertex]``.

    NOTE: This is the GeoPT-buggy method, retained only for parity
    (round-1 plan finding F0.7 / improvement I3). The principled
    answer is the face-barycentric normal returned by
    :func:`physicsnemo.models.domino.utils.utils.sample_points_on_mesh`,
    which is also stored under the ``normals_face_barycentric`` key.

    Parameters
    ----------
    tm : trimesh.Trimesh
        The source mesh (pre-alignment in our pipeline).
    surface_points : np.ndarray, shape (M, 3)
        Surface query points in the **same frame** as ``tm`` — i.e.
        pre-alignment, since the caller computes these before the
        alignment is applied.

    Returns
    -------
    np.ndarray, shape (M, 3), float32
        Vertex-nearest unit normal per surface point.
    """
    import trimesh

    # Mirror GeoPT (GeoPT_PreTraining_Data.py:30-34) verbatim: build a
    # PointCloud over the mesh's *vertices*, query its KD-tree for the
    # nearest vertex per surface point, and read the per-vertex normal
    # at that index. Touching ``vertex_normals`` warms trimesh's cache.
    _ = tm.vertex_normals
    point_cloud = trimesh.PointCloud(vertices=np.asarray(tm.vertices))
    _distances, indices = point_cloud.kdtree.query(
        np.asarray(surface_points), k=1
    )
    nearest_vertex_indices = np.asarray(indices).reshape(-1)
    return np.asarray(tm.vertex_normals)[nearest_vertex_indices].astype(
        np.float32, copy=False
    )


def _scalar_tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
    """Cast a Python scalar (incl. bool) to a 0-d tensor of the given dtype."""
    return torch.tensor(value, dtype=dtype)


def _alignment_record_to_tensordict(
    record: AlignmentRecord,
) -> dict[str, torch.Tensor]:
    """Flatten an :class:`AlignmentRecord` into 0-d tensors (TensorDict-friendly).

    Bool flags are stored as int8 because TensorDict's tensorclass
    save/load chokes on torch.bool zero-dim tensors in some versions;
    int8 round-trips cleanly. The ``axis_flipped`` integer is
    interpreted as 1=flip, 0=no-flip — matching the General-variant
    convention (always flips, but the bit is preserved for
    forensic/inverse use).
    """
    return {
        "axis_flipped": _scalar_tensor(int(bool(record.axis_flipped)), torch.int8),
        "y_min_post_flip": _scalar_tensor(float(record.y_min_post_flip), torch.float32),
        "scale": _scalar_tensor(float(record.scale), torch.float32),
        "x_mean_post_scale": _scalar_tensor(
            float(record.x_mean_post_scale), torch.float32
        ),
        "z_mean_post_scale": _scalar_tensor(
            float(record.z_mean_post_scale), torch.float32
        ),
        "oversize_safety_applied": _scalar_tensor(
            int(bool(record.oversize_safety_applied)), torch.int8
        ),
    }


def _rejection_sample_volume_points(
    aligned_vertices: torch.Tensor,
    aligned_faces: torch.Tensor,
    n_volume_points: int,
    bbox_padding: float,
    rng: np.random.Generator,
    *,
    batch_size: int = 65_536,
    max_iter: int = 50,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Uniform-bbox rejection sample, keeping points outside the geometry.

    Implements improvement I2 at the builder level. Convention §A:5 —
    ``signed_distance_field(use_sign_winding_number=True)`` returns
    ``sdf > 0`` for points outside the mesh; we keep those.

    Builder-level mesh hoist (improvement I15): we keep
    ``aligned_vertices`` / ``aligned_faces`` hot in memory and pass
    them to each SDF call rather than reloading the OBJ. The
    ``signed_distance_field_impl`` op still rebuilds ``wp.Mesh``
    internally — that hoist requires touching the op signature, which
    is out of scope for round-1 (kept on the I15 carry-forward list).

    Returns ``(volume_points, sdf_values, closest_points)`` for
    exactly ``n_volume_points`` accepted samples.
    """
    bbox_min = aligned_vertices.min(dim=0).values - bbox_padding
    bbox_max = aligned_vertices.max(dim=0).values + bbox_padding
    extent = bbox_max - bbox_min

    accepted_pts: list[torch.Tensor] = []
    accepted_sdf: list[torch.Tensor] = []
    accepted_closest: list[torch.Tensor] = []
    n_accepted = 0

    for _ in range(max_iter):
        if n_accepted >= n_volume_points:
            break
        # Sample a fresh batch of candidate points uniformly in the padded bbox.
        u = torch.from_numpy(rng.random((batch_size, 3), dtype=np.float32))
        candidate_pts = bbox_min + u * extent

        sdf, closest = signed_distance_field(
            aligned_vertices,
            aligned_faces,
            candidate_pts,
            use_sign_winding_number=True,
        )
        # Convention §A:5 — positive SDF means outside the geometry.
        outside_mask = sdf > 0.0
        accepted_pts.append(candidate_pts[outside_mask])
        accepted_sdf.append(sdf[outside_mask])
        accepted_closest.append(closest[outside_mask])
        n_accepted += int(outside_mask.sum().item())

    if n_accepted == 0:
        raise RuntimeError(
            "Volume-point rejection sampling produced 0 outside points after "
            f"{max_iter} iterations. Mesh may be degenerate or the bbox "
            f"padding ({bbox_padding}) is exhausted by interior."
        )
    if n_accepted < n_volume_points:
        # GeoPT warns and continues; we mirror that. The schema still
        # asks for exactly n_volume_points, but a short fall here would
        # cause a downstream shape error, so we re-pad by repetition.
        warnings.warn(
            f"Volume-point rejection sampling collected {n_accepted} / "
            f"{n_volume_points} points after {max_iter} iterations of "
            f"batch_size={batch_size}. Padding with sample-with-replacement "
            f"to maintain schema shape.",
            stacklevel=2,
        )

    pts = torch.cat(accepted_pts, dim=0)
    sdf = torch.cat(accepted_sdf, dim=0)
    closest = torch.cat(accepted_closest, dim=0)

    if pts.shape[0] >= n_volume_points:
        pts = pts[:n_volume_points]
        sdf = sdf[:n_volume_points]
        closest = closest[:n_volume_points]
    else:
        # Sample-with-replacement padding to maintain schema shape.
        idx = torch.from_numpy(
            rng.integers(0, pts.shape[0], size=(n_volume_points - pts.shape[0]))
        ).long()
        pts = torch.cat([pts, pts[idx]], dim=0)
        sdf = torch.cat([sdf, sdf[idx]], dim=0)
        closest = torch.cat([closest, closest[idx]], dim=0)

    return pts.contiguous(), sdf.contiguous(), closest.contiguous()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_pretraining_sample(
    obj_path: "str | Path | _trimesh_t.Trimesh",
    *,
    n_volume_points: int = 32_768,
    n_surface_points: int = 4_096,
    n_independent_walks: int = 10,
    n_jittered_per_base: int = 9,
    perturb_sigma: float = 0.05,
    n_steps: int = 3,
    max_step: float = 2.0,
    target_length: float = 5.0,
    seed: int = 0,
    device: str | torch.device = "cpu",
    bbox_padding: float = 0.5,
    compute_geopt_compatibility_normals: bool = True,
) -> DomainMesh:
    """Build a single GeoPT-style pretraining sample as a ``DomainMesh``.

    See module docstring for the full output schema. The orchestrator
    is per-geometry by design (round-1 plan §5); corpus-scale fan-out
    is a round-2 concern.

    Parameters
    ----------
    obj_path
        Path to an OBJ (or any other format ``trimesh.load`` handles)
        **or** an already-loaded :class:`trimesh.Trimesh`. Scenes with
        multiple objects are rejected with a clear error.
    n_volume_points, n_surface_points
        Counts for the two interior point partitions. Defaults match
        GeoPT (32768 / 4096).
    n_independent_walks, n_jittered_per_base, perturb_sigma
        Walk-diversity knobs forwarded to
        :func:`physicsnemo.experimental.pnm_pretraining.ops.constrained_walk.generate_walks`.
        Total walk count is ``n_independent_walks * (1 + n_jittered_per_base)``.
        Defaults match GeoPT (10 + 90).
    n_steps, max_step
        Constrained-walk parameters forwarded to ``generate_walks``.
    target_length
        X-extent target for ``align_mesh_geopt_general``. GeoPT uses 5.0.
    seed
        Single seed driving NumPy, PyTorch, and Warp (improvement I5).
    device
        Where to place the output tensors. ``"cpu"`` for the round-1
        builder; CUDA-side construction is supported but the Warp
        ops will dispatch on the input device automatically.
    bbox_padding
        Padding (in aligned-frame units) around the mesh bbox for
        volume-point rejection sampling.
    compute_geopt_compatibility_normals
        If ``True`` (default), additionally compute the GeoPT
        vertex-nearest normals for the surface points. Stored under
        ``normals_vertex_nearest``. Improvement I3 keeps both.

    Returns
    -------
    DomainMesh
        See module docstring for the schema.

    Notes
    -----
    Builder-level mesh hoist (improvement I15): the aligned mesh
    tensors are kept hot in memory across all SDF and walk calls.
    Hoisting the ``wp.Mesh`` BVH itself across calls requires editing
    the op signature and is deferred to round 2 per §8 I15.
    """
    rng = _seed_everything(seed)
    device_t = torch.device(device)

    # Step 2 — load mesh (or accept pre-built).
    tm = _ensure_trimesh_input(obj_path)

    # Step 3 — mesh-quality diagnostic (improvement I8).
    is_watertight = bool(tm.is_watertight)
    n_vertices_pre_alignment = int(np.asarray(tm.vertices).shape[0])
    n_faces = int(np.asarray(tm.faces).shape[0])

    # Step 4 — surface-point sampling pre-alignment.
    # GeoPT order (line 549 / 718): sample on the original mesh, then
    # apply the alignment to the samples *and* the mesh. This keeps
    # the area-weighting honest (alignment is an affine + per-axis
    # scale; the relative areas survive).
    surf_vertices_t = torch.from_numpy(
        np.asarray(tm.vertices, dtype=np.float32)
    ).contiguous()
    surf_faces_t = torch.from_numpy(np.asarray(tm.faces, dtype=np.int64)).contiguous()
    (
        surf_pts_pre,
        _tri_idx,
        _areas,
        surf_normals_face_barycentric_pre,
    ) = sample_points_on_mesh(surf_vertices_t, surf_faces_t, n_surface_points)
    surf_pts_pre = surf_pts_pre.to(torch.float32)
    surf_normals_face_barycentric_pre = surf_normals_face_barycentric_pre.to(
        torch.float32
    )

    if compute_geopt_compatibility_normals:
        # GeoPT-buggy method retained for parity (improvement I3 / F0.7).
        surf_normals_vertex_nearest_pre = torch.from_numpy(
            _compute_geopt_vertex_nearest_normals(tm, surf_pts_pre.cpu().numpy())
        ).to(torch.float32)
    else:
        surf_normals_vertex_nearest_pre = torch.zeros_like(
            surf_normals_face_barycentric_pre
        )

    # Step 5 — apply alignment (subagent C; improvement I4).
    aligned_tm, alignment_record = align_mesh_geopt_general(
        tm, target_length=target_length
    )

    aligned_vertices_t = torch.from_numpy(
        np.asarray(aligned_tm.vertices, dtype=np.float32)
    ).contiguous()
    aligned_faces_t = torch.from_numpy(
        np.asarray(aligned_tm.faces, dtype=np.int64)
    ).contiguous()

    # Apply the alignment to the pre-sampled surface points and surface
    # normals. record.apply handles points; for normals we apply the
    # rotation/flip part only (X-flip negates the X component; isotropic
    # scale is identity for direction vectors). Both face-barycentric
    # and vertex-nearest normal arrays are unit on input; the X-flip
    # leaves both flipped components unit, so we do NOT renormalize.
    surf_pts_post = torch.from_numpy(
        alignment_record.apply(surf_pts_pre.cpu().numpy())
    ).to(torch.float32)

    surf_normals_fb = surf_normals_face_barycentric_pre.clone()
    surf_normals_vn = surf_normals_vertex_nearest_pre.clone()
    if int(bool(alignment_record.axis_flipped)) == 1:
        surf_normals_fb[:, 0] = -surf_normals_fb[:, 0]
        surf_normals_vn[:, 0] = -surf_normals_vn[:, 0]

    # Step 6 — volume-point rejection sampling (I2 + I15 builder hoist).
    vol_pts, vol_sdf, vol_closest = _rejection_sample_volume_points(
        aligned_vertices_t,
        aligned_faces_t,
        n_volume_points,
        bbox_padding=float(bbox_padding),
        rng=rng,
    )

    # Step 7 — supervise_step0 = closest_point - position (Convention §A:1).
    # We have the closest-point output for the kept volume points already
    # (via the rejection-sampling pass). For surface points, the closest
    # point of an on-surface query is itself, so supervise_step0 == 0.
    vol_supervise_step0 = vol_closest - vol_pts
    surf_supervise_step0 = torch.zeros_like(surf_pts_post)
    supervise_step0 = torch.cat([vol_supervise_step0, surf_supervise_step0], dim=0)

    # Step 8 — constrained-walk supervision (M2).
    walks = generate_walks(
        aligned_vertices_t,
        aligned_faces_t,
        vol_pts,
        surf_pts_post,
        n_independent=int(n_independent_walks),
        n_jittered_per_base=int(n_jittered_per_base),
        perturb_sigma=float(perturb_sigma),
        n_steps=int(n_steps),
        max_step=float(max_step),
        seed=int(seed),
    )
    n_walks = int(walks["supervise"].shape[0])

    # Step 9 — assemble interior point cloud.
    interior_points = torch.cat([vol_pts, surf_pts_post], dim=0).to(torch.float32)
    n_total = interior_points.shape[0]

    region = torch.cat(
        [
            torch.zeros(vol_pts.shape[0], dtype=torch.int8),
            torch.ones(surf_pts_post.shape[0], dtype=torch.int8),
        ],
        dim=0,
    )
    sdf_all = torch.cat(
        [
            vol_sdf.to(torch.float32),
            torch.zeros(surf_pts_post.shape[0], dtype=torch.float32),
        ],
        dim=0,
    )
    normals_fb_all = torch.cat(
        [
            torch.zeros((vol_pts.shape[0], 3), dtype=torch.float32),
            surf_normals_fb.to(torch.float32),
        ],
        dim=0,
    )
    normals_vn_all = torch.cat(
        [
            torch.zeros((vol_pts.shape[0], 3), dtype=torch.float32),
            surf_normals_vn.to(torch.float32),
        ],
        dim=0,
    )

    # All persisted tensors fp32 (improvement I9).
    # Per-point fields stay in point_data (TensorDict batch_size=[N+M]
    # invariant). Walk-level arrays move to interior.global_data —
    # see I16 finding in the module docstring.
    point_data: dict[str, torch.Tensor] = {
        "region": region,
        "sdf": sdf_all,
        "normals_face_barycentric": normals_fb_all,
        "normals_vertex_nearest": normals_vn_all,
        "supervise_step0": supervise_step0.to(torch.float32),
    }
    interior_global_data: dict[str, torch.Tensor] = {
        "walks_supervise": walks["supervise"].to(torch.float32),
        "walks_directions": walks["directions"].to(torch.float32),
        "walks_step_lengths": walks["step_lengths"].to(torch.float32),
        "walks_is_independent": walks["is_independent"].to(torch.int8),
    }

    # Sanity: keep the consumer schema honest. Surface walks_step_lengths
    # already pinned to 0 by generate_walks (M2 G8), but verify cheaply
    # rather than rely on that as an unstated postcondition. Use an
    # explicit RuntimeError because `assert` is stripped under
    # `python -O` and this invariant is load-bearing for downstream
    # consumers.
    if surf_pts_post.shape[0] > 0:
        if not torch.all(
            interior_global_data["walks_step_lengths"][:, vol_pts.shape[0] :] == 0
        ):
            raise RuntimeError(
                "Surface-pin schema invariant violated: walks_step_lengths "
                "must be 0 for surface rows. This is a bug in generate_walks."
            )

    interior_mesh = Mesh(
        points=interior_points.to(device_t),
        # Empty cells: interior is a point cloud (volume + surface
        # samples are disjoint). DomainMesh.save will round-trip an
        # empty cells tensor via the (0, 1) sentinel.
        cells=torch.zeros((0, 1), dtype=torch.int64, device=device_t),
        point_data={k: v.to(device_t) for k, v in point_data.items()},
        global_data={k: v.to(device_t) for k, v in interior_global_data.items()},
    )

    # Boundaries: single "geometry" key carrying the aligned mesh.
    geometry_mesh = Mesh(
        points=aligned_vertices_t.to(device_t),
        cells=aligned_faces_t.to(device_t),
    )
    boundaries: dict[str, Mesh] = {"geometry": geometry_mesh}

    # global_data: nested TensorDict with config / alignment / mesh_quality.
    # The DomainMesh ctor coerces dicts to TensorDicts; nested dicts are
    # similarly coerced.
    global_data: dict[str, dict[str, torch.Tensor]] = {
        "config": {
            "n_volume_points": _scalar_tensor(int(n_volume_points), torch.int64),
            "n_surface_points": _scalar_tensor(int(n_surface_points), torch.int64),
            "n_walks": _scalar_tensor(int(n_walks), torch.int64),
            "n_steps": _scalar_tensor(int(n_steps), torch.int64),
            "target_length": _scalar_tensor(float(target_length), torch.float32),
            "max_step": _scalar_tensor(float(max_step), torch.float32),
            "perturb_sigma": _scalar_tensor(float(perturb_sigma), torch.float32),
            "seed": _scalar_tensor(int(seed), torch.int64),
        },
        "alignment": _alignment_record_to_tensordict(alignment_record),
        "mesh_quality": {
            "is_watertight": _scalar_tensor(int(is_watertight), torch.int8),
            "n_vertices_pre_alignment": _scalar_tensor(
                int(n_vertices_pre_alignment), torch.int64
            ),
            "n_faces": _scalar_tensor(int(n_faces), torch.int64),
        },
    }

    domain_mesh = DomainMesh(
        interior=interior_mesh,
        boundaries=boundaries,
        global_data=global_data,
    )

    # Light internal sanity: the schema is expected to land on
    # exactly n_total interior points.
    assert domain_mesh.interior.n_points == n_total

    return domain_mesh


def save_pretraining_sample(
    domain_mesh: DomainMesh,
    prefix: str | Path,
    *,
    atomic: bool = True,
) -> Path:
    """Persist a built ``DomainMesh`` to disk as a ``.pdmsh`` directory.

    GeoPT's reference data-gen has no atomic write discipline:
    ``GeoPT_PreTraining_Data.py:663`` skips when ``x.npy`` exists,
    which silently leaves partial walk data on disk after a mid-loop
    crash. Improvement I7 fixes this with a ``.pdmsh.tmp`` →
    ``.pdmsh`` atomic rename.

    Parameters
    ----------
    domain_mesh
        The built sample. Must be the output of
        :func:`build_pretraining_sample` (or schema-equivalent).
    prefix
        Destination prefix. The on-disk directory is
        ``f"{prefix}.pdmsh/"`` (matching ``DomainMeshReader``'s
        default ``**/*.pdmsh`` glob). If ``prefix`` already ends in
        ``.pdmsh``, the suffix is not duplicated.
    atomic
        When ``True`` (default), write to ``f"{prefix}.pdmsh.tmp/"``
        and ``os.rename`` to ``f"{prefix}.pdmsh/"`` on success. The
        rename is atomic on POSIX same-filesystem writes. On failure
        the ``.tmp`` directory is left for forensics; any
        pre-existing ``.pdmsh`` directory is cleared *only after*
        the staging write succeeds.

    Returns
    -------
    Path
        Absolute path to the final ``.pdmsh`` directory.
    """
    prefix = Path(prefix)
    if prefix.suffix == ".pdmsh":
        final = prefix
    else:
        final = prefix.with_suffix(prefix.suffix + ".pdmsh")
    final = final.resolve()

    if not atomic:
        # Allow the caller to opt out of atomic-rename for testing the
        # GeoPT failure mode; still fine for one-shot writes when no
        # corruption recovery is needed.
        if final.exists():
            shutil.rmtree(final)
        final.parent.mkdir(parents=True, exist_ok=True)
        domain_mesh.save(str(final))
        return final

    staging = final.parent / (final.name + ".tmp")
    final.parent.mkdir(parents=True, exist_ok=True)
    # Clean any stale staging from a prior crash.
    if staging.exists():
        shutil.rmtree(staging)

    # tensorclass.memmap creates the directory; we let it do that.
    domain_mesh.save(str(staging))

    # Now that staging succeeded, swap it in for any prior final
    # directory. ``os.replace`` is atomic on the same filesystem; for
    # directory targets it requires the target be empty or also a
    # directory — we explicitly clear the old final first.
    if final.exists():
        shutil.rmtree(final)
    os.rename(staging, final)
    return final


def load_pretraining_sample(prefix: str | Path) -> DomainMesh:
    """Thin wrapper for :meth:`DomainMesh.load`.

    Loads a ``.pdmsh`` directory written by
    :func:`save_pretraining_sample` (or any schema-equivalent
    producer). The expected schema is documented in the module
    docstring; this loader does not validate against it — schema
    drift surfaces at first-key-access time.
    """
    prefix = Path(prefix)
    if prefix.suffix != ".pdmsh":
        prefix = prefix.with_suffix(prefix.suffix + ".pdmsh")
    return DomainMesh.load(str(prefix))
