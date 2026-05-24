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

"""Parity tests for ``MeshRayIntersection``.

Mirrors ``test/nn/functional/geometry/test_sdf.py`` in structure. Validates
the Warp-backed ray-cast against analytic ground truth on the three
fixture meshes (sphere, cube, torus) and against ``trimesh`` on a sphere
when the optional ray-intersector backends are importable.

See ``geopt-datagen-round1-plan.md`` §3 (Milestone 1) for the M1 exit
criteria this file ticks off.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from physicsnemo.experimental.pnm_pretraining.ops import (
    MeshRayIntersection,
    mesh_ray_intersection,
)
from test.conftest import requires_module

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fibonacci_sphere(n: int, radius: float = 1.0) -> np.ndarray:
    """Return ``n`` quasi-uniform points on a sphere of the given radius."""
    # Golden-angle Fibonacci lattice; deterministic for parity tests.
    indices = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / n)
    theta = math.pi * (1.0 + 5.0**0.5) * indices
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return radius * np.stack([x, y, z], axis=-1)


# ---------------------------------------------------------------------------
# (a) Analytic sphere
# ---------------------------------------------------------------------------


# Validate the warp-backed ray-cast against an analytic sphere of radius 1.
@requires_module("warp")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mesh_ray_intersection_sphere(dtype: torch.dtype, device: str, sphere_mesh):
    device = torch.device(device)
    verts, faces = sphere_mesh.to_torch(device=device)
    verts = verts.to(dtype=dtype)

    # 100 ray origins on a Fibonacci-sphere shell at radius 2, all aimed
    # back at the world origin (i.e. inward toward the sphere).
    origins_np = _fibonacci_sphere(100, radius=2.0)
    origins = torch.as_tensor(origins_np, device=device, dtype=dtype)
    origin_norm = torch.linalg.norm(origins, dim=-1, keepdim=True)
    directions = -origins / origin_norm

    hit_mask, hit_distance, hit_point = mesh_ray_intersection(
        verts, faces, origins, directions, max_dist=10.0
    )

    # All 100 rays must hit the sphere.
    assert hit_mask.dtype == torch.int32
    assert torch.all(hit_mask == 1), (
        f"some rays missed the sphere: {(hit_mask == 0).sum().item()}/100"
    )

    # Expected hit distance: |origin| - radius = 2 - 1 = 1, with a loose
    # tolerance because the UV-sphere is a faceted approximation. The
    # worst-case chord error for a UV-tessellation with ``n_segments``
    # equator divisions is ~``1 - cos(pi / n_segments)``; for the conftest
    # default of 32 segments that is ~4.8e-3, so we use a 1e-2 envelope.
    expected = torch.full((100,), 1.0, device=device, dtype=dtype)
    rms = torch.sqrt(torch.mean((hit_distance - expected) ** 2)).item()
    assert rms < 1e-2, f"sphere hit-distance RMS too large: {rms}"

    # Hit points should be at radius ~1 from world origin (same envelope).
    radii = torch.linalg.norm(hit_point, dim=-1)
    radii_rms = torch.sqrt(torch.mean((radii - 1.0) ** 2)).item()
    assert radii_rms < 1e-2, f"sphere hit-point radius RMS too large: {radii_rms}"


# ---------------------------------------------------------------------------
# (b) Analytic cube — face-center hits and outward-pointing misses
# ---------------------------------------------------------------------------


# Validate axis-aligned ray-cast against the unit cube spanning [-1, 1]^3.
@requires_module("warp")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mesh_ray_intersection_cube(dtype: torch.dtype, device: str, cube_mesh):
    device = torch.device(device)
    verts, faces = cube_mesh.to_torch(device=device)
    verts = verts.to(dtype=dtype)

    # 6 axis-aligned rays from origin out through the face centers.
    directions_inward = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        device=device,
        dtype=dtype,
    )
    origins_inward = torch.zeros(6, 3, device=device, dtype=dtype)

    hit_mask, hit_distance, hit_point = mesh_ray_intersection(
        verts, faces, origins_inward, directions_inward, max_dist=10.0
    )

    assert torch.all(hit_mask == 1), f"face-center mask: {hit_mask.tolist()}"
    expected = torch.full((6,), 1.0, device=device, dtype=dtype)
    rms = torch.sqrt(torch.mean((hit_distance - expected) ** 2)).item()
    assert rms < 1e-5, f"cube face-center hit-distance RMS too large: {rms}"

    # 6 rays sitting *outside* the cube and pointing further outward must
    # miss; on miss, hit_distance == inf and hit_point == origin.
    origins_outward = 2.0 * directions_inward
    hit_mask_m, hit_distance_m, hit_point_m = mesh_ray_intersection(
        verts, faces, origins_outward, directions_inward, max_dist=10.0
    )

    assert torch.all(hit_mask_m == 0), f"outward miss mask: {hit_mask_m.tolist()}"
    assert torch.all(torch.isinf(hit_distance_m)), (
        f"outward miss distances: {hit_distance_m.tolist()}"
    )
    torch.testing.assert_close(hit_point_m, origins_outward)


# ---------------------------------------------------------------------------
# (c) Analytic torus — donut-hole miss and ring-edge hit
# ---------------------------------------------------------------------------


# Validate ray-cast against a torus (R=2, r=0.5, axis along Z).
@requires_module("warp")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mesh_ray_intersection_torus(dtype: torch.dtype, device: str, torus_mesh):
    device = torch.device(device)
    verts, faces = torus_mesh.to_torch(device=device)
    verts = verts.to(dtype=dtype)

    # Hit case: ray from (4, 0, 0) pointing in -X must hit the torus's
    # outer ring at distance 4 - (R + r) = 4 - 2.5 = 1.5.
    hit_origins = torch.tensor([[4.0, 0.0, 0.0]], device=device, dtype=dtype)
    hit_dirs = torch.tensor([[-1.0, 0.0, 0.0]], device=device, dtype=dtype)
    hit_mask, hit_distance, _ = mesh_ray_intersection(
        verts, faces, hit_origins, hit_dirs, max_dist=10.0
    )
    assert hit_mask.tolist() == [1]
    expected = torch.tensor([1.5], device=device, dtype=dtype)
    rms = torch.sqrt(torch.mean((hit_distance - expected) ** 2)).item()
    assert rms < 1e-3, f"torus outer-ring hit RMS too large: {rms}"

    # Miss case: ray from (0, 0, 5) pointing in -Z passes through the
    # donut hole (the torus axis), so it never hits the surface.
    miss_origins = torch.tensor([[0.0, 0.0, 5.0]], device=device, dtype=dtype)
    miss_dirs = torch.tensor([[0.0, 0.0, -1.0]], device=device, dtype=dtype)
    miss_mask, miss_distance, miss_point = mesh_ray_intersection(
        verts, faces, miss_origins, miss_dirs, max_dist=10.0
    )
    assert miss_mask.tolist() == [0]
    assert torch.all(torch.isinf(miss_distance))
    torch.testing.assert_close(miss_point, miss_origins)


# ---------------------------------------------------------------------------
# (d) Index-layout compatibility — flat (3F,) vs face-triplet (F, 3)
# ---------------------------------------------------------------------------


# Validate ray-cast index-shape compatibility paths.
@requires_module("warp")
def test_mesh_ray_intersection_index_layout_compatibility(device: str, cube_mesh):
    device = torch.device(device)
    verts, faces_2d = cube_mesh.to_torch(device=device)
    faces_flat = faces_2d.reshape(-1)

    origins = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], device=device, dtype=torch.float32
    )
    directions = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device, dtype=torch.float32
    )

    out_flat = mesh_ray_intersection(verts, faces_flat, origins, directions)
    out_faces = mesh_ray_intersection(verts, faces_2d, origins, directions)
    torch.testing.assert_close(out_flat[0], out_faces[0])
    torch.testing.assert_close(out_flat[1], out_faces[1])
    torch.testing.assert_close(out_flat[2], out_faces[2])


# ---------------------------------------------------------------------------
# (e) make_inputs_forward smoke — exercise all three benchmark sizes
# ---------------------------------------------------------------------------


# Validate benchmark input generation contract for MeshRayIntersection.
@requires_module("warp")
def test_mesh_ray_intersection_make_inputs_forward(device: str):
    seen_labels: list[str] = []
    for label, args, kwargs in MeshRayIntersection.make_inputs_forward(device=device):
        assert isinstance(label, str)
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)
        seen_labels.append(label)

        hit_mask, hit_distance, hit_point = MeshRayIntersection.dispatch(
            *args, implementation="warp", **kwargs
        )
        # The fourth positional arg is ray_directions, same shape as origins.
        n_rays = args[2].shape[0]
        assert hit_mask.shape == (n_rays,)
        assert hit_distance.shape == (n_rays,)
        assert hit_point.shape == (n_rays, 3)
        assert hit_mask.dtype == torch.int32
        assert hit_distance.dtype == torch.float32
        assert hit_point.dtype == torch.float32

    # All three benchmark sizes were exercised.
    assert any(label.startswith("small-") for label in seen_labels)
    assert any(label.startswith("medium-") for label in seen_labels)
    assert any(label.startswith("large-") for label in seen_labels)


# ---------------------------------------------------------------------------
# (f) Error handling — bad direction shape, bad index shape
# ---------------------------------------------------------------------------


# Validate ray-cast input and shape error handling paths.
@requires_module("warp")
def test_mesh_ray_intersection_error_handling(device: str, cube_mesh):
    device = torch.device(device)
    verts, faces = cube_mesh.to_torch(device=device)
    origins = torch.zeros(4, 3, device=device, dtype=torch.float32)
    directions = torch.tensor([[1.0, 0.0, 0.0]] * 4, device=device, dtype=torch.float32)

    # Ray direction must have last dimension of size 3.
    bad_dirs = torch.randn(4, device=device, dtype=torch.float32)
    with pytest.raises(ValueError, match="last dimension of size 3"):
        mesh_ray_intersection(verts, faces, origins, bad_dirs)

    # Ray origin must have last dimension of size 3.
    bad_origins = torch.randn(4, 2, device=device, dtype=torch.float32)
    with pytest.raises(ValueError, match="last dimension of size 3"):
        mesh_ray_intersection(verts, faces, bad_origins, directions)

    # 2D mesh indices must be shaped as (n_faces, 3).
    bad_connectivity_shape = torch.zeros(4, 4, device=device, dtype=torch.int32)
    with pytest.raises(ValueError, match="shape \\(n_faces, 3\\)"):
        mesh_ray_intersection(verts, bad_connectivity_shape, origins, directions)

    # Connectivity may be 1D flattened or 2D triangular faces only.
    bad_connectivity_rank = torch.zeros(1, 2, 3, device=device, dtype=torch.int32)
    with pytest.raises(ValueError, match="1D flattened indices or 2D"):
        mesh_ray_intersection(verts, bad_connectivity_rank, origins, directions)

    # ray_origins and ray_directions must have identical shapes.
    mismatched_dirs = torch.randn(8, 3, device=device, dtype=torch.float32)
    with pytest.raises(ValueError, match="identical shapes"):
        mesh_ray_intersection(verts, faces, origins, mismatched_dirs)


# ---------------------------------------------------------------------------
# (g) trimesh parity — sphere cross-check
# ---------------------------------------------------------------------------


# Validate ray-cast against trimesh's first-hit ray intersector on a sphere.
# Skipped cleanly if trimesh or its ray modules are unavailable. The
# ShapeNet subset is not strictly required for round-1 M1 — see
# geopt-datagen-round1-plan.md §3.4.
@requires_module("warp")
@requires_module("trimesh")
def test_mesh_ray_intersection_trimesh_parity_sphere(device: str, sphere_mesh):
    pytest.importorskip("trimesh")
    import trimesh  # noqa: F401

    # Prefer the Embree backend if available; fall back to the slower pure
    # python intersector otherwise. Skip cleanly if neither imports.
    intersector_cls = None
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector as _RPE

        intersector_cls = _RPE
    except Exception:
        try:
            from trimesh.ray.ray_triangle import RayMeshIntersector as _RT

            intersector_cls = _RT
        except Exception:
            pytest.skip("trimesh.ray backends not available")

    device = torch.device(device)

    tm = trimesh.Trimesh(
        vertices=sphere_mesh.vertices.astype(np.float64),
        faces=sphere_mesh.indices.astype(np.int64),
        process=False,
    )
    intersector = intersector_cls(tm)

    n_rays = 1000
    origins_np = _fibonacci_sphere(n_rays, radius=2.0).astype(np.float32)
    origin_norms = np.linalg.norm(origins_np, axis=-1, keepdims=True)
    dirs_np = (-origins_np / origin_norms).astype(np.float32)

    # trimesh: intersects_first returns the per-ray index of the first hit
    # face, or -1 on miss. intersects_location returns the world-space
    # hit points and a per-hit ray index.
    first_face_idx = intersector.intersects_first(
        ray_origins=origins_np, ray_directions=dirs_np
    )
    locations, ray_indices, _ = intersector.intersects_location(
        ray_origins=origins_np, ray_directions=dirs_np, multiple_hits=False
    )

    trimesh_mask = (first_face_idx >= 0).astype(np.int32)
    trimesh_distance = np.full(n_rays, np.inf, dtype=np.float32)
    if len(ray_indices) > 0:
        diffs = locations.astype(np.float32) - origins_np[ray_indices]
        trimesh_distance[ray_indices] = np.linalg.norm(diffs, axis=-1)

    # PhysicsNeMo Warp implementation.
    verts = torch.as_tensor(sphere_mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.as_tensor(sphere_mesh.indices, dtype=torch.int32, device=device)
    origins_t = torch.as_tensor(origins_np, dtype=torch.float32, device=device)
    dirs_t = torch.as_tensor(dirs_np, dtype=torch.float32, device=device)

    pnm_mask, pnm_distance, _ = mesh_ray_intersection(
        verts, faces, origins_t, dirs_t, max_dist=10.0
    )
    pnm_mask_np = pnm_mask.cpu().numpy().astype(np.int32)
    pnm_distance_np = pnm_distance.cpu().numpy().astype(np.float32)

    # Hit-mask agreement > 99% (allow boundary-grazing rays to disagree).
    agreement = float(np.mean(pnm_mask_np == trimesh_mask))
    assert agreement > 0.99, f"hit-mask agreement {agreement:.4f} below 99%"

    # Hit-distance RMS over the hit subset where both backends agree.
    both_hit = (pnm_mask_np == 1) & (trimesh_mask == 1)
    if both_hit.any():
        diff = pnm_distance_np[both_hit] - trimesh_distance[both_hit]
        rms = float(np.sqrt(np.mean(diff**2)))
        assert rms < 1e-4, f"hit-distance RMS {rms} above 1e-4"
