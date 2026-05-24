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

"""Mesh-ray intersection ``FunctionSpec`` for the GeoPT data-gen pipeline.

This op casts a batch of rays against a triangular surface mesh and reports,
per ray, whether it hit the mesh, the parametric distance to the first hit,
and the world-space hit point. The implementation mirrors the BVH-backed
warp kernel pattern used by
:mod:`physicsnemo.nn.functional.geometry.sdf`.

Direction convention
--------------------
Per ``geopt-datagen-round1-plan.md`` §A *Convention 3 — ray direction*,
``ray_directions`` are **unit vectors** that the caller is responsible for
normalizing. The kernel does **not** internally renormalize, matching the
``signed_distance_field`` discipline. The hit point is therefore
``origin + t * direction`` for the returned ``t = hit_distance``. On a
miss, the reference behavior (mirrored from
``external-repos/GeoPT/data_generation/GeoPT_PreTraining_Data.py:127-131``)
is ``hit_distance = inf``, ``hit_point = origin``, ``hit_mask = 0``.
"""

import torch
import warp as wp
from jaxtyping import Float

from physicsnemo.core.function_spec import FunctionSpec

# Warp is a required dependency in v2.0+.

wp.config.quiet = True


@wp.kernel
def _mesh_ray_intersection_kernel(
    mesh_id: wp.uint64,
    ray_origins: wp.array(dtype=wp.vec3f),
    ray_directions: wp.array(dtype=wp.vec3f),
    max_dist: wp.float32,
    hit_mask: wp.array(dtype=wp.int32),
    hit_distance: wp.array(dtype=wp.float32),
    hit_point: wp.array(dtype=wp.vec3f),
):
    """Per-ray BVH ray-cast against the given triangular mesh.

    On hit, writes ``hit_mask=1``, ``hit_distance=t`` (parametric distance
    along the unit-vector direction), and ``hit_point=origin + t*direction``.
    On miss, writes ``hit_mask=0``, ``hit_distance=inf``, ``hit_point=origin``
    — matching the GeoPT reference's miss semantics.

    Parameters
    ----------
    mesh_id : wp.uint64
        Identifier of the warp ``wp.Mesh`` to query.
    ray_origins : wp.array(dtype=wp.vec3f)
        Per-ray origin in world space.
    ray_directions : wp.array(dtype=wp.vec3f)
        Per-ray unit-vector direction in world space.
    max_dist : wp.float32
        Maximum traversal distance along each ray.
    hit_mask : wp.array(dtype=wp.int32)
        Output: ``1`` on hit, ``0`` on miss.
    hit_distance : wp.array(dtype=wp.float32)
        Output: parametric ``t`` on hit, ``inf`` on miss.
    hit_point : wp.array(dtype=wp.vec3f)
        Output: ``origin + t*direction`` on hit, ``origin`` on miss.
    """
    tid = wp.tid()

    origin = ray_origins[tid]
    direction = ray_directions[tid]

    query = wp.mesh_query_ray(mesh_id, origin, direction, max_dist)

    if query.result:
        hit_mask[tid] = wp.int32(1)
        hit_distance[tid] = query.t
        hit_point[tid] = origin + query.t * direction
    else:
        hit_mask[tid] = wp.int32(0)
        hit_distance[tid] = wp.float32(wp.inf)
        hit_point[tid] = origin


@torch.library.custom_op("physicsnemo::mesh_ray_intersection", mutates_args=())
def mesh_ray_intersection_impl(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    max_dist: float = 1e8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the first-hit ray-mesh intersection for a batch of rays.

    The mesh must be a triangular surface mesh. Uses NVIDIA Warp for GPU
    acceleration (via the BVH built by :class:`warp.Mesh`).

    Parameters
    ----------
    mesh_vertices : torch.Tensor
        Vertex positions, shape ``(n_vertices, 3)``.
    mesh_indices : torch.Tensor
        Triangle connectivity. Either flattened ``(3 * n_faces,)`` or
        face-triplet ``(n_faces, 3)``.
    ray_origins : torch.Tensor
        Ray origins with last dimension of size 3, shape ``(..., 3)``.
    ray_directions : torch.Tensor
        Ray directions, must be **unit vectors** with last dimension of size
        3 and same leading shape as ``ray_origins``. The kernel does not
        internally normalize — see ``geopt-datagen-round1-plan.md`` §A
        Convention 3.
    max_dist : float, optional
        Maximum traversal distance along each ray. Default is ``1e8``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(hit_mask, hit_distance, hit_point)`` where:

        - ``hit_mask`` is ``int32``, leading shape of the rays; ``1`` on
          hit, ``0`` on miss.
        - ``hit_distance`` is ``float32`` (cast to caller dtype on return),
          leading shape of the rays; parametric ``t`` on hit, ``inf`` on
          miss.
        - ``hit_point`` is shape ``(..., 3)`` matching the ray origins;
          ``origin + t*direction`` on hit, ``origin`` on miss.
    """
    if ray_origins.shape[-1] != 3:
        raise ValueError("ray_origins must have last dimension of size 3")
    if ray_directions.shape[-1] != 3:
        raise ValueError("ray_directions must have last dimension of size 3")
    if ray_origins.shape != ray_directions.shape:
        raise ValueError(
            "ray_origins and ray_directions must have identical shapes; "
            f"got {tuple(ray_origins.shape)} vs {tuple(ray_directions.shape)}"
        )

    # Accept either flattened indices or face-triplet connectivity.
    if mesh_indices.ndim == 2:
        if mesh_indices.shape[-1] != 3:
            raise ValueError(
                "mesh_indices with 2 dimensions must have shape (n_faces, 3)"
            )
        mesh_indices = mesh_indices.reshape(-1)
    elif mesh_indices.ndim != 1:
        raise ValueError(
            "mesh_indices must be either 1D flattened indices or 2D (n_faces, 3)"
        )

    input_shape = ray_origins.shape

    # Flatten rays to (N, 3).
    ray_origins_flat = ray_origins.reshape(-1, 3)
    ray_directions_flat = ray_directions.reshape(-1, 3)

    N = len(ray_origins_flat)

    # Allocate output tensors with torch.
    hit_mask = torch.zeros(N, dtype=torch.int32, device=ray_origins.device)
    hit_distance = torch.zeros(N, dtype=torch.float32, device=ray_origins.device)
    hit_point = torch.zeros(N, 3, dtype=torch.float32, device=ray_origins.device)

    wp_launch_device, wp_launch_stream = FunctionSpec.warp_launch_context(ray_origins)

    with wp.ScopedStream(wp_launch_stream):
        wp.init()

        # Zero-copy mesh and rays into warp.
        wp_vertices = wp.from_torch(mesh_vertices.to(torch.float32), dtype=wp.vec3)
        wp_indices = wp.from_torch(
            mesh_indices.to(torch.int32).contiguous(), dtype=wp.int32
        )
        wp_origins = wp.from_torch(
            ray_origins_flat.to(torch.float32).contiguous(), dtype=wp.vec3
        )
        wp_directions = wp.from_torch(
            ray_directions_flat.to(torch.float32).contiguous(), dtype=wp.vec3
        )

        # Outputs.
        wp_hit_mask = wp.from_torch(hit_mask, dtype=wp.int32)
        wp_hit_distance = wp.from_torch(hit_distance, dtype=wp.float32)
        wp_hit_point = wp.from_torch(hit_point, dtype=wp.vec3f)

        mesh = wp.Mesh(points=wp_vertices, indices=wp_indices)

        wp.launch(
            kernel=_mesh_ray_intersection_kernel,
            dim=N,
            inputs=[
                mesh.id,
                wp_origins,
                wp_directions,
                max_dist,
                wp_hit_mask,
                wp_hit_distance,
                wp_hit_point,
            ],
            device=wp_launch_device,
            stream=wp_launch_stream,
        )

    # Unflatten outputs.
    leading_shape = input_shape[:-1]
    hit_mask = hit_mask.reshape(leading_shape)
    hit_distance = hit_distance.reshape(leading_shape)
    hit_point = hit_point.reshape(input_shape)

    # hit_mask stays int32 (it is a categorical flag, not a value in the
    # ray's float dtype). hit_distance and hit_point follow the caller dtype.
    return (
        hit_mask,
        hit_distance.to(ray_origins.dtype),
        hit_point.to(ray_origins.dtype),
    )


@mesh_ray_intersection_impl.register_fake
def mesh_ray_intersection_impl_fake(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    max_dist: float = 1e8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mesh_vertices.device != ray_origins.device:
        raise RuntimeError("mesh_vertices and ray_origins must be on the same device")

    if mesh_vertices.device != mesh_indices.device:
        raise RuntimeError("mesh_vertices and mesh_indices must be on the same device")

    if ray_origins.device != ray_directions.device:
        raise RuntimeError("ray_origins and ray_directions must be on the same device")

    leading_shape = ray_origins.shape[:-1]

    hit_mask_out = torch.empty(
        leading_shape, device=ray_origins.device, dtype=torch.int32
    )
    hit_distance_out = torch.empty(
        leading_shape, device=ray_origins.device, dtype=ray_origins.dtype
    )
    hit_point_out = torch.empty(
        ray_origins.shape, device=ray_origins.device, dtype=ray_origins.dtype
    )

    return hit_mask_out, hit_distance_out, hit_point_out


class MeshRayIntersection(FunctionSpec):
    """Cast rays against a triangular surface mesh and return first-hit info.

    The kernel uses Warp's BVH (``wp.Mesh`` + ``wp.mesh_query_ray``) for
    accelerated execution. Direction vectors must be **unit vectors** —
    the kernel does not internally normalize, matching the convention
    fixed in ``geopt-datagen-round1-plan.md`` §A Convention 3.

    Parameters
    ----------
    mesh_vertices : torch.Tensor
        Vertex positions, shape ``(n_vertices, 3)``.
    mesh_indices : torch.Tensor
        Triangle connectivity. Either flattened ``(3 * n_faces,)`` or
        ``(n_faces, 3)``.
    ray_origins : torch.Tensor
        Ray origins with shape ``(..., 3)``.
    ray_directions : torch.Tensor
        Unit-vector ray directions with the same shape as ``ray_origins``.
    max_dist : float, optional
        Maximum traversal distance. Default is ``1e8``.
    implementation : str, optional
        Explicit implementation name. Defaults to the registered Warp
        implementation.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(hit_mask, hit_distance, hit_point)``:

        - ``hit_mask`` (``int32``, leading shape of the rays): ``1`` on hit,
          ``0`` on miss.
        - ``hit_distance`` (caller dtype): parametric ``t`` on hit, ``inf``
          on miss.
        - ``hit_point`` (caller dtype, shape ``(..., 3)``):
          ``origin + t*direction`` on hit, ``origin`` on miss.

    Examples
    --------
    >>> mesh_vertices = torch.tensor(
    ...     [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    ... )
    >>> mesh_indices = torch.tensor([(0, 1, 2)])
    >>> ray_origins = torch.tensor([(0.25, 0.25, 1.0)])
    >>> ray_directions = torch.tensor([(0.0, 0.0, -1.0)])
    >>> hit_mask, hit_distance, hit_point = mesh_ray_intersection(
    ...     mesh_vertices, mesh_indices, ray_origins, ray_directions
    ... )
    """

    _BENCHMARK_CASES = (
        ("small", 8, 20, 4096),
        ("medium", 16, 40, 16384),
        ("large", 32, 80, 65536),
    )

    @FunctionSpec.register(
        name="warp", required_imports=("warp>=0.6.0",), rank=0, baseline=True
    )
    def warp_forward(
        mesh_vertices: Float[torch.Tensor, "num_vertices 3"],
        mesh_indices: torch.Tensor,
        ray_origins: Float[torch.Tensor, "... 3"],
        ray_directions: Float[torch.Tensor, "... 3"],
        max_dist: float = 1e8,
    ) -> tuple[
        torch.Tensor,
        Float[torch.Tensor, "..."],
        Float[torch.Tensor, "... 3"],
    ]:
        return mesh_ray_intersection_impl(
            mesh_vertices,
            mesh_indices,
            ray_origins,
            ray_directions,
            max_dist=max_dist,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        device = torch.device(device)
        for label, n_rings, n_segments, num_points in cls._BENCHMARK_CASES:
            # Build UV-sphere vertex positions (mirrors SignedDistanceField).
            phi = torch.linspace(0, torch.pi, n_rings + 2, device=device)[1:-1]
            theta = torch.linspace(0, 2 * torch.pi, n_segments + 1, device=device)[:-1]
            phi_g, theta_g = torch.meshgrid(phi, theta, indexing="ij")

            sin_phi = phi_g.sin()
            ring_points = torch.stack(
                [sin_phi * theta_g.cos(), sin_phi * theta_g.sin(), phi_g.cos()],
                dim=-1,
            ).reshape(-1, 3)

            mesh_vertices = torch.cat(
                [
                    torch.tensor([[0.0, 0.0, 1.0]], device=device),
                    ring_points,
                    torch.tensor([[0.0, 0.0, -1.0]], device=device),
                ]
            ).to(torch.float32)

            # Build UV-sphere triangle connectivity (vectorized).
            south_idx = n_rings * n_segments + 1
            j = torch.arange(n_segments, device=device)
            j_next = (j + 1) % n_segments

            north_fan = torch.stack([torch.zeros_like(j), 1 + j, 1 + j_next], dim=1)

            r = torch.arange(n_rings - 1, device=device).unsqueeze(1)
            base = 1 + r * n_segments
            p00, p01 = base + j, base + j_next
            p10, p11 = base + n_segments + j, base + n_segments + j_next
            body_tris = torch.stack(
                [
                    torch.stack([p00, p10, p11], dim=-1),
                    torch.stack([p00, p11, p01], dim=-1),
                ],
                dim=2,
            ).reshape(-1, 3)

            last = south_idx - n_segments
            south_fan = torch.stack(
                [last + j, torch.full_like(j, south_idx), last + j_next], dim=1
            )

            mesh_indices = (
                torch.cat([north_fan, body_tris, south_fan]).to(torch.int32).reshape(-1)
            )

            # Sample ray origins outside the unit-sphere bbox; aim back at
            # the origin so most rays hit the sphere.
            ray_origins = 3.0 * torch.rand(num_points, 3, device=device) - 1.5
            # Reject the (numerically) zero-origin if it ever slips in:
            origin_norm = torch.linalg.norm(
                ray_origins, dim=-1, keepdim=True
            ).clamp_min(1e-12)
            ray_directions = (-ray_origins / origin_norm).to(torch.float32)
            ray_origins = ray_origins.to(torch.float32)

            yield (
                f"{label}-uv-sphere-tris{2 * n_rings * n_segments}-rays{num_points}",
                (mesh_vertices, mesh_indices, ray_origins, ray_directions),
                {"max_dist": 10.0},
            )


mesh_ray_intersection = MeshRayIntersection.make_function("mesh_ray_intersection")


__all__ = ["MeshRayIntersection", "mesh_ray_intersection"]
