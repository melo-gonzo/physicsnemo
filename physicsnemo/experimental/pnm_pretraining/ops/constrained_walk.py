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

"""GeoPT-style constrained-walk composite operator.

Ports ``multi_step_constrained_walk_with_surface`` from
``external-repos/GeoPT/data_generation/GeoPT_PreTraining_Data.py:252-363``
to a fused Warp kernel. Each step does, in a single launch:

1. Closest-point query against the mesh (winding-number sign-aware).
2. Supervision output ``v(p) = closest - p`` (surface-pointing; see
   convention block below).
3. For volume points (``surface_mask == 0``): a ray-cast in
   ``direction`` against the mesh; if it hits within the requested
   ``step_length``, snap the post-step position to ``hit_distance *
   0.99`` along the ray (the GeoPT 0.99 sticking haircut on
   ``GeoPT_PreTraining_Data.py:333``); otherwise advance by the full
   step length.
4. For surface points (``surface_mask == 1``): the position is pinned —
   no motion.

Surface points are also re-pinned by the Python orchestrator to be
bit-exact equal to the original ``surf_pts`` after each step (the
kernel itself does not move surface rows, so this is belt-and-braces;
matches the GeoPT reference at line 344).

Geometry-direction convention (load-bearing)
--------------------------------------------
This module emits the **surface-pointing** vector-distance feature

```
v(p) := closest_point(p) − p
```

per ``geopt-datagen-round1-plan.md`` §A *Convention 1*. This is the
**opposite sign** of the GeoPT reference, which emits ``positions −
closest`` at line 319. Parity tests against GeoPT negate one or the
other before comparing. This convention is non-negotiable across
round-1 code, tests, reports, and the ``.pdmsh`` schema; downstream
consumers that disagree are wrong, not the data-gen pipeline.

Walk-diversity API (improvement I10)
------------------------------------
GeoPT's "100 random walks" is **not** 100 independently-sampled
trajectories; ``GeoPT_PreTraining_Data.py:585-608`` defines
``BASE_WALKS = 10`` and ``PERTURB_SIGMA = 0.05``: walks 0–9 are
sampled fresh, walks 10–99 reuse ``base_directions[j % 10]`` plus a
σ=0.05 Gaussian jitter (renormalized) and reuse the base walk's
``step_lengths`` *verbatim*. We expose this honestly via

```python
generate_walks(
    ...,
    n_independent=10,         # was BASE_WALKS
    n_jittered_per_base=9,    # was (n_random_walks - BASE_WALKS) // BASE_WALKS
    perturb_sigma=0.05,
)
```

Defaults match GeoPT for parity. ``is_independent: (n_walks,) bool``
in the return dict makes the structure trivially auditable.

References
----------
* GeoPT reference:
  ``external-repos/GeoPT/data_generation/GeoPT_PreTraining_Data.py``
  - lines 252-363: ``multi_step_constrained_walk_with_surface``
  - lines 318-319: closest-point + supervise (opposite-sign).
  - line 333: ``actual_step = where(collision, hit_d * 0.99, vol_L)``.
  - line 344: surface re-pin.
  - lines 585-608: walk-orchestration loop.
* PhysicsNeMo SDF kernel:
  ``physicsnemo/nn/functional/geometry/sdf.py:28-72`` — barycentric
  closest-point reconstruction we mirror here.
"""

from __future__ import annotations

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec

wp.config.quiet = True


# ---------------------------------------------------------------------------
# GeoPT walk-orchestration constants — values from the released reference
# at external-repos/GeoPT/data_generation/GeoPT_PreTraining_Data.py:
#   line 333 — sticking factor (clamps actual step to 99% of hit distance).
#   line 585 — BASE_WALKS = 10 independent direction draws per geometry.
#   line 586 — PERTURB_SIGMA = 0.05 σ-jitter applied to the 90 jittered walks.
# The kernel below references the literal 0.99 inline (Warp kernels resolve
# constants at compile time and the kernel-typing path is finicky); the
# Python-side default for the haircut is named here to make the value
# discoverable from a single source. Keep the two in lockstep.
# ---------------------------------------------------------------------------
_STICKING_FACTOR: float = 0.99
_DEFAULT_BASE_WALKS: int = 10
_DEFAULT_PERTURB_SIGMA: float = 0.05


# ---------------------------------------------------------------------------
# Fused Warp kernel — one step of the constrained walk.
# ---------------------------------------------------------------------------


@wp.kernel
def _constrained_walk_step_kernel(
    mesh_id: wp.uint64,
    positions: wp.array(dtype=wp.vec3f),
    directions: wp.array(dtype=wp.vec3f),
    step_lengths: wp.array(dtype=wp.float32),
    surface_mask: wp.array(dtype=wp.int32),
    max_dist: wp.float32,
    positions_out: wp.array(dtype=wp.vec3f),
    supervise_out: wp.array(dtype=wp.vec3f),
):
    """Per-particle fused closest-point + ray-cast + position update.

    Closest point is reconstructed from the barycentric coordinates of
    ``wp.mesh_query_point_sign_winding_number`` using the same
    ``(u, v, 1-u-v)`` idiom as
    ``physicsnemo/nn/functional/geometry/sdf.py:65-72`` (compatible with
    Warp 1.12+ and avoids relying on ``wp.mesh_eval_position``, which
    may be unavailable in some Warp builds).

    Parameters
    ----------
    mesh_id : wp.uint64
        Identifier of the warp ``wp.Mesh`` (built with
        ``support_winding_number=True``).
    positions : wp.array(dtype=wp.vec3f)
        Per-particle position at the start of this step.
    directions : wp.array(dtype=wp.vec3f)
        Per-particle unit direction (caller-normalized; see
        ``geopt-datagen-round1-plan.md`` §A Convention 3).
    step_lengths : wp.array(dtype=wp.float32)
        Per-particle requested step length. Must be 0 for surface rows.
    surface_mask : wp.array(dtype=wp.int32)
        1 for surface (pinned) rows, 0 for volume (mobile) rows.
    max_dist : wp.float32
        Maximum search distance for the closest-point query.
    positions_out : wp.array(dtype=wp.vec3f)
        Output: post-step position.
    supervise_out : wp.array(dtype=wp.vec3f)
        Output: ``closest - position`` (surface-pointing).
    """
    tid = wp.tid()

    p = positions[tid]
    d = directions[tid]
    L = step_lengths[tid]

    # --- closest-point query (always, for both surface and volume). ---
    qp = wp.mesh_query_point_sign_winding_number(mesh_id, p, max_dist)
    mesh = wp.mesh_get(mesh_id)
    p0 = mesh.points[mesh.indices[3 * qp.face + 0]]
    p1 = mesh.points[mesh.indices[3 * qp.face + 1]]
    p2 = mesh.points[mesh.indices[3 * qp.face + 2]]
    closest = qp.u * p0 + qp.v * p1 + (1.0 - qp.u - qp.v) * p2

    # supervise = closest - position (surface-pointing; plan §A Convention 1).
    supervise_out[tid] = closest - p

    # --- position update. Surface rows pin; volume rows ray-cast + advance. ---
    if surface_mask[tid] == 1:
        positions_out[tid] = p
    else:
        qr = wp.mesh_query_ray(mesh_id, p, d, L)
        if qr.result and qr.t < L:
            # GeoPT 0.99 sticking haircut: stop short of the surface.
            # 0.99 matches _STICKING_FACTOR (module constant); inlined here
            # because Warp kernels resolve module-scope constants at
            # compile time and inlining keeps the kernel signature simple.
            positions_out[tid] = p + d * (qr.t * 0.99)
        else:
            # Free flight by the full requested step.
            positions_out[tid] = p + d * L


# ---------------------------------------------------------------------------
# Per-step Python orchestrator. Builds wp.Mesh, launches kernel, returns.
# ---------------------------------------------------------------------------


def constrained_walk_step(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    positions: torch.Tensor,
    directions: torch.Tensor,
    step_lengths: torch.Tensor,
    surface_mask: torch.Tensor,
    *,
    max_dist: float = 1.0e6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one fused step of the GeoPT constrained walk.

    Parameters
    ----------
    mesh_vertices : torch.Tensor
        ``(V, 3)`` vertex positions.
    mesh_indices : torch.Tensor
        Triangle connectivity, either ``(F, 3)`` or flat ``(3F,)``.
    positions : torch.Tensor
        ``(N, 3)`` particle positions at the start of this step.
    directions : torch.Tensor
        ``(N, 3)`` unit-vector directions (caller-normalized).
    step_lengths : torch.Tensor
        ``(N,)`` requested step lengths; must be 0 for surface rows.
    surface_mask : torch.Tensor
        ``(N,)`` ``int32``: 1 for surface (pinned) rows, 0 for volume
        rows.
    max_dist : float, optional
        Maximum search distance for the closest-point query. Defaults
        to ``1.0e6`` (matches the GeoPT FCPW reference's effectively-
        unbounded radius at ``GeoPT_PreTraining_Data.py:94``).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(positions_next, supervise)`` both shape ``(N, 3)``,
        ``float32``.

        ``supervise[i] = closest_point(positions[i]) - positions[i]``
        (surface-pointing; see module-level docstring).
    """
    if positions.shape[-1] != 3:
        raise ValueError("positions must have last dimension of size 3")
    if positions.shape != directions.shape:
        raise ValueError(
            "positions and directions must have identical shapes; "
            f"got {tuple(positions.shape)} vs {tuple(directions.shape)}"
        )
    if step_lengths.shape != positions.shape[:-1]:
        raise ValueError(
            "step_lengths must have shape positions.shape[:-1]; "
            f"got {tuple(step_lengths.shape)} vs {tuple(positions.shape[:-1])}"
        )
    if surface_mask.shape != positions.shape[:-1]:
        raise ValueError(
            "surface_mask must have shape positions.shape[:-1]; "
            f"got {tuple(surface_mask.shape)} vs {tuple(positions.shape[:-1])}"
        )

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

    device = positions.device
    n = positions.shape[0]

    positions_f32 = positions.to(torch.float32).contiguous()
    directions_f32 = directions.to(torch.float32).contiguous()
    step_lengths_f32 = step_lengths.to(torch.float32).contiguous()
    surface_mask_i32 = surface_mask.to(torch.int32).contiguous()

    positions_out = torch.zeros(n, 3, dtype=torch.float32, device=device)
    supervise_out = torch.zeros(n, 3, dtype=torch.float32, device=device)

    wp_launch_device, wp_launch_stream = FunctionSpec.warp_launch_context(positions)

    with wp.ScopedStream(wp_launch_stream):
        wp.init()

        wp_vertices = wp.from_torch(mesh_vertices.to(torch.float32), dtype=wp.vec3)
        wp_indices = wp.from_torch(
            mesh_indices.to(torch.int32).contiguous(), dtype=wp.int32
        )
        wp_positions = wp.from_torch(positions_f32, dtype=wp.vec3)
        wp_directions = wp.from_torch(directions_f32, dtype=wp.vec3)
        wp_step_lengths = wp.from_torch(step_lengths_f32, dtype=wp.float32)
        wp_surface_mask = wp.from_torch(surface_mask_i32, dtype=wp.int32)
        wp_positions_out = wp.from_torch(positions_out, dtype=wp.vec3f)
        wp_supervise_out = wp.from_torch(supervise_out, dtype=wp.vec3f)

        mesh = wp.Mesh(
            points=wp_vertices,
            indices=wp_indices,
            support_winding_number=True,
        )

        wp.launch(
            kernel=_constrained_walk_step_kernel,
            dim=n,
            inputs=[
                mesh.id,
                wp_positions,
                wp_directions,
                wp_step_lengths,
                wp_surface_mask,
                float(max_dist),
                wp_positions_out,
                wp_supervise_out,
            ],
            device=wp_launch_device,
            stream=wp_launch_stream,
        )

    return positions_out, supervise_out


# ---------------------------------------------------------------------------
# Direction sampling — uniform on S² via (φ, cos θ).
# ---------------------------------------------------------------------------


def _sample_uniform_s2(n: int, rng: torch.Generator | None = None) -> torch.Tensor:
    """Return ``(n, 3)`` float32 unit vectors uniform on the 2-sphere.

    Uses the (φ, cos θ) parameterization from the GeoPT reference at
    ``GeoPT_PreTraining_Data.py:302-309``: ``phi ~ U[0, 2π)``,
    ``cos_theta ~ U[-1, 1]``. The resulting Cartesian vectors are
    unit-norm by construction.
    """
    phi = torch.empty(n, 1).uniform_(0.0, 2.0 * torch.pi, generator=rng)
    cos_theta = torch.empty(n, 1).uniform_(-1.0, 1.0, generator=rng)
    sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta * cos_theta, min=0.0))
    out = torch.cat(
        [sin_theta * torch.cos(phi), sin_theta * torch.sin(phi), cos_theta],
        dim=1,
    ).to(torch.float32)
    return out


# ---------------------------------------------------------------------------
# Multi-step orchestration: one walk.
# ---------------------------------------------------------------------------


def constrained_walk(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    vol_pts: torch.Tensor,
    surf_pts: torch.Tensor,
    *,
    n_steps: int = 3,
    directions: torch.Tensor | None = None,
    step_lengths: torch.Tensor | None = None,
    max_step: float = 2.0,
    seed: int | None = None,
) -> dict:
    """Run a single ``n_steps``-step constrained walk over volume + surface points.

    This is a faithful port of GeoPT's
    ``multi_step_constrained_walk_with_surface``: at each of the
    ``n_steps`` recorded positions, a closest-point query produces a
    supervise vector ``v(p) = closest - p`` (surface-pointing). Between
    consecutive recorded positions, volume rows advance by their
    direction × step_length, with the GeoPT 0.99 sticking haircut on
    ray-mesh collisions; surface rows pin.

    Parameters
    ----------
    mesh_vertices : torch.Tensor
        ``(V, 3)`` vertex positions.
    mesh_indices : torch.Tensor
        ``(F, 3)`` or flat triangle connectivity.
    vol_pts : torch.Tensor
        ``(N, 3)`` volume sample positions.
    surf_pts : torch.Tensor
        ``(M, 3)`` surface sample positions.
    n_steps : int, optional
        Number of supervised steps. Default 3.
    directions : torch.Tensor, optional
        ``(N+M, 3)`` pre-sampled unit directions. If ``None``, sampled
        uniform on S² via the (φ, cos θ) parameterization that matches
        the GeoPT reference.
    step_lengths : torch.Tensor, optional
        ``(N+M,)`` pre-sampled step lengths. If ``None``, sampled
        ``U[0, max_step]`` for volume rows; surface rows are forced to
        0.
    max_step : float, optional
        Upper bound for the step-length distribution. Default 2.0
        (matches GeoPT).
    seed : int, optional
        Seed for the torch RNG used to sample directions and step
        lengths. If ``None``, the default global RNG is used.

    Returns
    -------
    dict
        Keys:

        - ``supervise`` (``(N+M, n_steps, 3)`` float32): per-particle
          per-step supervise vectors, surface-pointing.
        - ``positions_final`` (``(N+M, 3)`` float32): final positions
          after the walk. Surface rows equal ``surf_pts`` bit-exact.
        - ``directions`` (``(N+M, 3)`` float32): the directions used.
        - ``step_lengths`` (``(N+M,)`` float32): the step lengths used.
    """
    if vol_pts.ndim != 2 or vol_pts.shape[-1] != 3:
        raise ValueError(f"vol_pts must be (N, 3); got {tuple(vol_pts.shape)}")
    if surf_pts.ndim != 2 or surf_pts.shape[-1] != 3:
        raise ValueError(f"surf_pts must be (M, 3); got {tuple(surf_pts.shape)}")

    device = vol_pts.device
    n_vol = vol_pts.shape[0]
    n_surf = surf_pts.shape[0]
    n_tot = n_vol + n_surf

    rng = None
    if seed is not None:
        rng = torch.Generator(device="cpu")
        rng.manual_seed(int(seed))

    if directions is None:
        directions = _sample_uniform_s2(n_tot, rng=rng).to(device)
    else:
        directions = directions.to(device=device, dtype=torch.float32).contiguous()
        if directions.shape != (n_tot, 3):
            raise ValueError(
                f"directions must have shape ({n_tot}, 3); got {tuple(directions.shape)}"
            )

    if step_lengths is None:
        step_lengths = (
            torch.empty(n_tot).uniform_(0.0, max_step, generator=rng).to(torch.float32)
        )
        step_lengths[n_vol:] = 0.0
        step_lengths = step_lengths.to(device)
    else:
        step_lengths = step_lengths.to(device=device, dtype=torch.float32).contiguous()
        if step_lengths.shape != (n_tot,):
            raise ValueError(
                f"step_lengths must have shape ({n_tot},); got {tuple(step_lengths.shape)}"
            )

    # surface_mask: 0 for volume rows, 1 for surface rows.
    surface_mask = torch.zeros(n_tot, dtype=torch.int32, device=device)
    surface_mask[n_vol:] = 1

    positions = torch.cat(
        [vol_pts.to(torch.float32), surf_pts.to(torch.float32)], dim=0
    ).contiguous()
    surf_pts_anchor = surf_pts.to(torch.float32).contiguous()

    supervise_steps: list[torch.Tensor] = []
    for step_idx in range(n_steps):
        positions_next, supervise = constrained_walk_step(
            mesh_vertices,
            mesh_indices,
            positions,
            directions,
            step_lengths,
            surface_mask,
        )
        supervise_steps.append(supervise)

        # Mirror GeoPT: do not move on the final recorded step (line 321-322).
        if step_idx == n_steps - 1:
            break

        # Advance + re-pin surface rows bit-exact (line 344).
        positions = positions_next
        positions[n_vol:] = surf_pts_anchor

    supervise_stacked = torch.stack(supervise_steps, dim=1)  # (N+M, n_steps, 3)

    return {
        "supervise": supervise_stacked,
        "positions_final": positions,
        "directions": directions,
        "step_lengths": step_lengths,
    }


# ---------------------------------------------------------------------------
# Walk orchestration: GeoPT's 10 base + 90 jittered structure.
# ---------------------------------------------------------------------------


def generate_walks(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    vol_pts: torch.Tensor,
    surf_pts: torch.Tensor,
    *,
    n_independent: int = _DEFAULT_BASE_WALKS,
    n_jittered_per_base: int = 9,
    perturb_sigma: float = _DEFAULT_PERTURB_SIGMA,
    n_steps: int = 3,
    max_step: float = 2.0,
    seed: int | None = None,
) -> dict:
    """Run ``n_independent * (1 + n_jittered_per_base)`` constrained walks.

    Reproduces GeoPT's walk-orchestration layout from
    ``GeoPT_PreTraining_Data.py:585-608`` with explicit, honest knobs
    (improvement I10 in ``geopt-datagen-round1-plan.md`` §8). For each
    of the ``n_independent`` base walks: sample fresh directions and
    step lengths and run a walk. Then emit ``n_jittered_per_base``
    perturbed walks per base — directions are the base directions plus
    a per-particle Gaussian jitter of standard deviation
    ``perturb_sigma``, renormalized; step lengths are reused from the
    base walk **verbatim**. The walks are emitted in interleaved order:
    base[0], jitter[0,0], jitter[0,1], …, jitter[0, n_jittered-1],
    base[1], jitter[1,0], … (so consumers can stream walks per base
    without buffering the full set).

    Defaults (``n_independent=10, n_jittered_per_base=9,
    perturb_sigma=0.05``) match the GeoPT reference. ``is_independent``
    in the return dict makes the structure auditable.

    Parameters
    ----------
    mesh_vertices, mesh_indices, vol_pts, surf_pts, n_steps, max_step
        See :func:`constrained_walk`.
    n_independent : int, optional
        Number of base (freshly-sampled) walks. Default 10.
    n_jittered_per_base : int, optional
        Number of jittered walks emitted per base walk. Default 9.
    perturb_sigma : float, optional
        Gaussian standard deviation of the per-component direction
        jitter (pre-renormalization). Default 0.05.
    seed : int, optional
        Seed for the torch RNG controlling base sampling and jitter.

    Returns
    -------
    dict
        Keys:

        - ``supervise`` (``(n_walks, N+M, n_steps, 3)`` float32):
          stacked per-walk supervise tensors.
        - ``directions`` (``(n_walks, N+M, 3)`` float32): per-walk
          directions (unit-norm).
        - ``step_lengths`` (``(n_walks, N+M)`` float32): per-walk step
          lengths.
        - ``is_independent`` (``(n_walks,)`` bool): True for base
          walks, False for jittered walks.

        where ``n_walks = n_independent * (1 + n_jittered_per_base)``.
    """
    if n_independent <= 0:
        raise ValueError(f"n_independent must be > 0; got {n_independent}")
    if n_jittered_per_base < 0:
        raise ValueError(f"n_jittered_per_base must be >= 0; got {n_jittered_per_base}")

    n_vol = vol_pts.shape[0]
    n_surf = surf_pts.shape[0]
    n_tot = n_vol + n_surf

    rng = None
    if seed is not None:
        rng = torch.Generator(device="cpu")
        rng.manual_seed(int(seed))

    supervise_walks: list[torch.Tensor] = []
    directions_walks: list[torch.Tensor] = []
    step_lengths_walks: list[torch.Tensor] = []
    is_independent: list[bool] = []

    for base_idx in range(n_independent):
        # Sample fresh directions and step lengths for the base walk.
        base_dirs = _sample_uniform_s2(n_tot, rng=rng).to(vol_pts.device)
        base_steps = (
            torch.empty(n_tot).uniform_(0.0, max_step, generator=rng).to(torch.float32)
        )
        base_steps[n_vol:] = 0.0
        base_steps = base_steps.to(vol_pts.device)

        result = constrained_walk(
            mesh_vertices,
            mesh_indices,
            vol_pts,
            surf_pts,
            n_steps=n_steps,
            directions=base_dirs,
            step_lengths=base_steps,
        )
        supervise_walks.append(result["supervise"])
        directions_walks.append(result["directions"])
        step_lengths_walks.append(result["step_lengths"])
        is_independent.append(True)

        # Jittered walks: reuse base step lengths verbatim, jitter
        # directions by perturb_sigma, renormalize. (GeoPT lines
        # 600-603.)
        for _ in range(n_jittered_per_base):
            jitter = (
                torch.empty(n_tot, 3)
                .normal_(mean=0.0, std=float(perturb_sigma), generator=rng)
                .to(vol_pts.device, dtype=torch.float32)
            )
            jittered_dirs = base_dirs + jitter
            norms = torch.linalg.norm(jittered_dirs, dim=-1, keepdim=True).clamp_min(
                1.0e-10
            )
            jittered_dirs = (jittered_dirs / norms).to(torch.float32)

            result = constrained_walk(
                mesh_vertices,
                mesh_indices,
                vol_pts,
                surf_pts,
                n_steps=n_steps,
                directions=jittered_dirs,
                step_lengths=base_steps,  # verbatim reuse
            )
            supervise_walks.append(result["supervise"])
            directions_walks.append(result["directions"])
            step_lengths_walks.append(result["step_lengths"])
            is_independent.append(False)

    return {
        "supervise": torch.stack(supervise_walks, dim=0),
        "directions": torch.stack(directions_walks, dim=0),
        "step_lengths": torch.stack(step_lengths_walks, dim=0),
        "is_independent": torch.tensor(is_independent, dtype=torch.bool),
    }


__all__ = [
    "constrained_walk",
    "constrained_walk_step",
    "generate_walks",
]
