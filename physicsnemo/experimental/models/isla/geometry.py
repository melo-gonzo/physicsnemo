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

"""ISLA's two geometric statistics of a sampled boundary.

:func:`rms_pairwise_distance` is the model's one length unit and
:func:`nearest_boundary_geometry` the prediction points' built-in signed distance and
direction to the boundary. Both are built from pairwise differences only, so they
refer to no frame origin."""

import importlib

import torch
from jaxtyping import Float

from physicsnemo.core.version_check import is_package_available

### The exact nearest-boundary search below forms the squared distance from the
### *difference* x_q - y_i, whose rounding error is set by the coordinates' own
### precision. The library's CUDA search (cuML) expands it as
### |x|^2 + |y|^2 - 2 x.y in float32, whose error scales with |x|^2: measured
### on native AirfRANS cases (2026-10-03) it returned a wrong in-range nearest
### cell for 0.45% of the near-wall queries (0.70% once the case was rigidly
### translated) and, within ~1e-4 of the wall at |x| ~ 1, its untouched
### sentinel (index -1), so the trained model's error moved by 5-7% under a
### translation that the class specifies as a symmetry. No unexpanded L2 is
### exposed through cuML's NearestNeighbors (its "euclidean"/"l2"/"sqeuclidean"
### all map to the expanded forms), hence the kernels here.
HAS_TRITON = is_package_available("triton")
if HAS_TRITON:
    try:
        triton = importlib.import_module("triton")
        tl = importlib.import_module("triton.language")
    except Exception:  # pragma: no cover - a broken triton install is treated as absent
        HAS_TRITON = False

if HAS_TRITON:

    @triton.jit
    def _nearest_index_kernel(
        q_ptr,
        y_ptr,
        idx_ptr,
        Q,
        N,
        DIM: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """One program owns ``BLOCK_Q`` queries and streams the ``N`` boundary
        samples in ``BLOCK_N`` tiles with a running minimum, so no (Q, N) block is
        ever written; ties go to the smallest index, as ``torch.argmin``."""
        offs_q = tl.program_id(0) * BLOCK_Q + tl.arange(0, BLOCK_Q)
        mask_q = offs_q < Q
        best = tl.full((BLOCK_Q,), float("inf"), q_ptr.dtype.element_ty)
        best_i = tl.zeros((BLOCK_Q,), tl.int32)
        for n0 in range(0, N, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N
            d2 = tl.zeros((BLOCK_Q, BLOCK_N), q_ptr.dtype.element_ty)
            for k in tl.static_range(DIM):
                qk = tl.load(q_ptr + offs_q * DIM + k, mask=mask_q, other=0.0)
                ### padded samples sit at infinity
                yk = tl.load(y_ptr + offs_n * DIM + k, mask=mask_n, other=float("inf"))
                diff = qk[:, None] - yk[None, :]
                d2 += diff * diff
            tile_min = tl.min(d2, axis=1)
            tile_arg = tl.argmin(d2, axis=1)
            take = tile_min < best  # strict: the earliest tile keeps a tie
            best = tl.where(take, tile_min, best)
            best_i = tl.where(take, (n0 + tile_arg).to(tl.int32), best_i)
        tl.store(idx_ptr + offs_q, best_i.to(tl.int64), mask=mask_q)


### Queries per chunk of the torch search; bounds its (chunk, N) distance block.
_TORCH_SEARCH_CHUNK = 4096


def _nearest_index(
    query_points: Float[torch.Tensor, "queries dim"],
    points: Float[torch.Tensor, "tokens dim"],
) -> torch.Tensor:
    """Index ``(Q,)`` of the nearest of ``points`` to each query, by the exact
    (difference-form) squared distance in the inputs' precision; ties to the
    smallest index. Triton on CUDA when available, a chunked
    :func:`torch.cdist` argmin otherwise (CPU included); half types are
    searched in float32."""
    q, y = query_points.contiguous(), points.contiguous()
    if q.dtype in (torch.float16, torch.bfloat16):
        q, y = q.float(), y.float()
    nq, dim = q.shape
    if q.is_cuda and HAS_TRITON:
        idx = torch.empty(nq, dtype=torch.int64, device=q.device)
        ### fastest of the tilings measured on a GB300 at 200k x 200k (2026-10-03):
        ### 69 ms against 118 ms for 64 x 128 and 36 ms for cuML's expanded form
        block_q, block_n = 64, 256
        _nearest_index_kernel[(triton.cdiv(nq, block_q),)](
            q,
            y,
            idx,
            nq,
            y.shape[0],
            DIM=dim,
            BLOCK_Q=block_q,
            BLOCK_N=block_n,
            num_warps=4,
        )
        return idx
    return torch.cat(
        [
            torch.cdist(
                q[s : s + _TORCH_SEARCH_CHUNK],
                y,
                compute_mode="donot_use_mm_for_euclid_dist",
            ).argmin(-1)
            for s in range(0, nq, _TORCH_SEARCH_CHUNK)
        ]
    )


def rms_pairwise_distance(
    points: torch.Tensor, measure_weights: torch.Tensor | None, eps: float = 1e-8
) -> torch.Tensor:
    """Measure-weighted RMS pairwise distance of a sampled boundary, shape (B, 1, 1).

    ``L^2 = sum_ij w~_i w~_j |x_i - x_j|^2`` with ``w~ = w / sum(w)`` (uniform
    weights when ``measure_weights`` is ``None``), evaluated in O(N) through the
    variance identity ``L^2 = 2 sum_i w~_i |x_i - c|^2`` with ``c`` the weighted
    mean. A normalized two-point statistic: it converges under refinement of any
    boundary whose normalized measure converges (including fractal boundaries
    whose total measure diverges), is invariant to a uniform rescale of the
    weights, and is covariant under a geometric rescale. The mean ``c`` is an
    intermediate of this scalar only; no centre is exposed.
    """
    b, n = points.shape[0], points.shape[1]
    if measure_weights is None:
        w = points.new_ones(b, n, 1)
    else:
        w = measure_weights.reshape(b, n, 1).to(points.dtype)
    w_n = w / w.sum(dim=1, keepdim=True).clamp_min(eps)
    c = (w_n * points).sum(dim=1, keepdim=True)
    var = (w_n * (points - c).square().sum(-1, keepdim=True)).sum(dim=1, keepdim=True)
    return (2.0 * var).sqrt().clamp_min(eps)


def nearest_boundary_geometry(
    query_points: Float[torch.Tensor, "batch queries dim"],
    points: Float[torch.Tensor, "batch tokens dim"],
    normals: Float[torch.Tensor, "batch tokens dim"],
    *,
    eps: float = 1e-12,
) -> tuple[
    Float[torch.Tensor, "batch queries"], Float[torch.Tensor, "batch queries dim"]
]:
    r"""Signed distance and direction from the nearest boundary sample to each query.

    For every query :math:`x_q` the nearest boundary point :math:`y_{i^*}` is
    found by an exact brute-force search on the difference-form squared
    distance (:func:`_nearest_index`: a Triton streaming-minimum kernel on
    CUDA, a chunked :func:`torch.cdist` argmin otherwise; any spatial
    dimension; ties to the smallest index); then

    .. math::

        u_q = \frac{x_q - y_{i^*}}{\lvert x_q - y_{i^*} \rvert}, \qquad
        d_q = \operatorname{sign}(u_q \cdot \hat n_{i^*})\,\lvert x_q - y_{i^*} \rvert ,

    the direction from the boundary to the query and the distance signed by
    the nearest sample's normal: **positive on the side the normal points to**
    (outside, for outward normals), which is the convention of the stored
    signed-distance fields of the program's datasets. A query that coincides
    with its nearest sample takes that sample's normal as :math:`u_q` and
    :math:`d_q = 0`. Both quantities are functions of pairwise differences,
    so :math:`d_q` is invariant and :math:`u_q` covariant under every rigid
    motion and reflection, and :math:`d_q` scales with the geometry.

    The search runs without gradient (an argmin); :math:`d_q, u_q` are then
    rebuilt from the gathered sample, so they are differentiable in the
    positions. ``normals`` are renormalized.

    Because the boundary is a *sample* (200,000 of up to 17 million cells on
    a DrivAerML car), :math:`\lvert d_q \rvert` overestimates the true wall
    distance by up to about the sample spacing near the wall and :math:`u_q`
    deviates from the true SDF gradient by the same order; far from the wall
    both agree with the full-mesh field.
    """
    b, nq, _ = query_points.shape
    with torch.autocast(query_points.device.type, enabled=False), torch.no_grad():
        ### the search is unbatched; the batch axis is one in practice.
        idx = torch.stack(
            [_nearest_index(query_points[i], points[i]) for i in range(b)]
        )  # (B, Q)
    gather = idx[..., None].expand(b, nq, points.shape[-1])
    y = torch.gather(points, 1, gather)
    n_star = torch.gather(normals, 1, gather)
    n_star = n_star / n_star.norm(dim=-1, keepdim=True).clamp_min(eps)
    diff = query_points - y
    dist = diff.norm(dim=-1, keepdim=True)
    on_wall = dist <= eps
    u = torch.where(on_wall, n_star, diff / dist.clamp_min(eps))
    ### The tangent case (u . n = 0) counts as the positive side, so d never
    ### vanishes off the wall.
    sign = torch.where((u * n_star).sum(-1, keepdim=True) >= 0, 1.0, -1.0).to(dist)
    d = torch.where(on_wall, torch.zeros_like(dist), sign * dist)
    return d.squeeze(-1), u
