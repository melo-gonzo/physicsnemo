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

"""ISLA's geometric statistics of a sampled boundary.

:func:`rms_pairwise_distance` is the model's one length unit;
:func:`exact_boundary_geometry` gives the prediction points' built-in signed
distance and direction to the boundary surface from the exact closest point on
its cells. Both are built from differences of positions only, so they refer to
no frame origin."""

import importlib

import torch
from jaxtyping import Float, Int

from physicsnemo.core.version_check import is_package_available
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.geometry._cell_normals import compute_cell_normals
from physicsnemo.mesh.spatial import signed_distance_field
from physicsnemo.nn.functional.geometry.sdf import _closest_point_weights

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
### Not under torch.compile on CPU (or CUDA without Triton): there the search
### builds one (queries, N) block, whose memory grows as their product (about
### 160 GB at 200k x 200k in float32), so compiling on CPU suits small inputs only.
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
    if torch.compiler.is_compiling():
        ### One block: a loop over query chunks would specialize torch.compile's
        ### dynamic query count (each row's argmin is the same either way).
        return torch.cdist(q, y, compute_mode="donot_use_mm_for_euclid_dist").argmin(-1)
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


def _weld(
    points: Float[torch.Tensor, "vertices dim"],
) -> Int[torch.Tensor, " vertices"]:
    """For every vertex, the index of one representative of the vertices at
    bitwise the same position (a stable lexicographic sort; static shapes), so
    that a surface split into pieces, or stored as a triangle soup, has the
    connectivity of the surface it describes."""
    n = points.shape[0]
    order = torch.arange(n, device=points.device)
    for k in reversed(range(points.shape[1])):
        order = order[torch.argsort(points[order, k], stable=True)]
    s = points[order]
    new = torch.cat([s.new_ones(1, dtype=torch.bool), (s[1:] != s[:-1]).any(-1)])
    first = torch.cummax(
        torch.where(new, torch.arange(n, device=points.device), 0), 0
    ).values
    return torch.zeros_like(order).index_copy(0, order, order[first])


def _vertex_pseudo_normals(cell_points, cells, face_normals, n_points: int):
    """The (unnormalized) pseudo-normal of every vertex: the sum of the unit
    normals of its cells, each weighted by the cell's angle at the vertex in
    three dimensions (Baerentzen and Aanaes, 2005) and equally in two, where a
    polyline vertex's pseudo-normal is the sum of its two segments' normals."""
    d = cell_points.shape[-1]
    if d == 2:
        w = face_normals[:, None, :].expand(-1, 2, -1)
    else:
        tri = cell_points
        e1, e2 = tri.roll(-1, 1) - tri, tri.roll(1, 1) - tri
        angle = torch.atan2(
            torch.linalg.cross(e1, e2, dim=-1).norm(dim=-1), (e1 * e2).sum(-1)
        )
        w = angle[..., None] * face_normals[:, None, :]
    return cell_points.new_zeros(n_points, d).index_add(
        0, cells.reshape(-1), w.reshape(-1, d)
    )


def _edge_pseudo_normals(cells, face_normals, n_points: int):
    """The (unnormalized) pseudo-normal of every triangle's edges, ``(F, 3, 3)``:
    the sum of the unit normals of every triangle that shares the edge
    (Baerentzen and Aanaes, 2005); edge ``k`` of a triangle is the one opposite
    its corner ``k``. Edges are matched by a sort of their vertex pairs."""
    f = cells.shape[0]
    pairs = torch.stack([cells[:, [1, 2, 0]], cells[:, [2, 0, 1]]], -1)  # (F, 3, 2)
    key = (pairs.amin(-1) * n_points + pairs.amax(-1)).reshape(-1)
    order = torch.argsort(key)
    ks = key[order]
    first = torch.searchsorted(ks, ks)
    contrib = face_normals[:, None, :].expand(f, 3, 3).reshape(-1, 3)[order]
    sums = torch.zeros_like(contrib).index_add(0, first, contrib)[first]
    return torch.zeros_like(sums).index_copy(0, order, sums).reshape(f, 3, 3)


def _segment_t(q, a, b):
    """The parameter ``t`` in [0, 1] of the point ``a + t (b - a)`` of each
    segment closest to ``q`` (broadcast); exactly 0 or 1 in a vertex's region."""
    ab = b - a
    len2 = (ab * ab).sum(-1)
    t = ((q - a) * ab).sum(-1) / len2.clamp_min(torch.finfo(len2.dtype).tiny)
    return torch.where(len2 > 0, t, 0).clamp(0, 1)


def _segment_distance2(q, a, b):
    """Squared distance from points ``q`` to segments ``[a, b]`` (broadcast)."""
    r = q - a - _segment_t(q, a, b)[..., None] * (b - a)
    return (r * r).sum(-1)


def _closest_cell(query_points, points, cells) -> torch.Tensor:
    """Index ``(Q,)`` of the cell holding each query's closest surface point (no
    gradient). In three dimensions a Warp BVH query
    (:func:`physicsnemo.mesh.spatial.signed_distance_field`, float32, a torch
    custom op); in two an exact brute force over the segments in the inputs'
    precision (half types in float32), chunked over the queries."""
    if query_points.shape[-1] == 3:
        return signed_distance_field(Mesh(points=points, cells=cells), query_points)[2]
    q, y = query_points, points
    if q.dtype in (torch.float16, torch.bfloat16):
        q, y = q.float(), y.float()
    a, b = y[cells[:, 0]][None], y[cells[:, 1]][None]
    if torch.compiler.is_compiling():
        return _segment_distance2(q[:, None], a, b).argmin(-1)
    return torch.cat(
        [
            _segment_distance2(q[s : s + _TORCH_SEARCH_CHUNK, None], a, b).argmin(-1)
            for s in range(0, q.shape[0], _TORCH_SEARCH_CHUNK)
        ]
    )


def _closest_features(
    query_points: Float[torch.Tensor, "queries dim"],
    points: Float[torch.Tensor, "vertices dim"],
    cells: Int[torch.Tensor, "cells dim"],
):
    """The closest surface point of each query and its feature: the point
    ``(Q, d)``, the unit pseudo-normal of the feature it lies on ``(Q, d)``,
    whether that feature is the interior of a cell ``(Q,)`` and the cell's unit
    normal ``(Q, d)``; see :func:`exact_boundary_geometry`."""
    dim = query_points.shape[-1]
    points = points.to(query_points.dtype)
    cells = _weld(points)[cells]
    with torch.no_grad():
        idx = _closest_cell(query_points, points, cells)
    corners = cells[idx]  # (Q, d)
    simplex = points[corners]  # (Q, d, d)
    face_n = compute_cell_normals(simplex[:, 1:] - simplex[:, :1])
    with torch.no_grad():
        ### The closest point's barycentric weights on its cell, in float64 (the
        ### face-region weights are ratios of Gram determinants that cancel on
        ### thin cells); a vertex or edge region gets exact zeros.
        q64, s64 = query_points.double(), simplex.double()
        if dim == 3:
            weights = _closest_point_weights(q64, s64)
        else:
            t = _segment_t(q64, s64[:, 0], s64[:, 1])
            weights = torch.stack([1 - t, t], -1)
        nonzero = weights > 0
        n_nonzero = nonzero.sum(-1)
    closest = (weights.to(simplex.dtype)[..., None] * simplex).sum(1)
    on_cell = n_nonzero == dim
    ### The pseudo-normal of a vertex or an edge (the closest point's feature
    ### when it is not a cell's interior) from every cell that shares it.
    cell_points = points[cells]
    all_n = compute_cell_normals(cell_points[:, 1:] - cell_points[:, :1])
    vertex = corners.gather(1, weights.argmax(-1, keepdim=True))[:, 0]
    feature_n = _vertex_pseudo_normals(cell_points, cells, all_n, points.shape[0])[
        vertex
    ]
    if dim == 3:
        edge = (~nonzero).to(torch.int8).argmax(-1)  # the corner opposite the edge
        edge_n = _edge_pseudo_normals(cells, all_n, points.shape[0])[idx, edge]
        feature_n = torch.where((n_nonzero == 2)[:, None], edge_n, feature_n)
    feature_n = torch.where(on_cell[:, None], face_n, feature_n)
    norm = feature_n.norm(dim=-1, keepdim=True)
    ### A feature whose cells' normals cancel (both sides of a zero-thickness
    ### sheet) keeps the closest cell's normal.
    feature_n = torch.where(
        norm > 0, feature_n / norm.clamp_min(torch.finfo(norm.dtype).tiny), face_n
    )
    return closest, feature_n, on_cell, face_n


def exact_boundary_geometry(
    query_points: Float[torch.Tensor, "queries dim"],
    points: Float[torch.Tensor, "vertices dim"],
    cells: Int[torch.Tensor, "cells dim"],
    *,
    eps: float = 1e-12,
) -> tuple[Float[torch.Tensor, " queries"], Float[torch.Tensor, "queries dim"]]:
    r"""Signed distance and direction from a boundary surface to each query.

    The surface is the union of the cells ``cells`` (``(F, dim)`` vertex
    indices into ``points``): segments in two dimensions, triangles in three.
    For every query :math:`x` the closest point :math:`p` on the surface is
    found exactly (a Warp BVH in three dimensions, a brute force over the
    segments in two), and

    .. math::

        d = \operatorname{sign}\bigl((x - p) \cdot \hat N\bigr)\,\lvert x - p \rvert,
        \qquad
        u = \begin{cases}
            \operatorname{sign}(d)\, \hat n_f & p \text{ interior to the cell } f, \\
            (x - p) / \lvert x - p \rvert & \text{otherwise,}
        \end{cases}

    the distance to the surface signed by the pseudo-normal :math:`\hat N` of
    the feature :math:`p` lies on (the cell's normal :math:`\hat n_f` on a
    cell's interior; on an edge or a vertex the sum of the normals of every
    cell sharing it, angle-weighted at a vertex in three dimensions; Baerentzen
    and Aanaes, 2005), so **positive on the side the normals point to**, and
    the direction from the surface to the query (``sign(0) = +1``). A query on
    the surface (:math:`\lvert x - p \rvert \le` ``eps``) takes
    :math:`d = 0` and :math:`u = \hat N`. Coincident vertices are welded
    first, so a surface stored as pieces or as a triangle soup has the
    pseudo-normals of the surface it describes.

    :math:`d` and :math:`u` are functions of the surface alone (not of how a
    sample of it was drawn): invariant to splitting a cell, converging as
    :math:`O(h^2)` under refinement of a smooth surface, :math:`d` invariant
    and :math:`u` covariant under every rigid motion and reflection (with the
    cells re-oriented so their normals follow the map), :math:`d` scaling
    with the geometry.

    The search runs without gradient; the closest point is then rebuilt from
    its cell's vertices with the barycentric weights of its Voronoi region
    (computed in float64, Ericson 2005, Sec. 5.1.5), so :math:`d` and
    :math:`u` are differentiable in the query and the vertex positions, with
    the gradient of the envelope theorem. The three-dimensional search is in
    float32; its closest cell differs from the exact one only at a tie at
    float32 resolution.
    """
    if query_points.shape[-1] not in (2, 3):
        raise NotImplementedError(
            f"the exact boundary geometry is implemented in two and three dimensions; got {query_points.shape[-1]}"
        )
    closest, feature_n, on_cell, face_n = _closest_features(query_points, points, cells)
    diff = query_points - closest
    dist = diff.norm(dim=-1)
    plane = (diff * face_n).sum(-1)  # the signed distance on a cell's interior
    sign = torch.where((diff * feature_n).sum(-1) >= 0, 1.0, -1.0).to(dist)
    on_wall = dist <= eps
    u = torch.where(
        on_cell[:, None],
        torch.where((plane >= 0)[:, None], face_n, -face_n),
        diff / dist.clamp_min(eps)[:, None],
    )
    u = torch.where(on_wall[:, None], feature_n, u)
    d = torch.where(
        on_wall, torch.zeros_like(dist), torch.where(on_cell, plane, sign * dist)
    )
    return d, u
