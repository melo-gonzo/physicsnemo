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

"""ISLA's geometric statistics of the boundary.

:func:`rms_pairwise_distance` is the model's one length unit, a quadrature over
the boundary tokens; :func:`exact_boundary_geometry` gives the prediction
points' built-in signed distance and direction to the boundary *surface*, from
the exact closest point on its cells. Both are built from differences of
positions only, so they refer to no frame origin."""

import torch
from jaxtyping import Float, Int

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.geometry._cell_normals import compute_cell_normals
from physicsnemo.mesh.spatial import signed_distance_field
from physicsnemo.nn.functional.geometry.sdf import _closest_point_weights

### Queries per chunk of the two-dimensional search; bounds its (chunk, cells)
### blocks. Not under torch.compile, where the search is one (queries, cells)
### block (a loop over query chunks would specialize the dynamic query count).
_TORCH_SEARCH_CHUNK = 4096

#: The softening of the logarithmic length channels (:func:`soft_log_length`), in
#: the frame's length unit.
LENGTH_SOFTENING = 1e-6


def soft_log_length(s: torch.Tensor, eps: float = LENGTH_SOFTENING) -> torch.Tensor:
    r"""The logarithmic channel of a signed length ``s`` in the frame's length unit:
    :math:`\operatorname{asinh}(s / \varepsilon)`.

    It is :math:`\operatorname{sign}(s) \log(2 \lvert s \rvert / \varepsilon)` up to
    :math:`O((\varepsilon / s)^2)` when :math:`\lvert s \rvert \gg \varepsilon`, so
    it resolves the distances to a wall over many decades, as a logarithm does, and
    it is linear, :math:`s / \varepsilon`, near zero: odd, smooth and continuous
    through :math:`s = 0`, with slope at most :math:`1 / \varepsilon` (Lupton, Gunn
    and Szalay 1999, "asinh magnitudes", soften a logarithm the same way at the
    noise level of what it measures). A signed logarithm
    :math:`\operatorname{sign}(s) \log(\lvert s \rvert + \epsilon)` instead jumps
    by :math:`2 \lvert \log \epsilon \rvert` at :math:`s = 0`, so a point on a wall,
    whose distance is zero or a rounding error of either sign, gets a code that
    rounding or a rigid motion can change by tens.

    The softening :math:`\varepsilon = 10^{-6}` (in the frame's length unit, so
    the channel is invariant to a uniform scaling of the geometry) is a numerical
    resolution, not a model parameter. It sits about 16 times above the float32
    resolution of a distance relative to the frame's length (a few units of
    :math:`2^{-24}`), so rounding moves the channel by at most about 0.1, and below
    the finest wall spacing of the reference datasets (AirfRANS's first nodes off
    the airfoil, about :math:`5 \times 10^{-6}` of the frame's length), so the
    near-wall layers keep their logarithmic resolution.
    """
    return torch.asinh(s / eps)


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

