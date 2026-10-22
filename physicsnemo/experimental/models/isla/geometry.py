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

import itertools

import torch
from jaxtyping import Float, Int

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.geometry._cell_normals import compute_cell_normals
from physicsnemo.mesh.spatial import signed_distance_field
from physicsnemo.nn.functional.geometry.sdf import _closest_point_weights

### (Query, cell) pairs per chunk of the closest-feature search; bounds its
### (chunk, cells) blocks. The search is a custom op, so the loop over query chunks
### does not specialize a dynamic query count under torch.compile.
_SEARCH_PAIRS = 2**20

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
    points: torch.Tensor,
    measure_weights: torch.Tensor | None,
    eps: float = 1e-8,
    second_moments: torch.Tensor | None = None,
) -> torch.Tensor:
    """Measure-weighted RMS pairwise distance of a sampled boundary, shape (B, 1, 1).

    ``L^2 = sum_ij w~_i w~_j |x_i - x_j|^2`` with ``w~ = w / sum(w)`` (uniform
    weights when ``measure_weights`` is ``None``), evaluated in O(N) through the
    variance identity ``L^2 = 2 sum_i w~_i (|x_i - c|^2 + J_i)`` with ``c`` the weighted
    mean and ``J_i`` each point's own second moment about itself, per unit measure
    (``second_moments`` ``(B, N)``, :func:`cell_second_moments`; ``None``: points).
    With the cells' centroids, measures and second moments it is the exact RMS
    distance between two points of the piecewise-flat surface, the same for every
    subdivision of its cells. A normalized two-point statistic: it converges under refinement of any
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
    sq = (points - c).square().sum(-1, keepdim=True)
    if second_moments is not None:
        sq = sq + second_moments.reshape(b, n, 1).to(points.dtype)
    var = (w_n * sq).sum(dim=1, keepdim=True)
    return (2.0 * var).sqrt().clamp_min(eps)


def cell_second_moments(mesh) -> torch.Tensor:
    r"""Each simplex cell's second moment about its centroid per unit measure,
    :math:`\frac{1}{|K|}\int_K |x - g|^2 = \frac{1}{(n+1)(n+2)} \sum_k |v_k - g|^2` for an
    :math:`n`-simplex with vertices :math:`v_k` and centroid :math:`g` (a segment's
    :math:`\ell^2 / 12`), shape ``(n_cells,)``: the parallel-axis term that makes
    :func:`rms_pairwise_distance` exact for the piecewise-flat surface."""
    verts = mesh.points[mesh.cells]  # (C, n + 1, d)
    k = verts.shape[1]
    g = verts.mean(1, keepdim=True)
    return (verts - g).square().sum((1, 2)) / (k * (k + 1))


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


def _canonical_cells(points, cells):
    """Each cell's vertices in the lexicographic order of their positions, ``(F,
    d)``, and the parity of that permutation ``(F,)`` (+1 or -1): a function of the
    positions alone, the same for every numbering of the vertices and every
    rotation of a cell's vertex list, so the closest points computed on it are too;
    a winding flip changes only the parity, which carries the orientation."""
    k = cells.shape[1]
    order = torch.arange(k, device=cells.device).expand_as(cells)
    for j in reversed(range(points.shape[1])):
        key = points[cells.gather(1, order), j]
        order = order.gather(1, torch.argsort(key, dim=1, stable=True))
    inversions = sum((order[:, a] > order[:, b]).long() for a in range(k) for b in range(a + 1, k))
    return cells.gather(1, order), 1 - 2 * (inversions % 2)


def _closest_weights(q, simplex):
    """The barycentric weights ``(P, d)`` of the point of each simplex closest to
    its query (Ericson 2005, Sec. 5.1.5), exact zeros outside the face region; call
    in float64 (the face-region weights are ratios of Gram determinants that cancel
    on thin cells). An edge-region point of a triangle is recomputed on the edge
    alone, so every triangle sharing the edge gives it bitwise."""
    if q.shape[-1] == 2:
        t = _segment_t(q, simplex[:, 0], simplex[:, 1])
        return torch.stack([1 - t, t], -1)
    w = _closest_point_weights(q, simplex)
    edge = (w > 0).sum(-1) == 2
    ends = torch.tensor([[1, 2], [0, 2], [0, 1]], device=q.device)[(w[edge] == 0).to(torch.int8).argmax(-1)]
    s = simplex[edge]
    a, b = s.gather(1, ends[:, :1, None].expand(-1, 1, 3))[:, 0], s.gather(1, ends[:, 1:, None].expand(-1, 1, 3))[:, 0]
    t = _segment_t(q[edge], a, b)
    w[edge] = torch.zeros_like(s[..., 0]).scatter(1, ends, torch.stack([1 - t, t], -1))
    return w


def _feature_normals(weights, cell, corners, face_n, vertex_n, edge_n):
    """The unit pseudo-normal ``(P, d)`` of the feature each closest point lies on
    (``weights`` on the cells ``cell`` with vertices ``corners``), and whether that
    feature is a cell's interior ``(P,)``. A feature whose cells' normals cancel
    (both sides of a zero-thickness sheet) keeps the cell's normal."""
    nonzero = weights > 0
    n_nonzero = nonzero.sum(-1)
    on_cell = n_nonzero == corners.shape[-1]
    n = vertex_n[corners.gather(1, weights.argmax(-1, keepdim=True))[:, 0]]
    if edge_n is not None:
        edge = (~nonzero).to(torch.int8).argmax(-1)  # the corner opposite the edge
        n = torch.where((n_nonzero == 2)[:, None], edge_n[cell, edge], n)
    n = torch.where(on_cell[:, None], face_n[cell], n)
    norm = n.norm(dim=-1, keepdim=True)
    return torch.where(norm > 0, n / norm.clamp_min(torch.finfo(norm.dtype).tiny), face_n[cell]), on_cell


def _min_norm_point(c: torch.Tensor) -> torch.Tensor:
    """The point of the convex hull of the rows of ``c`` ``(K, d)`` (distinct unit
    vectors, ``K >= 2``) closest to the origin. It lies in the relative interior of
    a face spanned by at most ``d`` of them (Caratheodory), where it is that face's
    least-norm affine point with non-negative weights, and every such point lies in
    the hull: so it is the least-norm one over those faces (a pair's midpoint,
    since the vectors have equal norms), unless the origin is in the hull, which
    the optimality condition ``c_k . x >= |x|^2`` detects. ``c`` in a canonical
    order, so the result is a function of the set."""
    k, d = c.shape
    best = None
    for m in range(2, min(k, d) + 1):
        for sub in itertools.combinations(range(k), m):
            p = c[list(sub)]
            if m == 2:
                x = 0.5 * (p[0] + p[1])
            else:
                lam, info = torch.linalg.solve_ex(p @ p.T, torch.ones(m, dtype=c.dtype, device=c.device))
                if info or not bool(torch.isfinite(lam).all() and (lam >= 0).all()):
                    continue
                x = (lam / lam.sum()) @ p
            if best is None or x.norm() < best.norm():
                best = x
    ### Rounding of a d-term dot product of unit vectors, on each side.
    slack = 2 * d * torch.finfo(c.dtype).eps
    return torch.zeros_like(best) if bool((c @ best < best @ best - slack).any()) else best


@torch.library.custom_op("physicsnemo::isla_closest_candidates", mutates_args=())
def _closest_candidates(
    query_points: torch.Tensor,
    points: torch.Tensor,
    cells: torch.Tensor,
    face_normals: torch.Tensor,
    vertex_normals: torch.Tensor,
    edge_normals: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The closest-feature search (no gradient): for every query the cell of its
    closest point ``(Q,)``, whether several features tie for it ``(Q,)``, and at a
    tie the least-norm point of the hull of their directions from the surface
    ``(Q, d)`` and the sign the features' pseudo-normals vote for ``(Q,)``; see
    :func:`exact_boundary_geometry`. ``cells`` in canonical order
    (:func:`_canonical_cells`), the normals in the queries' precision.

    Every candidate distance is computed in float64 on the canonical cell; the
    features tie when their distances are equal within the error bound ``E_k =
    (u + (3 d + 4) u_64)(|x| + max_{v in k} |v|)`` of the distance to cell ``k``,
    ``u`` the unit roundoff of the queries' precision (the rounding of the
    coordinates) and ``u_64`` float64's (the arithmetic): cell ``k`` ties when
    ``D_k - E_k <= min_j (D_j + E_j)``, i.e. when it may be the closest one, and
    its closest point is a local minimum of the distance on the surface: on a
    vertex or an edge, every cell sharing that feature reaches the same point
    (otherwise a neighbouring cell holds a closer point of the same patch, as
    beside the foot of the perpendicular on a cell next to the vertex). In two
    dimensions the search is a brute force over every segment; in three a
    float32 Warp BVH query gives an upper bound on the closest distance, and the
    cells whose bounding boxes lie within it are searched exactly. Both are
    chunked over the queries."""
    nq, dim = query_points.shape
    dev = query_points.device
    q64, simplex = query_points.double(), points.double()[cells]
    face_n, vertex_n = face_normals.double(), vertex_normals.double()
    edge_n = None if edge_normals is None else edge_normals.double()
    coef = torch.finfo(query_points.dtype).eps / 2 + (3 * dim + 4) * torch.finfo(torch.float64).eps / 2
    reach, q_norm = simplex.norm(dim=-1).amax(-1), q64.norm(dim=-1)
    if dim == 3:
        seed = signed_distance_field(Mesh(points=points, cells=cells), query_points)[2]
        w0 = _closest_weights(q64, simplex[seed])
        bound = (q64 - (w0[..., None] * simplex[seed]).sum(1)).norm(dim=-1) + coef * (q_norm + reach[seed])
        lo, hi = simplex.amin(1), simplex.amax(1)
    ### Every vertex's and edge's number of cells, keyed as in the loop below.
    n_vertices = points.shape[0]
    valence = torch.bincount(cells.reshape(-1), minlength=n_vertices)
    if dim == 3:
        pairs = torch.stack([cells[:, [0, 0, 1]], cells[:, [1, 2, 2]]], -1).sort(-1).values
        edge_keys, edge_valence = torch.unique(
            n_vertices + pairs[..., 0] * n_vertices + pairs[..., 1], return_counts=True
        )
    idx = torch.empty(nq, dtype=torch.long, device=dev)
    tie_q, tie_c, tie_v = [], [], []
    chunk = max(1, _SEARCH_PAIRS // cells.shape[0])
    for s in range(0, nq, chunk):
        qc = q64[s : s + chunk]
        if dim == 3:
            gap = ((lo[None] - qc[:, None]).clamp_min(0) + (qc[:, None] - hi[None]).clamp_min(0)).norm(dim=-1)
            keep = gap <= (bound[s : s + chunk, None] + coef * (q_norm[s : s + chunk, None] + reach.max()))
        else:
            keep = torch.ones(qc.shape[0], cells.shape[0], dtype=torch.bool, device=dev)
        li, pk = keep.nonzero(as_tuple=True)
        tri = simplex[pk]
        w = _closest_weights(qc[li], tri)
        diff = qc[li] - (w[..., None] * tri).sum(1)
        dist = diff.norm(dim=-1)
        err = coef * (q_norm[s + li] + reach[pk])
        inf = dist.new_full((qc.shape[0],), torch.inf)
        d_min = inf.scatter_reduce(0, li, dist, "amin")
        upper = inf.scatter_reduce(0, li, dist + err, "amin")
        pair = torch.arange(li.shape[0], device=dev)
        first = torch.full((qc.shape[0],), li.shape[0], dtype=torch.long, device=dev).scatter_reduce(
            0, li, torch.where(dist == d_min[li], pair, li.shape[0]), "amin"
        )
        idx[s : s + chunk] = pk[first]
        ### The feature of each closest point (a vertex, an edge, or -1 for a cell's
        ### interior), and whether all of its cells reach it.
        nonzero = w > 0
        n_nonzero = nonzero.sum(-1)
        corners = cells[pk]
        key = torch.where(n_nonzero == 1, corners.gather(1, w.argmax(-1, keepdim=True))[:, 0], -1)
        count = valence[key.clamp_min(0)]
        if dim == 3:
            ends = corners.masked_fill(~nonzero, n_vertices)
            ends = ends.sort(-1).values  # the edge's two vertices, ascending
            edge = n_vertices + ends[:, 0] * n_vertices + ends[:, 1]
            key = torch.where(n_nonzero == 2, edge, key)
            count = torch.where(
                n_nonzero == 2,
                edge_valence[torch.searchsorted(edge_keys, edge).clamp_max(edge_keys.shape[0] - 1)],
                count,
            )
        reached = torch.ones_like(nonzero[:, 0])
        on_feature = key >= 0
        group = li[on_feature] * (n_vertices + n_vertices**2) + key[on_feature]
        _, inverse, counts = torch.unique(group, return_inverse=True, return_counts=True)
        reached[on_feature] = counts[inverse] == count[on_feature]
        tie = (dist - err <= upper[li]) & reached
        fn, on_cell = _feature_normals(w[tie], pk[tie], cells[pk[tie]], face_n, vertex_n, edge_n)
        dt = diff[tie]
        plane = (dt * fn).sum(-1)
        tie_c.append(
            torch.where(
                on_cell[:, None],
                torch.where((plane >= 0)[:, None], fn, -fn),
                dt / dist[tie].clamp_min(torch.finfo(dist.dtype).tiny)[:, None],
            )
        )
        tie_v.append(torch.where(plane >= 0, 1, -1))
        tie_q.append(s + li[tie])
    tq, tc, tv = torch.cat(tie_q), torch.cat(tie_c), torch.cat(tie_v)
    ### A tie is two or more distinct directions (the cells sharing a vertex or an
    ### edge give its direction bitwise).
    head = torch.full((nq,), tq.shape[0], dtype=torch.long, device=dev).scatter_reduce(
        0, tq, torch.arange(tq.shape[0], device=dev), "amin"
    )
    distinct = (tc != tc[head[tq]]).any(-1).long()
    tied = torch.zeros(nq, dtype=torch.long, device=dev).scatter_reduce(0, tq, distinct, "amax").bool()
    tie_u = q64.new_zeros(nq, dim)
    tie_sign = q64.new_ones(nq)
    for i in tied.nonzero()[:, 0].tolist():
        here = tq == i
        tie_u[i] = _min_norm_point(torch.unique(tc[here], dim=0))
        tie_sign[i] = 1.0 if int(tv[here].sum()) >= 0 else -1.0
    return idx, tied, tie_u.to(query_points.dtype), tie_sign.to(query_points.dtype)


@_closest_candidates.register_fake
def _(query_points, points, cells, face_normals, vertex_normals, edge_normals):
    nq = query_points.shape[0]
    return (
        query_points.new_empty(nq, dtype=torch.long),
        query_points.new_empty(nq, dtype=torch.bool),
        query_points.new_empty(query_points.shape),
        query_points.new_empty(nq),
    )


def _closest_features(
    query_points: Float[torch.Tensor, "queries dim"],
    points: Float[torch.Tensor, "vertices dim"],
    cells: Int[torch.Tensor, "cells dim"],
):
    """The closest surface point of each query and its feature: the point
    ``(Q, d)``, the unit pseudo-normal of the feature it lies on ``(Q, d)``,
    whether that feature is the interior of a cell ``(Q,)``, the cell's unit
    normal ``(Q, d)``, and at a closest-feature tie (:func:`_closest_candidates`)
    whether there is one ``(Q,)``, the direction ``(Q, d)`` and the sign ``(Q,)``;
    see :func:`exact_boundary_geometry`."""
    points = points.to(query_points.dtype)
    cells, parity = _canonical_cells(points, _weld(points)[cells])
    cell_points = points[cells]
    all_n = parity[:, None].to(points.dtype) * compute_cell_normals(cell_points[:, 1:] - cell_points[:, :1])
    vertex_n = _vertex_pseudo_normals(cell_points, cells, all_n, points.shape[0])
    edge_n = _edge_pseudo_normals(cells, all_n, points.shape[0]) if points.shape[-1] == 3 else None
    with torch.no_grad():
        q, y = query_points, points
        if q.dtype in (torch.float16, torch.bfloat16):
            q, y = q.float(), y.float()
        idx, tied, tie_u, tie_sign = _closest_candidates(
            q, y, cells, all_n.to(q.dtype), vertex_n.to(q.dtype), None if edge_n is None else edge_n.to(q.dtype)
        )
        corners = cells[idx]  # (Q, d)
        weights = _closest_weights(query_points.double(), points.double()[corners])
    simplex = points[corners]  # (Q, d, d)
    closest = (weights.to(simplex.dtype)[..., None] * simplex).sum(1)
    feature_n, on_cell = _feature_normals(weights, idx, corners, all_n, vertex_n, edge_n)
    dtype = query_points.dtype
    return closest, feature_n, on_cell, all_n[idx], tied, tie_u.to(dtype), tie_sign.to(dtype)


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

    Where several features tie for the closest point (the medial surface: a
    plate's mid-plane, a corner's bisector), within the error bound of the
    distance computation (:func:`_closest_candidates`), :math:`u` is the point of
    the convex hull of their directions :math:`u_k` closest to the origin, so
    :math:`\operatorname{sign}(d)\, u` is the least-norm element of the Clarke
    generalized gradient of :math:`d`: the midpoint of two directions, zero when
    they cancel (no preferred direction), and :math:`d` takes the sign the
    features' pseudo-normals vote for. It is a function of the tie set, not of
    the order of the vertices, the cells or the search, and a winding flip
    negates :math:`d` and leaves :math:`u` bitwise (every closest point is
    computed on its cell's vertices in the lexicographic order of their
    positions).

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
    the gradient of the envelope theorem (at a tie, :math:`u` carries none).
    """
    if query_points.shape[-1] not in (2, 3):
        raise NotImplementedError(
            f"the exact boundary geometry is implemented in two and three dimensions; got {query_points.shape[-1]}"
        )
    closest, feature_n, on_cell, face_n, tied, tie_u, tie_sign = _closest_features(query_points, points, cells)
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
    u = torch.where(tied[:, None], tie_u, u)
    u = torch.where(on_wall[:, None], feature_n, u)
    d = torch.where(
        on_wall,
        torch.zeros_like(dist),
        torch.where(tied, tie_sign * dist, torch.where(on_cell, plane, sign * dist)),
    )
    return d, u

