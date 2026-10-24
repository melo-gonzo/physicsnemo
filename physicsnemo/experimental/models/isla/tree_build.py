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

r"""The fast build of ISLA-DTT's boundary forest (:class:`~.tree_read.BoundaryForest`).

The reference build makes one GLOBE
:class:`~physicsnemo.mesh.spatial.cluster_tree.ClusterTree` per normal octant, and each
tree's Morton topology splits its ranges level by level with a host sync per level:
about 180 syncs per case. This build makes the same forest for every octant at once
with one sync. The Morton codes are the reference's expression, evaluated with each
cell's own octant's bounds; two stable sorts give the reference's order (octant, then
code, then cell); one Warp kernel finds every node from the positions (each sorted
position walks down from its tree's root, computing the reference's split at every
node on its way, and emits the nodes whose range starts there), a count pass and a
fill pass; a sort of the nodes by (octant, depth, start) is the reference's node
numbering. The topology is integer arithmetic, so it is the reference's exactly.
Every floating-point value is computed by the reference's own expressions: the boxes by
:func:`~physicsnemo.mesh.spatial.cluster_tree._node_boxes` on the whole forest (an exact
min and max), the prefix sums of the measure and of the weighted points, normals and
rank-1 inputs by the same ``cumsum`` calls on the same per-tree inputs, and the node
means by the same elementwise operations on the joined table. The forest's ``trees``
are real ``ClusterTree`` objects over slices of the table, so the reference walk
(:meth:`~.tree_read.BoundaryForest.point_pairs`) runs on them unchanged.
"""

import logging

import torch
import torch.nn.functional as F

from physicsnemo.core.function_spec import FunctionSpec
from physicsnemo.mesh.spatial._lbvh import LBVHTopology
from physicsnemo.mesh.spatial.bvh import _MORTON_SPREAD_STEPS
from physicsnemo.mesh.spatial.cluster_tree import _MORTON_BALANCE, ClusterTree, _node_boxes

try:
    import warp as wp

    HAS_WARP = True
except ImportError:  # pragma: no cover - warp is a dependency of the geometry
    HAS_WARP = False

logger = logging.getLogger(__name__)

#: Bits of a node key below the depth: the node's first sorted position.
_START_BITS = 40
#: Depth bins of the one readback; a forest with internal nodes at depth
#: ``_DEPTH_BINS - 2`` (whose children's paths would not fit an int64 key) is left to
#: the reference build.
_DEPTH_BINS = 64

_LOGGED: set = set()


def resolve_tree_build(build: str, points: torch.Tensor) -> str:
    """The forest build to run: ``"reference"`` or ``"fast"`` when ``build`` forces
    it; under ``"auto"`` the fast build for CUDA points when Warp imports and the
    dimension is 2 or 3, else the reference. Logged once per (device type, choice)."""
    if build not in ("auto", "reference", "fast"):
        raise ValueError(f"build must be 'auto', 'reference' or 'fast', got {build!r}")
    if build != "auto":
        return build
    ok = points.is_cuda and HAS_WARP and points.shape[-1] in (2, 3)
    chosen = "fast" if ok else "reference"
    key = (points.device.type, chosen)
    if key not in _LOGGED and not torch.compiler.is_compiling():
        _LOGGED.add(key)
        logger.info(
            "ISLA tree_build=auto resolved to %r on %s (cuda=%s, warp=%s, dim=%d)",
            chosen,
            points.device.type,
            points.is_cuda,
            HAS_WARP,
            points.shape[-1],
        )
    return chosen


if HAS_WARP:

    @wp.func
    def _morton_split(codes: wp.array(dtype=wp.int64), s: wp.int64, e: wp.int64, balance: wp.int64):
        """The split of the range ``[s, e)`` of sorted ``codes``, as
        :func:`~physicsnemo.mesh.spatial._lbvh.build_morton_topology` makes it: the
        first position in ``[lo, hi]`` whose code reaches ``codes[hi]`` with the bits
        below the highest bit in which ``codes[lo - 1]`` and ``codes[hi]`` differ
        cleared, or the midpoint when they are equal."""
        size = e - s
        margin = wp.int64(1)
        if balance > wp.int64(0):
            margin = wp.max(size // balance, wp.int64(1))
        lo = s + margin
        hi = e - margin
        last = codes[hi]
        diff = codes[lo - wp.int64(1)] ^ last
        q = s + size // wp.int64(2)
        if diff > wp.int64(0):
            m = diff
            m = m | (m >> wp.int64(1))
            m = m | (m >> wp.int64(2))
            m = m | (m >> wp.int64(4))
            m = m | (m >> wp.int64(8))
            m = m | (m >> wp.int64(16))
            m = m | (m >> wp.int64(32))
            key = last - (last & (m >> wp.int64(1)))
            a = lo
            b = hi
            while a < b:
                mid = a + (b - a) // wp.int64(2)
                if codes[mid] < key:
                    a = mid + wp.int64(1)
                else:
                    b = mid
            q = a
        return q

    @wp.kernel
    def _node_count_kernel(
        codes: wp.array(dtype=wp.int64),
        tree: wp.array(dtype=wp.int64),
        tree_start: wp.array(dtype=wp.int64),
        tree_end: wp.array(dtype=wp.int64),
        leaf_size: wp.int64,
        balance: wp.int64,
        count: wp.array(dtype=wp.int64),
        leaf_depth: wp.array(dtype=wp.int64),
    ):
        """The number of nodes whose range starts at sorted position ``tid`` (a chain
        of left children ending at a leaf) and the depth of that leaf."""
        p = wp.int64(wp.tid())
        t = tree[p]
        s = tree_start[t]
        e = tree_end[t]
        depth = wp.int64(0)
        n = wp.int64(0)
        if s == p:
            n = wp.int64(1)
        while e - s > leaf_size:
            q = _morton_split(codes, s, e, balance)
            if p < q:
                e = q
            else:
                s = q
            depth = depth + wp.int64(1)
            if s == p:
                n = n + wp.int64(1)
        count[p] = n
        leaf_depth[p] = depth

    @wp.kernel
    def _node_fill_kernel(
        codes: wp.array(dtype=wp.int64),
        tree: wp.array(dtype=wp.int64),
        tree_start: wp.array(dtype=wp.int64),
        tree_end: wp.array(dtype=wp.int64),
        leaf_size: wp.int64,
        balance: wp.int64,
        offset: wp.array(dtype=wp.int64),
        node_depth: wp.array(dtype=wp.int64),
        node_count: wp.array(dtype=wp.int64),
        node_path: wp.array(dtype=wp.int64),
    ):
        """The nodes whose range starts at sorted position ``tid``, from ``offset[tid]``
        on: depth, item count and path (bit ``k`` set when the node lies in the right
        child of its ancestor at depth ``k``)."""
        p = wp.int64(wp.tid())
        t = tree[p]
        s = tree_start[t]
        e = tree_end[t]
        depth = wp.int64(0)
        path = wp.int64(0)
        j = offset[p]
        if s == p:
            node_depth[j] = depth
            node_count[j] = e - s
            node_path[j] = path
            j = j + wp.int64(1)
        while e - s > leaf_size:
            q = _morton_split(codes, s, e, balance)
            if p < q:
                e = q
            else:
                s = q
                path = path | (wp.int64(1) << depth)
            depth = depth + wp.int64(1)
            if s == p:
                node_depth[j] = depth
                node_count[j] = e - s
                node_path[j] = path
                j = j + wp.int64(1)


def _launch(kernel, n: int, inputs: list, ref: torch.Tensor) -> None:
    device, stream = FunctionSpec.warp_launch_context(ref)
    with FunctionSpec.warp_stream_scope(stream):
        wp.init()
        arrays = [wp.from_torch(x, dtype=wp.int64) if isinstance(x, torch.Tensor) else x for x in inputs]
        wp.launch(kernel, dim=n, inputs=arrays, device=device, stream=stream)


def _morton_codes(points_f: torch.Tensor, octant: torch.Tensor, n_trees: int) -> torch.Tensor:
    """Each cell's code in its own octant's tree: the isotropic quantization of
    :func:`~physicsnemo.mesh.spatial.bvh._compute_morton_codes` with the octant's
    bounds (exact minima and maxima), then the same bit spreading."""
    n, d = points_f.shape
    n_bits = min(62, 63 // d)
    max_val = (1 << n_bits) - 1
    q = points_f.to(torch.float64 if points_f.dtype == torch.float64 else torch.float32)
    idx = octant[:, None].expand(-1, d)
    cmin = q.new_full((n_trees, d), float("inf")).scatter_reduce(0, idx, q, "amin")
    cmax = q.new_full((n_trees, d), float("-inf")).scatter_reduce(0, idx, q, "amax")
    extent = (cmax - cmin).amax(dim=1)
    safe = torch.where(extent != 0, extent, torch.ones_like(extent))
    coords = ((q - cmin[octant]) / safe[octant, None] * max_val).long().clamp(0, max_val)
    for shift, mask in _MORTON_SPREAD_STEPS[d]:
        coords = (coords | (coords << shift)) & mask
    return (coords << torch.arange(d, device=q.device)).sum(dim=1)


def forest_tables(
    points: torch.Tensor,
    measure: torch.Tensor,
    normals: torch.Tensor,
    vecs: torch.Tensor | None,
    points_f: torch.Tensor,
    octant: torch.Tensor,
    *,
    leaf_size: int,
) -> dict | None:
    """The fields of :class:`~.tree_read.BoundaryForest` that its trees make, with the
    nodes' raw mean normal as ``mean_n``, for the cells at lab ``points`` ``(N, d)``
    (``points_f`` in the boundary's frame, ``octant`` their normals' octants), or
    ``None`` where the reference build must make them (no cells, or a tree too deep
    for a node path)."""
    n, d = points_f.shape
    if n == 0 or leaf_size < 1:
        return None
    device = points.device
    n_oct = 2**d
    ### Cells by octant then code then index: the trees' sorted orders, end to end.
    codes = _morton_codes(points_f, octant, n_oct)
    by_code = torch.argsort(codes, stable=True)
    cells = by_code[torch.argsort(octant[by_code], stable=True)]
    sorted_codes = codes[cells].contiguous()
    pos_tree = octant[cells].contiguous()
    cells_per = torch.bincount(octant, minlength=n_oct)
    tree_end = torch.cumsum(cells_per, 0)
    tree_start = tree_end - cells_per

    ### Count pass: the nodes starting at each position; then one readback.
    count = torch.empty(n, dtype=torch.long, device=device)
    leaf_depth = torch.empty(n, dtype=torch.long, device=device)
    fixed = [sorted_codes, pos_tree, tree_start, tree_end, int(leaf_size), int(_MORTON_BALANCE)]
    _launch(_node_count_kernel, n, [*fixed, count, leaf_depth], points)
    nodes_per = torch.zeros(n_oct, dtype=torch.long, device=device).index_add_(0, pos_tree, count)
    depth_per = torch.zeros(n_oct, dtype=torch.long, device=device).scatter_reduce_(
        0, pos_tree, leaf_depth, "amax"
    )
    ### The internal nodes starting at position p have the depths leaf_depth - count + 1
    ### to leaf_depth - 1: counted per depth by a difference array.
    starts_here = (count > 0).long()
    bins = torch.zeros(_DEPTH_BINS, dtype=torch.long, device=device)
    bins.index_add_(0, (leaf_depth - count + 1).clamp_max(_DEPTH_BINS - 1), starts_here)
    bins.index_add_(0, leaf_depth.clamp_max(_DEPTH_BINS - 1), -starts_here)
    stats = torch.cat([cells_per, nodes_per, depth_per, torch.cumsum(bins, 0)]).tolist()
    cells_per_h = stats[:n_oct]
    nodes_per_h = stats[n_oct : 2 * n_oct]
    depth_per_h = stats[2 * n_oct : 3 * n_oct]
    internal_per_depth = stats[3 * n_oct :]
    if internal_per_depth[_DEPTH_BINS - 2]:
        return None
    m_total = sum(nodes_per_h)
    n_internal = sum(internal_per_depth)
    n_levels = sum(1 for c in internal_per_depth if c)

    ### Fill pass, then the nodes in the reference's numbering: tree by tree, each
    ### tree level by level from its root, a level by first position.
    offset = torch.cumsum(count, 0) - count
    depth = torch.empty(m_total, dtype=torch.long, device=device)
    size = torch.empty(m_total, dtype=torch.long, device=device)
    path = torch.empty(m_total, dtype=torch.long, device=device)
    _launch(_node_fill_kernel, n, [*fixed, offset, depth, size, path], points)
    start = torch.repeat_interleave(torch.arange(n, device=device), count, output_size=m_total)
    key = ((pos_tree[start] << 6 | depth) << _START_BITS) | start
    key, order = torch.sort(key)
    range_start, range_count = start[order], size[order]
    depth, path = depth[order], path[order]
    tree = pos_tree[range_start]
    is_leaf = range_count <= leaf_size
    ### A node's left child is the node one level deeper with its first position;
    ### the right child follows it.
    child = torch.searchsorted(key, key + (1 << _START_BITS))
    left = torch.where(is_leaf, -1, child)
    right = torch.where(is_leaf, -1, child + 1)
    leaf_ids = torch.nonzero_static(is_leaf, size=m_total - n_internal)[:, 0]
    leaf_ids = leaf_ids[torch.argsort(range_start[leaf_ids])]
    position_leaf = torch.repeat_interleave(
        torch.arange(leaf_ids.shape[0], device=device), range_count[leaf_ids], output_size=n
    )
    ### Internal nodes by depth from the roots, as the reference's frontier lists them:
    ### at each depth by the child bits from the deepest up, then by tree.
    internal = torch.nonzero_static(~is_leaf, size=n_internal)[:, 0]
    internal = internal[torch.argsort(path[internal], stable=True)]
    internal = internal[torch.argsort(depth[internal], stable=True, descending=True)]
    by_depth = list(torch.split(internal, [c for c in reversed(internal_per_depth) if c]))[::-1]
    levels = [(nodes, left[nodes], right[nodes]) for nodes in reversed(by_depth)]

    ### Boxes in the frame, from the sorted points: exact minima and maxima.
    topo = LBVHTopology(
        left_child=left,
        right_child=right,
        leaf_start=None,
        leaf_count=None,
        range_start=range_start,
        range_count=range_count,
        node_count=m_total,
        max_nodes=m_total,
        internal_nodes_per_level=by_depth,
        leaf_node_ids=leaf_ids,
        leaf_starts=range_start[leaf_ids],
        leaf_sizes=range_count[leaf_ids],
        max_depth=n_levels,
    )
    sorted_f = points_f[cells]
    aabb_min, aabb_max = _node_boxes(topo, sorted_f, leaf_size)

    ### Range sums: per tree, the reference's prefix sums of the sorted measure and of
    ### the weighted points, normals and rank-1 inputs (same inputs, same scans); the
    ### node sums and means on the joined table.
    sorted_w = measure[cells]
    parts = [points[cells], normals[cells]]
    if vecs is not None:
        parts.append(vecs[cells].flatten(1))
    weighted = (torch.cat(parts, dim=1) * sorted_w.unsqueeze(-1)).double()
    trees_h = [k for k in range(n_oct) if cells_per_h[k]]
    c_bounds, m_bounds, a = [0], [0], 0
    for k in trees_h:
        a += cells_per_h[k]
        c_bounds.append(a)
        m_bounds.append(m_bounds[-1] + nodes_per_h[k])
    area_prefix, prefix = [], []
    for t in range(len(trees_h)):
        a, b = c_bounds[t], c_bounds[t + 1]
        area_prefix.append(F.pad(torch.cumsum(sorted_w[a:b].double(), dim=0), (1, 0)))
        prefix.append(F.pad(torch.cumsum(weighted[a:b].t(), dim=1), (1, 0)))
    area_prefix = torch.cat(area_prefix)
    prefix = torch.cat(prefix, dim=1)
    ordinal = torch.cumsum(cells_per > 0, 0) - 1
    starts = range_start + ordinal[tree]
    ends = starts + range_count
    total_64 = area_prefix[ends] - area_prefix[starts]
    nonzero_total = total_64 != 0
    safe_64 = total_64.where(nonzero_total, 1)
    means = ((prefix[:, ends] - prefix[:, starts]) / safe_64).where(nonzero_total, 0).t()
    data_dtype = normals.dtype
    center = means[:, :d].to(points.dtype).contiguous()
    mean_n = means[:, d : 2 * d].to(data_dtype).contiguous()
    node_vecs = None
    if vecs is not None:
        node_vecs = means[:, 2 * d :].to(data_dtype).reshape(m_total, *vecs.shape[1:]).contiguous()
    node_measure = total_64.to(points_f.dtype)

    ### One ClusterTree per octant over its slice of the table (local ids), for the
    ### reference walk.
    by_octant = torch.argsort(octant, stable=True)
    local = torch.empty_like(cells)
    local[by_octant] = torch.arange(n, device=device) - tree_start[octant[by_octant]]
    trees, tree_cells, diameter_sq = [], [], []
    for t, k in enumerate(trees_h):
        a, b = c_bounds[t], c_bounds[t + 1]
        ma, mb = m_bounds[t], m_bounds[t + 1]
        sel = by_octant[a:b]
        leaf = is_leaf[ma:mb]
        rs = range_start[ma:mb] - a
        lo, hi = aabb_min[ma:mb], aabb_max[ma:mb]
        diameter_sq.append((hi - lo).pow(2).sum(dim=-1))
        trees.append(
            ClusterTree(
                node_aabb_min=lo,
                node_aabb_max=hi,
                node_diameter_sq=diameter_sq[-1],
                node_left_child=torch.where(leaf, -1, left[ma:mb] - ma),
                node_right_child=torch.where(leaf, -1, right[ma:mb] - ma),
                leaf_start=torch.where(leaf, rs, -1),
                leaf_count=torch.where(leaf, range_count[ma:mb], 0),
                node_range_start=rs,
                node_range_count=range_count[ma:mb],
                node_total_area=node_measure[ma:mb],
                sorted_source_order=local[cells[a:b]],
                source_points=points_f[sel],
                max_depth=torch.full((), depth_per_h[k], dtype=torch.long, device=device),
                root_ids=torch.zeros(1, dtype=torch.long, device=device),
                batch_size=torch.Size([]),
            )
        )
        tree_cells.append(sel)
    return {
        "trees": trees,
        "tree_cells": tree_cells,
        "node_offset": torch.tensor(m_bounds, device=device),
        "cells": cells,
        "range_start": range_start,
        "range_count": range_count,
        "left": left,
        "right": right,
        "leaf_ids": leaf_ids,
        "position_leaf": position_leaf,
        "levels": levels,
        "aabb_min": aabb_min,
        "aabb_max": aabb_max,
        "diameter_sq": torch.cat(diameter_sq),
        "center": center,
        "mean_n": mean_n,
        "vecs": node_vecs,
        "measure": node_measure,
    }
