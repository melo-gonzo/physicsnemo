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

r"""The forest walk of :meth:`~physicsnemo.experimental.models.isla.tree_read.BoundaryForest.point_pairs`
in two Warp kernels (``walk="fast"``), with the reference's pairs in the reference's
order.

The reference walks each tree level by level for every point at once
(:meth:`~physicsnemo.mesh.spatial.cluster_tree.ClusterTree.find_point_interaction_pairs`):
at each level the opened nodes hand on their left children, then their right
children, so a point's entries at one depth are in the lexicographic order of their
paths read from the deepest step up, and after the sort by point each point holds,
tree by tree, its near cells (leaves by depth and that order, then the leaf's cells
in order) and then its far nodes (by depth and that order). Here one thread walks
one tree for one point. A depth-first count pass sizes each thread's segment of the
output; a breadth-first fill pass writes the entries in that order, each level a
buffer of the opened nodes' (left, right) pairs read even entries first, so no sort
follows. The opening test is the reference's arithmetic, operation by operation in
the boxes' dtype, in a module compiled without contracted multiply-adds, so every
decision is the reference's. The forest is built and walked outside autocast, in its
inputs' precision (:meth:`~.tree_read.BoundaryForest.build`), float32 or float64; a
forest of reduced-precision boxes is left to the reference walk (:func:`walk_fits`).
"""

from typing import Any

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec

### Each product and sum of the opening test is rounded on its own, as the
### reference's separate torch kernels round it.
wp.set_module_options({"fuse_fp": False, "enable_backward": False})

#: Entries of the count pass's depth-first stack; a forest deeper than this is walked
#: by the reference (:func:`walk_fits`).
WALK_STACK = 96
_walk_stack = wp.types.vector(length=WALK_STACK, dtype=wp.int32)
#: Warp refuses an array of this many elements or more, and its thread ids are int32
#: (:func:`walk_fits`, :func:`~.tree_build.forest_tables`).
WARP_MAX_ELEMENTS = 2**31


@wp.func
def _gap_sq(
    pts: wp.array(dtype=Any),
    q: int,
    lo: wp.array(dtype=Any),
    hi: wp.array(dtype=Any),
    m: int,
    d: int,
    k: int,
    zero: Any,
):
    """``clamp(maximum(p - max, min - p), 0) ** 2`` on axis ``k``, NaN where torch's
    maximum and clamp would propagate one."""
    p = pts[q * d + k]
    a = p - hi[m * d + k]
    b = lo[m * d + k] - p
    g = wp.max(wp.max(a, b), zero)
    sq = g * g
    if wp.isnan(a) or wp.isnan(b):
        sq = type(zero)(wp.nan)
    return sq


@wp.func
def _is_far(
    pts: wp.array(dtype=Any),
    q: int,
    lo: wp.array(dtype=Any),
    hi: wp.array(dtype=Any),
    diam_sq: wp.array(dtype=Any),
    coherent: wp.array(dtype=wp.bool),
    m: int,
    d: int,
    theta_sq: Any,
):
    """The reference's admission of node ``m`` for point ``q``: the squared gap from the
    point to the box summed over the axes in torch's order, times ``theta_sq``, above
    the squared diagonal, and the node coherent; a NaN gap is never far."""
    zero = type(theta_sq)(0.0)
    s0 = _gap_sq(pts, q, lo, hi, m, d, 0, zero)
    s1 = _gap_sq(pts, q, lo, hi, m, d, 1, zero)
    s2 = zero
    if d == 3:
        s2 = _gap_sq(pts, q, lo, hi, m, d, 2, zero)
    dist_sq = (s0 + s1) + s2
    far = bool(False)
    if dist_sq * theta_sq > diam_sq[m]:
        far = coherent[m]
    return far


@wp.kernel
def _count_kernel(
    pts: wp.array(dtype=Any),
    lo: wp.array(dtype=Any),
    hi: wp.array(dtype=Any),
    diam_sq: wp.array(dtype=Any),
    coherent: wp.array(dtype=wp.bool),
    left: wp.array(dtype=wp.int64),
    right: wp.array(dtype=wp.int64),
    count: wp.array(dtype=wp.int64),
    roots: wp.array(dtype=wp.int64),
    theta_sq: Any,
    d: int,
    n_q: int,
    n_t: int,
    n_near: wp.array(dtype=wp.int64),
    n_far: wp.array(dtype=wp.int64),
    n_visit: wp.array(dtype=wp.int64),
):
    """Thread ``t * n_q + q``: point ``q``'s near cells, far nodes and visited nodes in
    tree ``t``, written at ``q * n_t + t``."""
    tid = wp.tid()
    t = tid // n_q
    q = tid - t * n_q
    stack = _walk_stack()
    stack[0] = wp.int32(roots[t])
    top = int(1)
    near = wp.int64(0)
    far = wp.int64(0)
    visit = wp.int64(0)
    while top > 0:
        top -= 1
        m = int(stack[top])
        visit += wp.int64(1)
        if _is_far(pts, q, lo, hi, diam_sq, coherent, m, d, theta_sq):
            far += wp.int64(1)
        elif left[m] < wp.int64(0):
            near += count[m]
        else:
            stack[top] = wp.int32(right[m])
            stack[top + 1] = wp.int32(left[m])
            top += 2
    i = q * n_t + t
    n_near[i] = near
    n_far[i] = far
    n_visit[i] = visit


@wp.kernel
def _fill_kernel(
    pts: wp.array(dtype=Any),
    lo: wp.array(dtype=Any),
    hi: wp.array(dtype=Any),
    diam_sq: wp.array(dtype=Any),
    coherent: wp.array(dtype=wp.bool),
    left: wp.array(dtype=wp.int64),
    right: wp.array(dtype=wp.int64),
    start: wp.array(dtype=wp.int64),
    count: wp.array(dtype=wp.int64),
    cells: wp.array(dtype=wp.int64),
    roots: wp.array(dtype=wp.int64),
    theta_sq: Any,
    d: int,
    n_q: int,
    n_t: int,
    n_cells: wp.int64,
    seg_start: wp.array(dtype=wp.int64),
    n_near: wp.array(dtype=wp.int64),
    visit_start: wp.array(dtype=wp.int64),
    n_visit: wp.array(dtype=wp.int64),
    scratch: wp.array(dtype=wp.int32),
    qi: wp.array(dtype=wp.int64),
    si: wp.array(dtype=wp.int64),
):
    """Thread ``t * n_q + q``: point ``q``'s entries in tree ``t``, level by level, near
    cells from ``seg_start`` and far nodes after them. A level's buffer holds the
    previous level's opened nodes as (left, right) pairs in its order, so its even
    entries then its odd ones are the reference's order at that depth. The levels
    alternate between the front of the thread's scratch (entry ``j`` at ``base + j``)
    and its back (at ``last - j``); two consecutive levels are disjoint sets of the
    thread's visited nodes, so they fit together."""
    tid = wp.tid()
    t = tid // n_q
    q = tid - t * n_q
    i = q * n_t + t
    near_at = seg_start[i]
    far_at = near_at + n_near[i]
    base = visit_start[i]
    last = base + n_visit[i] - wp.int64(1)
    scratch[base] = wp.int32(roots[t])
    width = int(1)
    front = bool(True)
    q64 = wp.int64(q)
    while width > 0:
        opened = int(0)
        for parity in range(2):
            j = int(parity)
            while j < width:
                m = int(0)
                if front:
                    m = int(scratch[base + wp.int64(j)])
                else:
                    m = int(scratch[last - wp.int64(j)])
                if _is_far(pts, q, lo, hi, diam_sq, coherent, m, d, theta_sq):
                    qi[far_at] = q64
                    si[far_at] = n_cells + wp.int64(m)
                    far_at += wp.int64(1)
                elif left[m] < wp.int64(0):
                    s = start[m]
                    for c in range(int(count[m])):
                        qi[near_at] = q64
                        si[near_at] = cells[s + wp.int64(c)]
                        near_at += wp.int64(1)
                else:
                    if front:
                        scratch[last - wp.int64(opened)] = wp.int32(left[m])
                        scratch[last - wp.int64(opened + 1)] = wp.int32(right[m])
                    else:
                        scratch[base + wp.int64(opened)] = wp.int32(left[m])
                        scratch[base + wp.int64(opened + 1)] = wp.int32(right[m])
                    opened += 2
                j += 2
        width = opened
        front = not front


def walk_fits(forest) -> bool:
    """The forest is two- or three-dimensional, its depth fits the count pass's stack,
    its boxes are float32 or float64 (the precisions the model builds a forest in; the
    reference walk takes any other) and their entries (the node tables' largest Warp
    arrays, up to 2 N d for N cells) stay below Warp's ``WARP_MAX_ELEMENTS`` (2**31).
    The walk's other arrays (the points' coordinates, the per-thread counts, the pairs
    and the scratch) must also each stay below it; the pairs' and the scratch's sizes
    are known only after the count pass, so :func:`walk_pairs` checks them and returns
    ``None``."""
    fields = (forest.aabb_min, forest.aabb_max, forest.diameter_sq)
    return (
        forest.aabb_min.shape[1] in (2, 3)
        and len(forest.levels) + 2 <= WALK_STACK
        and all(x.dtype in (torch.float32, torch.float64) for x in fields)
        and forest.aabb_min.numel() < WARP_MAX_ELEMENTS
    )


def walk_pairs(
    forest, points_f: torch.Tensor, theta: float, count_only: bool = False
) -> tuple[torch.Tensor, torch.Tensor, int] | torch.Tensor | None:
    """Every point's list ``(qi, si, n_node)`` for the points ``points_f`` ``(Q, d)`` in
    the forest's frame at ``theta > 0``, equal to the reference's
    (:meth:`BoundaryForest.point_pairs`) entry for entry; one host read, of the sizes.
    ``None`` when an array would pass Warp's limit (:func:`walk_fits`). With
    ``count_only`` the count pass alone: each point's list length ``(Q,)``, no host read."""
    device = forest.aabb_min.device
    n_q, d = points_f.shape
    n_t = int(forest.node_offset.shape[0]) - 1
    if n_q == 0 or n_t == 0:
        if count_only:
            return torch.zeros(n_q, dtype=torch.long, device=device)
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty.clone(), 0
    if n_q * max(d, n_t) >= WARP_MAX_ELEMENTS:
        return None
    work = forest.aabb_min.dtype
    pts = points_f.detach().to(work)
    scalar = wp.float64 if work == torch.float64 else wp.float32
    wtype = wp.dtype_from_torch(work)
    as_long = lambda x: wp.from_torch(x.contiguous(), dtype=wp.int64)  # noqa: E731
    as_real = lambda x: wp.from_torch(x.to(work).contiguous().reshape(-1), dtype=wtype)  # noqa: E731
    tree = [
        as_real(pts),
        as_real(forest.aabb_min),
        as_real(forest.aabb_max),
        as_real(forest.diameter_sq),
        wp.from_torch(forest.coherent.contiguous(), dtype=wp.bool),
        as_long(forest.left),
        as_long(forest.right),
    ]
    roots = as_long(forest.node_offset[:-1])
    walk = [scalar(theta * theta), d, n_q, n_t]
    n_near, n_far, n_visit = (torch.empty(n_q * n_t, dtype=torch.long, device=device) for _ in range(3))
    wp_device, stream = FunctionSpec.warp_launch_context(pts)
    with FunctionSpec.warp_stream_scope(stream):
        wp.init()
        wp.launch(
            _count_kernel,
            dim=n_q * n_t,
            inputs=[*tree, as_long(forest.range_count), roots, *walk, as_long(n_near), as_long(n_far), as_long(n_visit)],
            device=wp_device,
            stream=stream,
        )
    if count_only:
        return (n_near + n_far).view(n_q, n_t).sum(1)
    seg_end = (n_near + n_far).cumsum(0)
    visit_end = n_visit.cumsum(0)
    n_pairs, n_node, n_scratch = torch.stack([seg_end[-1], n_far.sum(), visit_end[-1]]).tolist()
    if max(n_pairs, n_scratch) >= WARP_MAX_ELEMENTS:
        return None
    qi = torch.empty(n_pairs, dtype=torch.long, device=device)
    si = torch.empty(n_pairs, dtype=torch.long, device=device)
    scratch = torch.empty(n_scratch, dtype=torch.int32, device=device)
    with FunctionSpec.warp_stream_scope(stream):
        wp.launch(
            _fill_kernel,
            dim=n_q * n_t,
            inputs=[
                *tree,
                as_long(forest.range_start),
                as_long(forest.range_count),
                as_long(forest.cells),
                roots,
                *walk,
                wp.int64(forest.n_cells),
                as_long(seg_end - (n_near + n_far)),
                as_long(n_near),
                as_long(visit_end - n_visit),
                as_long(n_visit),
                wp.from_torch(scratch, dtype=wp.int32),
                as_long(qi),
                as_long(si),
            ],
            device=wp_device,
            stream=stream,
        )
    return qi, si, int(n_node)
