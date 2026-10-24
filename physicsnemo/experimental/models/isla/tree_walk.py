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
decision is the reference's. Under autocast the forest's coordinates are bfloat16
(or float16), whose torch kernels compute each operation in float32 and round its
result, and sum in float32 with one rounding at the end; the kernels compute in
float32 and round at the same places.
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


@wp.func
def _low(x: Any, low: int):
    """``x`` rounded to the forest's storage precision: bfloat16 (``low = 1``),
    float16 (``2``), or kept (``0``)."""
    y = x
    if low == 1:
        y = type(x)(wp.bfloat16(wp.float32(x)))
    elif low == 2:
        y = type(x)(wp.float16(wp.float32(x)))
    return y


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
    low: int,
):
    """The reference's admission of node ``m`` for point ``q``: the squared gap from the
    point to the box, ``clamp(maximum(p - max, min - p), 0)`` squared and summed over
    the axes in order from zero, times ``theta_sq`` above the squared diagonal, and the
    node coherent. A NaN gap (torch's maximum and clamp propagate it) is never far."""
    zero = type(theta_sq)(0.0)
    dist_sq = zero
    nan = bool(False)
    for k in range(d):
        p = pts[q * d + k]
        a = _low(p - hi[m * d + k], low)
        b = _low(lo[m * d + k] - p, low)
        if wp.isnan(a) or wp.isnan(b):
            nan = True
        g = wp.max(wp.max(a, b), zero)
        dist_sq = dist_sq + _low(g * g, low)
    far = bool(False)
    if not nan:
        if _low(_low(dist_sq, low) * theta_sq, low) > diam_sq[m]:
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
    low: int,
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
        if _is_far(pts, q, lo, hi, diam_sq, coherent, m, d, theta_sq, low):
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
    low: int,
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
                if _is_far(pts, q, lo, hi, diam_sq, coherent, m, d, theta_sq, low):
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


#: The kernels' rounding mode for each storage dtype of the forest's boxes.
_LOW = {torch.bfloat16: 1, torch.float16: 2}


def walk_fits(forest) -> bool:
    """The forest's depth fits the count pass's stack and its boxes' dtype is one the
    kernels reproduce."""
    dtypes = (torch.float32, torch.float64, *_LOW)
    return len(forest.levels) + 2 <= WALK_STACK and forest.aabb_min.dtype in dtypes


def walk_pairs(forest, points_f: torch.Tensor, theta: float) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Every point's list ``(qi, si, n_node)`` for the points ``points_f`` ``(Q, d)`` in
    the forest's frame at ``theta > 0``, equal to the reference's
    (:meth:`BoundaryForest.point_pairs`) entry for entry; one host read, of the sizes."""
    lo = forest.aabb_min
    dtype, device = lo.dtype, lo.device
    n_q, d = points_f.shape
    n_t = int(forest.node_offset.shape[0]) - 1
    if n_q == 0 or n_t == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty.clone(), 0
    ### Reduced-precision coordinates are widened exactly to float32 and the kernels
    ### round where torch's kernels do.
    work = torch.float64 if dtype == torch.float64 else torch.float32
    low = _LOW.get(dtype, 0)
    pts = points_f.detach().to(dtype)
    scalar = wp.float64 if work == torch.float64 else wp.float32
    wtype = wp.dtype_from_torch(work)
    as_long = lambda x: wp.from_torch(x.contiguous(), dtype=wp.int64)  # noqa: E731
    as_real = lambda x: wp.from_torch(x.to(work).contiguous().reshape(-1), dtype=wtype)  # noqa: E731
    tree = [
        as_real(pts),
        as_real(lo),
        as_real(forest.aabb_max),
        as_real(forest.diameter_sq),
        wp.from_torch(forest.coherent.contiguous(), dtype=wp.bool),
        as_long(forest.left),
        as_long(forest.right),
    ]
    roots = as_long(forest.node_offset[:-1])
    walk = [scalar(theta * theta), low, d, n_q, n_t]
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
    seg_end = (n_near + n_far).cumsum(0)
    visit_end = n_visit.cumsum(0)
    n_pairs, n_node, n_scratch = torch.stack([seg_end[-1], n_far.sum(), visit_end[-1]]).tolist()
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
