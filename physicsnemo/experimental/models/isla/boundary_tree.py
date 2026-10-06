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

r"""A balanced cluster tree over the boundary cells for ISLA's tree-accelerated read.

Every node is a contiguous range of the cells in the tree's order and is split at its
median along a principal axis: of its cells' normals while the cluster's normals are
spread wider than the read's tolerance ``theta`` (so a thin body's two faces, or a
closed body's sides, part before anything else), of its cells' positions otherwise.
The clusters are therefore coherent in orientation first and tight in space second,
and the tree follows the geometry, not the coordinate axes: a rotation or reflection
of the case rotates the splits with it, up to ties (equal projections on the axis, or
a cluster whose two largest spreads are equal, where the chosen axis is a numerical
accident). All leaves are at the same depth and hold at most ``leaf_size`` cells.

Every node carries a bounding sphere, its measure-weighted centroid and the largest
distance of one of its cells from it; its normal spread, the measure-weighted RMS
chord between its cells' normals and their mean direction, which is
:math:`\sqrt{2 (1 - |\bar n|)}` with :math:`\bar n` the measure-weighted mean normal
before renormalization (1 for aligned normals, 0 for a fully mixed cluster; an RMS,
not a maximum, so a sliver cell with a stray normal does not condemn every cluster
above it); and the read's virtual cell: that centroid, the renormalized mean normal,
the measure-weighted mean rank-1 inputs and the total measure. The read admits a node
when both its angular size seen from the point and its normal spread are below
``theta``. The node means of any per-cell quantity are built by
:func:`~physicsnemo.experimental.models.isla.tree_read.node_means`.

Why normals: the kernel reads a cell's normal through three cosines, and the mean
normal of a cluster that spans both faces of a thin airfoil is the renormalization of
a near-zero vector. On the dense-trained AirfRANS checkpoints, replacing the normals
of the cells inside admitted clusters by the cluster mean reproduced the whole
deviation of the position-only tree read from the dense read (12% on pressure at
``theta = 0.5``), while replacing the positions by the centroid cost 1% to 2%.

GLOBE's :class:`~physicsnemo.mesh.spatial.cluster_tree.ClusterTree` (a Morton LBVH whose
nodes are count-midpoint ranges of the Morton order, with axis-aligned boxes) was the
first tree of the read. Its nodes straddle Morton boundaries at every level, so their
boxes are loose for a surface: on a 200,000-cell car the walk admitted 1,700 sources
per point at ``theta = 1`` against about 150 for tight clusters.
"""

import math
from dataclasses import dataclass

import torch

_PAD = (0, 0, 1, 0)
_EIGH_BATCH = 32768


def _prefix(x: torch.Tensor) -> torch.Tensor:
    """Leading-zero-padded float64 prefix sums along the first axis, ``(N + 1, ...)``."""
    flat = x.reshape(x.shape[0], -1).double()
    return torch.nn.functional.pad(torch.cumsum(flat, dim=0), _PAD)


@dataclass
class BoundaryTree:
    """A balanced binary tree over ``N`` cells in heap layout: node ``i`` has children
    ``2 i + 1`` and ``2 i + 2``; the ``2 ** depth`` leaves are the last nodes."""

    order: torch.Tensor  # (N,) the cells in tree order
    node_start: torch.Tensor  # (M,) each node's first position in that order
    node_count: torch.Tensor  # (M,) each node's cell count
    center: torch.Tensor  # (M, d) measure-weighted centroid
    radius: torch.Tensor  # (M,) largest distance of a cell from the centroid
    measure: torch.Tensor  # (M,) total measure
    normal: torch.Tensor  # (M, d) renormalized measure-weighted mean normal
    spread: torch.Tensor  # (M,) RMS chord of the cells' normals about ``normal``, sqrt(2 (1 - |mean|))
    vecs: torch.Tensor | None  # (M, P, d) measure-weighted mean rank-1 inputs
    depth: int  # the leaves' level (the root is level 0)
    theta: float  # the tolerance the splits were chosen for

    @property
    def n_cells(self) -> int:
        return int(self.order.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.node_start.shape[0])

    @property
    def first_leaf(self) -> int:
        return 2**self.depth - 1

    @classmethod
    def build(
        cls,
        points: torch.Tensor,
        measure: torch.Tensor,
        normals: torch.Tensor,
        vecs: torch.Tensor | None,
        *,
        leaf_size: int,
        theta: float,
        eps: float,
    ) -> "BoundaryTree":
        """Build the tree over ``points`` ``(N, d)`` with the cells' ``measure`` ``(N,)``,
        ``normals`` ``(N, d)`` and rank-1 inputs ``vecs`` ``(N, P, d)`` or ``None``, for
        the read's tolerance ``theta`` (a cluster whose normals' RMS chord about their
        mean exceeds it is split by normals). Runs without gradient (the tree is a
        numerical device)."""
        if leaf_size < 1:
            raise ValueError(f"leaf_size must be >= 1, got {leaf_size!r}")
        with torch.no_grad():
            n, d = points.shape
            device = points.device
            depth = 0
            while math.ceil(n / 2**depth) > leaf_size:
                depth += 1
            x_all = points.detach().double()
            n_all = normals.detach().double()
            w_all = measure.detach().double()
            ### A cluster's normals are within the tolerance when the length of their
            ### measure-weighted mean exceeds this (RMS chord sqrt(2 (1 - |mean|)) < theta).
            coherent_length = 1.0 - 0.5 * float(theta) ** 2
            order = torch.arange(n, device=device)
            starts = [torch.zeros(1, dtype=torch.long, device=device)]
            counts = [torch.full((1,), n, dtype=torch.long, device=device)]

            def principal(y, seg, k):
                """Per node: the centered rows, and the principal axis of their covariance."""
                mean = torch.zeros(k, d, dtype=y.dtype, device=device).index_add_(0, seg, y)
                mean = mean / counts[-1][:, None].to(y.dtype)
                yc = y - mean[seg]
                cov = torch.zeros(k, d, d, dtype=y.dtype, device=device).index_add_(
                    0, seg, yc[:, :, None] * yc[:, None, :]
                )
                ### Batched in slices: cuSOLVER's batched eigensolver fails beyond
                ### about 65,000 matrices (a tree of 200,000 cells with small leaves).
                axes = [torch.linalg.eigh(cov[a : a + _EIGH_BATCH])[1][..., -1] for a in range(0, k, _EIGH_BATCH)]
                return yc, torch.cat(axes)

            for level in range(depth):
                s, c = starts[-1], counts[-1]
                k = s.shape[0]
                seg = torch.repeat_interleave(torch.arange(k, device=device), c, output_size=n)
                xc, axis_x = principal(x_all[order], seg, k)
                nc, axis_n = principal(n_all[order], seg, k)
                ### The measure-weighted mean normal's length decides the split: a cluster
                ### whose normals are spread wider than theta splits by normals.
                w_o = w_all[order]
                wn = torch.zeros(k, d, dtype=x_all.dtype, device=device).index_add_(
                    0, seg, n_all[order] * w_o[:, None]
                )
                wsum = torch.zeros(k, dtype=x_all.dtype, device=device).index_add_(0, seg, w_o)
                length = wn.norm(dim=-1) / wsum.clamp_min(torch.finfo(torch.float64).tiny)
                by_normals = (length <= coherent_length)[seg]
                t = torch.where(by_normals, (nc * axis_n[seg]).sum(-1), (xc * axis_x[seg]).sum(-1))
                ### The axis's sign is a numerical accident; point it along the cells'
                ### third moment, so the children's order is the geometry's own too.
                skew = torch.zeros(k, dtype=t.dtype, device=device).index_add_(0, seg, t**3)
                t = torch.where(skew[seg] < 0, -t, t)
                ### Sort the cells of every node by their projection (a stable sort by
                ### projection, then a stable sort by node keeps the nodes contiguous).
                by_t = torch.argsort(t, stable=True)
                order = order[by_t][torch.argsort(seg[by_t], stable=True)]
                c_left = c // 2
                starts.append(torch.stack([s, s + c_left], -1).reshape(-1))
                counts.append(torch.stack([c_left, c - c_left], -1).reshape(-1))
            node_start, node_count = torch.cat(starts), torch.cat(counts)
            node_end = node_start + node_count
            tiny = torch.finfo(torch.float64).tiny
            ### Measure-weighted aggregates over every node's range.
            w = measure.detach()[order].double()
            p_w = _prefix(w)
            m = (p_w[node_end] - p_w[node_start])[:, 0]
            m_safe = m.clamp_min(tiny)[:, None]

            def mean_of(v):
                p = _prefix(v.detach()[order] * w.reshape(-1, *([1] * (v.ndim - 1))))
                out = (p[node_end] - p[node_start]) / m_safe
                return out.reshape(node_start.shape[0], *v.shape[1:])

            center = mean_of(x_all)
            nrm = mean_of(n_all)
            length = nrm.norm(dim=-1)
            spread = torch.sqrt((2.0 * (1.0 - length)).clamp_min(0.0))
            nrm = nrm / length[:, None].clamp_min(eps)
            vmean = None if vecs is None or not vecs.shape[1] else mean_of(vecs.double())
            ### The bounding radius: level by level, every cell against its node's centroid.
            radius = torch.zeros(node_start.shape[0], dtype=torch.float64, device=device)
            xs = x_all[order]
            for level in range(depth + 1):
                lo, hi = 2**level - 1, 2 ** (level + 1) - 1
                seg = lo + torch.repeat_interleave(
                    torch.arange(hi - lo, device=device), node_count[lo:hi], output_size=n
                )
                radius.scatter_reduce_(
                    0, seg, (xs - center[seg]).norm(dim=-1), reduce="amax", include_self=True
                )
            dt = points.dtype
            return cls(
                order=order,
                node_start=node_start,
                node_count=node_count,
                center=center.to(dt),
                radius=radius.to(dt),
                measure=m.to(measure.dtype),
                normal=nrm.to(normals.dtype),
                spread=spread.to(dt),
                vecs=None if vmean is None else vmean.to(vecs.dtype),
                depth=depth,
                theta=float(theta),
            )
