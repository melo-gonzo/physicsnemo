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

r"""ISLA-DTT's boundary read: passive prediction points read every boundary cell.

With ``query_independent=True`` (ISLA-DTT) every passive prediction point :math:`q`
gets one more residual update after every slice block: a read of every boundary cell
:math:`j`, a softmax over the cells of

.. math::

    z_{qjh} = K_h(\iota_{qj}) + \langle q_{qh}, k_{jh} \rangle / \sqrt{c} + \log \omega_j

weighting the cells' values :math:`v_{jh}`, with :math:`\omega_j` the cell's share of
the boundary measure, so the read is a measure-weighted mean over the boundary (a
quadrature). The kernel inputs :math:`\iota_{qj}` are invariants of the pair's relative
geometry (:func:`tree_pair_invariants`); one MLP maps them to a basis of
``KERNEL_READ_BASIS`` functions per pair once per forward pass (:class:`_KernelBasis`),
and each block's kernel :math:`K_h` is a learned linear combination of that basis per
head. The values, keys and queries are linear maps of invariant token states (the
cell's and the point's own), so the read is invariant and the equivariant heads
assemble every output. Read pair by pair, its cost is the number of (point, cell)
pairs, :math:`Q N` per block: 1e8 on an airfoil of a thousand cells, 4e10 on a car of
200,000.

The read is evaluated through GLOBE's cluster tree
(:class:`~physicsnemo.mesh.spatial.cluster_tree.ClusterTree`), walked once per
prediction point
(:meth:`~physicsnemo.mesh.spatial.cluster_tree.ClusterTree.find_point_interaction_pairs`)
as an N-body code walks its tree once per body. A point reads the cells of nearby
leaves one by one, and a distant cluster :math:`C` through one *virtual cell*: the
cluster's measure-weighted centroid, its measure-weighted mean normal (renormalized),
its total measure :math:`\Omega_C = \sum_{j \in C} \omega_j`, and the measure-weighted
means of its cells' keys and values at the block's depth. Because the read is a
quadrature, the virtual cell is the cluster's monopole: were the pair logit constant
over the cluster, :math:`\sum_{j \in C} \omega_j e^{z_{qj}} v_j` would equal
:math:`\Omega_C e^{z_{qC}} v_C` exactly.

The kernel reads a cell's normal through three cosines, so a cluster is a usable
virtual cell only when its normals agree: the mean normal of a cluster spanning both
faces of a thin body is a random direction. The boundary is therefore held as a
*forest*: the cells are split by the octant of their normal (the orientation split;
opposite normals never share a tree) and one cluster tree is built per octant
(:class:`BoundaryForest`). A point walks every tree. A cluster is admitted when its
box diagonal is below :math:`\theta` times its distance to the point (the tree's
criterion) and its normals are coherent: the measure-weighted RMS chord of the
normals about their mean, :math:`\sqrt{2 (1 - |\bar n|)}`, is below :math:`\theta`
(the tree's ``source_admissible`` mask). A leaf that is not admitted contributes its
cells; any other node hands the point to its two children. At :math:`\theta = 0` no
cluster is admitted and the read is exact, pair for pair. The model reads at
``TREE_READ_THETA``.

Symmetries. A point's list depends on the forest and the point alone, so the read is
exactly query independent at every :math:`\theta` (a point's prediction depends on
the boundary and the point, never on the other prediction points). At
:math:`\theta = 0` every contract of the exact read holds exactly: Euclidean
equivariance, the similarity gauge, boundary-refinement and measure-scale invariance.
At finite :math:`\theta` the forest is built in a frame the boundary carries with it
(:func:`canonical_frame`: the principal axes of the cells' measure-weighted
covariance, signed by their third moments), so a rotation, reflection, translation
or rescale of the case gives the same clusters and the same far-field list for every
point, up to ties of that frame (equal variances, a symmetric axis); the read is then
the same approximation in every pose, and the posed protocol equals the native one.
Trees built on lab coordinates were not: on AirfRANS at theta 1 their error rose
1.9x / 1.25x / 3.3x on pressure / velocity / viscosity under a rigid motion.

Constants. ``TREE_READ_LEAF_SIZE`` (cells per leaf), the orientation split into the
:math:`2^d` sign octants and ``TREE_READ_PAIR_CHUNK`` (pairs per chunk of the basis
evaluation) are implementation constants: at a fixed :math:`\theta` they change the
cost and the grouping of the exact pairs, and at :math:`\theta = 0` nothing at all.

Earlier forms, measured and retired on DrivAerML Volume (200,000 cells, 200,000
points): the dual-tree plan with a tree over the points (a leaf of points in a
sparse region is never well separated from the boundary near it: 1.6e9 pairs per
block at theta 0.5); the per-point walk of the count-midpoint Morton tree (loose
boxes on a surface: 1,700 sources per point at theta 1); a dedicated principal-axis
tree with normal-first splits and bounding spheres (980 sources per point at theta 1,
within 10% of this forest, which builds five times faster and shares its tree with
GLOBE).
"""

import dataclasses
import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.utils.checkpoint import checkpoint

from physicsnemo.experimental.models.isla import fused_basis, fused_read
from physicsnemo.experimental.models.isla.segment_softmax import (
    PairPlan,
    segment_softmax_read,
)
from physicsnemo.experimental.models.isla.tree_build import forest_tables, resolve_tree_build
from physicsnemo.experimental.models.isla.tree_walk import walk_fits, walk_pairs
from physicsnemo.mesh.spatial._ragged import _ragged_arange
from physicsnemo.mesh.spatial.cluster_tree import ClusterTree

logger = logging.getLogger(__name__)
#: Why the fused read used the reference kernel basis, each reason logged once.
_BASIS_FALLBACKS_LOGGED: set[str] = set()

#: Heads of the read; each head owns ``KERNEL_READ_WIDTH // KERNEL_READ_HEADS`` value
#: channels.
KERNEL_READ_HEADS = 8
#: Value channels of the read (all heads together).
KERNEL_READ_WIDTH = 64
#: Query/key channels per head.
KERNEL_READ_KEY = 4
#: Hidden width of the shared kernel MLP on the pair invariants.
KERNEL_READ_HIDDEN = 64
#: Kernel basis functions of the pair invariants, shared by every block; each block's
#: kernel is a learned linear combination of them per head.
KERNEL_READ_BASIS = 16
#: The Barnes-Hut opening angle of the model's read. Every ISLA-DTT result was trained
#: and evaluated at 1; a model is read at the angle it was trained at (a cluster's
#: aggregate is what the kernel learns to read).
TREE_READ_THETA = 1.0

#: Boundary cells per tree leaf. Implementation: at a fixed ``theta`` it sets how many
#: cells an opened leaf contributes; at ``theta = 0`` it changes nothing.
TREE_READ_LEAF_SIZE = 4
#: Pairs per chunk of the kernel-basis evaluation (memory only; each chunk's
#: invariants are built once, its MLP activations are rebuilt in backward).
TREE_READ_PAIR_CHUNK = 1 << 22


#: Below this measure-weighted normalized third moment along a principal axis, the
#: axis's sign is a tie (a symmetric body) and is fixed by the lab axis it most
#: overlaps instead. Computed in float64, so a body's skewness is reproduced under a
#: rigid motion to 1e-12 and only a genuinely symmetric axis falls back.
CANONICAL_SKEW_TIE = 1e-9


def canonical_frame(points: torch.Tensor, measure: torch.Tensor) -> torch.Tensor:
    r"""A frame the boundary carries with it, ``(d, d)`` with the axes as columns: the
    principal axes of the measure-weighted covariance of the cell positions, in
    ascending variance, each signed so that the measure-weighted third moment of the
    positions along it is positive. Under a rotation, reflection, translation or
    rescale of the boundary the frame moves with it (up to ties: equal variances, or a
    symmetric axis whose third moment vanishes, where the sign follows the lab axis
    of largest overlap), so coordinates ``points @ frame`` are invariant. The Morton
    trees and the normal octants are built in these coordinates, which makes the
    forest, and so every point's far-field list at finite ``theta``, the same in
    every pose of the case; a tree built on lab coordinates changes its clusters with
    the pose (on AirfRANS at theta 1, errors 1.9x / 1.25x / 3.3x on pressure /
    velocity / viscosity under the posed protocol against the native one)."""
    d = points.shape[1]
    x = points.detach().double()
    w = measure.detach().double().clamp_min(0)
    w = w / w.sum().clamp_min(torch.finfo(torch.float64).tiny)
    x = x - (w[:, None] * x).sum(0)
    wx = w[:, None] * x
    ### Column sums, not a GEMM (see frame_coordinates).
    cov = torch.stack([(wx * x[:, j, None]).sum(0) for j in range(d)], 1)
    _, axes = torch.linalg.eigh(cov)
    proj = frame_coordinates(x, axes)
    m2 = (w[:, None] * proj.square()).sum(0)
    m3 = (w[:, None] * proj.pow(3)).sum(0)
    skew = m3 / m2.clamp_min(torch.finfo(torch.float64).tiny).pow(1.5)
    lab = axes[axes.abs().argmax(dim=0), torch.arange(d, device=points.device)]
    sign = torch.where(skew.abs() > CANONICAL_SKEW_TIE, torch.sign(skew), torch.sign(lab))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return (axes * sign[None, :]).to(points.dtype)


def frame_coordinates(points: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """``points @ frame`` ``(N, d)`` as ``d`` column products. A float64 GEMM this thin
    (or the ``(d, N) @ (N, d)`` of a covariance) runs far below memory bandwidth on
    GPUs with few float64 units: on GB300 at 17.7M cells, 17.8 ms (34 ms) against
    0.9 ms (1.3 ms) as columns."""
    out = points[:, :1] * frame[0]
    for k in range(1, points.shape[1]):
        out = out + points[:, k : k + 1] * frame[k]
    return out


def normal_octants(normals: torch.Tensor) -> torch.Tensor:
    """The sign octant of each normal, ``0`` to ``2**d - 1`` (bit ``k`` set when
    component ``k`` is positive; a zero component takes the sign of the next nonzero
    one, cyclically). Antipodal: ``-n`` lands in the octant ``2**d - 1`` minus
    that of ``n``, so opposite normals never share an octant and a winding flip
    maps every octant's cells to the antipodal octant."""
    d = normals.shape[1]
    code = torch.zeros(normals.shape[0], dtype=torch.long, device=normals.device)
    for k in range(d):
        c = normals[:, k]
        for j in range(1, d):
            c = torch.where(c == 0, normals[:, (k + j) % d], c)
        code = code * 2 + (c > 0).long()
    return code


@dataclass
class BoundaryForest:
    """One case's boundary as a forest of cluster trees, one per normal octant, with
    the trees' nodes in one table (tree by tree, each tree's nodes in its own order)
    and the cells in one order (tree by tree, each tree's cells in its sorted order).

    Node ``m`` holds the cells ``cells[range_start[m] : range_start[m] +
    range_count[m]]``; its virtual cell is ``center[m]``, ``normal[m]``,
    ``vecs[m]`` and ``measure[m]``; ``coherent[m]`` says whether a point may read it
    as one (RMS chord of its normals below ``theta``). ``levels`` groups the internal
    nodes by depth, deepest first, with their children, for the bottom-up sums of
    :func:`node_means`.
    """

    n_cells: int
    theta: float
    frame: torch.Tensor  # (d, d) the boundary's own frame (canonical_frame); trees and boxes live in points @ frame
    trees: list[ClusterTree]
    tree_cells: list[torch.Tensor]  # per tree, the cells it holds (its sources, in its input order)
    node_offset: torch.Tensor  # (T + 1,) each tree's first node in the table
    cells: torch.Tensor  # (N,) every cell once, in forest order
    range_start: torch.Tensor  # (M,) each node's first position in ``cells``
    range_count: torch.Tensor  # (M,)
    left: torch.Tensor  # (M,) child in the table, -1 at a leaf
    right: torch.Tensor  # (M,)
    leaf_ids: torch.Tensor  # (L,) the leaves, in position order
    position_leaf: torch.Tensor  # (N,) the leaf (counted in ``leaf_ids``) of each position
    levels: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    aabb_min: torch.Tensor  # (M, d)
    aabb_max: torch.Tensor  # (M, d)
    diameter_sq: torch.Tensor  # (M,)
    center: torch.Tensor  # (M, d) measure-weighted centroid
    normal: torch.Tensor  # (M, d) renormalized measure-weighted mean normal
    vecs: torch.Tensor | None  # (M, P, d) measure-weighted mean rank-1 inputs
    measure: torch.Tensor  # (M,) total measure
    coherent: torch.Tensor  # (M,) bool

    @property
    def n_nodes(self) -> int:
        return int(self.range_start.shape[0])

    @classmethod
    def build(
        cls,
        points: torch.Tensor,
        measure: torch.Tensor,
        normals: torch.Tensor,
        vecs: torch.Tensor | None,
        *,
        theta: float,
        leaf_size: int = TREE_READ_LEAF_SIZE,
        eps: float = 1e-12,
        build: str = "auto",
    ) -> "BoundaryForest":
        """The forest over cells at ``points`` ``(N, d)`` with ``measure`` ``(N,)``,
        unit ``normals`` ``(N, d)`` and rank-1 inputs ``vecs`` ``(N, P, d)`` or
        ``None``. No gradient flows into it (a numerical device). ``build`` picks how
        it is built, not what it is (:func:`~.tree_build.resolve_tree_build`): the
        per-octant cluster trees below, or the same forest from one pass over all
        octants (:func:`~.tree_build.forest_tables`)."""
        points, measure, normals = points.detach(), measure.detach(), normals.detach()
        vecs = None if vecs is None else vecs.detach()
        n, d = points.shape
        device = points.device
        coherent_length = 1.0 - 0.5 * float(theta) ** 2
        ### The trees and the octants live in the boundary's own frame, so the forest
        ### is the same in every pose of the case (up to ties); the virtual cells
        ### (centroids, normals, rank-1 means) stay in the frame the read works in.
        frame = canonical_frame(points, measure)
        points_f, normals_f = points @ frame, normals @ frame
        octant = normal_octants(normals_f)
        if resolve_tree_build(build, points) == "fast":
            tables = forest_tables(
                points, measure, normals, vecs, points_f, octant, leaf_size=leaf_size
            )
            if tables is not None:
                mean_n = tables.pop("mean_n")
                length = mean_n.norm(dim=-1)
                return cls(
                    n_cells=n,
                    theta=float(theta),
                    frame=frame,
                    **tables,
                    normal=mean_n / length[:, None].clamp_min(eps),
                    coherent=length > coherent_length,
                )
        trees, tree_cells, cells, offsets = [], [], [], [0]
        starts, counts, lefts, rights, boxes = [], [], [], [], []
        centers, nrm, vec, meas = [], [], [], []
        c_off = m_off = 0
        for k in range(2**d):
            sel = (octant == k).nonzero().squeeze(-1)
            if sel.numel() == 0:
                continue
            tree = ClusterTree.from_points(points_f[sel], leaf_size=leaf_size, areas=measure[sel])
            data = {"n": normals[sel]}
            if vecs is not None:
                data["u"] = vecs[sel]
            agg = tree.compute_source_aggregates(
                points[sel], measure[sel], TensorDict(data, batch_size=[int(sel.numel())])
            )
            m = tree.n_nodes
            trees.append(tree)
            tree_cells.append(sel)
            cells.append(sel[tree.sorted_source_order])
            starts.append(tree.node_range_start + c_off)
            counts.append(tree.node_range_count)
            lefts.append(torch.where(tree.node_left_child >= 0, tree.node_left_child + m_off, -1))
            rights.append(torch.where(tree.node_right_child >= 0, tree.node_right_child + m_off, -1))
            boxes.append((tree.node_aabb_min, tree.node_aabb_max, tree.node_diameter_sq))
            centers.append(agg.node_centroid)
            nrm.append(agg.node_source_data["n"])
            if vecs is not None:
                vec.append(agg.node_source_data["u"])
            meas.append(tree.node_total_area)
            c_off += int(sel.numel())
            m_off += m
            offsets.append(m_off)
        left = torch.cat(lefts)
        right = torch.cat(rights)
        range_start = torch.cat(starts)
        range_count = torch.cat(counts)
        ### The leaves tile the cells: sorted by their first position, their counts
        ### concatenate to the forest order.
        leaf_ids = (left < 0).nonzero().squeeze(-1)
        leaf_ids = leaf_ids[torch.argsort(range_start[leaf_ids])]
        position_leaf = torch.repeat_interleave(
            torch.arange(leaf_ids.shape[0], device=device), range_count[leaf_ids], output_size=n
        )
        ### Internal nodes by depth from the roots; a node's children are one level deeper.
        node_offset = torch.tensor(offsets, device=device)
        frontier = node_offset[:-1]
        by_depth = []
        while frontier.numel():
            internal = frontier[left[frontier] >= 0]
            if internal.numel() == 0:
                break
            by_depth.append(internal)
            frontier = torch.cat([left[internal], right[internal]])
        levels = [(nodes, left[nodes], right[nodes]) for nodes in reversed(by_depth)]
        mean_n = torch.cat(nrm)
        length = mean_n.norm(dim=-1)
        return cls(
            n_cells=n,
            theta=float(theta),
            frame=frame,
            trees=trees,
            tree_cells=tree_cells,
            node_offset=node_offset,
            cells=torch.cat(cells),
            range_start=range_start,
            range_count=range_count,
            left=left,
            right=right,
            leaf_ids=leaf_ids,
            position_leaf=position_leaf,
            levels=levels,
            aabb_min=torch.cat([b[0] for b in boxes]),
            aabb_max=torch.cat([b[1] for b in boxes]),
            diameter_sq=torch.cat([b[2] for b in boxes]),
            center=torch.cat(centers),
            normal=mean_n / length[:, None].clamp_min(eps),
            vecs=torch.cat(vec) if vecs is not None else None,
            measure=torch.cat(meas),
            coherent=length > coherent_length,
        )

    def node_cells(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The cells of the listed nodes: ``(cell, which)`` with ``which`` the index into
        ``nodes`` each cell belongs to."""
        pos, which = _ragged_arange(self.range_start[nodes], self.range_count[nodes])
        return self.cells[pos], which

    def point_pairs(
        self, points: torch.Tensor, theta: float | None = None, walk: str = "auto"
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Every point's list: ``(qi, si, n_node)`` sorted by point, where ``si`` below
        ``n_cells`` names a cell and ``n_cells + m`` names node ``m``; ``n_node`` is the
        number of node entries. Each tree is walked once per point at ``theta``
        (the forest's own by default); a node is admitted only when its box is well
        separated from the point and its normals are coherent, so each point's list
        holds every cell exactly once, as itself or inside one node. ``walk`` names
        how, not what (:func:`resolve_walk`): the fast walk gives the same lists in
        the same order. Lists past Warp's array limit are walked by the reference
        under ``"auto"`` and refused under ``"fast"``."""
        theta = self.theta if theta is None else float(theta)
        points_f = points.detach() @ self.frame  # the trees' coordinates
        if resolve_walk(walk, points, theta, self) == "fast":
            pairs = walk_pairs(self, points_f, theta)
            if pairs is not None:
                return pairs
            if walk == "fast":
                raise ValueError(f"the fast walk's arrays for {points.shape[0]} points pass Warp's 2**31 elements")
            key = (points.device.type, "reference past Warp's array limit")
            if key not in _WALKS_LOGGED:
                _WALKS_LOGGED.add(key)
                logger.info("ISLA tree walk=auto walks %d points by the reference: past Warp's array limit", points.shape[0])
        qs, ss, n_node = [], [], 0
        for t, tree in enumerate(self.trees):
            a, b = int(self.node_offset[t]), int(self.node_offset[t + 1])
            plan = tree.find_point_interaction_pairs(
                points_f, theta=theta, validate=False, source_admissible=self.coherent[a:b]
            )
            qs += [plan.near_target_ids, plan.far_target_ids]
            ss += [self.tree_cells[t][plan.near_source_ids], self.n_cells + a + plan.far_source_node_ids]
            n_node += plan.n_far
        qi, si = torch.cat(qs), torch.cat(ss)
        order = torch.argsort(qi, stable=True)
        return qi[order], si[order], n_node


_WALKS_LOGGED: set[tuple[str, str]] = set()


def resolve_walk(walk: str, points: torch.Tensor, theta: float, forest: BoundaryForest) -> str:
    """The walk :meth:`BoundaryForest.point_pairs` runs: ``"reference"`` (GLOBE's
    cluster-tree traversal, tree by tree) or ``"fast"`` (:mod:`tree_walk`'s Warp
    kernels, the same pairs in the same order). ``"auto"`` picks the fast walk for
    CUDA points when the forest's depth fits its stack, else the reference, logged
    once per device type and choice; a forced name is kept (``"fast"`` runs on CPU
    too). At ``theta = 0`` there is no walk (the plan is every pair) and the
    reference's construction runs either way. Under autocast the forest's
    coordinates are in the autocast dtype (``points @ frame``), and the fast walk
    rounds as its torch kernels do."""
    if walk not in ("auto", "reference", "fast"):
        raise ValueError(f"walk must be 'auto', 'reference' or 'fast', got {walk!r}")
    if theta == 0:
        return "reference"
    if walk == "fast" and not walk_fits(forest):
        raise ValueError(
            f"the fast walk does not take this forest ({len(forest.levels)} levels, {forest.aabb_min.dtype} boxes)"
        )
    if walk != "auto":
        return walk
    fits = walk_fits(forest)
    chosen = "fast" if points.is_cuda and fits else "reference"
    key = (points.device.type, chosen)
    if key not in _WALKS_LOGGED:
        _WALKS_LOGGED.add(key)
        logger.info(
            "ISLA tree walk=auto resolved to %r on %s (cuda=%s, forest depth %d, %s boxes, fits=%s)",
            chosen,
            points.device.type,
            points.is_cuda,
            len(forest.levels),
            forest.aabb_min.dtype,
            fits,
        )
    return chosen


@dataclass
class TreeReadPairs:
    """The tree read's pairs of one forward pass, every case flattened: ``plan`` over
    the ``B * Q`` points and the virtual sources (each case's ``N`` cells followed by
    its forest's nodes), the kernel basis and log measure of every pair, the cells'
    measure fractions, and each case's forest."""

    plan: PairPlan
    basis: torch.Tensor  # (P, KERNEL_READ_BASIS)
    log_m: torch.Tensor  # (P,)
    omega: torch.Tensor  # (B * N,) the cells' measure fractions
    forests: list[BoundaryForest]
    src_offsets: list[int]
    n_cells: int
    n_queries: int
    n_pairs_node: int  # (point, node) pairs over every case
    kernel: str = "reference"  # the blocks' read: "reference" or "fused" (fused_read.py)
    src_log_m: torch.Tensor | None = None  # (n_b,) every source's log measure; log_m = src_log_m[plan.bj]
    basis_sink: tuple | None = None  # the fused reads' shared basis gradient (fused_read.basis_sink)

    @property
    def n_pairs(self) -> int:
        return int(self.plan.qi.shape[0])

    @property
    def n_pairs_cell(self) -> int:
        return self.n_pairs - self.n_pairs_node


def node_means(x: torch.Tensor, forest: BoundaryForest, omega: torch.Tensor) -> torch.Tensor:
    """The measure-weighted mean of the cells' rows ``x`` ``(N, ...)`` over every node
    of the forest, ``(M, ...)``, differentiable in ``x``: the leaves' sums by one
    scatter over the forest order, then every internal node as the sum of its two
    children, depth by depth up to the roots (a tree of pairwise sums, no
    cancellation), in float64 for float64 rows and in float32 otherwise (float32,
    bfloat16 and float16 rows alike)."""
    work = torch.float64 if x.dtype == torch.float64 else torch.float32
    cells = forest.cells
    flat = x.reshape(x.shape[0], -1).to(work)[cells]
    weighted = flat * omega[cells, None].to(work)
    n_leaves = forest.leaf_ids.shape[0]
    leaf_sums = weighted.new_zeros(n_leaves, weighted.shape[1]).index_add(
        0, forest.position_leaf, weighted
    )
    sums = weighted.new_zeros(forest.n_nodes, weighted.shape[1]).index_put(
        (forest.leaf_ids,), leaf_sums
    )
    for nodes, left, right in forest.levels:
        sums = sums.index_put((nodes,), sums[left] + sums[right])
    measure = forest.measure.to(work).clamp_min(torch.finfo(work).tiny)
    return (sums / measure[:, None]).to(x.dtype).reshape(forest.n_nodes, *x.shape[1:])


def tree_pair_invariants(
    qi: torch.Tensor,
    si: torch.Tensor,
    *,
    query_r: torch.Tensor,
    query_n: torch.Tensor,
    query_wall: torch.Tensor,
    src_r: torch.Tensor,
    src_n: torch.Tensor,
    src_vecs: torch.Tensor | None,
    g_unit: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    r"""The kernel inputs of the listed (point, source) pairs of one case, ``(P, G)``,
    the sources either cells or virtual cells.

    ``query_r``, ``query_n`` ``(Q, d)``, ``query_wall`` ``(Q,)`` the point's built-in
    wall distance :math:`|d_q|` (frame units); ``src_r``, ``src_n`` ``(S, d)``, ``src_vecs`` ``(S, P_b, d)`` or ``None``;
    ``g_unit`` ``(K, d)``. With :math:`\rho = (x_q - y_j) / L`, :math:`r = |\rho|` and
    :math:`\hat\rho = \rho / r` the invariants are :math:`[r, \log r, \log(r / |d_q|),
    \hat\rho\cdot n_j, \hat\rho\cdot n_q, n_q\cdot n_j, \hat\rho\cdot g_k, n_j\cdot g_k,
    \hat\rho\cdot u, n_q\cdot u]` (``u`` the source's rank-1 inputs): lengths in units of :math:`L` or of the point's
    own wall distance, and cosines, so no length scale is chosen. The only constant is
    ``eps``, a floor on the lengths inside the logarithms and the normalization (a
    numerical guard: :math:`r \ge |d_q| > 0` for every point off the boundary and every
    cell centroid source, since :math:`|d_q|` is the distance to the boundary surface,
    on which the cell centroids lie)."""
    with torch.autocast(query_r.device.type, enabled=False):
        rel = query_r[qi] - src_r[si]
        dist = rel.norm(dim=-1)
        dist_g = dist.clamp_min(eps)
        rho = rel / dist_g[:, None]
        n_q, n_j = query_n[qi], src_n[si]
        log_r = torch.log(dist_g)
        feats = [
            dist,
            log_r,
            log_r - torch.log(query_wall[qi].clamp_min(eps)),
            (rho * n_j).sum(-1),
            (rho * n_q).sum(-1),
            (n_q * n_j).sum(-1),
        ]
        cols = [torch.stack(feats, -1), rho @ g_unit.t(), n_j @ g_unit.t()]
        if src_vecs is not None and src_vecs.shape[1]:
            u = src_vecs[si]
            cols += [(rho[:, None] * u).sum(-1), (n_q[:, None] * u).sum(-1)]
        return torch.cat(cols, -1)


def kernel_geo_width(n_global_vectors: int, n_boundary_vecs: int) -> int:
    """Width of :func:`tree_pair_invariants`: r, log r, log(r / |d_q|), rho.n_j, rho.n_q,
    n_q.n_j, then rho.g_k and n_j.g_k (2K), rho.u and n_q.u per boundary vector (2P_b)."""
    return 6 + 2 * int(n_global_vectors) + 2 * int(n_boundary_vecs)


class _KernelBasis(nn.Module):
    """The kernel basis ``phi(iota)`` ``(P, KERNEL_READ_BASIS)``: one MLP of the pair
    invariants, evaluated once per forward pass and shared by every block's read (the
    per-pair work of a block is then a small linear map, not an MLP)."""

    def __init__(self, n_geo: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_geo, KERNEL_READ_HIDDEN),
            nn.GELU(),
            nn.Linear(KERNEL_READ_HIDDEN, KERNEL_READ_HIDDEN),
            nn.GELU(),
            nn.Linear(KERNEL_READ_HIDDEN, KERNEL_READ_BASIS),
        )

    def forward(self, inv: torch.Tensor) -> torch.Tensor:
        return self.net(inv)


def _chunked_basis(basis: _KernelBasis, qi, si, geometry: dict) -> torch.Tensor:
    """The kernel basis of every pair ``(P, KERNEL_READ_BASIS)`` in chunks of
    ``TREE_READ_PAIR_CHUNK`` pairs. The pair invariants are geometry of the inputs (no
    gradient reaches them), so they are computed once and kept; under gradient only
    the basis MLP of each chunk is checkpointed (its hidden activations are rebuilt in
    backward, the gathers of the invariants are not)."""
    grad = torch.is_grad_enabled() and any(p.requires_grad for p in basis.parameters())
    parts = []
    for a in range(0, int(qi.shape[0]), TREE_READ_PAIR_CHUNK):
        with torch.no_grad():
            inv = tree_pair_invariants(
                qi[a : a + TREE_READ_PAIR_CHUNK], si[a : a + TREE_READ_PAIR_CHUNK], **geometry
            )
        ### Kept in the MLP's own dtype (bfloat16 under autocast): the fused read consumes it in every block and its gradient
        ### accumulates over the blocks in that dtype; 2e8 to 7e8 rows at DrivAerML scale.
        parts.append(checkpoint(basis, inv, use_reentrant=False) if grad else basis(inv))
    if not parts:
        return geometry["query_r"].new_zeros(0, KERNEL_READ_BASIS)
    return torch.cat(parts)


def _basis_of_pairs(basis: _KernelBasis, qi, si, q_off: int, s_off: int, geometry: dict) -> torch.Tensor:
    with torch.no_grad():
        inv = tree_pair_invariants(qi.long() - q_off, si.long() - s_off, **geometry)
    return basis(inv)


def _chunked_basis_from_plan(
    basis: _KernelBasis,
    plan: PairPlan,
    a: int,
    b: int,
    q_off: int,
    s_off: int,
    geometry: dict,
    weights: tuple | None = None,
) -> torch.Tensor:
    """The kernel basis of one case's pairs, the plan's rows ``a:b``, in the chunks of
    :func:`_chunked_basis` and with its values, but each chunk's invariants inside the
    checkpoint too: backward rebuilds them from the plan's indices (``q_off`` and
    ``s_off`` the case's first point and source), so nothing per pair but the basis
    is kept. With ``weights`` (:func:`fused_basis.mlp_weights`) each chunk is one
    fused kernel of the invariants and the MLP (:mod:`fused_basis`)."""
    grad = torch.is_grad_enabled() and any(p.requires_grad for p in basis.parameters())
    parts = []
    for c in range(a, b, TREE_READ_PAIR_CHUNK):
        e = min(c + TREE_READ_PAIR_CHUNK, b)
        if weights is not None:
            parts.append(fused_basis.fused_kernel_basis(weights, plan.qi[c:e], plan.bj[c:e], q_off, s_off, geometry))
            continue
        args = (basis, plan.qi[c:e], plan.bj[c:e], q_off, s_off, geometry)
        parts.append(checkpoint(_basis_of_pairs, *args, use_reentrant=False) if grad else _basis_of_pairs(*args))
    if not parts:
        return geometry["query_r"].new_zeros(0, KERNEL_READ_BASIS)
    return torch.cat(parts)


@dataclass
class TreeReadWalk:
    """Every case's forest and point lists (:func:`tree_read_walk`), before the kernel
    basis: per case the pairs ``(qi, si, n_node)`` sorted by point, the inputs of their
    invariants (:func:`tree_pair_invariants`) and the sources' log measures.
    :func:`walk_range_pairs` assembles the pairs of any range of points from it;
    ``kernel`` is the read's implementation (:func:`tree_read_walk`)."""

    forests: list[BoundaryForest]
    lists: list[tuple[torch.Tensor, torch.Tensor, int]]
    geometry: list[dict]
    src_logm: list[torch.Tensor]
    omega: torch.Tensor  # (B * N,) the cells' measure fractions
    n_cells: int
    n_queries: int
    kernel: str


def tree_read_pairs(basis: _KernelBasis, **inputs) -> TreeReadPairs:
    """The tree read's pairs of every point of a batch of cases, with their kernel
    basis: :func:`walk_range_pairs` of :func:`tree_read_walk` (which takes the
    keyword arguments)."""
    return walk_range_pairs(basis, tree_read_walk(**inputs))


def tree_read_walk(
    *,
    query_r: torch.Tensor,
    query_n: torch.Tensor,
    query_wall: torch.Tensor,
    boundary_r: torch.Tensor,
    boundary_n: torch.Tensor,
    boundary_vecs: torch.Tensor | None,
    log_frac: torch.Tensor,
    g_unit: torch.Tensor,
    theta: float,
    eps: float,
    walk: str = "auto",
    build: str = "auto",
    kernel: str = "reference",
) -> TreeReadWalk:
    r"""Walk the tree read's forests for a batch of cases.

    ``query_r``, ``query_n`` ``(B, Q, d)`` (frame units; the point's built-in
    direction), ``query_wall`` ``(B, Q)`` the point's built-in wall distance;
    ``boundary_r``, ``boundary_n``
    ``(B, N, d)``, ``boundary_vecs`` ``(B, N, P_b, d)`` or ``None``; ``log_frac``
    ``(B, N)`` the cells' log measure fractions; ``g_unit`` ``(B, K, d)``.

    Per case: the forest over the cells (:class:`BoundaryForest`, with the measure
    fractions as the measure), each point's walk of it at ``theta``
    (:meth:`BoundaryForest.point_pairs`), so each point's list is a set of cells and
    a set of nodes that together cover every cell exactly once; the nodes' virtual
    cells (centroid, renormalized mean normal, mean rank-1 inputs, total measure)
    come with the forest. The forest and the walk run without gradient (a numerical
    device); the invariants and the basis (:func:`walk_range_pairs`) carry gradients
    to the kernel MLP. ``walk`` names the walk's implementation
    (:func:`resolve_walk`); ``build`` is the forest's build
    (:meth:`BoundaryForest.build`); ``kernel`` is the read's, recorded for
    :func:`walk_range_pairs` (with ``"fused"`` the blocks read through
    :mod:`fused_read`, and the basis is built after the plan from its indices, the
    same values with the invariants rebuilt in backward:
    :func:`_chunked_basis_from_plan`, by one fused kernel per chunk where
    :func:`fused_basis.supported` holds).
    """
    b, nq, d = query_r.shape
    n = boundary_r.shape[1]
    tiny = torch.finfo(torch.float32).tiny
    has_vecs = boundary_vecs is not None and bool(boundary_vecs.shape[2])
    forests, lists, geometries, src_logms = [], [], [], []
    for i in range(b):
        with torch.no_grad():
            omega = torch.exp(log_frac[i])
            forest = BoundaryForest.build(
                boundary_r[i],
                omega,
                boundary_n[i],
                boundary_vecs[i] if has_vecs else None,
                theta=theta,
                leaf_size=TREE_READ_LEAF_SIZE,
                eps=eps,
                build=build,
            )
            qi, si, n_node = forest.point_pairs(query_r[i], walk=walk)
            node_measure = forest.measure.to(log_frac.dtype)
        src_r = torch.cat([boundary_r[i], forest.center.to(boundary_r.dtype)])
        src_n = torch.cat([boundary_n[i], forest.normal.to(boundary_n.dtype)])
        src_vecs = (
            torch.cat([boundary_vecs[i], forest.vecs.to(boundary_vecs.dtype)])
            if has_vecs
            else None
        )
        src_logm = torch.cat([log_frac[i], torch.log(node_measure.clamp_min(tiny))])
        geometry = {
            "query_r": query_r[i],
            "query_n": query_n[i],
            "query_wall": query_wall[i],
            "src_r": src_r,
            "src_n": src_n,
            "src_vecs": src_vecs,
            "g_unit": g_unit[i],
            "eps": eps,
        }
        forests.append(forest)
        lists.append((qi, si, n_node))
        geometries.append(geometry)
        src_logms.append(src_logm)
    return TreeReadWalk(
        forests=forests,
        lists=lists,
        geometry=geometries,
        src_logm=src_logms,
        omega=torch.exp(log_frac).reshape(-1),
        n_cells=n,
        n_queries=nq,
        kernel=kernel,
    )


def walk_range_pairs(
    basis: _KernelBasis, walk: TreeReadWalk, start: int = 0, stop: int | None = None
) -> TreeReadPairs:
    """The pairs of the points ``start`` to ``stop`` (default: every point) of each
    case of ``walk``, numbered from ``start``, with the kernel basis of every pair in
    chunks: each point's list is the walk's, so a range of points reads exactly what
    it reads among all of them."""
    n, nq_all = walk.n_cells, walk.n_queries
    stop = nq_all if stop is None else stop
    nq = stop - start
    qis, sis, bases, logms, offsets = [], [], [], [], []
    deferred = []
    src_off = 0
    pair_off = 0
    n_pairs_node = 0
    for i, forest in enumerate(walk.forests):
        qi, si, n_node = walk.lists[i]
        if (start, stop) != (0, nq_all):
            a, b = torch.searchsorted(qi, qi.new_tensor([start, stop])).tolist()
            qi, si = qi[a:b], si[a:b]
            n_node = int((si >= n).sum())
        if walk.kernel == "fused":
            deferred.append((pair_off, pair_off + int(qi.shape[0]), i * nq - start, src_off, walk.geometry[i]))
        else:
            bases.append(_chunked_basis(basis, qi, si, walk.geometry[i]))
        pair_off += int(qi.shape[0])
        logms.append(walk.src_logm[i][si])
        qis.append(qi + (i * nq - start))
        sis.append(si + src_off)
        offsets.append(src_off)
        src_off += n + forest.n_nodes
        n_pairs_node += n_node
    plan_all = PairPlan.build(torch.cat(qis), torch.cat(sis), len(walk.forests) * nq, src_off)
    if plan_all.qi.shape[0] < 2**31:
        ### 32-bit pair indices (the fused read widens them on load): half the plan's memory.
        plan_all = dataclasses.replace(
            plan_all,
            qi=plan_all.qi.int(),
            bj=plan_all.bj.int(),
            perm=plan_all.perm.int(),
            qi_by_b=plan_all.qi_by_b.int(),
        )
    ### One cast of the MLP's weights for every case and chunk, as autocast's weight cache gives the reference.
    reason = next(filter(None, (fused_basis.unsupported_reason(basis, item[-1]) for item in deferred)), None)
    weights = fused_basis.mlp_weights(basis) if deferred and reason is None else None
    if reason is not None and reason not in _BASIS_FALLBACKS_LOGGED:
        _BASIS_FALLBACKS_LOGGED.add(reason)
        logger.info("ISLA fused tree read: the reference kernel basis is used (%s)", reason)
    for a, e, q_off, s_off, geometry in deferred:
        bases.append(_chunked_basis_from_plan(basis, plan_all, a, e, q_off, s_off, geometry, weights))
    basis_all = torch.cat(bases)
    return TreeReadPairs(
        plan=plan_all,
        basis=basis_all,
        log_m=torch.cat(logms),
        omega=walk.omega,
        forests=walk.forests,
        src_offsets=offsets,
        n_cells=n,
        n_queries=nq,
        n_pairs_node=n_pairs_node,
        kernel=walk.kernel,
        src_log_m=torch.cat(walk.src_logm),
        basis_sink=fused_read.basis_sink(basis_all) if walk.kernel == "fused" else None,
    )


class _TreeKernelRead(nn.Module):
    """One block's read for the passive prediction points, through the forest.

    A softmax over each point's cells and nodes (:class:`TreeReadPairs`) with the
    logits ``K_h(iota_qj) + <q_h, k_jh> / sqrt(c) + log omega_j`` (ISLA's quadrature
    form: the log measure as bias, so it is a measure-weighted mean over the
    boundary), with ``K_h`` a linear combination of the shared kernel basis. Logits
    linear in ``log r`` or ``log(r / |d_q|)`` are power laws, so the read can learn any
    locality; there is no radius, falloff or neighbour count. The cells' keys and
    values are linear maps of their states; a node's are the measure-weighted means of
    its cells' at this depth
    (:func:`~physicsnemo.experimental.models.isla.segment_softmax.segment_softmax_read`,
    which keeps nothing per pair between forward and backward). ``forward(h_b, h_q,
    pairs)`` with the boundary and point states flattened over the batch returns the
    points' residual update; the output projection starts at zero, so the read starts
    as no update.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        h, w = KERNEL_READ_HEADS, KERNEL_READ_WIDTH
        self.norm_b = nn.LayerNorm(hidden)
        self.norm_q = nn.LayerNorm(hidden)
        self.value = nn.Linear(hidden, w)
        self.key = nn.Linear(hidden, h * KERNEL_READ_KEY, bias=False)
        self.query = nn.Linear(hidden, h * KERNEL_READ_KEY, bias=False)
        ### No bias: a per-head constant cancels in the softmax.
        self.kernel = nn.Linear(KERNEL_READ_BASIS, h, bias=False)
        self.out = nn.Linear(w, hidden, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, h_b, h_q, pairs: TreeReadPairs):
        return self.read(self.sources(h_b, pairs), h_q, pairs)

    def sources(self, h_b, pairs: TreeReadPairs | TreeReadWalk):
        """The virtual sources' keys and values at this depth, ``(sources, H KC + W)``,
        from the boundary states ``h_b``."""
        hb = self.norm_b(h_b)
        v_c, k_c = self.value(hb), self.key(hb)
        ### The virtual sources, case by case: the cells, then the forest's node means
        ### of the keys and values together (one scatter and one sweep per case).
        n = pairs.n_cells
        kv_parts = []
        for i, forest in enumerate(pairs.forests):
            sl = slice(i * n, (i + 1) * n)
            kv_c = torch.cat([k_c[sl], v_c[sl]], dim=-1)
            means = (
                fused_read.fused_node_means
                if pairs.kernel == "fused" and fused_read.node_means_supported(kv_c)
                else node_means
            )
            kv_parts += [kv_c, means(kv_c, forest, pairs.omega[sl])]
        return torch.cat(kv_parts)

    def read(self, kv_src, h_q, pairs: TreeReadPairs):
        """The points' residual update from the sources' keys and values (:meth:`sources`)."""
        nh, c, kc = KERNEL_READ_HEADS, KERNEL_READ_WIDTH // KERNEL_READ_HEADS, KERNEL_READ_KEY
        q = self.query(self.norm_q(h_q)).view(-1, nh, kc)
        k_src = kv_src[:, : nh * kc].reshape(-1, nh, kc)
        v_src = kv_src[:, nh * kc :].reshape(-1, nh, c)
        ### The pairs' kernel terms at this block: the shared basis through the block's
        ### kernel weights (a matmul; its gradient to the basis accumulates over blocks).
        if pairs.kernel == "fused" and fused_read.supported(pairs.basis, q, k_src, v_src):
            ### The same read with the kernel term computed inside it (fused_read.py), its
            ### output rounded where the reference's is: to h_q's dtype if that is narrow,
            ### else at the output projection's input under autocast.
            low = (torch.bfloat16, torch.float16)
            dev = h_q.device.type
            if h_q.dtype in low:
                out_dtype = h_q.dtype
            elif torch.is_autocast_enabled(dev) and torch.get_autocast_dtype(dev) in low:
                out_dtype = torch.get_autocast_dtype(dev)
            else:
                out_dtype = torch.float32
            read = fused_read.fused_kernel_read(
                pairs.basis,
                self.kernel.weight.to(pairs.basis.dtype),
                pairs.src_log_m,
                q,
                k_src,
                v_src,
                pairs.plan,
                scale=1.0 / math.sqrt(kc),
                sink=pairs.basis_sink,
                out_dtype=out_dtype,
            )
        else:
            ker = pairs.basis @ self.kernel.weight.t().to(pairs.basis.dtype)
            read = segment_softmax_read(
                ker, pairs.log_m, q, k_src, v_src, pairs.plan, scale=1.0 / math.sqrt(kc)
            )
        return self.out(read.reshape(-1, nh * c).to(h_q.dtype))
