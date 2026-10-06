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

r"""ISLA's tree-accelerated global boundary read (Barnes-Hut over the boundary cells).

The dense global read (:mod:`~physicsnemo.experimental.models.isla.kernel_read`,
``kernel_read_radius=None``) has every passive prediction point :math:`q` read every
boundary cell :math:`j`: a softmax over the cells of

.. math::

    z_{qjh} = K_h(\iota_{qj}) + \langle q_{qh}, k_{jh} \rangle / \sqrt{c} + \log \omega_j

weighting the cells' values :math:`v_{jh}`, with :math:`\omega_j` the cell's share of
the boundary measure, so the read is a measure-weighted mean over the boundary (a
quadrature). Its cost is the number of (point, cell) pairs, :math:`Q N` per block:
1e8 on an airfoil of a thousand cells, 4e10 on a car of 200,000.

This module evaluates the same read through a cluster tree over the boundary cells
(:class:`~physicsnemo.experimental.models.isla.boundary_tree.BoundaryTree`, balanced,
split along each cluster's principal axis of normals while its normals are spread
wider than :math:`\theta` and of positions otherwise, with bounding spheres), walked
once per prediction point as an N-body code walks its tree once per body. A point
reads the cells of nearby leaves one by one, and a distant cluster :math:`C` through
one *virtual cell*: the cluster's measure-weighted centroid, its measure-weighted mean
normal (renormalized), its total measure :math:`\Omega_C = \sum_{j \in C} \omega_j`,
and the measure-weighted means of its cells' keys and values at the block's depth. A
cluster is admitted when its angular size and its normal spread are both below
:math:`\theta`: :math:`2 R_C < \theta\, r` (the diameter of its bounding sphere
against the gap between the point and that sphere) and the largest chord between one
of its normals and its mean normal below :math:`\theta`; a leaf that is not admitted
contributes its cells; any other node hands the point to its two children. Because
the read is a quadrature, the virtual
cell is the cluster's monopole: were the pair logit constant over the cluster,
:math:`\sum_{j \in C} \omega_j e^{z_{qj}} v_j` would equal
:math:`\Omega_C e^{z_{qC}} v_C` exactly, and the error at finite :math:`\theta` is the
variation of the pair invariants, keys and values over the admitted clusters. At
:math:`\theta = 0` no cluster is admitted and the read is the dense read, pair for
pair. :math:`\theta` is therefore a numerical tolerance with a convergence guarantee,
not a model constant: the model is *defined* as the dense read and the tree is how it
is computed.

Symmetries. A point's list depends on the boundary tree and the point alone, so the
read is exactly query independent at every :math:`\theta` (a point's prediction
depends on the boundary and the point, never on the other prediction points). At
:math:`\theta = 0` every contract of the dense read holds exactly: Euclidean
equivariance, the similarity gauge, boundary-refinement and measure-scale invariance.
At finite :math:`\theta` the tree is built from the clusters' own principal axes and
bounding spheres, so a rotation, reflection, translation or rescale of the case
reproduces the same clusters and the same lists up to ties (a cluster whose two
largest spreads are equal, or cells with equal projections); where a tie breaks
differently the differences are of the approximation's order and vanish with
:math:`\theta`.

Constants. ``TREE_READ_LEAF_SIZE`` (cells per leaf) and ``TREE_READ_PAIR_CHUNK``
(pairs per chunk of the basis evaluation) are implementation constants: at a fixed
:math:`\theta` they change the cost and the grouping of the exact pairs, and at
:math:`\theta = 0` nothing at all.

Two earlier forms are not used. The dual-tree form (GLOBE's
:meth:`~physicsnemo.mesh.spatial.cluster_tree.ClusterTree.find_dual_interaction_pairs`
with a tree over the points as well, the symmetric criterion
:math:`(D_T + D_S) / r < \theta`, far-field entries expanded to their points): a leaf
of points that spans a large box in a sparse region of the volume is never well
separated from the boundary leaves near it, and reads the whole boundary cell by cell
(1.6e9 pairs at :math:`\theta = 0.5` on a 200,000-cell car with 200,000 points, out of
memory on a 277 GiB device). The per-point walk of GLOBE's Morton tree with its
axis-aligned boxes: its count-midpoint nodes straddle Morton boundaries and their boxes
are loose for a surface (1,700 sources per point at :math:`\theta = 1` on the same
car, 3.4e8 pairs per block).
"""

import math
from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint

from physicsnemo.experimental.models.isla.boundary_tree import BoundaryTree
from physicsnemo.experimental.models.isla.kernel_read import (
    KERNEL_READ_BASIS,
    KERNEL_READ_HEADS,
    KERNEL_READ_KEY,
    KERNEL_READ_WIDTH,
    _GlobalKernelRead,
    _KernelBasis,
)
from physicsnemo.experimental.models.isla.pair_kernels import (
    PairPlan,
    pair_scores,
    pair_weighted_sum,
)
from physicsnemo.mesh.spatial._ragged import _ragged_arange

#: Boundary cells per tree leaf. Implementation: at a fixed ``theta`` it sets how many
#: cells an opened leaf contributes; at ``theta = 0`` it changes nothing.
TREE_READ_LEAF_SIZE = 8
#: Pairs per chunk of the kernel-basis evaluation (memory only; each chunk's
#: invariants and MLP activations are rebuilt in backward).
TREE_READ_PAIR_CHUNK = 1 << 22


@dataclass
class TreeReadCase:
    """One case's tree: what the per-block aggregation of the cells' keys and values
    over every node needs, plus the pair counts for diagnostics."""

    n_cells: int
    n_nodes: int
    order: torch.Tensor  # (N,) the cells in tree order
    node_start: torch.Tensor  # (M,) each node's first cell in that order
    node_count: torch.Tensor  # (M,) each node's cell count
    node_measure: torch.Tensor  # (M,) each node's total measure fraction
    n_pairs_cell: int  # (point, cell) pairs
    n_pairs_node: int  # (point, node) pairs


@dataclass
class TreeReadPairs:
    """The tree read's pairs of one forward pass, every case flattened: ``plan`` over
    the ``B * Q`` points and the virtual sources (each case's ``N`` cells followed by
    its ``M`` nodes), the kernel basis and log measure of every pair, the cells' measure
    fractions, and each case's tree."""

    plan: PairPlan
    basis: torch.Tensor  # (P, KERNEL_READ_BASIS)
    log_m: torch.Tensor  # (P,)
    omega: torch.Tensor  # (B * N,) the cells' measure fractions
    cases: list[TreeReadCase]
    src_offsets: list[int]
    n_cells: int
    n_queries: int

    @property
    def n_pairs(self) -> int:
        return int(self.plan.qi.shape[0])

    @property
    def n_pairs_node(self) -> int:
        return sum(c.n_pairs_node for c in self.cases)


def point_tree_pairs(
    tree: BoundaryTree, points: torch.Tensor, theta: float
) -> tuple[torch.Tensor, torch.Tensor, int]:
    r"""Each point's Barnes-Hut list over the boundary tree.

    Returns ``(qi, si, n_node_pairs)``: the pairs sorted by point, ``si`` a cell index
    (below the cell count) or the cell count plus a node index. The walk is level
    synchronous from the root: a node whose bounding sphere (radius :math:`R`)
    satisfies :math:`2 R < \theta r`, with :math:`r` the gap between the point and the
    sphere, and whose normal spread is below :math:`\theta`, is admitted as one virtual
    cell; a leaf that is not admitted contributes its cells; any other node hands the
    point to its two children. ``theta = 0`` admits nothing, so the list is every cell.
    A point's list depends on the tree and the point alone.
    """
    n = tree.n_cells
    device = points.device
    theta = float(theta)
    first_leaf = tree.first_leaf
    coherent = tree.spread < theta  # (M,) clusters whose normals are within the tolerance
    pq = torch.arange(points.shape[0], device=device)
    pn = torch.zeros_like(pq)
    out_q, out_s = [], []
    n_node_pairs = 0
    while pq.numel():
        radius = tree.radius[pn]
        gap = ((points[pq] - tree.center[pn]).norm(dim=-1) - radius).clamp_min(0)
        if theta > 0:
            admit = (2 * radius < theta * gap) & coherent[pn]
        else:
            admit = torch.zeros_like(gap, dtype=torch.bool)
        leaf = pn >= first_leaf
        q_adm, n_adm = pq[admit], pn[admit]
        out_q.append(q_adm)
        out_s.append(n_adm + n)
        n_node_pairs += int(q_adm.shape[0])
        opened = leaf & ~admit
        q_leaf, n_leaf = pq[opened], pn[opened]
        pos, pid = _ragged_arange(tree.node_start[n_leaf], tree.node_count[n_leaf])
        out_q.append(q_leaf[pid])
        out_s.append(tree.order[pos])
        down = ~leaf & ~admit
        q_down, n_down = pq[down], pn[down]
        pq = torch.cat([q_down, q_down])
        pn = torch.cat([2 * n_down + 1, 2 * n_down + 2])
    qi, si = torch.cat(out_q), torch.cat(out_s)
    order = torch.argsort(qi, stable=True)
    return qi[order], si[order], n_node_pairs


def node_means(x: torch.Tensor, case: TreeReadCase, omega: torch.Tensor) -> torch.Tensor:
    """The measure-weighted mean of the cells' rows ``x`` ``(N, ...)`` over every node's
    subtree, ``(M, ...)``: prefix sums over the tree order in float64 (float32 cancels
    at a few hundred thousand cells), differentiable in ``x``, O(N) per feature."""
    flat = x.reshape(case.n_cells, -1)
    weighted = (flat[case.order] * omega[case.order, None]).double()
    prefix = torch.nn.functional.pad(torch.cumsum(weighted, dim=0), (0, 0, 1, 0))
    sums = prefix[case.node_start + case.node_count] - prefix[case.node_start]
    measure = case.node_measure.double().clamp_min(torch.finfo(torch.float64).tiny)
    return (sums / measure[:, None]).to(x.dtype).reshape(case.n_nodes, *x.shape[1:])


def tree_pair_invariants(
    qi: torch.Tensor,
    si: torch.Tensor,
    *,
    query_r: torch.Tensor,
    query_n: torch.Tensor,
    query_wall: torch.Tensor,
    query_vecs: torch.Tensor | None,
    src_r: torch.Tensor,
    src_n: torch.Tensor,
    src_vecs: torch.Tensor | None,
    g_unit: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    r"""The kernel inputs of the listed (point, source) pairs, ``(P, G)``: the dense
    read's :func:`~physicsnemo.experimental.models.isla.kernel_read.global_pair_invariants`
    column for column (``r``, ``log r``, ``log(r / |d_q|)``, the cosines with the
    normals, the global directions and the rank-1 inputs), for one case, with the
    sources either cells or virtual cells."""
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
        if query_vecs is not None and query_vecs.shape[1]:
            v = query_vecs[qi]
            cols += [(rho[:, None] * v).sum(-1), (n_j[:, None] * v).sum(-1)]
        if src_vecs is not None and src_vecs.shape[1]:
            u = src_vecs[si]
            cols += [(rho[:, None] * u).sum(-1), (n_q[:, None] * u).sum(-1)]
        return torch.cat(cols, -1)


def _chunked_basis(basis: _KernelBasis, qi, si, geometry: dict) -> torch.Tensor:
    """The kernel basis of every pair ``(P, KERNEL_READ_BASIS)`` in chunks of
    ``TREE_READ_PAIR_CHUNK`` pairs; under gradient each chunk is checkpointed."""
    grad = torch.is_grad_enabled() and any(p.requires_grad for p in basis.parameters())

    def chunk(q_idx, s_idx):
        return basis(tree_pair_invariants(q_idx, s_idx, **geometry))

    parts = []
    for a in range(0, int(qi.shape[0]), TREE_READ_PAIR_CHUNK):
        args = (qi[a : a + TREE_READ_PAIR_CHUNK], si[a : a + TREE_READ_PAIR_CHUNK])
        parts.append(checkpoint(chunk, *args, use_reentrant=False) if grad else chunk(*args))
    if not parts:
        return geometry["query_r"].new_zeros(0, KERNEL_READ_BASIS)
    return torch.cat(parts)


def tree_read_pairs(
    basis: _KernelBasis,
    *,
    query_r: torch.Tensor,
    query_n: torch.Tensor,
    query_wall: torch.Tensor,
    query_vecs: torch.Tensor | None,
    boundary_r: torch.Tensor,
    boundary_n: torch.Tensor,
    boundary_vecs: torch.Tensor | None,
    log_frac: torch.Tensor,
    g_unit: torch.Tensor,
    theta: float,
    eps: float,
) -> TreeReadPairs:
    r"""Build the tree read's pairs for a batch of cases.

    ``query_r``, ``query_n`` ``(B, Q, d)`` (frame units; the point's built-in
    direction), ``query_wall`` ``(B, Q)`` the point's built-in wall distance,
    ``query_vecs`` ``(B, Q, P_q, d)`` or ``None``; ``boundary_r``, ``boundary_n``
    ``(B, N, d)``, ``boundary_vecs`` ``(B, N, P_b, d)`` or ``None``; ``log_frac``
    ``(B, N)`` the cells' log measure fractions; ``g_unit`` ``(B, K, d)``.

    Per case: a cluster tree over the cells (:class:`BoundaryTree`, with the measure
    fractions as the measure), each point's walk of it at ``theta``
    (:func:`point_tree_pairs`), so each point's list is a set of cells and a set of
    nodes that together cover every cell exactly once; the nodes' virtual cells
    (centroid, renormalized mean normal, mean rank-1 inputs, total measure) come with
    the tree; the kernel basis of every pair in chunks. The tree and the walk run
    without gradient (a numerical device); the invariants and the basis carry
    gradients to the kernel MLP.
    """
    b, nq, d = query_r.shape
    n = boundary_r.shape[1]
    tiny = torch.finfo(torch.float32).tiny
    qis, sis, bases, logms, cases, offsets = [], [], [], [], [], []
    src_off = 0
    for i in range(b):
        with torch.no_grad():
            omega = torch.exp(log_frac[i])
            has_vecs = boundary_vecs is not None and bool(boundary_vecs.shape[2])
            tree_b = BoundaryTree.build(
                boundary_r[i],
                omega,
                boundary_n[i],
                boundary_vecs[i] if has_vecs else None,
                leaf_size=TREE_READ_LEAF_SIZE,
                theta=theta,
                eps=eps,
            )
            qi, si, n_pairs_node = point_tree_pairs(tree_b, query_r[i].detach(), theta)
            m = tree_b.n_nodes
            node_measure = tree_b.measure.to(log_frac.dtype)
        src_r = torch.cat([boundary_r[i], tree_b.center.to(boundary_r.dtype)])
        src_n = torch.cat([boundary_n[i], tree_b.normal.to(boundary_n.dtype)])
        src_vecs = (
            torch.cat([boundary_vecs[i], tree_b.vecs.to(boundary_vecs.dtype)])
            if has_vecs
            else None
        )
        src_logm = torch.cat([log_frac[i], torch.log(node_measure.clamp_min(tiny))])
        geometry = {
            "query_r": query_r[i],
            "query_n": query_n[i],
            "query_wall": query_wall[i],
            "query_vecs": None if query_vecs is None else query_vecs[i],
            "src_r": src_r,
            "src_n": src_n,
            "src_vecs": src_vecs,
            "g_unit": g_unit[i],
            "eps": eps,
        }
        bases.append(_chunked_basis(basis, qi, si, geometry))
        logms.append(src_logm[si])
        qis.append(qi + i * nq)
        sis.append(si + src_off)
        cases.append(
            TreeReadCase(
                n_cells=n,
                n_nodes=m,
                order=tree_b.order,
                node_start=tree_b.node_start,
                node_count=tree_b.node_count,
                node_measure=node_measure,
                n_pairs_cell=int(qi.shape[0]) - n_pairs_node,
                n_pairs_node=n_pairs_node,
            )
        )
        offsets.append(src_off)
        src_off += n + m
    plan_all = PairPlan.build(torch.cat(qis), torch.cat(sis), b * nq, src_off)
    return TreeReadPairs(
        plan=plan_all,
        basis=torch.cat(bases),
        log_m=torch.cat(logms),
        omega=torch.exp(log_frac).reshape(-1),
        cases=cases,
        src_offsets=offsets,
        n_cells=n,
        n_queries=nq,
    )


class _TreeKernelRead(_GlobalKernelRead):
    """One block's global read evaluated through the tree: the dense read's
    parameters (a dense checkpoint loads as it stands) applied to the pair list of
    :class:`TreeReadPairs`. The cells' keys and values are those of the dense read;
    a node's are the measure-weighted means of its cells' at this depth; the softmax
    runs over each point's cells and nodes with the log measure of each as bias. The
    ``(P, .)`` pair tensors are rebuilt in backward, not stored."""

    def forward(self, h_b, h_q, pairs):
        if torch.is_grad_enabled() and (h_b.requires_grad or h_q.requires_grad):
            return checkpoint(self._read, h_b, h_q, pairs, use_reentrant=False)
        return self._read(h_b, h_q, pairs)

    def _read(self, h_b, h_q, pairs: TreeReadPairs):
        nh, c, kc = KERNEL_READ_HEADS, KERNEL_READ_WIDTH // KERNEL_READ_HEADS, KERNEL_READ_KEY
        acc = torch.promote_types(h_q.dtype, torch.float32)
        hb = self.norm_b(h_b)
        v_c, k_c = self.value(hb), self.key(hb)
        q = self.query(self.norm_q(h_q)).view(-1, nh, kc)
        ### The virtual sources, case by case: the cells, then the nodes' means.
        n = pairs.n_cells
        v_parts, k_parts = [], []
        for i, case in enumerate(pairs.cases):
            sl = slice(i * n, (i + 1) * n)
            om = pairs.omega[sl]
            v_parts += [v_c[sl], node_means(v_c[sl], case, om)]
            k_parts += [k_c[sl], node_means(k_c[sl], case, om)]
        v_src = torch.cat(v_parts).view(-1, nh, c).to(acc)
        k_src = torch.cat(k_parts).view(-1, nh, kc)
        plan = pairs.plan
        ker = pairs.basis.to(acc) @ self.kernel.weight.to(acc).t()  # (P, heads)
        score = pair_scores(q, k_src, plan, acc) / math.sqrt(kc)
        logit = ker + score + pairs.log_m.to(acc)[:, None]
        ### A softmax over each point's pairs; the shift is the point's largest logit
        ### (detached: the softmax does not depend on it).
        counts = plan.q_offsets[1:] - plan.q_offsets[:-1]
        seg_max = torch.segment_reduce(logit.detach(), "max", lengths=counts, axis=0, unsafe=True)
        e = torch.exp(logit - seg_max[plan.qi])
        ones = e.new_ones(plan.n_b, nh, 1)
        den = pair_weighted_sum(e, ones, plan)[..., 0]
        read = pair_weighted_sum(e, v_src, plan) / den[..., None]
        return self.out(read.reshape(-1, nh * c).to(h_q.dtype))
