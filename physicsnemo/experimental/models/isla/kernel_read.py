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

r"""ISLA's boundary kernel read: passive prediction points read individual boundary cells.

A GLOBE-style read for the read-only interior task (``query_independent=True``). At a
slice block, each prediction point :math:`q` sums a learned kernel over every boundary
cell :math:`j` within a radius :math:`R` (in units of the length unit :math:`L`),
weighted by the cell's share of the boundary measure and a smooth compact falloff:

.. math::

    \omega_j = \mu_j / \textstyle\sum_i \mu_i, \qquad
    b_{qj} = (1 - s_{qj}^2)^3_+, \quad s_{qj} = |x_q - y_j| / (R L).

``"sum"`` (GLOBE-like, an unnormalized quadrature):
:math:`\text{read}_{qh} = e^{\gamma_h} \sum_j \omega_j b_{qj} K_h(\iota_{qj}) v_{jh}`.
``"mean"`` (ISLA-like, a softmax with the log measure as bias and a null of learned
mass :math:`e^{\beta_h}`, a fraction of the whole boundary measure, carrying value zero):
:math:`\text{read}_{qh} = \sum_j a_{qjh} v_{jh}`,
:math:`a_{qjh} \propto \omega_j b_{qj} \exp(K_h(\iota_{qj}) + \langle q_h, k_{jh}\rangle / \sqrt{c})`
normalized against :math:`e^{\beta_h} + \sum_j (\cdot)`. Either way the read vanishes
smoothly as a point leaves the boundary's neighbourhood, and a point with no cell
within :math:`R` reads exactly zero (the slices carry the far field).

The kernel inputs :math:`\iota_{qj}` are invariants of the pair's relative geometry
(:func:`pair_invariants`); one MLP maps them to a basis of ``KERNEL_READ_BASIS``
functions per pair once per forward pass (:class:`_KernelBasis`), and each block's
kernel :math:`K_h` is a learned linear combination of that basis per head. The values :math:`v_j`, keys :math:`k_j` and queries
:math:`q` are linear maps of invariant token states (the boundary cell's and the point's
own), so the read is invariant and the equivariant heads assemble every output. The
neighbourhood is the exact radius set, found by :func:`boundary_pairs` (never a fixed
neighbour count), so splitting a cell into coincident copies of half the measure, a
uniform rescale of the measures and a similarity of the geometry change nothing, and a
prediction depends on its own point and the boundary only.
"""

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from physicsnemo.experimental.models.isla.pair_kernels import (
    pair_scores,
    pair_weighted_sum,
)

#: Heads of the kernel read; each head owns ``KERNEL_READ_WIDTH // KERNEL_READ_HEADS``
#: value channels.
KERNEL_READ_HEADS = 8
#: Value channels of the kernel read (all heads together).
KERNEL_READ_WIDTH = 64
#: Query/key channels per head of the ``"mean"`` read.
KERNEL_READ_KEY = 4
#: Hidden width of the shared kernel MLP on the pair invariants.
KERNEL_READ_HIDDEN = 64
#: Kernel basis functions of the pair invariants, shared by every block; each block's
#: kernel is a learned linear combination of them per head.
KERNEL_READ_BASIS = 16
#: Query rows per chunk of the exact (torch) radius search.
_EXACT_SEARCH_CHUNK = 4096


def kernel_geo_width(
    n_global_vectors: int, n_query_vecs: int, n_boundary_vecs: int
) -> int:
    """Width of :func:`pair_invariants`: s, log s, rho.n_j, rho.n_q, n_q.n_j, then
    rho.g_k and n_j.g_k (2K), rho.v and n_j.v per query vector (2P_q), rho.u and n_q.u
    per boundary vector (2P_b)."""
    return (
        5 + 2 * int(n_global_vectors) + 2 * int(n_query_vecs) + 2 * int(n_boundary_vecs)
    )


def _exact_radius_pairs(q: torch.Tensor, y: torch.Tensor, radius: float):
    """All (query, cell) pairs with distance below ``radius``: a chunked cdist in the
    input precision, any dimension. Pairs come grouped by query."""
    qi, bj = [], []
    for start in range(0, q.shape[0], _EXACT_SEARCH_CHUNK):
        d = torch.cdist(
            q[start : start + _EXACT_SEARCH_CHUNK],
            y,
            compute_mode="donot_use_mm_for_euclid_dist",
        )
        i, j = (d < radius).nonzero(as_tuple=True)
        qi.append(i + start)
        bj.append(j)
    return torch.cat(qi), torch.cat(bj)


def boundary_pairs(
    query_r: torch.Tensor, boundary_r: torch.Tensor, radius: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Every (prediction point, boundary cell) pair closer than ``radius``.

    ``query_r`` ``(B, Q, d)`` and ``boundary_r`` ``(B, N, d)`` in the frame's unit;
    returns flat int64 indices ``(qi, bj)`` into the ``B * Q`` points and the ``B * N``
    cells, grouped by point. The set is exact up to the search's precision: on CUDA in
    float32 or lower the library's hash-grid radius search
    (:func:`physicsnemo.nn.functional.neighbors.radius_search.radius_search`, Warp;
    a planar problem is padded with a zero coordinate), else an exact chunked cdist in
    the input precision. Membership can differ between the two only at the rim, where
    the falloff weight is of order ``(1 - s^2)^3 ~ 1e-20``; :func:`pair_invariants`
    drops every pair at or beyond the radius in the model's precision. Runs without
    gradient.
    """
    b, nq, d = query_r.shape
    n = boundary_r.shape[1]
    qis, bjs = [], []
    with torch.no_grad():
        for i in range(b):
            q, y = query_r[i].contiguous(), boundary_r[i].contiguous()
            if q.is_cuda and q.dtype != torch.float64 and d <= 3:
                from physicsnemo.nn.functional.neighbors.radius_search.radius_search import (
                    radius_search,
                )

                if d < 3:
                    q = torch.nn.functional.pad(q.float(), (0, 3 - d))
                    y = torch.nn.functional.pad(y.float(), (0, 3 - d))
                idx = radius_search(
                    y.float(), q.float(), float(radius), implementation="warp"
                )
                qi, bj = idx[0].long(), idx[1].long()
            else:
                qi, bj = _exact_radius_pairs(q, y, float(radius))
            qis.append(qi + i * nq)
            bjs.append(bj + i * n)
    return torch.cat(qis), torch.cat(bjs)


def pair_invariants(
    qi: torch.Tensor,
    bj: torch.Tensor,
    *,
    query_r: torch.Tensor,
    query_n: torch.Tensor,
    query_vecs: torch.Tensor | None,
    boundary_r: torch.Tensor,
    boundary_n: torch.Tensor,
    boundary_vecs: torch.Tensor | None,
    g_unit: torch.Tensor,
    radius: float,
    eps: float,
):
    r"""The kernel inputs of every pair and its log falloff weight.

    Token tensors are flattened over the batch: ``query_r``, ``query_n`` ``(B*Q, d)``,
    ``query_vecs`` ``(B*Q, P_q, d)`` (the point's own rank-1 inputs) or ``None``;
    ``boundary_r``, ``boundary_n`` ``(B*N, d)``, ``boundary_vecs`` ``(B*N, P_b, d)`` or
    ``None``; ``g_unit`` ``(B, K, d)``. With :math:`\rho = (x_q - y_j) / L`,
    :math:`s = |\rho| / R` and :math:`\hat\rho = \rho / |\rho|` the invariants are
    :math:`[s, \log s, \hat\rho\cdot n_j, \hat\rho\cdot n_q, n_q\cdot n_j,
    \hat\rho\cdot g_k, n_j\cdot g_k, \hat\rho\cdot v, n_j\cdot v, \hat\rho\cdot u,
    n_q\cdot u]` (``v`` the point's vectors, ``u`` the cell's), all functions of
    relative geometry, so rotations, reflections, translations and similarities leave
    them unchanged. Pairs at or beyond the radius in this precision are dropped.

    Returns ``(qi, bj, inv (P, G), log_b (P,))`` with ``log_b = 3 log(1 - s^2)``.
    """
    with torch.autocast(query_r.device.type, enabled=False):
        rel = query_r[qi] - boundary_r[bj]
        dist = rel.norm(dim=-1)
        s = dist / radius
        keep = s < 1.0
        if not bool(keep.all()):
            qi, bj, rel, dist, s = qi[keep], bj[keep], rel[keep], dist[keep], s[keep]
        rho = rel / dist.clamp_min(eps)[:, None]
        n_q, n_j = query_n[qi], boundary_n[bj]
        nq_ = query_r.shape[0] // g_unit.shape[0]
        g = g_unit[qi // nq_]  # (P, K, d)
        feats = [
            s,
            torch.log(s.clamp_min(1e-6)),
            (rho * n_j).sum(-1),
            (rho * n_q).sum(-1),
            (n_q * n_j).sum(-1),
        ]
        cols = [
            torch.stack(feats, -1),
            (rho[:, None] * g).sum(-1),
            (n_j[:, None] * g).sum(-1),
        ]
        if query_vecs is not None and query_vecs.shape[1]:
            v = query_vecs[qi]
            cols += [(rho[:, None] * v).sum(-1), (n_j[:, None] * v).sum(-1)]
        if boundary_vecs is not None and boundary_vecs.shape[1]:
            u = boundary_vecs[bj]
            cols += [(rho[:, None] * u).sum(-1), (n_q[:, None] * u).sum(-1)]
        inv = torch.cat(cols, -1)
        log_b = 3.0 * torch.log1p(-s.square())
    return qi, bj, inv, log_b


def reference_log_fraction(radius: float, dim: int) -> float:
    r"""Log of the falloff-weighted boundary fraction near a flat wall, for initialization.

    A point on a flat wall sees :math:`\int b\,dA = V_R / 4` of measure (3-D;
    :math:`V_R = \pi R^2`) or :math:`\int b\,dl = 16 R / 35` (2-D; ``V_R = 2R``) in
    units of :math:`L^{d-1}`, out of a whole boundary of reference measure
    :math:`A_0`: the sphere's (circle's), :math:`2\pi` (3-D) or :math:`\sqrt 2 \pi`
    (2-D) in units of its own :math:`L^{d-1}`. The ``"sum"`` read starts with
    :math:`\gamma = -\log` of this fraction and the ``"mean"`` read's null with
    :math:`\beta = \log` of it, so both start at order one at the wall.
    """
    if dim == 2:
        near, total = 16.0 * radius / 35.0, math.sqrt(2.0) * math.pi
    else:
        near, total = math.pi * radius**2 / 4.0, 2.0 * math.pi
    return math.log(near / total)


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


class _BoundaryKernelRead(nn.Module):
    """One block's kernel read for the passive prediction-point tokens.

    ``forward(h_b, h_q, pairs)`` with the boundary token states ``h_b`` ``(B*N, H)``,
    the point states ``h_q`` ``(B*Q, H)`` at the block's depth and ``pairs`` =
    ``(PairPlan, kernel basis (P, KERNEL_READ_BASIS), log weight (P,))`` returns the residual
    update ``(B*Q, H)`` of the points. The output projection starts at zero, so the
    model starts as the plain read-only model.
    """

    def __init__(self, hidden: int, mode: str, radius: float, dim: int) -> None:
        super().__init__()
        if mode not in ("mean", "sum"):
            raise ValueError(f"kernel_read must be 'mean' or 'sum', got {mode!r}")
        self.mode = mode
        #: Rebuild the (P, .) pair tensors in backward instead of storing them (about
        #: 100 to 200 bytes per pair and block stored otherwise; recomputing costs about
        #: a fifth of the read's time).
        self.checkpoint = False
        h, w = KERNEL_READ_HEADS, KERNEL_READ_WIDTH
        self.norm_b = nn.LayerNorm(hidden)
        self.value = nn.Linear(hidden, w)
        self.kernel = nn.Linear(KERNEL_READ_BASIS, h)
        log_f = reference_log_fraction(radius, dim)
        if mode == "mean":
            self.norm_q = nn.LayerNorm(hidden)
            self.key = nn.Linear(hidden, h * KERNEL_READ_KEY, bias=False)
            self.query = nn.Linear(hidden, h * KERNEL_READ_KEY, bias=False)
            self.null_logw = nn.Parameter(torch.full((h,), log_f))
        else:
            self.log_scale = nn.Parameter(torch.full((h,), -log_f))
        ### No bias: a point with no cell within the radius reads exactly zero.
        self.out = nn.Linear(w, hidden, bias=False)
        nn.init.zeros_(self.out.weight)

    def _read(self, h_b, h_q, basis, log_m, plan):
        nh, c = KERNEL_READ_HEADS, KERNEL_READ_WIDTH // KERNEL_READ_HEADS
        ### Weights and sums in at least float32 (float64 stays float64).
        acc = torch.promote_types(h_q.dtype, torch.float32)
        hb = self.norm_b(h_b)
        v = self.value(hb).view(-1, nh, c)
        ker = self.kernel(basis).to(acc)  # (P, heads)
        if self.mode == "mean":
            k = self.key(hb).view(-1, nh, KERNEL_READ_KEY)
            q = self.query(self.norm_q(h_q)).view(-1, nh, KERNEL_READ_KEY)
            score = pair_scores(q, k, plan, acc) / math.sqrt(KERNEL_READ_KEY)
            logit = ker + score + log_m.to(acc)[:, None]
            null = self.null_logw.to(acc)[None].expand(plan.n_q, nh)
            ### A softmax over each point's pairs and the null: the shift is the
            ### larger of the null and the point's largest logit (detached; the
            ### softmax does not depend on it).
            counts = plan.q_offsets[1:] - plan.q_offsets[:-1]
            seg_max = torch.segment_reduce(
                logit.detach(), "max", lengths=counts, axis=0, unsafe=True
            )
            m = torch.maximum(null.detach(), seg_max)
            e = torch.exp(logit - m[plan.qi])
            ones = e.new_ones(plan.n_b, nh, 1)
            den = torch.exp(null - m) + pair_weighted_sum(e, ones, plan)[..., 0]
            read = (
                pair_weighted_sum(e, v, plan) / den[..., None]
            )  # (n_q, heads, c) in acc
        else:
            wgt = ker * torch.exp(self.log_scale.to(acc)[None] + log_m.to(acc)[:, None])
            read = pair_weighted_sum(wgt, v, plan)
        return self.out(read.view(-1, nh * c).to(h_q.dtype))

    def forward(self, h_b, h_q, pairs):
        plan, basis, log_m = pairs
        if (
            self.checkpoint
            and torch.is_grad_enabled()
            and (h_b.requires_grad or h_q.requires_grad)
        ):
            ### The (P, .) pair tensors are rebuilt in backward, not stored.
            return checkpoint(
                self._read, h_b, h_q, basis, log_m, plan, use_reentrant=False
            )
        return self._read(h_b, h_q, basis, log_m, plan)


# ---------------------------------------------------------------------------
# The global read (``kernel_read_radius=None``): every point reads every cell.
# ---------------------------------------------------------------------------

#: Pairs per dense tile of the global read: the tile's query rows are
#: ``max(1, GLOBAL_READ_TILE_PAIRS // N)`` for ``N`` boundary cells. A memory knob
#: only: every query row is computed alone (the softmax is over the row's cells), so
#: the tiling changes nothing but roundoff.
GLOBAL_READ_TILE_PAIRS = 1 << 24


def global_geo_width(
    n_global_vectors: int, n_query_vecs: int, n_boundary_vecs: int
) -> int:
    """Width of :func:`global_pair_invariants`: :func:`kernel_geo_width`'s invariants
    with ``s = r / L`` and ``log s`` and the near-wall invariant ``log(r / |d_q|)``."""
    return 1 + kernel_geo_width(n_global_vectors, n_query_vecs, n_boundary_vecs)


def global_pair_invariants(
    query_r: torch.Tensor,
    query_n: torch.Tensor,
    query_wall: torch.Tensor,
    query_vecs: torch.Tensor | None,
    boundary_r: torch.Tensor,
    boundary_n: torch.Tensor,
    boundary_vecs: torch.Tensor | None,
    g_unit: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    r"""The kernel inputs of every (point, cell) pair of a tile, ``(Q, N, G)``.

    One case: ``query_r``, ``query_n`` ``(Q, d)``, ``query_wall`` ``(Q,)`` the point's
    built-in wall distance :math:`|d_q|` (frame units), ``query_vecs`` ``(Q, P_q, d)``
    or ``None``; ``boundary_r``, ``boundary_n`` ``(N, d)``, ``boundary_vecs``
    ``(N, P_b, d)`` or ``None``; ``g_unit`` ``(K, d)``. With
    :math:`\rho = (x_q - y_j) / L`, :math:`r = |\rho|` and :math:`\hat\rho = \rho / r`
    the invariants are :math:`[r, \log r, \log(r / |d_q|), \hat\rho\cdot n_j,
    \hat\rho\cdot n_q, n_q\cdot n_j, \hat\rho\cdot g_k, n_j\cdot g_k, \hat\rho\cdot v,
    n_j\cdot v, \hat\rho\cdot u, n_q\cdot u]`: lengths in units of :math:`L` or of
    the point's own wall distance, and cosines, so no length scale is chosen. The
    only constant is ``eps``, a floor on the lengths inside the logarithms and the
    normalization (a numerical guard: :math:`r \ge |d_q| > 0` for every point off
    the boundary sample, since :math:`|d_q|` is the distance to the nearest cell).
    """
    with torch.autocast(query_r.device.type, enabled=False):
        rel = query_r[:, None] - boundary_r[None]  # (Q, N, d)
        dist = rel.norm(dim=-1)
        dist_g = dist.clamp_min(eps)
        rho = rel / dist_g[..., None]
        nq, nb = query_r.shape[0], boundary_r.shape[0]
        n_q = query_n[:, None].expand(nq, nb, -1)
        n_j = boundary_n[None].expand(nq, nb, -1)
        log_r = torch.log(dist_g)
        feats = [
            dist,
            log_r,
            log_r - torch.log(query_wall.clamp_min(eps))[:, None],
            (rho * n_j).sum(-1),
            (rho * n_q).sum(-1),
            (n_q * n_j).sum(-1),
        ]
        cols = [
            torch.stack(feats, -1),
            torch.einsum("qnd,kd->qnk", rho, g_unit),
            torch.einsum("nd,kd->nk", boundary_n, g_unit)[None].expand(nq, nb, -1),
        ]
        if query_vecs is not None and query_vecs.shape[1]:
            cols += [
                torch.einsum("qnd,qpd->qnp", rho, query_vecs),
                torch.einsum("nd,qpd->qnp", boundary_n, query_vecs),
            ]
        if boundary_vecs is not None and boundary_vecs.shape[1]:
            cols += [
                torch.einsum("qnd,npd->qnp", rho, boundary_vecs),
                torch.einsum("qd,npd->qnp", query_n, boundary_vecs),
            ]
        return torch.cat(cols, -1)


def global_tile_rows(n_cells: int) -> int:
    """Query rows per dense tile for a boundary of ``n_cells`` cells."""
    return max(1, GLOBAL_READ_TILE_PAIRS // max(1, int(n_cells)))


def global_kernel_basis(basis: "_KernelBasis", geometry: dict) -> torch.Tensor:
    """The kernel basis of every (point, cell) pair of one case, ``(BASIS, Q * N)``.

    Evaluated tile by tile over the query rows; with gradient enabled each tile is
    checkpointed (its invariants and the MLP's activations are rebuilt in backward),
    so only the basis itself, ``KERNEL_READ_BASIS`` numbers per pair, is stored.
    ``geometry`` holds :func:`global_pair_invariants`' arguments.
    """
    q_keys = ("query_r", "query_n", "query_wall", "query_vecs")
    nq, nb = geometry["query_r"].shape[0], geometry["boundary_r"].shape[0]
    rows = global_tile_rows(nb)

    def tile(*q_parts):
        inv = global_pair_invariants(
            **dict(zip(q_keys, q_parts)),
            **{k: v for k, v in geometry.items() if k not in q_keys},
        )
        return basis(inv).reshape(-1, KERNEL_READ_BASIS).t()

    grad = torch.is_grad_enabled() and any(p.requires_grad for p in basis.parameters())
    tiles = []
    for a in range(0, nq, rows):
        q_parts = [
            None if geometry[k] is None else geometry[k][a : a + rows] for k in q_keys
        ]
        if grad:
            tiles.append(checkpoint(tile, *q_parts, use_reentrant=False))
        else:
            tiles.append(tile(*q_parts))
    return torch.cat(tiles, dim=1)


class _GlobalSoftmaxRead(torch.autograd.Function):
    r"""The global read of one case, ``(H, Q, C)``: for every point, a softmax over
    every boundary cell of ``w_h . phi_qj + <q_qh, k_jh> * scale + log omega_j``
    weighting the cells' values.

    ``phi`` ``(BASIS, Q * N)`` the kernel basis; ``w`` ``(H, BASIS)``; ``q``
    ``(H, Q, c)``; ``k`` ``(H, N, c)``; ``v`` ``(H, N, C)``; ``log_frac`` ``(N,)``.
    Dense tiles of query rows, computed in at least float32 with autocast off; no
    ``(pairs, heads)`` tensor is stored: backward recomputes each tile's softmax
    (as flash attention does) and accumulates the gradients tile by tile.
    """

    @staticmethod
    def _tile(phi, w, q, k_t, log_frac, a, b, n, scale, acc):
        ph = phi[:, a * n : b * n].to(acc)
        z = (w @ ph).view(w.shape[0], b - a, n) + log_frac
        logit = torch.baddbmm(z, q[:, a:b], k_t, alpha=scale)
        return ph, torch.softmax(logit, dim=-1)

    @staticmethod
    def forward(ctx, phi, w, q, k, v, log_frac, scale):
        dev = q.device.type
        with torch.autocast(dev, enabled=False):
            acc = torch.promote_types(q.dtype, torch.float32)
            w_, q_, k_, v_ = (t.to(acc) for t in (w, q, k, v))
            lf = log_frac.to(acc)
            nh, nq, n = q.shape[0], q.shape[1], k.shape[1]
            k_t = k_.transpose(1, 2)
            out = q_.new_empty(nh, nq, v.shape[-1])
            rows = global_tile_rows(n)
            for a in range(0, nq, rows):
                b = min(nq, a + rows)
                _, att = _GlobalSoftmaxRead._tile(
                    phi, w_, q_, k_t, lf, a, b, n, scale, acc
                )
                out[:, a:b] = torch.bmm(att, v_)
        ctx.save_for_backward(phi, w, q, k, v, log_frac, out)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, d_out):
        phi, w, q, k, v, log_frac, out = ctx.saved_tensors
        scale = ctx.scale
        need = ctx.needs_input_grad
        with torch.autocast(q.device.type, enabled=False):
            acc = out.dtype
            w_, q_, k_, v_ = (t.to(acc) for t in (w, q, k, v))
            lf = log_frac.to(acc)
            d_out = d_out.to(acc)
            nh, nq, n = q.shape[0], q.shape[1], k.shape[1]
            k_t, v_t = k_.transpose(1, 2), v_.transpose(1, 2)
            d_phi = torch.empty_like(phi) if need[0] else None
            d_w = torch.zeros_like(w_)
            d_q = torch.empty_like(q_)
            d_k = torch.zeros_like(k_)
            d_v = torch.zeros_like(v_)
            d_lf = torch.zeros_like(lf)
            ### D = sum_c dO * O, the softmax's row correction.
            dd = (d_out * out).sum(-1, keepdim=True)
            rows = global_tile_rows(n)
            for a in range(0, nq, rows):
                b = min(nq, a + rows)
                ph, att = _GlobalSoftmaxRead._tile(
                    phi, w_, q_, k_t, lf, a, b, n, scale, acc
                )
                do = d_out[:, a:b]
                d_v += torch.bmm(att.transpose(1, 2), do)
                ### dlogit = A * (dO V^T - D), in place over the tile.
                dl = torch.bmm(do, v_t).sub_(dd[:, a:b]).mul_(att)
                del att
                d_q[:, a:b] = torch.bmm(dl, k_) * scale
                d_k += torch.bmm(dl.transpose(1, 2), q_[:, a:b]) * scale
                if need[5]:
                    d_lf += dl.sum((0, 1))
                dl2 = dl.view(nh, (b - a) * n)
                d_w += dl2 @ ph.t()
                if d_phi is not None:
                    d_phi[:, a * n : b * n] = (w_.t() @ dl2).to(phi.dtype)
        return (
            d_phi,
            d_w.to(w.dtype) if need[1] else None,
            d_q.to(q.dtype) if need[2] else None,
            d_k.to(k.dtype) if need[3] else None,
            d_v.to(v.dtype) if need[4] else None,
            d_lf.to(log_frac.dtype) if need[5] else None,
            None,
        )


def global_softmax_read_reference(phi, w, q, k, v, log_frac, scale):
    """Dense eager reference of :class:`_GlobalSoftmaxRead` (tests)."""
    nh, nq, n = q.shape[0], q.shape[1], k.shape[1]
    z = (w @ phi).view(nh, nq, n) + log_frac
    att = torch.softmax(z + scale * q @ k.transpose(1, 2), dim=-1)
    return att @ v


class _GlobalKernelRead(nn.Module):
    """One block's global read: every prediction point reads every boundary cell.

    A softmax over the boundary cells with the logits ``K_h(iota_qj) + <q_h, k_jh> /
    sqrt(c) + log omega_j`` (ISLA's quadrature form: the log measure fraction as
    bias, so it is a measure-weighted mean over the boundary), with ``K_h`` a linear
    combination of the shared kernel basis. Logits linear in ``log r`` or
    ``log(r / |d_q|)`` are power laws, so the read can learn any locality; there is no
    radius, falloff or neighbour count. ``forward(h_b, h_q, pairs)`` with the
    boundary and point states flattened over the batch and ``pairs = (basis per case
    (BASIS, Q*N), log fraction (B, N))`` returns the points' residual update; the
    output projection starts at zero (the model starts as the plain read-only
    model).
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

    def forward(self, h_b, h_q, pairs):
        basis, log_frac = pairs
        b, n = log_frac.shape
        nh, c, kc = (
            KERNEL_READ_HEADS,
            KERNEL_READ_WIDTH // KERNEL_READ_HEADS,
            KERNEL_READ_KEY,
        )
        hb = self.norm_b(h_b)
        v = self.value(hb).view(b, n, nh, c).permute(0, 2, 1, 3)
        k = self.key(hb).view(b, n, nh, kc).permute(0, 2, 1, 3)
        q = self.query(self.norm_q(h_q)).view(b, -1, nh, kc).permute(0, 2, 1, 3)
        reads = [
            _GlobalSoftmaxRead.apply(
                basis[i],
                self.kernel.weight,
                q[i],
                k[i],
                v[i],
                log_frac[i],
                1.0 / math.sqrt(kc),
            )
            for i in range(b)
        ]
        read = torch.stack(reads).permute(0, 2, 1, 3).reshape(-1, nh * c)
        return self.out(read.to(h_q.dtype))
