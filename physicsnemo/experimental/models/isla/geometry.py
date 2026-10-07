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
:func:`nearest_boundary_geometry` and :func:`soft_boundary_geometry` are the two
forms of the prediction points' built-in signed distance and direction to the
boundary (the exact nearest sample, and a measure-weighted kernel average over the
samples that converges under refinement of the boundary sample). All are built from
pairwise differences only, so they refer to no frame origin."""

import importlib
from typing import Literal

import torch
from jaxtyping import Float

from physicsnemo.core.version_check import is_package_available

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


### "nearest": the exact nearest sample; "soft": the one-sided kernel average
### (soft_boundary_geometry, one_sided=True); "smooth": the two-sided kernel
### average (one_sided=False). The two kernel forms are research arms of one
### candidate (2026-10-07); one of them replaces the other after the verdict.
QueryGeometry = Literal["nearest", "soft", "smooth"]


if HAS_TRITON:

    @triton.jit
    def _soft_geometry_kernel(
        q_ptr,
        y_ptr,
        n_ptr,
        lw_ptr,
        nstar_ptr,
        side_ptr,
        idx_ptr,
        iw_ptr,
        out_ptr,
        Q,
        N,
        DIM: tl.constexpr,
        ONE_SIDED: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """The accumulation pass of :func:`soft_boundary_geometry`: one program owns
        ``BLOCK_Q`` queries, each with its nearest sample's index (``idx``), normal
        (``nstar``) and side (``side``, +1 or -1), and streams the ``N`` boundary
        samples in ``BLOCK_N`` tiles, keeping the running kernel mass, plane-distance
        mass and normal mass of the samples that face the nearest sample's way (and,
        when ``ONE_SIDED``, lie on its side of their tangent planes), rescaled online
        as a streamed softmax; no (Q, N) block is ever written. The nearest sample is
        always kept: the host decided the side from its plane distance in its own
        rounding, and a query on the wall must not lose its own sample to the
        kernel's. ``DIM`` is 2 or 3; the inverse width is read from ``iw_ptr`` in the
        inputs' precision."""
        offs_q = tl.program_id(0) * BLOCK_Q + tl.arange(0, BLOCK_Q)
        mask_q = offs_q < Q
        dt = q_ptr.dtype.element_ty
        inf = float("inf")
        inv_width = tl.load(iw_ptr)
        q0 = tl.load(q_ptr + offs_q * DIM + 0, mask=mask_q, other=0.0)
        q1 = tl.load(q_ptr + offs_q * DIM + 1, mask=mask_q, other=0.0)
        ns0 = tl.load(nstar_ptr + offs_q * DIM + 0, mask=mask_q, other=0.0)
        ns1 = tl.load(nstar_ptr + offs_q * DIM + 1, mask=mask_q, other=0.0)
        if DIM == 3:
            q2 = tl.load(q_ptr + offs_q * DIM + 2, mask=mask_q, other=0.0)
            ns2 = tl.load(nstar_ptr + offs_q * DIM + 2, mask=mask_q, other=0.0)
        side = tl.load(side_ptr + offs_q, mask=mask_q, other=1.0)
        want_pos = side > 0
        idx_q = tl.load(idx_ptr + offs_q, mask=mask_q, other=-1)
        m = tl.full((BLOCK_Q,), -inf, dt)
        w = tl.zeros((BLOCK_Q,), dt)
        sacc = tl.zeros((BLOCK_Q,), dt)
        a0 = tl.zeros((BLOCK_Q,), dt)
        a1 = tl.zeros((BLOCK_Q,), dt)
        a2 = tl.zeros((BLOCK_Q,), dt)
        for n0 in range(0, N, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N
            y0 = tl.load(y_ptr + offs_n * DIM + 0, mask=mask_n, other=0.0)
            y1 = tl.load(y_ptr + offs_n * DIM + 1, mask=mask_n, other=0.0)
            nn0 = tl.load(n_ptr + offs_n * DIM + 0, mask=mask_n, other=0.0)
            nn1 = tl.load(n_ptr + offs_n * DIM + 1, mask=mask_n, other=0.0)
            diff0 = q0[:, None] - y0[None, :]
            diff1 = q1[:, None] - y1[None, :]
            d2 = diff0 * diff0 + diff1 * diff1
            s = diff0 * nn0[None, :] + diff1 * nn1[None, :]
            f = ns0[:, None] * nn0[None, :] + ns1[:, None] * nn1[None, :]
            if DIM == 3:
                y2 = tl.load(y_ptr + offs_n * DIM + 2, mask=mask_n, other=0.0)
                nn2 = tl.load(n_ptr + offs_n * DIM + 2, mask=mask_n, other=0.0)
                diff2 = q2[:, None] - y2[None, :]
                d2 += diff2 * diff2
                s += diff2 * nn2[None, :]
                f += ns2[:, None] * nn2[None, :]
            r = tl.sqrt(d2)
            lw = tl.load(lw_ptr + offs_n, mask=mask_n, other=-inf)
            keep = mask_n[None, :] & (f > 0)
            if ONE_SIDED:
                pos = s >= 0
                keep = keep & ((pos & want_pos[:, None]) | ((~pos) & (~want_pos[:, None])))
            keep = keep | (mask_n[None, :] & (offs_n[None, :] == idx_q[:, None]))
            f = tl.where(keep, f, 1.0)
            l = lw[None, :] - r * inv_width + tl.log(f)
            l = tl.where(keep, l, -inf)
            m_new = tl.maximum(m, tl.max(l, axis=1))
            ### Until a query's stream has met its first kept sample, m and m_new are
            ### both -inf and exp(m - m_new) would be NaN; the accumulators are zero
            ### there, so the rescale is 0. Likewise e is 0 for every excluded sample.
            live = m_new > -inf
            scale = tl.where(live, tl.exp(m - tl.where(live, m_new, 0.0)), 0.0)
            e = tl.where(keep, tl.exp(l - tl.where(live, m_new, 0.0)[:, None]), 0.0)
            w = w * scale + tl.sum(e, axis=1)
            sacc = sacc * scale + tl.sum(e * s, axis=1)
            a0 = a0 * scale + tl.sum(e * nn0[None, :], axis=1)
            a1 = a1 * scale + tl.sum(e * nn1[None, :], axis=1)
            if DIM == 3:
                a2 = a2 * scale + tl.sum(e * nn2[None, :], axis=1)
            m = m_new
        d = tl.where(w > 0, sacc / tl.where(w > 0, w, 1.0), 0.0)
        norm = tl.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
        inv = tl.where(norm > 0, 1.0 / tl.where(norm > 0, norm, 1.0), 0.0)
        tl.store(out_ptr + offs_q * (DIM + 1) + 0, d, mask=mask_q)
        tl.store(out_ptr + offs_q * (DIM + 1) + 1, a0 * inv, mask=mask_q)
        tl.store(out_ptr + offs_q * (DIM + 1) + 2, a1 * inv, mask=mask_q)
        if DIM == 3:
            tl.store(out_ptr + offs_q * (DIM + 1) + 3, a2 * inv, mask=mask_q)


### Queries per chunk of the torch soft geometry; bounds its (chunk, N, dim) blocks.
_TORCH_SOFT_CHUNK = 256


def _soft_geometry_torch(
    q: torch.Tensor,
    y: torch.Tensor,
    n: torch.Tensor,
    lw: torch.Tensor,
    idx: torch.Tensor,
    n_star: torch.Tensor,
    side: torch.Tensor,
    width: torch.Tensor,
    eps: float,
    one_sided: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The chunked PyTorch form of the accumulation pass of
    :func:`soft_boundary_geometry` for one case: ``q`` (Q, d), ``y``/``n`` (N, d),
    ``lw`` (N,), ``idx`` (Q,), ``n_star`` (Q, d) and ``side`` (Q,) from the nearest
    sample, ``width`` a scalar tensor."""
    inf = float("inf")
    ds, us = [], []
    ar = torch.arange(y.shape[0], device=y.device)
    for s0 in range(0, q.shape[0], _TORCH_SOFT_CHUNK):
        sl = slice(s0, s0 + _TORCH_SOFT_CHUNK)
        qc, nsc, sdc, ic = q[sl], n_star[sl], side[sl], idx[sl]
        diff = qc[:, None, :] - y[None, :, :]  # (c, N, d)
        r = diff.norm(dim=-1)  # (c, N)
        s = (diff * n[None]).sum(-1)  # signed distance to each sample's tangent plane
        f = (nsc[:, None, :] * n[None]).sum(-1)  # the sample's normal on the nearest sample's normal
        keep = f > 0
        if one_sided:
            keep = keep & ((s >= 0) == (sdc[:, None] > 0))
        keep = keep | (ar[None, :] == ic[:, None])  # the nearest sample is always kept
        l = lw[None, :] - r / width + torch.log(torch.where(keep, f, torch.ones_like(f)))
        l = torch.where(keep, l, torch.full_like(l, -inf))
        e = torch.exp(l - l.max(dim=1, keepdim=True).values)
        w = e.sum(dim=1)
        d = (e * s).sum(dim=1) / w
        nbar = (e[:, :, None] * n[None]).sum(dim=1)  # (c, d)
        u = nbar / nbar.norm(dim=-1, keepdim=True).clamp_min(eps)
        ds.append(d)
        us.append(u)
    return torch.cat(ds), torch.cat(us)


def soft_boundary_geometry(
    query_points: Float[torch.Tensor, "batch queries dim"],
    points: Float[torch.Tensor, "batch tokens dim"],
    normals: Float[torch.Tensor, "batch tokens dim"],
    log_measures: Float[torch.Tensor, "batch tokens"] | None = None,
    *,
    width: float | torch.Tensor,
    eps: float = 1e-12,
    one_sided: bool = True,
) -> tuple[
    Float[torch.Tensor, "batch queries"], Float[torch.Tensor, "batch queries dim"]
]:
    r"""Signed distance and direction to the boundary as a measure-weighted kernel
    average over the boundary samples, the sample-independent form of
    :func:`nearest_boundary_geometry`.

    For every query :math:`x_q` the exact nearest sample :math:`i^*` is found
    first (:func:`_nearest_index`); it fixes the local orientation
    :math:`\hat n_{i^*}` and, for the one-sided form, the side of the boundary
    the query is on, :math:`\sigma^* = \operatorname{sign}((x_q - y_{i^*})
    \cdot \hat n_{i^*})` (positive on the side the normals point to; a tangent
    or coincident query counts as positive). Every sample
    :math:`(y_i, \hat n_i, w_i)` with :math:`w_i` its measure then gets the
    weight

    .. math::

        a_i = w_i\, e^{-|x_q - y_i| / \tau}\, \max(0, \hat n_i \cdot \hat n_{i^*})
        \,[\operatorname{sign}(s_i) = \sigma^*]^{\text{one-sided}},
        \qquad s_i = (x_q - y_i) \cdot \hat n_i,

    with :math:`\tau` the ``width`` and :math:`s_i` the signed distance from the
    query to the sample's tangent plane (the nearest sample is always kept), and

    .. math::

        d_q = \frac{\sum_i a_i s_i}{\sum_i a_i}, \qquad
        u_q = \frac{\sum_i a_i \hat n_i}{\lvert \sum_i a_i \hat n_i \rvert}.

    :math:`d_q > 0` outside for outward normals, the stored signed-distance
    fields' convention and the nearest form's. :math:`u_q` is the kernel-mean
    normal, which is the gradient of a signed distance: it points outward on
    both sides of the wall and is continuous across it (the nearest form's
    direction points from the boundary to the query, so inside a body the two
    forms' directions are opposite). The side enters :math:`u_q` nowhere: a
    query on the wall itself (the volume meshes of the program's datasets
    carry the wall nodes among their interior points) has a side decided by
    rounding, and a direction that flipped with it made the trained model's
    error move by 1 to 6 percent under a rigid motion in float32 (measured
    2026-10-07; the nearest form's direction never flips there).

    The sums are Monte Carlo estimates of integrals over the boundary with the
    samples' measures as quadrature weights, so under refinement of the
    boundary sample they converge for any sampling law whose normalized measure
    converges, uniformly in the wall distance, the property the model's slice
    attention has; the nearest sample's Euclidean distance and direction do not
    (they read the sample spacing, by an amount that grows as the query
    approaches the wall). Next to a flat wall every :math:`s_i` equals the wall
    distance, so :math:`d_q` is exact there whatever the sample. The facing
    factor keeps the far face of a thin body and the far wall of a concave
    corner out of the average (their normals point the other way) and at a
    convex edge selects the nearest face, as the nearest form does.

    On a curved wall the tangent-plane distances of the neighbouring samples
    differ from the wall distance by the curvature times half the squared
    tangential offset, so the kernel average carries a curvature bias of order
    :math:`\kappa \tau^2`. ``one_sided=True`` (the form the 2026-10-06 AirfRANS
    lanes trained with) excludes the samples whose tangent plane the query lies
    behind, which keeps :math:`d_q` the sign of the nearest sample's and removes
    the bias on the convex side of the wall where the kept set shrinks towards
    the query's foot, but leaves the bias on the concave side, so :math:`d_q`
    jumps by about :math:`\kappa \tau^2` across the wall. ``one_sided=False``
    averages every facing sample: :math:`d_q` is then continuous and smooth
    across the wall and carries the bias on both sides (its zero level set
    sits about :math:`\kappa \tau^2` outside a convex wall).

    All quantities are functions of pairwise differences, so :math:`d_q` is
    invariant and :math:`u_q` covariant under every rigid motion and
    reflection, and :math:`d_q` scales with the geometry when ``width`` does.
    ``log_measures`` are the samples' log measures (``None``: equal
    measures); ``width`` is a length in the coordinates' units, a scalar or
    one value per batch element. ``normals`` are renormalized.

    On CUDA with Triton available and no input requiring a gradient, the
    exact nearest search and a streaming accumulation kernel (two or three
    spatial dimensions) compute the result without materializing the (Q, N)
    block; otherwise a chunked PyTorch form, differentiable in every input
    (the side and the nearest index are discrete), is used. Half types are
    computed in float32.
    """
    b, nq, dim = query_points.shape
    width_t = torch.as_tensor(width, dtype=torch.float64, device=query_points.device)
    width_t = width_t.reshape(-1)
    if width_t.numel() == 1:
        width_t = width_t.expand(b)
    if width_t.numel() != b or not torch.compiler.is_compiling() and bool(
        (width_t <= 0).any()
    ):
        raise ValueError(
            f"width must be a positive length, a scalar or one per batch element ({b}), got {width!r}"
        )
    needs_grad = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad
        for t in (query_points, points, normals, log_measures)
    )
    ds, us = [], []
    for i in range(b):
        q, y, nrm = query_points[i], points[i], normals[i]
        work = q.dtype
        if work in (torch.float16, torch.bfloat16):
            work = torch.float32
        q, y, nrm = q.to(work).contiguous(), y.to(work).contiguous(), nrm.to(work)
        nrm = (nrm / nrm.norm(dim=-1, keepdim=True).clamp_min(eps)).contiguous()
        lw = (
            q.new_zeros(y.shape[0])
            if log_measures is None
            else log_measures[i].reshape(-1).to(work).contiguous()
        )
        tau = width_t[i].to(work)
        with torch.no_grad():
            idx = _nearest_index(q.detach(), y.detach())
        n_star = nrm[idx]  # (Q, d)
        s_star = ((q - y[idx]) * n_star).sum(-1)
        side = torch.where(s_star >= 0, 1.0, -1.0).to(work)
        if q.is_cuda and HAS_TRITON and not needs_grad and dim in (2, 3):
            out = torch.empty(nq, dim + 1, dtype=work, device=q.device)
            inv_width = (1.0 / tau).reshape(1).contiguous()
            block_q, block_n = 32, 128
            _soft_geometry_kernel[(triton.cdiv(nq, block_q),)](
                q,
                y,
                nrm,
                lw,
                n_star.contiguous(),
                side.contiguous(),
                idx.contiguous(),
                inv_width,
                out,
                nq,
                y.shape[0],
                DIM=dim,
                ONE_SIDED=one_sided,
                BLOCK_Q=block_q,
                BLOCK_N=block_n,
                num_warps=4,
            )
            d, u = out[:, 0], out[:, 1:]
        else:
            d, u = _soft_geometry_torch(
                q, y, nrm, lw, idx, n_star, side, tau, eps, one_sided
            )
        ds.append(d.to(query_points.dtype))
        us.append(u.to(query_points.dtype))
    return torch.stack(ds), torch.stack(us)
