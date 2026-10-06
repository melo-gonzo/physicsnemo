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

r"""Triton kernels of the global read: a softmax over every boundary cell, fused.

The global read (:class:`~physicsnemo.experimental.models.isla.kernel_read._GlobalKernelRead`)
gives every prediction point :math:`q` and head :math:`h`

.. math::

    o_{qh} = \sum_j a_{qjh} v_{jh}, \qquad a_{qjh} = \operatorname{softmax}_j
    \big(z_{hqj} + s \langle q_{qh}, k_{jh}\rangle + \log\omega_j\big),

with :math:`z_h = w_h \cdot \phi` the per-pair kernel. :func:`global_softmax_read`
computes it as flash attention does: :math:`z` (``(H, Q * N)``, one cuBLAS product of
the shared kernel basis) is the only per-pair tensor read; the softmax is computed
online over tiles of cells and never stored (the row's log-sum-exp is kept for
backward). Backward recomputes the softmax per tile: one kernel over query tiles
writes :math:`\partial z` (the per-pair logit gradient, needed for the kernel basis
and its weights) and :math:`\partial q`; one over cell tiles accumulates
:math:`\partial k, \partial v` over every query. No atomics, deterministic. All
accumulation in float32; the tile products take bfloat16 operands on the tensor cores
when the states are stored in bfloat16 (autocast training, as flash attention does)
and are exact float32 otherwise (``input_precision="ieee"``, and the query-key match
as rank-1 updates); float64 inputs take the eager path instead.
"""

from __future__ import annotations

import importlib

import torch

from physicsnemo.core.version_check import is_package_available

HAS_TRITON = is_package_available("triton")
if HAS_TRITON:
    try:
        triton = importlib.import_module("triton")
        tl = importlib.import_module("triton.language")
    except Exception:  # pragma: no cover - a broken triton install is treated as absent
        HAS_TRITON = False

#: Query rows and cells per tile of the kernels.
BLOCK_Q = 64
BLOCK_N = 64
#: Cells per program of the key / value gradient kernel.
BLOCK_N_KV = 32
#: Channel dimension padding of the tile products (``tl.dot`` needs 16).
PAD = 16


def supported(*tensors: torch.Tensor) -> bool:
    """The kernels apply: CUDA tensors, Triton importable, no float64."""
    return HAS_TRITON and all(t.is_cuda and t.dtype != torch.float64 for t in tensors)


if HAS_TRITON:

    @triton.jit
    def _dot(a, b, LOWP: tl.constexpr):
        """A tile product with float32 accumulation: bfloat16 operands on the tensor
        cores (``LOWP``, the storage is bfloat16 already) or exact float32."""
        if LOWP:
            return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
        else:
            return tl.dot(a, b, input_precision="ieee")

    @triton.jit
    def _qk(q_ptr, k_ptr, qv, kv, rq, rn, mq, mn, Q, N, h,
            CK: tl.constexpr, LOWP: tl.constexpr):  # fmt: skip
        """<q, k> of a tile: a tile product, or (float32) CK exact rank-1 updates."""
        if LOWP:
            return tl.dot(qv.to(tl.bfloat16), tl.trans(kv).to(tl.bfloat16))
        else:
            qo = (h * Q + rq.to(tl.int64)) * CK
            ko = (h * N + rn.to(tl.int64)) * CK
            s = tl.zeros([rq.shape[0], rn.shape[0]], tl.float32)
            for c in tl.static_range(CK):
                qc = tl.load(q_ptr + qo + c, mask=mq, other=0.0).to(tl.float32)
                kc = tl.load(k_ptr + ko + c, mask=mn, other=0.0).to(tl.float32)
                s += qc[:, None] * kc[None, :]
            return s

    @triton.jit
    def _tile_logits(z_ptr, q_ptr, k_ptr, lf_ptr, qv, kv, rq, rn, mq, mn, Q, N, h,
                     scale, CK: tl.constexpr, LOWP: tl.constexpr):  # fmt: skip
        """The logits of one (query tile, cell tile) for head ``h`` (masked cells -inf)."""
        s = _qk(q_ptr, k_ptr, qv, kv, rq, rn, mq, mn, Q, N, h, CK, LOWP)
        zo = h * Q.to(tl.int64) * N + rq.to(tl.int64)[:, None] * N + rn[None, :]
        z = tl.load(z_ptr + zo, mask=mq[:, None] & mn[None, :], other=0.0)
        lf = tl.load(lf_ptr + rn, mask=mn, other=0.0)
        logit = z.to(tl.float32) + s * scale + lf.to(tl.float32)[None, :]
        return tl.where(mn[None, :], logit, float("-inf")), zo

    @triton.jit
    def _load_rows(ptr, rows, mask, n_rows, h, C: tl.constexpr, P: tl.constexpr):
        """Rows ``(len(rows), P)`` of a ``(H, n_rows, C)`` tensor, zero-padded to P."""
        c = tl.arange(0, P)
        off = (h * n_rows + rows.to(tl.int64))[:, None] * C + c[None, :]
        return tl.load(ptr + off, mask=mask[:, None] & (c[None, :] < C), other=0.0).to(
            tl.float32
        )

    @triton.jit
    def _fwd_kernel(
        z_ptr, q_ptr, k_ptr, v_ptr, lf_ptr, o_ptr, lse_ptr, Q, N, scale,
        H: tl.constexpr, CK: tl.constexpr, CV: tl.constexpr, P: tl.constexpr,
        BQ: tl.constexpr, BN: tl.constexpr, LOWP: tl.constexpr,
    ):  # fmt: skip
        """o (H, Q, CV) and the row log-sum-exp (H, Q); one program per (tile, head)."""
        pid = tl.program_id(0)
        h = pid % H
        rq = (pid // H) * BQ + tl.arange(0, BQ)
        mq = rq < Q
        qv = _load_rows(q_ptr, rq, mq, Q, h, CK, P)
        m_i = tl.full([BQ], float("-inf"), tl.float32)
        l_i = tl.zeros([BQ], tl.float32)
        acc = tl.zeros([BQ, P], tl.float32)
        for n0 in range(0, N, BN):
            rn = n0 + tl.arange(0, BN)
            mn = rn < N
            kv = _load_rows(k_ptr, rn, mn, N, h, CK, P)
            vv = _load_rows(v_ptr, rn, mn, N, h, CV, P)
            logit, _ = _tile_logits(
                z_ptr,
                q_ptr,
                k_ptr,
                lf_ptr,
                qv,
                kv,
                rq,
                rn,
                mq,
                mn,
                Q,
                N,
                h,
                scale,
                CK,
                LOWP,
            )
            m_new = tl.maximum(m_i, tl.max(logit, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(logit - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + _dot(p, vv, LOWP)
            m_i = m_new
        c = tl.arange(0, P)
        oo = (h * Q + rq.to(tl.int64))[:, None] * CV + c[None, :]
        tl.store(o_ptr + oo, acc / l_i[:, None], mask=mq[:, None] & (c[None, :] < CV))
        tl.store(lse_ptr + h * Q + rq, m_i + tl.log(l_i), mask=mq)

    @triton.jit
    def _bwd_q_kernel(
        z_ptr, q_ptr, k_ptr, v_ptr, lf_ptr, lse_ptr, dd_ptr, do_ptr, dz_ptr, dq_ptr,
        Q, N, scale,
        H: tl.constexpr, CK: tl.constexpr, CV: tl.constexpr, P: tl.constexpr,
        BQ: tl.constexpr, BN: tl.constexpr, LOWP: tl.constexpr,
    ):  # fmt: skip
        """dz (H, Q*N) = a * (dO . v - D) per pair, and dq (H, Q, CK)."""
        pid = tl.program_id(0)
        h = pid % H
        rq = (pid // H) * BQ + tl.arange(0, BQ)
        mq = rq < Q
        qv = _load_rows(q_ptr, rq, mq, Q, h, CK, P)
        dov = _load_rows(do_ptr, rq, mq, Q, h, CV, P)
        lse = tl.load(lse_ptr + h * Q + rq, mask=mq, other=0.0)
        dd = tl.load(dd_ptr + h * Q + rq, mask=mq, other=0.0)
        dq = tl.zeros([BQ, P], tl.float32)
        for n0 in range(0, N, BN):
            rn = n0 + tl.arange(0, BN)
            mn = rn < N
            kv = _load_rows(k_ptr, rn, mn, N, h, CK, P)
            vv = _load_rows(v_ptr, rn, mn, N, h, CV, P)
            logit, zo = _tile_logits(
                z_ptr,
                q_ptr,
                k_ptr,
                lf_ptr,
                qv,
                kv,
                rq,
                rn,
                mq,
                mn,
                Q,
                N,
                h,
                scale,
                CK,
                LOWP,
            )
            a = tl.exp(logit - lse[:, None])
            dp = _dot(dov, tl.trans(vv), LOWP)
            dl = a * (dp - dd[:, None])
            tl.store(
                dz_ptr + zo,
                dl.to(dz_ptr.dtype.element_ty),
                mask=mq[:, None] & mn[None, :],
            )
            dq += _dot(dl, kv, LOWP)
        c = tl.arange(0, P)
        oo = (h * Q + rq.to(tl.int64))[:, None] * CK + c[None, :]
        tl.store(dq_ptr + oo, dq * scale, mask=mq[:, None] & (c[None, :] < CK))

    @triton.jit
    def _bwd_kv_kernel(
        z_ptr, q_ptr, k_ptr, v_ptr, lf_ptr, lse_ptr, dd_ptr, do_ptr, dk_ptr, dv_ptr,
        Q, N, scale,
        H: tl.constexpr, CK: tl.constexpr, CV: tl.constexpr, P: tl.constexpr,
        BQ: tl.constexpr, BN: tl.constexpr, LOWP: tl.constexpr,
    ):  # fmt: skip
        """dk (H, N, CK) and dv (H, N, CV), summed over every query; one program per
        (cell tile, head)."""
        pid = tl.program_id(0)
        h = pid % H
        rn = (pid // H) * BN + tl.arange(0, BN)
        mn = rn < N
        kv = _load_rows(k_ptr, rn, mn, N, h, CK, P)
        vv = _load_rows(v_ptr, rn, mn, N, h, CV, P)
        dk = tl.zeros([BN, P], tl.float32)
        dv = tl.zeros([BN, P], tl.float32)
        for q0 in range(0, Q, BQ):
            rq = q0 + tl.arange(0, BQ)
            mq = rq < Q
            qv = _load_rows(q_ptr, rq, mq, Q, h, CK, P)
            dov = _load_rows(do_ptr, rq, mq, Q, h, CV, P)
            lse = tl.load(lse_ptr + h * Q + rq, mask=mq, other=0.0)
            dd = tl.load(dd_ptr + h * Q + rq, mask=mq, other=0.0)
            logit, _ = _tile_logits(
                z_ptr,
                q_ptr,
                k_ptr,
                lf_ptr,
                qv,
                kv,
                rq,
                rn,
                mq,
                mn,
                Q,
                N,
                h,
                scale,
                CK,
                LOWP,
            )
            a = tl.where(mq[:, None], tl.exp(logit - lse[:, None]), 0.0)
            dv += _dot(tl.trans(a), dov, LOWP)
            dp = _dot(dov, tl.trans(vv), LOWP)
            dl = a * (dp - dd[:, None])
            dk += _dot(tl.trans(dl), qv, LOWP)
        c = tl.arange(0, P)
        ko = (h * N + rn.to(tl.int64))[:, None] * CK + c[None, :]
        tl.store(dk_ptr + ko, dk * scale, mask=mn[:, None] & (c[None, :] < CK))
        vo = (h * N + rn.to(tl.int64))[:, None] * CV + c[None, :]
        tl.store(dv_ptr + vo, dv, mask=mn[:, None] & (c[None, :] < CV))


def _consts(q, v):
    nh, nq, ck = q.shape
    cv = v.shape[-1]
    if ck > PAD or cv > PAD:
        raise ValueError(f"channels per head must be <= {PAD}, got {ck}, {cv}")
    return nh, nq, ck, cv


def _fwd(z, q, k, v, lf, scale):
    nh, nq, ck, cv = _consts(q, v)
    n = k.shape[1]
    lowp = q.dtype in (torch.bfloat16, torch.float16)
    o = torch.empty(nh, nq, cv, device=q.device, dtype=torch.float32)
    lse = torch.empty(nh, nq, device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(nq, BLOCK_Q) * nh,)
    _fwd_kernel[grid](
        z, q, k, v, lf, o, lse, nq, n, float(scale),
        H=nh, CK=ck, CV=cv, P=PAD, BQ=BLOCK_Q, BN=BLOCK_N, LOWP=lowp,
    )  # fmt: skip
    return o, lse


class _FusedGlobalSoftmaxRead(torch.autograd.Function):
    """``(phi, w, q, k, v, log_frac, scale) -> o (H, Q, C)`` float32, as
    :class:`~physicsnemo.experimental.models.isla.kernel_read._GlobalSoftmaxRead`."""

    @staticmethod
    def forward(ctx, phi, w, q, k, v, log_frac, scale):
        with torch.autocast("cuda", enabled=False):
            q_, k_, v_ = (t.contiguous() for t in (q, k, v))
            lf = log_frac.contiguous()
            z = torch.mm(w.to(phi.dtype), phi)
            o, lse = _fwd(z, q_, k_, v_, lf, scale)
        ctx.save_for_backward(phi, w, q_, k_, v_, lf, o, lse)
        ctx.scale = scale
        return o

    @staticmethod
    def backward(ctx, d_out):
        phi, w, q, k, v, lf, o, lse = ctx.saved_tensors
        need = ctx.needs_input_grad
        with torch.autocast("cuda", enabled=False):
            nh, nq, ck, cv = _consts(q, v)
            n = k.shape[1]
            z = torch.mm(w.to(phi.dtype), phi)
            do = d_out.float().contiguous()
            dd = (do * o).sum(-1).contiguous()
            dz = torch.empty_like(z)
            dq = torch.empty(nh, nq, ck, device=q.device, dtype=torch.float32)
            dk = torch.empty(nh, n, ck, device=q.device, dtype=torch.float32)
            dv = torch.empty(nh, n, cv, device=q.device, dtype=torch.float32)
            s = float(ctx.scale)
            lowp = q.dtype in (torch.bfloat16, torch.float16)
            _bwd_q_kernel[(triton.cdiv(nq, BLOCK_Q) * nh,)](
                z, q, k, v, lf, lse, dd, do, dz, dq, nq, n, s,
                H=nh, CK=ck, CV=cv, P=PAD, BQ=BLOCK_Q, BN=BLOCK_N, LOWP=lowp,
            )  # fmt: skip
            _bwd_kv_kernel[(triton.cdiv(n, BLOCK_N_KV) * nh,)](
                z, q, k, v, lf, lse, dd, do, dk, dv, nq, n, s,
                H=nh, CK=ck, CV=cv, P=PAD, BQ=BLOCK_Q, BN=BLOCK_N_KV, LOWP=lowp,
            )  # fmt: skip
            d_phi = torch.mm(w.to(dz.dtype).t(), dz) if need[0] else None
            d_w = None
            if need[1]:
                d_w = (
                    torch.mm(dz, phi.t())
                    if dz.dtype == torch.float32
                    else torch.mm(dz, phi.t(), out_dtype=torch.float32)
                )
            d_lf = dz.float().view(nh, nq, n).sum((0, 1)) if need[5] else None
        return (
            d_phi,
            None if d_w is None else d_w.to(w.dtype),
            dq.to(q.dtype) if need[2] else None,
            dk.to(k.dtype) if need[3] else None,
            dv.to(v.dtype) if need[4] else None,
            None if d_lf is None else d_lf.to(lf.dtype),
            None,
        )


def global_softmax_read(phi, w, q, k, v, log_frac, scale):
    """The fused global read (see the module docstring); CUDA, float32 or lower."""
    return _FusedGlobalSoftmaxRead.apply(phi, w, q, k, v, log_frac, scale)


# ---------------------------------------------------------------------------
# The kernel basis of every pair: invariants and MLP fused, never stored.
# ---------------------------------------------------------------------------

#: Pairs per tile of the basis kernels.
BLOCK_P = 64
#: Warps per program of the basis kernels.
BASIS_WARPS = 8
#: Padded width of the pair invariants (``tl.dot`` needs 16) and of a position.
GEO_PAD = 16
DIM_PAD = 4
_SQRT1_2 = tl.constexpr(0.7071067811865476) if HAS_TRITON else None
_INV_SQRT_2PI = tl.constexpr(0.3989422804014327) if HAS_TRITON else None

if HAS_TRITON:

    @triton.jit
    def _rows4(ptr, rows, mask, D: tl.constexpr):
        """Rows ``(len(rows), 4)`` of a ``(n, D)`` tensor, zero-padded."""
        c = tl.arange(0, 4)
        off = rows.to(tl.int64)[:, None] * D + c[None, :]
        return tl.load(ptr + off, mask=mask[:, None] & (c[None, :] < D), other=0.0).to(
            tl.float32
        )

    @triton.jit
    def _put(F, f, g: tl.constexpr, GP: tl.constexpr):
        """Column ``g`` of the feature tile ``F`` set to ``f``."""
        col = tl.arange(0, GP)
        return F + f[:, None] * (col[None, :] == g).to(tl.float32)

    @triton.jit
    def _pair_features(
        p, mp, N, eps, qr_ptr, qn_ptr, qw_ptr, qv_ptr, br_ptr, bn_ptr, bv_ptr, g_ptr,
        D: tl.constexpr, K: tl.constexpr, PQ: tl.constexpr, PB: tl.constexpr,
        GP: tl.constexpr, BP: tl.constexpr,
    ):  # fmt: skip
        """The invariants of the pairs ``p`` (flat ``q * N + j``) as a ``(BP, GP)``
        tile, in the order of :func:`~physicsnemo.experimental.models.isla.
        kernel_read.global_pair_invariants`."""
        qi = p // N
        nj = p - qi * N
        xq = _rows4(qr_ptr, qi, mp, D)
        nq = _rows4(qn_ptr, qi, mp, D)
        yj = _rows4(br_ptr, nj, mp, D)
        nb = _rows4(bn_ptr, nj, mp, D)
        rel = xq - yj
        dist = tl.sqrt(tl.sum(rel * rel, 1))
        dist_g = tl.maximum(dist, eps)
        rho = rel / dist_g[:, None]
        log_r = tl.log(dist_g)
        wall = tl.load(qw_ptr + qi, mask=mp, other=1.0).to(tl.float32)
        F = tl.zeros([BP, GP], tl.float32)
        F = _put(F, dist, 0, GP)
        F = _put(F, log_r, 1, GP)
        F = _put(F, log_r - tl.log(tl.maximum(wall, eps)), 2, GP)
        F = _put(F, tl.sum(rho * nb, 1), 3, GP)
        F = _put(F, tl.sum(rho * nq, 1), 4, GP)
        F = _put(F, tl.sum(nq * nb, 1), 5, GP)
        c = tl.arange(0, 4)
        for k in tl.static_range(K):
            gk = tl.load(g_ptr + k * D + c, mask=c < D, other=0.0).to(tl.float32)
            F = _put(F, tl.sum(rho * gk[None, :], 1), 6 + k, GP)
            F = _put(F, tl.sum(nb * gk[None, :], 1), 6 + K + k, GP)
        for i in tl.static_range(PQ):
            v = _rows4(qv_ptr, qi * PQ + i, mp, D)
            F = _put(F, tl.sum(rho * v, 1), 6 + 2 * K + i, GP)
            F = _put(F, tl.sum(nb * v, 1), 6 + 2 * K + PQ + i, GP)
        for i in tl.static_range(PB):
            u = _rows4(bv_ptr, nj * PB + i, mp, D)
            F = _put(F, tl.sum(rho * u, 1), 6 + 2 * K + 2 * PQ + i, GP)
            F = _put(F, tl.sum(nq * u, 1), 6 + 2 * K + 2 * PQ + PB + i, GP)
        return F

    @triton.jit
    def _gelu(x):
        return 0.5 * x * (1.0 + tl.erf(x * _SQRT1_2))

    @triton.jit
    def _gelu_grad(x):
        return 0.5 * (1.0 + tl.erf(x * _SQRT1_2)) + x * _INV_SQRT_2PI * tl.exp(
            -0.5 * x * x
        )

    @triton.jit
    def _wt(ptr, n_out, n_in, R: tl.constexpr, C: tl.constexpr):
        """The transpose ``(R, C)`` of a Linear weight ``(n_out, n_in)``, zero-padded:
        element ``[i, o] = W[o, i]``."""
        i = tl.arange(0, R)
        o = tl.arange(0, C)
        m = (i[:, None] < n_in) & (o[None, :] < n_out)
        return tl.load(ptr + o[None, :] * n_in + i[:, None], mask=m, other=0.0)

    @triton.jit
    def _mlp_forward(F, w1_ptr, b1_ptr, w2_ptr, b2_ptr, G,
                     GP: tl.constexpr, HID: tl.constexpr, LOWP: tl.constexpr):  # fmt: skip
        """The first two layers: pre-activations a1, a2 and activations x1, x2."""
        hc = tl.arange(0, HID)
        a1 = _dot(F, _wt(w1_ptr, HID, G, GP, HID), LOWP) + tl.load(b1_ptr + hc)[None, :]
        x1 = _gelu(a1)
        a2 = _dot(x1, _wt(w2_ptr, HID, HID, HID, HID), LOWP)
        a2 = a2 + tl.load(b2_ptr + hc)[None, :]
        return a1, x1, a2, _gelu(a2)

    @triton.jit
    def _basis_fwd_kernel(
        qr_ptr, qn_ptr, qw_ptr, qv_ptr, br_ptr, bn_ptr, bv_ptr, g_ptr,
        w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, b3_ptr, phi_ptr, Q, N, G, eps,
        D: tl.constexpr, K: tl.constexpr, PQ: tl.constexpr, PB: tl.constexpr,
        GP: tl.constexpr, HID: tl.constexpr, NB: tl.constexpr, BP: tl.constexpr,
        LOWP: tl.constexpr,
    ):  # fmt: skip
        """phi (NB, Q*N): the basis of every pair, one program per BP pairs."""
        total = Q.to(tl.int64) * N
        p = tl.program_id(0).to(tl.int64) * BP + tl.arange(0, BP)
        mp = p < total
        F = _pair_features(
            p, mp, N, eps, qr_ptr, qn_ptr, qw_ptr, qv_ptr, br_ptr, bn_ptr, bv_ptr,
            g_ptr, D, K, PQ, PB, GP, BP,
        )  # fmt: skip
        _, _, _, x2 = _mlp_forward(F, w1_ptr, b1_ptr, w2_ptr, b2_ptr, G, GP, HID, LOWP)
        jc = tl.arange(0, NB)
        phi = _dot(x2, _wt(w3_ptr, NB, HID, HID, NB), LOWP)
        phi = phi + tl.load(b3_ptr + jc)[None, :]
        tl.store(
            phi_ptr + jc[None, :].to(tl.int64) * total + p[:, None],
            phi.to(phi_ptr.dtype.element_ty),
            mask=mp[:, None],
        )

    @triton.jit
    def _basis_bwd_kernel(
        qr_ptr, qn_ptr, qw_ptr, qv_ptr, br_ptr, bn_ptr, bv_ptr, g_ptr,
        w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, dphi_ptr, part_ptr,
        Q, N, G, eps, n_tiles,
        D: tl.constexpr, K: tl.constexpr, PQ: tl.constexpr, PB: tl.constexpr,
        GP: tl.constexpr, HID: tl.constexpr, NB: tl.constexpr, BP: tl.constexpr,
        LOWP: tl.constexpr,
    ):  # fmt: skip
        """This program's share of the MLP's parameter gradients, written to its row
        of ``part`` (flat dW1^T (GP, HID), db1, dW2^T (HID, HID), db2, dW3^T (HID,
        NB), db3); the tiles are dealt round-robin over the programs."""
        total = Q.to(tl.int64) * N
        hc = tl.arange(0, HID)
        jc = tl.arange(0, NB)
        gc = tl.arange(0, GP)
        w2 = _wt(w2_ptr, HID, HID, HID, HID)  # W2^T (in, out)
        w3 = _wt(w3_ptr, NB, HID, HID, NB)  # W3^T (in, out)
        d_w1 = tl.zeros([GP, HID], tl.float32)
        d_b1 = tl.zeros([HID], tl.float32)
        d_w2 = tl.zeros([HID, HID], tl.float32)
        d_b2 = tl.zeros([HID], tl.float32)
        d_w3 = tl.zeros([HID, NB], tl.float32)
        d_b3 = tl.zeros([NB], tl.float32)
        for t in range(tl.program_id(0), n_tiles, tl.num_programs(0)):
            p = t.to(tl.int64) * BP + tl.arange(0, BP)
            mp = p < total
            F = _pair_features(
                p, mp, N, eps, qr_ptr, qn_ptr, qw_ptr, qv_ptr, br_ptr, bn_ptr,
                bv_ptr, g_ptr, D, K, PQ, PB, GP, BP,
            )  # fmt: skip
            a1, x1, a2, x2 = _mlp_forward(
                F, w1_ptr, b1_ptr, w2_ptr, b2_ptr, G, GP, HID, LOWP
            )
            dphi = tl.load(
                dphi_ptr + jc[None, :].to(tl.int64) * total + p[:, None],
                mask=mp[:, None],
                other=0.0,
            ).to(tl.float32)
            d_w3 += _dot(tl.trans(x2), dphi, LOWP)
            d_b3 += tl.sum(dphi, 0)
            da2 = _dot(dphi, tl.trans(w3), LOWP) * _gelu_grad(a2)
            d_w2 += _dot(tl.trans(x1), da2, LOWP)
            d_b2 += tl.sum(da2, 0)
            da1 = _dot(da2, tl.trans(w2), LOWP) * _gelu_grad(a1)
            d_w1 += _dot(tl.trans(F), da1, LOWP)
            d_b1 += tl.sum(da1, 0)
        base = part_ptr + tl.program_id(0).to(tl.int64) * (
            GP * HID + HID + HID * HID + HID + HID * NB + NB
        )
        tl.store(base + gc[:, None] * HID + hc[None, :], d_w1)
        base += GP * HID
        tl.store(base + hc, d_b1)
        base += HID
        tl.store(base + hc[:, None] * HID + hc[None, :], d_w2)
        base += HID * HID
        tl.store(base + hc, d_b2)
        base += HID
        tl.store(base + hc[:, None] * NB + jc[None, :], d_w3)
        base += HID * NB
        tl.store(base + jc, d_b3)


def _basis_args(geometry):
    """Contiguous float32 geometry and its constexpr shapes."""
    f = lambda t: None if t is None else t.detach().float().contiguous()  # noqa: E731
    qr, qn, qw = (
        f(geometry["query_r"]),
        f(geometry["query_n"]),
        f(geometry["query_wall"]),
    )
    br, bn, g = (
        f(geometry["boundary_r"]),
        f(geometry["boundary_n"]),
        f(geometry["g_unit"]),
    )
    qv, bv = f(geometry["query_vecs"]), f(geometry["boundary_vecs"])
    pq = 0 if qv is None else qv.shape[1]
    pb = 0 if bv is None else bv.shape[1]
    qv = qr.new_zeros(1) if qv is None or pq == 0 else qv
    bv = qr.new_zeros(1) if bv is None or pb == 0 else bv
    shape = {"D": qr.shape[1], "K": g.shape[0], "PQ": pq, "PB": pb}
    return (qr, qn, qw, qv, br, bn, bv, g), shape, float(geometry["eps"])


class _FusedKernelBasis(torch.autograd.Function):
    """``(geometry tensors..., W1, b1, W2, b2, W3, b3) -> phi (NB, Q * N)``: the
    shared kernel basis of every pair, invariants and MLP in one kernel; backward
    recomputes both per tile and returns the MLP's parameter gradients only (the
    geometry carries none)."""

    @staticmethod
    def forward(ctx, geo, shape, eps, dtype, lowp, w1, b1, w2, b2, w3, b3):
        qr, br = geo[0], geo[4]
        nq, n = qr.shape[0], br.shape[0]
        hid, g_width, nb = w1.shape[0], w1.shape[1], w3.shape[0]
        phi = torch.empty(nb, nq * n, device=qr.device, dtype=dtype)
        n_tiles = triton.cdiv(nq * n, BLOCK_P)
        _basis_fwd_kernel[(n_tiles,)](
            *geo, w1, b1, w2, b2, w3, b3, phi, nq, n, g_width, eps, **shape,
            GP=GEO_PAD, HID=hid, NB=nb, BP=BLOCK_P, LOWP=lowp, num_warps=BASIS_WARPS,
        )  # fmt: skip
        ctx.save_for_backward(*geo, w1, b1, w2, b2, w3)
        ctx.shape, ctx.eps, ctx.lowp = shape, eps, lowp
        return phi

    @staticmethod
    def backward(ctx, d_phi):
        *geo, w1, b1, w2, b2, w3 = ctx.saved_tensors
        qr, br = geo[0], geo[4]
        nq, n = qr.shape[0], br.shape[0]
        hid, g_width, nb = w1.shape[0], w1.shape[1], w3.shape[0]
        n_tiles = triton.cdiv(nq * n, BLOCK_P)
        sms = torch.cuda.get_device_properties(qr.device).multi_processor_count
        n_prog = max(1, min(n_tiles, 4 * sms))
        width = GEO_PAD * hid + hid + hid * hid + hid + hid * nb + nb
        part = torch.empty(n_prog, width, device=qr.device, dtype=torch.float32)
        _basis_bwd_kernel[(n_prog,)](
            *geo, w1, b1, w2, b2, w3, d_phi.contiguous(), part, nq, n, g_width,
            ctx.eps, n_tiles, **ctx.shape, GP=GEO_PAD, HID=hid, NB=nb, BP=BLOCK_P,
            LOWP=ctx.lowp, num_warps=BASIS_WARPS,
        )  # fmt: skip
        s = part.sum(0)
        sizes = [GEO_PAD * hid, hid, hid * hid, hid, hid * nb, nb]
        dw1, db1, dw2, db2, dw3, db3 = s.split(sizes)
        dw1 = dw1.view(GEO_PAD, hid)[:g_width].t()
        dw2 = dw2.view(hid, hid).t()
        dw3 = dw3.view(hid, nb).t()
        return (None, None, None, None, None, dw1, db1, dw2, db2, dw3, db3)


def fused_basis_supported(basis, geometry) -> bool:
    """The fused basis applies: CUDA float32 geometry or lower, Triton, an MLP of
    the expected shape (three Linear layers with exact GELU), invariants at most
    ``GEO_PAD`` wide, at most three dimensions, and no geometry gradient."""
    net = getattr(basis, "net", None)
    t = geometry["query_r"]
    return (
        HAS_TRITON
        and t.is_cuda
        and t.dtype != torch.float64
        and t.shape[1] <= DIM_PAD
        and net is not None
        and len(net) == 5
        and getattr(net[1], "approximate", None) == "none"
        and net[0].in_features <= GEO_PAD
        and net[0].out_features in (16, 32, 64, 128)
        and net[4].out_features in (16, 32, 64)
        and not any(
            isinstance(v, torch.Tensor) and v.requires_grad for v in geometry.values()
        )
    )


def fused_kernel_basis(basis, geometry) -> torch.Tensor:
    """The kernel basis ``(NB, Q * N)`` of every pair of one case, fused; stored in
    bfloat16 under CUDA bfloat16 autocast (the dtype the eager MLP would return),
    else float32."""
    geo, shape, eps = _basis_args(geometry)
    lowp = torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") in (
        torch.bfloat16,
        torch.float16,
    )
    dtype = torch.get_autocast_dtype("cuda") if lowp else torch.float32
    net = basis.net
    params = [net[0].weight, net[0].bias, net[2].weight, net[2].bias]
    params += [net[4].weight, net[4].bias]
    params = [p.contiguous() for p in params]
    return _FusedKernelBasis.apply(geo, shape, eps, dtype, lowp, *params)
