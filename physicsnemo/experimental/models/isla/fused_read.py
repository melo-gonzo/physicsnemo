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

r"""The tree read with its kernel term fused in (Triton): nothing per pair is kept
between forward and backward but the kernel basis and the plan.

The reference read (:mod:`segment_softmax` behind
:class:`~physicsnemo.experimental.models.isla.tree_read._TreeKernelRead`) takes each
block's kernel term :math:`\kappa_{ph} = \sum_f \phi_{pf} W_{hf}` as a ``(P, H)``
tensor, a matmul of the shared basis :math:`\phi` ``(P, F)`` with the block's kernel
weights, which autograd keeps for backward in every block (2e8 to 7e8 pairs at
DrivAerML scale). Here the read's kernels compute :math:`\kappa` per tile from the
basis and the weights instead, in forward and again in backward (as flash attention
recomputes its logits), so a block keeps only its per-point output and log-sum-exp.

The arithmetic is the reference's but for summation order: :math:`\kappa` is a sum of
exact products in float32 rounded to the basis's dtype (bfloat16 under autocast, as
the reference matmul's output is), its gradient is rounded to that dtype before it
reaches the basis and the weights (as the reference read's kernel-term gradient is),
the basis gradient of each block is rounded to that dtype (the reference matmul's
backward), and the weights' gradient is a float32 sum over every pair rounded once.
The blocks' basis gradients are summed in place, in one buffer, in the order and at
the rounding points of autograd's own sum over the basis's uses (each block's
contribution rounded, then added to the running sum and rounded again), and handed to
the basis once, after the last read's backward (:func:`basis_sink`): no block's
gradient is materialized beside the sum.
The small products are float32 broadcast sums, each over the last axis: Triton (3.7,
GB300) turns a sum over the middle axis, ``(M, K, 1) * (1, K, N)`` summed over ``K``,
into a ``tl.dot`` at TF32 precision (relative error 1e-3 against float64).
"""

from __future__ import annotations

import importlib

import torch

from physicsnemo.core.version_check import is_package_available
from physicsnemo.experimental.models.isla.segment_softmax import (
    BLOCK_PAIRS,
    NUM_WARPS,
    SOURCE_CHUNK,
    PairPlan,
    _source_items,
)

### Triton is an optional dependency, loaded through importlib (as in geo_kernel.py).
HAS_TRITON = is_package_available("triton")
if HAS_TRITON:
    try:
        triton = importlib.import_module("triton")
        tl = importlib.import_module("triton.language")
    except Exception:  # pragma: no cover - a broken triton install is treated as absent
        HAS_TRITON = False


def _pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def supported(basis: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    """The fused read applies: CUDA tensors, Triton importable, no float64 (the
    reference read runs float64 eagerly), and power-of-two widths."""
    h, kc, c, f = q.shape[1], q.shape[2], v.shape[2], basis.shape[1]
    return (
        HAS_TRITON
        and all(t.is_cuda for t in (basis, q, k, v))
        and torch.float64 not in (basis.dtype, q.dtype, k.dtype, v.dtype)
        and all(_pow2(n) for n in (h, kc, c, f, h * kc, h * c))
    )


if HAS_TRITON:

    @triton.jit
    def _weights(w_ptr, H: tl.constexpr, F: tl.constexpr):
        """The kernel weights ``(H, F)`` in float32."""
        return tl.load(w_ptr + tl.arange(0, H)[:, None] * F + tl.arange(0, F)[None, :]).to(tl.float32)

    @triton.jit
    def _weights_t(w_ptr, H: tl.constexpr, F: tl.constexpr):
        """The kernel weights transposed, ``(F, H)`` in float32."""
        return tl.load(w_ptr + tl.arange(0, H)[None, :] * F + tl.arange(0, F)[:, None]).to(tl.float32)

    @triton.jit
    def _kernel_term(basis_ptr, p, msk, w, F: tl.constexpr):
        """The pairs' basis rows ``(BLOCK, F)`` and kernel terms ``(BLOCK, H)``, the
        latter rounded to the basis's dtype (both returned in float32)."""
        ff = tl.arange(0, F)
        bas = tl.load(basis_ptr + p[:, None] * F + ff[None, :], mask=msk[:, None], other=0.0).to(tl.float32)
        ker = tl.sum(bas[:, None, :] * w[None, :, :], axis=2)
        return bas, ker.to(basis_ptr.dtype.element_ty).to(tl.float32)

    @triton.jit
    def _fkr_fwd_kernel(
        off_ptr, bj_ptr, basis_ptr, w_ptr, logm_ptr, q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
        scale, H: tl.constexpr, C: tl.constexpr, KC: tl.constexpr, F: tl.constexpr, BLOCK: tl.constexpr,
    ):  # fmt: skip
        """One point: online softmax over its pairs; out[s] (H, C), lse[s] (H)."""
        s = tl.program_id(0).to(tl.int64)
        start = tl.load(off_ptr + s)
        end = tl.load(off_ptr + s + 1)
        hh = tl.arange(0, H)
        w = _weights(w_ptr, H, F)
        q = tl.reshape(tl.load(q_ptr + s * (H * KC) + tl.arange(0, H * KC)).to(tl.float32), [H, KC])
        m_run = tl.full([H], float("-inf"), tl.float32)
        den = tl.zeros([H], tl.float32)
        acc = tl.zeros([H, C], tl.float32)
        for p0 in range(start, end, BLOCK):
            p = p0 + tl.arange(0, BLOCK)
            msk = p < end
            bj = tl.load(bj_ptr + p, mask=msk, other=0).to(tl.int64)
            _, ker = _kernel_term(basis_ptr, p, msk, w, F)
            k = tl.reshape(
                tl.load(
                    k_ptr + bj[:, None] * (H * KC) + tl.arange(0, H * KC)[None, :],
                    mask=msk[:, None], other=0.0,
                ).to(tl.float32),
                [BLOCK, H, KC],
            )
            score = tl.sum(k * q[None, :, :], axis=2)  # (BLOCK, H)
            logm = tl.load(logm_ptr + p, mask=msk, other=0.0).to(tl.float32)
            z = ker + score * scale + logm[:, None]
            z = tl.where(msk[:, None], z, float("-inf"))
            m_new = tl.maximum(m_run, tl.max(z, axis=0))
            alpha = tl.exp(m_run - m_new)
            pe = tl.exp(z - m_new[None, :])
            den = den * alpha + tl.sum(pe, axis=0)
            v = tl.reshape(
                tl.load(
                    v_ptr + bj[:, None] * (H * C) + tl.arange(0, H * C)[None, :],
                    mask=msk[:, None], other=0.0,
                ).to(tl.float32),
                [BLOCK, H, C],
            )
            acc = acc * alpha[:, None] + tl.sum(pe[:, :, None] * v, axis=0)
            m_run = m_new
        safe = tl.where(den > 0, den, 1.0)
        out = acc / safe[:, None]
        tl.store(out_ptr + s * (H * C) + tl.arange(0, H * C), tl.reshape(out, [H * C]))
        tl.store(lse_ptr + s * H + hh, m_run + tl.log(safe))

    @triton.jit
    def _fkr_bwd_point_kernel(
        off_ptr, bj_ptr, basis_ptr, w_ptr, logm_ptr, q_ptr, k_ptr, v_ptr, lse_ptr, dsum_ptr, dout_ptr,
        dq_ptr, dbasis_ptr, dw_ptr,
        scale, H: tl.constexpr, C: tl.constexpr, KC: tl.constexpr, F: tl.constexpr, BLOCK: tl.constexpr,
        BASIS_GRAD: tl.constexpr,
    ):  # fmt: skip
        """One point: dq[s], the basis gradient of its pairs through this block's
        weights added to dbasis (BASIS_GRAD), and its part of the weights' gradient,
        dw[s] (H, F) float32."""
        s = tl.program_id(0).to(tl.int64)
        start = tl.load(off_ptr + s)
        end = tl.load(off_ptr + s + 1)
        hh = tl.arange(0, H)
        ff = tl.arange(0, F)
        w = _weights(w_ptr, H, F)
        wt = _weights_t(w_ptr, H, F)
        q = tl.reshape(tl.load(q_ptr + s * (H * KC) + tl.arange(0, H * KC)).to(tl.float32), [H, KC])
        lse = tl.load(lse_ptr + s * H + hh)
        dsum = tl.load(dsum_ptr + s * H + hh)
        dout = tl.reshape(tl.load(dout_ptr + s * (H * C) + tl.arange(0, H * C)).to(tl.float32), [H, C])
        dq = tl.zeros([H, KC], tl.float32)
        dw = tl.zeros([H, F], tl.float32)
        for p0 in range(start, end, BLOCK):
            p = p0 + tl.arange(0, BLOCK)
            msk = p < end
            bj = tl.load(bj_ptr + p, mask=msk, other=0).to(tl.int64)
            bas, ker = _kernel_term(basis_ptr, p, msk, w, F)
            k = tl.reshape(
                tl.load(
                    k_ptr + bj[:, None] * (H * KC) + tl.arange(0, H * KC)[None, :],
                    mask=msk[:, None], other=0.0,
                ).to(tl.float32),
                [BLOCK, H, KC],
            )
            score = tl.sum(k * q[None, :, :], axis=2)
            logm = tl.load(logm_ptr + p, mask=msk, other=0.0).to(tl.float32)
            z = ker + score * scale + logm[:, None]
            pr = tl.exp(z - lse[None, :])
            pr = tl.where(msk[:, None], pr, 0.0)  # (BLOCK, H)
            v = tl.reshape(
                tl.load(
                    v_ptr + bj[:, None] * (H * C) + tl.arange(0, H * C)[None, :],
                    mask=msk[:, None], other=0.0,
                ).to(tl.float32),
                [BLOCK, H, C],
            )
            dp = tl.sum(v * dout[None, :, :], axis=2)  # (BLOCK, H)
            dz = pr * (dp - dsum[None, :])  # (BLOCK, H)
            dq += tl.sum(dz[:, :, None] * k, axis=0) * scale
            ### The kernel-term gradient in the basis's dtype.
            dker = dz.to(basis_ptr.dtype.element_ty).to(tl.float32)
            if BASIS_GRAD:
                dt = dbasis_ptr.dtype.element_ty
                dbas = tl.sum(dker[:, None, :] * wt[None, :, :], axis=2).to(dt).to(tl.float32)  # (BLOCK, F)
                rows = dbasis_ptr + p[:, None] * F + ff[None, :]
                run = tl.load(rows, mask=msk[:, None], other=0.0).to(tl.float32)
                tl.store(rows, (run + dbas).to(dt), mask=msk[:, None])
            dw += tl.sum(tl.trans(dker)[:, None, :] * tl.trans(bas)[None, :, :], axis=2)
        tl.store(dq_ptr + s * (H * KC) + tl.arange(0, H * KC), tl.reshape(dq, [H * KC]))
        tl.store(dw_ptr + s * (H * F) + hh[:, None] * F + ff[None, :], dw)

    @triton.jit
    def _fkr_bwd_source_kernel(
        boff_ptr, item_src_ptr, item_chunk_ptr, perm_ptr, qib_ptr, basis_ptr, w_ptr, logm_ptr, q_ptr, k_ptr,
        v_ptr, lse_ptr, dsum_ptr, dout_ptr, dk_ptr, dv_ptr,
        scale, CHUNK, H: tl.constexpr, C: tl.constexpr, KC: tl.constexpr, F: tl.constexpr, BLOCK: tl.constexpr,
    ):  # fmt: skip
        """One chunk of one source's pairs (the plan's by-source order): its part of
        dk[j] and dv[j], added atomically."""
        item = tl.program_id(0).to(tl.int64)
        j = tl.load(item_src_ptr + item).to(tl.int64)
        chunk = tl.load(item_chunk_ptr + item).to(tl.int64)
        start = tl.load(boff_ptr + j) + chunk * CHUNK
        end = tl.minimum(tl.load(boff_ptr + j + 1), start + CHUNK)
        hh = tl.arange(0, H)
        w = _weights(w_ptr, H, F)
        kj = tl.reshape(tl.load(k_ptr + j * (H * KC) + tl.arange(0, H * KC)).to(tl.float32), [H, KC])
        vj = tl.reshape(tl.load(v_ptr + j * (H * C) + tl.arange(0, H * C)).to(tl.float32), [H, C])
        dk = tl.zeros([H, KC], tl.float32)
        dv = tl.zeros([H, C], tl.float32)
        for i0 in range(start, end, BLOCK):
            i = i0 + tl.arange(0, BLOCK)
            msk = i < end
            p = tl.load(perm_ptr + i, mask=msk, other=0).to(tl.int64)
            qi = tl.load(qib_ptr + i, mask=msk, other=0).to(tl.int64)
            _, ker = _kernel_term(basis_ptr, p, msk, w, F)
            q = tl.reshape(
                tl.load(
                    q_ptr + qi[:, None] * (H * KC) + tl.arange(0, H * KC)[None, :],
                    mask=msk[:, None], other=0.0,
                ).to(tl.float32),
                [BLOCK, H, KC],
            )
            score = tl.sum(q * kj[None, :, :], axis=2)
            logm = tl.load(logm_ptr + p, mask=msk, other=0.0).to(tl.float32)
            z = ker + score * scale + logm[:, None]
            lse = tl.load(lse_ptr + qi[:, None] * H + hh[None, :], mask=msk[:, None], other=0.0)
            dsum = tl.load(dsum_ptr + qi[:, None] * H + hh[None, :], mask=msk[:, None], other=0.0)
            pr = tl.exp(z - lse)
            pr = tl.where(msk[:, None], pr, 0.0)
            dout = tl.reshape(
                tl.load(
                    dout_ptr + qi[:, None] * (H * C) + tl.arange(0, H * C)[None, :],
                    mask=msk[:, None], other=0.0,
                ).to(tl.float32),
                [BLOCK, H, C],
            )
            dp = tl.sum(dout * vj[None, :, :], axis=2)
            dz = pr * (dp - dsum)
            dv += tl.sum(pr[:, :, None] * dout, axis=0)
            dk += tl.sum(dz[:, :, None] * q, axis=0) * scale
        tl.atomic_add(dk_ptr + j * (H * KC) + tl.arange(0, H * KC), tl.reshape(dk, [H * KC]))
        tl.atomic_add(dv_ptr + j * (H * C) + tl.arange(0, H * C), tl.reshape(dv, [H * C]))


class _BasisGrad:
    """The running sum of the blocks' basis gradients of one forward pass."""

    acc: torch.Tensor | None = None


class _BasisSink(torch.autograd.Function):
    """The basis's one consumer in autograd's eyes: an empty token every fused read takes,
    so its backward runs after every read's and hands the summed gradient to the basis."""

    @staticmethod
    def forward(ctx, basis, grad: _BasisGrad):
        ctx.grad = grad
        return basis.new_empty(0)

    @staticmethod
    def backward(ctx, _):
        acc, ctx.grad.acc = ctx.grad.acc, None
        return acc, None


def basis_sink(basis: torch.Tensor) -> tuple[torch.Tensor, _BasisGrad] | None:
    """The token and running sum through which every fused read of one forward pass
    passes its gradient to ``basis`` (``None`` when ``basis`` needs none): give it to
    each :func:`fused_kernel_read` of that basis."""
    if not (torch.is_grad_enabled() and basis.requires_grad):
        return None
    grad = _BasisGrad()
    return _BasisSink.apply(basis, grad), grad


class _FusedKernelRead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, basis, token, grad: _BasisGrad | None, w, logm, q, k, v, plan: PairPlan, scale: float):
        n_q, h, kc = q.shape
        c, f = v.shape[2], basis.shape[1]
        basis, w, logm, q, k, v = (t.contiguous() for t in (basis, w, logm, q, k, v))
        out = torch.empty(n_q, h, c, device=q.device, dtype=torch.float32)
        lse = torch.empty(n_q, h, device=q.device, dtype=torch.float32)
        if n_q:
            _fkr_fwd_kernel[(n_q,)](
                plan.q_offsets, plan.bj, basis, w, logm, q, k, v, out, lse, scale,
                H=h, C=c, KC=kc, F=f, BLOCK=BLOCK_PAIRS, num_warps=NUM_WARPS,
            )  # fmt: skip
        ctx.save_for_backward(basis, w, logm, q, k, v, out, lse)
        ctx.plan, ctx.scale, ctx.grad = plan, scale, grad
        return out

    @staticmethod
    def backward(ctx, dout):
        basis, w, logm, q, k, v, out, lse = ctx.saved_tensors
        plan, scale = ctx.plan, ctx.scale
        n_q, h, kc = q.shape
        c, f = v.shape[2], basis.shape[1]
        dout = dout.contiguous().float()
        dsum = (dout * out).sum(-1)  # (n_q, H)
        dq = torch.empty(n_q, h, kc, device=q.device, dtype=torch.float32)
        grad = ctx.grad
        if grad is not None and grad.acc is None:
            grad.acc = torch.zeros_like(basis)
        dbasis = grad.acc if grad is not None else basis
        dw_parts = torch.empty(n_q, h, f, device=q.device, dtype=torch.float32)
        dk = torch.zeros(plan.n_b, h, kc, device=q.device, dtype=torch.float32)
        dv = torch.zeros(plan.n_b, h, c, device=q.device, dtype=torch.float32)
        if n_q:
            _fkr_bwd_point_kernel[(n_q,)](
                plan.q_offsets, plan.bj, basis, w, logm, q, k, v, lse, dsum, dout, dq, dbasis, dw_parts, scale,
                H=h, C=c, KC=kc, F=f, BLOCK=BLOCK_PAIRS, BASIS_GRAD=grad is not None, num_warps=NUM_WARPS,
            )  # fmt: skip
        if plan.n_b:
            src, chunk_idx = _source_items(plan, SOURCE_CHUNK)
            if src.shape[0]:
                _fkr_bwd_source_kernel[(src.shape[0],)](
                    plan.b_offsets, src, chunk_idx, plan.perm, plan.qi_by_b, basis, w, logm, q, k, v, lse, dsum,
                    dout, dk, dv, scale, SOURCE_CHUNK,
                    H=h, C=c, KC=kc, F=f, BLOCK=BLOCK_PAIRS, num_warps=NUM_WARPS,
                )  # fmt: skip
        dw = dw_parts.sum(0).to(w.dtype)
        dtoken = None if grad is None else basis.new_empty(0)
        return None, dtoken, None, dw, None, dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None


def fused_kernel_read(
    basis: torch.Tensor,
    w: torch.Tensor,
    logm: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: PairPlan,
    scale: float,
    sink: tuple[torch.Tensor, _BasisGrad] | None = None,
) -> torch.Tensor:
    """``segment_softmax_read(basis @ w.T, logm, q, k, v, plan, scale)`` with the kernel
    term computed inside the read: ``basis`` ``(P, F)`` the pairs' kernel basis, ``w``
    ``(H, F)`` the block's kernel weights in the basis's dtype; the rest as
    :func:`~physicsnemo.experimental.models.isla.segment_softmax.segment_softmax_read`.
    Differentiable in ``basis``, ``w``, ``q``, ``k`` and ``v``; requires
    :func:`supported`. Reads of one basis share its :func:`basis_sink` (``sink``);
    without one, the read's basis gradient is its own."""
    if sink is None:
        sink = basis_sink(basis)
    token, grad = sink if sink is not None else (None, None)
    return _FusedKernelRead.apply(basis.detach(), token, grad, w, logm, q, k, v, plan, float(scale))
