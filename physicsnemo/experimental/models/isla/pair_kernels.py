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

r"""Segment kernels for the kernel read's sums over (point, cell) pairs.

The kernel read sums, for every prediction point, a weighted value row of every
boundary cell within the radius: ``out[q, h, :] = sum_p w[p, h] X[j_p, h, :]`` over
the point's pairs ``p``, and the mean read matches point and cell states per pair,
``s[p, h] = <A[q_p, h, :], B[j_p, h, :]>``. In eager PyTorch both gather a
``(pairs, heads, channels)`` tensor and, in backward, scatter one with atomics (and a
boundary of a thousand cells read by millions of pairs serializes those atomics). The
Triton kernels here read the rows in place and write only the results:

- :func:`segment_weighted_sum`: one program per segment of a sorted pair list
  (pairs grouped by point, or by cell through a permutation), looping over its pairs;
  no atomics, deterministic.
- :func:`pair_dot`: one program per tile of pairs.

:class:`PairPlan` holds the two orderings of one pair list; :func:`pair_weighted_sum`
and :func:`pair_scores` are the differentiable operations (each backward is the other
two kernels). CUDA only; :func:`supported` says when they apply, and the eager
reference (:func:`_eager_weighted_sum`, :func:`_eager_scores`) serves everything else.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import torch

from physicsnemo.core.version_check import is_package_available

HAS_TRITON = is_package_available("triton")
if HAS_TRITON:
    try:
        triton = importlib.import_module("triton")
        tl = importlib.import_module("triton.language")
    except Exception:  # pragma: no cover - a broken triton install is treated as absent
        HAS_TRITON = False

BLOCK_P = 32


@dataclass
class PairPlan:
    """A pair list ``(qi, bj)`` grouped by point, with its regrouping by cell.

    ``q_offsets`` ``(n_q + 1,)`` delimit each point's pairs; ``perm`` orders the pairs
    by cell (stable) and ``b_offsets`` ``(n_b + 1,)`` delimit each cell's pairs in that
    order; ``qi_by_b = qi[perm]``.
    """

    qi: torch.Tensor
    bj: torch.Tensor
    n_q: int
    n_b: int
    q_offsets: torch.Tensor
    perm: torch.Tensor
    b_offsets: torch.Tensor
    qi_by_b: torch.Tensor

    @classmethod
    def build(
        cls, qi: torch.Tensor, bj: torch.Tensor, n_q: int, n_b: int
    ) -> "PairPlan":
        """Index the pair list (``qi`` non-decreasing) by point and by cell."""

        def offsets(idx, n):
            o = idx.new_zeros(n + 1)
            o[1:] = torch.cumsum(torch.bincount(idx, minlength=n), 0)
            return o

        perm = torch.argsort(bj, stable=True)
        return cls(qi, bj, n_q, n_b, offsets(qi, n_q), perm, offsets(bj, n_b), qi[perm])


def supported(*tensors: torch.Tensor) -> bool:
    """The Triton kernels apply: CUDA tensors and Triton importable."""
    return HAS_TRITON and all(t.is_cuda for t in tensors)


if HAS_TRITON:

    @triton.jit
    def _segment_sum_kernel(
        off_ptr, idx_ptr, w_ptr, x_ptr, out_ptr,
        H: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr, F64: tl.constexpr,
    ):  # fmt: skip
        """out[s, h, c] = sum over the pairs p of segment s of w[p, h] * x[idx[p], h, c]."""
        s = tl.program_id(0)
        start = tl.load(off_ptr + s)
        end = tl.load(off_ptr + s + 1)
        hc = tl.arange(0, H * C)
        head = hc // C
        ACC = tl.float64 if F64 else tl.float32
        acc = tl.zeros([H * C], dtype=ACC)
        for p0 in range(start, end, BLOCK):
            p = p0 + tl.arange(0, BLOCK)
            m = p < end
            row = tl.load(idx_ptr + p, mask=m, other=0)
            w = tl.load(
                w_ptr + p[:, None] * H + head[None, :], mask=m[:, None], other=0.0
            )
            x = tl.load(
                x_ptr + row[:, None] * (H * C) + hc[None, :], mask=m[:, None], other=0.0
            )
            acc += tl.sum(w.to(ACC) * x.to(ACC), axis=0)
        tl.store(out_ptr + s * (H * C) + hc, acc.to(out_ptr.dtype.element_ty))

    @triton.jit
    def _pair_dot_kernel(
        ia_ptr, ib_ptr, a_ptr, b_ptr, out_ptr, P,
        H: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr, F64: tl.constexpr,
    ):  # fmt: skip
        """out[p, h] = sum_c a[ia[p], h, c] * b[ib[p], h, c]."""
        ### 64-bit pair offsets: ``p * H`` passes 2**31 above 2.7e8 pairs (the tree
        ### read at DrivAerML scale), where 32-bit offsets wrapped and read garbage.
        p = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = p < P
        ia = tl.load(ia_ptr + p, mask=m, other=0)
        ib = tl.load(ib_ptr + p, mask=m, other=0)
        hc = tl.arange(0, H * C)
        a = tl.load(
            a_ptr + ia[:, None] * (H * C) + hc[None, :], mask=m[:, None], other=0.0
        )
        b = tl.load(
            b_ptr + ib[:, None] * (H * C) + hc[None, :], mask=m[:, None], other=0.0
        )
        ACC = tl.float64 if F64 else tl.float32
        prod = tl.reshape(a.to(ACC) * b.to(ACC), [BLOCK, H, C])
        res = tl.sum(prod, axis=2)
        hh = tl.arange(0, H)
        tl.store(
            out_ptr + p[:, None] * H + hh[None, :],
            res.to(out_ptr.dtype.element_ty),
            mask=m[:, None],
        )


def segment_weighted_sum(offsets, idx, w, x, n_seg: int, out_dtype) -> torch.Tensor:
    """``out[s, h, :] = sum_{p in segment s} w[p, h] x[idx[p], h, :]`` (Triton)."""
    _, h, c = x.shape
    out = torch.empty(n_seg, h, c, device=x.device, dtype=out_dtype)
    if n_seg:
        _segment_sum_kernel[(n_seg,)](
            offsets, idx, w.contiguous(), x.contiguous(), out, H=h, C=c, BLOCK=BLOCK_P,
            F64=torch.float64 in (w.dtype, x.dtype, out_dtype),
        )  # fmt: skip
    return out


def pair_dot(ia, ib, a, b, out_dtype) -> torch.Tensor:
    """``out[p, h] = <a[ia[p], h, :], b[ib[p], h, :]>`` (Triton)."""
    p = ia.shape[0]
    h, c = a.shape[1], a.shape[2]
    out = torch.empty(p, h, device=a.device, dtype=out_dtype)
    if p:
        _pair_dot_kernel[(triton.cdiv(p, BLOCK_P),)](
            ia, ib, a.contiguous(), b.contiguous(), out, p, H=h, C=c, BLOCK=BLOCK_P,
            F64=torch.float64 in (a.dtype, b.dtype, out_dtype),
        )  # fmt: skip
    return out


class _PairWeightedSum(torch.autograd.Function):
    """``out[q] = sum_{p of q} w[p, h] x[bj[p], h, :]`` with Triton forward and backward."""

    @staticmethod
    def forward(ctx, w, x, plan: PairPlan):
        ctx.plan = plan
        ctx.save_for_backward(w, x)
        return segment_weighted_sum(plan.q_offsets, plan.bj, w, x, plan.n_q, w.dtype)

    @staticmethod
    def backward(ctx, g):
        w, x = ctx.saved_tensors
        plan = ctx.plan
        g = g.contiguous()
        gw = (
            pair_dot(plan.qi, plan.bj, g, x, w.dtype)
            if ctx.needs_input_grad[0]
            else None
        )
        gx = None
        if ctx.needs_input_grad[1]:
            gx = segment_weighted_sum(
                plan.b_offsets,
                plan.qi_by_b,
                w[plan.perm],
                g,
                plan.n_b,
                torch.promote_types(x.dtype, torch.float32),
            ).to(x.dtype)
        return gw, gx, None


class _PairScores(torch.autograd.Function):
    """``s[p, h] = <a[qi[p], h, :], b[bj[p], h, :]>`` with Triton forward and backward."""

    @staticmethod
    def forward(ctx, a, b, plan: PairPlan, out_dtype):
        ctx.plan = plan
        ctx.save_for_backward(a, b)
        return pair_dot(plan.qi, plan.bj, a, b, out_dtype)

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_tensors
        plan = ctx.plan
        g = g.contiguous()
        ga = gb = None
        if ctx.needs_input_grad[0]:
            ga = segment_weighted_sum(
                plan.q_offsets,
                plan.bj,
                g,
                b,
                plan.n_q,
                torch.promote_types(a.dtype, torch.float32),
            ).to(a.dtype)
        if ctx.needs_input_grad[1]:
            gb = segment_weighted_sum(
                plan.b_offsets,
                plan.qi_by_b,
                g[plan.perm],
                a,
                plan.n_b,
                torch.promote_types(b.dtype, torch.float32),
            ).to(b.dtype)
        return ga, gb, None, None


def _eager_weighted_sum(w, x, plan: PairPlan) -> torch.Tensor:
    out = w.new_zeros(plan.n_q, x.shape[1], x.shape[2])
    return out.index_add(0, plan.qi, w[..., None] * x[plan.bj].to(w.dtype))


def _eager_scores(a, b, plan: PairPlan, out_dtype) -> torch.Tensor:
    return (a[plan.qi].to(out_dtype) * b[plan.bj].to(out_dtype)).sum(-1)


def pair_weighted_sum(w: torch.Tensor, x: torch.Tensor, plan: PairPlan) -> torch.Tensor:
    """Per point, the ``w``-weighted sum of its cells' rows: ``(n_q, H, C)`` in ``w``'s
    dtype from ``w`` ``(P, H)`` and ``x`` ``(n_b, H, C)``. ``H * C`` must be a power of
    two on the Triton path."""
    hc = x.shape[1] * x.shape[2]
    if supported(w, x) and hc & (hc - 1) == 0:
        return _PairWeightedSum.apply(w, x, plan)
    return _eager_weighted_sum(w, x, plan)


def pair_scores(
    a: torch.Tensor, b: torch.Tensor, plan: PairPlan, out_dtype
) -> torch.Tensor:
    """Per pair and head, the dot product of the point's row of ``a`` ``(n_q, H, C)``
    with the cell's row of ``b`` ``(n_b, H, C)``: ``(P, H)``."""
    hc = a.shape[1] * a.shape[2]
    if supported(a, b) and hc & (hc - 1) == 0:
        return _PairScores.apply(a, b, plan, out_dtype)
    return _eager_scores(a, b, plan, out_dtype)
