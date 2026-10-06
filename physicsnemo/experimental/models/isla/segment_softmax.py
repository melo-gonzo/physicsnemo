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

r"""The tree read's softmax over each point's pair list, fused (Triton) or eager.

For every prediction point :math:`q` with pairs :math:`p` (a cell or a virtual cell
:math:`j_p` each) the read is

.. math::

    z_{ph} = \kappa_{ph} + \langle q_{qh}, k_{j_p h} \rangle / \sqrt{c} + \log m_p,
    \qquad
    \mathrm{read}_{qh} = \frac{\sum_p e^{z_{ph}} v_{j_p h}}{\sum_p e^{z_{ph}}},

with :math:`\kappa_{ph}` the pair's kernel term (its basis through the block's kernel
weights, a matmul outside this module) and :math:`\log m_p` the source's log measure.
Materialized pair by pair this is several ``(pairs, heads)`` tensors per block, and at
2e8 pairs on a car those tensors (and the gathers that build them) were most of a
training step. The Triton path here is the attention kernel over variable-length
segments: one program per point in forward (online softmax over tiles of its pairs,
the keys and values read in place), and in backward one program per point for the
query gradient and the kernel term's gradient, and one program per chunk of a source's
pairs (the pairs regrouped by source through the plan's permutation; a source read by
many points is split over several programs that add their parts atomically) for the
key and value gradients. Only the per-point output and log-sum-exp are kept between
forward and backward; every per-pair quantity is recomputed from the kernel term, the
logits' inputs and the plan. :func:`segment_softmax_read` dispatches; the eager form
serves CPU and float64.
"""

from __future__ import annotations

import math

import torch

from physicsnemo.experimental.models.isla.pair_kernels import (
    HAS_TRITON,
    PairPlan,
    supported,
)
from physicsnemo.mesh.spatial._ragged import _ragged_arange

#: Pairs per tile of a program's loop, warps per program, and pairs per program of
#: the by-source backward (launch constants).
BLOCK_PAIRS = 128
NUM_WARPS = 4
SOURCE_CHUNK = 2048

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _ssr_fwd_kernel(
        off_ptr, bj_ptr, ker_ptr, logm_ptr, q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
        scale, H: tl.constexpr, C: tl.constexpr, KC: tl.constexpr, BLOCK: tl.constexpr,
    ):  # fmt: skip
        """One point: online softmax over its pairs; out[s] (H, C), lse[s] (H)."""
        s = tl.program_id(0).to(tl.int64)
        start = tl.load(off_ptr + s)
        end = tl.load(off_ptr + s + 1)
        hh = tl.arange(0, H)
        q = tl.reshape(tl.load(q_ptr + s * (H * KC) + tl.arange(0, H * KC)).to(tl.float32), [H, KC])
        m_run = tl.full([H], float("-inf"), tl.float32)
        den = tl.zeros([H], tl.float32)
        acc = tl.zeros([H, C], tl.float32)
        for p0 in range(start, end, BLOCK):
            p = p0 + tl.arange(0, BLOCK)
            msk = p < end
            bj = tl.load(bj_ptr + p, mask=msk, other=0).to(tl.int64)
            ker = tl.load(ker_ptr + p[:, None] * H + hh[None, :], mask=msk[:, None], other=0.0).to(tl.float32)
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
    def _ssr_bwd_point_kernel(
        off_ptr, bj_ptr, ker_ptr, logm_ptr, q_ptr, k_ptr, v_ptr, lse_ptr, dsum_ptr, dout_ptr, dq_ptr, dker_ptr,
        scale, H: tl.constexpr, C: tl.constexpr, KC: tl.constexpr, BLOCK: tl.constexpr,
    ):  # fmt: skip
        """One point: dq[s] and the kernel-term gradient of its pairs."""
        s = tl.program_id(0).to(tl.int64)
        start = tl.load(off_ptr + s)
        end = tl.load(off_ptr + s + 1)
        hh = tl.arange(0, H)
        q = tl.reshape(tl.load(q_ptr + s * (H * KC) + tl.arange(0, H * KC)).to(tl.float32), [H, KC])
        lse = tl.load(lse_ptr + s * H + hh)
        dsum = tl.load(dsum_ptr + s * H + hh)
        dout = tl.reshape(tl.load(dout_ptr + s * (H * C) + tl.arange(0, H * C)).to(tl.float32), [H, C])
        dq = tl.zeros([H, KC], tl.float32)
        for p0 in range(start, end, BLOCK):
            p = p0 + tl.arange(0, BLOCK)
            msk = p < end
            bj = tl.load(bj_ptr + p, mask=msk, other=0).to(tl.int64)
            ker = tl.load(ker_ptr + p[:, None] * H + hh[None, :], mask=msk[:, None], other=0.0).to(tl.float32)
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
            tl.store(
                dker_ptr + p[:, None] * H + hh[None, :], dz.to(dker_ptr.dtype.element_ty), mask=msk[:, None]
            )
        tl.store(dq_ptr + s * (H * KC) + tl.arange(0, H * KC), tl.reshape(dq, [H * KC]))

    @triton.jit
    def _ssr_bwd_source_kernel(
        boff_ptr, item_src_ptr, item_chunk_ptr, perm_ptr, qib_ptr, ker_ptr, logm_ptr, q_ptr, k_ptr, v_ptr,
        lse_ptr, dsum_ptr, dout_ptr, dk_ptr, dv_ptr,
        scale, CHUNK, H: tl.constexpr, C: tl.constexpr, KC: tl.constexpr, BLOCK: tl.constexpr,
    ):  # fmt: skip
        """One chunk of one source's pairs (the plan's by-source order): its part of
        dk[j] and dv[j], added atomically."""
        item = tl.program_id(0).to(tl.int64)
        j = tl.load(item_src_ptr + item).to(tl.int64)
        chunk = tl.load(item_chunk_ptr + item).to(tl.int64)
        start = tl.load(boff_ptr + j) + chunk * CHUNK
        end = tl.minimum(tl.load(boff_ptr + j + 1), start + CHUNK)
        hh = tl.arange(0, H)
        kj = tl.reshape(tl.load(k_ptr + j * (H * KC) + tl.arange(0, H * KC)).to(tl.float32), [H, KC])
        vj = tl.reshape(tl.load(v_ptr + j * (H * C) + tl.arange(0, H * C)).to(tl.float32), [H, C])
        dk = tl.zeros([H, KC], tl.float32)
        dv = tl.zeros([H, C], tl.float32)
        for i0 in range(start, end, BLOCK):
            i = i0 + tl.arange(0, BLOCK)
            msk = i < end
            p = tl.load(perm_ptr + i, mask=msk, other=0).to(tl.int64)
            qi = tl.load(qib_ptr + i, mask=msk, other=0).to(tl.int64)
            ker = tl.load(ker_ptr + p[:, None] * H + hh[None, :], mask=msk[:, None], other=0.0).to(tl.float32)
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


def _source_items(plan: PairPlan, chunk: int):
    """The by-source work items: (source, chunk index) for every chunk of ``chunk``
    pairs of every source's segment."""
    counts = plan.b_offsets[1:] - plan.b_offsets[:-1]
    n_chunks = (counts + (chunk - 1)) // chunk
    chunk_idx, src = _ragged_arange(torch.zeros_like(n_chunks), n_chunks)
    return src, chunk_idx


class _SegmentSoftmaxRead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ker, logm, q, k, v, plan: PairPlan, scale: float):
        n_q, h, kc = q.shape
        c = v.shape[2]
        ker, logm, q, k, v = (t.contiguous() for t in (ker, logm, q, k, v))
        out = torch.empty(n_q, h, c, device=q.device, dtype=torch.float32)
        lse = torch.empty(n_q, h, device=q.device, dtype=torch.float32)
        if n_q:
            _ssr_fwd_kernel[(n_q,)](
                plan.q_offsets, plan.bj, ker, logm, q, k, v, out, lse, scale,
                H=h, C=c, KC=kc, BLOCK=BLOCK_PAIRS, num_warps=NUM_WARPS,
            )  # fmt: skip
        ctx.save_for_backward(ker, logm, q, k, v, out, lse)
        ctx.plan, ctx.scale = plan, scale
        return out

    @staticmethod
    def backward(ctx, dout):
        ker, logm, q, k, v, out, lse = ctx.saved_tensors
        plan, scale = ctx.plan, ctx.scale
        n_q, h, kc = q.shape
        c = v.shape[2]
        dout = dout.contiguous().float()
        dsum = (dout * out).sum(-1)  # (n_q, H)
        dq = torch.empty(n_q, h, kc, device=q.device, dtype=torch.float32)
        dker = torch.empty_like(ker)
        dk = torch.zeros(plan.n_b, h, kc, device=q.device, dtype=torch.float32)
        dv = torch.zeros(plan.n_b, h, c, device=q.device, dtype=torch.float32)
        if n_q:
            _ssr_bwd_point_kernel[(n_q,)](
                plan.q_offsets, plan.bj, ker, logm, q, k, v, lse, dsum, dout, dq, dker, scale,
                H=h, C=c, KC=kc, BLOCK=BLOCK_PAIRS, num_warps=NUM_WARPS,
            )  # fmt: skip
        if plan.n_b:
            src, chunk_idx = _source_items(plan, SOURCE_CHUNK)
            if src.shape[0]:
                _ssr_bwd_source_kernel[(src.shape[0],)](
                    plan.b_offsets, src, chunk_idx, plan.perm, plan.qi_by_b, ker, logm, q, k, v, lse, dsum, dout,
                    dk, dv, scale, SOURCE_CHUNK,
                    H=h, C=c, KC=kc, BLOCK=BLOCK_PAIRS, num_warps=NUM_WARPS,
                )  # fmt: skip
        return dker, None, dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None


def _eager_segment_softmax_read(ker, logm, q, k, v, plan: PairPlan, scale: float):
    """The same read pair by pair (CPU, float64, the reference of the Triton path)."""
    acc = torch.promote_types(q.dtype, torch.float32)
    qi, bj = plan.qi.long(), plan.bj.long()
    score = (q[qi].to(acc) * k[bj].to(acc)).sum(-1)
    logit = ker.to(acc) + score * scale + logm.to(acc)[:, None]
    counts = plan.q_offsets[1:] - plan.q_offsets[:-1]
    seg_max = torch.segment_reduce(logit.detach(), "max", lengths=counts, axis=0, unsafe=True)
    e = torch.exp(logit - seg_max[qi])
    den = e.new_zeros(plan.n_q, e.shape[1]).index_add(0, qi, e)
    num = e.new_zeros(plan.n_q, e.shape[1], v.shape[2]).index_add(0, qi, e[..., None] * v[bj].to(acc))
    return num / den.clamp_min(torch.finfo(acc).tiny)[..., None]


def segment_softmax_read(
    ker: torch.Tensor,
    logm: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: PairPlan,
    scale: float | None = None,
) -> torch.Tensor:
    """Each point's softmax-weighted read of its pairs' values, ``(n_q, H, C)`` in float32
    (float64 if the inputs are): ``ker`` ``(P, H)`` the pairs' kernel terms, ``logm``
    ``(P,)``, ``q`` ``(n_q, H, KC)``, ``k`` ``(n_b, H, KC)``, ``v`` ``(n_b, H, C)``;
    ``scale`` defaults to ``1 / sqrt(KC)``. Differentiable in ``ker``, ``q``, ``k`` and
    ``v``."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[2])
    h, kc, c = q.shape[1], q.shape[2], v.shape[2]
    pow2 = all(n & (n - 1) == 0 for n in (h, kc, c, h * kc, h * c))
    if supported(ker, q, k, v) and pow2 and q.dtype != torch.float64:
        return _SegmentSoftmaxRead.apply(ker, logm, q, k, v, plan, float(scale))
    return _eager_segment_softmax_read(ker, logm, q, k, v, plan, float(scale))
