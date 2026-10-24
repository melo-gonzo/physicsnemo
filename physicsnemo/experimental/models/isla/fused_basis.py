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

r"""The kernel basis with its pair invariants fused in (Triton): nothing per pair is
kept between forward and backward but the basis the fused read consumes.

The reference (:func:`~physicsnemo.experimental.models.isla.tree_read._chunked_basis_from_plan`)
builds each chunk's pair invariants (:func:`~physicsnemo.experimental.models.isla.tree_read.tree_pair_invariants`)
and runs the basis MLP (:class:`~physicsnemo.experimental.models.isla.tree_read._KernelBasis`)
on them under a checkpoint, a dozen tensor ops per chunk and per layer, each a pass
over every pair. Here one kernel computes, per tile of pairs, the invariants from the
plan's indices and the points' and sources' geometry, the three layers and the basis;
the backward kernel rebuilds them per tile and sums the MLP's weight and bias
gradients.

The arithmetic is the reference's but for summation order. The invariants are float32
with IEEE division and square root and libdevice's logarithm, each sum of two or three
products in the order of PyTorch's reductions. Each linear layer is a dot product of
operands in the MLP's dtype accumulated in float32, plus its bias, rounded to that
dtype: under autocast the invariants, weights and biases are rounded to bfloat16 (or
float16) and every layer's output too, as autocast's GEMMs do; in float32 the dots are
IEEE (``input_precision="ieee"``, no TF32). GELU is the erf form computed in float32
from its rounded input and rounded to the MLP's dtype, forward and backward, as
PyTorch's CUDA kernels compute it. Backward rounds every layer's input gradient to the
MLP's dtype, as the reference GEMMs' outputs are; each chunk's weight and bias
gradients are float32 sums over its pairs rounded once to that dtype, and autograd sums
the chunks' through one cast of the weights per forward pass (:func:`mlp_weights`), as
it sums the reference's through autocast's cached cast.
"""

from __future__ import annotations

import importlib

import torch
from torch.autograd.function import once_differentiable

from physicsnemo.experimental.models.isla import fused_read

HAS_TRITON = fused_read.HAS_TRITON
if HAS_TRITON:
    triton, tl = fused_read.triton, fused_read.tl
    try:
        libdevice = importlib.import_module("triton.language.extra.libdevice")
    except Exception:  # pragma: no cover - a Triton without libdevice is treated as absent
        HAS_TRITON = False

#: Pairs per tile and warps per program of the basis kernels, and the backward's
#: programs per multiprocessor (each sums the weight gradients of its tiles); launch
#: constants.
BASIS_BLOCK = 64
BASIS_WARPS = 4
BACKWARD_PROGRAMS_PER_SM = 4


def _pow2(n: int) -> bool:
    return n >= 16 and n & (n - 1) == 0


def supported(basis: torch.nn.Module, geometry: dict) -> bool:
    """The fused basis applies to ``basis`` (a ``_KernelBasis``) on one case's
    ``geometry`` (the keyword arguments of ``tree_pair_invariants``): Triton importable,
    CUDA float32 geometry and parameters, two or three dimensions, power-of-two hidden
    and basis widths, and either no autocast or a bfloat16 or float16 autocast with its
    weight cache on (the reference then casts the weights once per forward pass)."""
    if not HAS_TRITON:
        return False
    lin = (basis.net[0], basis.net[2], basis.net[4])
    params = [t for m in lin for t in (m.weight, m.bias)]
    geo = [geometry[k] for k in ("query_r", "query_n", "query_wall", "src_r", "src_n", "g_unit")]
    if geometry["src_vecs"] is not None:
        geo.append(geometry["src_vecs"])
    if not all(t.is_cuda and t.dtype == torch.float32 for t in geo + params):
        return False
    if torch.is_autocast_enabled("cuda") and not (
        torch.get_autocast_dtype("cuda") in (torch.bfloat16, torch.float16) and torch.is_autocast_cache_enabled()
    ):
        return False
    return geometry["query_r"].shape[1] in (2, 3) and _pow2(lin[0].out_features) and _pow2(lin[2].out_features)


def mlp_weights(basis: torch.nn.Module) -> tuple[torch.Tensor, ...]:
    """The basis MLP's weights and biases ``(W1, b1, W2, b2, W3, b3)`` as its layers
    read them: under autocast on their device cast to its dtype, once per forward pass
    (autocast caches one cast of each for every chunk), else the parameters themselves."""
    params = [t for m in (basis.net[0], basis.net[2], basis.net[4]) for t in (m.weight, m.bias)]
    dev = params[0].device.type
    if torch.is_autocast_enabled(dev):
        dt = torch.get_autocast_dtype(dev)
        params = [t.to(dt) for t in params]
    return tuple(params)


if HAS_TRITON:

    @triton.jit
    def _sum3(a, b, c, D: tl.constexpr):
        """A sum over a last axis of D = 2 or 3 rounded terms in the order of PyTorch's
        CUDA reduction (two threads: the first takes terms 0 and 2)."""
        if D == 3:
            return (a + c) + b
        else:
            return a + b

    @triton.jit
    def _mm3(ax, ay, az, bx, by, bz, D: tl.constexpr):
        """A row of a float32 matmul with an inner dimension of D = 2 or 3: a chain of
        fused multiply-adds from the first term, as cuBLAS's."""
        acc = tl.fma(ay, by, ax * bx)
        if D == 3:
            acc = tl.fma(az, bz, acc)
        return acc

    @triton.jit
    def _load3(ptr, row, stride, msk, D: tl.constexpr):
        """The D components of rows ``row`` of a float32 ``(.., D)`` table (the third
        zero when D = 2)."""
        x = tl.load(ptr + row * stride, mask=msk, other=0.0)
        y = tl.load(ptr + row * stride + 1, mask=msk, other=0.0)
        z = tl.zeros_like(x)
        if D == 3:
            z = tl.load(ptr + row * stride + 2, mask=msk, other=0.0)
        return x, y, z

    @triton.jit
    def _invariants(
        qi, si, msk, qr_ptr, qn_ptr, qw_ptr, sr_ptr, sn_ptr, sv_ptr, g_ptr, eps,
        D: tl.constexpr, K: tl.constexpr, PB: tl.constexpr, GP: tl.constexpr, BLOCK: tl.constexpr,
    ):  # fmt: skip
        """The pairs' invariants ``(BLOCK, GP)`` float32, ``tree_pair_invariants``'s
        columns then zeros (and zero rows where ``msk`` is off)."""
        qx, qy, qz = _load3(qr_ptr, qi, D, msk, D)
        sx, sy, sz = _load3(sr_ptr, si, D, msk, D)
        rx, ry, rz = qx - sx, qy - sy, qz - sz
        dist = tl.sqrt_rn(_sum3(rx * rx, ry * ry, rz * rz, D))
        dist_g = tl.maximum(dist, eps)
        hx, hy, hz = tl.div_rn(rx, dist_g), tl.div_rn(ry, dist_g), tl.div_rn(rz, dist_g)
        ax, ay, az = _load3(qn_ptr, qi, D, msk, D)
        jx, jy, jz = _load3(sn_ptr, si, D, msk, D)
        log_r = libdevice.log(dist_g)
        log_w = libdevice.log(tl.maximum(tl.load(qw_ptr + qi, mask=msk, other=1.0), eps))
        cols = tl.arange(0, GP)[None, :]
        inv = tl.zeros([BLOCK, GP], tl.float32)
        inv = tl.where(cols == 0, dist[:, None], inv)
        inv = tl.where(cols == 1, log_r[:, None], inv)
        inv = tl.where(cols == 2, (log_r - log_w)[:, None], inv)
        inv = tl.where(cols == 3, _sum3(hx * jx, hy * jy, hz * jz, D)[:, None], inv)
        inv = tl.where(cols == 4, _sum3(hx * ax, hy * ay, hz * az, D)[:, None], inv)
        inv = tl.where(cols == 5, _sum3(ax * jx, ay * jy, az * jz, D)[:, None], inv)
        for k in tl.static_range(K):
            gx = tl.load(g_ptr + k * D)
            gy = tl.load(g_ptr + k * D + 1)
            gz = 0.0
            if D == 3:
                gz = tl.load(g_ptr + k * D + 2)
            inv = tl.where(cols == 6 + k, _mm3(hx, hy, hz, gx, gy, gz, D)[:, None], inv)
            inv = tl.where(cols == 6 + K + k, _mm3(jx, jy, jz, gx, gy, gz, D)[:, None], inv)
        for u in tl.static_range(PB):
            ux, uy, uz = _load3(sv_ptr + u * D, si, PB * D, msk, D)
            inv = tl.where(cols == 6 + 2 * K + u, _sum3(hx * ux, hy * uy, hz * uz, D)[:, None], inv)
            inv = tl.where(cols == 6 + 2 * K + PB + u, _sum3(ax * ux, ay * uy, az * uz, D)[:, None], inv)
        return tl.where(msk[:, None], inv, 0.0)

    @triton.jit
    def _dot(a, b, acc, NARROW: tl.constexpr):
        """``acc + a @ b`` in float32: exact products of narrow operands on the tensor
        cores, IEEE float32 products otherwise."""
        if NARROW:
            return tl.dot(a, b, acc)
        else:
            return tl.dot(a, b, acc, input_precision="ieee")

    @triton.jit
    def _linear(x, w_ptr, b_ptr, N_IN: tl.constexpr, IN: tl.constexpr, OUT: tl.constexpr, NARROW: tl.constexpr):
        """``x @ W.T + b`` in float32 for ``x`` ``(BLOCK, IN)`` and ``W`` ``(OUT, N_IN)``
        (``N_IN <= IN``, the rest zero), both in the MLP's dtype."""
        ii = tl.arange(0, IN)
        oo = tl.arange(0, OUT)
        wt = tl.load(w_ptr + oo[None, :] * N_IN + ii[:, None], mask=(ii < N_IN)[:, None], other=0.0)
        acc = _dot(x, wt, tl.zeros([x.shape[0], OUT], tl.float32), NARROW)
        return acc + tl.load(b_ptr + oo).to(tl.float32)[None, :]

    @triton.jit
    def _input_grad(dy, w_ptr, OUT: tl.constexpr, IN: tl.constexpr, NARROW: tl.constexpr):
        """``dy @ W`` in float32 for ``dy`` ``(BLOCK, OUT)`` and ``W`` ``(OUT, IN)``."""
        w = tl.load(w_ptr + tl.arange(0, OUT)[:, None] * IN + tl.arange(0, IN)[None, :])
        return _dot(dy, w, tl.zeros([dy.shape[0], IN], tl.float32), NARROW)

    @triton.jit
    def _gelu(x):
        """GELU (erf form) of float32 ``x``, as PyTorch's CUDA kernel."""
        return x * 0.5 * (1.0 + libdevice.erf(x * 0.7071067690849304))

    @triton.jit
    def _gelu_grad(dy, x):
        """GELU's backward at float32 ``x``, as PyTorch's CUDA kernel (its last
        multiply-add fused)."""
        cdf = 0.5 * (1.0 + libdevice.erf(x * 0.7071067690849304))
        pdf = libdevice.exp(-0.5 * x * x) * 0.3989422917366028
        return dy * tl.fma(x, pdf, cdf)

    @triton.jit
    def _hidden(inv, w1_ptr, b1_ptr, w2_ptr, b2_ptr, G: tl.constexpr, GP: tl.constexpr, HID: tl.constexpr,
                NARROW: tl.constexpr):  # fmt: skip
        """The MLP's input and hidden activations in its dtype: the rounded invariants,
        the first layer's output and GELU, the second's and GELU."""
        dt = w1_ptr.dtype.element_ty
        x0 = inv.to(dt)
        h1 = _linear(x0, w1_ptr, b1_ptr, G, GP, HID, NARROW).to(dt)
        a1 = _gelu(h1.to(tl.float32)).to(dt)
        h2 = _linear(a1, w2_ptr, b2_ptr, HID, HID, HID, NARROW).to(dt)
        a2 = _gelu(h2.to(tl.float32)).to(dt)
        return x0, h1, a1, h2, a2

    @triton.jit
    def _pair_rows(qi_ptr, si_ptr, tile, n, q_off, s_off, BLOCK: tl.constexpr):
        """The pair rows of tile ``tile`` (int64), their mask, and the case's point and
        source of each."""
        p = tile * BLOCK + tl.arange(0, BLOCK)
        msk = p < n
        qi = tl.where(msk, tl.load(qi_ptr + p, mask=msk, other=0).to(tl.int64) - q_off, 0)
        si = tl.where(msk, tl.load(si_ptr + p, mask=msk, other=0).to(tl.int64) - s_off, 0)
        return p, msk, qi, si

    @triton.jit
    def _inv_kernel(
        qi_ptr, si_ptr, n, q_off, s_off, qr_ptr, qn_ptr, qw_ptr, sr_ptr, sn_ptr, sv_ptr, g_ptr, out_ptr, eps,
        D: tl.constexpr, K: tl.constexpr, PB: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, BLOCK: tl.constexpr,
    ):  # fmt: skip
        """One tile's invariants ``(BLOCK, G)`` float32 (:func:`pair_invariants`)."""
        p, msk, qi, si = _pair_rows(qi_ptr, si_ptr, tl.program_id(0).to(tl.int64), n, q_off, s_off, BLOCK)
        inv = _invariants(qi, si, msk, qr_ptr, qn_ptr, qw_ptr, sr_ptr, sn_ptr, sv_ptr, g_ptr, eps, D, K, PB, GP, BLOCK)
        cc = tl.arange(0, GP)
        tl.store(out_ptr + p[:, None] * G + cc[None, :], inv, mask=msk[:, None] & (cc < G)[None, :])

    @triton.jit
    def _basis_fwd_kernel(
        qi_ptr, si_ptr, n, q_off, s_off, qr_ptr, qn_ptr, qw_ptr, sr_ptr, sn_ptr, sv_ptr, g_ptr,
        w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, b3_ptr, out_ptr, eps,
        D: tl.constexpr, K: tl.constexpr, PB: tl.constexpr, G: tl.constexpr, GP: tl.constexpr,
        HID: tl.constexpr, F: tl.constexpr, BLOCK: tl.constexpr, NARROW: tl.constexpr,
    ):  # fmt: skip
        """One tile of pairs: invariants, MLP, basis ``(BLOCK, F)`` in the MLP's dtype."""
        p, msk, qi, si = _pair_rows(qi_ptr, si_ptr, tl.program_id(0).to(tl.int64), n, q_off, s_off, BLOCK)
        inv = _invariants(qi, si, msk, qr_ptr, qn_ptr, qw_ptr, sr_ptr, sn_ptr, sv_ptr, g_ptr, eps, D, K, PB, GP, BLOCK)
        _, _, _, _, a2 = _hidden(inv, w1_ptr, b1_ptr, w2_ptr, b2_ptr, G, GP, HID, NARROW)
        out = _linear(a2, w3_ptr, b3_ptr, HID, HID, F, NARROW).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + p[:, None] * F + tl.arange(0, F)[None, :], out, mask=msk[:, None])

    @triton.jit
    def _basis_bwd_kernel(
        qi_ptr, si_ptr, n, n_tiles, q_off, s_off, qr_ptr, qn_ptr, qw_ptr, sr_ptr, sn_ptr, sv_ptr, g_ptr,
        w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, b3_ptr, dout_ptr, part_ptr, eps,
        D: tl.constexpr, K: tl.constexpr, PB: tl.constexpr, G: tl.constexpr, GP: tl.constexpr,
        HID: tl.constexpr, F: tl.constexpr, BLOCK: tl.constexpr, NARROW: tl.constexpr,
    ):  # fmt: skip
        """Tiles ``pid, pid + programs, ...``: the MLP rebuilt, its backward from the
        basis gradient, and the program's float32 sums of the weight and bias gradients
        to ``part_ptr[pid]`` (W1 ``(HID, GP)``, b1, W2, b2, W3 ``(F, HID)``, b3)."""
        pid = tl.program_id(0).to(tl.int64)
        dt = w1_ptr.dtype.element_ty
        hh = tl.arange(0, HID)
        ff = tl.arange(0, F)
        dw1 = tl.zeros([HID, GP], tl.float32)
        dw2 = tl.zeros([HID, HID], tl.float32)
        dw3 = tl.zeros([F, HID], tl.float32)
        db1 = tl.zeros([HID], tl.float32)
        db2 = tl.zeros([HID], tl.float32)
        db3 = tl.zeros([F], tl.float32)
        for tile in range(pid, n_tiles, tl.num_programs(0)):
            p, msk, qi, si = _pair_rows(qi_ptr, si_ptr, tile, n, q_off, s_off, BLOCK)
            inv = _invariants(
                qi, si, msk, qr_ptr, qn_ptr, qw_ptr, sr_ptr, sn_ptr, sv_ptr, g_ptr, eps, D, K, PB, GP, BLOCK
            )
            x0, h1, a1, h2, a2 = _hidden(inv, w1_ptr, b1_ptr, w2_ptr, b2_ptr, G, GP, HID, NARROW)
            dy = tl.load(dout_ptr + p[:, None] * F + ff[None, :], mask=msk[:, None], other=0.0)
            dw3 = _dot(tl.trans(dy), a2, dw3, NARROW)
            db3 += tl.sum(dy.to(tl.float32), axis=0)
            da2 = _input_grad(dy, w3_ptr, F, HID, NARROW).to(dt)
            dh2 = _gelu_grad(da2.to(tl.float32), h2.to(tl.float32)).to(dt)
            dw2 = _dot(tl.trans(dh2), a1, dw2, NARROW)
            db2 += tl.sum(dh2.to(tl.float32), axis=0)
            da1 = _input_grad(dh2, w2_ptr, HID, HID, NARROW).to(dt)
            dh1 = _gelu_grad(da1.to(tl.float32), h1.to(tl.float32)).to(dt)
            dw1 = _dot(tl.trans(dh1), x0, dw1, NARROW)
            db1 += tl.sum(dh1.to(tl.float32), axis=0)
        part = part_ptr + pid * (HID * GP + 2 * HID + HID * HID + F * HID + F)
        tl.store(part + hh[:, None] * GP + tl.arange(0, GP)[None, :], dw1)
        part += HID * GP
        tl.store(part + hh, db1)
        part += HID
        tl.store(part + hh[:, None] * HID + hh[None, :], dw2)
        part += HID * HID
        tl.store(part + hh, db2)
        part += HID
        tl.store(part + ff[:, None] * HID + hh[None, :], dw3)
        part += F * HID
        tl.store(part + ff, db3)


_SM_COUNT: dict = {}


def _sm_count(device: torch.device) -> int:
    key = device.index if device.index is not None else torch.cuda.current_device()
    if key not in _SM_COUNT:
        _SM_COUNT[key] = torch.cuda.get_device_properties(key).multi_processor_count
    return _SM_COUNT[key]


def _meta(qr, gu, pb, w1, w3) -> dict:
    hid, g = w1.shape
    return dict(
        D=int(qr.shape[1]), K=int(gu.shape[0]), PB=int(pb), G=int(g), GP=max(16, triton.next_power_of_2(int(g))),
        HID=int(hid), F=int(w3.shape[0]),
    )  # fmt: skip


def _geometry_args(geometry: dict) -> tuple[tuple[torch.Tensor, ...], int]:
    """The case's geometry tensors in the kernels' order, contiguous, and the number
    of rank-1 inputs per source (``src_vecs`` a placeholder when there are none)."""
    sv = geometry["src_vecs"]
    pb = 0 if sv is None else int(sv.shape[1])
    keys = ("query_r", "query_n", "query_wall", "src_r", "src_n")
    geo = tuple(geometry[k].contiguous() for k in keys)
    return geo + ((sv if pb else geometry["src_r"]).contiguous(), geometry["g_unit"].contiguous()), pb


class _FusedBasis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qi, si, q_off, s_off, eps, pb, qr, qn, qw, sr, sn, sv, gu, w1, b1, w2, b2, w3, b3):
        meta = _meta(qr, gu, pb, w1, w3)
        if meta["G"] != 6 + 2 * meta["K"] + 2 * pb:
            raise ValueError(f"the basis MLP takes {meta['G']} invariants; the geometry gives {6 + 2 * meta['K'] + 2 * pb}")
        meta["NARROW"] = w1.dtype in (torch.bfloat16, torch.float16)
        n = int(qi.shape[0])
        out = torch.empty(n, meta["F"], device=qi.device, dtype=w1.dtype)
        if n:
            _basis_fwd_kernel[(triton.cdiv(n, BASIS_BLOCK),)](
                qi, si, n, q_off, s_off, qr, qn, qw, sr, sn, sv, gu, w1, b1, w2, b2, w3, b3, out, eps,
                BLOCK=BASIS_BLOCK, num_warps=BASIS_WARPS, enable_fp_fusion=False, **meta,
            )  # fmt: skip
        ctx.save_for_backward(qi, si, qr, qn, qw, sr, sn, sv, gu, w1, b1, w2, b2, w3, b3)
        ctx.args = (q_off, s_off, eps, meta)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        qi, si, qr, qn, qw, sr, sn, sv, gu, w1, b1, w2, b2, w3, b3 = ctx.saved_tensors
        q_off, s_off, eps, meta = ctx.args
        hid, g, gp, f = meta["HID"], meta["G"], meta["GP"], meta["F"]
        sizes = (hid * gp, hid, hid * hid, hid, f * hid, f)
        n = int(qi.shape[0])
        n_tiles = triton.cdiv(n, BASIS_BLOCK)
        progs = max(1, min(n_tiles, BACKWARD_PROGRAMS_PER_SM * _sm_count(qi.device)))
        parts = torch.zeros(progs, sum(sizes), device=qi.device, dtype=torch.float32)
        if n:
            _basis_bwd_kernel[(progs,)](
                qi, si, n, n_tiles, q_off, s_off, qr, qn, qw, sr, sn, sv, gu, w1, b1, w2, b2, w3, b3,
                dout.contiguous(), parts, eps, BLOCK=BASIS_BLOCK, num_warps=BASIS_WARPS, enable_fp_fusion=False, **meta,
            )  # fmt: skip
        dw1, db1, dw2, db2, dw3, db3 = parts.sum(0).split(sizes)
        ### Each a float32 sum over the chunk's pairs, rounded once, as the reference GEMMs' are.
        grads = (dw1.view(hid, gp)[:, :g], db1, dw2.view(hid, hid), db2, dw3.view(f, hid), db3)
        return (None,) * 13 + tuple(t.to(w1.dtype) for t in grads)


def fused_kernel_basis(
    weights: tuple[torch.Tensor, ...], qi: torch.Tensor, si: torch.Tensor, q_off: int, s_off: int, geometry: dict
) -> torch.Tensor:
    """``_KernelBasis`` of ``tree_pair_invariants(qi - q_off, si - s_off, **geometry)``
    ``(P, F)`` in the MLP's dtype, the invariants and MLP in one kernel: ``weights`` from
    :func:`mlp_weights`, ``qi`` and ``si`` the pairs' points and sources in the plan's
    numbering (int32 or int64), ``q_off`` and ``s_off`` the case's first point and
    source. Differentiable in ``weights`` (the geometry gets no gradient, as in the
    reference); requires :func:`supported`."""
    geo, pb = _geometry_args(geometry)
    return _FusedBasis.apply(qi, si, int(q_off), int(s_off), float(geometry["eps"]), pb, *geo, *weights)


def pair_invariants(qi: torch.Tensor, si: torch.Tensor, q_off: int, s_off: int, geometry: dict) -> torch.Tensor:
    """The fused kernels' invariants ``(P, G)`` float32 of the same pairs, for tests."""
    geo, pb = _geometry_args(geometry)
    k = int(geometry["g_unit"].shape[0])
    g = 6 + 2 * k + 2 * pb
    n = int(qi.shape[0])
    out = torch.empty(n, g, device=qi.device, dtype=torch.float32)
    if n:
        _inv_kernel[(triton.cdiv(n, BASIS_BLOCK),)](
            qi, si, n, int(q_off), int(s_off), *geo, out, float(geometry["eps"]), D=int(geo[0].shape[1]), K=k, PB=pb,
            G=g, GP=max(16, triton.next_power_of_2(g)), BLOCK=BASIS_BLOCK, num_warps=BASIS_WARPS,
            enable_fp_fusion=False,
        )  # fmt: skip
    return out
