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

"""Exact brute-force nearest neighbour for ISLA's internal interior geometry.

For every query the index of the nearest boundary sample, by the exact
float32 squared distance ``|x_q - y_i|^2`` (formed from the *difference*, whose
rounding error is set by the coordinates' own precision and not, as in the
matmul expansion ``|x|^2 + |y|^2 - 2 x.y``, by the body radius squared). One
Triton program owns ``BLOCK_Q`` queries and streams the ``N`` boundary samples
in ``BLOCK_N`` tiles, keeping a running minimum and its index, so no
``(Q, N)`` block is ever written: the pass is compute-bound at about eight
flops per pair (4e10 pairs at 200,000 x 200,000, tens of milliseconds on a
data-centre GPU) where ``torch.cdist``'s element-wise path took 46 s and a
float64 matmul expansion about a second.

Triton is optional; :func:`nearest_index` reports availability through
:data:`HAS_TRITON` and the caller falls back to the float64 expansion.
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


if HAS_TRITON:

    @triton.jit
    def _nn_kernel(
        q_ptr,
        y_ptr,
        idx_ptr,
        Q,
        N,
        q_sb,
        y_sb,
        BLOCK_Q: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_q = tl.program_id(0)
        pid_b = tl.program_id(1)
        offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
        mask_q = offs_q < Q
        qb = q_ptr + pid_b * q_sb + offs_q * 3
        qx = tl.load(qb, mask=mask_q, other=0.0)
        qy = tl.load(qb + 1, mask=mask_q, other=0.0)
        qz = tl.load(qb + 2, mask=mask_q, other=0.0)
        best = tl.full((BLOCK_Q,), float("inf"), tl.float32)
        best_i = tl.zeros((BLOCK_Q,), tl.int32)
        for n0 in range(0, N, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N
            yb = y_ptr + pid_b * y_sb + offs_n * 3
            yx = tl.load(yb, mask=mask_n, other=float("inf"))
            yy = tl.load(yb + 1, mask=mask_n, other=0.0)
            yz = tl.load(yb + 2, mask=mask_n, other=0.0)
            dx = qx[:, None] - yx[None, :]
            dy = qy[:, None] - yy[None, :]
            dz = qz[:, None] - yz[None, :]
            d2 = dx * dx + dy * dy + dz * dz  # (BLOCK_Q, BLOCK_N); inf on padded samples
            tile_min = tl.min(d2, axis=1)
            tile_arg = tl.argmin(d2, axis=1)
            ### strict '<' keeps the earliest tile on ties, as torch.argmin does
            take = tile_min < best
            best = tl.where(take, tile_min, best)
            best_i = tl.where(take, (n0 + tile_arg).to(tl.int32), best_i)
        tl.store(idx_ptr + pid_b * Q + offs_q, best_i.to(tl.int64), mask=mask_q)


def nearest_index(
    query_points: torch.Tensor, points: torch.Tensor
) -> torch.Tensor:
    """Index ``(B, Q)`` of the nearest of ``points`` ``(B, N, 3)`` to each of
    ``query_points`` ``(B, Q, 3)``, by exact float32 distance. CUDA and Triton
    only (``HAS_TRITON``); inputs are made contiguous float32."""
    if not HAS_TRITON:  # pragma: no cover
        raise RuntimeError("nearest_index needs triton")
    q = query_points.contiguous().float()
    y = points.contiguous().float()
    b, nq, _ = q.shape
    n = y.shape[1]
    idx = torch.empty(b, nq, dtype=torch.int64, device=q.device)
    block_q, block_n = 64, 128
    grid = (triton.cdiv(nq, block_q), b)
    _nn_kernel[grid](
        q, y, idx, nq, n, q.stride(0), y.stride(0), BLOCK_Q=block_q, BLOCK_N=block_n, num_warps=4
    )
    return idx
