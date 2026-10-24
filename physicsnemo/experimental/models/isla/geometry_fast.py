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

r"""Triton kernels of the fast boundary-geometry search (``kernel="fast"`` of
:func:`~physicsnemo.experimental.models.isla.geometry.exact_boundary_geometry`).

The mesh-only steps before the search compute integers from comparisons of the
vertex coordinates: the representative of each welded vertex and each cell's
vertices in the lexicographic order of their positions with the parity of that
permutation. The kernels here compute the same integers as the reference's torch
code in one pass each, where the torch code launches tens of kernels (the
canonical order) or a slow scan (the weld's running maximum). Each function falls
back to the reference's torch code off CUDA or without Triton, so its result never
depends on the path taken.
"""

from __future__ import annotations

import importlib

import torch

from physicsnemo.core.version_check import is_package_available

### Triton is an optional dependency, loaded through importlib as in geo_kernel.py.
HAS_TRITON = is_package_available("triton")
if HAS_TRITON:
    try:
        triton = importlib.import_module("triton")
        tl = importlib.import_module("triton.language")
    except Exception:  # pragma: no cover - a broken triton install is treated as absent
        HAS_TRITON = False

_SCAN_BLOCK = 1024
_CELL_BLOCK = 256


def _usable(t: torch.Tensor) -> bool:
    return HAS_TRITON and t.is_cuda


if HAS_TRITON:

    @triton.jit
    def _max_combine(a, b):
        return tl.maximum(a, b)

    @triton.jit
    def _block_last_start_kernel(new, out, n, BLOCK: tl.constexpr):
        """The last position ``i`` with ``new[i]`` in each block (0 if none)."""
        b = tl.program_id(0)
        i = b * BLOCK + tl.arange(0, BLOCK)
        flag = tl.load(new + i, mask=i < n, other=0)
        tl.store(out + b, tl.max(tl.where(flag != 0, i, 0).to(tl.int64), 0))

    @triton.jit
    def _run_start_kernel(new, before, out, n, BLOCK: tl.constexpr):
        """The running maximum of ``where(new, i, 0)``: the block's own scan, against the
        greatest start of the blocks before it."""
        b = tl.program_id(0)
        i = b * BLOCK + tl.arange(0, BLOCK)
        flag = tl.load(new + i, mask=i < n, other=0)
        scan = tl.associative_scan(tl.where(flag != 0, i, 0).to(tl.int64), 0, _max_combine)
        tl.store(out + i, tl.maximum(scan, tl.load(before + b)), mask=i < n)

    @triton.jit
    def _canonical_kernel(points, cells, vert_out, parity_out, n, K: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
        """Each cell's vertices sorted stably by their coordinates, the last first, by
        adjacent swaps of strictly smaller keys (NaN last), as
        :func:`~physicsnemo.experimental.models.isla.geometry._canonical_cells`; and the
        parity of the permutation."""
        c = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = c < n
        v0 = tl.load(cells + K * c, mask=m, other=0)
        v1 = tl.load(cells + K * c + 1, mask=m, other=0)
        s0 = tl.zeros([BLOCK], dtype=tl.int32)
        s1 = s0 + 1
        if K == 3:
            v2 = tl.load(cells + K * c + 2, mask=m, other=0)
            s2 = s0 + 2
        for jj in tl.static_range(D):
            j = D - 1 - jj
            k0 = tl.load(points + D * v0 + j, mask=m, other=0.0).to(tl.float64)
            k1 = tl.load(points + D * v1 + j, mask=m, other=0.0).to(tl.float64)
            ### the pairs (0, 1), then for three vertices (1, 2) and (0, 1) again
            swap = (k1 < k0) | ((k0 != k0) & (k1 == k1))
            v0, v1 = tl.where(swap, v1, v0), tl.where(swap, v0, v1)
            s0, s1 = tl.where(swap, s1, s0), tl.where(swap, s0, s1)
            k0, k1 = tl.where(swap, k1, k0), tl.where(swap, k0, k1)
            if K == 3:
                k2 = tl.load(points + D * v2 + j, mask=m, other=0.0).to(tl.float64)
                swap = (k2 < k1) | ((k1 != k1) & (k2 == k2))
                v1, v2 = tl.where(swap, v2, v1), tl.where(swap, v1, v2)
                s1, s2 = tl.where(swap, s2, s1), tl.where(swap, s1, s2)
                k1, k2 = tl.where(swap, k2, k1), tl.where(swap, k1, k2)
                swap = (k1 < k0) | ((k0 != k0) & (k1 == k1))
                v0, v1 = tl.where(swap, v1, v0), tl.where(swap, v0, v1)
                s0, s1 = tl.where(swap, s1, s0), tl.where(swap, s0, s1)
        inv = (s0 > s1).to(tl.int64)
        tl.store(vert_out + K * c, v0, mask=m)
        tl.store(vert_out + K * c + 1, v1, mask=m)
        if K == 3:
            inv += (s0 > s2).to(tl.int64) + (s1 > s2).to(tl.int64)
            tl.store(vert_out + K * c + 2, v2, mask=m)
        tl.store(parity_out + c, 1 - 2 * (inv % 2), mask=m)


def run_start(new: torch.Tensor) -> torch.Tensor:
    """``torch.cummax(torch.where(new, arange, 0), 0).values`` for a boolean ``(N,)``:
    for each position, the last position at or before it where ``new`` holds."""
    n = new.shape[0]
    if not _usable(new) or n == 0:
        return torch.cummax(torch.where(new, torch.arange(n, device=new.device), 0), 0).values
    flags = new.to(torch.int8).contiguous()
    blocks = triton.cdiv(n, _SCAN_BLOCK)
    last = torch.empty(blocks, dtype=torch.int64, device=new.device)
    _block_last_start_kernel[(blocks,)](flags, last, n, BLOCK=_SCAN_BLOCK)
    before = torch.cat([last.new_zeros(1), torch.cummax(last, 0).values[:-1]])
    out = torch.empty(n, dtype=torch.int64, device=new.device)
    _run_start_kernel[(blocks,)](flags, before, out, n, BLOCK=_SCAN_BLOCK)
    return out


def canonical_cells(points: torch.Tensor, cells: torch.Tensor, reference):
    """``reference(points, cells)`` (the reference's
    :func:`~physicsnemo.experimental.models.isla.geometry._canonical_cells`) by one
    Triton kernel on CUDA, for segments and triangles in two and three dimensions."""
    k, d = cells.shape[1], points.shape[1]
    if not _usable(points) or k not in (2, 3) or d not in (2, 3) or cells.shape[0] == 0:
        return reference(points, cells)
    n = cells.shape[0]
    vert = torch.empty_like(cells)
    parity = torch.empty(n, dtype=torch.int64, device=cells.device)
    _canonical_kernel[(triton.cdiv(n, _CELL_BLOCK),)](
        points.contiguous(), cells.contiguous(), vert, parity, n, K=k, D=d, BLOCK=_CELL_BLOCK
    )
    return vert, parity
