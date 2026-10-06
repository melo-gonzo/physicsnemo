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

"""Shared indexing helpers for datapipes."""

from __future__ import annotations

from typing import Literal

import torch


def _cyclic_block_indices(
    total: int,
    k: int,
    start: int | None = None,
    device: torch.device | str | None = None,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return ``k`` consecutive indices in ``[0, total)``, wrapping at the end.

    When ``start`` is omitted, it is drawn uniformly from every index in the
    range. Consequently, every element has inclusion probability exactly
    ``k / total``. A non-wrapping block cannot provide that property because
    elements near either end occur in fewer valid blocks.

    The result contains one or two ascending contiguous runs, retaining the
    locality needed by readers backed by memory-mapped or chunked storage.

    Parameters
    ----------
    total : int
        Number of available elements.
    k : int
        Number of consecutive elements to select.
    start : int, optional
        Explicit first index. Primarily useful when a caller owns its random
        number generator or for deterministic tests. Must not be supplied with
        ``generator``.
    device : torch.device or str, optional
        Device on which to create the returned indices.
    generator : torch.Generator, optional
        Generator used to draw a random start. ``None`` uses PyTorch's global
        default generator.

    Returns
    -------
    torch.Tensor
        Integer indices with shape ``(k,)``.

    Raises
    ------
    ValueError
        If the sizes are invalid, an explicit start is out of bounds, or both
        ``start`` and ``generator`` are supplied.
    """
    if total < 0 or k < 0:
        raise ValueError(f"total and k must be non-negative, got {total=} and {k=}.")
    if k > total:
        raise ValueError(
            f"Range size {total} is less than the requested block size {k}."
        )
    if start is not None and generator is not None:
        raise ValueError("start and generator are mutually exclusive.")
    if start is not None and (start < 0 or start >= total):
        raise ValueError(f"start must be in [0, {total}), got {start}.")

    # There is no sampling to do in either case. In particular, keep the
    # natural order when selecting the full range instead of returning a
    # randomly rotated view of the same elements.
    if k == 0 or k == total:
        return torch.arange(k, device=device)

    output_device = torch.device(device) if device is not None else None
    if start is None:
        random_device = generator.device if generator is not None else output_device
        start_index: int | torch.Tensor = torch.randint(
            0,
            total,
            (1,),
            generator=generator,
            device=random_device,
        )
        if output_device is None:
            output_device = start_index.device
        elif start_index.device != output_device:
            start_index = start_index.to(output_device)
    else:
        start_index = start

    return (torch.arange(k, device=output_device) + start_index) % total


SubsampleMode = Literal["uniform", "block"]


def _uniform_indices(
    total: int,
    k: int,
    device: torch.device | str | None = None,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return ``k`` distinct indices drawn uniformly from ``[0, total)``, sorted.

    Every ``k``-subset is equally likely, so the selection is spatially
    representative regardless of storage order. Sorting keeps memory-mapped and
    chunked reads moving forward through the file.

    For ``k`` much smaller than ``total`` this avoids an ``O(total)``
    permutation: it draws with replacement until ``k`` distinct values exist,
    then keeps a uniformly random ``k``-subset of them, which is again a
    uniform ``k``-subset of the range by symmetry.

    Parameters
    ----------
    total : int
        Number of available elements.
    k : int
        Number of indices to select, ``0 <= k <= total``.
    device : torch.device or str, optional
        Device on which to create the returned indices.
    generator : torch.Generator, optional
        Random generator. ``None`` uses PyTorch's global default generator.

    Returns
    -------
    torch.Tensor
        Ascending integer indices with shape ``(k,)``.
    """
    if total < 0 or k < 0 or k > total:
        raise ValueError(f"Need 0 <= k <= total, got {total=} and {k=}.")
    if k == total:
        return torch.arange(k, device=device)
    random_device = generator.device if generator is not None else "cpu"
    if 4 * k >= total:
        idx = torch.randperm(total, generator=generator, device=random_device)[:k]
    else:
        idx = torch.unique(
            torch.randint(
                total, (k + k // 8,), generator=generator, device=random_device
            )
        )
        while idx.numel() < k:
            extra = torch.randint(
                total, (k,), generator=generator, device=random_device
            )
            idx = torch.unique(torch.cat([idx, extra]))
        idx = idx[
            torch.randperm(idx.numel(), generator=generator, device=random_device)[:k]
        ]
    return torch.sort(idx).values.to(device)


def _subsample_indices(
    total: int,
    k: int,
    mode: SubsampleMode = "uniform",
    device: torch.device | str | None = None,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Select ``k`` of ``total`` indices with :func:`_uniform_indices` or
    :func:`_cyclic_block_indices`, depending on ``mode``.

    ``"block"`` reads one contiguous run in storage order. It is only spatially
    representative when storage order is shuffled; on solver- or patch-ordered
    meshes every sample is a spatial slab. ``"uniform"`` is order-independent.
    """
    if mode == "uniform":
        return _uniform_indices(total, k, device, generator=generator)
    if mode == "block":
        return _cyclic_block_indices(total, k, device=device, generator=generator)
    raise ValueError(f"subsample mode must be 'uniform' or 'block', got {mode!r}.")
