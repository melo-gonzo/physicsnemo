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

"""Exact duplicate cells of a boundary mesh, and the cached index the reader drops them with.

A cell is an *extra copy* when its sorted vertex-index triple equals that of a
lower-indexed cell; the lowest-indexed cell of each such group is kept. The
HiLiftAeroML wings store about 18.9 M extra copies of about 25,700 root cells per
case (copies carry the same fields up to float32 rounding), which hold about 46 %
of the summed wing area, so every measure-weighted quantity of the stored wing is
biased; dropping them is a data correction applied by the recipe, the stored files
stay as they are.

Finding the copies of a 280 M-cell wing takes about 4 minutes, so the reader
(:class:`~merge_global_data.MeshReaderWithGlobalData`, ``drop_duplicate_cells``)
reads a cached index instead: per case, ``<case>.npy`` holds
``np.packbits(extra)`` and ``<case>.json`` its provenance, including
:func:`connectivity_fingerprint` of the connectivity it was built from, which the
reader checks at every load (a mismatch is an error) and the md5 of the whole
connectivity array. Without a cache the reader computes :func:`duplicate_cells` at
load (exact, slow; used by the tests).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

FINGERPRINT_ROWS = 1024
CHUNK = 20_000_000
_U = np.uint64


def _numpy(cells) -> np.ndarray:
    return cells.numpy() if isinstance(cells, torch.Tensor) else np.asarray(cells)


def _strided_fingerprint(rows, dtype: str) -> str:
    """``"<rows>x<cols>:<md5>"`` of ``FINGERPRINT_ROWS`` evenly strided rows of ``rows``
    as ``dtype``."""
    n, d = int(rows.shape[0]), int(rows.shape[1])
    idx = (np.arange(FINGERPRINT_ROWS, dtype=np.int64) * max(n - 1, 0)) // (
        FINGERPRINT_ROWS - 1
    )
    if isinstance(rows, torch.Tensor):
        sample = rows[torch.from_numpy(idx)].numpy() if n else np.zeros((0, d), dtype)
    else:
        sample = np.asarray(rows[idx]) if n else np.zeros((0, d), dtype)
    digest = hashlib.md5(
        np.ascontiguousarray(sample, dtype=dtype).tobytes()
    ).hexdigest()
    return f"{n}x{d}:{digest}"


def connectivity_fingerprint(cells) -> str:
    """``"<rows>x<cols>:<md5>"`` of ``FINGERPRINT_ROWS`` evenly strided rows of the
    connectivity (int64, little-endian): cheap enough for every load, and it changes
    with the cell count, the column count or any sampled row."""
    return _strided_fingerprint(cells, "<i8")


def points_fingerprint(points) -> str:
    """The same for the points (float32, little-endian), sampled with the same striding."""
    return _strided_fingerprint(points, "<f4")


def _splitmix(x: np.ndarray) -> np.ndarray:
    x = x + _U(0x9E3779B97F4A7C15)
    x = (x ^ (x >> _U(30))) * _U(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> _U(27))) * _U(0x94D049BB133111EB)
    return x ^ (x >> _U(31))


def _triple_hash(c: np.ndarray) -> np.ndarray:
    s = np.sort(c, axis=1).astype(_U)
    return _splitmix(_splitmix(s[:, 0] | (s[:, 1] << _U(32))) ^ s[:, 2])


def duplicate_cells(cells, md5: hashlib._Hash | None = None) -> np.ndarray:
    """The extra copies of a triangle connectivity ``(N, 3)``, as a bool array.

    Exact: every cell's sorted index triple is hashed to 64 bits to find the
    candidates (cells whose hash repeats), which are then grouped exactly on the
    triple by a stable lexsort, so a hash collision only adds a candidate that stays
    alone. Vertex indices must be below 2**32. Reads ``cells`` in chunks (a memmap is
    fine); ``md5``, when given, is updated with the connectivity's bytes as read.
    """
    n = int(cells.shape[0])
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError(
            f"duplicate_cells expects (N, 3) triangles, got {tuple(cells.shape)}"
        )
    h = np.empty(n, dtype=_U)
    for s in range(0, n, CHUNK):
        c = np.ascontiguousarray(_numpy(cells[s : s + CHUNK]), dtype=np.int64)
        if md5 is not None:
            md5.update(c.astype("<i8", copy=False).tobytes())
        if len(c) and (c.min() < 0 or c.max() >= 2**32):
            raise ValueError("duplicate_cells needs vertex indices in [0, 2**32)")
        h[s : s + CHUNK] = _triple_hash(c)
    hs = np.sort(h)
    repeated = np.unique(hs[1:][hs[1:] == hs[:-1]])
    del hs
    extra = np.zeros(n, dtype=bool)
    if len(repeated) == 0:
        return extra
    cand = []
    for s in range(0, n, CHUNK):
        x = h[s : s + CHUNK]
        i = np.searchsorted(repeated, x).clip(max=len(repeated) - 1)
        cand.append(np.flatnonzero(repeated[i] == x) + s)
    del h
    cand = np.concatenate(cand)
    t = np.sort(
        _numpy(cells[cand])
        if not isinstance(cells, torch.Tensor)
        else cells[torch.from_numpy(cand)].numpy(),
        axis=1,
    )
    order = np.lexsort(t.T[::-1])
    ts = t[order]
    dup = np.zeros(len(ts), dtype=bool)
    dup[1:] = (ts[1:] == ts[:-1]).all(axis=1)
    extra[cand[order[dup]]] = True
    return extra


def duplicate_cells_reference(cells) -> np.ndarray:
    """The same as :func:`duplicate_cells` by one exact lexsort of every cell (no
    hashing): slower and about twice the memory, an independent check."""
    c = np.sort(_numpy(cells), axis=1)
    order = np.lexsort(c.T[::-1])
    cs = c[order]
    dup = np.zeros(len(cs), dtype=bool)
    dup[1:] = (cs[1:] == cs[:-1]).all(axis=1)
    extra = np.zeros(len(cs), dtype=bool)
    extra[order[dup]] = True
    return extra


def write_mask(out_dir: Path | str, case: str, cells, source: str) -> dict:
    """Build ``case``'s mask from ``cells`` and write ``<case>.npy`` and ``<case>.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md5 = hashlib.md5()
    extra = duplicate_cells(cells, md5=md5)
    packed = np.packbits(extra)
    np.save(out_dir / f"{case}.npy", packed)
    meta = {
        "case": case,
        "source": source,
        "n_cells": int(cells.shape[0]),
        "n_extra": int(extra.sum()),
        "fingerprint": connectivity_fingerprint(cells),
        "connectivity_md5": md5.hexdigest(),
        "mask_md5": hashlib.md5((out_dir / f"{case}.npy").read_bytes()).hexdigest(),
        "builder": "dedup.duplicate_cells",
    }
    (out_dir / f"{case}.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


class DuplicateMask:
    """One case's cached extra-copy index, checked against the mesh it is applied to."""

    def __init__(self, mask_dir: Path | str, case: str, cells) -> None:
        mask_dir = Path(mask_dir)
        meta_path, mask_path = mask_dir / f"{case}.json", mask_dir / f"{case}.npy"
        if not meta_path.exists() or not mask_path.exists():
            raise FileNotFoundError(
                f"drop_duplicate_cells: no cached mask for {case!r} in {str(mask_dir)!r}"
            )
        meta = json.loads(meta_path.read_text())
        got = connectivity_fingerprint(cells)
        if meta.get("fingerprint") != got:
            raise ValueError(
                f"drop_duplicate_cells: the mask of {case!r} was built from another connectivity "
                f"(cached {meta.get('fingerprint')!r}, this case {got!r}); rebuild it"
            )
        self.n_cells = int(cells.shape[0])
        self.n_extra = int(meta["n_extra"])
        self._packed = np.load(mask_path, mmap_mode="r")
        if len(self._packed) != (self.n_cells + 7) // 8:
            raise ValueError(
                f"drop_duplicate_cells: the mask of {case!r} has {len(self._packed)} bytes for {self.n_cells} cells"
            )

    def __getitem__(self, idx: torch.Tensor) -> torch.Tensor:
        """Whether each cell of ``idx`` (int64) is an extra copy."""
        i = idx.cpu().numpy()
        lo, hi = int(i.min()) >> 3, (int(i.max()) >> 3) + 1
        window = np.asarray(self._packed[lo:hi])
        bits = (window[(i >> 3) - lo] >> (7 - (i & 7))) & 1
        return torch.from_numpy(bits.astype(bool)).to(idx.device)

    def all(self) -> torch.Tensor:
        """Every cell's flag, ``(N,)`` bool."""
        return torch.from_numpy(
            np.unpackbits(np.asarray(self._packed), count=self.n_cells).astype(bool)
        )


class ComputedDuplicates:
    """The mask-free path: the extra copies computed at load (exact, slow)."""

    def __init__(self, cells) -> None:
        self._extra = torch.from_numpy(duplicate_cells(cells))
        self.n_cells = int(cells.shape[0])
        self.n_extra = int(self._extra.sum())

    def __getitem__(self, idx: torch.Tensor) -> torch.Tensor:
        return self._extra.to(idx.device)[idx]

    def all(self) -> torch.Tensor:
        """Every cell's flag, ``(N,)`` bool."""
        return self._extra
