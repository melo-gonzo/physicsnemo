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

"""Tests for ``drop_duplicate_cells`` (src/dedup.py and MeshReaderWithGlobalData)."""

import numpy as np
import pytest
import torch
from dedup import (
    ComputedDuplicates,
    connectivity_fingerprint,
    duplicate_cells,
    duplicate_cells_reference,
    write_mask,
)
from merge_global_data import MeshReaderWithGlobalData, drop_duplicates_and_subsample

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import cell_measures


def _grid(n: int = 12, seed: int = 0) -> tuple[torch.Tensor, np.ndarray]:
    rng = np.random.default_rng(seed)
    xy = np.stack(
        np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 2, n + 1)), -1
    ).reshape(-1, 2)
    points = np.concatenate([xy, rng.random((len(xy), 1)) * 0.1], 1)
    v = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    a, b, c, d = (
        v[:-1, :-1].ravel(),
        v[:-1, 1:].ravel(),
        v[1:, :-1].ravel(),
        v[1:, 1:].ravel(),
    )
    cells = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])
    return torch.tensor(points, dtype=torch.float32), cells.astype(np.int64)


def _with_copies(cells: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    copies = [
        np.repeat(cells[3:4], 40, 0),
        np.roll(cells[10:20], 1, axis=1),
        cells[30:35][:, ::-1],
        cells[50:60],
    ]
    out = np.concatenate([cells, *copies])
    return out[rng.permutation(len(out))]


def _save_case(root, name: str, points, cells) -> None:
    Mesh(points=points, cells=torch.from_numpy(cells)).save(
        root / name / "surface.pmsh"
    )


def test_builder_is_exact_and_keeps_the_first_copy():
    """The hash-first builder equals the lexsort reference and keeps the lowest index of each group."""
    _, cells = _grid()
    cells = _with_copies(cells)
    extra = duplicate_cells(cells)
    np.testing.assert_array_equal(extra, duplicate_cells_reference(cells))
    first = {}
    for i, row in enumerate(map(tuple, np.sort(cells, axis=1))):
        first.setdefault(row, i)
        assert extra[i] == (first[row] != i)
    assert extra.sum() == 40 + 10 + 5 + 10


def test_without_copies_the_draw_equals_meshreader(tmp_path):
    """With no copies, the option draws exactly what MeshReader draws, epoch by epoch."""
    points, cells = _grid()
    _save_case(tmp_path, "case_a", points, cells)
    kw = dict(pattern="case_*/*.pmsh", subsample_n_cells=50)
    plain, dedup = (
        MeshReaderWithGlobalData(tmp_path, **kw),
        MeshReaderWithGlobalData(tmp_path, drop_duplicate_cells=True, **kw),
    )
    for r in (plain, dedup):
        r.set_generator(torch.Generator().manual_seed(7))
    for epoch in range(3):
        plain.set_epoch(epoch)
        dedup.set_epoch(epoch)
        a, _ = plain[0]
        b, _ = dedup[0]
        torch.testing.assert_close(a.points, b.points)
        torch.testing.assert_close(a.cells, b.cells)
        torch.testing.assert_close(cell_measures(a), cell_measures(b))


def test_measure_is_unbiased_for_the_deduplicated_mesh():
    """Over every block start, the kept measure averages to the deduplicated area."""
    points, cells = _grid()
    cells = _with_copies(cells)
    mesh = Mesh(points=points, cells=torch.from_numpy(cells))
    duplicates = ComputedDuplicates(mesh.cells)
    true_area = float(cell_measures(mesh.slice_cells(~duplicates.all())).double().sum())
    stored_area = float(cell_measures(mesh).double().sum())
    assert stored_area > 1.1 * true_area
    k = 37
    sums = [
        float(
            cell_measures(drop_duplicates_and_subsample(mesh, duplicates, k, start=s))
            .double()
            .sum()
        )
        for s in range(mesh.n_cells)
    ]
    assert np.mean(sums) == pytest.approx(true_area, rel=1e-6)
    full = drop_duplicates_and_subsample(mesh, duplicates, None)
    assert full.n_cells == mesh.n_cells - duplicates.n_extra
    assert float(cell_measures(full).double().sum()) == pytest.approx(
        true_area, rel=1e-6
    )


def test_cached_mask_equals_the_computed_path_and_is_checked(tmp_path):
    """A cached mask gives the computed path's draw, and a changed case is refused."""
    points, cells = _grid()
    cells = _with_copies(cells)
    _save_case(tmp_path / "data", "case_a", points, cells)
    meta = write_mask(tmp_path / "masks", "case_a", cells, "synthetic")
    assert meta["n_extra"] == 65 and meta["fingerprint"] == connectivity_fingerprint(
        torch.from_numpy(cells)
    )
    kw = dict(pattern="case_*/*.pmsh", subsample_n_cells=60, drop_duplicate_cells=True)
    computed = MeshReaderWithGlobalData(tmp_path / "data", **kw)
    cached = MeshReaderWithGlobalData(
        tmp_path / "data", duplicate_cell_masks=str(tmp_path / "masks"), **kw
    )
    for r in (computed, cached):
        r.set_generator(torch.Generator().manual_seed(3))
    for epoch in range(4):
        computed.set_epoch(epoch)
        cached.set_epoch(epoch)
        a, _ = computed[0]
        b, _ = cached[0]
        torch.testing.assert_close(a.cells, b.cells)
        torch.testing.assert_close(cell_measures(a), cell_measures(b))
        assert b.n_cells <= 60
    ### the case changes under its cached mask: refused, no fallback
    changed = cells.copy()
    changed[[0, -1]] = changed[[-1, 0]]
    _save_case(tmp_path / "data", "case_a", points, changed)
    with pytest.raises(ValueError, match="another connectivity"):
        cached[0]
    with pytest.raises(FileNotFoundError):
        MeshReaderWithGlobalData(
            tmp_path / "data", duplicate_cell_masks=str(tmp_path / "none"), **kw
        )[0]


def test_masks_require_the_flag(tmp_path):
    """A mask directory without drop_duplicate_cells is an error."""
    with pytest.raises(ValueError, match="drop_duplicate_cells is false"):
        MeshReaderWithGlobalData(
            tmp_path, pattern="*", duplicate_cell_masks=str(tmp_path)
        )
