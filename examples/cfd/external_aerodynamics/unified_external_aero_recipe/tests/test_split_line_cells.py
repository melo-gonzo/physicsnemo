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

"""Tests for ``SplitLineCells`` (the AF-GEO refinement protocol, 2026-10-06): a
count-raising re-panelization of a planar line boundary that keeps the geometry
and the represented measure exactly."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from domain_transforms import SplitLineCells  # noqa: E402

from physicsnemo.mesh import Mesh  # noqa: E402
from physicsnemo.mesh.calculus.measure import EFFECTIVE_MEASURE_KEY, cell_measures  # noqa: E402


def _polygon(n: int = 12, radius: float = 1.0) -> Mesh:
    """A closed polygon in the plane as line cells, with outward normals and a
    per-cell field; the x range is [-radius, radius] so mid-chord is x = 0."""
    t = torch.linspace(0, 2 * torch.pi, n + 1, dtype=torch.float64)[:-1]
    pts = torch.stack([radius * torch.cos(t), radius * torch.sin(t)], -1)
    cells = torch.stack([torch.arange(n), torch.roll(torch.arange(n), -1)], -1)
    mid = 0.5 * (pts[cells[:, 0]] + pts[cells[:, 1]])
    normals = mid / mid.norm(dim=-1, keepdim=True)
    return Mesh(
        points=pts,
        cells=cells,
        cell_data={"normals": normals, "tag": torch.arange(n, dtype=torch.float64)},
        point_data={"phi": t.clone()},
    )


@pytest.mark.parametrize("front,back", [(2, 2), (3, 1), (1, 4)])
def test_split_preserves_geometry_measure_and_cell_data(front, back):
    mesh = _polygon()
    out = SplitLineCells(pieces_front=front, pieces_back=back)(mesh)
    cx = mesh.cell_centroids[:, 0]
    k = torch.where(cx < 0.0, front, back)
    assert out.n_cells == int(k.sum())
    ### the total length is exact; every piece is collinear with its parent
    assert torch.allclose(cell_measures(out).sum(), cell_measures(mesh).sum(), atol=1e-12)
    assert torch.allclose(out.points[: mesh.n_points], mesh.points)
    ### cell data copied to every piece; the per-piece tag identifies the parent
    tags = out.cell_data["tag"]
    counts = torch.bincount(tags.long(), minlength=mesh.n_cells)
    assert torch.equal(counts, k)
    assert torch.allclose(out.cell_data["normals"].norm(dim=-1), torch.ones(out.n_cells, dtype=torch.float64))
    ### the pieces of a parent lie on the parent's segment
    for parent in range(mesh.n_cells):
        a, b = mesh.points[mesh.cells[parent]]
        pieces = out.cells[tags.long() == parent]
        mids = 0.5 * (out.points[pieces[:, 0]] + out.points[pieces[:, 1]])
        d = b - a
        cross = (mids - a)[:, 0] * d[1] - (mids - a)[:, 1] * d[0]
        assert torch.allclose(cross, torch.zeros_like(cross), atol=1e-12)
    ### new point data is linearly interpolated along each parent segment
    assert out.point_data["phi"].shape[0] == out.n_points
    assert torch.allclose(out.point_data["phi"][: mesh.n_points], mesh.point_data["phi"])


def test_split_divides_an_explicit_measure_by_the_piece_count():
    mesh = _polygon()
    mesh.cell_data[EFFECTIVE_MEASURE_KEY] = 3.0 * mesh.cell_areas  # a Horvitz-Thompson-like weight
    out = SplitLineCells(pieces_front=2, pieces_back=2)(mesh)
    assert torch.allclose(cell_measures(out).sum(), cell_measures(mesh).sum(), atol=1e-12)
    assert torch.allclose(cell_measures(out), 3.0 * out.cell_areas, atol=1e-12)


def test_split_leaves_point_clouds_and_surfaces_alone():
    cloud = Mesh(points=torch.rand(20, 2, dtype=torch.float64))
    assert SplitLineCells()(cloud) is cloud
    tri = Mesh(points=torch.rand(4, 3, dtype=torch.float64), cells=torch.tensor([[0, 1, 2], [0, 2, 3]]))
    assert SplitLineCells()(tri) is tri


def test_split_rejects_non_positive_counts():
    with pytest.raises(ValueError):
        SplitLineCells(pieces_front=0)
