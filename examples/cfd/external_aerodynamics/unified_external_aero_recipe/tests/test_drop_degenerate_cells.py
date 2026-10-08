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

"""DropDegenerateCells (2026-10-02): zero-measure cells (repeated vertices) and their cell data are removed;
healthy meshes pass through unchanged; measures of the kept cells are untouched."""

import torch
from tensordict import TensorDict

from domain_transforms import DropDegenerateCells
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import cell_measures


def _mesh():
    pts = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 0]])
    cells = torch.tensor(
        [[0, 1, 2], [1, 3, 2], [1, 1, 4], [3, 3, 3]]
    )  # two valid, two degenerate
    cd = TensorDict(
        {"pressure": torch.arange(4.0), "normals": torch.randn(4, 3)}, batch_size=[4]
    )
    return Mesh(points=pts, cells=cells, cell_data=cd)


def test_degenerate_cells_and_their_data_are_dropped():
    """Zero-measure cells and their cell data go; the vertices and the other cells stay."""
    m = _mesh()
    out = DropDegenerateCells()(m)
    assert out.n_cells == 2 and out.n_points == m.n_points
    torch.testing.assert_close(out.cell_data["pressure"], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(cell_measures(out), cell_measures(m)[:2])
    assert (cell_measures(out) > 0).all()


def test_healthy_mesh_passes_through():
    """A mesh without degenerate cells comes back unchanged."""
    m = _mesh().slice_cells(torch.tensor([True, True, False, False]))
    out = DropDegenerateCells()(m)
    assert out.n_cells == 2
    torch.testing.assert_close(out.cells, m.cells)
