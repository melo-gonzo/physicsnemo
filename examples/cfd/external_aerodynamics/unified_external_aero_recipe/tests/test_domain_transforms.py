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

"""Tests for the recipe-local surface mesh transforms."""

from pathlib import Path

import pytest
import torch
from datasets import build_dataset
from omegaconf import OmegaConf

from physicsnemo.datapipes.protocols import DatasetBase
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.calculus import cell_measures, point_measures


def _two_triangles() -> Mesh:
    """Two triangles in the xy-plane."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
        ]
    )
    return Mesh(
        points=points,
        cells=torch.tensor([[0, 1, 2], [3, 4, 5]]),
        cell_data={"pressure": torch.tensor([10.0, 20.0])},
    )


@pytest.mark.parametrize("augment", [False, True])
def test_drivaer_dataset_pipeline_preserves_thin_face_and_aligns_targets(
    tmp_path, augment
):
    """Run the actual reader, configured transforms, and target conversion; a thin
    (1e-4 high) face keeps a positive measure and its target row."""
    vehicle = Mesh(
        points=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1e-4, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [3.0, 1.0, 0.0],
                [5.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [5.0, 1.0, 0.0],
            ]
        ),
        cells=torch.arange(9).reshape(3, 3),
        cell_data={
            "pMeanTrim": torch.tensor([100.0, 200.0, 300.0]),
            "wallShearStressMeanTrim": torch.zeros(3, 3),
        },
        global_data={"TimeValue": torch.tensor(0.0)},
    )
    source = DomainMesh(
        interior=Mesh(points=torch.zeros(0, 3)),
        boundaries={"vehicle": vehicle},
        global_data={
            "U_inf": torch.tensor([10.0, 0.0, 0.0]),
            "rho_inf": torch.tensor(2.0),
            "p_inf": torch.tensor(0.0),
            "L_ref": torch.tensor(1.0),
        },
    )
    sample_path = tmp_path / "run_001" / "sample.pdmsh"
    sample_path.parent.mkdir()
    source.save(sample_path)
    recipe = Path(__file__).resolve().parent.parent
    cfg = OmegaConf.merge(
        OmegaConf.load(recipe / "datasets" / "drivaer_ml_surface.yaml"),
        {
            "dataset_paths": {"drivaer_ml": str(tmp_path)},
            "sampling_resolution": 3,
        },
    )
    dataset = build_dataset(cfg, augment=augment, device=None, num_workers=1)
    try:
        domain, _ = dataset[0]
    finally:
        # MeshReader has no close() hook; release the dataset's prefetch pool.
        DatasetBase.close(dataset)

    boundary = domain.boundaries["vehicle"]
    assert boundary.n_cells == domain.interior.n_points == 3
    torch.testing.assert_close(
        domain.interior.point_data["pressure"], torch.tensor([1.0, 2.0, 3.0])
    )
    torch.testing.assert_close(domain.interior.points, boundary.cell_centroids)
    torch.testing.assert_close(
        boundary.cell_data["normals"].norm(dim=-1), torch.ones(3)
    )
    measures = point_measures(domain.interior)
    assert torch.isfinite(measures).all() and (measures > 0).all()
    torch.testing.assert_close(measures, cell_measures(boundary))
    # A measure-weighted loss must include every target and backpropagate.
    prediction = torch.zeros(3, requires_grad=True)
    error = prediction - domain.interior.point_data["pressure"]
    loss = domain.interior.integrate_samples(error.square()) / measures.sum()
    weights = measures / measures.sum()
    torch.testing.assert_close(loss, (weights * torch.tensor([1.0, 4.0, 9.0])).sum())
    loss.backward()
    torch.testing.assert_close(prediction.grad, 2 * weights * error.detach())
    assert "TimeValue" not in domain.global_data


def test_stack_scalar_fields_builds_a_zero_padded_vector():
    """Two stored components become one (N, 3) field with z = 0; sources are kept."""
    from domain_transforms import StackScalarFields

    mesh = _two_triangles()
    mesh.cell_data["bc_x"] = torch.tensor([1.0, 2.0])
    mesh.cell_data["bc_y"] = torch.tensor([[3.0], [4.0]])
    out = StackScalarFields(fields=["bc_x", "bc_y"], output_field="bc")(mesh)
    assert torch.equal(
        out.cell_data["bc"], torch.tensor([[1.0, 3.0, 0.0], [2.0, 4.0, 0.0]])
    )
    assert "bc_x" in out.cell_data and "bc_y" in out.cell_data
    with pytest.raises(KeyError):
        StackScalarFields(fields=["missing"], output_field="bc")(mesh)
    with pytest.raises(ValueError):
        StackScalarFields(fields=["a", "b", "c", "d"], output_field="bc")



def test_flip_boundary_orientation_reverses_the_named_boundaries():
    """The named boundary's cells are rewound (every normal reversed, centroids and
    measures unchanged) and its stored normals negated; other boundaries, the
    interior, the other fields and the stored files' meshes are untouched; an
    unknown boundary is refused, a stored-normal name a boundary lacks is skipped.
    A two-dimensional (segment) boundary flips the same way."""
    from domain_transforms import FlipBoundaryOrientation

    wall = _two_triangles()
    wall = wall.with_data(cell_data=wall.cell_data.update({"normals": wall.cell_normals.clone()}))
    seg = Mesh(points=torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]), cells=torch.tensor([[0, 1], [1, 2]]))
    inlet = _two_triangles()
    domain = DomainMesh(
        interior=Mesh(points=torch.randn(5, 3)),
        boundaries={"wall": wall, "inlet": inlet},
        global_data={"U": torch.ones(3)},
    )
    before = wall.cell_normals.clone()
    out = FlipBoundaryOrientation(boundaries=["wall"], normal_fields=["normals", "absent"]).apply_to_domain(domain)
    flipped = out.boundaries["wall"]
    assert torch.equal(flipped.cell_normals, -before)
    assert torch.equal(flipped.cell_data["normals"], -before)
    assert torch.equal(flipped.cell_data["pressure"], wall.cell_data["pressure"])
    assert torch.allclose(flipped.cell_centroids, wall.cell_centroids)
    assert torch.allclose(cell_measures(flipped), cell_measures(wall))
    assert torch.equal(wall.cell_normals, before)  # the input mesh is not modified
    assert out.boundaries["inlet"] is inlet and out.interior is domain.interior
    assert torch.equal(out.global_data["U"], torch.ones(3))
    twice = FlipBoundaryOrientation(boundaries=["wall"], normal_fields=["normals"]).apply_to_domain(out)
    assert torch.equal(twice.boundaries["wall"].cells, wall.cells)
    d2 = DomainMesh(interior=Mesh(points=torch.randn(3, 2)), boundaries={"airfoil": seg})
    seg_n = seg.cell_normals.clone()
    assert torch.equal(FlipBoundaryOrientation(boundaries=["airfoil"]).apply_to_domain(d2).boundaries["airfoil"].cell_normals, -seg_n)
    with pytest.raises(KeyError, match="'outlet'"):
        FlipBoundaryOrientation(boundaries=["outlet"]).apply_to_domain(domain)
