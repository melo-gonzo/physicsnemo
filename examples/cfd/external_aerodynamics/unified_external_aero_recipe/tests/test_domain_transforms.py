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


#: The tunnel boundaries DrivAerML's case files store beside the vehicle.
DOMAIN_BOUNDARIES = ("inlet", "no_slip", "outlet", "slip")


def _tunnel_boundaries() -> dict[str, Mesh]:
    """Four two-triangle squares standing for the tunnel's boundaries (no cell data,
    as in the case files), at distinct places so that a move of one is visible."""
    square = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    cells = torch.tensor([[0, 1, 2], [0, 2, 3]])
    return {
        name: Mesh(
            points=square * (i + 2) + torch.tensor([-5.0, -4.0, -1.0 - i]), cells=cells
        )
        for i, name in enumerate(DOMAIN_BOUNDARIES)
    }


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
        boundaries={"vehicle": vehicle, **_tunnel_boundaries()},
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
    assert set(domain.boundary_names) == {"vehicle", *DOMAIN_BOUNDARIES}


def _drivaer_surface_case(tmp_path, n_cells: int = 30) -> DomainMesh:
    """One DrivAerML-shaped case file: a random vehicle surface with targets and the
    four tunnel boundaries, saved as run_001/sample.pdmsh."""
    g = torch.Generator().manual_seed(0)
    vehicle = Mesh(
        points=torch.randn(3 * n_cells, 3, generator=g),
        cells=torch.arange(3 * n_cells).reshape(n_cells, 3),
        cell_data={
            "pMeanTrim": torch.randn(n_cells, generator=g),
            "wallShearStressMeanTrim": torch.randn(n_cells, 3, generator=g),
        },
        global_data={"TimeValue": torch.tensor(0.0)},
    )
    source = DomainMesh(
        interior=Mesh(points=torch.zeros(0, 3)),
        boundaries={"vehicle": vehicle, **_tunnel_boundaries()},
        global_data={
            "U_inf": torch.tensor([10.0, 0.0, 0.0]),
            "rho_inf": torch.tensor(2.0),
            "p_inf": torch.tensor(0.0),
            "L_ref": torch.tensor(5.0),
        },
    )
    path = tmp_path / "run_001" / "sample.pdmsh"
    path.parent.mkdir()
    source.save(path)
    return source


def test_surface_reader_loads_the_domain_boundaries_whole_beside_the_subsampled_vehicle(
    tmp_path,
):
    """With domain_boundaries the reader returns a DomainMesh: the vehicle, subsampled
    exactly as without them (same draw), as its interior, the named siblings whole as its
    boundaries, the case's global data at the domain level."""
    from merge_global_data import MeshReaderWithGlobalData

    source = _drivaer_surface_case(tmp_path)
    kw = dict(
        pattern="run_*/*.pdmsh/_tensordict/boundaries/vehicle",
        subsample_n_cells=10,
        merge_global_data_from="../../global_data",
    )
    bare = MeshReaderWithGlobalData(tmp_path, **kw)
    whole = MeshReaderWithGlobalData(
        tmp_path, domain_boundaries=list(DOMAIN_BOUNDARIES), **kw
    )
    for r in (bare, whole):
        r.set_generator(torch.Generator().manual_seed(7))
    mesh, _ = bare[0]
    domain, meta = whole[0]
    assert isinstance(domain, DomainMesh) and domain.interior.n_cells == 10
    assert torch.equal(domain.interior.points, mesh.points)
    assert torch.equal(domain.interior.cells, mesh.cells)
    assert (
        set(domain.boundary_names)
        == set(DOMAIN_BOUNDARIES)
        == set(meta["boundary_names"])
    )
    for name in DOMAIN_BOUNDARIES:
        assert torch.equal(
            domain.boundaries[name].points, source.boundaries[name].points
        )
        assert torch.equal(domain.boundaries[name].cells, source.boundaries[name].cells)
    assert set(domain.global_data.keys()) == set(mesh.global_data.keys())
    with pytest.raises(FileNotFoundError, match="domain boundary 'roof'"):
        MeshReaderWithGlobalData(tmp_path, domain_boundaries=["roof"], **kw)[0]


@pytest.mark.parametrize("dataset", ["drivaer_ml_surface", "drivaer_ml_surface_posed"])
def test_drivaer_surface_pipeline_moves_the_domain_boundaries_with_the_vehicle(
    tmp_path, dataset
):
    """Through the dataset's whole chain, augmentations included, the tunnel boundaries
    undergo the vehicle's rigid motion and scaling: the one similarity that maps the
    stored tunnel onto the output tunnel maps every output vehicle point onto a stored
    vehicle point. The targets sit at the vehicle's cell centroids, and the tunnel
    boundaries keep every cell."""
    source = _drivaer_surface_case(tmp_path)
    recipe = Path(__file__).resolve().parent.parent
    cfg = OmegaConf.merge(
        OmegaConf.load(recipe / "datasets" / f"{dataset}.yaml"),
        {"dataset_paths": {"drivaer_ml": str(tmp_path)}, "sampling_resolution": 10},
    )
    dataset_ = build_dataset(cfg, augment=True, device=None, num_workers=1)
    try:
        domain, _ = dataset_[0]
    finally:
        DatasetBase.close(dataset_)
    assert set(domain.boundary_names) == {"vehicle", *DOMAIN_BOUNDARIES}
    vehicle = domain.boundaries["vehicle"]
    assert vehicle.n_cells == domain.interior.n_points == 10
    torch.testing.assert_close(domain.interior.points, vehicle.cell_centroids)
    raw = torch.cat([source.boundaries[n].points for n in DOMAIN_BOUNDARIES]).double()
    out = torch.cat([domain.boundaries[n].points for n in DOMAIN_BOUNDARIES]).double()
    assert all(domain.boundaries[n].n_cells == 2 for n in DOMAIN_BOUNDARIES)
    ### The similarity out = s R raw + t, fitted on the tunnel (Kabsch with scale).
    rc, oc = raw - raw.mean(0), out - out.mean(0)
    u, sv, vt = torch.linalg.svd(rc.T @ oc)
    rot = (u @ vt).T
    scale = sv.sum() / rc.square().sum()
    shift = out.mean(0) - scale * raw.mean(0) @ rot.T
    torch.testing.assert_close(scale * raw @ rot.T + shift, out, atol=1e-5, rtol=0)
    torch.testing.assert_close(
        scale, torch.tensor(0.2, dtype=scale.dtype), atol=1e-6, rtol=0
    )  # 1 / L_ref
    mapped = scale * source.boundaries["vehicle"].points.double() @ rot.T + shift
    nearest = torch.cdist(vehicle.points.double(), mapped).min(dim=1).values
    assert nearest.max() < 1e-5


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
    wall = wall.with_data(
        cell_data=wall.cell_data.update({"normals": wall.cell_normals.clone()})
    )
    seg = Mesh(
        points=torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
        cells=torch.tensor([[0, 1], [1, 2]]),
    )
    inlet = _two_triangles()
    domain = DomainMesh(
        interior=Mesh(points=torch.randn(5, 3)),
        boundaries={"wall": wall, "inlet": inlet},
        global_data={"U": torch.ones(3)},
    )
    before = wall.cell_normals.clone()
    out = FlipBoundaryOrientation(
        boundaries=["wall"], normal_fields=["normals", "absent"]
    ).apply_to_domain(domain)
    flipped = out.boundaries["wall"]
    assert torch.equal(flipped.cell_normals, -before)
    assert torch.equal(flipped.cell_data["normals"], -before)
    assert torch.equal(flipped.cell_data["pressure"], wall.cell_data["pressure"])
    assert torch.allclose(flipped.cell_centroids, wall.cell_centroids)
    assert torch.allclose(cell_measures(flipped), cell_measures(wall))
    assert torch.equal(wall.cell_normals, before)  # the input mesh is not modified
    assert out.boundaries["inlet"] is inlet and out.interior is domain.interior
    assert torch.equal(out.global_data["U"], torch.ones(3))
    twice = FlipBoundaryOrientation(
        boundaries=["wall"], normal_fields=["normals"]
    ).apply_to_domain(out)
    assert torch.equal(twice.boundaries["wall"].cells, wall.cells)
    d2 = DomainMesh(
        interior=Mesh(points=torch.randn(3, 2)), boundaries={"airfoil": seg}
    )
    seg_n = seg.cell_normals.clone()
    assert torch.equal(
        FlipBoundaryOrientation(boundaries=["airfoil"])
        .apply_to_domain(d2)
        .boundaries["airfoil"]
        .cell_normals,
        -seg_n,
    )
    with pytest.raises(KeyError, match="'outlet'"):
        FlipBoundaryOrientation(boundaries=["outlet"]).apply_to_domain(domain)
