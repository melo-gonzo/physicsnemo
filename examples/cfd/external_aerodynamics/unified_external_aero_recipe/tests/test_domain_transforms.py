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
    ### The ground placed from the whole vehicle, which the subsample does not hold: the
    ### tunnel moves up by the whole vehicle's lowest z minus the ground's, the vehicle
    ### draw is the same.
    from domain_transforms import PlaceGroundUnderBody

    place = PlaceGroundUnderBody(
        ground="no_slip", body="vehicle", boundaries=list(DOMAIN_BOUNDARIES)
    )
    placed = MeshReaderWithGlobalData(
        tmp_path,
        domain_boundaries=list(DOMAIN_BOUNDARIES),
        whole_surface_transforms=[place],
        **kw,
    )
    placed.set_generator(torch.Generator().manual_seed(7))
    pdomain, _ = placed[0]
    lowest = source.boundaries["vehicle"].points[:, 2].min()
    assert (
        pdomain.interior.points[:, 2].min() > lowest
    )  # the draw misses the lowest vertex
    assert torch.equal(pdomain.interior.points, mesh.points)
    dz = lowest - source.boundaries["no_slip"].points[0, 2]
    assert dz != 0
    for name in DOMAIN_BOUNDARIES:
        moved = source.boundaries[name].points + torch.tensor([0.0, 0.0, 1.0]) * dz
        torch.testing.assert_close(
            pdomain.boundaries[name].points, moved, atol=0, rtol=0
        )
    assert bool((pdomain.boundaries["no_slip"].points[:, 2] == lowest).all())
    with pytest.raises(KeyError, match="not found"):  # no domain boundaries to place
        MeshReaderWithGlobalData(tmp_path, whole_surface_transforms=[place], **kw)[0]
    with pytest.raises(FileNotFoundError, match="domain boundary 'roof'"):
        MeshReaderWithGlobalData(tmp_path, domain_boundaries=["roof"], **kw)[0]


@pytest.mark.parametrize("dataset", ["drivaer_ml_surface", "drivaer_ml_surface_posed"])
def test_drivaer_surface_pipeline_moves_the_domain_boundaries_with_the_vehicle(
    tmp_path, dataset
):
    """Through the dataset's whole chain, augmentations included, the tunnel boundaries
    undergo the vehicle's rigid motion and scaling: the one similarity that maps the
    stored tunnel onto the output tunnel maps every output vehicle point onto a stored
    vehicle point, the stored tunnel being first placed with its ground through the whole
    vehicle's lowest point (PlaceGroundUnderBody in the reader). The targets sit at the
    vehicle's cell centroids, and the tunnel boundaries keep every cell."""
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
    dz = (
        source.boundaries["vehicle"].points[:, 2].min()
        - source.boundaries["no_slip"].points[0, 2]
    )
    raw = torch.cat([source.boundaries[n].points for n in DOMAIN_BOUNDARIES]).double()
    raw[:, 2] += float(dz)
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


def _tunnel_case(ground: float, roof: float, lowest: float) -> DomainMesh:
    """A DrivAerML-like case: a body whose lowest point is at z = lowest, interior nodes
    from lowest to lowest + 20, and the four box meshes with their ground at z = ground
    and their roof at z = roof (float32, as stored)."""

    def rect(corners):
        return Mesh(
            points=torch.tensor(corners), cells=torch.tensor([[0, 1, 2], [0, 2, 3]])
        )

    x0, x1, y0, y1, xb = -40.0, 80.0, -22.0, 22.0, -2.339
    walls = {
        "no_slip": rect(
            [[xb, y0, ground], [x1, y0, ground], [x1, y1, ground], [xb, y1, ground]]
        ),
        "slip": rect([[x0, y0, roof], [x0, y1, roof], [x1, y1, roof], [x1, y0, roof]]),
        "inlet": rect(
            [[x0, y0, ground], [x0, y1, ground], [x0, y1, roof], [x0, y0, roof]]
        ),
        "outlet": rect(
            [[x1, y0, ground], [x1, y0, roof], [x1, y1, roof], [x1, y1, ground]]
        ),
    }
    body = Mesh(
        points=torch.tensor(
            [[0.0, -1.0, lowest], [3.0, -1.0, lowest + 0.1], [1.0, 1.0, 1.1]]
        ),
        cells=torch.tensor([[0, 1, 2]]),
    )
    interior = Mesh(
        points=torch.tensor(
            [[x0, y0, lowest], [x1, y1, lowest + 20.0], [0.0, 0.0, 0.5]]
        )
    )
    return DomainMesh(
        interior=interior,
        boundaries={"vehicle": body, **walls},
        global_data={"L_ref": torch.tensor(5.0)},
    )


def test_place_ground_under_body_moves_the_box_onto_the_cfd_domain():
    """On a case stored 2.4 mm low (ground -0.32, roof 19.68, wheels at -0.317597) the four
    box meshes move up by the body's lowest z minus the ground's, in float32: the ground lands
    on the wheels and the roof on the interior's highest node exactly, x, y and the cells stay;
    the body, the interior and global_data are the inputs'. A matching case moves by zero."""
    from domain_transforms import PlaceGroundUnderBody

    walls = ["no_slip", "slip", "inlet", "outlet"]
    t = PlaceGroundUnderBody(ground="no_slip", body="vehicle", boundaries=walls)
    lowest = -0.317597  # the stored vehicle's lowest z, as float32
    low = _tunnel_case(-0.32, 19.68, lowest)
    out = t.apply_to_domain(low)
    zmin = float(low.interior.points[:, 2].min())
    zmax = float(low.interior.points[:, 2].max())
    assert zmax == 19.682403564453125
    for name in walls:
        before, after = low.boundaries[name], out.boundaries[name]
        assert torch.equal(after.points[:, :2], before.points[:, :2])
        assert torch.equal(after.cells, before.cells)
        dz = after.points[:, 2].double() - before.points[:, 2].double()
        assert float(dz.min()) > 0.0024 and float(dz.max()) < 0.00241
    ### Ground on the wheels, roof (here all of slip) on the highest node, the end walls between.
    for name, (lo, hi) in {
        "no_slip": (zmin, zmin),
        "slip": (zmax, zmax),
        "inlet": (zmin, zmax),
        "outlet": (zmin, zmax),
    }.items():
        z = out.boundaries[name].points[:, 2]
        assert (float(z.min()), float(z.max())) == (lo, hi), name
    assert (
        out.boundaries["vehicle"] is low.boundaries["vehicle"]
        and out.interior is low.interior
    )
    assert torch.equal(out.global_data["L_ref"], torch.tensor(5.0))
    ### A matching file (the 187): zero shift, every point as stored.
    ok = _tunnel_case(lowest, zmax, lowest)
    same = t.apply_to_domain(ok)
    for name in walls:
        assert torch.equal(same.boundaries[name].points, ok.boundaries[name].points)
    ### A ground above the body moves down onto it.
    high = t.apply_to_domain(_tunnel_case(-0.30, 19.70, lowest))
    assert float(high.boundaries["no_slip"].points[:, 2].max()) == zmin
    assert t(low.interior) is low.interior  # a bare Mesh passes through


def test_place_ground_under_body_refuses_what_it_cannot_place():
    """A tilted ground, a missing boundary, a ground outside the moved set or a body inside it
    are refused rather than guessed."""
    from domain_transforms import PlaceGroundUnderBody

    walls = ["no_slip", "slip", "inlet", "outlet"]
    case = _tunnel_case(-0.32, 19.68, -0.317597)
    pts = case.boundaries["no_slip"].points.clone()
    pts[0, 2] += 1e-3
    tilted = Mesh(points=pts, cells=case.boundaries["no_slip"].cells)
    bad = DomainMesh(
        interior=case.interior, boundaries={**case.boundaries, "no_slip": tilted}
    )
    t = PlaceGroundUnderBody(ground="no_slip", body="vehicle", boundaries=walls)
    with pytest.raises(ValueError, match="not one plane"):
        t.apply_to_domain(bad)
    with pytest.raises(KeyError, match="'roof'"):
        PlaceGroundUnderBody(
            ground="no_slip", body="vehicle", boundaries=[*walls, "roof"]
        ).apply_to_domain(case)
    with pytest.raises(ValueError, match="among the boundaries"):
        PlaceGroundUnderBody(ground="floor", body="vehicle", boundaries=walls)
    with pytest.raises(ValueError, match="must not be among"):
        PlaceGroundUnderBody(
            ground="no_slip", body="vehicle", boundaries=[*walls, "vehicle"]
        )
