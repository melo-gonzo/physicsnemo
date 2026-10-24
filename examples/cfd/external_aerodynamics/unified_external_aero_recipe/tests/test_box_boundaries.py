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

"""The boundaries of the PDE domain that a case file does not store: SHIFT-SUV's CFD box, built by
``AddBoxBoundaries`` from its dataset files' constants, and HiLift's far field (inlet and outlet),
built by ``AddHemisphereBoundaries`` from its dataset file's constants and each case's freestream."""

import os
from pathlib import Path

import pytest
import torch
from datasets import build_dataset
from domain_transforms import AddBoxBoundaries
from omegaconf import OmegaConf

from physicsnemo.datapipes.protocols import DatasetBase
from physicsnemo.mesh import DomainMesh, Mesh

_RECIPE_ROOT = Path(__file__).resolve().parents[1]

#: SHIFT-SUV's CFD box (the interior's bounding box on all 1,991 case files) and its pieces.
SUV_BOX = [[-40.0, 80.0], [-20.0, 20.0], [-0.024, 19.976]]
SUV_FACES = {
    "inlet": ["x_min"],
    "outlet": ["x_max"],
    "floor": ["z_min"],
    "side_walls": ["y_min", "y_max"],
    "roof": ["z_max"],
}

#: The SHIFT-SUV dataset files and the subdirectory of the dataset root each reads.
SUV_DATASETS = {
    "shift_suv_surface": "",
    "shift_suv_surface_area": "",
    "shift_suv_surface_posed": "",
    "shift_suv_estate_surface": "estate",
    "shift_suv_fastback_surface": "fastback",
}


def _suv_vehicle(n_cells: int = 30) -> Mesh:
    """A random vehicle surface with SHIFT-SUV's raw field names, inside the box."""
    g = torch.Generator().manual_seed(0)
    return Mesh(
        points=torch.rand(3 * n_cells, 3, generator=g) * torch.tensor([4.6, 2.0, 1.6])
        + torch.tensor([-0.8, -1.0, -0.0228]),
        cells=torch.arange(3 * n_cells).reshape(n_cells, 3),
        cell_data={
            "pressure_average": torch.randn(n_cells, generator=g) + 101325.0,
            "wall_shear_stress_average": torch.randn(n_cells, 3, generator=g),
        },
        global_data={"TimeValue": torch.tensor(0.0)},
    )


def test_add_box_boundaries_builds_the_closed_box_into_the_fluid():
    """On a bare surface the transform returns the reader's DomainMesh shape: the surface as the
    interior, the case's global_data at the domain level, and the six faces of the box as the
    named boundaries, each a 20 x 20 grid (722 triangles; side_walls two of them) on its plane,
    every normal into the box, the faces covering the box's surface once."""
    vehicle = _suv_vehicle()
    out = AddBoxBoundaries(bounds=SUV_BOX, faces=SUV_FACES)(vehicle)
    assert isinstance(out, DomainMesh)
    assert torch.equal(out.interior.points, vehicle.points)
    assert len(out.interior.global_data.keys()) == 0
    assert torch.equal(out.global_data["TimeValue"], torch.tensor(0.0))
    assert set(out.boundary_names) == set(SUV_FACES)
    box = torch.tensor(SUV_BOX)
    centre = box.mean(dim=1)
    area = 0.0
    for name, faces in SUV_FACES.items():
        b = out.boundaries[name]
        assert b.n_cells == 722 * len(faces) and b.n_points == 400 * len(faces)
        for f in faces:
            axis, side = "xyz".index(f[0]), 0 if f.endswith("min") else 1
            assert int((b.points[:, axis] == box[axis, side]).sum()) == 400
        ### Into the box: each normal is its face's axis, signed toward the centre.
        inward = ((centre - b.cell_centroids) * b.cell_normals).sum(dim=1)
        assert bool((inward > 0).all())
        torch.testing.assert_close(
            b.cell_normals.abs().max(dim=1).values, torch.ones(b.n_cells)
        )
        assert bool((b.points >= box[:, 0]).all() and (b.points <= box[:, 1]).all())
        area += float(b.cell_areas.double().sum())
    ext = (box[:, 1] - box[:, 0]).double()
    total = 2 * float(ext[0] * ext[1] + ext[1] * ext[2] + ext[0] * ext[2])
    assert area == pytest.approx(total, rel=1e-6)
    ### curator's grid (box_face_patch): linspace along both in-plane axes.
    inlet = out.boundaries["inlet"].points
    assert torch.equal(torch.unique(inlet[:, 1]), torch.linspace(-20.0, 20.0, 20))
    assert torch.equal(torch.unique(inlet[:, 2]), torch.linspace(-0.024, 19.976, 20))


def test_add_box_boundaries_on_a_domain_and_what_it_refuses():
    """On a DomainMesh the faces join its boundaries; a name it has, a face missing or held
    twice, and an empty box are refused."""
    t = AddBoxBoundaries(bounds=SUV_BOX, faces=SUV_FACES, n_per_side=(3, 4))
    domain = t(_suv_vehicle())
    assert domain.boundaries["roof"].n_cells == 2 * 2 * 3
    with_vehicle = DomainMesh(
        interior=domain.interior,
        boundaries={"vehicle": domain.interior},
        global_data=domain.global_data,
    )
    both = t.apply_to_domain(with_vehicle)
    assert set(both.boundary_names) == {"vehicle", *SUV_FACES}
    assert torch.equal(both.boundaries["vehicle"].points, domain.interior.points)
    with pytest.raises(ValueError, match="exist already"):
        t.apply_to_domain(both)
    with pytest.raises(ValueError, match="exactly once"):
        AddBoxBoundaries(bounds=SUV_BOX, faces={**SUV_FACES, "roof": ["z_min"]})
    with pytest.raises(ValueError, match="exactly once"):
        AddBoxBoundaries(
            bounds=SUV_BOX, faces={k: v for k, v in SUV_FACES.items() if k != "roof"}
        )
    with pytest.raises(ValueError, match="min < max"):
        AddBoxBoundaries(bounds=[[0.0, 0.0], [0.0, 1.0], [0.0, 1.0]], faces=SUV_FACES)


@pytest.mark.parametrize("dataset", sorted(SUV_DATASETS))
def test_shift_suv_pipeline_moves_the_box_with_the_vehicle(tmp_path, dataset):
    """Each SHIFT-SUV dataset file builds the box first, from its constants, and ends with
    SurfaceMeshToDomainMesh. Through its whole chain, augmentations included, the box undergoes
    the vehicle's rigid motion and scaling: the one similarity that maps the raw box onto the
    output box maps every output vehicle point onto a stored one. The box keeps every cell, and
    the targets sit at the vehicle's cell centroids. (The estate and fastback files subsample the
    vehicle with a SubsampleMesh step, which acts on every boundary of the DomainMesh: the
    resolution here, as the configured 200,000, is above the box's 1,444 cells per piece.)"""
    vehicle = _suv_vehicle(n_cells=2000)
    source = DomainMesh(
        interior=Mesh(points=torch.zeros(0, 3)),
        boundaries={"vehicle": vehicle},
        global_data={
            "U_inf": torch.tensor([30.0, 0.0, 0.0]),
            "rho_inf": torch.tensor(1.225),
            "p_inf": torch.tensor(101325.0),
            "L_ref": torch.tensor(4.6),
        },
    )
    path = tmp_path / SUV_DATASETS[dataset] / "run_00001" / "domain.pdmsh"
    path.parent.mkdir(parents=True)
    source.save(path)
    cfg = OmegaConf.merge(
        OmegaConf.load(_RECIPE_ROOT / "datasets" / f"{dataset}.yaml"),
        {"dataset_paths": {"shift_suv": str(tmp_path)}, "sampling_resolution": 1500},
    )
    raw_cfg = OmegaConf.to_container(cfg.pipeline.transforms, resolve=False)
    assert str(raw_cfg[0]["_target_"]) == "${dp:AddBoxBoundaries}"
    assert raw_cfg[0]["bounds"] == SUV_BOX and raw_cfg[0]["faces"] == SUV_FACES
    assert "n_per_side" not in raw_cfg[0]  # curator's 20 x 20, the class default
    assert sum("AddBoxBoundaries" in str(t["_target_"]) for t in raw_cfg) == 1
    assert str(raw_cfg[-1]["_target_"]) == "${dp:SurfaceMeshToDomainMesh}"
    dataset_ = build_dataset(cfg, augment=True, device=None, num_workers=1)
    try:
        domain, _ = dataset_[0]
    finally:
        DatasetBase.close(dataset_)
    assert set(domain.boundary_names) == {"vehicle", *SUV_FACES}
    out_vehicle = domain.boundaries["vehicle"]
    assert domain.interior.n_points == out_vehicle.n_cells
    torch.testing.assert_close(domain.interior.points, out_vehicle.cell_centroids)
    box = AddBoxBoundaries(bounds=SUV_BOX, faces=SUV_FACES)(vehicle)
    names = sorted(SUV_FACES)
    for n in names:
        assert domain.boundaries[n].n_cells == box.boundaries[n].n_cells
    raw = torch.cat([box.boundaries[n].points for n in names]).double()
    out = torch.cat([domain.boundaries[n].points for n in names]).double()
    ### The similarity out = s R raw + t, fitted on the box (Kabsch with scale).
    rc, oc = raw - raw.mean(0), out - out.mean(0)
    u, sv, vt = torch.linalg.svd(rc.T @ oc)
    rot = (u @ vt).T
    scale = sv.sum() / rc.square().sum()
    shift = out.mean(0) - scale * raw.mean(0) @ rot.T
    torch.testing.assert_close(scale * raw @ rot.T + shift, out, atol=1e-4, rtol=0)
    torch.testing.assert_close(
        scale, torch.tensor(1 / 4.6, dtype=scale.dtype), atol=1e-6, rtol=0
    )  # 1 / L_ref
    mapped = scale * vehicle.points.double() @ rot.T + shift
    nearest = torch.cdist(out_vehicle.points.double(), mapped).min(dim=1).values
    assert nearest.max() < 1e-5


#: HiLift's far field (the stored data's radius, curator's divisions) and the stored U_inf of three
#: angles of attack, bit for bit (every case of an angle stores the same vector; 10 vectors in all).
HL_RADIUS = 24533.41796875
HL_U_INF = {
    4: "a50f274500000000ade93a43",
    12: "39cf2345000000006d460b44",
    22: "4d461b4500000000a6f07a44",
}


def _u_inf(aoa: int) -> torch.Tensor:
    return torch.frombuffer(
        bytearray(bytes.fromhex(HL_U_INF[aoa])), dtype=torch.float32
    )


def _hl_wing(u_inf: torch.Tensor, n_cells: int = 2000) -> Mesh:
    """A random wing surface with HiLift's raw field names and freestream, in inches, y >= 0."""
    g = torch.Generator().manual_seed(1)
    return Mesh(
        points=torch.rand(3 * n_cells, 3, generator=g)
        * torch.tensor([2500.0, 1500.0, 600.0])
        + torch.tensor([90.0, 0.0, 90.0]),
        cells=torch.arange(3 * n_cells).reshape(n_cells, 3),
        cell_data={
            "PROJ(AVG(P))": 0.1 + 0.01 * torch.rand(n_cells, generator=g),
            "PROJ(AVG(T))": 500.0 + torch.rand(n_cells, generator=g),
            "PROJ(AVG(RHO))": 2e-3 + 1e-5 * torch.rand(n_cells, generator=g),
            "PROJ(AVG(U))": torch.randn(n_cells, 3, generator=g),
            "AVG(TAU_WALL)": 1e-4 * torch.randn(n_cells, 3, generator=g),
        },
        global_data={
            "U_inf": u_inf.clone(),
            "p_inf": torch.tensor(0.1),
            "rho_inf": torch.tensor(2e-3),
            "T_inf": torch.tensor(500.0),
            "L_ref": torch.tensor(275.8),
        },
    )


def _canonical(m: Mesh) -> tuple[torch.Tensor, torch.Tensor]:
    """A mesh's points in sorted order and its oriented triangles as sorted rows of vertex
    coordinates, each rotated to start at its smallest vertex: equal for two meshes that hold the
    same points and the same oriented cells in any order."""
    import numpy as np

    p = m.points.numpy()
    t = p[m.cells.numpy()]
    first = np.lexsort((t[:, :, 2], t[:, :, 1], t[:, :, 0]), axis=-1)[:, 0]
    t = np.take_along_axis(
        t, ((first[:, None] + np.arange(3)) % 3)[:, :, None], axis=1
    ).reshape(len(t), 9)
    return (
        torch.from_numpy(p[np.lexsort(p.T[::-1])]),
        torch.from_numpy(t[np.lexsort(t.T[::-1])]),
    )


@pytest.mark.parametrize("aoa", sorted(HL_U_INF))
def test_add_hemisphere_boundaries_builds_the_far_field_into_the_fluid(aoa):
    """On a bare surface the transform returns the reader's DomainMesh shape with the inlet and
    the outlet: together curator's whole hemisphere (64 x 32 divisions, 3,904 cells), every point
    on the sphere of the radius with y >= 0, every normal toward the origin (into the fluid), the
    inlet's cells upstream (centroid . U_inf <= 0) and the outlet's downstream, sharing only their
    seam. The wing does not enter: another wing with the same U_inf gives the same far field."""
    from domain_transforms import AddHemisphereBoundaries

    u = _u_inf(aoa)
    wing = _hl_wing(u)
    t = AddHemisphereBoundaries(radius=HL_RADIUS)
    out = t(wing)
    assert isinstance(out, DomainMesh)
    assert torch.equal(out.interior.points, wing.points)
    assert len(out.interior.global_data.keys()) == 0
    assert torch.equal(out.global_data["U_inf"], u)
    assert set(out.boundary_names) == {"inlet", "outlet"}
    inlet, outlet = out.boundaries["inlet"], out.boundaries["outlet"]
    assert inlet.n_cells + outlet.n_cells == 64 + 2 * 64 * 30
    direction = (u / u.norm()).double()
    for m, upstream in ((inlet, True), (outlet, False)):
        assert m.points.dtype == torch.float32
        r = m.points.double().norm(dim=1)
        torch.testing.assert_close(r, torch.full_like(r, HL_RADIUS), atol=0.02, rtol=0)
        assert bool((m.points[:, 1] >= 0).all())
        assert bool(((m.cell_centroids * m.cell_normals).sum(dim=1) < 0).all())
        along = m.cell_centroids.double() @ direction
        assert bool((along <= 0).all() if upstream else (along > 0).all())
        assert torch.unique(m.cells).numel() == m.n_points  # no unused point
    seam = (inlet.points[:, None, :] == outlet.points[None, :, :]).all(-1)
    assert 0 < int(seam.sum()) < min(inlet.n_points, outlet.n_points)
    other = t(_hl_wing(u, n_cells=50))
    for n in ("inlet", "outlet"):
        assert torch.equal(other.boundaries[n].points, out.boundaries[n].points)
        assert torch.equal(other.boundaries[n].cells, out.boundaries[n].cells)


def test_add_hemisphere_boundaries_on_a_domain_and_what_it_refuses():
    """On a DomainMesh the inlet and outlet join its boundaries, split by the domain-level U_inf;
    a name it has, a missing freestream and a radius <= 0 are refused."""
    from domain_transforms import AddHemisphereBoundaries

    t = AddHemisphereBoundaries(radius=HL_RADIUS)
    wing = _hl_wing(_u_inf(12))
    bare = t(wing)
    domain = DomainMesh(
        interior=bare.interior,
        boundaries={"vehicle": wing},
        global_data=bare.global_data,
    )
    both = t.apply_to_domain(domain)
    assert set(both.boundary_names) == {"vehicle", "inlet", "outlet"}
    for n in ("inlet", "outlet"):
        assert torch.equal(both.boundaries[n].points, bare.boundaries[n].points)
    with pytest.raises(ValueError, match="exist already"):
        t.apply_to_domain(both)
    with pytest.raises(KeyError, match="U_inf"):
        t(Mesh(points=wing.points, cells=wing.cells))
    with pytest.raises(ValueError, match="radius"):
        AddHemisphereBoundaries(radius=0.0)


def _hl_dataset_cfg(tmp_path, posed: bool):
    cfg = OmegaConf.merge(
        OmegaConf.load(_RECIPE_ROOT / "datasets" / "highlift_surface.yaml"),
        {
            "dataset_paths": {
                "highlift_aero_ml": str(tmp_path),
                "highlift_aero_ml_duplicate_masks": None,
            },
            "sampling_resolution": 1500,
        },
    )
    if posed:
        ### The posed protocol of the other datasets' _posed files: the pose right after the far
        ### field, seeded by the case key.
        cfg.pipeline.reader.store_case_key = True
        transforms = OmegaConf.to_container(cfg.pipeline.transforms, resolve=False)
        transforms.insert(
            1, {"_target_": "${dp:FixedRandomPose}", "salt": 0, "translate": 1.0}
        )
        cfg.pipeline.transforms = transforms
    return cfg


@pytest.mark.parametrize("posed", [False, True], ids=["native", "posed"])
def test_highlift_pipeline_moves_the_far_field_with_the_wing(tmp_path, posed):
    """highlift_surface.yaml builds the far field first, from its constants and the case's U_inf,
    and ends with SurfaceMeshToDomainMesh. Through its whole chain (the deduplicating reader, the
    posed protocol's FixedRandomPose where set, the augmentations, CenterMesh and the L_ref scaling)
    the far field undergoes the wing's rigid motion and scaling: the one similarity that maps the
    raw far field onto the output one maps every output wing point onto a stored one."""
    from domain_transforms import AddHemisphereBoundaries

    wing = _hl_wing(_u_inf(22))
    source = DomainMesh(
        interior=Mesh(points=torch.zeros(0, 3)),
        boundaries={"vehicle": wing.with_data(global_data={})},
        global_data=wing.global_data,
    )
    path = tmp_path / "geo_LHC001_AoA_22" / "domain_LHC001_AoA_22.pdmsh"
    path.parent.mkdir(parents=True)
    source.save(path)
    cfg = _hl_dataset_cfg(tmp_path, posed)
    raw_cfg = OmegaConf.to_container(cfg.pipeline.transforms, resolve=False)
    assert str(raw_cfg[0]["_target_"]) == "${dp:AddHemisphereBoundaries}"
    assert raw_cfg[0]["radius"] == HL_RADIUS
    assert (raw_cfg[0]["n_theta"], raw_cfg[0]["n_phi"]) == (64, 32)
    assert sum("AddHemisphereBoundaries" in str(t["_target_"]) for t in raw_cfg) == 1
    assert str(raw_cfg[-1]["_target_"]) == "${dp:SurfaceMeshToDomainMesh}"
    dataset_ = build_dataset(cfg, augment=True, device=None, num_workers=1)
    try:
        domain, _ = dataset_[0]
    finally:
        DatasetBase.close(dataset_)
    assert set(domain.boundary_names) == {"vehicle", "inlet", "outlet"}
    out_wing = domain.boundaries["vehicle"]
    torch.testing.assert_close(domain.interior.points, out_wing.cell_centroids)
    far = AddHemisphereBoundaries(radius=HL_RADIUS)(wing)
    names = ("inlet", "outlet")
    for n in names:
        assert torch.equal(domain.boundaries[n].cells, far.boundaries[n].cells)
    raw = torch.cat([far.boundaries[n].points for n in names]).double()
    out = torch.cat([domain.boundaries[n].points for n in names]).double()
    rc, oc = raw - raw.mean(0), out - out.mean(0)
    u, sv, vt = torch.linalg.svd(rc.T @ oc)
    rot = (u @ vt).T
    scale = sv.sum() / rc.square().sum()
    shift = out.mean(0) - scale * raw.mean(0) @ rot.T
    torch.testing.assert_close(scale * raw @ rot.T + shift, out, atol=1e-3, rtol=0)
    torch.testing.assert_close(
        scale, torch.tensor(1 / 275.8, dtype=scale.dtype), atol=1e-7, rtol=0
    )  # 1 / L_ref
    mapped = scale * wing.points.double() @ rot.T + shift
    nearest = torch.cdist(out_wing.points.double(), mapped).min(dim=1).values
    assert nearest.max() < 1e-4


_HL_ROOT = os.environ.get("HIGHLIFT_AERO_ML_ROOT")


@pytest.mark.skipif(
    _HL_ROOT is None,
    reason="needs HiLiftAeroML case files (set HIGHLIFT_AERO_ML_ROOT to the dataset root)",
)
@pytest.mark.parametrize(
    "case,stored",
    [
        ("geo_LHC001_AoA_10", "geo_LHC001_AoA_10"),
        ("geo_LHC001_AoA_12", "geo_LHC001_AoA_12"),
        ("geo_LHC002_AoA_10", "geo_LHC001_AoA_10"),
    ],
)
def test_add_hemisphere_boundaries_reproduces_the_stored_far_field(case, stored):
    """On real case files, the far field the transform builds from a case's own U_inf holds the
    points and oriented cells, bit for bit, of the far field curator stored on LHC001 at the same
    angle of attack (the stored files' order is curator's random permutation, so the comparison is
    order-free). LHC002_AoA_10 stores no far field: its U_inf equals LHC001_AoA_10's bit for bit."""
    from domain_transforms import AddHemisphereBoundaries
    from tensordict import TensorDict

    def pdmsh(c):
        (p,) = sorted((Path(_HL_ROOT) / c).glob("domain_*.pdmsh"))
        return p / "_tensordict"

    u_inf = TensorDict.load_memmap(pdmsh(case) / "global_data")["U_inf"].clone()
    built = AddHemisphereBoundaries(radius=HL_RADIUS)(
        Mesh(points=torch.zeros(1, 3), global_data={"U_inf": u_inf})
    )
    for n in ("inlet", "outlet"):
        ref = Mesh.load(pdmsh(stored) / "boundaries" / n)
        (pg, tg), (pr, tr) = _canonical(built.boundaries[n]), _canonical(ref)
        assert torch.equal(pg, pr), n
        assert torch.equal(tg, tr), n
