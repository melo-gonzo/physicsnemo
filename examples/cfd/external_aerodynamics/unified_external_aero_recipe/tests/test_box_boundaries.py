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
``AddBoxBoundaries`` from its dataset files' constants, and HiLift's far field and symmetry plane, read
from a boundary-only sidecar by ``MeshReaderWithGlobalData(domain_boundaries_root=...)``."""

from pathlib import Path

import pytest
import torch
from datasets import build_dataset
from domain_transforms import AddBoxBoundaries
from merge_global_data import MeshReaderWithGlobalData
from omegaconf import OmegaConf
from test_domain_transforms import _drivaer_surface_case

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


def test_surface_reader_reads_the_domain_boundaries_from_a_sidecar_root(tmp_path):
    """With domain_boundaries_root the named boundaries are read whole from a separate directory
    in the case files' relative layout (a boundary-only sidecar, as HiLift's far field), beside
    the subsampled surface read from the data; the surface's draw is the same as without them."""
    data, side = tmp_path / "data", tmp_path / "side"
    data.mkdir()
    source = _drivaer_surface_case(data)
    rel = Path("run_001") / "sample.pdmsh" / "_tensordict" / "boundaries"
    for name in ("inlet", "outlet"):
        source.boundaries[name].save(side / rel / name)
    kw = dict(
        pattern="run_*/*.pdmsh/_tensordict/boundaries/vehicle",
        subsample_n_cells=10,
        merge_global_data_from="../../global_data",
    )
    bare = MeshReaderWithGlobalData(data, **kw)
    whole = MeshReaderWithGlobalData(
        data,
        domain_boundaries=["inlet", "outlet"],
        domain_boundaries_root=str(side),
        **kw,
    )
    for r in (bare, whole):
        r.set_generator(torch.Generator().manual_seed(7))
    mesh, _ = bare[0]
    domain, _ = whole[0]
    assert torch.equal(domain.interior.points, mesh.points)
    assert set(domain.boundary_names) == {"inlet", "outlet"}
    for name in ("inlet", "outlet"):
        assert torch.equal(
            domain.boundaries[name].points, source.boundaries[name].points
        )
        assert torch.equal(domain.boundaries[name].cells, source.boundaries[name].cells)
    with pytest.raises(FileNotFoundError, match="domain boundary 'slip'"):
        MeshReaderWithGlobalData(
            data, domain_boundaries=["slip"], domain_boundaries_root=str(side), **kw
        )[0]
    with pytest.raises(ValueError, match="none are named"):
        MeshReaderWithGlobalData(data, domain_boundaries_root=str(side), **kw)


def test_highlift_whole_reads_the_sidecar_beside_the_wing():
    """highlift_surface_whole.yaml is highlift_surface.yaml with the far field and the symmetry
    plane read from the sidecar root and SurfaceMeshToDomainMesh as its terminal; every other
    step is the same."""
    body = OmegaConf.load(_RECIPE_ROOT / "datasets" / "highlift_surface.yaml")
    whole = OmegaConf.load(_RECIPE_ROOT / "datasets" / "highlift_surface_whole.yaml")
    reader = OmegaConf.to_container(whole.pipeline.reader, resolve=False)
    assert reader["domain_boundaries"] == ["inlet", "outlet", "symmetry"]
    assert (
        reader["domain_boundaries_root"]
        == "${dataset_paths.highlift_aero_ml_boundaries}"
    )
    rest = {
        k: v
        for k, v in reader.items()
        if k not in ("domain_boundaries", "domain_boundaries_root")
    }
    assert rest == OmegaConf.to_container(body.pipeline.reader, resolve=False)
    tb = OmegaConf.to_container(body.pipeline.transforms, resolve=False)
    tw = OmegaConf.to_container(whole.pipeline.transforms, resolve=False)
    assert str(tw[-1]["_target_"]) == "${dp:SurfaceMeshToDomainMesh}"
    assert str(tb[-1]["_target_"]) == "${dp:MeshToDomainMesh}"
    assert tw[:-1] == tb[:-1]
    assert OmegaConf.to_container(
        whole.pipeline.augmentations, resolve=False
    ) == OmegaConf.to_container(body.pipeline.augmentations, resolve=False)
    assert OmegaConf.to_container(whole.targets) == OmegaConf.to_container(body.targets)
