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

"""The ISLA recipe configurations compose to the mesh contract and run end to end.

``model=isla_surface`` (surface task: predictions at the vehicle's cells, the whole boundary read) and
``model=isla_volume`` (interior task: predictions at the
dataset's interior points) all declare ``input_type: mesh`` / ``output_type: mesh`` and
map the DomainMesh's boundary and global data onto ISLA's keyword-only mesh forward.
Each config is composed with Hydra, instantiated at a reduced width, and driven through
collate -> resolve_forward_kwargs -> ISLA.forward -> normalize_output_to_tensordict ->
loss on a tiny synthetic DomainMesh shaped like the paired dataset's output."""

import sys
from pathlib import Path

import hydra
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

_RECIPE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RECIPE_ROOT / "src"))

from collate import build_collate_fn  # noqa: E402
from conftest import make_volume_domain_mesh  # noqa: E402
from domain_transforms import ComposeQuadratureMeasure  # noqa: E402
from loss import LossCalculator  # noqa: E402
from merge_global_data import SurfaceMeshToDomainMesh  # noqa: E402
from output_normalize import normalize_output_to_tensordict  # noqa: E402

from physicsnemo.mesh import DomainMesh, Mesh  # noqa: E402
from physicsnemo.mesh.calculus.measure import (  # noqa: E402
    cell_measures,
    scale_measures,
)

ISLA_TARGET = "physicsnemo.experimental.models.isla.ISLA"
SMALL = [
    "model.hidden=16",
    "model.n_layers=1",
    "model.n_slices=4",
]
CONFIGS = [
    ("isla_surface", "drivaer_ml_surface"),
    ("isla_volume", "drivaer_ml_volume_reference"),
]


def _compose(model, dataset, *extra):
    """The recipe's model configs are `@package _global_`: the architecture block lands at cfg.model and
    the data-to-model mapping at cfg.forward_kwargs."""
    with initialize_config_dir(
        config_dir=str(_RECIPE_ROOT / "conf"), version_base=None
    ):
        return compose(
            config_name="train",
            overrides=[f"model={model}", f"dataset={dataset}", "+out_dim=5", *extra],
        )


def _surface_domain(n_cells: int = 12) -> DomainMesh:
    """What drivaer_ml_surface.yaml hands the collate: a 1-in-3 cell subsample of a surface
    with its Horvitz-Thompson measure and the four tunnel boundaries read whole beside it,
    then SurfaceMeshToDomainMesh (interior = the vehicle's cell centroids carrying the
    targets; boundaries.vehicle = the surface, its cell_data keeping the measure and the
    `normals` field the tensor-input models read; the tunnel boundaries as they were)."""
    g = torch.Generator().manual_seed(0)
    mesh = Mesh(
        points=torch.randn(n_cells * 3, 3, generator=g),
        cells=torch.arange(n_cells * 3).reshape(n_cells, 3),
        cell_data={
            "pressure": torch.randn(n_cells, generator=g),
            "wss": torch.randn(n_cells, 3, generator=g),
        },
        global_data={
            "U_inf": torch.tensor([30.0, 0.0, 0.0]),
            "U_inf_dir": torch.tensor([1.0, 0.0, 0.0]),
            "L_ref": torch.tensor(5.0),
        },
    )
    scale_measures(mesh, 3.0)
    mesh.cell_data["normals"] = mesh.cell_normals
    walls = {
        name: _wall(axis, c, up)
        for name, axis, c, up in zip(
            DOMAIN_WALLS,
            (2, 2, 0, 0),
            (-3.0, 3.0, -4.0, 4.0),
            (True, False, True, False),
        )
    }
    gd = mesh.global_data
    surface = Mesh(points=mesh.points, cells=mesh.cells, cell_data=mesh.cell_data)
    sample = DomainMesh(interior=surface, boundaries=walls, global_data=gd)
    return SurfaceMeshToDomainMesh(
        cell_data_targets=["pressure", "wss"]
    ).apply_to_domain(sample)


DOMAIN_WALLS = ("no_slip", "slip", "inlet", "outlet")


def _wall(axis: int, c: float, up: bool) -> Mesh:
    """A square of two triangles at coordinate c along axis, its normals +axis (up)
    or -axis."""
    cells = [[0, 1, 2], [0, 2, 3]] if up else [[0, 2, 1], [0, 3, 2]]
    square = torch.tensor([[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]])
    points = torch.empty(4, 3)
    points[:, axis] = c
    points[:, (axis + 1) % 3], points[:, (axis + 2) % 3] = square[:, 0], square[:, 1]
    return Mesh(points=points, cells=torch.tensor(cells))


def _volume_domain() -> DomainMesh:
    """What drivaer_ml_volume_reference.yaml hands the collate: an interior point cloud
    with sdf / sdf_normals and the targets, the whole vehicle boundary and the four
    domain meshes the case files store (no_slip, the ground; slip; inlet; outlet; no
    cell data, normals into the fluid), each with its measure materialized by
    ComposeQuadratureMeasure, and U_inf_dir in global_data."""
    domain = make_volume_domain_mesh(n_pts=64)
    boundaries = {"vehicle": domain.boundaries["vehicle"]}
    for name, axis, c, up in zip(
        DOMAIN_WALLS, (2, 2, 0, 0), (-3.0, 3.0, -4.0, 4.0), (True, False, True, False)
    ):
        boundaries[name] = _wall(axis, c, up)
    gd = domain.global_data.clone()
    gd["U_inf_dir"] = torch.tensor([1.0, 0.0, 0.0])
    return DomainMesh(
        interior=domain.interior,
        boundaries={k: ComposeQuadratureMeasure()(m) for k, m in boundaries.items()},
        global_data=gd,
    )


def _targets(dataset: str) -> dict:
    return OmegaConf.to_container(
        OmegaConf.load(_RECIPE_ROOT / "datasets" / f"{dataset}.yaml").targets
    )


def _run(cfg, domain: DomainMesh, targets: dict):
    """collate -> forward -> normalize -> loss for one composed config."""
    model = hydra.utils.instantiate(cfg.model)
    spec = OmegaConf.to_container(cfg.forward_kwargs, resolve=True)
    batch = build_collate_fn(cfg.input_type, spec, targets)([(domain, {})])
    with torch.no_grad():
        out = model(**batch["forward_kwargs"])
    assert isinstance(out, Mesh)
    pred = normalize_output_to_tensordict(out, targets, cfg.output_type)
    total, td = LossCalculator(targets, loss_type="mse")(
        pred.float(), batch["targets"].float(), batch.get("target_measure")
    )
    assert torch.isfinite(total)
    assert set(td.keys()) >= {f"loss/{name}" for name in targets}
    return batch, out, pred


@pytest.mark.parametrize("model, dataset", CONFIGS)
def test_isla_configs_declare_the_mesh_contract(model, dataset):
    """Every ISLA config is mesh in, mesh out; the model's outputs are the dataset's
    targets by name and rank, and the freestream direction is declared under the
    dataset's own leaf name so the model can select it from the pass-through
    global_data."""
    cfg = _compose(model, dataset)
    m = cfg.model
    assert m._target_ == ISLA_TARGET
    assert cfg.input_type == "mesh" and cfg.output_type == "mesh"
    ### Both read the whole boundary: the vehicle and the four domain meshes of the files.
    names = ("vehicle", *DOMAIN_WALLS)
    assert cfg.forward_kwargs.boundary_meshes == {n: f"boundaries.{n}" for n in names}
    assert OmegaConf.to_container(m.boundary_schemas) == {n: {} for n in names}
    assert cfg.forward_kwargs.global_data == "global_data"
    assert OmegaConf.to_container(m.global_schema) == {"U_inf_dir": {"rank": 1}}
    assert (
        m.hidden == 512 and m.n_layers == 12 and m.n_slices == 512
    )  # REF-80 reference (2026-10-04)
    ### Removed options never come back through a config.
    for gone in (
        "scale_mode",
        "query_tokens",
        "query_mass",
        "support_tokens",
        "kernel_read",
        "kernel_read_radius",
        "kernel_read_blocks",
        "kernel_read_theta",
        "mlp_ratio",
        "geo_checkpoint",
        "geo_kernel",
        "eps",
        "n_decoder_layers",
        "interior_geometry",
        "use_measure_weights",
    ):
        assert gone not in m, gone
    for gone in (
        "points",
        "normals",
        "measure_weights",
        "global_inputs",
        "query_points",
        "query_normals",
        "query_inputs",
    ):
        assert gone not in cfg.forward_kwargs, gone
    ### Every output name is a dataset target.
    targets = _targets(dataset)
    assert set(OmegaConf.to_container(m.output_schema)) == set(targets)


def test_isla_surface_composes_and_runs():
    """Surface task: no prediction points; the model reads the vehicle and the four
    tunnel boundaries and predicts at the vehicle's cells only; the output Mesh's points
    are the vehicle's cells in cell order, which is where SurfaceMeshToDomainMesh put the
    targets, and the vehicle measure ISLA reads is the corrected measure aligned with the
    target measure. The vehicle is reoriented into the fluid for ISLA only."""
    cfg = _compose("isla_surface", "drivaer_ml_surface", *SMALL)
    assert "prediction_points" not in cfg.forward_kwargs
    assert not cfg.model.get("interior", False)
    assert set(cfg.forward_kwargs) == {
        "boundary_meshes",
        "prediction_boundaries",
        "global_data",
    }
    assert OmegaConf.to_container(cfg.forward_kwargs.prediction_boundaries) == {
        "literal": ["vehicle"]
    }
    appended = OmegaConf.to_container(cfg.dataset_append_transforms, resolve=False)
    assert [str(t["_target_"]) for t in appended] == ["${dp:FlipBoundaryOrientation}"]
    assert [t["boundaries"] for t in appended] == [["vehicle"]]
    assert OmegaConf.to_container(cfg.model.output_schema) == {
        "pressure": {"rank": 0},
        "wss": {"rank": 1},
    }
    domain = _surface_domain()
    batch, out, pred = _run(cfg, domain, _targets("drivaer_ml_surface"))
    assert batch["forward_kwargs"]["prediction_boundaries"] == ["vehicle"]
    assert set(batch["forward_kwargs"]["boundary_meshes"]) == {"vehicle", *DOMAIN_WALLS}
    torch.testing.assert_close(out.points, domain.interior.points)
    torch.testing.assert_close(
        cell_measures(batch["forward_kwargs"]["boundary_meshes"]["vehicle"]),
        batch["target_measure"],
    )
    assert pred["pressure"].shape == (12,) and pred["wss"].shape == (12, 3)
    ### The pass-through global_data is what the model selected its direction from.
    assert torch.equal(out.global_data["U_inf_dir"], domain.global_data["U_inf_dir"])


def _surface_sample(
    cell_data: dict, walls: dict[str, Mesh], n_cells: int = 12
) -> DomainMesh:
    """A surface dataset's output: a 1-in-3 subsample of a random surface carrying ``cell_data``
    beside the domain's other boundaries ``walls``, through SurfaceMeshToDomainMesh."""
    g = torch.Generator().manual_seed(0)
    mesh = Mesh(
        points=torch.randn(n_cells * 3, 3, generator=g),
        cells=torch.arange(n_cells * 3).reshape(n_cells, 3),
        cell_data={
            k: torch.randn(n_cells, *shape, generator=g)
            for k, shape in cell_data.items()
        },
    )
    scale_measures(mesh, 3.0)
    mesh.cell_data["normals"] = mesh.cell_normals
    gd = {"U_inf_dir": torch.tensor([1.0, 0.0, 0.0]), "L_ref": torch.tensor(5.0)}
    sample = DomainMesh(interior=mesh, boundaries=walls, global_data=gd)
    return SurfaceMeshToDomainMesh(cell_data_targets=list(cell_data)).apply_to_domain(
        sample
    )


def test_isla_surface_shift_suv_reads_the_box_and_predicts_at_the_vehicle():
    """SHIFT-SUV: the model reads the vehicle and the five pieces of the CFD box its datasets
    build (AddBoxBoundaries, normals into the fluid), each a condition by its name, and predicts
    at the vehicle's cells only. The stored vehicle points into the fluid already, so nothing
    is reversed."""
    from domain_transforms import AddBoxBoundaries

    box = ["inlet", "outlet", "floor", "side_walls", "roof"]
    cfg = _compose("isla_surface_shift_suv", "shift_suv_surface", *SMALL)
    assert cfg.forward_kwargs.boundary_meshes == {
        n: f"boundaries.{n}" for n in ("vehicle", *box)
    }
    assert OmegaConf.to_container(cfg.model.boundary_schemas) == {
        n: {} for n in ("vehicle", *box)
    }
    assert OmegaConf.to_container(cfg.forward_kwargs.prediction_boundaries) == {
        "literal": ["vehicle"]
    }
    assert "dataset_append_transforms" not in cfg
    ds = OmegaConf.load(_RECIPE_ROOT / "datasets" / "shift_suv_surface.yaml")
    built = OmegaConf.to_container(ds.pipeline.transforms, resolve=False)[0]
    assert sorted(built["faces"]) == sorted(box)
    walls = AddBoxBoundaries(
        bounds=[[-8.0, 16.0], [-4.0, 4.0], [-1.0, 3.0]],
        faces=built["faces"],
        n_per_side=(3, 3),
    )(Mesh(points=torch.zeros(1, 3))).boundaries
    domain = _surface_sample(
        {"pressure": (), "wss": (3,)},
        {n: ComposeQuadratureMeasure()(walls[n]) for n in walls.keys()},
    )
    batch, out, pred = _run(cfg, domain, _targets("shift_suv_surface"))
    assert set(batch["forward_kwargs"]["boundary_meshes"]) == {"vehicle", *box}
    torch.testing.assert_close(out.points, domain.interior.points)
    assert pred["pressure"].shape == (12,) and pred["wss"].shape == (12, 3)


def test_isla_surface_highlift_reads_the_far_field_and_predicts_at_the_wing():
    """HiLift: the model reads the wing, the inlet and outlet of the far field its dataset builds
    (AddHemisphereBoundaries) and the symmetry plane its reader builds (AddSymmetryPlane) by name,
    predicts at the wing's cells only, and reverses the wing's winding (into the body as stored) for
    ISLA only."""
    from domain_transforms import AddHemisphereBoundaries, AddSymmetryPlane

    names = ("vehicle", "inlet", "outlet", "symmetry")
    cfg = _compose("isla_surface_highlift", "highlift_surface", *SMALL)
    assert cfg.forward_kwargs.boundary_meshes == {n: f"boundaries.{n}" for n in names}
    assert OmegaConf.to_container(cfg.model.boundary_schemas) == {n: {} for n in names}
    assert OmegaConf.to_container(cfg.forward_kwargs.prediction_boundaries) == {
        "literal": ["vehicle"]
    }
    appended = OmegaConf.to_container(cfg.dataset_append_transforms, resolve=False)
    assert [t["boundaries"] for t in appended] == [["vehicle"]]
    ds = OmegaConf.load(_RECIPE_ROOT / "datasets" / "highlift_surface.yaml")
    built = OmegaConf.to_container(ds.pipeline.transforms, resolve=False)[0]
    assert str(built["_target_"]) == "${dp:AddHemisphereBoundaries}"
    far = AddHemisphereBoundaries(radius=4.0, n_theta=6, n_phi=3)(
        Mesh(
            points=torch.zeros(1, 3),
            global_data={"U_inf": torch.tensor([0.98, 0.0, 0.17])},
        )
    ).boundaries
    far = {
        **{n: far[n] for n in far.keys()},
        "symmetry": AddSymmetryPlane(
            radius=4.0, cap_root="unused", cap_md5=[], n_theta=6, n_phi=3
        ).plane(
            np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.5, 0.5],
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [-0.5, 0.5],
                ]
            )
        ),
    }
    targets = _targets("highlift_surface")
    domain = _surface_sample(
        {k: () if v == "scalar" else (3,) for k, v in targets.items()},
        {n: ComposeQuadratureMeasure()(far[n]) for n in far.keys()},
    )
    batch, out, pred = _run(cfg, domain, targets)
    assert set(batch["forward_kwargs"]["boundary_meshes"]) == set(names)
    torch.testing.assert_close(out.points, domain.interior.points)
    assert pred["velocity"].shape == (12, 3) and pred["density"].shape == (12,)


def test_isla_volume_composes_and_runs():
    """Interior task: the interior Mesh is the prediction-point cloud, whose points
    alone the model reads (not the stored sdf / sdf_normals)."""
    cfg = _compose("isla_volume", "drivaer_ml_volume_reference", *SMALL)
    assert cfg.model.interior is True
    assert cfg.forward_kwargs.prediction_points == "interior"
    assert OmegaConf.to_container(cfg.model.output_schema) == {
        "velocity": {"rank": 1},
        "pressure": {"rank": 0},
        "nut": {"rank": 0},
    }
    ### The whole boundary of the domain: the full-resolution vehicle and the four domain
    ### meshes the files store, each a condition by its file name without fields, the
    ### surface of the prediction points' geometry and the source of the tokens the
    ### model draws.
    assert set(cfg.forward_kwargs) == {
        "boundary_meshes",
        "prediction_points",
        "global_data",
    }
    names = ("vehicle", *DOMAIN_WALLS)
    assert OmegaConf.to_container(cfg.forward_kwargs.boundary_meshes) == {
        name: f"boundaries.{name}" for name in names
    }
    assert OmegaConf.to_container(cfg.model.boundary_schemas) == {
        name: {} for name in names
    }
    domain = _volume_domain()
    batch, out, pred = _run(cfg, domain, _targets("drivaer_ml_volume_reference"))
    assert set(batch["forward_kwargs"]["boundary_meshes"]) == set(names)
    assert batch["forward_kwargs"]["prediction_points"] is domain.interior
    torch.testing.assert_close(out.points, domain.interior.points)
    assert pred["velocity"].shape == (64, 3) and pred["nut"].shape == (64,)


@pytest.mark.parametrize(
    "model",
    [
        "isla_volume",
        "isla_volume_aerozoo",
        "isla_volume_airfrans",
        "isla_tensile2d",
        "isla_lug",
    ],
)
def test_the_interior_configs_build_their_model(model):
    """One input contract: an interior ISLA reads its prediction points' positions
    alone and computes their wall distance and direction itself from the boundary
    meshes (the dataset's stored sdf and sdf_normals are the baselines' inputs);
    every interior config builds its model."""
    from datasets import load_dataset_config
    from output_normalize import model_dataset_constants

    paired = {
        "isla_volume_aerozoo": "aerozoo_uvolume",
        "isla_volume_airfrans": "airfrans_volume_2d",
    }
    dataset = paired.get(model, "drivaer_ml_volume_reference")
    cfg = _compose(model, dataset, *SMALL)
    ds = load_dataset_config(_RECIPE_ROOT / "datasets" / f"{dataset}.yaml")
    built = hydra.utils.instantiate(cfg.model, **model_dataset_constants(cfg.model, ds))
    assert built.interior is True
    assert built.query_independent is True  # ISLA-DTT


def test_isla_volume_composes_with_the_corrected_measure_dataset():
    """The volume pipeline materializes the boundary measure ISLA reads from cell_data."""
    ds = OmegaConf.load(_RECIPE_ROOT / "datasets" / "drivaer_ml_volume_reference.yaml")
    raw = OmegaConf.to_container(
        ds.pipeline.transforms, resolve=False
    )  # ${dp:...} nodes kept as strings
    targets = [t["_target_"] for t in raw]
    assert "ComposeQuadratureMeasure" in targets[-1], targets
    assert ds.pipeline.reader.boundary_subsample == "cells"


@pytest.mark.parametrize("suffix", ["", "_1case", "_pose", "_posed"])
def test_surface_datasets_read_the_tunnel_beside_the_vehicle(suffix):
    """Every DrivAerML Surface dataset file reads the four tunnel boundaries beside the
    subsampled vehicle, places their ground from the whole vehicle in the reader (before
    the subsample and every pose), and ends with SurfaceMeshToDomainMesh."""
    ds = OmegaConf.load(_RECIPE_ROOT / "datasets" / f"drivaer_ml_surface{suffix}.yaml")
    reader = OmegaConf.to_container(ds.pipeline.reader, resolve=False)
    assert reader["subsample_n_cells"] == "${sampling_resolution}"
    assert sorted(reader["domain_boundaries"]) == sorted(DOMAIN_WALLS)
    assert reader["whole_surface_transforms"] == [
        {
            "_target_": "${dp:PlaceGroundUnderBody}",
            "ground": "no_slip",
            "body": "vehicle",
            "boundaries": ["inlet", "no_slip", "outlet", "slip"],
        }
    ]
    raw = OmegaConf.to_container(ds.pipeline.transforms, resolve=False)
    assert str(raw[-1]["_target_"]) == "${dp:SurfaceMeshToDomainMesh}"
    assert not any("PlaceGroundUnderBody" in str(t["_target_"]) for t in raw)


@pytest.mark.parametrize("suffix", ["", "_area", "_posed"])
def test_reference_volume_datasets_read_the_vehicle_whole_into_the_fluid(suffix):
    """ISLA draws its boundary tokens inside the model, so every reference volume
    dataset file reads the boundaries whole (no boundary cell cap, no subsampling
    transform), keeps the four domain meshes of the files beside the vehicle, reverses
    the vehicle's stored winding so the normals point into the fluid (the domain
    meshes' point into it as stored), and drops only the STL, once the stored SDF is
    computed from it."""
    ds = OmegaConf.load(
        _RECIPE_ROOT / "datasets" / f"drivaer_ml_volume_reference{suffix}.yaml"
    )
    assert ds.pipeline.reader.get("subsample_n_cells") is None
    assert ds.pipeline.reader.drop_in_file_boundaries is False
    raw = OmegaConf.to_container(ds.pipeline.transforms, resolve=False)
    targets = [str(t["_target_"]) for t in raw]
    assert not any("Subsample" in t for t in targets), targets
    flips = [t for t in raw if "FlipBoundaryOrientation" in str(t["_target_"])]
    assert [t["boundaries"] for t in flips] == [["vehicle"]]
    dropped = [t.get("names", []) for t in raw if "DropBoundary" in str(t["_target_"])]
    assert dropped == [["stl_geometry"]]
    sdf = targets.index("${dp:ComputeSDFFromBoundary}")
    assert sdf < targets.index("${dp:DropBoundary}")


@pytest.mark.parametrize("suffix", ["", "_area", "_posed"])
def test_reference_volume_datasets_place_the_ground_under_the_vehicle_first(suffix):
    """The four domain meshes are stored 2.4 mm low on 297 of the 484 DrivAerML files, so
    every reference volume dataset moves them onto the CFD mesh's domain from the vehicle
    read whole, as its first transform: in the raw metres, with z vertical, before the pose,
    the centering and the nondimensionalization."""
    ds = OmegaConf.load(
        _RECIPE_ROOT / "datasets" / f"drivaer_ml_volume_reference{suffix}.yaml"
    )
    raw = OmegaConf.to_container(ds.pipeline.transforms, resolve=False)
    assert raw[0] == {
        "_target_": "${dp:PlaceGroundUnderBody}",
        "ground": "no_slip",
        "body": "vehicle",
        "boundaries": list(DOMAIN_WALLS),
    }
    assert sum("PlaceGroundUnderBody" in str(t["_target_"]) for t in raw) == 1
    assert ds.pipeline.reader.get("subsample_n_cells") is None


@pytest.mark.parametrize(
    "model",
    sorted(p.stem for p in (_RECIPE_ROOT / "conf" / "model").glob("isla_*.yaml")),
)
def test_interior_isla_configs_use_the_class_default_boundary_budget(model):
    """Every interior ISLA config draws the class default of boundary tokens (no config
    overrides it); the surface configs, which sample nothing, do not set it."""
    import inspect

    from physicsnemo.experimental.models.isla import ISLA

    cfg = _compose(model, "drivaer_ml_volume_reference")
    default = inspect.signature(ISLA.__init__).parameters["boundary_tokens"].default
    if cfg.model.get("interior", False):
        assert cfg.model.boundary_tokens == default == 200_000
    else:
        assert "boundary_tokens" not in cfg.model


@pytest.mark.parametrize(
    "model",
    sorted(p.stem for p in (_RECIPE_ROOT / "conf" / "model").glob("isla*.yaml")),
)
def test_interior_isla_configs_run_isla_dtt_compiled_per_block(model):
    """Every interior ISLA config (found by its `interior: true`, so a new one is covered)
    states `model.query_independent: true`, ISLA-DTT, the interior task's reference form
    and the class default, and compiles it per block (`compile: false`,
    `model.compile_blocks: true`), since its boundary read breaks the whole-model
    fullgraph trace; it keeps its `compile_options`, interacting ISLA's whole-model
    compile. No surface config sets `query_independent: true`."""
    path = _RECIPE_ROOT / "conf" / "model" / f"{model}.yaml"
    raw = OmegaConf.load(path)
    cfg = _compose(model, "drivaer_ml_volume_reference")
    if raw.model.get("interior", False):
        assert raw.model.get("query_independent") is True  # stated in the file
        assert cfg.model.query_independent is True
        assert cfg.model.compile_blocks is True
        assert cfg.compile is False
        assert raw.compile is False and raw.model.compile_blocks is True
        assert raw.compile_options.fullgraph is True
    else:
        assert raw.model.get("query_independent") is not True
        assert cfg.model.get("query_independent") is not True


def test_corrected_measure_sees_the_surface_area():
    """End to end on a toy surface: raw areas of a 1-in-5 subsample sum to a fifth of the
    body; the corrected field ISLA reads sums to the body, so its measure-weighted length
    unit is the body's and not the subsample's."""
    g = torch.Generator().manual_seed(0)
    pts = torch.rand(3000, 3, generator=g)
    mesh = Mesh(points=pts, cells=torch.arange(3000).reshape(1000, 3))
    full = float(mesh.cell_areas.sum())
    sub = mesh.slice_cells(torch.arange(0, 1000, 5))
    scale_measures(sub, 5.0)
    sub = ComposeQuadratureMeasure()(sub)
    raw = float(sub.cell_areas.sum())
    corrected = float(cell_measures(sub).sum())
    assert abs(corrected - full) / full < 0.15
    assert raw / full < 0.3


@pytest.mark.parametrize("model, dataset", CONFIGS)
def test_isla_global_vector_is_produced_by_the_dataset_pipeline(model, dataset):
    """The ISLA configs declare the unit freestream direction as ``U_inf_dir``; the paired
    dataset must compute it (``ComputeFreestreamDirection`` -> ``U_inf_dir``)."""
    cfg = _compose(model, dataset)
    assert OmegaConf.to_container(cfg.model.global_schema) == {"U_inf_dir": {"rank": 1}}
    ds = OmegaConf.load(_RECIPE_ROOT / "datasets" / f"{dataset}.yaml")
    raw = OmegaConf.to_container(ds.pipeline.transforms, resolve=False)
    producers = [
        t
        for t in raw
        if "ComputeFreestreamDirection" in t["_target_"]
        and t.get("output_field") == "U_inf_dir"
    ]
    assert producers, (
        f"{dataset}.yaml never computes U_inf_dir; transforms: {[t['_target_'] for t in raw]}"
    )


@pytest.mark.parametrize("model, dataset", CONFIGS)
def test_isla_configs_pin_the_protocol_learning_rate(model, dataset):
    """The recipe's default rate (3e-3, GeoTransolver's) drives ISLA to a mean-field
    plateau within five epochs, so every ISLA config carries its own rate: 3.75e-4, the
    reference configuration's (REF-80, 2026-10-04; the width rule 1e-3 x 192 / 512)."""
    cfg = _compose(model, dataset)
    assert cfg.training.optimizer.lr == pytest.approx(3.75e-4)


@pytest.mark.parametrize(
    "model",
    sorted(p.stem for p in (_RECIPE_ROOT / "conf" / "model").glob("isla_*.yaml")),
)
def test_every_isla_model_config_pins_the_reference_rate(model):
    """Every ISLA model config, whatever its dataset, trains at the reference rate
    3.75e-4: with the normalized slice reads no dataset keeps a rate exception. The
    dataset key is train.yaml's default; it does not enter the rate."""
    cfg = _compose(model, "drivaer_ml_volume")
    assert cfg.training.optimizer.lr == pytest.approx(3.75e-4)


def test_cli_override_still_wins_over_the_pinned_rate():
    """Explicit learning-rate overrides take precedence over model defaults."""
    cfg = _compose("isla_surface", "drivaer_ml_surface", "training.optimizer.lr=5e-4")
    assert cfg.training.optimizer.lr == pytest.approx(5e-4)


def _airfrans_case(s: float = 1.0, c: float = 1.0) -> DomainMesh:
    """The synthetic AirfRANS case, with the far field the reader adds, under x -> s x, U_inf -> c U_inf:
    velocity times c, kinematic pressure c^2, nut s c, the boundaries' lengths s."""
    from test_airfrans_configs import make_airfrans_case, make_airfrans_freestream

    d = make_airfrans_case()
    free = make_airfrans_freestream()
    pd = d.interior.point_data.clone()
    for name, k in (("velocity", c), ("pressure", c * c), ("nut", s * c)):
        pd[name] = pd[name] * k
    foil = d.boundaries["airfoil"]
    cd = foil.cell_data.clone()
    cd["quadrature_measure"] = cd["quadrature_measure"] * s
    gd = d.global_data.clone()
    for name in ("U_inf", "U_inf_mag", "U_inf_freestream_mean"):
        gd[name] = gd[name] * c
    return DomainMesh(
        interior=Mesh(points=d.interior.points * s, point_data=pd),
        boundaries={
            "airfoil": Mesh(points=foil.points * s, cells=foil.cells, cell_data=cd),
            "freestream": Mesh(points=free.points * s, cells=free.cells),
        },
        global_data=gd,
    )


def _airfrans_chain(cfg) -> list:
    from datasets import (
        _apply_dataset_transform_appends,
        _apply_dataset_transform_drops,
        load_dataset_config,
    )

    ds = OmegaConf.merge(
        load_dataset_config(_RECIPE_ROOT / "datasets" / "airfrans_volume_2d.yaml"),
        {"sampling_resolution": 64},
    )
    ds = _apply_dataset_transform_appends(_apply_dataset_transform_drops(ds, cfg), cfg)
    return [hydra.utils.instantiate(t, _convert_="all") for t in ds.pipeline.transforms]


def test_isla_airfrans_scales_follow_the_dataset_s_nondimensionalization():
    """The guard that keeps the two from drifting apart: the dataset's own NonDimensionalizeByMetadata, run
    on a mesh whose global_data hold only ISLA's declared velocity source (and the dataset's freestream
    density and pressure), divides the velocity target by that source's magnitude and the pressure target by
    its square, so the two outputs ISLA declares dimensionless are dimensionless in ISLA's own U; nut, which
    the dataset leaves physical, has the units U L, and nu is the constant the config sets."""
    cfg = _compose("isla_volume_airfrans", "airfrans_volume_2d")
    gs = OmegaConf.to_container(cfg.model.global_schema)
    (source,) = [k for k, v in gs.items() if v.get("velocity_scale")]
    assert gs["nu"]["role"] == "viscosity" and not any(
        v["rank"] == 0 and "role" not in v for v in gs.values()
    )
    ds = OmegaConf.load(_RECIPE_ROOT / "datasets" / "airfrans_volume_2d.yaml")
    raw = OmegaConf.to_container(ds.pipeline.transforms, resolve=False)
    (nondim_cfg,) = [t for t in raw if "NonDimensionalizeByMetadata" in t["_target_"]]
    nondim = hydra.utils.instantiate(nondim_cfg, _convert_="all")
    outs = OmegaConf.to_container(cfg.model.output_schema)
    divided = {
        name
        for name, kind in nondim_cfg["fields"].items()
        if kind in ("velocity", "pressure")
    }
    assert divided == {"velocity", "pressure"} and all(
        "units" not in outs[n] for n in divided
    )
    assert outs["nut"]["units"] == {"U": 1, "L": 1}
    n = 5
    pd = {"velocity": torch.ones(n, 2), "pressure": torch.ones(n), "nut": torch.ones(n)}
    for speed in (1.0, 4.0):
        mesh = Mesh(
            points=torch.zeros(n, 2),
            point_data=dict(pd),
            global_data={
                source: torch.tensor([0.6, 0.8]) * speed,
                "rho_inf": torch.tensor(1.0),
                "p_inf": torch.tensor(0.0),
            },
        )
        out = nondim(mesh).point_data
        torch.testing.assert_close(out["velocity"], pd["velocity"] / speed)
        torch.testing.assert_close(out["pressure"], pd["pressure"] / (0.5 * speed**2))
        torch.testing.assert_close(out["nut"], pd["nut"])


def test_isla_airfrans_similarity_end_to_end():
    """One synthetic AirfRANS case through the dataset's chain and ISLA: x -> s x, U_inf -> c U_inf and
    nu -> s c nu leave the dataset's velocity and Cp targets and ISLA's velocity and pressure outputs
    unchanged, and scale nut (the target and the output, in physical units) by s c."""
    from datasets import load_dataset_config
    from output_normalize import (
        check_model_output_stats,
        model_dataset_constants,
        model_output_stats,
    )

    cfg = _compose("isla_volume_airfrans", "airfrans_volume_2d", *SMALL)
    targets = _targets("airfrans_volume_2d")
    spec = OmegaConf.to_container(cfg.forward_kwargs, resolve=True)
    chain = _airfrans_chain(cfg)
    (normalizer,) = [t for t in chain if type(t).__name__ == "NormalizeMeshFields"]
    torch.manual_seed(0)
    ds = load_dataset_config(_RECIPE_ROOT / "datasets" / "airfrans_volume_2d.yaml")
    model = hydra.utils.instantiate(
        cfg.model,
        _convert_="partial",
        **model_output_stats(cfg.model, normalizer),
        **model_dataset_constants(cfg.model, ds),
    )
    check_model_output_stats(model, normalizer)
    model.eval()
    s, c = 0.3, 2.5
    runs = []
    for k, (si, ci) in enumerate(((1.0, 1.0), (s, c))):
        d = _airfrans_case(si, ci)
        for t in chain:
            d = t.apply_to_domain(d)
        gd = d.global_data.clone()
        gd["nu"] = gd["nu"] * (si * ci)
        d = DomainMesh(interior=d.interior, boundaries=d.boundaries, global_data=gd)
        batch = build_collate_fn(cfg.input_type, spec, targets)([(d, {})])
        with torch.no_grad():
            out = model(**batch["forward_kwargs"])
        runs.append(
            (
                batch["targets"],
                normalize_output_to_tensordict(out, targets, cfg.output_type),
            )
        )
    (t0, y0), (t1, y1) = runs
    stats = normalizer.stats["nut"]
    for name in ("velocity", "pressure"):
        torch.testing.assert_close(t1[name], t0[name], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(y1[name], y0[name], atol=1e-5, rtol=1e-4)
    for td in (t0, t1, y0, y1):
        td["nut"] = td["nut"] * stats["std"] + stats["mean"]
    torch.testing.assert_close(t1["nut"], s * c * t0["nut"], atol=1e-9, rtol=1e-4)
    torch.testing.assert_close(y1["nut"], s * c * y0["nut"], atol=1e-9, rtol=1e-4)


@pytest.mark.parametrize(
    "dataset", ["airfrans_uvolume_2d", "airfrans_uvolume_2d_posed"]
)
def test_isla_airfrans_reads_the_airfoil_and_the_far_field(dataset, tmp_path):
    """AF-WHOLE (2026-10-23): through the real reader, chain and collate, isla_volume_airfrans takes the airfoil
    and the far field by name (no fields on either) and predicts at the interior points; its length unit is the
    whole boundary's, which the far field sets (the synthetic far field is a 16-gon in a circle of radius 20,
    whose RMS pairwise distance is 20 sqrt(2) for the exact circle, less a few percent for the polygon and the
    airfoil's share of the measure)."""
    from datasets import (
        _apply_dataset_transform_appends,
        _apply_dataset_transform_drops,
        build_dataset,
        load_dataset_config,
    )
    from output_normalize import model_dataset_constants, model_output_stats
    from test_airfrans_configs import write_airfrans_case

    from physicsnemo.datapipes.protocols import DatasetBase

    write_airfrans_case(tmp_path)
    cfg = _compose("isla_volume_airfrans", dataset, *SMALL)
    assert OmegaConf.to_container(cfg.forward_kwargs.boundary_meshes) == {
        "airfoil": "boundaries.airfoil",
        "freestream": "boundaries.freestream",
    }
    assert OmegaConf.to_container(cfg.model.boundary_schemas) == {
        "airfoil": {},
        "freestream": {},
    }
    ds_yaml = load_dataset_config(_RECIPE_ROOT / "datasets" / f"{dataset}.yaml")
    ds_cfg = OmegaConf.merge(
        ds_yaml,
        {
            "train_datadir": str(tmp_path),
            "manifest": str(tmp_path / "manifest.json"),
            "raw_datadir": str(tmp_path / "raw"),
            "sampling_resolution": 64,
        },
    )
    ds_cfg = _apply_dataset_transform_appends(
        _apply_dataset_transform_drops(ds_cfg, cfg), cfg
    )
    ds = build_dataset(ds_cfg, device=None, num_workers=1)
    try:
        sample = ds[0][0]
    finally:
        DatasetBase.close(ds)
    (normalizer,) = [
        hydra.utils.instantiate(t, _convert_="all")
        for t in ds_cfg.pipeline.transforms
        if "NormalizeMeshFields" in str(t["_target_"])
    ]
    torch.manual_seed(0)
    model = hydra.utils.instantiate(
        cfg.model,
        _convert_="partial",
        **model_output_stats(cfg.model, normalizer),
        **model_dataset_constants(cfg.model, ds_yaml),
    ).eval()
    targets = _targets(dataset)
    spec = OmegaConf.to_container(cfg.forward_kwargs, resolve=True)
    batch = build_collate_fn(cfg.input_type, spec, targets)([(sample, {})])
    meshes = batch["forward_kwargs"]["boundary_meshes"]
    assert set(meshes) == {"airfoil", "freestream"}
    measures = [
        cell_measures(meshes[n].to(torch.float64)) for n in ("airfoil", "freestream")
    ]
    gauge = float(
        model._gauge(
            [meshes[n].to(torch.float64) for n in ("airfoil", "freestream")], measures
        )
    )
    assert gauge == pytest.approx(20.0 * 2**0.5, rel=0.03)
    assert gauge > 50 * float(
        model._gauge([meshes["airfoil"].to(torch.float64)], measures[:1])
    )
    with torch.no_grad():
        out = model(**batch["forward_kwargs"])
    torch.testing.assert_close(out.points, sample.interior.points)
    pred = normalize_output_to_tensordict(out, targets, cfg.output_type)
    assert pred["velocity"].shape == (64, 2) and pred["pressure"].shape == (64,)
    assert all(torch.isfinite(pred[k]).all() for k in targets)


@pytest.mark.parametrize(
    "model, dataset",
    [
        ("isla_volume_aerozoo", f"aerozoo_uvolume{suffix}")
        for suffix in (
            "",
            "_nonbox",
            "_reynolds",
            "_blobs",
            "_bank_zero_to_cad",
            "_profile_crossflow",
            "_profile_dual_inlet",
        )
    ]
    + [
        ("isla_volume_airfrans", f"airfrans_volume_2d{suffix}")
        for suffix in ("", "_area", "_posed")
    ]
    + [
        ("isla_volume_airfrans", f"airfrans_uvolume_2d_{task}{posed}")
        for task in ("reynolds", "aoa")
        for posed in ("", "_posed")
    ]
    + [
        ("isla_volume_airfrans", "airfrans_uvolume_2d_s504"),
        ("isla_volume_airfrans", "airfrans_uvolume_2d_reynolds_lre"),
    ],
)
def test_isla_takes_its_split_s_constants_from_the_dataset_file(model, dataset):
    """ISLA's AeroZoo and AirfRANS configs leave reynolds and each output's scale to the dataset file, whose
    isla_constants are fitted on its split's training side; the model is built with them, a dataset file
    without them is refused, and the config's own null values refuse to build alone."""
    from datasets import load_dataset_config
    from output_normalize import model_dataset_constants

    cfg = _compose(model, dataset, *SMALL)
    ds = load_dataset_config(_RECIPE_ROOT / "datasets" / f"{dataset}.yaml")
    block = OmegaConf.to_container(ds.isla_constants)
    m = hydra.utils.instantiate(cfg.model, **model_dataset_constants(cfg.model, ds))
    assert m.reynolds == pytest.approx(block["reynolds"])
    assert m._output_scale == pytest.approx(block["output_scale"])
    for cond, scales in block.get("input_scale", {}).items():
        assert m._boundary_scale[cond] == pytest.approx(scales)
    with pytest.raises(ValueError, match="isla_constants"):
        model_dataset_constants(cfg.model, OmegaConf.create({}))
    with pytest.raises(Exception, match="positive finite number; got None|reynolds"):
        hydra.utils.instantiate(cfg.model)
