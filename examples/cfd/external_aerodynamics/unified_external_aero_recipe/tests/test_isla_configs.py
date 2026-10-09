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

``model=isla_surface`` (surface task: predictions at the boundary cells) and
``model=isla_volume`` / ``model=isla_volume_sdf`` (interior task: predictions at the
dataset's interior points) all declare ``input_type: mesh`` / ``output_type: mesh`` and
map the DomainMesh's boundary and global data onto ISLA's keyword-only mesh forward.
Each config is composed with Hydra, instantiated at a reduced width, and driven through
collate -> resolve_forward_kwargs -> ISLA.forward -> normalize_output_to_tensordict ->
loss on a tiny synthetic DomainMesh shaped like the paired dataset's output."""

import sys
from pathlib import Path

import hydra
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
from output_normalize import normalize_output_to_tensordict  # noqa: E402

from physicsnemo.datapipes.transforms.mesh import MeshToDomainMesh  # noqa: E402
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
    "model.geo_kernel=eager",
]
CONFIGS = [
    ("isla_surface", "drivaer_ml_surface"),
    ("isla_volume", "drivaer_ml_volume_reference"),
    ("isla_volume_sdf", "drivaer_ml_volume_reference"),
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
    with its Horvitz-Thompson measure, then MeshToDomainMesh (interior = cell centroids
    carrying the targets; boundaries.vehicle = the surface, its cell_data keeping the
    measure and the `normals` field the tensor-input models read)."""
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
    return MeshToDomainMesh(cell_data_targets=["pressure", "wss"])(mesh)


def _volume_domain() -> DomainMesh:
    """What drivaer_ml_volume_reference.yaml hands the collate: an interior point cloud
    with sdf / sdf_normals and the targets, a vehicle boundary with the corrected measure
    materialized by ComposeQuadratureMeasure, and U_inf_dir in global_data."""
    domain = make_volume_domain_mesh(n_pts=64)
    vehicle = domain.boundaries["vehicle"]
    scale_measures(vehicle, 4.0)
    vehicle = ComposeQuadratureMeasure()(vehicle)
    gd = domain.global_data.clone()
    gd["U_inf_dir"] = torch.tensor([1.0, 0.0, 0.0])
    return DomainMesh(
        interior=domain.interior, boundaries={"vehicle": vehicle}, global_data=gd
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
    assert cfg.forward_kwargs.boundary_meshes == {"vehicle": "boundaries.vehicle"}
    assert cfg.forward_kwargs.global_data == "global_data"
    assert OmegaConf.to_container(m.global_schema) == {"U_inf_dir": {"rank": 1}}
    assert (
        m.hidden == 512 and m.n_layers == 12 and m.n_slices == 512 and m.mlp_ratio == 4
    )  # REF-80 reference (2026-10-04)
    assert m.geo_checkpoint is True and m.geo_kernel is None
    ### Removed options never come back through a config.
    for gone in (
        "scale_mode",
        "query_tokens",
        "query_mass",
        "query_independent",
        "support_tokens",
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
    """Surface task: no prediction points; the output Mesh's points are the boundary cells
    in cell order, which is where MeshToDomainMesh put the targets, and the boundary
    measure ISLA reads is the corrected measure aligned with the target measure."""
    cfg = _compose("isla_surface", "drivaer_ml_surface", *SMALL)
    assert "prediction_points" not in cfg.forward_kwargs
    assert not cfg.model.get("interior", False)
    assert OmegaConf.to_container(cfg.model.output_schema) == {
        "pressure": {"rank": 0},
        "wss": {"rank": 1},
    }
    domain = _surface_domain()
    batch, out, pred = _run(cfg, domain, _targets("drivaer_ml_surface"))
    torch.testing.assert_close(out.points, domain.interior.points)
    torch.testing.assert_close(
        cell_measures(batch["forward_kwargs"]["boundary_meshes"]["vehicle"]),
        batch["target_measure"],
    )
    assert pred["pressure"].shape == (12,) and pred["wss"].shape == (12, 3)
    ### The pass-through global_data is what the model selected its direction from.
    assert torch.equal(out.global_data["U_inf_dir"], domain.global_data["U_inf_dir"])


def test_isla_volume_composes_and_runs():
    """Interior task: the interior Mesh is the prediction-point cloud; no query inputs
    are declared, so the stored sdf / sdf_normals are not read."""
    cfg = _compose("isla_volume", "drivaer_ml_volume_reference", *SMALL)
    assert cfg.model.interior is True
    assert cfg.forward_kwargs.prediction_points == "interior"
    assert cfg.model.get("query_schema") is None
    assert OmegaConf.to_container(cfg.model.output_schema) == {
        "velocity": {"rank": 1},
        "pressure": {"rank": 0},
        "nut": {"rank": 0},
    }
    domain = _volume_domain()
    batch, out, pred = _run(cfg, domain, _targets("drivaer_ml_volume_reference"))
    assert batch["forward_kwargs"]["prediction_points"] is domain.interior
    torch.testing.assert_close(out.points, domain.interior.points)
    assert pred["velocity"].shape == (64, 3) and pred["nut"].shape == (64,)


def test_isla_volume_sdf_declares_the_stored_sdf_as_query_inputs():
    """The stored SDF and its gradient enter as declared rank-0 / rank-1 query inputs read
    from the prediction Mesh's point_data; everything else is isla_volume."""
    cfg = _compose("isla_volume_sdf", "drivaer_ml_volume_reference", *SMALL)
    base = _compose("isla_volume", "drivaer_ml_volume_reference", *SMALL)
    assert OmegaConf.to_container(cfg.model.query_schema) == {
        "sdf": {"rank": 0},
        "sdf_normals": {"rank": 1},
    }
    assert OmegaConf.to_container(cfg.forward_kwargs) == OmegaConf.to_container(
        base.forward_kwargs
    )
    domain = _volume_domain()
    _, out, _ = _run(cfg, domain, _targets("drivaer_ml_volume_reference"))
    torch.testing.assert_close(out.points, domain.interior.points)
    ### Without the declared leaves the model refuses the prediction Mesh.
    model = hydra.utils.instantiate(cfg.model)
    bare = Mesh(points=domain.interior.points)
    with pytest.raises(ValueError, match="missing the declared fields"):
        model(
            boundary_meshes={"vehicle": domain.boundaries["vehicle"]},
            prediction_points=bare,
            global_data=domain.global_data,
        )


def test_isla_volume_composes_with_the_corrected_measure_dataset():
    """The volume pipeline materializes the boundary measure ISLA reads from cell_data."""
    ds = OmegaConf.load(_RECIPE_ROOT / "datasets" / "drivaer_ml_volume_reference.yaml")
    raw = OmegaConf.to_container(
        ds.pipeline.transforms, resolve=False
    )  # ${dp:...} nodes kept as strings
    targets = [t["_target_"] for t in raw]
    assert "ComposeQuadratureMeasure" in targets[-1], targets
    assert ds.pipeline.reader.boundary_subsample == "cells"


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


def test_cli_override_still_wins_over_the_pinned_rate():
    """Explicit learning-rate overrides take precedence over model defaults."""
    cfg = _compose("isla_surface", "drivaer_ml_surface", "training.optimizer.lr=5e-4")
    assert cfg.training.optimizer.lr == pytest.approx(5e-4)
