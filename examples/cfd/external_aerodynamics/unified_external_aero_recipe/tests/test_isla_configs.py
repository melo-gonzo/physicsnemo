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
``model=isla_volume`` (interior task: predictions at the
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
    with sdf / sdf_normals and the targets, the whole vehicle boundary with its measure
    materialized by ComposeQuadratureMeasure, and U_inf_dir in global_data."""
    domain = make_volume_domain_mesh(n_pts=64)
    vehicle = ComposeQuadratureMeasure()(domain.boundaries["vehicle"])
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
    ### One boundary input: the full-resolution vehicle, the surface of the prediction
    ### points' geometry and the source of the tokens the model draws.
    assert set(cfg.forward_kwargs) == {
        "boundary_meshes",
        "prediction_points",
        "global_data",
    }
    assert OmegaConf.to_container(cfg.forward_kwargs.boundary_meshes) == {
        "vehicle": "boundaries.vehicle"
    }
    domain = _volume_domain()
    batch, out, pred = _run(cfg, domain, _targets("drivaer_ml_volume_reference"))
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
    assert (
        hydra.utils.instantiate(
            cfg.model, **model_dataset_constants(cfg.model, ds)
        ).interior
        is True
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


@pytest.mark.parametrize("suffix", ["", "_area", "_posed"])
def test_reference_volume_datasets_read_the_vehicle_whole_into_the_fluid(suffix):
    """ISLA draws its boundary tokens inside the model, so every reference volume
    dataset file reads the vehicle whole (no boundary cell cap, no subsampling
    transform), reverses its stored winding so the normals point into the fluid, and
    drops the STL once the stored SDF is computed from it."""
    ds = OmegaConf.load(
        _RECIPE_ROOT / "datasets" / f"drivaer_ml_volume_reference{suffix}.yaml"
    )
    assert ds.pipeline.reader.get("subsample_n_cells") is None
    raw = OmegaConf.to_container(ds.pipeline.transforms, resolve=False)
    targets = [str(t["_target_"]) for t in raw]
    assert not any("Subsample" in t for t in targets), targets
    flips = [t for t in raw if "FlipBoundaryOrientation" in str(t["_target_"])]
    assert [t["boundaries"] for t in flips] == [["vehicle"]]
    dropped = [t.get("names", []) for t in raw if "DropBoundary" in str(t["_target_"])]
    assert ["stl_geometry"] in dropped
    sdf = targets.index("${dp:ComputeSDFFromBoundary}")
    assert sdf < targets.index("${dp:DropBoundary}")


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
    """The synthetic AirfRANS case under x -> s x, U_inf -> c U_inf: velocity times c, kinematic pressure c^2,
    nut s c, the airfoil's lengths s."""
    from test_airfrans_configs import make_airfrans_case

    d = make_airfrans_case()
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
            "airfoil": Mesh(points=foil.points * s, cells=foil.cells, cell_data=cd)
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
