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

"""AeroZoo v1.2, interior task: the dataset files datasets/aerozoo_uvolume*.yaml (the PDE data only, in SI units,
2026-10-19) run on a synthetic release-shaped case (four boundaries, one case without slip faces) by the real
reader and by their transform chain, and the GeoTransolver and Transolver configs composed and driven through
collate -> forward -> loss on the output. The label check: every label perturbed or deleted leaves every model
input bitwise equal. The ISLA config brings its own scales and runs on the files' physical data."""

import sys
from pathlib import Path

import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict

_RECIPE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RECIPE_ROOT / "src"))

from collate import build_collate_fn  # noqa: E402
from datasets import (  # noqa: E402
    _apply_dataset_transform_appends,
    _apply_dataset_transform_drops,
    build_dataset,
    load_dataset_config,
)
from forward_kwargs import extract_target_measure  # noqa: E402
from loss import LossCalculator  # noqa: E402
from metrics import MetricCalculator  # noqa: E402
from output_normalize import (  # noqa: E402
    check_model_output_stats,
    model_dataset_constants,
    model_output_stats,
    normalize_output_to_tensordict,
)
from utils import field_dim  # noqa: E402

from physicsnemo.datapipes.protocols import DatasetBase  # noqa: E402
from physicsnemo.mesh import DomainMesh, Mesh  # noqa: E402
from physicsnemo.mesh.calculus.measure import EFFECTIVE_MEASURE_KEY  # noqa: E402

DATASET = "aerozoo_uvolume"
TARGETS = {
    "velocity": "vector",
    "pressure": "scalar",
    "nut": "scalar",
    "tke": "scalar",
    "omega": "scalar",
}
### Every dataset file and the split part aerozoo_pstats.py filled its statistics from.
DATASET_FILES = {
    f"aerozoo_uvolume{suffix}": split
    for suffix, split in (
        ("", "full"),
        ("_nonbox", "family_non_box"),
        ("_reynolds", "reynolds"),
        ("_blobs", "bank_blobs"),
        ("_bank_zero_to_cad", "bank_zero_to_cad"),
        ("_profile_crossflow", "profile_crossflow"),
        ("_profile_dual_inlet", "profile_dual_inlet"),
    )
}
BC_FIELDS = {
    "bc_inlet": 0,
    "bc_outlet": 0,
    "bc_no_slip": 0,
    "bc_slip": 0,
    "bc_velocity": 1,
    "bc_velocity_reverse": 1,
    "bc_pressure": 0,
    "bc_tke": 0,
    "bc_omega": 0,
    "bc_intermittency": 0,
    "bc_retheta": 0,
    "bc_tke_reverse": 0,
    "bc_omega_reverse": 0,
    "bc_intermittency_reverse": 0,
    "bc_retheta_reverse": 0,
}

### The case: U_inf = 10 m/s along x, rho 1.2, L_ref 2 m, nu 1.5e-5 (Re 1.33e6); the inlet prescribes k = 0.5,
### omega = 50, intermittency 1, Re_theta_t 300; the outlet p = 0 and reverse-flow values; the body is a
### closed box (the SDF sign is defined without the winding-number pass).
U_INF, RHO, L_REF, NU = 10.0, 1.2, 2.0, 1.5e-5
Q_INF = 0.5 * RHO * U_INF**2
INLET_TURBULENCE = {
    "turbulent_kinetic_energy": 0.5,
    "specific_dissipation_rate": 50.0,
    "intermittency": 1.0,
    "momentum_thickness_reynolds_number": 300.0,
}
N_INTERIOR, SAMPLING = 300, 64

### Every global_data group the AeroZoo release labels (the 2026-10-15 ruling: a model reads the boundary meshes
### with their condition types, the prescribed values and nu; rho only to convert pressure and shear).
LABEL_GROUPS = ("reference", "identity", "diagnostics", "solution_definition")


def _box(lo, hi) -> Mesh:
    """A closed axis-aligned box, 12 outward-wound triangles."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = torch.tensor(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )
    f = torch.tensor(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ]
    )
    return Mesh(points=v, cells=f)


def _plane(x: float | None = None, z: float | None = None, n: int = 4) -> Mesh:
    """``n`` triangles tiling a strip of the plane x = const (an inlet / outlet) or z = const (a floor)."""
    g = torch.Generator().manual_seed(7)
    a = torch.rand(3 * n, generator=g) * 20 - 10
    b = torch.rand(3 * n, generator=g) * 20 - 10
    if x is not None:
        pts = torch.stack([torch.full_like(a, x), a, b], -1)
    else:
        pts = torch.stack([a, b, torch.full_like(a, z)], -1)
    return Mesh(points=pts, cells=torch.arange(3 * n).reshape(n, 3))


def _solution(n: int, g: torch.Generator) -> dict:
    return {
        "velocity": torch.randn(n, 3, generator=g) * 3
        + torch.tensor([U_INF, 0.0, 0.0]),
        "gauge_pressure": torch.randn(n, generator=g) * 20,
        "turbulent_kinetic_energy": torch.rand(n, generator=g),
        "specific_dissipation_rate": torch.rand(n, generator=g) * 100 + 1,
        "turbulent_kinematic_viscosity": torch.rand(n, generator=g) * 1e-3,
        "intermittency": torch.ones(n),
        "momentum_thickness_reynolds_number": torch.full((n,), 300.0),
    }


def make_aerozoo_case(with_slip: bool, seed: int = 0) -> DomainMesh:
    """A release-shaped AeroZoo case: dimensional SI, the four boundaries and their cell_data trees."""
    g = torch.Generator().manual_seed(seed)
    interior = Mesh(
        points=torch.rand(N_INTERIOR, 3, generator=g) * torch.tensor([20.0, 10.0, 6.0])
        - torch.tensor([10.0, 5.0, 0.0]),
        point_data={
            "_target_quadrature_measure": torch.rand(N_INTERIOR, generator=g) * 1e-3,
            "solution": _solution(N_INTERIOR, g),
        },
    )
    inlet = _plane(x=-10.0)
    inlet.cell_data["solution"] = _solution(inlet.n_cells, g)
    inlet.cell_data["prescribed", "dirichlet", "velocity"] = (
        torch.tensor([U_INF, 0.0, 0.0]).expand(inlet.n_cells, 3).clone()
    )
    for name, value in INLET_TURBULENCE.items():
        inlet.cell_data["prescribed", "dirichlet", name] = torch.full(
            (inlet.n_cells,), value
        )
    outlet = _plane(x=10.0)
    outlet.cell_data["solution"] = _solution(outlet.n_cells, g)
    outlet.cell_data["prescribed", "dirichlet", "gauge_pressure"] = torch.zeros(
        outlet.n_cells
    )
    outlet.cell_data["prescribed", "reverse_flow", "velocity"] = (
        torch.tensor([U_INF, 0.0, 0.0]).expand(outlet.n_cells, 3).clone()
    )
    for name, value in INLET_TURBULENCE.items():
        outlet.cell_data["prescribed", "reverse_flow", name] = torch.full(
            (outlet.n_cells,), value
        )
    no_slip = _box((-1.0, -0.5, 0.2), (3.0, 0.5, 1.2))
    no_slip.cell_data["solution"] = _solution(no_slip.n_cells, g)
    no_slip.cell_data["prescribed", "dirichlet", "velocity"] = torch.zeros(
        no_slip.n_cells, 3
    )
    no_slip.cell_data["derived", "wall_shear_stress"] = torch.randn(
        no_slip.n_cells, 3, generator=g
    )
    no_slip.cell_data["derived", "wall_y_plus"] = torch.rand(
        no_slip.n_cells, generator=g
    )
    no_slip.cell_data["annotation", "body_id"] = torch.zeros(
        no_slip.n_cells, dtype=torch.int64
    )
    slip = (
        _plane(z=0.0)
        if with_slip
        else Mesh(points=torch.zeros(0, 3), cells=torch.zeros(0, 3, dtype=torch.long))
    )
    slip.cell_data["solution"] = _solution(slip.n_cells, g)
    slip.cell_data["annotation", "body_id"] = torch.full(
        (slip.n_cells,), -1, dtype=torch.int64
    )
    return DomainMesh(
        interior=interior,
        boundaries={"inlet": inlet, "outlet": outlet, "no_slip": no_slip, "slip": slip},
        global_data={
            "physics": {
                "density": torch.tensor(RHO),
                "kinematic_viscosity": torch.tensor(NU),
                "body_acceleration": torch.tensor([0.0, 0.0, -9.81]),
                "closure_model_id": torch.tensor(3),
            },
            "reference": {
                "velocity": torch.tensor([U_INF, 0.0, 0.0]),
                "speed": torch.tensor(U_INF),
                "length": torch.tensor(L_REF),
                "dynamic_pressure": torch.tensor(Q_INF),
                "force_projection_direction": torch.tensor([1.0, 0.0, 0.0]),
                "secondary_inlet_velocities": torch.zeros(0, 3),
            },
            "diagnostics": {"continuity_residual": torch.tensor(1e-6)},
        },
    )


def _compose(model, *extra, dataset: str = DATASET):
    out_dim = sum(field_dim(t) for t in TARGETS.values())
    with initialize_config_dir(
        config_dir=str(_RECIPE_ROOT / "conf"), version_base=None
    ):
        return compose(
            config_name="train",
            overrides=[
                f"model={model}",
                f"dataset={dataset}",
                f"+out_dim={out_dim}",
                *extra,
            ],
        )


#: The fields each condition prescribes, as ISLA declares them (rank by name).
ISLA_CONDITIONS = {
    "inlet": {
        "bc_velocity": 1,
        "bc_tke": 0,
        "bc_omega": 0,
        "bc_intermittency": 0,
        "bc_retheta": 0,
    },
    "outlet": {
        "bc_pressure": 0,
        "bc_velocity_reverse": 1,
        "bc_tke_reverse": 0,
        "bc_omega_reverse": 0,
        "bc_intermittency_reverse": 0,
        "bc_retheta_reverse": 0,
    },
    "no_slip": {},
    "slip": {},
}


def _chain(dataset: str, model_cfg=None) -> list:
    """The dataset file's transforms (with ``model_cfg``'s drops and appends), instantiated; the reader's draw is
    left out."""
    ds = OmegaConf.merge(
        load_dataset_config(_RECIPE_ROOT / "datasets" / f"{dataset}.yaml"),
        {
            "dataset_paths": {"aerozoo": "/unused", "aerozoo_manifest": "/unused"},
            "sampling_resolution": SAMPLING,
        },
    )
    if model_cfg is not None:
        ds = _apply_dataset_transform_appends(
            _apply_dataset_transform_drops(ds, model_cfg), model_cfg
        )
    return [hydra.utils.instantiate(t, _convert_="all") for t in ds.pipeline.transforms]


def _run(chain: list, domain: DomainMesh, seed: int = 0) -> DomainMesh:
    torch.manual_seed(seed)
    for t in chain:
        domain = t.apply_to_domain(domain)
    return domain


def _labelled_case(with_slip: bool) -> DomainMesh:
    """The synthetic case with every kind of AeroZoo label: reference, identity, diagnostics,
    solution_definition, the closure id and the body ids."""
    d = make_aerozoo_case(with_slip, seed=1)
    gd = d.global_data.clone()
    gd["identity", "family_id"] = torch.tensor(4)
    gd["identity", "geometry_scale"] = torch.tensor(1.7)
    gd["solution_definition", "residual_target"] = torch.tensor(1e-5)
    return DomainMesh(interior=d.interior, boundaries=d.boundaries, global_data=gd)


def _relabel(d: DomainMesh, delete: bool) -> DomainMesh:
    """Every label leaf perturbed (x 7.3 plus noise; integers + 3; body ids reversed + 1) or deleted."""
    g = torch.Generator().manual_seed(3)
    gd = d.global_data.clone()
    keys = [
        k if isinstance(k, tuple) else (k,)
        for k in gd.keys(include_nested=True, leaves_only=True)
    ]
    for k in keys:
        if k[0] in LABEL_GROUPS or k == ("physics", "closure_model_id"):
            if delete:
                del gd[k]
            elif gd[k].is_floating_point():
                gd[k] = gd[k] * 7.3 + torch.randn(gd[k].shape, generator=g)
            else:
                gd[k] = gd[k] + 3
    if delete:
        for group in LABEL_GROUPS:
            if group in gd.keys():
                del gd[group]
    boundaries = {}
    for name in d.boundary_names:
        b = d.boundaries[name]
        cd = b.cell_data.clone()
        if ("annotation", "body_id") in cd.keys(include_nested=True):
            if delete:
                del cd["annotation"]
            else:
                cd["annotation", "body_id"] = cd["annotation", "body_id"].flip(0) + 1
        boundaries[name] = b.with_data(cell_data=cd)
    return DomainMesh(interior=d.interior, boundaries=boundaries, global_data=gd)


def _is_label(key: str) -> bool:
    return (
        (key.split("/")[-1].split(".")[0] in LABEL_GROUPS and key.startswith("global/"))
        or key in ("global/physics.closure_model_id",)
        or "annotation" in key
    )


def _leaves(d: DomainMesh) -> dict[str, torch.Tensor]:
    """Every tensor a model could be handed, keyed by its place; the label leaves are skipped."""
    out = {"interior/points": d.interior.points}

    def add(prefix, td):
        for k in td.keys(include_nested=True, leaves_only=True):
            k = k if isinstance(k, tuple) else (k,)
            out[prefix + ".".join(k)] = td[k]

    add("interior/point_data/", d.interior.point_data)
    for name in d.boundary_names:
        b = d.boundaries[name]
        out[f"{name}/points"], out[f"{name}/cells"] = b.points, b.cells
        add(f"{name}/cell_data/", b.cell_data)
    add("global/", d.global_data)
    return {k: v for k, v in out.items() if not _is_label(k)}


def _flat(value, prefix="") -> dict[str, torch.Tensor]:
    """The tensors of a collated batch, keyed by their path."""
    if isinstance(value, torch.Tensor):
        return {prefix: value}
    if isinstance(value, dict):
        return {
            path: leaf
            for k, v in value.items()
            for path, leaf in _flat(v, f"{prefix}/{k}").items()
        }
    if hasattr(value, "keys"):  # TensorDict
        return {
            f"{prefix}/{k}": value[k]
            for k in value.keys(include_nested=True, leaves_only=True)
        }
    raise TypeError(type(value))


@pytest.mark.parametrize("dataset", sorted(DATASET_FILES))
def test_dataset_files_read_no_label_and_derive_no_case_scale(dataset):
    """Every dataset file is the same chain with its split's statistics: no transform reads a label or a
    reference scale, nothing is nondimensionalized or centred, and the constants are the split's (targets
    scale-only, mean 0); the reader reads the boundaries whole and SubsampleMesh draws first."""
    split = DATASET_FILES[dataset]
    path = _RECIPE_ROOT / "datasets" / f"{dataset}.yaml"
    assert path.read_text().startswith(
        f"# Statistics filled by aerozoo_pstats.py fill from pstats_{split}.train.json: split part {split}.train, "
    )
    ds = OmegaConf.load(path)
    assert set(ds.keys()) == {
        "train_datadir",
        "manifest",
        "pipeline",
        "targets",
        "isla_constants",
    }
    reader = OmegaConf.to_container(ds.pipeline.reader, resolve=False)
    assert "subsample_n_cells" not in reader and reader["boundary_subsample"] == "cells"
    assert reader["subsample_n_points"] == "${sampling_resolution}"
    chain = OmegaConf.to_container(ds.pipeline.transforms, resolve=False)
    assert chain[0] == {"_target_": "${dp:SubsampleMesh}", "n_cells": 50000}
    names = [t["_target_"].split(":")[-1].rstrip("}") for t in chain]
    assert names == [
        "SubsampleMesh",
        "HarmonizeBoundaryFields",
        "RenameMeshFields",
        "ComputeSDFFromBoundary",
        "StandardizeGlobalLeaf",
        "FlipBoundaryOrientation",
        "RenameMeshFields",
        "DropMeshFields",
        "NormalizeMeshFields",
        "ComposeQuadratureMeasure",
        "MergeBoundaries",
    ]
    text = OmegaConf.to_yaml(ds.pipeline)
    for label in (
        *LABEL_GROUPS,
        "closure_model_id",
        "body_id",
        "U_inf",
        "L_ref",
        "sdf_slip",
    ):
        assert label not in text, label
    assert (
        chain[4]["field"] == "physics.kinematic_viscosity" and chain[4]["log10"] is True
    )
    ### scale-only standardization of the targets and of the token values (an offset would put a small
    ### case below the float32 / bf16 step of the offset)
    for stats in [*chain[8]["fields"].values(), *chain[-1]["standardize"].values()]:
        assert stats["mean"] in (0, [0, 0, 0])
    assert OmegaConf.to_container(ds.targets) == TARGETS
    for t in chain:
        hydra.utils.instantiate(t, _convert_="all")


@pytest.mark.parametrize("with_slip", [True, False])
def test_pipeline_is_physical_and_merges_the_faces(with_slip):
    """SI values throughout: raw coordinates, the prescribed values in m/s and the case's own units on the
    four boundaries, nu_std from per-dataset constants; the merged boundary is every face in the listed
    order with its one-hot condition and standardized values, and no label."""
    case = _labelled_case(with_slip)
    chain = _chain(DATASET)
    d = _run(chain, case)
    torch.testing.assert_close(d.interior.points, case.interior.points)
    inlet = d.boundaries["inlet"]
    torch.testing.assert_close(
        inlet.cell_data["bc_velocity"],
        torch.tensor([U_INF, 0.0, 0.0]).expand(inlet.n_cells, 3),
    )
    torch.testing.assert_close(
        inlet.cell_data["bc_omega"], torch.full((inlet.n_cells,), 50.0)
    )
    std_leaf = next(t for t in chain if type(t).__name__ == "StandardizeGlobalLeaf")
    torch.testing.assert_close(
        d.global_data["nu_std"],
        (torch.log10(torch.tensor(NU)) - std_leaf.mean) / std_leaf.std,
    )
    merged = d.boundaries["merged"]
    names = ["inlet", "outlet", "no_slip", "slip"]
    assert merged.n_cells == sum(d.boundaries[n].n_cells for n in names)
    torch.testing.assert_close(
        merged.cell_centroids,
        torch.cat([d.boundaries[n].cell_centroids for n in names]),
    )
    merge = chain[-1]
    v_std = float(merge.normalize.stats["bc_velocity"]["std"])
    torch.testing.assert_close(
        merged.cell_data["bc_velocity"][: inlet.n_cells],
        inlet.cell_data["bc_velocity"] / (v_std + 1e-8),
    )
    assert torch.equal(
        merged.cell_data["bc_inlet"],
        torch.cat([d.boundaries[n].cell_data["bc_inlet"] for n in names]),
    )
    assert (
        "annotation" not in merged.cell_data.keys()
        and "solution" not in merged.cell_data.keys()
    )


def test_transforms_move_every_tensor_they_hold():
    """MeshDataset moves each transform with ``t.to(device)``: every tensor a recipe transform here holds, its nested
    standardization included, follows (on GPU the merged boundary's standardization stayed on the CPU, LHR
    job 287322). Device-agnostic: the meta device."""

    def tensors(obj, seen=()):
        if isinstance(obj, torch.Tensor):
            yield obj
        elif isinstance(obj, dict):
            for v in obj.values():
                yield from tensors(v, seen)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                yield from tensors(v, seen)
        elif hasattr(obj, "__dict__") and id(obj) not in seen:
            for v in vars(obj).values():
                yield from tensors(v, (*seen, id(obj)))

    chain = _chain(DATASET)
    merge = next(t for t in chain if type(t).__name__ == "MergeBoundaries")
    assert merge.normalize is not None and any(True for _ in tensors(merge.normalize))
    for t in chain:
        if type(t).__name__ in ("MergeBoundaries", "StandardizeGlobalLeaf"):
            assert t.to("meta") is t
            held = list(tensors(t))
            assert all(x.device.type == "meta" for x in held), type(t).__name__


def test_geotransolver_option_b_and_transolver_configs_run():
    """GeoTransolver reads every boundary face as a token [xyz, one-hot (4), the prescribed values (15)] with
    the ball query off, the query point with its no_slip SDF and normal, and nu_std; Transolver the query
    point with the no_slip SDF and normal, and nu_std broadcast. Both set the per-case relative loss."""
    targets = TARGETS
    d = _run(_chain(DATASET), _labelled_case(False))
    n_faces = d.boundaries["merged"].n_cells
    for model, shapes in (
        (
            "geotransolver_volume_aerozoo",
            {
                "geometry": (1, n_faces, 22),
                "local_embedding": (1, N_INTERIOR, 7),
                "global_embedding": (1, 1, 1),
            },
        ),
        (
            "transolver_volume_aerozoo",
            {"embedding": (1, N_INTERIOR, 7), "fx": (1, N_INTERIOR, 1)},
        ),
    ):
        cfg = _compose(model, dataset=DATASET)
        assert cfg.training.loss_type == "case_relative_l2"
        ### velocity weighs 3: one term per field, the per-component losses' channel balance
        assert OmegaConf.to_container(cfg.training.field_weights) == {"velocity": 3}
        assert cfg.input_type == "tensors" and cfg.model.out_dim == 7
        if model.startswith("geo"):
            assert cfg.model.include_local_features is False
            assert (
                cfg.model.geometry_dim,
                cfg.model.functional_dim,
                cfg.model.global_dim,
            ) == (22, 7, 1)
            assert cfg.training.optimizer.lr == pytest.approx(1.0e-3)
        spec = OmegaConf.to_container(cfg.forward_kwargs, resolve=True)
        batch = build_collate_fn("tensors", spec, targets)([(d, {})])
        fk = batch["forward_kwargs"]
        assert {k: tuple(v.shape) for k, v in fk.items()} == shapes
        small = OmegaConf.merge(
            cfg.model, {"n_layers": 2, "n_hidden": 32, "slice_num": 16, "n_head": 2}
        )
        net = hydra.utils.instantiate(small, _convert_="partial")
        with torch.no_grad():
            out = net(**fk)
        pred = normalize_output_to_tensordict(out, targets, "tensors")
        total, losses = LossCalculator(
            targets,
            loss_type=cfg.training.loss_type,
            field_weights=OmegaConf.to_container(cfg.training.field_weights),
        )(pred, batch["targets"])
        assert torch.isfinite(total)
        unweighted = LossCalculator(targets, loss_type=cfg.training.loss_type)(
            pred, batch["targets"]
        )[1]
        torch.testing.assert_close(
            losses["loss/velocity"], 3 * unweighted["loss/velocity"]
        )


@pytest.mark.parametrize("with_slip", [True, False])
def test_every_model_input_is_blind_to_every_label(with_slip):
    """On the synthetic case: with every label perturbed, and with every label
    deleted, the pipeline runs and every non-label leaf it hands over (the baselines' view and ISLA's, which
    drops the draw, the SDF, the harmonized fields and the merge) is bitwise equal, and so is every tensor of
    the GeoTransolver and Transolver batches."""
    case = _labelled_case(with_slip)
    isla_cfg = _compose("isla_volume_aerozoo", dataset=DATASET)
    views = {
        "baselines": _chain(DATASET),
        "isla": _chain(DATASET, isla_cfg),
    }
    assert not any(type(t).__name__ == "MergeBoundaries" for t in views["isla"])
    specs = {
        m: OmegaConf.to_container(
            _compose(m, dataset=DATASET).forward_kwargs, resolve=True
        )
        for m in ("geotransolver_volume_aerozoo", "transolver_volume_aerozoo")
    }
    for delete in (False, True):
        other = _relabel(case, delete)
        for name, chain in views.items():
            a, b = _leaves(_run(chain, case)), _leaves(_run(chain, other))
            assert a.keys() == b.keys(), (name, delete, a.keys() ^ b.keys())
            differ = [
                k for k in a if a[k].shape != b[k].shape or not torch.equal(a[k], b[k])
            ]
            assert not differ, (name, delete, differ)
        for model, spec in specs.items():
            a, b = (
                _flat(
                    build_collate_fn("tensors", spec, TARGETS)(
                        [(_run(views["baselines"], x), {})]
                    )
                )
                for x in (case, other)
            )
            assert a.keys() == b.keys()
            assert all(torch.equal(a[k], b[k]) for k in a), (model, delete)
        ### the perturbation reached the labels
        if not delete:
            assert not torch.equal(
                other.global_data["reference", "velocity"],
                case.global_data["reference", "velocity"],
            )


def test_the_reader_and_the_chain_end_to_end(tmp_path):
    """The file through the real reader: every boundary face is read and SubsampleMesh (capped at 5 here)
    keeps a uniform random subset with the Horvitz-Thompson measure N/k; the interior is the reader's point
    draw with no target measure (an unweighted mean over the sampled points, as on DrivAerML Volume); the four
    boundaries come out reversed (normals into the fluid), the merged tokens in SI order, the targets
    standardized by scale only."""
    path = tmp_path / "shard-000" / "case-0001" / "domain.pdmsh"
    path.parent.mkdir(parents=True)
    stored = make_aerozoo_case(True, seed=1)
    stored.save(path)
    cfg = OmegaConf.merge(
        load_dataset_config(_RECIPE_ROOT / "datasets" / f"{DATASET}.yaml"),
        {
            "dataset_paths": {"aerozoo": str(tmp_path), "aerozoo_manifest": "/unused"},
            "sampling_resolution": SAMPLING,
        },
    )
    cfg.pipeline.transforms[0].n_cells = 5
    ds = build_dataset(cfg, device=None, num_workers=1)
    try:
        assert ds.reader[0][0].boundaries["no_slip"].n_cells == 12  # read whole
        ds.set_generator(torch.Generator().manual_seed(3))
        d = ds[0][0]
    finally:
        DatasetBase.close(ds)
    for name, n in (("inlet", 4), ("outlet", 4), ("no_slip", 12), ("slip", 4)):
        bnd = d.boundaries[name]
        assert bnd.n_cells == min(n, 5), name
        torch.testing.assert_close(
            bnd.cell_data[EFFECTIVE_MEASURE_KEY], bnd.cell_areas * max(n / 5, 1.0)
        )
        ### reversed: each kept face's normal is the stored one's negative
        src = stored.boundaries[name]
        index = (
            (bnd.cell_centroids[:, None, :] - src.cell_centroids[None]).norm(dim=-1)
        ).argmin(dim=1)
        torch.testing.assert_close(bnd.cell_normals, -src.cell_normals[index])
    assert d.boundaries["merged"].n_cells == 4 + 4 + 5 + 4
    assert d.interior.n_points == SAMPLING and extract_target_measure(d) is None
    std = OmegaConf.to_container(cfg.pipeline.transforms[8].fields)["pressure"]["std"]
    raw = stored.interior.point_data["solution", "gauge_pressure"]
    p = d.interior.point_data["pressure"] * std
    assert all(bool(torch.isclose(x, raw, rtol=1e-5).any()) for x in p[:5])


def test_scale_only_standardization_makes_the_metric_physical():
    """With mean 0 (scale-only standardization, every target of these files), the recipe's training-space
    relative L2 of a field is its relative L2 in SI units, per field and case (the batch is one case); with a
    nonzero mean it would not be. So training-space scores are physical scores, and no label is needed."""
    chain = _chain(DATASET)
    normalizer = next(t for t in chain if type(t).__name__ == "NormalizeMeshFields")
    stats = normalizer.stats
    assert all(
        float(torch.as_tensor(s["mean"]).abs().max()) == 0 for s in stats.values()
    )
    g = torch.Generator().manual_seed(0)
    phys_t = {
        "velocity": torch.randn(1, 200, 3, generator=g) * 7 + 3,
        "pressure": torch.randn(1, 200, generator=g) * 40 + 25,
        "nut": torch.rand(1, 200, generator=g) * 1e-3,
        "tke": torch.rand(1, 200, generator=g) * 0.4,
        "omega": torch.rand(1, 200, generator=g) * 900,
    }
    phys_p = {
        k: v * (1 + 0.1 * torch.randn(v.shape, generator=g)) for k, v in phys_t.items()
    }
    td = lambda d: TensorDict(d, batch_size=[1, 200])  # noqa: E731
    std = {
        k: float(stats[k]["std"]) + 1e-8 for k in TARGETS
    }  # NormalizeMeshFields' eps
    calc = MetricCalculator(TARGETS, metrics=["l2"])
    physical = calc(td(phys_p), td(phys_t))
    training = calc(
        td({k: v / std[k] for k, v in phys_p.items()}),
        td({k: v / std[k] for k, v in phys_t.items()}),
    )
    for key in physical.keys():
        torch.testing.assert_close(training[key], physical[key], rtol=1e-5, atol=0.0)
    ### the case-relative loss is the same quantity per field (velocity's components jointly)
    loss = LossCalculator(TARGETS, loss_type="case_relative_l2")(
        td({k: v / std[k] for k, v in phys_p.items()}),
        td({k: v / std[k] for k, v in phys_t.items()}),
    )[1]
    for name in TARGETS:
        torch.testing.assert_close(
            loss[f"loss/{name}"], physical[f"{name}_l2"], rtol=1e-4, atol=0.0
        )
    ### an offset breaks it: standardizing pressure with a mean changes its relative L2
    shifted = calc(
        td({**phys_p, "pressure": (phys_p["pressure"] - 25) / 40}),
        td({**phys_t, "pressure": (phys_t["pressure"] - 25) / 40}),
    )
    assert not torch.isclose(shifted["pressure_l2"], physical["pressure_l2"], rtol=1e-2)


def test_isla_config_brings_its_own_scales_and_runs():
    """ISLA declares the four conditions by name with the values each prescribes (in SI units, with their
    units), drops the steps only the baselines read, consumes rho by its role and derives its own scales: U
    and the direction g from the inlet's velocity (K = 1, no label-derived global), L from the four
    boundaries. Its Reynolds number re_z is the recipe's, appended to the chain with the dataset file's
    isla_constants.reynolds, and is bitwise the one the model computed up to 0.12.0 from the same U and L (its
    arithmetic copied here); a dataset file without the constants is refused. Its outputs are standardized by the dataset's own NormalizeMeshFields stats (injected as
    output_stats), so the per-case relative loss with velocity weighed 3 (the baselines') compares them with
    the same targets."""
    cfg = _compose(
        "isla_volume_aerozoo",
        "model.hidden=16",
        "model.n_layers=1",
        "model.n_slices=4",
        "model.compile_blocks=false",
        "model.n_boundary_tokens=64",
    )
    assert {
        name: {k: v["rank"] for k, v in schema.items()}
        for name, schema in OmegaConf.to_container(cfg.model.boundary_schemas).items()
    } == ISLA_CONDITIONS
    assert cfg.training.loss_type == "case_relative_l2"
    assert OmegaConf.to_container(cfg.training.field_weights) == {"velocity": 3}
    ds = load_dataset_config(_RECIPE_ROOT / "datasets" / f"{DATASET}.yaml")
    kept = [
        str(t["_target_"])
        for t in _apply_dataset_transform_drops(ds, cfg).pipeline.transforms
    ]
    dropped = (
        ".SubsampleMesh",
        ".ComputeSDFFromBoundary",
        ".HarmonizeBoundaryFields",
        ".MergeBoundaries",
    )
    assert not any(t.endswith(dropped) for t in kept)
    assert len(kept) == len(ds.pipeline.transforms) - len(dropped)
    chain = _chain(DATASET, cfg)
    (normalizer,) = [t for t in chain if type(t).__name__ == "NormalizeMeshFields"]
    d = _run(chain, make_aerozoo_case(True))
    spec = OmegaConf.to_container(cfg.forward_kwargs, resolve=True)
    batch = build_collate_fn("mesh", spec, TARGETS)([(d, {})])
    model = hydra.utils.instantiate(
        cfg.model,
        _convert_="partial",
        **model_output_stats(cfg.model, normalizer),
        **model_dataset_constants(cfg.model, ds),
    )
    check_model_output_stats(model, normalizer)
    assert model.n_global_vectors == 1 and model.n_global_scalars == 1  # g; re_z
    assert "reynolds" not in model._args["__args__"]
    (log_re,) = [t for t in chain if type(t).__name__ == "ComputeLogReynolds"]
    re = OmegaConf.to_container(ds.isla_constants.reynolds)
    assert (log_re.mean, log_re.std) == (re["mean"], re["std"])
    no_constants = ds.copy()
    del no_constants["isla_constants"]
    with pytest.raises(ValueError, match="isla_constants.reynolds"):
        _apply_dataset_transform_appends(no_constants, cfg)
    assert model.output_stats["pressure"] == {
        "mean": 0.0,
        "std": pytest.approx(1129.26),
    }
    seen = []
    scales = model._case_scales
    model._case_scales = lambda *a: seen.append(scales(*a)) or seen[-1]
    with torch.no_grad():
        out = model(**batch["forward_kwargs"])
    torch.testing.assert_close(seen[0]["U"], torch.tensor(U_INF, dtype=torch.float64))
    nu = d.global_data["physics", "kinematic_viscosity"].float().double()
    z = (torch.log10(seen[0]["U"] * seen[0]["L"] / nu) - re["mean"]) / re["std"]
    assert torch.equal(d.global_data["re_z"], z)
    torch.testing.assert_close(
        seen[0]["g"][0, 0], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    )
    pred = normalize_output_to_tensordict(out, TARGETS, "mesh")
    total, _ = LossCalculator(
        TARGETS,
        loss_type=cfg.training.loss_type,
        field_weights=OmegaConf.to_container(cfg.training.field_weights),
    )(pred, batch["targets"])
    assert torch.isfinite(total)
    model.output_stats = {**model.output_stats, "nut": {"mean": 0.0, "std": 1.0}}
    with pytest.raises(ValueError, match="not the dataset's NormalizeMeshFields stats"):
        check_model_output_stats(model, normalizer)
