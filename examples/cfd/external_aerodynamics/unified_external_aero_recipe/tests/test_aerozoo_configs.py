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

"""AeroZoo v1.2 (2026-10-09): datasets/aerozoo_volume.yaml run end to end by the real reader on a
synthetic release-shaped case (four boundaries, one case without slip faces), and the two model configs
(isla_volume_aerozoo, geotransolver_volume_aerozoo) composed, driven through collate -> forward -> loss on
the pipeline's output, and pinned to their protocol values."""

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
from datasets import (  # noqa: E402
    _apply_dataset_reader_overrides,
    build_dataset,
    load_dataset_config,
)
from forward_kwargs import extract_target_measure  # noqa: E402
from infer import build_redim_field_types, redimensionalize  # noqa: E402
from loss import LossCalculator  # noqa: E402
from metrics import MetricCalculator  # noqa: E402
from nondim import NonDimensionalizeByMetadata  # noqa: E402
from output_normalize import normalize_output_to_tensordict  # noqa: E402
from utils import field_dim  # noqa: E402

from physicsnemo.datapipes.protocols import DatasetBase  # noqa: E402
from physicsnemo.mesh import DomainMesh, Mesh  # noqa: E402
from physicsnemo.mesh.calculus.measure import EFFECTIVE_MEASURE_KEY  # noqa: E402

DATASET = "aerozoo_volume"
TARGETS = {
    "velocity": "vector",
    "pressure": "scalar",
    "nut": "scalar",
    "tke": "scalar",
    "omega": "scalar",
}
TARGETS_VP = {"velocity": "vector", "pressure": "scalar"}
### Every AeroZoo dataset file: the split part fill_stats.py filled its statistics from, and its targets. The
### `_vp` variants keep the pipeline (nut, tke and omega become unused leaves) and declare two targets.
DATASET_FILES = {
    "aerozoo_volume": ("full", TARGETS),
    "aerozoo_volume_nonbox": ("family_non_box", TARGETS),
    "aerozoo_volume_reynolds": ("reynolds", TARGETS),
    "aerozoo_volume_blobs": ("bank_blobs", TARGETS),
    **{
        f"aerozoo_volume_{split}": (split, TARGETS)
        for split in (
            "bank_airframes",
            "bank_blender_bluff",
            "bank_blender_streamlined",
            "bank_cars",
            "bank_streamlined",
            "bank_zero_to_cad",
            "family_duct",
            "family_opposed_hemisphere",
            "family_opposed_sphere",
            "layout",
            "profile_confined_opposed",
            "profile_crossflow",
            "profile_dual_inlet",
            "profile_ground_effect",
            "profile_open_opposed",
            "reynolds_interior",
            "scarce",
            "super_scarce",
        )
    },
    "aerozoo_volume_vp": ("full", TARGETS_VP),
    "aerozoo_volume_nonbox_vp": ("family_non_box", TARGETS_VP),
}
### The ISLA lanes on a `_vp` dataset drop the three turbulence outputs from the model's output_schema.
ISLA_VP_OVERRIDES = {
    "aerozoo_volume": (),
    "aerozoo_volume_vp": (
        "~model.output_schema.nut",
        "~model.output_schema.tke",
        "~model.output_schema.omega",
    ),
}
### AZ-UNIFORM (2026-10-14): every file's uniform-boundary twin aerozoo_uvolume<suffix>.yaml (the reader reads the
### boundaries whole, SubsampleMesh draws a uniform 50,000 faces per boundary first), tested like its source.
UNIFORM_TWINS = {
    name.replace("aerozoo_volume", "aerozoo_uvolume", 1): name for name in DATASET_FILES
}
DATASET_FILES.update({twin: DATASET_FILES[src] for twin, src in UNIFORM_TWINS.items()})
ISLA_VP_OVERRIDES.update(
    {
        twin: ISLA_VP_OVERRIDES[src]
        for twin, src in UNIFORM_TWINS.items()
        if src in ISLA_VP_OVERRIDES
    }
)
SOLUTION = (
    "velocity",
    "gauge_pressure",
    "turbulent_kinetic_energy",
    "specific_dissipation_rate",
    "turbulent_kinematic_viscosity",
    "intermittency",
    "momentum_thickness_reynolds_number",
)
TURBULENCE = (
    "turbulent_kinetic_energy",
    "specific_dissipation_rate",
    "intermittency",
    "momentum_thickness_reynolds_number",
)
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

### Placeholders of the dataset yaml (the stats script fills the real ones).
STATS = {
    "velocity": {"mean": [0.5, 0.0, 0.0], "std": 0.3},
    "pressure": {"mean": -0.1, "std": 0.5},
    "nut": {"mean": 1.0e-4, "std": 2.0e-4},
    "tke": {"mean": 1.0e-3, "std": 2.0e-3},
    "omega": {"mean": 10.0, "std": 20.0},
}
INLET_MEAN, INLET_STD = [1.0e-3, 5.0, 1.0, 300.0], [1.0e-3, 2.0, 1.0, 100.0]


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


def _filled_dataset_yaml(root: Path, dataset: str = DATASET):
    """The dataset yaml pointed at ``root`` with the stats placeholders filled."""
    ds = load_dataset_config(_RECIPE_ROOT / "datasets" / f"{dataset}.yaml")
    transforms = OmegaConf.to_container(ds.pipeline.transforms, resolve=False)
    for t in transforms:
        if "NormalizeMeshFields" in t["_target_"]:
            for name, s in STATS.items():
                t["fields"][name].update(s)
        if "SplitGlobalVector" in t["_target_"]:
            t["mean"], t["std"] = INLET_MEAN, INLET_STD
    return OmegaConf.merge(
        ds,
        {
            "dataset_paths": {
                "aerozoo": str(root),
                "aerozoo_manifest": str(root / "manifest.json"),
            },
            "sampling_resolution": SAMPLING,
            "pipeline": {"transforms": transforms},
        },
    )


def _read_back(tmp_path_factory, dataset: str) -> dict[str, DomainMesh]:
    """Two cases written in the release tree and read back through the yaml's reader and transforms."""
    root = tmp_path_factory.mktemp(dataset)
    for case, with_slip in (("case-0001", True), ("case-0002", False)):
        path = root / "shard-000" / case / "domain.pdmsh"
        path.parent.mkdir(parents=True)
        make_aerozoo_case(with_slip, seed=int(case[-1])).save(path)
    ds = build_dataset(_filled_dataset_yaml(root, dataset), device=None, num_workers=1)
    try:
        return {"with_slip": ds[0][0], "no_slip_walls": ds[1][0]}
    finally:
        DatasetBase.close(ds)


@pytest.fixture(scope="module")
def pipeline_output(tmp_path_factory) -> dict[str, DomainMesh]:
    return _read_back(tmp_path_factory, DATASET)


@pytest.fixture(scope="module")
def pipeline_output_vp(tmp_path_factory) -> dict[str, DomainMesh]:
    return _read_back(tmp_path_factory, "aerozoo_volume_vp")


@pytest.fixture(scope="module")
def pipeline_output_uniform(tmp_path_factory) -> dict[str, DomainMesh]:
    return _read_back(tmp_path_factory, "aerozoo_uvolume")


@pytest.fixture(scope="module")
def pipeline_output_uniform_vp(tmp_path_factory) -> dict[str, DomainMesh]:
    return _read_back(tmp_path_factory, "aerozoo_uvolume_vp")


def _outputs(request, dataset: str) -> dict[str, DomainMesh]:
    return request.getfixturevalue(
        {
            "aerozoo_volume": "pipeline_output",
            "aerozoo_volume_vp": "pipeline_output_vp",
            "aerozoo_uvolume": "pipeline_output_uniform",
            "aerozoo_uvolume_vp": "pipeline_output_uniform_vp",
        }[dataset]
    )


def _out_dim(targets: dict[str, str]) -> int:
    """GeoTransolver's out_dim as build_dataloaders derives it from the dataset's `targets:` block."""
    return sum(field_dim(t) for t in targets.values())


def _compose(model, *extra, dataset: str = DATASET):
    out_dim = _out_dim(DATASET_FILES[dataset][1])
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


### ---------------------------------------------------------------------------
### The dataset pipeline
### ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["with_slip", "no_slip_walls"])
def test_pipeline_output_contract(pipeline_output, case):
    """Every boundary carries the fifteen harmonized fields at their ranks (zero rows on the case without
    slip faces); the interior carries the five standardized targets, the two signed distances and no raw
    solution group; the globals are the two models' inputs."""
    d = pipeline_output[case]
    assert d.boundary_names == ["inlet", "outlet", "no_slip", "slip"] or set(
        d.boundary_names
    ) == {"inlet", "outlet", "no_slip", "slip"}
    for name, bnd in d.boundaries.items():
        n = bnd.n_cells
        for field, rank in BC_FIELDS.items():
            assert bnd.cell_data[field].shape == (n,) + (3,) * rank, (name, field)
        for role in ("inlet", "outlet", "no_slip", "slip"):
            assert torch.equal(
                bnd.cell_data[f"bc_{role}"], torch.full((n,), float(role == name))
            )
        assert "prescribed" not in bnd.cell_data.keys()
    assert d.boundaries["slip"].n_cells == (4 if case == "with_slip" else 0)
    assert d.interior.n_points == SAMPLING  # the reader's point subsample
    pd = d.interior.point_data
    assert pd["velocity"].shape == (SAMPLING, 3)
    for name in ("pressure", "nut", "tke", "omega", "_target_quadrature_measure"):
        assert pd[name].shape == (SAMPLING,), name
    assert pd["sdf_wall"].shape == (SAMPLING, 1) and pd["sdf_wall_normals"].shape == (
        SAMPLING,
        3,
    )
    assert pd["sdf_slip"].shape == (SAMPLING, 1)
    assert "solution" not in pd.keys()
    gd = d.global_data
    torch.testing.assert_close(gd["U_inf_dir"], torch.tensor([1.0, 0.0, 0.0]))
    for name in (
        "log_re_std",
        "inlet_tke_std",
        "inlet_omega_std",
        "inlet_intermittency_std",
        "inlet_retheta_std",
    ):
        assert gd[name].shape == (), name
    assert gd["global_vector"].shape == (8,)
    torch.testing.assert_close(
        gd["global_vector"],
        torch.cat(
            [gd["U_inf_dir"], gd["log_re_std"].reshape(1), gd["inlet_turbulence_std"]]
        ),
    )
    ### U_inf stays physical for re-dimensionalization; p_inf = 0 makes `pressure` Cp.
    torch.testing.assert_close(gd["U_inf"], torch.tensor([U_INF, 0.0, 0.0]))
    assert float(gd["p_inf"]) == 0.0 and float(gd["L_ref"]) == L_REF


def test_pipeline_values(pipeline_output):
    """The numbers: nondimensional, centred, standardized as the yaml declares."""
    d = pipeline_output["with_slip"]
    gd, pd = d.global_data, d.interior.point_data
    log_re = torch.log10(torch.tensor(U_INF * L_REF / NU))
    torch.testing.assert_close(gd["log_re"], log_re)
    torch.testing.assert_close(gd["log_re_std"], (log_re - 5.5) / 1.5)
    ### The inlet's uniform turbulence, nondimensional, then standardized by the placeholders.
    nondim = torch.tensor(
        [
            INLET_TURBULENCE["turbulent_kinetic_energy"] / U_INF**2,
            INLET_TURBULENCE["specific_dissipation_rate"] * L_REF / U_INF,
            1.0,
            300.0,
        ]
    )
    torch.testing.assert_close(gd["inlet_turbulence"], nondim)
    torch.testing.assert_close(
        gd["inlet_turbulence_std"],
        (nondim - torch.tensor(INLET_MEAN)) / torch.tensor(INLET_STD),
    )
    torch.testing.assert_close(gd["inlet_omega_std"], gd["inlet_turbulence_std"][1])
    ### Boundary values: the inlet velocity is 1 along the freestream, the outlet pressure Cp = 0, the
    ### outlet's reverse-flow k nondimensional; the body prescribes zero velocity.
    inlet, outlet, body = (
        d.boundaries["inlet"],
        d.boundaries["outlet"],
        d.boundaries["no_slip"],
    )
    torch.testing.assert_close(
        inlet.cell_data["bc_velocity"],
        torch.tensor([1.0, 0.0, 0.0]).expand(inlet.n_cells, 3),
    )
    torch.testing.assert_close(
        inlet.cell_data["bc_tke"], torch.full((inlet.n_cells,), 0.5 / U_INF**2)
    )
    assert torch.equal(
        inlet.cell_data["bc_velocity_reverse"], torch.zeros(inlet.n_cells, 3)
    )
    torch.testing.assert_close(
        outlet.cell_data["bc_pressure"], torch.zeros(outlet.n_cells)
    )
    torch.testing.assert_close(
        outlet.cell_data["bc_tke_reverse"],
        torch.full((outlet.n_cells,), 0.5 / U_INF**2),
    )
    assert torch.equal(body.cell_data["bc_velocity"], torch.zeros(body.n_cells, 3))
    ### Geometry: divided by L_ref and centred on the interior sample's mean; the body moved with it.
    torch.testing.assert_close(
        d.interior.points.mean(0), torch.zeros(3), atol=1e-5, rtol=0
    )
    extent = body.points.amax(0) - body.points.amin(0)
    torch.testing.assert_close(extent, torch.tensor([4.0, 1.0, 1.0]) / L_REF)
    ### Targets: nondimensional then (x - mean) / std with the placeholders; the raw data is recovered
    ### from the same case written to disk.
    raw = make_aerozoo_case(True, seed=1).interior.point_data["solution"]
    v = pd["velocity"] * STATS["velocity"]["std"] + torch.tensor(
        STATS["velocity"]["mean"]
    )
    ### The reader's draw is a contiguous block, so the kept rows are a subset of the raw rows.
    assert all(
        any(torch.allclose(x * U_INF, r, atol=1e-4) for r in raw["velocity"])
        for x in v[:5]
    )
    p = pd["pressure"] * STATS["pressure"]["std"] + STATS["pressure"]["mean"]
    assert all(
        any(torch.allclose(x * Q_INF, r, atol=1e-3) for r in raw["gauge_pressure"])
        for x in p[:5]
    )
    omega = pd["omega"] * STATS["omega"]["std"] + STATS["omega"]["mean"]
    assert all(
        any(
            torch.allclose(x * U_INF / L_REF, r, atol=1e-3)
            for r in raw["specific_dissipation_rate"]
        )
        for x in omega[:5]
    )
    ### SDF: outside the closed body everywhere (the sample points are drawn outside it with
    ### overwhelming probability), unit normals; the slip floor sits below every point.
    assert bool((pd["sdf_wall"] > -1e-5).all())
    torch.testing.assert_close(
        pd["sdf_wall_normals"].norm(dim=-1), torch.ones(SAMPLING), atol=1e-4, rtol=0
    )


def test_interior_measure_wiring_matches_drivaer_volume(pipeline_output):
    """The cell volumes stay an inert leaf: the interior has no `_effective_measure`, so the collate
    hands the loss no target measure and the loss is an unweighted mean over the sampled points, as on
    DrivAerML Volume. The subsampled-boundary measure path is untouched."""
    d = pipeline_output["with_slip"]
    assert EFFECTIVE_MEASURE_KEY not in d.interior.point_data.keys()
    assert "_target_quadrature_measure" in d.interior.point_data.keys()
    assert extract_target_measure(d) is None
    ### The boundaries carry the materialized measure ISLA reads; no boundary is above the 50,000-face
    ### cap here, so it is the plain geometric area; a boundary without cells has none.
    for name, bnd in d.boundaries.items():
        if bnd.n_cells:
            torch.testing.assert_close(
                bnd.cell_data[EFFECTIVE_MEASURE_KEY], bnd.cell_areas
            )
        else:
            assert EFFECTIVE_MEASURE_KEY not in bnd.cell_data.keys(), name


def test_absent_slip_walls_get_the_far_fill(pipeline_output):
    d = pipeline_output["no_slip_walls"]
    pts = d.interior.points
    diagonal = (pts.amax(0) - pts.amin(0)).norm()
    torch.testing.assert_close(
        d.interior.point_data["sdf_slip"], diagonal.expand(SAMPLING, 1)
    )
    assert d.boundaries["slip"].n_cells == 0 and d.boundaries["slip"].cells.shape == (
        0,
        3,
    )


### ---------------------------------------------------------------------------
### The model configs
### ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset", sorted(DATASET_FILES))
def test_dataset_yaml_pins_the_reader_and_the_chain(dataset):
    split, targets = DATASET_FILES[dataset]
    path = _RECIPE_ROOT / "datasets" / f"{dataset}.yaml"
    assert path.read_text().startswith(
        f"# Statistics filled by fill_stats.py from stats_{split}.train.json: split part {split}.train, "
    )
    ds = OmegaConf.load(path)
    reader = OmegaConf.to_container(ds.pipeline.reader, resolve=False)
    if dataset in UNIFORM_TWINS:
        ### The twin: the reader reads the boundaries whole; the cap is the first transform's.
        assert "subsample_n_cells" not in reader
        reader["subsample_n_cells"] = ds.pipeline.transforms[0].n_cells
    assert (
        reader["subsample_n_cells"] == 50000
    )  # a literal: 4 x 50,000 boundary tokens at any interior resolution
    assert reader["subsample_n_points"] == "${sampling_resolution}"
    assert reader["boundary_subsample"] == "cells"
    assert (
        reader["drop_interior_cells"] is True
        and reader["drop_in_file_boundaries"] is False
    )
    assert reader["pattern"] == "shard-*/case-*/domain.pdmsh"
    chain = [
        t["_target_"]
        for t in OmegaConf.to_container(ds.pipeline.transforms, resolve=False)
    ]
    assert "ComposeQuadratureMeasure" in chain[-1]
    assert OmegaConf.to_container(ds.targets) == targets
    ### fill_stats.py has filled the normalization statistics (the yaml's first line names the split part and the sample
    ### counts), so the standardizing transform instantiates with no mandatory value missing.
    assert "???" not in OmegaConf.to_yaml(ds.pipeline.transforms[-2])
    hydra.utils.instantiate(ds.pipeline.transforms[-2], _convert_="all")


@pytest.mark.parametrize("twin", sorted(UNIFORM_TWINS))
def test_uniform_twin_is_its_source_but_the_boundary_draw(twin):
    """aerozoo_uvolume<suffix>.yaml is aerozoo_volume<suffix>.yaml with the reader's boundary cap moved to a
    SubsampleMesh first in the chain (before the SDFs, the measures and the harmonized fields read the boundaries)
    and a header line after the statistics line; the interior draw, the chain and the statistics are the source's."""
    src = UNIFORM_TWINS[twin]
    datasets = _RECIPE_ROOT / "datasets"
    twin_lines = (datasets / f"{twin}.yaml").read_text().splitlines()
    src_lines = (datasets / f"{src}.yaml").read_text().splitlines()
    assert twin_lines[0] == src_lines[0]
    assert twin_lines[1].startswith(
        f"# AZ-UNIFORM (2026-10-14): {src}.yaml with uniform"
    )
    a = OmegaConf.to_container(OmegaConf.load(datasets / f"{twin}.yaml"), resolve=False)
    b = OmegaConf.to_container(OmegaConf.load(datasets / f"{src}.yaml"), resolve=False)
    assert b["pipeline"]["reader"].pop("subsample_n_cells") == 50000
    first = a["pipeline"]["transforms"].pop(0)
    assert first == {"_target_": "${dp:SubsampleMesh}", "n_cells": 50000}
    assert a == b


def test_uniform_twin_draws_a_uniform_subset_of_the_whole_boundary(tmp_path):
    """The twin's reader hands over every boundary face and its SubsampleMesh (capped at 5 faces here) keeps a
    uniform random subset, not the reader's cyclic contiguous block, with the Horvitz-Thompson measure N/k through
    the whole chain; the interior's point draw is the source file's for the same seed."""
    path = tmp_path / "shard-000" / "case-0001" / "domain.pdmsh"
    path.parent.mkdir(parents=True)
    make_aerozoo_case(True, seed=1).save(path)
    cfg = _filled_dataset_yaml(tmp_path, "aerozoo_uvolume")
    cfg.pipeline.transforms[0].n_cells = 5
    twin = build_dataset(cfg, device=None, num_workers=1)
    src = build_dataset(
        _filled_dataset_yaml(tmp_path, "aerozoo_volume"), device=None, num_workers=1
    )
    try:
        raw = twin.reader[0][0]
        body = raw.boundaries["no_slip"]
        assert body.n_cells == 12  # read whole
        y_plus = body.cell_data["derived", "wall_y_plus"]  # a face's identity
        blocks = {frozenset((s + i) % 12 for i in range(5)) for s in range(12)}
        subsample, draws = twin.transforms[0], set()
        for seed in range(20):
            subsample.set_generator(torch.Generator().manual_seed(seed))
            kept = subsample.apply_to_domain(raw).boundaries["no_slip"]
            index = (
                kept.cell_data["derived", "wall_y_plus"][:, None] == y_plus
            ).nonzero()[:, 1]
            assert kept.n_cells == 5 and index.unique().numel() == 5
            torch.testing.assert_close(
                kept.cell_data[EFFECTIVE_MEASURE_KEY], body.cell_areas[index] * 12 / 5
            )
            draws.add(frozenset(index.tolist()))
        assert len(draws) > 12 and not draws <= blocks
        ### End to end with one seed for both files: the boundaries capped at 5 faces with the N/k measure in
        ### nondimensional units, the interior sample (points and targets) the source file's.
        for ds in (twin, src):
            ds.set_generator(torch.Generator().manual_seed(3))
        d, s = twin[0][0], src[0][0]
        for name, n in (("inlet", 4), ("outlet", 4), ("no_slip", 12), ("slip", 4)):
            bnd = d.boundaries[name]
            assert bnd.n_cells == min(n, 5), name
            torch.testing.assert_close(
                bnd.cell_data[EFFECTIVE_MEASURE_KEY], bnd.cell_areas * max(n / 5, 1.0)
            )
        torch.testing.assert_close(d.interior.points, s.interior.points)
        for name in ("velocity", "pressure", "nut", "tke", "omega"):
            torch.testing.assert_close(
                d.interior.point_data[name], s.interior.point_data[name]
            )
    finally:
        DatasetBase.close(twin)
        DatasetBase.close(src)


@pytest.mark.parametrize("dataset", sorted(ISLA_VP_OVERRIDES))
def test_isla_config_declares_the_harmonized_grammar_and_runs(request, dataset):
    pipeline_output = _outputs(request, dataset)
    targets = DATASET_FILES[dataset][1]
    cfg = _compose(
        "isla_volume_aerozoo",
        "model.hidden=16",
        "model.n_layers=1",
        "model.n_slices=4",
        "model.geo_kernel=eager",
        "model.compile_blocks=false",
        *ISLA_VP_OVERRIDES[dataset],
        dataset=dataset,
    )
    m = cfg.model
    assert cfg.input_type == "mesh" and cfg.output_type == "mesh"
    assert cfg.training.optimizer.lr == pytest.approx(3.75e-4)
    assert OmegaConf.to_container(cfg.forward_kwargs.boundary_meshes) == {
        r: f"boundaries.{r}" for r in ("inlet", "outlet", "no_slip", "slip")
    }
    assert {
        k: v["rank"] for k, v in OmegaConf.to_container(m.boundary_schema).items()
    } == BC_FIELDS
    assert OmegaConf.to_container(m.global_schema) == {
        "U_inf_dir": {"rank": 1},
        "log_re_std": {"rank": 0},
        "inlet_tke_std": {"rank": 0},
        "inlet_omega_std": {"rank": 0},
        "inlet_intermittency_std": {"rank": 0},
        "inlet_retheta_std": {"rank": 0},
    }
    assert set(OmegaConf.to_container(m.output_schema)) == set(targets)
    assert m.get("query_schema") is None and m.interior is True
    model = hydra.utils.instantiate(m)
    spec = OmegaConf.to_container(cfg.forward_kwargs, resolve=True)
    collate = build_collate_fn("mesh", spec, targets)
    ### The scorer's re-dimensionalization (infer.py) from the dataset yaml's own chain.
    ds = _filled_dataset_yaml(Path("unused"), dataset)
    normalizer = hydra.utils.instantiate(ds.pipeline.transforms[-2], _convert_="all")
    field_types = build_redim_field_types(ds)
    for case in (
        "with_slip",
        "no_slip_walls",
    ):  # the second hands ISLA a boundary with zero faces
        domain = pipeline_output[case]
        ### The pipeline output still carries all five standardized fields; only the declared ones are read.
        assert {"nut", "tke", "omega"} <= set(domain.interior.point_data.keys())
        batch = collate([(domain, {})])
        assert "target_measure" not in batch
        assert set(batch["targets"].keys()) == set(targets)
        with torch.no_grad():
            out = model(**batch["forward_kwargs"])
        assert isinstance(out, Mesh)
        torch.testing.assert_close(out.points, domain.interior.points)
        pred = normalize_output_to_tensordict(out, targets, "mesh")
        assert set(pred.keys()) == set(targets)
        assert pred["velocity"].shape == (SAMPLING, 3)
        assert pred["pressure"].shape == (SAMPLING,)
        total, loss_td = LossCalculator(targets, loss_type="mse")(
            pred.float(), batch["targets"].float()
        )
        assert torch.isfinite(total)
        metrics = MetricCalculator(targets, metrics=["l2"])(
            pred.float(), batch["targets"].float()
        )
        assert {k.split("_")[0] for k in metrics.keys()} == set(targets)
        assert set(loss_td.keys()) == {f"loss/{n}" for n in (*targets, "total")}
        phys = redimensionalize(
            pred,
            normalizer=normalizer,
            nondim=NonDimensionalizeByMetadata(fields=field_types),
            field_types=field_types,
            global_data=domain.global_data,
        )
        assert set(phys.keys()) == set(targets)
        assert all(torch.isfinite(v).all() for v in phys.values())


@pytest.mark.parametrize("dataset", sorted(ISLA_VP_OVERRIDES))
def test_geotransolver_config_reads_the_sdf_and_global_vector_and_runs(
    request, dataset
):
    pipeline_output = _outputs(request, dataset)
    targets = DATASET_FILES[dataset][1]
    cfg = _compose("geotransolver_volume_aerozoo", dataset=dataset)
    assert cfg.model.out_dim == (4 if dataset.endswith("_vp") else 7)
    assert cfg.input_type == "tensors" and cfg.output_type == "tensors"
    assert cfg.training.optimizer.lr == pytest.approx(
        1.0e-3
    )  # the CHAR campaign's GeoTransolver rate
    assert cfg.model.functional_dim == 8 and cfg.model.global_dim == 8
    assert (
        cfg.model.n_hidden == 256
        and cfg.model.n_layers == 12
        and cfg.model.slice_num == 256
    )
    assert cfg.forward_kwargs.global_embedding == "global_data.global_vector"
    ### The boundaries stay in the reader: the SDF transforms need them.
    ds = OmegaConf.load(_RECIPE_ROOT / "datasets" / f"{dataset}.yaml")
    assert (
        _apply_dataset_reader_overrides(ds, cfg).pipeline.reader.drop_in_file_boundaries
        is False
    )
    small = OmegaConf.merge(
        cfg.model,
        {
            "n_layers": 2,
            "n_hidden": 32,
            "slice_num": 16,
            "n_head": 2,
            "include_local_features": False,
        },
    )
    model = hydra.utils.instantiate(small, _convert_="partial")
    spec = OmegaConf.to_container(cfg.forward_kwargs, resolve=True)
    batch = build_collate_fn("tensors", spec, targets)(
        [(pipeline_output["no_slip_walls"], {})]
    )
    fk = batch["forward_kwargs"]
    assert fk["local_embedding"].shape == (1, SAMPLING, cfg.model.functional_dim)
    assert fk["global_embedding"].shape == (1, 1, cfg.model.global_dim)
    with torch.no_grad():
        out = model(**fk)
    assert out.shape[-1] == cfg.model.out_dim
    pred = normalize_output_to_tensordict(out, targets, "tensors")
    assert set(pred.keys()) == set(batch["targets"].keys()) == set(targets)
    for name in targets:
        assert pred[name].shape == batch["targets"][name].shape, name
    total, _ = LossCalculator(targets, loss_type="mse")(pred, batch["targets"])
    assert torch.isfinite(total)
    metrics = MetricCalculator(targets, metrics=["l2"])(pred, batch["targets"])
    assert {k.split("_")[0] for k in metrics.keys()} == set(targets)
