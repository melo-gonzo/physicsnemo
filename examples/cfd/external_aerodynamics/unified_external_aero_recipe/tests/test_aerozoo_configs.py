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
synthetic release-shaped case (four boundaries, one case without slip faces), and the ISLA config composed,
driven through collate -> forward -> loss on the pipeline's output, and pinned to its protocol values.

AeroZoo v2 (2026-10-19): the physical-input files datasets/aerozoo_puvolume*.yaml, the GeoTransolver
(Option B) and Transolver configs on them, the label check (every label perturbed or deleted leaves every
model input bitwise equal) and the label-reference score map (test D: exact against the label-era file)."""

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
    _apply_dataset_transform_drops,
    build_dataset,
    load_dataset_config,
)
from forward_kwargs import extract_target_measure  # noqa: E402
from infer import (  # noqa: E402
    LABEL_REFERENCE_PREFIX,
    LabelReferenceMap,
    LabelReferenceMetricCalculator,
    build_redim_field_types,
    redimensionalize,
)
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

### AeroZoo v2 (2026-10-19): the physical-input files, the split part aerozoo_pstats.py filled each from, and the
### label-era file of the same split whose space the scorer maps into.
V2_FILES = {
    f"aerozoo_puvolume{suffix}": (split, TARGETS, f"aerozoo_uvolume{suffix}")
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


def _read_back(tmp_path_factory, dataset: str, model_cfg=None) -> dict[str, DomainMesh]:
    """Two cases written in the release tree and read back through the yaml's reader and transforms
    (with ``model_cfg``'s reader overrides and transform drops, the model's view of the dataset)."""
    root = tmp_path_factory.mktemp(dataset)
    for case, with_slip in (("case-0001", True), ("case-0002", False)):
        path = root / "shard-000" / case / "domain.pdmsh"
        path.parent.mkdir(parents=True)
        make_aerozoo_case(with_slip, seed=int(case[-1])).save(path)
    ds_yaml = _filled_dataset_yaml(root, dataset)
    if model_cfg is not None:
        ds_yaml = _apply_dataset_transform_drops(
            _apply_dataset_reader_overrides(ds_yaml, model_cfg), model_cfg
        )
    ds = build_dataset(ds_yaml, device=None, num_workers=1)
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


@pytest.fixture(scope="module")
def isla_pipeline_outputs(tmp_path_factory):
    """The ISLA config's view of a dataset (its reader overrides and transform drops), per dataset."""
    cache = {}

    def get(dataset: str) -> dict[str, DomainMesh]:
        if dataset not in cache:
            cfg = _compose("isla_volume_aerozoo", dataset=dataset)
            cache[dataset] = _read_back(tmp_path_factory, dataset, cfg)
        return cache[dataset]

    return get


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
    out_dim = _out_dim((DATASET_FILES.get(dataset) or V2_FILES[dataset])[1])
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


@pytest.mark.parametrize("dataset", sorted(ISLA_VP_OVERRIDES))
def test_isla_config_declares_each_condition_s_fields_and_runs(isla_pipeline_outputs, dataset):
    """ISLA declares the four conditions by name, each with the prescribed values it alone carries;
    its view of the dataset has no role labels and no zero-filled copies (HarmonizeBoundaryFields
    dropped), and every declared field is the condition's own prescribed value."""
    pipeline_output = isla_pipeline_outputs(dataset)
    targets = DATASET_FILES[dataset][1]
    cfg = _compose(
        "isla_volume_aerozoo",
        "model.hidden=16",
        "model.n_layers=1",
        "model.n_slices=4",
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
        name: {k: v["rank"] for k, v in schema.items()}
        for name, schema in OmegaConf.to_container(m.boundary_schemas).items()
    } == ISLA_CONDITIONS
    for domain in pipeline_output.values():
        for name, bnd in domain.boundaries.items():
            present = set(bnd.cell_data.keys())
            assert not present & {f"bc_{role}" for role in ISLA_CONDITIONS}, name
            assert set(ISLA_CONDITIONS[name]) <= present, name
            ### no zero-filled copy of another condition's field: the inlet carries no
            ### outlet value, and the slip walls, which prescribe nothing, carry none
            assert not present & set(ISLA_CONDITIONS["outlet" if name == "inlet" else "inlet"]) or name == "no_slip"
            assert name != "slip" or not present & set(BC_FIELDS)
    inlet = pipeline_output["with_slip"].boundaries["inlet"]
    torch.testing.assert_close(
        inlet.cell_data["bc_velocity"],
        torch.tensor([1.0, 0.0, 0.0]).expand(inlet.n_cells, 3),
    )
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


@pytest.mark.parametrize("dataset", sorted(DATASET_FILES))
def test_isla_reads_every_boundary_whole_and_geotransolver_keeps_its_draw(dataset):
    """ISLA draws its boundary tokens inside the model, so its config lifts the reader's
    per-boundary cap (aerozoo_volume*) and drops the uniform twins' SubsampleMesh, the
    SDF passes only GeoTransolver reads and the harmonized role labels it does not read;
    GeoTransolver's pipeline is the file's.
    Both reverse the four boundaries after the SDF passes."""
    ds = load_dataset_config(_RECIPE_ROOT / "datasets" / f"{dataset}.yaml")
    targets = lambda d: [str(t.get("_target_")) for t in d.pipeline.transforms]  # noqa: E731
    gt_cfg, isla_cfg = _compose("geotransolver_volume_aerozoo", dataset=dataset), _compose(
        "isla_volume_aerozoo", dataset=dataset
    )
    gt = _apply_dataset_transform_drops(_apply_dataset_reader_overrides(ds, gt_cfg), gt_cfg)
    isla = _apply_dataset_transform_drops(_apply_dataset_reader_overrides(ds, isla_cfg), isla_cfg)
    assert targets(gt) == targets(ds)
    capped = gt.pipeline.reader.get("subsample_n_cells") == 50000 or targets(gt)[0].endswith(".SubsampleMesh")
    assert capped
    assert isla.pipeline.reader.get("subsample_n_cells") is None
    chain = targets(isla)
    dropped = (".SubsampleMesh", ".ComputeSDFFromBoundary", ".HarmonizeBoundaryFields")
    assert not any(t.endswith(dropped) for t in chain), chain
    assert [t for t in targets(ds) if t not in chain] == [t for t in targets(ds) if t.endswith(dropped)]
    flips = [t for t in ds.pipeline.transforms if str(t["_target_"]).endswith(".FlipBoundaryOrientation")]
    assert [list(t.boundaries) for t in flips] == [["inlet", "outlet", "no_slip", "slip"]]
    order = targets(ds)
    assert max(i for i, t in enumerate(order) if t.endswith(".ComputeSDFFromBoundary")) < order.index(
        str(flips[0]["_target_"])
    )


def test_the_pipeline_reverses_the_four_boundaries(pipeline_output):
    """The release's boundaries are wound with their normals out of the fluid; the
    pipeline output's normals are the stored ones reversed, face for face."""
    for case, with_slip, seed in (("with_slip", True, 1), ("no_slip_walls", False, 2)):
        stored = make_aerozoo_case(with_slip, seed=seed)
        for name in ("inlet", "outlet", "no_slip", "slip"):
            out, src = pipeline_output[case].boundaries[name], stored.boundaries[name]
            assert out.n_cells == src.n_cells
            torch.testing.assert_close(out.cell_normals, -src.cell_normals)


### ---------------------------------------------------------------------------
### AeroZoo v2: physical inputs (datasets/aerozoo_puvolume*.yaml)
### ---------------------------------------------------------------------------


def _chain(dataset: str, model_cfg=None) -> list:
    """The dataset file's transforms (with ``model_cfg``'s drops), instantiated; the reader's draw is left out."""
    ds = OmegaConf.merge(
        load_dataset_config(_RECIPE_ROOT / "datasets" / f"{dataset}.yaml"),
        {
            "dataset_paths": {"aerozoo": "/unused", "aerozoo_manifest": "/unused"},
            "sampling_resolution": SAMPLING,
        },
    )
    if model_cfg is not None:
        ds = _apply_dataset_transform_drops(ds, model_cfg)
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
            k2: v2
            for k, v in value.items()
            for k2, v2 in _flat(v, f"{prefix}/{k}").items()
        }
    if hasattr(value, "keys"):  # TensorDict
        return {
            f"{prefix}/{k}": value[k]
            for k in value.keys(include_nested=True, leaves_only=True)
        }
    raise TypeError(type(value))


@pytest.mark.parametrize("dataset", sorted(V2_FILES))
def test_v2_dataset_files_read_no_label_and_derive_no_case_scale(dataset):
    """Every v2 file is the template's chain with its split's statistics: no transform reads a label or a
    reference scale, nothing is nondimensionalized or centred, and the constants are the split's (targets
    scale-only, mean 0); the label_reference block names the label-era file of the same split."""
    split, targets, twin = V2_FILES[dataset]
    path = _RECIPE_ROOT / "datasets" / f"{dataset}.yaml"
    assert path.read_text().startswith(
        f"# Statistics filled by aerozoo_pstats.py fill from pstats_{split}.train.json: split part {split}.train, "
    )
    ds = OmegaConf.load(path)
    chain = OmegaConf.to_container(ds.pipeline.transforms, resolve=False)
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
    for stats in chain[8]["fields"].values():
        assert stats["mean"] in (0, [0, 0, 0])
    assert OmegaConf.to_container(ds.targets) == targets
    assert ds.label_reference.dataset == twin
    twin_text = (_RECIPE_ROOT / "datasets" / f"{twin}.yaml").read_text()
    assert f"split part {split}.train" in twin_text.splitlines()[0]
    for t in chain:
        hydra.utils.instantiate(t, _convert_="all")


@pytest.mark.parametrize("with_slip", [True, False])
def test_v2_pipeline_is_physical_and_merges_the_faces(with_slip):
    """SI values throughout: raw coordinates, the prescribed values in m/s and the case's own units on the
    four boundaries, nu_std from per-dataset constants; the merged boundary is every face in the listed
    order with its one-hot condition and standardized values, and no label."""
    case = _labelled_case(with_slip)
    chain = _chain("aerozoo_puvolume")
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


def test_v2_geotransolver_option_b_and_transolver_configs_run():
    """GeoTransolver reads every boundary face as a token [xyz, one-hot (4), the prescribed values (15)] with
    the ball query off, the query point with its no_slip SDF and normal, and nu_std; Transolver the query
    point with the no_slip SDF and normal, and nu_std broadcast. Both set the per-case relative loss."""
    targets = TARGETS
    d = _run(_chain("aerozoo_puvolume"), _labelled_case(False))
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
        cfg = _compose(model, dataset="aerozoo_puvolume")
        assert cfg.training.loss_type == "case_relative_l2"
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
        total, _ = LossCalculator(targets, loss_type=cfg.training.loss_type)(
            pred, batch["targets"]
        )
        assert torch.isfinite(total)


@pytest.mark.parametrize("with_slip", [True, False])
def test_v2_every_model_input_is_blind_to_every_label(with_slip):
    """v2/check_v2.py B1 and B2 on the synthetic case: with every label perturbed, and with every label
    deleted, the pipeline runs and every non-label leaf it hands over (the baselines' view and ISLA's, which
    drops the draw, the SDF, the harmonized fields and the merge) is bitwise equal, and so is every tensor of
    the GeoTransolver and Transolver batches."""
    case = _labelled_case(with_slip)
    isla_cfg = _compose("isla_volume_aerozoo", dataset="aerozoo_puvolume")
    views = {
        "baselines": _chain("aerozoo_puvolume"),
        "isla": _chain("aerozoo_puvolume", isla_cfg),
    }
    assert not any(type(t).__name__ == "MergeBoundaries" for t in views["isla"])
    specs = {
        m: OmegaConf.to_container(
            _compose(m, dataset="aerozoo_puvolume").forward_kwargs, resolve=True
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


@pytest.mark.parametrize("with_slip", [True, False])
def test_v2_label_reference_map_is_exact(with_slip):
    """Test D: the v2 targets, de-standardized to SI units and mapped with the case's labels into the
    label-era space of aerozoo_uvolume.yaml, are that file's targets for the same case to float32 roundoff;
    the label-reference metrics of a prediction are the plain metrics in that space."""
    case = _labelled_case(with_slip)
    v2_chain, old_chain = _chain("aerozoo_puvolume"), _chain("aerozoo_uvolume")
    d_v2, d_old = _run(v2_chain, case), _run(old_chain, case)
    block = OmegaConf.to_container(
        load_dataset_config(
            _RECIPE_ROOT / "datasets" / "aerozoo_puvolume.yaml"
        ).label_reference
    )
    normalizer = next(t for t in v2_chain if type(t).__name__ == "NormalizeMeshFields")
    spec = {"embedding": "interior.points"}
    batch = build_collate_fn("tensors", spec, TARGETS, score_globals=block["globals"])(
        [(d_v2, {})]
    )
    expected = build_collate_fn("tensors", spec, TARGETS)([(d_old, {})])["targets"]
    assert set(batch["score_globals"].keys()) == {"U_inf", "rho_inf", "L_ref", "p_inf"}
    label_map = LabelReferenceMap.from_block(block, normalizer)
    mapped = label_map(batch["targets"], batch["score_globals"])
    for name in TARGETS:
        scale = expected[name].abs().max()
        torch.testing.assert_close(
            mapped[name], expected[name], rtol=0, atol=2e-6 * float(scale)
        )
    ### the metric calculator of score_space=label_reference
    calc = LabelReferenceMetricCalculator(TARGETS, metrics=["l2"], label_map=label_map)
    pred = batch["targets"].apply(lambda x: x * 1.1 + 0.01)
    out = calc(pred, batch["targets"], score_globals=batch["score_globals"])
    assert set(out.keys()) == set(calc.expected_keys())
    plain = MetricCalculator(TARGETS, metrics=["l2"])(
        label_map(pred, batch["score_globals"]), mapped
    )
    for k in plain.keys():
        torch.testing.assert_close(out[f"{LABEL_REFERENCE_PREFIX}/{k}"], plain[k])
    with pytest.raises(ValueError, match="score_globals"):
        calc(pred, batch["targets"])
