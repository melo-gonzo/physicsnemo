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

"""AeroZoo v1.2, derived-globals arm (datasets/aerozoo_uvolume_casescale.yaml): ISLA's per-case scales in the
datapipe. The scales are ISLA's (bitwise its L, U and re_z); a case rescaled in length, speed and viscosity at the
same Reynolds number gives every model the same inputs and targets; every derived-globals config predicts the seven
fields; and inference's re-dimensionalization returns the physical targets."""

import copy

import hydra
import pytest
import torch
from infer import build_redim_field_types, redimensionalize
from nondim import NonDimensionalizeByMetadata
from omegaconf import OmegaConf
from output_normalize import normalize_output_to_tensordict
from test_aerozoo_configs import (
    _RECIPE_ROOT,
    TARGETS,
    _chain,
    _compose,
    _labelled_case,
    _run,
    make_aerozoo_case,
)
from test_aerozoo_geometry_configs import BC_CONFIGS, GEOMETRY_CONFIGS, _tensors

from datasets import load_dataset_config
from physicsnemo.datapipes.transforms.mesh import NormalizeMeshFields
from physicsnemo.mesh import DomainMesh, Mesh

DATASET = "aerozoo_uvolume_casescale"
#: Each derived-globals config and the main-arm config whose CPU-test knobs it takes.
CONFIGS = {f"{m}_casescale": m for m in [*GEOMETRY_CONFIGS, *BC_CONFIGS]}
KNOBS = {**GEOMETRY_CONFIGS, **BC_CONFIGS}


def _batch(cfg, case: DomainMesh, dataset: str = DATASET) -> dict:
    from collate import build_collate_fn

    spec = OmegaConf.to_container(cfg.forward_kwargs, resolve=True)
    if "geometry_seed" in spec:
        gd = case.global_data.clone()
        gd["smart_query_seed"], gd["smart_geometry_seed"] = (
            torch.tensor(1),
            torch.tensor(2),
        )
        case = DomainMesh(
            interior=case.interior, boundaries=case.boundaries, global_data=gd
        )
    d = _run(_chain(dataset, cfg), case)
    return build_collate_fn(cfg.input_type, spec, TARGETS)([(d, {})])


def _similar(d: DomainMesh, s: float, t: float) -> DomainMesh:
    """The same flow at the same Reynolds number: lengths times s, velocities times t, the viscosities times s t,
    pressure times t^2 (density fixed), k times t^2, omega times t / s."""
    per_unit = {
        "velocity": t,
        "gauge_pressure": t * t,
        "turbulent_kinematic_viscosity": s * t,
        "turbulent_kinetic_energy": t * t,
        "specific_dissipation_rate": t / s,
    }

    def scaled(td):
        td = td.clone()
        for key in td.keys(include_nested=True, leaves_only=True):
            name = key[-1] if isinstance(key, tuple) else key
            if name in per_unit and key[0] in ("solution", "prescribed"):
                td[key] = td[key] * per_unit[name]
        return td

    def mesh(m: Mesh) -> Mesh:
        return Mesh(
            points=m.points * s,
            cells=m.cells,
            point_data=scaled(m.point_data),
            cell_data=scaled(m.cell_data),
        )

    gd = d.global_data.clone()
    gd["physics", "kinematic_viscosity"] = gd["physics", "kinematic_viscosity"] * (
        s * t
    )
    return DomainMesh(
        interior=mesh(d.interior),
        boundaries={n: mesh(d.boundaries[n]) for n in d.boundary_names},
        global_data=gd,
    )


#: The steps the derived-inputs file runs before the main file's chain, in order.
SCALE_STEPS = [
    "ComputeBoundaryScales",
    "ComputeLogReynolds",
    "StandardizeGlobalLeaf",
    "ScalesAsFreestream",
    "NonDimensionalizeByMetadata",
    "NonDimensionalizeByMetadata",
]


def test_the_shared_part_is_the_main_files():
    """datasets/aerozoo_uvolume_casescale.yaml is aerozoo_uvolume.yaml with the scale steps first, the main file's
    nu_std step dropped, and its own standardization scales. Set those aside and every other key and step is the main
    file's, so a revision of the main file fails here until it reaches this file too."""

    def load(name):
        path = _RECIPE_ROOT / "datasets" / f"{name}.yaml"
        return OmegaConf.to_container(OmegaConf.load(path), resolve=False)

    def name(step):
        return step["_target_"].split(":")[-1].rstrip("}")

    def unscaled(step):
        """The step without the standardization scales this arm refits."""
        step = copy.deepcopy(step)
        key = {"NormalizeMeshFields": "fields", "MergeBoundaries": "standardize"}.get(
            name(step)
        )
        for stats in step[key].values() if key else ():
            del stats["std"]
        return step

    ours, main = load(DATASET), load("aerozoo_uvolume")
    ours_steps, main_steps = (
        ours["pipeline"].pop("transforms"),
        main["pipeline"].pop("transforms"),
    )
    assert [name(t) for t in ours_steps[: len(SCALE_STEPS)]] == SCALE_STEPS
    (nu_std,) = [t for t in main_steps if name(t) == "StandardizeGlobalLeaf"]
    assert nu_std["output_field"] == "nu_std"
    assert [unscaled(t) for t in ours_steps[len(SCALE_STEPS) :]] == [
        unscaled(t) for t in main_steps if t is not nu_std
    ]
    assert ours == main


def test_the_scales_are_islas():
    """L, U and re_z are the ones ISLA's own config computes on the same boundaries, and U_inf = U g."""
    case = make_aerozoo_case(True)
    d = _run(_chain(DATASET)[:4], case)
    isla = _run(_chain("aerozoo_uvolume", _compose("isla_volume_aerozoo")), case)
    for ours, theirs in (("L", "L"), ("U", "U"), ("re_z_float64", "re_z")):
        assert torch.equal(d.global_data[ours], isla.global_data[theirs]), ours
    assert d.global_data["re_z"].dtype == torch.float32
    assert torch.allclose(
        d.global_data["re_z"].double(), isla.global_data["re_z"], rtol=1e-6
    )
    assert torch.allclose(
        d.global_data["g"], torch.tensor([1.0, 0, 0], dtype=torch.float64)
    )
    assert torch.equal(d.global_data["U_inf"], d.global_data["U"] * d.global_data["g"])
    assert d.global_data["L_ref"] == d.global_data["L"] and d.global_data["p_inf"] == 0


def test_a_similar_case_gives_the_same_inputs_and_targets():
    """Every model input and target is unchanged, to float32 rounding, when the case is rescaled at fixed
    Reynolds number; the main arm's inputs change (control)."""
    case = _labelled_case(True)
    other = _similar(case, 2.5, 0.4)
    for model in CONFIGS:
        cfg = _compose(model, dataset=DATASET)
        a, b = (_batch(cfg, x) for x in (case, other))
        if cfg.input_type == "mesh":
            ### A mesh model is handed whole meshes but reads only its declared leaves (GLOBE: the no-slip
            ### geometry and re_z), so the check is on its prediction.
            torch.manual_seed(0)
            net = hydra.utils.instantiate(
                OmegaConf.merge(cfg.model, KNOBS[CONFIGS[model]]), _convert_="partial"
            )
            with torch.no_grad():
                a["forward_kwargs"], b["forward_kwargs"] = (
                    normalize_output_to_tensordict(
                        net(**x["forward_kwargs"]), TARGETS, cfg.output_type
                    )
                    for x in (a, b)
                )
        for part in ("forward_kwargs", "targets"):
            ta, tb = _tensors(a[part]), _tensors(b[part])
            assert ta.keys() == tb.keys()
            for k in ta:
                if ta[k].is_floating_point():
                    assert torch.allclose(ta[k], tb[k], rtol=1e-4, atol=1e-5), (
                        model,
                        part,
                        k,
                    )
                else:
                    assert torch.equal(ta[k], tb[k]), (model, part, k)
    from test_aerozoo_geometry_configs import _batch as main_batch

    cfg = _compose("geotransolver_volume_aerozoo_local")
    a, b = (_tensors(main_batch(cfg, x)["forward_kwargs"]) for x in (case, other))
    assert not torch.allclose(a["/geometry"], b["/geometry"])


@pytest.mark.parametrize("model", sorted(CONFIGS))
def test_casescale_config_predicts_the_seven_fields(model):
    """The config composes with the protocol loss and re_z as its global input, and predicts every target."""
    cfg = _compose(model, dataset=DATASET)
    parent = _compose(CONFIGS[model])
    assert cfg.training == parent.training
    reads = str(OmegaConf.to_container(cfg.forward_kwargs)) + str(
        OmegaConf.to_container(cfg.model)
    )
    assert "re_z" in reads and "nu_std" not in reads
    ### Nothing but the global input differs from the parent's model and inputs.
    swap = lambda c: str(OmegaConf.to_container(c)).replace("nu_std", "re_z")
    assert swap(parent.model) == str(OmegaConf.to_container(cfg.model))
    assert swap(parent.forward_kwargs) == str(
        OmegaConf.to_container(cfg.forward_kwargs)
    )
    batch = _batch(cfg, _labelled_case(False))
    net = hydra.utils.instantiate(
        OmegaConf.merge(cfg.model, KNOBS[CONFIGS[model]]), _convert_="partial"
    )
    with torch.no_grad():
        out = net(**batch["forward_kwargs"])
    pred = normalize_output_to_tensordict(out, TARGETS, cfg.output_type)
    assert set(pred.keys()) == set(TARGETS)
    assert all(torch.isfinite(v).all() for v in pred.values())


def test_redimensionalization_returns_the_physical_targets():
    """The normalized targets, re-dimensionalized as inference does, are the case's physical fields."""
    case = make_aerozoo_case(True)
    chain = _chain(DATASET)
    d = _run(chain, case)
    ds = load_dataset_config(_RECIPE_ROOT / "datasets" / f"{DATASET}.yaml")
    types = build_redim_field_types(ds)
    assert {k: types[k] for k in ("velocity", "pressure", "nut", "tke", "omega")} == {
        "velocity": "velocity",
        "pressure": "pressure",
        "nut": "kinematic_viscosity",
        "tke": "turbulent_kinetic_energy",
        "omega": "specific_dissipation_rate",
    }
    normalizer = next(t for t in chain if isinstance(t, NormalizeMeshFields))
    pred = d.interior.point_data.select(*TARGETS)
    phys = redimensionalize(
        pred,
        normalizer=normalizer,
        nondim=NonDimensionalizeByMetadata(fields=types),
        field_types=types,
        global_data=d.global_data,
    )
    ### The points the chain kept, matched by position (the chain divides them by L).
    pts = d.interior.points.double() * d.global_data["L"]
    idx = torch.cdist(pts, case.interior.points.double()).argmin(-1)
    sol = case.interior.point_data["solution"]
    names = {
        "velocity": "velocity",
        "pressure": "gauge_pressure",
        "nut": "turbulent_kinematic_viscosity",
        "tke": "turbulent_kinetic_energy",
        "omega": "specific_dissipation_rate",
        "intermittency": "intermittency",
        "retheta": "momentum_thickness_reynolds_number",
    }
    for ours, theirs in names.items():
        assert torch.allclose(phys[ours], sol[theirs][idx], rtol=1e-4, atol=1e-6), ours
