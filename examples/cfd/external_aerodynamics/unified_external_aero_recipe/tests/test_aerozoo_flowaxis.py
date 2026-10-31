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


"""AeroZoo v1.2, derived globals with the flow axis (datasets/aerozoo_uvolume_flowaxis.yaml and the *_flowaxis
model configs). ComputeFlowAxis takes its hand values, inlets weighted by volume flow rate and outlets by area,
turns with the case, does not depend on the normals' orientation and counts an inlet face's outflow against the
inflow; the dataset
file is the derived-globals file plus that one step; and each flow-axis config differs from its parent only in its
global input, and predicts the seven fields."""

import hydra
import pytest
import torch
from omegaconf import OmegaConf
from output_normalize import (
    model_dataset_constants,
    model_output_stats,
    normalize_output_to_tensordict,
)
from test_aerozoo_casescale import _batch
from test_aerozoo_configs import (
    _RECIPE_ROOT,
    TARGETS,
    _chain,
    _compose,
    _labelled_case,
    _run,
)
from test_aerozoo_geometry_configs import BC_CONFIGS, GEOMETRY_CONFIGS

from collate import build_collate_fn
from datasets import load_dataset_config
from domain_transforms import ComputeFlowAxis
from physicsnemo.mesh import DomainMesh, Mesh

DATASET = "aerozoo_uvolume_flowaxis"
#: Each flow-axis config, its derived-globals parent and the main-arm config whose CPU-test knobs it takes.
CONFIGS = {f"{m}_flowaxis": m for m in [*GEOMETRY_CONFIGS, *BC_CONFIGS]}
KNOBS = {**GEOMETRY_CONFIGS, **BC_CONFIGS}
#: The one model setting each config changes: the width of its global input, re_z (1) + flow_axis (3).
WIDTHS = {"functional_dim", "global_dim", "parameter_channels"}


def _square(axis: int, at: float, u, flip: bool = False) -> Mesh:
    """The unit square in the plane x_axis = at as two triangles, normal +x_axis, the velocity u on both; flip
    reverses their winding."""
    pts = torch.zeros(4, 3)
    pts[:, axis] = at
    pts[:, [(axis + 1) % 3, (axis + 2) % 3]] = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    )
    cells = torch.tensor([[0, 1, 2], [0, 2, 3]])
    return Mesh(
        points=pts,
        cells=cells.flip(-1) if flip else cells,
        cell_data={"u": torch.tensor(u, dtype=torch.float32).expand(2, 3).clone()},
    )


def _join(*meshes: Mesh) -> Mesh:
    """One mesh of several (one boundary made of patches, as a dual-inlet box's inlet)."""
    offsets = torch.tensor([0, *[m.n_points for m in meshes]]).cumsum(0)
    return Mesh(
        points=torch.cat([m.points for m in meshes]),
        cells=torch.cat([m.cells + o for m, o in zip(meshes, offsets)]),
        cell_data={"u": torch.cat([m.cell_data["u"] for m in meshes])},
    )


def _domain(inlet: Mesh, outlet: Mesh) -> DomainMesh:
    return DomainMesh(
        interior=Mesh(points=torch.zeros(3, 3)),
        boundaries={"inlet": inlet, "outlet": outlet},
        global_data={},
    )


AXIS = ComputeFlowAxis(["inlet"], ["outlet"], "u", "flow_axis")


def _dual_inlet(flip: bool = False) -> DomainMesh:
    """A dual-inlet box: one inlet patch at x = 0 at speed 1 along +x, a second at y = 0 at speed 2 along +y,
    the outlet at x = 3."""
    inlet = _join(
        _square(0, 0.0, [1.0, 0, 0], flip), _square(1, 0.0, [0, 2.0, 0], flip)
    )
    return _domain(inlet, _square(0, 3.0, [0, 0, 0]))


def test_the_axis_takes_its_hand_values():
    """Opposed faces: the axis is the inflow direction. Two inlets: c_in weights each patch by its flow rate,
    1 and 2, so c_in = ((0, .5, .5) + 2 (.5, 0, .5)) / 3 and the axis is (c_out - c_in) / |c_out - c_in| with
    c_out = (3, .5, .5); an area weighting would give another axis. In the meshes' dtype, unit length."""
    opposed = AXIS.apply_to_domain(
        _domain(_square(0, 0.0, [2.0, 0, 0]), _square(0, 3.0, [0, 0, 0]))
    ).global_data["flow_axis"]
    torch.testing.assert_close(opposed, torch.tensor([1.0, 0.0, 0.0]))
    axis = AXIS.apply_to_domain(_dual_inlet()).global_data["flow_axis"]
    d = torch.tensor([3.0, 0.5, 0.5]) - torch.tensor([1.0, 0.5, 1.5]) / 3
    torch.testing.assert_close(axis, d / d.norm())
    assert axis.dtype == torch.float32
    by_area = torch.tensor([3.0, 0.5, 0.5]) - torch.tensor([0.25, 0.25, 0.5])
    assert not torch.allclose(axis, by_area / by_area.norm(), atol=1e-2)


def test_the_axis_turns_with_the_case_and_ignores_the_normals_orientation():
    """A rotation of the points and the velocities rotates the axis; reversing every face's winding (the
    normals into or out of the fluid, before or after FlipBoundaryOrientation) leaves it unchanged."""
    q, _ = torch.linalg.qr(
        torch.randn(3, 3, generator=torch.Generator().manual_seed(3))
    )
    q = q * torch.linalg.det(q).sign()
    d = _dual_inlet()
    turned = _domain(
        *(
            Mesh(
                points=m.points @ q.T,
                cells=m.cells,
                cell_data={"u": m.cell_data["u"] @ q.T},
            )
            for m in (d.boundaries["inlet"], d.boundaries["outlet"])
        )
    )
    axis = AXIS.apply_to_domain(d).global_data["flow_axis"]
    torch.testing.assert_close(
        AXIS.apply_to_domain(turned).global_data["flow_axis"], q @ axis
    )
    torch.testing.assert_close(
        AXIS.apply_to_domain(_dual_inlet(flip=True)).global_data["flow_axis"], axis
    )


def test_an_inlet_face_with_outflow_counts_against_the_inflow():
    """The rim of a spherical inlet cap: one face of the inlet square carries flow in at speed 2, the other out at
    speed 1, so their weights are 1 and -1/2 (area 1/2 each) and c_in = (c_0 - c_1 / 2) / (1/2) = (0, 1, 0) from
    the triangles' centroids (0, 2/3, 1/3) and (0, 1/3, 2/3)."""
    inlet = _square(0, 0.0, [2.0, 0, 0])
    inlet.cell_data["u"][1] = torch.tensor([-1.0, 0, 0])
    axis = AXIS.apply_to_domain(_domain(inlet, _square(0, 3.0, [0, 0, 0]))).global_data[
        "flow_axis"
    ]
    d = torch.tensor([3.0, 0.5, 0.5]) - torch.tensor([0.0, 1.0, 0.0])
    torch.testing.assert_close(axis, d / d.norm())


def test_the_axis_refuses_missing_inputs():
    """A missing boundary, an inlet without net flow, and an existing leaf raise."""
    with pytest.raises(ValueError, match="carry no flow"):
        AXIS.apply_to_domain(
            _domain(_square(0, 0.0, [0, 1.0, 0]), _square(0, 3.0, [0, 0, 0]))
        )
    with pytest.raises(KeyError, match="not found"):
        ComputeFlowAxis(["inlet"], ["exit"], "u", "flow_axis").apply_to_domain(
            _dual_inlet()
        )
    with pytest.raises(ValueError, match="already has"):
        AXIS.apply_to_domain(AXIS.apply_to_domain(_dual_inlet()))


def test_the_dataset_file_is_the_derived_file_plus_the_axis():
    """datasets/aerozoo_uvolume_flowaxis.yaml is aerozoo_uvolume_casescale.yaml with ComputeFlowAxis after
    ComputeBoundaryScales, on the boundaries as read and the inlet's prescribed velocity, and nothing else."""
    load = lambda n: OmegaConf.to_container(
        load_dataset_config(_RECIPE_ROOT / "datasets" / f"{n}.yaml")
    )
    ours, theirs = load(DATASET), load("aerozoo_uvolume_casescale")
    steps = ours["pipeline"].pop("transforms")
    base = theirs["pipeline"].pop("transforms")
    assert ours == theirs
    i = ["ComputeFlowAxis" in s["_target_"] for s in steps].index(True)
    assert "ComputeBoundaryScales" in base[i - 1]["_target_"]
    assert steps[:i] + steps[i + 1 :] == base
    assert {k: v for k, v in steps[i].items() if k != "_target_"} == {
        "inlets": ["inlet"],
        "outlets": ["outlet"],
        "velocity_field": "prescribed.dirichlet.velocity",
        "output_field": "flow_axis",
    }


def _flat(d, prefix=""):
    if isinstance(d, dict):
        return {
            k: v
            for key, val in d.items()
            for k, v in _flat(val, f"{prefix}{key}.").items()
        }
    return {prefix[:-1]: d}


def _oriented(d: DomainMesh) -> DomainMesh:
    """The synthetic case with its inlet and outlet faces wound consistently, normals out of the fluid (the
    release's convention); the shared fixture draws them at random."""
    out = {"inlet": torch.tensor([-1.0, 0, 0]), "outlet": torch.tensor([1.0, 0, 0])}
    bounds = {}
    for name in d.boundary_names:
        m = d.boundaries[name]
        if name in out:
            flip = (m.cell_normals @ out[name]) < 0
            m = Mesh(
                points=m.points,
                cells=torch.where(flip[:, None], m.cells.flip(-1), m.cells),
                point_data=m.point_data,
                cell_data=m.cell_data,
            )
        bounds[name] = m
    return DomainMesh(interior=d.interior, boundaries=bounds, global_data=d.global_data)


@pytest.mark.parametrize("model", sorted(CONFIGS))
def test_flowaxis_config_adds_the_axis_and_predicts_the_seven_fields(model):
    """Against its derived-globals parent the config changes only its global input, re_z to re_z and flow_axis
    (and the input's width by 3; GLOBE declares flow_axis as a rank-1 global), and predicts every target."""
    cfg = _compose(model, dataset=DATASET)
    parent = _compose(
        f"{CONFIGS[model]}_casescale", dataset="aerozoo_uvolume_casescale"
    )
    assert cfg.training == parent.training
    ours, theirs = (_flat(OmegaConf.to_container(c.model)) for c in (cfg, parent))
    changed = {k for k in ours.keys() | theirs.keys() if ours.get(k) != theirs.get(k)}
    assert changed <= WIDTHS | {"global_schema.flow_axis.rank"}, changed
    assert all(ours[k] == theirs[k] + 3 for k in changed & WIDTHS)
    swap = lambda c: str(OmegaConf.to_container(c)).replace(
        "'global_data.re_z'", "['global_data.re_z', 'global_data.flow_axis']"
    )
    assert swap(parent.forward_kwargs) == str(
        OmegaConf.to_container(cfg.forward_kwargs)
    )
    assert "flow_axis" in str(OmegaConf.to_container(cfg.forward_kwargs)) + str(ours)
    batch = _batch(cfg, _oriented(_labelled_case(False)), DATASET)
    net = hydra.utils.instantiate(
        OmegaConf.merge(cfg.model, KNOBS[CONFIGS[model]]), _convert_="partial"
    )
    with torch.no_grad():
        out = net(**batch["forward_kwargs"])
    pred = normalize_output_to_tensordict(out, TARGETS, cfg.output_type)
    assert set(pred.keys()) == set(TARGETS)
    assert all(torch.isfinite(v).all() for v in pred.values())


def test_isla_flowaxis_config_takes_the_axis_as_a_second_direction():
    """ISLA's flow-axis config appends ComputeFlowAxis to its parent's two appended steps on the meshes it
    receives (normals into the fluid), declares flow_axis as a global direction beside the inlet's g (K = 2),
    and runs; the axis equals the one the flow-axis dataset file computes on the stored orientation."""
    knobs = [
        "model.hidden=16",
        "model.n_layers=1",
        "model.n_slices=4",
        "model.compile_blocks=false",
        "model.n_boundary_tokens=64",
    ]
    cfg = _compose("isla_volume_aerozoo_flowaxis", *knobs, dataset="aerozoo_uvolume")
    parent = _compose("isla_volume_aerozoo", *knobs, dataset="aerozoo_uvolume")
    assert OmegaConf.to_container(cfg.dataset_append_transforms)[
        :2
    ] == OmegaConf.to_container(parent.dataset_append_transforms)
    ours, theirs = (_flat(OmegaConf.to_container(c.model)) for c in (cfg, parent))
    assert {k for k in ours.keys() | theirs.keys() if ours.get(k) != theirs.get(k)} == {
        "global_schema.flow_axis.rank"
    }
    case = _oriented(_labelled_case(False))
    chain = _chain("aerozoo_uvolume", cfg)
    d = _run(chain, case)
    torch.testing.assert_close(
        d.global_data["flow_axis"],
        _run(_chain(DATASET)[:2], case).global_data["flow_axis"],
    )
    ds = load_dataset_config(_RECIPE_ROOT / "datasets" / "aerozoo_uvolume.yaml")
    (normalizer,) = [t for t in chain if type(t).__name__ == "NormalizeMeshFields"]
    model = hydra.utils.instantiate(
        cfg.model,
        _convert_="partial",
        **model_output_stats(cfg.model, normalizer),
        **model_dataset_constants(cfg.model, ds),
    )
    assert (
        model.n_global_vectors == 2 and model.n_global_scalars == 1
    )  # flow_axis, g; re_z
    batch = build_collate_fn(
        "mesh", OmegaConf.to_container(cfg.forward_kwargs, resolve=True), TARGETS
    )([(d, {})])
    with torch.no_grad():
        out = model(**batch["forward_kwargs"])
    pred = normalize_output_to_tensordict(out, TARGETS, "mesh")
    assert set(pred.keys()) == set(TARGETS)
    assert all(torch.isfinite(v).all() for v in pred.values())
