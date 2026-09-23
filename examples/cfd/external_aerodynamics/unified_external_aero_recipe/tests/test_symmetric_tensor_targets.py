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

"""The ``fields`` output type and the ``symmetric_tensor`` target type (2026-09-22).

A ``fields`` model (ISLA) returns ``{name: tensor}`` keyed by the dataset's target
names, each tensor of its declared rank. A dataset that stores the components of a
stress tensor as separate scalar fields declares them as one tensor target::

    stress: {type: symmetric_tensor, components: {sig11: xx, sig22: yy, sig12: xy}}

The recipe reads the listed laboratory components off the predicted ``(B, N, 3, 3)``
tensor; the loss and the per-component metric names are exactly those of the same
components declared as scalars, so tensor-head lanes stay comparable with the
scalar-head lanes. A flat ``(B, N, C)`` model cannot carry a tensor target.
"""

from __future__ import annotations

import pytest
import torch
from collate import build_collate_fn
from forward_kwargs import extract_targets
from loss import LossCalculator
from metrics import MetricCalculator
from output_normalize import (
    explode_tensor_targets,
    normalize_output_to_tensordict,
    select_fields_by_target,
    split_concat_by_target,
)
from utils import (
    field_dim,
    field_kind,
    field_loss_channels,
    select_tensor_components,
    target_field_names,
)

from physicsnemo.experimental.models import ISLA
from physicsnemo.mesh import DomainMesh, Mesh

STRESS = {
    "type": "symmetric_tensor",
    "components": {"sig11": "xx", "sig22": "yy", "sig12": "xy"},
}
TENSOR_TARGETS = {"q": "scalar", "stress": STRESS, "displacement": "vector"}
SCALAR_TARGETS = {
    "q": "scalar",
    "sig11": "scalar",
    "sig22": "scalar",
    "sig12": "scalar",
    "displacement": "vector",
}


def _domain(n: int = 16) -> DomainMesh:
    g = torch.Generator().manual_seed(0)
    pd = {
        k: torch.randn(n, generator=g) for k in ("q", "sig11", "sig22", "sig12", "sdf")
    }
    pd["displacement"] = torch.randn(n, 3, generator=g)
    return DomainMesh(
        interior=Mesh(points=torch.randn(n, 3, generator=g), point_data=pd),
        boundaries={},
        global_data={},
    )


def _fields_output(n: int = 16) -> dict[str, torch.Tensor]:
    """A ``fields`` model output: q (1, N), a symmetric stress (1, N, 3, 3), a vector (1, N, 3)."""
    g = torch.Generator().manual_seed(1)
    t = torch.randn(1, n, 3, 3, generator=g)
    return {
        "q": torch.randn(1, n, generator=g),
        "stress": t + t.transpose(-1, -2),
        "displacement": torch.randn(1, n, 3, generator=g),
    }


def _as_scalar_output(out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The same predictions as the scalar-target config would see them."""
    s = out["stress"]
    return {
        "q": out["q"],
        "sig11": s[..., 0, 0],
        "sig22": s[..., 1, 1],
        "sig12": s[..., 0, 1],
        "displacement": out["displacement"],
    }


def test_type_helpers():
    assert field_kind(STRESS) == "symmetric_tensor"
    assert field_loss_channels(STRESS) == 3
    with pytest.raises(ValueError, match="no flat channel layout"):
        field_dim(STRESS)
    assert target_field_names(TENSOR_TARGETS) == [
        "q",
        "sig11",
        "sig22",
        "sig12",
        "displacement",
    ]
    assert target_field_names(SCALAR_TARGETS) == list(SCALAR_TARGETS)
    with pytest.raises(ValueError):
        field_kind({"type": "tensor"})
    with pytest.raises(ValueError):
        field_kind({"type": "symmetric_tensor", "components": {"a": "xx", "b": "xx"}})
    with pytest.raises(ValueError):
        field_kind({"type": "symmetric_tensor", "components": {"a": "rr"}})
    t = _fields_output()["stress"]
    assert torch.equal(
        select_tensor_components(t, STRESS),
        torch.stack([t[..., 0, 0], t[..., 1, 1], t[..., 0, 1]], -1),
    )


def test_extract_and_select_agree_on_layout():
    domain = _domain()
    targets = extract_targets(domain, TENSOR_TARGETS)
    pd = domain.interior.point_data
    assert torch.equal(
        targets["stress"], torch.stack([pd["sig11"], pd["sig22"], pd["sig12"]], -1)
    )
    assert torch.equal(targets["q"], pd["q"]) and torch.equal(
        targets["displacement"], pd["displacement"]
    )
    assert list(targets.keys()) == ["q", "stress", "displacement"]
    out = _fields_output()
    pred = select_fields_by_target(out, TENSOR_TARGETS)
    assert pred["stress"].shape == (1, 16, 3) and pred["q"].shape == (1, 16)
    assert torch.equal(pred["stress"], out["stress"][..., [0, 1, 0], [0, 1, 1]])
    assert torch.equal(
        normalize_output_to_tensordict(out, TENSOR_TARGETS, "fields")["q"], out["q"]
    )
    with pytest.raises(KeyError, match="missing target fields"):
        select_fields_by_target({"q": out["q"]}, TENSOR_TARGETS)
    with pytest.raises(ValueError, match="must have 4 dims"):
        select_fields_by_target({**out, "stress": out["displacement"]}, TENSOR_TARGETS)
    with pytest.raises(TypeError):
        normalize_output_to_tensordict(torch.zeros(1, 16, 4), TENSOR_TARGETS, "fields")
    ### A flat model has no layout for a tensor target.
    with pytest.raises(ValueError, match="no flat channel layout"):
        split_concat_by_target(torch.zeros(1, 16, 10), TENSOR_TARGETS)


@pytest.mark.parametrize("loss_type", ["huber", "mse", "relative_mse"])
def test_loss_equals_the_same_components_as_scalars(loss_type):
    """One tensor target and three scalar targets give the same per-field and total loss."""
    domain = _domain()
    out = _fields_output()
    t_pred = select_fields_by_target(out, TENSOR_TARGETS)
    t_true = extract_targets(domain, TENSOR_TARGETS).unsqueeze(0)
    s_pred = select_fields_by_target(_as_scalar_output(out), SCALAR_TARGETS)
    s_true = extract_targets(domain, SCALAR_TARGETS).unsqueeze(0)
    t_total, t_td = LossCalculator(TENSOR_TARGETS, loss_type=loss_type)(t_pred, t_true)
    s_total, s_td = LossCalculator(SCALAR_TARGETS, loss_type=loss_type)(s_pred, s_true)
    assert torch.allclose(t_total, s_total, atol=1e-6)
    stress = t_td["loss/stress"]
    three = s_td["loss/sig11"] + s_td["loss/sig22"] + s_td["loss/sig12"]
    assert torch.allclose(stress, three, atol=1e-6)


def test_metrics_keep_the_component_names():
    domain = _domain()
    out = _fields_output()
    t_pred = select_fields_by_target(out, TENSOR_TARGETS)
    t_true = extract_targets(domain, TENSOR_TARGETS).unsqueeze(0)
    s_pred = select_fields_by_target(_as_scalar_output(out), SCALAR_TARGETS)
    s_true = extract_targets(domain, SCALAR_TARGETS).unsqueeze(0)
    calc = MetricCalculator(TENSOR_TARGETS, metrics=["l2", "l1"])
    t_m = calc(t_pred, t_true)
    s_m = MetricCalculator(SCALAR_TARGETS, metrics=["l2", "l1"])(s_pred, s_true)
    assert set(t_m.keys()) == set(calc.expected_keys())
    for k in (
        "sig11_l2",
        "sig22_l2",
        "sig12_l2",
        "sig11_l1",
        "q_l2",
        "displacement_l2",
    ):
        assert torch.allclose(t_m[k], s_m[k]), k
    assert "stress_l2" in t_m.keys()
    ### A measure-weighted call has the same key set.
    w = torch.rand(1, 16) + 0.5
    assert set(calc(t_pred, t_true, w).keys()) == set(calc.expected_keys())


def test_explode_tensor_targets_round_trips_to_component_fields():
    domain = _domain()
    t_true = extract_targets(domain, TENSOR_TARGETS)
    s_true = extract_targets(domain, SCALAR_TARGETS)
    exploded = explode_tensor_targets(t_true, TENSOR_TARGETS)
    assert list(exploded.keys()) == list(SCALAR_TARGETS)
    for k in SCALAR_TARGETS:
        assert torch.equal(exploded[k], s_true[k])
    assert explode_tensor_targets(s_true, SCALAR_TARGETS) is s_true


def test_isla_end_to_end_through_collate_and_loss():
    """The recipe's dict-valued forward_kwargs blocks reach ISLA's named inputs, and the
    named outputs reach the loss by target name, including a rank-1 boundary input and a
    symmetric-tensor target read off the rank-2 output."""
    g = torch.Generator().manual_seed(0)
    n = 24
    surface = Mesh(
        points=torch.randn(n * 2, 3, generator=g),
        cells=torch.randint(0, n * 2, (n, 3), generator=g),
        cell_data={
            "normals": torch.nn.functional.normalize(
                torch.randn(n, 3, generator=g), dim=-1
            ),
            "measure": torch.rand(n, generator=g) + 0.5,
            "traction": torch.randn(n, 3, generator=g),
            "fixed": torch.rand(n, generator=g),
        },
    )
    pd = {k: torch.randn(n, generator=g) for k in ("q", "sig11", "sig22", "sig12")}
    pd["displacement"] = torch.randn(n, 3, generator=g)
    domain = DomainMesh(
        interior=Mesh(points=surface.cell_centroids, point_data=pd),
        boundaries={"outer": surface},
        global_data={"load_dir": torch.tensor([0.0, 1.0, 0.0])},
    )
    spec = {
        "points": "boundaries.outer.cell_centroids",
        "normals": "boundaries.outer.cell_data.normals",
        "measure_weights": "boundaries.outer.cell_data.measure",
        "global_inputs": {"load_dir": "global_data.load_dir"},
        "boundary_inputs": {
            "traction": "boundaries.outer.cell_data.traction",
            "fixed": "boundaries.outer.cell_data.fixed",
        },
    }
    batch = build_collate_fn("tensors", spec, TENSOR_TARGETS)([(domain, {})])
    fk = batch["forward_kwargs"]
    assert fk["boundary_inputs"]["traction"].shape == (1, n, 3)
    model = ISLA(
        outputs={"q": 0, "stress": {"rank": 2, "symmetric": True}, "displacement": 1},
        global_inputs={"load_dir": 1},
        boundary_inputs={"traction": 1, "fixed": 0},
        hidden=16,
        n_layers=1,
        n_slices=4,
        geo_kernel="eager",
    )
    with torch.no_grad():
        out = model(**fk)
    assert out["stress"].shape == (1, n, 3, 3) and out["q"].shape == (1, n)
    pred = normalize_output_to_tensordict(out, TENSOR_TARGETS, "fields")
    assert pred["stress"].shape == (1, n, 3)
    total, td = LossCalculator(TENSOR_TARGETS, loss_type="mse")(pred, batch["targets"])
    assert torch.isfinite(total) and set(td.keys()) >= {
        "loss/q",
        "loss/stress",
        "loss/displacement",
    }
