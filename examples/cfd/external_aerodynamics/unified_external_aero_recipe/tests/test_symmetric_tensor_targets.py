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

"""The ``symmetric_tensor`` target type (TENSOR-HEAD, 2026-09-22).

A dataset stores the components of a stress tensor as separate scalar fields; the
target block declares them as one tensor::

    stress: {type: symmetric_tensor, components: {sig11: xx, sig22: yy, sig12: xy}}

The model emits six components (``xx, yy, zz, xy, xz, yz``); the recipe reads the
listed ones, and the loss and the per-component metric names are exactly those of the
same components declared as scalars, so tensor-head lanes stay comparable with the
scalar-head lanes.
"""

from __future__ import annotations

import pytest
import torch
from forward_kwargs import extract_targets
from loss import LossCalculator
from metrics import MetricCalculator
from output_normalize import explode_tensor_targets, split_concat_by_target
from utils import (
    field_dim,
    field_kind,
    field_loss_channels,
    target_field_names,
    tensor_component_indices,
)

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


def _model_output(n: int = 16) -> torch.Tensor:
    """A (1, N, 1 + 6 + 3) row in the tensor layout: q, then xx..yz, then the vector."""
    return torch.randn(1, n, 1 + 6 + 3, generator=torch.Generator().manual_seed(1))


def _as_scalar_layout(out: torch.Tensor) -> torch.Tensor:
    """The same predictions re-laid out for the scalar-target config: q, sig11, sig22, sig12, u."""
    idx = tensor_component_indices(STRESS)
    return torch.cat([out[..., :1], out[..., 1:7][..., idx], out[..., 7:]], dim=-1)


def test_type_helpers():
    assert field_kind(STRESS) == "symmetric_tensor"
    assert field_dim(STRESS) == 6 and field_loss_channels(STRESS) == 3
    assert tensor_component_indices(STRESS) == [0, 1, 3]
    assert target_field_names(TENSOR_TARGETS) == ["q", "sig11", "sig22", "sig12", "displacement"]
    assert target_field_names(SCALAR_TARGETS) == list(SCALAR_TARGETS)
    with pytest.raises(ValueError):
        field_kind({"type": "tensor"})
    with pytest.raises(ValueError):
        field_kind({"type": "symmetric_tensor", "components": {"a": "xx", "b": "xx"}})
    with pytest.raises(ValueError):
        field_kind({"type": "symmetric_tensor", "components": {"a": "rr"}})


def test_extract_and_split_agree_on_layout():
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
    pred = split_concat_by_target(_model_output(), TENSOR_TARGETS)
    assert pred["stress"].shape == (1, 16, 3) and pred["q"].shape == (1, 16)
    ### xx, yy, xy of the six-channel block, in the YAML's component order.
    out = _model_output()
    assert torch.equal(pred["stress"], out[..., [1, 2, 4]])
    with pytest.raises(ValueError):
        split_concat_by_target(out[..., :-1], TENSOR_TARGETS)


@pytest.mark.parametrize("loss_type", ["huber", "mse", "relative_mse"])
def test_loss_equals_the_same_components_as_scalars(loss_type):
    """One tensor target and three scalar targets give the same per-field and total loss."""
    domain = _domain()
    out = _model_output()
    t_pred = split_concat_by_target(out, TENSOR_TARGETS)
    t_true = extract_targets(domain, TENSOR_TARGETS).unsqueeze(0)
    s_pred = split_concat_by_target(_as_scalar_layout(out), SCALAR_TARGETS)
    s_true = extract_targets(domain, SCALAR_TARGETS).unsqueeze(0)
    t_total, t_td = LossCalculator(TENSOR_TARGETS, loss_type=loss_type)(t_pred, t_true)
    s_total, s_td = LossCalculator(SCALAR_TARGETS, loss_type=loss_type)(s_pred, s_true)
    assert torch.allclose(t_total, s_total, atol=1e-6)
    stress = t_td["loss/stress"]
    three = s_td["loss/sig11"] + s_td["loss/sig22"] + s_td["loss/sig12"]
    assert torch.allclose(stress, three, atol=1e-6)


def test_metrics_keep_the_component_names():
    domain = _domain()
    out = _model_output()
    t_pred = split_concat_by_target(out, TENSOR_TARGETS)
    t_true = extract_targets(domain, TENSOR_TARGETS).unsqueeze(0)
    s_pred = split_concat_by_target(_as_scalar_layout(out), SCALAR_TARGETS)
    s_true = extract_targets(domain, SCALAR_TARGETS).unsqueeze(0)
    calc = MetricCalculator(TENSOR_TARGETS, metrics=["l2", "l1"])
    t_m = calc(t_pred, t_true)
    s_m = MetricCalculator(SCALAR_TARGETS, metrics=["l2", "l1"])(s_pred, s_true)
    assert set(t_m.keys()) == set(calc.expected_keys())
    for k in ("sig11_l2", "sig22_l2", "sig12_l2", "sig11_l1", "q_l2", "displacement_l2"):
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
