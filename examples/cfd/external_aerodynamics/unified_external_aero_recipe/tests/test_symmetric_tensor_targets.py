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

"""The ``symmetric_tensor`` target type on the ``mesh`` output path (2026-09-22).

A mesh-output model (ISLA) returns a point-cloud ``Mesh`` whose ``point_data`` holds
one leaf per target name at its natural rank. A dataset that stores the components
of a stress tensor as separate scalar fields declares them as one tensor target::

    stress: {type: symmetric_tensor, components: {sig11: xx, sig22: yy, sig12: xy}}

The recipe reads the listed laboratory components off the predicted ``(N, 3, 3)``
leaf; the loss and the per-component metric names are exactly those of the same
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
    select_mesh_fields_by_target,
    split_concat_by_target,
)
from utils import (
    field_dim,
    field_kind,
    field_loss_channels,
    select_tensor_components,
    target_field_names,
)

from physicsnemo.experimental.models.isla import ISLA
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.calculus.measure import set_cell_measures

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


def _mesh_output(n: int = 16) -> Mesh:
    """A mesh-output model's Mesh: q (N,), a symmetric stress (N, 3, 3), a vector (N, 3)."""
    g = torch.Generator().manual_seed(1)
    t = torch.randn(n, 3, 3, generator=g)
    return Mesh(
        points=torch.randn(n, 3, generator=g),
        point_data={
            "q": torch.randn(n, generator=g),
            "stress": t + t.transpose(-1, -2),
            "displacement": torch.randn(n, 3, generator=g),
        },
    )


def _as_scalar_output(out: Mesh) -> Mesh:
    """The same predictions as the scalar-target config would see them."""
    s = out.point_data["stress"]
    return Mesh(
        points=out.points,
        point_data={
            "q": out.point_data["q"],
            "sig11": s[..., 0, 0],
            "sig22": s[..., 1, 1],
            "sig12": s[..., 0, 1],
            "displacement": out.point_data["displacement"],
        },
    )


def test_type_helpers():
    """field_kind, field_loss_channels, field_dim and target_field_names on a symmetric-tensor target."""
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
    t = _mesh_output().point_data["stress"]
    assert torch.equal(
        select_tensor_components(t, STRESS),
        torch.stack([t[..., 0, 0], t[..., 1, 1], t[..., 0, 1]], -1),
    )


def test_extract_and_select_agree_on_layout():
    """Targets and predictions lay a symmetric tensor out as its listed components, in target order."""
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
    out = _mesh_output()
    pred = select_mesh_fields_by_target(out, TENSOR_TARGETS)
    assert pred["stress"].shape == (16, 3) and pred["q"].shape == (16,)
    assert pred.batch_size == torch.Size([16])
    assert torch.equal(
        pred["stress"], out.point_data["stress"][..., [0, 1, 0], [0, 1, 1]]
    )
    assert torch.equal(
        normalize_output_to_tensordict(out, TENSOR_TARGETS, "mesh")["q"],
        out.point_data["q"],
    )
    with pytest.raises(KeyError, match="missing target fields"):
        select_mesh_fields_by_target(
            Mesh(points=out.points, point_data={"q": out.point_data["q"]}),
            TENSOR_TARGETS,
        )
    with pytest.raises(ValueError, match="must have 3 dims"):
        bad = Mesh(
            points=out.points,
            point_data={
                **dict(out.point_data.items()),
                "stress": out.point_data["displacement"],
            },
        )
        select_mesh_fields_by_target(bad, TENSOR_TARGETS)
    with pytest.raises(TypeError):
        normalize_output_to_tensordict(torch.zeros(16, 4), TENSOR_TARGETS, "mesh")
    ### A flat model has no layout for a tensor target.
    with pytest.raises(ValueError, match="no flat channel layout"):
        split_concat_by_target(torch.zeros(1, 16, 10), TENSOR_TARGETS)


@pytest.mark.parametrize("loss_type", ["huber", "mse", "relative_mse"])
def test_loss_equals_the_same_components_as_scalars(loss_type):
    """One tensor target and three scalar targets give the same per-field and total loss."""
    domain = _domain()
    out = _mesh_output()
    t_pred = select_mesh_fields_by_target(out, TENSOR_TARGETS)
    t_true = extract_targets(domain, TENSOR_TARGETS)
    s_pred = select_mesh_fields_by_target(_as_scalar_output(out), SCALAR_TARGETS)
    s_true = extract_targets(domain, SCALAR_TARGETS)
    t_total, t_td = LossCalculator(TENSOR_TARGETS, loss_type=loss_type)(t_pred, t_true)
    s_total, s_td = LossCalculator(SCALAR_TARGETS, loss_type=loss_type)(s_pred, s_true)
    assert torch.allclose(t_total, s_total, atol=1e-6)
    stress = t_td["loss/stress"]
    three = s_td["loss/sig11"] + s_td["loss/sig22"] + s_td["loss/sig12"]
    assert torch.allclose(stress, three, atol=1e-6)


def test_metrics_keep_the_component_names():
    """Tensor-target metrics carry the component names the same fields have as scalar targets."""
    domain = _domain()
    out = _mesh_output()
    t_pred = select_mesh_fields_by_target(out, TENSOR_TARGETS)
    t_true = extract_targets(domain, TENSOR_TARGETS)
    s_pred = select_mesh_fields_by_target(_as_scalar_output(out), SCALAR_TARGETS)
    s_true = extract_targets(domain, SCALAR_TARGETS)
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
    w = torch.rand(16) + 0.5
    assert set(calc(t_pred, t_true, w).keys()) == set(calc.expected_keys())


def test_explode_tensor_targets_round_trips_to_component_fields():
    """Exploding a tensor target recovers its component fields; scalar-only targets pass through."""
    domain = _domain()
    t_true = extract_targets(domain, TENSOR_TARGETS)
    s_true = extract_targets(domain, SCALAR_TARGETS)
    exploded = explode_tensor_targets(t_true, TENSOR_TARGETS)
    assert list(exploded.keys()) == list(SCALAR_TARGETS)
    for k in SCALAR_TARGETS:
        assert torch.equal(exploded[k], s_true[k])
    assert explode_tensor_targets(s_true, SCALAR_TARGETS) is s_true


def test_isla_end_to_end_through_collate_and_loss():
    """The recipe's mesh contract reaches ISLA end to end on a surface task: the boundary
    Mesh (with a rank-1 and a rank-0 boundary input in its cell_data and an explicit cell
    measure) and the case's global_data go in, the output Mesh's point_data reaches the
    loss by target name, and a symmetric-tensor target is read off the rank-2 leaf."""
    g = torch.Generator().manual_seed(0)
    n = 24
    surface = Mesh(
        points=torch.randn(n * 2, 3, generator=g),
        cells=torch.randint(0, n * 2, (n, 3), generator=g),
        cell_data={
            "traction": torch.randn(n, 3, generator=g),
            "fixed": torch.rand(n, generator=g),
        },
    )
    set_cell_measures(surface, torch.rand(n, generator=g) + 0.5)
    pd = {k: torch.randn(n, generator=g) for k in ("q", "sig11", "sig22", "sig12")}
    pd["displacement"] = torch.randn(n, 3, generator=g)
    domain = DomainMesh(
        interior=Mesh(points=surface.cell_centroids, point_data=pd),
        boundaries={"outer": surface},
        global_data={"load_dir": torch.tensor([0.0, 1.0, 0.0])},
    )
    spec = {
        "boundary_meshes": {"outer": "boundaries.outer"},
        "prediction_boundaries": {"literal": ["outer"]},
        "global_data": "global_data",
    }
    batch = build_collate_fn("mesh", spec, TENSOR_TARGETS)([(domain, {})])
    fk = batch["forward_kwargs"]
    assert fk["boundary_meshes"]["outer"] is surface
    model = ISLA(
        output_schema={
            "q": {"rank": 0},
            "stress": {"rank": 2, "symmetric": True},
            "displacement": {"rank": 1},
        },
        global_schema={"load_dir": {"rank": 1}},
        boundary_schemas={"outer": {"traction": {"rank": 1}, "fixed": {"rank": 0}}},
        hidden=16,
        n_layers=1,
        n_slices=4,
    )
    with torch.no_grad():
        out = model(**fk)
    assert isinstance(out, Mesh)
    ### Surface task: the predictions sit at the boundary cells, in cell order, which is
    ### where the dataset contract puts the targets (interior = cell centroids).
    torch.testing.assert_close(out.points, domain.interior.points)
    assert out.point_data["stress"].shape == (n, 3, 3)
    assert out.point_data["q"].shape == (n,)
    pred = normalize_output_to_tensordict(out, TENSOR_TARGETS, "mesh")
    assert pred["stress"].shape == (n, 3)
    total, td = LossCalculator(TENSOR_TARGETS, loss_type="mse")(pred, batch["targets"])
    assert torch.isfinite(total) and set(td.keys()) >= {
        "loss/q",
        "loss/stress",
        "loss/displacement",
    }


def test_select_tensor_components_accepts_a_planar_2x2_tensor():
    """A planar problem restricted to its plane (ISLA n_spatial_dims=2) predicts a (N, 2, 2) tensor; the in-plane
    labels read off it, out-of-plane labels are refused (2026-10-03)."""
    import pytest
    import torch
    from utils import select_tensor_components

    t = torch.randn(5, 2, 2)
    out = select_tensor_components(t, STRESS)
    assert out.shape == (5, 3)
    assert (
        torch.equal(out[:, 0], t[:, 0, 0])
        and torch.equal(out[:, 1], t[:, 1, 1])
        and torch.equal(out[:, 2], t[:, 0, 1])
    )
    with pytest.raises(ValueError, match="does not have"):
        select_tensor_components(
            t, {"type": "symmetric_tensor", "components": {"s33": "zz"}}
        )
    with pytest.raises(ValueError, match="d in"):
        select_tensor_components(torch.randn(5, 4, 4), STRESS)


def test_stress_l2_is_one_frobenius_formula_for_both_target_paths_and_rotation_invariant():
    """CHAR-PROTOCOLS (2026-10-04): a dataset declaring sig11/sig22/sig12 as scalars gets a derived
    ``stress_l2`` equal (to roundoff) to the ``symmetric_tensor`` path's on the same arrays; the
    aggregate is the Frobenius relative L2 (off-diagonal counted twice), invariant under an in-plane
    rotation of pred and target, while the per-component relative errors move."""
    import math

    import torch
    from metrics import MetricCalculator
    from tensordict import TensorDict

    g = torch.Generator().manual_seed(5)
    n = 64
    t = torch.randn(n, 3, generator=g)  # xx, yy, xy
    p = t + 0.3 * torch.randn(n, 3, generator=g)
    scalars = {"q": "scalar", "sig11": "scalar", "sig22": "scalar", "sig12": "scalar"}
    tensor = {"q": "scalar", "stress": STRESS}
    q_p, q_t = torch.randn(n, generator=g), torch.randn(n, generator=g)
    m_s = MetricCalculator(scalars, metrics=["l2"])(
        TensorDict({"q": q_p, "sig11": p[:, 0], "sig22": p[:, 1], "sig12": p[:, 2]}),
        TensorDict({"q": q_t, "sig11": t[:, 0], "sig22": t[:, 1], "sig12": t[:, 2]}),
    )
    m_t = MetricCalculator(tensor, metrics=["l2"])(
        TensorDict({"q": q_p, "stress": p}), TensorDict({"q": q_t, "stress": t})
    )
    assert (
        "stress_l2" in m_s.keys()
        and "stress_l2" in MetricCalculator(scalars, metrics=["l2"]).expected_keys()
    )
    torch.testing.assert_close(m_s["stress_l2"], m_t["stress_l2"])
    for k in ("sig11_l2", "sig22_l2", "sig12_l2"):
        torch.testing.assert_close(m_s[k], m_t[k])

    ### Frobenius by hand: off-diagonal twice.
    def fro(a, b):
        return math.sqrt(
            float(((a - b)[:, :2] ** 2).sum() + 2 * ((a - b)[:, 2] ** 2).sum())
        ) / math.sqrt(float((b[:, :2] ** 2).sum() + 2 * (b[:, 2] ** 2).sum()))

    assert abs(float(m_t["stress_l2"]) - fro(p, t)) < 1e-5
    ### Rotation invariance of the aggregate; per-component errors change.
    c, s_ = math.cos(0.7), math.sin(0.7)

    def rot(v):
        xx, yy, xy = v[:, 0], v[:, 1], v[:, 2]
        return torch.stack(
            [
                c * c * xx + s_ * s_ * yy - 2 * c * s_ * xy,
                s_ * s_ * xx + c * c * yy + 2 * c * s_ * xy,
                c * s_ * (xx - yy) + (c * c - s_ * s_) * xy,
            ],
            -1,
        )

    m_r = MetricCalculator(tensor, metrics=["l2"])(
        TensorDict({"q": q_p, "stress": rot(p)}),
        TensorDict({"q": q_t, "stress": rot(t)}),
    )
    torch.testing.assert_close(m_r["stress_l2"], m_t["stress_l2"], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(m_r["sig11_l2"], m_t["sig11_l2"])
