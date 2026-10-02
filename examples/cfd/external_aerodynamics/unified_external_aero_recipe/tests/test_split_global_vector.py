# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SplitGlobalVector (2026-10-02): a per-case global vector becomes named rank-0 global leaves (standardized when
statistics are given) plus, optionally, the standardized vector whole; the raw field stays; collisions and
mismatched lengths are refused; a bare Mesh and a DomainMesh are both served."""

import pytest
import torch
from tensordict import TensorDict

from domain_transforms import SplitGlobalVector
from physicsnemo.mesh import DomainMesh, Mesh

MEAN = [0.0124862, 0.100103, 0.0225110, 75300663.3]
STD = [0.00144079, 0.0115957, 0.00423982, 14360825.0]
NAMES = ["hole_radius_std", "length_std", "thickness_std", "pressure_std"]


def _global_data() -> TensorDict:
    return TensorDict(
        {
            "params": torch.tensor([0.0139, 0.11, 0.019, 80.0e6]),
            "pressure": torch.tensor(80.0e6),
            "load_type": torch.tensor([0.0, 1.0]),
        },
        batch_size=[],
    )


def _domain() -> DomainMesh:
    interior = Mesh(points=torch.rand(20, 3), cells=torch.zeros(0, 1, dtype=torch.long))
    bnd = Mesh(points=torch.rand(4, 3), cells=torch.tensor([[0, 1, 2], [1, 2, 3]]))
    return DomainMesh(interior=interior, boundaries={"surface": bnd}, global_data=_global_data())


def test_domain_vector_is_standardized_split_and_kept_whole():
    d = _domain()
    out = SplitGlobalVector(field="params", names=NAMES, mean=MEAN, std=STD, output_field="params_std").apply_to_domain(d)
    gd = out.global_data
    expected = (d.global_data["params"] - torch.tensor(MEAN)) / torch.tensor(STD)
    for i, name in enumerate(NAMES):
        assert gd[name].shape == ()  # a rank-0 field: a 0-d tensor
        torch.testing.assert_close(gd[name], expected[i])
    torch.testing.assert_close(gd["params_std"], expected)
    torch.testing.assert_close(gd["params"], d.global_data["params"])  # the raw vector stays
    torch.testing.assert_close(gd["pressure"], d.global_data["pressure"])  # untouched raw leaf
    assert out.interior is d.interior and out.boundaries["surface"] is d.boundaries["surface"]


def test_one_hot_is_split_without_statistics_on_a_bare_mesh():
    mesh = Mesh(points=torch.rand(4, 3), cells=torch.tensor([[0, 1, 2], [1, 2, 3]]), global_data=_global_data())
    out = SplitGlobalVector(field="load_type", names=["is_force", "is_torque"])(mesh)
    assert float(out.global_data["is_force"]) == 0.0 and float(out.global_data["is_torque"]) == 1.0
    assert "load_type" in out.global_data.keys()
    assert torch.equal(out.points, mesh.points)


def test_scalar_statistics_broadcast_over_the_names():
    d = _domain()
    out = SplitGlobalVector(field="load_type", names=["a", "b"], mean=0.5, std=0.5).apply_to_domain(d)
    torch.testing.assert_close(out.global_data["a"], torch.tensor(-1.0))
    torch.testing.assert_close(out.global_data["b"], torch.tensor(1.0))


def test_collisions_lengths_and_statistics_are_refused():
    d = _domain()
    with pytest.raises(ValueError, match="already has"):
        SplitGlobalVector(field="params", names=["hole_radius", "length", "thickness", "pressure"]).apply_to_domain(d)
    with pytest.raises(ValueError, match="entries"):
        SplitGlobalVector(field="params", names=["a", "b"]).apply_to_domain(d)
    with pytest.raises(ValueError, match="mean has 3"):
        SplitGlobalVector(field="params", names=NAMES, mean=[0.0, 0.0, 0.0], std=1.0)
    with pytest.raises(ValueError, match="positive"):
        SplitGlobalVector(field="params", names=NAMES, std=[1.0, 0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="distinct"):
        SplitGlobalVector(field="params", names=["a", "a", "b", "c"])
    with pytest.raises(KeyError, match="needs global_data"):
        SplitGlobalVector(field="missing", names=["a"]).apply_to_domain(d)
