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

"""Bounded (cosine) routing logits."""

import torch

from physicsnemo.experimental.models import ISLA
from physicsnemo.experimental.models.isla.model import (
    ROUTING_TEMPERATURE,
    _RoutingLogits,
)

PV = {"p": 0, "v": 1}


def _cloud(n=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn(1, n, 3, dtype=torch.float64, generator=g) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64, generator=g), dim=-1
    )
    drv = torch.nn.functional.normalize(
        torch.randn(1, 3, dtype=torch.float64, generator=g), dim=-1
    )
    w = torch.rand(1, n, dtype=torch.float64, generator=g) + 0.5
    return pts, nrm, drv, w


def test_cosine_routing_logits_are_bounded_and_bias_free():
    torch.manual_seed(0)
    r = _RoutingLogits(64, 32)
    assert r.weight.shape == (32, 64)
    assert set(dict(r.named_parameters())) == {"weight"}
    with torch.no_grad():
        r.weight.mul_(1000.0)  # however large the weights grow ...
        x = torch.randn(2, 50, 64) * 100.0  # ... or the state
        out = r(x)
    assert ROUTING_TEMPERATURE == 10.0
    assert out.abs().max() <= ROUTING_TEMPERATURE + 1e-5


def test_cosine_routing_keeps_the_contracts_and_all_parameters_train():
    torch.manual_seed(0)
    m = (
        ISLA(outputs=PV, global_inputs={"g": 1}, hidden=32, n_layers=2, n_slices=8)
        .double()
        .eval()
    )
    pts, nrm, drv, w = _cloud()
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    with torch.no_grad():
        base = m(points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w)
        moved = m(
            points=pts @ q.T + shift,
            normals=nrm @ q.T,
            global_inputs={"g": drv @ q.T},
            measure_weights=w,
        )
        rescaled = m(
            points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=137.0 * w
        )
    assert torch.allclose(moved["p"], base["p"], atol=1e-10)
    assert torch.allclose(moved["v"], base["v"] @ q.T, atol=1e-10)
    assert all(torch.allclose(rescaled[k], base[k], atol=1e-9) for k in base.keys())
    m.train()
    out = m(
        points=pts,
        normals=nrm,
        global_inputs={"g": drv},
        measure_weights=w,
        query_points=pts[:, :20] * 1.5,
    )
    sum(o.square().mean() for o in out.values()).backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, missing
