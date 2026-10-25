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
from physicsnemo.experimental.models.isla.blocks import (
    ROUTING_TEMPERATURE,
    _RoutingLogits,
)
from test.experimental.models.isla._meshes import (
    cloud,
    live,
    orthogonal,
    outputs,
    rescale_measures,
    soup,
    transform,
)

PV = {"p": {"rank": 0}, "v": {"rank": 1}}


def test_cosine_routing_logits_are_bounded_and_bias_free():
    torch.manual_seed(0)
    r = _RoutingLogits(64, 32)
    assert r.weight.shape == (32, 64)
    # the seeds, the key and value maps of the prototype synthesis and its gain; no bias anywhere
    assert set(dict(r.named_parameters())) == {
        "weight",
        "k_proj.weight",
        "v_proj.weight",
        "gain",
    }
    with torch.no_grad():
        r.weight.mul_(1000.0)  # however large the seeds grow ...
        r.k_proj.weight.mul_(1000.0)
        r.v_proj.weight.mul_(1000.0)
        r.gain.fill_(1000.0)  # ... or the synthesis maps ...
        x = torch.randn(2, 50, 64) * 100.0  # ... or the state
        out = r(x)
    assert ROUTING_TEMPERATURE == 10.0
    assert out.abs().max() <= ROUTING_TEMPERATURE + 1e-5


def test_cosine_routing_keeps_the_contracts_and_all_parameters_train():
    torch.manual_seed(0)
    m = (
        ISLA(
            output_schema=PV,
            boundary_schemas={"wall": {}},
            global_schema={"g": {"rank": 1}},
            hidden=32,
            n_layers=2,
            n_slices=8,
            interior=True,
        )
        .double()
        .eval()
    )
    live(m)
    pts, nrm, w, drv = cloud(300)
    mesh = soup(pts, nrm, w)
    q = orthogonal(3, 0)
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    probe = pts[:20] * 1.5

    def run(mesh, g, q):
        with torch.no_grad():
            return outputs(
                m(
                    boundary_meshes={"wall": mesh},
                    prediction_points=q,
                    global_data={"g": g},
                )
            )

    base = run(mesh, drv, probe)
    moved = run(transform(mesh, q, shift), drv @ q.T, probe @ q.T + shift)
    rescaled = run(rescale_measures(mesh, 137.0), drv, probe)
    assert torch.allclose(moved["p"], base["p"], atol=1e-10)
    assert torch.allclose(moved["v"], base["v"] @ q.T, atol=1e-10)
    assert all(torch.allclose(rescaled[k], base[k], atol=1e-9) for k in base)
    m.train()
    out = outputs(
        m(
            boundary_meshes={"wall": mesh},
            prediction_points=probe,
            global_data={"g": drv},
        )
    )
    sum(o.square().mean() for o in out.values()).backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, missing
