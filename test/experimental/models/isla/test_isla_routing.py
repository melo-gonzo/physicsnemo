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

"""Bounded (cosine) routing logits, and the softmax over points that normalizes them."""

import torch

from physicsnemo.experimental.models.isla.blocks import (
    ROUTING_TEMPERATURE,
    _RoutingLogits,
    _softmax_over_points,
)


def test_routing_logits_are_bounded_and_bias_free():
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
        out = r(x, torch.zeros(2, 50, 1))
    assert ROUTING_TEMPERATURE == 10.0
    assert out.abs().max() <= ROUTING_TEMPERATURE + 1e-5


def test_softmax_over_points_matches_middle_dim_softmax():
    """The point softmax (reduction along a contiguous last dimension after a
    transpose) is the same arithmetic as a softmax over the middle dimension: exact in
    float64, at roundoff in float32."""
    x = torch.randn(2, 500, 64, dtype=torch.float64)
    assert torch.allclose(_softmax_over_points(x), torch.softmax(x, dim=1), atol=1e-14)
    assert torch.allclose(
        _softmax_over_points(x).sum(dim=1),
        torch.ones(2, 64, dtype=torch.float64),
        atol=1e-14,
    )
    x32 = x.float()
    rel = (
        (_softmax_over_points(x32) - torch.softmax(x32, dim=1)).norm()
        / torch.softmax(x32, dim=1).norm()
    ).item()
    assert rel < 1e-6
