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

"""The point softmax (reduction along a contiguous last dimension after a transpose)
is the same arithmetic as a softmax over the middle dimension: exact in float64, at
roundoff in float32."""

import torch

from physicsnemo.experimental.models.isla.model import _softmax_over_points


def test_softmax_over_points_matches_middle_dim_softmax():
    x = torch.randn(2, 500, 64, dtype=torch.float64)
    assert torch.allclose(_softmax_over_points(x), torch.softmax(x, dim=1), atol=1e-14)
    assert torch.allclose(_softmax_over_points(x).sum(dim=1), torch.ones(2, 64, dtype=torch.float64), atol=1e-14)
    x32 = x.float()
    rel = (
        (_softmax_over_points(x32) - torch.softmax(x32, dim=1)).norm()
        / torch.softmax(x32, dim=1).norm()
    ).item()
    assert rel < 1e-6
