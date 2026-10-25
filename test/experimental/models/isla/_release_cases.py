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

"""The release configurations and inputs that test_isla_release.py's goldens were
recorded with, shared with the tests that compare against those goldens."""

import torch

from physicsnemo.experimental.models import ISLA
from test.experimental.models.isla._meshes import soup

#: Every configuration declares one scalar and one vector output, one boundary condition
#: without fields and one global direction.
IO = dict(
    output_schema={"p": {"rank": 0}, "v": {"rank": 1}},
    boundary_schemas={"wall": {}},
    global_schema={"g": {"rank": 1}},
)

#: One small configuration in its two forms: a surface task and an interior task
#: (the constructor's ``interior`` flag), which is ISLA-DTT.
CONFIGS = {
    "default": dict(hidden=32, n_layers=2, n_slices=8, **IO),
    "dtt": dict(hidden=32, n_layers=2, n_slices=8, interior=True, **IO),
}


def _instantiate_model(seed: int = 0, **kwargs) -> ISLA:
    """Build an ISLA with reproducible random parameters (MOD-008b pattern), forced to
    the eager geometry region (the one the goldens were recorded with)."""
    model = ISLA(**kwargs)
    model.geo_kernel = "eager"
    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * 0.1)
    return model.eval()


def _make_inputs(device, query: bool = False, seed: int = 1234) -> dict:
    """Deterministic mesh inputs: one case, a 64-cell boundary soup with unit normals
    and positive effective measures, one global direction, and 40 prediction points
    for the interior task."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = 64
    pts = torch.randn(n, 3, generator=gen) * torch.tensor([3.0, 2.0, 1.0])
    nrm = torch.nn.functional.normalize(torch.randn(n, 3, generator=gen), dim=-1)
    w = torch.rand(n, generator=gen) + 0.5
    g = torch.nn.functional.normalize(torch.randn(3, generator=gen), dim=-1)
    inputs = dict(
        boundary_meshes={"wall": soup(pts, nrm, w).to(device)},
        global_data={"g": g.to(device)},
    )
    if query:
        inputs["prediction_points"] = (torch.randn(40, 3, generator=gen) * 2.0).to(
            device
        )
    else:
        inputs["prediction_boundaries"] = ["wall"]
    return inputs
