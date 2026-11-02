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

import math

import pytest
import torch

from physicsnemo.models.geotransolver import GeoTransolver
from physicsnemo.models.geotransolver.streaming import streamed_forward
from physicsnemo.models.transolver import Transolver

N_POINTS = 1500
N_GEO = 700
CHUNK = 400  # neither count is a multiple, so every pass ends on a short chunk


def _geotransolver(local: bool, local_geometry_dims: int | None = None):
    geometry_dim = 3 if local_geometry_dims is None else 5
    kwargs = dict(
        radii=[0.3, 1.0],
        neighbors_in_radius=[4, 16],  # truncates: most balls hold more points
        n_hidden_local=8,
        local_geometry_dims=local_geometry_dims,
    )
    model = GeoTransolver(
        functional_dim=4,
        out_dim=3,
        geometry_dim=geometry_dim,
        global_dim=2,
        n_layers=2,
        n_hidden=32,
        n_head=4,
        slice_num=8,
        use_te=False,
        include_local_features=local,
        **(kwargs if local else {}),
    )
    return model.eval(), geometry_dim


def _inputs(geometry_dim: int):
    positions = torch.rand(1, N_POINTS, 3) * 4
    return dict(
        local_embedding=torch.cat([positions, torch.randn(1, N_POINTS, 1)], -1),
        local_positions=positions,
        geometry=torch.cat(
            [torch.rand(1, N_GEO, 3) * 4, torch.randn(1, N_GEO, geometry_dim - 3)], -1
        ),
        global_embedding=torch.randn(1, 1, 2),
    )


@pytest.mark.parametrize(
    "local, local_geometry_dims",
    [(False, None), (True, None), (True, 3)],
    ids=["tokens", "local-features", "local-features-xyz-of-5"],
)
def test_geotransolver_streamed_matches_one_pass(local, local_geometry_dims):
    torch.manual_seed(0)
    model, geometry_dim = _geotransolver(local, local_geometry_dims)
    kwargs = _inputs(geometry_dim)
    with torch.no_grad():
        full = model(**kwargs)
    streamed = streamed_forward(model, CHUNK, **kwargs)
    torch.testing.assert_close(streamed, full, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("kind", ["geotransolver", "transolver"])
def test_each_block_projects_features_in_its_first_pass_only(kind):
    """The second pass over a block's chunks needs only the slice weights, so
    the feature projection runs once per chunk and block."""
    torch.manual_seed(0)
    if kind == "geotransolver":
        model, geometry_dim = _geotransolver(False)
        kwargs = _inputs(geometry_dim)
    else:
        model = Transolver(
            functional_dim=2,
            out_dim=3,
            embedding_dim=3,
            n_layers=3,
            n_hidden=32,
            n_head=4,
            slice_num=8,
            use_te=False,
            unified_pos=False,
        ).eval()
        kwargs = dict(
            fx=torch.randn(1, N_POINTS, 2), embedding=torch.randn(1, N_POINTS, 3)
        )
    calls = []
    for block in model.blocks:
        block.Attn.in_project_fx.register_forward_hook(lambda *_: calls.append(1))
    streamed_forward(model, CHUNK, **kwargs)
    assert len(calls) == len(model.blocks) * math.ceil(N_POINTS / CHUNK)


def test_transolver_streamed_matches_one_pass():
    torch.manual_seed(0)
    model = Transolver(
        functional_dim=2,
        out_dim=3,
        embedding_dim=3,
        n_layers=3,
        n_hidden=32,
        n_head=4,
        slice_num=8,
        use_te=False,
        unified_pos=False,
    ).eval()
    kwargs = dict(fx=torch.randn(1, N_POINTS, 2), embedding=torch.randn(1, N_POINTS, 3))
    with torch.no_grad():
        full = model(**kwargs)
    torch.testing.assert_close(
        streamed_forward(model, CHUNK, **kwargs), full, atol=1e-5, rtol=1e-5
    )


def test_streamed_forward_refuses_training_mode_and_plus():
    model, geometry_dim = _geotransolver(False)
    with pytest.raises(ValueError, match="eval mode"):
        streamed_forward(model.train(), CHUNK, **_inputs(geometry_dim))
    plus = GeoTransolver(
        functional_dim=4,
        out_dim=3,
        geometry_dim=3,
        n_layers=1,
        n_hidden=32,
        n_head=4,
        slice_num=8,
        use_te=False,
        plus=True,
    ).eval()
    with pytest.raises(NotImplementedError, match="plus=True"):
        streamed_forward(plus, CHUNK, **_inputs(3))


def test_streamed_sums_stay_float32_under_autocast():
    """Under bf16 autocast the chunked slice sums still accumulate in float32."""
    from physicsnemo.models.geotransolver.streaming import _SliceSums

    torch.manual_seed(0)
    w = torch.rand(1, 20_000, 2, 4).softmax(-1)
    x = torch.rand(1, 20_000, 2, 3)
    sums = _SliceSums()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        for r in range(0, 20_000, 100):
            sums.add(w[:, r : r + 100], x[:, r : r + 100])
    exact = torch.einsum("bnhs,bnhd->bhsd", w.double(), x.double())
    assert sums.num.dtype == torch.float32
    torch.testing.assert_close(sums.num.double(), exact, rtol=1e-5, atol=0)
