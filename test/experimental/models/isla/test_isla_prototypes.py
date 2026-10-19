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

"""Sample-conditioned slice prototypes (the routing of every slice block). Contracts (rotation, translation,
reflection, token permutation, uniform measure rescale), parameter accounting, exactness of the centered,
scale-bounded construction and collapse-proofness."""

import torch

from physicsnemo.experimental.models.isla import ISLA
from physicsnemo.experimental.models.isla.blocks import _RoutingLogits
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import EFFECTIVE_MEASURE_KEY
from test.experimental.models.isla._meshes import (
    orthogonal,
    outputs,
    rescale_measures,
    soup,
    transform,
)


def _problem(n=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = (torch.rand(n, 3, generator=g) * 2 - 1).double()
    nrm = torch.nn.functional.normalize(torch.randn(n, 3, generator=g), dim=-1).double()
    w = (torch.rand(n, generator=g) + 0.5).double()
    drv = torch.tensor([0.6, 0.8, 0.0]).double()
    return soup(r, nrm, w), drv


KW = dict(
    output_schema={"p": {"rank": 0}, "v": {"rank": 1}},
    boundary_schemas={"wall": {}},
    global_schema={"g": {"rank": 1}},
    hidden=32,
    n_layers=2,
    n_slices=8,
)


def test_centered_prototypes_keep_every_contract():
    m = ISLA(**KW).double().eval()
    mesh, drv = _problem()
    q = orthogonal(3, 7)
    refl = torch.diag(torch.tensor([1.0, -1.0, 1.0])).double()
    perm = torch.randperm(mesh.n_cells, generator=torch.Generator().manual_seed(7))
    permuted = Mesh(
        points=mesh.points,
        cells=mesh.cells[perm],
        cell_data={EFFECTIVE_MEASURE_KEY: mesh.cell_data[EFFECTIVE_MEASURE_KEY][perm]},
    )

    def run(mesh, g):
        with torch.no_grad():
            return outputs(m(boundary_meshes={"wall": mesh}, global_data={"g": g}))

    base = run(mesh, drv)
    rot = run(transform(mesh, q), drv @ q.T)
    sh = run(transform(mesh, shift=torch.full((3,), 3.0).double()), drv)
    rf = run(transform(mesh, refl), drv @ refl.T)
    pm = run(permuted, drv)
    sc = run(rescale_measures(mesh, 3.7), drv)
    tol = dict(rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(rot["p"], base["p"], **tol)
    torch.testing.assert_close(rot["v"], base["v"] @ q.T, **tol)
    torch.testing.assert_close(sh["p"], base["p"], **tol)
    torch.testing.assert_close(sh["v"], base["v"], **tol)
    torch.testing.assert_close(rf["p"], base["p"], **tol)
    torch.testing.assert_close(rf["v"], base["v"] @ refl.T, **tol)
    torch.testing.assert_close(pm["p"], base["p"][perm], **tol)
    torch.testing.assert_close(pm["v"], base["v"][perm], **tol)
    torch.testing.assert_close(sc["p"], base["p"], **tol)
    torch.testing.assert_close(sc["v"], base["v"], **tol)


def test_prototypes_depend_on_the_sample_and_are_accounted_for():
    torch.manual_seed(0)
    rl = _RoutingLogits(16, 4)
    x1, x2 = torch.randn(1, 50, 16), torch.randn(1, 50, 16)
    lw = torch.zeros(1, 50, 1)
    p1, p2 = rl.prototypes(x1, lw), rl.prototypes(x2, lw)
    assert p1.shape == (1, 4, 16) and not torch.allclose(p1, p2)
    assert rl(x1, lw).abs().max() <= 10.0 + 1e-5  # bounded by the temperature
    # per block: the seeds (S x H), the key and value maps (2 H^2) and the gain
    n_routing = sum(p.numel() for b in ISLA(**KW).blocks for p in b.assign.parameters())
    assert n_routing == KW["n_layers"] * (
        KW["n_slices"] * KW["hidden"] + 2 * KW["hidden"] ** 2 + 1
    )


def test_centered_prototypes_are_centered_and_scale_bounded():
    torch.manual_seed(0)
    rl = _RoutingLogits(16, 4).eval()
    x = torch.randn(2, 50, 16)
    lw = torch.randn(2, 50, 1)
    with torch.no_grad():
        delta = rl.prototypes(x, lw) - rl.weight[None]  # (B, S, H)
        # the unbounded centered term, recomputed from the module's own maps
        att = torch.softmax(
            torch.einsum("sh,bnh->bsn", rl.weight, rl.k_proj(x)) * rl.scale
            + lw[..., 0][:, None, :],
            -1,
        )
        raw = torch.einsum("bsn,bnh->bsh", att, rl.v_proj(x))
        raw = raw - raw.mean(dim=1, keepdim=True)
    assert delta.shape == (2, 4, 16)
    torch.testing.assert_close(delta.mean(dim=1), torch.zeros(2, 16), rtol=0, atol=1e-5)
    rms_delta = delta.square().mean(dim=(1, 2)).sqrt()
    rms_raw = raw.square().mean(dim=(1, 2)).sqrt()
    bound = rl.gain * rl.weight.square().mean().sqrt()
    torch.testing.assert_close(
        rms_delta, bound * rms_raw / (rms_raw + bound), rtol=1e-4, atol=0
    )
    assert (rms_delta <= bound).all()  # saturates at g rms(seeds)


def test_centered_prototypes_equal_the_seeds_on_identical_tokens():
    torch.manual_seed(0)
    rl = _RoutingLogits(16, 4).eval()
    x = (
        torch.randn(1, 1, 16).expand(1, 50, 16).clone()
    )  # every token identical: one value for every seed
    lw = torch.randn(1, 50, 1)
    with torch.no_grad():
        protos = rl.prototypes(x, lw)
    torch.testing.assert_close(
        protos, rl.weight[None].expand_as(protos), rtol=0, atol=1e-6
    )
