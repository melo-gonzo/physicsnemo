# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PROTO-COND (2026-09-29): sample-conditioned slice prototypes. Contracts (rotation, translation, reflection, token
permutation, uniform measure rescale), parameter accounting, and the cosine-only restriction."""

import pytest
import torch

from physicsnemo.experimental.models.isla import ISLA


def _problem(n=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = torch.rand(1, n, 3, generator=g) * 2 - 1
    nrm = torch.randn(1, n, 3, generator=g); nrm = nrm / nrm.norm(dim=-1, keepdim=True)
    w = torch.rand(1, n, generator=g) + 0.5
    drv = torch.tensor([[0.6, 0.8, 0.0]])
    return r, nrm, w, drv


def _rotation(g):
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g).double())  # orthogonal to double precision
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


KW = dict(outputs={"p": 0, "v": 1}, global_inputs={"g": 1}, hidden=32, n_layers=2, n_slices=8)


def test_conditioned_prototypes_keep_every_contract():
    m = ISLA(**KW, slice_prototypes="conditioned").double().eval()
    r, nrm, w, drv = _problem()
    r, nrm, w, drv = r.double(), nrm.double(), w.double(), drv.double()
    g = torch.Generator().manual_seed(7)
    q = _rotation(g); refl = torch.diag(torch.tensor([1.0, -1.0, 1.0])).double()
    perm = torch.randperm(r.shape[1], generator=g)
    with torch.no_grad():
        base = m(points=r, normals=nrm, global_inputs={"g": drv}, measure_weights=w)
        rot = m(points=r @ q.T, normals=nrm @ q.T, global_inputs={"g": drv @ q.T}, measure_weights=w)
        sh = m(points=r + 3.0, normals=nrm, global_inputs={"g": drv}, measure_weights=w)
        rf = m(points=r @ refl.T, normals=nrm @ refl.T, global_inputs={"g": drv @ refl.T}, measure_weights=w)
        pm = m(points=r[:, perm], normals=nrm[:, perm], global_inputs={"g": drv}, measure_weights=w[:, perm])
        sc = m(points=r, normals=nrm, global_inputs={"g": drv}, measure_weights=3.7 * w)
    tol = dict(rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(rot["p"], base["p"], **tol); torch.testing.assert_close(rot["v"], base["v"] @ q.T, **tol)
    torch.testing.assert_close(sh["p"], base["p"], **tol); torch.testing.assert_close(sh["v"], base["v"], **tol)
    torch.testing.assert_close(rf["p"], base["p"], **tol); torch.testing.assert_close(rf["v"], base["v"] @ refl.T, **tol)
    torch.testing.assert_close(pm["p"], base["p"][:, perm], **tol); torch.testing.assert_close(pm["v"], base["v"][:, perm], **tol)
    torch.testing.assert_close(sc["p"], base["p"], **tol); torch.testing.assert_close(sc["v"], base["v"], **tol)


def test_prototypes_depend_on_the_sample_and_static_is_unchanged():
    from physicsnemo.experimental.models.isla.model import _RoutingLogits

    torch.manual_seed(0)
    rl = _RoutingLogits(16, 4, "cosine", 10.0, conditioned=True)
    x1, x2 = torch.randn(1, 50, 16), torch.randn(1, 50, 16)
    lw = torch.zeros(1, 50, 1)
    p1, p2 = rl.prototypes(x1, lw), rl.prototypes(x2, lw)
    assert p1.shape == (1, 4, 16) and not torch.allclose(p1, p2)
    assert rl(x1, lw).abs().max() <= 10.0 + 1e-5  # bounded by the temperature
    n_static = sum(p.numel() for p in ISLA(**KW).parameters())
    n_cond = sum(p.numel() for p in ISLA(**KW, slice_prototypes="conditioned").parameters())
    assert n_cond - n_static == KW["n_layers"] * 2 * KW["hidden"] ** 2  # the key and value maps per block


def test_conditioned_requires_cosine_routing():
    with pytest.raises(ValueError, match="cosine"):
        ISLA(**KW, slice_prototypes="conditioned", routing_norm="linear")
    with pytest.raises(ValueError, match="slice_prototypes"):
        ISLA(**KW, slice_prototypes="dynamic")
