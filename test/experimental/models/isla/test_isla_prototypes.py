# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sample-conditioned slice prototypes (``slice_prototypes="centered"``). Contracts (rotation, translation,
reflection, token permutation, uniform measure rescale), parameter accounting, exactness of the centered,
scale-bounded construction and collapse-proofness."""

import pytest
import torch

from physicsnemo.experimental.models.isla import ISLA
from physicsnemo.experimental.models.isla.model import _RoutingLogits


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


def test_centered_prototypes_keep_every_contract():
    m = ISLA(**KW, slice_prototypes="centered").double().eval()
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
    torch.manual_seed(0)
    rl = _RoutingLogits(16, 4, slice_prototypes="centered")
    x1, x2 = torch.randn(1, 50, 16), torch.randn(1, 50, 16)
    lw = torch.zeros(1, 50, 1)
    p1, p2 = rl.prototypes(x1, lw), rl.prototypes(x2, lw)
    assert p1.shape == (1, 4, 16) and not torch.allclose(p1, p2)
    assert rl(x1, lw).abs().max() <= 10.0 + 1e-5  # bounded by the temperature
    n_static = sum(p.numel() for p in ISLA(**KW).parameters())
    n_cent = sum(p.numel() for p in ISLA(**KW, slice_prototypes="centered").parameters())
    assert n_cent - n_static == KW["n_layers"] * (2 * KW["hidden"] ** 2 + 1)  # key and value maps plus the gain, per block
    assert not hasattr(_RoutingLogits(16, 4), "k_proj")


def test_centered_prototypes_are_centered_and_scale_bounded():
    torch.manual_seed(0)
    rl = _RoutingLogits(16, 4, slice_prototypes="centered").eval()
    x = torch.randn(2, 50, 16)
    lw = torch.randn(2, 50, 1)
    with torch.no_grad():
        delta = rl.prototypes(x, lw) - rl.weight[None]  # (B, S, H)
        # the unbounded centered term, recomputed from the module's own maps
        att = torch.softmax(torch.einsum("sh,bnh->bsn", rl.weight, rl.k_proj(x)) * rl.scale + lw[..., 0][:, None, :], -1)
        raw = torch.einsum("bsn,bnh->bsh", att, rl.v_proj(x)); raw = raw - raw.mean(dim=1, keepdim=True)
    assert delta.shape == (2, 4, 16)
    torch.testing.assert_close(delta.mean(dim=1), torch.zeros(2, 16), rtol=0, atol=1e-5)
    rms_delta = delta.square().mean(dim=(1, 2)).sqrt()
    rms_raw = raw.square().mean(dim=(1, 2)).sqrt()
    bound = rl.gain * rl.weight.square().mean().sqrt()
    torch.testing.assert_close(rms_delta, bound * rms_raw / (rms_raw + bound), rtol=1e-4, atol=0)
    assert (rms_delta <= bound).all()  # saturates at g rms(seeds)


def test_centered_prototypes_equal_the_seeds_on_identical_tokens():
    torch.manual_seed(0)
    rl = _RoutingLogits(16, 4, slice_prototypes="centered").eval()
    x = torch.randn(1, 1, 16).expand(1, 50, 16).clone()  # every token identical: one value for every seed
    lw = torch.randn(1, 50, 1)
    with torch.no_grad():
        protos = rl.prototypes(x, lw)
    torch.testing.assert_close(protos, rl.weight[None].expand_as(protos), rtol=0, atol=1e-6)


def test_slice_prototypes_is_validated():
    for mode in ("static", "centered"):
        ISLA(**KW, slice_prototypes=mode)
    for mode in ("conditioned", "dynamic"):
        with pytest.raises(ValueError, match="'static' or 'centered'"):
            ISLA(**KW, slice_prototypes=mode)
