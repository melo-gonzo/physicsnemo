# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LOCAL-LPR (2026-09-28): polynomial-reproduction stencil weights and finite-difference channels of the local aggregation.
Exact affine reproduction of the pooled value and of the normal-projected gradient; contracts of a model using them;
default path parameter count unchanged; the dense path refuses the options."""

import pytest
import torch

from physicsnemo.experimental.models.isla import ISLA
from physicsnemo.experimental.models.isla.model import _knn_indices, _local_region_knn


def _cloud(n=400, d=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = torch.rand(1, n, d, generator=g) * 2 - 1
    nrm = torch.randn(1, n, d, generator=g)
    nrm = nrm / nrm.norm(dim=-1, keepdim=True)
    log_w = torch.log(torch.rand(1, n, 1, generator=g) + 0.5)
    return r, nrm, log_w


@pytest.mark.parametrize("d", [2, 3])
def test_lpr_weights_reproduce_affine_fields_and_their_normal_derivative(d):
    r, nrm, log_w = _cloud(d=d)
    g = torch.Generator().manual_seed(1)
    G = torch.randn(d, 6, generator=g)
    c = torch.randn(6, generator=g)
    h = r @ G + c  # affine field, 6 channels
    idx = _knn_indices(r, 16)
    rho = 0.2
    g_hat = torch.zeros(1, r.shape[1], 0, d)
    pooled, geo = _local_region_knn(h, log_w, r, nrm, g_hat, idx, rho, 1e-12, "lpr", True)
    val, fd = pooled[..., :6], pooled[..., 6:]
    torch.testing.assert_close(val, h, rtol=1e-4, atol=1e-4)  # value weights reproduce affine fields exactly
    expected = rho * torch.einsum("bnd,dh->bnh", nrm, G)  # rho * (n . grad h)
    torch.testing.assert_close(fd, expected, rtol=1e-3, atol=1e-3)
    # the Gaussian smoother does not reproduce the affine field (first-moment bias)
    pooled_g, _ = _local_region_knn(h, log_w, r, nrm, g_hat, idx, rho, 1e-12, "gaussian", False)
    assert (pooled_g - h).abs().max() > 1e-2
    assert geo.shape == (1, r.shape[1], 5)


def _rotation(d, g):
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_model_with_lpr_and_gradient_keeps_the_contracts_and_parameter_count():
    kw = dict(outputs={"p": 0, "v": 1}, global_inputs={"g": 1}, hidden=32, n_layers=2, n_slices=8, local_rho=0.3, local_k=16)
    m = ISLA(**kw, local_weights="lpr", local_gradient=True)
    base_model = ISLA(**kw)
    lpr_only = ISLA(**kw, local_weights="lpr")
    n0, n1, n2 = (sum(p.numel() for p in mm.parameters()) for mm in (base_model, lpr_only, m))
    assert n0 == n1 and n2 > n0  # lpr weights add no parameters; the gradient channels widen local_out
    r, nrm, log_w = _cloud(n=200)
    w = log_w.exp()[..., 0]
    drv = torch.tensor([[0.6, 0.8, 0.0]])
    g = torch.Generator().manual_seed(5)
    q = _rotation(3, g)
    refl = torch.diag(torch.tensor([1.0, -1.0, 1.0]))
    with torch.no_grad():
        base = m(points=r, normals=nrm, global_inputs={"g": drv}, measure_weights=w)
        rot = m(points=r @ q.T, normals=nrm @ q.T, global_inputs={"g": drv @ q.T}, measure_weights=w)
        sh = m(points=r + 3.0, normals=nrm, global_inputs={"g": drv}, measure_weights=w)
        rf = m(points=r @ refl.T, normals=nrm @ refl.T, global_inputs={"g": drv @ refl.T}, measure_weights=w)
    tol = dict(rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(rot["p"], base["p"], **tol)
    torch.testing.assert_close(rot["v"], base["v"] @ q.T, **tol)
    torch.testing.assert_close(sh["p"], base["p"], **tol)
    torch.testing.assert_close(sh["v"], base["v"], **tol)
    torch.testing.assert_close(rf["p"], base["p"], **tol)
    torch.testing.assert_close(rf["v"], base["v"] @ refl.T, **tol)
    assert torch.isfinite(base["p"]).all()


def test_lpr_options_are_refused_without_the_knn_path():
    with pytest.raises(ValueError, match="local_k"):
        ISLA(outputs={"p": 0}, local_rho=0.3, local_k=0, local_weights="lpr")
    with pytest.raises(ValueError, match="local_k"):
        ISLA(outputs={"p": 0}, local_gradient=True)
    with pytest.raises(ValueError, match="local_weights"):
        ISLA(outputs={"p": 0}, local_rho=0.3, local_weights="cubic")
