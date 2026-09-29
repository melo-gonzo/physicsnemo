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

"""ISLA in two and four spatial dimensions (ND-GENERIC, 2026-09-27): the contracts
of the three-dimensional tests, on a native two-dimensional problem and on a
four-dimensional even-parity smoke test, plus the invariants of the dimension
handling (Levi-Civita symbol, pseudovector counts, refusals)."""

import pytest
import torch

from physicsnemo.experimental.models.isla import ISLA
from physicsnemo.experimental.models.isla.model import _levi_civita, _pseudovectors

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _rotation(d: int, generator: torch.Generator) -> torch.Tensor:
    """A random proper rotation in ``d`` dimensions (det +1)."""
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=generator, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.float()


def _reflection(d: int) -> torch.Tensor:
    r = torch.eye(d)
    r[0, 0] = -1.0
    return r


def _problem(d: int, n: int = 160, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn(1, n, d, generator=g) * (1.0 + torch.arange(d, dtype=torch.float32))
    nrm = torch.nn.functional.normalize(torch.randn(1, n, d, generator=g), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, d, generator=g), dim=-1)
    w = torch.rand(1, n, generator=g) + 0.5
    return pts, nrm, drv, w


def _model(d: int, outputs, **kw):
    torch.manual_seed(1)
    return ISLA(
        outputs=outputs, global_inputs={"g": 1}, hidden=32, n_layers=2, n_slices=8, dim=d, **kw
    )


OUT2 = {"p": 0, "v": 1, "w": {"rank": 1, "parity": "odd"}, "t": {"rank": 2, "symmetric": True},
        "a": {"rank": 2, "parity": "odd"}}


def test_levi_civita_symbol_and_pseudovector_counts():
    e2 = _levi_civita(2)
    assert torch.equal(e2, torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64))
    e3 = _levi_civita(3)
    assert e3[0, 1, 2] == 1 and e3[0, 2, 1] == -1 and e3[1, 2, 0] == 1 and e3[0, 0, 1] == 0
    v = [torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])]
    ps = _pseudovectors(v, 2)
    assert len(ps) == 2 and torch.equal(ps[0], torch.tensor([[0.0, 1.0]])) and torch.equal(ps[1], torch.tensor([[-1.0, 0.0]]))
    v3 = [torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[0.0, 1.0, 0.0]]), torch.tensor([[0.0, 0.0, 1.0]])]
    assert len(_pseudovectors(v3, 3)) == 3 and _pseudovectors(v3, 4) == []
    m2 = _model(2, {"p": 0})
    m3 = _model(3, {"p": 0})
    assert (m2.n_true_basis, m2.n_pseudo_basis) == (5, 3)
    assert (m3.n_true_basis, m3.n_pseudo_basis) == (5, 3)
    m2k = _model(2, {"p": 0})
    assert m2k.output_layouts["p"].dim == 2


def test_two_dimensional_contracts_rotation_translation_magnitude_and_reflection():
    """Rotation equivariance of every rank and parity under a random SO(2) map,
    translation invariance, magnitude invariance of the global direction, and the
    reflection behaviour: even fields transform as tensors, odd fields pick up the
    extra sign."""
    d = 2
    m = _model(d, OUT2)
    pts, nrm, drv, w = _problem(d)
    g = torch.Generator().manual_seed(3)
    q = _rotation(d, g)
    with torch.no_grad():
        base = m(points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w)
        rot = m(points=pts @ q.T, normals=nrm @ q.T, global_inputs={"g": drv @ q.T}, measure_weights=w)
        shifted = m(points=pts + 7.0, normals=nrm, global_inputs={"g": drv}, measure_weights=w)
        mag = m(points=pts, normals=nrm, global_inputs={"g": 3.5 * drv}, measure_weights=w)
        r = _reflection(d)
        ref = m(points=pts @ r.T, normals=nrm @ r.T, global_inputs={"g": drv @ r.T}, measure_weights=w)
    tol = dict(rtol=1e-4, atol=1e-5)
    assert base["p"].shape == (1, pts.shape[1]) and base["v"].shape == (1, pts.shape[1], 2)
    assert base["t"].shape == (1, pts.shape[1], 2, 2)
    torch.testing.assert_close(rot["p"], base["p"], **tol)
    torch.testing.assert_close(rot["v"], base["v"] @ q.T, **tol)
    torch.testing.assert_close(rot["w"], base["w"] @ q.T, **tol)
    torch.testing.assert_close(rot["t"], torch.einsum("ij,bnjk,lk->bnil", q, base["t"], q), **tol)
    torch.testing.assert_close(rot["a"], torch.einsum("ij,bnjk,lk->bnil", q, base["a"], q), **tol)
    for k in base.keys():
        ### a shift of 7 length units moves the float32 positions' roundoff by ~1e-5
        torch.testing.assert_close(shifted[k], base[k], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(mag[k], base[k], **tol)
    ### det(r) = -1: even fields transform as tensors, odd fields flip in addition.
    torch.testing.assert_close(ref["p"], base["p"], **tol)
    torch.testing.assert_close(ref["v"], base["v"] @ r.T, **tol)
    torch.testing.assert_close(ref["w"], -(base["w"] @ r.T), **tol)
    torch.testing.assert_close(ref["t"], torch.einsum("ij,bnjk,lk->bnil", r, base["t"], r), **tol)
    torch.testing.assert_close(ref["a"], -torch.einsum("ij,bnjk,lk->bnil", r, base["a"], r), **tol)
    ### the symmetric spec is symmetric, the odd rank-2 field is nonzero (its basis has
    ### the perpendiculars and the 2-D Levi-Civita symbol as content)
    torch.testing.assert_close(base["t"], base["t"].transpose(-1, -2))
    assert base["a"].abs().max() > 0 and base["w"].abs().max() > 0


def test_two_dimensional_gradients_and_interior_mode():
    d = 2
    pts, nrm, drv, w = _problem(d)
    ### interior query tokens with the internal nearest-boundary geometry in 2-D
    mq = _model(d, {"p": 0, "v": 1})
    q = torch.randn(1, 50, d) * 0.5
    out = mq(points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w, query_points=q)
    sum(v.square().mean() for v in out.values()).backward()
    missing = [k for k, p in mq.named_parameters() if p.grad is None]
    assert not missing, missing
    mq.eval()
    g = torch.Generator().manual_seed(5)
    rq = _rotation(d, g)
    with torch.no_grad():
        o1 = mq(points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w, query_points=q)
        o2 = mq(points=pts @ rq.T, normals=nrm @ rq.T, global_inputs={"g": drv @ rq.T}, measure_weights=w, query_points=q @ rq.T)
    assert o1["p"].shape == (1, 50) and o1["v"].shape == (1, 50, 2)
    torch.testing.assert_close(o2["p"], o1["p"], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(o2["v"], o1["v"] @ rq.T, rtol=1e-4, atol=1e-5)


def test_two_dimensional_rank_inputs_keep_the_contracts():
    """Rank-1 and rank-2 boundary inputs in two dimensions: the invariant seeds and
    the block contractions are built for d components."""
    d = 2
    pts, nrm, drv, w = _problem(d)
    g = torch.Generator().manual_seed(9)
    trac = torch.randn(1, pts.shape[1], d, generator=g)
    sym = torch.randn(1, pts.shape[1], d, d, generator=g)
    sym = sym + sym.transpose(-1, -2)
    m = _model(d, {"p": 0, "v": 1}, boundary_inputs={"trac": 1, "sig": {"rank": 2, "symmetric": True}})
    q = _rotation(d, g)
    with torch.no_grad():
        base = m(points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w, boundary_inputs={"trac": trac, "sig": sym})
        rot = m(points=pts @ q.T, normals=nrm @ q.T, global_inputs={"g": drv @ q.T}, measure_weights=w,
                boundary_inputs={"trac": trac @ q.T, "sig": torch.einsum("ij,bnjk,lk->bnil", q, sym, q)})
    torch.testing.assert_close(rot["p"], base["p"], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(rot["v"], base["v"] @ q.T, rtol=1e-4, atol=1e-5)


def test_four_dimensional_even_parity_smoke_and_odd_refused():
    d = 4
    pts, nrm, drv, w = _problem(d, n=120)
    m = _model(d, {"p": 0, "v": 1, "t": 2})
    assert m.n_pseudo_basis == 0 and m.output_layouts["t"].iso[0].shape == (4, 4)
    g = torch.Generator().manual_seed(11)
    q = _rotation(d, g)
    with torch.no_grad():
        base = m(points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w)
        rot = m(points=pts @ q.T, normals=nrm @ q.T, global_inputs={"g": drv @ q.T}, measure_weights=w)
    assert base["v"].shape == (1, 120, 4) and base["t"].shape == (1, 120, 4, 4)
    torch.testing.assert_close(rot["p"], base["p"], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(rot["v"], base["v"] @ q.T, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(rot["t"], torch.einsum("ij,bnjk,lk->bnil", q, base["t"], q), rtol=1e-4, atol=1e-5)
    with pytest.raises(NotImplementedError):
        _model(d, {"w": {"rank": 1, "parity": "odd"}})
    with pytest.raises(ValueError):
        ISLA(outputs={"p": 0}, dim=1)
    with pytest.raises(ValueError):
        ISLA(outputs={"p": 0}, global_inputs={"g": 1}, dim=2, geo_kernel="fused")


def test_dimension_mismatch_is_refused_and_two_d_is_not_embedded_three_d():
    pts, nrm, drv, w = _problem(3)
    m2 = _model(2, {"p": 0})
    with pytest.raises(ValueError):
        m2(points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w)
    ### The 3-D model with a K = 1 direction keeps its parameter count (the head has
    ### m = 5 true and p = 3 pseudo vectors in both dimensions for K = 1, so the
    ### 2-D model of the same spec has the same count too). 42549 = 41364 for the
    ### backbone and heads plus 1185 for the query side (qt.logw 1, qt.type_offset 32,
    ### qt.embed 2 x 32 + 32 + 32 x 32 + 32).
    m3 = _model(3, {"p": 0, "v": 1})
    assert sum(p.numel() for p in m3.parameters()) == 42549
    assert sum(p.numel() for p in _model(2, {"p": 0, "v": 1}).parameters()) == 42549
