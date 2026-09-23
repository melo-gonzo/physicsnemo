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

"""The internal interior geometry (``interior_geometry="nearest_boundary"``, 2026-09-23).

For every interior token the model derives its direction and signed distance from
the boundary sample itself instead of reading a stored signed-distance field: the
nearest boundary point gives the distance, the difference direction the "normal",
and the nearest sample's normal the sign (positive on the side the normal points
to). These tests pin the geometry against a brute-force computation, its sign
convention, the contracts in the internal mode, and the exact equivalence with
``interior_geometry="provided"`` fed the same quantities.
"""

import pytest
import torch

from physicsnemo.experimental.models import ISLA
from physicsnemo.experimental.models.isla.model import nearest_boundary_geometry

D = torch.float64
PV = {"p": 0, "v": 1}


def _cloud(n, seed, batch=1, scale=(3.0, 2.0, 1.0)):
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn(batch, n, 3, dtype=D, generator=g) * torch.tensor(scale, dtype=D)
    nrm = torch.nn.functional.normalize(
        torch.randn(batch, n, 3, dtype=D, generator=g), dim=-1
    )
    w = torch.rand(batch, n, dtype=D, generator=g) + 0.5
    return pts, nrm, w


def _brute_force(q, pts, nrm):
    """The full (B, Q, N) distance matrix, for a small case."""
    diff = q[:, :, None, :] - pts[:, None, :, :]
    dist = diff.norm(dim=-1)
    idx = dist.argmin(-1)
    d = dist.gather(-1, idx[..., None])[..., 0]
    y = pts.gather(1, idx[..., None].expand(*idx.shape, 3))
    n_star = nrm.gather(1, idx[..., None].expand(*idx.shape, 3))
    u = torch.nn.functional.normalize(q - y, dim=-1)
    sign = torch.where((u * n_star).sum(-1) >= 0, 1.0, -1.0).to(D)
    return sign * d, u


@pytest.mark.parametrize("chunk", [None, 1, 7, 1000])
def test_nearest_boundary_geometry_matches_brute_force(chunk):
    pts, nrm, _ = _cloud(150, 0, batch=2)
    q, _, _ = _cloud(45, 1, batch=2, scale=(4.0, 4.0, 4.0))
    d, u = nearest_boundary_geometry(q, pts, nrm, chunk=chunk)
    d_ref, u_ref = _brute_force(q, pts, nrm)
    assert d.shape == (2, 45) and u.shape == (2, 45, 3)
    assert torch.equal(d, d_ref) or torch.allclose(d, d_ref, atol=1e-14, rtol=0)
    assert torch.allclose(u, u_ref, atol=1e-14, rtol=0)
    assert torch.allclose(u.norm(dim=-1), torch.ones(2, 45, dtype=D), atol=1e-14)


def test_sign_convention_is_positive_on_the_normal_side():
    """A sphere with outward normals: queries outside have d > 0 and u radial outward,
    queries inside d < 0 and u radial inward, and |d| is within the sample spacing
    of the exact distance to the sphere."""
    g = torch.Generator().manual_seed(0)
    n = 20000
    dirs = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=D, generator=g), dim=-1
    )
    pts, nrm = 2.0 * dirs, dirs  # radius 2, outward normals
    q_dirs = torch.nn.functional.normalize(
        torch.randn(1, 30, 3, dtype=D, generator=g), dim=-1
    )
    radii = torch.cat(
        [torch.full((15,), 3.0, dtype=D), torch.full((15,), 1.0, dtype=D)]
    )
    q = q_dirs * radii[None, :, None]
    d, u = nearest_boundary_geometry(q, pts, nrm)
    assert (d[0, :15] > 0).all() and (d[0, 15:] < 0).all()
    ### the sample spacing on the sphere is about sqrt(4 pi r^2 / n) ~ 0.05
    assert torch.allclose(d[0, :15], torch.ones(15, dtype=D), atol=0.03)
    assert torch.allclose(d[0, 15:], -torch.ones(15, dtype=D), atol=0.03)
    assert ((u[0, :15] * q_dirs[0, :15]).sum(-1) > 0.99).all()
    assert ((u[0, 15:] * q_dirs[0, 15:]).sum(-1) < -0.99).all()


def test_on_wall_queries_take_the_sample_normal():
    pts, nrm, _ = _cloud(60, 0)
    d, u = nearest_boundary_geometry(pts[:, :10], pts, nrm)
    assert torch.equal(d, torch.zeros(1, 10, dtype=D))
    assert torch.allclose(u, nrm[:, :10], atol=1e-15)


def test_nearest_boundary_geometry_is_covariant_and_scales():
    pts, nrm, _ = _cloud(100, 0)
    q, _, _ = _cloud(20, 1, scale=(4.0, 4.0, 4.0))
    R, _ = torch.linalg.qr(torch.randn(3, 3, dtype=D))
    if torch.det(R) > 0:
        R[:, 0] = -R[:, 0]  # a reflection
    t = torch.tensor([1.0, -2.0, 0.5], dtype=D)
    d, u = nearest_boundary_geometry(q, pts, nrm)
    d2, u2 = nearest_boundary_geometry(q @ R.T + t, pts @ R.T + t, nrm @ R.T)
    assert torch.allclose(d2, d, atol=1e-12) and torch.allclose(u2, u @ R.T, atol=1e-12)
    d3, u3 = nearest_boundary_geometry(3.0 * q, 3.0 * pts, nrm)
    assert torch.allclose(d3, 3.0 * d, atol=1e-12) and torch.allclose(u3, u, atol=1e-12)


def test_geometry_is_differentiable_in_the_positions():
    pts, nrm, _ = _cloud(50, 0)
    q = _cloud(9, 1, scale=(4.0, 4.0, 4.0))[0].requires_grad_(True)
    d, u = nearest_boundary_geometry(q, pts, nrm)
    (d.sum() + u.sum()).backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


def _model(**kw):
    torch.manual_seed(0)
    return (
        ISLA(
            outputs=PV, global_inputs={"g": 1}, hidden=32, n_layers=2, n_slices=8, **kw
        )
        .double()
        .eval()
    )


@pytest.mark.parametrize(
    "kw",
    [
        dict(query_tokens=True),
        dict(query_independent=True, n_decoder_layers=1),
        dict(query_independent=True, n_decoder_layers=1, support_tokens=True),
        dict(query_tokens=True, query_inputs={"tag": 0, "flux": 1}),
    ],
    ids=["query_tokens", "passive", "support", "query_tokens+inputs"],
)
def test_internal_mode_equals_provided_mode_fed_its_own_geometry(kw):
    """The built-in distance channel has the layout of one declared rank-0 query
    input under query_scalar_scale="length" (the pair [d/L, sign(d) log|d|/L]), so
    a "provided" model with the same weights, handed the internally computed
    direction as query_normals and the internal distance as its first query input,
    is the same function (to roundoff: the provided path renormalizes the unit
    direction it is handed)."""
    internal = _model(**kw)
    declared = {"_d": 0, **kw.get("query_inputs", {})}
    provided = _model(**{**kw, "query_inputs": declared}, interior_geometry="provided")
    provided.load_state_dict(internal.state_dict())
    pts, nrm, w = _cloud(120, 0)
    drv = torch.tensor([[0.3, 1.0, 0.2]], dtype=D)
    q, _, _ = _cloud(25, 1, scale=(4.0, 4.0, 4.0))
    s, _, _ = _cloud(15, 2, scale=(4.0, 4.0, 4.0))
    qi = {}
    if "query_inputs" in kw:
        qi = {"tag": torch.rand(1, 25, dtype=D), "flux": torch.randn(1, 25, 3, dtype=D)}
    extra_i = dict(query_points=q, query_inputs=qi or None)
    d_q, u_q = nearest_boundary_geometry(q, pts, nrm)
    extra_p = dict(query_points=q, query_normals=u_q, query_inputs={"_d": d_q, **qi})
    if kw.get("support_tokens"):
        d_s, u_s = nearest_boundary_geometry(s, pts, nrm)
        extra_i.update(support_points=s)
        extra_p.update(
            support_points=s, support_normals=u_s, support_inputs={"_d": d_s}
        )
    common = dict(points=pts, normals=nrm, measure_weights=w, global_inputs={"g": drv})
    with torch.no_grad():
        a = internal(**common, **extra_i)
        b = provided(**common, **extra_p)
    assert all(torch.allclose(a[k], b[k], atol=1e-12, rtol=0) for k in a.keys())
    assert a["p"].shape == (1, 25)


def test_internal_mode_refuses_normals_and_provided_mode_requires_them():
    pts, nrm, w = _cloud(40, 0)
    q, qn, _ = _cloud(5, 1)
    common = dict(
        points=pts,
        normals=nrm,
        measure_weights=w,
        global_inputs={"g": torch.ones(1, 3, dtype=D)},
    )
    m = _model(query_tokens=True)
    with pytest.raises(ValueError, match="derives the query directions"):
        m(**common, query_points=q, query_normals=qn)
    m_sup = _model(query_independent=True, n_decoder_layers=1, support_tokens=True)
    with pytest.raises(ValueError, match="derives the support directions"):
        m_sup(**common, query_points=q, support_points=q, support_normals=qn)
    p = _model(query_tokens=True, interior_geometry="provided")
    with pytest.raises(ValueError, match="needs query_normals"):
        p(**common, query_points=q)
    with pytest.raises(ValueError, match="interior_geometry must be"):
        _model(interior_geometry="sdf")


def test_internal_mode_contracts_with_all_interior_arrangements():
    """Rotation and reflection covariance, translation invariance and geometric-scale
    invariance in the internal mode, where no direction is supplied for the queries;
    the queries of a subset decode identically on the passive path."""
    pts, nrm, w = _cloud(200, 0)
    drv = torch.tensor([[0.3, 1.0, 0.2]], dtype=D)
    q, _, _ = _cloud(40, 1, scale=(4.0, 4.0, 4.0))
    s, _, _ = _cloud(20, 2, scale=(4.0, 4.0, 4.0))
    for kw, extra in (
        (dict(query_tokens=True), dict(query_points=q)),
        (dict(query_independent=True, n_decoder_layers=1), dict(query_points=q)),
        (
            dict(query_independent=True, n_decoder_layers=1, support_tokens=True),
            dict(query_points=q, support_points=s),
        ),
    ):
        m = _model(**kw)

        def call(R, t, k=1.0):
            moved = {name: k * v @ R.T + t for name, v in extra.items()}
            with torch.no_grad():
                return m(
                    points=k * pts @ R.T + t,
                    normals=nrm @ R.T,
                    measure_weights=k * k * w,
                    global_inputs={"g": drv @ R.T},
                    **moved,
                )

        base = call(torch.eye(3, dtype=D), torch.zeros(3, dtype=D))
        for improper in (False, True):
            R, _ = torch.linalg.qr(
                torch.randn(3, 3, dtype=D, generator=torch.Generator().manual_seed(4))
            )
            if (torch.det(R) < 0) != improper:
                R[:, 0] = -R[:, 0]
            out = call(R, torch.tensor([2.0, -5.0, 1.0], dtype=D), k=2.5)
            assert torch.allclose(out["p"], base["p"], atol=1e-11), (kw, improper)
            assert torch.allclose(out["v"], base["v"] @ R.T, atol=1e-11), (kw, improper)
        ### the distance channel is live: moving the queries off the boundary changes p
        with torch.no_grad():
            far = m(
                points=pts,
                normals=nrm,
                measure_weights=w,
                global_inputs={"g": drv},
                **{**extra, "query_points": q * 3.0},
            )
        assert not torch.allclose(far["p"], base["p"], atol=1e-6)
        if kw.get("query_independent"):
            with torch.no_grad():
                sub = m(
                    points=pts,
                    normals=nrm,
                    measure_weights=w,
                    global_inputs={"g": drv},
                    **{**extra, "query_points": q[:, :7]},
                )
            assert torch.allclose(sub["p"], base["p"][:, :7], atol=1e-12, rtol=0)


def test_fused_kernel_stays_available_in_the_internal_mode():
    """The built-in distance enters the embedding, not the geometry region, so a model
    without rank-1 or rank-2 per-token inputs keeps the fused kernel."""
    m = ISLA(
        outputs=PV,
        global_inputs={"g": 1},
        hidden=32,
        n_layers=1,
        n_slices=8,
        query_tokens=True,
        geo_kernel="fused",
    )
    assert m.n_query_builtin == 2 and m.qt_embed[0].in_features == 2
    assert not (m.n_block_vecs or m.n_block_tens)


def test_checkpoint_round_trip_keeps_the_mode(tmp_path):
    from physicsnemo.core.module import Module

    m = _model(query_tokens=True, interior_nn_chunk=3)
    m.save(tmp_path / "m.mdlus")
    m2 = Module.from_checkpoint(str(tmp_path / "m.mdlus")).double().eval()
    assert m2.interior_geometry == "nearest_boundary" and m2.interior_nn_chunk == 3
    pts, nrm, w = _cloud(50, 0)
    q, _, _ = _cloud(8, 1)
    kw = dict(
        points=pts,
        normals=nrm,
        measure_weights=w,
        global_inputs={"g": torch.ones(1, 3, dtype=D)},
        query_points=q,
    )
    with torch.no_grad():
        a, b = m(**kw), m2(**kw)
    assert all(torch.equal(a[k], b[k]) for k in a.keys())


def test_nearest_boundary_geometry_near_wall_dense_cluster_matches_brute_force():
    """The matmul candidate search must return the true nearest sample where it is
    hardest: queries a hair off a dense cluster of boundary samples far from the
    cloud's centre (large |x|^2 relative to the separations), in float32 inputs,
    against a float64 brute force; every candidate count, including 1, must agree
    (the expansion runs in float64, so its resolution is below fp32 coordinate ties)."""
    g = torch.Generator().manual_seed(3)
    n, q_n = 5000, 400
    pts64, nrm64, _ = _cloud(n, 0)
    ### a dense cluster of 1,000 samples within 1e-3 of a point 4 radii out
    centre = torch.tensor([[[4.0, 0.0, 0.0]]], dtype=D)
    cluster = centre + 1e-3 * torch.randn(1, 1000, 3, dtype=D, generator=g)
    pts64 = torch.cat([pts64, cluster], dim=1)
    nrm64 = torch.cat(
        [nrm64, torch.nn.functional.normalize(torch.randn(1, 1000, 3, dtype=D, generator=g), dim=-1)], dim=1
    )
    q64 = centre + 2e-3 * torch.randn(1, q_n, 3, dtype=D, generator=g)
    pts, nrm, q = pts64.float(), nrm64.float(), q64.float()
    d_ref, u_ref = _brute_force(q.double(), pts.double(), nrm.double())
    for cand in (1, 2, 16):
        d, u = nearest_boundary_geometry(q, pts, nrm, chunk=64, candidates=cand)
        assert torch.allclose(d.double(), d_ref, atol=1e-9, rtol=1e-5), cand
        assert torch.allclose(u.double(), u_ref, atol=1e-5, rtol=0), cand
