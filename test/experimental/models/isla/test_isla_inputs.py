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

"""Contract tests for ISLA's rank-spec per-token inputs (2026-09-22).

Rank-1 and rank-2 boundary and query fields enter only through invariants formed
with true vectors (the normal, the global directions, the token-to-anchor
direction), so every rank-0 output stays invariant and every rank >= 1 output
stays covariant when the geometry AND the fields are rotated, reflected or
translated together; the checks are exact (float64, 1e-12).
"""

import pytest
import torch

from physicsnemo.experimental.models import ISLA

D = torch.float64
OUT = {"p": 0, "v": 1, "sig": {"rank": 2, "symmetric": True}}
BIN = {"trac": 1, "T": {"rank": 2, "symmetric": True}, "A": 2, "tag": 0}
QIN = {"sdf": 0, "u": 1, "S": {"rank": 2, "symmetric": True}}


def _cloud(n, seed):
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn(1, n, 3, dtype=D, generator=g) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=D
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=D, generator=g), dim=-1
    )
    w = torch.rand(1, n, dtype=D, generator=g) + 0.5
    return pts, nrm, w


def _fields(n, seed, symmetric_names, vector_names, scalar_names):
    g = torch.Generator().manual_seed(seed)
    f = {}
    for k in vector_names:
        f[k] = torch.randn(1, n, 3, dtype=D, generator=g)
    for k in symmetric_names:
        t = torch.randn(1, n, 3, 3, dtype=D, generator=g)
        f[k] = t + t.transpose(-1, -2)
    for k in scalar_names:
        f[k] = torch.rand(1, n, dtype=D, generator=g)
    return f


def _boundary_fields(n, seed=1):
    f = _fields(n, seed, ["T"], ["trac"], ["tag"])
    f["A"] = torch.randn(
        1, n, 3, 3, dtype=D, generator=torch.Generator().manual_seed(seed + 10)
    )
    return f


def _query_fields(n, seed=2):
    return _fields(n, seed, ["S"], ["u"], ["sdf"])


def _move(fields, specs, R, shift=None):
    """Transform every field by its rank: rank 0 unchanged, rank 1 ``v R^T``, rank 2
    ``R T R^T``."""
    out = {}
    for k, v in fields.items():
        r = specs[k].rank
        out[k] = v if r == 0 else (v @ R.T if r == 1 else R @ v @ R.T)
    return out


def _orthogonal(seed, improper):
    q, _ = torch.linalg.qr(
        torch.randn(3, 3, dtype=D, generator=torch.Generator().manual_seed(seed))
    )
    if (torch.det(q) < 0) != improper:
        q[:, 0] = -q[:, 0]
    return q


def _law(x, R, rank):
    y = x
    for k in range(rank):
        y = torch.movedim(torch.tensordot(y, R, dims=([2 + k], [1])), -1, 2 + k)
    return y


def _model(**kw):
    torch.manual_seed(0)
    base = dict(outputs=OUT, global_inputs={"g": 1}, hidden=32, n_layers=2, n_slices=8)
    base.update(kw)
    return ISLA(**base).double().eval()


def _assert_covariant(m, call, improper):
    """``call(R, shift)`` runs the model on the transformed problem."""
    base = call(torch.eye(3, dtype=D), torch.zeros(3, dtype=D))
    R = _orthogonal(4, improper)
    out = call(R, torch.tensor([2.0, -5.0, 1.0], dtype=D))
    for name, spec in m.output_specs.items():
        assert torch.allclose(out[name], _law(base[name], R, spec.rank), atol=1e-12), (
            name,
            improper,
        )
    return base


@pytest.mark.parametrize("improper", [False, True], ids=["rotation", "reflection"])
def test_boundary_inputs_covariance(improper):
    m = _model(boundary_inputs=BIN)
    pts, nrm, w = _cloud(200, 0)
    drv = torch.tensor([[0.3, 1.0, 0.2]], dtype=D)
    bf = _boundary_fields(200)

    def call(R, shift):
        with torch.no_grad():
            return m(
                points=pts @ R.T + shift,
                normals=nrm @ R.T,
                measure_weights=w,
                global_inputs={"g": drv @ R.T},
                boundary_inputs=_move(bf, m.boundary_specs, R),
            )

    base = _assert_covariant(m, call, improper)
    ### the fields are live: scaling the traction changes every output
    with torch.no_grad():
        other = m(
            points=pts,
            normals=nrm,
            measure_weights=w,
            global_inputs={"g": drv},
            boundary_inputs={**bf, "trac": 2.0 * bf["trac"]},
        )
    assert all(not torch.allclose(other[k], base[k], atol=1e-6) for k in base)
    ### a rotation of the fields ALONE (geometry fixed) is not a symmetry
    R = _orthogonal(6, False)
    with torch.no_grad():
        skew = m(
            points=pts,
            normals=nrm,
            measure_weights=w,
            global_inputs={"g": drv},
            boundary_inputs=_move(bf, m.boundary_specs, R),
        )
    assert not torch.allclose(skew["p"], base["p"], atol=1e-6)


@pytest.mark.parametrize("improper", [False, True], ids=["rotation", "reflection"])
@pytest.mark.parametrize(
    "mode",
    ["query_tokens", "passive_queries", "passive_boundary", "support"],
)
def test_query_inputs_covariance(mode, improper):
    """Rank-0/1/2 per-query fields on the interacting query tokens, on passive queries
    at distinct points, on passive queries that ARE the boundary points, and on
    support tokens (with their own field set)."""
    pts, nrm, w = _cloud(200, 0)
    drv = torch.tensor([[0.3, 1.0, 0.2]], dtype=D)
    q_pts, q_nrm, _ = _cloud(40, 3)
    kw = {
        "query_tokens": dict(query_tokens=True),
        "passive_queries": dict(query_independent=True, n_decoder_layers=1),
        "passive_boundary": dict(query_independent=True, n_decoder_layers=1),
        "support": dict(
            query_independent=True, n_decoder_layers=1, support_tokens=True
        ),
    }[mode]
    m = _model(query_inputs=QIN, boundary_inputs={"trac": 1}, **kw)
    bf = {
        "trac": torch.randn(
            1, 200, 3, dtype=D, generator=torch.Generator().manual_seed(5)
        )
    }
    n_q = 200 if mode == "passive_boundary" else 40
    qf = _query_fields(n_q)
    sf = _query_fields(30, seed=7)
    s_pts, s_nrm, _ = _cloud(30, 8)

    def call(R, shift):
        extra = dict(query_inputs=_move(qf, m.query_specs, R))
        if mode != "passive_boundary":
            extra.update(query_points=q_pts @ R.T + shift, query_normals=q_nrm @ R.T)
        if mode == "support":
            extra.update(
                support_points=s_pts @ R.T + shift,
                support_normals=s_nrm @ R.T,
                support_inputs=_move(sf, m.query_specs, R),
            )
        with torch.no_grad():
            return m(
                points=pts @ R.T + shift,
                normals=nrm @ R.T,
                measure_weights=w,
                global_inputs={"g": drv @ R.T},
                boundary_inputs=_move(bf, m.boundary_specs, R),
                **extra,
            )

    base = _assert_covariant(m, call, improper)
    assert base["p"].shape == (1, n_q) and base["sig"].shape == (1, n_q, 3, 3)
    ### the query vector field is live
    qf2 = {**qf, "u": 3.0 * qf["u"]}
    extra = dict(query_inputs=qf2)
    if mode != "passive_boundary":
        extra.update(query_points=q_pts, query_normals=q_nrm)
    if mode == "support":
        extra.update(support_points=s_pts, support_normals=s_nrm, support_inputs=sf)
    with torch.no_grad():
        other = m(
            points=pts,
            normals=nrm,
            measure_weights=w,
            global_inputs={"g": drv},
            boundary_inputs=bf,
            **extra,
        )
    assert not torch.allclose(other["p"], base["p"], atol=1e-6)


def test_geometry_width_and_seed_width():
    """One rank-1 input and one symmetric rank-2 input over K = 1: the geometry region
    gains P' = 1 + (1 + K) contracted vectors and Q = 1 quadratic form; the seed gains
    the frame cosines, the norm, the frame contractions, the trace and the norm."""
    for K in (0, 1, 2):
        m = _model(
            global_inputs={f"g{k}": 1 for k in range(K)},
            boundary_inputs={"v": 1, "T": {"rank": 2, "symmetric": True}},
        )
        F = 1 + K
        assert m.blocks[0].n_geo == 5 + K + (1 + F) + 1
        assert m.n_block_vecs == 1 + F and m.n_block_tens == 1
        assert m.bin.n_features == (F + 1) + (F * (F + 1) // 2 + 2)
        assert m.embed[0].in_features == K + m.bin.n_features
    ### a non-symmetric rank-2 input contracts both T a and T^T a and all F^2 pairs
    m = _model(boundary_inputs={"A": 2})
    assert m.n_block_vecs == 2 * 2 and m.bin.n_features == 4 + 2
    ### query inputs widen the union stack for every token, boundary ones included
    m = _model(query_tokens=True, query_inputs={"u": 1}, boundary_inputs={"v": 1})
    assert m.blocks[0].n_geo == 5 + 1 + 2


def test_field_validation_errors():
    m = _model(boundary_inputs={"trac": 1, "tag": 0})
    pts, nrm, w = _cloud(20, 0)
    common = dict(
        points=pts,
        normals=nrm,
        measure_weights=w,
        global_inputs={"g": torch.ones(1, 3, dtype=D)},
    )
    trac, tag = torch.randn(1, 20, 3, dtype=D), torch.rand(1, 20, dtype=D)
    with pytest.raises(ValueError, match="missing \\['tag'\\]"):
        m(**common, boundary_inputs={"trac": trac})
    with pytest.raises(ValueError, match="unexpected \\['extra'\\]"):
        m(**common, boundary_inputs={"trac": trac, "tag": tag, "extra": tag})
    with pytest.raises(ValueError, match="pass boundary_inputs"):
        m(**common)
    with pytest.raises(ValueError, match="expected a tensor of shape \\(1, 20, 3\\)"):
        m(**common, boundary_inputs={"trac": trac[..., :2], "tag": tag})
    m0 = _model()
    with pytest.raises(ValueError, match="declares no boundary_inputs"):
        m0(**common, boundary_inputs={"trac": trac})
    ### query fields are refused where the model declares none, and required where it does
    mq = _model(query_tokens=True, query_inputs={"sdf": 0})
    q, qn, _ = _cloud(7, 1)
    with pytest.raises(ValueError, match="pass query_inputs"):
        mq(**common, query_points=q, query_normals=qn)
    with pytest.raises(ValueError, match="unexpected"):
        mq(
            **common,
            query_points=q,
            query_normals=qn,
            query_inputs={"sdf": q[..., 0], "x": q[..., 0]},
        )


def test_fused_kernel_refuses_tensor_inputs_and_auto_falls_back():
    base = dict(outputs=OUT, global_inputs={"g": 1}, hidden=32, n_layers=1, n_slices=8)
    with pytest.raises(ValueError, match="does not support rank-1 or rank-2"):
        ISLA(boundary_inputs={"trac": 1}, geo_kernel="fused", **base)
    with pytest.raises(ValueError, match="does not support rank-1 or rank-2"):
        ISLA(query_inputs={"S": 2}, query_tokens=True, geo_kernel="fused", **base)
    ### rank-0 inputs never touch the geometry region: the fused kernel stays available
    ISLA(boundary_inputs={"tag": 0}, geo_kernel="fused", **base)
    m = ISLA(boundary_inputs={"trac": 1}, **base)
    assert m.geo_kernel == "auto"
    assert m.resolve_geo_kernel(torch.zeros(1, 4, 3)) == "eager"


def test_rank0_query_inputs_are_scaled_by_the_gauge():
    """With query_scalar_scale="length" a rank-0 query field is a length: rescaling the
    geometry (points x s, weights x s^2) together with the field leaves the output
    unchanged, while rank-1 query fields are never scaled (they are physical
    vectors, not lengths) and a rescale of the geometry alone changes the output."""
    m = _model(query_tokens=True, query_inputs={"sdf": 0, "u": 1})
    pts, nrm, w = _cloud(60, 0)
    q, qn, _ = _cloud(12, 1)
    drv = torch.tensor([[1.0, 0.3, 0.0]], dtype=D)
    sdf = torch.randn(1, 12, dtype=D) * 0.3
    u = torch.randn(1, 12, 3, dtype=D)
    s = 3.7

    def call(k_geo, k_sdf):
        with torch.no_grad():
            return m(
                points=k_geo * pts,
                normals=nrm,
                measure_weights=w * k_geo**2,
                global_inputs={"g": drv},
                query_points=k_geo * q,
                query_normals=qn,
                query_inputs={"sdf": k_sdf * sdf, "u": u},
            )

    base, scaled, geo_only = call(1.0, 1.0), call(s, s), call(s, 1.0)
    assert all(torch.allclose(scaled[k], base[k], atol=1e-10) for k in base)
    assert not torch.allclose(geo_only["p"], base["p"], atol=1e-6)
    ### query_scalar_scale="none" enters the raw value (one channel instead of the
    ### scaled pair, so its own parameters): the geometric rescale then changes it
    m_raw = _model(
        query_tokens=True, query_inputs={"sdf": 0, "u": 1}, query_scalar_scale="none"
    )
    assert m_raw.qt_embed[0].in_features == m.qt_embed[0].in_features - 1
    with torch.no_grad():
        a = m_raw(
            points=pts,
            normals=nrm,
            measure_weights=w,
            global_inputs={"g": drv},
            query_points=q,
            query_normals=qn,
            query_inputs={"sdf": sdf, "u": u},
        )
        b = m_raw(
            points=s * pts,
            normals=nrm,
            measure_weights=w * s**2,
            global_inputs={"g": drv},
            query_points=s * q,
            query_normals=qn,
            query_inputs={"sdf": s * sdf, "u": u},
        )
    assert not torch.allclose(a["p"], b["p"], atol=1e-6)


def test_geo_checkpoint_is_exact_with_tensor_inputs():
    """The recompute path carries the extra vectors and quadratic forms bitwise."""
    torch.manual_seed(0)
    ref = _model(boundary_inputs=BIN, query_tokens=True, query_inputs=QIN)
    ckp = _model(
        boundary_inputs=BIN, query_tokens=True, query_inputs=QIN, geo_checkpoint=True
    )
    ckp.load_state_dict(ref.state_dict())
    pts, nrm, w = _cloud(50, 0)
    q, qn, _ = _cloud(9, 1)
    kw = dict(
        points=pts,
        normals=nrm,
        measure_weights=w,
        global_inputs={"g": torch.tensor([[1.0, 0.2, 0.0]], dtype=D)},
        boundary_inputs=_boundary_fields(50),
        query_points=q,
        query_normals=qn,
        query_inputs=_query_fields(9),
    )
    outs = []
    for m in (ref, ckp):
        m.train()
        out = m(**kw)
        sum(o.square().sum() for o in out.values()).backward()
        outs.append(out)
    assert all(torch.equal(outs[0][k], outs[1][k]) for k in outs[0])
    for (n, p), (_, g) in zip(ref.named_parameters(), ckp.named_parameters()):
        assert p.grad is not None and g.grad is not None, n
        assert torch.allclose(p.grad, g.grad, atol=1e-11, rtol=1e-9), n
