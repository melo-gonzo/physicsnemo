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

"""Contract tests for ISLA (Invariant Slice Attention): every guarantee is exact by construction.

The model under test declares one scalar output ``p`` and one vector output ``v``
and one global direction ``g``; ``forward`` returns ``{"p": (B, N), "v": (B, N, 3)}``.
"""

import pytest
import torch

from physicsnemo.experimental.models import ISLA

# The frame is origin-free (positions enter only as point-to-anchor differences) with
# one data-derived length unit: the measure-weighted RMS pairwise distance
# (scale_mode="rms_distance", the default) or the square root of the total measure
# (scale_mode="total_measure"). Unless a test says otherwise it exercises the default.

PV = {"p": 0, "v": 1}
G = {"g": 1}


def _isla(**kw):
    return ISLA(outputs=PV, global_inputs=G, **kw)


def _same(a, b, atol, rtol=1e-5):
    """Two output dicts agree key by key."""
    assert a.keys() == b.keys()
    return all(torch.allclose(a[k], b[k], atol=atol, rtol=rtol) for k in a.keys())


def _rows(d, sl):
    """Slice the token axis of every output."""
    return {k: v[:, sl] for k, v in d.items()}


def _assert_rotated(rot, base, q, atol=1e-10):
    assert torch.allclose(rot["p"], base["p"], atol=atol)
    assert torch.allclose(rot["v"], base["v"] @ q.T, atol=atol)


def _rotation():
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


@pytest.fixture
def setup():
    torch.manual_seed(0)
    m = _isla(hidden=64, n_layers=3, n_slices=32).double().eval()
    n = 500
    ### Anisotropic cloud: the principal-axis frame is exactly covariant
    ### only where the covariance spectrum is non-degenerate (generic for
    ### vehicle geometry; an isotropic cloud is the degenerate corner).
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    with torch.no_grad():
        base = m(points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w)
    return m, pts, nrm, drv, w, base


def test_rotation_equivariance(setup):
    m, pts, nrm, drv, w, base = setup
    q = _rotation()
    with torch.no_grad():
        rot = m(
            points=pts @ q.T,
            normals=nrm @ q.T,
            global_inputs={"g": drv @ q.T},
            measure_weights=w,
        )
    _assert_rotated(rot, base, q)


def test_translation_invariance(setup):
    m, pts, nrm, drv, w, base = setup
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    with torch.no_grad():
        tr = m(
            points=pts + shift, normals=nrm, global_inputs={"g": drv}, measure_weights=w
        )
    assert _same(tr, base, atol=1e-10)


@pytest.mark.parametrize("k", [0.5, 2.0, 4.0])
def test_global_vector_magnitude_invariance(setup, k):
    """Rank-1 global inputs enter as unit directions: rescaling one leaves every
    output unchanged (a physically meaningful magnitude is a rank-0 global input)."""
    m, pts, nrm, drv, w, base = setup
    with torch.no_grad():
        sc = m(points=pts, normals=nrm, global_inputs={"g": drv * k}, measure_weights=w)
    assert _same(sc, base, atol=1e-10)


def test_measure_weight_scale_invariance(setup):
    m, pts, nrm, drv, w, base = setup
    with torch.no_grad():
        ws = m(
            points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w * 137.0
        )
    assert _same(ws, base, atol=1e-9)


def test_collated_input_shapes(setup):
    m, pts, nrm, drv, w, base = setup
    with torch.no_grad():
        out = m(
            points=pts,
            normals=nrm,
            global_inputs={"g": drv[:, None, :]},
            measure_weights=w[..., None].squeeze(-1),
        )
    assert _same(out, base, atol=1e-12)


@pytest.fixture
def setup_qi():
    torch.manual_seed(0)
    m = (
        _isla(
            hidden=64,
            n_layers=2,
            n_slices=16,
            query_independent=True,
            n_decoder_layers=2,
        )
        .double()
        .eval()
    )
    n = 400
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    return m, pts, nrm, drv, w


def test_query_set_independence(setup_qi):
    """THE v5a contract: same source, different query companions ->
    bitwise-identical predictions at shared queries."""
    m, pts, nrm, drv, w = setup_qi
    qa = pts[:, :50]
    q_big = torch.cat([qa, pts[:, 200:300]], dim=1)
    with torch.no_grad():
        out_small = m(
            points=pts,
            normals=nrm,
            global_inputs={"g": drv},
            measure_weights=w,
            query_points=qa,
        )
        out_big = m(
            points=pts,
            normals=nrm,
            global_inputs={"g": drv},
            measure_weights=w,
            query_points=q_big,
        )
    ### Mathematically exact; allclose(1e-12) rather than bitwise because
    ### GEMM tiling reorders reductions when the query count changes.
    assert _same(out_small, _rows(out_big, slice(0, 50)), atol=1e-12, rtol=0.0)


def test_qi_rotation_equivariance(setup_qi):
    m, pts, nrm, drv, w = setup_qi
    q = _rotation()
    with torch.no_grad():
        base = m(points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w)
        rot = m(
            points=pts @ q.T,
            normals=nrm @ q.T,
            global_inputs={"g": drv @ q.T},
            measure_weights=w,
        )
    _assert_rotated(rot, base, q)


def test_qi_global_vector_magnitude_invariance(setup_qi):
    m, pts, nrm, drv, w = setup_qi
    with torch.no_grad():
        base = m(points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w)
        sc = m(
            points=pts, normals=nrm, global_inputs={"g": drv * 2.0}, measure_weights=w
        )
    assert _same(sc, base, atol=1e-10)


def test_boundary_scalar_channel_contracts():
    torch.manual_seed(0)
    m = (
        _isla(
            hidden=64,
            n_layers=2,
            n_slices=16,
            boundary_inputs={"bc": {"rank": 0, "channels": 2}},
        )
        .double()
        .eval()
    )
    n = 300
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    bs = torch.randn(1, n, 2, dtype=torch.float64)
    with torch.no_grad():
        base = m(
            points=pts,
            normals=nrm,
            global_inputs={"g": drv},
            measure_weights=w,
            boundary_inputs={"bc": bs},
        )
    q = _rotation()
    with torch.no_grad():
        rot = m(
            points=pts @ q.T,
            normals=nrm @ q.T,
            global_inputs={"g": drv @ q.T},
            measure_weights=w,
            boundary_inputs={"bc": bs},
        )
    _assert_rotated(rot, base, q)


@pytest.mark.parametrize("kw", [{}, {"scale_mode": "total_measure"}])
def test_head_runs_under_bf16_autocast(kw):
    """Mixed-precision smoke: the head must survive bf16 autocast under both length
    units (the geometry is kept in the input precision by construction)."""
    torch.manual_seed(0)
    m = _isla(hidden=32, n_layers=1, n_slices=8, **kw)
    pts = torch.randn(1, 128, 3)
    nrm = torch.nn.functional.normalize(torch.randn(1, 128, 3), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3), dim=-1)
    w = torch.rand(1, 128) + 0.5
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w)
    sum(o.float().sum() for o in out.values()).backward()
    assert all(torch.isfinite(o.float()).all() for o in out.values())


def test_passive_interior_queries_contracts():
    """Passive decoding at off-surface query points under the default internal
    geometry (direction and signed distance from the nearest boundary sample):
    the prediction is exactly rotation- and translation-covariant, query-set
    independent and finite; query normals are refused in this mode, and the
    "provided" mode refuses distinct query points WITHOUT normals."""
    torch.manual_seed(0)
    n, nq = 400, 150
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    qpts = torch.randn(1, nq, 3, dtype=torch.float64) * 4.0  # interior/exterior points
    qnrm = torch.nn.functional.normalize(
        torch.randn(1, nq, 3, dtype=torch.float64), dim=-1
    )
    m = (
        _isla(
            hidden=64,
            n_layers=2,
            n_slices=16,
            query_independent=True,
            n_decoder_layers=2,
        )
        .double()
        .eval()
    )
    q = _rotation()
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    with torch.no_grad():
        base = m(
            points=pts,
            normals=nrm,
            global_inputs={"g": drv},
            measure_weights=w,
            query_points=qpts,
        )
        rot = m(
            points=pts @ q.T + shift,
            normals=nrm @ q.T,
            global_inputs={"g": drv @ q.T},
            measure_weights=w,
            query_points=qpts @ q.T + shift,
        )
        sub = m(
            points=pts,
            normals=nrm,
            global_inputs={"g": drv},
            measure_weights=w,
            query_points=qpts[:, :50],
        )
    assert base["p"].shape == (1, nq) and base["v"].shape == (1, nq, 3)
    assert all(torch.isfinite(o).all() for o in base.values())
    _assert_rotated(rot, base, q)
    assert _same(sub, _rows(base, slice(0, 50)), atol=1e-12, rtol=0.0)
    with pytest.raises(ValueError, match="derives the query directions"):
        with torch.no_grad():
            m(
                points=pts,
                normals=nrm,
                global_inputs={"g": drv},
                measure_weights=w,
                query_points=qpts,
                query_normals=qnrm,
            )
    provided = _isla(
        hidden=32,
        n_layers=1,
        n_slices=8,
        query_independent=True,
        n_decoder_layers=1,
        interior_geometry="provided",
    ).double()
    with pytest.raises(ValueError, match="needs query_normals"):
        with torch.no_grad():
            provided(
                points=pts,
                normals=nrm,
                global_inputs={"g": drv},
                measure_weights=w,
                query_points=qpts,
            )


def test_no_relational_geo_ablation_runs_and_differs():
    """A35b: use_relational_geo=False (Transolver-style feature-only slicing)
    stays exactly equivariant, runs, is finite, and changes the output."""
    torch.manual_seed(0)
    n = 300
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    q = _rotation()
    torch.manual_seed(1)
    ref = _isla(hidden=64, n_layers=2, n_slices=16).double().eval()
    torch.manual_seed(1)
    nogeo = (
        _isla(hidden=64, n_layers=2, n_slices=16, use_relational_geo=False)
        .double()
        .eval()
    )
    with torch.no_grad():
        o_ref = ref(
            points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w
        )
        o_ng = nogeo(
            points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w
        )
        o_ng_rot = nogeo(
            points=pts @ q.T,
            normals=nrm @ q.T,
            global_inputs={"g": drv @ q.T},
            measure_weights=w,
        )
    assert all(torch.isfinite(o).all() for o in o_ng.values())
    assert not _same(o_ng, o_ref, atol=1e-6)
    _assert_rotated(o_ng_rot, o_ng, q)


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"use_relational_geo": False},
        {"scale_mode": "total_measure"},
        {"geo_checkpoint": True},
        {
            "scale_mode": "total_measure",
            "query_independent": True,
            "n_decoder_layers": 1,
        },
    ],
)
def test_all_parameters_receive_gradients(kw):
    """DDP requires every parameter to take part in the loss; a module built
    but skipped in forward crashes distributed training (A35b nogeo incident)."""
    torch.manual_seed(0)
    m = _isla(hidden=32, n_layers=2, n_slices=8, **kw)
    pts = torch.randn(1, 128, 3)
    nrm = torch.nn.functional.normalize(torch.randn(1, 128, 3), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3), dim=-1)
    w = torch.rand(1, 128) + 0.5
    out = m(points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w)
    sum(o.sum() for o in out.values()).backward()
    unused = [n for n, p in m.named_parameters() if p.grad is None]
    assert not unused, unused


def test_geo_pool_then_project_is_exact():
    """Pooling the relational invariants over slices and then projecting is
    exactly the old project-then-pool (the projection is affine and the slice
    mix is a softmax over slices, so the bias passes through unchanged). This
    is the identity behind the memory saving: the saved activation is
    (B, N, geo) instead of (B, N, S, hidden/2)."""
    from physicsnemo.experimental.models.isla.model import _ReadBlock, _SliceBlock

    torch.manual_seed(0)
    for blk in (_SliceBlock(64, 32).double(), _ReadBlock(64, 32).double()):
        geo = torch.randn(2, 50, 32, blk.n_geo, dtype=torch.float64)
        mix = torch.softmax(torch.randn(2, 50, 32, dtype=torch.float64), dim=-1)
        old = torch.einsum("bns,bnsg->bng", mix, blk.geo_feat(geo))
        new = blk.geo_feat(torch.einsum("bns,bnsg->bng", mix, geo))
        assert torch.allclose(old, new, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("extra", [{}, {"query_independent": True}])
def test_geo_checkpoint_is_exact(extra):
    """Recomputing the per-slice invariants in backward (geo_checkpoint=True)
    changes what autograd stores, not what it computes: identical forward
    output and gradients (to roundoff) against the stored-activation path,
    for the encoder blocks and the passive decoder blocks."""
    torch.manual_seed(0)
    kw = dict(hidden=32, n_layers=2, n_slices=8, **extra)
    ref = _isla(**kw).double()
    ckp = _isla(geo_checkpoint=True, **kw).double()
    ckp.load_state_dict(ref.state_dict())
    pts = torch.randn(1, 40, 3, dtype=torch.float64)
    nrm = torch.nn.functional.normalize(
        torch.randn(1, 40, 3, dtype=torch.float64), dim=-1
    )
    drive = torch.tensor([[1.0, 0.2, 0.0]], dtype=torch.float64)
    w = torch.rand(1, 40, dtype=torch.float64) + 0.5
    fk = {}
    if extra:
        fk = dict(query_points=torch.randn(1, 17, 3, dtype=torch.float64))
    outs = []
    for m in (ref, ckp):
        out = m(
            points=pts,
            normals=nrm,
            global_inputs={"g": drive},
            measure_weights=w,
            **fk,
        )
        sum(o.square().sum() for o in out.values()).backward()
        outs.append(out)
    assert all(torch.equal(outs[0][k], outs[1][k]) for k in outs[0].keys())
    for (n, p), (_, q) in zip(ref.named_parameters(), ckp.named_parameters()):
        assert p.grad is not None and q.grad is not None, n
        assert torch.allclose(p.grad, q.grad, atol=1e-11, rtol=1e-9), n


def test_query_scalars_contracts():
    """Per-query rank-0 inputs on the query tokens (e.g. the signed distance
    to the wall): exact rotation/translation covariance is untouched (scalars
    are invariants), the channel changes the output and receives gradients,
    geometric-scale equivariance holds with the "length" scaling of the
    scalars (they are divided by the frame's length unit), and misuse raises."""
    torch.manual_seed(0)
    kw = dict(
        hidden=32,
        n_layers=2,
        n_slices=8,
        query_tokens=True,
        query_inputs={"sdf": 0},
        interior_geometry="provided",
    )
    m = _isla(**kw).double()
    pts = torch.randn(1, 40, 3, dtype=torch.float64)
    nrm = torch.nn.functional.normalize(
        torch.randn(1, 40, 3, dtype=torch.float64), dim=-1
    )
    drive = torch.tensor([[1.0, 0.3, 0.0]], dtype=torch.float64)
    w = torch.rand(1, 40, dtype=torch.float64) + 0.5
    q = torch.randn(1, 12, 3, dtype=torch.float64)
    qn = torch.nn.functional.normalize(
        torch.randn(1, 12, 3, dtype=torch.float64), dim=-1
    )
    sdf = torch.randn(1, 12, dtype=torch.float64) * 0.3
    out = m(
        points=pts,
        normals=nrm,
        global_inputs={"g": drive},
        measure_weights=w,
        query_points=q,
        query_normals=qn,
        query_inputs={"sdf": sdf},
    )
    assert out["p"].shape == (1, 12) and out["v"].shape == (1, 12, 3)
    ### scalar channel is live
    out2 = m(
        points=pts,
        normals=nrm,
        global_inputs={"g": drive},
        measure_weights=w,
        query_points=q,
        query_normals=qn,
        query_inputs={"sdf": sdf * 2},
    )
    assert not _same(out, out2, atol=1e-8)
    ### rotation + translation covariance (scalars ride along unchanged)
    R = _rotation()
    t = torch.tensor([0.4, -1.1, 2.0], dtype=torch.float64)
    rot = m(
        points=pts @ R.T + t,
        normals=nrm @ R.T,
        global_inputs={"g": drive @ R.T},
        measure_weights=w,
        query_points=q @ R.T + t,
        query_normals=qn @ R.T,
        query_inputs={"sdf": sdf},
    )
    _assert_rotated(rot, out, R)
    ### geometric-scale equivariance: lengths scale, so must the scalar
    s = 3.7
    sc = m(
        points=pts * s,
        normals=nrm,
        global_inputs={"g": drive},
        measure_weights=w * s**2,
        query_points=q * s,
        query_normals=qn,
        query_inputs={"sdf": sdf * s},
    )
    assert _same(sc, out, atol=1e-10)
    ### gradients reach the query-input embedding
    sum(o.square().sum() for o in out.values()).backward()
    assert all(
        p.grad is not None and p.grad.abs().sum() > 0 for p in m.qt_embed.parameters()
    )
    with pytest.raises(ValueError):
        m(
            points=pts,
            normals=nrm,
            global_inputs={"g": drive},
            measure_weights=w,
            query_points=q,
            query_normals=qn,
        )
    with pytest.raises(ValueError, match="query_inputs require"):
        _isla(hidden=32, n_layers=1, n_slices=8, query_inputs={"sdf": 0})


def test_query_tokens_contracts():
    """Query-token mode (interior queries as interacting tokens): exactly
    rotation/translation-covariant, finite, live (differs from the passive
    interior decode), and -- by design -- NOT query-set independent."""
    torch.manual_seed(0)
    n, nq = 400, 150
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    qpts = torch.randn(1, nq, 3, dtype=torch.float64) * 4.0
    m = _isla(hidden=64, n_layers=2, n_slices=16, query_tokens=True).double().eval()
    q = _rotation()
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    with torch.no_grad():
        base = m(
            points=pts,
            normals=nrm,
            global_inputs={"g": drv},
            measure_weights=w,
            query_points=qpts,
        )
        rot = m(
            points=pts @ q.T + shift,
            normals=nrm @ q.T,
            global_inputs={"g": drv @ q.T},
            measure_weights=w,
            query_points=qpts @ q.T + shift,
        )
        sub = m(
            points=pts,
            normals=nrm,
            global_inputs={"g": drv},
            measure_weights=w,
            query_points=qpts[:, :50],
        )
    assert base["p"].shape == (1, nq) and base["v"].shape == (1, nq, 3)
    assert all(torch.isfinite(o).all() for o in base.values())
    _assert_rotated(rot, base, q)
    # interacting mode: predictions at shared queries depend on the query set
    assert not _same(sub, _rows(base, slice(0, 50)), atol=1e-6)
    # every parameter receives a gradient (DDP safety)
    m.train()
    out = m(
        points=pts,
        normals=nrm,
        global_inputs={"g": drv},
        measure_weights=w,
        query_points=qpts,
    )
    sum(o.square().mean() for o in out.values()).backward()
    missing = [k for k, p in m.named_parameters() if p.grad is None]
    assert not missing, missing
    with pytest.raises(ValueError):
        _isla(
            hidden=32,
            n_layers=1,
            n_slices=8,
            query_tokens=True,
            query_independent=True,
        )


@pytest.mark.parametrize(
    "extra", [{}, {"query_inputs": {"sdf": 0}, "query_scalar_scale": "length"}]
)
def test_query_mass_source_total_refinement_invariance(extra):
    """Splitting every source token into two half-weight copies leaves the
    discrete source measure unchanged (positions, normals, total area, every
    integral). With query_mass="source_total" (the default) the query tokens'
    weight is a fraction of the total source measure, so the output is exactly
    unchanged and the geometric-scale contract still holds; the
    "geometric_mean" option is refinement-dependent and is asserted to differ."""
    torch.manual_seed(42)
    pts = torch.randn(1, 60, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(torch.randn_like(pts), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, 60, dtype=torch.float64) + 0.5
    q = pts[:, :17]
    fk = dict(query_points=q)
    if extra:
        fk["query_inputs"] = {"sdf": q.norm(dim=-1)}
    kw = dict(
        hidden=32,
        n_layers=2,
        n_slices=8,
        query_tokens=True,
        **extra,
    )
    torch.manual_seed(0)
    total = _isla(**kw).double().eval()
    torch.manual_seed(0)
    default = _isla(query_mass="geometric_mean", **kw).double().eval()
    refined = dict(
        points=pts.repeat_interleave(2, 1),
        normals=nrm.repeat_interleave(2, 1),
        global_inputs={"g": drv},
        measure_weights=w.repeat_interleave(2, 1) / 2,
    )
    s = 3.7
    scaled_fk = {**fk, "query_points": q * s}
    if extra:
        scaled_fk["query_inputs"] = {"sdf": fk["query_inputs"]["sdf"] * s}
    with torch.no_grad():
        a = total(
            points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w, **fk
        )
        b = total(**refined, **fk)
        c = total(
            points=pts * s,
            normals=nrm,
            global_inputs={"g": drv},
            measure_weights=w * s**2,
            **scaled_fk,
        )
        a0 = default(
            points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w, **fk
        )
        b0 = default(**refined, **fk)
    assert _same(b, a, atol=1e-10, rtol=0.0)
    assert _same(c, a, atol=1e-10, rtol=0.0)
    ### the geometric-mean option is refinement-dependent; the two conventions are
    ### not the same model
    assert not _same(b0, a0, atol=1e-3)
    assert not _same(a0, a, atol=1e-6)
    with pytest.raises(ValueError):
        _isla(
            hidden=32,
            n_layers=1,
            n_slices=8,
            query_tokens=True,
            query_mass="mean",
        )


@pytest.mark.parametrize("scale_mode", ["rms_distance", "total_measure"])
@pytest.mark.parametrize(
    "extra", [{}, {"query_independent": True, "n_decoder_layers": 2}]
)
def test_geometric_scale_equivariance(scale_mode, extra):
    """A similarity of the geometry (points x k + shift, areas x k^2) leaves the
    output exactly unchanged under either length unit, for the encoder alone and
    with the passive decoder reading queries that are moved along."""
    torch.manual_seed(0)
    n = 400
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    m = (
        _isla(hidden=64, n_layers=2, n_slices=16, scale_mode=scale_mode, **extra)
        .double()
        .eval()
    )
    k = 2.7
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    fk = dict(query_points=pts[:, :50]) if extra else {}
    fk_sc = {**fk, "query_points": k * fk["query_points"] + shift} if extra else {}
    with torch.no_grad():
        a = m(
            points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w, **fk
        )
        b = m(
            points=k * pts + shift,
            normals=nrm,
            global_inputs={"g": drv},
            measure_weights=k * k * w,
            **fk_sc,
        )
    assert _same(b, a, atol=1e-10, rtol=0.0)


def test_passive_decode_boundary_scalars():
    """Audit 2026-09-08: boundary inputs are per-boundary-cell data. Passive
    decoding of the boundary itself (query_points=None) carries them into the
    query seeds and runs; distinct query points carry zeros in the channel
    (GLOBAL INPUTS, 2026-09-14) and decode without a shape error."""
    torch.manual_seed(0)
    pts = torch.randn(1, 60, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(torch.randn_like(pts), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, 60, dtype=torch.float64) + 0.5
    bs = torch.randn(1, 60, 2, dtype=torch.float64)
    m = (
        _isla(
            hidden=32,
            n_layers=2,
            n_slices=8,
            query_independent=True,
            n_decoder_layers=2,
            boundary_inputs={"bc": {"rank": 0, "channels": 2}},
        )
        .double()
        .eval()
    )
    with torch.no_grad():
        out = m(
            points=pts,
            normals=nrm,
            global_inputs={"g": drv},
            measure_weights=w,
            boundary_inputs={"bc": bs},
        )
        out2 = m(
            points=pts,
            normals=nrm,
            global_inputs={"g": drv},
            measure_weights=w,
            boundary_inputs={"bc": bs * 2},
        )
    assert out["p"].shape == (1, 60) and out["v"].shape == (1, 60, 3)
    assert all(torch.isfinite(o).all() for o in out.values())
    assert not _same(out, out2, atol=1e-6)  # the channel is live on the query side
    with torch.no_grad():
        out_q = m(
            points=pts,
            normals=nrm,
            global_inputs={"g": drv},
            measure_weights=w,
            boundary_inputs={"bc": bs},
            query_points=pts[:, :17] * 0.5,
        )
    assert out_q["p"].shape == (1, 17) and out_q["v"].shape == (1, 17, 3)
    assert all(torch.isfinite(o).all() for o in out_q.values())


def _biased_poisson_subsample(pts, w, n_expected, bias, generator):
    """10:1 Poisson subsample with exact Horvitz-Thompson weights, mirroring the
    recipe's PoissonBiasedSubsampleMesh (bias toward x below the median)."""
    x = pts[0, :, 0]
    b = torch.where(x < x.median(), torch.full_like(x, bias), torch.ones_like(x))
    pi = (n_expected / b.sum() * b).clamp(max=1.0)
    keep = torch.rand(x.shape[0], dtype=pi.dtype, generator=generator) < pi
    idx = keep.nonzero(as_tuple=True)[0]
    return pts[:, idx], (w[:, idx] / pi[idx]), idx


def _frame_cloud(n=300, seed=0):
    torch.manual_seed(seed)
    pts = (
        torch.randn(1, n, 3, dtype=torch.float64)
        * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
        + 5.0
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    return pts, nrm, drv, w


def test_frame_contracts():
    """The origin-free frame: exact translation invariance without centering
    (1e-12, fp64), rotation equivariance, the feature widths (one seed invariant
    n.g, six relational invariants in the slice blocks and the passive read
    blocks), geometric scale equivariance under scale_mode='total_measure'
    (points x s, weights x s^2), and the refused options."""
    pts, nrm, drv, w = _frame_cloud()
    shift = torch.tensor([300.0, -70.0, 1100.0], dtype=torch.float64)
    torch.manual_seed(1)
    m = (
        _isla(
            hidden=64,
            n_layers=2,
            n_slices=16,
            query_independent=True,
            n_decoder_layers=1,
        )
        .double()
        .eval()
    )
    torch.manual_seed(1)
    m_tm = (
        _isla(hidden=64, n_layers=2, n_slices=16, scale_mode="total_measure")
        .double()
        .eval()
    )
    assert m.embed[0].in_features == 1
    assert m.blocks[0].geo_logit.in_features == 6
    assert m.read_blocks[0].geo_logit.in_features == 6
    with torch.no_grad():
        for model in (m, m_tm):
            base = model(
                points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w
            )
            assert all(torch.isfinite(o).all() for o in base.values())
            assert _same(
                model(
                    points=pts + shift,
                    normals=nrm,
                    global_inputs={"g": drv},
                    measure_weights=w,
                ),
                base,
                atol=1e-12,
            )
            q = _rotation()
            rot = model(
                points=pts @ q.T,
                normals=nrm @ q.T,
                global_inputs={"g": drv @ q.T},
                measure_weights=w,
            )
            _assert_rotated(rot, base, q)
        base = m_tm(
            points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w
        )
        assert _same(
            m_tm(
                points=pts * 2.5,
                normals=nrm,
                global_inputs={"g": drv},
                measure_weights=w * 2.5**2,
            ),
            base,
            atol=1e-10,
        )
        assert not _same(
            m_tm(
                points=pts * 2.5,
                normals=nrm,
                global_inputs={"g": drv},
                measure_weights=w,
            ),
            base,
            atol=1e-3,
        )  # the total measure IS the scale
        with pytest.raises(ValueError):
            m_tm(
                points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=None
            )
    with pytest.raises(ValueError, match="scale_mode must be"):
        _isla(hidden=64, n_layers=2, n_slices=16, scale_mode="reference_length")


@pytest.mark.parametrize("scale_mode", ["rms_distance", "total_measure"])
def test_frame_is_sampling_consistent(scale_mode):
    """The discriminating contract of the origin-free frame. Under a 10:1 biased
    Poisson subsample with exact HT weights, a model whose frame carries no
    sample statistic moves its predictions at fixed queries between the uniform
    and the biased draw by no more than the measure noise -- the same order as
    between two independent uniform draws."""
    torch.manual_seed(0)
    n_full = 100000
    pts = torch.randn(1, n_full, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n_full, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n_full, dtype=torch.float64) + 0.5
    q = pts[:, :40]
    ### local_readout_rho 1 keeps the passive readout a smooth measure-weighted
    ### average on this volumetric toy cloud (its surface default, 0.02, underflows
    ### to the clamp here and its noise would swamp the frame).
    torch.manual_seed(1)
    m = (
        _isla(
            hidden=32,
            n_layers=2,
            n_slices=8,
            query_independent=True,
            n_decoder_layers=1,
            local_readout_rho=1.0,
            scale_mode=scale_mode,
        )
        .double()
        .eval()
    )
    n_sub = 10000
    d_bias, d_noise = [], []

    def _flat(o):
        return torch.cat([o["p"][..., None], o["v"]], dim=-1)

    for seed in range(4):
        g = torch.Generator().manual_seed(100 + seed)
        p_u, w_u, iu = _biased_poisson_subsample(pts, w, n_sub, 1.0, g)
        p_u2, w_u2, iu2 = _biased_poisson_subsample(pts, w, n_sub, 1.0, g)
        p_b, w_b, ib = _biased_poisson_subsample(pts, w, n_sub, 10.0, g)
        with torch.no_grad():
            o_u = m(
                points=p_u,
                normals=nrm[:, iu],
                global_inputs={"g": drv},
                measure_weights=w_u,
                query_points=q,
            )
            o_u2 = m(
                points=p_u2,
                normals=nrm[:, iu2],
                global_inputs={"g": drv},
                measure_weights=w_u2,
                query_points=q,
            )
            o_b = m(
                points=p_b,
                normals=nrm[:, ib],
                global_inputs={"g": drv},
                measure_weights=w_b,
                query_points=q,
            )
        d_bias.append(float((_flat(o_b) - _flat(o_u)).norm()))
        d_noise.append(float((_flat(o_u2) - _flat(o_u)).norm()))
    ### measure-noise level: the biased draw's HT estimates have a larger variance
    ### than a uniform draw's (10x fewer points in the undersampled half), so the
    ### bias-vs-uniform difference may exceed the uniform-vs-uniform one by a
    ### bounded factor -- not the order of magnitude a centroid shift produces.
    ratio = sum(d_bias) / sum(d_noise)
    assert ratio < 4.0, (ratio, d_bias, d_noise)


def test_default_is_rms_distance():
    """The class default is the reference configuration: the RMS-pairwise-distance
    length unit (positions divided by the measure-weighted root-mean-square
    pairwise distance of the sample; validated and adopted 2026-09-18, LEN-RMS),
    one seed invariant n.g and six relational invariants. Checkpoints store their
    constructor arguments, so models saved under the earlier total-measure default
    are unaffected; this test pins the default itself."""
    m = _isla(hidden=32, n_layers=2, n_slices=8)
    assert m.scale_mode == "rms_distance"
    assert (
        m.geo_kernel == "auto"
    )  # 2026-09-19 ruling: fused wherever it can run, eager otherwise
    assert m.embed[0].in_features == 1
    assert m.blocks[0].geo_logit.in_features == 6


def test_constructor_and_forward_are_keyword_only():
    """Every constructor argument and every forward input is keyword-only
    (ruling 2026-09-11: inputs read as the recipe's forward_kwargs mapping and
    models can be hot-swapped without positional bookkeeping)."""
    pts = torch.randn(1, 16, 3)
    nrm = torch.nn.functional.normalize(torch.randn(1, 16, 3), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3), dim=-1)
    w = torch.rand(1, 16) + 0.5
    with pytest.raises(TypeError):
        ISLA(PV, G)
    with pytest.raises(TypeError):
        ISLA(hidden=32, n_layers=1, n_slices=8)  # outputs is required
    m = _isla(hidden=32, n_layers=1, n_slices=8)
    with pytest.raises(TypeError):
        m(pts, nrm, {"g": drv}, w)
    out = m(points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w)
    assert out["p"].shape == (1, 16) and out["v"].shape == (1, 16, 3)
    assert all(torch.isfinite(o).all() for o in out.values())


def test_unknown_constructor_argument_raises_type_error():
    with pytest.raises(TypeError, match="unexpected keyword argument 'nonsense'"):
        _isla(hidden=32, n_layers=1, n_slices=8, nonsense=1)


def test_rms_distance_scale_contracts():
    """LEN-RMS: the RMS-pairwise-distance length unit keeps exact translation
    invariance and rotation covariance, is invariant to a geometric rescale of the
    points (with or without the matching weight rescale) AND to a uniform rescale
    of the weights alone (which the total-measure unit is not), and needs no
    centre."""
    pts, nrm, drv, w = _frame_cloud()
    torch.manual_seed(1)
    m = (
        _isla(hidden=64, n_layers=2, n_slices=16, scale_mode="rms_distance")
        .double()
        .eval()
    )
    shift = torch.tensor([300.0, -70.0, 1100.0], dtype=torch.float64)
    gi = {"g": drv}
    with torch.no_grad():
        base = m(points=pts, normals=nrm, global_inputs=gi, measure_weights=w)
        assert all(torch.isfinite(o).all() for o in base.values())
        assert _same(
            m(points=pts + shift, normals=nrm, global_inputs=gi, measure_weights=w),
            base,
            atol=1e-12,
        )
        q = _rotation()
        rot = m(
            points=pts @ q.T,
            normals=nrm @ q.T,
            global_inputs={"g": drv @ q.T},
            measure_weights=w,
        )
        _assert_rotated(rot, base, q)
        assert _same(
            m(
                points=pts * 2.5,
                normals=nrm,
                global_inputs=gi,
                measure_weights=w * 2.5**2,
            ),
            base,
            atol=1e-10,
        )
        assert _same(
            m(points=pts, normals=nrm, global_inputs=gi, measure_weights=w * 137.0),
            base,
            atol=1e-10,
        )  # weight rescale alone: invariant (the unit is normalized)
        assert _same(
            m(points=pts * 2.5, normals=nrm, global_inputs=gi, measure_weights=w),
            base,
            atol=1e-10,
        )  # a rescale of the points alone is invariant too: the unit is a length of the sample and the
        # routing bias log w is uniform-rescale invariant, so weights and positions decouple
    torch.manual_seed(1)
    m_tm = (
        _isla(hidden=64, n_layers=2, n_slices=16, scale_mode="total_measure")
        .double()
        .eval()
    )
    with torch.no_grad():
        base_tm = m_tm(points=pts, normals=nrm, global_inputs=gi, measure_weights=w)
        assert not _same(
            m_tm(points=pts, normals=nrm, global_inputs=gi, measure_weights=w * 137.0),
            base_tm,
            atol=1e-3,
        )  # the total-measure unit is NOT invariant to a weight rescale; the contrast this option removes


def _koch_curve(level: int):
    """Koch curve (plane z = 0) as a polyline of 4**level segments on the unit base."""
    import math

    pts = [torch.tensor([0.0, 0.0]), torch.tensor([1.0, 0.0])]
    rot = torch.tensor(
        [
            [math.cos(math.pi / 3), -math.sin(math.pi / 3)],
            [math.sin(math.pi / 3), math.cos(math.pi / 3)],
        ]
    )
    for _ in range(level):
        new = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            d = b - a
            p1 = a + d / 3
            p3 = a + 2 * d / 3
            p2 = p1 + rot @ (d / 3)
            new += [p1, p2, p3, b]
        pts = new
    xy = torch.stack(pts).double()
    mids = (xy[:-1] + xy[1:]) / 2
    seg = (xy[1:] - xy[:-1]).norm(dim=-1)
    return torch.cat([mids, torch.zeros(len(mids), 1, dtype=torch.float64)], dim=-1)[
        None
    ], seg[None]


def test_rms_distance_converges_on_koch_curve_where_total_measure_diverges():
    """LEN-RMS: on a fractal boundary refined dyadically, the total measure grows
    without bound ((4/3)^level for the Koch curve) while the RMS pairwise distance
    converges: the relative differences between successive levels fall below 1e-3
    from level 4 on, and the last two levels agree to 1e-4."""
    from physicsnemo.experimental.models.isla.model import rms_pairwise_distance

    totals, rms = [], []
    for level in range(0, 7):
        pts, w = _koch_curve(level)
        totals.append(w.sum().item())
        rms.append(rms_pairwise_distance(pts, w).item())
    for a, b in zip(totals[:-1], totals[1:]):
        assert (
            abs(b / a - 4.0 / 3.0) < 1e-6
        )  # the measure diverges geometrically (float64 polyline arithmetic)
    rel = [abs(b - a) / a for a, b in zip(rms[:-1], rms[1:])]
    assert all(r < 1e-3 for r in rel[4:]), rel
    assert rel[-1] < 1e-4, rel
    assert rms[-1] > 0.3  # a finite length of the order of the base


def test_geo_kernel_auto_resolves_to_eager_off_cuda_and_honors_explicit_choices():
    """geo_kernel=None (auto) picks the fused Triton region only where it can run: on CUDA with one to four
    global directions and at most 1024 slices. On CPU it must resolve to eager at the first forward pass, cache the
    decision, and leave the outputs identical to an explicit eager model with the same weights."""
    torch.manual_seed(0)
    pts = torch.randn(1, 64, 3, dtype=torch.float64)
    nrm = torch.nn.functional.normalize(
        torch.randn(1, 64, 3, dtype=torch.float64), dim=-1
    )
    drv = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float64)
    w = torch.rand(1, 64, dtype=torch.float64) + 0.1
    auto = _isla(hidden=32, n_layers=2, n_slices=8).double().eval()
    assert auto.geo_kernel == "auto" and auto.resolve_geo_kernel(pts) == "eager"
    eager = _isla(hidden=32, n_layers=2, n_slices=8, geo_kernel="eager").double().eval()
    eager.load_state_dict(auto.state_dict())
    with torch.no_grad():
        ya = auto(points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w)
        ye = eager(points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w)
    assert all(torch.equal(ya[k], ye[k]) for k in ya.keys())
    assert all(b.geo_kernel == "eager" for b in auto.blocks)
    assert auto._geo_kernel_cache[1] == "eager"
    # five global directions: auto silently stays eager (off CUDA it does anyway); an explicit fused request is
    # refused at construction (the kernel is unrolled for one to four)
    two = (
        ISLA(
            outputs=PV,
            global_inputs={"g1": 1, "g2": 1, "g3": 1, "g4": 1, "g5": 1},
            hidden=32,
            n_layers=2,
            n_slices=8,
        )
        .double()
        .eval()
    )
    assert two.resolve_geo_kernel(pts) == "eager"
    with pytest.raises(ValueError, match="one to four per-point vectors"):
        ISLA(
            outputs=PV,
            global_inputs={"g1": 1, "g2": 1, "g3": 1, "g4": 1, "g5": 1},
            hidden=32,
            n_layers=2,
            n_slices=8,
            geo_kernel="fused",
        )
    with pytest.raises(ValueError, match="geo_kernel must be"):
        _isla(hidden=32, n_layers=2, n_slices=8, geo_kernel="turbo")


def test_kernel_readout_recompute_matches_dense_gradient():
    """The passive decoder's local Gaussian readout rebuilds each query chunk's
    kernel in the backward pass instead of storing it; outputs and gradients must
    equal the dense formula to roundoff (float64)."""
    from physicsnemo.experimental.models.isla.model import _kernel_readout

    torch.manual_seed(0)
    b, nq, ns, d = 2, 300, 500, 8
    q_r = torch.randn(b, nq, 3, dtype=torch.float64, requires_grad=True)
    src_r = torch.randn(b, ns, 3, dtype=torch.float64, requires_grad=True)
    src_h = torch.randn(b, ns, d, dtype=torch.float64, requires_grad=True)
    src_logw = torch.randn(b, ns, dtype=torch.float64)
    rho = 0.7
    out = _kernel_readout(q_r, src_r, src_h, src_logw, rho, 1e-12)
    loss = (out * torch.linspace(0.5, 1.5, d, dtype=torch.float64)).sum()
    g_q, g_r, g_h = torch.autograd.grad(loss, (q_r, src_r, src_h))
    # dense reference
    q2, r2, h2 = (t.detach().clone().requires_grad_(True) for t in (q_r, src_r, src_h))
    d2 = torch.cdist(q2, r2).square()
    k = torch.softmax(-d2 / (rho * rho) + src_logw[:, None, :], dim=-1)
    ref = torch.einsum("bcn,bnh->bch", k, h2)
    ref_loss = (ref * torch.linspace(0.5, 1.5, d, dtype=torch.float64)).sum()
    r_q, r_r, r_h = torch.autograd.grad(ref_loss, (q2, r2, h2))
    assert torch.allclose(out, ref, atol=1e-12, rtol=0)
    for a, c in ((g_q, r_q), (g_r, r_r), (g_h, r_h)):
        assert torch.allclose(a, c, atol=1e-10, rtol=0)


def test_geo_bias_mlp_is_equivariant_and_runs_on_both_kernels():
    """geo_bias='mlp' (a nonlinear map of the pair invariants to the routing bias) keeps
    the contracts (rotation equivariance, translation invariance) because it consumes the
    same invariants; on CPU it resolves to the eager region, and the fused kernel accepts
    it (its exactness is tested in test_isla_kernels.py)."""
    torch.manual_seed(0)
    m = _isla(hidden=64, n_layers=2, n_slices=16, geo_bias="mlp").double().eval()
    n = 300
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn(1, n, 3, dtype=torch.float64), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    q = _rotation()
    with torch.no_grad():
        base = m(points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w)
        rot = m(points=pts @ q.T, normals=nrm @ q.T, global_inputs={"g": drv @ q.T}, measure_weights=w)
        sh = m(points=pts + 5.0, normals=nrm, global_inputs={"g": drv}, measure_weights=w)
    _assert_rotated(rot, base, q)
    assert all(torch.allclose(sh[k], base[k], atol=1e-9, rtol=1e-9) for k in base.keys())
    assert m.resolve_geo_kernel(pts) == "eager"  # CPU points: no Triton path
    fused = _isla(hidden=64, n_layers=1, n_slices=16, geo_bias="mlp", geo_kernel="fused")
    assert fused.geo_kernel == "fused" and fused.geo_bias == "mlp"
    with pytest.raises(ValueError, match="geo_bias must be"):
        _isla(hidden=64, n_layers=1, n_slices=16, geo_bias="quadratic")


def test_local_rho_aggregation_keeps_the_contracts_and_is_off_by_default():
    """local_rho (LOCAL-AGG, 2026-09-27): the measure-weighted Gaussian aggregation of
    token states and pair invariants before the slice routing keeps rotation
    equivariance, translation invariance and measure-rescale invariance (its kernel is
    a softmax over sources with log-weight logits, its features are pair invariants);
    a model built without it has no local parameters and rejects a non-positive radius."""
    torch.manual_seed(0)
    m = _isla(hidden=64, n_layers=2, n_slices=16, local_rho=0.05).double().eval()
    n = 300
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn(1, n, 3, dtype=torch.float64), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    q = _rotation()
    with torch.no_grad():
        base = m(points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w)
        rot = m(points=pts @ q.T, normals=nrm @ q.T, global_inputs={"g": drv @ q.T}, measure_weights=w)
        sh = m(points=pts + 5.0, normals=nrm, global_inputs={"g": drv}, measure_weights=w)
        sc = m(points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w * 2.5)
    _assert_rotated(rot, base, q)
    assert all(torch.allclose(sh[k], base[k], atol=1e-9, rtol=1e-9) for k in base.keys())
    assert all(torch.allclose(sc[k], base[k], atol=1e-9, rtol=1e-9) for k in base.keys())
    assert sum(1 for k, _ in m.named_parameters() if ".local_" in k) == 16  # norm, value, geo, out: 4 modules x (weight, bias) per block, 2 blocks
    plain = _isla(hidden=64, n_layers=2, n_slices=16)
    assert not any(".local_" in k for k, _ in plain.named_parameters()) and plain.local_rho is None
    with pytest.raises(ValueError, match="local_rho must be positive"):
        _isla(hidden=64, n_layers=1, n_slices=16, local_rho=0.0)


def test_local_k_nearest_form_equals_the_dense_form_at_k_equal_n():
    """local_k = N reproduces the dense O(N^2) local aggregation (local_k = 0) to
    roundoff; the default k = 32 keeps the contracts (covered above) and a negative k is
    rejected."""
    torch.manual_seed(0)
    n = 250
    dense = _isla(hidden=64, n_layers=2, n_slices=16, local_rho=0.05, local_k=0).double().eval()
    knn = _isla(hidden=64, n_layers=2, n_slices=16, local_rho=0.05, local_k=n).double().eval()
    knn.load_state_dict(dense.state_dict())
    pts = torch.randn(1, n, 3, dtype=torch.float64) * torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    nrm = torch.nn.functional.normalize(torch.randn(1, n, 3, dtype=torch.float64), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, n, dtype=torch.float64) + 0.5
    with torch.no_grad():
        a = dense(points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w)
        b = knn(points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w)
    assert all(torch.allclose(a[k], b[k], atol=1e-12, rtol=1e-12) for k in a.keys())
    with pytest.raises(ValueError, match="local_k must be"):
        _isla(hidden=64, n_layers=1, n_slices=16, local_rho=0.05, local_k=-1)


def test_scalar_only_model_has_no_frame_routing_and_every_parameter_gets_a_gradient():
    """A model whose outputs are all rank 0 never reads the token basis, so the head's
    frame routing is not built (a parameter without a gradient stops DDP at the second
    step; found on the OLBC Poisson lanes, 2026-09-27); with a vector output the frame
    routing exists. In both cases every parameter receives a gradient, and the
    scalar-only outputs equal the head coefficients directly."""
    torch.manual_seed(0)
    n = 200
    pts = torch.randn(1, n, 3) * torch.tensor([3.0, 2.0, 1.0])
    nrm = torch.nn.functional.normalize(torch.randn(1, n, 3), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3), dim=-1)
    w = torch.rand(1, n) + 0.5
    for outs in ({"u": 0}, {"u": 0, "q": {"rank": 0, "channels": 3}}, {"u": 0, "v": 1}):
        m = ISLA(outputs=outs, global_inputs={"g": 1}, hidden=32, n_layers=2, n_slices=8)
        has_vec = any(v == 1 or (isinstance(v, dict) and v.get("rank", 0) >= 1) for v in outs.values())
        assert m.needs_frame == has_vec and (m.frame_assign is not None) == has_vec
        out = m(points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w)
        sum(v.square().mean() for v in out.values()).backward()
        missing = [k for k, p in m.named_parameters() if p.grad is None]
        assert not missing, missing
        assert out["u"].shape == (1, n)
        if "q" in outs:
            assert out["q"].shape == (1, n, 3)
