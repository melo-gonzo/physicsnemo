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
# one data-derived length unit, the measure-weighted RMS pairwise distance.

PV = {"p": 0, "v": 1}
G = {"g": 1}

#: The query-side submodule: its parameters are unused in a boundary-only forward pass.
QUERY_PARAMS = ("qt",)


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


def _is_query_param(name: str) -> bool:
    return name.split(".")[0] in QUERY_PARAMS


@pytest.fixture
def setup():
    torch.manual_seed(0)
    m = _isla(hidden=64, n_layers=3, n_slices=32).double().eval()
    n = 500
    ### Anisotropic cloud: generic for vehicle geometry; an isotropic cloud is the
    ### degenerate corner.
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


def test_no_measure_weights_is_the_uniform_quadrature(setup):
    """``measure_weights=None`` gives every cell the same weight: the same output as
    any constant weight vector."""
    m, pts, nrm, drv, w, base = setup
    with torch.no_grad():
        none = m(points=pts, normals=nrm, global_inputs={"g": drv})
        const = m(
            points=pts,
            normals=nrm,
            global_inputs={"g": drv},
            measure_weights=torch.full_like(w, 0.37),
        )
    assert _same(none, const, atol=1e-10)
    assert not _same(none, base, atol=1e-3)


def test_boundary_scalar_contracts():
    """Two rank-0 boundary fields (one leaf per scalar) ride along under a rotation."""
    torch.manual_seed(0)
    m = (
        _isla(
            hidden=64,
            n_layers=2,
            n_slices=16,
            boundary_inputs={"bc_a": 0, "bc_b": 0},
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
    bs = {"bc_a": torch.randn(1, n, dtype=torch.float64), "bc_b": torch.randn(1, n, dtype=torch.float64)}
    with torch.no_grad():
        base = m(
            points=pts,
            normals=nrm,
            global_inputs={"g": drv},
            measure_weights=w,
            boundary_inputs=bs,
        )
    q = _rotation()
    with torch.no_grad():
        rot = m(
            points=pts @ q.T,
            normals=nrm @ q.T,
            global_inputs={"g": drv @ q.T},
            measure_weights=w,
            boundary_inputs=bs,
        )
    _assert_rotated(rot, base, q)
    assert m.embed[0].in_features == 1 + 2


def test_head_runs_under_bf16_autocast():
    """Mixed-precision smoke: the head must survive bf16 autocast (the geometry is
    kept in the input precision by construction)."""
    torch.manual_seed(0)
    m = _isla(hidden=32, n_layers=1, n_slices=8)
    pts = torch.randn(1, 128, 3)
    nrm = torch.nn.functional.normalize(torch.randn(1, 128, 3), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3), dim=-1)
    w = torch.rand(1, 128) + 0.5
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w)
    sum(o.float().sum() for o in out.values()).backward()
    assert all(torch.isfinite(o.float()).all() for o in out.values())


@pytest.mark.parametrize("kw", [{}, {"geo_checkpoint": True}])
@pytest.mark.parametrize("interior", [False, True], ids=["boundary", "interior"])
def test_parameters_receive_gradients(kw, interior):
    """DDP requires every parameter to take part in the loss. In the interior mode
    every parameter receives a gradient; a boundary-only forward leaves exactly the
    query-side parameters (the query token type, routing mass and embedding) unused."""
    torch.manual_seed(0)
    m = _isla(hidden=32, n_layers=2, n_slices=8, **kw)
    pts = torch.randn(1, 128, 3)
    nrm = torch.nn.functional.normalize(torch.randn(1, 128, 3), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3), dim=-1)
    w = torch.rand(1, 128) + 0.5
    fk = dict(query_points=torch.randn(1, 20, 3)) if interior else {}
    out = m(points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w, **fk)
    sum(o.sum() for o in out.values()).backward()
    unused = [n for n, p in m.named_parameters() if p.grad is None]
    if interior:
        assert not unused, unused
    else:
        assert unused and all(_is_query_param(n) for n in unused), unused


def test_geo_pool_then_project_is_exact():
    """Pooling the relational invariants over slices and then projecting is
    exactly the old project-then-pool (the projection is affine and the slice
    mix is a softmax over slices, so the bias passes through unchanged). This
    is the identity behind the memory saving: the saved activation is
    (B, N, geo) instead of (B, N, S, hidden/2)."""
    from physicsnemo.experimental.models.isla.model import _SliceBlock

    torch.manual_seed(0)
    blk = _SliceBlock(64, 32).double()
    geo = torch.randn(2, 50, 32, blk.n_geo, dtype=torch.float64)
    mix = torch.softmax(torch.randn(2, 50, 32, dtype=torch.float64), dim=-1)
    old = torch.einsum("bns,bnsg->bng", mix, blk.geo_feat(geo))
    new = blk.geo_feat(torch.einsum("bns,bnsg->bng", mix, geo))
    assert torch.allclose(old, new, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("interior", [False, True], ids=["boundary", "interior"])
def test_geo_checkpoint_is_exact(interior):
    """Recomputing the per-slice invariants in backward (geo_checkpoint=True)
    changes what autograd stores, not what it computes: identical forward
    output and gradients (to roundoff) against the stored-activation path."""
    torch.manual_seed(0)
    kw = dict(hidden=32, n_layers=2, n_slices=8)
    ref = _isla(**kw).double()
    ckp = _isla(geo_checkpoint=True, **kw).double()
    ckp.load_state_dict(ref.state_dict())
    pts = torch.randn(1, 40, 3, dtype=torch.float64)
    nrm = torch.nn.functional.normalize(
        torch.randn(1, 40, 3, dtype=torch.float64), dim=-1
    )
    drive = torch.tensor([[1.0, 0.2, 0.0]], dtype=torch.float64)
    w = torch.rand(1, 40, dtype=torch.float64) + 0.5
    fk = dict(query_points=torch.randn(1, 17, 3, dtype=torch.float64)) if interior else {}
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
        if not interior and _is_query_param(n):
            continue
        assert p.grad is not None and q.grad is not None, n
        assert torch.allclose(p.grad, q.grad, atol=1e-11, rtol=1e-9), n


def test_query_scalars_contracts():
    """Per-query rank-0 inputs on the query tokens (e.g. a stored signed distance),
    with the directions supplied as query_normals: exact rotation/translation
    covariance is untouched (scalars are invariants), the channel changes the
    output and receives gradients, geometric-scale equivariance holds (rank-0 query
    inputs are lengths and are divided by the frame's length unit), and misuse
    raises."""
    torch.manual_seed(0)
    m = _isla(hidden=32, n_layers=2, n_slices=8, query_inputs={"sdf": 0}).double()
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
        p.grad is not None and p.grad.abs().sum() > 0 for p in m.qt.embed.parameters()
    )
    with pytest.raises(ValueError, match="pass query_inputs"):
        m(
            points=pts,
            normals=nrm,
            global_inputs={"g": drive},
            measure_weights=w,
            query_points=q,
            query_normals=qn,
        )
    with pytest.raises(ValueError, match="without query_points"):
        m(
            points=pts,
            normals=nrm,
            global_inputs={"g": drive},
            measure_weights=w,
            query_inputs={"sdf": sdf},
        )
    with pytest.raises(ValueError, match="without query_points"):
        m(
            points=pts,
            normals=nrm,
            global_inputs={"g": drive},
            measure_weights=w,
            query_normals=qn,
        )


def test_interior_mode_contracts():
    """The interior mode (query_points given; interior queries as interacting
    tokens): exactly rotation/translation-covariant, finite, and -- by design --
    NOT query-set independent; every parameter receives a gradient."""
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
    m = _isla(hidden=64, n_layers=2, n_slices=16).double().eval()
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


@pytest.mark.parametrize("extra", [{}, {"query_inputs": {"sdf": 0}}])
def test_query_mass_refinement_invariance(extra):
    """Splitting every boundary token into two half-weight copies leaves the
    discrete boundary measure unchanged (positions, normals, total area, every
    integral). The query tokens' weight is a fraction of the total boundary
    measure, so the output is exactly unchanged and the geometric-scale contract
    still holds."""
    torch.manual_seed(42)
    pts = torch.randn(1, 60, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(torch.randn_like(pts), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, 60, dtype=torch.float64) + 0.5
    q = pts[:, :17] * 1.3
    fk = dict(query_points=q)
    if extra:
        fk["query_inputs"] = {"sdf": q.norm(dim=-1)}
    torch.manual_seed(0)
    m = _isla(hidden=32, n_layers=2, n_slices=8, **extra).double().eval()
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
        a = m(points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w, **fk)
        b = m(**refined, **fk)
        c = m(
            points=pts * s,
            normals=nrm,
            global_inputs={"g": drv},
            measure_weights=w * s**2,
            **scaled_fk,
        )
    assert _same(b, a, atol=1e-10, rtol=0.0)
    assert _same(c, a, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("interior", [False, True], ids=["boundary", "interior"])
def test_geometric_scale_equivariance(interior):
    """A similarity of the geometry (points x k + shift, areas x k^2) leaves the
    output exactly unchanged, for the boundary alone and with interior queries
    that are moved along."""
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
    m = _isla(hidden=64, n_layers=2, n_slices=16).double().eval()
    k = 2.7
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    fk = dict(query_points=pts[:, :50] * 1.5) if interior else {}
    fk_sc = {"query_points": k * fk["query_points"] + shift} if interior else {}
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


def test_boundary_scalars_reach_the_interior_prediction():
    """Boundary inputs are per-boundary-cell data: query tokens carry zeros in the
    channel, and the interior prediction depends on the boundary data."""
    torch.manual_seed(0)
    pts = torch.randn(1, 60, 3, dtype=torch.float64) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64
    )
    nrm = torch.nn.functional.normalize(torch.randn_like(pts), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3, dtype=torch.float64), dim=-1)
    w = torch.rand(1, 60, dtype=torch.float64) + 0.5
    bs = {"bc_a": torch.randn(1, 60, dtype=torch.float64), "bc_b": torch.randn(1, 60, dtype=torch.float64)}
    m = (
        _isla(hidden=32, n_layers=2, n_slices=8, boundary_inputs={"bc_a": 0, "bc_b": 0})
        .double()
        .eval()
    )
    q = pts[:, :17] * 0.5
    with torch.no_grad():
        out = m(
            points=pts,
            normals=nrm,
            global_inputs={"g": drv},
            measure_weights=w,
            boundary_inputs=bs,
            query_points=q,
        )
        out2 = m(
            points=pts,
            normals=nrm,
            global_inputs={"g": drv},
            measure_weights=w,
            boundary_inputs={k: 2 * v for k, v in bs.items()},
            query_points=q,
        )
    assert out["p"].shape == (1, 17) and out["v"].shape == (1, 17, 3)
    assert all(torch.isfinite(o).all() for o in out.values())
    assert not _same(out, out2, atol=1e-6)


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
    (1e-12, fp64), rotation equivariance, and the feature widths (one seed
    invariant n.g, six relational invariants in the slice blocks)."""
    pts, nrm, drv, w = _frame_cloud()
    shift = torch.tensor([300.0, -70.0, 1100.0], dtype=torch.float64)
    torch.manual_seed(1)
    m = _isla(hidden=64, n_layers=2, n_slices=16).double().eval()
    assert m.embed[0].in_features == 1
    assert m.blocks[0].geo_logit.in_features == 6
    with torch.no_grad():
        base = m(points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w)
        assert all(torch.isfinite(o).all() for o in base.values())
        assert _same(
            m(points=pts + shift, normals=nrm, global_inputs={"g": drv}, measure_weights=w),
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


def test_frame_is_sampling_consistent():
    """The discriminating contract of the origin-free frame. Under a 10:1 biased
    Poisson subsample with exact HT weights, a model whose frame carries no
    sample statistic moves its predictions at fixed interior queries between the
    uniform and the biased draw by no more than the measure noise -- the same
    order as between two independent uniform draws."""
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
    q = pts[:, :40] * 1.2
    torch.manual_seed(1)
    m = _isla(hidden=32, n_layers=2, n_slices=8).double().eval()
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


def test_defaults_are_the_reference_configuration():
    """The class defaults: the automatic geometry kernel, static prototypes, one
    seed invariant n.g and six relational invariants."""
    m = _isla(hidden=32, n_layers=2, n_slices=8)
    assert m.geo_kernel == "auto"
    assert m.slice_prototypes == "static"
    assert m.embed[0].in_features == 1
    assert m.blocks[0].geo_logit.in_features == 6
    assert m.qt.embed[0].in_features == 2  # the built-in distance pair


def test_constructor_and_forward_are_keyword_only():
    """Every constructor argument and every forward input is keyword-only
    (inputs read as the recipe's forward_kwargs mapping and models can be
    hot-swapped without positional bookkeeping)."""
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


@pytest.mark.parametrize(
    "name",
    [
        "nonsense",
        "routing_norm",
        "routing_temperature",
        "scale_mode",
        "use_measure_weights",
        "use_relational_geo",
        "fast_point_softmax",
        "query_mass",
        "query_scalar_scale",
        "local_rho",
        "query_independent",
        "query_tokens",
        "support_tokens",
        "interior_geometry",
    ],
)
def test_unknown_constructor_argument_raises_type_error(name):
    with pytest.raises(TypeError, match=f"unexpected keyword argument '{name}'"):
        _isla(hidden=32, n_layers=1, n_slices=8, **{name: 1})


def test_rms_distance_scale_contracts():
    """The RMS-pairwise-distance length unit keeps exact translation invariance and
    rotation covariance, is invariant to a geometric rescale of the points (with or
    without the matching weight rescale) AND to a uniform rescale of the weights
    alone, and needs no centre."""
    pts, nrm, drv, w = _frame_cloud()
    torch.manual_seed(1)
    m = _isla(hidden=64, n_layers=2, n_slices=16).double().eval()
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
    """On a fractal boundary refined dyadically, the total measure grows without
    bound ((4/3)^level for the Koch curve) while the RMS pairwise distance
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


def test_scalar_only_model_has_no_frame_routing_and_every_parameter_gets_a_gradient():
    """A model whose outputs are all rank 0 never reads the token basis, so the head's
    frame routing is not built (a parameter without a gradient stops DDP at the second
    step); with a vector output the frame routing exists. In both cases every
    parameter receives a gradient in the interior mode, and the scalar-only outputs
    equal the head coefficients directly."""
    torch.manual_seed(0)
    n = 200
    pts = torch.randn(1, n, 3) * torch.tensor([3.0, 2.0, 1.0])
    nrm = torch.nn.functional.normalize(torch.randn(1, n, 3), dim=-1)
    drv = torch.nn.functional.normalize(torch.randn(1, 3), dim=-1)
    w = torch.rand(1, n) + 0.5
    q = torch.randn(1, 30, 3) * 2.0
    for outs in ({"u": 0}, {"u": 0, "q1": 0, "q2": 0}, {"u": 0, "v": 1}):
        m = ISLA(outputs=outs, global_inputs={"g": 1}, hidden=32, n_layers=2, n_slices=8)
        has_vec = any(v == 1 for v in outs.values())
        assert m.needs_frame == has_vec and (m.frame_assign is not None) == has_vec
        out = m(points=pts, normals=nrm, global_inputs={"g": drv}, measure_weights=w, query_points=q)
        sum(v.square().mean() for v in out.values()).backward()
        missing = [k for k, p in m.named_parameters() if p.grad is None]
        assert not missing, missing
        assert out["u"].shape == (1, 30)
        if "q1" in outs:
            assert out["q2"].shape == (1, 30) and m.head.out_features == 3
