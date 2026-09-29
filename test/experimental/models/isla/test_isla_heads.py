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

"""Contract tests for ISLA's rank-spec output heads (2026-09-22).

Every output is declared with a rank, a symmetry and a parity and is assembled from
invariant coefficients times tensor products of the token's basis vectors (true
vectors for even parity, exactly one pseudovector factor for odd parity) plus the
isotropic tensors of that rank. The checks are exact (float64, 1e-12): a rank-r
output transforms as ``R^{(x)r}`` under every orthogonal map, with the extra sign
``det R`` for odd parity, and is invariant under translations.
"""

from math import comb

import pytest
import torch

from physicsnemo.experimental.models import ISLA

D = torch.float64

OUTPUTS = {
    "s": 0,
    "s2": 0,
    "v": 1,
    "pv": {"rank": 1, "parity": "odd"},
    "t": 2,
    "ts": {"rank": 2, "symmetric": True},
    "pt": {"rank": 2, "parity": "odd"},
    "t3": 3,
    "t3o": {"rank": 3, "parity": "odd"},
    "t3s": {"rank": 3, "symmetric": True},
    "t4s": {"rank": 4, "symmetric": True},
}


def _cloud(n=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn(1, n, 3, dtype=D, generator=g) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=D
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=D, generator=g), dim=-1
    )
    drv = torch.nn.functional.normalize(
        torch.randn(1, 2, 3, dtype=D, generator=g), dim=-1
    )
    w = torch.rand(1, n, dtype=D, generator=g) + 0.5
    return pts, nrm, drv, w


def _globals(K):
    return {f"g{k}": 1 for k in range(K)}


def _model(K=1, outputs=OUTPUTS, **kw):
    torch.manual_seed(0)
    base = dict(
        hidden=32, n_layers=2, n_slices=8, outputs=outputs, global_inputs=_globals(K)
    )
    base.update(kw)
    return ISLA(**base).double().eval()


def _orthogonal(seed, improper=False):
    q, _ = torch.linalg.qr(
        torch.randn(3, 3, dtype=D, generator=torch.Generator().manual_seed(seed))
    )
    if (torch.det(q) < 0) != improper:
        q[:, 0] = -q[:, 0]
    return q


def _forward(m, pts, nrm, drv, w, K, R=None, shift=None, **extra):
    """Forward pass with the geometry (and every vector input) transformed by ``R`` and
    translated by ``shift``."""
    R = torch.eye(3, dtype=D, device=pts.device) if R is None else R.to(pts.device)
    shift = (
        torch.zeros(3, dtype=D, device=pts.device)
        if shift is None
        else shift.to(pts.device)
    )
    gi = {f"g{k}": drv[:, k] @ R.T for k in range(K)}
    with torch.no_grad():
        return m(
            points=pts @ R.T + shift,
            normals=nrm @ R.T,
            measure_weights=w,
            global_inputs=gi,
            **extra,
        )


def _transform(x, R, spec):
    """``R^{(x)r} x`` on the trailing ``rank`` axes of ``x`` (leading axes batch), with
    the pseudotensor sign for odd parity."""
    if spec.rank == 0:
        return x
    y = x
    for k in range(spec.rank):
        y = torch.movedim(torch.tensordot(y, R, dims=([2 + k], [1])), -1, 2 + k)
    return y * (torch.det(R) if spec.parity == "odd" else 1.0)


@pytest.mark.parametrize("K", [0, 1, 2])
def test_coefficient_counts_match_the_docstring(K):
    """m = 2K + 3 true vectors, p = C(K+2, 2) pseudovectors, plus the isotropic terms."""
    m = _model(K)
    mm, p = 2 * K + 3, comb(K + 2, 2)
    assert m.n_true_basis == mm and m.n_pseudo_basis == p
    expected = {
        "s": 1,
        "s2": 1,
        "v": mm,
        "pv": p,
        "t": mm**2 + 1,
        "ts": mm * (mm + 1) // 2 + 1,
        "pt": 2 * p * mm,
        "t3": mm**3,
        "t3o": 3 * p * mm**2 + 1,
        "t3s": comb(mm + 2, 3),
        "t4s": comb(mm + 3, 4) + 1,
    }
    for name, n in expected.items():
        assert m.output_layouts[name].n_coeffs == n, name
    assert m.head.out_features == sum(expected.values())
    assert m.n_head_coeffs == m.head.out_features


def test_minimal_head_width():
    """One scalar and one true vector over K = 1: a head of width 1 + 5."""
    m = _model(1, outputs={"p": 0, "v": 1})
    assert m.head.out_features == 6


@pytest.mark.parametrize("K", [0, 1, 2])
def test_output_shapes(K):
    m = _model(K)
    pts, nrm, drv, w = _cloud()
    out = _forward(m, pts, nrm, drv, w, K)
    assert set(out.keys()) == set(OUTPUTS)
    assert out["s"].shape == (1, 300) and out["s2"].shape == (1, 300)
    assert out["v"].shape == out["pv"].shape == (1, 300, 3)
    assert out["t"].shape == out["ts"].shape == out["pt"].shape == (1, 300, 3, 3)
    assert out["t3"].shape == out["t3o"].shape == out["t3s"].shape == (1, 300, 3, 3, 3)
    assert out["t4s"].shape == (1, 300, 3, 3, 3, 3)
    assert all(torch.isfinite(o).all() for o in out.values())


@pytest.mark.parametrize("K", [0, 1, 2])
@pytest.mark.parametrize("improper", [False, True], ids=["rotation", "reflection"])
def test_covariance_of_every_rank_and_parity(K, improper):
    """Under a random orthogonal map R (proper or improper) together with a
    translation, a rank-r even output transforms as R^{(x)r}, an odd one picks up
    det R, and a rank-0 output is unchanged (1e-12, float64)."""
    m = _model(K)
    pts, nrm, drv, w = _cloud()
    base = _forward(m, pts, nrm, drv, w, K)
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=D)
    for seed in range(2):
        R = _orthogonal(seed, improper)
        assert (torch.det(R) < 0) == improper
        out = _forward(m, pts, nrm, drv, w, K, R=R, shift=shift)
        for name, spec in m.output_specs.items():
            assert torch.allclose(
                out[name], _transform(base[name], R, spec), atol=1e-12
            ), (name, K, improper)


def test_translation_invariance_alone():
    m = _model(1)
    pts, nrm, drv, w = _cloud()
    base = _forward(m, pts, nrm, drv, w, 1)
    tr = _forward(
        m, pts, nrm, drv, w, 1, shift=torch.tensor([3.0, -7.0, 11.0], dtype=D)
    )
    for k in base.keys():
        assert torch.allclose(tr[k], base[k], atol=1e-12), k


def test_symmetric_outputs_are_symmetric():
    m = _model(1)
    pts, nrm, drv, w = _cloud()
    out = _forward(m, pts, nrm, drv, w, 1)
    ts, t3s, t4s = out["ts"], out["t3s"], out["t4s"]
    assert torch.allclose(ts, ts.transpose(-1, -2), atol=1e-12)
    for perm in ((0, 1, 3, 4, 2), (0, 1, 2, 4, 3), (0, 1, 4, 3, 2)):
        assert torch.allclose(t3s, t3s.permute(*perm), atol=1e-12)
    for perm in ((0, 1, 3, 2, 4, 5), (0, 1, 2, 3, 5, 4), (0, 1, 5, 4, 3, 2)):
        assert torch.allclose(t4s, t4s.permute(*perm), atol=1e-12)
    ### the general rank-2 output is generically NOT symmetric
    assert not torch.allclose(out["t"], out["t"].transpose(-1, -2), atol=1e-6)


def test_odd_outputs_are_pseudotensors_and_even_ones_are_not():
    """A pure inversion x -> -x flips every odd output's sign relative to the even
    law and leaves the even outputs on the even law; the contract is not vacuous
    because the odd basis is non-trivial (the outputs are not zero)."""
    m = _model(1)
    pts, nrm, drv, w = _cloud()
    base = _forward(m, pts, nrm, drv, w, 1)
    R = -torch.eye(3, dtype=D)
    out = _forward(m, pts, nrm, drv, w, 1, R=R)
    assert torch.allclose(out["pv"], base["pv"], atol=1e-12)  # (-1)^1 * det = +1
    assert torch.allclose(out["v"], -base["v"], atol=1e-12)
    assert torch.allclose(out["pt"], -base["pt"], atol=1e-12)  # (-1)^2 * det = -1
    assert torch.allclose(out["t"], base["t"], atol=1e-12)
    assert base["pv"].norm() > 1e-3 and base["pt"].norm() > 1e-3
    assert base["t3o"].norm() > 1e-3


def test_rank_spec_grammar_is_validated():
    base = dict(hidden=32, n_layers=1, n_slices=8, global_inputs={"g": 1})
    with pytest.raises(ValueError, match="rank 5 is not supported"):
        ISLA(outputs={"x": 5}, **base)
    with pytest.raises(ValueError, match="unknown rank-spec keys"):
        ISLA(outputs={"x": {"rank": 1, "parity_odd": True}}, **base)
    with pytest.raises(TypeError, match="bool"):
        ISLA(outputs={"x": True}, **base)
    with pytest.raises(ValueError, match="parity must be"):
        ISLA(outputs={"x": {"rank": 1, "parity": "mixed"}}, **base)
    with pytest.raises(ValueError, match="unknown rank-spec keys \\['channels'\\]"):
        ISLA(outputs={"x": {"rank": 0, "channels": 2}}, **base)
    with pytest.raises(ValueError, match="at least one field"):
        ISLA(outputs={}, **base)
    with pytest.raises(ValueError, match="pseudotensor"):
        ISLA(
            outputs={"p": 0},
            boundary_inputs={"w": {"rank": 1, "parity": "odd"}},
            **base,
        )
    with pytest.raises(ValueError, match="rank 3 is not supported"):
        ISLA(outputs={"p": 0}, boundary_inputs={"T": 3}, **base)
    with pytest.raises(ValueError, match="rank 2 is not supported"):
        ISLA(
            outputs={"p": 0}, hidden=32, n_layers=1, n_slices=8, global_inputs={"T": 2}
        )
    with pytest.raises(TypeError, match="mapping"):
        ISLA(outputs=[0, 1], **base)


@pytest.mark.parametrize("supplied", [False, True], ids=["internal", "supplied_normals"])
def test_rank_two_output_in_the_interior_mode(supplied):
    """The heads sit on the shared readout, so the interior mode carries a rank-2
    output with the same covariance (rotation and reflection), with the query
    directions computed inside or supplied."""
    pts, nrm, drv, w = _cloud(200)
    q_pts, q_nrm, _, _ = _cloud(50, seed=1)
    outputs = {"p": 0, "v": 1, "sig": {"rank": 2, "symmetric": True}, "t": 2}
    m = _model(1, outputs=outputs, query_inputs={"sdf": 0})
    qs = torch.rand(1, 50, dtype=D)

    def call(R):
        extra = dict(query_points=q_pts @ R.T, query_inputs={"sdf": qs})
        if supplied:
            extra["query_normals"] = q_nrm @ R.T
        return _forward(m, pts, nrm, drv, w, 1, R=R, **extra)

    base = call(torch.eye(3, dtype=D))
    for improper in (False, True):
        R = _orthogonal(3, improper)
        out = call(R)
        assert base["sig"].shape == (1, 50, 3, 3)
        for name, spec in m.output_specs.items():
            assert torch.allclose(
                out[name], _transform(base[name], R, spec), atol=1e-12
            ), (name, supplied, improper)


def test_all_parameters_receive_gradients():
    m = _model(1).train()
    pts, nrm, drv, w = _cloud(100)
    out = m(
        points=pts,
        normals=nrm,
        measure_weights=w,
        global_inputs={"g0": drv[:, 0]},
        query_points=pts[:, :20] * 1.5,
    )
    sum(o.square().sum() for o in out.values()).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_checkpoint_round_trip_keeps_the_specs(tmp_path):
    from physicsnemo.core.module import Module

    m = _model(1)
    m.save(tmp_path / "m.mdlus")
    m2 = Module.from_checkpoint(str(tmp_path / "m.mdlus")).double().eval()
    assert m2.output_specs == m.output_specs and m2.global_specs == m.global_specs
    pts, nrm, drv, w = _cloud(64)
    a, b = _forward(m, pts, nrm, drv, w, 1), _forward(m2, pts, nrm, drv, w, 1)
    assert all(torch.equal(a[k], b[k]) for k in a.keys())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernel needs CUDA")
def test_fused_kernel_does_not_touch_the_heads():
    """The fused geometry region is per layer; the heads are downstream of it, so the
    rank-2 outputs of the fused and eager models agree to roundoff in float64."""
    pytest.importorskip("triton")
    pts, nrm, drv, w = _cloud(400)
    outputs = {"p": 0, "v": 1, "sig": {"rank": 2, "symmetric": True}}
    outs = []
    for kernel in ("eager", "fused"):
        m = _model(1, outputs=outputs, geo_kernel=kernel).cuda()
        outs.append(
            {
                k: v.cpu()
                for k, v in _forward(
                    m, pts.cuda(), nrm.cuda(), drv.cuda(), w.cuda(), 1
                ).items()
            }
        )
    for k in outs[0].keys():
        assert torch.allclose(outs[0][k], outs[1][k], atol=1e-10), k
