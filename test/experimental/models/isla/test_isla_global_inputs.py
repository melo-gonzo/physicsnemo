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

"""ISLA global inputs (2026-09-14; rank-spec grammar 2026-09-22): zero or more rank-1
global inputs (directions) and zero or more rank-0 global inputs (scalars, with
``channels`` for a parameter vector). The contracts tested here are exact (float64,
tolerance 1e-10): SE(3) covariance and measure-scale invariance for K = 0 and K = 2
directions, magnitude invariance of every direction, scalar conditioning that
changes the output while preserving every covariance contract, the K = 1 parameter
layout of the former single-direction model, and the fused kernel refusing
K != 1."""

import pytest
import torch

from physicsnemo.experimental.models import ISLA

D = torch.float64
PV = {"p": 0, "v": 1}


def _rotation(seed: int):
    R, _ = torch.linalg.qr(
        torch.randn(3, 3, dtype=D, generator=torch.Generator().manual_seed(seed))
    )
    if torch.det(R) < 0:
        R[:, 0] = -R[:, 0]
    return R


def _sample(b=2, n=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn(b, n, 3, dtype=D, generator=g)
    nrm = torch.nn.functional.normalize(
        torch.randn(b, n, 3, dtype=D, generator=g), dim=-1
    )
    w = torch.rand(b, n, dtype=D, generator=g) + 0.1
    return pts, nrm, w


def _model(global_inputs=None, **kw):
    torch.manual_seed(0)
    if global_inputs is None:
        global_inputs = {"g": 1}
    return (
        ISLA(
            outputs=PV,
            global_inputs=global_inputs,
            hidden=32,
            n_layers=2,
            n_slices=8,
            **kw,
        )
        .to(D)
        .eval()
    )


def _same(a, b, atol):
    assert a.keys() == b.keys()
    return all(torch.allclose(a[k], b[k], atol=atol) for k in a.keys())


def _rotate_globals(m, gi, R):
    """Rotate the rank-1 entries of a global_inputs dict, leave the rank-0 ones."""
    return {k: (v @ R.T if m.global_specs[k].rank == 1 else v) for k, v in gi.items()}


def _assert_se3(m, pts, nrm, w, gi, extra=None, atol=1e-10):
    extra = extra or {}
    R = _rotation(5)
    t = torch.tensor([2.0, -5.0, 1.0], dtype=D)
    with torch.no_grad():
        base = m(points=pts, normals=nrm, measure_weights=w, global_inputs=gi, **extra)
        moved_extra = {
            k: (
                v @ R.T + t
                if k.endswith("points")
                else (v @ R.T if k.endswith("normals") else v)
            )
            for k, v in extra.items()
        }  # rank-0 fields (query_inputs, boundary_inputs) are invariants and pass through unchanged
        moved = m(
            points=pts @ R.T + t,
            normals=nrm @ R.T,
            measure_weights=w,
            global_inputs=_rotate_globals(m, gi, R),
            **moved_extra,
        )
    assert torch.allclose(moved["p"], base["p"], atol=atol)
    assert torch.allclose(moved["v"], base["v"] @ R.T, atol=atol)
    if m.scale_mode != "total_measure":
        ### Measure-scale invariance holds under the RMS-distance unit (normalized
        ### weights); under the total-measure unit the weights carry physical area
        ### and a rescale is a different geometry.
        with torch.no_grad():
            rescaled = m(
                points=pts,
                normals=nrm,
                measure_weights=137.0 * w,
                global_inputs=gi,
                **extra,
            )
        assert _same(rescaled, base, atol=1e-9)
    return base


def test_zero_global_vectors():
    """K = 0: no direction anywhere in the problem (steady conduction with pinned
    boundary values). The seed is constant and the model is still exactly
    SE(3)-covariant and measure-scale invariant."""
    m = _model(global_inputs={})
    pts, nrm, w = _sample()
    assert m.constant_seed
    assert m.embed[0].in_features == 1
    assert m.n_true_basis == 3  # n, u, and e_theta(u, n)
    assert m.n_pseudo_basis == 1  # n x u
    base = _assert_se3(m, pts, nrm, w, {})
    assert all(torch.isfinite(o).all() for o in base.values())
    with pytest.raises(ValueError, match="unexpected"):
        m(
            points=pts,
            normals=nrm,
            measure_weights=w,
            global_inputs={"g": torch.randn(2, 3, dtype=D)},
        )


def test_two_global_vectors():
    """K = 2 (a freestream and a gravity direction): both directions rotate with the
    geometry, the output rotates with them, and rescaling either's magnitude
    changes nothing."""
    m = _model(global_inputs={"g1": 1, "g2": 1})
    pts, nrm, w = _sample()
    gv = torch.randn(2, 2, 3, dtype=D)
    gi = {"g1": gv[:, 0], "g2": gv[:, 1]}
    assert m.embed[0].in_features == 2  # n.g_1, n.g_2
    assert m.n_true_basis == 7 and m.n_pseudo_basis == 6
    base = _assert_se3(m, pts, nrm, w, gi)
    with torch.no_grad():
        scaled = m(
            points=pts,
            normals=nrm,
            measure_weights=w,
            global_inputs={"g1": 3.0 * gv[:, 0], "g2": 0.2 * gv[:, 1]},
        )
        swapped = m(
            points=pts,
            normals=nrm,
            measure_weights=w,
            global_inputs={"g1": gv[:, 1], "g2": gv[:, 0]},
        )
        shared = m(
            points=pts,
            normals=nrm,
            measure_weights=w,
            global_inputs={"g1": gv[:1, 0], "g2": gv[:1, 1]},
        )
    assert _same(scaled, base, atol=1e-10)
    assert not _same(swapped, base, atol=1e-3)  # the two inputs are distinct channels
    assert all(
        torch.allclose(shared[k][:1], base[k][:1], atol=1e-10) for k in base.keys()
    )  # a single set is shared across the batch
    with pytest.raises(ValueError, match="3 values per case"):
        m(
            points=pts,
            normals=nrm,
            measure_weights=w,
            global_inputs={"g1": torch.randn(5, dtype=D), "g2": gv[:, 1]},
        )


def test_two_vectors_passive_and_query_token_paths():
    """The interior paths carry the K directions to the query side as well."""
    pts, nrm, w = _sample()
    q = torch.randn(2, 40, 3, dtype=D)
    gv = torch.randn(2, 2, 3, dtype=D)
    gi = {"g1": gv[:, 0], "g2": gv[:, 1]}
    two = {"g1": 1, "g2": 1}
    m_passive = _model(two, query_independent=True, n_decoder_layers=1)
    _assert_se3(m_passive, pts, nrm, w, gi, extra=dict(query_points=q))
    m_qt = _model(two, query_tokens=True, query_mass="source_total")
    _assert_se3(m_qt, pts, nrm, w, gi, extra=dict(query_points=q))


def test_global_scalars_condition_every_token():
    """S = 2 scalar inputs (a diffusivity and a modulus, say) change the output,
    reach the query side too, and leave every covariance contract intact."""
    pts, nrm, w = _sample()
    spec = {"g": 1, "kappa": 0, "E": 0}
    m = _model(spec)
    assert m.embed[0].in_features == 1 + 2  # n.g plus the two scalars
    assert m.n_global_scalars == 2
    gv = torch.randn(2, 3, dtype=D)
    gs = torch.tensor([[0.3, -1.2], [2.0, 0.1]], dtype=D)
    gi = {"g": gv, "kappa": gs[:, 0], "E": gs[:, 1]}
    base = _assert_se3(m, pts, nrm, w, gi)
    with torch.no_grad():
        other = m(
            points=pts,
            normals=nrm,
            measure_weights=w,
            global_inputs={"g": gv, "kappa": gs[:, 0] + 1.0, "E": gs[:, 1] + 1.0},
        )
        shared = m(
            points=pts,
            normals=nrm,
            measure_weights=w,
            global_inputs={"g": gv, "kappa": gs[:1, 0], "E": gs[:1, 1]},
        )
    assert not _same(other, base, atol=1e-3)
    assert all(
        torch.allclose(shared[k][:1], base[k][:1], atol=1e-10) for k in base.keys()
    )
    with pytest.raises(ValueError, match="missing"):
        m(points=pts, normals=nrm, measure_weights=w, global_inputs={"g": gv})
    m0 = _model()
    with pytest.raises(ValueError, match="unexpected"):
        m0(points=pts, normals=nrm, measure_weights=w, global_inputs=gi)
    ### a parameter vector packs the scalars under one name with `channels`
    m_ch = _model({"g": 1, "params": {"rank": 0, "channels": 2}})
    assert m_ch.embed[0].in_features == 3 and m_ch.n_global_scalars == 2
    _assert_se3(m_ch, pts, nrm, w, {"g": gv, "params": gs})
    ### the passive and interacting query paths take the scalars as well
    q = torch.randn(2, 30, 3, dtype=D)
    m_passive = _model(spec, query_independent=True, n_decoder_layers=1)
    _assert_se3(m_passive, pts, nrm, w, gi, extra=dict(query_points=q))
    m_qt = _model(spec, query_tokens=True, query_mass="source_total")
    _assert_se3(m_qt, pts, nrm, w, gi, extra=dict(query_points=q))


def test_no_global_inputs_at_all_with_scalars_only():
    """K = 0 with S = 1: a parameterized PDE family with no preferred direction."""
    pts, nrm, w = _sample()
    m = _model({"kappa": 0})
    assert not m.constant_seed and m.embed[0].in_features == 1
    gs = torch.tensor([[0.5], [1.5]], dtype=D)
    base = _assert_se3(m, pts, nrm, w, {"kappa": gs})
    with torch.no_grad():
        other = m(
            points=pts,
            normals=nrm,
            measure_weights=w,
            global_inputs={"kappa": 2.0 * gs},
        )
    assert not _same(other, base, atol=1e-3)


def test_one_vector_layout_matches_the_former_single_vector_model():
    """K = 1, S = 0 keeps the former backbone layout: seed width 1, geometry width 6;
    the head has 5 true basis vectors and 3 pseudovectors."""
    m = _model()
    assert (m.n_global_vectors, m.n_global_scalars) == (1, 0)
    assert m.embed[0].in_features == 1
    assert m.blocks[0].geo_logit.in_features == 6
    assert m.n_true_basis == 5 and m.n_pseudo_basis == 3


def test_legacy_vector_layouts_still_accepted():
    """For K = 1 the recipe delivers (3,), (B, 3) or (B, 1, 3); all are one direction."""
    pts, nrm, w = _sample()
    m = _model()
    d = torch.tensor([1.0, 0.3, 0.0], dtype=D)
    with torch.no_grad():
        a = m(points=pts, normals=nrm, measure_weights=w, global_inputs={"g": d})
        b = m(
            points=pts,
            normals=nrm,
            measure_weights=w,
            global_inputs={"g": d[None].expand(2, 3)},
        )
        c = m(
            points=pts,
            normals=nrm,
            measure_weights=w,
            global_inputs={"g": d[None, None].expand(2, 1, 3)},
        )
    assert all(torch.equal(a[k], b[k]) and torch.equal(a[k], c[k]) for k in a.keys())
    with pytest.raises(ValueError, match="missing"):
        m(points=pts, normals=nrm, measure_weights=w)


def test_fused_kernel_refuses_other_vector_counts():
    """The fused Triton region is written for one global direction."""
    for spec in ({}, {"g1": 1, "g2": 1}):
        with pytest.raises(ValueError, match="exactly one global direction"):
            ISLA(
                outputs=PV,
                global_inputs=spec,
                hidden=32,
                n_layers=1,
                n_slices=8,
                geo_kernel="fused",
            )


def test_bad_global_specs_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        ISLA(outputs=PV, global_inputs={"g": -1}, hidden=32, n_layers=1, n_slices=8)
    with pytest.raises(ValueError, match="rank 2 is not supported"):
        ISLA(outputs=PV, global_inputs={"T": 2}, hidden=32, n_layers=1, n_slices=8)


def test_boundary_data_reaches_every_interior_path():
    """A boundary-value problem needs the per-cell boundary data (Dirichlet trace)
    on the boundary tokens and zeros on every token that is not a boundary cell.
    All three interior paths accept it, the covariance contracts hold, and the
    interior prediction depends on the boundary data."""
    pts, nrm, w = _sample()
    bsc = torch.randn(2, pts.shape[1], dtype=D)
    q = torch.randn(2, 30, 3, dtype=D) * 0.5
    gi = {"kappa": torch.tensor([[0.7], [2.0]], dtype=D)}
    common = dict(global_inputs={"kappa": 0}, boundary_inputs={"trace": 0})
    arms = [
        (
            _model(query_tokens=True, query_mass="source_total", **common),
            dict(query_points=q),
        ),
        (
            _model(query_independent=True, n_decoder_layers=1, **common),
            dict(query_points=q),
        ),
        (
            _model(
                query_independent=True,
                support_tokens=True,
                query_mass="source_total",
                n_decoder_layers=1,
                **common,
            ),
            dict(query_points=q, support_points=q),
        ),
    ]
    for m, extra in arms:
        assert (
            m.embed[0].in_features == 2
        )  # boundary scalar + global scalar (no directions)
        base = _assert_se3(
            m, pts, nrm, w, gi, extra=dict(extra, boundary_inputs={"trace": bsc})
        )
        with torch.no_grad():
            other = m(
                points=pts,
                normals=nrm,
                measure_weights=w,
                global_inputs=gi,
                boundary_inputs={"trace": bsc + 1.0},
                **extra,
            )
        assert base["p"].shape[1] == 30 and not _same(other, base, atol=1e-3)
