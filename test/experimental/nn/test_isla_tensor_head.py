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

"""Contract tests for ISLA's symmetric rank-2 tensor head (``out_tensors``, 2026-09-22).

The tensor head expands each output in the symmetrized outer products of the head's TRUE
vectors plus the identity, so the assembled tensor is a true tensor: ``sigma -> R sigma R^T``
for proper rotations AND reflections (the vector heads, whose basis includes the ``e_phi``
cross products, are covariant under proper rotations only). Every check is exact (float64).
"""

import pytest
import torch

from physicsnemo.experimental.nn.isla import ISLA

D = torch.float64
COMPONENTS = ("xx", "yy", "zz", "xy", "xz", "yz")


def _cloud(n=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn(1, n, 3, dtype=D, generator=g) * torch.tensor(
        [3.0, 2.0, 1.0], dtype=D
    )
    nrm = torch.nn.functional.normalize(
        torch.randn(1, n, 3, dtype=D, generator=g), dim=-1
    )
    drv = torch.nn.functional.normalize(
        torch.randn(1, 1, 3, dtype=D, generator=g), dim=-1
    )
    w = torch.rand(1, n, dtype=D, generator=g) + 0.5
    return pts, nrm, drv, w


def _model(**kw):
    torch.manual_seed(0)
    base = dict(
        hidden=32, n_layers=2, n_slices=8, out_scalars=2, out_vectors=1, out_tensors=2
    )
    base.update(kw)
    return ISLA(**base).double().eval()


def _tensors(out, m):
    """The symmetric tensors of an output row as (B, N, out_tensors, 3, 3)."""
    start = m.out_scalars + 3 * m.out_vectors
    c = out[..., start:].reshape(*out.shape[:-1], m.out_tensors, 6)
    xx, yy, zz, xy, xz, yz = c.unbind(-1)
    return torch.stack(
        [
            torch.stack([xx, xy, xz], -1),
            torch.stack([xy, yy, yz], -1),
            torch.stack([xz, yz, zz], -1),
        ],
        -2,
    )


def _proper_rotation(seed):
    q, _ = torch.linalg.qr(
        torch.randn(3, 3, dtype=D, generator=torch.Generator().manual_seed(seed))
    )
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _forward(m, pts, nrm, drv, w, K):
    kw = dict(points=pts, normals=nrm, measure_weights=w)
    if K:
        kw["global_vectors"] = drv[:, :K]
    with torch.no_grad():
        return m(**kw)


@pytest.mark.parametrize("K", [0, 1, 2])
def test_layout_and_basis_counts(K):
    m = _model(n_global_vectors=K)
    pts, nrm, drv, w = _cloud()
    drv = torch.cat([drv, torch.tensor([[[0.0, 0.0, 1.0]]], dtype=D)], dim=1)
    out = _forward(m, pts, nrm, drv, w, K)
    assert out.shape == (1, 300, m.out_scalars + 3 * m.out_vectors + 6 * m.out_tensors)
    ### True vectors: g_1..g_K, n, rhat, e_th_n, e_th_g1..e_th_gK -> m = 2K + 3.
    mm = 2 * K + 3
    assert m.n_true_basis == mm
    assert m.n_tbasis == mm * (mm + 1) // 2 + 1
    assert (
        m.head.out_features
        == m.out_scalars + m.out_vectors * m.n_basis + m.out_tensors * m.n_tbasis
    )
    ### K = 0: 7 tensor basis elements; K = 1: 16.
    assert {0: 7, 1: 16, 2: 29}[K] == m.n_tbasis


@pytest.mark.parametrize("K", [0, 1])
def test_rotation_covariance_of_tensors(K):
    """R sigma R^T under random proper rotations (scalars and vectors keep their contracts)."""
    m = _model(n_global_vectors=K)
    pts, nrm, drv, w = _cloud()
    base = _forward(m, pts, nrm, drv, w, K)
    for seed in range(3):
        q = _proper_rotation(seed)
        rot = _forward(m, pts @ q.T, nrm @ q.T, drv @ q.T, w, K)
        s = m.out_scalars
        assert torch.allclose(rot[..., :s], base[..., :s], atol=1e-12)
        v0, v1 = base[..., s : s + 3], rot[..., s : s + 3]
        assert torch.allclose(v1, v0 @ q.T, atol=1e-12)
        t0, t1 = _tensors(base, m), _tensors(rot, m)
        assert torch.allclose(t1, q @ t0 @ q.T, atol=1e-12)


@pytest.mark.parametrize("K", [0, 1])
def test_reflection_covariance_of_tensors(K):
    """Under a mirror the tensor output is a TRUE tensor (sigma -> M sigma M^T), the scalars
    are invariant, and the vector head (with its e_phi pseudovectors) generically is not."""
    m = _model(n_global_vectors=K)
    pts, nrm, drv, w = _cloud()
    base = _forward(m, pts, nrm, drv, w, K)
    for mirror in (
        torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=D)),
        torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=D)),
    ):
        ### A general improper orthogonal matrix: mirror composed with a rotation.
        M = _proper_rotation(7) @ mirror
        assert torch.det(M) < 0
        ref = _forward(m, pts @ M.T, nrm @ M.T, drv @ M.T, w, K)
        s = m.out_scalars
        assert torch.allclose(ref[..., :s], base[..., :s], atol=1e-12)
        t0, t1 = _tensors(base, m), _tensors(ref, m)
        assert torch.allclose(t1, M @ t0 @ M.T, atol=1e-12)
        v0, v1 = base[..., s : s + 3], ref[..., s : s + 3]
        assert not torch.allclose(v1, v0 @ M.T, atol=1e-6)


def test_translation_invariance_and_symmetry():
    m = _model(n_global_vectors=1)
    pts, nrm, drv, w = _cloud()
    base = _forward(m, pts, nrm, drv, w, 1)
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=D)
    tr = _forward(m, pts + shift, nrm, drv, w, 1)
    assert torch.allclose(tr, base, atol=1e-12)
    t = _tensors(base, m)
    assert torch.allclose(t, t.transpose(-1, -2))
    assert t.shape == (1, 300, m.out_tensors, 3, 3)


def test_tensor_head_in_the_interior_modes():
    """The tensor head sits on the shared readout, so the query-token and passive paths carry
    it with the same covariance."""
    pts, nrm, drv, w = _cloud(200)
    q_pts, q_nrm, _, _ = _cloud(50, seed=1)
    q = _proper_rotation(3)
    for kw in (
        dict(query_tokens=True, n_query_scalars=1),
        dict(query_independent=True, n_decoder_layers=1, n_query_scalars=1),
    ):
        m = _model(
            n_global_vectors=1, out_scalars=1, out_vectors=1, out_tensors=1, **kw
        )
        qs = torch.rand(1, 50, 1, dtype=D)
        with torch.no_grad():
            base = m(
                points=pts,
                normals=nrm,
                global_vectors=drv,
                measure_weights=w,
                query_points=q_pts,
                query_normals=q_nrm,
                query_scalars=qs,
            )
            rot = m(
                points=pts @ q.T,
                normals=nrm @ q.T,
                global_vectors=drv @ q.T,
                measure_weights=w,
                query_points=q_pts @ q.T,
                query_normals=q_nrm @ q.T,
                query_scalars=qs,
            )
        assert base.shape == (1, 50, 1 + 3 + 6)
        assert torch.allclose(_tensors(rot, m), q @ _tensors(base, m) @ q.T, atol=1e-12)


def test_out_tensors_zero_is_the_previous_model():
    """Default out_tensors=0: same parameter names and shapes, same output width, no tensor
    channels; the goldens in test_isla_release.py pin the values."""
    torch.manual_seed(0)
    a = ISLA(hidden=32, n_layers=2, n_slices=8)
    torch.manual_seed(0)
    b = ISLA(hidden=32, n_layers=2, n_slices=8, out_tensors=0)
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert a.head.out_features == 1 + 7
    with pytest.raises(ValueError):
        ISLA(hidden=32, n_layers=2, n_slices=8, out_tensors=-1)


def test_all_parameters_receive_gradients_with_tensors():
    m = _model(n_global_vectors=1).train()
    pts, nrm, drv, w = _cloud(100)
    out = m(points=pts, normals=nrm, global_vectors=drv, measure_weights=w)
    out[..., -6:].square().sum().backward()
    ### Only the tensor coefficients' rows of the head see this loss; every other
    ### parameter feeds them through the trunk.
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernel needs CUDA")
def test_fused_kernel_does_not_touch_the_tensor_head():
    """The fused geometry region is per layer; the head is downstream of it, so the tensor
    outputs of the fused and eager models agree to roundoff in float64."""
    pytest.importorskip("triton")
    pts, nrm, drv, w = _cloud(400)
    outs = []
    for kernel in ("eager", "fused"):
        m = _model(n_global_vectors=1, geo_kernel=kernel).cuda()
        outs.append(_forward(m, pts.cuda(), nrm.cuda(), drv.cuda(), w.cuda(), 1).cpu())
    assert torch.allclose(outs[0], outs[1], atol=1e-10)
