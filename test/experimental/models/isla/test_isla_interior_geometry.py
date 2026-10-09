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

"""The interior geometry of the prediction-point tokens.

For every prediction point the model derives its signed distance from the boundary
tokens itself (the nearest boundary token gives the distance, that token's normal
the sign, positive on the side the normal points to) and its direction from the same
search; neither is an input. These tests pin the geometry against a brute-force
computation, its sign convention, and the contracts of the interior task.
"""

import math
import os

import pytest
import torch

from physicsnemo.experimental.models import ISLA
from physicsnemo.experimental.models.isla import geometry
from physicsnemo.experimental.models.isla.geometry import (
    _nearest_index,
    _soft_geometry_torch,
    nearest_boundary_geometry,
    patch_boundary_geometry,
    soft_boundary_geometry,
)
from test.experimental.models.isla._meshes import (
    cloud,
    orthogonal,
    outputs,
    points_mesh,
    soup,
    transform,
)

D = torch.float64
PV = {"p": {"rank": 0}, "v": {"rank": 1}}


def _cloud(n, seed, batch=1, scale=(3.0, 2.0, 1.0)):
    """Batched tensors for the tensor-level geometry function."""
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn(batch, n, 3, dtype=D, generator=g) * torch.tensor(scale, dtype=D)
    nrm = torch.nn.functional.normalize(torch.randn(batch, n, 3, dtype=D, generator=g), dim=-1)
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


def test_nearest_boundary_geometry_matches_brute_force():
    pts, nrm, _ = _cloud(150, 0, batch=2)
    q, _, _ = _cloud(45, 1, batch=2, scale=(4.0, 4.0, 4.0))
    d, u = nearest_boundary_geometry(q, pts, nrm)
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
    dirs = torch.nn.functional.normalize(torch.randn(1, n, 3, dtype=D, generator=g), dim=-1)
    pts, nrm = 2.0 * dirs, dirs  # radius 2, outward normals
    q_dirs = torch.nn.functional.normalize(torch.randn(1, 30, 3, dtype=D, generator=g), dim=-1)
    radii = torch.cat([torch.full((15,), 3.0, dtype=D), torch.full((15,), 1.0, dtype=D)])
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
    R = orthogonal(3, 0, improper=True)
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
    return ISLA(output_schema=PV, global_schema={"g": {"rank": 1}}, hidden=32, n_layers=2, n_slices=8, interior=True, **kw).double().eval()


def test_interior_contracts():
    """Rotation and reflection covariance, translation invariance and geometric-scale
    invariance in the interior task, with the prediction-point directions computed
    inside; the distance channel is live."""
    pts, nrm, w, _ = cloud(200)
    mesh = soup(pts, nrm, w)
    drv = torch.tensor([0.3, 1.0, 0.2], dtype=D)
    q, _, _, _ = cloud(40, seed=1, scale=(4.0, 4.0, 4.0))
    m = _model()

    def call(R=None, t=None, k=1.0, q=q):
        R = torch.eye(3, dtype=D) if R is None else R
        t = torch.zeros(3, dtype=D) if t is None else t
        with torch.no_grad():
            return outputs(
                m(boundary_meshes={"wall": transform(mesh, R, t, scale=k)}, prediction_points=k * q @ R.T + t, global_data={"g": drv @ R.T})
            )

    base = call()
    for improper in (False, True):
        R = orthogonal(3, 4, improper)
        out = call(R, torch.tensor([2.0, -5.0, 1.0], dtype=D), k=2.5)
        assert torch.allclose(out["p"], base["p"], atol=1e-11), improper
        assert torch.allclose(out["v"], base["v"] @ R.T, atol=1e-11), improper
    ### the distance channel is live: moving the queries off the boundary changes p
    far = call(q=q * 3.0)
    assert not torch.allclose(far["p"], base["p"], atol=1e-6)


def test_stored_distance_is_an_ordinary_query_input():
    """A dataset's stored signed distance (and its gradient) enter as query_schema
    fields of rank 0 and 1 from the prediction Mesh's point_data; the built-in
    geometry is computed regardless, so the two channels coexist and both are live."""
    pts, nrm, w, _ = cloud(120)
    mesh = soup(pts, nrm, w)
    drv = torch.tensor([0.3, 1.0, 0.2], dtype=D)
    q, _, _, _ = cloud(25, seed=1, scale=(4.0, 4.0, 4.0))
    d, grad = nearest_boundary_geometry(q[None], pts[None], nrm[None])
    m = _model(query_schema={"geometry.sdf": {"rank": 0}, "geometry.grad": {"rank": 1}})
    assert m.qt.embed[0].in_features == 2 + (2 + (2 + 1))  # built-in pair, scaled scalar pair, vector cosines and norm

    def call(sdf, g):
        with torch.no_grad():
            return outputs(
                m(
                    boundary_meshes={"wall": mesh},
                    prediction_points=points_mesh(q, {"geometry": {"sdf": sdf, "grad": g}}),
                    global_data={"g": drv},
                )
            )

    base = call(d[0], grad[0])
    assert base["p"].shape == (25,)
    assert not torch.allclose(call(2 * d[0], grad[0])["p"], base["p"], atol=1e-6)
    assert not torch.allclose(call(d[0], -grad[0])["p"], base["p"], atol=1e-6)


def test_no_query_normals_argument():
    """The prediction-point direction is never an input."""
    pts, nrm, w, _ = cloud(20)
    m = _model()
    with pytest.raises(TypeError, match="query_normals"):
        m(boundary_meshes={"wall": soup(pts, nrm, w)}, prediction_points=pts[:3], global_data={"g": nrm[0]}, query_normals=nrm[:3])
    with pytest.raises(TypeError, match="interior_geometry"):
        _model(interior_geometry="provided")


def test_fused_kernel_stays_available_with_the_built_in_distance():
    """The built-in distance enters the embedding, not the geometry region, so a model
    without rank-1 or rank-2 per-token inputs keeps the fused kernel."""
    m = ISLA(output_schema=PV, global_schema={"g": {"rank": 1}}, hidden=32, n_layers=1, n_slices=8, interior=True, geo_kernel="fused")
    assert m.n_query_builtin == 2 and m.qt.embed[0].in_features == 2
    assert not (m.n_block_vecs or m.n_block_tens)


def test_checkpoint_round_trip_keeps_the_task(tmp_path):
    from physicsnemo.core.module import Module

    m = _model()
    m.save(tmp_path / "m.mdlus")
    m2 = Module.from_checkpoint(str(tmp_path / "m.mdlus")).double().eval()
    assert m2.interior is True
    pts, nrm, w, _ = cloud(50)
    q, _, _, _ = cloud(8, seed=1)
    kw = dict(boundary_meshes={"wall": soup(pts, nrm, w)}, prediction_points=q, global_data={"g": torch.ones(3, dtype=D)})
    with torch.no_grad():
        a, b = outputs(m(**kw)), outputs(m2(**kw))
    assert all(torch.equal(a[k], b[k]) for k in a)


def _assert_indices_exact(idx, q, pts):
    """``idx`` equals the float64 brute-force argmin on the same inputs, except
    where the two candidates are a tie at the inputs' own precision (their
    float64 distances agree to within four float32 ulp); returns the tie count."""
    dist = torch.cdist(q.double(), pts.double())
    ref = dist.argmin(-1)
    diff = idx != ref
    d_idx = dist.gather(-1, idx[..., None])[..., 0]
    d_ref = dist.gather(-1, ref[..., None])[..., 0]
    tie = (d_idx - d_ref).abs() <= 4 * torch.finfo(torch.float32).eps * d_ref
    assert not (diff & ~tie).any(), int((diff & ~tie).sum())
    return int(diff.sum())


def test_nearest_boundary_geometry_near_wall_dense_cluster_matches_brute_force():
    """Queries a hair off a dense cluster of boundary samples far from the cloud's
    centre (large |x|^2 relative to the separations), in float32 inputs, against a
    float64 brute force on the same inputs: the exact difference-form search
    returns the true nearest sample on CPU and CUDA alike (the expanded form of
    the library's CUDA search resolved these only to the cluster's spacing)."""
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
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devices:
        d, u = nearest_boundary_geometry(q.to(dev), pts.to(dev), nrm.to(dev))
        assert torch.allclose(d.cpu().double(), d_ref, atol=1e-9, rtol=1e-6), dev
        assert torch.allclose(u.cpu().double(), u_ref, atol=1e-5, rtol=0), dev
        assert _assert_indices_exact(_nearest_index(q[0].to(dev), pts[0].to(dev)).cpu(), q[0], pts[0]) == 0


def _check_against_cdist_argmin(device, dim):
    """d and u from ``nearest_boundary_geometry`` against ``torch.cdist(...).argmin``
    on a random ``dim``-dimensional cloud, on ``device``."""
    g = torch.Generator().manual_seed(dim)
    pts = torch.randn(1, 3000, dim, dtype=D, generator=g).to(device)
    nrm = torch.nn.functional.normalize(torch.randn(1, 3000, dim, dtype=D, generator=g), dim=-1).to(device)
    q = 1.5 * torch.randn(1, 500, dim, dtype=D, generator=g).to(device)
    d, u = nearest_boundary_geometry(q, pts, nrm)
    dist = torch.cdist(q, pts)
    idx = dist.argmin(-1)
    y = pts.gather(1, idx[..., None].expand(1, 500, dim))
    n_star = nrm.gather(1, idx[..., None].expand(1, 500, dim))
    u_ref = torch.nn.functional.normalize(q - y, dim=-1)
    d_ref = torch.where((u_ref * n_star).sum(-1) >= 0, 1.0, -1.0).to(D) * dist.gather(-1, idx[..., None])[..., 0]
    assert d.shape == (1, 500) and u.shape == (1, 500, dim)
    assert torch.allclose(d, d_ref, atol=1e-12, rtol=0)
    assert torch.allclose(u, u_ref, atol=1e-12, rtol=0)


@pytest.mark.parametrize("dim", [2, 3])
def test_nearest_boundary_geometry_matches_cdist_argmin_on_cpu(dim):
    _check_against_cdist_argmin("cpu", dim)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA search")
@pytest.mark.parametrize("dim", [2, 3])
def test_nearest_boundary_geometry_matches_cdist_argmin_on_cuda(dim):
    _check_against_cdist_argmin("cuda", dim)


def _wall_problem(dim: int, seed: int = 5, n: int = 4000, n_q: int = 2000, offsets=(1e-5, 1e-4, 1e-3)):
    """A unit circle (polyline) or sphere sampled at ``n`` points with outward normals
    and ``n_q`` queries a sample of ``offsets`` off the wall on both sides, in
    float32: the near-wall picture of a unit-chord airfoil."""
    g = torch.Generator().manual_seed(seed)
    dirs = torch.nn.functional.normalize(torch.randn(n, dim, dtype=D, generator=g), dim=-1)
    q_dirs = torch.nn.functional.normalize(torch.randn(n_q, dim, dtype=D, generator=g), dim=-1)
    side = torch.where(torch.rand(n_q, dtype=D, generator=g) < 0.5, 1.0, -1.0)
    off = torch.tensor(offsets, dtype=D)[torch.randint(len(offsets), (n_q,), generator=g)]
    q = q_dirs * (1.0 + side * off)[:, None]
    return dirs.float(), dirs.float(), q.float()


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("shift", [0.0, 1.0, 100.0])
def test_search_is_exact_near_a_translated_wall(device, dim, shift):
    """Regression (AirfRANS posed inference, 2026-10-03): the library's CUDA search
    (cuML, float32 expanded distance) returned a wrong in-range nearest cell for
    0.45% of the near-wall queries of native AirfRANS cases, 0.70% of translated
    ones, and the sentinel -1 within ~1e-4 of the wall at |x| ~ 1, so the trained
    model's error moved by 5-7% under a rigid translation. Near-wall float32 queries
    on a unit circle / sphere translated by 0, 1 and 100 radii: the index equals the
    float64 brute force on the same float32 inputs (up to ties at float32
    resolution) and |d| equals the exact distance, on every device."""
    pts, nrm, q = _wall_problem(dim)
    t = torch.full((dim,), shift)
    pts, q, nrm = (pts + t).to(device), (q + t).to(device), nrm.to(device)
    idx = _nearest_index(q, pts)
    ties = _assert_indices_exact(idx.cpu(), q.cpu(), pts.cpu())
    assert ties <= 2, ties
    d, u = nearest_boundary_geometry(q[None], pts[None], nrm[None])
    d_ref = torch.cdist(q[None].double(), pts[None].double()).min(-1).values
    assert torch.allclose(d.double().abs(), d_ref, atol=1e-10, rtol=1e-6)
    assert torch.allclose(u.norm(dim=-1), torch.ones_like(d), atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dim", [2, 3])
def test_search_indices_are_translation_invariant(device, dim):
    """The same near-wall problem natively and translated by (100, ..., 100) in
    float32: the returned indices agree except where the float64 brute force on the
    two (differently rounded) float32 inputs itself disagrees."""
    pts, _, q = _wall_problem(dim, n=3000, n_q=5000)
    t = torch.full((dim,), 100.0)
    native = _nearest_index(q.to(device), pts.to(device)).cpu()
    moved = _nearest_index((q + t).to(device), (pts + t).to(device)).cpu()
    ref_native = torch.cdist(q.double(), pts.double()).argmin(-1)
    ref_moved = torch.cdist((q + t).double(), (pts + t).double()).argmin(-1)
    changed = native != moved
    assert not (changed & (ref_native == ref_moved)).any(), int((changed & (ref_native == ref_moved)).sum())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dim", [2, 3])
def test_search_is_exact_on_random_clouds(device, dim):
    """Float32 random clouds (queries both inside and well outside the boundary
    cloud) against the float64 brute force on the same inputs, in 2-D and 3-D."""
    g = torch.Generator().manual_seed(dim)
    pts = (torch.randn(6000, dim, dtype=D, generator=g) * torch.arange(dim, 0, -1, dtype=D)).float()
    q = (torch.randn(3000, dim, dtype=D, generator=g) * 2.0).float()
    idx = _nearest_index(q.to(device), pts.to(device)).cpu()
    assert _assert_indices_exact(idx, q, pts) == 0


@pytest.mark.parametrize("device", DEVICES)
def test_torch_search_path_chunks_and_matches_the_kernel(device, monkeypatch):
    """The chunked torch path (CPU; CUDA without Triton) returns the exact indices
    across chunk boundaries, in float32 and float64, and agrees with the Triton
    kernel where both run."""
    pts, _, q = _wall_problem(3, n=1500, n_q=2 * geometry._TORCH_SEARCH_CHUNK + 17)
    pts, q = pts.to(device), q.to(device)
    kernel = _nearest_index(q, pts) if device == "cuda" and geometry.HAS_TRITON else None
    monkeypatch.setattr(geometry, "HAS_TRITON", False)
    for dtype in (torch.float32, torch.float64):
        idx = _nearest_index(q.to(dtype), pts.to(dtype))
        assert idx.shape == (q.shape[0],) and idx.dtype == torch.int64
        assert _assert_indices_exact(idx.cpu(), q.cpu().to(dtype), pts.cpu().to(dtype)) <= 2
    if kernel is not None:
        ### the two float32 sums may round differently on a tie at float32 resolution
        torch_idx = _nearest_index(q, pts)
        dist = torch.cdist(q.double(), pts.double())
        d_a = dist.gather(-1, torch_idx[:, None])[:, 0]
        d_b = dist.gather(-1, kernel[:, None])[:, 0]
        assert ((torch_idx == kernel) | ((d_a - d_b).abs() <= 4 * torch.finfo(torch.float32).eps * d_a)).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the saved case runs on the CUDA search")
@pytest.mark.skipif("ISLA_AF_POSED_SAMPLE" not in os.environ, reason="set ISLA_AF_POSED_SAMPLE to a forward-kwargs .pt saved by the posed-inference reproduction")
def test_saved_posed_airfrans_case_runs():
    """The AirfRANS posed validation case that crashed the 2026-10-03 inference, as
    saved forward kwargs (boundary polyline, 100,000 interior points with the stored
    sdf / sdf_normals, inlet vectors), runs through a model of its schema."""
    saved = torch.load(os.environ["ISLA_AF_POSED_SAMPLE"], weights_only=False)
    kw = saved["forward_kwargs"]
    kw = {
        "boundary_meshes": {k: v.to("cuda") for k, v in kw["boundary_meshes"].items()},
        "prediction_points": kw["prediction_points"].to("cuda"),
        "global_data": kw["global_data"].to("cuda"),
    }
    torch.manual_seed(0)
    m = ISLA(
        output_schema={"velocity": {"rank": 1}, "pressure": {"rank": 0}, "nut": {"rank": 0}},
        global_schema={"U_inf_dir": {"rank": 1}, "U_inf_mag": {"rank": 0}},
        query_schema={"sdf": {"rank": 0}, "sdf_normals": {"rank": 1}},
        dim=2, interior=True, hidden=32, n_layers=2, n_slices=8,
    ).cuda().eval()
    with torch.no_grad():
        out = outputs(m(**kw))
    assert all(torch.isfinite(v).all() for v in out.values())


### ---------------------------------------------------------------------------
### soft_boundary_geometry: the measure-weighted kernel average over the samples
### ---------------------------------------------------------------------------


def _wall_samples(n, seed, clustered: bool, extent=12.0):
    """A flat wall z = 0 with outward normal +z, sampled uniformly or clustered
    towards x = 0, each sample carrying a Horvitz-Thompson-like measure (inverse
    inclusion density), so both samples represent the same wall measure."""
    g = torch.Generator().manual_seed(seed)
    if clustered:
        x = torch.empty(0, dtype=D)
        while x.numel() < n:
            cand = (torch.rand(4 * n, dtype=D, generator=g) * 2 - 1) * extent
            keep = torch.rand(4 * n, dtype=D, generator=g) < torch.exp(-cand.abs() / 3.0)
            x = torch.cat([x, cand[keep]])[:n]
        w = torch.exp(x.abs() / 3.0)
    else:
        x = (torch.rand(n, dtype=D, generator=g) * 2 - 1) * extent
        w = torch.ones(n, dtype=D)
    w = w / w.sum() * (2 * extent) ** 2
    y = (torch.rand(n, dtype=D, generator=g) * 2 - 1) * extent
    pts = torch.stack([x, y, torch.zeros_like(x)], -1)
    nrm = torch.zeros_like(pts)
    nrm[:, 2] = 1.0
    return pts[None], nrm[None], torch.log(w)[None]


def test_soft_geometry_is_exact_at_a_flat_wall_for_any_sample():
    """Every sample's tangent plane is the wall, so the kernel-mean plane distance is
    the wall distance and the kernel-mean normal the wall normal, whatever the sample
    and its law; the nearest sample's Euclidean distance is not."""
    q = torch.tensor([[[0.3, -0.2, 1e-3], [0.1, 0.4, 0.05], [-0.7, 0.2, 0.5], [0.2, 0.1, 2.0], [0.0, 0.0, -0.3]]], dtype=D)
    expect = q[0, :, 2]
    for clustered in (False, True):
        pts, nrm, lw = _wall_samples(4000, 0 if not clustered else 1, clustered)
        d, u = soft_boundary_geometry(q, pts, nrm, lw, width=0.5)
        assert torch.allclose(d[0], expect, atol=1e-12, rtol=0), (clustered, d[0], expect)
        ### the direction is the wall's outward normal on both sides (the side is in d's sign)
        assert torch.allclose(u[0], torch.tensor([0.0, 0.0, 1.0], dtype=D).expand(5, 3), atol=1e-12)
        d_near, _ = nearest_boundary_geometry(q, pts, nrm)
        ### the nearest-sample distance carries the sample spacing (~0.4 here)
        assert (d_near[0, :3].abs() - expect[:3].abs()).max() > 0.05


@pytest.mark.parametrize("one_sided", [True, False])
def test_soft_geometry_sphere_sign_direction_and_distance(one_sided):
    """A sphere with outward normals: outside queries have d > 0, inside queries
    d < 0, and u is the outward radial on BOTH sides (the kernel-mean normal is the
    gradient of the signed distance, which points outward everywhere); |d| is within
    the kernel's curvature bias of the exact distance and u within a milliradian of
    radial, for the one-sided and the two-sided average alike."""
    g = torch.Generator().manual_seed(0)
    n = 20000
    dirs = torch.nn.functional.normalize(torch.randn(1, n, 3, dtype=D, generator=g), dim=-1)
    pts, nrm = 2.0 * dirs, dirs
    lw = torch.full((1, n), math.log(4 * math.pi * 4.0 / n), dtype=D)
    q_dirs = torch.nn.functional.normalize(torch.randn(1, 30, 3, dtype=D, generator=g), dim=-1)
    radii = torch.cat([torch.full((15,), 3.0, dtype=D), torch.full((15,), 1.0, dtype=D)])
    q = q_dirs * radii[None, :, None]
    d, u = soft_boundary_geometry(q, pts, nrm, lw, width=0.1, one_sided=one_sided)
    assert (d[0, :15] > 0).all() and (d[0, 15:] < 0).all()
    assert torch.allclose(d[0, :15], torch.ones(15, dtype=D), atol=0.1)
    assert torch.allclose(d[0, 15:], -torch.ones(15, dtype=D), atol=0.1)
    assert ((u[0] * q_dirs[0]).sum(-1) > 0.999).all()
    ### the nearest form's direction points from the boundary to the query instead:
    ### it agrees outside and is opposite inside
    _, u_near = nearest_boundary_geometry(q, pts, nrm)
    assert ((u[0, :15] * u_near[0, :15]).sum(-1) > 0.99).all()
    assert ((u[0, 15:] * u_near[0, 15:]).sum(-1) < -0.99).all()


def test_soft_geometry_two_sided_form_is_continuous_across_a_curved_wall():
    """On a curved wall the one-sided average jumps across the wall by about the
    curvature times the mean squared tangential offset of the kernel (the inside
    branch averages tangent planes the query lies behind, the outside branch only
    those it lies in front of); the two-sided average is continuous there, with unit
    slope. Queries at +-delta of a sphere of radius 2 (curvature 0.5) sampled at a
    spacing h = 0.018 and read at width 0.2: delta = 2e-4 is above the sampling
    offset kappa h^2 / 8 = 2e-5 (below it even an outside query sits behind its
    nearest sample's plane and both forms call it inside) and far below the
    kernel's curvature bias, about 0.06 here."""
    g = torch.Generator().manual_seed(1)
    n = 160000
    dirs = torch.nn.functional.normalize(torch.randn(1, n, 3, dtype=D, generator=g), dim=-1)
    pts, nrm = 2.0 * dirs, dirs
    lw = torch.full((1, n), math.log(4 * math.pi * 4.0 / n), dtype=D)
    q_dirs = torch.nn.functional.normalize(torch.randn(1, 12, 3, dtype=D, generator=g), dim=-1)
    delta = 2e-4
    q_out, q_in = q_dirs * (2.0 + delta), q_dirs * (2.0 - delta)
    for one_sided in (True, False):
        d_out, u_out = soft_boundary_geometry(q_out, pts, nrm, lw, width=0.2, one_sided=one_sided)
        d_in, u_in = soft_boundary_geometry(q_in, pts, nrm, lw, width=0.2, one_sided=one_sided)
        step = d_out - d_in
        ### the direction is continuous in both forms (outward radial on both sides)
        assert ((u_out[0] * u_in[0]).sum(-1) > 0.9999).all(), one_sided
        if one_sided:
            assert (d_out > 0).all() and (d_in < 0).all()
            assert step.min() > 20 * 2 * delta, step  # a jump of order kappa tau^2, not a 2 delta displacement
        else:
            assert ((step - 2 * delta).abs() < 0.2 * 2 * delta).all(), step  # continuous with unit slope
            assert (d_in < 0).all() and (d_out < 0).all()  # both within the bias of the wall: the zero set sits outside


def test_soft_geometry_thin_plate_reads_the_near_face_only():
    """Two faces of a plate (thickness 0.01, far below the kernel width 0.5). A query
    above reads the upper face (d = 0.3, u = +z), a query below the lower face
    (d = 0.3, u = -z); a query inside reads the nearer face (d = -0.004, u = +z, the
    upper face's outward normal): the facing factor keeps the far face, whose normal
    points the other way, out of the average, even though the query lies behind both
    faces' planes. Both kernel forms agree here (flat faces carry no curvature)."""
    g = torch.Generator().manual_seed(0)
    n = 3000
    xy = (torch.rand(n, 2, dtype=D, generator=g) * 2 - 1) * 10.0
    up = torch.cat([xy, torch.zeros(n, 1, dtype=D)], -1)
    lo = torch.cat([xy, torch.full((n, 1), -0.01, dtype=D)], -1)
    pts = torch.cat([up, lo])[None]
    nrm = torch.cat([torch.tensor([[0.0, 0.0, 1.0]]).to(D).expand(n, 3), torch.tensor([[0.0, 0.0, -1.0]]).to(D).expand(n, 3)])[None]
    lw = torch.full((1, 2 * n), math.log(400.0 / n), dtype=D)
    q = torch.tensor([[[0.2, 0.1, 0.3], [-0.3, 0.4, -0.31], [0.1, -0.2, -0.004]]], dtype=D)
    for one_sided in (True, False):
        d, u = soft_boundary_geometry(q, pts, nrm, lw, width=0.5, one_sided=one_sided)
        assert torch.allclose(d[0], torch.tensor([0.3, 0.3, -0.004], dtype=D), atol=1e-12), one_sided
        assert torch.allclose(u[0, 0], torch.tensor([0.0, 0.0, 1.0], dtype=D), atol=1e-12)
        assert torch.allclose(u[0, 1], torch.tensor([0.0, 0.0, -1.0], dtype=D), atol=1e-12)
        assert torch.allclose(u[0, 2], torch.tensor([0.0, 0.0, 1.0], dtype=D), atol=1e-12)


def test_soft_geometry_concave_corner_reads_the_nearest_wall():
    """Two walls meeting at a right angle (floor z = 0 facing +z, wall x = 0 facing
    +x), both in front of a query in the corner region: the facing factor excludes
    the wall whose normal is perpendicular to the nearest sample's, so the distance
    is the nearest wall's and the direction its normal, as the exact field has."""
    g = torch.Generator().manual_seed(0)
    n = 4000
    ab = torch.rand(n, 2, dtype=D, generator=g) * 10.0
    floor = torch.stack([ab[:, 0], ab[:, 1] - 5.0, torch.zeros(n, dtype=D)], -1)
    wall = torch.stack([torch.zeros(n, dtype=D), ab[:, 1] - 5.0, ab[:, 0]], -1)
    pts = torch.cat([floor, wall])[None]
    nrm = torch.cat([torch.tensor([[0.0, 0.0, 1.0]]).to(D).expand(n, 3), torch.tensor([[1.0, 0.0, 0.0]]).to(D).expand(n, 3)])[None]
    lw = torch.full((1, 2 * n), math.log(100.0 / n), dtype=D)
    q = torch.tensor([[[0.4, 0.0, 0.1], [0.1, 0.3, 0.4]]], dtype=D)  # nearer the floor; nearer the wall
    d, u = soft_boundary_geometry(q, pts, nrm, lw, width=0.5)
    assert torch.allclose(d[0], torch.tensor([0.1, 0.1], dtype=D), atol=1e-12)
    assert torch.allclose(u[0, 0], torch.tensor([0.0, 0.0, 1.0], dtype=D), atol=1e-12)
    assert torch.allclose(u[0, 1], torch.tensor([1.0, 0.0, 0.0], dtype=D), atol=1e-12)


def test_soft_geometry_is_covariant_and_scales():
    pts, nrm, w = _cloud(100, 0)
    q, _, _ = _cloud(20, 1, scale=(4.0, 4.0, 4.0))
    lw = torch.log(w)
    R = orthogonal(3, 0, improper=True)
    t = torch.tensor([1.0, -2.0, 0.5], dtype=D)
    d, u = soft_boundary_geometry(q, pts, nrm, lw, width=0.7)
    d2, u2 = soft_boundary_geometry(q @ R.T + t, pts @ R.T + t, nrm @ R.T, lw, width=0.7)
    assert torch.allclose(d2, d, atol=1e-12) and torch.allclose(u2, u @ R.T, atol=1e-12)
    d3, u3 = soft_boundary_geometry(3.0 * q, 3.0 * pts, nrm, lw, width=3.0 * 0.7)
    assert torch.allclose(d3, 3.0 * d, atol=1e-12) and torch.allclose(u3, u, atol=1e-12)
    ### a uniform rescale of the measures changes nothing
    d4, u4 = soft_boundary_geometry(q, pts, nrm, lw + 2.0, width=0.7)
    assert torch.allclose(d4, d, atol=1e-12) and torch.allclose(u4, u, atol=1e-12)
    ### equal measures by default
    d5, _ = soft_boundary_geometry(q, pts, nrm, None, width=0.7)
    d6, _ = soft_boundary_geometry(q, pts, nrm, torch.zeros_like(lw), width=0.7)
    assert torch.equal(d5, d6)
    ### the sign agrees with the nearest form's; the direction is the outward normal,
    ### so it agrees with the nearest form's boundary-to-query direction outside and
    ### opposes it inside
    d_near, u_near = nearest_boundary_geometry(q, pts, nrm)
    assert torch.equal(torch.sign(d), torch.sign(d_near))
    assert (torch.sign(d) * (u * u_near).sum(-1) > 0).all()


def test_soft_geometry_width_is_a_length_per_batch_element():
    pts, nrm, w = _cloud(80, 0, batch=2)
    q, _, _ = _cloud(12, 1, batch=2, scale=(4.0, 4.0, 4.0))
    d_a, _ = soft_boundary_geometry(q, pts, nrm, torch.log(w), width=torch.tensor([0.5, 1.5], dtype=D))
    d_b, _ = soft_boundary_geometry(q[:1], pts[:1], nrm[:1], torch.log(w[:1]), width=0.5)
    d_c, _ = soft_boundary_geometry(q[1:], pts[1:], nrm[1:], torch.log(w[1:]), width=1.5)
    assert torch.allclose(d_a, torch.cat([d_b, d_c]), atol=1e-12)
    with pytest.raises(ValueError, match="positive length"):
        soft_boundary_geometry(q, pts, nrm, None, width=0.0)
    with pytest.raises(ValueError, match="positive length"):
        soft_boundary_geometry(q, pts, nrm, None, width=torch.tensor([0.5, 0.5, 0.5], dtype=D))


def test_soft_geometry_is_differentiable_in_the_positions():
    pts, nrm, w = _cloud(50, 0)
    q = _cloud(9, 1, scale=(4.0, 4.0, 4.0))[0].requires_grad_(True)
    d, u = soft_boundary_geometry(q, pts, nrm, torch.log(w), width=0.8)
    (d.sum() + u.sum()).backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


def test_soft_geometry_two_laws_agree_where_the_nearest_form_does_not():
    """The same sphere sampled under two laws (uniform on the sphere; clustered towards
    one pole, with inverse-density measures), at a kernel width of a few spacings
    under the sparser law: the kernel average moves far less between the two samples
    than the nearest-sample geometry does, in distance and in direction."""
    g = torch.Generator().manual_seed(3)
    n = 20000

    def sample(clustered):
        dirs = torch.nn.functional.normalize(torch.randn(8 * n, 3, dtype=D, generator=g), dim=-1)
        if clustered:
            dens = torch.exp(1.0 * dirs[:, 2])  # inclusion density, up to a constant (e^2 range)
            keep = torch.rand(8 * n, dtype=D, generator=g) < dens / dens.max()
            dirs, dens = dirs[keep][:n], dens[keep][:n]
        else:
            dirs, dens = dirs[:n], torch.ones(n, dtype=D)
        w = 1.0 / dens
        w = w / w.sum() * 4 * math.pi * 4.0
        return (2.0 * dirs)[None], dirs[None], torch.log(w)[None]

    q_dirs = torch.nn.functional.normalize(torch.randn(1, 200, 3, dtype=D, generator=g), dim=-1)
    q = q_dirs * (2.0 + 0.02 + 0.3 * torch.rand(1, 200, 1, dtype=D, generator=g))
    a, b = sample(False), sample(True)
    d_soft_a, u_soft_a = soft_boundary_geometry(q, *a, width=0.3)
    d_soft_b, u_soft_b = soft_boundary_geometry(q, *b, width=0.3)
    d_near_a, u_near_a = nearest_boundary_geometry(q, a[0], a[1])
    d_near_b, u_near_b = nearest_boundary_geometry(q, b[0], b[1])
    shift_soft = torch.log(d_soft_a.abs() / d_soft_b.abs()).pow(2).mean().sqrt()
    shift_near = torch.log(d_near_a.abs() / d_near_b.abs()).pow(2).mean().sqrt()
    ### measured 2026-10-06 on GB300: 0.036 against 0.138 (a factor 3.9); the bar is a factor 2
    assert shift_soft < 0.5 * shift_near, (shift_soft, shift_near)
    turn_soft = (1 - (u_soft_a * u_soft_b).sum(-1)).mean()
    turn_near = (1 - (u_near_a * u_near_b).sum(-1)).mean()
    assert turn_soft < 0.5 * turn_near, (turn_soft, turn_near)


@pytest.mark.parametrize("dim", [2, 3])
def test_soft_geometry_torch_path_chunks_and_matches_the_unchunked_form(dim, monkeypatch):
    pts, nrm, w, _ = cloud(300, d=dim, seed=0)
    q, _, _, _ = cloud(700, d=dim, seed=1, scale=tuple(4.0 for _ in range(dim)))
    lw = torch.log(w)
    monkeypatch.setattr(geometry, "_TORCH_SOFT_CHUNK", 64)
    d, u = soft_boundary_geometry(q[None], pts[None], nrm[None], lw[None], width=0.9)
    monkeypatch.setattr(geometry, "_TORCH_SOFT_CHUNK", 100000)
    d2, u2 = soft_boundary_geometry(q[None], pts[None], nrm[None], lw[None], width=0.9)
    assert torch.allclose(d, d2, atol=1e-12) and torch.allclose(u, u2, atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available() or not geometry.HAS_TRITON, reason="CUDA + Triton")
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_soft_geometry_kernel_matches_the_torch_form_on_cuda(dim, dtype):
    """The streaming kernel against the chunked PyTorch form (taken by forcing a
    gradient requirement) on random clouds, in both precisions."""
    dev = torch.device("cuda")
    pts, nrm, w, _ = cloud(5000, d=dim, seed=0, dtype=dtype)
    q, _, _, _ = cloud(3000, d=dim, seed=1, dtype=dtype, scale=tuple(4.0 for _ in range(dim)))
    lw = torch.log(w)
    q, pts, nrm, lw = (t.to(dev) for t in (q, pts, nrm, lw))
    d, u = soft_boundary_geometry(q[None], pts[None], nrm[None], lw[None], width=0.6)
    with torch.enable_grad():
        d_ref, u_ref = soft_boundary_geometry(q[None].clone().requires_grad_(True), pts[None], nrm[None], lw[None], width=0.6)
    tol = 1e-5 if dtype == torch.float32 else 1e-12
    assert torch.allclose(d, d_ref.detach(), atol=tol, rtol=tol), (d - d_ref).abs().max()
    assert torch.allclose(u, u_ref.detach(), atol=tol, rtol=tol), (u - u_ref).abs().max()


def test_soft_geometry_model_contracts_and_checkpoint(tmp_path):
    """The interior model with query_geometry='soft' keeps every covariance contract,
    records its geometry arguments in the checkpoint, and differs from the nearest
    form (the channel is live)."""
    from physicsnemo.core.module import Module

    pts, nrm, w, _ = cloud(200)
    mesh = soup(pts, nrm, w)
    drv = torch.tensor([0.3, 1.0, 0.2], dtype=D)
    q, _, _, _ = cloud(40, seed=1, scale=(4.0, 4.0, 4.0))
    m = _model(query_geometry="soft", query_geometry_width=0.05)
    assert m.query_geometry == "soft" and m.query_geometry_width == 0.05

    def call(model, R=None, t=None, k=1.0):
        R = torch.eye(3, dtype=D) if R is None else R
        t = torch.zeros(3, dtype=D) if t is None else t
        with torch.no_grad():
            return outputs(model(boundary_meshes={"wall": transform(mesh, R, t, scale=k)}, prediction_points=k * q @ R.T + t, global_data={"g": drv @ R.T}))

    base = call(m)
    for improper in (False, True):
        R = orthogonal(3, 4, improper)
        out = call(m, R, torch.tensor([2.0, -5.0, 1.0], dtype=D), k=2.5)
        assert torch.allclose(out["p"], base["p"], atol=1e-11), improper
        assert torch.allclose(out["v"], base["v"] @ R.T, atol=1e-11), improper
    near = call(_model())
    assert not torch.allclose(near["p"], base["p"], atol=1e-6)
    ### the two-sided form is a third, distinct geometry with the same contracts
    torch.manual_seed(0)
    smooth = _model(query_geometry="smooth", query_geometry_width=0.05)
    smooth.load_state_dict(m.state_dict())
    base_s = call(smooth)
    assert not torch.allclose(base_s["p"], base["p"], atol=1e-6)
    R = orthogonal(3, 5, True)
    out_s = call(smooth, R, torch.tensor([1.0, 2.0, -3.0], dtype=D), k=0.5)
    assert torch.allclose(out_s["p"], base_s["p"], atol=1e-11)
    assert torch.allclose(out_s["v"], base_s["v"] @ R.T, atol=1e-11)

    m.save(str(tmp_path / "soft.mdlus"))
    loaded = Module.from_checkpoint(str(tmp_path / "soft.mdlus")).double().eval()
    assert loaded.query_geometry == "soft" and loaded.query_geometry_width == 0.05
    assert torch.allclose(call(loaded)["p"], base["p"], atol=1e-12)
    smooth.save(str(tmp_path / "smooth.mdlus"))
    assert Module.from_checkpoint(str(tmp_path / "smooth.mdlus")).query_geometry == "smooth"
    with pytest.raises(ValueError, match="query_geometry must be"):
        _model(query_geometry="plane")
    with pytest.raises(ValueError, match="query_geometry_width"):
        _model(query_geometry="soft", query_geometry_width=0.0)


@pytest.mark.skipif(not torch.cuda.is_available() or not geometry.HAS_TRITON, reason="CUDA + Triton")
def test_soft_geometry_kernel_is_finite_when_the_first_tiles_hold_no_kept_sample():
    """The streaming kernel's online rescale must stay finite for a query whose first
    token tiles are all excluded by the side and facing rules (the far face of a plate
    listed first): the first soft lanes trained on AirfRANS went NaN at epoch 3 through
    exp(-inf - (-inf)) in exactly this case (2026-10-06)."""
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(0)
    n = 3000
    xy = (torch.rand(n, 2, dtype=torch.float32, generator=g) * 2 - 1) * 10.0
    lo = torch.cat([xy, torch.full((n, 1), -0.01)], -1)  # far face first in the token order
    up = torch.cat([xy, torch.zeros(n, 1)], -1)
    pts = torch.cat([lo, up])[None].to(dev)
    nrm = torch.cat([torch.tensor([[0.0, 0.0, -1.0]]).expand(n, 3), torch.tensor([[0.0, 0.0, 1.0]]).expand(n, 3)])[None].to(dev)
    q = torch.tensor([[[0.2, 0.1, 0.3], [-3.0, 4.0, 0.05], [0.1, -0.2, 0.001]]], device=dev)
    d, u = soft_boundary_geometry(q, pts, nrm, None, width=0.5)
    assert torch.isfinite(d).all() and torch.isfinite(u).all()
    assert torch.allclose(d[0], q[0, :, 2], atol=1e-5)
    assert torch.allclose(u[0], torch.tensor([[0.0, 0.0, 1.0]], device=dev).expand(3, 3), atol=1e-5)
    with torch.enable_grad():
        d_ref, u_ref = soft_boundary_geometry(q.clone().requires_grad_(True), pts, nrm, None, width=0.5)
    assert torch.allclose(d, d_ref.detach(), atol=1e-5) and torch.allclose(u, u_ref.detach(), atol=1e-5)


def _circle_with_wall_nodes(n: int, dtype):
    """A closed polygon of n equal segments on the unit circle (tokens at the segment
    centroids, outward normals, equal lengths) and queries AT its vertices: points
    exactly on the sampled boundary, as the volume meshes of the program's datasets
    carry the wall nodes among their interior points."""
    t = torch.linspace(0, 2 * math.pi, n + 1, dtype=torch.float64)[:-1]
    verts = torch.stack([torch.cos(t), torch.sin(t)], -1)
    cent = 0.5 * (verts + verts.roll(-1, 0))
    nrm = cent / cent.norm(dim=-1, keepdim=True)
    length = (verts.roll(-1, 0) - verts).norm(dim=-1)
    return cent.to(dtype)[None], nrm.to(dtype)[None], torch.log(length).to(dtype)[None], verts.to(dtype)[None]


@pytest.mark.parametrize("one_sided", [True, False])
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_soft_geometry_wall_nodes_keep_a_unit_direction_that_survives_a_pose_in_float32(device, one_sided):
    """For a query on the wall the plane distance to its nearest sample is a
    roundoff-level number, so its side is decided by rounding and may differ between
    the host and the kernel, and may change under a rigid motion in float32. Measured
    on AirfRANS (2026-10-07): the first soft form returned a zero direction at ~3% of
    the wall nodes (the nearest sample dropped from an otherwise empty kept set) and
    reversed the direction at ~45% of them under the posed protocol, which moved the
    trained model's error by 1 to 6 percent. Now the nearest sample is always kept and
    the direction is the kernel-mean outward normal, independent of the side: unit
    length everywhere, within a degree of the exact radial, and the same (mapped back)
    after a rigid motion; the two-sided form's distance also survives the motion."""
    dev = torch.device(device)
    pts, nrm, lw, q = (t.to(dev) for t in _circle_with_wall_nodes(720, torch.float32))
    d, u = soft_boundary_geometry(q, pts, nrm, lw, width=0.05, one_sided=one_sided)
    assert torch.isfinite(d).all()
    assert torch.allclose(u.norm(dim=-1), torch.ones_like(d), atol=1e-5)
    radial = q / q.norm(dim=-1, keepdim=True)
    assert ((u * radial).sum(-1) > math.cos(math.radians(1.0))).all()
    ### |d| at a wall node is at most the curvature bias kappa tau^2 (0.0025 here)
    assert (d.abs() < 0.01).all()
    theta = 0.7
    R = torch.tensor([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=torch.float32, device=dev)
    t = torch.tensor([3.0, -1.5], dtype=torch.float32, device=dev)
    d2, u2 = soft_boundary_geometry(q @ R.T + t, pts @ R.T + t, nrm @ R.T, lw, width=0.05, one_sided=one_sided)
    assert ((u2 @ R) * u).sum(-1).min() > math.cos(math.radians(1.0))
    if not one_sided:
        assert torch.allclose(d2, d, atol=1e-4)


### ---------------------------------------------------------------------------
### The patch form: the exact signed distance to the finite patches the tokens'
### measures define, with the winding-number side (two dimensions).


def _polygon(vertices):
    """Segment tokens of a closed counter-clockwise polygon: centroids, outward
    unit normals, lengths, and the vertex pairs."""
    a, b = vertices, torch.roll(vertices, -1, 0)
    tangent = b - a
    length = tangent.norm(dim=-1)
    normal = torch.stack([tangent[:, 1], -tangent[:, 0]], -1) / length[:, None]
    return 0.5 * (a + b), normal, length, a, b


def _polyline_distance(q, a, b):
    """Exact distance from each query to a closed polyline, and the foot."""
    ab = b - a
    t = (((q[:, None, :] - a[None]) * ab[None]).sum(-1) / (ab * ab).sum(-1)[None]).clamp(0, 1)
    p = a[None] + t[..., None] * ab[None]
    dist, i = (q[:, None, :] - p).norm(dim=-1).min(dim=1)
    return dist, p[torch.arange(q.shape[0]), i]


def _naca_polygon(n_per_side=400):
    """A closed symmetric airfoil with a sharp trailing edge (NACA 0012 ordinates),
    counter-clockwise."""
    xs = torch.linspace(0, 1, n_per_side + 1, dtype=D)
    th = 0.12 / 0.2 * (0.2969 * xs.sqrt() - 0.1260 * xs - 0.3516 * xs**2 + 0.2843 * xs**3 - 0.1036 * xs**4)
    upper = torch.stack([xs, th], -1)
    lower = torch.stack([xs.flip(0), -th.flip(0)], -1)
    v = torch.cat([upper[1:], lower[1:-1]])
    if (v[:, 0] * torch.roll(v, -1, 0)[:, 1] - torch.roll(v, -1, 0)[:, 0] * v[:, 1]).sum() < 0:
        v = v.flip(0)
    return v


def test_patch_geometry_is_the_exact_signed_distance_of_a_closed_polygon():
    """On its own cells the patch form is the polygon's exact signed distance:
    distance to rounding, sign by inside/outside, direction the gradient."""
    th = torch.linspace(0, 2 * math.pi, 301, dtype=D)[:-1]
    v = torch.stack([1.5 * torch.cos(th), torch.sin(th)], -1)  # an ellipse
    y, n, length, a, b = _polygon(v)
    g = torch.Generator().manual_seed(3)
    q = torch.randn(4000, 2, dtype=D, generator=g) * 1.6
    d, u = patch_boundary_geometry(q[None], y[None], n[None], torch.log(length)[None])
    exact, foot = _polyline_distance(q, a, b)
    inside = (q[:, 0] / 1.5) ** 2 + q[:, 1] ** 2 < 1.0
    away = (exact > 1e-3)  # off the facets' own thickness
    assert torch.allclose(d[0].abs(), exact, atol=1e-10)
    assert bool(((d[0] < 0) == inside)[away].all())
    grad = torch.nn.functional.normalize(q - foot, dim=-1) * torch.where(inside, -1.0, 1.0)[:, None]
    assert float((1 - (u[0] * grad).sum(-1))[away & (exact > 0.05)].max()) < 1e-3


def test_patch_geometry_side_behind_a_sharp_edge_is_the_winding_number():
    """Behind the trailing edge of a thin closed body the nearest patch's plane
    puts a third of the wake points inside; the winding number puts them all
    outside, and the distance is the distance to the edge."""
    v = _naca_polygon()
    y, n, length, a, b = _polygon(v)
    g = torch.Generator().manual_seed(4)
    L = torch.rand(3000, dtype=D, generator=g) * 0.5 + 2e-3
    ang = (torch.rand(3000, dtype=D, generator=g) - 0.5) * math.radians(40)
    q = torch.stack([1.0 + L * torch.cos(ang), L * torch.sin(ang)], -1)
    d, u = patch_boundary_geometry(q[None], y[None], n[None], torch.log(length)[None])
    exact, _ = _polyline_distance(q, a, b)
    assert bool((d[0] > 0).all())
    assert torch.allclose(d[0], exact, atol=1e-10)
    ### the discriminating control: the nearest patch's plane alone reads a side
    diff = q[:, None, :] - y[None]
    s = (diff * n[None]).sum(-1)
    tang = (diff - s[..., None] * n[None]).norm(dim=-1)
    dist = torch.sqrt(s * s + (tang - 0.5 * length[None]).clamp_min(0) ** 2)
    s_star = s.gather(1, dist.argmin(1, keepdim=True))[:, 0]
    assert float((s_star < 0).double().mean()) > 0.2
    ### the direction is radial from the edge
    radial = torch.nn.functional.normalize(q - torch.tensor([1.0, 0.0], dtype=D), dim=-1)
    assert float((1 - (u[0] * radial).sum(-1)).max()) < 1e-3


def test_patch_geometry_keeps_the_wake_side_under_a_measure_weighted_subsample():
    """A Poisson subsample of the cells with inverse-inclusion measures (the
    patches tile the surface in expectation) keeps the side of every wake point
    and of the near-wall points farther than the facet thickness outside the
    band the sample cannot resolve; the nearest form flips wake points."""
    v = _naca_polygon(1000)
    y, n, length, a, b = _polygon(v)
    g = torch.Generator().manual_seed(5)
    keep = torch.rand(y.shape[0], generator=g) < 0.4
    ys, ns, ls = y[keep], n[keep], length[keep] / 0.4
    L = torch.rand(2000, dtype=D, generator=g) * 0.5 + 2e-3
    ang = (torch.rand(2000, dtype=D, generator=g) - 0.5) * math.radians(40)
    wake = torch.stack([1.0 + L * torch.cos(ang), L * torch.sin(ang)], -1)
    xs = torch.rand(4000, dtype=D, generator=g) * 0.96 + 0.02
    th = 0.12 / 0.2 * (0.2969 * xs.sqrt() - 0.1260 * xs - 0.3516 * xs**2 + 0.2843 * xs**3 - 0.1036 * xs**4)
    h = 10 ** (torch.rand(4000, dtype=D, generator=g) * 3 - 5)
    side = torch.where(torch.rand(4000, generator=g) < 0.5, 1.0, -1.0).to(D)
    near = torch.stack([xs, side * (th + h)], -1)
    d_w, _ = patch_boundary_geometry(wake[None], ys[None], ns[None], torch.log(ls)[None])
    d_n, u_n = patch_boundary_geometry(near[None], ys[None], ns[None], torch.log(ls)[None])
    assert bool((d_w[0] > 0).all())
    exact, _ = _polyline_distance(near, a, b)
    ### the resolution limit: within a kept-cell spacing of the trailing edge the
    ### body is thinner than the sample resolves, a band of a few points in a thousand
    assert float((d_n[0][exact > 1e-3] < 0).double().mean()) < 5e-3
    ### the subsample's wall normal, not a tilt towards a gap's rim
    assert float((u_n[0][:, 1] * side).median()) > 0.95
    d_near_form, _ = nearest_boundary_geometry(wake[None], ys[None], ns[None])
    assert float((d_near_form[0] < 0).double().mean()) > 0.05


def test_patch_geometry_covariance_and_scale():
    """Invariance of the distance and covariance of the direction under a rotation,
    a reflection and a translation; the distance scales with the geometry."""
    v = _naca_polygon(200)
    y, n, length, _, _ = _polygon(v)
    g = torch.Generator().manual_seed(6)
    q = torch.randn(500, 2, dtype=D, generator=g) * 0.6 + torch.tensor([0.5, 0.0], dtype=D)
    d, u = patch_boundary_geometry(q[None], y[None], n[None], torch.log(length)[None])
    ### within one cell of a vertex two patches tie exactly and rounding picks the
    ### normal of one or the other; everywhere else the direction is single-valued
    clear = d[0].abs() > 2 * length.max()
    for improper in (False, True):
        R = orthogonal(2, 7 + int(improper), improper)
        t = torch.tensor([3.0, -2.0], dtype=D)
        k = 2.5
        d2, u2 = patch_boundary_geometry(
            (k * q @ R.T + t)[None], (k * y @ R.T + t)[None], (n @ R.T)[None], torch.log(k * length)[None]
        )
        assert torch.allclose(d2, k * d, atol=1e-10), improper
        assert torch.allclose(u2[0][clear], (u @ R.T)[0][clear], atol=1e-9), improper


def test_patch_geometry_rejects_three_dimensions_and_a_negative_reach():
    pts, nrm, w, _ = cloud(50)
    with pytest.raises(NotImplementedError, match="two spatial dimensions"):
        patch_boundary_geometry(pts[None], pts[None], nrm[None], torch.log(w)[None])
    pts2, nrm2, w2, _ = cloud(50, d=2)
    with pytest.raises(ValueError, match="reach"):
        patch_boundary_geometry(pts2[None], pts2[None], nrm2[None], torch.log(w2)[None], reach=-1.0)


def test_patch_geometry_model_contracts_and_checkpoint(tmp_path):
    """The two-dimensional interior model with query_geometry='patch' keeps every
    covariance contract, records the option in the checkpoint, and differs from
    the nearest form (the channel is live)."""
    from physicsnemo.core.module import Module

    pts, nrm, w, _ = cloud(120, d=2)
    mesh = soup(pts, nrm, w)
    drv = torch.tensor([0.3, 1.0], dtype=D)
    q, _, _, _ = cloud(30, d=2, seed=1, scale=(3.0, 3.0))
    torch.manual_seed(0)
    m = ISLA(output_schema=PV, global_schema={"g": {"rank": 1}}, dim=2, hidden=32, n_layers=2, n_slices=8, interior=True, query_geometry="patch").double().eval()
    assert m.query_geometry == "patch"

    def call(model, R=None, t=None, k=1.0):
        R = torch.eye(2, dtype=D) if R is None else R
        t = torch.zeros(2, dtype=D) if t is None else t
        with torch.no_grad():
            return outputs(model(boundary_meshes={"wall": transform(mesh, R, t, scale=k)}, prediction_points=k * q @ R.T + t, global_data={"g": drv @ R.T}))

    base = call(m)
    for improper in (False, True):
        R = orthogonal(2, 4, improper)
        out = call(m, R, torch.tensor([2.0, -5.0], dtype=D), k=2.5)
        assert torch.allclose(out["p"], base["p"], atol=1e-11), improper
        assert torch.allclose(out["v"], base["v"] @ R.T, atol=1e-11), improper
    torch.manual_seed(0)
    near = ISLA(output_schema=PV, global_schema={"g": {"rank": 1}}, dim=2, hidden=32, n_layers=2, n_slices=8, interior=True).double().eval()
    near.load_state_dict(m.state_dict())
    assert not torch.allclose(call(near)["p"], base["p"], atol=1e-6)
    m.save(str(tmp_path / "patch.mdlus"))
    loaded = Module.from_checkpoint(str(tmp_path / "patch.mdlus")).double().eval()
    assert loaded.query_geometry == "patch"
    assert torch.allclose(call(loaded)["p"], base["p"], atol=1e-12)
