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

"""Contract tests for ISLA's global boundary read (``kernel_read_radius=None``).

Passive prediction points (``query_independent=True``) read every boundary cell: a
softmax over all cells with the log measure fraction as bias, a learned kernel of
the pair invariants (no radius, falloff or neighbour count) and a query-key match.
Every contract of the read-only model must survive: exact query independence,
rotation / reflection / translation covariance, the similarity gauge,
boundary-refinement and measure-scale invariance, a gradient for every parameter,
finite bf16 autocast, and the CUDA fused-kernel and compiled-block training path.
The read's output projection starts at zero, so the tests give every read parameter
a random value first.
"""

import pytest
import torch

import physicsnemo.experimental.models.isla.kernel_read as kernel_read
from physicsnemo.experimental.models import ISLA
from physicsnemo.experimental.models.isla.kernel_read import (
    _GlobalSoftmaxRead,
    global_pair_invariants,
    global_softmax_read_reference,
)
from physicsnemo.mesh.calculus.measure import EFFECTIVE_MEASURE_KEY
from test.experimental.models.isla._meshes import (
    cloud,
    orthogonal,
    outputs,
    points_mesh,
    refine,
    same,
    soup,
    transform,
)

PV = {"p": {"rank": 0}, "v": {"rank": 1}}
G = {"g": {"rank": 1}}
QS = {"sdf": {"rank": 0}, "sdf_normals": {"rank": 1}}
D = torch.float64
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


def _plain(**kw):
    kw = {"hidden": 32, "n_layers": 2, "n_slices": 8, **kw}
    return ISLA(
        output_schema=PV,
        global_schema=G,
        query_schema=QS,
        interior=True,
        query_independent=True,
        **kw,
    )


def _model(dim=3, randomize=True, **kw):
    torch.manual_seed(0)
    m = _plain(kernel_read="mean", kernel_read_radius=None, dim=dim, **kw)
    if randomize:
        with torch.no_grad():
            for p in [*m.kernel_reads.parameters(), *m.kernel_basis.parameters()]:
                p.normal_(0.0, 0.3)
    return m


def _case(dim=3, n=300, nq=120, seed=0):
    """A boundary cloud and prediction points near it plus a few far away, with a
    per-point sdf and a unit vector field."""
    pts, nrm, w, drv = cloud(n, d=dim, seed=seed)
    g = torch.Generator().manual_seed(seed + 1)
    near = pts[: nq - 10] + 0.15 * torch.randn(nq - 10, dim, generator=g, dtype=D)
    far = 40.0 + torch.randn(10, dim, generator=g, dtype=D)
    q = torch.cat([near, far])
    sdf = torch.randn(nq, generator=g, dtype=D) * 0.3
    sn = torch.nn.functional.normalize(
        torch.randn(nq, dim, generator=g, dtype=D), dim=-1
    )
    return soup(pts, nrm, w), drv, q, sdf, sn


def _run(m, mesh, drv, q, sdf, sn):
    with torch.no_grad():
        return m(
            boundary_meshes={"wall": mesh},
            prediction_points=points_mesh(q, {"sdf": sdf, "sdf_normals": sn}),
            global_data={"g": drv},
        )


def _rows(out, idx):
    return {k: v[idx] for k, v in outputs(out).items()}


def test_global_read_is_exactly_query_independent(monkeypatch):
    """A prediction depends on its own point and the boundary only: alone, a prefix,
    any order, any chunking of the request and any tiling of the pairs give the same
    values to float64 roundoff."""
    m = _model().double().eval()
    mesh, drv, q, sdf, sn = _case()
    base = _run(m, mesh, drv, q, sdf, sn)
    assert m.kernel_read_pairs == q.shape[0] * mesh.n_cells
    nq = q.shape[0]
    perm = torch.randperm(nq, generator=torch.Generator().manual_seed(1))
    assert same(
        _run(m, mesh, drv, q[:40], sdf[:40], sn[:40]),
        _rows(base, slice(0, 40)),
        atol=1e-12,
    )
    assert same(
        _run(m, mesh, drv, q[7:8], sdf[7:8], sn[7:8]),
        _rows(base, slice(7, 8)),
        atol=1e-12,
    )
    assert same(
        _run(m, mesh, drv, q[perm], sdf[perm], sn[perm]), _rows(base, perm), atol=1e-12
    )
    chunks = [
        outputs(_run(m, mesh, drv, *c))
        for c in zip(q.split(25), sdf.split(25), sn.split(25))
    ]
    assert same(
        {k: torch.cat([c[k] for c in chunks]) for k in chunks[0]}, base, atol=1e-12
    )
    monkeypatch.setattr(kernel_read, "GLOBAL_READ_TILE_PAIRS", 7 * mesh.n_cells)
    assert same(_run(m, mesh, drv, q, sdf, sn), base, atol=1e-12)


@pytest.mark.parametrize("improper", [False, True], ids=["rotation", "reflection"])
@pytest.mark.parametrize("dim", [3, 2])
def test_global_read_covariance(improper, dim):
    """Rotations, reflections and translations: the outputs follow exactly."""
    m = _model(dim=dim).double().eval()
    mesh, drv, q, sdf, sn = _case(dim=dim)
    base = _run(m, mesh, drv, q, sdf, sn)
    R = orthogonal(dim, 3, improper=improper)
    t = torch.arange(1.0, dim + 1, dtype=D) * 2.5
    posed = _run(m, transform(mesh, R, t), drv @ R.T, q @ R.T + t, sdf, sn @ R.T)
    assert torch.allclose(posed.point_data["p"], base.point_data["p"], atol=1e-10)
    assert torch.allclose(posed.point_data["v"], base.point_data["v"] @ R.T, atol=1e-10)


def test_global_read_similarity_refinement_and_measure_scale():
    """x -> k x + shift with areas x k^2 (and the sdf, a length, x k); every cell
    split into two coincident half-measure copies (every copy is read); every
    measure x 137: nothing changes. Lengths enter in units of L or of the point's
    wall distance, and the weights are measure fractions."""
    m = _model().double().eval()
    mesh, drv, q, sdf, sn = _case()
    base = _run(m, mesh, drv, q, sdf, sn)
    k, t = 2.7, torch.tensor([3.0, -7.0, 11.0], dtype=D)
    assert same(
        _run(m, transform(mesh, shift=t, scale=k), drv, k * q + t, k * sdf, sn),
        base,
        atol=1e-10,
        rtol=0,
    )
    pairs = m.kernel_read_pairs
    assert same(_run(m, refine(mesh), drv, q, sdf, sn), base, atol=1e-10, rtol=0)
    assert m.kernel_read_pairs == 2 * pairs
    w = mesh.cell_data[EFFECTIVE_MEASURE_KEY]
    assert same(
        _run(
            m,
            mesh.with_data(cell_data={EFFECTIVE_MEASURE_KEY: w * 137.0}),
            drv,
            q,
            sdf,
            sn,
        ),
        base,
        atol=1e-10,
        rtol=0,
    )


def test_global_read_is_live_everywhere_and_starts_as_the_plain_model():
    """With the zero-initialized output the model equals the plain read-only model;
    with trained-like weights the read changes every point, the far ones included
    (there is no radius)."""
    mesh, drv, q, sdf, sn = _case()
    torch.manual_seed(0)
    plain = _plain().double().eval()
    for randomize in (False, True):
        m = _model(randomize=randomize).double().eval()
        plain.load_state_dict(
            {
                k: v
                for k, v in m.state_dict().items()
                if not k.startswith(("kernel_reads", "kernel_basis"))
            }
        )
        a, b = _run(m, mesh, drv, q, sdf, sn), _run(plain, mesh, drv, q, sdf, sn)
        if not randomize:
            assert same(a, outputs(b), atol=1e-12, rtol=0)
        else:
            far = slice(q.shape[0] - 10, None)
            diff = (a.point_data["p"] - b.point_data["p"]).abs()
            assert bool((diff[far] > 1e-8).all()), diff[far]
            assert bool((diff > 1e-8).all())


def test_global_pair_invariants_are_scale_free_and_the_near_wall_invariant():
    """The invariants: r / L, log(r / L) and log(r / |d_q|), which is zero at the
    point's nearest cell and positive elsewhere; a point on a cell centroid
    (d_q = 0) stays finite (the numerical floor)."""
    g = torch.Generator().manual_seed(0)
    y = torch.randn(50, 3, generator=g, dtype=D)
    n = torch.nn.functional.normalize(torch.randn(50, 3, generator=g, dtype=D), dim=-1)
    x = torch.cat([y[:1], torch.randn(20, 3, generator=g, dtype=D)])
    d = torch.cdist(x, y).min(-1).values
    nq = torch.nn.functional.normalize(x - y[torch.cdist(x, y).argmin(-1)], dim=-1)
    gu = torch.nn.functional.normalize(torch.randn(1, 3, generator=g, dtype=D), dim=-1)
    inv = global_pair_invariants(x, nq, d, None, y, n, None, gu, 1e-12)
    assert inv.shape == (21, 50, 6 + 2) and torch.isfinite(inv).all()
    r = torch.cdist(x, y)
    assert torch.allclose(inv[..., 0], r) and torch.allclose(inv[1:, :, 1], r[1:].log())
    near = inv[1:, :, 2]
    assert near.min() > -1e-12 and torch.allclose(
        near.min(-1).values, torch.zeros(20, dtype=D), atol=1e-12
    )
    k = 3.1
    inv_k = global_pair_invariants(k * x, nq, k * d, None, k * y, n, None, gu, 1e-12)
    assert torch.allclose(inv_k[1:, :, 2:], inv[1:, :, 2:], atol=1e-12)


def test_global_softmax_read_matches_the_dense_reference(monkeypatch):
    """The tiled read (forward and every gradient) equals the dense softmax
    reference, whatever the tiling."""
    g = torch.Generator().manual_seed(0)
    nh, nq, n, c, cv = 3, 11, 7, 2, 4

    def mk(*s):
        return torch.randn(*s, generator=g, dtype=D).requires_grad_()

    args = (mk(16, nq * n), mk(nh, 16), mk(nh, nq, c), mk(nh, n, c), mk(nh, n, cv))
    args += (mk(n), 0.7)
    for tile in (5 * n, 1 << 24):
        monkeypatch.setattr(kernel_read, "GLOBAL_READ_TILE_PAIRS", tile)
        assert torch.autograd.gradcheck(_GlobalSoftmaxRead.apply, args)
        a = _GlobalSoftmaxRead.apply(*args)
        b = global_softmax_read_reference(*args)
        assert torch.allclose(a, b, atol=1e-13)
        go = torch.randn(a.shape, generator=g, dtype=D)
        ga = torch.autograd.grad(a, args[:6], go)
        gb = torch.autograd.grad(b, args[:6], go)
        assert all(torch.allclose(u, v, atol=1e-12) for u, v in zip(ga, gb))


def test_global_read_tiled_basis_gradients_match_one_tile(monkeypatch):
    """The checkpointed, tiled kernel basis gives the gradients of one dense tile."""
    mesh, drv, q, sdf, sn = _case(nq=60)
    grads = []
    for tile in (1 << 24, 9 * mesh.n_cells):
        monkeypatch.setattr(kernel_read, "GLOBAL_READ_TILE_PAIRS", tile)
        m = _model().double()
        out = m(
            boundary_meshes={"wall": mesh},
            prediction_points=points_mesh(q, {"sdf": sdf, "sdf_normals": sn}),
            global_data={"g": drv},
        )
        sum(o.square().sum() for o in outputs(out).values()).backward()
        grads.append([p.grad.clone() for p in m.parameters()])
    assert all(torch.allclose(a, b, atol=1e-12, rtol=1e-10) for a, b in zip(*grads))


@pytest.mark.parametrize("blocks", [None, 1])
def test_global_read_every_parameter_receives_a_gradient(blocks):
    """DDP: every parameter takes part in the loss, with the read at every block or
    only the last one, from the zero-initialized start included."""
    m = _model(randomize=False, kernel_read_blocks=blocks)
    assert len(m.kernel_reads) == (2 if blocks is None else 1)
    mesh, drv, q, sdf, sn = _case()
    f = lambda t: t.float()  # noqa: E731
    out = m(
        boundary_meshes={
            "wall": soup(
                f(mesh.cell_centroids),
                f(mesh.cell_normals),
                f(mesh.cell_data[EFFECTIVE_MEASURE_KEY]),
            )
        },
        prediction_points=points_mesh(f(q), {"sdf": f(sdf), "sdf_normals": f(sn)}),
        global_data={"g": f(drv)},
    )
    sum(o.square().mean() for o in outputs(out).values()).backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, missing


def test_global_read_bf16_autocast_is_finite():
    m = _model()
    mesh, drv, q, sdf, sn = _case()
    q = torch.cat([q, mesh.cell_centroids[:2]])  # points on a cell centroid: d_q = 0
    sdf, sn = torch.cat([sdf, sdf[:2]]), torch.cat([sn, sn[:2]])
    f = lambda t: t.float()  # noqa: E731
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(
            boundary_meshes={
                "wall": soup(
                    f(mesh.cell_centroids),
                    f(mesh.cell_normals),
                    f(mesh.cell_data[EFFECTIVE_MEASURE_KEY]),
                )
            },
            prediction_points=points_mesh(f(q), {"sdf": f(sdf), "sdf_normals": f(sn)}),
            global_data={"g": f(drv)},
        )
    sum(o.float().sum() for o in outputs(out).values()).backward()
    assert all(torch.isfinite(o.float()).all() for o in outputs(out).values())
    assert all(
        torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None
    )


def test_global_read_boundary_vector_inputs_enter_the_pair_invariants():
    """A rank-1 boundary input joins the pair invariants; covariance still holds."""
    m = _model(boundary_schema={"tr": {"rank": 1}}).double().eval()
    assert m.kernel_basis.net[0].in_features == 6 + 2 + 2 + 2
    mesh, drv, q, sdf, sn = _case()
    tr = torch.randn(
        mesh.n_cells, 3, generator=torch.Generator().manual_seed(5), dtype=D
    )
    w = mesh.cell_data[EFFECTIVE_MEASURE_KEY]
    msh = mesh.with_data(cell_data={EFFECTIVE_MEASURE_KEY: w, "tr": tr})
    base = _run(m, msh, drv, q, sdf, sn)
    R, t = orthogonal(3, 7, improper=True), torch.tensor([1.0, 2.0, -3.0], dtype=D)
    posed = _run(
        m,
        transform(msh, R, t, specs=m.boundary_schema),
        drv @ R.T,
        q @ R.T + t,
        sdf,
        sn @ R.T,
    )
    assert torch.allclose(posed.point_data["v"], base.point_data["v"] @ R.T, atol=1e-10)


def test_global_read_construction():
    """The global read is the ``"mean"`` form only; without a read a model owns no
    read parameter."""
    with pytest.raises(ValueError, match="global read"):
        _plain(kernel_read="sum", kernel_read_radius=None)
    m = _model(randomize=False)
    assert m.kernel_read_radius is None and len(m.kernel_reads) == 2


@cuda_only
def test_global_read_cuda_matches_cpu_float64():
    """The same tiled read on CUDA and CPU, float64."""
    m = _model(dim=2).double().eval()
    mesh, drv, q, sdf, sn = _case(dim=2)
    base = _run(m, mesh, drv, q, sdf, sn)
    dev = "cuda"
    got = _run(m.to(dev), mesh.to(dev), drv.to(dev), q.to(dev), sdf.to(dev), sn.to(dev))
    assert same({k: v.cpu() for k, v in outputs(got).items()}, base, atol=1e-10)


@cuda_only
def test_global_read_cuda_fused_and_compiled_training_path():
    """The protocol's training path on CUDA: fused geometry kernel, compiled blocks,
    bf16 autocast. One training step is finite with every parameter's gradient
    present, the compiled forward matches the eager one to bf16 roundoff, and the
    prediction stays query-independent (float32 roundoff)."""
    dev = "cuda"
    m_e = _model(geo_kernel="fused").to(dev)
    m_c = _model(geo_kernel="fused", compile_blocks=True).to(dev)
    m_c.load_state_dict(m_e.state_dict())
    mesh, drv, q, sdf, sn = _case(n=600, nq=400)
    f = lambda t: t.float().to(dev)  # noqa: E731
    bm = {
        "wall": soup(
            mesh.cell_centroids.float(),
            mesh.cell_normals.float(),
            mesh.cell_data[EFFECTIVE_MEASURE_KEY].float(),
        ).to(dev)
    }
    pp = points_mesh(f(q), {"sdf": f(sdf), "sdf_normals": f(sn)})
    outs = []
    for m in (m_e, m_c):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = outputs(
                m(boundary_meshes=bm, prediction_points=pp, global_data={"g": f(drv)})
            )
        sum(o.float().square().mean() for o in out.values()).backward()
        assert all(torch.isfinite(o.float()).all() for o in out.values())
        missing = [n for n, p in m.named_parameters() if p.grad is None]
        assert not missing, missing
        outs.append(out)
    for k in outs[0]:
        a, b = outs[0][k].float(), outs[1][k].float()
        assert (a - b).norm() <= 3e-2 * a.norm() + 1e-6, k
    with torch.no_grad():
        full = outputs(
            m_e(boundary_meshes=bm, prediction_points=pp, global_data={"g": f(drv)})
        )
        part = outputs(
            m_e(
                boundary_meshes=bm,
                prediction_points=points_mesh(
                    f(q[:50]), {"sdf": f(sdf[:50]), "sdf_normals": f(sn[:50])}
                ),
                global_data={"g": f(drv)},
            )
        )
    assert all(
        torch.allclose(part[k], full[k][:50], atol=1e-4, rtol=1e-4) for k in part
    )


@cuda_only
@pytest.mark.parametrize("shape", [(37, 53, 2, 1, 1, 0), (100, 301, 3, 1, 1, 1)])
def test_fused_kernel_basis_matches_the_eager_basis(shape):
    """The fused Triton basis (invariants and MLP per tile, never stored) equals the
    eager float64 basis in float32, forward and every parameter gradient; in bf16
    autocast to bf16 roundoff."""
    from physicsnemo.experimental.models.isla import global_read_kernels as grk
    from physicsnemo.experimental.models.isla.kernel_read import (
        _KernelBasis,
        global_geo_width,
    )

    nq, n, d, k, pq, pb = shape
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)

    def mk(*s):
        return torch.randn(*s, generator=g, device=dev)

    def unit(*s):
        return torch.nn.functional.normalize(mk(*s), dim=-1)

    geo = {
        "query_r": mk(nq, d),
        "query_n": unit(nq, d),
        "query_wall": mk(nq).abs() + 0.01,
        "query_vecs": unit(nq, pq, d) if pq else None,
        "boundary_r": mk(n, d),
        "boundary_n": unit(n, d),
        "boundary_vecs": unit(n, pb, d) if pb else None,
        "g_unit": unit(k, d),
        "eps": 1e-12,
    }
    torch.manual_seed(0)
    basis = _KernelBasis(global_geo_width(k, pq, pb)).to(dev)
    ref = _KernelBasis(global_geo_width(k, pq, pb)).to(dev).double()
    ref.load_state_dict(basis.state_dict())
    assert grk.fused_basis_supported(basis, geo)
    geo64 = {
        key: v.double() if isinstance(v, torch.Tensor) else v for key, v in geo.items()
    }
    phi_r = ref(global_pair_invariants(**geo64)).reshape(-1, 16).t()
    go = torch.randn(phi_r.shape, generator=g, device=dev, dtype=D)
    g_r = torch.autograd.grad(phi_r, list(ref.parameters()), go)
    for lowp, tol in ((False, 1e-5), (True, 2e-2)):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=lowp):
            phi = grk.fused_kernel_basis(basis, geo)
        assert phi.dtype == (torch.bfloat16 if lowp else torch.float32)
        g_f = torch.autograd.grad(phi, list(basis.parameters()), go.to(phi.dtype))
        scale = phi_r.abs().max()
        assert (phi.double() - phi_r).abs().max() <= tol * scale
        for a, b in zip(g_f, g_r):
            assert (a.double() - b).abs().max() <= tol * b.abs().max()


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_global_softmax_read_matches_the_dense_reference(dtype):
    """The fused Triton read (online softmax over cell tiles; backward by recompute,
    no atomics) equals the dense float64 reference, forward and every gradient,
    with ragged tiles (Q, N not multiples of the tile)."""
    from physicsnemo.experimental.models.isla.global_read_kernels import (
        global_softmax_read,
    )

    g = torch.Generator(device="cuda").manual_seed(0)
    tol = 1e-5 if dtype == torch.float32 else 3e-2
    for nh, nq, n, c, cv in ((8, 203, 77, 4, 8), (3, 64, 130, 2, 4)):

        def mk(*s, dt=dtype):
            return torch.randn(*s, generator=g, device="cuda", dtype=dt)

        args = [
            mk(16, nq * n),
            mk(nh, 16, dt=torch.float32) * 0.5,
            mk(nh, nq, c),
            mk(nh, n, c),
            mk(nh, n, cv),
            mk(n, dt=torch.float32),
        ]
        args = [a.requires_grad_() for a in args]
        ref = [a.detach().double().requires_grad_() for a in args]
        out = global_softmax_read(*args, 0.7)
        want = global_softmax_read_reference(*ref, 0.7)
        assert (out.double() - want).abs().max() <= tol * want.abs().max()
        go = torch.randn(out.shape, generator=g, device="cuda")
        g_f = torch.autograd.grad(out, args, go)
        g_r = torch.autograd.grad(want, ref, go.double())
        for a, b in zip(g_f, g_r):
            assert (a.double() - b).abs().max() <= tol * b.abs().max()


@cuda_only
def test_global_read_fused_model_matches_the_eager_model(monkeypatch):
    """The whole model on CUDA in float32: the fused basis and read against the
    eager tiles, outputs and every parameter gradient."""
    from physicsnemo.experimental.models.isla import global_read_kernels as grk

    mesh, drv, q, sdf, sn = _case(dim=2, n=200, nq=150)
    dev = "cuda"
    f = lambda t: t.float().to(dev)  # noqa: E731
    bm = {
        "wall": soup(
            mesh.cell_centroids.float(),
            mesh.cell_normals.float(),
            mesh.cell_data[EFFECTIVE_MEASURE_KEY].float(),
        ).to(dev)
    }
    pp = points_mesh(f(q), {"sdf": f(sdf), "sdf_normals": f(sn)})
    res = []
    for fused in (True, False):
        monkeypatch.setattr(grk, "HAS_TRITON", fused and grk.HAS_TRITON)
        m = _model(dim=2).to(dev)
        out = outputs(
            m(boundary_meshes=bm, prediction_points=pp, global_data={"g": f(drv)})
        )
        sum(o.square().sum() for o in out.values()).backward()
        res.append((out, [p.grad.clone() for p in m.parameters()]))
    (oa, ga), (ob, gb) = res
    for k in oa:
        assert (oa[k] - ob[k]).norm() <= 1e-4 * ob[k].norm(), k
    ### Every gradient, against the scale of the whole gradient (some parameters'
    ### gradients cancel to float32 noise).
    ga, gb = torch.cat([a.flatten() for a in ga]), torch.cat([b.flatten() for b in gb])
    assert (ga - gb).norm() <= 1e-4 * gb.norm()
