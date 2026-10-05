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

"""Contract tests for ISLA's boundary kernel read (``kernel_read``, kernel_read.py).

Passive prediction points (``query_independent=True``) also read the individual
boundary cells within a radius in units of L. Every contract of the read-only model
must survive: exact query independence, rotation / reflection / translation
covariance, the similarity gauge, boundary-refinement and measure-scale invariance,
a gradient for every parameter, finite bf16 autocast, and the CUDA fused-kernel and
compiled-block training path. The kernel read's output projection starts at zero (the
model starts as the plain read-only model), so the tests give every kernel-read
parameter a random value first.
"""

import pytest
import torch

from physicsnemo.experimental.models import ISLA
from physicsnemo.experimental.models.isla.geometry import rms_pairwise_distance
from physicsnemo.experimental.models.isla.kernel_read import (
    _exact_radius_pairs,
    boundary_pairs,
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
MODES = ["mean", "sum"]
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


def _model(mode, dim=3, radius=0.3, **kw):
    torch.manual_seed(0)
    kw = {"hidden": 32, "n_layers": 2, "n_slices": 8, **kw}
    m = ISLA(
        output_schema=PV,
        global_schema=G,
        query_schema=QS,
        interior=True,
        query_independent=True,
        kernel_read=mode,
        kernel_read_radius=radius,
        dim=dim,
        **kw,
    )
    with torch.no_grad():
        for p in [*m.kernel_reads.parameters(), *m.kernel_basis.parameters()]:
            p.normal_(0.0, 0.3)
    return m


def _case(dim=3, n=300, nq=120, seed=0):
    """A boundary cloud and prediction points near it (most within the radius of some
    cell) plus a few far away, with a per-point sdf and a unit vector field."""
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


@pytest.mark.parametrize("mode", MODES)
def test_kernel_read_is_exactly_query_independent(mode):
    """A prediction depends on its own point and the boundary only: alone, a prefix,
    any order and any chunking give the same values to float64 roundoff."""
    m = _model(mode).double().eval()
    mesh, drv, q, sdf, sn = _case()
    base = _run(m, mesh, drv, q, sdf, sn)
    assert m.kernel_read_pairs > 0
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


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("improper", [False, True], ids=["rotation", "reflection"])
@pytest.mark.parametrize("dim", [3, 2])
def test_kernel_read_covariance(mode, improper, dim):
    """Rotations, reflections and translations: the outputs follow exactly (the
    prediction points, the boundary, the global direction and the vector input
    move together)."""
    m = _model(mode, dim=dim).double().eval()
    mesh, drv, q, sdf, sn = _case(dim=dim)
    base = _run(m, mesh, drv, q, sdf, sn)
    R = orthogonal(dim, 3, improper=improper)
    t = torch.arange(1.0, dim + 1, dtype=D) * 2.5
    posed = _run(m, transform(mesh, R, t), drv @ R.T, q @ R.T + t, sdf, sn @ R.T)
    assert torch.allclose(posed.point_data["p"], base.point_data["p"], atol=1e-10)
    assert torch.allclose(posed.point_data["v"], base.point_data["v"] @ R.T, atol=1e-10)


@pytest.mark.parametrize("mode", MODES)
def test_kernel_read_similarity_refinement_and_measure_scale(mode):
    """x -> k x + shift with areas x k^2 (and the sdf, a length, x k); every cell
    split into two coincident half-measure copies; every measure x 137: nothing
    changes. The radius is in units of L and the weights are measure fractions."""
    m = _model(mode).double().eval()
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
    assert m.kernel_read_pairs == 2 * pairs  # every copy is read, none is dropped
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


@pytest.mark.parametrize("mode", MODES)
def test_kernel_read_is_live_and_local(mode):
    """The read changes the prediction near the boundary and nothing beyond the
    radius: a point with no cell within R gets exactly the plain read-only model's
    value (same weights), and a boundary cell moved far from every point changes no
    point it no longer reaches through the read (only through the slices)."""
    m = _model(mode).double().eval()
    plain = (
        ISLA(
            output_schema=PV,
            global_schema=G,
            query_schema=QS,
            interior=True,
            query_independent=True,
            hidden=32,
            n_layers=2,
            n_slices=8,
        )
        .double()
        .eval()
    )
    plain.load_state_dict(
        {
            k: v
            for k, v in m.state_dict().items()
            if not k.startswith(("kernel_reads", "kernel_basis"))
        }
    )
    mesh, drv, q, sdf, sn = _case()
    a, b = _run(m, mesh, drv, q, sdf, sn), _run(plain, mesh, drv, q, sdf, sn)
    far = slice(q.shape[0] - 10, None)
    assert same(_rows(a, far), _rows(b, far), atol=1e-12, rtol=0)
    near = slice(0, q.shape[0] - 10)
    assert not same(_rows(a, near), _rows(b, near), atol=1e-6)


def test_kernel_read_falloff_vanishes_smoothly_at_the_radius():
    """A point just inside the radius of its only cell gets the plain read-only
    model's value to within the falloff there, (1 - s^2)^3 ~ 1e-11 at s = 1 - 1e-4:
    the read switches on continuously, with no jump at R."""
    m = _model("mean", radius=0.05).double().eval()
    plain = (
        ISLA(
            output_schema=PV,
            global_schema=G,
            query_schema=QS,
            interior=True,
            query_independent=True,
            hidden=32,
            n_layers=2,
            n_slices=8,
        )
        .double()
        .eval()
    )
    plain.load_state_dict(
        {
            k: v
            for k, v in m.state_dict().items()
            if not k.startswith(("kernel_reads", "kernel_basis"))
        }
    )
    pts, nrm, w, drv = cloud(50, seed=4)
    mesh = soup(pts * 30.0, nrm, w)  # cells far apart against the radius
    L = float(
        rms_pairwise_distance(
            mesh.cell_centroids[None], mesh.cell_data[EFFECTIVE_MEASURE_KEY][None]
        ).reshape(())
    )
    y, n0 = mesh.cell_centroids[0], mesh.cell_normals[0]
    gap = []
    for s in (0.05 * (1 - 1e-4), 0.025):
        q, z = (y + s * L * n0)[None], torch.zeros(1, dtype=D)
        a, b = (
            outputs(_run(m, mesh, drv, q, z, n0[None])),
            outputs(_run(plain, mesh, drv, q, z, n0[None])),
        )
        assert m.kernel_read_pairs == 1
        gap.append(max(float((a[k] - b[k]).abs().max()) for k in a))
    assert gap[0] < 1e-8 < 1e-3 < gap[1], gap


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("blocks", [None, 1])
def test_kernel_read_every_parameter_receives_a_gradient(mode, blocks):
    """DDP: every parameter takes part in the loss, with the kernel read at every
    block or only the last one, from the zero-initialized start included."""
    torch.manual_seed(0)
    m = ISLA(
        output_schema=PV,
        global_schema=G,
        query_schema=QS,
        interior=True,
        query_independent=True,
        kernel_read=mode,
        kernel_read_radius=0.3,
        kernel_read_blocks=blocks,
        hidden=32,
        n_layers=2,
        n_slices=8,
    )
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


@pytest.mark.parametrize("mode", MODES)
def test_kernel_read_bf16_autocast_is_finite(mode):
    m = _model(mode)
    mesh, drv, q, sdf, sn = _case()
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


def test_kernel_read_boundary_vector_inputs_enter_the_pair_invariants():
    """A rank-1 boundary input joins the pair invariants; covariance still holds."""
    torch.manual_seed(0)
    m = (
        ISLA(
            output_schema=PV,
            global_schema=G,
            query_schema=QS,
            boundary_schema={"tr": {"rank": 1}},
            interior=True,
            query_independent=True,
            kernel_read="sum",
            kernel_read_radius=0.3,
            hidden=32,
            n_layers=2,
            n_slices=8,
        )
        .double()
        .eval()
    )
    with torch.no_grad():
        for p in [*m.kernel_reads.parameters(), *m.kernel_basis.parameters()]:
            p.normal_(0.0, 0.3)
    assert m.kernel_basis.net[0].in_features == 5 + 2 + 2 + 2
    mesh, drv, q, sdf, sn = _case()
    tr = torch.randn(
        mesh.n_cells, 3, generator=torch.Generator().manual_seed(5), dtype=D
    )
    msh = mesh.with_data(
        cell_data={
            EFFECTIVE_MEASURE_KEY: mesh.cell_data[EFFECTIVE_MEASURE_KEY],
            "tr": tr,
        }
    )
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
    assert not same(
        _run(
            m,
            msh.with_data(
                cell_data={
                    EFFECTIVE_MEASURE_KEY: mesh.cell_data[EFFECTIVE_MEASURE_KEY],
                    "tr": -tr,
                }
            ),
            drv,
            q,
            sdf,
            sn,
        ),
        base,
        atol=1e-8,
    )


def test_kernel_read_construction_errors_and_default_absence():
    """The read exists only for passive prediction points; without it a model owns
    no kernel-read parameter."""
    kw = {
        "output_schema": PV,
        "global_schema": G,
        "hidden": 32,
        "n_layers": 2,
        "n_slices": 8,
    }
    with pytest.raises(ValueError, match="query_independent=True"):
        ISLA(interior=True, kernel_read="mean", **kw)
    with pytest.raises(ValueError, match="'mean' or 'sum'"):
        ISLA(interior=True, query_independent=True, kernel_read="max", **kw)
    with pytest.raises(ValueError, match="kernel_read_blocks"):
        ISLA(
            interior=True,
            query_independent=True,
            kernel_read="sum",
            kernel_read_blocks=3,
            **kw,
        )
    with pytest.raises(ValueError, match="kernel_read_radius"):
        ISLA(
            interior=True,
            query_independent=True,
            kernel_read="sum",
            kernel_read_radius=0.0,
            **kw,
        )
    plain = ISLA(interior=True, query_independent=True, **kw)
    assert plain.kernel_reads is None and not any(
        "kernel_read" in n for n, _ in plain.named_parameters()
    )


def test_exact_radius_pairs_match_brute_force():
    g = torch.Generator().manual_seed(0)
    q, y = (
        torch.randn(300, 3, generator=g, dtype=D),
        torch.randn(200, 3, generator=g, dtype=D),
    )
    qi, bj = _exact_radius_pairs(q, y, 0.5)
    want = ((q[:, None] - y[None]).norm(dim=-1) < 0.5).nonzero(as_tuple=True)
    assert torch.equal(qi, want[0]) and torch.equal(bj, want[1])


@cuda_only
@pytest.mark.parametrize("dim", [3, 2])
def test_hash_grid_radius_search_matches_the_exact_set(dim):
    """On CUDA in float32 the library's hash-grid search returns the exact radius set
    (planar points padded with a zero coordinate), up to pairs at the rim."""
    g = torch.Generator().manual_seed(0)
    q = torch.randn(1, 3000, dim, generator=g).cuda()
    y = torch.randn(1, 2000, dim, generator=g).cuda()
    qi, bj = boundary_pairs(q, y, 0.2)
    eq, ej = boundary_pairs(q.double(), y.double(), 0.2)
    got = set(zip(qi.tolist(), bj.tolist()))
    want = set(zip(eq.tolist(), ej.tolist()))
    d = (q[0][:, None].double() - y[0][None].double()).norm(dim=-1)
    rim = {(i, j) for i, j in got ^ want if abs(float(d[i, j]) - 0.2) > 1e-6}
    assert not rim and len(want) > 1000


@cuda_only
@pytest.mark.parametrize("mode", MODES)
def test_kernel_read_cuda_fused_and_compiled_training_path(mode):
    """The protocol's training path on CUDA: fused geometry kernel, compiled blocks,
    bf16 autocast, the hash-grid search. One training step is finite with every
    parameter's gradient present, the compiled forward matches the eager one to
    bf16 roundoff, and the prediction stays query-independent (float32 roundoff)."""
    dev = "cuda"
    m_e = _model(mode, geo_kernel="fused").to(dev)
    m_c = _model(mode, geo_kernel="fused", compile_blocks=True).to(dev)
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
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_pair_kernels_match_the_eager_reference(dtype):
    """The Triton segment sum and pair dot (forward and both gradients) equal the
    eager gather / index_add reference; points without pairs read zero and a cell
    read by thousands of pairs (a thousand-cell boundary) is summed without atomics."""
    from physicsnemo.experimental.models.isla.pair_kernels import (
        PairPlan,
        _eager_scores,
        _eager_weighted_sum,
        pair_scores,
        pair_weighted_sum,
    )

    g = torch.Generator().manual_seed(0)
    nq, nb = 500, 40
    counts = torch.randint(0, 120, (nq,), generator=g)
    counts[::7] = 0
    qi = torch.repeat_interleave(torch.arange(nq), counts).cuda()
    bj = torch.randint(0, nb, (qi.shape[0],), generator=g).cuda()
    plan = PairPlan.build(qi, bj, nq, nb)
    p = qi.shape[0]
    mk = lambda *s: torch.randn(*s, generator=g, dtype=dtype).cuda().requires_grad_()  # noqa: E731
    w, x, a, b = mk(p, 8), mk(nb, 8, 8), mk(nq, 8, 4), mk(nb, 8, 4)
    tol = (
        {"atol": 1e-10, "rtol": 1e-10}
        if dtype == torch.float64
        else {"atol": 1e-4, "rtol": 1e-4}
    )
    for fast, ref, args in (
        (pair_weighted_sum, _eager_weighted_sum, (w, x, plan)),
        (pair_scores, _eager_scores, (a, b, plan, dtype)),
    ):
        out_f = fast(*args)
        out_r = ref(*args)
        assert torch.allclose(out_f, out_r, **tol)
        go = torch.randn(out_r.shape, generator=g, dtype=dtype).cuda()
        ins = [t for t in args if isinstance(t, torch.Tensor)]
        gf = torch.autograd.grad(out_f, ins, go)
        gr = torch.autograd.grad(out_r, ins, go)
        assert all(torch.allclose(u, v, **tol) for u, v in zip(gf, gr))
    assert pair_weighted_sum(w, x, plan)[::7].abs().max() == 0


@pytest.mark.parametrize("mode", MODES)
def test_kernel_read_checkpoint_changes_nothing(mode):
    """Rebuilding the pair tensors in backward (checkpoint) or storing them gives the
    same outputs and gradients."""
    mesh, drv, q, sdf, sn = _case()
    grads = []
    for ck in (False, True):
        m = _model(mode).double()
        for r in m.kernel_reads:
            r.checkpoint = ck
        out = m(
            boundary_meshes={"wall": mesh},
            prediction_points=points_mesh(q, {"sdf": sdf, "sdf_normals": sn}),
            global_data={"g": drv},
        )
        sum(o.square().sum() for o in outputs(out).values()).backward()
        grads.append([p.grad.clone() for p in m.parameters()])
    assert all(torch.allclose(a, b, atol=1e-12, rtol=1e-10) for a, b in zip(*grads))
