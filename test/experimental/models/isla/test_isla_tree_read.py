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

"""Contract tests for ISLA's tree-accelerated global read (``kernel_read_theta``).

The tree read is the dense global read evaluated through a forest of Barnes-Hut
cluster trees over the boundary cells (GLOBE's ``ClusterTree``, one tree per normal
octant), walked once per prediction point. At ``theta = 0`` it is the dense read pair
for pair, so every contract of the dense read must hold exactly there: equality with
the dense form, Euclidean covariance, the similarity gauge, refinement and
measure-scale invariance, a gradient for every parameter, CUDA against CPU. Query
independence holds exactly at every ``theta`` (a point's walk depends on the forest
and the point alone). At a finite ``theta`` the read is an approximation whose error
falls with ``theta``; the tests measure that on a smooth boundary and check that the
pair list still covers the whole boundary measure exactly once per point and admits
only well-separated, orientation-coherent clusters.
"""

import math

import pytest
import torch

import physicsnemo.experimental.models.isla.tree_read as tree_read
from physicsnemo.experimental.models import ISLA
from physicsnemo.experimental.models.isla.kernel_read import _KernelBasis, global_geo_width
from physicsnemo.experimental.models.isla.tree_read import (
    BoundaryForest,
    _TreeKernelRead,
    normal_octants,
    tree_read_pairs,
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


def _model(dim=3, theta=None, randomize=True, seed=0, **kw):
    torch.manual_seed(seed)
    kw = {"hidden": 32, "n_layers": 2, "n_slices": 8, **kw}
    m = ISLA(
        output_schema=PV,
        global_schema=G,
        query_schema=QS,
        interior=True,
        query_independent=True,
        kernel_read="mean",
        kernel_read_radius=None,
        kernel_read_theta=theta,
        dim=dim,
        **kw,
    )
    if randomize:
        with torch.no_grad():
            for p in [*m.kernel_reads.parameters(), *m.kernel_basis.parameters()]:
                p.normal_(0.0, 0.3)
    return m


def _pair(dim=3, theta=0.0, **kw):
    """A dense model and a tree model with identical weights."""
    dense = _model(dim, None, **kw)
    tree = _model(dim, theta, randomize=False, **kw)
    tree.load_state_dict(dense.state_dict())
    return dense.double().eval(), tree.double().eval()


def _case(dim=3, n=300, nq=120, seed=0):
    pts, nrm, w, drv = cloud(n, d=dim, seed=seed)
    g = torch.Generator().manual_seed(seed + 1)
    near = pts[: nq - 10] + 0.15 * torch.randn(nq - 10, dim, generator=g, dtype=D)
    far = 40.0 + torch.randn(10, dim, generator=g, dtype=D)
    q = torch.cat([near, far])
    sdf = torch.randn(nq, generator=g, dtype=D) * 0.3
    sn = torch.nn.functional.normalize(torch.randn(nq, dim, generator=g, dtype=D), dim=-1)
    return soup(pts, nrm, w), drv, q, sdf, sn


def _sphere_case(n=4000, nq=200, seed=0):
    """A smooth boundary: a sphere of radius 2 with outward normals and areas, and
    points at radii 2.05 to 6 plus a few far away."""
    g = torch.Generator().manual_seed(seed)
    dirs = torch.nn.functional.normalize(torch.randn(n, 3, dtype=D, generator=g), dim=-1)
    pts, nrm = 2.0 * dirs, dirs
    w = torch.full((n,), 4 * math.pi * 4.0 / n, dtype=D)
    drv = torch.tensor([0.3, 1.0, 0.2], dtype=D)
    q_dirs = torch.nn.functional.normalize(torch.randn(nq, 3, dtype=D, generator=g), dim=-1)
    radii = torch.cat([2.05 + 3.95 * torch.rand(nq - 10, dtype=D, generator=g), torch.full((10,), 40.0, dtype=D)])
    q = q_dirs * radii[:, None]
    sdf = radii - 2.0
    return soup(pts, nrm, w), drv, q, sdf, q_dirs


def _run(m, mesh, drv, q, sdf, sn, surface=None):
    """``surface``: the boundary_geometry, when the cells' measures exceed their own."""
    with torch.no_grad():
        return m(
            boundary_meshes={"wall": mesh},
            prediction_points=points_mesh(q, {"sdf": sdf, "sdf_normals": sn}),
            boundary_geometry=None if surface is None else {"wall": surface},
            global_data={"g": drv},
        )


def _rows(out, idx):
    """Rows ``idx`` of an outputs mapping."""
    return {k: v[idx] for k, v in out.items()}


def _rel(a, b):
    return max(
        float((a[k] - b[k]).norm() / b[k].norm().clamp_min(1e-30)) for k in b
    )


@pytest.mark.parametrize("dim", [3, 2])
def test_tree_read_at_theta_zero_is_the_dense_read(dim):
    dense, tree = _pair(dim, 0.0)
    mesh, drv, q, sdf, sn = _case(dim)
    a = outputs(_run(dense, mesh, drv, q, sdf, sn))
    b = outputs(_run(tree, mesh, drv, q, sdf, sn))
    assert _rel(b, a) < 1e-10
    assert tree.kernel_read_pairs == q.shape[0] * mesh.n_cells
    assert isinstance(tree.kernel_reads[0], _TreeKernelRead)


def test_tree_read_dense_checkpoint_loads_into_the_tree_form(tmp_path):
    """The two forms own the same parameters, so a dense checkpoint loads strictly."""
    dense = _model(3, None)
    tree = _model(3, 0.5, randomize=False)
    missing = tree.load_state_dict(dense.state_dict(), strict=True)
    assert not missing.missing_keys and not missing.unexpected_keys
    assert tree.kernel_read_theta == 0.5 and dense.kernel_read_theta is None


@pytest.mark.parametrize("theta", [0.0, 0.5, 1.0])
def test_tree_read_pairs_cover_the_boundary_measure_once(theta):
    """Every point's pair list (cells and nodes) represents the whole boundary measure
    exactly once, whatever theta; at theta = 0 there are no node pairs."""
    torch.manual_seed(0)
    pts, nrm, w, drv = cloud(500, d=3, seed=3)
    q, _, _, _ = cloud(150, d=3, seed=4, scale=(4.0, 4.0, 4.0))
    log_frac = torch.log(w / w.sum())[None]
    basis = _KernelBasis(global_geo_width(1, 1, 0)).double()
    pairs = tree_read_pairs(
        basis,
        query_r=q[None],
        query_n=torch.nn.functional.normalize(q, dim=-1)[None],
        query_wall=torch.rand(1, 150, dtype=D) + 0.1,
        query_vecs=torch.nn.functional.normalize(torch.randn(1, 150, 1, 3, dtype=D), dim=-1),
        boundary_r=pts[None],
        boundary_n=nrm[None],
        boundary_vecs=None,
        log_frac=log_frac,
        g_unit=torch.nn.functional.normalize(drv, dim=-1)[None, None],
        theta=theta,
        eps=1e-12,
    )
    plan = pairs.plan
    mass = torch.zeros(plan.n_q, dtype=D).index_add(0, plan.qi, torch.exp(pairs.log_m))
    assert torch.allclose(mass, torch.ones(plan.n_q, dtype=D), atol=1e-12)
    assert pairs.basis.shape == (pairs.n_pairs, tree_read.KERNEL_READ_BASIS)
    assert pairs.basis.isfinite().all()
    if theta == 0.0:
        assert pairs.n_pairs_node == 0 and pairs.n_pairs == 150 * 500
    else:
        assert pairs.n_pairs_node > 0 and pairs.n_pairs < 150 * 500


def _node_cells(f, i):
    cells, _ = f.node_cells(torch.tensor([i]))
    return cells


@pytest.mark.parametrize("dim", [3, 2])
def test_boundary_forest_splits_by_normal_octant_and_its_nodes_are_tight(dim):
    """One cluster tree per normal octant: the trees' cells partition the cells and
    each tree's cells share an octant; every node's box holds its cells; its virtual
    cell is their measure-weighted centroid, renormalized mean normal, mean rank-1
    input and total measure; its coherence is the RMS-chord test; the children split
    their parent's range; the leaves tile the cells with at most the leaf size each."""
    pts, nrm, w, _ = cloud(1000, d=dim, seed=8)
    vecs = torch.nn.functional.normalize(torch.randn(1000, 2, dim, dtype=D), dim=-1)
    theta = 0.5
    f = BoundaryForest.build(pts, w, nrm, vecs, theta=theta, leaf_size=8, eps=1e-12)
    n, m = f.n_cells, f.n_nodes
    assert n == 1000 and len(f.trees) == len(f.tree_cells) == int(f.node_offset.shape[0]) - 1
    assert len(f.trees) == 2**dim
    assert sum(t.n_nodes for t in f.trees) == m
    assert torch.equal(torch.sort(torch.cat(f.tree_cells)).values, torch.arange(n))
    assert torch.equal(torch.sort(f.cells).values, torch.arange(n))
    ### Octants and boxes live in the boundary's own frame.
    assert torch.allclose(f.frame.t() @ f.frame, torch.eye(dim, dtype=D), atol=1e-12)
    pts_f = pts @ f.frame
    octant = normal_octants(nrm @ f.frame)
    for sel in f.tree_cells:
        assert len(set(octant[sel].tolist())) == 1
    for i in range(m):
        cells = _node_cells(f, i)
        assert bool((pts_f[cells] >= f.aabb_min[i] - 1e-9).all())
        assert bool((pts_f[cells] <= f.aabb_max[i] + 1e-9).all())
        assert torch.allclose(f.measure[i], w[cells].sum())
        assert torch.allclose(f.center[i], (pts[cells] * w[cells, None]).sum(0) / w[cells].sum())
        mean_n = (nrm[cells] * w[cells, None]).sum(0) / w[cells].sum()
        assert torch.allclose(f.normal[i], mean_n / mean_n.norm(), atol=1e-9)
        mean_u = (vecs[cells] * w[cells, None, None]).sum(0) / w[cells].sum()
        assert torch.allclose(f.vecs[i], mean_u, atol=1e-9)
        rms_chord = math.sqrt(2 * (1 - float(mean_n.norm())))
        assert bool(f.coherent[i]) == (rms_chord < theta)
        left, right = int(f.left[i]), int(f.right[i])
        if left >= 0:
            a, b = sorted(
                [(int(f.range_start[c]), int(f.range_count[c])) for c in (left, right)]
            )
            assert a[0] == int(f.range_start[i]) and a[0] + a[1] == b[0]
            assert b[0] + b[1] == int(f.range_start[i] + f.range_count[i])
        else:
            assert 1 <= int(f.range_count[i]) <= 8
    leaves = f.leaf_ids
    ends = torch.cumsum(f.range_count[leaves], 0)
    assert torch.equal(f.range_start[leaves][1:], ends[:-1]) and int(ends[-1]) == n
    assert torch.allclose(f.normal.norm(dim=-1), torch.ones(m, dtype=D))
    assert f.vecs.shape == (m, 2, dim)


def test_boundary_forest_never_mixes_opposite_normals():
    """A thin plate in a random orientation (two faces a hair apart, opposite normals)
    lands in two trees, one per face, so no node holds cells of both faces whatever
    its position, and every node is coherent with its face's normal."""
    g = torch.Generator().manual_seed(11)
    xy = torch.rand(400, 2, generator=g, dtype=D)
    up = torch.cat([xy, torch.full((400, 1), 1e-3, dtype=D)], -1)
    down = torch.cat([xy, torch.full((400, 1), -1e-3, dtype=D)], -1)
    R = orthogonal(3, 5, False)
    pts = torch.cat([up, down]) @ R.T
    nrm = torch.cat([torch.tensor([[0.0, 0.0, 1.0]] * 400, dtype=D), torch.tensor([[0.0, 0.0, -1.0]] * 400, dtype=D)]) @ R.T
    w = torch.ones(800, dtype=D)
    f = BoundaryForest.build(pts, w, nrm, None, theta=0.5, leaf_size=8, eps=1e-12)
    assert len(f.trees) == 2
    for sel in f.tree_cells:
        assert len(set((sel >= 400).tolist())) == 1
    assert bool(f.coherent.all())
    assert torch.allclose(f.normal, nrm[f.cells[f.range_start]], atol=1e-9)


def test_node_means_are_the_measure_weighted_means_of_every_node():
    """The scatter-and-sum-up node means equal the explicit weighted mean of each node's
    cells, and carry a gradient to the cells' rows."""
    pts, nrm, w, _ = cloud(300, d=3, seed=9)
    f = BoundaryForest.build(pts, w, nrm, None, theta=0.5, leaf_size=8, eps=1e-12)
    x = torch.randn(300, 4, 5, dtype=D, requires_grad=True)
    means = tree_read.node_means(x, f, w)
    assert means.shape == (f.n_nodes, 4, 5)
    for i in range(f.n_nodes):
        cells = _node_cells(f, i)
        ref = (x[cells] * w[cells, None, None]).sum(0) / w[cells].sum()
        assert torch.allclose(means[i], ref, atol=1e-5), i
    means.sum().backward()
    ### d(sum of all node means)/dx_j = w_j * sum over the nodes holding j of 1 / measure.
    ref = torch.zeros(300, dtype=D)
    for i in range(f.n_nodes):
        cells = _node_cells(f, i)
        ref[cells] += w[cells] / w[cells].sum()
    assert torch.allclose(x.grad, ref[:, None, None].expand_as(x), atol=1e-5)


@pytest.mark.parametrize("theta", [0.0, 0.5, 1.0])
def test_point_pairs_admit_only_separated_coherent_nodes_and_cover_every_cell(theta):
    """Each point's walk admits a node only when the node's box diagonal is below theta
    times the distance from the point to the box and the node is coherent (RMS normal
    chord below theta), and its cells and nodes together hold every cell exactly once;
    a point's list is the same whether it walks alone or with the others."""
    pts, nrm, w, _ = cloud(700, d=3, seed=6)
    q, _, _, _ = cloud(90, d=3, seed=7, scale=(3.0, 3.0, 3.0))
    f = BoundaryForest.build(pts, w, nrm, None, theta=theta)
    qi, si, n_node = f.point_pairs(q)
    n = f.n_cells
    assert torch.equal(qi, qi.sort().values)
    is_node = si >= n
    assert int(is_node.sum()) == n_node
    node = si[is_node] - n
    p = (q @ f.frame)[qi[is_node]]  # the boxes live in the boundary's own frame
    gap = torch.clamp(torch.maximum(p - f.aabb_max[node], f.aabb_min[node] - p), min=0)
    assert bool((gap.square().sum(-1) * theta**2 > f.diameter_sq[node]).all())
    assert bool(f.coherent[node].all())
    ### Coverage: a cell is counted through itself or through exactly one node.
    counts = torch.zeros(q.shape[0], n, dtype=torch.long)
    counts.index_put_((qi[~is_node], si[~is_node]), torch.ones_like(qi[~is_node]), accumulate=True)
    cells, which = f.node_cells(node)
    counts.index_put_((qi[is_node][which], cells), torch.ones_like(cells), accumulate=True)
    assert torch.equal(counts, torch.ones_like(counts))
    if theta == 0.0:
        assert n_node == 0
    ### Query independence of the list itself.
    qi_one, si_one, _ = f.point_pairs(q[17:18])
    assert torch.equal(torch.sort(si_one).values, torch.sort(si[qi == 17]).values)
    assert bool((qi_one == 0).all())


@pytest.mark.parametrize("improper", [False, True], ids=["rotation", "reflection"])
def test_forest_and_every_list_are_the_same_in_every_pose_at_finite_theta(improper):
    """The trees live in the boundary's own frame (principal axes signed by the third
    moments), so a rotated, reflected, translated and rescaled copy of a generic case
    gives the same frame (moved with the case), the same cells in the same order, the
    same nodes and, for every point, the same far-field list at theta 1."""
    pts, nrm, w, _ = cloud(700, d=3, seed=6)
    q, _, _, _ = cloud(90, d=3, seed=7, scale=(3.0, 3.0, 3.0))
    R = orthogonal(3, 4, improper)
    t = torch.arange(1.0, 4.0, dtype=D)
    k = 2.5
    f0 = BoundaryForest.build(pts, w, nrm, None, theta=1.0)
    f1 = BoundaryForest.build(k * pts @ R.T + t, 3.0 * w, nrm @ R.T, None, theta=1.0)
    assert torch.allclose(f1.frame, R @ f0.frame, atol=1e-9)
    assert torch.equal(f1.cells, f0.cells)
    assert torch.equal(f1.range_start, f0.range_start) and torch.equal(f1.range_count, f0.range_count)
    assert torch.equal(f1.coherent, f0.coherent)
    assert torch.allclose(f1.center, k * f0.center @ R.T + t, atol=1e-9)
    assert torch.allclose(f1.normal, f0.normal @ R.T, atol=1e-9)
    qi0, si0, n0 = f0.point_pairs(q)
    qi1, si1, n1 = f1.point_pairs(k * q @ R.T + t)
    assert n0 == n1 and torch.equal(qi0, qi1) and torch.equal(si0, si1)


@pytest.mark.parametrize("improper", [False, True], ids=["rotation", "reflection"])
def test_tree_read_is_covariant_in_every_pose_at_finite_theta(improper):
    """At theta 1 the read is an approximation, but the same approximation in every
    pose of the case, so a rotated, reflected and translated copy gives the same
    predictions to roundoff (the posed protocol equals the native one)."""
    _, tree = _pair(3, 1.0)
    mesh, drv, q, sdf, sn = _case(3)
    base = outputs(_run(tree, mesh, drv, q, sdf, sn))
    R = orthogonal(3, 4, improper)
    t = torch.arange(1.0, 4.0, dtype=D)
    out = outputs(_run(tree, transform(mesh, R, t), drv @ R.T, q @ R.T + t, sdf, sn @ R.T))
    assert torch.allclose(out["p"], base["p"], atol=1e-9)
    assert torch.allclose(out["v"], base["v"] @ R.T, atol=1e-9)


def test_tree_read_error_falls_with_theta_on_a_smooth_boundary():
    """Against the dense read, the tree read's deviation at theta = 0.25 is below the
    deviation at theta = 1 and both are small on a sphere (smooth normals, so the
    cluster monopoles are good far-field surrogates); theta = 0 is exact."""
    dense, _ = _pair(3, 0.0, seed=1)
    mesh, drv, q, sdf, sn = _sphere_case()
    ref = outputs(_run(dense, mesh, drv, q, sdf, sn))
    devs = {}
    for theta in (0.0, 0.25, 1.0):
        tree = _model(3, theta, randomize=False, seed=1).double().eval()
        tree.load_state_dict(dense.state_dict())
        devs[theta] = _rel(outputs(_run(tree, mesh, drv, q, sdf, sn)), ref)
    assert devs[0.0] < 1e-10, devs
    assert devs[0.25] < devs[1.0], devs
    assert devs[1.0] < 0.1, devs


@pytest.mark.parametrize("theta", [0.0, 0.5, 1.0])
def test_tree_read_is_exactly_query_independent_at_every_theta(theta):
    """A prediction depends on its own point and the boundary only, whatever theta (a
    point's walk of the tree does not see the other points): alone, a prefix, permuted,
    the same rows to 1e-11."""
    _, tree = _pair(3, theta)
    mesh, drv, q, sdf, sn = _case()
    full = outputs(_run(tree, mesh, drv, q, sdf, sn))
    idx = torch.tensor([3, 50, 117])
    alone = outputs(_run(tree, mesh, drv, q[idx], sdf[idx], sn[idx]))
    assert same(alone, _rows(full, idx), atol=1e-11)
    perm = torch.randperm(q.shape[0], generator=torch.Generator().manual_seed(5))
    permuted = outputs(_run(tree, mesh, drv, q[perm], sdf[perm], sn[perm]))
    assert same(permuted, _rows(full, perm), atol=1e-11)
    half = q.shape[0] // 2
    prefix = outputs(_run(tree, mesh, drv, q[:half], sdf[:half], sn[:half]))
    assert same(prefix, _rows(full, torch.arange(half)), atol=1e-11)


@pytest.mark.parametrize("improper", [False, True], ids=["rotation", "reflection"])
@pytest.mark.parametrize("dim", [3, 2])
def test_tree_read_covariance_at_theta_zero(improper, dim):
    _, tree = _pair(dim, 0.0)
    mesh, drv, q, sdf, sn = _case(dim)
    base = outputs(_run(tree, mesh, drv, q, sdf, sn))
    R = orthogonal(dim, 4, improper)
    t = torch.arange(1.0, dim + 1.0, dtype=D)
    out = outputs(_run(tree, transform(mesh, R, t), drv @ R.T, q @ R.T + t, sdf, sn @ R.T))
    assert torch.allclose(out["p"], base["p"], atol=1e-10)
    assert torch.allclose(out["v"], base["v"] @ R.T, atol=1e-10)


def test_tree_read_similarity_refinement_and_measure_scale_at_theta_zero():
    _, tree = _pair(3, 0.0)
    mesh, drv, q, sdf, sn = _case()
    base = outputs(_run(tree, mesh, drv, q, sdf, sn))
    k = 2.5
    scaled = outputs(_run(tree, transform(mesh, scale=k), drv, k * q, k * sdf, sn))
    assert same(scaled, base, atol=1e-10)
    refined = outputs(_run(tree, refine(mesh), drv, q, sdf, sn))
    assert same(refined, base, atol=1e-10)
    data = dict(mesh.cell_data.items())
    data[EFFECTIVE_MEASURE_KEY] = data[EFFECTIVE_MEASURE_KEY] * 3.0
    rescaled = outputs(_run(tree, mesh.with_data(cell_data=data), drv, q, sdf, sn, surface=mesh))
    assert same(rescaled, base, atol=1e-10)


def test_tree_read_every_parameter_receives_a_gradient():
    m = _model(3, 0.5).double()
    mesh, drv, q, sdf, sn = _case()
    out = m(
        boundary_meshes={"wall": mesh},
        prediction_points=points_mesh(q, {"sdf": sdf, "sdf_normals": sn}),
        global_data={"g": drv},
    )
    loss = sum(v.square().sum() for v in outputs(out).values())
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert all(
        p.grad.abs().sum() > 0
        for n_, p in m.named_parameters()
        if n_.startswith("kernel_reads") or n_.startswith("kernel_basis")
    )


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_fused_segment_softmax_read_matches_eager(dtype):
    """The Triton read (values and every gradient) against the eager float64 form on a
    random pair list with ragged segments, including empty sources."""
    from physicsnemo.experimental.models.isla.pair_kernels import PairPlan
    from physicsnemo.experimental.models.isla.segment_softmax import (
        _eager_segment_softmax_read,
        segment_softmax_read,
    )

    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(3)
    n_q, n_b, h, kc, c = 37, 50, 8, 4, 8
    counts = torch.randint(0, 5000, (n_q,), generator=g)  # segments longer than a source chunk too
    qi = torch.repeat_interleave(torch.arange(n_q), counts)
    bj = torch.randint(0, n_b - 5, (int(counts.sum()),), generator=g)  # sources 45..49 get no pairs
    bj[: int(counts.sum()) // 2] = 0  # one source read by half of all pairs (many chunks)
    plan = PairPlan.build(qi.to(dev), bj.to(dev), n_q, n_b)
    ker = torch.randn(int(counts.sum()), h, generator=g, dtype=D)
    logm = -torch.rand(int(counts.sum()), generator=g, dtype=D) * 8
    q = torch.randn(n_q, h, kc, generator=g, dtype=D)
    k = torch.randn(n_b, h, kc, generator=g, dtype=D)
    v = torch.randn(n_b, h, c, generator=g, dtype=D)
    plan_cpu = PairPlan.build(qi, bj, n_q, n_b)
    ref_in = [t.clone().requires_grad_(True) for t in (ker, q, k, v)]
    ref = _eager_segment_softmax_read(ref_in[0], logm, ref_in[1], ref_in[2], ref_in[3], plan_cpu, 0.5)
    seed = torch.randn_like(ref)
    (ref * seed).sum().backward()
    dev_in = [t.to(dev, dtype).requires_grad_(True) for t in (ker, q, k, v)]
    out = segment_softmax_read(dev_in[0], logm.to(dev, torch.float32), dev_in[1], dev_in[2], dev_in[3], plan, 0.5)
    (out * seed.to(dev, out.dtype)).sum().backward()
    tol = 1e-4 if dtype == torch.float32 else 5e-2
    rows = counts > 0
    assert torch.allclose(out.cpu().double()[rows], ref[rows], atol=tol, rtol=tol)
    for a, b, name in zip(dev_in, ref_in, ("ker", "q", "k", "v")):
        ga, gb_ = a.grad.cpu().double(), b.grad
        assert torch.allclose(ga, gb_, atol=tol * gb_.abs().max(), rtol=tol), name


@cuda_only
def test_tree_read_cuda_float32_matches_cpu_float64():
    """The whole tree read on CUDA in float32 (the fused kernel) against the CPU float64
    eager form, at a finite theta."""
    _, tree = _pair(3, 0.5)
    mesh, drv, q, sdf, sn = _sphere_case(n=1500, nq=120)
    cpu = outputs(_run(tree, mesh, drv, q, sdf, sn))
    dev = torch.device("cuda")
    tree_c = tree.to(dev).float()
    mesh_c = mesh.to(dev).apply(lambda t: t.float() if t.is_floating_point() else t)
    gpu = outputs(
        _run(tree_c, mesh_c, drv.to(dev).float(), q.to(dev).float(), sdf.to(dev).float(), sn.to(dev).float())
    )
    assert _rel({k: v.cpu().double() for k, v in gpu.items()}, cpu) < 1e-3


@cuda_only
@pytest.mark.parametrize("theta", [0.0, 0.5])
def test_tree_read_cuda_matches_cpu_float64(theta):
    _, tree = _pair(3, theta)
    mesh, drv, q, sdf, sn = _sphere_case(n=1500, nq=120)
    cpu = outputs(_run(tree, mesh, drv, q, sdf, sn))
    dev = torch.device("cuda")
    tree_c = tree.to(dev)
    gpu = outputs(
        _run(tree_c, mesh.to(dev), drv.to(dev), q.to(dev), sdf.to(dev), sn.to(dev))
    )
    assert same({k: v.cpu() for k, v in gpu.items()}, cpu, atol=1e-9)
