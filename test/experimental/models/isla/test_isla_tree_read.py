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

The tree read is the dense global read evaluated through a Barnes-Hut cluster tree
over the boundary cells, walked once per prediction point. At ``theta = 0`` it is the
dense read pair for pair, so every contract of the dense read must hold exactly there:
equality with the dense form, Euclidean covariance, the similarity gauge, refinement
and measure-scale invariance, a gradient for every parameter, CUDA against CPU. Query
independence holds exactly at every ``theta`` (a point's walk depends on the tree and
the point alone). At a finite ``theta`` the read is an approximation whose error falls
with ``theta``; the tests measure that on a smooth boundary and check that the pair
list still covers the whole boundary measure exactly once per point and admits only
well-separated clusters.
"""

import math

import pytest
import torch

import physicsnemo.experimental.models.isla.tree_read as tree_read
from physicsnemo.experimental.models import ISLA
from physicsnemo.experimental.models.isla.kernel_read import _KernelBasis, global_geo_width
from physicsnemo.experimental.models.isla.boundary_tree import BoundaryTree
from physicsnemo.experimental.models.isla.tree_read import (
    _TreeKernelRead,
    point_tree_pairs,
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


def _run(m, mesh, drv, q, sdf, sn):
    with torch.no_grad():
        return m(
            boundary_meshes={"wall": mesh},
            prediction_points=points_mesh(q, {"sdf": sdf, "sdf_normals": sn}),
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


@pytest.mark.parametrize("dim", [3, 2])
def test_boundary_tree_is_balanced_tight_and_frame_free(dim):
    """Every leaf sits at the same depth with at most the leaf size of cells; every
    node's cells lie inside its bounding sphere and sum to its measure; the leaves
    partition the cells; a rotated, reflected, translated and rescaled copy of the cells
    gives the same clusters (as sets of cells)."""
    pts, nrm, w, _ = cloud(1000, d=dim, seed=8)
    vecs = torch.nn.functional.normalize(torch.randn(1000, 2, dim, dtype=D), dim=-1)
    tree = BoundaryTree.build(pts, w, nrm, vecs, leaf_size=8, eps=1e-12)
    n, m = tree.n_cells, tree.n_nodes
    assert tree.depth == 7 and m == 2**8 - 1 and n == 1000
    leaves = torch.arange(tree.first_leaf, m)
    assert int(tree.node_count[leaves].max()) <= 8 and int(tree.node_count[leaves].min()) >= 4
    assert int(tree.node_count[leaves].sum()) == n
    assert torch.equal(torch.sort(tree.order).values, torch.arange(n))
    for i in range(m):
        cells = tree.order[tree.node_start[i] : tree.node_start[i] + tree.node_count[i]]
        d2 = (pts[cells] - tree.center[i]).norm(dim=-1)
        assert float(d2.max()) <= float(tree.radius[i]) * (1 + 1e-12) + 1e-12
        assert torch.allclose(tree.measure[i], w[cells].sum())
        c = (pts[cells] * w[cells, None]).sum(0) / w[cells].sum()
        assert torch.allclose(tree.center[i], c)
        if i < tree.first_leaf:
            left, right = 2 * i + 1, 2 * i + 2
            assert int(tree.node_start[left]) == int(tree.node_start[i])
            assert int(tree.node_count[left] + tree.node_count[right]) == int(tree.node_count[i])
    assert torch.allclose(tree.normal.norm(dim=-1), torch.ones(m, dtype=D))
    assert tree.vecs.shape == (m, 2, dim)
    R = orthogonal(dim, 3, True)
    t = torch.arange(1.0, dim + 1.0, dtype=D)
    k = 3.0
    moved = BoundaryTree.build(k * pts @ R.T + t, 2.0 * w, nrm @ R.T, vecs @ R.T, leaf_size=8, eps=1e-12)
    for i in range(m):
        a = set(tree.order[tree.node_start[i] : tree.node_start[i] + tree.node_count[i]].tolist())
        b = set(moved.order[moved.node_start[i] : moved.node_start[i] + moved.node_count[i]].tolist())
        assert a == b, i
    assert torch.allclose(moved.radius, k * tree.radius, rtol=1e-9, atol=1e-12)
    assert torch.allclose(moved.center, k * tree.center @ R.T + t, atol=1e-9)
    assert torch.allclose(moved.normal, tree.normal @ R.T, atol=1e-9)
    assert torch.allclose(moved.measure, 2.0 * tree.measure)


@pytest.mark.parametrize("theta", [0.0, 0.5, 1.0])
def test_point_tree_pairs_admit_only_separated_clusters_and_cover_every_cell(theta):
    """Each point's walk admits a node only when the node's sphere diameter is below
    theta times the gap between the point and the sphere, and its cells and nodes
    together hold every cell exactly once."""
    pts, nrm, w, _ = cloud(700, d=3, seed=6)
    q, _, _, _ = cloud(90, d=3, seed=7, scale=(3.0, 3.0, 3.0))
    tree = BoundaryTree.build(pts, w, nrm, None, leaf_size=tree_read.TREE_READ_LEAF_SIZE, eps=1e-12)
    qi, si, n_node = point_tree_pairs(tree, q, theta)
    n = pts.shape[0]
    assert torch.equal(qi, qi.sort().values)
    is_node = si >= n
    assert int(is_node.sum()) == n_node
    node = si[is_node] - n
    p = q[qi[is_node]]
    gap = (p - tree.center[node]).norm(dim=-1) - tree.radius[node]
    assert bool((2 * tree.radius[node] < theta * gap).all())
    ### Coverage: a cell is counted through itself or through exactly one node.
    counts = torch.zeros(q.shape[0], n, dtype=torch.long)
    counts.index_put_((qi[~is_node], si[~is_node]), torch.ones_like(qi[~is_node]), accumulate=True)
    starts, cnts = tree.node_start[node], tree.node_count[node]
    pos, pid = tree_read._ragged_arange(starts, cnts)
    counts.index_put_((qi[is_node][pid], tree.order[pos]), torch.ones_like(pos), accumulate=True)
    assert torch.equal(counts, torch.ones_like(counts))
    if theta == 0.0:
        assert n_node == 0


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
    rescaled = outputs(_run(tree, mesh.with_data(cell_data=data), drv, q, sdf, sn))
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
