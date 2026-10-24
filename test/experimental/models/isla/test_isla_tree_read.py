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

"""Contract tests for ISLA-DTT's boundary read (``query_independent=True``).

Passive prediction points read every boundary cell after every block: a softmax over
the cells with the log measure fraction as bias, a learned kernel of the pair
invariants and a query-key match, evaluated through a forest of Barnes-Hut cluster
trees over the boundary cells (GLOBE's ``ClusterTree``, one tree per normal octant),
walked once per prediction point at the opening angle ``TREE_READ_THETA``. At
``theta = 0`` (the model's private ``_tree_read_theta``, set by these tests) it reads
every cell pair for pair, so every contract must hold exactly there: equality with a
brute-force all-pairs reference kept in this file, Euclidean covariance, the
similarity gauge, refinement and measure-scale invariance, CUDA against CPU. Query
independence holds exactly at every ``theta`` (a point's walk depends on the forest
and the point alone). At a finite ``theta`` the read is an approximation whose error
falls with ``theta``; the tests measure that on a smooth boundary and check that the
pair list still covers the whole boundary measure exactly once per point and admits
only well-separated, orientation-coherent clusters. The read's output projection
starts at zero, so most tests give every read parameter a random value first.
"""

import copy
import itertools
import logging
import math
import types

import pytest
import torch

import physicsnemo.experimental.models.isla.tree_read as tree_read
from physicsnemo.experimental.models import ISLA
from physicsnemo.experimental.models.isla.tree_read import (
    KERNEL_READ_HEADS,
    KERNEL_READ_KEY,
    KERNEL_READ_WIDTH,
    TREE_READ_THETA,
    BoundaryForest,
    _KernelBasis,
    _TreeKernelRead,
    kernel_geo_width,
    normal_octants,
    tree_pair_invariants,
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
D = torch.float64
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


def _model(dim=3, theta=TREE_READ_THETA, randomize=True, seed=0, **kw):
    torch.manual_seed(seed)
    kw = {
        "hidden": 32,
        "n_layers": 2,
        "n_slices": 8,
        "boundary_schemas": {"wall": {}},
        **kw,
    }
    m = ISLA(
        output_schema=PV,
        global_schema=G,
        interior=True,
        query_independent=True,
        dim=dim,
        **kw,
    )
    m._tree_read_theta = theta
    if randomize:
        with torch.no_grad():
            for p in [*m.kernel_reads.parameters(), *m.kernel_basis.parameters()]:
                p.normal_(0.0, 0.3)
    return m


def _tree(dim=3, theta=0.0, **kw):
    """A float64 model in eval mode, its read at ``theta``."""
    return _model(dim, theta, **kw).double().eval()


def _exact_pairs(bnd, qry, q_wall, g_unit):
    """What the brute-force read needs of the tokens (one case)."""
    return {"bnd": bnd, "qry": qry, "q_wall": q_wall, "g_unit": g_unit}


def _exact_read(model, read, h_b, h_q, tok):
    r"""The boundary read of one block over every (point, cell) pair, written out
    densely: the invariants :math:`[r, \log r, \log(r / |d_q|), \hat\rho\cdot n_j,
    \hat\rho\cdot n_q, n_q\cdot n_j, \hat\rho\cdot g_k, n_j\cdot g_k, \hat\rho\cdot u,
    n_q\cdot u]`, the kernel basis, and a softmax over the
    cells of the kernel term, the query-key match and the log measure fraction."""
    bnd, qry, wall, g = tok["bnd"], tok["qry"], tok["q_wall"][0], tok["g_unit"][0]
    nb = model.n_boundary_vecs
    y, n_j = bnd["r"][0], bnd["n_hat"][0]
    x, n_q = qry["r"][0], qry["n_hat"][0]
    rel = x[:, None] - y[None]
    dist = rel.norm(dim=-1)
    rho = rel / dist.clamp_min(1e-12)[..., None]
    nq_, nb_ = x.shape[0], y.shape[0]
    log_r = dist.clamp_min(1e-12).log()
    cols = [
        torch.stack(
            [
                dist,
                log_r,
                log_r - wall.clamp_min(1e-12).log()[:, None],
                (rho * n_j[None]).sum(-1),
                (rho * n_q[:, None]).sum(-1),
                (n_q[:, None] * n_j[None]).sum(-1),
            ],
            -1,
        ),
        torch.einsum("qnd,kd->qnk", rho, g),
        (n_j @ g.t())[None].expand(nq_, nb_, -1),
    ]
    if "vecs" in bnd.keys() and nb:
        u = bnd["vecs"][0][:, :nb]
        cols += [
            torch.einsum("qnd,npd->qnp", rho, u),
            torch.einsum("qd,npd->qnp", n_q, u),
        ]
    phi = model.kernel_basis(torch.cat(cols, -1))  # (Q, N, BASIS)
    nh, c, kc = (
        KERNEL_READ_HEADS,
        KERNEL_READ_WIDTH // KERNEL_READ_HEADS,
        KERNEL_READ_KEY,
    )
    hb = read.norm_b(h_b)
    v = read.value(hb).view(nb_, nh, c)
    k = read.key(hb).view(nb_, nh, kc)
    q = read.query(read.norm_q(h_q)).view(nq_, nh, kc)
    log_w = bnd["log_w"][0, :, 0]
    log_frac = log_w - torch.logsumexp(log_w, 0)
    logit = phi @ read.kernel.weight.t() + torch.einsum(
        "qhc,nhc->qnh", q, k
    ) / math.sqrt(kc)
    att = torch.softmax(logit + log_frac[None, :, None], dim=1)
    return read.out(torch.einsum("qnh,nhc->qhc", att, v).reshape(nq_, nh * c))


def _with_exact_read(model):
    """A copy of ``model`` whose boundary read is :func:`_exact_read` (the reference)."""
    ref = copy.deepcopy(model)
    ref._tree_read_pairs = _exact_pairs
    for read in ref.kernel_reads:
        read.forward = types.MethodType(
            lambda self, h_b, h_q, tok: _exact_read(ref, self, h_b, h_q, tok), read
        )
    return ref


def _case(dim=3, n=300, nq=120, seed=0):
    pts, nrm, w, drv = cloud(n, d=dim, seed=seed)
    g = torch.Generator().manual_seed(seed + 1)
    near = pts[: nq - 10] + 0.15 * torch.randn(nq - 10, dim, generator=g, dtype=D)
    far = 40.0 + torch.randn(10, dim, generator=g, dtype=D)
    q = torch.cat([near, far])
    return soup(pts, nrm, w), drv, q


def _sphere_case(n=4000, nq=200, seed=0):
    """A smooth boundary: a sphere of radius 2 with outward normals and areas, and
    points at radii 2.05 to 6 plus a few far away."""
    g = torch.Generator().manual_seed(seed)
    dirs = torch.nn.functional.normalize(
        torch.randn(n, 3, dtype=D, generator=g), dim=-1
    )
    pts, nrm = 2.0 * dirs, dirs
    w = torch.full((n,), 4 * math.pi * 4.0 / n, dtype=D)
    drv = torch.tensor([0.3, 1.0, 0.2], dtype=D)
    q_dirs = torch.nn.functional.normalize(
        torch.randn(nq, 3, dtype=D, generator=g), dim=-1
    )
    radii = torch.cat(
        [
            2.05 + 3.95 * torch.rand(nq - 10, dtype=D, generator=g),
            torch.full((10,), 40.0, dtype=D),
        ]
    )
    q = q_dirs * radii[:, None]
    return soup(pts, nrm, w), drv, q


def _run(m, mesh, drv, q):
    with torch.no_grad():
        return m(
            boundary_meshes={"wall": mesh},
            prediction_points=points_mesh(q),
            global_data={"g": drv},
        )


def _rows(out, idx):
    """Rows ``idx`` of an outputs mapping."""
    return {k: v[idx] for k, v in out.items()}


def _rel(a, b):
    return max(float((a[k] - b[k]).norm() / b[k].norm().clamp_min(1e-30)) for k in b)


@pytest.mark.parametrize("vectors", [False, True], ids=["plain", "boundary_vectors"])
@pytest.mark.parametrize("dim", [3, 2])
def test_tree_read_at_theta_zero_is_the_exact_read(dim, vectors):
    """At theta 0 the forest read is the all-pairs read (the brute-force reference
    above), with and without rank-1 boundary inputs; at the model's angle it is not."""
    extra = {"boundary_schemas": {"wall": {"tr": {"rank": 1}}}} if vectors else {}
    tree = _tree(dim, 0.0, **extra)
    assert (
        isinstance(tree.kernel_reads[0], _TreeKernelRead)
        and len(tree.kernel_reads) == 2
    )
    mesh, drv, q = _case(dim)
    if vectors:
        tr = torch.randn(
            mesh.n_cells, dim, generator=torch.Generator().manual_seed(5), dtype=D
        )
        mesh = mesh.with_data(
            cell_data={
                EFFECTIVE_MEASURE_KEY: mesh.cell_data[EFFECTIVE_MEASURE_KEY],
                "tr": tr,
            }
        )
    ref = outputs(_run(_with_exact_read(tree), mesh, drv, q))
    assert _rel(outputs(_run(tree, mesh, drv, q)), ref) < 1e-10
    tree._tree_read_theta = TREE_READ_THETA
    assert _rel(outputs(_run(tree, mesh, drv, q)), ref) > 1e-6


@pytest.mark.parametrize("theta", [0.0, 0.5, 1.0])
def test_tree_read_pairs_cover_the_boundary_measure_once(theta):
    """Every point's pair list (cells and nodes) represents the whole boundary measure
    exactly once, whatever theta; at theta = 0 there are no node pairs."""
    torch.manual_seed(0)
    pts, nrm, w, drv = cloud(500, d=3, seed=3)
    q, _, _, _ = cloud(150, d=3, seed=4, scale=(4.0, 4.0, 4.0))
    log_frac = torch.log(w / w.sum())[None]
    basis = _KernelBasis(kernel_geo_width(1, 0)).double()
    pairs = tree_read_pairs(
        basis,
        query_r=q[None],
        query_n=torch.nn.functional.normalize(q, dim=-1)[None],
        query_wall=torch.rand(1, 150, dtype=D) + 0.1,
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
    assert (
        n == 1000
        and len(f.trees) == len(f.tree_cells) == int(f.node_offset.shape[0]) - 1
    )
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
        assert torch.allclose(
            f.center[i], (pts[cells] * w[cells, None]).sum(0) / w[cells].sum()
        )
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
    nrm = (
        torch.cat(
            [
                torch.tensor([[0.0, 0.0, 1.0]] * 400, dtype=D),
                torch.tensor([[0.0, 0.0, -1.0]] * 400, dtype=D),
            ]
        )
        @ R.T
    )
    w = torch.ones(800, dtype=D)
    f = BoundaryForest.build(pts, w, nrm, None, theta=0.5, leaf_size=8, eps=1e-12)
    assert len(f.trees) == 2
    for sel in f.tree_cells:
        assert len(set((sel >= 400).tolist())) == 1
    assert bool(f.coherent.all())
    assert torch.allclose(f.normal, nrm[f.cells[f.range_start]], atol=1e-9)


@pytest.mark.parametrize("dim", [2, 3])
def test_normal_octants_are_antipodal_at_zero_components(dim):
    """Every normal with components in {-1, 0, 1}: -n lands in the octant antipodal
    to n's (a zero component takes the sign of the next nonzero one, so it is not
    coded negative for n and -n alike)."""
    grid = torch.tensor(list(itertools.product((-1.0, 0.0, 1.0), repeat=dim)), dtype=D)
    nrm = torch.nn.functional.normalize(grid[grid.abs().sum(-1) > 0], dim=-1)
    assert torch.equal(normal_octants(-nrm), 2**dim - 1 - normal_octants(nrm))


def test_boundary_forest_partition_is_the_same_under_a_winding_flip():
    """A 2 x 1 rectangle, symmetric about both axes, so its own frame is exact and
    its normals there have zero components: reversing every normal maps each tree's
    cells to the antipodal octant, so the forest partitions the cells the same way."""
    s = (torch.arange(8, dtype=D) + 0.5) / 8
    bottom = torch.stack([2 * s - 1, torch.full_like(s, -0.5)], -1)
    left = torch.stack([torch.full_like(s, -1.0), s - 0.5], -1)
    pts = torch.cat([bottom, -bottom, left, -left])
    axes = torch.tensor([[0.0, -1.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]], dtype=D)
    nrm = axes.repeat_interleave(8, 0)
    w = torch.cat([torch.full((16,), 0.25), torch.full((16,), 0.125)]).to(D)
    forests = [
        BoundaryForest.build(pts, w, n, None, theta=0.5, leaf_size=4, eps=1e-12)
        for n in (nrm, -nrm)
    ]
    assert bool(((nrm @ forests[0].frame) == 0).any())
    blocks = [{frozenset(c.tolist()) for c in f.tree_cells} for f in forests]
    assert blocks[0] == blocks[1]


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
    counts.index_put_(
        (qi[~is_node], si[~is_node]), torch.ones_like(qi[~is_node]), accumulate=True
    )
    cells, which = f.node_cells(node)
    counts.index_put_(
        (qi[is_node][which], cells), torch.ones_like(cells), accumulate=True
    )
    assert torch.equal(counts, torch.ones_like(counts))
    if theta == 0.0:
        assert n_node == 0
    ### Query independence of the list itself.
    qi_one, si_one, _ = f.point_pairs(q[17:18])
    assert torch.equal(torch.sort(si_one).values, torch.sort(si[qi == 17]).values)
    assert bool((qi_one == 0).all())


@pytest.mark.parametrize("theta", [0.5, 1.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, "bfloat16 autocast"])
@pytest.mark.parametrize("dim", [2, 3])
def test_fast_walk_gives_the_reference_lists_in_the_reference_order(dim, dtype, theta):
    """The fast walk (Warp; forced here, on CPU) gives the reference walk's lists
    entry for entry and in its order, with the same node count, for points near and
    far from the boundary, a NaN point (opened down to every cell) and an infinite
    one; under bfloat16 autocast (as in training) too, in whatever precision the
    reference forest is built."""
    autocast = dtype == "bfloat16 autocast"
    dtype = torch.float32 if autocast else dtype
    pts, nrm, w, _ = cloud(900, d=dim, seed=3, dtype=dtype)
    q, _, _, _ = cloud(150, d=dim, seed=4, dtype=dtype, scale=(3.0,) * dim)
    q[5] = float("nan")
    q[6, 0] = float("inf")
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        f = BoundaryForest.build(pts, w, nrm, None, theta=theta)
        qi, si, n_node = f.point_pairs(q, walk="reference")
        qi_f, si_f, n_node_f = f.point_pairs(q, walk="fast")
    assert torch.equal(qi_f, qi) and torch.equal(si_f, si)
    assert n_node_f == n_node > 0
    assert torch.equal(torch.sort(si[qi == 5]).values, torch.arange(f.n_cells))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=cuda_only)])
@pytest.mark.parametrize(
    "boxes,diagonals,autocast",
    [
        (torch.float32, torch.float32, False),
        (torch.float64, torch.float64, False),
        (torch.bfloat16, torch.bfloat16, False),
        (torch.bfloat16, torch.float32, True),
    ],
    ids=["float32", "float64", "bfloat16", "bfloat16_boxes_under_autocast"],
)
def test_fast_walk_opening_test_is_the_references_at_the_threshold(
    device, boxes, diagonals, autocast
):
    """Every node a one-leaf tree whose squared diagonal is the reference's own
    ``min_dist_sq * theta^2`` for one point (not far) or one ulp below it (far): each
    (point, node) decision of the fast walk is that of GLOBE's traversal, so the walk
    rounds the test as torch does (its sum order, and the steps autocast runs in
    float32 for a forest built under it)."""
    from types import SimpleNamespace

    from physicsnemo.mesh.spatial.cluster_tree import ClusterTree

    n, d, theta, dev = 256, 3, 0.7, torch.device(device)
    g = torch.Generator().manual_seed(0)
    scale = torch.logspace(-3, 3, n, dtype=D)[torch.randperm(n, generator=g)][:, None]
    p, c = (
        torch.randn(n, d, generator=g, dtype=D) * scale,
        torch.randn(n, d, generator=g, dtype=D) * 3 * scale,
    )
    half = torch.rand(n, d, generator=g, dtype=D) * scale
    p, lo, hi = (x.to(boxes).to(dev) for x in (p, c - half, c + half))
    ar, one = torch.arange(n, device=dev), torch.ones(n, dtype=torch.long, device=dev)
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=autocast):
        pt = p.t().contiguous()
        gap = torch.clamp(
            torch.maximum(pt - hi.t().contiguous(), lo.t().contiguous() - pt), min=0
        )
        at = (gap.pow(2).sum(dim=0) * theta**2).to(diagonals)
        bits = {
            torch.bfloat16: torch.int16,
            torch.float32: torch.int32,
            D: torch.int64,
        }[diagonals]
        below = torch.where(
            at > 0, (at.view(bits) - 1).view(diagonals), at
        )  # one ulp toward zero
        diam_sq = torch.where(ar % 2 == 0, at, below)
        leaves = torch.full((n,), -1, device=dev)
        tree = ClusterTree(
            node_aabb_min=lo, node_aabb_max=hi, node_diameter_sq=diam_sq, node_left_child=leaves,
            node_right_child=leaves, leaf_start=ar, leaf_count=one, node_range_start=ar, node_range_count=one,
            node_total_area=one.float(), sorted_source_order=ar, source_points=p.float(),
            max_depth=torch.zeros((), dtype=torch.long, device=dev), root_ids=ar, batch_size=torch.Size([]),
        )  # fmt: skip
        plan = tree.find_point_interaction_pairs(p, theta=theta, validate=False)
        forest = SimpleNamespace(
            aabb_min=lo, aabb_max=hi, diameter_sq=diam_sq, coherent=torch.ones(n, dtype=torch.bool, device=dev),
            left=leaves, right=leaves, node_offset=torch.arange(n + 1, device=dev), range_start=ar, range_count=one,
            cells=ar, n_cells=n, levels=[],
        )  # fmt: skip
        qi, si, _ = tree_read.walk_pairs(forest, p, theta)
    ref = torch.zeros(n, n, dtype=torch.bool, device=dev)
    ref[plan.far_target_ids, plan.far_source_node_ids] = True
    fast = torch.zeros_like(ref)
    fast[qi[si >= n], si[si >= n] - n] = True
    assert 0 < int(ref[ar, ar].sum()) < n
    assert torch.equal(fast, ref)


def test_tree_walk_resolution_and_the_models_walks_agree(monkeypatch):
    """``auto`` walks by the reference on CPU and at theta 0, a forced walk is kept and
    an unknown one refused; the model's two walks (the resolver patched to force
    each) give the same pairs, so the same outputs bitwise."""
    pts, nrm, w, _ = cloud(300, d=3, seed=1)
    f = BoundaryForest.build(pts, w, nrm, None, theta=1.0)
    assert tree_read.resolve_walk("auto", pts, 1.0, f) == "reference"
    assert tree_read.resolve_walk("fast", pts, 1.0, f) == "fast"
    assert tree_read.resolve_walk("fast", pts, 0.0, f) == "reference"
    with pytest.raises(ValueError, match="walk must be"):
        tree_read.resolve_walk("warp", pts, 1.0, f)
    tree = _tree(3, 0.5)
    mesh, drv, q = _sphere_case(n=1500, nq=120)
    runs = {}
    for walk in ("reference", "fast"):
        monkeypatch.setattr(tree_read, "resolve_walk", lambda *a, w=walk: w)
        runs[walk] = outputs(_run(tree, mesh, drv, q))
    assert all(
        torch.equal(runs["fast"][k], runs["reference"][k]) for k in runs["reference"]
    )


def test_fast_walk_falls_back_to_the_reference_past_its_limits(monkeypatch):
    """A forest too deep for the count pass's stack is walked by the reference under
    ``auto`` and refused under a forced ``"fast"``; so are lists past Warp's array
    limit, found after the count pass."""
    import physicsnemo.experimental.models.isla.tree_walk as tree_walk

    pts, nrm, w, _ = cloud(300, d=3, seed=1)
    q, _, _, _ = cloud(40, d=3, seed=2, scale=(3.0,) * 3)
    f = BoundaryForest.build(pts, w, nrm, None, theta=1.0)
    qi, si, n_node = f.point_pairs(q, walk="reference")
    on_cuda = types.SimpleNamespace(is_cuda=True, device=torch.device("cuda"))
    assert tree_read.resolve_walk("auto", on_cuda, 1.0, f) == "fast"
    with monkeypatch.context() as m:
        m.setattr(tree_walk, "WALK_STACK", len(f.levels) + 1)
        assert tree_read.resolve_walk("auto", on_cuda, 1.0, f) == "reference"
        with pytest.raises(ValueError, match="does not take this forest"):
            tree_read.resolve_walk("fast", q, 1.0, f)
    monkeypatch.setattr(tree_walk, "WARP_MAX_ELEMENTS", qi.numel())
    with pytest.raises(ValueError, match="Warp's 2\\*\\*31"):
        f.point_pairs(q, walk="fast")
    monkeypatch.setattr(tree_read, "resolve_walk", lambda *a: "fast")
    qi_a, si_a, n_node_a = f.point_pairs(q)
    assert torch.equal(qi_a, qi) and torch.equal(si_a, si) and n_node_a == n_node


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
    assert torch.equal(f1.range_start, f0.range_start) and torch.equal(
        f1.range_count, f0.range_count
    )
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
    tree = _tree(3, TREE_READ_THETA)
    mesh, drv, q = _case(3)
    base = outputs(_run(tree, mesh, drv, q))
    R = orthogonal(3, 4, improper)
    t = torch.arange(1.0, 4.0, dtype=D)
    out = outputs(_run(tree, transform(mesh, R, t), drv @ R.T, q @ R.T + t))
    assert torch.allclose(out["p"], base["p"], atol=1e-9)
    assert torch.allclose(out["v"], base["v"] @ R.T, atol=1e-9)


def test_tree_read_error_falls_with_theta_on_a_smooth_boundary():
    """Against the exact read (theta = 0), the tree read's deviation at theta = 0.25 is
    below the deviation at theta = 1 and both are small on a sphere (smooth normals, so
    the cluster monopoles are good far-field surrogates)."""
    tree = _tree(3, 0.0, seed=1)
    mesh, drv, q = _sphere_case()
    ref = outputs(_run(tree, mesh, drv, q))
    devs = {}
    for theta in (0.25, 1.0):
        tree._tree_read_theta = theta
        devs[theta] = _rel(outputs(_run(tree, mesh, drv, q)), ref)
    assert devs[0.25] < devs[1.0], devs
    assert devs[1.0] < 0.1, devs


@pytest.mark.parametrize("theta", [0.0, 0.5, 1.0])
def test_tree_read_is_exactly_query_independent_at_every_theta(theta):
    """A prediction depends on its own point and the boundary only, whatever theta (a
    point's walk of the tree does not see the other points): alone, a prefix, permuted,
    the same rows to 1e-11."""
    tree = _tree(3, theta)
    mesh, drv, q = _case()
    full = outputs(_run(tree, mesh, drv, q))
    idx = torch.tensor([3, 50, 117])
    alone = outputs(_run(tree, mesh, drv, q[idx]))
    assert same(alone, _rows(full, idx), atol=1e-11)
    perm = torch.randperm(q.shape[0], generator=torch.Generator().manual_seed(5))
    permuted = outputs(_run(tree, mesh, drv, q[perm]))
    assert same(permuted, _rows(full, perm), atol=1e-11)
    half = q.shape[0] // 2
    prefix = outputs(_run(tree, mesh, drv, q[:half]))
    assert same(prefix, _rows(full, torch.arange(half)), atol=1e-11)


@pytest.mark.parametrize("improper", [False, True], ids=["rotation", "reflection"])
@pytest.mark.parametrize("dim", [3, 2])
def test_tree_read_covariance_at_theta_zero(improper, dim):
    tree = _tree(dim, 0.0)
    mesh, drv, q = _case(dim)
    base = outputs(_run(tree, mesh, drv, q))
    R = orthogonal(dim, 4, improper)
    t = torch.arange(1.0, dim + 1.0, dtype=D)
    out = outputs(_run(tree, transform(mesh, R, t), drv @ R.T, q @ R.T + t))
    assert torch.allclose(out["p"], base["p"], atol=1e-10)
    assert torch.allclose(out["v"], base["v"] @ R.T, atol=1e-10)


def test_tree_read_similarity_refinement_and_measure_scale_at_theta_zero():
    tree = _tree(3, 0.0)
    mesh, drv, q = _case()
    base = outputs(_run(tree, mesh, drv, q))
    k = 2.5
    scaled = outputs(_run(tree, transform(mesh, scale=k), drv, k * q))
    assert same(scaled, base, atol=1e-10)
    refined = outputs(_run(tree, refine(mesh), drv, q))
    assert same(refined, base, atol=1e-10)
    data = dict(mesh.cell_data.items())
    data[EFFECTIVE_MEASURE_KEY] = data[EFFECTIVE_MEASURE_KEY] * 3.0
    rescaled = outputs(_run(tree, mesh.with_data(cell_data=data), drv, q))
    assert same(rescaled, base, atol=1e-10)


@pytest.mark.parametrize("randomize", [False, True], ids=["initial", "random"])
def test_tree_read_every_parameter_receives_a_gradient(randomize):
    """DDP: every parameter takes part in the loss, from the zero-initialized start
    of the read's output projection included."""
    m = _model(3, 0.5, randomize=randomize).double()
    mesh, drv, q = _case()
    out = m(
        boundary_meshes={"wall": mesh},
        prediction_points=points_mesh(q),
        global_data={"g": drv},
    )
    loss = sum(v.square().sum() for v in outputs(out).values())
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    if randomize:
        assert all(
            p.grad.abs().sum() > 0
            for n_, p in m.named_parameters()
            if n_.startswith("kernel_reads") or n_.startswith("kernel_basis")
        )


def test_tree_read_starts_as_no_update_and_is_live_everywhere():
    """With the zero-initialized output projection the read adds nothing (the model
    equals itself without the read); with trained-like weights it changes every
    point, the far ones included (there is no radius)."""
    mesh, drv, q = _case()
    for randomize in (False, True):
        m = _model(randomize=randomize).double().eval()
        no_read = copy.deepcopy(m)
        no_read.kernel_reads = None
        a, b = _run(m, mesh, drv, q), _run(no_read, mesh, drv, q)
        if not randomize:
            assert same(a, outputs(b), atol=1e-12, rtol=0)
        else:
            far = slice(q.shape[0] - 10, None)
            diff = (a.point_data["p"] - b.point_data["p"]).abs()
            assert bool((diff[far] > 1e-8).all()), diff[far]
            assert bool((diff > 1e-8).all())


def test_tree_pair_invariants_are_scale_free_and_the_near_wall_invariant():
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
    qi, si = (
        t.reshape(-1)
        for t in torch.meshgrid(torch.arange(21), torch.arange(50), indexing="ij")
    )

    def inv_at(k):
        geo = dict(
            query_r=k * x,
            query_n=nq,
            query_wall=k * d,
            src_r=k * y,
            src_n=n,
            src_vecs=None,
            g_unit=gu,
            eps=1e-12,
        )
        return tree_pair_invariants(qi, si, **geo).reshape(21, 50, -1)

    inv = inv_at(1.0)
    assert inv.shape == (21, 50, kernel_geo_width(1, 0)) and torch.isfinite(inv).all()
    r = torch.cdist(x, y)
    assert torch.allclose(inv[..., 0], r) and torch.allclose(inv[1:, :, 1], r[1:].log())
    near = inv[1:, :, 2]
    assert near.min() > -1e-12 and torch.allclose(
        near.min(-1).values, torch.zeros(20, dtype=D), atol=1e-12
    )
    assert torch.allclose(inv_at(3.1)[1:, :, 2:], inv[1:, :, 2:], atol=1e-12)


def test_tree_read_bf16_autocast_is_finite():
    m = _model()
    mesh, drv, q = _case()
    q = torch.cat([q, mesh.cell_centroids[:2]])  # points on a cell centroid: d_q = 0
    f = lambda t: t.float()  # noqa: E731
    bm = {
        "wall": soup(
            f(mesh.cell_centroids),
            f(mesh.cell_normals),
            f(mesh.cell_data[EFFECTIVE_MEASURE_KEY]),
        )
    }
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(
            boundary_meshes=bm,
            prediction_points=points_mesh(f(q)),
            global_data={"g": f(drv)},
        )
    sum(o.float().sum() for o in outputs(out).values()).backward()
    assert all(torch.isfinite(o.float()).all() for o in outputs(out).values())
    assert all(
        torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None
    )


def test_tree_read_rank2_inputs_are_refused():
    with pytest.raises(NotImplementedError, match="rank-2"):
        _model(boundary_schemas={"wall": {"S": {"rank": 2}}}, randomize=False)


@cuda_only
def test_tree_read_cuda_fused_and_compiled_training_path():
    """The protocol's training path on CUDA: fused geometry kernel, compiled blocks,
    bf16 autocast. One training step is finite with every parameter's gradient
    present, the compiled forward matches the eager one to bf16 roundoff, and the
    prediction stays query-independent (float32 roundoff)."""
    dev = "cuda"
    m_e = _model().to(dev)
    m_c = _model(compile_blocks=True).to(dev)
    m_c.load_state_dict(m_e.state_dict())
    mesh, drv, q = _case(n=600, nq=400)
    f = lambda t: t.float().to(dev)  # noqa: E731
    bm = {
        "wall": soup(
            mesh.cell_centroids.float(),
            mesh.cell_normals.float(),
            mesh.cell_data[EFFECTIVE_MEASURE_KEY].float(),
        ).to(dev)
    }
    pp = points_mesh(f(q))
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
    assert m_e.resolve_geo_kernel(bm["wall"].points[None]) == "fused"
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
                prediction_points=points_mesh(f(q[:50])),
                global_data={"g": f(drv)},
            )
        )
    assert all(
        torch.allclose(part[k], full[k][:50], atol=1e-4, rtol=1e-4) for k in part
    )


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_fused_segment_softmax_read_matches_eager(dtype):
    """The Triton read (values and every gradient) against the eager float64 form on a
    random pair list with ragged segments, including empty sources."""
    from physicsnemo.experimental.models.isla.segment_softmax import (
        PairPlan,
        _eager_segment_softmax_read,
        segment_softmax_read,
    )

    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(3)
    n_q, n_b, h, kc, c = 37, 50, 8, 4, 8
    counts = torch.randint(
        0, 5000, (n_q,), generator=g
    )  # segments longer than a source chunk too
    qi = torch.repeat_interleave(torch.arange(n_q), counts)
    bj = torch.randint(
        0, n_b - 5, (int(counts.sum()),), generator=g
    )  # sources 45..49 get no pairs
    bj[: int(counts.sum()) // 2] = (
        0  # one source read by half of all pairs (many chunks)
    )
    plan = PairPlan.build(qi.to(dev), bj.to(dev), n_q, n_b)
    ker = torch.randn(int(counts.sum()), h, generator=g, dtype=D)
    logm = -torch.rand(int(counts.sum()), generator=g, dtype=D) * 8
    q = torch.randn(n_q, h, kc, generator=g, dtype=D)
    k = torch.randn(n_b, h, kc, generator=g, dtype=D)
    v = torch.randn(n_b, h, c, generator=g, dtype=D)
    plan_cpu = PairPlan.build(qi, bj, n_q, n_b)
    ref_in = [t.clone().requires_grad_(True) for t in (ker, q, k, v)]
    ref = _eager_segment_softmax_read(
        ref_in[0], logm, ref_in[1], ref_in[2], ref_in[3], plan_cpu, 0.5
    )
    seed = torch.randn_like(ref)
    (ref * seed).sum().backward()
    dev_in = [t.to(dev, dtype).requires_grad_(True) for t in (ker, q, k, v)]
    out = segment_softmax_read(
        dev_in[0],
        logm.to(dev, torch.float32),
        dev_in[1],
        dev_in[2],
        dev_in[3],
        plan,
        0.5,
    )
    (out * seed.to(dev, out.dtype)).sum().backward()
    tol = 1e-4 if dtype == torch.float32 else 5e-2
    rows = counts > 0
    assert torch.allclose(out.cpu().double()[rows], ref[rows], atol=tol, rtol=tol)
    for a, b, name in zip(dev_in, ref_in, ("ker", "q", "k", "v")):
        ga, gb_ = a.grad.cpu().double(), b.grad
        assert torch.allclose(ga, gb_, atol=tol * gb_.abs().max(), rtol=tol), name


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("dim", [2, 3])
def test_fast_walk_on_cuda_gives_the_reference_lists_in_the_reference_order(dim, dtype):
    """On CUDA, ``auto`` picks the fast walk, which gives the reference walk's lists
    on the same device entry for entry and in its order."""
    dev = torch.device("cuda")
    pts, nrm, w, _ = cloud(3000, d=dim, seed=8, dtype=dtype)
    q, _, _, _ = cloud(2000, d=dim, seed=9, dtype=dtype, scale=(3.0,) * dim)
    pts, nrm, w, q = (t.to(dev) for t in (pts, nrm, w, q))
    f = BoundaryForest.build(pts, w, nrm, None, theta=TREE_READ_THETA)
    assert tree_read.resolve_walk("auto", q, TREE_READ_THETA, f) == "fast"
    qi, si, n_node = f.point_pairs(q, walk="reference")
    qi_f, si_f, n_node_f = f.point_pairs(q)
    assert torch.equal(qi_f, qi) and torch.equal(si_f, si)
    assert n_node_f == n_node > 0


@cuda_only
def test_tree_read_cuda_float32_matches_cpu_float64():
    """The whole tree read on CUDA in float32 (the fused kernel) against the CPU float64
    eager form, at a finite theta."""
    tree = _tree(3, 0.5)
    mesh, drv, q = _sphere_case(n=1500, nq=120)
    cpu = outputs(_run(tree, mesh, drv, q))
    dev = torch.device("cuda")
    tree_c = tree.to(dev).float()
    mesh_c = mesh.to(dev).apply(lambda t: t.float() if t.is_floating_point() else t)
    gpu = outputs(_run(tree_c, mesh_c, drv.to(dev).float(), q.to(dev).float()))
    assert _rel({k: v.cpu().double() for k, v in gpu.items()}, cpu) < 1e-3


@cuda_only
@pytest.mark.parametrize("theta", [0.0, 0.5])
def test_tree_read_cuda_matches_cpu_float64(theta, monkeypatch):
    """The float64 model on CUDA against CPU: both walk the forest to the same pair
    lists (every opening decision agrees), and the outputs agree to 1e-9, at
    theta = 0 (cells only) and at theta > 0, whose virtual cells' means are float64
    in a float64 model (node_means)."""
    import physicsnemo.experimental.models.isla.model as isla_model

    plans = []

    def capture(*args, **kwargs):
        pairs = tree_read.walk_range_pairs(*args, **kwargs)
        plans.append(set(zip(pairs.plan.qi.tolist(), pairs.plan.bj.tolist())))
        return pairs

    monkeypatch.setattr(isla_model, "walk_range_pairs", capture)
    tree = _tree(3, theta)
    mesh, drv, q = _sphere_case(n=1500, nq=120)
    cpu = outputs(_run(tree, mesh, drv, q))
    dev = torch.device("cuda")
    tree_c = tree.to(dev)
    gpu = outputs(_run(tree_c, mesh.to(dev), drv.to(dev), q.to(dev)))
    assert len(plans) == 2 and plans[0] == plans[1]  # the same opening decisions
    gpu = {k: v.cpu() for k, v in gpu.items()}
    for k in cpu:
        d = (gpu[k] - cpu[k]).abs()
        print(
            f"{k}: max abs {d.max().item():.3e}, of the allclose bound {(d / (1e-9 + 1e-5 * cpu[k].abs())).max().item():.3e}"
        )
    assert same(gpu, cpu, atol=1e-9)


_TREE_FIELDS = (
    "node_aabb_min",
    "node_aabb_max",
    "node_diameter_sq",
    "node_left_child",
    "node_right_child",
    "leaf_start",
    "leaf_count",
    "node_range_start",
    "node_range_count",
    "node_total_area",
    "sorted_source_order",
    "source_points",
    "max_depth",
    "root_ids",
)
_FOREST_FIELDS = (
    "frame",
    "node_offset",
    "cells",
    "range_start",
    "range_count",
    "left",
    "right",
    "leaf_ids",
    "position_leaf",
    "aabb_min",
    "aabb_max",
    "diameter_sq",
    "center",
    "normal",
    "vecs",
    "measure",
    "coherent",
)


def _bitwise(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.is_floating_point():
        ints = {8: torch.int64, 4: torch.int32, 2: torch.int16}[a.element_size()]
        a, b = a.contiguous().view(ints), b.contiguous().view(ints)
    return torch.equal(a, b)


def _forest_differences(ref, fast) -> list[str]:
    """The fields in which two forests differ in any bit, their trees' included."""
    bad = [k for k in _FOREST_FIELDS if not _bitwise(getattr(ref, k), getattr(fast, k))]
    if len(ref.levels) != len(fast.levels) or not all(
        _bitwise(x, y) for a, b in zip(ref.levels, fast.levels) for x, y in zip(a, b)
    ):
        bad.append("levels")
    if len(ref.trees) != len(fast.trees):
        return [*bad, "trees"]
    for t, (a, b) in enumerate(zip(ref.trees, fast.trees)):
        bad += [
            f"trees[{t}].{k}"
            for k in _TREE_FIELDS
            if not _bitwise(getattr(a, k), getattr(b, k))
        ]
        if not _bitwise(ref.tree_cells[t], fast.tree_cells[t]):
            bad.append(f"tree_cells[{t}]")
    return bad


def _build_cases(dim, dtype):
    """Boundaries for the fast build: a random cloud with rank-1 inputs, half of it on
    one point (equal Morton codes, midpoint splits), and tiny ones (a tree of one cell,
    trees no larger than a leaf), at three leaf sizes."""
    pts, nrm, w, _ = cloud(1500, d=dim, seed=21, dtype=dtype)
    vecs = torch.randn(1500, 2, dim, dtype=dtype)
    dup = pts.clone()
    dup[:700] = dup[0]
    yield pts, w, nrm, vecs, 4
    yield dup, w, nrm, None, 4
    yield pts, w, nrm, vecs[:, :1], 1
    yield pts, w, nrm, None, 8
    for n in (1, 3, 9, 40):
        yield pts[:n], w[:n], nrm[:n], vecs[:n], 4


def _check_fast_build(dim, dtype, device, autocast):
    for pts, w, nrm, vecs, leaf in _build_cases(dim, dtype):
        pts, w, nrm = pts.to(device), w.to(device), nrm.to(device)
        vecs = None if vecs is None else vecs.to(device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
            forests = [
                BoundaryForest.build(
                    pts, w, nrm, vecs, theta=1.0, leaf_size=leaf, build=b
                )
                for b in ("reference", "fast")
            ]
            q = pts[:50] * 1.5 + 0.25
            (qa, sa, na), (qb, sb, nb) = (f.point_pairs(q) for f in forests)
        ### The fast build follows the precision the reference builds in (under
        ### autocast too): _forest_differences compares dtypes as well as bits.
        assert not _forest_differences(*forests), (pts.shape[0], leaf)
        assert torch.equal(qa, qb) and torch.equal(sa, sb) and na == nb


_PRECISIONS = pytest.mark.parametrize(
    "dtype, autocast",
    [(torch.float32, False), (D, False), (torch.float32, True)],
    ids=["float32", "float64", "bf16-autocast"],
)


@_PRECISIONS
@pytest.mark.parametrize("dim", [3, 2])
def test_fast_forest_build_is_the_reference_forest(dim, dtype, autocast):
    """The fast build (one pass over all octants, Warp's CPU backend here) gives the
    reference's forest in every bit, under bfloat16 autocast too: node table, levels,
    boxes, diameters, virtual cells, measures, coherence and every tree's own fields,
    so the reference walk makes the same pair lists on it."""
    _check_fast_build(dim, dtype, "cpu", autocast)


def test_tree_build_auto_resolves_to_the_reference_off_cuda(monkeypatch):
    """The build's "auto" is the reference build off CUDA, a forced "fast" (the
    resolver patched) gives the model the same outputs, and an unknown name is
    refused."""
    from physicsnemo.experimental.models.isla.tree_build import resolve_tree_build

    pts = torch.zeros(4, 3)
    assert resolve_tree_build("auto", pts) == "reference"
    assert resolve_tree_build("fast", pts) == "fast"
    with pytest.raises(ValueError, match="build must be"):
        resolve_tree_build("gpu", pts)
    m = _tree(3, 1.0)
    mesh, drv, q = _case(n=400, nq=60)
    ref = outputs(_run(m, mesh, drv, q))
    monkeypatch.setattr(tree_read, "resolve_tree_build", lambda *a: "fast")
    fast = outputs(_run(m, mesh, drv, q))
    assert all(torch.equal(v, fast[k]) for k, v in ref.items())


def test_tree_read_kernel_is_the_tree_reads_one_execution_attribute(monkeypatch):
    """``tree_read_kernel`` is an execution choice, not in the state dict: "auto" by
    default hands "auto" to the forest build and the walk, "reference" hands them
    "reference" and keeps the reference read, and any other name is refused."""
    seen = []
    build, walk = tree_read.resolve_tree_build, tree_read.resolve_walk
    monkeypatch.setattr(
        tree_read, "resolve_tree_build", lambda b, *a: seen.append(b) or build(b, *a)
    )
    monkeypatch.setattr(
        tree_read, "resolve_walk", lambda w, *a: seen.append(w) or walk(w, *a)
    )
    m = _tree(3, 1.0)
    keys = list(m.state_dict())
    mesh, drv, q = _case(n=400, nq=60)
    assert m.tree_read_kernel == "auto"
    for mode in ("auto", "reference"):
        m.tree_read_kernel = mode
        seen.clear()
        _run(m, mesh, drv, q)
        assert len(seen) == 2 and set(seen) == {mode}
        assert m.resolve_tree_read_kernel(q) == "reference"
    assert list(m.state_dict()) == keys
    for bad in ("fused", "fast"):
        m.tree_read_kernel = bad
        with pytest.raises(ValueError, match="tree_read_kernel must be"):
            _run(m, mesh, drv, q)


def test_fast_forest_build_refuses_what_it_cannot_build(monkeypatch):
    """A forced "fast" build refuses a dimension it has no Morton code for ("auto"
    picks the reference there), and from its cell limit (Warp's int32 thread ids) on
    it leaves the forest to the reference build."""
    import physicsnemo.experimental.models.isla.tree_build as tree_build

    with pytest.raises(ValueError, match="does not take 4-D points"):
        tree_build.resolve_tree_build("fast", torch.zeros(4, 4))
    assert tree_build.resolve_tree_build("auto", torch.zeros(4, 4)) == "reference"
    tables = []

    def spy(*args, **kwargs):
        tables.append(tree_build.forest_tables(*args, **kwargs))
        return tables[-1]

    monkeypatch.setattr(tree_read, "forest_tables", spy)
    pts, nrm, w, _ = cloud(300, d=3, seed=5, dtype=torch.float32)
    for limit in (301, 300):
        monkeypatch.setattr(tree_build, "_MAX_CELLS", limit)
        forests = [
            BoundaryForest.build(pts, w, nrm, None, theta=1.0, leaf_size=4, build=b)
            for b in ("reference", "fast")
        ]
        assert not _forest_differences(*forests)
    assert tables[0] is not None and tables[1] is None


@cuda_only
@_PRECISIONS
@pytest.mark.parametrize("dim", [3, 2])
def test_fast_forest_build_is_the_reference_forest_on_cuda(dim, dtype, autocast):
    """On CUDA, where "auto" picks it, the fast build gives the reference's forest in
    every bit, under bfloat16 autocast too."""
    from physicsnemo.experimental.models.isla.tree_build import resolve_tree_build

    assert resolve_tree_build("auto", torch.zeros(1, dim, device="cuda")) == "fast"
    _check_fast_build(dim, dtype, "cuda", autocast)


def _step_grads(m, mesh, drv, q, autocast=False):
    """One training step's outputs and every parameter's gradient, in float64."""
    m.zero_grad()
    dev = q.device.type
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=autocast):
        out = outputs(
            m(
                boundary_meshes={"wall": mesh},
                prediction_points=points_mesh(q),
                global_data={"g": drv},
            )
        )
    sum(o.double().square().mean() for o in out.values()).backward()
    grads = {
        n: p.grad.double().cpu() for n, p in m.named_parameters() if p.grad is not None
    }
    return {k: v.detach().double().cpu() for k, v in out.items()}, grads


def test_tree_read_kernel_resolves_to_the_reference_on_cpu_and_its_basis_flow_is_exact(
    monkeypatch,
):
    """On CPU ``tree_read_kernel="auto"`` resolves to the reference read. Forcing
    ``"fused"`` there (the resolver patched) keeps the reference read (the fused
    kernels need CUDA) but builds the basis as the fused path does, from the plan
    with the invariants rebuilt in backward: outputs and every gradient bitwise equal
    to the reference's."""
    m = _model(3, 0.5).double()
    mesh, drv, q = _case()
    assert m.tree_read_kernel == "auto" and m.resolve_tree_read_kernel(q) == "reference"
    threads = torch.get_num_threads()
    ### Some CPU reductions are not run-to-run bitwise across threads.
    torch.set_num_threads(1)
    try:
        ref = _step_grads(m, mesh, drv, q)
        monkeypatch.setattr(ISLA, "resolve_tree_read_kernel", lambda self, p: "fused")
        fused = _step_grads(m, mesh, drv, q)
    finally:
        torch.set_num_threads(threads)
    for a, b in zip(ref, fused):
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_fused_kernel_read_matches_the_reference_read(dtype):
    """The fused read (kernel term inside the kernels) against the reference read fed
    ``basis @ W.T``, values and every gradient, on a ragged random plan with empty
    sources and one source read by half the pairs: in float32 and bfloat16 each is as
    close to the float64 eager read as the reference is (the same rounding points,
    another summation order)."""
    from physicsnemo.experimental.models.isla.fused_read import (
        fused_kernel_read,
        supported,
    )
    from physicsnemo.experimental.models.isla.segment_softmax import (
        PairPlan,
        _eager_segment_softmax_read,
        segment_softmax_read,
    )

    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(4)
    n_q, n_b, h, kc, c, f = 37, 50, 8, 4, 8, 16
    counts = torch.randint(0, 5000, (n_q,), generator=g)
    counts[3] = 0
    p = int(counts.sum())
    qi = torch.repeat_interleave(torch.arange(n_q), counts)
    bj = torch.randint(0, n_b - 5, (p,), generator=g)
    bj[: p // 2] = 0
    leaves = [
        torch.randn(p, f, generator=g, dtype=D),
        0.3 * torch.randn(h, f, generator=g, dtype=D),
        torch.randn(n_q, h, kc, generator=g, dtype=D),
        torch.randn(n_b, h, kc, generator=g, dtype=D),
        torch.randn(n_b, h, c, generator=g, dtype=D),
    ]
    logm = -torch.rand(n_b, generator=g, dtype=D) * 8  # per source
    seed = torch.randn(n_q, h, c, generator=g, dtype=D)

    def run(kind, device, dt):
        plan = PairPlan.build(qi.to(device), bj.to(device), n_q, n_b)
        b, w, q, k, v = ins = [
            t.detach().clone().to(device, dt).requires_grad_(True) for t in leaves
        ]
        lm = logm.to(device, torch.float32 if dt != D else D)
        if kind == "fused":
            assert supported(b, q, k, v)
            out = fused_kernel_read(b, w, lm, q, k, v, plan, 0.5)
        elif kind == "reference":
            out = segment_softmax_read(
                b @ w.t(), lm[plan.bj.long()], q, k, v, plan, 0.5
            )
        else:
            out = _eager_segment_softmax_read(
                b @ w.t(), lm[plan.bj.long()], q, k, v, plan, 0.5
            )
        (out.double() * seed.to(device)).sum().backward()
        return [out.detach().double().cpu()] + [t.grad.double().cpu() for t in ins]

    exact = run("eager", "cpu", D)
    fused, ref = run("fused", dev, dtype), run("reference", dev, dtype)
    rows = counts > 0
    floor = 1e-6 if dtype == torch.float32 else 1e-3
    for name, a, b, e in zip(("out", "basis", "w", "q", "k", "v"), fused, ref, exact):
        if name == "out":
            a, b, e = a[rows], b[rows], e[rows]
        scale = e.abs().max()
        d_fused, d_ref = (a - e).abs().max() / scale, (b - e).abs().max() / scale
        print(
            f"{name}: fused {d_fused:.2e}, reference {d_ref:.2e}, fused-reference {(a - b).abs().max() / scale:.2e}"
        )
        assert d_fused <= 2 * d_ref + floor, name


@cuda_only
@pytest.mark.parametrize("autocast", [False, True], ids=["fp32", "bf16_autocast"])
def test_tree_read_fused_matches_the_reference_on_cuda(autocast):
    """The model on CUDA reads through the fused read by default; one training step's
    outputs and every parameter's gradient are as close to the CPU float64 model's as
    the reference read's are (float32, and bf16 autocast)."""
    m = _model(3, 0.5)
    mesh, drv, q = _sphere_case(n=1500, nq=200)
    exact = _step_grads(copy.deepcopy(m).double(), mesh, drv, q)
    dev = torch.device("cuda")
    m = m.to(dev).float()
    mesh_c = mesh.to(dev).apply(lambda t: t.float() if t.is_floating_point() else t)
    args = (mesh_c, drv.to(dev).float(), q.to(dev).float())
    assert m.resolve_tree_read_kernel(args[2]) == "fused"
    fused = _step_grads(m, *args, autocast=autocast)
    m.tree_read_kernel = "reference"
    ref = _step_grads(m, *args, autocast=autocast)
    floor = 1e-6 if not autocast else 1e-3
    for a, b, e in zip(fused, ref, exact):
        groups = {k.split(".")[0] for k in e}
        for grp in groups:
            keys = [k for k in e if k.split(".")[0] == grp]
            scale = max(float(e[k].abs().max()) for k in keys)
            d_fused = max(float((a[k] - e[k]).abs().max()) for k in keys) / scale
            d_ref = max(float((b[k] - e[k]).abs().max()) for k in keys) / scale
            print(f"{grp}: fused {d_fused:.2e}, reference {d_ref:.2e}")
            assert d_fused <= 2 * d_ref + floor, grp


@cuda_only
def test_fused_reads_sum_their_basis_gradients_as_autograd_does():
    """Fused reads of one basis that share its sink sum their basis gradients in place
    bitwise as autograd sums them when each read passes its own (bfloat16, the
    running sum rounded after every read), and the step leaves no memory behind."""
    from physicsnemo.experimental.models.isla.fused_read import (
        _plan_source_items,
        basis_sink,
        fused_kernel_read,
    )
    from physicsnemo.experimental.models.isla.segment_softmax import PairPlan

    dev, bf = torch.device("cuda"), torch.bfloat16
    g = torch.Generator(device=dev).manual_seed(5)
    n_q, n_b, h, kc, c, f = 300, 400, 8, 4, 8, 16
    counts = torch.randint(1, 600, (n_q,), generator=g, device=dev)
    qi = torch.repeat_interleave(torch.arange(n_q, device=dev), counts)
    bj = torch.randint(0, n_b, (qi.shape[0],), generator=g, device=dev)
    plan = PairPlan.build(qi, bj, n_q, n_b)
    logm = -8 * torch.rand(n_b, generator=g, device=dev)
    reads = [
        [
            torch.randn(*s, generator=g, device=dev).to(bf)
            for s in ((h, f), (n_q, h, kc), (n_b, h, kc), (n_b, h, c))
        ]
        for _ in range(3)
    ]
    basis0 = torch.randn(qi.shape[0], f, generator=g, device=dev).to(bf)
    _plan_source_items(plan)  # the plan keeps its by-source work items for every block
    base = torch.cuda.memory_allocated()
    grads = []
    for shared in (True, False):
        basis = basis0.clone().requires_grad_(True)
        sink = basis_sink(basis) if shared else None
        out = sum(
            fused_kernel_read(basis, *r[:1], logm, *r[1:], plan, 0.5, sink=sink)
            for r in reads
        )
        out.square().sum().backward()
        grads.append(basis.grad)
        del basis, sink, out
    assert torch.equal(grads[0], grads[1])
    del grads
    assert torch.cuda.memory_allocated() == base


@pytest.mark.parametrize("dim", [3, 2])
def test_tree_read_pairs_basis_from_the_plan_is_exact_for_a_batch(
    dim, caplog, monkeypatch
):
    """Two cases in one batch: the basis built from the plan (the fused path's) has the
    reference's plan, values and basis-MLP gradients, bitwise, so each case's point,
    source and pair offsets are right. The fallback to the reference basis (here, on
    CPU) is logged once."""
    monkeypatch.setattr(tree_read, "_BASIS_FALLBACKS_LOGGED", set())
    caplog.set_level(logging.INFO, logger=tree_read.__name__)
    torch.manual_seed(0)
    cases = [cloud(400 + 60 * i, d=dim, seed=3 + i) for i in range(2)]
    n = min(c[0].shape[0] for c in cases)
    pts = torch.stack([c[0][:n] for c in cases])
    nrm = torch.stack([c[1][:n] for c in cases])
    w = torch.stack([c[2][:n] for c in cases])
    q = torch.stack(
        [cloud(90, d=dim, seed=7 + i, scale=(4.0,) * dim)[0] for i in range(2)]
    )
    g = torch.stack([torch.nn.functional.normalize(c[3], dim=-1)[None] for c in cases])
    basis = _KernelBasis(kernel_geo_width(1, 0)).double()
    kw = dict(
        query_r=q,
        query_n=torch.nn.functional.normalize(q, dim=-1),
        query_wall=torch.rand(
            2, 90, dtype=D, generator=torch.Generator().manual_seed(1)
        )
        + 0.1,
        boundary_r=pts,
        boundary_n=nrm,
        boundary_vecs=None,
        log_frac=torch.log(w / w.sum(1, keepdim=True)),
        g_unit=g,
        theta=0.5,
        eps=1e-12,
    )
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        res = []
        for kernel in ("reference", "fused"):
            basis.zero_grad()
            pairs = tree_read_pairs(basis, kernel=kernel, **kw)
            seed = torch.randn(
                pairs.basis.shape, dtype=D, generator=torch.Generator().manual_seed(2)
            )
            (pairs.basis * seed).sum().backward()
            res.append((pairs, [p.grad.clone() for p in basis.parameters()]))
    finally:
        torch.set_num_threads(threads)
    (ref, g_ref), (fus, g_fus) = res
    assert (
        ref.plan.n_q == 180
        and torch.equal(ref.plan.qi, fus.plan.qi)
        and torch.equal(ref.plan.bj, fus.plan.bj)
    )
    assert (
        len({int(x) for x in ref.plan.qi[ref.plan.qi >= 90]}) > 0
    )  # the second case has pairs
    assert torch.equal(ref.basis, fus.basis)
    assert all(torch.equal(a, b) for a, b in zip(g_ref, g_fus))
    tree_read_pairs(basis, kernel="fused", **kw)
    logged = [r.getMessage() for r in caplog.records]
    assert sum("reference kernel basis" in m for m in logged) == 1, logged


class _SinkProbeRead(torch.autograd.Function):
    """``basis @ w.T``, passing its basis gradient through the fused read's sink as
    the fused read does (a stand-in for its Triton kernels on CPU)."""

    @staticmethod
    def forward(ctx, basis, token, grad, w):
        ctx.save_for_backward(basis, w)
        ctx.grad, ctx.sink_node = grad, token.grad_fn
        return basis @ w.t()

    @staticmethod
    def backward(ctx, dout):
        from physicsnemo.experimental.models.isla.fused_read import _basis_grad_target

        basis, w = ctx.saved_tensors
        acc = _basis_grad_target(ctx, basis)
        if acc is not None:
            acc += dout @ w
        return None, None if acc is None else basis.new_empty(0), None, dout.t() @ basis


def _partial_backward_sequences(read, theta, ws, x, loss_of):
    """theta.grad after each partial-backward sequence, ``read(basis, w)`` the read."""
    out = []
    for seq in ("grad_then_backward", "inputs_w_then_theta"):
        theta.grad = None
        loss = loss_of(read, theta, ws, x)
        if seq == "grad_then_backward":
            torch.autograd.grad(loss, [x], retain_graph=True)
            loss.backward(inputs=[theta])
        else:
            loss.backward(inputs=[ws[0]], retain_graph=True)
            loss.backward(inputs=[theta])
        out.append(theta.grad.clone())
    return out


def test_fused_read_sink_ignores_backward_passes_that_skip_the_basis():
    """A backward pass that runs the reads but not the basis (autograd.grad of another
    input, backward(inputs=[a read weight])) leaves nothing in the sink's sum: the
    next pass's basis gradient equals plain autograd's, with the fused read's sink and
    its backward's choice of target (a stand-in read, CPU)."""
    from physicsnemo.experimental.models.isla.fused_read import basis_sink

    g = torch.Generator().manual_seed(8)
    inv = torch.randn(50, 16, generator=g, dtype=D)
    theta = torch.randn(16, generator=g, dtype=D).requires_grad_(True)
    ws = [
        torch.randn(8, 16, generator=g, dtype=D).requires_grad_(True) for _ in range(3)
    ]
    x = torch.randn(50, 8, generator=g, dtype=D).requires_grad_(True)
    sinks = []

    def loss_of(read, theta, ws, x):
        basis = inv * theta
        sinks.append(basis_sink(basis) if read is not None else None)
        outs = [
            basis @ w.t()
            if read is None
            else _SinkProbeRead.apply(basis.detach(), *sinks[-1], w)
            for w in ws
        ]
        return (sum(outs) * x).square().sum()

    ref = _partial_backward_sequences(None, theta, ws, x, loss_of)
    fused = _partial_backward_sequences(_SinkProbeRead, theta, ws, x, loss_of)
    for a, b in zip(fused, ref):
        assert torch.allclose(a, b, rtol=1e-12, atol=0)
    assert all(s[1].acc is None for s in sinks[2:])


@cuda_only
def test_fused_read_partial_backward_matches_the_reference_read():
    """The same partial-backward sequences through the fused reads (float32) against the
    reference read fed ``basis @ W.T``."""
    from physicsnemo.experimental.models.isla.fused_read import (
        basis_sink,
        fused_kernel_read,
    )
    from physicsnemo.experimental.models.isla.segment_softmax import (
        PairPlan,
        segment_softmax_read,
    )

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(9)
    n_q, n_b, h, kc, c, f = 200, 300, 8, 4, 8, 16
    counts = torch.randint(1, 400, (n_q,), generator=g, device=dev)
    qi = torch.repeat_interleave(torch.arange(n_q, device=dev), counts)
    bj = torch.randint(0, n_b, (qi.shape[0],), generator=g, device=dev)
    plan = PairPlan.build(qi, bj, n_q, n_b)
    logm = -8 * torch.rand(n_b, generator=g, device=dev)
    inv = torch.randn(qi.shape[0], f, generator=g, device=dev)
    theta = torch.randn(f, generator=g, device=dev).requires_grad_(True)
    ws = [
        (0.3 * torch.randn(h, f, generator=g, device=dev)).requires_grad_(True)
        for _ in range(3)
    ]
    qkv = [
        torch.randn(*s, generator=g, device=dev)
        for s in ((n_q, h, kc), (n_b, h, kc), (n_b, h, c))
    ]
    x = torch.randn(n_q, h, c, generator=g, device=dev).requires_grad_(True)

    def loss_of(fused, theta, ws, x):
        basis = inv * theta
        sink = basis_sink(basis) if fused else None
        if fused:
            outs = [
                fused_kernel_read(basis, w, logm, *qkv, plan, 0.5, sink=sink)
                for w in ws
            ]
        else:
            outs = [
                segment_softmax_read(basis @ w.t(), logm[plan.bj], *qkv, plan, 0.5)
                for w in ws
            ]
        return (sum(outs) * x).square().sum()

    ref = _partial_backward_sequences(False, theta, ws, x, loss_of)
    fused = _partial_backward_sequences(True, theta, ws, x, loss_of)
    for a, b in zip(fused, ref):
        assert (a - b).abs().max() <= 1e-4 * b.abs().max(), (
            (a - b).abs().max(),
            b.abs().max(),
        )


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_fused_node_means_match_the_reference(dtype):
    """The fused node means against ``node_means`` on a forest of a sphere: the
    backward bitwise (the same float32 additions, one per node), the forward to
    float32 roundoff before the cast (the reference sums each leaf's cells by atomics,
    in no fixed order)."""
    from physicsnemo.experimental.models.isla.fused_read import fused_node_means

    mesh, _, _ = _sphere_case(n=3000, nq=10)
    dev = torch.device("cuda")
    pts = mesh.cell_centroids.float().to(dev)
    nrm = mesh.cell_normals.float().to(dev)
    w = mesh.cell_data[EFFECTIVE_MEASURE_KEY].float().to(dev)
    omega = w / w.sum()
    forest = BoundaryForest.build(pts, omega, nrm, None, theta=1.0)
    g = torch.Generator(device=dev).manual_seed(6)
    x0 = torch.randn(pts.shape[0], 96, generator=g, device=dev).to(dtype)
    gout = torch.randn(forest.n_nodes, 96, generator=g, device=dev).to(dtype)
    res = []
    for fn in (tree_read.node_means, fused_node_means):
        x = x0.clone().requires_grad_(True)
        y = fn(x, forest, omega)
        y.backward(gout)
        res.append((y.detach().float(), x.grad))
    tol = 1e-6 if dtype == torch.float32 else 8e-3
    assert torch.allclose(
        res[1][0], res[0][0], rtol=tol, atol=tol * res[0][0].abs().max()
    )
    assert torch.equal(res[1][1], res[0][1])


def _chunked_step(m, kw, loss_of, size):
    """One step through :meth:`ISLA.query_chunks` in chunks of ``size`` points: the
    chunks' outputs, concatenated."""
    chunks = m.query_chunks(**kw)
    parts = []
    for a in range(0, chunks.n_points, size):
        rows = slice(a, min(a + size, chunks.n_points))
        out = outputs(chunks(rows.start, rows.stop))
        loss_of(out, rows).backward()
        parts.append({k: v.detach() for k, v in out.items()})
    chunks.backward()
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


@pytest.mark.parametrize("vectors", [False, True], ids=["plain", "boundary_vectors"])
@pytest.mark.parametrize("theta", [0.0, TREE_READ_THETA])
def test_query_chunks_equal_the_forward_pass_and_its_gradients(theta, vectors):
    """ISLA.query_chunks (the boundary side once, the points chunk by chunk, then the
    boundary side's backward with the leaves' gradients): in training mode, from the
    same token draw, the outputs of forward and every parameter's gradient of a loss
    summed over the points, to float64 roundoff. The forward pads its token axis, the
    chunks' boundary side pads its own (both unaligned here) and the points run
    unpadded."""
    import physicsnemo.experimental.models.isla.model as isla_model

    extra = {"boundary_schemas": {"wall": {"tr": {"rank": 1}}}} if vectors else {}
    m = _model(3, theta, **extra).double()
    with torch.no_grad():
        for p in m.blocks.parameters():
            p.add_(0.05 * torch.randn_like(p))
    mesh, drv, q = _case()
    assert (mesh.n_cells + q.shape[0]) % isla_model.TOKEN_ALIGNMENT
    assert mesh.n_cells % isla_model.TOKEN_ALIGNMENT
    if vectors:
        g = torch.Generator().manual_seed(4)
        data = dict(mesh.cell_data.items())
        data["tr"] = torch.randn(mesh.n_cells, 3, generator=g, dtype=D)
        mesh = mesh.with_data(cell_data=data)
    kw = {
        "boundary_meshes": {"wall": mesh},
        "prediction_points": points_mesh(q),
        "global_data": {"g": drv},
    }
    target = torch.randn(
        q.shape[0], 4, generator=torch.Generator().manual_seed(9), dtype=D
    )

    def loss_of(out, rows):
        return (out["p"] - target[rows, 0]).square().sum() + (
            out["v"] - target[rows, 1:]
        ).square().sum()

    torch.manual_seed(3)
    full = outputs(m(**kw))
    loss_of(full, slice(None)).backward()
    ref = {n: p.grad.clone() for n, p in m.named_parameters()}
    m.zero_grad(set_to_none=True)
    torch.manual_seed(3)
    chunked = _chunked_step(m, kw, loss_of, 37)
    assert same(chunked, {k: v.detach() for k, v in full.items()}, atol=1e-12)
    ### gradients that vanish analytically (a bias every slice logit shares) are roundoff on both sides
    scale = max(float(g.abs().max()) for g in ref.values())
    for name, p in m.named_parameters():
        assert p.grad is not None, name
        assert torch.allclose(p.grad, ref[name], rtol=1e-9, atol=1e-12 * scale), name


def test_query_chunks_pad_the_boundary_side_to_the_token_alignment():
    """The boundary side of :meth:`ISLA.query_chunks` runs the slice blocks on its
    tokens completed to a multiple of TOKEN_ALIGNMENT, as forward runs its own."""
    import physicsnemo.experimental.models.isla.model as isla_model

    m = _model(3, TREE_READ_THETA).double()
    mesh, drv, q = _case(n=300, nq=40)
    seen = []
    hook = m.blocks[0].register_forward_pre_hook(
        lambda mod, args, kwargs: seen.append(args[0].shape[1]), with_kwargs=True
    )
    try:
        m.query_chunks(
            boundary_meshes={"wall": mesh},
            prediction_points=points_mesh(q),
            global_data={"g": drv},
        )
    finally:
        hook.remove()
    align = isla_model.TOKEN_ALIGNMENT
    assert mesh.n_cells % align
    assert seen == [mesh.n_cells + (-mesh.n_cells) % align]


def test_query_chunks_need_query_independent_points():
    m = ISLA(
        output_schema=PV,
        global_schema=G,
        boundary_schemas={"wall": {}},
        interior=True,
        hidden=16,
        n_layers=1,
    )
    mesh, drv, q = _case(n=50, nq=20)
    with pytest.raises(ValueError, match="query_independent"):
        m.query_chunks(
            boundary_meshes={"wall": mesh}, prediction_points=q, global_data={"g": drv}
        )


def _basis_case(dim, n_vecs, n_pairs=20000, seed=0):
    """One case's geometry for the kernel basis (float64, CPU; two global directions,
    ``n_vecs`` rank-1 inputs per source), a random plan of ``n_pairs`` pairs and a
    basis MLP."""
    g = torch.Generator().manual_seed(seed)
    nq, ns = 300, 500

    def unit(*s):
        return torch.nn.functional.normalize(
            torch.randn(*s, generator=g, dtype=D), dim=-1
        )

    geometry = dict(
        query_r=torch.randn(nq, dim, generator=g, dtype=D),
        query_n=unit(nq, dim),
        query_wall=torch.rand(nq, generator=g, dtype=D) + 0.01,
        src_r=torch.randn(ns, dim, generator=g, dtype=D),
        src_n=unit(ns, dim),
        src_vecs=torch.randn(ns, n_vecs, dim, generator=g, dtype=D) if n_vecs else None,
        g_unit=unit(2, dim),
        eps=1e-12,
    )
    plan = types.SimpleNamespace(
        qi=torch.randint(0, nq, (n_pairs,), generator=g).int(),
        bj=torch.randint(0, ns, (n_pairs,), generator=g).int(),
    )
    return geometry, plan, _KernelBasis(kernel_geo_width(2, n_vecs)).double()


def _to(x, device, dtype):
    if torch.is_tensor(x):
        return x.to(device, dtype) if x.is_floating_point() else x.to(device)
    if isinstance(x, dict):
        return {k: _to(v, device, dtype) for k, v in x.items()}
    if isinstance(x, types.SimpleNamespace):
        return types.SimpleNamespace(**_to(vars(x), device, dtype))
    return x


def _basis_step(basis, plan, geometry, seed, autocast=False, weights=False):
    """The chunked basis of every pair from the plan, and its MLP's gradients for the
    basis gradient ``seed``."""
    basis.zero_grad(set_to_none=True)
    dev = geometry["query_r"].device.type
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=autocast):
        w = tree_read.fused_basis.mlp_weights(basis) if weights else None
        out = tree_read._chunked_basis_from_plan(
            basis, plan, 0, int(plan.qi.shape[0]), 0, 0, geometry, w
        )
    out.backward(seed.to(out.dtype))
    return [out.detach()] + [p.grad.clone() for p in basis.parameters()]


class _MlpProbe(torch.autograd.Function):
    """The basis MLP of one chunk's invariants with given weights, its backward the
    reference's ops (a CPU stand-in for the fused basis kernels)."""

    @staticmethod
    def mlp(inv, w1, b1, w2, b2, w3, b3):
        lin, gelu = torch.nn.functional.linear, torch.nn.functional.gelu
        h = gelu(lin(gelu(lin(inv.to(w1.dtype), w1, b1)), w2, b2))
        return lin(h, w3, b3)

    @staticmethod
    def forward(ctx, inv, *ws):
        ctx.save_for_backward(inv, *ws)
        return _MlpProbe.mlp(inv, *ws)

    @staticmethod
    def backward(ctx, dout):
        inv, *ws = ctx.saved_tensors
        with torch.enable_grad():
            ws = [w.detach().requires_grad_(True) for w in ws]
            grads = torch.autograd.grad(_MlpProbe.mlp(inv, *ws), ws, dout)
        return (None, *grads)


def test_fused_basis_chunks_sum_their_weight_gradients_as_the_reference(monkeypatch):
    """Under bf16 autocast the reference rounds each chunk's weight and bias gradients
    to bfloat16 and sums the chunks' in bfloat16, through autocast's one cached cast
    of each weight. One autograd node per chunk fed one cast per forward pass
    (``mlp_weights``), as the fused basis is, gives the reference's basis and
    gradients bitwise (a stand-in node, CPU)."""

    def stand_in(weights, qi, si, q_off, s_off, geometry):
        with torch.no_grad():
            inv = tree_pair_invariants(qi.long() - q_off, si.long() - s_off, **geometry)
        return _MlpProbe.apply(inv, *weights)

    monkeypatch.setattr(tree_read, "TREE_READ_PAIR_CHUNK", 3000)
    monkeypatch.setattr(tree_read.fused_basis, "fused_kernel_basis", stand_in)
    geometry, plan, basis = _basis_case(3, 1, n_pairs=10000)
    geometry, basis = _to(geometry, "cpu", torch.float32), basis.float()
    seed = 1e-3 * torch.randn(10000, 16, generator=torch.Generator().manual_seed(1))
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        ref = _basis_step(basis, plan, geometry, seed, autocast=True)
        fused = _basis_step(basis, plan, geometry, seed, autocast=True, weights=True)
    finally:
        torch.set_num_threads(threads)
    assert ref[0].dtype == torch.bfloat16
    assert all(torch.equal(a, b) for a, b in zip(fused, ref))


@cuda_only
@pytest.mark.parametrize(
    "gelu_tables", [False, True], ids=["gelu_computed", "gelu_tables"]
)
@pytest.mark.parametrize("dim,n_vecs", [(3, 1), (2, 0)])
@pytest.mark.parametrize("autocast", [False, True], ids=["fp32", "bf16_autocast"])
def test_fused_kernel_basis_matches_the_reference_basis(
    autocast, dim, n_vecs, gelu_tables, monkeypatch
):
    """The fused basis (invariants and MLP in one kernel per chunk) against the
    reference's chunked basis on CUDA, the basis and every MLP gradient for a given
    basis gradient: each as close to the CPU float64 reference as the reference is
    (float32, and bf16 autocast, its backward's GELU computed or from the tables);
    the fused invariants equal ``tree_pair_invariants``'s to float32 roundoff."""
    from physicsnemo.experimental.models.isla import fused_basis

    if gelu_tables:
        monkeypatch.setattr(
            fused_basis, "_computed_gelu_is_pytorchs", lambda device, dtype: False
        )
    monkeypatch.setattr(tree_read, "TREE_READ_PAIR_CHUNK", 4096)
    geometry, plan, basis = _basis_case(dim, n_vecs)
    seed = 1e-3 * torch.randn(20000, 16, generator=torch.Generator().manual_seed(1))
    seed = seed.bfloat16().double()
    exact = _basis_step(basis, plan, geometry, seed)
    dev = torch.device("cuda")
    geo_c, plan_c = _to(geometry, dev, torch.float32), _to(plan, dev, torch.float32)
    basis_c = copy.deepcopy(basis).float().to(dev)
    assert fused_basis.supported(basis_c, geo_c)
    inv = fused_basis.pair_invariants(plan_c.qi, plan_c.bj, 0, 0, geo_c)
    inv_ref = tree_pair_invariants(plan_c.qi.long(), plan_c.bj.long(), **geo_c)
    assert torch.allclose(inv, inv_ref, rtol=1e-6, atol=1e-6)
    seed_c = seed.float().to(dev)
    ref = _basis_step(basis_c, plan_c, geo_c, seed_c, autocast=autocast)
    fused = _basis_step(basis_c, plan_c, geo_c, seed_c, autocast=autocast, weights=True)
    floor = 1e-6 if not autocast else 1e-3
    names = ["basis"] + [n for n, _ in basis.named_parameters()]
    for name, a, b, e in zip(names, fused, ref, exact):
        assert a.dtype == b.dtype, name
        scale = e.abs().max()
        d_fused = (a.double().cpu() - e).abs().max() / scale
        d_ref = (b.double().cpu() - e).abs().max() / scale
        print(f"{name}: fused {d_fused:.2e}, reference {d_ref:.2e}")
        assert d_fused <= 2 * d_ref + floor, name
    if autocast:
        ### The reference's rounding points, which a lost rounding would pass above: the
        ### basis bitwise but where summation order flips a last bit, and the gradients
        ### within a few bf16 ulps (at each tensor's largest value) of the reference's.
        same = (fused[0] == ref[0]).double().mean()
        ulps = [
            float((a - b).abs().max() / 2.0 ** (b.abs().max().log2().floor() - 7))
            for a, b in zip(fused[1:], ref[1:])
        ]
        print(f"basis bitwise {same:.6f}; gradient ulps {ulps}")
        assert same >= 0.99
        assert max(ulps) <= 4, dict(zip(names[1:], ulps))


@cuda_only
def test_fused_basis_leaves_the_geometrys_graph_alone():
    """The fused basis gives its geometry no gradient and keeps no edge into the
    geometry's graph: rank-1 inputs from a learned map (as ISLA's are) survive one
    backward per range of points, as :meth:`ISLA.query_chunks` takes, and get no
    gradient."""
    from physicsnemo.experimental.models.isla import fused_basis

    dev = torch.device("cuda")
    geometry, plan, basis = _basis_case(3, 1)
    geo, plan = _to(geometry, dev, torch.float32), _to(plan, dev, torch.float32)
    basis = basis.float().to(dev)
    raw = geo["src_vecs"].clone().requires_grad_(True)
    geo["src_vecs"] = raw.exp()
    for a, b in ((0, 7000), (7000, 20000)):
        weights = fused_basis.mlp_weights(basis)
        fused_basis.fused_kernel_basis(
            weights, plan.qi[a:b], plan.bj[a:b], 0, 0, geo
        ).sum().backward()
    assert raw.grad is None
    assert all(p.grad is not None for p in basis.parameters())


@cuda_only
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_fused_basis_gelu_tables_are_pytorchs_gelu_on_every_input(dtype):
    """The fused basis's GELU tables give PyTorch's 16-bit GELU and its backward
    bitwise at every input value (the backward for random gradients)."""
    from physicsnemo.experimental.models.isla import fused_basis

    dev = torch.device("cuda")
    f, fg = fused_basis._gelu_tables(dev, dtype)
    f_bwd, g = fg.view(torch.float32).view(-1, 2).unbind(1)
    bits = torch.arange(1 << 16, dtype=torch.int32, device=dev)
    x = bits.to(torch.int16).view(dtype)
    ok = torch.isfinite(x)
    gen = torch.Generator(device=dev).manual_seed(0)
    assert torch.equal(f[ok], torch.nn.functional.gelu(x)[ok])
    assert torch.equal(f_bwd[ok], f.float()[ok])
    for _ in range(8):
        dy = torch.randn(x.shape, generator=gen, device=dev).mul(4).to(dtype)
        ref = torch.ops.aten.gelu_backward(dy, x)
        assert torch.equal((dy.float() * g).to(dtype)[ok], ref[ok])


@cuda_only
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_fused_basis_backward_computes_pytorchs_gelu_on_every_input(dtype):
    """The fused basis backward's computed GELU (CUDA 13's erff, ported) is PyTorch's
    16-bit GELU, and its factor PyTorch's float32 ``gelu_backward(1, x)``, bitwise at
    every finite input on PyTorch's CUDA 13 builds, where the check that chooses it
    over the tables passes; it passes only where they hold."""
    from physicsnemo.experimental.models.isla import fused_basis

    dev = torch.device("cuda")
    x = torch.arange(1 << 16, dtype=torch.int32, device=dev).to(torch.int16).view(dtype)
    out = torch.empty(1 << 16, 2, device=dev)
    fused_basis._gelu_check_kernel[(64,)](
        x, out, 1 << 16, BLOCK=1024, enable_fp_fusion=False
    )
    ok, xf = torch.isfinite(x), x.float()
    gelu = torch.nn.functional.gelu(xf).to(dtype).float()
    factor = torch.ops.aten.gelu_backward(torch.ones_like(xf), xf)
    same = torch.equal(out[ok, 0], gelu[ok]) and torch.equal(out[ok, 1], factor[ok])
    fused_basis._GELU_COMPUTED.clear()
    computed = fused_basis._computed_gelu_is_pytorchs(dev, dtype)
    assert same or not computed
    if int(torch.version.cuda.split(".")[0]) >= 13:
        assert same and computed


@cuda_only
def test_tree_read_pairs_fused_on_cuda_builds_the_basis_in_the_fused_kernel(
    monkeypatch,
):
    """On CUDA ``tree_read_pairs(kernel="fused")`` builds the basis with the fused
    basis kernel: the reference's plan, and its basis to float32 roundoff; so does
    one range of points (``walk_range_pairs``, as :meth:`ISLA.query_chunks` reads)."""
    from physicsnemo.experimental.models.isla import fused_basis

    calls = []
    real = fused_basis.fused_kernel_basis

    def spy(*args):
        calls.append(1)
        return real(*args)

    monkeypatch.setattr(fused_basis, "fused_kernel_basis", spy)
    torch.manual_seed(0)
    dev = torch.device("cuda")
    pts, nrm, w, g = (t.float().to(dev) for t in cloud(500, seed=3))
    q = cloud(150, seed=7, scale=(4.0,) * 3)[0].float().to(dev)
    kw = dict(
        query_r=q[None],
        query_n=torch.nn.functional.normalize(q, dim=-1)[None],
        query_wall=0.1 + torch.rand(1, 150, device=dev),
        boundary_r=pts[None],
        boundary_n=nrm[None],
        boundary_vecs=None,
        log_frac=torch.log(w / w.sum())[None],
        g_unit=torch.nn.functional.normalize(g, dim=-1)[None, None],
        theta=0.5,
        eps=1e-12,
    )
    basis = _KernelBasis(kernel_geo_width(1, 0)).to(dev)
    ref = tree_read_pairs(basis, kernel="reference", **kw)
    assert not calls
    fus = tree_read_pairs(basis, kernel="fused", **kw)
    assert len(calls) == 1
    assert torch.equal(ref.plan.qi.long(), fus.plan.qi.long())
    assert torch.equal(ref.plan.bj.long(), fus.plan.bj.long())
    scale = float(ref.basis.abs().max())
    assert torch.allclose(fus.basis, ref.basis, rtol=1e-5, atol=1e-5 * scale)
    ref = tree_read.walk_range_pairs(
        basis, tree_read.tree_read_walk(kernel="reference", **kw), 40, 110
    )
    fus = tree_read.walk_range_pairs(
        basis, tree_read.tree_read_walk(kernel="fused", **kw), 40, 110
    )
    assert len(calls) == 2
    assert torch.equal(ref.plan.qi.long(), fus.plan.qi.long())
    assert torch.equal(ref.plan.bj.long(), fus.plan.bj.long())
    assert torch.allclose(fus.basis, ref.basis, rtol=1e-5, atol=1e-5 * scale)
