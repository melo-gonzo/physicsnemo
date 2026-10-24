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
    one; under bfloat16 autocast (as in training) the forest's coordinates are
    bfloat16, and the walk rounds as torch does."""
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
    assert f.aabb_min.dtype == (torch.bfloat16 if autocast else dtype)
    assert torch.equal(qi_f, qi) and torch.equal(si_f, si)
    assert n_node_f == n_node > 0
    assert torch.equal(torch.sort(si[qi == 5]).values, torch.arange(f.n_cells))


def test_tree_walk_resolution_and_the_models_walks_agree():
    """``auto`` walks by the reference on CPU and at theta 0, a forced walk is kept and
    an unknown one refused; the model's ``tree_walk`` reaches the walk, and its two
    walks give the same pairs, so the same outputs bitwise."""
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
        tree.tree_walk = walk
        runs[walk] = outputs(_run(tree, mesh, drv, q))
    assert all(
        torch.equal(runs["fast"][k], runs["reference"][k]) for k in runs["reference"]
    )


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
        pairs = tree_read_pairs(*args, **kwargs)
        plans.append(set(zip(pairs.plan.qi.tolist(), pairs.plan.bj.tolist())))
        return pairs

    monkeypatch.setattr(isla_model, "tree_read_pairs", capture)
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
