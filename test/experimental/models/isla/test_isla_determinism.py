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

"""Run-to-run determinism of ISLA's eager paths, as they are (TEST-AUDIT's
DETERMINISM.md, measured in LHR job 377000; the supervisor's decision 5).

No test sets a global deterministic mode (``torch.use_deterministic_algorithms``):
that would hide what a user meets by default. Each test runs a path twice (or five
times) in one process on the same inputs and states what repeats:

- bitwise: the fast exact geometry (d, u, both gradients), the fused geometry
  region, the fused node means and the fused read with their backward passes, the
  segment-softmax read's forward and its by-point gradients, the whole model (the
  surface task and ISLA-DTT with its draw, in 3-D and 2-D, in float32 and under
  bfloat16 autocast, forward and backward) on CPU with one thread and on CUDA with
  kernels auto;
- to a stated tolerance, where the code sums in an order CUDA does not fix: the
  draw's float64 Horvitz-Thompson weights (CUDA's floating-point ``cumsum``,
  ulps), the segment-softmax read's key and value gradients (atomic adds), and the
  reference (unfused) tree read on CUDA (``node_means``' ``index_add``).

When a source of order is made exact (the draw in int64 fixed point,
DETERMINISM.md (a)), its test here tightens to bitwise.
"""

import pytest
import torch

from physicsnemo.experimental.models import ISLA

### The modules that own each function run here (LIB-MOVE retargets these imports).
from physicsnemo.experimental.models.isla import (
    fused_read,
    geo_kernel,
    sampling,
    segment_softmax,
    tree_build,
    tree_read,
    tree_walk,
)
from physicsnemo.experimental.models.isla.geometry import exact_boundary_geometry
from physicsnemo.experimental.models.isla.segment_softmax import PairPlan
from physicsnemo.experimental.models.isla.tree_read import (
    BoundaryForest,
    _KernelBasis,
    _TreeKernelRead,
    kernel_geo_width,
)
from physicsnemo.mesh.sampling.systematic_sampling import _cumsum
from test.experimental.models.isla._forced import ran
from test.experimental.models.isla._geometry_cases import (
    _near_wall,
    _sphere,
)
from test.experimental.models.isla._marks import requires_cuda, requires_cuda_triton
from test.experimental.models.isla._meshes import (
    cloud,
    live,
    outputs,
    points_mesh,
    soup,
)

D = torch.float64


def _bytes(t):
    return t.detach().contiguous().reshape(-1).view(torch.uint8)


def _bitwise(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return (
        a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bytes(a), _bytes(b))
    )


def _rel_norm(a, b) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def _rel_max(a, b) -> float:
    a, b = a.double(), b.double()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


# --------------------------------------------------------------------------- the draw
def _draw_case(n=300_000, seed=0):
    """About 300k float64 cells over 4 conditions (one smaller than its share),
    measures spread over three decades, anisotropic centroids."""
    sizes = (150_000, 90_000, 57_000, 3_000)
    g = torch.Generator().manual_seed(seed)
    measures = [(torch.rand(m, generator=g, dtype=D) * 3).exp() * 1e-3 for m in sizes]
    scale = torch.tensor([4.0, 2.0, 1.0], dtype=D)
    centroids = [torch.randn(m, 3, generator=g, dtype=D) * scale for m in sizes]
    return [m.cuda() for m in measures], [x.cuda() for x in centroids]


BUDGET = 100_000


@pytest.mark.cuda
@requires_cuda
def test_the_draw_twice_on_cuda_keeps_the_same_cells():
    """The draw run twice on CUDA keeps the same cells bitwise; its float64
    Horvitz-Thompson weights agree to 1e-15 relative but not always bitwise (CUDA's
    float64 cumsum sums in no fixed order: a few ulps, 2.2e-19 absolute in job 377000).
    Tightens to bitwise weights once the draw's sums are exact."""
    measures, centroids = _draw_case()
    start = torch.tensor(0.4137, dtype=D, device="cuda")
    runs = [
        sampling.sample_boundary_cells(measures, centroids, BUDGET, start)
        for _ in range(2)
    ]
    (kept0, w0), (kept1, w1) = runs
    assert kept0.shape == (BUDGET,) and torch.equal(kept0, kept1)
    spread = float(((w1 - w0).abs() / w0.abs()).max())
    print(f"the draw's weights, run-to-run: {spread:.2e} relative (bound 1e-15)")
    ### DRAW-EXACT (0.14.0 candidate, int64 sums): make this torch.equal(w1, w0).
    assert torch.allclose(w1, w0, rtol=1e-15, atol=0)


@pytest.mark.cuda
@requires_cuda
def test_hits_planted_on_line_boundaries_keep_one_of_the_two_cells_there():
    """Hits planted on cell boundaries of the systematic line (the start chosen so
    that ``start + j`` is a boundary ``line[i]`` as the CPU draw computes it): every
    CUDA run, and the CPU run, keeps the same cells except, at most, which of the two
    cells beside the planted boundary (found by moving the CPU's start a few ulps
    either way) takes that hit (the line's ulp-level wobble
    decides it on CUDA today: DETERMINISM.md (a)). Once the draw's sums are exact this
    tightens to: every run keeps the CPU's cells bitwise."""
    measures, centroids = _draw_case(seed=1)
    cpu_m, cpu_x = [m.cpu() for m in measures], [x.cpu() for x in centroids]
    with ran(_cumsum) as r:
        sampling.sample_boundary_cells(cpu_m, cpu_x, BUDGET, torch.tensor(0.5, dtype=D))
    line = r.returns(_cumsum)[-1]  # the draw's last scan is the line
    assert line.shape[0] == sum(m.shape[0] for m in cpu_m)
    for i in torch.linspace(1000, line.shape[0] - 1000, 6).long().tolist():
        j = torch.floor(line[i])
        start = line[i] - j
        ### (the hit start + j lands on line[i] to within two ulps, and is one of the k hits)
        assert (start + j - line[i]).abs() <= 2 * torch.finfo(D).eps * line[
            i
        ] and j < BUDGET
        kept_cpu, _ = sampling.sample_boundary_cells(cpu_m, cpu_x, BUDGET, start)
        ### The two cells beside the planted boundary: those the CPU draw swaps when
        ### the start moves 8 ulps of line[i] either way (no other hit is that close
        ### to a boundary).
        delta = 8 * torch.finfo(D).eps * line[i]
        lo, hi = (
            sampling.sample_boundary_cells(cpu_m, cpu_x, BUDGET, start + e)[0]
            for e in (-delta, delta)
        )
        beside = torch.cat([lo[~torch.isin(lo, hi)], hi[~torch.isin(hi, lo)]])
        assert beside.numel() == 2, (i, beside)
        for _ in range(3):
            kept, _ = sampling.sample_boundary_cells(
                measures, centroids, BUDGET, start.cuda()
            )
            kept = kept.cpu()
            assert kept.shape == kept_cpu.shape
            differ = torch.cat(
                [
                    kept[~torch.isin(kept, kept_cpu)],
                    kept_cpu[~torch.isin(kept_cpu, kept)],
                ]
            )
            ### DRAW-EXACT (0.14.0 candidate): make this torch.equal(kept, kept_cpu).
            assert differ.numel() in (0, 2), (i, differ)
            assert bool(torch.isin(differ, beside).all()), (i, differ, beside)


# --------------------------------------------------------------------------- module level
@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_the_fast_geometry_repeats_on_cuda(dtype):
    """``exact_boundary_geometry(kernel="fast")`` twice on CUDA: d, u and the query
    and vertex gradients bitwise (the gather's backward is CUDA's sort-based
    ``index_put_``, which sums in a fixed order)."""
    mesh = _sphere(3)
    q0 = _near_wall(3, 20000, 2, band=0.5).to("cuda", dtype)
    p0, cells = mesh.points.to("cuda", dtype), mesh.cells.cuda()
    g = torch.Generator().manual_seed(0)
    probe = torch.randn(q0.shape, generator=g, dtype=D).to("cuda", dtype)

    def run():
        q, p = q0.clone().requires_grad_(), p0.clone().requires_grad_()
        d, u = exact_boundary_geometry(q, p, cells, kernel="fast")
        gq, gp = torch.autograd.grad(d.sum() + (u * probe).sum(), (q, p))
        return d, u, gq, gp

    a, b = run(), run()
    for name, x, y in zip(("d", "u", "dq", "dp"), a, b):
        if not _bitwise(x, y):
            print(f"{name}: run-to-run {_rel_max(x, y):.2e}")
        assert _bitwise(x, y), name


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("autocast", [False, True], ids=["fp32", "bf16_autocast"])
def test_the_fused_geo_region_repeats_on_cuda(autocast):
    """The fused geometry region twice on CUDA, forward and every input gradient:
    bitwise (its per-slice reductions are per-tile partial sums reduced by torch, no
    atomics)."""
    g = torch.Generator().manual_seed(0)
    b, n, s = 2, 3000, 64
    x0 = [
        torch.randn(b, n, s, generator=g),
        torch.randn(b, n, 3, generator=g) * 0.1,
        torch.nn.functional.normalize(torch.randn(b, n, 3, generator=g), dim=-1),
        torch.nn.functional.normalize(torch.randn(b, n, 2, 3, generator=g), dim=-1),
        torch.randn(b, s, 3, generator=g) * 0.1,
        torch.nn.functional.normalize(torch.randn(b, s, 3, generator=g), dim=-1),
    ]
    wts = [
        torch.randn(b, n, s, generator=g).cuda(),
        torch.randn(b, n, s, generator=g).cuda(),
        torch.randn(b, n, 7, generator=g).cuda(),
    ]
    torch.manual_seed(0)
    lin0 = torch.nn.Linear(7, 1).cuda()

    def run():
        x = [t.cuda().requires_grad_() for t in x0]
        lin = torch.nn.Linear(7, 1).cuda()
        lin.load_state_dict(lin0.state_dict())
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            logits = x[0].bfloat16() if autocast else x[0]
            outs = geo_kernel.fused_geo_region(lin, logits, *x[1:], 1e-12)
        sum((o.float() * w).sum() for o, w in zip(outs, wts)).backward()
        return (
            [o.detach() for o in outs]
            + [t.grad for t in x]
            + [lin.weight.grad, lin.bias.grad]
        )

    for i, (x, y) in enumerate(zip(run(), run())):
        assert _bitwise(x, y), i


def _read_case(dtype, seed=0):
    """A forest over 4000 cells on CUDA, a random plan of the read's shape, and the
    leaves of one fused read with node means: the cells' keys and values ``(N, 96)``,
    the basis, the kernel weights and the points' queries."""
    g = torch.Generator().manual_seed(seed)
    pts, nrm, w, _ = cloud(4000, seed=seed)
    dev = torch.device("cuda")
    omega = (w / w.sum()).to(dev)
    forest = BoundaryForest.build(pts.to(dev), omega, nrm.to(dev), None, theta=1.0)
    n_src = 4000 + forest.n_nodes
    n_q, h, kc, c, f = 400, 8, 4, 8, 16
    counts = torch.randint(1, 300, (n_q,), generator=g)
    qi = torch.repeat_interleave(torch.arange(n_q), counts).to(dev)
    bj = torch.randint(0, n_src, (qi.shape[0],), generator=g).to(dev)
    plan = PairPlan.build(qi, bj, n_q, n_src)
    leaves = [
        torch.randn(4000, h * (kc + c), generator=g),
        torch.randn(qi.shape[0], f, generator=g),
        0.3 * torch.randn(h, f, generator=g),
        torch.randn(n_q, h, kc, generator=g),
    ]
    logm = -8 * torch.rand(n_src, generator=g)
    seed_ = torch.randn(n_q, h, c, generator=g)
    return (
        forest,
        omega,
        plan,
        [t.to(dev, dtype) for t in leaves],
        logm.to(dev),
        seed_.to(dev),
        (h, kc, c),
    )


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_the_fused_read_with_its_node_means_repeats_on_cuda(dtype):
    """The fused node means of the cells' keys and values, then the fused read over
    the cells and the nodes, twice on CUDA: the output and the gradients of the
    cells' keys and values, the basis, the kernel weights and the queries, bitwise
    (fixed-order Triton sums, no atomics)."""
    forest, omega, plan, leaves, logm, seed, (h, kc, c) = _read_case(dtype)

    def run():
        kv, basis, w, q = [t.clone().requires_grad_() for t in leaves]
        assert fused_read.node_means_supported(kv)
        src = torch.cat(
            [kv, fused_read.fused_node_means(kv, forest, omega.to(kv.dtype))]
        )
        k, v = src[:, : h * kc].reshape(-1, h, kc), src[:, h * kc :].reshape(-1, h, c)
        assert fused_read.supported(basis, q, k, v)
        out = fused_read.fused_kernel_read(basis, w, logm, q, k, v, plan, 0.5)
        (out * seed).sum().backward()
        return [out.detach(), kv.grad, basis.grad, w.grad, q.grad]

    for name, x, y in zip(("out", "kv", "basis", "w", "q"), run(), run()):
        assert _bitwise(x, y), name


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_the_segment_softmax_read_repeats_on_cuda_but_for_its_atomic_sums(dtype):
    """The Triton segment-softmax read twice on CUDA: the output and the by-point
    gradients (ker, q) bitwise; the keys' and values' gradients come from atomic adds
    in the by-source kernel, whose order CUDA does not fix: to 1e-5 relative to the
    tensor's largest entry (float32 sums of a few thousand pairs per source)."""
    g = torch.Generator().manual_seed(3)
    n_q, n_b, h, kc, c = 300, 400, 8, 4, 8
    counts = torch.randint(1, 3000, (n_q,), generator=g)
    qi = torch.repeat_interleave(torch.arange(n_q), counts)
    bj = torch.randint(0, n_b, (qi.shape[0],), generator=g)
    bj[::3] = 7  # one source read by a third of the pairs
    plan = PairPlan.build(qi.cuda(), bj.cuda(), n_q, n_b)
    leaves = [
        torch.randn(qi.shape[0], h, generator=g),
        torch.randn(n_q, h, kc, generator=g),
        torch.randn(n_b, h, kc, generator=g),
        torch.randn(n_b, h, c, generator=g),
    ]
    logm = (-8 * torch.rand(qi.shape[0], generator=g)).cuda()
    seed = torch.randn(n_q, h, c, generator=g).cuda()

    def run():
        ker, q, k, v = ins = [t.to("cuda", dtype).requires_grad_() for t in leaves]
        with ran(segment_softmax._SegmentSoftmaxRead) as r:
            out = segment_softmax.segment_softmax_read(ker, logm, q, k, v, plan, 0.5)
        assert r.count(segment_softmax._SegmentSoftmaxRead) == 1
        (out * seed).sum().backward()
        return [out.detach()] + [t.grad for t in ins]

    a, b = run(), run()
    for name, x, y in zip(("out", "ker", "q"), (a[0], a[1], a[2]), (b[0], b[1], b[2])):
        assert _bitwise(x, y), name
    for name, x, y in zip(("k", "v"), a[3:], b[3:]):
        print(f"{name}: run-to-run {_rel_max(x, y):.2e}")
        assert _rel_max(x, y) < 1e-5, name


@pytest.mark.cuda
@requires_cuda
def test_the_reference_tree_read_on_cuda_spreads_within_its_bound():
    """The reference (unfused) tree read on CUDA in float32, five runs of one block's
    read (node means by ``index_add``, whose atomic order CUDA does not fix, then the
    segment-softmax read), each tensor against the first run's, relative in norm: the
    output within 1e-5 (the bound of DETERMINISM.md; 7e-8 measured in LHR job 377654,
    2e-6 at the model's first read in job 377000), every parameter gradient within
    1e-4 (1.4e-5 measured at ``kernel.weight``, a sum over every pair whose terms
    nearly cancel, so the reads' run-to-run change is amplified there)."""
    torch.manual_seed(0)
    dev = torch.device("cuda")
    pts, nrm, w, g = (t.float().to(dev) for t in cloud(3000, seed=3))
    q = cloud(500, seed=7, scale=(4.0,) * 3)[0].float().to(dev)
    kw = dict(
        query_r=q[None],
        query_n=torch.nn.functional.normalize(q, dim=-1)[None],
        query_wall=0.1 + torch.rand(1, 500, device=dev),
        boundary_r=pts[None],
        boundary_n=nrm[None],
        boundary_vecs=None,
        log_frac=torch.log(w / w.sum())[None],
        g_unit=torch.nn.functional.normalize(g, dim=-1)[None, None],
        theta=0.5,
        eps=1e-12,
    )
    basis = _KernelBasis(kernel_geo_width(1, 0)).to(dev)
    with torch.no_grad():
        pairs = tree_read.tree_read_pairs(basis, kernel="reference", **kw)
    assert pairs.kernel == "reference" and pairs.n_pairs_node > 0
    read = _TreeKernelRead(64).to(dev)
    with torch.no_grad():
        for p in read.parameters():
            p.normal_(0.0, 0.3)
    gen = torch.Generator().manual_seed(1)
    h_b = torch.randn(3000, 64, generator=gen).to(dev)
    h_q = torch.randn(500, 64, generator=gen).to(dev)
    seed = torch.randn(500, 64, generator=gen).to(dev)

    def run():
        read.zero_grad(set_to_none=True)
        with ran(tree_read.node_means, fused_read.fused_node_means) as r:
            out = read(h_b, h_q, pairs)
        assert (
            r.count(tree_read.node_means) == 1
            and r.count(fused_read.fused_node_means) == 0
        )
        (out * seed).sum().backward()
        return [out.detach()] + [p.grad.clone() for p in read.parameters()]

    runs = [run() for _ in range(5)]
    names = ["out"] + [n for n, _ in read.named_parameters()]
    spread = [
        max(_rel_norm(r[i], runs[0][i]) for r in runs[1:]) for i in range(len(names))
    ]
    print(
        "reference tree read, five-run spread in norm:",
        {n: f"{x:.1e}" for n, x in zip(names, spread)},
    )
    assert spread[0] < 1e-5
    assert max(spread[1:]) < 1e-4, dict(zip(names[1:], spread[1:]))


# --------------------------------------------------------------------------- the whole model
#: The whole-model cases: the surface task and ISLA-DTT (its read live, its draw
#: keeping 300 of 400 cells), each in 3-D and in 2-D.
TASKS = ["surface", "dtt"]
DIMS = [3, 2]
PRECISIONS = [
    pytest.param(False, id="fp32"),
    pytest.param(True, id="bf16_autocast"),
]


def _model(task, dims):
    torch.manual_seed(0)
    kw = dict(
        output_schema={"p": {"rank": 0}, "v": {"rank": 1}},
        boundary_schemas={"wall": {}},
        global_schema={"g": {"rank": 1}},
        n_spatial_dims=dims,
        hidden=32,
        n_layers=2,
        n_slices=16,
    )
    if task == "dtt":
        kw.update(interior=True, n_boundary_tokens=300)
    return live(ISLA(**kw))


def _step(model, task, device, autocast, n=400, nq=120):
    """One training step of ``model`` on a boundary soup of ``n`` cells (and, for
    ISLA-DTT, ``nq`` prediction points), the global RNG seeded first (the draw's
    start in training): the outputs, every parameter's gradient and the draw's kept
    cells and weights, by name."""
    dims = model.n_spatial_dims
    pts, nrm, w, drv = cloud(n, d=dims, seed=0, dtype=torch.float32)
    mesh = soup(pts, nrm, w).to(device)
    if task == "dtt":
        q = cloud(nq, d=dims, seed=1, dtype=torch.float32, scale=(3.0,) * dims)[0]
        where = {"prediction_points": points_mesh(q.to(device))}
    else:
        where = {"prediction_boundaries": ["wall"]}
    model.zero_grad(set_to_none=True)
    torch.manual_seed(1)
    with (
        ran(sampling.sample_boundary_cells) as r,
        torch.autocast(device, dtype=torch.bfloat16, enabled=autocast),
    ):
        out = outputs(
            model(
                boundary_meshes={"wall": mesh},
                global_data={"g": drv.to(device)},
                **where,
            )
        )
        sum(o.double().square().mean() for o in out.values()).backward()
    draws = r.returns(sampling.sample_boundary_cells)
    assert len(draws) == (task == "dtt")
    draw = {}
    if draws:
        draw = dict(zip(("draw.kept", "draw.weights"), draws[0]))
        assert draw["draw.kept"].shape == (300,)  # the draw is active: 300 of 400 cells
    return {
        **{k: v.detach() for k, v in out.items()},
        **{n: p.grad for n, p in model.named_parameters()},
        **draw,
    }


def _not_bitwise(a, b) -> list:
    assert a.keys() == b.keys()
    bad = [k for k in a if not _bitwise(a[k], b[k])]
    for k in bad:
        print(f"{k}: run-to-run {_rel_max(a[k], b[k]):.2e}")
    return bad


@pytest.mark.parametrize("autocast", PRECISIONS)
@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("task", TASKS)
def test_the_model_repeats_bitwise_on_cpu_with_one_thread(task, dims, autocast):
    """The surface model and ISLA-DTT (read live, draw active), in 3-D and 2-D, in
    float32 and under bfloat16 autocast: one training step twice on CPU with one
    thread, the outputs, every parameter gradient and the draw bitwise (the baseline
    the cleanup's A/B harness assumes)."""
    model = _model(task, dims)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        a, b = (_step(model, task, "cpu", autocast) for _ in range(2))
    finally:
        torch.set_num_threads(threads)
    assert not _not_bitwise(a, b)


def _resolved(model, task, device):
    """The kernels "auto" resolves to for ``model`` on ``device`` (float32 points):
    the geometry region's, and for ISLA-DTT the exact geometry's and the read's."""
    p = torch.zeros(1, model.n_spatial_dims, device=device)
    kernels = {"geo": model.resolve_geo_kernel(p)}
    if task == "dtt":
        kernels["geometry"] = model.resolve_geometry_kernel(p)
        kernels["tree_read"] = model.resolve_tree_read_kernel(p)
    return kernels


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("autocast", PRECISIONS)
@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("task", TASKS)
def test_the_model_repeats_bitwise_on_cuda_with_kernels_auto(task, dims, autocast):
    """The surface model and ISLA-DTT (read live, draw active), in 3-D and 2-D, in
    float32 and under bfloat16 autocast, on CUDA with every kernel "auto": one
    training step twice, the outputs, every parameter gradient and the draw bitwise.
    The kernels that resolve are recorded and asserted: the fused geometry region in
    3-D and 2-D, and for ISLA-DTT the fast
    exact geometry, the fused read (seen running: the fused read and node means, not
    the segment softmax read), the fast forest build and the fast walk.

    DRAW-EXACT: ISLA-DTT's draw (its float64 weights included) is bitwise here at 400
    cells today; the 0.14.0 draw makes it so at any size, and this test stays."""
    model = _model(task, dims).cuda()
    reads = (
        fused_read.fused_kernel_read,
        fused_read.fused_node_means,
        segment_softmax.segment_softmax_read,
    )
    with ran(tree_build.forest_tables, tree_walk.walk_pairs, *reads) as r:
        a, b = (_step(model, task, "cuda", autocast) for _ in range(2))
    kernels = _resolved(model, task, "cuda")
    if task == "dtt":
        tables, walks = (
            r.returns(tree_build.forest_tables),
            r.returns(tree_walk.walk_pairs),
        )
        kernels["build"] = "fast" if tables and None not in tables else "reference"
        kernels["walk"] = "fast" if walks and None not in walks else "reference"
        kernels["read_ran"] = {f.__name__ for f in reads if r.count(f)}
    print("resolved:", kernels)
    want = {"geo": "fused"}
    if task == "dtt":
        want.update(geometry="fast", tree_read="fused", build="fast", walk="fast")
        want["read_ran"] = {"fused_kernel_read", "fused_node_means"}
    assert kernels == want
    assert not _not_bitwise(a, b)
