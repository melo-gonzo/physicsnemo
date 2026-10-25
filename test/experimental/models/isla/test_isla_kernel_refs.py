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

"""ISLA's kernels against their references, where the existing tests leave a gap
(TEST-AUDIT's plan, P3-2 to P3-8).

- The fused geometry region: its bfloat16 backward against the eager region under the
  same autocast, slice counts past each launch branch (257, 513, 1024), and every
  tile size.
- The segment-softmax read: the 2x-error rule in float32, bfloat16 and float16, two
  shapes; a width that is not a power of two (the eager fallback on CUDA); the log
  measure's gradient on both paths.
- The fused read and the fused basis under float16 (``_narrow`` and
  ``unsupported_reason`` accept it).
- The boundary-token draw on CUDA against CPU.
- Paths no test ran (COVERAGE.md): the ball search's second pass, the nearest-cell
  stack's refusal, the forest build's depth-bin fallback, the walk of an empty forest
  or of no points and its first size limit, ``fused_basis.unsupported_reason``'s
  branches, and the geometry kernel's disqualifiers on CUDA.

The 2x-error rule (as in test_isla_tree_read.py): per tensor, the fast path's largest
distance to float64 eager autograd, relative to the float64 tensor's largest entry,
is at most twice the eager path's in the same precision plus a floor at that
precision's roundoff: 1e-6 in float32, 1e-3 in bfloat16 and float16 (whose unit
roundoffs are 2**-8 and 2**-11; the floor stands for the eager path's own error
landing near zero by chance).
"""

import types

import pytest
import torch

from physicsnemo.experimental.models import ISLA

### The modules that own each function compared here (LIB-MOVE retargets these imports).
from physicsnemo.experimental.models.isla import (
    fused_basis,
    fused_read,
    geo_kernel,
    geometry,
    sampling,
    segment_softmax,
    tree_build,
    tree_read,
    tree_walk,
)
from physicsnemo.experimental.models.isla.blocks import _geo_region
from physicsnemo.experimental.models.isla.geometry import exact_boundary_geometry
from physicsnemo.experimental.models.isla.segment_softmax import PairPlan
from physicsnemo.experimental.models.isla.tree_read import BoundaryForest
from test.experimental.models.isla._forced import ran
from test.experimental.models.isla._geometry_cases import (
    _near_wall,
    _sphere,
)
from test.experimental.models.isla._kernel_cases import (
    _basis_case,
    _basis_step,
    _to,
    _two_x,
)
from test.experimental.models.isla._marks import requires_cuda, requires_cuda_triton
from test.experimental.models.isla._meshes import cloud

D = torch.float64
_DEVICES = ["cpu", pytest.param("cuda", marks=[pytest.mark.cuda, requires_cuda])]
_LOW = {torch.float32: 1e-6, torch.bfloat16: 1e-3, torch.float16: 1e-3}


def _rel(a, b) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm())


# --------------------------------------------------------------------------- fused geometry region
def _region_case(b, n, s, k, p, dtype, seed=0):
    """Leaves of one geometry region on CUDA (pre-logits, point and slice geometry,
    ``k`` global directions and ``p`` token vectors, the routing Linear) and the
    weights of a scalar loss over its three outputs."""
    g = torch.Generator().manual_seed(seed)

    def mk(*shape, scale=1.0, unit=False):
        x = torch.randn(*shape, generator=g, dtype=D)
        x = torch.nn.functional.normalize(x, dim=-1) if unit else scale * x
        return x.to("cuda", dtype).requires_grad_()

    leaves = dict(
        logits=mk(b, n, s),
        r=mk(b, n, 3, scale=0.1),
        nh=mk(b, n, 3, unit=True),
        d=mk(b, 1, k, 3, unit=True),
        v=mk(b, n, p, 3, scale=0.7),
        z=mk(b, s, 3, scale=0.1),
        ms=mk(b, s, 3, unit=True),
        weight=mk(1, 5 + k + p, scale=0.3),
        bias=mk(1, scale=0.3),
    )
    wts = [
        torch.randn(b, n, s, generator=g, dtype=D),
        torch.randn(b, n, s, generator=g, dtype=D),
    ]
    wts.append(torch.randn(b, n, 5 + k + p, generator=g, dtype=D))
    return leaves, wts


def _run_region(fused, leaves, wts, k, p, autocast=False, logits_bf16=False):
    """One region's outputs and every leaf's gradient (zeros for an empty leaf)."""
    x = {name: t.detach().clone().requires_grad_() for name, t in leaves.items()}
    lin = torch.nn.Linear(5 + k + p, 1).to(x["r"].device, x["r"].dtype)
    with torch.no_grad():
        lin.weight.copy_(x["weight"])
        lin.bias.copy_(x["bias"])
    lin.weight.requires_grad_(True)
    lin.bias.requires_grad_(True)
    b, n = x["r"].shape[:2]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        logits = x["logits"].bfloat16() if logits_bf16 else x["logits"]
        d_n = x["d"].expand(b, n, k, 3)
        if fused:
            vecs = torch.cat([d_n, x["v"]], 2) if p else d_n
            outs = geo_kernel.fused_geo_region(
                lin, logits, x["r"], x["nh"], vecs, x["z"], x["ms"], 1e-12, n_global=k
            )
        else:
            outs = _geo_region(
                lin,
                logits,
                x["r"],
                x["nh"],
                d_n,
                x["z"],
                x["ms"],
                1e-12,
                x["v"] if p else None,
            )
    sum((o.double() * w.to(o.device)).sum() for o, w in zip(outs, wts)).backward()
    grads = {
        name: t.grad if t.grad is not None else torch.zeros_like(t)
        for name, t in x.items()
        if name not in ("weight", "bias")
    }
    grads["weight"], grads["bias"] = lin.weight.grad, lin.bias.grad
    return [o.detach() for o in outs], grads


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("k, p", [(1, 0), (2, 1)], ids=["K1P0", "K2P1"])
def test_fused_geo_region_bf16_backward_obeys_the_2x_error_rule(k, p):
    """Under bfloat16 autocast (bfloat16 pre-logits, as the model's), the fused
    region's three outputs and the gradients of every leaf are as close to the
    float64 eager region on the same inputs as the eager region under the same
    autocast is (the 2x-error rule, floor 1e-3). The fused backward accumulates in
    float32 where autocast's GEMM backward rounds to bfloat16, so it is the closer."""
    leaves, wts = _region_case(1, 800, 64, k, p, torch.float32)
    exact_leaves = {n: t.detach().double() for n, t in leaves.items()}
    exact_leaves["logits"] = leaves["logits"].detach().bfloat16().double()
    exact = _run_region(False, exact_leaves, wts, k, p)
    eager = _run_region(False, leaves, wts, k, p, autocast=True, logits_bf16=True)
    fused = _run_region(True, leaves, wts, k, p, autocast=True, logits_bf16=True)
    for i, name in enumerate(("bias", "mix", "pooled")):
        _two_x(name, fused[0][i], eager[0][i], exact[0][i], 1e-3)
    for name in exact[1]:
        if p == 0 and name == "v":
            continue
        _two_x(f"grad {name}", fused[1][name], eager[1][name], exact[1][name], 1e-3)


def _assert_region_exact(fused, eager, tol=1e-11):
    """float64: outputs and every gradient at ``tol`` relative in norm (the same
    arithmetic in another summation order)."""
    for a, b in zip(fused[0], eager[0]):
        assert _rel(a, b) < tol, _rel(a, b)
    for name, g in eager[1].items():
        if g.numel() and g.norm() > 0:
            assert _rel(fused[1][name], g) < tol, (name, _rel(fused[1][name], g))


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("s", [257, 513, 1024])
def test_fused_geo_region_matches_eager_past_every_slice_branch(s):
    """Slice counts past the launch branches at 256 and 512 slices, and the most the
    kernel takes (``MAX_SLICES``): forward and backward in float64 at 1e-11 relative."""
    assert s <= geo_kernel.MAX_SLICES
    leaves, wts = _region_case(1, 64, s, 2, 1, D, seed=s)
    _assert_region_exact(
        _run_region(True, leaves, wts, 2, 1), _run_region(False, leaves, wts, 2, 1)
    )


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("block", [1, 2, 4, 8])
def test_fused_geo_region_matches_eager_at_every_tile_size(block, monkeypatch):
    """``FWD_BLOCK_N`` and ``BWD_BLOCK_N`` set (the launch overrides): every tile size
    gives the eager region's outputs and gradients in float64 at 1e-11 relative, on 100
    points (a partial last tile at each size but 1 and 4)."""
    monkeypatch.setattr(geo_kernel, "FWD_BLOCK_N", block)
    monkeypatch.setattr(geo_kernel, "BWD_BLOCK_N", block)
    leaves, wts = _region_case(2, 101, 40, 1, 1, D, seed=block)
    with ran(geo_kernel._fwd_block, geo_kernel._bwd_block) as r:
        fused = _run_region(True, leaves, wts, 1, 1)
    assert r.returns(geo_kernel._fwd_block) == [block] and r.returns(
        geo_kernel._bwd_block
    ) == [block]
    _assert_region_exact(fused, _run_region(False, leaves, wts, 1, 1))


# --------------------------------------------------------------------------- segment softmax
def _ssr_case(h, kc, c, n_q=37, n_b=50, seed=3):
    """A random ragged pair list (segments longer than a source chunk, sources read by
    none, one source read by half the pairs) and float64 leaves."""
    g = torch.Generator().manual_seed(seed)
    counts = torch.randint(0, 5000, (n_q,), generator=g)
    counts[5] = 0
    p = int(counts.sum())
    qi = torch.repeat_interleave(torch.arange(n_q), counts)
    bj = torch.randint(0, n_b - 5, (p,), generator=g)
    bj[: p // 2] = 0
    leaves = [
        torch.randn(p, h, generator=g, dtype=D),
        torch.randn(n_q, h, kc, generator=g, dtype=D),
        torch.randn(n_b, h, kc, generator=g, dtype=D),
        torch.randn(n_b, h, c, generator=g, dtype=D),
    ]
    logm = -torch.rand(p, generator=g, dtype=D) * 8
    seed_ = torch.randn(n_q, h, c, generator=g, dtype=D)
    return (qi, bj, n_q, n_b), leaves, logm, seed_, counts > 0


def _ssr_run(read, plan_args, leaves, logm, seed, device, dtype, logm_grad=False):
    """``read``'s output and the gradients of ker, q, k, v (and of logm with
    ``logm_grad``) for the loss ``(out * seed).sum()``."""
    plan = PairPlan.build(
        plan_args[0].to(device), plan_args[1].to(device), *plan_args[2:]
    )
    ins = [t.detach().to(device, dtype, copy=True).requires_grad_(True) for t in leaves]
    lm = logm.to(device, D if dtype == D else torch.float32).requires_grad_(logm_grad)
    out = read(ins[0], lm, ins[1], ins[2], ins[3], plan, 0.5)
    (out.double() * seed.to(device)).sum().backward()
    return [out.detach()] + [t.grad for t in ins] + ([lm.grad] if logm_grad else [])


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("h, kc, c", [(8, 4, 8), (4, 8, 16)])
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bfloat16, torch.float16],
    ids=["fp32", "bf16", "fp16"],
)
def test_segment_softmax_read_obeys_the_2x_error_rule(dtype, h, kc, c):
    """The Triton segment-softmax read (``_SegmentSoftmaxRead``) against the float64
    eager read on CPU: its output and the gradients of ker, q, k and v each as close
    as the eager read in the same precision on CUDA (the 2x-error rule; the rows of
    points with no pairs left out of the output's)."""
    plan_args, leaves, logm, seed, rows = _ssr_case(h, kc, c)
    with ran(segment_softmax._SegmentSoftmaxRead) as r:
        fast = _ssr_run(
            segment_softmax.segment_softmax_read,
            plan_args,
            leaves,
            logm,
            seed,
            "cuda",
            dtype,
        )
    assert r.count(segment_softmax._SegmentSoftmaxRead) == 1
    eager = _ssr_run(
        segment_softmax._eager_segment_softmax_read,
        plan_args,
        leaves,
        logm,
        seed,
        "cuda",
        dtype,
    )
    exact = _ssr_run(
        segment_softmax._eager_segment_softmax_read,
        plan_args,
        leaves,
        logm,
        seed,
        "cpu",
        D,
    )
    for i, name in enumerate(("out", "ker", "q", "k", "v")):
        a, b, e = fast[i], eager[i], exact[i]
        if name == "out":
            a, b, e = a.cpu()[rows], b.cpu()[rows], e[rows]
        _two_x(name, a, b, e, _LOW[dtype])


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("h, kc, c", [(3, 4, 8), (8, 3, 8), (8, 4, 6)])
def test_segment_softmax_read_of_a_width_not_a_power_of_two_is_the_eager_read(h, kc, c):
    """A head count, key width or value width that is not a power of two takes the
    eager read on CUDA (float32): ``_eager_segment_softmax_read`` runs and the Triton
    Function does not, and the output and every gradient agree with a direct eager
    run's to 1e-4 relative to each tensor's largest entry. (Not bitwise: the eager
    read's ``index_add`` sums up to 5000 pairs per point by float32 atomics on CUDA,
    in no fixed order, so it does not repeat bitwise itself; 1.1e-5 measured in LHR
    job 377613.)"""
    plan_args, leaves, logm, seed, _ = _ssr_case(h, kc, c, n_q=20)
    args = (plan_args, leaves, logm, seed, "cuda", torch.float32)
    eager = segment_softmax._eager_segment_softmax_read
    with ran(segment_softmax._SegmentSoftmaxRead, eager) as r:
        got = _ssr_run(segment_softmax.segment_softmax_read, *args)
    assert r.count(segment_softmax._SegmentSoftmaxRead) == 0 and r.count(eager) == 1
    want = _ssr_run(eager, *args)
    for name, a, b in zip(("out", "ker", "q", "k", "v"), got, want):
        diff = float((a - b).abs().max() / b.abs().max())
        print(f"{name}: against a direct eager run {diff:.2e}")
        assert diff < 1e-4, name


@pytest.mark.cuda
@requires_cuda_triton
def test_both_segment_softmax_reads_differentiate_the_log_measure_alike():
    """``segment_softmax_read`` is differentiable in ``logm`` on both paths: the
    Triton read's gradient of ``logm`` (the kernel term's summed over the heads, in
    float32) obeys the 2x-error rule against the eager read's in float32, beside the
    float64 eager read on CPU, and asking for it leaves the kernel term's and the
    query gradients bitwise (one by-point program per point, no atomics). The model's
    ``logm`` (the sources' log measure fractions) takes no gradient in training."""
    plan_args, leaves, logm, seed, _ = _ssr_case(8, 4, 8, n_q=20)
    args = (plan_args, leaves, logm, seed)
    read = segment_softmax.segment_softmax_read
    with ran(segment_softmax._SegmentSoftmaxRead) as r:
        triton = _ssr_run(read, *args, "cuda", torch.float32, True)
        plain = _ssr_run(read, *args, "cuda", torch.float32)
    assert r.count(segment_softmax._SegmentSoftmaxRead) == 2
    eager = _ssr_run(
        segment_softmax._eager_segment_softmax_read, *args, "cuda", torch.float32, True
    )
    exact = _ssr_run(segment_softmax._eager_segment_softmax_read, *args, "cpu", D, True)
    assert triton[-1] is not None and triton[-1].dtype == torch.float32
    _two_x("logm", triton[-1], eager[-1], exact[-1], _LOW[torch.float32])
    for i, name in ((1, "ker"), (2, "q")):
        assert torch.equal(triton[i], plain[i]), name


@pytest.mark.parametrize("h, kc, c", [(8, 4, 8), (4, 8, 16)])
def test_the_eager_segment_softmax_read_in_bfloat16_rounds_its_gradients_once(h, kc, c):
    """The eager read with bfloat16 ker, q, k and v (the CPU path, and CUDA's where
    the Triton read does not apply) against the float64 read of the same inputs: its
    output and the gradients of ker, q, k and v within one bfloat16 ulp of each
    float64 tensor's largest entry, 2**-7 (they are computed in float32 and rounded
    once to bfloat16, at most 2**-8 of it; measured at most 3.0e-3). Casting after the
    gathers instead added each source's pair gradients up in bfloat16 (BUGS_W1.md B1:
    0.58 for dv here)."""
    plan_args, leaves, logm, seed, _ = _ssr_case(h, kc, c, n_q=20)
    leaves = [t.bfloat16().double() for t in leaves]
    eager = segment_softmax._eager_segment_softmax_read
    low = _ssr_run(eager, plan_args, leaves, logm, seed, "cpu", torch.bfloat16)
    exact = _ssr_run(eager, plan_args, leaves, logm, seed, "cpu", D)
    for name, a, e in zip(("out", "ker", "q", "k", "v"), low, exact):
        err = float((a.double() - e).abs().max() / e.abs().max())
        print(f"{name}: {err:.2e} of max|float64|")
        assert err <= 2**-7, (name, err)


def test_the_fused_read_refuses_a_log_measure_that_requires_a_gradient(monkeypatch):
    """``fused_kernel_read`` is not differentiable in ``logm``: a ``logm`` that
    requires a gradient is refused, naming it, before any kernel runs (so on any
    device), and is passed on to the read where no gradient is recorded."""
    n_q, n_b, h, kc, c, f = 3, 4, 2, 4, 2, 8
    plan = PairPlan.build(torch.tensor([0, 0, 2]), torch.tensor([1, 3, 0]), n_q, n_b)
    basis, w = torch.ones(3, f), torch.ones(h, f)
    q, k, v = torch.ones(n_q, h, kc), torch.ones(n_b, h, kc), torch.ones(n_b, h, c)
    logm = torch.zeros(n_b, requires_grad=True)
    with pytest.raises(ValueError, match="logm"):
        fused_read.fused_kernel_read(basis, w, logm, q, k, v, plan, 0.5)
    monkeypatch.setattr(
        fused_read, "_FusedKernelRead", types.SimpleNamespace(apply=lambda *a: "read")
    )
    with torch.no_grad():
        assert (
            fused_read.fused_kernel_read(basis, w, logm, q, k, v, plan, 0.5) == "read"
        )


# --------------------------------------------------------------------------- float16 narrow paths
@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_fused_kernel_read_in_16_bits_obeys_the_2x_error_rule(dtype):
    """The fused read with float16 operands (``_narrow``: the basis and the weights in
    float16, as under float16 autocast; bfloat16 beside it, as test_isla_tree_read.py
    has it) against the reference read fed ``basis @ W.T`` in the same dtype: output
    and the gradients of basis, w, q, k and v each given, and each as close to the
    float64 eager read as the reference's (floor 1e-3)."""
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
    logm = -torch.rand(n_b, generator=g, dtype=D) * 8
    seed = torch.randn(n_q, h, c, generator=g, dtype=D)

    def run(kind, device, dt):
        plan = PairPlan.build(qi.to(device), bj.to(device), n_q, n_b)
        b, w, q, k, v = ins = [
            t.detach().to(device, dt, copy=True).requires_grad_(True) for t in leaves
        ]
        lm = logm.to(device, torch.float32 if dt != D else D)
        if kind == "fused":
            assert fused_read.supported(b, q, k, v) and fused_read._narrow(b)
            out = fused_read.fused_kernel_read(b, w, lm, q, k, v, plan, 0.5)
        else:
            read = (
                segment_softmax.segment_softmax_read
                if kind == "reference"
                else segment_softmax._eager_segment_softmax_read
            )
            out = read(b @ w.t(), lm[plan.bj.long()], q, k, v, plan, 0.5)
        (out.double() * seed.to(device)).sum().backward()
        return [out.detach()] + [t.grad for t in ins]

    exact = run("eager", "cpu", D)
    fused, ref = run("fused", "cuda", dtype), run("reference", "cuda", dtype)
    names = ("out", "basis", "w", "q", "k", "v")
    missing = [
        f"{kind} {n}"
        for kind, res in (("fused", fused), ("reference", ref))
        for n, t in zip(names, res)
        if t is None
    ]
    assert not missing, missing
    rows = counts > 0
    for i, name in enumerate(("out", "basis", "w", "q", "k", "v")):
        a, b, e = fused[i], ref[i], exact[i]
        if name == "out":
            a, b, e = a.cpu()[rows], b.cpu()[rows], e[rows]
        _two_x(name, a, b, e, 1e-3)


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("dim, n_vecs", [(3, 1), (2, 0)])
def test_fused_kernel_basis_under_float16_autocast_obeys_the_2x_error_rule(
    dim, n_vecs, monkeypatch
):
    """The fused basis under float16 autocast (``unsupported_reason`` accepts it with
    the weight cache on) against the reference's chunked basis under the same
    autocast: the basis and every MLP gradient each as close to the CPU float64
    basis as the reference's (floor 1e-3)."""
    monkeypatch.setattr(tree_read, "TREE_READ_PAIR_CHUNK", 4096)
    geometry_, plan, basis = _basis_case(dim, n_vecs)
    seed = 1e-3 * torch.randn(20000, 16, generator=torch.Generator().manual_seed(1))
    seed = seed.half().double()
    exact = _basis_step(basis, plan, geometry_, seed)
    dev = torch.device("cuda")
    geo_c, plan_c = _to(geometry_, dev, torch.float32), _to(plan, dev, torch.float32)
    basis_c = basis.float().to(dev)

    def step(weights):
        basis_c.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            assert fused_basis.unsupported_reason(basis_c, geo_c) is None
            w = fused_basis.mlp_weights(basis_c) if weights else None
            out = tree_read._chunked_basis_from_plan(
                basis_c, plan_c, 0, int(plan_c.qi.shape[0]), 0, 0, geo_c, w
            )
        out.backward(seed.to(dev, out.dtype))
        return [out.detach()] + [p.grad.clone() for p in basis_c.parameters()]

    with ran(fused_basis._FusedBasis) as r:
        fused = step(True)
    assert r.count(fused_basis._FusedBasis) > 0
    ref = step(False)
    assert fused[0].dtype == ref[0].dtype == torch.float16
    names = ["basis"] + [n for n, _ in basis.named_parameters()]
    for name, a, b, e in zip(names, fused, ref, exact):
        _two_x(name, a, b, e, 1e-3)


# --------------------------------------------------------------------------- the draw
def _draw_case(sizes=(9000, 6000, 3000, 400), seed=0):
    """Four conditions' float64 cell measures and centroids (the last smaller than its
    share, so kept whole)."""
    g = torch.Generator().manual_seed(seed)
    measures = [torch.rand(n, generator=g, dtype=D) ** 3 + 1e-3 for n in sizes]
    centroids = [
        torch.randn(n, 3, generator=g, dtype=D) * torch.tensor([3.0, 2.0, 1.0], dtype=D)
        for n in sizes
    ]
    return measures, centroids


@pytest.mark.cuda
@requires_cuda
def test_the_draw_on_cuda_keeps_the_cpu_draws_cells():
    """``sample_boundary_cells`` on CUDA keeps the CPU draw's cells, bitwise, and
    their Horvitz-Thompson weights agree to 1e-12 relative (float64 sums over the
    cells in another order: a few ulps of the line's length, far below a cell's
    interval, so a hit lands on the same cell but on a boundary)."""
    measures, centroids = _draw_case()
    start = torch.tensor(0.37, dtype=D)
    kept, weights = sampling.sample_boundary_cells(measures, centroids, 6000, start)
    kept_c, weights_c = sampling.sample_boundary_cells(
        [m.cuda() for m in measures], [x.cuda() for x in centroids], 6000, start.cuda()
    )
    assert kept.shape == (6000,) and torch.equal(kept_c.cpu(), kept)
    ### DRAW-EXACT (0.14.0 candidate, int64 sums): tighten to bitwise if its weights are.
    assert torch.allclose(weights_c.cpu(), weights, rtol=1e-12, atol=0)


# --------------------------------------------------------------------------- never-run paths
@pytest.mark.parametrize("kernel", ["reference", "fast"])
@pytest.mark.parametrize("device", _DEVICES)
def test_the_ball_search_second_pass_finds_the_one_pass_candidates(
    device, kernel, monkeypatch
):
    """With ``_BALL_SLOTS`` patched to 2, most queries' balls overflow the first pass
    and are searched again with as many slots as they need (``geometry.py``'s second
    pass): the search's outputs, d and u are bitwise those of the default slots, where
    no query overflows."""
    mesh = _sphere(2)
    q = torch.cat([_near_wall(3, 300, 4, band=0.5), torch.zeros(1, 3, dtype=D)]).to(
        device
    )
    pts, cells = mesh.points.to(device), mesh.cells.to(device)
    search = next(
        c
        for c in geometry._ball_candidates.__code__.co_consts
        if getattr(c, "co_name", "") == "search"
    )

    def run():
        outs = []
        op = {
            "reference": geometry._closest_candidates,
            "fast": geometry._closest_candidates_fast,
        }[kernel]

        def spy(*args):
            outs.append(op(*args))
            return outs[-1]

        with monkeypatch.context() as m:
            m.setattr(
                geometry,
                "_closest_candidates" + ("_fast" if kernel == "fast" else ""),
                spy,
            )
            with ran(search) as r:
                d, u = exact_boundary_geometry(q, pts, cells, kernel=kernel)
        return [d, u, *outs[0]], r.count(search)

    default, n_default = run()
    monkeypatch.setattr(geometry, "_BALL_SLOTS", 2)
    small, n_small = run()
    assert n_small > n_default  # the second pass ran
    for a, b in zip(small, default):
        assert torch.equal(a, b)


def test_a_tree_deeper_than_the_nearest_cell_stack_is_refused(monkeypatch):
    """``_nearest_cells`` refuses a tree as deep as its traversal stack (a midpoint
    tree of F cells is about log2 F deep, so 64 levels is unreachable in practice;
    the stack constant patched down to the tree's depth here)."""
    mesh = _sphere(2)
    s = mesh.points[mesh.cells]
    tree = geometry._cell_tree(s)
    monkeypatch.setattr(geometry, "_NEAREST_STACK", len(tree[3]))
    with pytest.raises(ValueError, match="exceeds the nearest-cell stack"):
        geometry._nearest_cells(mesh.cell_centroids, s, tree)


def _forest_fields(f):
    return [
        f.cells,
        f.left,
        f.right,
        f.range_start,
        f.range_count,
        f.aabb_min,
        f.aabb_max,
        f.center,
        f.measure,
    ]


def test_the_fast_build_leaves_a_forest_too_deep_for_its_depth_bins_to_the_reference(
    monkeypatch,
):
    """``forest_tables`` counts internal nodes by depth in ``_DEPTH_BINS`` bins and
    declines (None) when the depth ``_DEPTH_BINS - 2`` holds any: with the bins
    patched to the forest's internal depth plus 2 it builds the forest, at one fewer it
    declines and the reference builds it; both forests equal the reference build's."""
    pts, nrm, w, _ = cloud(400, d=3, seed=5)
    ref = BoundaryForest.build(
        pts, w, nrm, None, theta=1.0, leaf_size=4, build="reference"
    )
    depth = len(ref.levels)
    for bins, declines in ((depth + 2, False), (depth + 1, True)):
        monkeypatch.setattr(tree_build, "_DEPTH_BINS", bins)
        with ran(tree_build.forest_tables) as r:
            f = BoundaryForest.build(
                pts, w, nrm, None, theta=1.0, leaf_size=4, build="fast"
            )
        assert (r.returns(tree_build.forest_tables)[0] is None) == declines, bins
        assert all(
            torch.equal(a, b) for a, b in zip(_forest_fields(f), _forest_fields(ref))
        )


def test_the_fast_walk_of_no_points_or_no_trees_and_at_its_first_size_limit(
    monkeypatch,
):
    """``walk_pairs``: no points, or a forest of no trees, give empty lists (and zero
    counts with ``count_only``); points times trees (or dimensions) at Warp's element
    limit give None before any launch, counted or not."""
    pts, nrm, w, _ = cloud(300, d=3, seed=1)
    f = BoundaryForest.build(pts, w, nrm, None, theta=1.0)
    q = cloud(40, d=3, seed=2, scale=(3.0,) * 3)[0] @ f.frame
    qi, si, n_node = tree_walk.walk_pairs(f, q[:0], 1.0)
    assert qi.shape == si.shape == (0,) and n_node == 0
    assert torch.equal(
        tree_walk.walk_pairs(f, q[:0], 1.0, count_only=True),
        torch.zeros(0, dtype=torch.long),
    )
    no_trees = types.SimpleNamespace(
        aabb_min=f.aabb_min[:0], node_offset=f.node_offset[:1]
    )
    qi, si, n_node = tree_walk.walk_pairs(no_trees, q, 1.0)
    assert qi.shape == si.shape == (0,) and n_node == 0
    assert torch.equal(
        tree_walk.walk_pairs(no_trees, q, 1.0, count_only=True),
        torch.zeros(40, dtype=torch.long),
    )
    n_t = int(f.node_offset.shape[0]) - 1
    monkeypatch.setattr(tree_walk, "WARP_MAX_ELEMENTS", 40 * max(3, n_t))
    assert tree_walk.walk_pairs(f, q, 1.0) is None
    assert tree_walk.walk_pairs(f, q, 1.0, count_only=True) is None


def _basis_geometry(device, dim=3):
    geometry_, _, basis = _basis_case(dim, 1)
    return _to(geometry_, device, torch.float32), basis.float().to(device)


def test_fused_basis_unsupported_reasons_off_cuda(monkeypatch):
    """``unsupported_reason`` on CPU: CPU float32 geometry is refused, and without
    Triton that is the reason given first."""
    geometry_, basis = _basis_geometry("cpu")
    assert "not CUDA float32" in fused_basis.unsupported_reason(basis, geometry_)
    monkeypatch.setattr(fused_basis, "HAS_TRITON", False)
    assert "Triton" in fused_basis.unsupported_reason(basis, geometry_)


@pytest.mark.cuda
@requires_cuda_triton
def test_fused_basis_unsupported_reasons_on_cuda():
    """``unsupported_reason`` on CUDA float32: None for the supported case; a
    float64 parameter, an autocast without its weight cache or in float32, a 4-D
    geometry and a hidden width that is not a power of two are each refused with
    their reason."""
    geometry_, basis = _basis_geometry("cuda")
    assert fused_basis.unsupported_reason(basis, geometry_) is None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert fused_basis.unsupported_reason(basis, geometry_) is None
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        assert "autocast" in fused_basis.unsupported_reason(basis, geometry_)
    with torch.autocast("cuda", dtype=torch.float32):
        assert "autocast" in fused_basis.unsupported_reason(basis, geometry_)
    ### (Module.double() and .float() convert the basis in place: back to float32 here)
    assert "not CUDA float32" in fused_basis.unsupported_reason(
        basis.double(), geometry_
    )
    basis.float()
    four = {
        k: (
            torch.cat([v, v[:, :1]], 1)
            if k in ("query_r", "query_n", "src_r", "src_n", "g_unit")
            else v
        )
        for k, v in geometry_.items()
    }
    four["src_vecs"] = torch.cat(
        [geometry_["src_vecs"], geometry_["src_vecs"][..., :1]], -1
    )
    assert "two- or three-dimensional" in fused_basis.unsupported_reason(basis, four)
    basis.net[0] = torch.nn.Linear(basis.net[0].in_features, 24).cuda()
    basis.net[2] = torch.nn.Linear(24, basis.net[2].out_features).cuda()
    assert "power of two" in fused_basis.unsupported_reason(basis, geometry_)


_SURFACE = dict(output_schema={"p": {"rank": 0}}, hidden=16, n_layers=1)


def _geo_model(**kw):
    kw = {
        "boundary_schemas": {"wall": {}},
        "global_schema": {"g": {"rank": 1}},
        "n_slices": 8,
        **kw,
    }
    return ISLA(**_SURFACE, **kw)


@pytest.mark.cuda
@requires_cuda_triton
def test_geo_kernel_auto_disqualifiers_on_cuda():
    """On CUDA with Triton, ``geo_kernel="auto"`` picks the fused region for a model
    it fits and the eager region for each disqualifier: no global direction, five
    per-point vectors (five global directions, or two and three rank-1 boundary
    inputs), more slices than ``MAX_SLICES``, a rank-2 boundary input, two spatial
    dimensions, float16 autocast (the fused region's only autocast dtype is bfloat16;
    the choice is re-made when the autocast dtype changes)."""
    pts = torch.zeros(4, 3, device="cuda")
    m = _geo_model()
    assert m.resolve_geo_kernel(pts) == "fused"
    for dtype, want in (
        (torch.float16, "eager"),
        (torch.bfloat16, "fused"),
        (torch.float16, "eager"),
    ):
        with torch.autocast("cuda", dtype=dtype):
            assert m.resolve_geo_kernel(pts) == want, dtype
    assert m.resolve_geo_kernel(pts) == "fused"
    five = {f"g{i}": {"rank": 1} for i in range(5)}
    cases = {
        "no global direction": _geo_model(global_schema={"s": {"rank": 0}}),
        "five global directions": _geo_model(global_schema=five),
        "two directions, three boundary vectors": _geo_model(
            global_schema={"a": {"rank": 1}, "b": {"rank": 1}},
            boundary_schemas={"wall": {f"t{i}": {"rank": 1} for i in range(3)}},
        ),
        "slices past MAX_SLICES": _geo_model(n_slices=geo_kernel.MAX_SLICES + 1),
        "two dimensions": _geo_model(n_spatial_dims=2),
        "rank-2 boundary input": _geo_model(
            boundary_schemas={"wall": {"tau": {"rank": 2, "symmetric": True}}}
        ),
    }
    m = cases["no global direction"]
    assert m.n_global_vectors == 0
    assert cases["five global directions"].n_global_vectors == 5
    m = cases["two directions, three boundary vectors"]
    assert m.n_global_vectors + m.n_block_vecs == 5
    m = cases["rank-2 boundary input"]
    ### (symmetric: its two derived vectors keep K + P at 3, so the rank-2 input alone
    ### disqualifies; a general one adds four and passes MAX_DIRECTIONS too)
    assert m.n_block_tens == 1
    assert m.n_global_vectors + m.n_block_vecs <= geo_kernel.MAX_DIRECTIONS
    for name, m in cases.items():
        assert m.geo_kernel == "auto" and m.resolve_geo_kernel(pts) == "eager", name
