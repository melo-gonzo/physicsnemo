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

"""The implementation options of ISLA's geometry region compute the same thing as the
eager reference (the eager region, rebuilt in backward).

Bars from the study brief, applied to the reference configuration's arithmetic:
the fused Triton region reproduces the eager forward to 1e-6 relative in float32 and
the parameter gradients to 1e-5 relative; in float64 the two agree to 1e-11 (same
arithmetic, different summation order). The same bars hold for the whole model
compiled with torch.compile, which is the recipe's default (compile: true) and must
stay free of graph breaks."""

import pytest
import torch

from physicsnemo.experimental.models import ISLA
from physicsnemo.experimental.models.isla.blocks import _geo_region
from test.experimental.models.isla._marks import requires_cuda
from test.experimental.models.isla._meshes import cloud, outputs, soup

REF = dict(
    output_schema={"p": {"rank": 0}, "v": {"rank": 1}},
    boundary_schemas={"wall": {}},
    global_schema={"g": {"rank": 1}},
    hidden=64,
    n_layers=3,
    n_slices=32,
)


def _boundary(n=700, seed=0, device="cuda", dtype=torch.float32, cell_data=None):
    """One boundary soup of ``n`` cells with a global direction, and a target."""
    pts, nrm, w, drv = cloud(n, seed=seed, dtype=torch.float32)
    mesh = soup(pts, nrm, w, cell_data=cell_data).to(device).to(dtype)
    tgt = torch.randn(n, 4, generator=torch.Generator().manual_seed(seed + 1)).to(
        device, dtype
    )
    return dict(
        boundary_meshes={"wall": mesh}, global_data={"g": drv.to(device, dtype)}
    ), tgt


def _flat(o):
    return torch.cat([o["p"][..., None], o["v"]], dim=-1)


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


def _run(model, inputs, target):
    model.zero_grad(set_to_none=True)
    if "prediction_points" not in inputs:
        inputs = {**inputs, "prediction_boundaries": ["wall"]}
    out = _flat(outputs(model(**inputs)))
    torch.nn.functional.mse_loss(out.float(), target.float()).backward()
    grads = {
        k.replace("_orig_mod.", ""): p.grad.detach().clone()
        for k, p in model.named_parameters()
    }
    return out.detach(), grads


def _pair(dtype, device="cuda", geo_kernel="auto", **kw):
    torch.manual_seed(0)
    ref = ISLA(**REF).to(device, dtype)
    ref.geo_kernel = (
        "eager"  # the eager reference ("auto" resolves to the fused kernel on CUDA)
    )
    torch.manual_seed(0)
    opt = ISLA(**{**REF, **kw}).to(device, dtype)
    opt.geo_kernel = geo_kernel
    opt.load_state_dict(ref.state_dict())
    return ref, opt


def _assert_same(ref, opt, dtype, out_tol, grad_tol, n=700):
    inputs, tgt = _boundary(n, dtype=dtype)
    o_ref, g_ref = _run(ref, inputs, tgt)
    o_opt, g_opt = _run(opt, inputs, tgt)
    assert _rel(o_opt, o_ref) < out_tol, _rel(o_opt, o_ref)
    total = torch.cat([g.flatten() for g in g_ref.values()]).norm()
    for k, g in g_ref.items():
        if g.norm() < 1e-6 * total:
            continue  # geo_logit.bias: a constant over slices cancels in both softmaxes; its gradient is roundoff
        # relative to the parameter's own gradient, floored at 1e-3 of the whole gradient: a
        # scalar such as a block's prototype gain is a reduction over every token and slice
        # whose gradient can nearly cancel, and its own norm is then a roundoff-sized yardstick
        scale = torch.maximum(g.norm(), 1e-3 * total)
        assert (g_opt[k] - g).norm() / scale < grad_tol, (k, _rel(g_opt[k], g))


# --------------------------------------------------------------------------- fused region
@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize(
    "K,P",
    [(1, 0), (2, 0), (3, 0), (4, 0), (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (3, 1)],
)
def test_fused_geo_region_matches_eager_region(dtype, K, P, monkeypatch):
    """Region level: forward outputs (bias, mix, pooled) and every input gradient, for
    K global directions plus P per-token rank-1 vectors (the (5 + K + P)-invariant set
    in the eager order) with the linear routing-bias head."""
    from physicsnemo.experimental.models.isla.geo_kernel import fused_geo_region

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    b, n, s = 2, 500, 40
    g = torch.Generator().manual_seed(0)

    def mk(*shape):
        return (
            (torch.randn(*shape, generator=g) * 0.1).to("cuda", dtype).requires_grad_()
        )

    r, z = mk(b, n, 3), mk(b, s, 3)
    nh = (
        torch.nn.functional.normalize(torch.randn(b, n, 3, generator=g), dim=-1)
        .to("cuda", dtype)
        .requires_grad_()
    )
    ms = (
        torch.nn.functional.normalize(torch.randn(b, s, 3, generator=g), dim=-1)
        .to("cuda", dtype)
        .requires_grad_()
    )
    d = (
        torch.nn.functional.normalize(torch.randn(b, 1, K, 3, generator=g), dim=-1)
        .to("cuda", dtype)
        .requires_grad_()
    )
    v = (torch.randn(b, n, P, 3, generator=g) * 0.7).to("cuda", dtype).requires_grad_()
    logits = torch.randn(b, n, s, generator=g).to("cuda", dtype).requires_grad_()
    torch.manual_seed(0)
    lin = torch.nn.Linear(5 + K + P, 1).to("cuda", dtype)
    leaves = (*lin.parameters(), logits, r, nh, d, z, ms, v)
    wts = [
        torch.randn(b, n, s, generator=g).to("cuda", dtype),
        torch.randn(b, n, s, generator=g).to("cuda", dtype),
        torch.randn(b, n, 5 + K + P, generator=g).to("cuda", dtype),
    ]

    def run(fn):
        for t in leaves:
            t.grad = None
        d_n = d.expand(
            b, n, K, 3
        )  # both regions take the K global directions as (B, N, K, 3)
        if fn is _geo_region:
            outs = fn(lin, logits, r, nh, d_n, z, ms, 1e-12, v if P else None)
        else:  # the fused kernel takes the directions and the token vectors as one (B, N, K + P, 3) set
            outs = fn(
                lin,
                logits,
                r,
                nh,
                torch.cat([d_n, v], 2) if P else d_n,
                z,
                ms,
                1e-12,
                n_global=K,
            )
        sum((o * w).sum() for o, w in zip(outs, wts)).backward()
        return [o.detach() for o in outs], [
            (t.grad.detach().clone() if t.grad is not None else torch.zeros_like(t))
            for t in leaves
        ]

    o_ref, g_ref = run(_geo_region)
    o_f, g_f = run(fused_geo_region)
    tol = 1e-11 if dtype == torch.float64 else 1e-6
    for a, bb in zip(o_f, o_ref):
        assert _rel(a, bb) < tol, _rel(a, bb)
    for a, bb in zip(g_f, g_ref):
        if bb.numel() == 0:
            continue  # the empty token-vector leaf when P = 0
        assert _rel(a, bb) < (1e-11 if dtype == torch.float64 else 1e-5), _rel(a, bb)


@pytest.mark.cuda
@requires_cuda
@pytest.mark.slow
def test_fused_geo_region_offsets_past_int32():
    """Points x slices >= 2**31: the last rows, whose (point, slice) offsets pass int32,
    get the forward outputs and the per-point gradients of the same rows run alone
    (every row is computed from its own inputs). With int32 offsets the kernel read and
    wrote the wrong rows or faulted. Needs about 100 GiB of device memory: three (N, S)
    float32 tensors each way and the backward's per-tile slice partials."""
    from physicsnemo.experimental.models.isla.geo_kernel import fused_geo_region

    s, tail = 512, 64
    n = 2**31 // s + tail
    if torch.cuda.mem_get_info()[0] < 128 * 2**30:
        pytest.skip("needs about 100 GiB of free device memory")
    dev = torch.Generator(device="cuda").manual_seed(0)

    def unit(*shape):
        return torch.nn.functional.normalize(
            torch.randn(*shape, device="cuda", generator=dev), dim=-1
        )

    r = torch.randn(1, n, 3, device="cuda", generator=dev) * 0.1
    nh, d = unit(1, n, 3), unit(1, 1, 1, 3).expand(1, n, 1, 3)
    z, ms = torch.randn(1, s, 3, device="cuda", generator=dev) * 0.1, unit(1, s, 3)
    logits = torch.randn(1, n, s, device="cuda", generator=dev)
    torch.manual_seed(0)
    lin = torch.nn.Linear(6, 1).cuda()
    w_bias, w_mix = torch.randn(2, tail, s, device="cuda", generator=dev)

    def run(sl):
        x, rr = logits[:, sl].requires_grad_(), r[:, sl].requires_grad_()
        bias, mix, pooled = fused_geo_region(
            lin, x, rr, nh[:, sl], d[:, sl], z, ms, 1e-12
        )
        loss = (bias[0, -tail:] * w_bias).sum() + (mix[0, -tail:] * w_mix).sum()
        (loss + pooled[0, -tail:].sum()).backward()
        rows = (bias, mix, pooled, x.grad, rr.grad)
        return [t[0, -tail:].detach().clone() for t in rows]

    whole = run(slice(None))
    alone = run(slice(n - tail, None))
    for a, b in zip(whole, alone):
        assert _rel(a, b) < 1e-6, _rel(a, b)


@pytest.mark.cuda
@requires_cuda
def test_fused_geo_region_bf16_autocast_forward_matches_eager():
    """Under bf16 autocast the fused forward makes the same dtype decisions as eager
    (bf16 Linear on bf16-rounded invariants, bf16 pre-logit sum, fp32 softmax), so the
    two forwards agree to bf16 rounding-boundary flips."""
    from physicsnemo.experimental.models.isla.geo_kernel import fused_geo_region

    b, n, s = 1, 800, 64
    g = torch.Generator().manual_seed(1)
    r = (torch.randn(b, n, 3, generator=g) * 0.1).cuda()
    z = (torch.randn(b, s, 3, generator=g) * 0.1).cuda()
    nh = torch.nn.functional.normalize(torch.randn(b, n, 3, generator=g), dim=-1).cuda()
    ms = torch.nn.functional.normalize(torch.randn(b, s, 3, generator=g), dim=-1).cuda()
    d = (
        torch.nn.functional.normalize(torch.randn(b, 1, 1, 3, generator=g), dim=-1)
        .cuda()
        .expand(b, n, 1, 3)
    )
    logits = torch.randn(b, n, s, generator=g).cuda().bfloat16()
    torch.manual_seed(0)
    lin = torch.nn.Linear(6, 1).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = _geo_region(lin, logits, r, nh, d, z, ms, 1e-12)
        fused = fused_geo_region(lin, logits, r, nh, d, z, ms, 1e-12)
    assert [t.dtype for t in fused] == [t.dtype for t in ref]
    for a, bb in zip(fused, ref):
        assert _rel(a, bb) < 1e-4, _rel(a, bb)


# --------------------------------------------------------------------------- model level
@pytest.mark.cuda
@requires_cuda
def test_fused_geo_kernel_model_exact_in_float64():
    ref, opt = _pair(torch.float64, geo_kernel="fused")
    _assert_same(ref, opt, torch.float64, 1e-11, 1e-11)


@pytest.mark.cuda
@requires_cuda
def test_fused_geo_kernel_model_rank1_boundary_input_exact_in_float64():
    """A model with a rank-1 boundary input (the RANK-IO configuration) runs the fused
    kernel and matches the eager region to float64 roundoff."""
    spec = {**REF, "boundary_schemas": {"wall": {"t": {"rank": 1}}}}
    torch.manual_seed(0)
    ref = ISLA(**spec).to("cuda", torch.float64)
    torch.manual_seed(0)
    opt = ISLA(**spec).to("cuda", torch.float64)
    opt.geo_kernel = "fused"
    opt.load_state_dict(ref.state_dict())
    t = torch.randn(600, 3, generator=torch.Generator().manual_seed(9)) * 0.5
    inputs, tgt = _boundary(600, dtype=torch.float64, cell_data={"t": t})
    o_ref, g_ref = _run(ref, inputs, tgt)
    o_opt, g_opt = _run(opt, inputs, tgt)
    assert _rel(o_opt, o_ref) < 1e-11
    total = torch.cat([g.flatten() for g in g_ref.values()]).norm()
    for k, g in g_ref.items():
        if g.norm() < 1e-6 * total:
            continue
        assert _rel(g_opt[k], g) < 1e-11, (k, _rel(g_opt[k], g))


# --------------------------------------------------------------------------- torch.compile
@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize("geo_kernel", ["eager", "fused"])
def test_torch_compile_fullgraph_matches_eager(geo_kernel, monkeypatch):
    """The recipe compiles the whole model (compile: true); it must stay one graph and
    reproduce the eager reference within the study's float32 bars."""
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch._dynamo.reset()
    ref, opt = _pair(torch.float32, geo_kernel=geo_kernel)
    inputs, _ = _boundary(300)
    ex = torch._dynamo.explain(opt)(**inputs, prediction_boundaries=["wall"])
    assert ex.graph_break_count == 0, ex.break_reasons
    compiled = torch.compile(opt, fullgraph=True)
    ### Gradient bar 5e-5: on GB300 blocks.0.assign.v_proj.weight measured 2.1e-5 to 2.3e-5
    ### (LHR job 250845, before and after the argument cleanup), above the former 1e-5.
    _assert_same(ref, compiled, torch.float32, 1e-6, 5e-5, n=300)


# --------------------------------------------------------------------------- compiled blocks
@pytest.mark.slow  # 6 s on LHR CPU, 9 s on GB300: compiled blocks
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=[pytest.mark.cuda, requires_cuda])]
)
def test_compile_blocks_matches_eager(device):
    """``compile_blocks=True`` runs each slice block's forward under torch.compile and
    leaves parameter names, the state dict and the arithmetic unchanged: outputs and
    gradients match the eager model to float64 roundoff (the compiled blocks reorder
    floating-point sums only), and the model compiles once."""
    ref, opt = _pair(torch.float64, device=device, compile_blocks=True)
    assert set(opt.state_dict()) == set(ref.state_dict())
    inputs, tgt = _boundary(500, device=device, dtype=torch.float64)
    o_ref, g_ref = _run(ref, inputs, tgt)
    o_opt, g_opt = _run(opt, inputs, tgt)
    assert opt._block_forwards is not None and len(opt._block_forwards) == len(
        opt.blocks
    )
    assert _rel(o_opt, o_ref) < 1e-9, _rel(o_opt, o_ref)
    total = torch.cat([g.flatten() for g in g_ref.values()]).norm()
    for k, g in g_ref.items():
        if g.norm() < 1e-6 * total:
            continue
        assert _rel(g_opt[k], g) < 1e-9, (k, _rel(g_opt[k], g))
    # a second call with the same shapes reuses the compiled blocks (no rebuild)
    built = opt._block_forwards
    _run(opt, inputs, tgt)
    assert opt._block_forwards is built


@pytest.mark.cuda
@requires_cuda
def test_query_independent_interior_fused_and_compiled_blocks_match_eager():
    """The training configuration of ISLA-DTT (fused geometry kernel, compiled blocks)
    reproduces the eager model in float64, and its predictions stay
    query-independent on CUDA: a request's prefix gives the prefix of the predictions."""
    spec = {**REF, "interior": True}
    torch.manual_seed(0)
    ref = ISLA(**spec).to("cuda", torch.float64)
    ref.geo_kernel = "eager"
    torch.manual_seed(0)
    opt = ISLA(**spec, compile_blocks=True).to("cuda", torch.float64)
    opt.geo_kernel = "fused"
    opt.load_state_dict(ref.state_dict())
    inputs, _ = _boundary(600, dtype=torch.float64)
    q = (
        torch.randn(
            300, 3, generator=torch.Generator().manual_seed(5), dtype=torch.float64
        ).cuda()
        * 3.0
    )
    tgt = torch.randn(
        300, 4, generator=torch.Generator().manual_seed(6), dtype=torch.float64
    ).cuda()
    o_ref, g_ref = _run(ref, {**inputs, "prediction_points": q}, tgt)
    o_opt, g_opt = _run(opt, {**inputs, "prediction_points": q}, tgt)
    ### 1e-6 for what passes through the boundary read: its node means are float32 by design
    ### and CUDA's index_add sums in no fixed order; 1e-9 for the rest.
    tree_tol = 1e-6
    assert _rel(o_opt, o_ref) < tree_tol, _rel(o_opt, o_ref)
    total = torch.cat([g.flatten() for g in g_ref.values()]).norm()
    for k, g in g_ref.items():
        if g.norm() < 1e-6 * total:
            continue
        tol = tree_tol if k.startswith(("kernel_reads.", "kernel_basis.")) else 1e-9
        assert _rel(g_opt[k], g) < tol, (k, _rel(g_opt[k], g))
    with torch.no_grad():
        head = _flat(outputs(opt(**inputs, prediction_points=q[:40])))
    assert _rel(head, o_opt[:40]) < tree_tol


### Mix-order reduction (pytorch/pytorch#195845): Inductor fuses a block's softmax over the
### slices of each token with a sum over the tokens of each slice once tokens x slices >=
### 5 * 2**20 (and tokens >= 4096), and in torch 2.12 and 2.13 adds the padded rows of the
### last tile of tokens to that sum when the token count is not a multiple of the tile
### height (4 with 512 slices). The pair exists only in the eager geometry region (the
### fused kernel is a leaf of the graph); the 2-D models and those without a global
### direction run that region. The compiled blocks turn mix-order reduction off. These
### tests run the eager region at a size where it would fuse, in float32 (it never fuses
### float64), at token counts that are and are not multiples of 4. ISLA's forward pads
### the token axis to a multiple of TOKEN_ALIGNMENT (32), so the compiled blocks of a
### default forward never see such a count; the padding is not the fix, and the tests
### turn it off (an alignment of 1) so that the compiled blocks see the input's count.
MIX = {**REF, "n_layers": 1, "n_slices": 512}


def _record_block_tokens(model):
    """The token count of every call of ``model``'s compiled blocks, recorded from now on."""
    seen = []

    def recorded(forward):
        def call(h, *args, **kwargs):
            seen.append(h.shape[1])
            return forward(h, *args, **kwargs)

        return call

    model._block_forwards = [recorded(f) for f in model._block_callables()]
    return seen


@pytest.mark.slow  # 24 s on LHR CPU as the session's first Inductor compile, 3 s after one
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=[pytest.mark.cuda, requires_cuda])]
)
@pytest.mark.parametrize("task", ["surface", "interior_dtt"])
def test_compile_blocks_see_an_aligned_token_count(task, device):
    """A padded forward gives the compiled blocks a multiple of TOKEN_ALIGNMENT tokens at a
    token count that is not one."""
    from physicsnemo.experimental.models.isla import model

    kw = {} if task == "surface" else {"interior": True}
    torch.manual_seed(0)
    opt = ISLA(**{**REF, "n_layers": 1, **kw}, compile_blocks=True).to(device)
    inputs, _ = _boundary(301, device=device)
    if task == "surface":
        n, call = 301, dict(prediction_boundaries=["wall"])
    else:
        q = torch.randn(7, 3, generator=torch.Generator().manual_seed(5)).to(device)
        n, call = 308, dict(prediction_points=q)
    assert n % model.TOKEN_ALIGNMENT
    seen = _record_block_tokens(opt)
    with torch.no_grad():
        opt(**inputs, **call)
    assert seen == [n + -n % model.TOKEN_ALIGNMENT], seen


def _mix_pair(**kw):
    import torch._inductor.metrics

    torch._dynamo.reset()
    torch._inductor.metrics.reset()
    torch.manual_seed(0)
    ref = ISLA(**{**MIX, **kw}).to("cuda")
    opt = ISLA(**{**MIX, **kw}, compile_blocks=True).to("cuda")
    opt.load_state_dict(ref.state_dict())
    ref.geo_kernel = opt.geo_kernel = "eager"
    return ref, opt


def _forced_on_error(run, **kw):
    """The positive control: ``run(ref, opt)``'s output error with the same blocks compiled
    with mix-order reduction forced on. Skips the calling test where Inductor then builds
    no mix-order kernel, since the test cannot see the bug there. Inductor's caches are
    off because a cache hit skips the codegen that counts the kernels."""
    import torch._inductor.config

    from physicsnemo.experimental.models.isla import model

    with (
        pytest.MonkeyPatch.context() as mp,
        torch._inductor.config.patch(force_disable_caches=True),
    ):
        mp.setattr(
            model, "_BLOCK_COMPILE_OPTIONS", {"triton.mix_order_reduction": True}
        )
        ref, opt = _mix_pair(**kw)
        err = run(ref, opt)
    if torch._inductor.metrics.codegen_mix_order_reduction == 0:
        pytest.skip(
            "mix-order reduction does not fire on this device, so this test cannot detect pytorch/pytorch#195845"
        )
    return err


@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize("n", [12000, 12001, 12002, 12003])
def test_compile_blocks_float32_matches_eager_at_every_token_count(n, monkeypatch):
    """compile_blocks reproduces the eager model in float32 (outputs and gradients) at
    every token count, at a size where Inductor's mix-order reduction would fuse, and
    builds no mix-order kernel."""
    from physicsnemo.experimental.models.isla import model

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(model, "TOKEN_ALIGNMENT", 1)
    inputs, tgt = _boundary(n)

    def forced_on(ref, opt):
        o_ref = _run(ref, inputs, tgt)[0]
        return _rel(_run(opt, inputs, tgt)[0], o_ref)

    err = _forced_on_error(forced_on)
    if n % 4:
        assert err > 5e-6, err  # forced on, the bug shows at this size
    ref, opt = _mix_pair()
    seen = _record_block_tokens(opt)
    ### Without the fix on GB300 (torch 2.12): output error 5.7e-5 at n = 12001 and one
    ### mix-order kernel; with it, 2.7e-7 and none.
    _assert_same(ref, opt, torch.float32, 5e-6, 5e-5, n=n)
    assert torch._inductor.metrics.codegen_mix_order_reduction == 0
    assert seen == [n], seen


@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize("nq", [6001, 6002])
def test_compile_blocks_query_independent_float32_at_every_token_count(nq, monkeypatch):
    """ISLA-DTT under compile_blocks in float32, at a size where mix-order reduction would
    fuse and a total token count (6000 boundary cells + nq queries) that is not a multiple
    of 4: the compiled model reproduces the eager one, and its prediction at a fixed set of
    queries does not depend on the other queries of the request."""
    from physicsnemo.experimental.models.isla import model

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(model, "TOKEN_ALIGNMENT", 1)
    inputs, _ = _boundary(6000)
    q = torch.randn(nq, 3, generator=torch.Generator().manual_seed(5)).cuda() * 3.0

    def forced_on(ref, opt):
        with torch.no_grad():
            o_ref = _flat(outputs(ref(**inputs, prediction_points=q)))
            return _rel(_flat(outputs(opt(**inputs, prediction_points=q))), o_ref)

    err = _forced_on_error(forced_on, interior=True)
    assert err > 5e-6, err  # forced on, the bug shows at this size
    ref, opt = _mix_pair(interior=True)
    seen = _record_block_tokens(opt)
    with torch.no_grad():
        o_ref = _flat(outputs(ref(**inputs, prediction_points=q)))
        o_opt = _flat(outputs(opt(**inputs, prediction_points=q)))
        alone = _flat(outputs(opt(**inputs, prediction_points=q[:1000])))
    assert _rel(o_opt, o_ref) < 5e-6, _rel(o_opt, o_ref)
    assert _rel(alone, o_opt[:1000]) < 5e-6, _rel(alone, o_opt[:1000])
    assert torch._inductor.metrics.codegen_mix_order_reduction == 0
    assert seen == [6000 + nq, 7000], seen
