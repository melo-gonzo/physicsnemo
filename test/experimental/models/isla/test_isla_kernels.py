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
eager reference (geo_kernel="eager", geo_checkpoint=True).

Bars from the study brief, applied to the reference configuration's arithmetic:
the fused Triton region reproduces the eager forward to 1e-6 relative in float32 and
the parameter gradients to 1e-5 relative; in float64 the two agree to 1e-11 (same
arithmetic, different summation order). The same bars hold for the whole model
compiled with torch.compile, which is the recipe's default (compile: true) and must
stay free of graph breaks."""

import pytest
import torch

from physicsnemo.experimental.models import ISLA
from physicsnemo.experimental.models.isla.model import _geo_region
from test.experimental.models.isla._meshes import cloud, outputs, soup

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernels")

REF = dict(
    outputs={"p": {"rank": 0}, "v": {"rank": 1}},
    global_inputs={"g": {"rank": 1}},
    hidden=64,
    n_layers=3,
    n_slices=32,
    geo_checkpoint=True,
)


def _boundary(n=700, seed=0, device="cuda", dtype=torch.float32, cell_data=None):
    """One boundary soup of ``n`` cells with a global direction, and a target."""
    pts, nrm, w, drv = cloud(n, seed=seed, dtype=torch.float32)
    mesh = soup(pts, nrm, w, cell_data=cell_data).to(device).to(dtype)
    tgt = torch.randn(n, 4, generator=torch.Generator().manual_seed(seed + 1)).to(device, dtype)
    return dict(boundary_meshes={"wall": mesh}, global_data={"g": drv.to(device, dtype)}), tgt


def _flat(o):
    return torch.cat([o["p"][..., None], o["v"]], dim=-1)


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


def _run(model, inputs, target):
    model.zero_grad(set_to_none=True)
    out = _flat(outputs(model(**inputs)))
    torch.nn.functional.mse_loss(out.float(), target.float()).backward()
    grads = {k.replace("_orig_mod.", ""): p.grad.detach().clone() for k, p in model.named_parameters()}
    return out.detach(), grads


def _pair(dtype, device="cuda", **kw):
    torch.manual_seed(0)
    ref = ISLA(**REF).to(device, dtype)
    torch.manual_seed(0)
    opt = ISLA(**{**REF, **kw}).to(device, dtype)
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
        assert _rel(g_opt[k], g) < grad_tol, (k, _rel(g_opt[k], g))


# --------------------------------------------------------------------------- fused region
@cuda_only
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("K,P", [(1, 0), (2, 0), (3, 0), (4, 0), (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (3, 1)])
def test_fused_geo_region_matches_eager_region(dtype, K, P):
    """Region level: forward outputs (bias, mix, pooled) and every input gradient, for
    K global directions plus P per-token rank-1 vectors (the (5 + K + P)-invariant set
    in the eager order) with the linear routing-bias head. The fused kernel keeps a
    ``relative`` flag; the model always passes ``True``, which is what the eager region
    computes."""
    from physicsnemo.experimental.models.isla.geo_kernel import fused_geo_region

    torch.backends.cuda.matmul.allow_tf32 = False
    b, n, s = 2, 500, 40
    g = torch.Generator().manual_seed(0)

    def mk(*shape):
        return (torch.randn(*shape, generator=g) * 0.1).to("cuda", dtype).requires_grad_()

    r, z = mk(b, n, 3), mk(b, s, 3)
    nh = torch.nn.functional.normalize(torch.randn(b, n, 3, generator=g), dim=-1).to("cuda", dtype).requires_grad_()
    ms = torch.nn.functional.normalize(torch.randn(b, s, 3, generator=g), dim=-1).to("cuda", dtype).requires_grad_()
    d = torch.nn.functional.normalize(torch.randn(b, 1, K, 3, generator=g), dim=-1).to("cuda", dtype).requires_grad_()
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

    def run(fn, *extra):
        for t in leaves:
            t.grad = None
        d_n = d.expand(b, n, K, 3)  # both regions take the K global directions as (B, N, K, 3)
        if fn is _geo_region:
            outs = fn(lin, logits, r, nh, d_n, z, ms, 1e-12, v if P else None)
        else:  # the fused kernel takes the directions and the token vectors as one (B, N, K + P, 3) set
            outs = fn(lin, logits, r, nh, torch.cat([d_n, v], 2) if P else d_n, z, ms, 1e-12, *extra, n_global=K)
        sum((o * w).sum() for o, w in zip(outs, wts)).backward()
        return [o.detach() for o in outs], [
            (t.grad.detach().clone() if t.grad is not None else torch.zeros_like(t)) for t in leaves
        ]

    o_ref, g_ref = run(_geo_region)
    o_f, g_f = run(fused_geo_region, True)
    tol = 1e-11 if dtype == torch.float64 else 1e-6
    for a, bb in zip(o_f, o_ref):
        assert _rel(a, bb) < tol, _rel(a, bb)
    for a, bb in zip(g_f, g_ref):
        if bb.numel() == 0:
            continue  # the empty token-vector leaf when P = 0
        assert _rel(a, bb) < (1e-11 if dtype == torch.float64 else 1e-5), _rel(a, bb)


@cuda_only
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
    d = torch.nn.functional.normalize(torch.randn(b, 1, 1, 3, generator=g), dim=-1).cuda().expand(b, n, 1, 3)
    logits = torch.randn(b, n, s, generator=g).cuda().bfloat16()
    torch.manual_seed(0)
    lin = torch.nn.Linear(6, 1).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = _geo_region(lin, logits, r, nh, d, z, ms, 1e-12)
        fused = fused_geo_region(lin, logits, r, nh, d, z, ms, 1e-12, True)
    assert [t.dtype for t in fused] == [t.dtype for t in ref]
    for a, bb in zip(fused, ref):
        assert _rel(a, bb) < 1e-4, _rel(a, bb)


# --------------------------------------------------------------------------- model level
@cuda_only
def test_fused_geo_kernel_model_exact_in_float64():
    ref, opt = _pair(torch.float64, geo_kernel="fused")
    _assert_same(ref, opt, torch.float64, 1e-11, 1e-11)


@cuda_only
def test_fused_geo_kernel_model_rank1_boundary_input_exact_in_float64():
    """A model with a rank-1 boundary input (the RANK-IO configuration) runs the fused
    kernel and matches the eager region to float64 roundoff."""
    spec = {**REF, "boundary_inputs": {"t": {"rank": 1}}}
    torch.manual_seed(0)
    ref = ISLA(**spec).to("cuda", torch.float64)
    torch.manual_seed(0)
    opt = ISLA(**{**spec, "geo_kernel": "fused"}).to("cuda", torch.float64)
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


@cuda_only
def test_geo_kernel_argument_is_validated():
    with pytest.raises(ValueError):
        ISLA(**REF, geo_kernel="triton")


# --------------------------------------------------------------------------- torch.compile
@cuda_only
@pytest.mark.parametrize("geo_kernel", ["eager", "fused"])
def test_torch_compile_fullgraph_matches_eager(geo_kernel):
    """The recipe compiles the whole model (compile: true); it must stay one graph and
    reproduce the eager reference within the study's float32 bars."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch._dynamo.reset()
    ref, opt = _pair(torch.float32, geo_kernel=geo_kernel)
    inputs, _ = _boundary(300)
    ex = torch._dynamo.explain(opt)(**inputs)
    assert ex.graph_break_count == 0, ex.break_reasons
    compiled = torch.compile(opt, fullgraph=True)
    _assert_same(ref, compiled, torch.float32, 1e-6, 1e-5, n=300)
