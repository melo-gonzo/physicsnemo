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

"""Single-process tests for physicsnemo.experimental.optim.LookSAM."""

import copy
import warnings

import pytest
import torch

from physicsnemo.optim import CombinedOptimizer

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from physicsnemo.experimental.optim import LookLayerSAM, LookSAM


def _model(device, dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(6, 12), torch.nn.Tanh(), torch.nn.Linear(12, 3)
    ).to(device, dtype)


def _data(device, dtype=torch.float64, n=16, seed=1):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 6, generator=gen, dtype=dtype)
    y = torch.randn(n, 3, generator=gen, dtype=dtype)
    return x.to(device), y.to(device)


def _closure(opt, model, x, y):
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        return loss

    return closure


def _reference(model, x, y, steps, *, lr, rho, alpha, k, eps, layer):
    """Dense per-parameter LookSAM on SGD, written directly from the paper."""
    params = list(model.parameters())

    def grads():
        model.zero_grad()
        torch.nn.functional.mse_loss(model(x), y).backward()
        return [p.grad.clone() for p in params]

    def norm(ts):
        return torch.sqrt(sum((t * t).sum() for t in ts))

    g_v = None
    for t in range(steps):
        g = grads()
        with torch.no_grad():
            if t % k == 0:
                if layer:
                    e = [rho * gi / (gi.norm() + eps) for gi in g]
                else:
                    e = [rho * gi / (norm(g) + eps) for gi in g]
                w = [p.clone() for p in params]
                for p, ei in zip(params, e):
                    p.add_(ei)
                with torch.enable_grad():
                    g_s = grads()
                for p, wi in zip(params, w):
                    p.copy_(wi)
                dot = sum((gi * gsi).sum() for gi, gsi in zip(g, g_s))
                coef = dot / (norm(g) ** 2 + eps)
                g_v = [gsi - coef * gi for gi, gsi in zip(g, g_s)]
                update = g_s
            else:
                blend = alpha * norm(g) / (norm(g_v) + eps)
                update = [gi + blend * vi for gi, vi in zip(g, g_v)]
            for p, u in zip(params, update):
                p.sub_(lr * u)


@pytest.mark.parametrize("cls", [LookSAM, LookLayerSAM])
def test_matches_reference(device, cls):
    """SAM and fast steps match the dense per-parameter update rule."""
    hp = dict(rho=0.1, alpha=0.7, k=3, eps=1e-12)
    x, y = _data(device)
    model, ref = _model(device), _model(device)
    opt = cls(
        model.parameters(),
        base_optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        **hp,
    )
    for _ in range(8):
        opt.step(_closure(opt, model, x, y))
    _reference(ref, x, y, 8, lr=0.1, layer=cls is LookLayerSAM, **hp)

    assert (opt.global_step, opt.sam_step) == (8, 3)
    for p, q in zip(model.parameters(), ref.parameters()):
        torch.testing.assert_close(p, q, rtol=1e-12, atol=1e-12)


# Relative tolerance of a short trajectory against the float64 reference.
_DTYPE_RTOL = {torch.float32: 1e-6, torch.bfloat16: 2e-2, torch.float16: 3e-3}


def _rel_err(model, ref):
    num = sum(
        (p.double() - q).square().sum()
        for p, q in zip(model.parameters(), ref.parameters())
    )
    den = sum(q.square().sum() for q in ref.parameters())
    return (num / den).sqrt().item()


@pytest.mark.parametrize("cls", [LookSAM, LookLayerSAM])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_low_precision_parameters(device, cls, dtype):
    """Training directly in a lower precision tracks the float64 reference.

    Norms and dot products accumulate in float32, so the result only carries
    the rounding of the parameter dtype itself; g_v is stored in that dtype.
    """
    hp = dict(rho=0.1, alpha=0.7, k=3, eps=1e-12)
    ref = _model(device)
    model = _model(device).to(dtype)
    x, y = _data(device)
    opt = cls(
        model.parameters(),
        base_optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        **hp,
    )
    for _ in range(5):
        opt.step(_closure(opt, model, x.to(dtype), y.to(dtype)))
    _reference(ref, x, y, 5, lr=0.1, layer=cls is LookLayerSAM, **hp)

    assert all(opt.state[p]["g_v"].dtype == dtype for p in model.parameters())
    assert _rel_err(model, ref) < _DTYPE_RTOL[dtype]


class _MixedPrecisionMLP(torch.nn.Module):
    """Layers kept in different dtypes, as with a float32 head on a bfloat16 trunk."""

    def __init__(self, dtypes):
        super().__init__()
        torch.manual_seed(0)
        sizes = [6, 12, 12, 3]
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(a, b).to(dtype=dt)
            for a, b, dt in zip(sizes, sizes[1:], dtypes)
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x.to(layer.weight.dtype))
            if i < len(self.layers) - 1:
                x = torch.tanh(x)
        return x.to(torch.float64)


@pytest.mark.parametrize("cls", [LookSAM, LookLayerSAM])
def test_mixed_dtype_model(device, cls):
    """One model with float64, float32, and bfloat16 parameters: one bucket each."""
    hp = dict(rho=0.1, alpha=0.7, k=2, eps=1e-12)
    dtypes = (torch.bfloat16, torch.float32, torch.float64)
    model = _MixedPrecisionMLP(dtypes).to(device)
    ref = _MixedPrecisionMLP(dtypes).to(device).double()
    x, y = _data(device)
    opt = cls(
        model.parameters(),
        base_optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        **hp,
    )
    for _ in range(4):
        opt.step(_closure(opt, model, x, y))
    _reference(ref, x, y, 4, lr=0.1, layer=cls is LookLayerSAM, **hp)

    assert sorted(str(b.flat_v.dtype) for b in opt._buckets) == sorted(map(str, dtypes))
    assert _rel_err(model, ref) < _DTYPE_RTOL[torch.bfloat16]


def test_k1_calls_closure_twice(device):
    """k=1 is vanilla SAM: every step is a SAM step with two closure calls."""
    model = _model(device)
    x, y = _data(device)
    opt = LookSAM(
        model.parameters(),
        base_optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        k=1,
    )
    calls = 0
    inner = _closure(opt, model, x, y)

    def closure():
        nonlocal calls
        calls += 1
        return inner()

    for _ in range(3):
        opt.step(closure)
    assert calls == 6
    assert opt.sam_step == 3


def test_weights_restored_exactly(device):
    """The ascent perturbation is undone bit-exactly, also for bfloat16 weights."""
    model = _model(device, dtype=torch.bfloat16)
    x, y = _data(device, dtype=torch.bfloat16)
    before = [p.detach().clone() for p in model.parameters()]
    opt = LookSAM(
        model.parameters(),
        base_optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        rho=0.5,
    )
    for _ in range(4):
        opt.step(_closure(opt, model, x, y))
    for p, b in zip(model.parameters(), before):
        assert torch.equal(p, b)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_autocast(device, dtype):
    """Autocast steps on float32 weights stay finite (no GradScaler)."""
    model = _model(device, dtype=torch.float32)
    x, y = _data(device, dtype=torch.float32)
    opt = LookSAM(
        model.parameters(),
        base_optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        k=2,
    )

    def closure():
        opt.zero_grad()
        with torch.autocast(torch.device(device).type, dtype=dtype):
            loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        return loss

    for _ in range(4):
        opt.step(closure)
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_params_without_grad(device):
    """Frozen or unused parameters are skipped on SAM and fast steps."""
    model = _model(device)
    unused = torch.nn.Parameter(torch.ones(4, device=device, dtype=torch.float64))
    model[0].weight.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad] + [unused]
    x, y = _data(device)
    opt = LookSAM(params, base_optimizer=torch.optim.SGD(params, lr=0.1), k=2)
    for _ in range(4):
        opt.step(_closure(opt, model, x, y))
    assert torch.equal(unused, torch.ones_like(unused))
    assert "g_v" not in opt.state[unused]
    assert "g_v" in opt.state[model[2].weight]


def test_state_dict_roundtrip(device, tmp_path):
    """Save/load mid-cycle continues the identical trajectory."""
    x, y = _data(device)
    model = _model(device)
    opt = LookSAM(
        model.parameters(),
        base_optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
        k=3,
    )
    for _ in range(4):
        opt.step(_closure(opt, model, x, y))
    torch.save(
        {"model": model.state_dict(), "opt": opt.state_dict()}, tmp_path / "ck.pt"
    )
    for _ in range(4):
        opt.step(_closure(opt, model, x, y))

    ckpt = torch.load(tmp_path / "ck.pt", weights_only=False)
    model2 = _model(device, seed=7)
    model2.load_state_dict(ckpt["model"])
    opt2 = LookSAM(
        model2.parameters(),
        base_optimizer=torch.optim.Adam(model2.parameters(), lr=1e-2),
        k=3,
    )
    opt2.load_state_dict(ckpt["opt"])
    assert (opt2.global_step, opt2.sam_step) == (4, 2)
    for _ in range(4):
        opt2.step(_closure(opt2, model2, x, y))
    for p, q in zip(model.parameters(), model2.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_scheduler_after_load(device):
    """LR schedules still reach the base optimizer after load_state_dict."""
    model = _model(device)
    opt = LookSAM(
        model.parameters(), base_optimizer=torch.optim.SGD(model.parameters(), lr=1.0)
    )
    opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
    opt.step(_closure(opt, model, *_data(device)))
    sched.step()
    assert opt.base_optimizer.param_groups[0]["lr"] == pytest.approx(0.5)
    assert opt.param_groups is opt.base_optimizer.param_groups
    assert opt.param_groups[0]["rho"] == 0.05


def test_combined_optimizer_base(device):
    """LookSAM wraps a CombinedOptimizer, including LR scheduling and reload."""
    model = _model(device)
    base = CombinedOptimizer(
        [
            torch.optim.SGD(model[0].parameters(), lr=0.1),
            torch.optim.Adam(model[2].parameters(), lr=1e-2),
        ]
    )
    opt = LookSAM(model.parameters(), base_optimizer=base, k=2)
    x, y = _data(device)
    for _ in range(3):
        opt.step(_closure(opt, model, x, y))
    opt.load_state_dict(opt.state_dict())
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
    opt.step(_closure(opt, model, x, y))
    sched.step()
    assert base.optimizers[0].param_groups[0]["lr"] == pytest.approx(0.05)
    assert base.optimizers[1].param_groups[0]["lr"] == pytest.approx(5e-3)
    assert all(torch.isfinite(p).all() for p in model.parameters())


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(rho=0.0), "rho"),
        (dict(alpha=-1.0), "alpha"),
        (dict(k=0), "k"),
    ],
)
def test_invalid_hyperparameters(kwargs, match):
    p = torch.nn.Parameter(torch.zeros(2))
    with pytest.raises(ValueError, match=match):
        LookSAM([p], base_optimizer=torch.optim.SGD([p], lr=0.1), **kwargs)


def test_requires_closure_and_optimizer():
    p = torch.nn.Parameter(torch.zeros(2))
    with pytest.raises(ValueError, match="base_optimizer"):
        LookSAM([p], base_optimizer=None)
    opt = LookSAM([p], base_optimizer=torch.optim.SGD([p], lr=0.1))
    with pytest.raises(ValueError, match="closure"):
        opt.step(None)
