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

"""Parity and integration tests for physicsnemo.experimental.optim.LookSAM.

All tests run on cuda:0 — this is a Jetson Orin Nano (sm_87) with a real GPU.
Use bfloat16 for AMP; fp16 produces NaN on this device via the CC 8.0 PTX fallback.
"""

import json
import pathlib
import warnings

import pytest
import torch

warnings.filterwarnings("ignore", category=UserWarning)

from physicsnemo.experimental.optim import LookSAM, LookLayerSAM  # noqa: E402

DEVICE = "cuda:0"
FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "looksam"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make(seed: int = 42, device: str = DEVICE):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)
    ).to(device)
    base = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt = LookSAM(model.parameters(), base_optimizer=base, rho=0.1, alpha=0.7, k=5)
    return model, opt


def _closure(opt, model, x):
    def _fn():
        opt.zero_grad()
        loss = model(x).sum()
        loss.backward()
        return loss
    return _fn


@pytest.fixture(scope="module")
def fixture_data():
    pt = FIXTURE_DIR / "fixtures.pt"
    if not pt.exists():
        pytest.skip(f"fixtures not found at {pt}")
    return torch.load(pt, weights_only=False)


@pytest.fixture(scope="module")
def metadata():
    meta = FIXTURE_DIR / "metadata.json"
    if not meta.exists():
        pytest.skip(f"metadata not found at {meta}")
    return json.loads(meta.read_text())


# ---------------------------------------------------------------------------
# a. test_shape — output shapes match reference oracle
# ---------------------------------------------------------------------------


def test_shape(fixture_data):
    torch.manual_seed(42)
    model, opt = _make()
    x = torch.randn(4, 8, device=DEVICE)
    opt.step(_closure(opt, model, x))   # step 0 is a SAM step (0 % k == 0)

    params_ref = fixture_data["params_after_cycle"]
    for name, p in model.named_parameters():
        assert p.shape == params_ref[name].shape


# ---------------------------------------------------------------------------
# b. test_parity — numerical match against reference oracle
# ---------------------------------------------------------------------------


def test_parity(fixture_data, metadata):
    tolerance = metadata["tolerance"]
    seed = metadata["seed"]
    torch.manual_seed(seed)
    model, opt = _make(seed=seed)
    torch.manual_seed(seed)
    x = torch.randn(4, 8, device=DEVICE)

    opt.step(_closure(opt, model, x))

    params_ref = fixture_data["params_after_cycle"]
    for name, p in model.named_parameters():
        mse = ((p.data.cpu() - params_ref[name]) ** 2).mean().item()
        assert mse <= tolerance, f"{name}: MSE={mse:.2e} > tol={tolerance:.2e}"


# ---------------------------------------------------------------------------
# c. test_gradient_flow — grads are finite after a full cycle
# ---------------------------------------------------------------------------


def test_gradient_flow():
    torch.manual_seed(42)
    model, opt = _make()
    x = torch.randn(4, 8, device=DEVICE)

    opt.step(_closure(opt, model, x))

    # Another backward after the step should yield finite grads
    opt.zero_grad()
    model(x).sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"grad is None for {name}"
        assert not torch.isnan(p.grad).any(), f"NaN grad for {name}"


# ---------------------------------------------------------------------------
# d. test_amp_cuda — bfloat16 autocast on CUDA (no GradScaler needed)
# ---------------------------------------------------------------------------


def test_amp_cuda():
    """bf16 is the correct AMP dtype for Jetson Orin (sm_87 via CC 8.0 PTX).

    fp16 produces NaN gradients on this device. bf16 has the same dynamic
    range as fp32 and does not need a GradScaler.
    """
    torch.manual_seed(42)
    model, opt = _make()
    x = torch.randn(4, 8, device=DEVICE)

    def closure_bf16():
        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x).sum()
        loss.backward()
        return loss

    opt.step(closure_bf16)

    for name, p in model.named_parameters():
        assert not torch.isnan(p).any(), f"NaN in {name} after bf16 step"
        assert not torch.isinf(p).any(), f"Inf in {name} after bf16 step"


# ---------------------------------------------------------------------------
# e. test_k_reuse — fast steps (t % k != 0) update params without error
# ---------------------------------------------------------------------------


def test_k_reuse():
    torch.manual_seed(42)
    model, opt = _make()
    x = torch.randn(4, 8, device=DEVICE)
    closure = _closure(opt, model, x)

    opt.step(closure)  # step 0: SAM step, populates g_v cache

    params_before = {n: p.data.clone() for n, p in model.named_parameters()}

    opt.step(closure)  # step 1: fast step (1 % 5 != 0)

    for name, p in model.named_parameters():
        assert not torch.equal(p.data, params_before[name]), f"{name} did not update on fast step"
        assert not torch.isnan(p).any(), f"NaN in {name} after fast step"


# ---------------------------------------------------------------------------
# f. test_k1_equiv_sam — k=1 is equivalent to vanilla SAM
# ---------------------------------------------------------------------------


def test_k1_equiv_sam():
    """With k=1 every step is a full SAM step; closure is called twice each time."""
    torch.manual_seed(42)
    model, opt = _make()
    base = opt.base_optimizer
    x = torch.randn(4, 8, device=DEVICE)

    call_count = [0]

    def counted_closure():
        call_count[0] += 1
        opt.zero_grad()
        loss = model(x).sum()
        loss.backward()
        return loss

    # Override k=1
    for g in opt.param_groups:
        g["k"] = 1

    for _ in range(3):
        opt.step(counted_closure)

    assert call_count[0] == 6, f"expected 6 closure calls (2×3), got {call_count[0]}"
    assert opt.sam_step == 3


# ---------------------------------------------------------------------------
# g. test_state_dict_roundtrip — save/load produces identical trajectory
# ---------------------------------------------------------------------------


def test_state_dict_roundtrip():
    import copy

    torch.manual_seed(42)
    model, opt = _make()
    x = torch.randn(4, 8, device=DEVICE)
    closure = _closure(opt, model, x)

    for _ in range(5):
        opt.step(closure)

    state = copy.deepcopy(opt.state_dict())
    params_snap = {n: p.clone() for n, p in model.named_parameters()}

    for _ in range(3):
        opt.step(closure)
    params_forward = {n: p.clone() for n, p in model.named_parameters()}

    # Restore and replay
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(params_snap[n])

    model2 = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)
    ).to(DEVICE)
    with torch.no_grad():
        for (n, p_src), p_dst in zip(params_snap.items(), model2.parameters()):
            p_dst.copy_(p_src)
    base2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    opt2 = LookSAM(model2.parameters(), base_optimizer=base2, rho=0.1, alpha=0.7, k=5)
    opt2.load_state_dict(state)
    assert opt2.global_step == opt.global_step - 3

    closure2 = _closure(opt2, model2, x)
    for _ in range(3):
        opt2.step(closure2)

    for name, p in model2.named_parameters():
        torch.testing.assert_close(p, params_forward[name], atol=1e-5, rtol=0,
                                   msg=f"state_dict roundtrip mismatch for {name}")


# ---------------------------------------------------------------------------
# h. test_lr_scheduler_compat — LR scheduler propagates through wrapper
# ---------------------------------------------------------------------------


def test_lr_scheduler_compat():
    torch.manual_seed(42)
    model, opt = _make()
    base = opt.base_optimizer
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)

    lr_before = opt.param_groups[0]["lr"]

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sched.step()

    assert opt.param_groups[0]["lr"] == pytest.approx(lr_before * 0.5)
    assert base.param_groups[0]["lr"] == pytest.approx(lr_before * 0.5), \
        "lr change did not propagate to base optimizer"


# ---------------------------------------------------------------------------
# i. test_multiple_param_groups — two groups with different lr
# ---------------------------------------------------------------------------


def test_multiple_param_groups():
    torch.manual_seed(42)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)
    ).to(DEVICE)
    base = torch.optim.Adam([
        {"params": list(model[0].parameters()), "lr": 1e-3},
        {"params": list(model[2].parameters()), "lr": 1e-4},
    ])
    opt = LookSAM(model.parameters(), base_optimizer=base, rho=0.05, k=3)

    x = torch.randn(4, 8, device=DEVICE)
    for _ in range(4):
        opt.step(_closure(opt, model, x))

    for p in model.parameters():
        assert not torch.isnan(p).any()
