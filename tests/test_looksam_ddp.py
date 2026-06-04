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

"""DDP integration tests for LookSAM (closure-based API).

Single-process tests (world_size=1, gloo backend) run on the Jetson.
Multi-process tests are marked @pytest.mark.multigpu_static and require:

    torchrun --nproc-per-node 2 -m pytest --multigpu-static tests/test_looksam_ddp.py

Note: NCCL P2P is not supported on Jetson Orin Nano (nvmlDeviceGetP2PStatus fails).
Single-process DDP uses gloo with CUDA tensors instead.

DDP limitation: on SAM steps LookSAM calls the closure twice. The second call
(at perturbed weights) triggers an AllReduce under DDP. This is redundant but
not incorrect. Suppressing it requires passing ddp_no_sync_fn to LookSAM.__init__.
"""

import os
import warnings

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

warnings.filterwarnings("ignore")

from physicsnemo.experimental.optim import LookSAM, LookLayerSAM  # noqa: E402

DEVICE = "cuda:0"


# ---------------------------------------------------------------------------
# Process group helpers
# ---------------------------------------------------------------------------


def _init_gloo(port: str = "29510") -> None:
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("gloo", rank=0, world_size=1)


@pytest.fixture(autouse=True, scope="module")
def process_group():
    _init_gloo(port="29510")
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def _make_ddp(seed: int = 42):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)
    ).to(DEVICE)
    ddp = DistributedDataParallel(model, device_ids=[0])
    base = torch.optim.Adam(ddp.parameters(), lr=1e-3)
    opt = LookSAM(ddp.parameters(), base_optimizer=base, rho=0.1, alpha=0.7, k=5)
    return model, ddp, opt


def _closure(opt, ddp, x):
    def _fn():
        opt.zero_grad()
        loss = ddp(x).sum()
        loss.backward()
        return loss
    return _fn


# ---------------------------------------------------------------------------
# Single-process DDP tests (run on Jetson, world_size=1, gloo backend)
# ---------------------------------------------------------------------------


def test_ddp_wrapping():
    """LookSAM step() works when model is wrapped in DistributedDataParallel."""
    model, ddp, opt = _make_ddp()
    x = torch.randn(4, 8, device=DEVICE)

    for _ in range(6):  # 2 SAM steps (at 0, 5), 4 fast steps
        opt.step(_closure(opt, ddp, x))

    assert opt.global_step == 6
    assert opt.sam_step == 2
    for name, p in model.named_parameters():
        assert not torch.isnan(p).any(), f"NaN in {name}"


def test_ddp_no_sync_fn():
    """ddp_no_sync_fn suppresses the AllReduce on the second (perturbation) closure.

    With world_size=1 the AllReduce is a no-op, but the context manager must be
    accepted without error and the step must still converge correctly.
    """
    torch.manual_seed(42)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)
    ).to(DEVICE)
    ddp = DistributedDataParallel(model, device_ids=[0])
    base = torch.optim.Adam(ddp.parameters(), lr=1e-3)
    opt = LookSAM(
        ddp.parameters(),
        base_optimizer=base,
        rho=0.1,
        alpha=0.7,
        k=5,
        ddp_no_sync_fn=ddp.no_sync,  # suppress redundant AllReduce on perturbation pass
    )

    x = torch.randn(4, 8, device=DEVICE)
    for _ in range(6):
        opt.step(_closure(opt, ddp, x))

    assert opt.sam_step == 2
    for p in model.parameters():
        assert not torch.isnan(p).any()


def test_ddp_fast_steps():
    """Fast steps (t % k != 0) update params without error under DDP."""
    model, ddp, opt = _make_ddp()
    x = torch.randn(4, 8, device=DEVICE)
    closure = _closure(opt, ddp, x)

    opt.step(closure)  # SAM step — populates g_v
    params_after_sam = {n: p.data.clone() for n, p in model.named_parameters()}

    for _ in range(4):  # 4 fast steps (k=5)
        opt.step(closure)

    for name, p in model.named_parameters():
        assert not torch.equal(p.data, params_after_sam[name]), f"{name} frozen on fast step"
        assert not torch.isnan(p).any()


def test_ddp_bf16():
    """bf16 autocast works under DDP on sm_87."""
    model, ddp, opt = _make_ddp()
    x = torch.randn(4, 8, device=DEVICE)

    def closure_bf16():
        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = ddp(x).sum()
        loss.backward()
        return loss

    for _ in range(3):
        opt.step(closure_bf16)

    for p in model.parameters():
        assert not torch.isnan(p).any()


def test_ddp_looksam_layer():
    """LookLayerSAM works under DDP."""
    torch.manual_seed(42)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)
    ).to(DEVICE)
    ddp = DistributedDataParallel(model, device_ids=[0])
    base = torch.optim.Adam(ddp.parameters(), lr=1e-3)
    opt = LookLayerSAM(ddp.parameters(), base_optimizer=base, rho=0.1, k=3)
    x = torch.randn(4, 8, device=DEVICE)

    for _ in range(4):
        opt.step(_closure(opt, ddp, x))

    for p in model.parameters():
        assert not torch.isnan(p).any()


# ---------------------------------------------------------------------------
# Multi-process DDP test — requires torchrun --nproc-per-node 2
# ---------------------------------------------------------------------------


@pytest.mark.multigpu_static
@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="requires ≥2 GPUs — run with: torchrun --nproc-per-node 2 -m pytest --multigpu-static",
)
def test_ddp_gradient_sync():
    """After a full LookSAM cycle params are identical across all ranks.

    On the descent step DDP AllReduces gradients; all replicas must apply the
    same update and converge to identical parameter values.
    """
    from physicsnemo.distributed import DistributedManager

    dm = DistributedManager()
    device = f"cuda:{dm.local_rank}"

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)
    ).to(device)
    ddp = DistributedDataParallel(model, device_ids=[dm.local_rank])
    base = torch.optim.Adam(ddp.parameters(), lr=1e-3)
    opt = LookSAM(ddp.parameters(), base_optimizer=base, rho=0.05, k=3,
                  ddp_no_sync_fn=ddp.no_sync)

    torch.manual_seed(dm.rank + 1)
    x = torch.randn(4, 8, device=device)

    def closure():
        opt.zero_grad()
        ddp(x).sum().backward()
        return torch.tensor(0.0)

    for _ in range(6):
        opt.step(closure)

    for name, p in model.named_parameters():
        norm = p.data.norm().unsqueeze(0)
        gathered = [torch.zeros_like(norm) for _ in range(dm.world_size)]
        dist.all_gather(gathered, norm)
        norms = [g.item() for g in gathered]
        assert max(norms) - min(norms) < 1e-5, \
            f"{name} diverged across ranks after LookSAM: {norms}"
