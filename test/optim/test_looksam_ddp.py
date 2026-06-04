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

Sync contract under test
------------------------
On SAM steps LookSAM calls the closure twice, and *both* backward passes must
AllReduce. Unlike vanilla SAM, the ascent gradient is not discarded — it is
reused inside the cached ``g_v`` that drives fast steps — so skipping its sync
(e.g. via ``no_sync``) would desynchronize ``g_v`` and make ranks diverge. There
is no correct ``no_sync`` shortcut for LookSAM; ``test_ddp_gradient_sync`` guards
this by feeding each rank a *different* batch and asserting parameters stay
identical.

Running these tests
-------------------
All tests here are NCCL-based and marked ``@pytest.mark.multigpu_static``; they
use ``DistributedManager`` for backend/device assignment (one GPU per rank) and
require a real multi-GPU machine. Launch with::

    torchrun --nproc-per-node <N> -m pytest --multigpu-static \\
        test/optim/test_looksam_ddp.py

Under plain ``pytest`` (no ``--multigpu-static``) every test here is auto-skipped
by the repo conftest. Note: NCCL P2P is unsupported on Jetson sm_87, so these do
not run on that hardware — see the ``reference_ddp_testing_jetson`` note for the
gloo/shared-GPU workaround used during local development.
"""

import warnings

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

warnings.filterwarnings("ignore")

from physicsnemo.experimental.optim import LookSAM, LookLayerSAM  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _device():
    """Return (DistributedManager, device_str) and pin this rank to its GPU.

    Relies on the repo conftest having called ``DistributedManager.initialize()``
    under ``--multigpu-static`` (NCCL backend, one GPU per rank).
    """
    from physicsnemo.distributed import DistributedManager

    dm = DistributedManager()
    torch.cuda.set_device(dm.local_rank)
    return dm, str(dm.device)


def _make_ddp(seed: int = 42):
    dm, device = _device()
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)
    ).to(device)
    ddp = DistributedDataParallel(model, device_ids=[dm.local_rank])
    base = torch.optim.Adam(ddp.parameters(), lr=1e-3)
    opt = LookSAM(ddp.parameters(), base_optimizer=base, rho=0.1, alpha=0.7, k=5)
    return model, ddp, opt, device


def _closure(opt, ddp, x):
    def _fn():
        opt.zero_grad()
        loss = ddp(x).sum()
        loss.backward()
        return loss
    return _fn


# ---------------------------------------------------------------------------
# Per-rank smoke tests (run on every rank's own GPU)
# ---------------------------------------------------------------------------


@pytest.mark.multigpu_static
def test_ddp_wrapping():
    """LookSAM step() works when model is wrapped in DistributedDataParallel."""
    model, ddp, opt, device = _make_ddp()
    x = torch.randn(4, 8, device=device)

    for _ in range(6):  # 2 SAM steps (at 0, 5), 4 fast steps
        opt.step(_closure(opt, ddp, x))

    assert opt.global_step == 6
    assert opt.sam_step == 2
    for name, p in model.named_parameters():
        assert not torch.isnan(p).any(), f"NaN in {name}"


@pytest.mark.multigpu_static
def test_ddp_no_nosync_kwarg():
    """LookSAM exposes no no_sync escape hatch (it would desync g_v).

    LookSAM reuses the ascent gradient inside g_v, so both passes must sync and
    there is intentionally no ddp_no_sync_fn parameter. Guard against it silently
    reappearing.
    """
    dm, device = _device()
    torch.manual_seed(42)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)
    ).to(device)
    ddp = DistributedDataParallel(model, device_ids=[dm.local_rank])
    base = torch.optim.Adam(ddp.parameters(), lr=1e-3)

    with pytest.raises(TypeError):
        LookSAM(ddp.parameters(), base_optimizer=base, ddp_no_sync_fn=ddp.no_sync)

    # And the normal construction (both passes synced) runs cleanly.
    opt = LookSAM(ddp.parameters(), base_optimizer=base, rho=0.1, alpha=0.7, k=5)
    x = torch.randn(4, 8, device=device)
    for _ in range(6):
        opt.step(_closure(opt, ddp, x))

    assert opt.sam_step == 2
    for p in model.parameters():
        assert not torch.isnan(p).any()


@pytest.mark.multigpu_static
def test_ddp_fast_steps():
    """Fast steps (t % k != 0) update params without error under DDP."""
    model, ddp, opt, device = _make_ddp()
    x = torch.randn(4, 8, device=device)
    closure = _closure(opt, ddp, x)

    opt.step(closure)  # SAM step — populates g_v
    params_after_sam = {n: p.data.clone() for n, p in model.named_parameters()}

    for _ in range(4):  # 4 fast steps (k=5)
        opt.step(closure)

    for name, p in model.named_parameters():
        assert not torch.equal(p.data, params_after_sam[name]), f"{name} frozen on fast step"
        assert not torch.isnan(p).any()


@pytest.mark.multigpu_static
def test_ddp_bf16():
    """bf16 autocast works under DDP."""
    model, ddp, opt, device = _make_ddp()
    x = torch.randn(4, 8, device=device)

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


@pytest.mark.multigpu_static
def test_ddp_looksam_layer():
    """LookLayerSAM works under DDP."""
    dm, device = _device()
    torch.manual_seed(42)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)
    ).to(device)
    ddp = DistributedDataParallel(model, device_ids=[dm.local_rank])
    base = torch.optim.Adam(ddp.parameters(), lr=1e-3)
    opt = LookLayerSAM(ddp.parameters(), base_optimizer=base, rho=0.1, k=3)
    x = torch.randn(4, 8, device=device)

    for _ in range(4):
        opt.step(_closure(opt, ddp, x))

    for p in model.parameters():
        assert not torch.isnan(p).any()


# ---------------------------------------------------------------------------
# Cross-rank sync regression — requires world_size >= 2
# ---------------------------------------------------------------------------


@pytest.mark.multigpu_static
def test_ddp_gradient_sync():
    """Params identical across ranks after a LookSAM cycle (NCCL, device-per-rank).

    The regression core of the descent-gradient sync contract: each rank is fed a
    *different* batch, so replicas can only stay bit-for-bit identical if both
    backward passes AllReduce. If the ascent gradient were not synced (the removed
    ``no_sync`` shortcut), each rank would cache a different ``g_v`` and the fast
    steps would drive the parameters apart — failing the assert below.
    """
    dm, device = _device()
    if dm.world_size < 2:
        pytest.skip("requires world_size >= 2 (torchrun --nproc-per-node >= 2)")

    # Identical initial weights on every rank (DDP also broadcasts at
    # construction, but seed identically so the check is self-contained).
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)
    ).to(device)
    ddp = DistributedDataParallel(model, device_ids=[dm.local_rank])
    base = torch.optim.Adam(ddp.parameters(), lr=1e-3)
    opt = LookSAM(ddp.parameters(), base_optimizer=base, rho=0.05, k=3)

    # Distinct data per rank — this is what makes the sync contract observable.
    torch.manual_seed(dm.rank + 1)
    x = torch.randn(4, 8, device=device)

    def closure():
        opt.zero_grad()
        ddp(x).sum().backward()
        return torch.tensor(0.0)

    for _ in range(6):  # 2 SAM steps (at 0, 3) + 4 fast steps, k=3
        opt.step(closure)

    for name, p in model.named_parameters():
        norm = p.data.norm().unsqueeze(0)
        gathered = [torch.zeros_like(norm) for _ in range(dm.world_size)]
        dist.all_gather(gathered, norm)
        norms = [g.item() for g in gathered]
        assert max(norms) - min(norms) < 1e-5, \
            f"{name} diverged across ranks after LookSAM: {norms}"
