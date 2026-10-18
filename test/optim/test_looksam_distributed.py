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

r"""Distributed tests for LookSAM: every parallel layout matches one device.

Each test trains the same model with LookSAM under one parallel layout and
compares the final weights to a single-device run over the full batch:

- DDP: plain parameters, gradients all-reduced by DDP.
- FSDP2: parameters and gradients are ``Shard(0)`` DTensors (uneven shards).
- HSDP: FSDP2 on a 2-D mesh, ``(Replicate(), Shard(0))`` placements.
- Domain parallel: inputs are ``ShardTensor`` s sharded along the sequence;
  parameters stay plain and ShardTensor reduces their gradients.
- Domain parallel x FSDP2: sequence sharding on one mesh axis, FSDP2 on the
  other.

Each rank sees different data, so the ranks only agree with the reference if
both backward passes of a SAM step are reduced and the global norms and dot
products are reduced over the shard dimensions.

Run with::

    torchrun --nproc-per-node <N> -m pytest --multigpu-static \
        test/optim/test_looksam_distributed.py
"""

import warnings

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.nn.parallel import DistributedDataParallel

from physicsnemo.distributed import DistributedManager

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from physicsnemo.experimental.optim import LookLayerSAM, LookSAM

STEPS, BATCH, SEQ = 6, 8, 16
HPARAMS = dict(rho=0.1, alpha=0.7, k=2)


# Relative error of the final weights against the one-device float64 run: the
# reduction order differs across layouts, so only float64 is compared tightly.
RTOL = {torch.float64: 1e-10, torch.float32: 1e-5, torch.bfloat16: 2e-2}
# ShardTensor's domain reduction of float64 weight gradients is accurate to
# about float32 precision (plain SGD shows the same 1e-8 difference), so the
# domain-parallel tests compare at that level.
DOMAIN_RTOL = 1e-7


def _model(device, dtype=torch.float64):
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(6, 16), torch.nn.Tanh(), torch.nn.Linear(16, 3)
    ).to(device, dtype)


def _data(device, dtype=torch.float64):
    """Full (batch, sequence, channel) inputs and targets, identical on all ranks."""
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(BATCH, SEQ, 6, generator=gen, dtype=torch.float64)
    y = torch.randn(BATCH, SEQ, 3, generator=gen, dtype=torch.float64)
    return x.to(device, dtype), y.to(device, dtype)


def _optimizer(cls, model):
    base = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    return cls(model.parameters(), base_optimizer=base, **HPARAMS)


def _train(opt, model, x, y, steps=STEPS):
    def closure():
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        return loss

    for _ in range(steps):
        opt.step(closure)


def _reference(cls, device):
    model = _model(device)
    _train(_optimizer(cls, model), model, *_data(device))
    return [p.detach() for p in model.parameters()]


def _full(p):
    return p.full_tensor() if isinstance(p, DTensor) else p.detach()


def _assert_matches(model, reference, dtype=torch.float64, rtol=None):
    pairs = [(_full(p).double(), q) for p, q in zip(model.parameters(), reference)]
    if dtype == torch.float64 and rtol is None:
        for p, q in pairs:
            torch.testing.assert_close(p, q, rtol=1e-10, atol=1e-10)
        return
    err = sum((p - q).square().sum() for p, q in pairs).sqrt()
    rtol = RTOL[dtype] if rtol is None else rtol
    assert err / sum(q.square().sum() for _, q in pairs).sqrt() < rtol


def _batch_slice(x, index, count):
    n = x.shape[0] // count
    return x[index * n : (index + 1) * n]


def _dm(min_world=2, even=False):
    dm = DistributedManager()
    if dm.world_size < min_world or (even and dm.world_size % 2):
        pytest.skip(f"requires an even world size >= {min_world}")
    return dm


DTYPES = [torch.float64, torch.float32, torch.bfloat16]


@pytest.mark.multigpu_static
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("cls", [LookSAM, LookLayerSAM])
def test_ddp(cls, dtype):
    dm = _dm()
    x, y = _data(dm.device, dtype)
    model = _model(dm.device, dtype)
    ddp = DistributedDataParallel(
        model, device_ids=[dm.local_rank] if dm.device.type == "cuda" else None
    )
    opt = _optimizer(cls, ddp)
    _train(
        opt,
        ddp,
        _batch_slice(x, dm.rank, dm.world_size),
        _batch_slice(y, dm.rank, dm.world_size),
    )
    _assert_matches(model, _reference(cls, dm.device), dtype)


@pytest.mark.multigpu_static
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("cls", [LookSAM, LookLayerSAM])
def test_fsdp2(cls, dtype):
    dm = _dm()
    mesh = init_device_mesh(dm.device.type, (dm.world_size,))
    x, y = _data(dm.device, dtype)
    model = fully_shard(_model(dm.device, dtype), mesh=mesh)
    opt = _optimizer(cls, model)
    _train(
        opt,
        model,
        _batch_slice(x, dm.rank, dm.world_size),
        _batch_slice(y, dm.rank, dm.world_size),
    )
    assert all(isinstance(v, DTensor) for v in opt.state_dict()["g_v"][0])
    _assert_matches(model, _reference(cls, dm.device), dtype)


@pytest.mark.multigpu_static
def test_fsdp2_mixed_precision():
    """FSDP2 mixed precision: bfloat16 compute, float32 sharded weights and g_v."""
    from torch.distributed.fsdp import MixedPrecisionPolicy

    dm = _dm()
    mesh = init_device_mesh(dm.device.type, (dm.world_size,))
    x, y = _data(dm.device, torch.float32)
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    model = fully_shard(_model(dm.device, torch.float32), mesh=mesh, mp_policy=policy)
    opt = _optimizer(LookSAM, model)
    _train(
        opt,
        model,
        _batch_slice(x, dm.rank, dm.world_size),
        _batch_slice(y, dm.rank, dm.world_size),
    )
    assert all(v.dtype == torch.float32 for v in opt.state_dict()["g_v"][0])
    _assert_matches(model, _reference(LookSAM, dm.device), torch.bfloat16)


@pytest.mark.multigpu_static
def test_fsdp2_state_dict_roundtrip():
    """Reloading the DTensor g_v state mid-cycle continues the same trajectory."""
    dm = _dm()
    mesh = init_device_mesh(dm.device.type, (dm.world_size,))
    x, y = _data(dm.device)
    x = _batch_slice(x, dm.rank, dm.world_size)
    y = _batch_slice(y, dm.rank, dm.world_size)

    model = fully_shard(_model(dm.device), mesh=mesh)
    opt = _optimizer(LookSAM, model)
    _train(opt, model, x, y, steps=3)
    state = opt.state_dict()
    opt2 = _optimizer(LookSAM, model)
    opt2.load_state_dict(state)
    _train(opt2, model, x, y, steps=STEPS - 3)
    _assert_matches(model, _reference(LookSAM, dm.device))


@pytest.mark.multigpu_static
@pytest.mark.parametrize("cls", [LookSAM, LookLayerSAM])
def test_hsdp(cls):
    dm = _dm(min_world=4, even=True)
    mesh = init_device_mesh(
        dm.device.type, (2, dm.world_size // 2), mesh_dim_names=("replicate", "shard")
    )
    x, y = _data(dm.device)
    model = fully_shard(_model(dm.device), mesh=mesh)
    assert next(model.parameters()).placements == (Replicate(), Shard(0))
    opt = _optimizer(cls, model)
    _train(
        opt,
        model,
        _batch_slice(x, dm.rank, dm.world_size),
        _batch_slice(y, dm.rank, dm.world_size),
    )
    _assert_matches(model, _reference(cls, dm.device))


def _scatter(x, mesh):
    from physicsnemo.domain_parallel import scatter_tensor

    src = torch.distributed.get_global_rank(mesh.get_group(), 0)
    return scatter_tensor(
        x, src, mesh, placements=(Shard(1),), global_shape=x.shape, dtype=x.dtype
    )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("cls", [LookSAM, LookLayerSAM])
def test_domain_parallel(cls):
    """ShardTensor inputs sharded along the sequence; plain parameters."""
    dm = _dm()
    if dm.device.type != "cuda":
        pytest.skip("ShardTensor requires CUDA")
    mesh = init_device_mesh("cuda", (dm.world_size,), mesh_dim_names=("domain",))
    x, y = _data(dm.device)
    model = _model(dm.device)
    opt = _optimizer(cls, model)
    _train(opt, model, _scatter(x, mesh), _scatter(y, mesh))
    _assert_matches(model, _reference(cls, dm.device), rtol=DOMAIN_RTOL)


@pytest.mark.multigpu_static
@pytest.mark.parametrize("cls", [LookSAM, LookLayerSAM])
def test_domain_parallel_fsdp2(cls):
    """Sequence sharding on the domain axis, FSDP2 on the data-parallel axis."""
    dm = _dm(min_world=4, even=True)
    if dm.device.type != "cuda":
        pytest.skip("ShardTensor requires CUDA")
    mesh = init_device_mesh(
        "cuda", (2, dm.world_size // 2), mesh_dim_names=("ddp", "domain")
    )
    ddp_rank = mesh.get_local_rank("ddp")
    x, y = _data(dm.device)
    x, y = (_scatter(_batch_slice(t, ddp_rank, 2), mesh["domain"]) for t in (x, y))
    model = fully_shard(_model(dm.device), mesh=mesh["ddp"])
    opt = _optimizer(cls, model)
    _train(opt, model, x, y)
    _assert_matches(model, _reference(cls, dm.device), rtol=DOMAIN_RTOL)
