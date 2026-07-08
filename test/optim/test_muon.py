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

import copy

import pytest
import torch

from physicsnemo.optim import Muon

# torch.optim.Muon is the per-parameter reference implementation we batch.
_TORCH_MUON = getattr(torch.optim, "Muon", None)
_HAS_TORCH_MUON = _TORCH_MUON is not None


def _make_params(shapes, device, seed=0):
    """Create a list of 2-D parameters with reproducible random values."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return [
        torch.nn.Parameter(torch.randn(*s, generator=gen).to(device)) for s in shapes
    ]


def _make_grads(shapes, device, seed):
    """Create a list of gradients aligned with ``shapes``."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return [torch.randn(*s, generator=gen).to(device) for s in shapes]


@pytest.mark.skipif(not _HAS_TORCH_MUON, reason="torch.optim.Muon unavailable")
@pytest.mark.parametrize("nesterov", [True, False])
@pytest.mark.parametrize("adjust_lr_fn", ["original", "match_rms_adamw"])
@pytest.mark.parametrize("weight_decay", [0.01, 0.0])
def test_matches_torch_muon(device, nesterov, adjust_lr_fn, weight_decay):
    """Fused Muon matches torch.optim.Muon step-for-step within tolerance.

    Shapes include square, wide, tall, and a repeated shape so the batched
    Newton-Schulz path (group size > 1) is exercised. ``weight_decay=0``
    covers the skipped decay multiply.
    """
    shapes = [(8, 8), (8, 8), (8, 16), (16, 8)]

    ref_params = _make_params(shapes, device, seed=1)
    fused_params = [torch.nn.Parameter(p.detach().clone()) for p in ref_params]

    kwargs = dict(
        lr=0.02,
        weight_decay=weight_decay,
        momentum=0.95,
        nesterov=nesterov,
        adjust_lr_fn=adjust_lr_fn,
    )
    ref_opt = _TORCH_MUON(ref_params, **kwargs)
    fused_opt = Muon(fused_params, **kwargs)

    for step in range(5):
        grads = _make_grads(shapes, device, seed=100 + step)
        for p, g in zip(ref_params, grads):
            p.grad = g.clone()
        for p, g in zip(fused_params, grads):
            p.grad = g.clone()
        ref_opt.step()
        fused_opt.step()

    for ref_p, fused_p in zip(ref_params, fused_params):
        torch.testing.assert_close(fused_p, ref_p, atol=1e-3, rtol=1e-3)


def test_lazy_adjust_lr_proxy_resolves():
    """The private torch._adjust_lr is reachable via the lazy OptionalImport.

    physicsnemo.optim.muon imports torch.optim._muon lazily so a future
    rename/removal fails at step() runtime rather than at module import time.
    This asserts the proxy resolves the symbol on the installed torch.
    """
    from physicsnemo.optim.muon import _torch_muon_internal

    assert callable(_torch_muon_internal._adjust_lr)


def test_group_params_by_shape(device):
    """Equally-shaped params bucket together; distinct shapes stay separate."""
    shapes = [(8, 8), (8, 8), (8, 16), (16, 8), (8, 8)]
    params = _make_params(shapes, device, seed=2)

    groups = Muon._group_params_by_shape(params)

    sizes = sorted(len(idxs) for idxs in groups.values())
    # (8,8) x3, (8,16) x1, (16,8) x1
    assert sizes == [1, 1, 3]
    # The repeated (8,8) shape collapses to one group with the right indices.
    eight = [idxs for key, idxs in groups.items() if key[0] == (8, 8)]
    assert eight == [[0, 1, 4]]


def test_state_dict_roundtrip(device):
    """Saving and restoring state (incl. momentum buffers) resumes identically."""
    shapes = [(8, 8), (8, 16)]
    params = _make_params(shapes, device, seed=3)
    opt = Muon(params, lr=0.02, weight_decay=0.01, adjust_lr_fn="match_rms_adamw")

    # Two steps to populate momentum buffers.
    for step in range(2):
        grads = _make_grads(shapes, device, seed=200 + step)
        for p, g in zip(params, grads):
            p.grad = g.clone()
        opt.step()

    saved_state = copy.deepcopy(opt.state_dict())
    snapshot = [p.detach().clone() for p in params]

    # Continue one more step on the original optimizer (the reference).
    final_grads = _make_grads(shapes, device, seed=999)
    for p, g in zip(params, final_grads):
        p.grad = g.clone()
    opt.step()
    reference_final = [p.detach().clone() for p in params]

    # Fresh optimizer restored from the snapshot + saved state must match.
    restored_params = [torch.nn.Parameter(s.clone()) for s in snapshot]
    restored_opt = Muon(
        restored_params, lr=0.02, weight_decay=0.01, adjust_lr_fn="match_rms_adamw"
    )
    restored_opt.load_state_dict(saved_state)
    for p, g in zip(restored_params, final_grads):
        p.grad = g.clone()
    restored_opt.step()

    for ref_p, restored_p in zip(reference_final, restored_params):
        torch.testing.assert_close(restored_p, ref_p, atol=1e-6, rtol=1e-6)


def test_rejects_non_2d_params(device):
    """Muon only supports 2-D parameters."""
    param_1d = torch.nn.Parameter(torch.randn(8).to(device))
    with pytest.raises(ValueError, match="2D"):
        Muon([param_1d], lr=0.02)


def test_ns_scratch_buffer_reused_across_steps(device):
    """The bf16 Newton-Schulz stacks are allocated once and reused every step."""
    shapes = [(8, 8), (8, 8), (8, 16)]
    params = _make_params(shapes, device, seed=4)
    opt = Muon(params, lr=0.02)

    ptrs = None
    for step in range(3):
        grads = _make_grads(shapes, device, seed=400 + step)
        for p, g in zip(params, grads):
            p.grad = g.clone()
        opt.step()
        if step == 0:
            # One buffer per shape group: (2, 8, 8) and (1, 8, 16).
            assert len(opt._ns_buffers) == 2
            assert sorted(tuple(b.shape) for b in opt._ns_buffers.values()) == [
                (1, 8, 16),
                (2, 8, 8),
            ]
            assert all(b.dtype == torch.bfloat16 for b in opt._ns_buffers.values())
            ptrs = {k: v.data_ptr() for k, v in opt._ns_buffers.items()}

    assert {k: v.data_ptr() for k, v in opt._ns_buffers.items()} == ptrs


@pytest.mark.skipif(not _HAS_TORCH_MUON, reason="torch.optim.Muon unavailable")
def test_matches_torch_muon_changing_grad_pattern(device):
    """Steps stay correct when the set of params with gradients changes.

    Shrinking/growing a shape group across steps forces the cached
    Newton-Schulz stack to be reallocated, which must not affect results.
    """
    shapes = [(8, 8), (8, 8), (8, 8)]
    ref_params = _make_params(shapes, device, seed=5)
    fused_params = [torch.nn.Parameter(p.detach().clone()) for p in ref_params]

    kwargs = dict(lr=0.02, weight_decay=0.01, momentum=0.95)
    ref_opt = _TORCH_MUON(ref_params, **kwargs)
    fused_opt = Muon(fused_params, **kwargs)

    # Group size goes 3 -> 2 -> 3 across steps.
    grad_patterns = [{0, 1, 2}, {0, 2}, {0, 1, 2}]
    for step, with_grad in enumerate(grad_patterns):
        grads = _make_grads(shapes, device, seed=500 + step)
        for param_list in (ref_params, fused_params):
            for i, p in enumerate(param_list):
                p.grad = grads[i].clone() if i in with_grad else None
        ref_opt.step()
        fused_opt.step()

    for ref_p, fused_p in zip(ref_params, fused_params):
        torch.testing.assert_close(fused_p, ref_p, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not _HAS_TORCH_MUON, reason="torch.optim.Muon unavailable")
def test_matches_torch_muon_multiple_param_groups(device):
    """Same-shape params in different param groups get independent treatment.

    Two groups containing the same (8, 8) shape but different group sizes
    exercise the per-group scratch-buffer keying.
    """
    shapes_a = [(8, 8), (8, 8)]
    shapes_b = [(8, 8), (8, 16)]
    ref_a = _make_params(shapes_a, device, seed=6)
    ref_b = _make_params(shapes_b, device, seed=7)
    fused_a = [torch.nn.Parameter(p.detach().clone()) for p in ref_a]
    fused_b = [torch.nn.Parameter(p.detach().clone()) for p in ref_b]

    def _groups(a, b):
        return [
            {"params": a, "lr": 0.02, "weight_decay": 0.01},
            {"params": b, "lr": 0.005, "weight_decay": 0.0},
        ]

    ref_opt = _TORCH_MUON(_groups(ref_a, ref_b), lr=0.02)
    fused_opt = Muon(_groups(fused_a, fused_b), lr=0.02)

    for step in range(3):
        grads_a = _make_grads(shapes_a, device, seed=600 + step)
        grads_b = _make_grads(shapes_b, device, seed=700 + step)
        for params, grads in (
            (ref_a, grads_a),
            (ref_b, grads_b),
            (fused_a, grads_a),
            (fused_b, grads_b),
        ):
            for p, g in zip(params, grads):
                p.grad = g.clone()
        ref_opt.step()
        fused_opt.step()

    for ref_p, fused_p in zip(ref_a + ref_b, fused_a + fused_b):
        torch.testing.assert_close(fused_p, ref_p, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not _HAS_TORCH_MUON, reason="torch.optim.Muon unavailable")
def test_matches_torch_muon_non_contiguous_grads(device):
    """Non-contiguous gradients (e.g. transposed views) are handled correctly."""
    shapes = [(8, 16), (8, 16)]
    ref_params = _make_params(shapes, device, seed=8)
    fused_params = [torch.nn.Parameter(p.detach().clone()) for p in ref_params]

    ref_opt = _TORCH_MUON(ref_params, lr=0.02)
    fused_opt = Muon(fused_params, lr=0.02)

    for step in range(2):
        gen = torch.Generator(device="cpu").manual_seed(800 + step)
        bases = [torch.randn(16, 8, generator=gen).to(device) for _ in shapes]
        for p, base in zip(ref_params, bases):
            p.grad = base.t()
        for p, base in zip(fused_params, bases):
            p.grad = base.clone().t()
        assert all(not p.grad.is_contiguous() for p in fused_params)
        ref_opt.step()
        fused_opt.step()

    for ref_p, fused_p in zip(ref_params, fused_params):
        torch.testing.assert_close(fused_p, ref_p, atol=1e-3, rtol=1e-3)


def test_bfloat16_params_step(device):
    """bf16 parameters step without dtype errors and stay finite.

    Exercises the no-cast branch where the Newton-Schulz output dtype already
    matches the parameter dtype.
    """
    shapes = [(8, 8), (8, 8)]
    params = [
        torch.nn.Parameter(p.detach().to(torch.bfloat16))
        for p in _make_params(shapes, device, seed=9)
    ]
    before = [p.detach().clone() for p in params]
    opt = Muon(params, lr=0.02)

    grads = [g.to(torch.bfloat16) for g in _make_grads(shapes, device, seed=900)]
    for p, g in zip(params, grads):
        p.grad = g
    opt.step()

    for p, b in zip(params, before):
        assert p.dtype == torch.bfloat16
        assert torch.isfinite(p).all()
        assert not torch.equal(p, b)
