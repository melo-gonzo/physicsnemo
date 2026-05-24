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

"""Unit tests for :class:`TransolverPretrainBackbone`.

Covers the four core invariants of PR 2:

(a) **Output shape** is ``(B, N, n_steps * 3)``.
(b) **Backbone-only round-trip** — saving the wrapper's checkpoint and
    loading it into a plain :class:`Transolver` via
    :func:`load_pretrained_backbone` (with
    ``key_remap={"transolver.": ""}`` and
    ``exclude_layers=["trajectory_head"]``) preserves backbone weights
    bit-exact and leaves the fine-tune model's head untouched. This is
    the central PR 2 / PR 2.5 round-trip.
(c) **`.mdlus` round-trip** — wrapper saves and re-loads via
    :meth:`physicsnemo.core.module.Module.from_checkpoint` with
    bit-exact param recovery.
(d) **Gradient flow** — backward through one forward pass populates
    grads on every backbone *and* head parameter (no detach()).
"""

from __future__ import annotations

import pytest
import torch

from physicsnemo.experimental.pnm_pretraining.models import (
    TrajectoryHead,
    TransolverPretrainBackbone,
)
from physicsnemo.models.transolver import Transolver
from physicsnemo.utils.checkpoint import load_pretrained_backbone
from test.conftest import requires_module

# Small hyperparameters shared across tests. Picked to be the smallest
# valid Transolver config that still exercises the full forward path:
# n_layers >= 2 so the "last block" rewrite is visible, n_head divides
# n_hidden, slice_num >= 1, etc.
_BACKBONE_KWARGS: dict = {
    "functional_dim": 4,
    "embedding_dim": 7,
    "n_layers": 2,
    "n_hidden": 32,
    "n_head": 4,
    "slice_num": 16,
    "use_te": False,
}

_BATCH = 2
_N_POINTS = 64
_N_STEPS = 3


def _make_inputs(
    *, batch: int = _BATCH, n_points: int = _N_POINTS, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random ``(embedding, fx)`` of the wrapper's expected shapes."""
    g = torch.Generator().manual_seed(seed)
    embedding = torch.randn(
        batch, n_points, _BACKBONE_KWARGS["embedding_dim"], generator=g
    )
    fx = torch.randn(batch, n_points, _BACKBONE_KWARGS["functional_dim"], generator=g)
    return embedding, fx


# ---------------------------------------------------------------------
# Test (a): output shape.
# ---------------------------------------------------------------------


@requires_module("torch")
def test_wrapper_output_shape() -> None:
    """Wrapper emits ``(B, N, n_steps * 3)`` for the standard inputs."""
    model = TransolverPretrainBackbone(n_steps=_N_STEPS, **_BACKBONE_KWARGS).eval()
    embedding, fx = _make_inputs()
    with torch.no_grad():
        out = model(embedding=embedding, fx=fx)
    assert out.shape == (_BATCH, _N_POINTS, _N_STEPS * 3), (
        f"got {tuple(out.shape)}, expected ({_BATCH}, {_N_POINTS}, {_N_STEPS * 3})"
    )
    assert out.dtype == embedding.dtype


@requires_module("torch")
def test_wrapper_rejects_inconsistent_out_dim() -> None:
    """Explicit ``out_dim`` that disagrees with ``n_steps * 3`` raises."""
    with pytest.raises(ValueError, match="disagrees with n_steps"):
        TransolverPretrainBackbone(
            n_steps=3,
            out_dim=4,  # disagrees with 3 * 3 = 9
            **_BACKBONE_KWARGS,
        )


@requires_module("torch")
def test_wrapper_accepts_matching_out_dim() -> None:
    """Explicit ``out_dim`` matching ``n_steps * 3`` is tolerated."""
    model = TransolverPretrainBackbone(
        n_steps=2,
        out_dim=6,  # matches 2 * 3 = 6
        **_BACKBONE_KWARGS,
    ).eval()
    embedding, fx = _make_inputs()
    out = model(embedding=embedding, fx=fx)
    assert out.shape == (_BATCH, _N_POINTS, 6)


@requires_module("torch")
def test_trajectory_head_shape() -> None:
    """``TrajectoryHead`` projects ``(B, N, in_dim)`` to ``(B, N, out_dim)``."""
    head = TrajectoryHead(in_dim=32, out_dim=9, hidden_dim=16, n_layers=1)
    x = torch.randn(2, 8, 32)
    y = head(x)
    assert y.shape == (2, 8, 9)


# ---------------------------------------------------------------------
# Test (b): backbone-only round-trip via load_pretrained_backbone.
# ---------------------------------------------------------------------


@requires_module("torch")
def test_backbone_round_trip_via_load_pretrained_backbone(tmp_path) -> None:
    """Pretrain-then-fine-tune round trip — the PR 2 / PR 2.5 contract.

    Workflow:

    1. Build a wrapper (pretrain) and a plain Transolver (fine-tune)
       with **identical** backbone hyperparameters but different
       ``out_dim`` (4 in fine-tune, ignored in wrapper).
    2. Save the wrapper to ``.mdlus``.
    3. Snapshot the fine-tune's pre-load head params (``ln_mlp2``).
    4. Load the wrapper checkpoint into the fine-tune via
       :func:`load_pretrained_backbone` with
       ``key_remap={"transolver.": ""}`` (strip the wrapper prefix) and
       ``exclude_layers=["trajectory_head"]`` (drop the head).
    5. Assert:
       - 3 specific backbone params on fine-tune now equal the wrapper's
         counterparts bit-exact.
       - The fine-tune's head params (``blocks.{n-1}.ln_mlp2``) are
         **unchanged** from their initialization (the load did not
         touch them).
       - Wrapper and fine-tune produce **different** outputs on the
         same input (confirms the head differs).
    """
    torch.manual_seed(0)
    wrapper = TransolverPretrainBackbone(n_steps=_N_STEPS, **_BACKBONE_KWARGS).eval()

    torch.manual_seed(1)  # different init for the fine-tune
    plain = Transolver(out_dim=4, **_BACKBONE_KWARGS).eval()

    # 2. Save wrapper.
    ckpt_path = tmp_path / "pretrain.mdlus"
    wrapper.save(str(ckpt_path))
    assert ckpt_path.exists()

    # 3. Snapshot the fine-tune's pre-load head params.
    n_blocks = _BACKBONE_KWARGS["n_layers"]
    head_keys = [
        f"blocks.{n_blocks - 1}.ln_mlp2.0.weight",
        f"blocks.{n_blocks - 1}.ln_mlp2.0.bias",
        f"blocks.{n_blocks - 1}.ln_mlp2.1.weight",
        f"blocks.{n_blocks - 1}.ln_mlp2.1.bias",
    ]
    plain_sd_pre = {k: plain.state_dict()[k].clone() for k in head_keys}

    # 4. Load. Plain Transolver does not have the `transolver.` prefix,
    # so strip it; drop the wrapper's `trajectory_head.*` keys.
    report = load_pretrained_backbone(
        plain,
        str(ckpt_path),
        key_remap={"transolver.": ""},
        exclude_layers=["trajectory_head"],
        strict=False,
        verbose=False,
    )

    # The wrapper's checkpoint has no `blocks.{n-1}.ln_mlp2.*` keys
    # (they were replaced by Identity, which has no params), so the
    # plain Transolver's head is in `missing_in_source` (informational).
    # The wrapper's `trajectory_head.*` keys must all be in the
    # exclude bucket. Backbone keys (preprocess + non-last blocks +
    # last-block ln_1 / Attn / ln_mlp1) must all be in `loaded`.
    assert any("trajectory_head" in k for k in report["skipped_excluded"]), (
        f"trajectory_head should be excluded; report={report}"
    )
    # No shape mismatches.
    assert report["skipped_shape_mismatch"] == [], report["skipped_shape_mismatch"]
    # The loaded set must contain the backbone preprocess weights.
    assert "preprocess.layers.0.weight" in report["loaded"]
    assert f"blocks.{n_blocks - 1}.ln_1.weight" in report["loaded"]

    # 5a. Three specific backbone params match bit-exact.
    wrapper_sd = wrapper.state_dict()
    plain_sd_post = plain.state_dict()
    for plain_key in [
        "preprocess.layers.0.weight",
        f"blocks.{n_blocks - 1}.ln_1.weight",
        f"blocks.{n_blocks - 1}.ln_mlp1.0.weight",
    ]:
        wrapper_key = f"transolver.{plain_key}"
        assert torch.equal(wrapper_sd[wrapper_key], plain_sd_post[plain_key]), (
            f"backbone param {plain_key} did not load bit-exact"
        )

    # 5b. The fine-tune's head params are unchanged from init.
    for k in head_keys:
        assert torch.equal(plain_sd_pre[k], plain_sd_post[k]), (
            f"fine-tune head param {k} was overwritten by the backbone load"
        )

    # 5c. Outputs differ on the same input.
    embedding, fx = _make_inputs()
    with torch.no_grad():
        out_wrapper = wrapper(embedding=embedding, fx=fx)
        out_plain = plain(fx, embedding=embedding)
    assert out_wrapper.shape == (_BATCH, _N_POINTS, _N_STEPS * 3)
    assert out_plain.shape == (_BATCH, _N_POINTS, 4)
    # Different shapes already confirm the heads differ; a more
    # interesting check: feed plain's hidden through wrapper's head
    # and compare to wrapper's full forward — but that's not the
    # contract. Confirming non-equal shapes is sufficient.


# ---------------------------------------------------------------------
# Test (c): .mdlus round-trip via physicsnemo.Module.
# ---------------------------------------------------------------------


@requires_module("torch")
def test_mdlus_round_trip(tmp_path) -> None:
    """Save wrapper to ``.mdlus``, load into a fresh wrapper, params match bit-exact."""
    torch.manual_seed(0)
    model_a = TransolverPretrainBackbone(n_steps=_N_STEPS, **_BACKBONE_KWARGS).eval()

    ckpt_path = tmp_path / "wrapper.mdlus"
    model_a.save(str(ckpt_path))

    model_b = TransolverPretrainBackbone.from_checkpoint(str(ckpt_path)).eval()

    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    assert set(sd_a.keys()) == set(sd_b.keys()), (
        f"state-dict key mismatch: only-in-a={set(sd_a) - set(sd_b)}, "
        f"only-in-b={set(sd_b) - set(sd_a)}"
    )
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_b[k]), f"param {k} did not round-trip bit-exact"

    # Same forward output on the same input (confirms architectural identity).
    embedding, fx = _make_inputs()
    with torch.no_grad():
        out_a = model_a(embedding=embedding, fx=fx)
        out_b = model_b(embedding=embedding, fx=fx)
    assert torch.equal(out_a, out_b)


# ---------------------------------------------------------------------
# Test (d): gradient flow.
# ---------------------------------------------------------------------


@requires_module("torch")
def test_gradients_reach_backbone_and_head() -> None:
    """Backward populates grads on backbone *and* head params."""
    model = TransolverPretrainBackbone(n_steps=_N_STEPS, **_BACKBONE_KWARGS).train()
    embedding, fx = _make_inputs()
    out = model(embedding=embedding, fx=fx)
    out.sum().backward()

    # Sample one backbone param (early preprocess layer) and one head
    # param (trajectory head's first linear). Both must have grad.
    backbone_param = model.transolver.preprocess.layers[0].weight
    assert backbone_param.grad is not None, "preprocess.layers[0].weight has no grad"
    assert torch.isfinite(backbone_param.grad).all()
    assert backbone_param.grad.abs().sum() > 0

    head_param = model.trajectory_head.layers[0].weight
    assert head_param.grad is not None, "trajectory_head.layers[0].weight has no grad"
    assert torch.isfinite(head_param.grad).all()
    assert head_param.grad.abs().sum() > 0

    # Confirm every parameter that is `requires_grad=True` actually got
    # a grad — defensive against future changes that accidentally
    # detach a sub-block.
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"param {name} has no grad after backward"
