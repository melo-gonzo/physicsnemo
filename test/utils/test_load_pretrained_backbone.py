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

"""Tests for :func:`physicsnemo.utils.checkpoint.load_pretrained_backbone`.

Mirrors the spec in ``geopt-physicsnemo-engineering-plan.md`` §1.4 PR 2.5:

a. full match
b. prefix-stripped match (e.g. ``backbone.`` → ``""``)
c. callable key remap
d. ``exclude_layers`` filtering
e. graceful failure on shape mismatch (default + strict)
f. missing-key reporting
g. round-trip via a ``.pt`` file
h. wrapped checkpoint format (``{"model_state_dict": ..., "epoch": 5}``)
i. DDP-wrapped target (gated on ``WORLD_SIZE > 1``)
j. ``verbose=False`` is quiet
"""

import logging
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.checkpoint import load_pretrained_backbone

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_target() -> nn.Module:
    """Plain backbone-only target model.

    ``backbone.0`` is a ``Linear(8, 4)``; ``backbone.1`` is a ``Linear(4, 4)``.
    No head — fine-tuner will add its own elsewhere.
    """
    return nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 4))


def _make_target_with_head() -> nn.Module:
    """Target that does have a head (``Linear(4, 2)``).

    Useful for the ``exclude_layers`` test, which checks that exclusion
    leaves the *target's* head at its random init (rather than overwriting
    it with the source's head).
    """
    backbone = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 4))
    head = nn.Linear(4, 2)
    return nn.ModuleDict({"backbone": backbone, "head": head})


def _save_pt(state_dict: dict, path: Path) -> str:
    """Persist a state dict to a ``.pt`` file and return the string path."""
    torch.save(state_dict, path)
    return str(path)


def _ensure_dist_init() -> None:
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()


# ---------------------------------------------------------------------------
# (a) full match
# ---------------------------------------------------------------------------


def test_full_match(tmp_path: Path) -> None:
    """Every source key matches a target key by name and shape."""
    _ensure_dist_init()
    src_model = _make_target()
    tgt_model = _make_target()

    weights_path = _save_pt(src_model.state_dict(), tmp_path / "src.pt")

    # Seed-independent: assert tgt and src are *different* before load
    # (random init differs between the two constructions).
    src_sd = src_model.state_dict()
    tgt_sd_before = tgt_model.state_dict()
    assert not torch.allclose(src_sd["0.weight"], tgt_sd_before["0.weight"])

    report = load_pretrained_backbone(tgt_model, weights_path, verbose=False)

    assert set(report["loaded"]) == set(src_sd.keys())
    assert report["skipped_excluded"] == []
    assert report["skipped_missing_target"] == []
    assert report["skipped_shape_mismatch"] == []
    assert report["missing_in_source"] == []

    # State now matches source.
    tgt_sd_after = tgt_model.state_dict()
    for k, v in src_sd.items():
        assert torch.equal(tgt_sd_after[k], v), f"{k} did not match after load"


# ---------------------------------------------------------------------------
# (b) prefix-stripped match
# ---------------------------------------------------------------------------


def test_prefix_strip_match(tmp_path: Path) -> None:
    """``key_remap={"backbone.": ""}`` strips a prefix from the source side."""
    _ensure_dist_init()
    # Source has a "backbone." prefix on every key; target does not.
    src_model = nn.ModuleDict({"backbone": _make_target()})
    src_sd = src_model.state_dict()
    assert all(k.startswith("backbone.") for k in src_sd)

    weights_path = _save_pt(src_sd, tmp_path / "src.pt")

    tgt_model = _make_target()
    report = load_pretrained_backbone(
        tgt_model,
        weights_path,
        key_remap={"backbone.": ""},
        verbose=False,
    )

    expected_loaded = {k.removeprefix("backbone.") for k in src_sd}
    assert set(report["loaded"]) == expected_loaded
    assert report["skipped_missing_target"] == []
    assert report["skipped_shape_mismatch"] == []

    tgt_sd = tgt_model.state_dict()
    for src_key, src_val in src_sd.items():
        tgt_key = src_key.removeprefix("backbone.")
        assert torch.equal(tgt_sd[tgt_key], src_val)


# ---------------------------------------------------------------------------
# (c) callable key remap
# ---------------------------------------------------------------------------


def test_callable_remap(tmp_path: Path) -> None:
    """A callable ``key_remap`` produces equivalent behavior to a prefix dict."""
    _ensure_dist_init()
    # Source has a "module." prefix (DataParallel-style).
    src_model = nn.ModuleDict({"module": _make_target()})
    src_sd = src_model.state_dict()
    weights_path = _save_pt(src_sd, tmp_path / "src.pt")

    tgt_model = _make_target()
    report = load_pretrained_backbone(
        tgt_model,
        weights_path,
        key_remap=lambda k: k.replace("module.", ""),
        verbose=False,
    )

    assert set(report["loaded"]) == {k.replace("module.", "") for k in src_sd}

    tgt_sd = tgt_model.state_dict()
    for k, v in src_sd.items():
        assert torch.equal(tgt_sd[k.replace("module.", "")], v)


# ---------------------------------------------------------------------------
# (d) exclude_layers
# ---------------------------------------------------------------------------


def test_exclude_layers(tmp_path: Path) -> None:
    """``exclude_layers`` skips matching source keys; target retains its init."""
    _ensure_dist_init()
    # Source: backbone + head.
    src_model = _make_target_with_head()
    src_sd = src_model.state_dict()
    head_keys = [k for k in src_sd if k.startswith("head.")]
    backbone_keys = [k for k in src_sd if k.startswith("backbone.")]
    assert head_keys, "fixture invariant: source has head keys"
    assert backbone_keys, "fixture invariant: source has backbone keys"

    weights_path = _save_pt(src_sd, tmp_path / "src.pt")

    # Target: same shape (same fixture) but freshly initialized.
    tgt_model = _make_target_with_head()
    tgt_head_before = {
        k: v.clone() for k, v in tgt_model.state_dict().items() if k.startswith("head.")
    }

    report = load_pretrained_backbone(
        tgt_model,
        weights_path,
        exclude_layers=("head",),
        verbose=False,
    )

    # Head keys are excluded; backbone keys are loaded.
    assert set(report["skipped_excluded"]) == set(head_keys)
    assert set(report["loaded"]) == set(backbone_keys)
    assert report["skipped_missing_target"] == []
    assert report["skipped_shape_mismatch"] == []

    tgt_sd_after = tgt_model.state_dict()
    # Backbone params now match source.
    for k in backbone_keys:
        assert torch.equal(tgt_sd_after[k], src_sd[k])
    # Head params were *not* overwritten — they still match the pre-load values.
    for k, v in tgt_head_before.items():
        assert torch.equal(tgt_sd_after[k], v)


# ---------------------------------------------------------------------------
# (e) graceful failure on shape mismatch
# ---------------------------------------------------------------------------


def test_shape_mismatch_default_lenient(tmp_path: Path) -> None:
    """With ``strict=False`` (default), shape mismatches are reported, not raised."""
    _ensure_dist_init()
    # Source: Linear(8, 4) + Linear(4, 4).
    src_sd = _make_target().state_dict()
    weights_path = _save_pt(src_sd, tmp_path / "src.pt")

    # Target: first layer has output dim 6 instead of 4. Concretely:
    #   src "0.weight": (4, 8) vs tgt "0.weight": (6, 8)  -> mismatch
    #   src "0.bias":   (4,)   vs tgt "0.bias":   (6,)    -> mismatch
    #   src "1.weight": (4, 4) vs tgt "1.weight": (4, 6)  -> mismatch
    #   src "1.bias":   (4,)   vs tgt "1.bias":   (4,)    -> matches
    # So three of the four source keys mismatch on shape, one lands cleanly.
    tgt_model = nn.Sequential(nn.Linear(8, 6), nn.Linear(6, 4))

    report = load_pretrained_backbone(tgt_model, weights_path, verbose=False)

    expected_mismatch = {"0.weight", "0.bias", "1.weight"}
    assert set(report["skipped_shape_mismatch"]) == expected_mismatch
    assert report["loaded"] == ["1.bias"]
    assert report["skipped_missing_target"] == []


def test_shape_mismatch_strict_raises(tmp_path: Path) -> None:
    """With ``strict=True``, a shape mismatch raises ``RuntimeError``."""
    _ensure_dist_init()
    src_sd = _make_target().state_dict()
    weights_path = _save_pt(src_sd, tmp_path / "src.pt")
    tgt_model = nn.Sequential(nn.Linear(8, 6), nn.Linear(6, 4))

    with pytest.raises(RuntimeError, match="shape mismatch"):
        load_pretrained_backbone(tgt_model, weights_path, strict=True, verbose=False)


def test_strict_raises_on_missing_target(tmp_path: Path) -> None:
    """With ``strict=True``, a missing-target key also raises."""
    _ensure_dist_init()
    # Source has an extra "head" key the target does not have.
    src_model = _make_target_with_head()
    src_sd = src_model.state_dict()
    weights_path = _save_pt(src_sd, tmp_path / "src.pt")

    tgt_model = nn.ModuleDict({"backbone": _make_target()})

    with pytest.raises(RuntimeError, match="no target counterpart"):
        load_pretrained_backbone(tgt_model, weights_path, strict=True, verbose=False)


# ---------------------------------------------------------------------------
# (f) missing-key reporting
# ---------------------------------------------------------------------------


def test_missing_in_source_reported(tmp_path: Path) -> None:
    """Target keys with no source counterpart land in ``missing_in_source``."""
    _ensure_dist_init()
    # Source: just the first layer.
    partial_sd = {
        "0.weight": torch.randn(4, 8),
        "0.bias": torch.randn(4),
    }
    weights_path = _save_pt(partial_sd, tmp_path / "src.pt")

    tgt_model = _make_target()
    tgt_layer1_before = tgt_model.state_dict()["1.weight"].clone()

    report = load_pretrained_backbone(tgt_model, weights_path, verbose=False)

    assert set(report["loaded"]) == {"0.weight", "0.bias"}
    # Target's layer-1 keys should be reported as missing in source.
    assert set(report["missing_in_source"]) == {"1.weight", "1.bias"}
    # No exception raised, and target's layer 1 retains its init.
    tgt_layer1_after = tgt_model.state_dict()["1.weight"]
    assert torch.equal(tgt_layer1_before, tgt_layer1_after)


# ---------------------------------------------------------------------------
# (g) round-trip via .pt file
# ---------------------------------------------------------------------------


def test_pt_round_trip(tmp_path: Path) -> None:
    """Save a state dict directly to ``.pt``; load_pretrained_backbone loads it."""
    _ensure_dist_init()
    src_model = _make_target()
    src_sd = src_model.state_dict()

    weights_path = tmp_path / "round_trip.pt"
    torch.save(src_sd, weights_path)

    tgt_model = _make_target()
    report = load_pretrained_backbone(tgt_model, str(weights_path), verbose=False)

    assert set(report["loaded"]) == set(src_sd.keys())
    tgt_sd = tgt_model.state_dict()
    for k, v in src_sd.items():
        assert torch.equal(tgt_sd[k], v)


# ---------------------------------------------------------------------------
# (h) wrapped checkpoint format
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wrapper_key", ["model_state_dict", "state_dict", "model"])
def test_wrapped_checkpoint_format(tmp_path: Path, wrapper_key: str) -> None:
    """A ``{wrapper_key: state_dict, "epoch": 5}`` container is auto-unwrapped."""
    _ensure_dist_init()
    src_model = _make_target()
    src_sd = src_model.state_dict()

    weights_path = tmp_path / "wrapped.pt"
    torch.save({wrapper_key: src_sd, "epoch": 5}, weights_path)

    tgt_model = _make_target()
    report = load_pretrained_backbone(tgt_model, str(weights_path), verbose=False)

    assert set(report["loaded"]) == set(src_sd.keys())
    tgt_sd = tgt_model.state_dict()
    for k, v in src_sd.items():
        assert torch.equal(tgt_sd[k], v)


# ---------------------------------------------------------------------------
# (i) DDP-wrapped target (gated)
# ---------------------------------------------------------------------------


def test_ddp_wrapped_target(tmp_path: Path) -> None:
    """Loading into a DDP-wrapped model unwraps the ``module`` prefix.

    Gated on ``WORLD_SIZE > 1`` — single-process runs skip.  Mirrors the
    skip pattern in ``test/utils/test_checkpoint_distributed.py``.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        pytest.skip("Need at least 2 ranks for DDP-wrapped test")

    _ensure_dist_init()
    dm = DistributedManager()
    device = dm.device

    src_model = _make_target().to(device)
    src_sd = src_model.state_dict()

    weights_path = tmp_path / "ddp_src.pt"
    if dm.rank == 0:
        torch.save(src_sd, weights_path)
    torch.distributed.barrier()

    tgt_model = _make_target().to(device)
    ddp_model = nn.parallel.DistributedDataParallel(
        tgt_model, device_ids=[dm.local_rank] if device.type == "cuda" else None
    )

    report = load_pretrained_backbone(
        ddp_model, str(weights_path), device=device, verbose=False
    )

    assert set(report["loaded"]) == set(src_sd.keys())
    # Inner module's state matches source.
    inner_sd = ddp_model.module.state_dict()
    for k, v in src_sd.items():
        assert torch.equal(inner_sd[k].cpu(), v.cpu())


# ---------------------------------------------------------------------------
# (j) verbose=False is quiet
# ---------------------------------------------------------------------------


def test_verbose_false_is_quiet(tmp_path: Path, caplog) -> None:
    """``verbose=False`` should not emit the per-call summary log line."""
    _ensure_dist_init()
    src_sd = _make_target().state_dict()
    weights_path = _save_pt(src_sd, tmp_path / "src.pt")
    tgt_model = _make_target()

    # Capture every level on the physicsnemo / checkpoint logger and root.
    with caplog.at_level(logging.DEBUG):
        load_pretrained_backbone(tgt_model, weights_path, verbose=False)

    quiet_records = [
        r for r in caplog.records if "load_pretrained_backbone" in r.getMessage()
    ]
    assert quiet_records == [], (
        "verbose=False should not emit the [load_pretrained_backbone] summary; "
        f"got: {[r.getMessage() for r in quiet_records]}"
    )

    # As a sanity check, verbose=True does emit it. We construct a fresh
    # target so the load actually happens.
    tgt_model2 = _make_target()
    caplog.clear()
    with caplog.at_level(logging.INFO):
        load_pretrained_backbone(tgt_model2, weights_path, verbose=True)
    loud_records = [
        r for r in caplog.records if "load_pretrained_backbone" in r.getMessage()
    ]
    assert len(loud_records) >= 1, "verbose=True should emit the summary line"


# ---------------------------------------------------------------------------
# Additional: file-not-found
# ---------------------------------------------------------------------------


def test_missing_file_raises(tmp_path: Path) -> None:
    """A non-existent local path raises ``FileNotFoundError`` with a helpful message."""
    _ensure_dist_init()
    tgt_model = _make_target()
    bogus = str(tmp_path / "does_not_exist.pt")
    with pytest.raises(FileNotFoundError, match="not found"):
        load_pretrained_backbone(tgt_model, bogus, verbose=False)
