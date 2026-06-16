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

"""Synthetic smoke tests for the model-agnostic loss-landscape core.

These exercise the filter-normalization, grid evaluation, weight restoration,
and plotting on a tiny CPU model so the machinery is validated without a GPU,
a checkpoint, or the DrivaerML dataset (the real GeoTransolver pass runs later
on the GPU box via the ``main`` entry point).
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from plot_loss_landscape import (  # noqa: E402
    compare_landscapes,
    compute_landscape,
    evaluate_landscape,
    filter_normalized_directions,
    plot_landscape,
)


def _toy_model(seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)
    )


def _quadratic_loss_fn(model: torch.nn.Module):
    """Loss = sum of squared weights -> minimized exactly at theta_star.

    With a convex bowl centred on the trained weights, the landscape minimum
    must sit at the grid centre ``(alpha, beta) = (0, 0)``.
    """

    def loss_fn() -> float:
        return float(sum((p**2).sum() for p in model.parameters()))

    return loss_fn


def test_filter_normalized_directions_match_weight_norms():
    model = _toy_model()
    dir_a, dir_b = filter_normalized_directions(model, seed=1)
    params = list(model.parameters())
    assert len(dir_a) == len(dir_b) == len(params)

    for p, da, db in zip(params, dir_a, dir_b):
        assert da.shape == p.shape and db.shape == p.shape
        if p.dim() < 2:
            # Biases must be frozen (zero direction).
            assert torch.count_nonzero(da) == 0
            assert torch.count_nonzero(db) == 0
        else:
            # Per-filter norm of the direction matches the weight's.
            for d in (da, db):
                dn = d.reshape(d.shape[0], -1).norm(dim=1)
                wn = p.reshape(p.shape[0], -1).norm(dim=1)
                torch.testing.assert_close(dn, wn, rtol=1e-4, atol=1e-6)


def test_directions_are_reproducible_by_seed():
    model = _toy_model()
    a1, _ = filter_normalized_directions(model, seed=7)
    a2, _ = filter_normalized_directions(model, seed=7)
    a3, _ = filter_normalized_directions(model, seed=8)
    for x, y in zip(a1, a2):
        torch.testing.assert_close(x, y)
    # Different seed -> different directions (at least one weight tensor).
    assert any(not torch.allclose(x, z) for x, z in zip(a1, a3))


def test_evaluate_restores_weights_exactly():
    model = _toy_model()
    snapshot = [p.detach().clone() for p in model.parameters()]
    alphas = np.linspace(-1, 1, 5)
    betas = np.linspace(-1, 1, 5)
    dir_a, dir_b = filter_normalized_directions(model, seed=2)

    evaluate_landscape(
        model, _quadratic_loss_fn(model), dir_a, dir_b, alphas, betas,
        progress=False,
    )

    for p, w0 in zip(model.parameters(), snapshot):
        assert torch.equal(p, w0), "weights not restored bit-exactly"


def test_weights_restored_even_if_loss_raises():
    model = _toy_model()
    snapshot = [p.detach().clone() for p in model.parameters()]
    dir_a, dir_b = filter_normalized_directions(model, seed=3)

    calls = {"n": 0}

    def boom() -> float:
        calls["n"] += 1
        if calls["n"] >= 3:
            raise RuntimeError("synthetic failure mid-sweep")
        return 0.0

    with pytest.raises(RuntimeError):
        evaluate_landscape(
            model, boom, dir_a, dir_b,
            np.linspace(-1, 1, 4), np.linspace(-1, 1, 4), progress=False,
        )

    for p, w0 in zip(model.parameters(), snapshot):
        assert torch.equal(p, w0), "weights not restored after exception"


def test_grid_shape_and_minimum_at_center():
    model = _toy_model()
    alphas, betas, grid = compute_landscape(
        model, _quadratic_loss_fn(model), resolution=11, span=1.0, seed=4,
        progress=False,
    )
    assert grid.shape == (11, 11)
    assert np.all(np.isfinite(grid))

    # Convex bowl centred on theta_star -> global min at the centre cell.
    i, j = np.unravel_index(np.argmin(grid), grid.shape)
    assert (i, j) == (5, 5)
    # Centre value equals the unperturbed loss.
    center = float(sum((p**2).sum() for p in model.parameters()))
    np.testing.assert_allclose(grid[5, 5], center, rtol=1e-5)


def test_plot_landscape_writes_png_and_npz(tmp_path):
    model = _toy_model()
    alphas, betas, grid = compute_landscape(
        model, _quadratic_loss_fn(model), resolution=7, span=1.0, progress=False,
    )
    out = plot_landscape(alphas, betas, grid, tmp_path / "land.png", title="t")
    assert out.exists()
    assert out.with_suffix(".npz").exists()

    data = np.load(out.with_suffix(".npz"))
    np.testing.assert_array_equal(data["grid"], grid)


def test_compare_landscapes_writes_png(tmp_path):
    model = _toy_model()
    a, b, g1 = compute_landscape(
        model, _quadratic_loss_fn(model), resolution=7, seed=0, progress=False,
    )
    _, _, g2 = compute_landscape(
        model, _quadratic_loss_fn(model), resolution=7, seed=1, progress=False,
    )
    out = compare_landscapes(
        a, b, g1, g2, tmp_path / "cmp.png", titles=("Adam", "LookSAM")
    )
    assert out.exists()
    data = np.load(out.with_suffix(".npz"))
    np.testing.assert_array_equal(data["grid_a"], g1)
    np.testing.assert_array_equal(data["grid_b"], g2)
