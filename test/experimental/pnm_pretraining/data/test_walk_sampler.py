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

"""Unit tests for :class:`~physicsnemo.experimental.pnm_pretraining.data.WalkSampler`.

Pinned behaviors (PR 2 / improvement I20, with the PR 2 follow-up
per-step decomposition):

(a) Slicing produces ``n_steps`` per-point ``(N, 3)`` ``supervise_step{i}``
    fields plus ``directions: (N, 3)`` and ``step_lengths: (N, 1)`` in
    ``interior.point_data`` and removes the source ``(n_walks, …)``
    arrays from ``interior.global_data``. The flat ``supervise`` key
    is gone (it was replaced by the per-step decomposition so the
    recipe's ``vector`` FieldType, which eats 3 channels, can consume
    each step natively).
(b) Same seed → identical slice across all per-step + ancillary keys
    (per-instance generator is honored).
(c) Different seed on a many-walks DomainMesh → at least one of the
    per-step / directions / step_lengths fields differs (different
    walk index drawn).
(d) The per-step fields preserve trajectory order (``supervise_step0``
    is step 0, ``supervise_step1`` is step 1, …) — regression guard
    against an accidental dim swap.
(e) Missing walk arrays raise ``KeyError`` with a message that names
    the missing key and points at the M3 builder.
"""

from __future__ import annotations

import pytest
import torch

from physicsnemo.experimental.pnm_pretraining.data import (
    WalkSampler,
    build_pretraining_sample,
)
from physicsnemo.mesh import DomainMesh, Mesh
from test.experimental.pnm_pretraining.conftest import TriangleMesh

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_tiny_pretraining_dm(sphere_mesh: TriangleMesh, **overrides) -> DomainMesh:
    """Build a tiny pretraining ``DomainMesh`` from the conftest sphere fixture.

    Defaults match the prompt's tiny config (``n_volume_points=64,
    n_surface_points=16, n_independent_walks=2, n_jittered_per_base=1,
    n_steps=2, seed=0``). Overrides let tests bump ``n_walks`` for the
    variation test (test (c)).
    """
    import trimesh

    tm = trimesh.Trimesh(
        vertices=sphere_mesh.vertices,
        faces=sphere_mesh.indices,
        process=False,
    )
    kwargs = dict(
        n_volume_points=64,
        n_surface_points=16,
        n_independent_walks=2,
        n_jittered_per_base=1,
        n_steps=2,
        seed=0,
    )
    kwargs.update(overrides)
    return build_pretraining_sample(tm, **kwargs)


def _make_synthetic_walk_dm(
    n_points: int,
    n_walks: int,
    n_steps: int,
    *,
    deterministic: bool = False,
) -> DomainMesh:
    """Construct a minimal pretraining-shaped DomainMesh in-memory.

    No mesh-quality / alignment / config block — just enough scaffolding
    that :class:`WalkSampler` can find the three walk arrays in
    ``interior.global_data``. Used by tests (b)-(d) which need explicit
    control over walk-array contents (test (d) hand-crafts indices).
    """
    if deterministic:
        # walks_supervise[i, j, t, :] == [i, j, t]
        i_idx = (
            torch.arange(n_walks)
            .view(n_walks, 1, 1, 1)
            .expand(n_walks, n_points, n_steps, 1)
        )
        j_idx = (
            torch.arange(n_points)
            .view(1, n_points, 1, 1)
            .expand(n_walks, n_points, n_steps, 1)
        )
        t_idx = (
            torch.arange(n_steps)
            .view(1, 1, n_steps, 1)
            .expand(n_walks, n_points, n_steps, 1)
        )
        walks_supervise = torch.cat([i_idx, j_idx, t_idx], dim=-1).to(torch.float32)
        walks_directions = torch.arange(n_walks * n_points * 3, dtype=torch.float32)
        walks_directions = walks_directions.view(n_walks, n_points, 3)
        walks_step_lengths = torch.arange(n_walks * n_points, dtype=torch.float32).view(
            n_walks, n_points
        )
    else:
        torch.manual_seed(0)
        walks_supervise = torch.randn(n_walks, n_points, n_steps, 3)
        walks_directions = torch.randn(n_walks, n_points, 3)
        walks_step_lengths = torch.randn(n_walks, n_points)

    interior = Mesh(
        points=torch.zeros(n_points, 3),
        cells=torch.zeros((0, 1), dtype=torch.int64),
        global_data={
            "walks_supervise": walks_supervise,
            "walks_directions": walks_directions,
            "walks_step_lengths": walks_step_lengths,
        },
    )
    boundary = Mesh(
        points=torch.zeros(3, 3),
        cells=torch.tensor([[0, 1, 2]], dtype=torch.int64),
    )
    return DomainMesh(
        interior=interior,
        boundaries={"geometry": boundary},
    )


# ---------------------------------------------------------------------------
# Test (a): shape + key semantics on a real pretraining sample
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    pytest.importorskip("trimesh", reason="trimesh required for builder tests") is None,
    reason="trimesh not available",
)
class TestWalkSamplerOnPretrainingSample:
    """Pinned behavior on a real :func:`build_pretraining_sample` output."""

    def test_shapes_and_key_movement(self, sphere_mesh: TriangleMesh):
        """Per-point shapes correct; walk arrays gone from global_data."""
        dm = _build_tiny_pretraining_dm(sphere_mesh)

        # Sanity: the source arrays are where the docstring claims.
        n_total = dm.interior.n_points
        assert n_total == 64 + 16, f"expected 80 interior points, got {n_total}"
        for key in ("walks_supervise", "walks_directions", "walks_step_lengths"):
            assert key in dm.interior.global_data.keys(), (
                f"prerequisite missing: {key} not in interior.global_data"
            )

        sampler = WalkSampler(seed=0)
        out = sampler(dm)

        # ``directions`` and ``step_lengths`` keep their single-key shapes.
        assert out.interior.point_data["directions"].shape == (n_total, 3)
        assert out.interior.point_data["step_lengths"].shape == (n_total, 1)

        # ``supervise`` is now decomposed into ``n_steps`` per-step
        # ``(n_total, 3)`` vector fields. The flat ``supervise`` key
        # must be gone (we no longer emit it).
        n_steps = 2
        assert "supervise" not in out.interior.point_data.keys(), (
            "WalkSampler still emits the flat ``supervise`` key; the "
            "PR 2 follow-up replaces it with per-step ``supervise_step{i}``."
        )
        for step_idx in range(n_steps):
            key = f"supervise_step{step_idx}"
            assert key in out.interior.point_data.keys(), (
                f"WalkSampler did not emit per-step key {key!r}"
            )
            assert out.interior.point_data[key].shape == (n_total, 3), (
                f"per-step field {key!r} has wrong shape "
                f"{tuple(out.interior.point_data[key].shape)}; expected "
                f"({n_total}, 3)"
            )

        # And the source arrays are gone from interior.global_data.
        for key in ("walks_supervise", "walks_directions", "walks_step_lengths"):
            assert key not in out.interior.global_data.keys(), (
                f"WalkSampler did not drop {key!r} from interior.global_data"
            )


# ---------------------------------------------------------------------------
# Test (b): determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same seed → identical slice across two independent invocations."""

    def test_same_seed_same_slice(self):
        n_steps = 3
        dm = _make_synthetic_walk_dm(n_points=12, n_walks=8, n_steps=n_steps)

        a = WalkSampler(seed=42)(dm)
        b = WalkSampler(seed=42)(dm)

        keys = ["directions", "step_lengths"] + [
            f"supervise_step{i}" for i in range(n_steps)
        ]
        for key in keys:
            assert torch.equal(
                a.interior.point_data[key], b.interior.point_data[key]
            ), f"determinism violated on key {key!r}"


# ---------------------------------------------------------------------------
# Test (c): variation across seeds
# ---------------------------------------------------------------------------


class TestSeedVariation:
    """Different seeds on a many-walks DomainMesh draw different walks."""

    def test_different_seed_different_slice(self):
        # n_walks=10 keeps the chance of two distinct seeds picking the
        # same index small (1/10 worst-case for a uniform draw); we run
        # one pair and accept that on the rare collision the assertion
        # would fail loudly.
        n_steps = 3
        dm = _make_synthetic_walk_dm(n_points=12, n_walks=10, n_steps=n_steps)

        a = WalkSampler(seed=42)(dm)
        b = WalkSampler(seed=43)(dm)

        keys = ["directions", "step_lengths"] + [
            f"supervise_step{i}" for i in range(n_steps)
        ]
        differ = any(
            not torch.equal(a.interior.point_data[key], b.interior.point_data[key])
            for key in keys
        )
        assert differ, (
            "WalkSampler(seed=42) and WalkSampler(seed=43) produced "
            "identical slices on a 10-walk DomainMesh — generator "
            "almost certainly not honoring the seed."
        )


# ---------------------------------------------------------------------------
# Test (d): supervise reshape preserves trajectory order
# ---------------------------------------------------------------------------


class TestReshapeOrdering:
    """Per-step decomposition preserves trajectory order."""

    def test_trajectory_order_preserved(self):
        # Hand-crafted contents: walks_supervise[i, j, t, :] == [i, j, t].
        # If WalkSampler picks index 0, the per-step rows j become:
        #   supervise_step0[j, :] == [0, j, 0]   (step 0 vector)
        #   supervise_step1[j, :] == [0, j, 1]   (step 1 vector)
        # An accidental step↔point dim swap would break this directly.
        n_points = 5
        n_walks = 2
        n_steps = 2
        dm = _make_synthetic_walk_dm(
            n_points=n_points,
            n_walks=n_walks,
            n_steps=n_steps,
            deterministic=True,
        )

        # Force the draw to a known index by seeding to a value and
        # peeking at what the same-seeded generator draws first. We do
        # not depend on a specific seed-to-index mapping: instead, we
        # peek the index a fresh same-seeded generator would draw and
        # then assert the per-step fields against the expected pattern
        # for that index. Robust to torch RNG version skew.
        sampler = WalkSampler(seed=0)
        peek_gen = torch.Generator()
        peek_gen.manual_seed(0)
        expected_idx = int(torch.randint(0, n_walks, (), generator=peek_gen).item())

        out = sampler(dm)
        for t in range(n_steps):
            sup_t = out.interior.point_data[f"supervise_step{t}"]
            assert sup_t.shape == (n_points, 3), (
                f"per-step field supervise_step{t} has wrong shape "
                f"{tuple(sup_t.shape)}; expected ({n_points}, 3)"
            )
            for j in range(n_points):
                row = sup_t[j]
                assert torch.equal(
                    row, torch.tensor([expected_idx, j, t], dtype=torch.float32)
                ), (
                    f"per-step ordering broken at supervise_step{t}, j={j}: "
                    f"got {row.tolist()}, expected [{expected_idx}, {j}, {t}]."
                )


# ---------------------------------------------------------------------------
# Test (e): KeyError on a missing-walks DomainMesh
# ---------------------------------------------------------------------------


class TestMissingWalksRaise:
    """Applying WalkSampler to a DomainMesh without walk arrays raises."""

    def test_missing_walks_raises_keyerror(self):
        # A DrivAerML-shaped DomainMesh (interior + boundaries, no walks).
        interior = Mesh(
            points=torch.zeros(10, 3),
            cells=torch.zeros((0, 1), dtype=torch.int64),
            point_data={"velocity": torch.zeros(10, 3)},
        )
        boundary = Mesh(
            points=torch.zeros(3, 3),
            cells=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        )
        dm = DomainMesh(interior=interior, boundaries={"vehicle": boundary})

        sampler = WalkSampler(seed=0)
        with pytest.raises(KeyError) as excinfo:
            sampler(dm)
        msg = str(excinfo.value)
        assert "walks_supervise" in msg
        assert "build_pretraining_sample" in msg
