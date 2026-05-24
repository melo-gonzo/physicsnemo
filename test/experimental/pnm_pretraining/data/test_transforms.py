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

"""Alignment-sanity tests for ``align_mesh_geopt_general``.

Covers M3 deliverable 3 of ``geopt-datagen-round1-plan.md``: invariants
on the post-alignment mesh (X-extent, Y-min, X-mean, Z-mean), the
``AlignmentRecord.apply`` / ``inverse`` round-trip (improvement I4 in
plan §8), and the oversize-safety branch.
"""

from __future__ import annotations

import numpy as np
import pytest

from test.conftest import requires_module

trimesh = pytest.importorskip("trimesh")

from physicsnemo.experimental.pnm_pretraining.data.transforms import (  # noqa: E402
    AlignmentRecord,
    align_mesh_geopt_general,
)

# Tolerance: invariants are validated in float32 storage; ulp at scale ~5
# is ~5e-7. The 1e-5 slack covers float64 arithmetic round-off and the
# mean-subtraction summation error across vertex counts up to ~1e4.
_TOL = 1e-5


def _trimesh_from_dataclass(tm) -> "trimesh.Trimesh":
    """Convert the conftest ``TriangleMesh`` dataclass to a ``trimesh.Trimesh``.

    Uses ``process=False`` per plan §9: trimesh 3.x's default
    ``process=True`` runs vertex merging that would change vertex
    counts, breaking parity with the rest of the M1/M2 pipeline.
    """
    return trimesh.Trimesh(
        vertices=tm.vertices.astype(np.float64), faces=tm.indices, process=False
    )


def _make_asymmetric_box(
    x_range: tuple[float, float] = (0.0, 4.0),
    y_range: tuple[float, float] = (0.0, 1.0),
    z_range: tuple[float, float] = (-0.5, 0.5),
) -> "trimesh.Trimesh":
    """Axis-aligned box from (x0, x1) × (y0, y1) × (z0, z1)."""
    x0, x1 = x_range
    y0, y1 = y_range
    z0, z1 = z_range
    verts = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [1, 2, 6],
            [1, 6, 5],
            [0, 4, 7],
            [0, 7, 3],
        ],
        dtype=np.int64,
    )
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


# ---------------------------------------------------------------------------
# Test (a): asymmetric "car-like" box exercises X-flip + X-centering
# ---------------------------------------------------------------------------


@requires_module("trimesh")
def test_asymmetric_box_alignment() -> None:
    """A 4x1x1 box at +X should land centered at the origin with extent 5.

    Walks the X-axis arithmetic explicitly so the test doubles as
    documentation of the intended transform: original X ``[0, 4]`` →
    flip → ``[-4, 0]`` → scale by ``5/4 = 1.25`` → ``[-5, 0]`` →
    recenter on mean ``-2.5`` → ``[-2.5, +2.5]``.
    """
    mesh = _make_asymmetric_box(x_range=(0.0, 4.0))
    aligned, record = align_mesh_geopt_general(mesh, target_length=5.0)

    v = aligned.vertices
    extents = v.max(axis=0) - v.min(axis=0)

    assert extents[0] == pytest.approx(5.0, abs=_TOL)
    assert v[:, 1].min() == pytest.approx(0.0, abs=_TOL)
    assert v[:, 0].mean() == pytest.approx(0.0, abs=_TOL)
    assert v[:, 2].mean() == pytest.approx(0.0, abs=_TOL)

    # Walk-through: post-alignment X coords should be exactly {-2.5, +2.5}.
    unique_x = np.sort(np.unique(np.round(v[:, 0], 6)))
    np.testing.assert_allclose(unique_x, np.array([-2.5, 2.5]), atol=_TOL)

    assert record.axis_flipped is True
    assert record.oversize_safety_applied is False
    assert record.scale == pytest.approx(1.25, abs=_TOL)


# ---------------------------------------------------------------------------
# Test (b): invariants hold on all three analytic meshes
# ---------------------------------------------------------------------------


@requires_module("trimesh")
@pytest.mark.parametrize("mesh_fixture", ["sphere_mesh", "cube_mesh", "torus_mesh"])
def test_alignment_invariants_on_analytic_meshes(
    mesh_fixture: str, request: pytest.FixtureRequest
) -> None:
    """X-extent ≈ 5.0, Y-min ≈ 0, X-mean ≈ 0, Z-mean ≈ 0 on every mesh.

    Also verifies ``record.apply(original_vertices)`` reproduces the
    transformed mesh's vertices bit-exactly: both run through the same
    arithmetic, so any drift would point at an internal inconsistency
    (e.g. forgetting a step in :meth:`AlignmentRecord.apply`).
    """
    tm = request.getfixturevalue(mesh_fixture)
    original_vertices = tm.vertices.astype(np.float64).copy()
    mesh = _trimesh_from_dataclass(tm)

    # ``oversize_safety=False`` here: the analytic fixtures are unit-scale
    # primitives whose other axes blow past the GeoPT 3.0/6.0 hard limits at
    # ``target_length=5.0`` (e.g. sphere extents become (5, 5, 5); 5 > 3 on Y
    # → safety would fire and halve the scale, defeating the X-extent
    # invariant we want to assert here). Test (d) covers the safety branch
    # explicitly on a geometry that needs it.
    aligned, record = align_mesh_geopt_general(
        mesh, target_length=5.0, oversize_safety=False
    )
    v = np.asarray(aligned.vertices)
    extents = v.max(axis=0) - v.min(axis=0)

    assert extents[0] == pytest.approx(5.0, abs=_TOL), f"{mesh_fixture}: x_extent"
    assert v[:, 1].min() == pytest.approx(0.0, abs=_TOL), f"{mesh_fixture}: y_min"
    assert v[:, 0].mean() == pytest.approx(0.0, abs=_TOL), f"{mesh_fixture}: x_mean"
    assert v[:, 2].mean() == pytest.approx(0.0, abs=_TOL), f"{mesh_fixture}: z_mean"

    # ``record.apply`` should reproduce the mesh's transform exactly when
    # arithmetic is performed in the same float64 precision.
    replayed = record.apply(original_vertices)
    np.testing.assert_array_equal(replayed, v)


# ---------------------------------------------------------------------------
# Test (c): apply / inverse round-trip on random points
# ---------------------------------------------------------------------------


@requires_module("trimesh")
def test_apply_inverse_round_trip(cube_mesh) -> None:
    """``record.inverse(record.apply(p)) == p`` within float32 ulps.

    Validates I4: the named alignment record can recover original-frame
    coordinates, which round-2 evaluation needs to map fine-tuning
    predictions back into the user's physical-units frame.
    """
    mesh = _trimesh_from_dataclass(cube_mesh)
    _, record = align_mesh_geopt_general(mesh, target_length=5.0)

    rng = np.random.default_rng(seed=42)
    pts = rng.uniform(-10.0, 10.0, size=(100, 3)).astype(np.float64)
    recovered = record.inverse(record.apply(pts))
    err = np.max(np.abs(recovered - pts))
    assert err < _TOL, f"round-trip max abs err = {err}"


# ---------------------------------------------------------------------------
# Test (d): oversize-safety branch fires and is recorded
# ---------------------------------------------------------------------------


@requires_module("trimesh")
def test_oversize_safety_triggers_and_is_recorded() -> None:
    """A box with a huge Y-extent triggers the GeoPT *0.5 scale haircut.

    Geometry: ``x ∈ [-1, 1]``, ``y ∈ [-50, 50]``, ``z ∈ [-1, 1]``. The
    unmodified scale is ``5 / 2 = 2.5``; ``y_extent * scale = 100 *
    2.5 = 250`` blows past the GeoPT limit of 3.0, so the safety
    branch fires and halves the scale to ``1.25``. With safety
    triggered, the post-alignment X-extent is intentionally **half** of
    ``target_length`` (the hack gives up X-extent fidelity to keep the
    bounding box bounded).
    """
    mesh = _make_asymmetric_box(
        x_range=(-1.0, 1.0), y_range=(-50.0, 50.0), z_range=(-1.0, 1.0)
    )
    aligned, record = align_mesh_geopt_general(
        mesh, target_length=5.0, oversize_safety=True
    )

    assert record.oversize_safety_applied is True
    assert record.scale == pytest.approx(1.25, abs=_TOL)

    # X-extent should be ``target_length / 2 = 2.5`` after the haircut.
    v = aligned.vertices
    extents = v.max(axis=0) - v.min(axis=0)
    assert extents[0] == pytest.approx(2.5, abs=_TOL)

    # Y-floor and X/Z-centering invariants still hold.
    assert v[:, 1].min() == pytest.approx(0.0, abs=_TOL)
    assert v[:, 0].mean() == pytest.approx(0.0, abs=_TOL)
    assert v[:, 2].mean() == pytest.approx(0.0, abs=_TOL)


# ---------------------------------------------------------------------------
# Test (e): docstring smoke — divergence-from-category-variant is documented
# ---------------------------------------------------------------------------


def test_docstring_documents_geopt_variant() -> None:
    """Function docstring must call out which GeoPT variant we ported.

    Low-cost guard against a future refactor losing the documentation
    that distinguishes the General-variant X-flip path from the
    category-specific X↔Z-swap path (plan §0 F0.6).
    """
    doc = align_mesh_geopt_general.__doc__ or ""
    assert "X-flip" in doc, "docstring must mention X-flip"
    assert "General variant" in doc, "docstring must mention General variant"


# ---------------------------------------------------------------------------
# Sanity: AlignmentRecord is frozen / cannot be mutated
# ---------------------------------------------------------------------------


def test_alignment_record_is_frozen() -> None:
    """``AlignmentRecord`` is a frozen dataclass; mutations must raise."""
    rec = AlignmentRecord(
        axis_flipped=True,
        y_min_post_flip=0.0,
        scale=1.0,
        x_mean_post_scale=0.0,
        z_mean_post_scale=0.0,
        oversize_safety_applied=False,
    )
    with pytest.raises(Exception):
        rec.scale = 2.0  # type: ignore[misc]
