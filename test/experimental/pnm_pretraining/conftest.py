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

"""Shared pytest fixtures for ``pnm_pretraining`` parity tests.

Provides a fixed mesh set used identically across M1 (kernel parity)
and M2 (composite parity). Three analytic meshes (sphere, cube, torus)
generated at import time; ShapeNet meshes are loaded lazily via the
fetch script in ``.worktrees/exp-geopt-datagen-r1/scripts/`` and
skipped if unavailable.

Determinism: every test that consumes a mesh from this module also
consumes a seeded RNG returned by :func:`rng`. Round-1 parity tests
must seed both NumPy and PyTorch (and, where applicable, Warp) at the
top of every test for reproducibility.

See ``geopt-datagen-round1-plan.md`` §2 for the test-mesh-set
specification.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Analytic mesh generators
# ---------------------------------------------------------------------------


@dataclass
class TriangleMesh:
    """Plain mesh container for parity tests.

    Attributes
    ----------
    name
        Short label for reports (e.g. ``"sphere"``).
    vertices
        ``(V, 3)`` float32 vertex positions.
    indices
        ``(F, 3)`` int32 triangle indices.
    note
        Optional one-line description (e.g. ``"watertight"``).
    """

    name: str
    vertices: np.ndarray
    indices: np.ndarray
    note: str = ""

    def to_torch(
        self, device: str | torch.device = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(vertices, indices)`` as torch tensors on the given device."""
        v = torch.as_tensor(self.vertices, dtype=torch.float32, device=device)
        f = torch.as_tensor(self.indices, dtype=torch.int32, device=device)
        return v, f


def make_sphere(n_rings: int = 16, n_segments: int = 32) -> TriangleMesh:
    """UV-tessellated unit sphere centered at origin, radius 1."""
    rings = np.linspace(0, np.pi, n_rings + 1)
    segs = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    R, S = np.meshgrid(rings, segs, indexing="ij")
    x = np.sin(R) * np.cos(S)
    y = np.sin(R) * np.sin(S)
    z = np.cos(R)
    verts = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)

    faces: list[tuple[int, int, int]] = []
    for i in range(n_rings):
        for j in range(n_segments):
            a = i * n_segments + j
            b = i * n_segments + (j + 1) % n_segments
            c = (i + 1) * n_segments + j
            d = (i + 1) * n_segments + (j + 1) % n_segments
            faces.append((a, c, b))
            faces.append((b, c, d))
    idx = np.asarray(faces, dtype=np.int32)
    return TriangleMesh("sphere", verts, idx, note="UV-tessellated, radius 1")


def make_cube() -> TriangleMesh:
    """Axis-aligned unit cube spanning ``[-1, 1]^3``."""
    verts = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float32,
    )
    # 12 triangles, outward-CCW.
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],  # -Z
            [4, 5, 6],
            [4, 6, 7],  # +Z
            [0, 1, 5],
            [0, 5, 4],  # -Y
            [2, 3, 7],
            [2, 7, 6],  # +Y
            [1, 2, 6],
            [1, 6, 5],  # +X
            [0, 4, 7],
            [0, 7, 3],  # -X
        ],
        dtype=np.int32,
    )
    return TriangleMesh("cube", verts, faces, note="axis-aligned, half-extent 1")


def make_torus(
    R: float = 2.0, r: float = 0.5, n_major: int = 40, n_minor: int = 20
) -> TriangleMesh:
    """Torus with major radius ``R`` and minor radius ``r``, axis along Z."""
    u = np.linspace(0, 2 * np.pi, n_major, endpoint=False)
    v = np.linspace(0, 2 * np.pi, n_minor, endpoint=False)
    U, V = np.meshgrid(u, v, indexing="ij")
    x = (R + r * np.cos(V)) * np.cos(U)
    y = (R + r * np.cos(V)) * np.sin(U)
    z = r * np.sin(V)
    verts = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)

    faces: list[tuple[int, int, int]] = []
    for i in range(n_major):
        for j in range(n_minor):
            a = i * n_minor + j
            b = ((i + 1) % n_major) * n_minor + j
            c = i * n_minor + (j + 1) % n_minor
            d = ((i + 1) % n_major) * n_minor + (j + 1) % n_minor
            faces.append((a, b, c))
            faces.append((b, d, c))
    idx = np.asarray(faces, dtype=np.int32)
    return TriangleMesh("torus", verts, idx, note=f"R={R}, r={r}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sphere_mesh() -> TriangleMesh:
    return make_sphere()


@pytest.fixture(scope="session")
def cube_mesh() -> TriangleMesh:
    return make_cube()


@pytest.fixture(scope="session")
def torus_mesh() -> TriangleMesh:
    return make_torus()


@pytest.fixture(scope="session")
def analytic_meshes(
    sphere_mesh: TriangleMesh,
    cube_mesh: TriangleMesh,
    torus_mesh: TriangleMesh,
) -> list[TriangleMesh]:
    """All three analytic meshes used by M1 / M2 parity tests."""
    return [sphere_mesh, cube_mesh, torus_mesh]


@pytest.fixture
def rng() -> np.random.Generator:
    """Per-test seeded NumPy RNG. Tests should also seed torch separately."""
    return np.random.default_rng(seed=42)


def shapenet_subset_root() -> Path | None:
    """Return the local ShapeNet subset root if configured, else ``None``.

    Resolved from the ``PNM_SHAPENET_SUBSET`` env var. The fetch script
    (``scripts/fetch_shapenet_subset.py``) populates this directory.
    """
    p = os.environ.get("PNM_SHAPENET_SUBSET")
    return Path(p) if p else None


@pytest.fixture(scope="session")
def shapenet_subset_or_skip() -> Path:
    """Skip the test if the ShapeNet subset is not configured locally."""
    p = shapenet_subset_root()
    if p is None or not p.exists():
        pytest.skip(
            "ShapeNet subset not configured. "
            "Set PNM_SHAPENET_SUBSET to a populated directory; "
            "see scripts/fetch_shapenet_subset.py."
        )
    return p


def requires_fcpw():
    """Return a pytest mark that skips when ``fcpw`` is not importable."""
    try:
        import fcpw  # noqa: F401

        return pytest.mark.fcpw
    except ImportError:
        return pytest.mark.skip(reason="fcpw not installed; install for parity check")


def requires_geopt_reference():
    """Return a pytest mark that skips when the GeoPT clone is unavailable.

    Resolved from the ``PNM_GEOPT_REF`` env var.
    """
    p = os.environ.get("PNM_GEOPT_REF")
    if p and Path(p).exists():
        return pytest.mark.geopt_ref
    return pytest.mark.skip(
        reason="PNM_GEOPT_REF not set or path missing; "
        "clone Physics-Scaling/GeoPT and point PNM_GEOPT_REF at it."
    )
