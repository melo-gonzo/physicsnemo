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
"""Harmonic boundary measures (HARMONIC, 2026-09-27): accuracy against a series solution, partition of unity,
consistency of the extensions, rigid-motion invariance, and the transform / precompute round trip."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree
from tensordict import TensorDict

from domain_transforms import HarmonicBoundaryMeasures  # noqa: E402
from harmonic import harmonic_boundary_fields, knn_laplacian, node_types_from_boundary, node_types_from_tags  # noqa: E402
from physicsnemo.mesh import DomainMesh, Mesh  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src"


def _square(m: int = 60, seed: int = 0):
    """Jittered m x m grid on the unit square; type A = left edge (x = 0), type B = the other three edges."""
    rng = np.random.default_rng(seed)
    g = np.linspace(0.0, 1.0, m)
    X, Y = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), np.zeros(m * m)], 1)
    inner = (pts[:, 0] > 0) & (pts[:, 0] < 1) & (pts[:, 1] > 0) & (pts[:, 1] < 1)
    pts[inner, :2] += rng.uniform(-0.3, 0.3, (int(inner.sum()), 2)) / (m - 1)
    tA = (pts[:, 0] <= 1e-12).astype(float)
    tB = ((pts[:, 0] >= 1 - 1e-12) | (pts[:, 1] <= 1e-12) | (pts[:, 1] >= 1 - 1e-12)).astype(float)
    tB[tA > 0] = 0.0
    return pts, node_types_from_tags([tA, tB])


def _series(x, y, terms=200):
    """Laplace on the unit square, u = 1 on x = 0 and 0 on the other edges (sinh ratio written with exponentials
    so large n does not overflow)."""
    u = np.zeros_like(x)
    for n in range(1, terms, 2):
        a = n * np.pi
        ratio = (np.exp(-a * x) - np.exp(-a * (2 - x))) / (1 - np.exp(-2 * a))  # sinh(a(1-x)) / sinh(a)
        u += (4 / a) * ratio * np.sin(a * y)
    return u


def test_harmonic_measure_matches_the_series_solution_and_sums_to_one():
    pts, t = _square()
    phi, _ = harmonic_boundary_fields(pts, t, 2)
    rng = np.random.default_rng(1)
    probes = np.stack([rng.uniform(0.1, 0.9, 20), rng.uniform(0.1, 0.9, 20)], 1)
    j = cKDTree(pts[:, :2]).query(probes)[1]
    rms = np.sqrt(np.mean((phi[j, 0] - _series(pts[j, 0], pts[j, 1])) ** 2))
    assert rms < 0.03, rms  # relative to max = 1
    assert np.abs(phi.sum(1) - 1).max() < 1e-6
    assert phi.min() > -1e-9 and phi.max() < 1 + 1e-9
    assert np.all(phi[t == 0, 0] == 1.0) and np.all(phi[t == 1, 0] == 0.0)


def test_extensions_are_consistent_with_a_direct_solve_and_add_up():
    pts, t = _square(40)
    g = pts[:, :1].copy()  # value x on every boundary node
    phi, hx = harmonic_boundary_fields(pts, t, 2, values=g)
    # direct solve with the same operator: data g on type A, 0 elsewhere
    L = knn_laplacian(pts, 12)
    bnd, free = t >= 0, t < 0
    ub = np.where(t == 0, g[:, 0], 0.0)
    from scipy.sparse.linalg import spsolve

    uf = spsolve(L[free][:, free].tocsc(), -(L[free][:, bnd] @ ub[bnd]))
    direct = ub.copy()
    direct[free] = uf
    assert np.abs(hx[:, 0, 0] - direct).max() < 1e-6
    # the sum over types is the extension of the full data
    ub_all = np.where(bnd, g[:, 0], 0.0)
    uf_all = spsolve(L[free][:, free].tocsc(), -(L[free][:, bnd] @ ub_all[bnd]))
    full = ub_all.copy()
    full[free] = uf_all
    assert np.abs(hx.sum(1)[:, 0] - full).max() < 1e-6


def test_fields_are_invariant_under_rotation_and_translation():
    pts, t = _square(40)
    phi, hx = harmonic_boundary_fields(pts, t, 2, values=pts[:, :1])
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
    moved = pts @ R.T + np.array([3.0, -2.0, 0.0])
    phi2, hx2 = harmonic_boundary_fields(moved, t, 2, values=pts[:, :1])
    assert np.abs(phi2 - phi).max() < 1e-6
    assert np.abs(hx2 - hx).max() < 1e-6


def test_absent_type_gives_zero_fields():
    pts, t = _square(30)
    phi, hx = harmonic_boundary_fields(pts, t, 3, values=pts[:, :1])  # type 2 never occurs
    assert np.all(phi[:, 2] == 0.0) and np.all(hx[:, 2] == 0.0)
    assert np.abs(phi.sum(1) - 1).max() < 1e-6


def _domain(m: int = 30):
    pts, t = _square(m)
    interior = Mesh(
        points=torch.tensor(pts, dtype=torch.float32),
        point_data=TensorDict(
            {
                "tag_a": torch.tensor((t == 0).astype(np.float32)),
                "tag_b": torch.tensor((t == 1).astype(np.float32)),
                "g_node": torch.tensor(pts[:, 0].astype(np.float32)),
                "sdf": torch.tensor(np.minimum.reduce([pts[:, 0], 1 - pts[:, 0], pts[:, 1], 1 - pts[:, 1]]).astype(np.float32)),
            },
            batch_size=[pts.shape[0]],
        ),
    )
    # boundary mesh: line cells along the four edges, labeled A (left) / B (rest)
    n = 30
    s = np.linspace(0, 1, n + 1)
    edges = []
    for a, b in ((np.stack([np.zeros(n + 1), s], 1), "a"), (np.stack([np.ones(n + 1), s], 1), "b"),
                 (np.stack([s, np.zeros(n + 1)], 1), "b"), (np.stack([s, np.ones(n + 1)], 1), "b")):
        edges.append((a, b))
    P, cells, la, lb, gv = [], [], [], [], []
    off = 0
    for xy, lab in edges:
        P.append(np.concatenate([xy, np.zeros((n + 1, 1))], 1))
        cells.append(np.stack([np.arange(n), np.arange(1, n + 1)], 1) + off)
        la += [1.0 if lab == "a" else 0.0] * n
        lb += [0.0 if lab == "a" else 1.0] * n
        gv += list(0.5 * (xy[:-1, 0] + xy[1:, 0]))
        off += n + 1
    outer = Mesh(
        points=torch.tensor(np.concatenate(P), dtype=torch.float32),
        cells=torch.tensor(np.concatenate(cells), dtype=torch.long),
        cell_data=TensorDict(
            {"bc_a": torch.tensor(la), "bc_b": torch.tensor(lb), "bc_value_g": torch.tensor(gv, dtype=torch.float32)},
            batch_size=[4 * n],
        ),
    )
    return DomainMesh(interior=interior, boundaries={"outer": outer}, global_data=TensorDict({}, batch_size=[]))


def test_transform_writes_the_six_fields_from_node_tags_and_from_the_boundary_mesh():
    dm = _domain()
    out = HarmonicBoundaryMeasures(
        "outer", ["bc_a", "bc_b"], value_field="bc_value_g", node_tag_fields=["tag_a", "tag_b"], node_value_field="g_node"
    ).apply_to_domain(dm)
    pd = out.interior.point_data
    for k in ("hm_a", "hm_b", "hx_a", "hx_b"):
        assert k in pd.keys() and pd[k].shape[0] == dm.interior.points.shape[0]
    assert torch.allclose(pd["hm_a"] + pd["hm_b"], torch.ones_like(pd["hm_a"]), atol=1e-5)
    # typed from the boundary mesh instead of node tags: the same boundary set up to the four corner nodes, whose
    # type is ambiguous where two types meet, so the measures agree everywhere except next to those corners
    out2 = HarmonicBoundaryMeasures("outer", ["bc_a", "bc_b"], value_field="bc_value_g").apply_to_domain(dm)
    d = (out2.interior.point_data["hm_a"] - pd["hm_a"]).abs()
    assert d.mean() < 0.01 and (d > 0.02).float().mean() < 0.02


def test_precompute_script_round_trips_a_dataset(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    for name in ("train_000000", "test_000001"):
        _domain().save(src / name / f"{name}.pdmsh")
    (src / "manifest.json").write_text('{"train": ["train_000000"], "test": ["test_000001"]}')
    cmd = [sys.executable, str(SRC / "precompute_harmonic.py"), "--src", str(src), "--dst", str(dst), "--boundary", "outer",
           "--label-fields", "bc_a", "bc_b", "--node-tag-fields", "tag_a", "tag_b", "--value-field", "bc_value_g",
           "--node-value-field", "g_node", "--workers", "2"]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "DONE ok=2 failed=0" in r.stdout, r.stdout
    assert (dst / "manifest.json").exists()
    back = DomainMesh.load(dst / "train_000000" / "train_000000.pdmsh")
    pd = back.interior.point_data
    assert torch.allclose(pd["hm_a"] + pd["hm_b"], torch.ones_like(pd["hm_a"]), atol=1e-5)
    assert "hx_a" in pd.keys() and "u" not in pd.keys()
    # the source is untouched
    assert "hm_a" not in DomainMesh.load(src / "train_000000" / "train_000000.pdmsh").interior.point_data.keys()


def test_wall_from_sdf_mode_runs(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _domain().save(src / "case_000000" / "case_000000.pdmsh")
    cmd = [sys.executable, str(SRC / "precompute_harmonic.py"), "--src", str(src), "--dst", str(dst), "--boundary", "outer",
           "--wall-from-sdf", "--wall-sdf-tol", "1e-6", "--workers", "1"]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "DONE ok=1 failed=0" in r.stdout, r.stdout
    pd = DomainMesh.load(dst / "case_000000" / "case_000000.pdmsh").interior.point_data
    assert "hm_outer" in pd.keys() and "hm_farfield" in pd.keys() and not any(k.startswith("hx_") for k in pd.keys())
    assert torch.allclose(pd["hm_outer"] + pd["hm_farfield"], torch.ones_like(pd["hm_outer"]), atol=1e-5)
