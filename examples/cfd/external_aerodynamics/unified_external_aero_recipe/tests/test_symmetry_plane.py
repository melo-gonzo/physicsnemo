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

"""Tests for HiLift's symmetry plane: the source STL's root cap, its tolerance, the per-case check
(src/symmetry.py) and its record (tools/build_symmetry_check.py)."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import symmetry
import torch
from dedup import connectivity_fingerprint, points_fingerprint
from symmetry import (
    RootCap,
    cap_md5,
    check_root,
    load_root_cap,
    outline_distance,
    outline_tolerance,
    root_loop,
)

from physicsnemo.mesh import DomainMesh, Mesh

_RECIPE_ROOT = Path(__file__).resolve().parents[1]
_TOOL = _RECIPE_ROOT / "tools" / "build_symmetry_check.py"

#: The synthetic root section: an ellipse polygon in (x, z), in the wing's units.
CENTRE, AXES, N_CAP = (2000.0, 300.0), (600.0, 120.0), 48


def _ellipse(n: int = N_CAP, scale: float = 1.0) -> np.ndarray:
    t = 2 * np.pi * np.arange(n) / n
    return np.stack(
        [
            CENTRE[0] + scale * AXES[0] * np.cos(t),
            CENTRE[1] + scale * AXES[1] * np.sin(t),
        ],
        axis=1,
    ).astype(np.float32)


def _cone(ring_xz: np.ndarray, height: float = 800.0) -> tuple[np.ndarray, np.ndarray]:
    """A closed half-body (an STL): the ring on y = 0 fanned to its centre (the root cap) and to an
    apex at ``y = height``."""
    n = len(ring_xz)
    pts = np.zeros((n + 2, 3), np.float32)
    pts[:n, [0, 2]] = ring_xz
    pts[n, [0, 2]] = ring_xz.mean(axis=0)
    pts[n + 1] = [pts[n, 0], height, pts[n, 2]]
    i, j = np.arange(n), (np.arange(n) + 1) % n
    cap = np.stack([np.full(n, n), j, i], axis=1)
    side = np.stack([i, j, np.full(n, n + 1)], axis=1)
    return pts, np.concatenate([cap, side]).astype(np.int64)


def _wing(shift: float = 0.0, copies: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """A wing on the cap, open at its root: a band from the root (the cap's vertices and its chords'
    midpoints, twice as dense as the cap, as the CFD wings are denser than the STL) up to a smaller
    tip section, with ``copies`` repeats of every root cell, as the stored wings repeat theirs, and
    the root shifted along x by ``shift``."""
    cap = _ellipse().astype(np.float64)
    root = np.empty((2 * N_CAP, 2))
    root[0::2], root[1::2] = cap, 0.5 * (cap + np.roll(cap, -1, axis=0))
    m = len(root)
    tip = CENTRE + 0.5 * (root - CENTRE)
    pts = np.zeros((2 * m, 3), np.float32)
    pts[:m, [0, 2]] = root + [shift, 0.0]
    pts[m:, [0, 2]] = tip
    pts[m:, 1] = 400.0
    i, j = np.arange(m), (np.arange(m) + 1) % m
    band = np.concatenate(
        [np.stack([i, j, m + i], 1), np.stack([j, m + j, m + i], 1)]
    ).astype(np.int64)
    root_cells = band[:m]
    return pts, np.concatenate([band, *[root_cells[:, ::-1]] * copies])


def _write_case(root: Path, case: str, wing, stl=None) -> None:
    points, cells = wing
    vehicle = Mesh(points=torch.from_numpy(points), cells=torch.from_numpy(cells))
    DomainMesh(
        interior=Mesh(points=torch.zeros(0, 3)),
        boundaries={"vehicle": vehicle},
        global_data={"U_inf": torch.tensor([1.0, 0.0, 0.1])},
    ).save(root / case / f"domain_{case.removeprefix('geo_')}.pdmsh")
    if stl is not None:
        Mesh(points=torch.from_numpy(stl[0]), cells=torch.from_numpy(stl[1])).save(
            root / case / f"{case}.stl.pmsh"
        )


def _cap_key(stl) -> str:
    points, cells = stl
    return cap_md5(points, cells[(np.abs(points[cells, 1]) < 1e-3).all(axis=1)])


def _cap(outline: np.ndarray) -> RootCap:
    return RootCap(
        file="none",
        md5="none",
        n_cells=0,
        outline=outline,
        y_eps=1e-3,
        tolerance=outline_tolerance(outline),
    )


def test_load_root_cap_reads_the_first_stl_and_walks_its_outline(tmp_path):
    """The cap is the first STL's cells on the plane; its key is HL-DUP's (the md5 of the cap
    vertices' sorted float32 bits), and its outline is the cap's boundary loop in loop order, (x, z)
    as stored. A later STL with another cap is not read."""
    stl = _cone(_ellipse())
    _write_case(tmp_path, "geo_LHC001_AoA_10", _wing(), stl)
    _write_case(tmp_path, "geo_LHC002_AoA_10", _wing(), _cone(_ellipse(scale=0.5)))
    key = _cap_key(stl)
    bits = np.sort(stl[0][: N_CAP + 1].view(np.uint32), axis=0)
    assert key == hashlib.md5(np.ascontiguousarray(bits).tobytes()).hexdigest()
    cap = load_root_cap(tmp_path, [key])
    assert cap.file == "geo_LHC001_AoA_10/geo_LHC001_AoA_10.stl.pmsh"
    assert (cap.md5, cap.n_cells, cap.y_eps) == (key, N_CAP, 1e-3)
    assert cap.outline.dtype == np.float32 and cap.outline.shape == (N_CAP, 2)
    ring = _ellipse()
    start = int(np.flatnonzero((ring == cap.outline[0]).all(axis=1))[0])
    order = (start + np.arange(N_CAP)) % N_CAP
    assert np.array_equal(cap.outline, ring[order]) or np.array_equal(
        cap.outline, ring[(start - np.arange(N_CAP)) % N_CAP]
    )
    assert load_root_cap(str(tmp_path), (key,)) is cap  # read once per process


def test_load_root_cap_refuses_a_foreign_key_and_a_bad_outline(tmp_path):
    """A cap key outside the accepted set, two loops and a vertex of degree 4 are refused."""
    stl = _cone(_ellipse())
    _write_case(tmp_path, "geo_LHC001_AoA_10", _wing(), stl)
    with pytest.raises(ValueError, match="not among the accepted"):
        load_root_cap(tmp_path, ["0" * 32])
    a, b = _cone(_ellipse()), _cone(_ellipse(scale=0.3) + [5000.0, 0.0])
    two = np.concatenate([a[1][:N_CAP], b[1][:N_CAP] + len(a[0])])
    with pytest.raises(ValueError, match="more than one loop"):
        root_loop(two)
    bowtie = np.array([[0, 1, 2], [0, 3, 4]])  # two triangles touching at vertex 0
    with pytest.raises(ValueError, match="degree other than 2"):
        root_loop(bowtie)


def test_tolerance_bounds_the_sagitta_of_a_regular_polygon():
    """On a regular n-gon inscribed in a circle of radius R, every turning is 2 pi / n and every
    chord L = 2 R sin(pi / n), so h = L tan(beta) / 2 with beta = 2 pi / n + 2 eps / L; it is at least
    the true sagitta R (1 - cos(pi / n)), and tau = h + eps with eps = sqrt(2) spacing(float32(R)).
    The stored float32 vertices move each chord's bound by a few parts in a million; recomputed
    from them with the cot formula, chord by chord, the bound is exact. Points on the circle pass
    the check; points moved outward from a vertex by more than tau fail."""
    r, n = 1000.0, 64
    t = 2 * np.pi * np.arange(n) / n
    outline = (r * np.stack([np.cos(t), np.sin(t)], 1)).astype(np.float32)
    tol = outline_tolerance(outline)
    eps = np.sqrt(2.0) * float(np.spacing(np.abs(outline).max()))
    chord = 2 * r * np.sin(np.pi / n)
    h = chord * np.tan(2 * np.pi / n + 2 * eps / chord) / 2
    assert tol["n_chords"] == n
    np.testing.assert_allclose(tol["eps"], eps, rtol=1e-12)
    np.testing.assert_allclose(tol["max_h"], h, rtol=1e-5)
    v = outline.astype(np.float64)
    bounds = []
    for j in range(n):
        p, a, b, q = v[j - 1], v[j], v[(j + 1) % n], v[(j + 2) % n]

        def beta(
            u, w
        ):  # |turning| from chord u to chord w, plus its float32 uncertainty
            cos = u @ w / np.linalg.norm(u) / np.linalg.norm(w)
            return (
                np.arccos(min(1.0, cos))
                + eps / np.linalg.norm(u)
                + eps / np.linalg.norm(w)
            )

        cot = 1 / np.tan(beta(a - p, b - a)) + 1 / np.tan(beta(b - a, q - b))
        bounds.append(np.linalg.norm(b - a) / cot)
    np.testing.assert_allclose(tol["max_h"], max(bounds), rtol=1e-9)
    np.testing.assert_allclose(tol["tau"], tol["max_h"] + tol["eps"], rtol=1e-15)
    assert tol["max_h"] >= r * (1 - np.cos(np.pi / n))
    cap = _cap(outline)
    s = 2 * np.pi * np.arange(4000) / 4000
    on_circle = np.zeros((len(s), 3), np.float32)
    on_circle[:, 0], on_circle[:, 2] = r * np.cos(s), r * np.sin(s)
    got = check_root(on_circle, np.zeros((0, 3), np.int64), cap)
    assert got["passed"] and got["n_on_plane"] == len(s)
    assert got["max_vertex_distance"] <= r * (1 - np.cos(np.pi / n)) + 1e-3
    assert got["max_edge_distance"] is None and got["n_open_edges"] == 0
    out = on_circle[:1].copy()
    out[0, [0, 2]] = (r + 1.01 * tol["tau"]) * np.array([1.0, 0.0])
    got = check_root(np.concatenate([on_circle, out]), np.zeros((0, 3), np.int64), cap)
    assert not got["passed"]
    np.testing.assert_allclose(got["max_vertex_distance"], 1.01 * tol["tau"], rtol=1e-4)
    np.testing.assert_allclose(got["worst_vertex"], out[0].astype(np.float64))


def test_tolerance_refuses_a_cap_with_no_finite_bound():
    """A reversal (end angles summing to pi or more) leaves the triangle unbounded."""
    with pytest.raises(ValueError, match="no finite sagitta bound"):
        outline_tolerance(np.array([[0, 0], [10, 0], [10, 1], [9, 0.1]], np.float32))


def test_outline_distance_is_exact():
    """The point-to-polyline distance on a unit square: inside, outside a side, beyond a corner."""
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    q = np.array([[0.5, 0.25], [0.5, -2.0], [2.0, 3.0], [1.0, 0.5]])
    np.testing.assert_allclose(
        outline_distance(q, square), [0.25, 2.0, np.hypot(1, 2), 0.0], rtol=1e-15
    )


def test_check_root_passes_the_wing_on_the_cap_and_fails_a_shifted_root(monkeypatch):
    """The wing on the cap passes: its root vertices and its open edges' midpoints lie on the
    outline, the duplicate root cells do not hide its open edges, and an orphan on-plane point
    counts. Shifted by more than tau along x, it fails. Chunked reads give the same result."""
    cap = _cap(_ellipse())
    points, cells = _wing()
    got = check_root(points, cells, cap)
    m = 2 * N_CAP
    assert got["passed"]
    assert (got["n_on_plane"], got["n_root_cells"], got["n_open_edges"]) == (m, m, m)
    assert got["max_vertex_distance"] < 1e-3 and got["max_edge_distance"] < cap.tau
    torch_got = check_root(torch.from_numpy(points), torch.from_numpy(cells), cap)
    assert torch_got == got
    monkeypatch.setattr(symmetry, "CHUNK", 7)
    assert check_root(points, cells, cap) == got
    orphan = np.concatenate([points, [[CENTRE[0], 0.0, CENTRE[1]]]]).astype(np.float32)
    got_orphan = check_root(orphan, cells, cap)
    assert not got_orphan["passed"] and got_orphan["n_on_plane"] == m + 1
    shifted = check_root(*_wing(shift=3 * cap.tau), cap)
    assert not shifted["passed"] and shifted["max_vertex_distance"] > cap.tau
    no_root = points.copy()
    no_root[:, 1] += 1.0
    nothing = check_root(no_root, cells, cap)
    assert not nothing["passed"] and nothing["n_on_plane"] == 0
    assert nothing["max_vertex_distance"] is None


def test_points_fingerprint_samples_rows_as_the_connectivity_fingerprint():
    """The points' fingerprint is the md5 of the same strided rows as float32; it changes with a
    sampled row; the connectivity fingerprint is unchanged by the shared helper."""
    points, cells = _wing()
    fp = points_fingerprint(points)
    assert fp == points_fingerprint(torch.from_numpy(points))
    n = len(points)
    idx = (np.arange(1024) * (n - 1)) // 1023
    assert (
        fp == f"{n}x3:" + hashlib.md5(points[idx].astype("<f4").tobytes()).hexdigest()
    )
    moved = points.copy()
    moved[-1, 0] += 1.0
    assert points_fingerprint(moved) != fp
    m = len(cells)
    idx = (np.arange(1024) * (m - 1)) // 1023
    assert connectivity_fingerprint(cells) == (
        f"{m}x3:" + hashlib.md5(cells[idx].astype("<i8").tobytes()).hexdigest()
    )


def _tool(*args, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_TOOL), *map(str, args)],
        capture_output=True,
        text=True,
        check=check,
    )


def test_build_symmetry_check_shards_merge_and_refusals(tmp_path):
    """Two shards and their merge: one record with the cap, its tolerance and every case's verdict
    equal to the exact check, with both fingerprints; a rerun of a shard resumes; a missing part, a
    duplicate case and a part made with another cap are refused."""
    root = tmp_path / "root"
    stl = _cone(_ellipse())
    key = _cap_key(stl)
    _write_case(root, "geo_LHC001_AoA_10", _wing(), stl)
    _write_case(root, "geo_LHC002_AoA_10", _wing(copies=0))
    cap = load_root_cap(root, [key])
    _write_case(root, "geo_LHC003_AoA_10", _wing(shift=3 * cap.tau))
    common = ["--root", root, "--cap-md5", key, "0" * 32, "--cases", "all"]
    parts = [tmp_path / "p0.jsonl", tmp_path / "p1.jsonl"]
    _tool(*common, "--shard", "0/2", "--part", parts[0], "--workers", 2)
    _tool(*common, "--shard", "1/2", "--part", parts[1])
    lines0 = parts[0].read_text()
    assert (
        "0 cases to check, 2 already"
        in _tool(*common, "--shard", "0/2", "--part", parts[0]).stdout
    )
    assert parts[0].read_text() == lines0
    out = tmp_path / "record.json"
    _tool(*common, "--merge", *parts, "--out", out)
    rec = json.loads(out.read_text())
    assert rec["format"] == "hl-symmetry-check/1" and rec["dataset_root"] == str(root)
    assert rec["cap"] == {
        "file": cap.file,
        "md5": key,
        "n_cells": N_CAP,
        "n_outline_vertices": N_CAP,
        "y_eps": 1e-3,
    }
    assert rec["tolerance"] == json.loads(json.dumps(cap.tolerance))
    assert rec["tolerance"]["tau"] == cap.tau
    assert sorted(rec["cases"]) == [f"geo_LHC00{i}_AoA_10" for i in (1, 2, 3)]
    for case, entry in rec["cases"].items():
        points, cells = symmetry.read_memmap_mesh(
            next((root / case).glob("*.pdmsh"))
            / "_tensordict"
            / "boundaries"
            / "vehicle"
        )
        exact = json.loads(json.dumps(check_root(points, cells, cap)))
        assert {k: entry[k] for k in exact} == exact
        assert entry["connectivity_fingerprint"] == connectivity_fingerprint(cells)
        assert entry["points_fingerprint"] == points_fingerprint(points)
        assert entry["passed"] == (case != "geo_LHC003_AoA_10")
    bad = _tool(*common, "--merge", parts[0], "--out", out, check=False)
    assert bad.returncode != 0 and "miss 1 cases" in bad.stderr
    bad = _tool(
        *common, "--merge", parts[0], parts[0], parts[1], "--out", out, check=False
    )
    assert bad.returncode != 0 and "appears twice" in bad.stderr
    other = tmp_path / "other.jsonl"
    other.write_text(
        parts[1].read_text().replace(f'"tau": {json.dumps(cap.tau)}', '"tau": 1.0')
    )
    bad = _tool(*common, "--merge", parts[0], other, "--out", out, check=False)
    assert bad.returncode != 0 and "checked with cap" in bad.stderr
