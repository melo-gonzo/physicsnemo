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

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import domain_transforms
import numpy as np
import pytest
import symmetry
import torch
from datasets import build_dataset
from dedup import connectivity_fingerprint, points_fingerprint
from domain_transforms import (
    AddHemisphereBoundaries,
    AddSymmetryPlane,
    _curator_build_hemisphere,
)
from merge_global_data import MeshReaderWithGlobalData
from omegaconf import OmegaConf
from symmetry import (
    RootCap,
    cap_md5,
    check_root,
    load_root_cap,
    outline_distance,
    outline_tolerance,
    root_loop,
)

from physicsnemo.datapipes.protocols import DatasetBase
from physicsnemo.datapipes.transforms.mesh.base import MeshTransform
from physicsnemo.mesh import DomainMesh, Mesh

_RECIPE_ROOT = Path(__file__).resolve().parents[1]
_TOOL = _RECIPE_ROOT / "tools" / "build_symmetry_check.py"

#: The synthetic root section: an ellipse polygon in (x, z), in the wing's units.
CENTRE, AXES, N_CAP = (2000.0, 300.0), (600.0, 120.0), 48


def ellipse(n: int = N_CAP, scale: float = 1.0) -> np.ndarray:
    """The root section: ``n`` vertices of the ellipse (scaled about its centre), float32."""
    t = 2 * np.pi * np.arange(n) / n
    return np.stack(
        [
            CENTRE[0] + scale * AXES[0] * np.cos(t),
            CENTRE[1] + scale * AXES[1] * np.sin(t),
        ],
        axis=1,
    ).astype(np.float32)


def cone_stl(
    ring_xz: np.ndarray, height: float = 800.0
) -> tuple[np.ndarray, np.ndarray]:
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


def on_cap_wing(shift: float = 0.0, copies: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """A wing on the cap, open at its root: a band from the root (the cap's vertices and its chords'
    midpoints, twice as dense as the cap, as the CFD wings are denser than the STL) up to a smaller
    tip section, with ``copies`` repeats of every root cell, as the stored wings repeat theirs, and
    the root shifted along x by ``shift``."""
    cap = ellipse().astype(np.float64)
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


def write_case(root: Path, case: str, wing, stl=None) -> None:
    """A HiLift case folder: the wing with HiLift's raw cell fields in its DomainMesh file, with the
    case's freestream conditions, and the source STL when given."""
    points, cells = wing
    g = torch.Generator().manual_seed(0)
    n = len(cells)
    vehicle = Mesh(
        points=torch.from_numpy(points),
        cells=torch.from_numpy(cells),
        cell_data={
            "PROJ(AVG(P))": 0.1 + 0.01 * torch.rand(n, generator=g),
            "PROJ(AVG(T))": 500.0 + torch.rand(n, generator=g),
            "PROJ(AVG(RHO))": 2e-3 + 1e-5 * torch.rand(n, generator=g),
            "PROJ(AVG(U))": torch.randn(n, 3, generator=g),
            "AVG(TAU_WALL)": 1e-4 * torch.randn(n, 3, generator=g),
        },
    )
    DomainMesh(
        interior=Mesh(points=torch.zeros(0, 3)),
        boundaries={"vehicle": vehicle},
        global_data={
            "U_inf": torch.tensor([3000.0, 0.0, 600.0]),
            "p_inf": torch.tensor(0.1),
            "rho_inf": torch.tensor(2e-3),
            "T_inf": torch.tensor(500.0),
            "L_ref": torch.tensor(275.8),
        },
    ).save(root / case / f"domain_{case.removeprefix('geo_')}.pdmsh")
    if stl is not None:
        Mesh(points=torch.from_numpy(stl[0]), cells=torch.from_numpy(stl[1])).save(
            root / case / f"{case}.stl.pmsh"
        )


def cap_key(stl) -> str:
    """HL-DUP's key of an STL's root cap."""
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
    stl = cone_stl(ellipse())
    write_case(tmp_path, "geo_LHC001_AoA_10", on_cap_wing(), stl)
    write_case(
        tmp_path, "geo_LHC002_AoA_10", on_cap_wing(), cone_stl(ellipse(scale=0.5))
    )
    key = cap_key(stl)
    bits = np.sort(stl[0][: N_CAP + 1].view(np.uint32), axis=0)
    assert key == hashlib.md5(np.ascontiguousarray(bits).tobytes()).hexdigest()
    cap = load_root_cap(tmp_path, [key])
    assert cap.file == "geo_LHC001_AoA_10/geo_LHC001_AoA_10.stl.pmsh"
    assert (cap.md5, cap.n_cells, cap.y_eps) == (key, N_CAP, 1e-3)
    assert cap.outline.dtype == np.float32 and cap.outline.shape == (N_CAP, 2)
    ring = ellipse()
    start = int(np.flatnonzero((ring == cap.outline[0]).all(axis=1))[0])
    order = (start + np.arange(N_CAP)) % N_CAP
    assert np.array_equal(cap.outline, ring[order]) or np.array_equal(
        cap.outline, ring[(start - np.arange(N_CAP)) % N_CAP]
    )
    assert load_root_cap(str(tmp_path), (key,)) is cap  # read once per process


def test_load_root_cap_refuses_a_foreign_key_and_a_bad_outline(tmp_path):
    """A cap key outside the accepted set, two loops and a vertex of degree 4 are refused."""
    stl = cone_stl(ellipse())
    write_case(tmp_path, "geo_LHC001_AoA_10", on_cap_wing(), stl)
    with pytest.raises(ValueError, match="not among the accepted"):
        load_root_cap(tmp_path, ["0" * 32])
    a, b = cone_stl(ellipse()), cone_stl(ellipse(scale=0.3) + [5000.0, 0.0])
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
    cap = _cap(ellipse())
    points, cells = on_cap_wing()
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
    shifted = check_root(*on_cap_wing(shift=3 * cap.tau), cap)
    assert not shifted["passed"] and shifted["max_vertex_distance"] > cap.tau
    no_root = points.copy()
    no_root[:, 1] += 1.0
    nothing = check_root(no_root, cells, cap)
    assert not nothing["passed"] and nothing["n_on_plane"] == 0
    assert nothing["max_vertex_distance"] is None


def test_points_fingerprint_samples_rows_as_the_connectivity_fingerprint():
    """The points' fingerprint is the md5 of the same strided rows as float32; it changes with a
    sampled row; the connectivity fingerprint is unchanged by the shared helper."""
    points, cells = on_cap_wing()
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
    stl = cone_stl(ellipse())
    key = cap_key(stl)
    write_case(root, "geo_LHC001_AoA_10", on_cap_wing(), stl)
    write_case(root, "geo_LHC002_AoA_10", on_cap_wing(copies=0))
    cap = load_root_cap(root, [key])
    write_case(root, "geo_LHC003_AoA_10", on_cap_wing(shift=3 * cap.tau))
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


### --------------------------------------------------------------------------------------------
### AddSymmetryPlane, the reader hook and the configs
### --------------------------------------------------------------------------------------------

_HL = OmegaConf.load(_RECIPE_ROOT / "datasets" / "highlift_surface.yaml")
HL_RADIUS = float(_HL.hemisphere.radius)
CASE = "geo_LHC001_AoA_10"


def _transform(root, key, **kw) -> AddSymmetryPlane:
    return AddSymmetryPlane(
        radius=HL_RADIUS,
        cap_root=str(root),
        cap_md5=[key],
        n_theta=int(_HL.hemisphere.n_theta),
        n_phi=int(_HL.hemisphere.n_phi),
        **kw,
    )


def _domain(points, cells) -> DomainMesh:
    """What the reader hands a whole-surface transform: the whole wing under its name."""
    wing = Mesh(points=torch.from_numpy(points), cells=torch.from_numpy(cells))
    return DomainMesh(
        interior=Mesh(points=wing.points[:0]),
        boundaries={"vehicle": wing},
        global_data={"U_inf": torch.tensor([3000.0, 0.0, 600.0])},
    )


def _root(tmp_path, *shifted: str) -> tuple[Path, str, RootCap]:
    """A dataset root: CASE stores the STL and a wing on its cap; each of ``shifted`` a wing whose
    root lies 3 tau off it."""
    stl = cone_stl(ellipse())
    key = cap_key(stl)
    write_case(tmp_path, CASE, on_cap_wing(), stl)
    cap = load_root_cap(tmp_path, [key])
    for case in shifted:
        write_case(tmp_path, case, on_cap_wing(shift=3 * cap.tau))
    return tmp_path, key, cap


def _shoelace(xz: np.ndarray) -> float:
    x, z = xz[:, 0], xz[:, 1]
    return 0.5 * abs(float((x * np.roll(z, -1) - np.roll(x, -1) * z).sum()))


def _open_edges(*meshes: Mesh) -> set:
    """The edges used by exactly one cell of the meshes together, as pairs of vertex coordinates."""
    count: dict = {}
    for m in meshes:
        tri = m.points.numpy()[m.cells.numpy()]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            for e in zip(map(tuple, tri[:, a]), map(tuple, tri[:, b])):
                key = tuple(sorted(e))
                count[key] = count.get(key, 0) + 1
    return {e for e, c in count.items() if c == 1}


def test_add_symmetry_plane_builds_the_disk_minus_the_cap(tmp_path):
    """On a wing whose root lies on the cap, the plane is curator's disk: its outer loop is the
    hemisphere's equator bit for bit and its hole the cap's outline, as stored; its area equals the
    disk's minus the hole's to roundoff; every normal is +y (into the fluid); no cell lies in the
    hole; with the inlet and outlet it closes every edge but the outline's (a watertight seam)."""
    root, key, cap = _root(tmp_path)
    domain = _domain(*on_cap_wing())
    out = _transform(root, key).apply_to_domain(domain, case=CASE)
    assert set(out.boundary_names) == {"vehicle", "symmetry"}
    assert out.boundaries["vehicle"] is domain.boundaries["vehicle"]
    plane = out.boundaries["symmetry"]
    assert plane.points.dtype == torch.float32
    hemisphere, equator = _curator_build_hemisphere(HL_RADIUS, 64, 32)
    n_eq = len(equator)
    assert torch.equal(plane.points[:n_eq], hemisphere.points[equator])
    hole = torch.zeros(N_CAP, 3)
    hole[:, [0, 2]] = torch.from_numpy(cap.outline)
    assert torch.equal(plane.points[n_eq:], hole)
    p = plane.points.double()[plane.cells]
    normal = torch.linalg.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    assert bool((normal[:, 1] > 0).all())
    assert bool((normal[:, [0, 2]] == 0).all())
    disk = _shoelace(hemisphere.points[equator][:, [0, 2]].double().numpy())
    expected = disk - _shoelace(cap.outline.astype(np.float64))
    np.testing.assert_allclose(0.5 * float(normal[:, 1].sum()), expected, rtol=1e-12)
    centroid = p.mean(dim=1)[:, [0, 2]].numpy()
    v = cap.outline.astype(np.float64)[None] - centroid[:, None]
    w = np.roll(v, -1, axis=1)
    winding = np.arctan2(
        v[..., 0] * w[..., 1] - v[..., 1] * w[..., 0], (v * w).sum(-1)
    ).sum(axis=1)
    assert bool((np.abs(winding) < np.pi).all())  # every centroid outside the hole
    far = AddHemisphereBoundaries(radius=HL_RADIUS)(
        Mesh(
            points=torch.zeros(1, 3),
            global_data={"U_inf": torch.tensor([3.0, 0.0, 0.6])},
        )
    ).boundaries
    loop = [tuple(map(float, q)) for q in hole.numpy()]
    outline_edges = {
        tuple(sorted((loop[i], loop[(i + 1) % N_CAP]))) for i in range(N_CAP)
    }
    assert _open_edges(far["inlet"], far["outlet"], plane) == outline_edges


def test_a_root_off_the_cap_gets_no_plane_logged_once(tmp_path, caplog, monkeypatch):
    """A wing whose root lies more than tau off the cap gets no plane, and one warning per case with
    both maxima and tau; a second load does not log again. A plane already present, a call on a bare
    mesh and a cap key outside the accepted set are refused."""
    root, key, cap = _root(tmp_path)
    monkeypatch.setattr(domain_transforms, "_SYMMETRY_FAILURES_LOGGED", set())
    t = _transform(root, key)
    domain = _domain(*on_cap_wing(shift=3 * cap.tau))
    with caplog.at_level(logging.WARNING, logger="domain_transforms"):
        first = t.apply_to_domain(domain, case="geo_LHC003_AoA_10")
        second = t.apply_to_domain(domain, case="geo_LHC003_AoA_10")
    assert first is domain and second is domain
    logged = [
        r.getMessage() for r in caplog.records if "no symmetry plane" in r.getMessage()
    ]
    assert len(logged) == 1
    assert "'geo_LHC003_AoA_10'" in logged[0] and repr(cap.tau) in logged[0]
    exact = check_root(*on_cap_wing(shift=3 * cap.tau), cap)
    assert repr(exact["max_vertex_distance"]) in logged[0]
    placed = t.apply_to_domain(_domain(*on_cap_wing()), case=CASE)
    with pytest.raises(ValueError, match="exists already"):
        t.apply_to_domain(placed, case=CASE)
    with pytest.raises(TypeError, match="whole_surface_transforms"):
        t(placed.boundaries["vehicle"])
    with pytest.raises(ValueError, match="not among the accepted"):
        _transform(root, "0" * 32).apply_to_domain(_domain(*on_cap_wing()), case=CASE)


def _load_tool():
    spec = importlib.util.spec_from_file_location("build_symmetry_check", _TOOL)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def _record(root: Path, key: str, out: Path) -> dict:
    """The record of every case below ``root``, built in-process by the tool's shard and merge."""
    tool = _load_tool()
    common = dict(cap_md5=[key], y_eps=1e-3, cases="all")
    part = out.with_suffix(".jsonl")
    tool.run_shard(
        argparse.Namespace(shard="0/1", part=str(part), workers=1, **common), root
    )
    tool.merge(argparse.Namespace(merge=[str(part)], out=str(out), **common), root)
    return json.loads(out.read_text())


def test_the_record_agrees_with_the_exact_check_and_is_refused_on_mismatch(tmp_path):
    """With the record the transform gives what the exact check gives (the plane, bit for bit, or
    none); a case the record lacks, a wing whose connectivity or points differ from the record's,
    and a record made with another tolerance are refused, naming the case."""
    root, key, cap = _root(tmp_path, "geo_LHC003_AoA_10")
    path = tmp_path / "record.json"
    record = _record(root, key, path)
    assert (
        record["cases"][CASE]["passed"]
        and not record["cases"]["geo_LHC003_AoA_10"]["passed"]
    )
    exact, recorded = (
        _transform(root, key),
        _transform(root, key, check_record=str(path)),
    )
    for case, wing in (
        (CASE, on_cap_wing()),
        ("geo_LHC003_AoA_10", on_cap_wing(shift=3 * cap.tau)),
    ):
        a = exact.apply_to_domain(_domain(*wing), case=case)
        b = recorded.apply_to_domain(_domain(*wing), case=case)
        assert a.boundary_names == b.boundary_names
        if "symmetry" in a.boundary_names:
            assert torch.equal(
                a.boundaries["symmetry"].points, b.boundaries["symmetry"].points
            )
            assert torch.equal(
                a.boundaries["symmetry"].cells, b.boundaries["symmetry"].cells
            )
    with pytest.raises(KeyError, match="geo_LHC009_AoA_10"):
        recorded.apply_to_domain(_domain(*on_cap_wing()), case="geo_LHC009_AoA_10")
    with pytest.raises(ValueError, match=f"{CASE!r}.*points_fingerprint"):
        recorded.apply_to_domain(_domain(*on_cap_wing(shift=3 * cap.tau)), case=CASE)
    points, cells = on_cap_wing()
    with pytest.raises(ValueError, match=f"{CASE!r}.*connectivity_fingerprint"):
        recorded.apply_to_domain(_domain(points, cells[:, [1, 2, 0]].copy()), case=CASE)
    other = json.loads(path.read_text())
    other["tolerance"]["tau"] *= 2
    (tmp_path / "other.json").write_text(json.dumps(other))
    with pytest.raises(ValueError, match="made with"):
        _transform(
            root, key, check_record=str(tmp_path / "other.json")
        ).apply_to_domain(_domain(*on_cap_wing()), case=CASE)


class _DropSymmetry(MeshTransform):
    def __call__(self, mesh):
        return mesh

    def apply_to_domain(self, domain):
        return DomainMesh(
            interior=domain.interior,
            boundaries={
                n: domain.boundaries[n]
                for n in domain.boundary_names
                if n != "symmetry"
            },
            global_data=domain.global_data,
        )


def test_reader_whole_surface_transforms_add_boundaries_without_domain_boundaries(
    tmp_path,
):
    """Without domain_boundaries, a whole-surface transform that adds a boundary makes the sample a
    DomainMesh: the interior is the subsampled surface (the bare reader's draw), the boundaries what
    the transforms add; the transform gets the case name (the failing case's warning names it). A
    transform that drops a boundary is refused."""
    root, key, _ = _root(tmp_path, "geo_LHC003_AoA_10")
    kw = dict(
        pattern="geo_LHC*/*.pdmsh/_tensordict/boundaries/vehicle", subsample_n_cells=50
    )
    bare = MeshReaderWithGlobalData(root, **kw)
    reader = MeshReaderWithGlobalData(
        root, whole_surface_transforms=[_transform(root, key)], **kw
    )
    for r in (bare, reader):
        r.set_generator(torch.Generator().manual_seed(3))
    for i, names in enumerate((["symmetry"], [])):
        mesh, _ = bare[i]
        domain, meta = reader[i]
        assert isinstance(domain, DomainMesh)
        assert torch.equal(domain.interior.points, mesh.points)
        assert torch.equal(domain.interior.cells, mesh.cells)
        assert list(domain.boundary_names) == names == list(meta["boundary_names"])
    dropping = MeshReaderWithGlobalData(
        root, whole_surface_transforms=[_transform(root, key), _DropSymmetry()], **kw
    )
    with pytest.raises(ValueError, match=r"dropped the boundaries \['symmetry'\]"):
        dropping[0]


def test_highlift_pipeline_moves_the_symmetry_plane_with_the_wing(tmp_path):
    """highlift_surface.yaml builds the plane in the reader from the whole wing and the far field
    first in its transforms; through the whole chain (augmentations, CenterMesh, the L_ref scaling)
    the plane undergoes the far field's similarity, and keeps its cells."""
    root, key, _ = _root(tmp_path)
    cfg = OmegaConf.merge(
        _HL,
        {
            "dataset_paths": {
                "highlift_aero_ml": str(root),
                "highlift_aero_ml_duplicate_masks": None,
                "highlift_aero_ml_symmetry_check": None,
            },
            "sampling_resolution": 1500,
        },
    )
    cfg.pipeline.reader.whole_surface_transforms[0].cap_md5 = [key]
    dataset_ = build_dataset(cfg, augment=True, device=None, num_workers=1)
    try:
        domain, _ = dataset_[0]
    finally:
        DatasetBase.close(dataset_)
    assert set(domain.boundary_names) == {"vehicle", "inlet", "outlet", "symmetry"}
    u_inf = torch.tensor([3000.0, 0.0, 600.0])
    far = AddHemisphereBoundaries(radius=HL_RADIUS)(
        Mesh(points=torch.zeros(1, 3), global_data={"U_inf": u_inf})
    ).boundaries
    raw_plane = _transform(root, key).plane(load_root_cap(root, [key]).outline)
    assert torch.equal(domain.boundaries["symmetry"].cells, raw_plane.cells)
    names = ("inlet", "outlet")
    raw = torch.cat([far[n].points for n in names]).double()
    out = torch.cat([domain.boundaries[n].points for n in names]).double()
    rc, oc = raw - raw.mean(0), out - out.mean(0)
    u, sv, vt = torch.linalg.svd(rc.T @ oc)
    rot = (u @ vt).T
    scale = sv.sum() / rc.square().sum()
    shift = out.mean(0) - scale * raw.mean(0) @ rot.T
    mapped = scale * raw_plane.points.double() @ rot.T + shift
    torch.testing.assert_close(
        mapped, domain.boundaries["symmetry"].points.double(), atol=1e-3, rtol=0
    )


def test_highlift_dataset_reads_the_plane_with_the_far_fields_hemisphere():
    """The HiLift dataset file's reader names AddSymmetryPlane with the radius and divisions the
    far field's AddHemisphereBoundaries gets (both from the file's one hemisphere block), the dataset
    root as the cap's root and the record's dataset path; the far field stays the first transform."""
    reader = OmegaConf.to_container(_HL.pipeline.reader, resolve=False)
    (sym,) = reader["whole_surface_transforms"]
    hemi = OmegaConf.to_container(_HL.pipeline.transforms, resolve=False)[0]
    assert sym["_target_"] == "${dp:AddSymmetryPlane}"
    assert hemi["_target_"] == "${dp:AddHemisphereBoundaries}"
    for k in ("radius", "n_theta", "n_phi"):
        assert sym[k] == hemi[k] == f"${{hemisphere.{k}}}"
    assert (HL_RADIUS, _HL.hemisphere.n_theta, _HL.hemisphere.n_phi) == (
        24533.41796875,
        64,
        32,
    )
    assert sym["cap_root"] == "${train_datadir}"
    assert sym["check_record"] == "${dataset_paths.highlift_aero_ml_symmetry_check}"
    assert sorted(sym["cap_md5"]) == [
        "480578a2e673fc572e0b2d3194442721",
        "d3b5c68bdac50aa7c0cc688443795dc2",
    ]
    paths = OmegaConf.load(_RECIPE_ROOT / "datasets" / "dataset_paths.yaml")
    # Declared (??? here; a deployment fills it in, so its value is not asserted).
    assert "highlift_aero_ml_symmetry_check" in paths.keys()


_HL_ROOT = os.environ.get("HIGHLIFT_AERO_ML_ROOT")
_HL_RECORD = os.environ.get("HIGHLIFT_SYMMETRY_CHECK_RECORD")


@pytest.mark.skipif(
    _HL_ROOT is None or _HL_RECORD is None,
    reason="needs HiLiftAeroML case files and the symmetry check record (set HIGHLIFT_AERO_ML_ROOT "
    "to the -medium dataset root and HIGHLIFT_SYMMETRY_CHECK_RECORD to its record)",
)
@pytest.mark.parametrize(
    "case", ["geo_LHC001_AoA_10", "geo_LHC026_AoA_18", "geo_LHC105_AoA_22"]
)
def test_the_record_agrees_with_the_exact_check_on_real_wings(case):
    """On real wings (LHC026 and LHC105 have broken roots), the record's entry equals the exact
    check of the stored wing, and its fingerprints are the wing's."""
    from symmetry import load_check_record, recorded_verdict

    cap = load_root_cap(
        _HL_ROOT, list(_HL.pipeline.reader.whole_surface_transforms[0].cap_md5)
    )
    (wing,) = sorted(
        (Path(_HL_ROOT) / case).glob("*.pdmsh/_tensordict/boundaries/vehicle")
    )
    points, cells = symmetry.read_memmap_mesh(wing)
    entry = recorded_verdict(load_check_record(_HL_RECORD), cap, case, points, cells)
    exact = json.loads(json.dumps(check_root(points, cells, cap)))
    assert {k: entry[k] for k in exact} == exact
