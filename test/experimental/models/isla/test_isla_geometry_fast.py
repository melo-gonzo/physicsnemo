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

"""The fast search of the prediction points' boundary geometry against the reference.

``exact_boundary_geometry(kernel="fast")`` traverses the same tree in another
layout; every output of the search, the winding number, the distance, the direction
and their gradients must equal the reference's bitwise, on closed, open, folded and
repeated surfaces, at ties, on the wall and in float32 and float64.
"""

import pytest
import torch

from physicsnemo.experimental.models.isla import geometry, geometry_fast
from physicsnemo.experimental.models.isla.geometry import exact_boundary_geometry
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.geometry._cell_normals import compute_cell_normals
from physicsnemo.mesh.primitives.surfaces import cube_surface
from test.experimental.models.isla._meshes import orthogonal, outputs, transform
from test.experimental.models.isla.test_isla_geometry_thin import (
    _capped_cube,
    _face_queries,
    _rotation,
    _soup,
)
from test.experimental.models.isla.test_isla_interior_geometry import (
    _box_ties,
    _box_wall,
    _circle,
    _finned_cube,
    _interior,
    _naca,
    _near_wall,
    _polygon,
    _sphere,
    _torus,
)

D = torch.float64

_DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
    ),
]


def _cases():
    """(name, surface, queries), float64: closed, open, folded, repeated surfaces and
    a soup, with on-wall, tied and far queries; in 3-D, then in 2-D (a NACA section
    with its sharp trailing edge, a square's ties and walls in a generic pose, a
    circle's vertices and edge midpoints on the wall, a square with a folded fin)."""
    g = torch.Generator().manual_seed(0)
    sphere = _sphere(3)
    edge_mid = sphere.points[sphere.cells[:, :2]].mean(1)
    yield (
        "sphere",
        sphere,
        torch.cat(
            [
                _near_wall(3, 1500, 1, band=0.5),
                sphere.points,
                edge_mid,
                sphere.cell_centroids,
            ]
        ),
    )
    tor = _torus(24, 12)
    t = torch.linspace(0, 6.28, 64, dtype=D)
    core = torch.stack([torch.cos(t), torch.sin(t), 0 * t], -1)
    yield "torus", tor, torch.cat([torch.randn(1500, 3, dtype=D, generator=g), core])
    yield (
        "torus repeated",
        Mesh(points=tor.points, cells=torch.cat([tor.cells, tor.cells[:40]])),
        core,
    )
    R, t = orthogonal(3, 7), torch.tensor([1.0, 2.0, 3.0], dtype=D) / 3
    for name, (mesh, q) in (
        ("box ties", _box_ties(3)[:2]),
        ("box wall", _box_wall(3)[:2]),
    ):
        yield name, transform(mesh, R, t), q @ R.T + t
    yield (
        "finned cube",
        _finned_cube(),
        torch.rand(1500, 3, dtype=D, generator=g) * 2.4 - 0.7,
    )
    cube = cube_surface.load().subdivide(levels=2, filter="linear")
    open_cube = Mesh(
        points=cube.points.double(), cells=cube.cells[cube.cell_centroids[:, 2] < 0.49]
    )
    yield "open cube", open_cube, torch.randn(1500, 3, dtype=D, generator=g) * 2
    tri = torch.rand(60, 3, 3, dtype=D, generator=g)
    soup = Mesh(points=tri.reshape(-1, 3), cells=torch.arange(180).reshape(60, 3))
    yield "soup", soup, torch.rand(1500, 3, dtype=D, generator=g) * 1.4 - 0.2
    flat = Mesh(
        points=cube.points.double(), cells=cube.cells[cube.cell_centroids[:, 2] < -0.49]
    )
    bumped = flat.points.clone()
    bumped[:, 2] += 1e-9 * torch.randn(bumped.shape[0], dtype=D, generator=g)
    q = torch.cat([flat.points, flat.cell_centroids]) + torch.tensor(
        [0.0, 0.0, 1e-3], dtype=D
    )
    yield "near-duplicate normals", Mesh(points=bumped, cells=flat.cells), q
    naca = _polygon(_naca(65))
    yield (
        "2d naca",
        naca,
        torch.cat(
            [
                torch.rand(1500, 2, dtype=D, generator=g)
                * torch.tensor([1.6, 0.6], dtype=D)
                - torch.tensor([0.3, 0.3], dtype=D),
                naca.points,
                naca.cell_centroids,
            ]
        ),
    )
    R2, t2 = orthogonal(2, 7), torch.tensor([1.0, 2.0], dtype=D) / 3
    for name, (mesh, q) in (
        ("2d box ties", _box_ties(2)[:2]),
        ("2d box wall", _box_wall(2)[:2]),
    ):
        yield name, transform(mesh, R2, t2), q @ R2.T + t2
    circle = _circle(48)
    yield (
        "2d circle on the wall",
        circle,
        torch.cat(
            [circle.points, circle.cell_centroids, _near_wall(2, 500, 5, band=0.3)]
        ),
    )
    square = _polygon(
        torch.tensor([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], dtype=D)
    )
    fin = torch.tensor([[0.5, 0.0], [1.5, 0.0]], dtype=D)
    sheet = torch.tensor([[4, 5]])
    folded = Mesh(
        points=torch.cat([square.points, fin]),
        cells=torch.cat([square.cells, sheet, sheet.flip(-1)]),
    )
    off = torch.tensor([1e-4, 1e-2, 0.1], dtype=D)[
        torch.randint(3, (600,), generator=g)
    ]
    side = torch.where(torch.rand(600, dtype=D, generator=g) < 0.5, 1.0, -1.0)
    along = 0.6 + 0.8 * torch.rand(600, dtype=D, generator=g)
    yield (
        "2d folded fin",
        folded,
        torch.cat(
            [
                torch.stack([along, side * off], -1),
                torch.rand(600, 2, dtype=D, generator=g) * 2.4 - 0.9,
            ]
        ),
    )


def _run(kernel, q, points, cells, monkeypatch):
    """d, u, their gradients in the queries and the vertices, every output of the
    search and the winding number, under ``kernel``."""
    seen = {}
    for name in ("_closest_candidates", "_closest_candidates_fast"):
        op = getattr(geometry, name)

        def spy(*args, _op=op):
            seen["search"] = _op(*args)
            return seen["search"]

        monkeypatch.setattr(geometry, name, spy)
    sign, tree_w = geometry._winding_sign, geometry._tree_winding_numbers

    def spy_sign(query, simplex, parity, tree, w=None, oriented=None):
        if w is not None:
            seen["w"] = w / (4 * torch.pi)
        return sign(query, simplex, parity, tree, w, oriented)

    def spy_tree_w(*args):
        seen["w"] = tree_w(*args)
        return seen["w"]

    segment_w = geometry._segment_winding_numbers

    def spy_segment_w(*args):
        seen["w"] = segment_w(*args)
        return seen["w"]

    monkeypatch.setattr(geometry, "_winding_sign", spy_sign)
    monkeypatch.setattr(geometry, "_tree_winding_numbers", spy_tree_w)
    monkeypatch.setattr(geometry, "_segment_winding_numbers", spy_segment_w)
    q, points = q.clone().requires_grad_(), points.clone().requires_grad_()
    d, u = exact_boundary_geometry(q, points, cells, kernel=kernel)
    (
        d.sum()
        + (
            u
            * torch.linspace(0, 1, u.numel(), dtype=u.dtype, device=u.device).reshape(
                u.shape
            )
        ).sum()
    ).backward()
    monkeypatch.undo()
    return [d, u, q.grad, points.grad, *seen["search"], seen["w"]]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("case", [c[0] for c in _cases()])
def test_fast_search_equals_the_reference_bitwise(case, dtype, device, monkeypatch):
    _, mesh, q = next(c for c in _cases() if c[0] == case)
    args = (q.to(device, dtype), mesh.points.to(device, dtype), mesh.cells.to(device))
    ### (one thread: the vertex gradient sums over the cells in a thread-dependent order)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        ref = _run("reference", *args, monkeypatch)
        fast = _run("fast", *args, monkeypatch)
    finally:
        torch.set_num_threads(threads)
    names = [
        "d",
        "u",
        "dq",
        "dp",
        "cell",
        "tied",
        "tie direction",
        "sign",
        "on wall",
        "weights",
        "w",
    ]
    for name, a, b in zip(names, ref, fast):
        assert a.dtype == b.dtype, name
        assert (
            torch.equal(a.view(torch.uint8), b.view(torch.uint8))
            if a.is_floating_point()
            else torch.equal(a, b)
        ), name


def test_geometry_kernel_auto_resolves_to_the_reference_off_cuda_and_honors_explicit_choices():
    """geometry_kernel "auto" picks the fast search on CUDA only; on CPU it resolves
    to the reference, caches it, and a model forced to the fast search gives the same
    outputs bitwise."""
    sphere = _sphere(2)
    q = _near_wall(3, 200, 3, band=0.5)
    gd = {"g": torch.tensor([1.0, 0.0, 0.0], dtype=D)}
    auto = _interior(3)
    assert (
        auto.geometry_kernel == "auto"
        and auto.resolve_geometry_kernel(q) == "reference"
    )
    fast = _interior(3)
    fast.geometry_kernel = "fast"
    assert fast.resolve_geometry_kernel(q) == "fast"
    with torch.no_grad():
        ya = outputs(
            auto(boundary_meshes={"wall": sphere}, prediction_points=q, global_data=gd)
        )
        yf = outputs(
            fast(boundary_meshes={"wall": sphere}, prediction_points=q, global_data=gd)
        )
    assert all(torch.equal(ya[k], yf[k]) for k in ya)
    assert auto._geometry_kernel_cache[1] == "reference"
    if torch.cuda.is_available():
        assert auto.resolve_geometry_kernel(q.cuda()) == (
            "fast" if geometry_fast.HAS_TRITON else "reference"
        )


_TRITON_CUDA = pytest.mark.skipif(
    not (torch.cuda.is_available() and geometry_fast.HAS_TRITON),
    reason="CUDA and Triton",
)


@_TRITON_CUDA
@pytest.mark.parametrize("n", [1, 5, 1023, 1024, 1025, 70000])
def test_run_start_kernel_equals_the_torch_scan(n):
    g = torch.Generator().manual_seed(n)
    new = torch.rand(n, generator=g) < torch.rand(1, generator=g)
    new[0] = True
    ref = torch.cummax(torch.where(new, torch.arange(n), 0), 0).values
    assert torch.equal(geometry_fast.run_start(new.cuda()).cpu(), ref)


@_TRITON_CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("d, k", [(2, 2), (3, 3), (3, 2), (2, 3)])
def test_canonical_cells_kernel_equals_the_torch_sort(d, k, dtype):
    """Ties, signed zeros, NaN and repeated vertices: the same order and parity."""
    g = torch.Generator().manual_seed(d * 10 + k)
    v = 700
    pts = torch.randint(0, 4, (v, d), generator=g).to(dtype) * 0.5
    pts[torch.rand(v, d, generator=g) < 0.03] = float("nan")
    pts[torch.rand(v, d, generator=g) < 0.05] = -0.0
    cells = torch.randint(0, v, (5000, k), generator=g)
    cells[:50, 1] = cells[:50, 0]
    ref = geometry._canonical_cells(pts, cells)
    out = geometry_fast.canonical_cells(
        pts.cuda(), cells.cuda(), geometry._canonical_cells
    )
    assert all(torch.equal(a.cpu(), b) and a.dtype == b.dtype for a, b in zip(out, ref))


def _guard_meshes():
    """(name, queries, points, cells) for the thin-cell guard, float64: GEO-THIN's caps
    and needles in a rotated, shifted cube (float32-derived and float64) and sliver
    soups at h = 1e-16 and 1e-13, random ordinary triangles with queries on their
    faces, edges and vertices, near and far, and cells of zero area (coincident and
    collinear vertices) beside others."""
    for kind in ("cap", "needle"):
        for rho in (1e-3, 1e-6):
            p, c, _ = _capped_cube(rho, kind)
            g = torch.Generator().manual_seed(int(rho * 1e7) + len(kind))
            q = torch.cat(
                [
                    _face_queries(g, p[c[0]], rho, 40),
                    torch.rand(80, 3, dtype=D, generator=g) * 1.4 - 0.2,
                ]
            )
            rot, shift = _rotation(7), torch.tensor([0.3, -0.2, 0.7], dtype=D)
            q, p = q @ rot.T + shift, p @ rot.T + shift
            yield f"{kind} {rho} float32", q.float().double(), p.float().double(), c
            yield f"{kind} {rho} float64", q, p, c
    for seed in (0, 1):
        for h in (1e-16, 1e-13):
            p, c, q = _soup(seed, h)
            yield f"soup {seed} {h}", q, p, c
    g = torch.Generator().manual_seed(5)
    tri = torch.randn(64, 3, 3, dtype=D, generator=g)
    b = torch.rand(64, 3, dtype=D, generator=g)
    t = torch.rand(64, 1, dtype=D, generator=g)
    q = torch.cat(
        [
            ((b / b.sum(1, keepdim=True))[:, :, None] * tri).sum(1),
            tri[:, 0] + t * (tri[:, 1] - tri[:, 0]),
            tri.reshape(-1, 3),
            torch.randn(64, 3, dtype=D, generator=g),
            1e3 * torch.randn(64, 3, dtype=D, generator=g),
        ]
    )
    cells = torch.arange(192).reshape(-1, 3)
    yield (
        "ordinary float32",
        q.float().double(),
        tri.reshape(-1, 3).float().double(),
        cells,
    )
    yield "ordinary float64", q, tri.reshape(-1, 3), cells
    p = torch.randn(12, 3, dtype=D, generator=g)
    p[3] = p[2]
    p[5] = 0.25 * p[4] + 0.75 * p[6]
    p[8] = 0.5 * (p[7] + p[9])
    c = torch.tensor(
        [[0, 1, 2], [2, 3, 1], [4, 5, 6], [7, 8, 9], [9, 10, 11], [1, 1, 1], [0, 4, 8]]
    )
    yield "zero area", torch.cat([p, torch.randn(60, 3, dtype=D, generator=g)]), p, c


def _guard_pairs(q, p, c, device):
    """Every query against every canonical cell: queries, cells, zero-area flags."""
    cells, _ = geometry._canonical_cells(p, geometry._weld(p)[c])
    s = p[cells]
    flat, any_flat = geometry._zero_area(s, cells)
    qi = torch.arange(q.shape[0]).repeat_interleave(cells.shape[0])
    ki = torch.arange(cells.shape[0]).repeat(q.shape[0])
    return q[qi].to(device), s[ki].to(device), flat[ki].to(device) if any_flat else None


def _pairs_of(case, device="cuda"):
    """:func:`_guard_pairs` of the set ``case`` of :func:`_guard_meshes`."""
    return _guard_pairs(*next(m for m in _guard_meshes() if m[0] == case)[1:], device)


@_TRITON_CUDA
@pytest.mark.parametrize("switch", [1, float("inf")])
@pytest.mark.parametrize("case", [c[0] for c in _guard_meshes()] + ["empty"])
def test_guard_kernel_equals_the_torch_weights_bitwise(case, switch, monkeypatch):
    """The fused weights (Ericson's, the guard and the accurate point, the zero-area
    cells' segments) are the torch code's bitwise, with the switch as set and with it
    never taken; on the caps and needles the switch fires. A column-major ``q`` takes
    the torch code."""
    if case == "empty":
        q, s, flat = torch.zeros(0, 3, dtype=D), torch.zeros(0, 3, 3, dtype=D), None
        q, s = q.cuda(), s.cuda()
    else:
        q, s, flat = _pairs_of(case)
    monkeypatch.setattr(geometry, "_SWITCH_UNITS", switch)
    ref = geometry._closest_weights(q, s, flat)
    fast = geometry._closest_weights(q, s, flat, fast=True)
    assert torch.equal(ref.view(torch.int64), fast.view(torch.int64))
    q = q.T.contiguous().T
    ref = geometry._closest_weights(q, s, flat)
    fast = geometry._closest_weights(q, s, flat, fast=True)
    assert torch.equal(ref.view(torch.int64), fast.view(torch.int64))
    if switch == 1 and case.startswith(("cap", "needle")):
        monkeypatch.setattr(geometry, "_SWITCH_UNITS", float("inf"))
        assert bool((geometry._closest_weights(q, s, flat) != ref).any())


@_TRITON_CUDA
@pytest.mark.parametrize("n", [1, 7, 255])
def test_guard_kernel_on_a_partial_block(n, monkeypatch):
    """A single partial block of pairs, the switched ones first, gives the torch
    weights bitwise."""
    q, s, flat = _pairs_of("cap 0.001 float64")
    ref = geometry._closest_weights(q, s, flat)
    monkeypatch.setattr(geometry, "_SWITCH_UNITS", float("inf"))
    switched = (geometry._closest_weights(q, s, flat) != ref).any(-1)
    monkeypatch.undo()
    rows = torch.cat([switched.nonzero()[:, 0], (~switched).nonzero()[:, 0]])[:n]
    q, s, flat = q[rows], s[rows], None if flat is None else flat[rows]
    ref = geometry._closest_weights(q, s, flat)
    fast = geometry._closest_weights(q, s, flat, fast=True)
    assert bool(switched[rows[0]])
    assert torch.equal(ref.view(torch.int64), fast.view(torch.int64))


@_TRITON_CUDA
def test_guard_kernel_without_guarded_pairs(monkeypatch):
    """Pairs of which none is guarded (far queries of a well-shaped triangle) give the
    torch weights bitwise."""
    g = torch.Generator().manual_seed(3)
    q = (1e2 * torch.randn(300, 3, dtype=D, generator=g)).cuda()
    s = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=D).expand(300, 3, 3)
    s = s.contiguous().cuda()
    ref = geometry._closest_weights(q, s)

    def unguarded(*args):
        raise AssertionError("a guarded pair")

    monkeypatch.setattr(geometry, "_accurate_weights", unguarded)
    fast = geometry._closest_weights(q, s, fast=True)
    assert torch.equal(ref.view(torch.int64), fast.view(torch.int64))


@_TRITON_CUDA
@pytest.mark.parametrize("case", ["ordinary float64", "zero area"])
def test_guard_kernel_with_nan_coordinates(case):
    """NaN coordinates in queries and cells give the torch weights bitwise, NaN
    payloads included."""
    q, s, flat = _pairs_of(case)
    q[::97, 1] = float("nan")
    s[::89, 2, 0] = float("nan")
    ref = geometry._closest_weights(q, s, flat)
    fast = geometry._closest_weights(q, s, flat, fast=True)
    assert bool(ref.isnan().any())
    assert torch.equal(ref.view(torch.int64), fast.view(torch.int64))


@_TRITON_CUDA
@pytest.mark.parametrize("columns", [False, True], ids=["rows", "columns"])
@pytest.mark.parametrize("case", [c[0] for c in _guard_meshes()])
def test_fast_search_equals_the_reference_on_thin_cells(case, columns):
    """The fast op's outputs are the reference op's, bitwise, on the guard's sets,
    with the queries row-major and column-major."""
    _, q, p, c = next(m for m in _guard_meshes() if m[0] == case)
    q, p = q.cuda(), p.cuda()
    if columns:
        q = q.T.contiguous().T
    cells, parity = geometry._canonical_cells(p, geometry._weld(p)[c.cuda()])
    s = p[cells]
    n64 = parity[:, None].double() * compute_cell_normals(s[:, 1:] - s[:, :1])
    ref = geometry._closest_candidates(q, p, cells, parity, n64)
    fast = geometry._closest_candidates_fast(q, p, cells, parity, n64)
    for a, b in zip(ref, fast):
        assert a.dtype == b.dtype
        assert (
            torch.equal(a.view(torch.uint8), b.view(torch.uint8))
            if a.is_floating_point()
            else torch.equal(a, b)
        )


def test_guard_kernel_falls_back_to_the_torch_code_off_cuda(monkeypatch):
    """Off CUDA the fast weights are the torch code's, from the torch code itself."""
    q, s, flat = _pairs_of("zero area", "cpu")
    calls = []
    closest = geometry._closest_weights

    def spy(*args):
        calls.append(args)
        return closest(*args)

    monkeypatch.setattr(geometry, "_closest_weights", spy)
    fast = closest(q, s, flat, fast=True)
    assert len(calls) == 1
    assert torch.equal(fast.view(torch.int64), closest(q, s, flat).view(torch.int64))
