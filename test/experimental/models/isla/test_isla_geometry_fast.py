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

from physicsnemo.experimental.models.isla import geometry
from physicsnemo.experimental.models.isla.geometry import exact_boundary_geometry
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.surfaces import cube_surface
from test.experimental.models.isla._meshes import orthogonal, outputs, transform
from test.experimental.models.isla.test_isla_interior_geometry import (
    _box_ties,
    _box_wall,
    _finned_cube,
    _interior,
    _near_wall,
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
    a soup, with on-wall, tied and far queries."""
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

    monkeypatch.setattr(geometry, "_winding_sign", spy_sign)
    monkeypatch.setattr(geometry, "_tree_winding_numbers", spy_tree_w)
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
        assert auto.resolve_geometry_kernel(q.cuda()) == "fast"
