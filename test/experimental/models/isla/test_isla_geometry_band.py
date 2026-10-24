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

"""The error bound of the exact boundary geometry and the frame its arithmetic runs in.

The bound's arithmetic share is a worst case of the float64 pair distance in the
boundary's centred frame, and its storage share the rounding of the coordinates as
given: far from the origin the distance keeps the queries' relative precision, a
posed planar star is not a tie, the derived share holds against exact rational
distances, the outputs do not depend on the numbering, and the fast search gives
the reference's outputs bitwise.
"""

from fractions import Fraction

import pytest
import torch

from physicsnemo.experimental.models.isla import geometry
from physicsnemo.experimental.models.isla.geometry import exact_boundary_geometry
from physicsnemo.mesh.geometry._cell_normals import compute_cell_normals
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from test.experimental.models.isla._meshes import orthogonal
from test.experimental.models.isla.test_isla_geometry_thin import _exact_d2

D = torch.float64
U32, U64 = 2.0**-24, 2.0**-53

_DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
    ),
]


def _search(q, p, c):
    """Every output of the reference search op."""
    cells, parity = geometry._canonical_cells(p, geometry._weld(p)[c])
    s = p[cells].double()
    n64 = parity[:, None].double() * compute_cell_normals(s[:, 1:] - s[:, :1])
    return geometry._closest_candidates(q, p, cells, parity, n64)


def _far():
    """An icosphere of radius 1 translated by about 1e3 of its size, and float32
    queries 1e-3 to 2e-3 off its cells' centroids along their normals, either side
    (outside the storage share of the bound, ``u32 (|x| + R) ~ 1.2e-4``)."""
    m = sphere_icosahedral.load(radius=1.0, subdivisions=1)
    p, c = m.points.double(), m.cells
    tri = p[c]
    n = torch.nn.functional.normalize(
        torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1), dim=-1
    )
    g = torch.Generator().manual_seed(0)
    s = (1 + torch.rand(c.shape[0], 1, dtype=D, generator=g)) * 1e-3
    s = s * torch.where(torch.rand(c.shape[0], 1, generator=g) < 0.5, -1.0, 1.0)
    t = torch.tensor([1000.0, -700.0, 400.0], dtype=D)
    return (tri.mean(1) + s * n + t).float(), (p + t).float(), c, t.float()


@pytest.mark.parametrize("device", _DEVICES)
def test_far_from_the_origin_the_distance_keeps_its_relative_precision(device):
    """A float32 surface 1e3 of its size from the origin, queried 1e-3 off it: the
    distance is within 4 float32 units of itself of the exact one of the stored
    coordinates (float64, after an exact shift to the origin), as it is at the
    origin; a closest point formed in float32 loses it to the rounding of the
    coordinates' magnitude, 3e-5."""
    q, p, c, t = _far()
    d, _ = exact_boundary_geometry(q.to(device), p.to(device), c.to(device))
    t = t.double()
    exact, _ = exact_boundary_geometry(q.double() - t, p.double() - t, c)
    d = d.cpu().double()
    assert bool((d != 0).all()), "queries off the surface taken on it"
    err = ((d - exact).abs() / exact.abs()).max()
    assert err <= 4 * U32, f"relative distance error {float(err):.3g} > 4 u32"


def _star(posed: bool):
    """A planar star, seven triangles around a centre in the plane z = 0, as float32,
    rotated and translated when ``posed``, its plane's unit normal, and queries at
    its centre and at its spokes' midpoints (on the surface, every cell there
    coplanar)."""
    k = 7
    a = torch.arange(k, dtype=D) * 2 * torch.pi / k
    p = torch.cat(
        [torch.zeros(1, 3, dtype=D), torch.stack([a.cos(), a.sin(), 0 * a], -1)]
    )
    c = torch.stack(
        [
            torch.zeros(k, dtype=torch.long),
            torch.arange(1, k + 1),
            torch.arange(1, k + 1) % k + 1,
        ],
        -1,
    )
    q = torch.cat([p[:1], 0.5 * p[1:]])
    n = torch.tensor([0.0, 0.0, 1.0], dtype=D)
    if posed:
        R, t = orthogonal(3, 5), torch.tensor([0.7, -0.4, 1.3], dtype=D)
        p, q, n = p @ R.T + t, q @ R.T + t, n @ R.T
    return q.float(), p.float(), c, n


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("posed", [False, True])
def test_a_posed_planar_star_is_not_a_tie(posed, device):
    """At the centre and on the spokes of a planar star stored in float32 the
    query is on the surface and untied in either pose, u the plane's normal: the
    cells' float64 normals differ there only by the rounding of the stored
    vertices, which the certified test does not count as distinct directions."""
    q, p, c, n = _star(posed)
    q, p, c = q.to(device), p.to(device), c.to(device)
    _, tied, _, _, on_wall, _ = _search(q, p, c)
    assert bool(on_wall.all())
    assert not bool(tied.any()), f"{int(tied.sum())} of {q.shape[0]} queries tied"
    d, u = exact_boundary_geometry(q, p, c)
    assert bool((d == 0).all())
    assert float((u.cpu().double() - n).norm(dim=-1).max()) <= 4 * U32


def _thin(g, n, rho):
    """``n`` triangles of shape ``rho`` (a cap when the apex is near the long edge's
    middle, a needle near an end), posed 100 of their size from the origin, and a
    query per triangle on it, just off it, or beside its long edge."""
    a = torch.rand(n, 3, dtype=D, generator=g)
    b = a + torch.nn.functional.normalize(
        torch.randn(n, 3, dtype=D, generator=g), dim=-1
    )
    side = torch.nn.functional.normalize(
        torch.linalg.cross(b - a, torch.randn(n, 3, dtype=D, generator=g), dim=-1),
        dim=-1,
    )
    s = torch.rand(n, 1, dtype=D, generator=g)
    apex = a + s * (b - a) + rho * side
    tri = torch.stack([a, b, apex], 1)
    w = torch.rand(n, 3, dtype=D, generator=g)
    on = ((w / w.sum(1, keepdim=True))[..., None] * tri).sum(1)
    nrm = torch.nn.functional.normalize(
        torch.linalg.cross(b - a, apex - a, dim=-1), dim=-1
    )
    h = torch.rand(n, 1, dtype=D, generator=g) * 2 - 1
    kind = torch.arange(n)[:, None] % 3
    q = torch.where(
        kind == 0,
        on,
        torch.where(kind == 1, on + 1e-3 * h * nrm, a + s * (b - a) - 1e-2 * h * side),
    )
    t = torch.tensor([60.0, -80.0, 20.0], dtype=D)
    return q + t, tri + t


def _exact_segment_d2(x, a, b):
    """The squared distance from ``x`` to the segment ``[a, b]``, exactly."""
    q, a, b = ([Fraction(v) for v in p] for p in (x, a, b))
    e = [b[i] - a[i] for i in range(2)]
    y = [q[i] - a[i] for i in range(2)]
    t = (y[0] * e[0] + y[1] * e[1]) / (e[0] * e[0] + e[1] * e[1])
    t = min(max(t, Fraction(0)), Fraction(1))
    r = [y[i] - t * e[i] for i in range(2)]
    return r[0] * r[0] + r[1] * r[1]


@pytest.mark.parametrize("dim", [2, 3])
def test_the_arithmetic_share_bounds_the_pair_distance(dim):
    """Float64 cells and queries posed far from the origin, centred as the search
    centres them: the pair distance from the weights of ``_closest_weights`` is
    within ``C u64 (|x'| + R'_k)`` (``_BAND_UNITS``) of the exact distance from the
    query to the cell as given (exact rational arithmetic), on thin caps and
    needles in three dimensions, many of them guarded or switched, and on segments
    in two."""
    g = torch.Generator().manual_seed(dim)
    if dim == 3:
        q, tri = map(
            torch.cat, zip(*(_thin(g, 60, rho) for rho in (1e-2, 1e-6, 1e-10)))
        )
    else:
        a = torch.rand(200, 2, dtype=D, generator=g) + torch.tensor(
            [50.0, -30.0], dtype=D
        )
        tri = torch.stack([a, a + torch.randn(200, 2, dtype=D, generator=g)], 1)
        q = tri.mean(1) + torch.randn(
            200, 2, dtype=D, generator=g
        ) * 10.0 ** -torch.randint(0, 8, (200, 1), generator=g)
    origin = geometry._centroid(tri)
    qc, tc = q - origin, tri - origin
    w = geometry._closest_weights(qc, tc)
    dist = (qc - (w[..., None] * tc).sum(1)).norm(dim=-1)
    if dim == 3:
        exact = [
            float(_exact_d2(x.tolist(), t.tolist())) ** 0.5 for x, t in zip(q, tri)
        ]
    else:
        exact = [
            float(_exact_segment_d2(x.tolist(), *t.tolist())) ** 0.5
            for x, t in zip(q, tri)
        ]
    unit = U64 * (qc.norm(dim=-1) + tc.norm(dim=-1).amax(-1))
    ratio = ((dist - torch.tensor(exact, dtype=D)).abs() / unit).max()
    assert ratio <= geometry._BAND_UNITS[dim], f"{float(ratio):.3g} units"
    assert geometry._BAND_UNITS[3] >= geometry._SWITCH_UNITS + geometry._FOOT_UNITS


def _numbered():
    """A posed float32 icosphere, queries around it, on its vertices and its
    edges' midpoints, and the same surface with its vertices and cells permuted and
    each cell's vertex list rotated."""
    m = sphere_icosahedral.load(radius=1.0, subdivisions=2)
    R, t = orthogonal(3, 3), torch.tensor([30.0, 10.0, -20.0], dtype=D)
    p, c = m.points.double() @ R.T + t, m.cells
    g = torch.Generator().manual_seed(4)
    q = torch.cat(
        [
            torch.randn(300, 3, dtype=D, generator=g) * 0.8 @ R.T + t,
            p[:40],
            p[c[:40, :2]].mean(1),
        ]
    ).float()
    perm = torch.randperm(p.shape[0], generator=g)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(p.shape[0])
    c2 = inv[c][torch.randperm(c.shape[0], generator=g)].roll(1, -1)
    return q, p.float(), c, p[perm].float(), c2


@pytest.mark.parametrize("device", _DEVICES)
def test_the_outputs_do_not_depend_on_the_numbering(device):
    """Permuting the vertices and the cells and rotating each cell's vertex list
    leaves d and u bitwise (the centred frame's origin is a function of the set of
    cells)."""
    q, p, c, p2, c2 = (x.to(device) for x in _numbered())
    d, u = exact_boundary_geometry(q, p, c)
    d2, u2 = exact_boundary_geometry(q, p2, c2)
    assert torch.equal(d, d2) and torch.equal(u, u2)


def _band_cases():
    q, p, c, _ = _far()
    yield "far", q, p, c
    for posed in (False, True):
        q, p, c, _ = _star(posed)
        yield f"star {posed}", q, p, c
    g = torch.Generator().manual_seed(7)
    q, tri = map(torch.cat, zip(*(_thin(g, 30, rho) for rho in (1e-3, 1e-9))))
    yield "thin", q, tri.reshape(-1, 3), torch.arange(tri.shape[0] * 3).reshape(-1, 3)
    q, p, c, p2, c2 = _numbered()
    yield "numbered", q, p, c
    yield "renumbered", q, p2, c2


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("case", [c[0] for c in _band_cases()])
def test_the_fast_search_equals_the_reference_bitwise(case, device):
    """On the cases above, d, u and their gradients in the queries and the
    vertices are the reference's bitwise under the fast search."""
    _, q, p, c = next(x for x in _band_cases() if x[0] == case)
    out = []
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for kernel in ("reference", "fast"):
            qq = q.to(device).clone().requires_grad_()
            pp = p.to(device).clone().requires_grad_()
            d, u = exact_boundary_geometry(qq, pp, c.to(device), kernel=kernel)
            probe = torch.linspace(0, 1, u.numel(), dtype=u.dtype, device=u.device)
            gq, gp = torch.autograd.grad(
                d.sum() + (u * probe.reshape(u.shape)).sum(), (qq, pp)
            )
            out.append((d, u, gq, gp))
    finally:
        torch.set_num_threads(threads)
    for name, a, b in zip(("d", "u", "dq", "dp"), *out):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), name
