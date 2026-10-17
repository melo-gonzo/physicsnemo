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

"""The interior geometry of the prediction-point tokens.

For every prediction point the model derives its signed distance and direction
from the boundary surface itself (the exact closest point on the surface's cells,
signed by the closest feature's pseudo-normal, positive on the side the normals
point to); neither is an input, and neither depends on how the boundary tokens
were sampled from the surface. These tests pin the geometry against brute-force
and exact references, its sign convention and convergence, and the contracts of
the interior task.
"""

import os

import pytest
import torch

from physicsnemo.experimental.models import ISLA
from physicsnemo.experimental.models.isla import geometry
from physicsnemo.experimental.models.isla.geometry import exact_boundary_geometry
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import EFFECTIVE_MEASURE_KEY
from physicsnemo.mesh.primitives.curves import circle_2d
from physicsnemo.mesh.primitives.surfaces import cube_surface, sphere_icosahedral, torus
from physicsnemo.nn.functional.geometry.sdf import _closest_point_weights
from test.experimental.models.isla._meshes import (
    cloud,
    orthogonal,
    outputs,
    points_mesh,
    soup,
    transform,
)

D = torch.float64
PV = {"p": {"rank": 0}, "v": {"rank": 1}}


def _model(**kw):
    torch.manual_seed(0)
    return ISLA(output_schema=PV, global_schema={"g": {"rank": 1}}, hidden=32, n_layers=2, n_slices=8, interior=True, **kw).double().eval()


def test_interior_contracts():
    """Rotation and reflection covariance, translation invariance and geometric-scale
    invariance in the interior task, with the prediction-point directions computed
    inside; the distance channel is live."""
    pts, nrm, w, _ = cloud(200)
    mesh = soup(pts, nrm, w)
    drv = torch.tensor([0.3, 1.0, 0.2], dtype=D)
    q, _, _, _ = cloud(40, seed=1, scale=(4.0, 4.0, 4.0))
    m = _model()

    def call(R=None, t=None, k=1.0, q=q):
        R = torch.eye(3, dtype=D) if R is None else R
        t = torch.zeros(3, dtype=D) if t is None else t
        with torch.no_grad():
            return outputs(
                m(boundary_meshes={"wall": transform(mesh, R, t, scale=k)}, prediction_points=k * q @ R.T + t, global_data={"g": drv @ R.T})
            )

    base = call()
    for improper in (False, True):
        R = orthogonal(3, 4, improper)
        out = call(R, torch.tensor([2.0, -5.0, 1.0], dtype=D), k=2.5)
        assert torch.allclose(out["p"], base["p"], atol=1e-11), improper
        assert torch.allclose(out["v"], base["v"] @ R.T, atol=1e-11), improper
    ### the distance channel is live: moving the queries off the boundary changes p
    far = call(q=q * 3.0)
    assert not torch.allclose(far["p"], base["p"], atol=1e-6)


def test_stored_distance_is_an_ordinary_query_input():
    """A dataset's stored signed distance (and its gradient) enter as query_schema
    fields of rank 0 and 1 from the prediction Mesh's point_data; the built-in
    geometry is computed regardless, so the two channels coexist and both are live."""
    pts, nrm, w, _ = cloud(120)
    mesh = soup(pts, nrm, w)
    drv = torch.tensor([0.3, 1.0, 0.2], dtype=D)
    q, _, _, _ = cloud(25, seed=1, scale=(4.0, 4.0, 4.0))
    d, grad = exact_boundary_geometry(q, mesh.points, mesh.cells)
    m = _model(query_schema={"geometry.sdf": {"rank": 0}, "geometry.grad": {"rank": 1}})
    assert m.qt.embed[0].in_features == 2 + (2 + (2 + 1))  # built-in pair, scaled scalar pair, vector cosines and norm

    def call(sdf, g):
        with torch.no_grad():
            return outputs(
                m(
                    boundary_meshes={"wall": mesh},
                    prediction_points=points_mesh(q, {"geometry": {"sdf": sdf, "grad": g}}),
                    global_data={"g": drv},
                )
            )

    base = call(d, grad)
    assert base["p"].shape == (25,)
    assert not torch.allclose(call(2 * d, grad)["p"], base["p"], atol=1e-6)
    assert not torch.allclose(call(d, -grad)["p"], base["p"], atol=1e-6)


def test_no_query_normals_argument():
    """The prediction-point direction is never an input."""
    pts, nrm, w, _ = cloud(20)
    m = _model()
    with pytest.raises(TypeError, match="query_normals"):
        m(boundary_meshes={"wall": soup(pts, nrm, w)}, prediction_points=pts[:3], global_data={"g": nrm[0]}, query_normals=nrm[:3])
    with pytest.raises(TypeError, match="interior_geometry"):
        _model(interior_geometry="provided")


def test_fused_kernel_stays_available_with_the_built_in_distance():
    """The built-in distance enters the embedding, not the geometry region, so a model
    without rank-1 or rank-2 per-token inputs keeps the fused kernel."""
    m = ISLA(output_schema=PV, global_schema={"g": {"rank": 1}}, hidden=32, n_layers=1, n_slices=8, interior=True)
    assert m.n_query_builtin == 2 and m.qt.embed[0].in_features == 2
    assert not (m.n_block_vecs or m.n_block_tens)


def test_checkpoint_round_trip_keeps_the_task(tmp_path):
    from physicsnemo.core.module import Module

    m = _model()
    m.save(tmp_path / "m.mdlus")
    m2 = Module.from_checkpoint(str(tmp_path / "m.mdlus")).double().eval()
    assert m2.interior is True
    pts, nrm, w, _ = cloud(50)
    q, _, _, _ = cloud(8, seed=1)
    kw = dict(boundary_meshes={"wall": soup(pts, nrm, w)}, prediction_points=q, global_data={"g": torch.ones(3, dtype=D)})
    with torch.no_grad():
        a, b = outputs(m(**kw)), outputs(m2(**kw))
    assert all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the saved case runs on CUDA")
@pytest.mark.skipif("ISLA_AF_POSED_SAMPLE" not in os.environ, reason="set ISLA_AF_POSED_SAMPLE to a forward-kwargs .pt saved by the posed-inference reproduction")
def test_saved_posed_airfrans_case_runs():
    """The AirfRANS posed validation case that crashed the 2026-10-03 inference, as
    saved forward kwargs (boundary polyline, 100,000 interior points with the stored
    sdf / sdf_normals, inlet vectors), runs through a model of its schema."""
    saved = torch.load(os.environ["ISLA_AF_POSED_SAMPLE"], weights_only=False)
    kw = saved["forward_kwargs"]
    kw = {
        "boundary_meshes": {k: v.to("cuda") for k, v in kw["boundary_meshes"].items()},
        "prediction_points": kw["prediction_points"].to("cuda"),
        "global_data": kw["global_data"].to("cuda"),
    }
    torch.manual_seed(0)
    m = ISLA(
        output_schema={"velocity": {"rank": 1}, "pressure": {"rank": 0}, "nut": {"rank": 0}},
        global_schema={"U_inf_dir": {"rank": 1}, "U_inf_mag": {"rank": 0}},
        query_schema={"sdf": {"rank": 0}, "sdf_normals": {"rank": 1}},
        dim=2, interior=True, hidden=32, n_layers=2, n_slices=8,
    ).cuda().eval()
    with torch.no_grad():
        out = outputs(m(**kw))
    assert all(torch.isfinite(v).all() for v in out.values())


# ---------------------------------------------------------------------------
# The exact geometry against the boundary surface (exact_boundary_geometry)
# ---------------------------------------------------------------------------


def _oriented(mesh: Mesh, outward) -> Mesh:
    """``mesh`` with every cell re-oriented so its normal has a non-negative dot with
    ``outward(centroid)``."""
    cells = mesh.cells.clone()
    flip = (mesh.cell_normals * outward(mesh.cell_centroids)).sum(-1) < 0
    cells[flip, -2:] = cells[flip, -2:].flip(-1)
    return Mesh(points=mesh.points, cells=cells)


def _polygon(verts) -> Mesh:
    """A closed polyline through ``verts`` (n, 2), oriented with outward normals."""
    n = verts.shape[0]
    cells = torch.stack([torch.arange(n), torch.arange(1, n + 1) % n], -1)
    mesh = Mesh(points=verts, cells=cells)
    ### the shoelace area is positive for a counterclockwise polygon, whose rotated
    ### edges (the cell normals) point inward
    x, y = verts[:, 0], verts[:, 1]
    if (x * y.roll(-1) - x.roll(-1) * y).sum() > 0:
        mesh = Mesh(points=verts, cells=cells.flip(-1))
    return mesh


def _polygon_sdf(q, verts):
    """The exact signed distance to a closed polygon (positive outside): the brute
    force over its segments in float64, signed by the crossing-number test."""
    a, b = verts, verts.roll(-1, 0)
    ab = b - a
    t = (((q[:, None] - a) * ab).sum(-1) / (ab * ab).sum(-1)).clamp(0, 1)
    dist = (q[:, None] - a - t[..., None] * ab).norm(dim=-1).min(-1).values
    ya, yb = a[:, 1], b[:, 1]
    crosses = (ya > q[:, None, 1]) != (yb > q[:, None, 1])
    x_at = a[:, 0] + (q[:, None, 1] - ya) / (yb - ya) * ab[:, 0]
    inside = (crosses & (q[:, None, 0] < x_at)).sum(-1) % 2 == 1
    return torch.where(inside, -dist, dist)


def _naca(n: int, thickness: float = 0.12):
    """A NACA 00xx section of unit chord with a sharp trailing edge, 2n - 2 cosine-
    clustered vertices (the open-trailing-edge coefficient replaced by -0.1036)."""
    beta = torch.linspace(0, torch.pi, n, dtype=D)
    x = 0.5 * (1 - torch.cos(beta))
    yt = 5 * thickness * (0.2969 * x.sqrt() - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    upper = torch.stack([x, yt], -1)
    lower = torch.stack([x, -yt], -1).flip(0)[1:-1]
    return torch.cat([upper, lower])


def _brute_force_surface(q, points, cells):
    """The unsigned distance to the closest point of any cell, float64, over every
    (query, cell) pair: segments in two dimensions, triangles in three."""
    q, simplex = q.double(), points.double()[cells]
    if q.shape[-1] == 2:
        a, b = simplex[None, :, 0], simplex[None, :, 1]
        ab = b - a
        t = (((q[:, None] - a) * ab).sum(-1) / (ab * ab).sum(-1)).clamp(0, 1)
        return (q[:, None] - a - t[..., None] * ab).norm(dim=-1).min(-1)
    nq, nf = q.shape[0], simplex.shape[0]
    qq = q[:, None].expand(nq, nf, 3).reshape(-1, 3)
    tri = simplex[None].expand(nq, nf, 3, 3).reshape(-1, 3, 3)
    p = (_closest_point_weights(qq, tri)[..., None] * tri).sum(1)
    return (qq - p).norm(dim=-1).reshape(nq, nf).min(-1)


def _sphere(subdivisions: int, radius: float = 1.0) -> Mesh:
    m = sphere_icosahedral.load(radius=radius, subdivisions=subdivisions)
    return _oriented(Mesh(points=m.points.double(), cells=m.cells), lambda c: c)


def _circle(n: int) -> Mesh:
    m = circle_2d.load(n_points=n)
    return _oriented(Mesh(points=m.points.double(), cells=m.cells), lambda c: c)


def _torus(n_major: int = 48, n_minor: int = 24) -> Mesh:
    m = torus.load(n_major=n_major, n_minor=n_minor)

    def outward(c):
        ring = torch.nn.functional.normalize(c * torch.tensor([1.0, 1.0, 0.0], dtype=c.dtype), dim=-1)
        return c - ring

    return _oriented(Mesh(points=m.points.double(), cells=m.cells), outward)


def _near_wall(dim: int, n: int, seed: int, band: float, radius: float = 1.0):
    g = torch.Generator().manual_seed(seed)
    dirs = torch.nn.functional.normalize(torch.randn(n, dim, dtype=D, generator=g), dim=-1)
    return dirs * (radius + band * (2 * torch.rand(n, 1, dtype=D, generator=g) - 1))


def _geometry(q, mesh):
    return exact_boundary_geometry(q, mesh.points, mesh.cells)


@pytest.mark.parametrize("dim", [2, 3])
def test_exact_geometry_matches_the_brute_force(dim):
    """|d| is the distance to the closest point of any cell (float64 brute force over
    every pair): exact in two dimensions, to float32 resolution in three, where the
    BVH searches in float32 (the closest cell differs only at a float32 tie); u is
    a unit vector and q - |d| u lies on the closest cell."""
    g = torch.Generator().manual_seed(dim)
    pts, nrm, _, _ = cloud(150, dim, seed=dim)
    mesh = soup(pts, nrm)
    q = torch.randn(400, dim, dtype=D, generator=g) * 3.0
    d, u = _geometry(q, mesh)
    ref = _brute_force_surface(q, mesh.points, mesh.cells).values
    assert d.shape == (400,) and u.shape == (400, dim) and d.dtype == D
    tol = 0.0 if dim == 2 else 8 * torch.finfo(torch.float32).eps * 3.0
    assert (d.abs() - ref).abs().max() <= tol + 1e-14
    assert torch.allclose(u.norm(dim=-1), torch.ones(400, dtype=D), atol=1e-12)
    foot = q - d.abs()[:, None] * u
    assert _brute_force_surface(foot, mesh.points, mesh.cells).values.max() <= tol + 1e-12


@pytest.mark.parametrize("dim", [2, 3])
def test_sign_and_direction_on_a_closed_surface(dim):
    """A circle / sphere with outward normals: d > 0 outside and d < 0 inside beyond
    the mesh's own deviation from the curved surface, u radial (outward outside,
    inward inside)."""
    mesh = _circle(256) if dim == 2 else _sphere(4)
    q = _near_wall(dim, 3000, 0, band=0.2)
    d, u = _geometry(q, mesh)
    r = q.norm(dim=-1)
    off = (r - 1).abs() > 2e-3  # beyond the inscribed polygon's sagitta
    assert (torch.sign(d[off]) == torch.sign(r - 1)[off]).all()
    radial = (u * q / r[:, None]).sum(-1)
    assert (radial[off] * torch.sign(r - 1)[off] > 0.99).all()


def test_convergence_is_second_order_under_refinement():
    """The circle at 64, 128, 256 segments and the sphere at icosahedral
    subdivisions 2, 3, 4 (h halving each step): the largest |d - d_exact| over a
    near-wall band falls by at least 3.5x per halving, and the largest deviation of
    the direction (on the side the mesh puts the query) from the radial one by at
    least 1.7x (O(h), the cells' normals against the surface's)."""
    for dim, meshes in ((2, [_circle(n) for n in (64, 128, 256)]), (3, [_sphere(s) for s in (2, 3, 4)])):
        q = _near_wall(dim, 4000, 1, band=0.05)
        r = q.norm(dim=-1)
        errs, dir_errs = [], []
        for mesh in meshes:
            d, u = _geometry(q, mesh)
            errs.append((d - (r - 1)).abs().max())
            dir_errs.append((torch.sign(d)[:, None] * u - q / r[:, None]).norm(dim=-1).max())
        for coarse, fine in zip(errs, errs[1:]):
            assert coarse / fine >= 3.5, (dim, errs)
        for coarse, fine in zip(dir_errs, dir_errs[1:]):
            assert coarse / fine >= 1.7, (dim, dir_errs)


@pytest.mark.parametrize("dim", [2, 3])
def test_exact_geometry_is_covariant_and_scales(dim):
    """Under a rotation, a reflection (cells re-oriented so the normals follow it), a
    translation and a rescale: d is invariant (scales with the geometry) and u
    covariant, to float rounding."""
    mesh = _polygon(_naca(80)) if dim == 2 else _torus()
    g = torch.Generator().manual_seed(2)
    q = torch.randn(600, dim, dtype=D, generator=g) * 0.6 + (0.5 if dim == 2 else 0.0)
    d, u = _geometry(q, mesh)
    for improper in (False, True):
        R = orthogonal(dim, 3, improper)
        t = torch.arange(1, dim + 1, dtype=D)
        moved = transform(mesh, R, t, scale=2.5)
        d2, u2 = _geometry(2.5 * q @ R.T + t, moved)
        tol = 1e-12 if dim == 2 else 1e-5  # the float32 search's ties in three dimensions
        assert torch.allclose(d2, 2.5 * d, atol=tol), improper
        assert torch.allclose(u2, u @ R.T, atol=tol), improper


@pytest.mark.parametrize("dim", [2, 3])
def test_exact_geometry_is_invariant_to_splitting_the_surface(dim):
    """The same surface split into two named pieces (each with its own copy of the
    seam's vertices), and with every cell split (segments in two, triangles in four
    by their midpoints), gives the same d and u to rounding: the geometry is a
    function of the surface, not of its cells or of how it is stored."""
    mesh = _polygon(_naca(60)) if dim == 2 else _torus(24, 12)
    g = torch.Generator().manual_seed(4)
    q = torch.randn(800, dim, dtype=D, generator=g) * 0.5 + (0.5 if dim == 2 else 0.0)
    d, u = _geometry(q, mesh)
    ### two pieces, re-indexed separately, concatenated as the model concatenates them
    half = mesh.n_cells // 2
    pieces = [mesh.slice_cells(torch.arange(half)), mesh.slice_cells(torch.arange(half, mesh.n_cells))]
    pieces = [p.slice_points(torch.unique(p.cells)) for p in pieces]
    pts = torch.cat([p.points for p in pieces])
    cells = torch.cat([pieces[0].cells, pieces[1].cells + pieces[0].n_points])
    d2, u2 = exact_boundary_geometry(q, pts, cells)
    assert torch.allclose(d2, d, atol=1e-12, rtol=0) and torch.allclose(u2, u, atol=1e-12, rtol=0)
    ### every cell split
    d3, u3 = _geometry(q, mesh.subdivide(levels=1, filter="linear"))
    tol = 1e-12 if dim == 2 else 1e-5
    assert torch.allclose(d3, d, atol=tol, rtol=0) and torch.allclose(u3, u, atol=tol, rtol=0)


def test_sharp_trailing_edge_thin_plate_and_concave_corner():
    """Polygons where a single cell's normal gives the wrong sign: the wake behind a
    sharp trailing edge (NACA 0012), a thin plate (a 1e-3-thick rectangle) and the
    concave corner of an L; d equals the exact polygon SDF everywhere."""
    plate = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1e-3], [0.0, 1e-3]], dtype=D)
    ell = torch.tensor([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]], dtype=D)
    g = torch.Generator().manual_seed(5)
    for verts, centre in ((_naca(100), (1.0, 0.0)), (plate, (1.0, 5e-4)), (ell, (1.0, 1.0))):
        mesh = _polygon(verts)
        c = torch.tensor(centre, dtype=D)
        q = torch.cat([c + 0.05 * torch.randn(2000, 2, dtype=D, generator=g), c + torch.randn(2000, 2, dtype=D, generator=g)])
        d, _ = _geometry(q, mesh)
        assert torch.allclose(d, _polygon_sdf(q, verts), atol=1e-12, rtol=0)


def test_cube_edges_and_corners_and_torus_saddles():
    """The exact box SDF at a cube's edges and corners (the vertex and edge pseudo-
    normals), and the sign around a torus, whose inner vertices are saddles."""
    cube = cube_surface.load()
    g = torch.Generator().manual_seed(6)
    q = torch.randn(5000, 3, dtype=D, generator=g) * 0.8
    d, _ = _geometry(q, Mesh(points=cube.points.double(), cells=cube.cells))
    qq = q.abs() - 0.5
    exact = qq.clamp_min(0).norm(dim=-1) + qq.amax(-1).clamp_max(0)
    assert torch.allclose(d, exact, atol=8 * torch.finfo(torch.float32).eps, rtol=0)
    tor = _torus()
    q = torch.randn(5000, 3, dtype=D, generator=g) * torch.tensor([1.2, 1.2, 0.4], dtype=D)
    d, _ = _geometry(q, tor)
    exact = ((q[:, :2].norm(dim=-1) - 1.0) ** 2 + q[:, 2] ** 2).sqrt() - 0.3
    off = exact.abs() > 5e-3  # beyond the polyhedral torus's deviation
    assert (torch.sign(d[off]) == torch.sign(exact[off])).all()


@pytest.mark.parametrize("dim", [2, 3])
def test_on_wall_queries_take_the_feature_pseudo_normal(dim):
    """Queries exactly at the vertices (and at cell centroids) of a circle / sphere:
    d = 0 and u is the feature's unit pseudo-normal, radial outward to O(h^2)."""
    mesh = _circle(64) if dim == 2 else _sphere(3)
    for q in (mesh.points, mesh.cell_centroids):
        d, u = _geometry(q, mesh)
        assert (d.abs() <= 1e-15).all()
        radial = (u * torch.nn.functional.normalize(q, dim=-1)).sum(-1)
        assert (radial > 0.99).all() and torch.allclose(u.norm(dim=-1), torch.ones_like(d))
    d, u = _geometry(mesh.points, mesh)
    assert torch.equal(d, torch.zeros_like(d))


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("shift", [0.0, 1.0, 100.0])
def test_exact_near_a_translated_wall_in_float32(dim, shift):
    """Regression (AirfRANS posed inference, 2026-10-03): near-wall float32 queries
    1e-5 to 1e-3 off a unit circle / sphere translated by 0, 1 and 100 radii: |d|
    equals the float64 brute-force distance to the same float32 surface to within
    a few float32 ulp of the coordinates, and the sign is right wherever the offset
    exceeds that resolution (the exact sign of the convex polygon / polyhedron)."""
    mesh = _circle(2000) if dim == 2 else _sphere(4)
    g = torch.Generator().manual_seed(7)
    dirs = torch.nn.functional.normalize(torch.randn(1000, dim, dtype=D, generator=g), dim=-1)
    side = torch.where(torch.rand(1000, dtype=D, generator=g) < 0.5, 1.0, -1.0)
    off = torch.tensor([1e-5, 1e-4, 1e-3], dtype=D)[torch.randint(3, (1000,), generator=g)]
    t = torch.full((dim,), shift, dtype=D)
    pts = (mesh.points + t).float()
    q = (dirs * (1 + side * off)[:, None] + t).float()
    d, u = exact_boundary_geometry(q, pts, mesh.cells)
    assert d.dtype == torch.float32
    ref = _brute_force_surface(q, pts, mesh.cells).values
    ulp = torch.finfo(torch.float32).eps * (1.0 + shift)
    assert (d.double().abs() - ref).abs().max() <= 4 * ulp
    ### the mesh is convex: a query is outside iff it is above some cell's plane
    simplex = pts.double()[mesh.cells]
    planes = ((q.double()[:, None] - simplex[None, :, 0]) * mesh.cell_normals.double()[None]).sum(-1)
    outside = planes.amax(-1) > 0
    resolved = ref > 8 * ulp
    assert (torch.where(outside, 1.0, -1.0)[resolved] == torch.sign(d.double())[resolved]).all()
    assert torch.allclose(u.norm(dim=-1), torch.ones_like(d), atol=1e-6)


@pytest.mark.parametrize("dim", [2, 3])
def test_exact_geometry_is_differentiable(dim):
    """The gradient of d in the query is sign(d) u (the envelope theorem), and the
    vertex positions receive a finite gradient."""
    mesh = _circle(64) if dim == 2 else _sphere(2)
    q = _near_wall(dim, 200, 8, band=0.3).requires_grad_(True)
    pts = mesh.points.clone().requires_grad_(True)
    d, u = exact_boundary_geometry(q, pts, mesh.cells)
    d.sum().backward()
    assert torch.allclose(q.grad, torch.sign(d.detach())[:, None] * u.detach(), atol=1e-12)
    assert pts.grad is not None and torch.isfinite(pts.grad).all() and pts.grad.abs().sum() > 0


def test_two_dimensional_search_chunks_over_the_queries():
    """More queries than one chunk of the 2-D search: the same answer as one block."""
    verts = _naca(50)
    g = torch.Generator().manual_seed(9)
    q = torch.randn(2 * geometry._TORCH_SEARCH_CHUNK + 17, 2, dtype=D, generator=g)
    d, _ = _geometry(q, _polygon(verts))
    assert torch.allclose(d, _polygon_sdf(q, verts), atol=1e-12, rtol=0)


def test_exact_geometry_refuses_other_dimensions():
    pts, nrm, _, _ = cloud(10, 4)
    mesh = soup(pts, nrm)
    with pytest.raises(NotImplementedError, match="two and three dimensions"):
        exact_boundary_geometry(pts, mesh.points, mesh.cells)


# ---------------------------------------------------------------------------
# The model's geometry comes from the surface, not from the token sample
# ---------------------------------------------------------------------------


def _with_measures(mesh: Mesh, keep, weights) -> Mesh:
    """The cells ``keep`` of ``mesh`` with effective measures ``weights`` (their
    Horvitz-Thompson weights), unreferenced vertices compacted out."""
    sub = mesh.slice_cells(keep)
    sub = sub.slice_points(torch.unique(sub.cells))
    return Mesh(points=sub.points, cells=sub.cells, cell_data={EFFECTIVE_MEASURE_KEY: weights})


def _draws(mesh: Mesh, k: int, seed: int = 0):
    """The token meshes of five sampling laws at about ``k`` of the ``N`` cells, and
    the surface with every cell split: uniform without replacement, area-
    proportional Poisson, systematic, one contiguous window, all with HT weights."""
    g = torch.Generator().manual_seed(seed)
    n, a = mesh.n_cells, mesh.cell_areas
    uniform = torch.randperm(n, generator=g)[:k].sort().values
    pi = (k * a / a.sum()).clamp(max=1.0)
    poisson = torch.nonzero(torch.rand(n, generator=g, dtype=a.dtype) < pi)[:, 0]
    step = n // k
    systematic = torch.arange(int(torch.randint(step, (1,), generator=g)), n, step)
    start = int(torch.randint(n, (1,), generator=g))
    window = (start + torch.arange(k)) % n
    return {
        "uniform": _with_measures(mesh, uniform, a[uniform] * n / k),
        "area": _with_measures(mesh, poisson, a[poisson] / pi[poisson]),
        "systematic": _with_measures(mesh, systematic, a[systematic] * step),
        "window": _with_measures(mesh, window.sort().values, a[window.sort().values] * n / k),
        "split": mesh.subdivide(levels=1, filter="linear"),
    }


def _captured_geometry(m, **kw):
    """The built-in geometry the model's batched routine receives, and the outputs."""
    seen = {}
    inner = m._forward_tokens

    def spy(boundary, query, *args):
        seen["d"], seen["u"] = query["wall_distance"].clone(), query["wall_direction"].clone()
        return inner(boundary, query, *args)

    m._forward_tokens = spy
    try:
        with torch.no_grad():
            out = outputs(m(**kw))
    finally:
        del m._forward_tokens
    return seen["d"], seen["u"], out


def test_builtin_geometry_is_the_same_for_every_draw_of_the_tokens():
    """A 798-segment NACA 0012 as boundary_geometry and its tokens drawn under five
    laws at 200 cells (and as the surface with every cell split): the distance and
    direction the model computes are bitwise identical for every draw, and equal to
    the exact geometry of the surface; only the quadratures (and so the outputs)
    change with the draw."""
    torch.manual_seed(0)
    m = ISLA(output_schema=PV, global_schema={"g": {"rank": 1}}, hidden=32, n_layers=2, n_slices=8, interior=True, dim=2).double().eval()
    surface = _polygon(_naca(400))
    g = torch.Generator().manual_seed(1)
    q = torch.tensor([0.5, 0.0], dtype=D) + torch.randn(300, 2, dtype=D, generator=g) * torch.tensor([0.6, 0.1], dtype=D)
    drv = torch.tensor([1.0, 0.1], dtype=D)
    ref = exact_boundary_geometry(q, surface.points, surface.cells)
    seen = {}
    for law, tokens in _draws(surface, 200).items():
        d, u, out = _captured_geometry(
            m, boundary_meshes={"airfoil": tokens}, boundary_geometry={"airfoil": surface}, prediction_points=q, global_data={"g": drv}
        )
        assert torch.equal(d[0], ref[0]) and torch.equal(u[0], ref[1]), law
        assert all(torch.isfinite(v).all() for v in out.values())
        seen[law] = out["p"]
    assert not torch.allclose(seen["uniform"], seen["window"])


def test_a_sampled_boundary_without_its_surface_is_refused():
    """Without boundary_geometry the tokens' cells are the surface, so an interior
    model refuses tokens whose effective measures exceed their cells' measures (a
    subsample's HT weights) and accepts the whole boundary, with or without stored
    effective measures; given the surface, the sample runs."""
    torch.manual_seed(0)
    m = ISLA(output_schema=PV, global_schema={"g": {"rank": 1}}, hidden=32, n_layers=1, n_slices=8, interior=True, dim=2).double().eval()
    surface = _polygon(_naca(100))
    kw = dict(prediction_points=torch.randn(20, 2, dtype=D), global_data={"g": torch.tensor([1.0, 0.0], dtype=D)})
    sample = _draws(surface, 50)["uniform"]
    with torch.no_grad():
        with pytest.raises(ValueError, match="is a sample of its boundary"):
            m(boundary_meshes={"airfoil": sample}, **kw)
        m(boundary_meshes={"airfoil": sample}, boundary_geometry={"airfoil": surface}, **kw)
        m(boundary_meshes={"airfoil": surface}, **kw)
        whole = Mesh(points=surface.points, cells=surface.cells, cell_data={EFFECTIVE_MEASURE_KEY: surface.cell_areas})
        m(boundary_meshes={"airfoil": whole}, **kw)


def test_boundary_geometry_is_validated():
    """The surface names the same boundaries as the tokens, each a non-empty Mesh of
    the boundary's dimensions; a surface model takes none; the interior task is
    refused outside two and three dimensions."""
    torch.manual_seed(0)
    m = ISLA(output_schema=PV, global_schema={"g": {"rank": 1}}, hidden=32, n_layers=1, n_slices=8, interior=True, dim=2).double().eval()
    surface = _polygon(_naca(60))
    kw = dict(boundary_meshes={"airfoil": surface}, prediction_points=torch.randn(10, 2, dtype=D), global_data={"g": torch.tensor([1.0, 0.0], dtype=D)})
    with torch.no_grad():
        with pytest.raises(ValueError, match="must name the same boundaries"):
            m(boundary_geometry={"wing": surface}, **kw)
        with pytest.raises(ValueError, match=r"must be a Mesh\[1, 2\]"):
            m(boundary_geometry={"airfoil": _sphere(1)}, **kw)
        with pytest.raises(ValueError, match="boundary_geometry has no cells"):
            m(boundary_geometry={"airfoil": Mesh(points=surface.points, cells=surface.cells[:0])}, **kw)
        m(boundary_geometry={"airfoil": surface}, **kw)
    surface_model = ISLA(output_schema=PV, global_schema={"g": {"rank": 1}}, hidden=32, n_layers=1, n_slices=8, dim=2)
    with pytest.raises(ValueError, match="interior=False"):
        surface_model(boundary_meshes={"airfoil": surface}, boundary_geometry={"airfoil": surface}, global_data=kw["global_data"])
    with pytest.raises(NotImplementedError, match="two and three dimensions"):
        ISLA(output_schema=PV, interior=True, dim=4)


def _flipped(mesh: Mesh) -> Mesh:
    """``mesh`` with every cell's last two vertices swapped (each normal reversed)."""
    d = mesh.cells.shape[-1]
    return Mesh(points=mesh.points, cells=mesh.cells[:, [*range(d - 2), d - 1, d - 2]])


@pytest.mark.parametrize("dim", [2, 3])
def test_the_surface_is_oriented_by_its_tokens(dim):
    """Each surface is oriented as its own tokens, so reversing every cell of
    boundary_geometry leaves every output bitwise identical (the model swaps them
    back), and reversing the tokens instead reverses the sign of the distance and
    leaves the direction (from the surface to the point) as it is."""
    torch.manual_seed(0)
    m = ISLA(output_schema=PV, global_schema={"g": {"rank": 1}}, hidden=32, n_layers=2, n_slices=8, interior=True, dim=dim).double().eval()
    surface = _polygon(_naca(200)) if dim == 2 else _sphere(3)
    tokens = _draws(surface, 100)["uniform"]
    g = torch.Generator().manual_seed(3)
    q = torch.randn(80, dim, dtype=D, generator=g) * 0.7
    kw = dict(boundary_meshes={"wall": tokens}, prediction_points=q, global_data={"g": torch.ones(dim, dtype=D)})
    d, u, out = _captured_geometry(m, boundary_geometry={"wall": surface}, **kw)
    d2, u2, out2 = _captured_geometry(m, boundary_geometry={"wall": _flipped(surface)}, **kw)
    assert torch.equal(d, d2) and torch.equal(u, u2)
    assert all(torch.equal(out[k], out2[k]) for k in out)
    reversed_tokens = dict(kw, boundary_meshes={"wall": _flipped(tokens)})
    d3, u3, _ = _captured_geometry(m, boundary_geometry={"wall": surface}, **reversed_tokens)
    ### (to rounding: the model's swap of the cells rebuilds each closest point from
    ### its vertices in the other order)
    assert torch.allclose(d3, -d, atol=1e-14, rtol=0) and torch.allclose(u3, u, atol=1e-14, rtol=0)


def test_the_sign_is_the_nearest_token_convention():
    """On a consistently oriented case (outward tokens and surface) the distance has
    the sign of the former built-in, the nearest boundary token's sign((x - y) . n),
    at every query farther from the wall than the token spacing."""
    torch.manual_seed(0)
    m = ISLA(output_schema=PV, global_schema={"g": {"rank": 1}}, hidden=32, n_layers=1, n_slices=8, interior=True).double().eval()
    surface = _sphere(4)
    tokens = _draws(surface, 600)["uniform"]
    q = _near_wall(3, 500, 4, band=0.5)
    d, _, _ = _captured_geometry(
        m, boundary_meshes={"wall": tokens}, boundary_geometry={"wall": surface}, prediction_points=q, global_data={"g": torch.ones(3, dtype=D)}
    )
    y, n = tokens.cell_centroids, tokens.cell_normals
    nearest = torch.cdist(q, y).argmin(-1)
    old_sign = torch.sign(((q - y[nearest]) * n[nearest]).sum(-1))
    far = (q.norm(dim=-1) - 1).abs() > 0.15  # beyond the 600-token sample's spacing
    assert far.sum() > 100 and torch.equal(torch.sign(d[0])[far], old_sign[far])


def test_eager_and_compiled_agree_on_an_oppositely_oriented_surface():
    """A 2-D case whose boundary_geometry is oriented against its tokens: the
    fullgraph, dynamic-shape compiled model gives the eager outputs (the orientation
    is a torch.where, not a Python branch), at two query counts."""
    torch.manual_seed(0)
    m = ISLA(output_schema=PV, global_schema={"g": {"rank": 1}}, hidden=32, n_layers=1, n_slices=8, interior=True, dim=2).eval()
    surface = _polygon(_naca(100))
    surface = Mesh(points=surface.points.float(), cells=surface.cells)
    tokens = _draws(surface, 60)["uniform"]
    compiled = torch.compile(m, fullgraph=True, dynamic=True)
    g = torch.Generator().manual_seed(5)
    for nq in (50, 71):
        kw = dict(
            boundary_meshes={"wall": tokens},
            boundary_geometry={"wall": _flipped(surface)},
            prediction_points=torch.randn(nq, 2, generator=g) * 0.6,
            global_data={"g": torch.tensor([1.0, 0.2])},
        )
        with torch.no_grad():
            eager, comp = outputs(m(**kw)), outputs(compiled(**kw))
        for k in eager:
            torch.testing.assert_close(comp[k], eager[k], atol=1e-4, rtol=1e-4)
