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

"""The two forms of the prediction points' wall geometry.

``wall_geometry="sampled"`` (default) builds each prediction point's signed distance
and direction from the nearest boundary token (cell centroid); ``"exact"`` from the
nearest point of the boundary triangles through the Warp signed-distance query.
These tests pin the exact form against the analytic distance to a sphere, its sign
convention against the sampled form, its covariance contracts, and its API guards.
The Warp query runs in float32, so the exact-form tolerances are float32-sized even
where the model runs in double.
"""

import pytest
import torch

pytest.importorskip("warp")

from physicsnemo.experimental.models import ISLA
from physicsnemo.experimental.models.isla.model import (
    exact_boundary_geometry,
    nearest_boundary_geometry,
)
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from test.experimental.models.isla._meshes import orthogonal, outputs, transform

D = torch.float64
PV = {"p": {"rank": 0}, "v": {"rank": 1}}
R = 2.0


def _sphere(subdivisions: int = 5, radius: float = R) -> Mesh:
    """A closed icosphere in double: outward cell normals, 20 * 4**subdivisions
    triangles (20,480 at 5, sample spacing about 0.05 R)."""
    m = sphere_icosahedral.load(radius=radius, subdivisions=subdivisions)
    return Mesh(points=m.points.to(D), cells=m.cells)


def _radial_queries(radii, n_per: int = 25, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    dirs = torch.nn.functional.normalize(torch.randn(len(radii) * n_per, 3, dtype=D, generator=g), dim=-1)
    r = torch.as_tensor(radii, dtype=D).repeat_interleave(n_per)
    return dirs * r[:, None], dirs, r


def _model(**kw):
    torch.manual_seed(0)
    return ISLA(outputs=PV, global_inputs={"g": {"rank": 1}}, hidden=32, n_layers=2, n_slices=8, interior=True, **kw).double().eval()


def test_exact_matches_the_analytic_sphere_distance_where_sampled_overestimates():
    """On a fine icosphere the exact |d| is the distance to the faceted surface
    (within the facet sag, about 4e-4 R at 20,480 triangles), while the sampled |d|
    overestimates by up to about half a sample spacing near the wall."""
    mesh = _sphere()
    q, _, r = _radial_queries([R + 0.02, R - 0.02, R + 0.5, R - 0.5])
    analytic = (r - R).abs()
    d_exact, u_exact = exact_boundary_geometry([mesh], q)
    d_sampled, _ = nearest_boundary_geometry(q[None], mesh.cell_centroids[None], mesh.cell_normals[None])
    err_exact = (d_exact.abs() - analytic).abs()
    err_sampled = (d_sampled[0].abs() - analytic).abs()
    assert d_exact.shape == (100,) and u_exact.shape == (100, 3)
    assert (err_exact < 1e-3 * R).all()
    assert err_exact.mean() < err_sampled.mean()
    ### the centroids lie on the surface, so the sampled |d| is never below the
    ### exact |d| (beyond float32 roundoff); near the wall it is first order in
    ### the spacing where the exact error is the facet sag
    assert (d_sampled[0].abs() - d_exact.abs() >= -1e-6).all()
    near = r.sub(R).abs() < 0.1
    assert err_sampled[near].max() > 10 * err_exact[near].max()
    ### the direction is the surface's outward normal, flipped inside
    assert torch.allclose(u_exact.norm(dim=-1), torch.ones(100, dtype=D), atol=1e-6)


def test_sign_convention_matches_the_sampled_form_on_both_sides():
    """Positive on the side the normals point to (outside the sphere), negative
    inside; the direction points from the wall to the query, so radially outward
    outside and radially inward inside, in both forms."""
    mesh = _sphere(subdivisions=4)
    q, dirs, r = _radial_queries([R + 0.3, R - 0.3])
    d_exact, u_exact = exact_boundary_geometry([mesh], q)
    d_sampled, u_sampled = nearest_boundary_geometry(q[None], mesh.cell_centroids[None], mesh.cell_normals[None])
    outside = r > R
    assert (d_exact[outside] > 0).all() and (d_exact[~outside] < 0).all()
    assert torch.equal(d_exact.sign(), d_sampled[0].sign())
    assert ((u_exact * dirs).sum(-1)[outside] > 0.999).all()
    assert ((u_exact * dirs).sum(-1)[~outside] < -0.999).all()
    ### the sampled direction deviates by up to about spacing / distance (0.1 / 0.3)
    assert ((u_exact * u_sampled[0]).sum(-1) > 0.9).all()


def test_multi_mesh_boundary_takes_the_closest_surface_per_query():
    """Two spheres as two boundary meshes: each query's geometry is that of the
    sphere with the smaller |d|, identical to the single-mesh result on the union."""
    a = _sphere(subdivisions=3)
    b = transform(_sphere(subdivisions=3), shift=torch.tensor([6.0, 0.0, 0.0], dtype=D))
    g = torch.Generator().manual_seed(1)
    q = torch.randn(60, 3, dtype=D, generator=g) * 2.5 + torch.tensor([3.0, 0.0, 0.0], dtype=D)
    d, u = exact_boundary_geometry([a, b], q)
    d_a, u_a = exact_boundary_geometry([a], q)
    d_b, u_b = exact_boundary_geometry([b], q)
    pick_a = d_a.abs() <= d_b.abs()
    assert torch.equal(d, torch.where(pick_a, d_a, d_b))
    assert torch.equal(u, torch.where(pick_a[:, None], u_a, u_b))
    assert pick_a.any() and (~pick_a).any()


def test_on_wall_queries_take_the_hit_cell_normal():
    mesh = _sphere(subdivisions=3)
    ### a vertex is exactly on the surface (float32-representable, so the Warp
    ### hit point is the vertex itself)
    q = mesh.points[:12].float().to(D)
    mesh32 = Mesh(points=mesh.points.float().to(D), cells=mesh.cells)
    d, u = exact_boundary_geometry([mesh32], q)
    assert torch.equal(d, torch.zeros(12, dtype=D))
    assert torch.allclose(u.norm(dim=-1), torch.ones(12, dtype=D), atol=1e-12)
    ### the hit cell's normal on a sphere is within a facet angle of the radial one
    assert ((u * torch.nn.functional.normalize(q, dim=-1)).sum(-1) > 0.98).all()


def test_exact_form_is_not_differentiable_and_sampled_form_is():
    mesh = _sphere(subdivisions=3)
    q = _radial_queries([R + 0.5])[0].requires_grad_(True)
    d, u = exact_boundary_geometry([mesh], q)
    assert not d.requires_grad and not u.requires_grad
    d_s, _ = nearest_boundary_geometry(q[None], mesh.cell_centroids[None], mesh.cell_normals[None])
    assert d_s.requires_grad


def test_interior_contracts_hold_with_exact_wall_geometry():
    """Rotation and reflection covariance, translation and geometric-scale
    invariance with the exact wall geometry. The Warp query runs in float32, so the
    geometry carries about 1e-7 relative error into a double model: tolerances are
    1e-4 here where the sampled form's test uses 1e-11."""
    mesh = _sphere(subdivisions=3)
    drv = torch.tensor([0.3, 1.0, 0.2], dtype=D)
    q = _radial_queries([R + 0.4, R - 0.4, R + 1.5], n_per=12, seed=2)[0]
    m = _model(wall_geometry="exact")

    def call(Rm=None, t=None, k=1.0, q=q):
        Rm = torch.eye(3, dtype=D) if Rm is None else Rm
        t = torch.zeros(3, dtype=D) if t is None else t
        with torch.no_grad():
            return outputs(
                m(boundary_meshes={"wall": transform(mesh, Rm, t, scale=k)}, prediction_points=k * q @ Rm.T + t, global_data={"g": drv @ Rm.T})
            )

    base = call()
    for improper in (False, True):
        Rm = orthogonal(3, 4, improper)
        out = call(Rm, torch.tensor([2.0, -5.0, 1.0], dtype=D), k=2.5)
        assert torch.allclose(out["p"], base["p"], atol=1e-4), improper
        assert torch.allclose(out["v"], base["v"] @ Rm.T, atol=1e-4), improper
    ### the distance channel is live
    far = call(q=q * 3.0)
    assert not torch.allclose(far["p"], base["p"], atol=1e-6)


def test_exact_and_sampled_differ_near_the_wall_and_sampled_is_the_default():
    """The option is live: the two forms give different outputs on the same case;
    the default is the sampled form, bitwise the explicit one."""
    mesh = _sphere(subdivisions=3)
    q = _radial_queries([R + 0.05, R - 0.05], n_per=10)[0]
    kw = dict(boundary_meshes={"wall": mesh}, prediction_points=q, global_data={"g": torch.tensor([0.0, 0.0, 1.0], dtype=D)})
    with torch.no_grad():
        default = outputs(_model()(**kw))
        sampled = outputs(_model(wall_geometry="sampled")(**kw))
        exact = outputs(_model(wall_geometry="exact")(**kw))
    assert all(torch.equal(default[k], sampled[k]) for k in default)
    assert not torch.allclose(exact["p"], sampled["p"], atol=1e-6)


def test_exact_requires_three_dimensions_and_cells():
    with pytest.raises(ValueError, match="wall_geometry='exact'.*three spatial dimensions"):
        ISLA(outputs=PV, dim=2, interior=True, wall_geometry="exact")
    with pytest.raises(ValueError, match="wall_geometry must be"):
        ISLA(outputs=PV, interior=True, wall_geometry="nearest")
    m = _model(wall_geometry="exact")
    empty = Mesh(points=torch.zeros(0, 3, dtype=D), cells=torch.zeros(0, 3, dtype=torch.long))
    with pytest.raises(ValueError, match="has no cells"):
        m(boundary_meshes={"wall": empty}, prediction_points=torch.zeros(2, 3, dtype=D), global_data={"g": torch.ones(3, dtype=D)})


def test_exact_option_survives_a_checkpoint_round_trip(tmp_path):
    from physicsnemo.core.module import Module

    m = _model(wall_geometry="exact")
    m.save(tmp_path / "m.mdlus")
    m2 = Module.from_checkpoint(str(tmp_path / "m.mdlus")).double().eval()
    assert m2.wall_geometry == "exact"
    mesh = _sphere(subdivisions=3)
    kw = dict(boundary_meshes={"wall": mesh}, prediction_points=_radial_queries([R + 0.3])[0], global_data={"g": torch.ones(3, dtype=D)})
    with torch.no_grad():
        a, b = outputs(m(**kw)), outputs(m2(**kw))
    assert all(torch.equal(a[k], b[k]) for k in a)
