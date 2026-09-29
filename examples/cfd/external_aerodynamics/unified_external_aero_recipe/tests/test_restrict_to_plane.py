# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RestrictToPlane (ND-GENERIC, 2026-09-28): a planar DomainMesh stored in three coordinates becomes a native
two-dimensional one; the boundary-value extension computed in the plane equals the embedded one; non-planar input is refused."""

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.mesh import DomainMesh, Mesh

from domain_transforms import BoundaryValueExtension, RestrictToPlane


def _square_domain(z: float = 0.0, n_side: int = 8, n_int: int = 200, seed: int = 0) -> DomainMesh:
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(-1, 1, n_side + 1)[:-1]
    corners = torch.cat([torch.stack([t, -torch.ones_like(t)], 1), torch.stack([torch.ones_like(t), t], 1),
                         torch.stack([-t, torch.ones_like(t)], 1), torch.stack([-torch.ones_like(t), -t], 1)])
    pts = torch.cat([corners, torch.full((corners.shape[0], 1), z)], 1)
    m = pts.shape[0]
    cells = torch.stack([torch.arange(m), (torch.arange(m) + 1) % m], 1)
    seg = pts[cells[:, 1]] - pts[cells[:, 0]]
    normals = torch.stack([seg[:, 1], -seg[:, 0], torch.zeros(m)], 1)
    normals = normals / normals.norm(dim=1, keepdim=True)
    lab = (torch.arange(m) % 2 == 0).float()
    cd = TensorDict({"normals": normals, "quadrature_measure": seg.norm(dim=1), "bc_dirichlet": lab,
                     "bc_neumann": 1 - lab, "bc_value_g": torch.randn(m, generator=g)}, batch_size=[m])
    bnd = Mesh(points=pts, cells=cells, cell_data=cd)
    xi = torch.rand(n_int, 2, generator=g) * 1.6 - 0.8
    interior_pts = torch.cat([xi, torch.full((n_int, 1), z)], 1)
    sdf_normals = torch.cat([xi / xi.norm(dim=1, keepdim=True), torch.zeros(n_int, 1)], 1)
    pd = TensorDict({"sdf_normals": sdf_normals, "u": torch.randn(n_int, 1, generator=g)}, batch_size=[n_int])
    interior = Mesh(points=interior_pts, cells=torch.zeros(0, 1, dtype=torch.long), point_data=pd)
    return DomainMesh(interior=interior, boundaries={"outer": bnd})


VEC = ["interior.point_data.sdf_normals", "boundaries.outer.cell_data.normals"]


def test_drops_the_constant_axis_from_points_and_vectors():
    d3 = _square_domain(z=0.25)
    d2 = RestrictToPlane(axis=2, vector_fields=VEC).apply_to_domain(d3)
    assert d2.interior.points.shape == (200, 2)
    assert d2.boundaries["outer"].points.shape[1] == 2
    assert d2.boundaries["outer"].cell_centroids.shape[1] == 2
    assert d2.interior.point_data["sdf_normals"].shape == (200, 2)
    assert d2.boundaries["outer"].cell_data["normals"].shape[1] == 2
    torch.testing.assert_close(d2.interior.points, d3.interior.points[:, :2])
    torch.testing.assert_close(d2.boundaries["outer"].cell_data["normals"], d3.boundaries["outer"].cell_data["normals"][:, :2])
    # scalars and connectivity untouched
    torch.testing.assert_close(d2.interior.point_data["u"], d3.interior.point_data["u"])
    torch.testing.assert_close(d2.boundaries["outer"].cells, d3.boundaries["outer"].cells)
    torch.testing.assert_close(d2.boundaries["outer"].cell_data["bc_value_g"], d3.boundaries["outer"].cell_data["bc_value_g"])


def test_extension_in_the_plane_equals_the_embedded_extension():
    d3 = _square_domain()
    ext = BoundaryValueExtension(boundary="outer", value_field="bc_value_g", label_fields=["bc_dirichlet", "bc_neumann"])
    e3 = ext.apply_to_domain(d3).interior.point_data
    e2 = ext.apply_to_domain(RestrictToPlane(vector_fields=VEC).apply_to_domain(d3)).interior.point_data
    for k in ("ext_dirichlet", "ext_dist_dirichlet", "ext_neumann", "ext_dist_neumann"):
        torch.testing.assert_close(e2[k], e3[k])


def test_non_planar_input_is_refused_and_missing_fields_are_skipped():
    d3 = _square_domain()
    d3.interior.points[0, 2] = 1e-3
    with pytest.raises(ValueError, match="not planar"):
        RestrictToPlane(vector_fields=VEC).apply_to_domain(d3)
    RestrictToPlane(atol=None, vector_fields=VEC + ["interior.point_data.absent"]).apply_to_domain(d3)
    with pytest.raises(ValueError, match="must be"):
        RestrictToPlane(vector_fields=["sdf_normals"]).apply_to_domain(_square_domain())


def test_isla_runs_natively_in_two_dimensions_on_the_restricted_domain():
    from physicsnemo.experimental.models import ISLA

    d2 = RestrictToPlane(vector_fields=VEC).apply_to_domain(_square_domain())
    model = ISLA(outputs={"u": 0}, boundary_inputs={"bc_dirichlet": 0, "bc_value_g": 0},
                 query_inputs={"sdf_normals": 1}, interior=True,
                 hidden=32, n_layers=1, n_slices=8, dim=2, geo_kernel="eager")
    ### The restricted boundary is a Mesh[1, 2] (a polyline): its cells carry the normals
    ### and lengths ISLA reads; the interior Mesh's point_data carries the declared query input.
    out = model(boundary_meshes={"outer": d2.boundaries["outer"]}, prediction_points=d2.interior)
    assert out.point_data["u"].shape == (200,)
    torch.testing.assert_close(out.points, d2.interior.points)
