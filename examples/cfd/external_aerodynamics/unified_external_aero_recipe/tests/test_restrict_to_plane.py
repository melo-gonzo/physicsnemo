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

"""RestrictToPlane (ND-GENERIC, 2026-09-28): a planar DomainMesh stored in three coordinates becomes a native
two-dimensional one; non-planar input is refused; ISLA runs natively on the restricted domain."""

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.mesh import DomainMesh, Mesh

from domain_transforms import RestrictToPlane


def _square_domain(
    z: float = 0.0, n_side: int = 8, n_int: int = 200, seed: int = 0
) -> DomainMesh:
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(-1, 1, n_side + 1)[:-1]
    corners = torch.cat(
        [
            torch.stack([t, -torch.ones_like(t)], 1),
            torch.stack([torch.ones_like(t), t], 1),
            torch.stack([-t, torch.ones_like(t)], 1),
            torch.stack([-torch.ones_like(t), -t], 1),
        ]
    )
    pts = torch.cat([corners, torch.full((corners.shape[0], 1), z)], 1)
    m = pts.shape[0]
    cells = torch.stack([torch.arange(m), (torch.arange(m) + 1) % m], 1)
    seg = pts[cells[:, 1]] - pts[cells[:, 0]]
    normals = torch.stack([seg[:, 1], -seg[:, 0], torch.zeros(m)], 1)
    normals = normals / normals.norm(dim=1, keepdim=True)
    lab = (torch.arange(m) % 2 == 0).float()
    cd = TensorDict(
        {
            "normals": normals,
            "quadrature_measure": seg.norm(dim=1),
            "bc_dirichlet": lab,
            "bc_neumann": 1 - lab,
            "bc_value_g": torch.randn(m, generator=g),
        },
        batch_size=[m],
    )
    bnd = Mesh(points=pts, cells=cells, cell_data=cd)
    xi = torch.rand(n_int, 2, generator=g) * 1.6 - 0.8
    interior_pts = torch.cat([xi, torch.full((n_int, 1), z)], 1)
    sdf_normals = torch.cat(
        [xi / xi.norm(dim=1, keepdim=True), torch.zeros(n_int, 1)], 1
    )
    pd = TensorDict(
        {"sdf_normals": sdf_normals, "u": torch.randn(n_int, 1, generator=g)},
        batch_size=[n_int],
    )
    interior = Mesh(
        points=interior_pts, cells=torch.zeros(0, 1, dtype=torch.long), point_data=pd
    )
    return DomainMesh(interior=interior, boundaries={"outer": bnd})


VEC = ["interior.point_data.sdf_normals", "boundaries.outer.cell_data.normals"]


def test_drops_the_constant_axis_from_points_and_vectors():
    """Points and the named vector fields lose the constant axis; scalars and connectivity are untouched."""
    d3 = _square_domain(z=0.25)
    d2 = RestrictToPlane(axis=2, vector_fields=VEC).apply_to_domain(d3)
    assert d2.interior.points.shape == (200, 2)
    assert d2.boundaries["outer"].points.shape[1] == 2
    assert d2.boundaries["outer"].cell_centroids.shape[1] == 2
    assert d2.interior.point_data["sdf_normals"].shape == (200, 2)
    assert d2.boundaries["outer"].cell_data["normals"].shape[1] == 2
    torch.testing.assert_close(d2.interior.points, d3.interior.points[:, :2])
    torch.testing.assert_close(
        d2.boundaries["outer"].cell_data["normals"],
        d3.boundaries["outer"].cell_data["normals"][:, :2],
    )
    # scalars and connectivity untouched
    torch.testing.assert_close(d2.interior.point_data["u"], d3.interior.point_data["u"])
    torch.testing.assert_close(
        d2.boundaries["outer"].cells, d3.boundaries["outer"].cells
    )
    torch.testing.assert_close(
        d2.boundaries["outer"].cell_data["bc_value_g"],
        d3.boundaries["outer"].cell_data["bc_value_g"],
    )


def test_non_planar_input_is_refused_and_missing_fields_are_skipped():
    """A domain off the plane raises; a named field the domain lacks is skipped."""
    d3 = _square_domain()
    d3.interior.points[0, 2] = 1e-3
    with pytest.raises(ValueError, match="not planar"):
        RestrictToPlane(vector_fields=VEC).apply_to_domain(d3)
    RestrictToPlane(
        atol=None, vector_fields=VEC + ["interior.point_data.absent"]
    ).apply_to_domain(d3)
    with pytest.raises(ValueError, match="must be"):
        RestrictToPlane(vector_fields=["sdf_normals"]).apply_to_domain(_square_domain())


def test_isla_runs_natively_in_two_dimensions_on_the_restricted_domain():
    """ISLA with dim=2 runs a forward pass on the restricted domain."""
    from physicsnemo.experimental.models.isla import ISLA

    d2 = RestrictToPlane(vector_fields=VEC).apply_to_domain(_square_domain())
    model = ISLA(
        output_schema={"u": {"rank": 0}},
        boundary_schema={"bc_dirichlet": {"rank": 0}, "bc_value_g": {"rank": 0}},
        query_schema={"sdf_normals": {"rank": 1}},
        interior=True,
        hidden=32,
        n_layers=1,
        n_slices=8,
        dim=2,
    )
    ### The restricted boundary is a Mesh[1, 2] (a polyline): its cells carry the normals
    ### and lengths ISLA reads; the interior Mesh's point_data carries the declared query input.
    out = model(
        boundary_meshes={"outer": d2.boundaries["outer"]}, prediction_points=d2.interior
    )
    assert out.point_data["u"].shape == (200,)
    torch.testing.assert_close(out.points, d2.interior.points)


def test_named_global_vectors_lose_the_axis_too():
    """Named per-case global vectors lose the constant axis; global scalars are untouched."""
    d = _square_domain()
    d = DomainMesh(
        interior=d.interior,
        boundaries=d.boundaries,
        global_data=TensorDict(
            {
                "U_inf_dir": torch.tensor([0.6, 0.8, 0.0]),
                "U_inf_mag": torch.tensor(30.0),
                "Re": torch.tensor(1e6),
            },
            batch_size=[],
        ),
    )
    out = RestrictToPlane(
        axis=2,
        vector_fields=VEC,
        global_fields=["global_data.U_inf_dir", "U_inf", "missing"],
    ).apply_to_domain(d)
    torch.testing.assert_close(out.global_data["U_inf_dir"], torch.tensor([0.6, 0.8]))
    assert (
        out.global_data["U_inf_mag"].shape == () and out.global_data["Re"].shape == ()
    )  # scalars untouched
    assert out.interior.points.shape[1] == 2
