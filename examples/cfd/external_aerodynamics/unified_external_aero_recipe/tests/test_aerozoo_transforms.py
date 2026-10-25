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

"""The AeroZoo adapters (2026-10-09): ComputeLogReynolds, HarmonizeBoundaryFields, BoundaryFieldToGlobal
and ConcatGlobalFields; ISLA's ComputeBoundaryScales (2026-10-24); the eddy-viscosity / k / omega non-dimensionalization recipes and their inverse; the
SDF transform's far fill for a boundary the case does not have."""

import math

import pytest
import torch
from domain_transforms import (
    BoundaryFieldToGlobal,
    ComputeBoundaryScales,
    ComputeLogReynolds,
    ConcatGlobalFields,
    HarmonizeBoundaryFields,
)
from nondim import NonDimensionalizeByMetadata, freestream_scales
from sdf import ComputeSDFFromBoundary
from tensordict import TensorDict

from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.calculus.measure import scale_measures

ONE_HOT = {"inlet": "bc_inlet", "wall": "bc_wall", "slip": "bc_slip"}
FIELDS = {"prescribed.dirichlet.velocity": 1, "prescribed.dirichlet.pressure": 0}


def _triangles(n: int, offset: float = 0.0) -> Mesh:
    """``n`` disjoint unit right triangles in the plane z = offset."""
    pts = torch.cat(
        [
            torch.tensor(
                [
                    [2.0 * i, 0.0, offset],
                    [2.0 * i + 1, 0.0, offset],
                    [2.0 * i, 1.0, offset],
                ]
            )
            for i in range(n)
        ]
    )
    return Mesh(points=pts, cells=torch.arange(3 * n).reshape(n, 3))


def _empty_triangles() -> Mesh:
    return Mesh(points=torch.zeros(0, 3), cells=torch.zeros(0, 3, dtype=torch.long))


def _global_data() -> TensorDict:
    return TensorDict(
        {
            "U_inf": torch.tensor([10.0, 0.0, 0.0]),
            "rho_inf": torch.tensor(1.2),
            "p_inf": torch.tensor(0.0),
            "L_ref": torch.tensor(2.0),
            "reference": {"speed": torch.tensor(10.0)},
            "physics": {"kinematic_viscosity": torch.tensor(1.5e-5)},
        },
        batch_size=[],
    )


def _domain(slip_faces: int = 2) -> DomainMesh:
    inlet = _triangles(2)
    inlet.cell_data["prescribed", "dirichlet", "velocity"] = torch.tensor(
        [[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
    )
    inlet.cell_data["prescribed", "dirichlet", "tke"] = torch.tensor([0.5, 0.5])
    wall = _triangles(3, offset=1.0)
    wall.cell_data["prescribed", "dirichlet", "velocity"] = torch.zeros(3, 3)
    wall.cell_data["derived", "y_plus"] = torch.ones(3)
    slip = _triangles(slip_faces, offset=2.0) if slip_faces else _empty_triangles()
    return DomainMesh(
        interior=Mesh(points=torch.rand(20, 3) * 4),
        boundaries={"inlet": inlet, "wall": wall, "slip": slip},
        global_data=_global_data(),
    )


### ---------------------------------------------------------------------------
### ComputeLogReynolds
### ---------------------------------------------------------------------------


def test_log_reynolds_is_written_raw_and_standardized():
    """log10(U L / nu) from nested leaves, standardized with the given constants, in float64 with the
    viscosity rounded to the meshes' dtype (float32) first; the sources stay."""
    out = ComputeLogReynolds(
        speed_field="reference.speed",
        length_field="L_ref",
        viscosity_field="physics.kinematic_viscosity",
        mean=5.5,
        std=1.5,
    ).apply_to_domain(_domain())
    gd = out.global_data
    nu = float(torch.tensor(1.5e-5, dtype=torch.float32))
    expected = math.log10(10.0 * 2.0 / nu)
    assert gd["log_re"].shape == () and gd["log_re_std"].shape == ()
    assert gd["log_re"].dtype == gd["log_re_std"].dtype == torch.float64
    torch.testing.assert_close(
        gd["log_re"], torch.tensor(expected, dtype=torch.float64), atol=1e-15, rtol=0
    )
    torch.testing.assert_close(
        gd["log_re_std"],
        torch.tensor((expected - 5.5) / 1.5, dtype=torch.float64),
        atol=1e-15,
        rtol=0,
    )
    assert float(gd["reference", "speed"]) == 10.0


def test_log_reynolds_refuses_missing_leaves_and_bad_constants():
    """A missing leaf and a non-positive std are refused."""
    with pytest.raises(KeyError, match="needs global_data"):
        ComputeLogReynolds("nope", "L_ref", "physics.kinematic_viscosity", 5.5, 1.5)(
            Mesh(points=torch.rand(4, 3), global_data=_global_data())
        )
    with pytest.raises(ValueError, match="positive"):
        ComputeLogReynolds(
            "reference.speed", "L_ref", "physics.kinematic_viscosity", 5.5, 0.0
        )


### ---------------------------------------------------------------------------
### ComputeBoundaryScales
### ---------------------------------------------------------------------------


def _unit_square() -> Mesh:
    """The unit square in the plane z = 0 as two triangles, a velocity (3, 4, 0) on the first, 0 on the second."""
    pts = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    return Mesh(
        points=pts,
        cells=torch.tensor([[0, 1, 2], [0, 2, 3]]),
        cell_data={"u": torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])},
    )


def _segments(x) -> Mesh:
    """A straight polyline in the plane through the points (x_i, 0), a velocity (1, 0) on each segment."""
    pts = torch.stack(
        [torch.as_tensor(x, dtype=torch.float32), torch.zeros(len(x))], -1
    )
    n = len(x) - 1
    cells = torch.stack([torch.arange(n), torch.arange(1, n + 1)], -1)
    return Mesh(
        points=pts,
        cells=cells,
        cell_data={"u": torch.tensor([[1.0, 0.0]]).expand(n, 2).clone()},
    )


def _scales_domain(boundaries: dict, global_data=None) -> DomainMesh:
    d = next(iter(boundaries.values())).n_spatial_dims
    return DomainMesh(
        interior=Mesh(points=torch.zeros(3, d)),
        boundaries=boundaries,
        global_data=global_data if global_data is not None else {},
    )


def test_boundary_scales_take_their_hand_values():
    """The unit square: L^2 = E|x - y|^2 = 1/3 for two uniform points, U = sqrt(sum a |v|^2 / sum a) =
    sqrt(25 / 2); a straight segment of length 2 cut unevenly in three: L^2 = 2/3, whatever the cut (each
    segment's own second moment), U = 1; a global source: U = |U_inf|. Both float64."""
    gd = _scales_domain({"wall": _unit_square()})
    out = ComputeBoundaryScales(
        ["wall"], "L", "U", sources=[["wall", "u"]]
    ).apply_to_domain(gd)
    assert out.global_data["L"].dtype == out.global_data["U"].dtype == torch.float64
    torch.testing.assert_close(
        out.global_data["L"], torch.tensor(3.0**-0.5, dtype=torch.float64)
    )
    torch.testing.assert_close(
        out.global_data["U"], torch.tensor(12.5**0.5, dtype=torch.float64)
    )
    line = _scales_domain(
        {"line": _segments([0.0, 0.25, 1.5, 2.0])}, {"U_inf": torch.tensor([3.0, 4.0])}
    )
    out = ComputeBoundaryScales(
        ["line"], "L", "U", sources=[["line", "u"]]
    ).apply_to_domain(line)
    torch.testing.assert_close(
        out.global_data["L"], torch.tensor((2 / 3) ** 0.5, dtype=torch.float64)
    )
    torch.testing.assert_close(
        out.global_data["U"], torch.tensor(1.0, dtype=torch.float64)
    )
    out = ComputeBoundaryScales(
        ["line"], "L", "U", global_source="U_inf"
    ).apply_to_domain(line)
    torch.testing.assert_close(
        out.global_data["U"], torch.tensor(5.0, dtype=torch.float64)
    )


def test_boundary_scales_refuse_bad_declarations_and_cases():
    """Neither or both speed sources, a source off the named boundaries, an absent boundary, a zero source
    and an existing leaf are refused."""
    with pytest.raises(ValueError, match="exactly one"):
        ComputeBoundaryScales(["wall"], "L", "U")
    with pytest.raises(ValueError, match="exactly one"):
        ComputeBoundaryScales(
            ["wall"], "L", "U", sources=[["wall", "u"]], global_source="U_inf"
        )
    with pytest.raises(ValueError, match="named boundaries"):
        ComputeBoundaryScales(["wall"], "L", "U", sources=[["inlet", "u"]])
    square = _unit_square()
    with pytest.raises(KeyError, match="not found"):
        ComputeBoundaryScales(
            ["wall", "inlet"], "L", "U", sources=[["wall", "u"]]
        ).apply_to_domain(_scales_domain({"wall": square}))
    zero = square.with_data(cell_data={"u": torch.zeros(2, 3)})
    with pytest.raises(ValueError, match="zero everywhere"):
        ComputeBoundaryScales(
            ["wall"], "L", "U", sources=[["wall", "u"]]
        ).apply_to_domain(_scales_domain({"wall": zero}))
    with pytest.raises(ValueError, match="already has"):
        ComputeBoundaryScales(
            ["wall"], "L", "U", sources=[["wall", "u"]]
        ).apply_to_domain(_scales_domain({"wall": square}, {"L": torch.tensor(1.0)}))


def _isla_case(seed: int = 0):
    """A float32 AeroZoo-like case: three boundaries of random triangle soups (the source on ``inlet``, a
    second rank-1 field on ``outlet``), the globals, every reference.* label, and prediction points."""
    g = torch.Generator().manual_seed(seed)
    boundaries = {}
    for k, (name, n) in enumerate((("outlet", 7), ("inlet", 11), ("wall", 13))):
        pts = torch.randn(3 * n, 3, generator=g) + 3.0 * k
        cd = {
            "bc_velocity": torch.randn(n, 3, generator=g)
            + torch.tensor([8.0, 1.0, 0.0])
        }
        boundaries[name] = Mesh(
            points=pts, cells=torch.arange(3 * n).reshape(n, 3), cell_data=cd
        )
    gd = TensorDict(
        {
            "physics": {
                "kinematic_viscosity": torch.tensor(1.37e-5),
                "density": torch.tensor(1.18),
            },
            "reference": {
                "velocity": torch.tensor([9.0, 0.0, 0.0]),
                "length": torch.tensor(2.5),
                "reynolds_number": torch.tensor(1.6e6),
            },
        },
        batch_size=[],
    )
    return DomainMesh(
        interior=Mesh(points=torch.randn(9, 3, generator=g)),
        boundaries=boundaries,
        global_data=gd,
    )


def _isla_transforms(mean=5.9, std=1.4):
    return [
        ComputeBoundaryScales(
            ["inlet", "outlet", "wall"], "L", "U", sources=[["inlet", "bc_velocity"]]
        ),
        ComputeLogReynolds(
            "U", "L", "physics.kinematic_viscosity", mean, std, "log_re", "re_z"
        ),
    ]


def test_boundary_scales_and_log_reynolds_are_isla_s_bitwise():
    """On a float32 case, L and U equal ISLA's own (its _case_scales in a forward pass, exactly as
    _case_inputs computes them), and re_z equals the standardized log Reynolds number ISLA computed in its
    model up to 0.12.0 (its arithmetic copied here: log10(U L / nu) with nu rounded to the meshes' dtype,
    then (. - mean) / std, in float64), bitwise in float64; the network's global scalar is its float32
    rounding, bitwise."""
    from physicsnemo.experimental.models import ISLA

    mean, std = 5.9, 1.4
    d = _isla_case()
    for t in _isla_transforms(mean, std):
        d = t.apply_to_domain(d)
    torch.manual_seed(0)
    model = ISLA(
        output_schema={"p": {"rank": 0, "units": {"U": 2}}},
        boundary_schemas={
            "inlet": {"bc_velocity": {"rank": 1, "units": {"U": 1}, "defines": "U"}},
            "outlet": {},
            "wall": {},
        },
        global_schema={"re_z": {"rank": 0}},
        interior=True,
        n_boundary_tokens=16,
        hidden=16,
        n_layers=1,
        n_slices=4,
    ).eval()
    seen, columns = [], []
    case_scales, global_fields = model._case_scales, model._global_fields
    model._case_scales = lambda *a: seen.append(case_scales(*a)) or seen[-1]
    model._global_fields = lambda *a: columns.append(global_fields(*a)) or columns[-1]
    with torch.no_grad():
        model(
            boundary_meshes={n: d.boundaries[n] for n in d.boundary_names},
            prediction_points=d.interior,
            global_data=d.global_data,
        )
    gd = d.global_data
    assert torch.equal(gd["L"], seen[0]["L"]) and torch.equal(gd["U"], seen[0]["U"])
    nu = gd["physics", "kinematic_viscosity"].to(torch.float32).double().reshape(())
    log_re = torch.log10(seen[0]["U"] * seen[0]["L"] / nu)
    z = (log_re - mean) / std
    assert gd["re_z"].dtype == torch.float64 and torch.equal(gd["re_z"], z)
    assert torch.equal(columns[0][1].reshape(()), z.to(torch.float32))


def test_boundary_scales_and_log_reynolds_read_no_label():
    """Every reference.* leaf scaled, or deleted, leaves L, U, log_re and re_z bitwise as they were."""
    case = _isla_case()

    def run(d):
        for t in _isla_transforms():
            d = t.apply_to_domain(d)
        return {k: d.global_data[k] for k in ("L", "U", "log_re", "re_z")}

    base = run(case)
    labels = list(
        case.global_data["reference"].keys(include_nested=True, leaves_only=True)
    )
    for leaf in labels:
        gd = case.global_data.clone()
        gd["reference", leaf] = gd["reference", leaf] * 7.0 + 1.0
        other = run(
            DomainMesh(
                interior=case.interior, boundaries=case.boundaries, global_data=gd
            )
        )
        assert all(torch.equal(other[k], base[k]) for k in base), leaf
    gd = case.global_data.clone()
    del gd["reference"]
    other = run(
        DomainMesh(interior=case.interior, boundaries=case.boundaries, global_data=gd)
    )
    assert all(torch.equal(other[k], base[k]) for k in base)


### ---------------------------------------------------------------------------
### HarmonizeBoundaryFields
### ---------------------------------------------------------------------------


@pytest.mark.parametrize("slip_faces", [2, 0])
def test_harmonize_gives_every_boundary_the_same_fields(slip_faces):
    """One-hot labels on every boundary, zero-filled copies of the fields a boundary lacks (also on a
    boundary without faces), and the values a boundary carries kept."""
    d = _domain(slip_faces)
    out = HarmonizeBoundaryFields(one_hot=ONE_HOT, fields=FIELDS).apply_to_domain(d)
    for name, bnd in out.boundaries.items():
        n = bnd.n_cells
        for other, label in ONE_HOT.items():
            assert bnd.cell_data[label].shape == (n,)
            assert torch.equal(
                bnd.cell_data[label], torch.full((n,), float(other == name))
            )
        assert bnd.cell_data["prescribed", "dirichlet", "velocity"].shape == (n, 3)
        assert bnd.cell_data["prescribed", "dirichlet", "pressure"].shape == (n,)
    assert out.boundaries["slip"].n_cells == slip_faces
    torch.testing.assert_close(
        out.boundaries["inlet"].cell_data["prescribed", "dirichlet", "velocity"],
        d.boundaries["inlet"].cell_data["prescribed", "dirichlet", "velocity"],
    )
    assert torch.equal(
        out.boundaries["slip"].cell_data["prescribed", "dirichlet", "velocity"],
        torch.zeros(slip_faces, 3),
    )
    ### Fields outside the declaration are untouched.
    assert torch.equal(
        out.boundaries["wall"].cell_data["derived", "y_plus"], torch.ones(3)
    )
    assert torch.equal(
        out.boundaries["inlet"].cell_data["prescribed", "dirichlet", "tke"],
        torch.tensor([0.5, 0.5]),
    )
    assert out.interior is d.interior


def test_harmonize_refuses_a_wrong_rank_and_an_unknown_boundary():
    """A field of the wrong rank and an unknown boundary are refused."""
    d = _domain()
    with pytest.raises(ValueError, match="rank 1"):
        HarmonizeBoundaryFields(
            one_hot=ONE_HOT, fields={"prescribed.dirichlet.tke": 1}
        ).apply_to_domain(d)
    with pytest.raises(KeyError, match="not found"):
        HarmonizeBoundaryFields(
            one_hot={"outlet": "bc_outlet"}, fields={}
        ).apply_to_domain(d)
    with pytest.raises(ValueError, match="distinct"):
        HarmonizeBoundaryFields(one_hot={"a": "bc", "b": "bc"}, fields={})
    ### A bare Mesh has no boundary name: pass-through.
    mesh = Mesh(points=torch.rand(4, 3))
    assert HarmonizeBoundaryFields(one_hot=ONE_HOT, fields=FIELDS)(mesh) is mesh


### ---------------------------------------------------------------------------
### BoundaryFieldToGlobal
### ---------------------------------------------------------------------------


def test_boundary_field_to_global_is_the_measure_weighted_mean():
    """A uniform value comes back exactly; a varying one is weighted by the effective cell measure
    (the reader's Horvitz-Thompson weights included); a vector field gives a (d,) leaf; collisions and
    missing fields are refused; the inlet's uniform turbulence is the use case."""
    d = _domain()
    inlet = d.boundaries["inlet"]
    inlet.cell_data["prescribed", "dirichlet", "omega"] = torch.tensor([1.0, 3.0])
    scale_measures(inlet, torch.tensor([3.0, 1.0]))  # unequal weights on equal areas
    out = BoundaryFieldToGlobal(
        boundary="inlet",
        fields={
            "prescribed.dirichlet.tke": "inlet_tke",
            "prescribed.dirichlet.omega": "inlet_omega",
            "prescribed.dirichlet.velocity": "inlet_velocity",
        },
    ).apply_to_domain(d)
    gd = out.global_data
    assert gd["inlet_tke"].shape == () and gd["inlet_omega"].shape == ()
    torch.testing.assert_close(gd["inlet_tke"], torch.tensor(0.5))
    torch.testing.assert_close(gd["inlet_omega"], torch.tensor((3 * 1.0 + 1 * 3.0) / 4))
    torch.testing.assert_close(gd["inlet_velocity"], torch.tensor([10.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="already has"):
        BoundaryFieldToGlobal(
            "inlet", {"prescribed.dirichlet.tke": "L_ref"}
        ).apply_to_domain(d)
    with pytest.raises(KeyError, match="lacks"):
        BoundaryFieldToGlobal("inlet", {"nope": "x"}).apply_to_domain(d)
    with pytest.raises(KeyError, match="not found"):
        BoundaryFieldToGlobal(
            "outlet", {"prescribed.dirichlet.tke": "x"}
        ).apply_to_domain(d)


def test_boundary_field_to_global_of_an_empty_boundary_is_zero():
    """A boundary without faces gives a zero global."""
    d = _domain(slip_faces=0)
    d.boundaries["slip"].cell_data["q"] = torch.zeros(0)
    out = BoundaryFieldToGlobal("slip", {"q": "slip_q"}).apply_to_domain(d)
    assert torch.equal(out.global_data["slip_q"], torch.tensor(0.0))


### ---------------------------------------------------------------------------
### ConcatGlobalFields
### ---------------------------------------------------------------------------


def test_concat_global_fields_flattens_in_order():
    """The listed globals are flattened and concatenated in the listed order."""
    d = _domain()
    gd = d.global_data.clone()
    gd["log_re_std"] = torch.tensor(0.4)
    gd["turb"] = torch.tensor([1.0, 2.0])
    d = DomainMesh(interior=d.interior, boundaries=d.boundaries, global_data=gd)
    out = ConcatGlobalFields(
        fields=["U_inf", "log_re_std", "turb"], output_field="global_vector"
    ).apply_to_domain(d)
    torch.testing.assert_close(
        out.global_data["global_vector"], torch.tensor([10.0, 0.0, 0.0, 0.4, 1.0, 2.0])
    )
    assert "U_inf" in out.global_data.keys()  # the sources stay
    with pytest.raises(ValueError, match="already has"):
        ConcatGlobalFields(["U_inf"], "L_ref").apply_to_domain(d)
    with pytest.raises(KeyError, match="needs global_data"):
        ConcatGlobalFields(["missing"], "v")(
            Mesh(points=torch.rand(4, 3), global_data=gd)
        )
    ### On a bare Mesh too.
    mesh = ConcatGlobalFields(["L_ref", "U_inf"], "v")(
        Mesh(points=torch.rand(4, 3), global_data=gd)
    )
    torch.testing.assert_close(
        mesh.global_data["v"], torch.tensor([2.0, 10.0, 0.0, 0.0])
    )


### ---------------------------------------------------------------------------
### Turbulence non-dimensionalization recipes
### ---------------------------------------------------------------------------


def test_turbulence_nondim_recipes_round_trip():
    """nu_t / (U L), k / U^2 and omega L / U forward, and inverse_td back; L_ref is required."""
    pd = {
        "nut": torch.tensor([3.0e-4, 6.0e-4]),
        "tke": torch.tensor([2.0, 4.0]),
        "omega": torch.tensor([50.0, 100.0]),
    }
    mesh = Mesh(points=torch.rand(2, 3), point_data=pd, global_data=_global_data())
    fields = {
        "nut": "kinematic_viscosity",
        "tke": "turbulent_kinetic_energy",
        "omega": "specific_dissipation_rate",
    }
    t = NonDimensionalizeByMetadata(fields=fields, association="point_data")
    out = t(mesh)
    torch.testing.assert_close(out.point_data["nut"], pd["nut"] / (10.0 * 2.0))
    torch.testing.assert_close(out.point_data["tke"], pd["tke"] / 100.0)
    torch.testing.assert_close(out.point_data["omega"], pd["omega"] * 2.0 / 10.0)
    torch.testing.assert_close(out.points, mesh.points / 2.0)  # x / L_ref
    q_inf, p_inf, U_inf_mag, rho_inf, T_inf = freestream_scales(mesh.global_data)
    back = t.inverse_td(
        out.point_data,
        fields,
        q_inf,
        p_inf,
        U_inf_mag,
        rho_inf=rho_inf,
        T_inf=T_inf,
        L_ref=mesh.global_data["L_ref"],
    )
    for name in pd:
        torch.testing.assert_close(back[name], pd[name])
    with pytest.raises(ValueError, match="L_ref required"):
        t.inverse_td(out.point_data, fields, q_inf, p_inf, U_inf_mag)
    no_length = Mesh(
        points=torch.rand(2, 3),
        point_data=pd,
        global_data=mesh.global_data.exclude("L_ref"),
    )
    with pytest.raises(ValueError, match="L_ref required"):
        t(no_length)


### ---------------------------------------------------------------------------
### ComputeSDFFromBoundary on a boundary the case does not have
### ---------------------------------------------------------------------------


def test_sdf_of_an_absent_boundary_is_the_far_fill_only_when_allowed():
    """A boundary without faces gives the far fill only when allowed, and is refused otherwise."""
    d = _domain(slip_faces=0)
    with pytest.raises(ValueError, match="no cell connectivity"):
        ComputeSDFFromBoundary(
            "slip", "sdf_slip", None, use_winding_number=False
        ).apply_to_domain(d)
    out = ComputeSDFFromBoundary(
        "slip",
        "sdf_slip",
        "sdf_slip_normals",
        use_winding_number=False,
        allow_empty_boundary=True,
    ).apply_to_domain(d)
    pts = d.interior.points
    diagonal = (pts.amax(0) - pts.amin(0)).norm()
    sdf = out.interior.point_data["sdf_slip"]
    assert sdf.shape == (20, 1)
    torch.testing.assert_close(sdf, diagonal.expand(20, 1))
    assert torch.equal(out.interior.point_data["sdf_slip_normals"], torch.zeros(20, 3))
    ### With faces present the flag changes nothing.
    with_faces = ComputeSDFFromBoundary(
        "slip", "sdf_slip", None, use_winding_number=False, allow_empty_boundary=True
    ).apply_to_domain(_domain(slip_faces=2))
    assert bool((with_faces.interior.point_data["sdf_slip"].abs() < diagonal).all())
