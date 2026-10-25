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

r"""ISLA brings its own scales (module docstring, *Scales*), in float64.

The length L (the full-resolution boundary's RMS pairwise distance), the velocity U
and direction g of the ``velocity_scale`` sources and the density role: a similarity
of the problem (with its dimensionless global inputs unchanged) leaves every internal
state unchanged and scales the outputs by their units; the condition that
carries the source can have any name; U, g and L are the same for every draw and
under refinement; a model that declares nothing is the release model; the scales
take their hand values and an absent or zero source is refused; ``output_stats`` is
the recipe's NormalizeMeshFields; and a global source (AirfRANS) adds no direction.
"""

import math
from pathlib import Path

import pytest
import torch

from physicsnemo.datapipes.transforms.mesh import NormalizeMeshFields
from physicsnemo.experimental.models import ISLA
from physicsnemo.experimental.models.isla.geometry import (
    cell_second_moments,
    rms_pairwise_distance,
)
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import cell_measures
from test.experimental.models.isla._meshes import (
    cloud,
    law,
    live,
    orthogonal,
    outputs,
    soup,
    transform,
)

D = torch.float64
DATA_DIR = Path(__file__).parent / "data"

#: An AeroZoo-like declaration (DESIGN_v2 section 4): the inlet's velocity is the source.
SCHEMAS = {
    "inlet": {
        "bc_velocity": {"rank": 1, "units": {"U": 1}, "velocity_scale": True},
        "bc_tke": {"rank": 0, "units": {"U": 2}},
        "bc_omega": {"rank": 0, "units": {"U": 1, "L": -1}},
        "bc_gamma": {"rank": 0},
    },
    "outlet": {"bc_pressure": {"rank": 0, "units": {"U": 2, "rho": 1}}},
    "wall": {},
}
GLOBALS = {
    "physics.density": {"rank": 0, "role": "density"},
    "re_z": {"rank": 0},  # a dimensionless global, the recipe's standardized log10 Re
}
OUTPUTS = {
    "velocity": {"rank": 1, "units": {"U": 1}},
    "pressure": {"rank": 0, "units": {"U": 2, "rho": 1}},
    "nut": {"rank": 0, "units": {"U": 1, "L": 1}},
    "omega": {"rank": 0, "units": {"U": 1, "L": -1}},
}
SIZES = {"inlet": 40, "outlet": 30, "wall": 120}
OFFSETS = {
    "inlet": (-6.0, 0.0, 0.0),
    "outlet": (6.0, 0.0, 0.0),
    "wall": (0.0, 0.0, 0.0),
}
RHO, RE_Z = 1.2, 0.3


def _units(declared: dict) -> dict:
    return {name: f.get("units", {}) for name, f in declared.items()}


def _factor(units: dict, s: float, c: float, r: float) -> float:
    """The similarity's factor on a field of these units: L -> s L, U -> c U, rho -> r rho."""
    base = {"L": s, "U": c, "rho": r}
    return math.prod(base[k] ** e for k, e in units.items())


def _model(
    interior=True,
    schemas=SCHEMAS,
    globals_=GLOBALS,
    outputs_=OUTPUTS,
    **kw,
):
    torch.manual_seed(0)
    m = ISLA(
        output_schema=outputs_,
        boundary_schemas=schemas,
        global_schema=globals_,
        interior=interior,
        hidden=32,
        n_layers=2,
        n_slices=8,
        **kw,
    )
    m.geo_kernel = "eager"
    m._tree_read_theta = 0.0  # the exact boundary read
    return live(m.double().eval())


def _case(schemas=SCHEMAS, sizes=SIZES, seed=0):
    """The boundary meshes (the inlet's velocity about 10 m/s along a direction, its k
    and omega positive, the outlet's pressure in Pa) and the globals."""
    g = torch.Generator().manual_seed(seed + 50)
    meshes = {}
    for k, (name, n) in enumerate(sizes.items()):
        pts, nrm, w, _ = cloud(n, seed=seed + k)
        data = {}
        for f, spec in schemas[name].items():
            shape = (n, 3) if spec["rank"] == 1 else (n,)
            data[f] = torch.rand(shape, dtype=D, generator=g) + 0.5
        if "bc_velocity" in data:
            data["bc_velocity"] = (
                10.0 * torch.tensor([0.6, 0.8, 0.0], dtype=D) + data["bc_velocity"]
            )
        meshes[name] = soup(
            pts + torch.tensor(OFFSETS.get(name, (0.0, 0.0, 0.0)), dtype=D),
            nrm,
            w,
            cell_data=data,
        )
    gd = {
        "physics.density": torch.tensor(RHO, dtype=D),
        "re_z": torch.tensor(RE_Z, dtype=D),
    }
    q = torch.randn(25, 3, dtype=D, generator=g) * 3
    return meshes, gd, q


def _similar(m, meshes, gd, q, schemas, s, c, r, R=None):
    """The case under the similarity x -> s R x, every field times its units' factor
    (a vector rotated with the geometry), rho -> r rho and the dimensionless re_z
    unchanged."""
    out = {}
    for name, mesh in meshes.items():
        moved = transform(mesh, R=R, scale=s, specs=m.boundary_schemas[name])
        data = moved.cell_data.clone()
        for f, u in _units(schemas[name]).items():
            data[f] = data[f] * _factor(u, s, c, r)
        out[name] = moved.with_data(cell_data=data)
    gd = {
        "physics.density": gd["physics.density"] * r,
        "re_z": gd["re_z"],
    }
    R = torch.eye(3, dtype=D) if R is None else R
    return out, gd, s * q @ R.T


def _spy(m):
    """Record the case's scales and the head's input of every forward pass."""
    seen = {"scales": [], "h": [], "gauge": []}
    case_scales, forward_tokens = m._case_scales, m._forward_tokens

    def scales(*a, **k):
        out = case_scales(*a, **k)
        seen["scales"].append(out)
        return out

    def tokens(*a, **k):
        seen["gauge"].append(k.get("gauge"))
        return forward_tokens(*a, **k)

    m._case_scales, m._forward_tokens = scales, tokens
    m.norm_out.register_forward_hook(
        lambda mod, i, o: seen["h"].append(o.detach().clone())
    )
    return seen


def _run(m, meshes, gd, q=None):
    with torch.no_grad():
        return outputs(
            m(
                boundary_meshes=meshes,
                prediction_points=q,
                prediction_boundaries=None if q is not None else list(meshes),
                global_data=gd,
            )
        )


@pytest.mark.parametrize("s, c, r", [(3.7, 0.29, 1.0), (0.05, 12.0, 830.0)])
def test_similarity_leaves_the_internals_and_scales_the_outputs(s, c, r):
    """(x, v, rho, k, omega, p) -> (s x, c v, r rho, c^2 k, (c/s) omega, r c^2 p), with
    a rotation and re_z unchanged: L -> s L, U -> c U, g rotates, every internal state is
    unchanged, and every output is scaled by its units (and rotated by its rank)."""
    m = _model()
    seen = _spy(m)
    meshes, gd, q = _case()
    R = orthogonal(3, 4)
    base = _run(m, meshes, gd, q)
    moved = _run(m, *_similar(m, meshes, gd, q, SCHEMAS, s, c, r, R))
    a, b = seen["scales"]
    torch.testing.assert_close(b["L"], s * a["L"], atol=0, rtol=1e-12)
    torch.testing.assert_close(b["U"], c * a["U"], atol=0, rtol=1e-12)
    torch.testing.assert_close(b["rho"], r * a["rho"], atol=0, rtol=1e-12)
    torch.testing.assert_close(b["g"], a["g"] @ R.T, atol=1e-12, rtol=0)
    torch.testing.assert_close(seen["h"][1], seen["h"][0], atol=1e-9, rtol=1e-9)
    for name, spec in m.output_schema.items():
        expected = law(base[name], R, spec.rank) * _factor(
            OUTPUTS[name].get("units", {}), s, c, r
        )
        torch.testing.assert_close(
            moved[name], expected, atol=1e-10 * expected.abs().max(), rtol=1e-9
        )


def test_the_source_s_condition_can_have_any_name():
    """The code names no condition: the same physics with the source on a condition
    named ``inlet`` or ``zz`` (another canonical position) gives the same outputs for
    the same parameters."""
    renamed = {("zz" if k == "inlet" else k): v for k, v in SCHEMAS.items()}
    a, b = _model(), _model(schemas=renamed)
    state = {
        k.replace("boundary_conditions.inlet.", "boundary_conditions.zz."): v
        for k, v in a.state_dict().items()
    }
    b.load_state_dict(state)
    meshes, gd, q = _case()
    out_a = _run(a, meshes, gd, q)
    out_b = _run(
        b, {("zz" if k == "inlet" else k): v for k, v in meshes.items()}, gd, q
    )
    for name in out_a:
        torch.testing.assert_close(out_b[name], out_a[name], atol=1e-10, rtol=1e-10)


def _full_length(meshes) -> torch.Tensor:
    """The exact RMS pairwise distance of the meshes' piecewise-flat cells."""
    names = sorted(meshes)
    return rms_pairwise_distance(
        torch.cat([meshes[k].cell_centroids for k in names])[None],
        torch.cat([cell_measures(meshes[k]) for k in names])[None],
        1e-12,
        torch.cat([cell_second_moments(meshes[k]) for k in names])[None],
    )


def test_scales_are_independent_of_the_draw_and_of_refinement():
    """U, g and L are computed on the full-resolution boundary before the token
    draw: two training draws (different starts) of 60 of the 190 cells, and a
    subdivision of the source condition's cells (each triangle into four, its data
    carried), give the same scales; L is the full mesh's exact one."""
    m = _model(n_boundary_tokens=60).train()
    seen = _spy(m)
    meshes, gd, q = _case()
    for seed in (0, 1):
        torch.manual_seed(seed)
        _run(m, meshes, gd, q)
    _run(
        m,
        {**meshes, "inlet": meshes["inlet"].subdivide(levels=1, filter="linear")},
        gd,
        q,
    )
    for later in seen["scales"][1:]:
        for key in ("L", "U", "g"):
            torch.testing.assert_close(
                later[key], seen["scales"][0][key], atol=1e-13, rtol=1e-13
            )
    for gauge in seen["gauge"]:
        torch.testing.assert_close(gauge, _full_length(meshes), atol=0, rtol=1e-13)
    assert seen["h"][0].shape[1] == 25 and not torch.equal(
        seen["h"][0], seen["h"][1]
    )  # the draws differ


@pytest.mark.parametrize("dim", [2, 3])
def test_the_length_unit_is_exact_under_subdivision(dim):
    """L is the RMS distance between two points of the piecewise-flat boundary, each
    cell with its own second moment: subdividing every cell (segments in two in two
    dimensions, triangles in four in three) at three levels leaves it unchanged to
    roundoff, where the centroids alone converge only as h^2 (a 4 % change here)."""
    from test.experimental.models.isla.test_isla_interior_geometry import (
        _polygon,
        _sphere,
    )

    if dim == 2:
        t = torch.linspace(0, 2 * math.pi, 13, dtype=D)[:-1]
        mesh = _polygon(torch.stack([2.0 * t.cos(), 0.7 * t.sin()], -1))
    else:
        mesh = _sphere(1)
        mesh = Mesh(
            points=mesh.points * torch.tensor([2.0, 1.0, 0.5], dtype=D),
            cells=mesh.cells,
        )
    torch.manual_seed(0)
    m = (
        ISLA(
            output_schema={"p": {"rank": 0}},
            boundary_schemas={"wall": {}},
            interior=True,
            n_spatial_dims=dim,
            hidden=16,
            n_layers=1,
            n_slices=4,
        )
        .double()
        .eval()
    )
    seen = []
    inner = m._forward_tokens
    m._forward_tokens = lambda *a, **k: seen.append(k["gauge"]) or inner(*a, **k)
    q = torch.zeros(3, dim, dtype=D)
    lengths, centroid_only = [], []
    for _ in range(4):
        with torch.no_grad():
            m(boundary_meshes={"wall": mesh}, prediction_points=q)
        lengths.append(seen[-1].reshape(()))
        centroid_only.append(
            rms_pairwise_distance(
                mesh.cell_centroids[None], cell_measures(mesh)[None]
            ).reshape(())
        )
        mesh = mesh.subdivide(levels=1, filter="linear")
    for L in lengths[1:]:
        torch.testing.assert_close(L, lengths[0], atol=0, rtol=1e-12)
    assert abs(float(centroid_only[0] / lengths[0]) - 1) > 1e-3


@pytest.mark.parametrize("config", ["default", "dtt"])
def test_declaring_nothing_is_the_release_model(config):
    """Keys that declare nothing (``units: {}``, ``velocity_scale: false``) leave the
    model the unscaled one: bitwise equal to the release configuration with the same
    parameters, and equal to the release goldens at their tolerance (the goldens are
    bitwise on the recording architecture and differ at the last bits across CPU
    architectures, BLAS reduction order)."""
    from test.experimental.models.isla.test_isla_release import (
        CONFIGS,
        _instantiate_model,
        _make_inputs,
    )

    cfg = dict(CONFIGS[config])
    cfg["output_schema"] = {"p": {"rank": 0, "units": {}}, "v": {"rank": 1}}
    cfg["global_schema"] = {"g": {"rank": 1, "velocity_scale": False}}
    model, release = _instantiate_model(**cfg), _instantiate_model(**CONFIGS[config])
    assert not model._scaled and not model._fp32_head
    inputs = _make_inputs("cpu", query=(config != "default"))
    with torch.no_grad():
        out, same = outputs(model(**inputs)), outputs(release(**inputs))
    ref = torch.load(DATA_DIR / f"isla_{config}_v1.pth", weights_only=True)["output"]
    for k in ref:
        assert torch.equal(out[k], same[k]), k
        torch.testing.assert_close(out[k], ref[k], atol=1e-5, rtol=1e-5)


def test_the_scales_take_their_hand_values():
    """U = sqrt(sum a |v|^2 / sum a) pooled over two sources, g = sum a v / |sum a v|,
    rho the density's; g is the last global direction."""
    schemas = {
        "a": {"v": {"rank": 1, "units": {"U": 1}, "velocity_scale": True}},
        "b": {"w": {"rank": 1, "units": {"U": 1}, "velocity_scale": True}},
    }
    m = _model(
        schemas=schemas, outputs_={"p": {"rank": 0, "units": {"U": 2, "rho": 1}}}
    )
    assert (
        m.n_global_vectors == 1 and m.n_global_scalars == 1
    )  # g, and re_z (the density role is consumed)
    seen = _spy(m)
    meshes, gd, q = _case(schemas=schemas, sizes={"a": 7, "b": 5})
    _run(m, meshes, gd, q)
    sc = seen["scales"][0]
    a = [cell_measures(meshes[k]) for k in "ab"]
    v = [meshes["a"].cell_data["v"], meshes["b"].cell_data["w"]]
    sum_a = a[0].sum() + a[1].sum()
    U = (
        ((a[0] * v[0].square().sum(-1)).sum() + (a[1] * v[1].square().sum(-1)).sum())
        / sum_a
    ).sqrt()
    m1 = (a[0][:, None] * v[0]).sum(0) + (a[1][:, None] * v[1]).sum(0)
    pts = torch.cat([meshes[k].cell_centroids for k in "ab"])
    w = torch.cat(a) / sum_a
    centre = (w[:, None] * pts).sum(0)
    J = torch.cat(
        [cell_second_moments(meshes[k]) for k in "ab"]
    )  # each triangle's own second moment
    L = (2 * (w * ((pts - centre).square().sum(-1) + J)).sum()).sqrt()
    torch.testing.assert_close(sc["L"], L, atol=0, rtol=1e-13)
    torch.testing.assert_close(sc["U"], U, atol=0, rtol=1e-13)
    torch.testing.assert_close(sc["g"][0, 0], m1 / m1.norm(), atol=1e-14, rtol=0)
    torch.testing.assert_close(sc["rho"], torch.tensor(RHO, dtype=D), atol=0, rtol=0)


def test_an_absent_or_zero_source_and_bad_declarations_are_refused():
    m = _model()
    meshes, gd, q = _case()
    with pytest.raises(ValueError, match="absent from this case"):
        _run(m, {k: v for k, v in meshes.items() if k != "inlet"}, gd, q)
    inlet = meshes["inlet"]
    zero = inlet.cell_data.clone()
    zero["bc_velocity"] = torch.zeros_like(zero["bc_velocity"])
    with pytest.raises(ValueError, match="zero everywhere"):
        _run(m, {**meshes, "inlet": inlet.with_data(cell_data=zero)}, gd, q)
    ### two equal cells with opposite velocities: U is defined, g is not
    pts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=D)
    nrm = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=D)
    vel = torch.tensor([[1.0, 2.0, 0.0], [-1.0, -2.0, 0.0]], dtype=D)
    data = {
        "bc_velocity": vel,
        "bc_tke": torch.ones(2, dtype=D),
        "bc_omega": torch.ones(2, dtype=D),
        "bc_gamma": torch.ones(2, dtype=D),
    }
    with pytest.raises(ValueError, match="cancel"):
        _run(
            m,
            {**meshes, "inlet": soup(pts, nrm, torch.ones(2, dtype=D), cell_data=data)},
            gd,
            q,
        )
    with pytest.raises(ValueError, match="density"):
        _run(m, meshes, {**gd, "physics.density": torch.tensor(0.0, dtype=D)}, q)
    with pytest.raises(NotImplementedError, match="pass the meshes to forward"):
        m._forward_tensors(
            points=torch.zeros(1, 4, 3, dtype=D),
            normals=torch.ones(1, 4, 3, dtype=D),
            condition="wall",
        )
    bad = [
        (
            dict(globals_={**GLOBALS, "U_inf": {"rank": 1, "velocity_scale": True}}),
            "not both",
        ),
        (
            dict(
                schemas={
                    **SCHEMAS,
                    "inlet": {"bc_velocity": {"rank": 1, "units": {"U": 1}}},
                }
            ),
            "no field is a velocity_scale",
        ),
        (dict(globals_={"re_z": GLOBALS["re_z"]}), "role 'density'"),
        (dict(outputs_={"p": {"rank": 0, "units": {"T": 1}}}), "base scales"),
        (dict(outputs_={"p": {"rank": 0, "velocity_scale": True}}), "an output cannot"),
        (
            dict(
                schemas={**SCHEMAS, "wall": {"k": {"rank": 0, "velocity_scale": True}}}
            ),
            "rank-1 field",
        ),
        (
            dict(schemas={**SCHEMAS, "wall": {"k": {"rank": 0, "role": "density"}}}),
            "rank-0 global",
        ),
        (
            dict(globals_={**GLOBALS, "rho2": {"rank": 0, "role": "density"}}),
            "two fields",
        ),
        (
            dict(
                globals_={
                    **GLOBALS,
                    "physics.density": {
                        "rank": 0,
                        "role": "density",
                        "units": {"U": 1},
                    },
                }
            ),
            "has the units",
        ),
        (
            dict(output_stats={"velocity": {"mean": [1.0, 0.0, 0.0], "std": 2.0}}),
            "must be 0",
        ),
        (
            dict(output_stats={"other": {"mean": 0.0, "std": 2.0}}),
            "not a declared output",
        ),
        (dict(outputs_={"p": {"rank": 0, "scale": 0.0}}), "positive finite"),
        (
            dict(
                globals_={
                    **GLOBALS,
                    "physics.density": {"rank": 0, "role": "density", "scale": 2.0},
                }
            ),
            "takes no scale",
        ),
        (
            dict(globals_={**GLOBALS, "d": {"rank": 1, "units": {"L": 1}}}),
            "direction only",
        ),
    ]
    for kw, match in bad:
        with pytest.raises((ValueError, TypeError), match=match):
            _model(**kw)


def test_output_stats_are_the_dataset_s_normalize_mesh_fields():
    """With the dataset's NormalizeMeshFields stats the model returns its physical
    outputs standardized by them (the targets' space, vector mean 0), the inverse gives
    the physical outputs back, and the checkpoint arguments keep the merged declaration."""
    norm = NormalizeMeshFields(
        association="point_data",
        fields={
            "velocity": {"type": "vector", "mean": [0, 0, 0], "std": 11.6},
            "pressure": {"type": "scalar", "mean": 3.0, "std": 1129.0},
            "nut": {"type": "scalar", "mean": 0, "std": 0.61},
        },
    )
    physical, standardized = _model(), _model(output_stats=norm.stats)
    meshes, gd, q = _case()
    y = _run(physical, meshes, gd, q)
    ystd = _run(standardized, meshes, gd, q)
    expected = norm(Mesh(points=q, point_data=y)).point_data
    for name in y:
        torch.testing.assert_close(
            ystd[name].to(expected[name].dtype), expected[name], atol=1e-6, rtol=1e-5
        )
    back = norm.inverse_td(
        Mesh(points=q, point_data={k: v.float() for k, v in ystd.items()}).point_data
    )
    for name in y:
        torch.testing.assert_close(
            back[name].double(), y[name], atol=1e-6 * y[name].abs().max(), rtol=1e-5
        )
    args = standardized._args["__args__"]
    assert args["output_stats"]["pressure"] == {"mean": 3.0, "std": 1129.0}
    assert args["boundary_schemas"]["inlet"]["bc_velocity"] == {
        "rank": 1,
        "units": {"U": 1},
        "velocity_scale": True,
    }
    assert args["global_schema"]["physics.density"] == {"rank": 0, "role": "density"}
    rebuilt = ISLA(**args)
    assert (
        rebuilt._boundary_sources == [("inlet", "bc_velocity")]
        and rebuilt.output_stats == standardized.output_stats
    )


def test_a_global_source_adds_no_direction():
    """AirfRANS: the global U_inf is the source, so U = |U_inf| and g is U_inf's own
    direction (K unchanged); x -> s x, U_inf -> c U_inf with re_z unchanged leaves the
    dimensionless outputs (v / U, Cp) unchanged and scales nut by s c."""
    globals_ = {
        "U_inf": {"rank": 1, "units": {"U": 1}, "velocity_scale": True},
        "re_z": {"rank": 0},
    }
    outs = {
        "velocity": {"rank": 1},
        "pressure": {"rank": 0},
        "nut": {"rank": 0, "units": {"U": 1, "L": 1}},
    }
    torch.manual_seed(0)
    m = (
        ISLA(
            output_schema=outs,
            boundary_schemas={"airfoil": {}},
            global_schema=globals_,
            interior=True,
            n_spatial_dims=2,
            hidden=32,
            n_layers=2,
            n_slices=8,
        )
        .double()
        .eval()
    )
    assert m.n_global_vectors == 1 and m.n_global_scalars == 1
    seen = _spy(m)
    pts, nrm, w, _ = cloud(80, d=2, seed=3)
    mesh = soup(pts, nrm, w)
    q = torch.randn(20, 2, dtype=D, generator=torch.Generator().manual_seed(1)) * 2
    u_inf = torch.tensor([40.0, 3.0], dtype=D)
    gd = {"U_inf": u_inf, "re_z": torch.tensor(RE_Z, dtype=D)}
    base = _run(m, {"airfoil": mesh}, gd, q)
    torch.testing.assert_close(seen["scales"][0]["U"], u_inf.norm(), atol=0, rtol=1e-15)
    assert seen["scales"][0]["g"] is None
    s, c = 0.3, 2.5
    moved = _run(
        m,
        {"airfoil": transform(mesh, scale=s)},
        {"U_inf": c * u_inf, "re_z": gd["re_z"]},
        s * q,
    )
    torch.testing.assert_close(
        moved["velocity"], base["velocity"], atol=1e-10, rtol=1e-9
    )
    torch.testing.assert_close(
        moved["pressure"], base["pressure"], atol=1e-10, rtol=1e-9
    )
    torch.testing.assert_close(
        moved["nut"],
        s * c * base["nut"],
        atol=1e-10 * base["nut"].abs().max(),
        rtol=1e-9,
    )


def test_an_output_s_scale_multiplies_the_head_s_prediction():
    """``scale: c`` is the typical size of the dimensionless output: the same
    parameters give c times the output, before output_stats, the vector covariant as
    before; the checkpoint arguments keep it with the declaration."""
    cs = (0.5, 0.2, 1e-4, 3e3)
    scaled = {name: {**f, "scale": c} for (name, f), c in zip(OUTPUTS.items(), cs)}
    a = _model()
    b = _model(outputs_=scaled, output_stats={"pressure": {"mean": 1.0, "std": 4.0}})
    b.load_state_dict(a.state_dict())
    meshes, gd, q = _case()
    ya, yb = _run(a, meshes, gd, q), _run(b, meshes, gd, q)
    for name, c in zip(OUTPUTS, cs):
        expected = c * ya[name] if name != "pressure" else (c * ya[name] - 1.0) / 4.0
        torch.testing.assert_close(
            yb[name], expected, atol=1e-12 * expected.abs().max(), rtol=1e-12
        )
    assert b._args["__args__"]["output_schema"]["nut"] == {
        "rank": 0,
        "units": {"U": 1, "L": 1},
        "scale": 1e-4,
    }


def test_an_input_s_scale_divides_it_after_its_units():
    """``scale: c`` on an input is the typical size of its dimensionless value: the
    model reads x / (c prod scale^exp), the same as the unscaled declaration given
    x / c, for a boundary scalar, a boundary vector and a rank-0 global."""
    scaled = {
        **SCHEMAS,
        "inlet": {
            **SCHEMAS["inlet"],
            "bc_omega": {**SCHEMAS["inlet"]["bc_omega"], "scale": 400.0},
            "bc_gamma": {"rank": 0, "scale": 0.5},
        },
        "outlet": {"bc_pressure": {**SCHEMAS["outlet"]["bc_pressure"], "scale": 0.1}},
    }
    globals_ = {**GLOBALS, "phi": {"rank": 0, "units": {"U": 1}, "scale": 3.0}}
    plain_globals = {**GLOBALS, "phi": {"rank": 0, "units": {"U": 1}}}
    a, b = _model(globals_=plain_globals), _model(schemas=scaled, globals_=globals_)
    b.load_state_dict(a.state_dict())
    meshes, gd, q = _case()
    gd = {**gd, "phi": torch.tensor(12.0, dtype=D)}
    divided = {}
    for name, mesh in meshes.items():
        data = mesh.cell_data.clone()
        for f, spec in scaled[name].items():
            if "scale" in spec:
                data[f] = data[f] / spec["scale"]
        divided[name] = mesh.with_data(cell_data=data)
    ya = _run(a, divided, {**gd, "phi": gd["phi"] / 3.0}, q)
    yb = _run(b, meshes, gd, q)
    for name in ya:
        torch.testing.assert_close(
            yb[name], ya[name], atol=1e-12 * ya[name].abs().max(), rtol=1e-11
        )
    assert b._args["__args__"]["boundary_schemas"]["inlet"]["bc_gamma"] == {
        "rank": 0,
        "scale": 0.5,
    }
