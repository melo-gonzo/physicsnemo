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

"""ISLA's boundary conditions by name (``boundary_schemas``).

Each condition declares its own per-cell fields (an empty schema for a condition
without any); ``forward`` takes any non-empty subset of the declared names in any
order. The symmetry and remeshing contracts hold with every condition carrying its
own fields; a condition's identity is in its parameters (swapping two conditions'
names changes the outputs), not in a tag field; an absent condition is a condition
with no cells; and every condition's parameters receive a gradient in every pass.
"""

import pytest
import torch

from physicsnemo.experimental.models import ISLA
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import EFFECTIVE_MEASURE_KEY
from test.experimental.models.isla._meshes import (
    cloud,
    law,
    orthogonal,
    outputs,
    points_mesh,
    soup,
    transform,
)

D = torch.float64
OUT = {"p": {"rank": 0}, "v": {"rank": 1}, "s": {"rank": 2, "symmetric": True}}
#: An AeroZoo-like set of conditions: different fields of every rank up to 2, and
#: two conditions without any.
SCHEMAS = {
    "inlet": {"u": {"rank": 1}, "k": {"rank": 0}},
    "outlet": {
        "p": {"rank": 0},
        "u_rev": {"rank": 1},
        "T": {"rank": 2, "symmetric": True},
    },
    "no_slip": {},
    "slip": {},
}
SIZES = {"inlet": 40, "outlet": 30, "no_slip": 120, "slip": 50}
#: The cells of each condition sit in their own region, so a condition is a part of
#: one boundary as in a dataset.
OFFSETS = {
    "inlet": (-6.0, 0, 0),
    "outlet": (6.0, 0, 0),
    "no_slip": (0, 0, 0),
    "slip": (0, 4.0, 0),
}


#: The same conditions with ISLA's own scales: the inlet's velocity is the source of U
#: and of the derived direction g, the fields and outputs carry units.
SCALED = {
    "inlet": {
        "u": {"rank": 1, "units": {"U": 1}, "velocity_scale": True},
        "k": {"rank": 0, "units": {"U": 2}},
    },
    "outlet": {
        "p": {"rank": 0, "units": {"U": 2}},
        "u_rev": {"rank": 1, "units": {"U": 1}},
        "T": {"rank": 2, "symmetric": True, "units": {"U": 2}},
    },
    "no_slip": {},
    "slip": {},
}
SCALED_OUT = {
    "p": {"rank": 0, "units": {"U": 2}},
    "v": {"rank": 1, "units": {"U": 1}},
    "s": {"rank": 2, "symmetric": True, "units": {"U": 1, "L": 1}},
}


def _model(
    interior=False, schemas=SCHEMAS, output_schema=OUT, query_independent=False, **kw
):
    torch.manual_seed(0)
    return (
        ISLA(
            output_schema=output_schema,
            boundary_schemas=schemas,
            global_schema={"g": {"rank": 1}, "re": {"rank": 0}},
            interior=interior,
            query_independent=query_independent,
            hidden=32,
            n_layers=2,
            n_slices=8,
            **kw,
        )
        .double()
        .eval()
    )


def _meshes(schemas=SCHEMAS, sizes=SIZES, seed=0) -> dict[str, Mesh]:
    g = torch.Generator().manual_seed(seed + 50)
    out = {}
    for k, (name, n) in enumerate(sizes.items()):
        pts, nrm, w, _ = cloud(n, seed=seed + k)
        data = {
            f: torch.randn(n, *(3,) * s["rank"], dtype=D, generator=g)
            for f, s in schemas[name].items()
        }
        if "T" in data:
            data["T"] = data["T"] + data["T"].transpose(-1, -2)
        out[name] = soup(
            pts + torch.tensor(OFFSETS.get(name, (0, 0, 0)), dtype=D),
            nrm,
            w,
            cell_data=data,
        )
    return out


GD = {"g": torch.tensor([0.6, 0.8, 0.0], dtype=D), "re": torch.tensor(0.4, dtype=D)}
Q = torch.randn(25, 3, dtype=D, generator=torch.Generator().manual_seed(9)) * 3


def _run(m, meshes, gd=GD, q=None):
    with torch.no_grad():
        return outputs(
            m(
                boundary_meshes=meshes,
                prediction_points=q,
                prediction_boundaries=None if q is not None else list(meshes),
                global_data=gd,
            )
        )


def _specs(m, name):
    return m.boundary_schemas[name]


@pytest.mark.parametrize("scaled", [False, True], ids=["unscaled", "scaled"])
@pytest.mark.parametrize("improper", [False, True], ids=["rotation", "reflection"])
@pytest.mark.parametrize("interior", [False, True], ids=["surface", "interior"])
def test_every_condition_s_fields_are_covariant(interior, improper, scaled):
    """Rotations and reflections with a translation: every condition's vector and
    tensor fields move with the geometry, and every output transforms by its rank;
    with ISLA's own scales the derived direction g moves with the source as well."""
    m = _model(
        interior, **(dict(schemas=SCALED, output_schema=SCALED_OUT) if scaled else {})
    )
    assert m.n_global_vectors == 1 + scaled
    meshes = _meshes()
    q = Q if interior else None
    base = _run(m, meshes, q=q)
    R = orthogonal(3, 3, improper)
    t = torch.tensor([1.5, -2.0, 0.7], dtype=D)
    moved = {k: transform(v, R, t, specs=_specs(m, k)) for k, v in meshes.items()}
    out = _run(
        m,
        moved,
        {"g": GD["g"] @ R.T, "re": GD["re"]},
        None if q is None else q @ R.T + t,
    )
    for name, spec in m.output_schema.items():
        assert torch.allclose(out[name], law(base[name], R, spec.rank), atol=1e-10), (
            name
        )


def test_a_condition_s_cells_permute_with_its_outputs():
    """Permuting one condition's cells (and its fields) permutes that condition's
    surface outputs and leaves the others as they are."""
    m = _model()
    meshes = _meshes()
    base = _run(m, meshes)
    inlet = meshes["inlet"]
    perm = torch.randperm(inlet.n_cells, generator=torch.Generator().manual_seed(1))
    permuted = Mesh(
        points=inlet.points, cells=inlet.cells[perm], cell_data=inlet.cell_data[perm]
    )
    out = _run(m, {**meshes, "inlet": permuted})
    ### The tokens are the conditions in canonical order: inlet, no_slip, outlet, slip.
    n = inlet.n_cells
    for k in base:
        assert torch.allclose(out[k][:n], base[k][perm], atol=1e-10)
        assert torch.allclose(out[k][n:], base[k][n:], atol=1e-10)


def _split(mesh: Mesh) -> Mesh:
    """Every cell as two coincident half-measure copies carrying its fields: the
    same discrete boundary measure and the same data, re-represented."""
    data = mesh.cell_data.apply(
        lambda t: t.repeat_interleave(2, 0), batch_size=[2 * mesh.n_cells]
    )
    data[EFFECTIVE_MEASURE_KEY] = (
        mesh.cell_data[EFFECTIVE_MEASURE_KEY].repeat_interleave(2) / 2
    )
    return Mesh(
        points=mesh.points, cells=mesh.cells.repeat_interleave(2, 0), cell_data=data
    )


@pytest.mark.parametrize("name", list(SCHEMAS))
def test_remeshing_one_condition_changes_no_interior_prediction(name):
    """Splitting every cell of one condition into two half-measure copies with its
    fields (the budget above every cell count, so the draw keeps every cell) leaves
    the interior predictions unchanged, for a condition with fields and without."""
    m = _model(interior=True)
    meshes = _meshes()
    base = _run(m, meshes, q=Q)
    out = _run(m, {**meshes, name: _split(meshes[name])}, q=Q)
    for k in base:
        assert torch.allclose(out[k], base[k], atol=1e-10), k


@pytest.mark.parametrize("interior", [False, True], ids=["surface", "interior"])
def test_the_order_of_the_input_mapping_changes_nothing(interior):
    """The tokens are the conditions in canonical (sorted) order whatever the order
    of the mapping: the outputs are bitwise the same."""
    m = _model(interior)
    meshes = _meshes()
    q = Q if interior else None
    base = _run(m, meshes, q=q)
    out = _run(m, dict(reversed(list(meshes.items()))), q=q)
    assert all(torch.equal(out[k], base[k]) for k in base)


@pytest.mark.parametrize("interior", [False, True], ids=["surface", "interior"])
def test_swapping_two_conditions_names_changes_the_outputs(interior):
    """Identity is structural: two conditions without fields read different input
    layers, so the same cells under the other name give other outputs (no tag field
    tells the model which is which)."""
    m = _model(interior)
    meshes = _meshes()
    q = Q if interior else None
    base = _run(m, meshes, q=q)
    swapped = {**meshes, "no_slip": meshes["slip"], "slip": meshes["no_slip"]}
    out = _run(m, swapped, q=q)
    if not interior:
        ### The swapped tokens in canonical order are inlet, the slip cells, outlet,
        ### the no-slip cells: put them back in the order of the base tokens.
        n_in, n_s, n_out, n_ns = (
            SIZES["inlet"],
            SIZES["slip"],
            SIZES["outlet"],
            SIZES["no_slip"],
        )
        order = [*range(n_in), *range(n_in + n_s + n_out, n_in + n_s + n_out + n_ns)]
        order += [*range(n_in + n_s, n_in + n_s + n_out), *range(n_in, n_in + n_s)]
        out = {k: v[order] for k, v in out.items()}
    assert max(float((out[k] - base[k]).abs().max()) for k in base) > 1e-3
    assert not torch.equal(
        m.boundary_conditions["no_slip"].seed.weight,
        m.boundary_conditions["slip"].seed.weight,
    )


@pytest.mark.parametrize("interior", [False, True], ids=["surface", "interior"])
def test_an_absent_condition_is_a_condition_without_cells(interior):
    """Any non-empty subset of the declared names is a valid input: an absent
    condition contributes no tokens, exactly as one given with no cells, and the
    present conditions' tokens are the same cells in the same order."""
    m = _model(interior)
    meshes = _meshes()
    q = Q if interior else None
    subset = {k: meshes[k] for k in ("inlet", "no_slip")}
    out = _run(m, subset, q=q)
    empty = Mesh(points=meshes["slip"].points, cells=meshes["slip"].cells[:0])
    empty_outlet = _meshes(sizes={"outlet": 0})["outlet"]
    assert all(
        torch.equal(
            v, _run(m, {**subset, "slip": empty, "outlet": empty_outlet}, q=q)[k]
        )
        for k, v in out.items()
    )
    n = SIZES["inlet"] + SIZES["no_slip"]
    assert out["p"].shape == ((25,) if interior else (n,))
    full = _run(m, meshes, q=q)
    if interior:
        assert not torch.allclose(out["p"], full["p"], atol=1e-6)


def test_unknown_and_missing_names_are_refused():
    m = _model()
    meshes = _meshes()
    with pytest.raises(
        ValueError,
        match="\\['wall'\\], which are not declared boundary conditions; this model declares \\['inlet', 'no_slip', 'outlet', 'slip'\\]",
    ):
        m(boundary_meshes={**meshes, "wall": meshes["slip"]}, global_data=GD)
    with pytest.raises(ValueError, match="non-empty mapping"):
        m(boundary_meshes={}, global_data=GD)
    ### each mesh carries its own condition's fields
    with pytest.raises(
        ValueError,
        match="boundary_meshes\\['outlet'\\].cell_data is missing the declared fields \\['T', 'p', 'u_rev'\\]",
    ):
        m(boundary_meshes={"outlet": meshes["slip"]}, global_data=GD)


@pytest.mark.parametrize(
    "schemas, match",
    [
        ({}, "non-empty mapping"),
        (None, "non-empty mapping"),
        ({"a.b": {}}, "without '.'"),
        ({"": {}}, "without '.'"),
    ],
)
def test_the_declaration_is_validated(schemas, match):
    with pytest.raises(ValueError, match=match):
        _model(schemas=schemas)


@pytest.mark.parametrize(
    "name", ["keys", "type", "update", "training", "items", "forward"]
)
def test_a_condition_name_that_is_a_module_dict_attribute_is_refused(name):
    """The conditions' parameters live in an nn.ModuleDict keyed by name, so a name
    that is one of its attributes is refused at construction, naming the argument."""
    with pytest.raises(
        ValueError, match=f"boundary_schemas: the condition name '{name}'"
    ):
        _model(schemas={**SCHEMAS, name: {}})


def test_the_declaration_is_canonical_and_per_condition():
    """Names in sorted order whatever the declaration's; each condition's input layer
    reads [1, n.g, its fields' invariants, re]; vector and tensor maps exist only for
    the conditions with such inputs and map onto the widest condition's channels."""
    m = _model(schemas=dict(reversed(list(SCHEMAS.items()))))
    assert m.boundary_names == ("inlet", "no_slip", "outlet", "slip")
    assert list(m.boundary_conditions) == list(m.boundary_names)
    widths = {k: c.seed.in_features for k, c in m.boundary_conditions.items()}
    ### inlet: u (2 frame cosines, norm) and k; outlet: p, u_rev (3), T (3 frame pairs, trace, norm)
    assert widths == {
        "inlet": 1 + 1 + 4 + 1,
        "no_slip": 3,
        "outlet": 1 + 1 + 9 + 1,
        "slip": 3,
    }
    c = m.boundary_conditions
    assert c["no_slip"].vector_map is None and c["slip"].tensor_map is None
    ### outlet: u_rev and T's two derived vectors (T n, T g); inlet: u
    assert m.n_boundary_vecs == 3 and tuple(c["outlet"].vector_map.shape) == (3, 3)
    assert (
        tuple(c["inlet"].vector_map.shape) == (3, 1) and c["inlet"].tensor_map is None
    )
    assert m.n_boundary_tens == 1 and tuple(c["outlet"].tensor_map.shape) == (1, 1)


def test_a_condition_s_fields_and_maps_change_the_outputs():
    """A condition's own fields reach the prediction, through the seed and through
    its vector map into the slice layers."""
    m = _model(interior=True)
    meshes = _meshes()
    base = _run(m, meshes, q=Q)
    inlet = meshes["inlet"]
    data = inlet.cell_data.clone()
    data["u"] = data["u"] * 2
    out = _run(m, {**meshes, "inlet": inlet.with_data(cell_data=data)}, q=Q)
    assert not torch.allclose(out["p"], base["p"], atol=1e-6)
    with torch.no_grad():
        m.boundary_conditions["inlet"].vector_map.mul_(-1.5)
    assert not torch.allclose(_run(m, meshes, q=Q)["p"], base["p"], atol=1e-6)


@pytest.mark.parametrize("query_independent", [False, True], ids=["isla", "isla_dtt"])
def test_every_condition_s_parameters_train_even_when_it_is_absent(query_independent):
    """DDP needs every parameter in every pass: with the slip condition absent and the
    outlet given without cells, every parameter still receives a gradient. ISLA-DTT
    takes no rank-2 inputs, so its outlet declares none."""
    schemas = (
        {**SCHEMAS, "outlet": {"p": {"rank": 0}, "u_rev": {"rank": 1}}}
        if query_independent
        else SCHEMAS
    )
    torch.manual_seed(0)
    m = ISLA(
        output_schema=OUT,
        boundary_schemas=schemas,
        global_schema={"g": {"rank": 1}, "re": {"rank": 0}},
        interior=True,
        query_independent=query_independent,
        hidden=32,
        n_layers=2,
        n_slices=8,
    )
    meshes = {k: v.to(torch.float32) for k, v in _meshes(schemas).items()}
    meshes["outlet"] = _meshes(schemas, sizes={"outlet": 0})["outlet"].to(torch.float32)
    del meshes["slip"]
    gd = {k: v.float() for k, v in GD.items()}
    out = outputs(
        m(
            boundary_meshes=meshes,
            prediction_points=points_mesh(Q.float()),
            global_data=gd,
        )
    )
    sum(v.square().mean() for v in out.values()).backward()
    assert not [n for n, p in m.named_parameters() if p.grad is None]
    assert all(torch.isfinite(v).all() for v in out.values())


def test_a_change_of_condition_subset_compiles_once_per_subset():
    """Under the whole-model compile, cases of one subset with other cell counts
    reuse a graph; another subset is another graph, compiled once, and the first
    subset's graph is reused when its cases come back."""
    torch._dynamo.reset()
    m = _model(interior=True).float()
    compiled = torch.compile(m, backend="eager", fullgraph=True, dynamic=True)
    gd = {k: v.float() for k, v in GD.items()}
    q = Q.float()

    def case(names, extra):
        sizes = {k: SIZES[k] + extra + 7 * i for i, k in enumerate(names)}
        return {
            k: v.to(torch.float32) for k, v in _meshes(sizes=sizes, seed=extra).items()
        }

    with torch.no_grad():
        compiled(boundary_meshes=case(SCHEMAS, 0), prediction_points=q, global_data=gd)
        compiled(
            boundary_meshes=case(["inlet", "outlet", "no_slip"], 0),
            prediction_points=q,
            global_data=gd,
        )
        with torch._dynamo.config.patch(error_on_recompile=True):
            compiled(
                boundary_meshes=case(SCHEMAS, 3), prediction_points=q, global_data=gd
            )
            compiled(
                boundary_meshes=case(["inlet", "outlet", "no_slip"], 5),
                prediction_points=q,
                global_data=gd,
            )
