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

"""Contract tests for ISLA (Invariant Slice Attention): every guarantee is exact by construction.

The model under test declares one scalar output ``p`` and one vector output ``v``
and one global direction ``g``. The boundary is a ``Mesh[2, 3]`` built by
:func:`soup` around a prescribed cloud (its cell centroids, normals and effective
measures are the cloud's points, normals and weights); ``forward`` returns a
point-cloud Mesh with ``point_data["p"]`` of shape ``(N,)`` and ``point_data["v"]``
of shape ``(N, 3)``.
"""

import pytest
import torch

from physicsnemo.experimental.models import ISLA
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import EFFECTIVE_MEASURE_KEY
from test.experimental.models.isla._meshes import (
    cloud,
    orthogonal,
    outputs,
    points_mesh,
    refine,
    same,
    soup,
    transform,
)

# The frame is origin-free (positions enter only as point-to-anchor differences) with
# one data-derived length unit, the measure-weighted RMS pairwise distance.

PV = {"p": {"rank": 0}, "v": {"rank": 1}}
G = {"g": {"rank": 1}}
D = torch.float64


#: One boundary condition without fields, the cloud's.
WALL = {"wall": {}}


def _isla(**kw):
    kw.setdefault("boundary_schemas", WALL)
    return ISLA(output_schema=PV, global_schema=G, **kw)


def _rows(out, sl):
    """Slice the point axis of every output."""
    return {k: v[sl] for k, v in outputs(out).items()}


def _assert_rotated(rot, base, q, atol=1e-10):
    rot, base = outputs(rot), outputs(base)
    assert torch.allclose(rot["p"], base["p"], atol=atol)
    assert torch.allclose(rot["v"], base["v"] @ q.T, atol=atol)


def _rotation(seed=0):
    return orthogonal(3, seed)


def _run(m, mesh, drv, q=None, **kw):
    with torch.no_grad():
        return m(
            boundary_meshes={"wall": mesh},
            prediction_points=q,
            global_data={"g": drv},
            **kw,
        )


@pytest.fixture
def setup():
    torch.manual_seed(0)
    m = _isla(hidden=64, n_layers=3, n_slices=32).double().eval()
    ### Anisotropic cloud: generic for vehicle geometry; an isotropic cloud is the
    ### degenerate corner.
    pts, nrm, w, drv = cloud(500)
    mesh = soup(pts, nrm, w)
    return m, mesh, drv, _run(m, mesh, drv)


def test_rotation_equivariance(setup):
    m, mesh, drv, base = setup
    q = _rotation()
    _assert_rotated(_run(m, transform(mesh, q), drv @ q.T), base, q)


def test_rotation_equivariance_with_scales():
    """With ISLA's own scales (a global velocity source, a vector output in its
    units) the output still rotates with the geometry and the source."""
    torch.manual_seed(0)
    m = (
        ISLA(
            output_schema={
                "p": {"rank": 0, "units": {"U": 2}},
                "v": {"rank": 1, "units": {"U": 1}},
            },
            boundary_schemas=WALL,
            global_schema={"g": {"rank": 1, "units": {"U": 1}, "velocity_scale": True}},
            hidden=32,
            n_layers=2,
            n_slices=8,
        )
        .double()
        .eval()
    )
    pts, nrm, w, drv = cloud(200)
    mesh = soup(pts, nrm, w)
    q = _rotation()
    base = _run(m, mesh, 7.0 * drv)
    _assert_rotated(_run(m, transform(mesh, q), 7.0 * drv @ q.T), base, q, atol=1e-9)


def test_translation_invariance(setup):
    m, mesh, drv, base = setup
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=D)
    assert same(_run(m, transform(mesh, shift=shift), drv), base, atol=1e-10)


@pytest.mark.parametrize("k", [0.5, 2.0, 4.0])
def test_global_vector_magnitude_invariance(setup, k):
    """Rank-1 global inputs enter as unit directions: rescaling one leaves every
    output unchanged (a physically meaningful magnitude is a rank-0 global input)."""
    m, mesh, drv, base = setup
    assert same(_run(m, mesh, drv * k), base, atol=1e-10)


def test_measure_scale_invariance(setup):
    m, mesh, drv, base = setup
    w = mesh.cell_data[EFFECTIVE_MEASURE_KEY]
    scaled = mesh.with_data(cell_data={EFFECTIVE_MEASURE_KEY: w * 137.0})
    assert same(_run(m, scaled, drv), base, atol=1e-9)


def test_token_permutation_equivariance(setup):
    """Reordering the boundary cells permutes the outputs and changes nothing else."""
    m, mesh, drv, base = setup
    perm = torch.randperm(mesh.n_cells, generator=torch.Generator().manual_seed(3))
    permuted = Mesh(
        points=mesh.points,
        cells=mesh.cells[perm],
        cell_data={EFFECTIVE_MEASURE_KEY: mesh.cell_data[EFFECTIVE_MEASURE_KEY][perm]},
    )
    out = _run(m, permuted, drv)
    assert torch.allclose(out.points, base.points[perm])
    assert same(outputs(out), _rows(base, perm), atol=1e-10)


def test_no_effective_measures_is_the_geometric_quadrature(setup):
    """Without ``_effective_measure`` the cells' geometric measures are the weights;
    a soup without prescribed measures has congruent cells, so this is the uniform
    quadrature: the same output as any constant measure vector."""
    m, mesh, drv, base = setup
    mesh = soup(mesh.cell_centroids, mesh.cell_normals)
    none = _run(m, mesh, drv)
    const = _run(
        m,
        mesh.with_data(
            cell_data={
                EFFECTIVE_MEASURE_KEY: torch.full((mesh.n_cells,), 0.37, dtype=D)
            }
        ),
        drv,
    )
    assert same(none, const, atol=1e-10)
    assert not same(none, base, atol=1e-3)


def test_boundary_scalar_contracts():
    """Two rank-0 boundary fields (one leaf per scalar) in ``cell_data`` ride along
    under a rotation."""
    torch.manual_seed(0)
    m = (
        _isla(
            hidden=64,
            n_layers=2,
            n_slices=16,
            boundary_schemas={"wall": {"bc_a": {"rank": 0}, "bc_b": {"rank": 0}}},
        )
        .double()
        .eval()
    )
    pts, nrm, w, drv = cloud(300)
    bs = {"bc_a": torch.randn(300, dtype=D), "bc_b": torch.randn(300, dtype=D)}
    mesh = soup(pts, nrm, w, cell_data=bs)
    base = _run(m, mesh, drv)
    q = _rotation()
    _assert_rotated(
        _run(m, transform(mesh, q, specs=m.boundary_schemas["wall"]), drv @ q.T),
        base,
        q,
    )
    assert (
        m.boundary_conditions["wall"].seed.in_features == 1 + 1 + 2
    )  # constant, n.g, the two fields


def test_head_runs_under_bf16_autocast():
    """Mixed-precision smoke: the head must survive bf16 autocast (the geometry is
    kept in the input precision by construction)."""
    torch.manual_seed(0)
    m = _isla(hidden=32, n_layers=1, n_slices=8)
    pts, nrm, w, drv = cloud(128, dtype=torch.float32)
    mesh = soup(pts, nrm, w)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = outputs(m(boundary_meshes={"wall": mesh}, global_data={"g": drv}))
    sum(o.float().sum() for o in out.values()).backward()
    assert all(torch.isfinite(o.float()).all() for o in out.values())


@pytest.mark.parametrize("interior", [False, True], ids=["surface", "interior"])
def test_every_parameter_receives_a_gradient(interior):
    """DDP requires every parameter to take part in the loss. A surface model owns
    no prediction-point parameters and an interior model uses all of its own, so
    in both tasks every parameter receives a gradient."""
    torch.manual_seed(0)
    m = _isla(hidden=32, n_layers=2, n_slices=8, interior=interior)
    pts, nrm, w, drv = cloud(128, dtype=torch.float32)
    q = torch.randn(20, 3) if interior else None
    out = outputs(
        m(
            boundary_meshes={"wall": soup(pts, nrm, w)},
            prediction_points=q,
            global_data={"g": drv},
        )
    )
    sum(o.sum() for o in out.values()).backward()
    unused = [n for n, p in m.named_parameters() if p.grad is None]
    assert not unused, unused
    assert (m.qt is not None) == interior
    assert not any(n.startswith("qt") for n, _ in m.named_parameters()) or interior


def test_geo_pool_then_project_is_exact():
    """Pooling the relational invariants over slices and then projecting is
    exactly the old project-then-pool (the projection is affine and the slice
    mix is a softmax over slices, so the bias passes through unchanged). This
    is the identity behind the memory saving: the saved activation is
    (B, N, geo) instead of (B, N, S, hidden/2)."""
    from physicsnemo.experimental.models.isla.blocks import _SliceBlock

    torch.manual_seed(0)
    blk = _SliceBlock(64, 32).double()
    geo = torch.randn(2, 50, 32, blk.n_geo, dtype=D)
    mix = torch.softmax(torch.randn(2, 50, 32, dtype=D), dim=-1)
    old = torch.einsum("bns,bnsg->bng", mix, blk.geo_feat(geo))
    new = blk.geo_feat(torch.einsum("bns,bnsg->bng", mix, geo))
    assert torch.allclose(old, new, atol=1e-12, rtol=0.0)


def test_interior_task_contracts():
    """The interior task (prediction points as interacting tokens): exactly
    rotation/translation-covariant, finite, and -- by design -- NOT query-set
    independent; every parameter receives a gradient."""
    torch.manual_seed(0)
    n, nq = 400, 150
    pts, nrm, w, drv = cloud(n)
    mesh = soup(pts, nrm, w)
    qpts = torch.randn(nq, 3, dtype=D) * 4.0
    m = _isla(hidden=64, n_layers=2, n_slices=16, interior=True).double().eval()
    q = _rotation()
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=D)
    base = _run(m, mesh, drv, qpts)
    rot = _run(m, transform(mesh, q, shift), drv @ q.T, qpts @ q.T + shift)
    sub = _run(m, mesh, drv, qpts[:50])
    assert base.point_data["p"].shape == (nq,) and base.point_data["v"].shape == (nq, 3)
    assert base.n_cells == 0 and torch.equal(base.points, qpts)
    assert all(torch.isfinite(o).all() for o in outputs(base).values())
    _assert_rotated(rot, base, q)
    # interacting mode: predictions at shared queries depend on the query set
    assert not same(outputs(sub), _rows(base, slice(0, 50)), atol=1e-6)
    # every parameter receives a gradient (DDP safety)
    m.train()
    out = outputs(
        m(
            boundary_meshes={"wall": mesh},
            prediction_points=qpts,
            global_data={"g": drv},
        )
    )
    sum(o.square().mean() for o in out.values()).backward()
    missing = [k for k, p in m.named_parameters() if p.grad is None]
    assert not missing, missing


def test_query_independent_interior_contracts():
    """query_independent=True (ISLA-DTT, here with its boundary read at its zero
    initialization; test_isla_tree_read.py exercises the read): the prediction points
    read every block's slices and never write into them, so a prediction is exactly
    independent of the other points of the request (alone, a prefix, any order, any
    chunking), while
    rotation/translation covariance, the similarity gauge and boundary-refinement
    invariance hold as for interacting tokens; there is no routing-mass parameter,
    every parameter receives a gradient, and bf16 autocast stays finite."""
    torch.manual_seed(0)
    pts, nrm, w, drv = cloud(400)
    mesh = soup(pts, nrm, w)
    qpts = torch.randn(150, 3, dtype=D) * 4.0
    m = (
        _isla(hidden=64, n_layers=2, n_slices=16, interior=True, query_independent=True)
        .double()
        .eval()
    )
    assert m.qt.logw is None and not any(
        n == "qt.logw" for n, _ in m.named_parameters()
    )
    base = _run(m, mesh, drv, qpts)
    perm = torch.randperm(150, generator=torch.Generator().manual_seed(1))
    assert same(_run(m, mesh, drv, qpts[:50]), _rows(base, slice(0, 50)), atol=1e-12)
    assert same(_run(m, mesh, drv, qpts[7:8]), _rows(base, slice(7, 8)), atol=1e-12)
    assert same(_run(m, mesh, drv, qpts[perm]), _rows(base, perm), atol=1e-12)
    chunks = [outputs(_run(m, mesh, drv, c)) for c in qpts.split(40)]
    assert same(
        {k: torch.cat([c[k] for c in chunks]) for k in chunks[0]}, base, atol=1e-12
    )
    ### the boundary still determines the prediction
    assert not same(_run(m, soup(pts * 1.1, nrm, w), drv, qpts), base, atol=1e-6)
    q = _rotation()
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=D)
    _assert_rotated(
        _run(m, transform(mesh, q, shift), drv @ q.T, qpts @ q.T + shift), base, q
    )
    k = 2.7
    assert same(
        _run(m, transform(mesh, shift=shift, scale=k), drv, k * qpts + shift),
        base,
        atol=1e-10,
        rtol=0.0,
    )
    assert same(_run(m, refine(mesh), drv, qpts), base, atol=1e-10, rtol=0.0)
    m.train()
    out = outputs(
        m(
            boundary_meshes={"wall": mesh},
            prediction_points=qpts,
            global_data={"g": drv},
        )
    )
    sum(o.square().mean() for o in out.values()).backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, missing
    m32 = _isla(
        hidden=32, n_layers=2, n_slices=8, interior=True, query_independent=True
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = outputs(
            m32(
                boundary_meshes={"wall": soup(pts.float(), nrm.float(), w.float())},
                prediction_points=qpts.float(),
                global_data={"g": drv.float()},
            )
        )
    sum(o.float().sum() for o in out.values()).backward()
    assert all(torch.isfinite(o.float()).all() for o in out.values())
    assert all(
        torch.isfinite(p.grad).all() for p in m32.parameters() if p.grad is not None
    )
    with pytest.raises(ValueError, match="interior=True"):
        _isla(hidden=32, n_layers=1, n_slices=8, query_independent=True)


def test_query_mass_refinement_invariance():
    """Splitting every boundary cell into two coincident half-measure copies leaves
    the discrete boundary measure unchanged (positions, normals, total area, every
    integral). The prediction-point tokens' weight is a fraction of the total
    boundary measure, so the output is exactly unchanged and the geometric-scale
    contract still holds."""
    pts, nrm, w, drv = cloud(60, seed=42)
    mesh = soup(pts, nrm, w)
    q = pts[:17] * 1.3
    qm = points_mesh(q)
    torch.manual_seed(0)
    m = _isla(hidden=32, n_layers=2, n_slices=8, interior=True).double().eval()
    s = 3.7
    qm_s = points_mesh(q * s)
    a = _run(m, mesh, drv, qm)
    b = _run(m, refine(mesh), drv, qm)
    c = _run(m, transform(mesh, scale=s), drv, qm_s)
    assert same(b, a, atol=1e-10, rtol=0.0)
    assert same(c, a, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("interior", [False, True], ids=["surface", "interior"])
def test_geometric_scale_equivariance(interior):
    """For a model without scales: a similarity of the geometry (points x k + shift,
    areas x k^2) leaves the output exactly unchanged, for the boundary alone and with
    prediction points that are moved along. (With declared scales a geometry scale
    at a fixed viscosity changes the Reynolds number, as it should: the similarity
    with every scale is test_isla_scales.py's.)"""
    torch.manual_seed(0)
    pts, nrm, w, drv = cloud(400)
    mesh = soup(pts, nrm, w)
    m = _isla(hidden=64, n_layers=2, n_slices=16, interior=interior).double().eval()
    assert not m._scaled
    k = 2.7
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=D)
    q = pts[:50] * 1.5 if interior else None
    a = _run(m, mesh, drv, q)
    b = _run(
        m,
        transform(mesh, shift=shift, scale=k),
        drv,
        k * q + shift if interior else None,
    )
    assert same(b, a, atol=1e-10, rtol=0.0)


def test_boundary_scalars_reach_the_interior_prediction():
    """Boundary inputs are per-boundary-cell data: prediction-point tokens carry
    zeros in the channel, and the interior prediction depends on the boundary data."""
    torch.manual_seed(0)
    pts, nrm, w, drv = cloud(60)
    bs = {"bc_a": torch.randn(60, dtype=D), "bc_b": torch.randn(60, dtype=D)}
    m = (
        _isla(
            hidden=32,
            n_layers=2,
            n_slices=8,
            interior=True,
            boundary_schemas={"wall": {"bc_a": {"rank": 0}, "bc_b": {"rank": 0}}},
        )
        .double()
        .eval()
    )
    q = pts[:17] * 0.5
    out = _run(m, soup(pts, nrm, w, cell_data=bs), drv, q)
    out2 = _run(
        m, soup(pts, nrm, w, cell_data={k: 2 * v for k, v in bs.items()}), drv, q
    )
    assert out.point_data["p"].shape == (17,) and out.point_data["v"].shape == (17, 3)
    assert all(torch.isfinite(o).all() for o in outputs(out).values())
    assert not same(out, out2, atol=1e-6)


def _biased_poisson_subsample(pts, w, n_expected, bias, generator):
    """10:1 Poisson subsample with exact Horvitz-Thompson weights, mirroring the
    recipe's PoissonBiasedSubsampleMesh (bias toward x below the median)."""
    x = pts[:, 0]
    b = torch.where(x < x.median(), torch.full_like(x, bias), torch.ones_like(x))
    pi = (n_expected / b.sum() * b).clamp(max=1.0)
    keep = torch.rand(x.shape[0], dtype=pi.dtype, generator=generator) < pi
    idx = keep.nonzero(as_tuple=True)[0]
    return idx, w[idx] / pi[idx]


def _frame_cloud(n=300, seed=0):
    pts, nrm, w, drv = cloud(n, seed=seed)
    return soup(pts + 5.0, nrm, w), drv


def test_frame_contracts():
    """The origin-free frame: exact translation invariance without centering
    (fp64), rotation equivariance, and the feature widths (one seed invariant
    n.g, six relational invariants in the slice blocks)."""
    mesh, drv = _frame_cloud()
    shift = torch.tensor([300.0, -70.0, 1100.0], dtype=D)
    torch.manual_seed(1)
    m = _isla(hidden=64, n_layers=2, n_slices=16).double().eval()
    assert (
        m.boundary_conditions["wall"].seed.in_features == 1 + 1
    )  # the constant and n.g
    assert m.blocks[0].geo_logit.in_features == 6
    base = _run(m, mesh, drv)
    assert all(torch.isfinite(o).all() for o in outputs(base).values())
    ### the cell normals are recomputed from the shifted vertices (a 1e3 shift on
    ### 0.2 cells costs ~1e-13 in the normals), hence 1e-10 rather than 1e-12
    assert same(_run(m, transform(mesh, shift=shift), drv), base, atol=1e-10)
    q = _rotation()
    _assert_rotated(_run(m, transform(mesh, q), drv @ q.T), base, q)


def test_frame_is_sampling_consistent():
    """The discriminating contract of the origin-free frame. Under a 10:1 biased
    Poisson subsample with exact HT weights, a model whose frame carries no
    sample statistic moves its predictions at fixed prediction points between the
    uniform and the biased draw by no more than the measure noise -- the same
    order as between two independent uniform draws."""
    pts, nrm, w, drv = cloud(100000)
    q = pts[:40] * 1.2
    torch.manual_seed(1)
    m = _isla(hidden=32, n_layers=2, n_slices=8, interior=True).double().eval()
    n_sub = 10000
    d_bias, d_noise = [], []

    def _flat(o):
        return torch.cat([o.point_data["p"][..., None], o.point_data["v"]], dim=-1)

    def _draw(bias, g):
        idx, w_ht = _biased_poisson_subsample(pts, w, n_sub, bias, g)
        return _run(m, soup(pts[idx], nrm[idx], w_ht), drv, q)

    for seed in range(4):
        g = torch.Generator().manual_seed(100 + seed)
        o_u, o_u2, o_b = _draw(1.0, g), _draw(1.0, g), _draw(10.0, g)
        d_bias.append(float((_flat(o_b) - _flat(o_u)).norm()))
        d_noise.append(float((_flat(o_u2) - _flat(o_u)).norm()))
    ### measure-noise level: the biased draw's HT estimates have a larger variance
    ### than a uniform draw's (10x fewer points in the undersampled half), so the
    ### bias-vs-uniform difference may exceed the uniform-vs-uniform one by a
    ### bounded factor -- not the order of magnitude a centroid shift produces.
    ratio = sum(d_bias) / sum(d_noise)
    assert ratio < 4.0, (ratio, d_bias, d_noise)


def test_defaults_are_the_reference_configuration():
    """The class defaults: a surface task, the automatic geometry kernel, one
    seed invariant n.g and six relational invariants; an interior model adds the
    prediction-point embedding over the built-in distance pair."""
    m = _isla(hidden=32, n_layers=2, n_slices=8)
    assert m.interior is False and m.qt is None
    assert m.geo_kernel == "auto"
    assert m.boundary_conditions["wall"].seed.in_features == 1 + 1
    assert m.blocks[0].geo_logit.in_features == 6
    mi = _isla(hidden=32, n_layers=2, n_slices=8, interior=True)
    assert mi.qt.seed.in_features == 1 + 2  # n.g and the built-in distance pair


def test_constructor_and_forward_are_keyword_only():
    """Every constructor argument and every forward input is keyword-only
    (inputs read as the recipe's forward_kwargs mapping and models can be
    hot-swapped without positional bookkeeping)."""
    pts, nrm, w, drv = cloud(16, dtype=torch.float32)
    mesh = soup(pts, nrm, w)
    with pytest.raises(TypeError):
        ISLA(PV, WALL, G)
    with pytest.raises(TypeError):
        ISLA(
            boundary_schemas=WALL, hidden=32, n_layers=1, n_slices=8
        )  # outputs is required
    with pytest.raises(TypeError):
        ISLA(
            output_schema=PV, hidden=32, n_layers=1, n_slices=8
        )  # so are the conditions
    m = _isla(hidden=32, n_layers=1, n_slices=8)
    with pytest.raises(TypeError):
        m({"wall": mesh}, None, {"g": drv})
    out = m(boundary_meshes={"wall": mesh}, global_data={"g": drv})
    assert out.point_data["p"].shape == (16,) and out.point_data["v"].shape == (16, 3)
    assert torch.allclose(out.points, mesh.cell_centroids)
    assert all(torch.isfinite(o).all() for o in outputs(out).values())


@pytest.mark.parametrize(
    "name",
    [
        "nonsense",
        "routing_norm",
        "routing_temperature",
        "scale_mode",
        "use_measure_weights",
        "use_relational_geo",
        "fast_point_softmax",
        "query_mass",
        "query_scalar_scale",
        "query_schema",
        "local_rho",
        "query_tokens",
        "support_tokens",
        "interior_geometry",
        "kernel_read",
        "kernel_read_radius",
        "kernel_read_blocks",
        "kernel_read_theta",
        "mlp_ratio",
        "geo_checkpoint",
        "geo_kernel",
        "eps",
    ],
)
def test_unknown_constructor_argument_raises_type_error(name):
    with pytest.raises(TypeError, match=f"unexpected keyword argument '{name}'"):
        _isla(hidden=32, n_layers=1, n_slices=8, **{name: 1})


def test_interior_flag_structures_the_forward():
    """A surface model refuses prediction points; an interior model requires
    them, as a point-cloud Mesh or a bare tensor."""
    pts, nrm, w, drv = cloud(20, dtype=torch.float32)
    mesh = soup(pts, nrm, w)
    surface = _isla(hidden=32, n_layers=1, n_slices=8)
    with pytest.raises(ValueError, match="interior=False"):
        surface(
            boundary_meshes={"wall": mesh},
            prediction_points=pts[:3],
            global_data={"g": drv},
        )
    interior = _isla(hidden=32, n_layers=1, n_slices=8, interior=True)
    with pytest.raises(ValueError, match="pass prediction_points"):
        interior(boundary_meshes={"wall": mesh}, global_data={"g": drv})
    out = interior(
        boundary_meshes={"wall": mesh},
        prediction_points=points_mesh(pts[:3] * 2),
        global_data={"g": drv},
    )
    assert out.point_data["p"].shape == (3,)
    bare = interior(
        boundary_meshes={"wall": mesh},
        prediction_points=pts[:3] * 2,
        global_data={"g": drv},
    )
    assert torch.equal(bare.point_data["p"], out.point_data["p"])
    with pytest.raises(ValueError, match="tensor of shape \\(N, 3\\)"):
        interior(
            boundary_meshes={"wall": mesh},
            prediction_points=pts[:3, :2],
            global_data={"g": drv},
        )


def test_mesh_validation_errors():
    """Dimension and type mismatches are refused at the API boundary with the
    offending mesh named."""
    pts, nrm, w, drv = cloud(20, dtype=torch.float32)
    m = _isla(hidden=32, n_layers=1, n_slices=8)
    with pytest.raises(ValueError, match="non-empty mapping"):
        m(boundary_meshes={}, global_data={"g": drv})
    with pytest.raises(TypeError, match="must be a physicsnemo.mesh.Mesh"):
        m(boundary_meshes={"wall": pts}, global_data={"g": drv})
    with pytest.raises(
        ValueError,
        match="boundary_meshes\\['wall'\\]: the boundary of a 3-dimensional domain is a Mesh\\[2, 3\\]",
    ):
        m(boundary_meshes={"wall": Mesh(points=pts)}, global_data={"g": drv})
    pts2, nrm2, w2, _ = cloud(20, d=2, dtype=torch.float32)
    with pytest.raises(
        ValueError, match="has 2 spatial dimensions; this model has dim=3"
    ):
        m(boundary_meshes={"wall": soup(pts2, nrm2, w2)}, global_data={"g": drv})
    with pytest.raises(
        ValueError, match="global_data is missing the declared fields \\['g'\\]"
    ):
        m(boundary_meshes={"wall": soup(pts, nrm, w)})
    with pytest.raises(TypeError, match="global_data must be"):
        m(boundary_meshes={"wall": soup(pts, nrm, w)}, global_data=drv)


def test_rms_distance_scale_contracts():
    """The RMS-pairwise-distance length unit keeps exact translation invariance and
    rotation covariance, is invariant to a geometric rescale of the boundary (with
    or without the matching measure rescale) AND to a uniform rescale of the
    measures alone, and needs no centre."""
    mesh, drv = _frame_cloud()
    torch.manual_seed(1)
    m = _isla(hidden=64, n_layers=2, n_slices=16).double().eval()
    shift = torch.tensor([300.0, -70.0, 1100.0], dtype=D)
    w = mesh.cell_data[EFFECTIVE_MEASURE_KEY]
    base = _run(m, mesh, drv)
    assert all(torch.isfinite(o).all() for o in outputs(base).values())
    assert same(_run(m, transform(mesh, shift=shift), drv), base, atol=1e-10)
    q = _rotation()
    _assert_rotated(_run(m, transform(mesh, q), drv @ q.T), base, q)
    assert same(_run(m, transform(mesh, scale=2.5), drv), base, atol=1e-10)
    # measure rescale alone: invariant (the unit is normalized)
    assert same(
        _run(m, mesh.with_data(cell_data={EFFECTIVE_MEASURE_KEY: w * 137.0}), drv),
        base,
        atol=1e-10,
    )
    # a rescale of the points alone is invariant too: the unit is a length of the
    # sample and the routing bias log w is uniform-rescale invariant, so measures
    # and positions decouple
    scaled_points_only = transform(mesh, scale=2.5).with_data(
        cell_data={EFFECTIVE_MEASURE_KEY: w}
    )
    assert same(_run(m, scaled_points_only, drv), base, atol=1e-10)


def _koch_curve(level: int):
    """Koch curve (plane z = 0) as a polyline of 4**level segments on the unit base."""
    import math

    pts = [torch.tensor([0.0, 0.0]), torch.tensor([1.0, 0.0])]
    rot = torch.tensor(
        [
            [math.cos(math.pi / 3), -math.sin(math.pi / 3)],
            [math.sin(math.pi / 3), math.cos(math.pi / 3)],
        ]
    )
    for _ in range(level):
        new = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            d = b - a
            p1 = a + d / 3
            p3 = a + 2 * d / 3
            p2 = p1 + rot @ (d / 3)
            new += [p1, p2, p3, b]
        pts = new
    xy = torch.stack(pts).double()
    mids = (xy[:-1] + xy[1:]) / 2
    seg = (xy[1:] - xy[:-1]).norm(dim=-1)
    return torch.cat([mids, torch.zeros(len(mids), 1, dtype=D)], dim=-1)[None], seg[
        None
    ]


def test_rms_distance_converges_on_koch_curve_where_total_measure_diverges():
    """On a fractal boundary refined dyadically, the total measure grows without
    bound ((4/3)^level for the Koch curve) while the RMS pairwise distance
    converges: the relative differences between successive levels fall below 1e-3
    from level 4 on, and the last two levels agree to 1e-4."""
    from physicsnemo.experimental.models.isla.geometry import rms_pairwise_distance

    totals, rms = [], []
    for level in range(0, 7):
        pts, w = _koch_curve(level)
        totals.append(w.sum().item())
        rms.append(rms_pairwise_distance(pts, w).item())
    for a, b in zip(totals[:-1], totals[1:]):
        assert abs(b / a - 4.0 / 3.0) < 1e-6  # the measure diverges geometrically
    rel = [abs(b - a) / a for a, b in zip(rms[:-1], rms[1:])]
    assert all(r < 1e-3 for r in rel[4:]), rel
    assert rel[-1] < 1e-4, rel
    assert rms[-1] > 0.3  # a finite length of the order of the base


def test_geo_kernel_auto_resolves_to_eager_off_cuda_and_honors_explicit_choices():
    """geo_kernel "auto" picks the fused Triton region only where it can run: on CUDA
    with one to four global directions and at most 1024 slices. On CPU it must
    resolve to eager at the first forward pass, cache the decision, and leave the
    outputs identical to a model forced to eager with the same weights."""
    torch.manual_seed(0)
    pts, nrm, w, _ = cloud(64)
    mesh = soup(pts, nrm, w)
    drv = torch.tensor([1.0, 0.0, 0.0], dtype=D)
    auto = _isla(hidden=32, n_layers=2, n_slices=8).double().eval()
    assert auto.geo_kernel == "auto" and auto.resolve_geo_kernel(pts) == "eager"
    eager = _isla(hidden=32, n_layers=2, n_slices=8).double().eval()
    eager.geo_kernel = "eager"
    eager.load_state_dict(auto.state_dict())
    ya, ye = outputs(_run(auto, mesh, drv)), outputs(_run(eager, mesh, drv))
    assert all(torch.equal(ya[k], ye[k]) for k in ya)
    assert all(b.geo_kernel == "eager" for b in auto.blocks)
    assert auto._geo_kernel_cache[1] == "eager"
    # five global directions: auto stays eager (off CUDA it does anyway; the kernel is
    # unrolled for one to four)
    five = {
        "g1": {"rank": 1},
        "g2": {"rank": 1},
        "g3": {"rank": 1},
        "g4": {"rank": 1},
        "g5": {"rank": 1},
    }
    two = (
        ISLA(
            output_schema=PV,
            boundary_schemas=WALL,
            global_schema=five,
            hidden=32,
            n_layers=2,
            n_slices=8,
        )
        .double()
        .eval()
    )
    assert two.resolve_geo_kernel(pts) == "eager"


def test_scalar_only_model_has_no_frame_routing_and_every_parameter_gets_a_gradient():
    """A model whose outputs are all rank 0 never reads the token basis, so the head's
    frame routing is not built (a parameter without a gradient stops DDP at the second
    step); with a vector output the frame routing exists. In both cases every
    parameter receives a gradient in the interior task, and the scalar-only outputs
    equal the head coefficients directly."""
    torch.manual_seed(0)
    pts, nrm, w, drv = cloud(200, dtype=torch.float32)
    mesh = soup(pts, nrm, w)
    q = torch.randn(30, 3) * 2.0
    for outs in (
        {"u": {"rank": 0}},
        {"u": {"rank": 0}, "q1": {"rank": 0}, "q2": {"rank": 0}},
        {"u": {"rank": 0}, "v": {"rank": 1}},
    ):
        m = ISLA(
            output_schema=outs,
            boundary_schemas=WALL,
            global_schema={"g": {"rank": 1}},
            hidden=32,
            n_layers=2,
            n_slices=8,
            interior=True,
        )
        has_vec = any(v["rank"] == 1 for v in outs.values())
        assert m.needs_frame == has_vec and (m.frame_assign is not None) == has_vec
        out = outputs(
            m(
                boundary_meshes={"wall": mesh},
                prediction_points=q,
                global_data={"g": drv},
            )
        )
        sum(v.square().mean() for v in out.values()).backward()
        missing = [k for k, p in m.named_parameters() if p.grad is None]
        assert not missing, missing
        assert out["u"].shape == (30,)
        if "q1" in outs:
            assert out["q2"].shape == (30,) and m.head.out_features == 3
