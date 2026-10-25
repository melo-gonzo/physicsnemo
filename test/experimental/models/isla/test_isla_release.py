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

r"""Release-readiness tests for ISLA (MODELS_IMPLEMENTATION.md MOD-008a/b/c).

- MOD-008a: the constructor and every public attribute, for the defaults in three
  and two dimensions and a configuration that sets every argument to a non-default
  value (:data:`PUBLIC_ATTRIBUTES` lists them; a new attribute fails until listed).
- MOD-008b: forward non-regression against golden outputs stored in ``data/``: the
  two small float32 configurations of 0.7.0 (``isla_default_v1.pth``,
  ``isla_dtt_v1.pth``), and in ``isla_release_v2.pth`` a float64 configuration per
  task at 1e-12, a rich surface model with scales, ISLA-DTT in two dimensions, the
  in-model token draw over about 2,000 cells, and ``query_chunks`` (outputs and
  gradients); the exactness of the Mesh wrapper over the internal batched routine;
  and the backbone golden.
- MOD-008c: committed reference checkpoints of both tasks (``isla_default_v1.mdlus``,
  ``isla_dtt_v1.mdlus``) through ``Module.from_checkpoint`` (attributes and
  forward), a checkpoint round trip over schemas, and the refusal of every earlier
  version on every loading path.

**Regenerating the data.** Every file in ``data/`` is written by one command, and
only by it (a missing file fails; nothing is created on the fly)::

    python -m test.experimental.models.isla.goldens.record --regenerate [FILE ...]

with no FILE every file (the module goldens included), else the named ones (e.g.
``isla_release_v2.pth``). Regenerate only what a change is meant to move, and say so
in the commit: the reference checkpoints stand for a checkpoint written by the code
of their version (regenerate them at a version bump), and ``isla_backbone_v1.pth``
only when the backbone changes by decision (a forward hook on ``norm_out`` under
``CONFIGS["default"]`` / ``_make_tensors``; last recorded when the boundary
conditions came to be declared by name, each with its own input layer, checkpoint
0.7.0). :func:`regenerate` below writes the files of this test.
"""

import inspect
import io
import json
import logging
import re
import warnings
import zipfile
from pathlib import Path

import pytest
import torch

from physicsnemo.core.module import Module
from physicsnemo.experimental.models import ISLA
from physicsnemo.experimental.models.isla.geometry import cell_second_moments
from physicsnemo.mesh.calculus.measure import cell_measures
from test.experimental.models.isla._meshes import outputs, soup
from test.experimental.models.isla.goldens import cases as C

DATA_DIR = Path(__file__).parent / "data"
REGENERATE = "python -m test.experimental.models.isla.goldens.record --regenerate"
D, F = torch.float64, torch.float32

#: Every configuration declares one scalar and one vector output, one boundary condition
#: without fields and one global direction.
IO = dict(
    output_schema={"p": {"rank": 0}, "v": {"rank": 1}},
    boundary_schemas={"wall": {}},
    global_schema={"g": {"rank": 1}},
)

#: One small configuration in its two forms: a surface task and an interior task
#: (the constructor's ``interior`` flag), which is ISLA-DTT.
CONFIGS = {
    "default": dict(hidden=32, n_layers=2, n_slices=8, **IO),
    "dtt": dict(hidden=32, n_layers=2, n_slices=8, interior=True, **IO),
}
#: The committed reference checkpoints: the file, its configuration and the seed of
#: its parameters. ISLA-DTT's is narrower (hidden 16) to keep ``data/`` within its
#: budget.
CHECKPOINTS = {
    "isla_default_v1.mdlus": (CONFIGS["default"], 7),
    "isla_dtt_v1.mdlus": (dict(CONFIGS["dtt"], hidden=16), 8),
}


def _instantiate_model(seed: int = 0, **kwargs) -> ISLA:
    """Build an ISLA with reproducible random parameters (MOD-008b pattern), forced to
    the eager geometry region (the one the goldens were recorded with)."""
    model = ISLA(**kwargs)
    model.geo_kernel = "eager"
    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * 0.1)
    return model.eval()


def _make_tensors(device, seed: int = 1234) -> dict:
    """Deterministic batched token tensors for the internal routine: two cases, 64
    cells, unit normals, positive measures (the backbone golden's inputs)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    b, n = 2, 64
    pts = torch.randn(b, n, 3, generator=gen) * torch.tensor([3.0, 2.0, 1.0])
    nrm = torch.nn.functional.normalize(torch.randn(b, n, 3, generator=gen), dim=-1)
    w = torch.rand(b, n, generator=gen) + 0.5
    g = torch.nn.functional.normalize(torch.randn(b, 1, 3, generator=gen), dim=-1)
    inputs = dict(points=pts, normals=nrm, measure_weights=w)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["global_inputs"] = {"g": g.to(device)}
    return inputs


def _make_inputs(device, query: bool = False, seed: int = 1234) -> dict:
    """Deterministic mesh inputs: one case, a 64-cell boundary soup with unit normals
    and positive effective measures, one global direction, and 40 prediction points
    for the interior task."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = 64
    pts = torch.randn(n, 3, generator=gen) * torch.tensor([3.0, 2.0, 1.0])
    nrm = torch.nn.functional.normalize(torch.randn(n, 3, generator=gen), dim=-1)
    w = torch.rand(n, generator=gen) + 0.5
    g = torch.nn.functional.normalize(torch.randn(3, generator=gen), dim=-1)
    inputs = dict(
        boundary_meshes={"wall": soup(pts, nrm, w).to(device)},
        global_data={"g": g.to(device)},
    )
    if query:
        inputs["prediction_points"] = (torch.randn(40, 3, generator=gen) * 2.0).to(
            device
        )
    else:
        inputs["prediction_boundaries"] = ["wall"]
    return inputs


def _load_reference(file_name: str, key: str | None = None) -> dict:
    """A committed golden (``data/<file_name>``, its entry ``key`` if given); a missing
    one fails with the command that writes it."""
    path = DATA_DIR / file_name
    if not path.exists():
        pytest.fail(f"{path} is missing: regenerate it with `{REGENERATE} {file_name}`")
    data = torch.load(path, weights_only=True)
    return data if key is None else data[key]


def _assert_dict_close(out: dict, ref: dict, atol: float, rtol: float) -> None:
    assert list(out.keys()) == list(ref.keys())
    for k in out.keys():
        torch.testing.assert_close(
            out[k], ref[k].to(out[k].device), atol=atol, rtol=rtol
        )


def _assert_scaled_close(out: dict, ref: dict, tol: float) -> None:
    """Every field within ``tol`` of its golden's largest magnitude (rtol 0: the
    error of an output is set by the field's scale, not by each element's)."""
    assert list(out.keys()) == list(ref.keys())
    for k, r in ref.items():
        r = r.to(out[k].device)
        torch.testing.assert_close(
            out[k],
            r,
            atol=tol * float(r.abs().max()),
            rtol=0.0,
            msg=lambda m: f"{k}: {m}",
        )


# ---------------------------------------------------------------------------
# MOD-008a: constructor and public attributes
# ---------------------------------------------------------------------------

#: Every public attribute an ISLA has (``vars(model)``: not a submodule, a parameter
#: or a buffer). A new one fails ``test_isla_constructor_attributes`` until it is
#: listed here and its value asserted there.
PUBLIC_ATTRIBUTES = (
    "training",
    "meta",
    "logger",
    "compile_blocks",
    "n_spatial_dims",
    "output_schema",
    "boundary_schemas",
    "global_schema",
    "output_stats",
    "interior",
    "n_boundary_tokens",
    "global_vector_names",
    "global_scalar_names",
    "n_global_vectors",
    "n_global_scalars",
    "geo_kernel",
    "geometry_kernel",
    "n_slices",
    "n_boundary_vecs",
    "n_boundary_tens",
    "n_block_vecs",
    "n_block_tens",
    "boundary_names",
    "needs_frame",
    "n_true_basis",
    "n_pseudo_basis",
    "output_layouts",
    "n_head_coeffs",
    "n_query_features",
    "qt",
    "tree_read_kernel",
    "kernel_basis",
    "kernel_reads",
    "frame_assign",
)

#: Every constructor argument at a value other than its default: an interior task in
#: two dimensions (rank-2 boundary inputs are refused there) with three conditions, a
#: rank-1 field defining the velocity scale, a rank-0 global defining a density, a
#: dimensionless global, dotted, odd and rank-2 outputs, compiled blocks, a token
#: budget and output_stats.
CUSTOM = dict(
    output_schema={
        "p": {"rank": 0, "units": {"U": 2, "rho": 1}},
        "q.s": {"rank": 0},
        "u": {"rank": 1, "units": {"U": 1}},
        "w": {"rank": 1, "parity": "odd"},
        "stress": {"rank": 2, "symmetric": True},
        "t": {"rank": 2},
    },
    boundary_schemas={
        "wall": {"trace": {"rank": 0}},
        "inlet": {"velocity": {"rank": 1, "units": {"U": 1}, "defines": "U"}},
        "outlet": {},
    },
    global_schema={
        "g1": {"rank": 1},
        "g2": {"rank": 1},
        "rho": {"rank": 0, "defines": "rho"},
        "re": {"rank": 0},
    },
    interior=True,
    n_boundary_tokens=1000,
    hidden=64,
    n_layers=3,
    n_slices=16,
    compile_blocks=True,
    n_spatial_dims=2,
    output_stats={"p": {"mean": 0.5, "std": 2.0}},
)

ATTRIBUTE_CONFIGS = {
    "with_defaults": IO,
    "with_defaults_2d": dict(IO, n_spatial_dims=2),
    "with_custom_args": CUSTOM,
}


def _ranks(schema) -> dict:
    return {k: (s.rank, s.symmetric, s.parity) for k, s in schema.items()}


def _expected_attributes(config: str) -> dict:
    """The expected value of every public attribute (a callable: a predicate)."""
    common = dict(
        training=True,
        meta=ISLA.MetaData(),
        logger=lambda v: isinstance(v, logging.Logger),
        geo_kernel="auto",
        geometry_kernel="auto",
        tree_read_kernel="auto",
        n_query_features=2,  # the built-in distance channels
        needs_frame=True,
        n_boundary_tens=0,
        n_block_tens=0,
    )
    if config != "with_custom_args":
        ### K = 1: m = 2K + 3 = 5 true basis vectors; (K+2)(K+1)/2 = 3 pseudovectors in
        ### three dimensions, K + 2 = 3 in two.
        return dict(
            common,
            compile_blocks=False,
            n_spatial_dims=3 if config == "with_defaults" else 2,
            output_schema=lambda v: (
                _ranks(v) == {"p": (0, False, "even"), "v": (1, False, "even")}
            ),
            boundary_schemas={"wall": {}},
            global_schema=lambda v: _ranks(v) == {"g": (1, False, "even")},
            output_stats=None,
            interior=False,
            n_boundary_tokens=200_000,
            global_vector_names=["g"],
            global_scalar_names=[],
            n_global_vectors=1,
            n_global_scalars=0,
            n_slices=512,  # the reference configuration (REF-80, 2026-10-04)
            n_boundary_vecs=0,
            n_block_vecs=0,
            boundary_names=("wall",),
            n_true_basis=5,
            n_pseudo_basis=3,
            output_layouts=lambda v: (
                {k: lay.n_coeffs for k, lay in v.items()} == {"p": 1, "v": 5}
            ),
            n_head_coeffs=1 + 5,
            qt=None,
            kernel_basis=None,
            kernel_reads=None,
        )
    ### K = 3 (g1, g2 and the direction the inlet's velocity defines) in two
    ### dimensions: m = 9 true basis vectors, K + 2 = 5 pseudovectors; a vector costs 9,
    ### a pseudovector 5, a symmetric rank-2 tensor m(m+1)/2 + 1 = 46, a general one
    ### m^2 + 1 = 82.
    return dict(
        common,
        compile_blocks=True,
        n_spatial_dims=2,
        output_schema=lambda v: (
            _ranks(v)
            == {
                "p": (0, False, "even"),
                "q.s": (0, False, "even"),
                "u": (1, False, "even"),
                "w": (1, False, "odd"),
                "stress": (2, True, "even"),
                "t": (2, False, "even"),
            }
        ),
        boundary_schemas=lambda v: (
            list(v) == ["inlet", "outlet", "wall"]  # canonical order
            and {k: _ranks(s) for k, s in v.items()}
            == {
                "inlet": {"velocity": (1, False, "even")},
                "outlet": {},
                "wall": {"trace": (0, False, "even")},
            }
        ),
        global_schema=lambda v: list(v) == ["g1", "g2", "rho", "re"],
        output_stats={"p": {"mean": 0.5, "std": 2.0}},
        interior=True,
        n_boundary_tokens=1000,
        global_vector_names=["g1", "g2"],
        global_scalar_names=["re"],  # rho defines a scale: consumed, not appended
        n_global_vectors=3,
        n_global_scalars=1,
        n_slices=16,
        n_boundary_vecs=1,
        n_block_vecs=1,
        boundary_names=("inlet", "outlet", "wall"),
        n_true_basis=9,
        n_pseudo_basis=5,
        output_layouts=lambda v: (
            {k: lay.n_coeffs for k, lay in v.items()}
            == {"p": 1, "q.s": 1, "u": 9, "w": 5, "stress": 46, "t": 82}
        ),
        n_head_coeffs=144,
    )


def _public_attributes(model) -> dict:
    return {
        k: v
        for k, v in vars(model).items()
        if not k.startswith("_")
        and not isinstance(v, (torch.nn.Module, torch.nn.Parameter, torch.Tensor))
    }


def test_the_custom_configuration_sets_every_argument():
    """``with_custom_args`` gives every constructor argument, each at a value other
    than its default."""
    params = inspect.signature(ISLA.__init__).parameters
    assert set(CUSTOM) == set(params) - {"self"}
    for name, value in CUSTOM.items():
        assert value != params[name].default, name


@pytest.mark.parametrize("config", list(ATTRIBUTE_CONFIGS))
def test_isla_constructor_attributes(config):
    """Every public attribute of the model (``vars``, not a submodule or a parameter)
    is in :data:`PUBLIC_ATTRIBUTES` and has its expected value, and the modules the
    configuration implies are built."""
    kw = ATTRIBUTE_CONFIGS[config]
    m = ISLA(**kw)
    found = _public_attributes(m)
    assert set(found) <= set(PUBLIC_ATTRIBUTES), "unlisted: " + str(
        sorted(set(found) - set(PUBLIC_ATTRIBUTES))
    )
    expected = _expected_attributes(config)
    assert set(found) == set(expected), sorted(set(found) ^ set(expected))
    for name, value in found.items():
        want = expected[name]
        assert want(value) if callable(want) else value == want, (name, value)
    assert isinstance(m, Module)
    hidden, n_layers = kw.get("hidden", 512), kw.get("n_layers", 12)
    assert len(m.blocks) == n_layers and m.norm_out.normalized_shape == (hidden,)
    assert m.head.out_features == m.n_head_coeffs
    assert list(m.boundary_conditions) == list(m.boundary_names)
    if m.interior:
        assert m.qt is not None and len(m.kernel_reads) == n_layers
        assert m.kernel_basis is not None


# ---------------------------------------------------------------------------
# MOD-008b: forward non-regression against stored golden outputs
# ---------------------------------------------------------------------------


def _forward_golden(config: str) -> dict:
    """What ``isla_<config>_v1.pth`` stores: the CPU outputs of ``CONFIGS[config]``."""
    model = _instantiate_model(**CONFIGS[config])
    with torch.no_grad():
        return outputs(model(**_make_inputs("cpu", query=(config != "default"))))


@pytest.mark.parametrize("config", list(CONFIGS))
def test_isla_forward_non_regression(device, config):
    """The outputs of ``CONFIGS[config]`` against ``isla_<config>_v1.pth``. Regenerate:
    ``python -m test.experimental.models.isla.goldens.record --regenerate
    isla_<config>_v1.pth``."""
    model = _instantiate_model(**CONFIGS[config]).to(device)
    inputs = _make_inputs(device, query=(config != "default"))
    with torch.no_grad():
        out = outputs(model(**inputs))
    n_out = 64 if config == "default" else 40
    assert out["p"].shape == (n_out,) and out["v"].shape == (n_out, 3)
    ref = _load_reference(f"isla_{config}_v1.pth", "output")
    tol = 1e-5 if device == "cpu" else 1e-4
    _assert_dict_close(out, ref, atol=tol, rtol=tol)


#: The goldens of ``isla_release_v2.pth``: a model case of ``goldens/cases.py`` (its
#: configuration, inputs from correctly rounded operations, the same bits on every
#: platform, and random parameters) and its dtype. In evaluation mode; the draw at
#: its fixed start, at the default tree-read angle.
RELEASE_V2 = "isla_release_v2.pth"
V2_CONFIGS = {
    "surface_f64": ("surface", D),
    "dtt_f64": ("dtt", D),
    "rich_surface_f32": ("rich_surface", F),
    "dtt_2d_f64": ("dtt_2d", D),
    "dtt_draw_f32": ("dtt_draw", F),  # DRAW-EXACT: today's draw
}
#: Tolerances relative to each field's largest magnitude. CPU (reference kernels):
#: float64 1e-12, which holds across BLAS builds, CPU architectures and thread counts
#: (measured at most 1e-14 on these outputs, G-MODEL); float32 1e-5 (measured at most
#: 3e-6). CUDA, kernels auto (the fused geometry region, the fast forest, walk and
#: geometry search, the fused read; TF32 off) against the CPU golden: float64 1e-10
#: and float32 1e-4, the reordered sums of the fused kernels and of CUDA's atomics.
V2_TOL = {("cpu", D): 1e-12, ("cpu", F): 1e-5, ("cuda", D): 1e-10, ("cuda", F): 1e-4}


def _v2_case(name: str, device: str):
    case, dtype = V2_CONFIGS[name]
    spec = C._MODEL_SPECS[case](dtype)
    variant = "reference" if device == "cpu" else "auto"
    return C._model_setup(spec, C._model_build(spec)(), device, variant), dtype


def _v2_outputs(name: str, device: str) -> dict:
    (model, kw), _ = _v2_case(name, device)
    with torch.no_grad():
        return outputs(model(**kw))


def _no_tf32():
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    return tf32


@pytest.mark.parametrize("config", list(V2_CONFIGS))
def test_isla_forward_non_regression_v2(device, config):
    """The outputs of each configuration of :data:`V2_CONFIGS` against
    ``isla_release_v2.pth`` (recorded on one CPU thread with the reference kernels),
    at :data:`V2_TOL`. ``dtt_draw_f32`` pins today's draw of 512 of its cells
    (DRAW-EXACT; its hits keep 1e-9 off the cells' boundaries, test_isla_goldens).
    Regenerate: ``python -m test.experimental.models.isla.goldens.record --regenerate
    isla_release_v2.pth``."""
    tf32 = _no_tf32()
    try:
        out = _v2_outputs(config, device)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    kind = "cpu" if device == "cpu" else "cuda"
    _assert_scaled_close(
        out,
        _load_reference(RELEASE_V2, config)["output"],
        V2_TOL[(kind, V2_CONFIGS[config][1])],
    )


#: query_chunks' golden: ISLA-DTT in float64 over three ranges of its 60 points.
CHUNKS = ((0, 20), (20, 45), (45, 60))


def _chunks_golden(device: str) -> dict:
    """``query_chunks`` of ``dtt_f64``: each range's outputs (a fixed cotangent on
    each), then, after ``chunks.backward()``, every parameter's gradient as its
    projection on a fixed random direction and the sum of the magnitudes of the
    projection's terms (the scale its rounding is set by)."""
    (model, kw), _ = _v2_case("dtt_f64", device)
    chunks = model.query_chunks(**kw)
    out = {"pair_counts": chunks.pair_counts.cpu()}
    for i, (a, b) in enumerate(CHUNKS):
        fields = outputs(chunks(a, b))
        C._cotangent_loss(fields, 90 + 10 * i).backward()
        out.update({f"range{i}.{k}": t.detach().cpu() for k, t in fields.items()})
    chunks.backward()
    gen = torch.Generator().manual_seed(95)
    proj, scale = [], []
    for p in model.parameters():
        r = torch.randn(p.shape, generator=gen, dtype=D)
        g = p.grad.detach().cpu().double()
        proj.append((g * r).sum())
        scale.append((g * r).abs().sum())
    out["grad_projections"], out["grad_scales"] = torch.stack(proj), torch.stack(scale)
    return out


def test_isla_query_chunks_non_regression(device):
    """``query_chunks`` (every public method has a golden): the outputs of three
    ranges of ISLA-DTT's points, its pair counts (exact), and every parameter's
    gradient after the chunks' backward passes and ``chunks.backward()`` (each as a
    projection on a fixed random direction, within the float64 bound of
    :data:`V2_TOL` of the largest sum of magnitudes of a projection's terms).
    Regenerate: ``python -m test.experimental.models.isla.goldens.record --regenerate
    isla_release_v2.pth``."""
    tf32 = _no_tf32()
    try:
        out = _chunks_golden(device)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    ref = _load_reference(RELEASE_V2, "query_chunks")
    tol = V2_TOL[("cpu" if device == "cpu" else "cuda", D)]
    assert torch.equal(out.pop("pair_counts"), ref["pair_counts"])
    proj, scale = out.pop("grad_projections"), out.pop("grad_scales")
    _assert_scaled_close(
        out, {k: v for k, v in ref.items() if k.startswith("range")}, tol
    )
    bound = tol * float(ref["grad_scales"].max())
    torch.testing.assert_close(proj, ref["grad_projections"], atol=bound, rtol=0.0)
    torch.testing.assert_close(scale, ref["grad_scales"], atol=bound, rtol=0.0)


@pytest.mark.parametrize("config", list(CONFIGS))
def test_mesh_wrapper_is_exact_over_the_batched_routine(device, config):
    """The Mesh API is a wrapper: the same parameters run through ``forward`` on the
    meshes and through the internal batched routine on the very tensors the wrapper
    reads (cell centroids, cell normals, effective cell measures and second moments, the cells as the
    surface of the interior geometry, a batch axis of one) give bitwise-identical
    outputs."""
    model = _instantiate_model(**CONFIGS[config]).to(device)
    inputs = _make_inputs(device, query=(config != "default"))
    mesh = inputs["boundary_meshes"]["wall"]
    q = inputs.get("prediction_points")
    with torch.no_grad():
        via_mesh = model(**inputs)
        direct = model._forward_tensors(
            points=mesh.cell_centroids[None],
            normals=mesh.cell_normals[None],
            measure_weights=cell_measures(mesh)[None],
            global_inputs={"g": inputs["global_data"]["g"]},
            query_points=None if q is None else q[None],
            surface_points=mesh.points[None],
            surface_cells=mesh.cells[None],
            cell_second_moments=cell_second_moments(mesh)[None],
        )
    expected_points = mesh.cell_centroids if q is None else q
    assert torch.equal(via_mesh.points, expected_points) and via_mesh.n_cells == 0
    for k, v in direct.items():
        if config == "dtt" and via_mesh.point_data[k].is_cuda:
            ### The boundary read's node means are float32 sums by CUDA's index_add, whose
            ### atomics add in no fixed order, so the two calls agree to float32 roundoff,
            ### not bitwise (non-bitwise results from atomics are fine within a small
            ### multiple of float32 eps). The bound is eps times the length of the reordered
            ### sums (at most the 64 boundary cells summed into one node), relative to each
            ### element and, for the elements near zero, to the field's RMS: about 1.2e-7
            ### absolute on p, six times the largest difference seen on a GB300 (2.05e-8).
            eps, depth = torch.finfo(torch.float32).eps, mesh.n_cells
            ref = v[0]
            torch.testing.assert_close(
                via_mesh.point_data[k],
                ref,
                rtol=depth * eps,
                atol=depth * eps * float(ref.square().mean().sqrt()),
            )
        else:
            assert torch.equal(via_mesh.point_data[k], v[0]), k
    assert torch.equal(via_mesh.global_data["g"], inputs["global_data"]["g"])


def _backbone_golden() -> torch.Tensor:
    """The pre-head token state (``norm_out``'s output) of ``CONFIGS["default"]`` on
    ``_make_tensors`` through the internal batched routine, on the CPU."""
    model = _instantiate_model(**CONFIGS["default"])
    captured = {}
    model.norm_out.register_forward_hook(
        lambda mod, i, o: captured.setdefault("x", o.detach().clone())
    )
    with torch.no_grad():
        model._forward_tensors(**_make_tensors("cpu"))
    return captured["x"]


def test_isla_backbone_matches_the_recorded_backbone():
    """With no per-token inputs, the pre-head token state (``norm_out`` output) of
    the surface model is the one recorded when the backbone was last changed by
    decision (the slice sublayer's normalized reads, H-RESID arm A2), for the same
    parameters and token tensors. ``data/isla_backbone_v1.pth``
    was recorded with a forward hook on ``norm_out`` (seed 0 parameters, seed 1234
    inputs, as here) on an x86 CPU; the tensors enter through the internal batched
    routine, which the Mesh wrapper calls. Compared at float32 tolerance: the same
    graph gives bitwise-identical values on the recording architecture and differs
    at the last bits across CPU architectures (BLAS reduction order), which is not a
    property of the model. Regenerate (only when the backbone changes by decision):
    ``python -m test.experimental.models.isla.goldens.record --regenerate
    isla_backbone_v1.pth``."""
    ref = _load_reference("isla_backbone_v1.pth", "default")
    torch.testing.assert_close(
        _backbone_golden(), ref["norm_out"], atol=1e-6, rtol=1e-5
    )


# ---------------------------------------------------------------------------
# MOD-008c: checkpoints: the committed references, round trips, earlier versions
# ---------------------------------------------------------------------------


def _stored_args(path) -> dict:
    """The constructor arguments a ``.mdlus`` stores (its ``args.json``)."""
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("args.json"))["__args__"]


def _comparable(model) -> dict:
    """The public attributes in a comparable form (the output layouts by their specs
    and coefficient indices, the logger by its name)."""
    out = {}
    for k, v in _public_attributes(model).items():
        if k == "output_layouts":
            v = {n: (lay.spec, lay.n_coeffs, lay.index) for n, lay in v.items()}
            v = {
                n: (s, c, None if i is None else i.tolist())
                for n, (s, c, i) in v.items()
            }
        elif k == "logger":
            v = v.name
        out[k] = v
    return out


def _checkpoint_output(file_name: str, device) -> dict:
    """The forward pass of a committed reference checkpoint on ``_make_inputs``."""
    model = Module.from_checkpoint(str(DATA_DIR / file_name)).to(device).eval()
    with torch.no_grad():
        return outputs(model(**_make_inputs(device, query=model.interior)))


@pytest.mark.parametrize("file_name", list(CHECKPOINTS))
def test_isla_reference_checkpoint_matches_golden(device, file_name):
    """A committed reference checkpoint (version 0.13.0) loads through
    ``Module.from_checkpoint`` without a warning, as the model of its configuration
    (every public attribute but the execution choices, which a loaded model starts at
    "auto"), and keeps its outputs (``<name>_checkpoint_output_v1.pth``). Regenerate
    (at a version bump): ``python -m test.experimental.models.isla.goldens.record
    --regenerate <name>_v1.mdlus <name>_checkpoint_output_v1.pth``."""
    path = DATA_DIR / file_name
    if not path.exists():
        pytest.fail(f"{path} is missing: regenerate it with `{REGENERATE} {file_name}`")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = Module.from_checkpoint(str(path)).eval()
    assert not [w for w in caught if "checkpoint" in str(w.message)]
    assert isinstance(model, ISLA)
    cfg, _ = CHECKPOINTS[file_name]
    fresh = ISLA(**cfg).eval()
    assert _comparable(model) == _comparable(fresh)
    assert model.geo_kernel == model.tree_read_kernel == model.geometry_kernel == "auto"
    assert len(model.blocks) == cfg["n_layers"]
    assert model.norm_out.normalized_shape == (cfg["hidden"],)
    out = _checkpoint_output(file_name, device)
    ref = _load_reference(
        file_name.replace("_v1.mdlus", "_checkpoint_output_v1.pth"), "output"
    )
    tol = 1e-5 if device == "cpu" else 1e-4
    _assert_dict_close(out, ref, atol=tol, rtol=tol)


#: The schemas a checkpoint must keep (each a model case of ``goldens/cases.py``):
#: outputs of every rank, symmetry and parity with two global directions; dotted
#: names of fields, globals and outputs; ISLA-DTT; a model with scales and
#: output_stats; and ISLA-DTT in two dimensions.
_HEADS_OUTPUTS = {
    "s": {"rank": 0},
    "v": {"rank": 1},
    "pv": {"rank": 1, "parity": "odd"},
    "t": {"rank": 2},
    "ts": {"rank": 2, "symmetric": True},
    "pt": {"rank": 2, "parity": "odd"},
    "t3": {"rank": 3},
    "t3o": {"rank": 3, "parity": "odd"},
    "t4s": {"rank": 4, "symmetric": True},
}
ROUND_TRIPS = {
    "heads": C._ModelSpec(
        dict(
            hidden=16,
            n_layers=1,
            n_slices=8,
            output_schema=_HEADS_OUTPUTS,
            boundary_schemas={"wall": {}},
            global_schema={"g0": {"rank": 1}, "g1": {"rank": 1}},
        ),
        {"wall": lambda: C._soup_cells(48, 3, 91)},
        prediction_boundaries=("wall",),
    ),
    "names": C._ModelSpec(
        dict(
            hidden=16,
            n_layers=1,
            n_slices=8,
            output_schema={
                "surface.pressure": {"rank": 0},
                "surface.wall_shear": {"rank": 1},
                "solid.stress": {"rank": 2, "symmetric": True},
                "nut": {"rank": 0},
            },
            global_schema={
                "conditions.freestream": {"rank": 1},
                "conditions.re": {"rank": 0},
            },
            boundary_schemas={
                "wall": {"bc.trace": {"rank": 0}, "bc.traction": {"rank": 1}}
            },
        ),
        {"wall": lambda: C._soup_cells(48, 3, 92)},
        prediction_boundaries=("wall",),
    ),
    "interior": C._MODEL_SPECS["dtt"](D),
    "scaled": C._MODEL_SPECS["rich_surface"](D),
    "two_dimensional": C._MODEL_SPECS["dtt_2d"](D),
}


@pytest.mark.parametrize("schema", list(ROUND_TRIPS))
def test_isla_checkpoint_round_trip(tmp_path, schema):
    """``save`` and ``Module.from_checkpoint`` keep the model: the checkpoint names
    every constructor argument (``Module.__new__`` stores the bound defaults, so it
    never depends on the class defaults), the loaded model has every public
    attribute of the saved one, and its outputs on the case are bitwise the saved
    model's (on the CPU; serialization is device-independent, and the reference
    checkpoints' tests run on every device)."""
    spec = ROUND_TRIPS[schema]
    model, kw = C._model_setup(spec, C._model_build(spec)(), "cpu", "reference")
    ### from_checkpoint builds the model in float32 and loads the weights into it:
    ### float32-representable parameters make the float64 round trip exact.
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(p.float())
    path = tmp_path / f"isla_{schema}.mdlus"
    model.save(path)
    assert set(_stored_args(path)) == set(
        inspect.signature(ISLA.__init__).parameters
    ) - {"self"}
    loaded = Module.from_checkpoint(str(path)).to(dtype=spec.dtype).eval()
    assert isinstance(loaded, ISLA)
    ### geo_kernel and its kin are execution choices the checkpoint does not store: the
    ### loaded model starts at "auto", so it takes the saved model's choices.
    assert (
        loaded.geo_kernel == loaded.tree_read_kernel == loaded.geometry_kernel == "auto"
    )
    C._model_kernels(loaded, "reference")
    assert _comparable(loaded) == _comparable(model)
    with torch.no_grad():
        a, b = outputs(loaded(**kw)), outputs(model(**kw))
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a), schema


#: Each earlier release and the state-dict version torch records in its checkpoints
#: (``ISLA._version``: 1 before 0.8.0, 10 the unmerged head-frame candidate).
EARLIER_RELEASES = {
    "0.3.0": 1,
    "0.4.0": 1,
    "0.5.0": 1,
    "0.6.0": 1,
    "0.7.0": 1,
    "0.8.0": 8,
    "0.9.0": 9,
    "0.10.0": 10,
    "0.11.0": 11,
    "0.12.0": 12,
}


def _rewritten(src: Path, dst: Path, version=None, state_dict=None) -> Path:
    """A copy of the checkpoint ``src`` with its metadata version and/or its weights
    (``model.pt``) replaced."""
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "metadata.json" and version is not None:
                meta = json.loads(data)
                meta["mdlus_file_version"] = version
                data = json.dumps(meta).encode()
            if item.filename == "model.pt" and state_dict is not None:
                buf = io.BytesIO()
                torch.save(state_dict, buf)
                data = buf.getvalue()
            zout.writestr(item, data)
    return dst


@pytest.mark.parametrize("file_name", list(CHECKPOINTS))
@pytest.mark.parametrize("release", list(EARLIER_RELEASES))
def test_isla_checkpoint_of_an_earlier_version_is_refused(tmp_path, file_name, release):
    """ISLA keeps no upgrade path. A reference checkpoint relabelled as an earlier
    release wrote it (its metadata version and its weights' state-dict version) is
    refused by ``Module.from_checkpoint`` with an error naming both versions, and by
    ``Module.load`` into a current model (the recipe's resume) with the state-dict
    version named; with only its weights' state-dict version earlier (the metadata
    current), ``from_checkpoint`` refuses it too."""
    assert ISLA.__model_checkpoint_version__ == "0.13.0" and ISLA._version == 13
    cfg, _ = CHECKPOINTS[file_name]
    src = DATA_DIR / file_name
    with zipfile.ZipFile(src) as z:
        sd = torch.load(io.BytesIO(z.read("model.pt")), weights_only=True)
    sd._metadata[""]["version"] = EARLIER_RELEASES[release]  # what that release wrote
    old = _rewritten(src, tmp_path / f"old_{release}.mdlus", release, sd)
    with pytest.raises(
        IOError,
        match=re.escape(
            f"version {release} is not compatible with current version 0.13.0"
        ),
    ):
        Module.from_checkpoint(str(old))
    sd_version = (
        rf"before version 0\.13\.0 \(state-dict version {EARLIER_RELEASES[release]},"
    )
    with pytest.raises(RuntimeError, match=sd_version):
        ISLA(**cfg).load(str(old), strict=True)
    weights_only = _rewritten(src, tmp_path / f"weights_{release}.mdlus", None, sd)
    with pytest.raises(RuntimeError, match=sd_version):
        Module.from_checkpoint(str(weights_only))


def test_isla_earlier_state_dict_is_refused_on_every_loading_path(tmp_path):
    """A checkpoint whose weights have the parameter layout before 0.7.0 (the shared
    seed embedding ``embed.0``, ``qt.type_offset``, ``qt.embed``), loaded the way the
    recipe resumes and initializes (``Module.load``, strict), or through
    ``Module.from_checkpoint``, fails with the version named rather than with a list
    of missing and unexpected keys; so does ``load_state_dict`` itself."""
    model = _instantiate_model(**CONFIGS["dtt"])
    path = tmp_path / "isla.mdlus"
    model.save(path)
    old = {
        k: v
        for k, v in model.state_dict().items()
        if not k.startswith(("boundary_conditions.", "qt.seed."))
    }
    old["embed.0.weight"] = torch.zeros(32, 2)
    old["qt.type_offset"] = torch.zeros(32)
    old["qt.embed.0.weight"] = torch.zeros(32, 2)
    stale = _rewritten(path, tmp_path / "isla_0.6.0_weights.mdlus", None, old)
    fresh = ISLA(**CONFIGS["dtt"])
    layout = "parameter layout of a checkpoint before version 0.7.0"
    with pytest.raises(RuntimeError, match=layout):
        fresh.load(str(stale), strict=True)
    with pytest.raises(RuntimeError, match=layout):
        Module.from_checkpoint(str(stale))
    with pytest.raises(RuntimeError, match="no upgrade path"):
        fresh.load_state_dict(old, strict=False)
    ### a DDP-style prefix is seen through
    with pytest.raises(RuntimeError, match="before version 0.7.0"):
        torch.nn.ModuleDict({"module": fresh}).load_state_dict(
            {f"module.{k}": v for k, v in old.items()}
        )
    fresh.load(str(path))  # the current layout loads


@pytest.mark.parametrize(
    "release, version",
    [(r, v) for r, v in EARLIER_RELEASES.items() if v > 1 or r == "0.7.0"],
)
def test_isla_0_7_0_to_0_12_0_state_dicts_are_refused_on_every_loading_path(
    tmp_path, release, version
):
    """0.7.0 to 0.12.0 and the unmerged head-frame candidate 0.10.0 have the current
    parameter layout but an earlier forward or constructor (the prediction points'
    geometry encoding before 0.8.0, the interior task's length unit measured on the
    drawn tokens before 0.9.0, the geometry before its robustness bundle in 0.11.0, the
    candidate's head frame, the exact geometry's first-order bands before 0.12.0, and
    the constructor's dim, boundary_tokens, query_independent and reynolds before 0.13.0): their state dicts,
    told apart by the state-dict version torch records (1 before 0.8.0, as every 0.7.0
    checkpoint carries, 8 for 0.8.0, 9 for 0.9.0, 10 for the candidate, 11 for 0.11.0
    and 12 for 0.12.0), are refused by ``Module.load``, ``load_state_dict`` and through
    a DDP-style wrapper, with the version named; the current state dict loads."""
    model = _instantiate_model(**CONFIGS["dtt"])
    path = tmp_path / "isla.mdlus"
    model.save(path)
    old = model.state_dict()
    assert old._metadata[""]["version"] == ISLA._version == 13
    old._metadata[""]["version"] = version  # what that release wrote
    stale = _rewritten(path, tmp_path / f"isla_{release}_weights.mdlus", None, old)
    fresh = ISLA(**CONFIGS["dtt"])
    with pytest.raises(
        RuntimeError, match=rf"before version 0\.13\.0 \(state-dict version {version}"
    ):
        fresh.load(str(stale), strict=True)
    with pytest.raises(RuntimeError, match="no upgrade path"):
        fresh.load_state_dict(old, strict=False)
    wrapped = torch.nn.ModuleDict({"module": model}).state_dict()
    wrapped._metadata["module"]["version"] = version
    with pytest.raises(RuntimeError, match="before version 0.13.0"):
        torch.nn.ModuleDict({"module": fresh}).load_state_dict(wrapped)
    fresh.load(str(path))  # the current state dict loads
    _assert_dict_close(fresh.state_dict(), model.state_dict(), atol=0.0, rtol=0.0)


#: Constructor arguments ISLA does not take: a nonsense name, and every argument an
#: earlier ISLA or a prototype took (0.13.0: dim, boundary_tokens, query_independent,
#: reynolds and the scales' velocity_scale, density and viscosity roles).
RETIRED_ARGUMENTS = (
    "nonsense",
    "dim",
    "boundary_tokens",
    "query_independent",
    "reynolds",
    "velocity_scale",
    "density",
    "viscosity",
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
)


@pytest.mark.parametrize("name", RETIRED_ARGUMENTS)
def test_isla_unknown_constructor_argument_is_a_type_error(name):
    """An argument ISLA does not take is a TypeError naming it (no silent ``**kwargs``),
    in either task."""
    for interior in (False, True):
        with pytest.raises(TypeError, match=f"unexpected keyword argument '{name}'"):
            ISLA(
                hidden=16, n_layers=1, n_slices=8, interior=interior, **IO, **{name: 1}
            )


# ---------------------------------------------------------------------------
# Regeneration (``goldens/record.py --regenerate``)
# ---------------------------------------------------------------------------


def _save_output(file_name: str, output: dict) -> None:
    torch.save(
        {"output": {k: v.detach().cpu() for k, v in output.items()}},
        DATA_DIR / file_name,
    )


def _write_checkpoint(file_name: str) -> None:
    cfg, seed = CHECKPOINTS[file_name]
    _instantiate_model(seed=seed, **cfg).save(DATA_DIR / file_name)


def _write_v2() -> None:
    golden = {name: {"output": _v2_outputs(name, "cpu")} for name in V2_CONFIGS}
    golden["query_chunks"] = _chunks_golden("cpu")
    torch.save(golden, DATA_DIR / RELEASE_V2)


#: Every file of ``data/`` this test reads, and how it is written (on the CPU).
WRITERS = {
    "isla_default_v1.pth": lambda: _save_output(
        "isla_default_v1.pth", _forward_golden("default")
    ),
    "isla_dtt_v1.pth": lambda: _save_output("isla_dtt_v1.pth", _forward_golden("dtt")),
    "isla_backbone_v1.pth": lambda: torch.save(
        {"default": {"norm_out": _backbone_golden()}}, DATA_DIR / "isla_backbone_v1.pth"
    ),
    "isla_default_v1.mdlus": lambda: _write_checkpoint("isla_default_v1.mdlus"),
    "isla_default_checkpoint_output_v1.pth": lambda: _save_output(
        "isla_default_checkpoint_output_v1.pth",
        _checkpoint_output("isla_default_v1.mdlus", "cpu"),
    ),
    "isla_dtt_v1.mdlus": lambda: _write_checkpoint("isla_dtt_v1.mdlus"),
    "isla_dtt_checkpoint_output_v1.pth": lambda: _save_output(
        "isla_dtt_checkpoint_output_v1.pth",
        _checkpoint_output("isla_dtt_v1.mdlus", "cpu"),
    ),
    RELEASE_V2: _write_v2,
}


def regenerate(names=None, log=print) -> None:
    """Write the named files of :data:`WRITERS` (all by default) in their order (a
    checkpoint before its output golden), on the CPU."""
    for name in WRITERS:
        if names is None or name in names:
            WRITERS[name]()
            log(f"wrote {DATA_DIR / name} ({(DATA_DIR / name).stat().st_size} bytes)")
