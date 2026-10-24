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

Constructor and public attributes for a minimal and a custom configuration; forward
non-regression against golden outputs stored next to this file; the exactness of the
Mesh wrapper over the internal batched routine; checkpoint save and
``Module.from_checkpoint`` round trip; a committed reference ``.mdlus`` (and the
refusal of a checkpoint of an earlier version); and the backbone golden.

**Regenerating the goldens**: delete ``data/isla_default_v1.pth``,
``data/isla_interior_v1.pth``, ``data/isla_dtt_v1.pth``, ``data/isla_default_v1.mdlus`` and
``data/isla_default_checkpoint_output_v1.pth`` and run

    uv run --no-sync python -m pytest test/experimental/models/isla/test_isla_release.py

twice: the first run recreates each missing file from the current code (and fails on
purpose), the second run must pass. ``data/isla_backbone_v1.pth`` is regenerated only
when the backbone changes by decision (a forward hook on ``norm_out`` under
``CONFIGS["default"]`` / ``_make_tensors`` below; last recorded when the boundary
conditions came to be declared by name, each with its own input layer, checkpoint
0.7.0) and pins the backbone through the internal batched routine.
"""

import re
import warnings
from pathlib import Path

import pytest
import torch

from physicsnemo.core.module import Module
from physicsnemo.experimental.models import ISLA
from physicsnemo.experimental.models.isla.geometry import cell_second_moments
from physicsnemo.mesh.calculus.measure import cell_measures
from test.experimental.models.isla._meshes import outputs, soup

DATA_DIR = Path(__file__).parent / "data"

#: Every configuration declares one scalar and one vector output, one boundary condition
#: without fields and one global direction.
IO = dict(
    output_schema={"p": {"rank": 0}, "v": {"rank": 1}},
    boundary_schemas={"wall": {}},
    global_schema={"g": {"rank": 1}},
)

#: One small configuration in its three forms: a surface task, an interior task (the
#: constructor's ``interior`` flag) and ISLA-DTT (``query_independent``).
CONFIGS = {
    "default": dict(hidden=32, n_layers=2, n_slices=8, **IO),
    "interior": dict(hidden=32, n_layers=2, n_slices=8, interior=True, **IO),
    "dtt": dict(
        hidden=32, n_layers=2, n_slices=8, interior=True, query_independent=True, **IO
    ),
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


def _load_or_create_reference(file_name: str, output: dict) -> dict:
    """Load a golden output dict, or create it and require a second run (sibling-model pattern)."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / file_name
    if not path.exists():
        torch.save({"output": {k: v.detach().cpu() for k, v in output.items()}}, path)
        raise IOError(
            f"Golden output {path} was missing and has been created; commit it and re-run."
        )
    ref = torch.load(path, weights_only=True)["output"]
    return {k: v.to(next(iter(output.values())).device) for k, v in ref.items()}


def _assert_dict_close(out: dict, ref: dict, atol: float, rtol: float) -> None:
    assert list(out.keys()) == list(ref.keys())
    for k in out.keys():
        torch.testing.assert_close(out[k], ref[k], atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# MOD-008a: constructor and public attributes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config", ["minimal", "custom"], ids=["with_defaults", "with_custom_args"]
)
def test_isla_constructor_attributes(config):
    if config == "minimal":
        m = ISLA(**IO)
        assert list(m.output_schema) == ["p", "v"]
        assert (m.output_schema["p"].rank, m.output_schema["v"].rank) == (0, 1)
        assert m.n_head_coeffs == 1 + 5  # a scalar and a true vector over m = 5
        assert m.n_slices == 512  # the reference configuration (REF-80, 2026-10-04)
        assert len(m.blocks) == 12
        assert m.norm_out.normalized_shape == (512,)  # the reference width
        assert m.geo_kernel == "auto"
        assert m.interior is False and m.qt is None
        assert m.n_global_vectors == 1 and m.n_global_scalars == 0
        assert m.boundary_schemas == {"wall": {}}
    else:
        m = ISLA(
            output_schema={
                "p": {"rank": 0},
                "q": {"rank": 0},
                "nut": {"rank": 0},
                "u": {"rank": 1},
                "tau": {"rank": 1},
                "stress": {"rank": 2, "symmetric": True},
            },
            global_schema={"g1": {"rank": 1}, "g2": {"rank": 1}, "re": {"rank": 0}},
            boundary_schemas={
                "wall": {"trace": {"rank": 0}, "traction": {"rank": 1}},
                "outlet": {},
            },
            interior=True,
            hidden=64,
            n_layers=3,
            n_slices=16,
        )
        assert len(m.output_schema) == 6 and m.output_schema["stress"].symmetric
        assert m.n_slices == 16
        assert len(m.blocks) == 3
        assert m.geo_kernel == "auto"
        assert m.n_global_vectors == 2 and m.n_global_scalars == 1
        assert list(m.boundary_schemas) == ["outlet", "wall"]  # canonical order
        assert list(m.boundary_schemas["wall"]) == ["trace", "traction"]
        assert list(m.boundary_conditions) == ["outlet", "wall"]
        assert m.interior is True and m.qt is not None
    assert isinstance(m, Module)


# ---------------------------------------------------------------------------
# MOD-008b: forward non-regression against stored golden outputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", list(CONFIGS))
def test_isla_forward_non_regression(device, config):
    model = _instantiate_model(**CONFIGS[config]).to(device)
    inputs = _make_inputs(device, query=(config != "default"))
    with torch.no_grad():
        out = outputs(model(**inputs))
    n_out = 64 if config == "default" else 40
    assert out["p"].shape == (n_out,) and out["v"].shape == (n_out, 3)
    ref = _load_or_create_reference(f"isla_{config}_v1.pth", out)
    tol = 1e-5 if device == "cpu" else 1e-4
    _assert_dict_close(out, ref, atol=tol, rtol=tol)


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
    property of the model."""
    ref = torch.load(DATA_DIR / "isla_backbone_v1.pth", weights_only=True)["default"]
    model = _instantiate_model(**CONFIGS["default"])
    captured = {}
    model.norm_out.register_forward_hook(
        lambda mod, i, o: captured.setdefault("x", o.detach().clone())
    )
    with torch.no_grad():
        model._forward_tensors(**_make_tensors("cpu"))
    torch.testing.assert_close(captured["x"], ref["norm_out"], atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# MOD-008c: checkpoint save / from_checkpoint round trip and a committed reference
# ---------------------------------------------------------------------------


def test_isla_save_from_checkpoint_round_trip(device, tmp_path):
    model = _instantiate_model(seed=3, **CONFIGS["interior"]).to(device)
    inputs = _make_inputs(device, query=True)
    path = tmp_path / "isla_interior.mdlus"
    model.save(path)
    loaded = Module.from_checkpoint(str(path)).to(device).eval()
    assert isinstance(loaded, ISLA)
    ### geo_kernel is an execution choice the checkpoint does not store: the loaded model
    ### starts at "auto" (fused on CUDA), so it takes the saved model's choice.
    assert loaded.geo_kernel == "auto"
    loaded.geo_kernel = model.geo_kernel
    for name in (
        "n_slices",
        "interior",
        "n_global_vectors",
        "output_schema",
        "boundary_schemas",
        "global_schema",
    ):
        assert getattr(loaded, name) == getattr(model, name), name
    with torch.no_grad():
        _assert_dict_close(
            outputs(loaded(**inputs)), outputs(model(**inputs)), atol=0.0, rtol=0.0
        )


def test_isla_reference_checkpoint_matches_golden(device):
    """The committed reference checkpoint (version 0.12.0) keeps loading, without a
    warning, and keeps its outputs."""
    DATA_DIR.mkdir(exist_ok=True)
    ckpt = DATA_DIR / "isla_default_v1.mdlus"
    if not ckpt.exists():
        _instantiate_model(seed=7, **CONFIGS["default"]).save(ckpt)
        raise IOError(
            f"Reference checkpoint {ckpt} was missing and has been created; commit it and re-run."
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = Module.from_checkpoint(str(ckpt)).to(device).eval()
    assert not [w for w in caught if "checkpoint" in str(w.message)]
    assert isinstance(model, ISLA)
    assert (
        model.n_slices == 8
        and model.geo_kernel == "auto"
        and list(model.output_schema) == ["p", "v"]
    )
    assert model.interior is False and model.boundary_schemas == {"wall": {}}
    inputs = _make_inputs(device)
    with torch.no_grad():
        out = outputs(model(**inputs))
    ref = _load_or_create_reference("isla_default_checkpoint_output_v1.pth", out)
    tol = 1e-5 if device == "cpu" else 1e-4
    _assert_dict_close(out, ref, atol=tol, rtol=tol)


@pytest.mark.parametrize(
    "version",
    ["0.3.0", "0.4.0", "0.5.0", "0.6.0", "0.7.0", "0.8.0", "0.9.0", "0.10.0", "0.11.0"],
)
def test_isla_checkpoint_of_an_earlier_version_is_refused(tmp_path, version):
    """ISLA keeps no upgrade path: a checkpoint written at an earlier version (here
    the reference checkpoint relabelled) fails to load with an error naming both
    versions."""
    import json
    import zipfile

    old = tmp_path / f"isla_{version}.mdlus"
    with (
        zipfile.ZipFile(DATA_DIR / "isla_default_v1.mdlus") as zin,
        zipfile.ZipFile(old, "w") as zout,
    ):
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "metadata.json":
                meta = json.loads(data)
                meta["mdlus_file_version"] = version
                data = json.dumps(meta).encode()
            zout.writestr(item, data)
    assert ISLA.__model_checkpoint_version__ == "0.12.0"
    with pytest.raises(
        IOError,
        match=re.escape(
            f"version {version} is not compatible with current version 0.12.0"
        ),
    ):
        Module.from_checkpoint(str(old))


def test_isla_earlier_state_dict_is_refused_on_every_loading_path(tmp_path):
    """A checkpoint whose weights have the parameter layout before 0.7.0 (the shared
    seed embedding ``embed.0``, ``qt.type_offset``, ``qt.embed``), loaded the way the
    recipe resumes and initializes (``Module.load``, strict), fails with the version
    named rather than with a list of missing and unexpected keys; so does
    ``load_state_dict`` itself."""
    import io
    import zipfile

    model = _instantiate_model(**CONFIGS["interior"])
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
    buf = io.BytesIO()
    torch.save(old, buf)
    stale = tmp_path / "isla_0.6.0_weights.mdlus"
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(stale, "w") as zout:
        for item in zin.infolist():
            zout.writestr(
                item,
                buf.getvalue()
                if item.filename == "model.pt"
                else zin.read(item.filename),
            )
    fresh = ISLA(**CONFIGS["interior"])
    with pytest.raises(
        RuntimeError, match="parameter layout of a checkpoint before version 0.7.0"
    ):
        fresh.load(str(stale), strict=True)
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
    [("0.7.0", 1), ("0.8.0", 8), ("0.9.0", 9), ("0.10.0", 10), ("0.11.0", 11)],
)
def test_isla_0_7_0_to_0_11_0_state_dicts_are_refused_on_every_loading_path(
    tmp_path, release, version
):
    """0.7.0 to 0.11.0 and the unmerged head-frame candidate 0.10.0 have the current
    parameter layout but an earlier forward (the prediction points' geometry encoding
    before 0.8.0, the interior task's length unit measured on the drawn tokens before
    0.9.0, the geometry before its robustness bundle in 0.11.0, the candidate's head
    frame, and the exact geometry's first-order bands before 0.12.0): their state
    dicts, told apart by the state-dict version torch records (1 before 0.8.0, as every
    0.7.0 checkpoint carries, 8 for 0.8.0, 9 for 0.9.0, 10 for the candidate and 11 for
    0.11.0), are refused by ``Module.load``, ``load_state_dict`` and through
    a DDP-style wrapper, with the version named; the current state dict loads."""
    import io
    import zipfile

    model = _instantiate_model(**CONFIGS["interior"])
    path = tmp_path / "isla.mdlus"
    model.save(path)
    old = model.state_dict()
    assert old._metadata[""]["version"] == ISLA._version == 12
    old._metadata[""]["version"] = version  # what that release wrote
    buf = io.BytesIO()
    torch.save(old, buf)
    stale = tmp_path / f"isla_{release}_weights.mdlus"
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(stale, "w") as zout:
        for item in zin.infolist():
            zout.writestr(
                item,
                buf.getvalue()
                if item.filename == "model.pt"
                else zin.read(item.filename),
            )
    fresh = ISLA(**CONFIGS["interior"])
    with pytest.raises(
        RuntimeError, match=rf"before version 0\.12\.0 \(state-dict version {version}"
    ):
        fresh.load(str(stale), strict=True)
    with pytest.raises(RuntimeError, match="no upgrade path"):
        fresh.load_state_dict(old, strict=False)
    wrapped = torch.nn.ModuleDict({"module": model}).state_dict()
    wrapped._metadata["module"]["version"] = version
    with pytest.raises(RuntimeError, match="before version 0.12.0"):
        torch.nn.ModuleDict({"module": fresh}).load_state_dict(wrapped)
    fresh.load(str(path))  # the current state dict loads
    _assert_dict_close(fresh.state_dict(), model.state_dict(), atol=0.0, rtol=0.0)


def test_isla_unknown_constructor_argument_is_a_type_error():
    with pytest.raises(TypeError, match="never_an_option"):
        ISLA(hidden=32, n_layers=1, n_slices=8, never_an_option=1, **IO)
