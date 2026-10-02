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
``Module.from_checkpoint`` round trip; a committed reference ``.mdlus``; and the
backbone golden.

**Regenerating the goldens**: delete ``data/isla_default_v1.pth``,
``data/isla_interior_v1.pth``, ``data/isla_default_v1.mdlus`` and
``data/isla_default_checkpoint_output_v1.pth`` and run

    uv run --no-sync python -m pytest test/experimental/models/isla/test_isla_release.py

twice: the first run recreates each missing file from the current code (and fails on
purpose), the second run must pass. ``data/isla_backbone_v1.pth`` is regenerated only
when the backbone changes by decision (a forward hook on ``norm_out`` under
``CONFIGS["default"]`` / ``_make_tensors`` below; last recorded 2026-10-02 when the
sample-conditioned prototypes became the routing of every block) and pins the
backbone bitwise through the internal batched routine.
"""

from pathlib import Path

import pytest
import torch

from physicsnemo.core.module import Module
from physicsnemo.experimental.models import ISLA
from physicsnemo.mesh.calculus.measure import cell_measures
from test.experimental.models.isla._meshes import outputs, soup

DATA_DIR = Path(__file__).parent / "data"

#: Every configuration declares one scalar and one vector output and one global direction.
IO = dict(output_schema={"p": {"rank": 0}, "v": {"rank": 1}}, global_schema={"g": {"rank": 1}})

#: One small configuration in its two task structures (a surface task and an interior
#: task; the constructor's ``interior`` flag).
CONFIGS = {
    "default": dict(hidden=32, n_layers=2, n_slices=8, geo_kernel="eager", **IO),
    "interior": dict(hidden=32, n_layers=2, n_slices=8, geo_kernel="eager", interior=True, **IO),
}


def _instantiate_model(seed: int = 0, **kwargs) -> ISLA:
    """Build an ISLA with reproducible random parameters (MOD-008b pattern)."""
    model = ISLA(**kwargs)
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
    inputs = dict(boundary_meshes={"wall": soup(pts, nrm, w).to(device)}, global_data={"g": g.to(device)})
    if query:
        inputs["prediction_points"] = (torch.randn(40, 3, generator=gen) * 2.0).to(device)
    return inputs


def _load_or_create_reference(file_name: str, output: dict) -> dict:
    """Load a golden output dict, or create it and require a second run (sibling-model pattern)."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / file_name
    if not path.exists():
        torch.save({"output": {k: v.detach().cpu() for k, v in output.items()}}, path)
        raise IOError(f"Golden output {path} was missing and has been created; commit it and re-run.")
    ref = torch.load(path, weights_only=True)["output"]
    return {k: v.to(next(iter(output.values())).device) for k, v in ref.items()}


def _assert_dict_close(out: dict, ref: dict, atol: float, rtol: float) -> None:
    assert list(out.keys()) == list(ref.keys())
    for k in out.keys():
        torch.testing.assert_close(out[k], ref[k], atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# MOD-008a: constructor and public attributes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", ["minimal", "custom"], ids=["with_defaults", "with_custom_args"])
def test_isla_constructor_attributes(config):
    if config == "minimal":
        m = ISLA(**IO)
        assert list(m.output_schema) == ["p", "v"]
        assert (m.output_schema["p"].rank, m.output_schema["v"].rank) == (0, 1)
        assert m.n_head_coeffs == 1 + 5  # a scalar and a true vector over m = 5
        assert m.n_slices == 256
        assert len(m.blocks) == 12
        assert m.norm_out.normalized_shape == (384,)  # the reference width
        assert m.geo_kernel == "auto"
        assert m.interior is False and m.qt is None
        assert m.n_global_vectors == 1 and m.n_global_scalars == 0
        assert m.boundary_schema == {} and m.query_schema == {}
        assert m.eps == 1e-12
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
            boundary_schema={"trace": {"rank": 0}, "traction": {"rank": 1}},
            query_schema={"sdf": {"rank": 0}},
            interior=True,
            hidden=64,
            n_layers=3,
            n_slices=16,
            geo_kernel="eager",
            geo_checkpoint=True,
            eps=1e-8,
        )
        assert len(m.output_schema) == 6 and m.output_schema["stress"].symmetric
        assert m.n_slices == 16
        assert len(m.blocks) == 3 and all(b.geo_checkpoint for b in m.blocks)
        assert m.geo_kernel == "eager"
        assert m.n_global_vectors == 2 and m.n_global_scalars == 1
        assert list(m.boundary_schema) == ["trace", "traction"]
        assert list(m.query_schema) == ["sdf"]
        assert m.interior is True and m.qt is not None
        assert m.eps == 1e-8
    assert isinstance(m, Module)


# ---------------------------------------------------------------------------
# MOD-008b: forward non-regression against stored golden outputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", list(CONFIGS))
def test_isla_forward_non_regression(device, config):
    model = _instantiate_model(**CONFIGS[config]).to(device)
    inputs = _make_inputs(device, query=(config == "interior"))
    with torch.no_grad():
        out = outputs(model(**inputs))
    n_out = 40 if config == "interior" else 64
    assert out["p"].shape == (n_out,) and out["v"].shape == (n_out, 3)
    ref = _load_or_create_reference(f"isla_{config}_v1.pth", out)
    tol = 1e-5 if device == "cpu" else 1e-4
    _assert_dict_close(out, ref, atol=tol, rtol=tol)


@pytest.mark.parametrize("config", list(CONFIGS))
def test_mesh_wrapper_is_exact_over_the_batched_routine(device, config):
    """The Mesh API is a wrapper: the same parameters run through ``forward`` on the
    meshes and through the internal batched routine on the very tensors the wrapper
    reads (cell centroids, cell normals, effective cell measures, a batch axis of
    one) give bitwise-identical outputs."""
    model = _instantiate_model(**CONFIGS[config]).to(device)
    inputs = _make_inputs(device, query=(config == "interior"))
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
        )
    expected_points = mesh.cell_centroids if q is None else q
    assert torch.equal(via_mesh.points, expected_points) and via_mesh.n_cells == 0
    for k, v in direct.items():
        assert torch.equal(via_mesh.point_data[k], v[0]), k
    assert torch.equal(via_mesh.global_data["g"], inputs["global_data"]["g"])


def test_isla_backbone_is_the_pre_refactor_backbone():
    """With no per-token inputs, the pre-head token state (``norm_out`` output) of
    the surface model is the one recorded when the backbone was last changed by
    decision, for the same parameters and token tensors. ``data/isla_backbone_v1.pth``
    was recorded with a forward hook on ``norm_out`` (seed 0 parameters, seed 1234
    inputs, as here) on an x86 CPU; the tensors enter through the internal batched
    routine, which the Mesh wrapper calls. Compared at float32 tolerance: the same
    graph gives bitwise-identical values on the recording architecture and differs
    at the last bits across CPU architectures (BLAS reduction order), which is not a
    property of the model."""
    ref = torch.load(DATA_DIR / "isla_backbone_v1.pth", weights_only=True)["default"]
    model = _instantiate_model(**CONFIGS["default"])
    captured = {}
    model.norm_out.register_forward_hook(lambda mod, i, o: captured.setdefault("x", o.detach().clone()))
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
    for name in ("n_slices", "geo_kernel", "interior", "n_global_vectors", "output_schema", "global_schema"):
        assert getattr(loaded, name) == getattr(model, name), name
    with torch.no_grad():
        _assert_dict_close(outputs(loaded(**inputs)), outputs(model(**inputs)), atol=0.0, rtol=0.0)


def test_isla_reference_checkpoint_matches_golden(device):
    """The committed reference checkpoint keeps loading and keeps its outputs."""
    DATA_DIR.mkdir(exist_ok=True)
    ckpt = DATA_DIR / "isla_default_v1.mdlus"
    if not ckpt.exists():
        _instantiate_model(seed=7, **CONFIGS["default"]).save(ckpt)
        raise IOError(f"Reference checkpoint {ckpt} was missing and has been created; commit it and re-run.")
    model = Module.from_checkpoint(str(ckpt)).to(device).eval()
    assert isinstance(model, ISLA)
    assert model.n_slices == 8 and model.geo_kernel == "eager" and list(model.output_schema) == ["p", "v"]
    assert model.interior is False
    inputs = _make_inputs(device)
    with torch.no_grad():
        out = outputs(model(**inputs))
    ref = _load_or_create_reference("isla_default_checkpoint_output_v1.pth", out)
    tol = 1e-5 if device == "cpu" else 1e-4
    _assert_dict_close(out, ref, atol=tol, rtol=tol)


def test_isla_0_1_0_checkpoint_declarations_are_renamed(tmp_path):
    """A 0.1.0 checkpoint (``outputs``, ``global_inputs``, ...) loads through the
    backward-compatibility mapper, which renames the field declarations to
    ``output_schema``, ``global_schema`` and so on. The fixture is the committed
    reference checkpoint rewritten to the 0.1.0 declarations and file version, so
    the mapper is exercised against the current architecture's weights."""
    import json
    import zipfile

    src = DATA_DIR / "isla_default_v1.mdlus"
    old = tmp_path / "isla_0_1_0.mdlus"
    renamed = {new: old_name for old_name, new in ISLA._RENAMED_ARGS.items()}
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(old, "w", zipfile.ZIP_STORED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "args.json":
                args = json.loads(data)
                args["__args__"] = {renamed.get(k, k): v for k, v in args["__args__"].items()}
                data = json.dumps(args).encode()
            elif item.filename == "metadata.json":
                meta = json.loads(data)
                meta["mdlus_file_version"] = "0.1.0"
                data = json.dumps(meta).encode()
            zout.writestr(item, data)
    with pytest.warns(UserWarning, match="0.1.0 ISLA checkpoint"):
        model = Module.from_checkpoint(str(old))
    assert list(model.output_schema) == ["p", "v"] and list(model.global_schema) == ["g"]
    assert model.boundary_schema == {} and model.query_schema == {}
    assert ISLA.__model_checkpoint_version__ == "0.2.0"


def test_isla_unknown_constructor_argument_is_a_type_error():
    with pytest.raises(TypeError, match="never_an_option"):
        ISLA(hidden=32, n_layers=1, n_slices=8, never_an_option=1, **IO)
