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

Constructor and public attributes for a default and a custom configuration; forward
non-regression against golden outputs stored next to this file; checkpoint save and
``Module.from_checkpoint`` round trip; a committed reference ``.mdlus``; and loading of
checkpoints written under the class's former name with retired research options at
their former defaults.
"""

import io
import json
import zipfile
from pathlib import Path

import pytest
import torch

from physicsnemo.core.module import Module
from physicsnemo.experimental.nn.isla import ISLA

DATA_DIR = Path(__file__).parent / "data"

#: Small configurations that exercise the three forward paths users see: the default
#: boundary-to-boundary model, the centered frame with a fixed reference length, and
#: the query-independent (interior) decoder.
CONFIGS = {
    "default": dict(hidden=32, n_layers=2, n_slices=8, geo_kernel="eager"),
    "centered": dict(
        hidden=32,
        n_layers=2,
        n_slices=8,
        frame_mode="centered",
        scale_mode="reference_length",
        reference_length=4.0,
        geo_kernel="eager",
    ),
    "query_independent": dict(
        hidden=32,
        n_layers=2,
        n_slices=8,
        query_independent=True,
        n_decoder_layers=2,
        geo_kernel="eager",
    ),
}


def _instantiate_model(seed: int = 0, **kwargs) -> ISLA:
    """Build an ISLA with reproducible random parameters (MOD-008b pattern)."""
    model = ISLA(**kwargs)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * 0.1)
    return model.eval()


def _make_inputs(device, query: bool = False, seed: int = 1234) -> dict:
    """Deterministic boundary inputs: two cases, 64 cells, unit normals, positive measures."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    b, n = 2, 64
    pts = torch.randn(b, n, 3, generator=gen) * torch.tensor([3.0, 2.0, 1.0])
    nrm = torch.nn.functional.normalize(torch.randn(b, n, 3, generator=gen), dim=-1)
    w = torch.rand(b, n, generator=gen) + 0.5
    g = torch.nn.functional.normalize(torch.randn(b, 1, 3, generator=gen), dim=-1)
    inputs = dict(points=pts, normals=nrm, measure_weights=w, global_vectors=g)
    if query:
        inputs["query_points"] = torch.randn(b, 40, 3, generator=gen) * 2.0
        # interior queries carry a direction (the SDF gradient in the recipe)
        inputs["query_normals"] = torch.nn.functional.normalize(
            torch.randn(b, 40, 3, generator=gen), dim=-1
        )
    return {k: v.to(device) for k, v in inputs.items()}


def _load_or_create_reference(file_name: str, output: torch.Tensor) -> torch.Tensor:
    """Load a golden output, or create it and require a second run (sibling-model pattern)."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / file_name
    if not path.exists():
        torch.save({"output": output.detach().cpu()}, path)
        raise IOError(
            f"Golden output {path} was missing and has been created; commit it and re-run."
        )
    return torch.load(path, weights_only=True)["output"].to(output.device)


# ---------------------------------------------------------------------------
# MOD-008a: constructor and public attributes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config", ["default", "custom"], ids=["with_defaults", "with_custom_args"]
)
def test_isla_constructor_attributes(config):
    if config == "default":
        m = ISLA()
        assert m.out_scalars == 1 and m.out_vectors == 1
        assert m.n_slices == 256
        assert len(m.blocks) == 12
        assert m.frame_mode == "relative" and m.relative_frame is True
        assert m.scale_mode == "rms_distance"
        assert m.geo_kernel == "auto"
        assert m.use_measure_weights is True
        assert m.n_global_vectors == 1 and m.n_global_scalars == 0
        assert m.n_boundary_scalars == 0
        assert m.query_independent is False
        assert m.query_tokens is False and m.support_tokens is False
        assert m.similarity_gauge is False
        assert m.eps == 1e-12
    else:
        m = ISLA(
            out_scalars=3,
            out_vectors=2,
            hidden=64,
            n_layers=3,
            n_slices=16,
            frame_mode="centered",
            scale_mode="reference_length",
            reference_length=2.5,
            geo_kernel="eager",
            n_global_vectors=2,
            n_global_scalars=1,
            n_boundary_scalars=2,
            query_independent=True,
            n_decoder_layers=2,
            local_readout_rho=0.05,
            eps=1e-8,
        )
        assert m.out_scalars == 3 and m.out_vectors == 2
        assert m.n_slices == 16
        assert len(m.blocks) == 3 and len(m.read_blocks) == 2
        assert m.frame_mode == "centered" and m.relative_frame is False
        assert m.scale_mode == "reference_length" and m.reference_length == 2.5
        assert m.geo_kernel == "eager"
        assert m.n_global_vectors == 2 and m.n_global_scalars == 1
        assert m.n_boundary_scalars == 2
        assert m.query_independent is True
        assert m.local_readout_rho == 0.05
        assert m.eps == 1e-8
    assert isinstance(m, Module)


# ---------------------------------------------------------------------------
# MOD-008b: forward non-regression against stored golden outputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", list(CONFIGS))
def test_isla_forward_non_regression(device, config):
    model = _instantiate_model(**CONFIGS[config]).to(device)
    inputs = _make_inputs(device, query=(config == "query_independent"))
    with torch.no_grad():
        out = model(**inputs)
    n_out = inputs["query_points"].shape[1] if "query_points" in inputs else 64
    assert out.shape == (2, n_out, 4)
    ref = _load_or_create_reference(f"isla_{config}_v1.pth", out)
    tol = 1e-5 if device == "cpu" else 1e-4
    torch.testing.assert_close(out, ref, atol=tol, rtol=tol)


# ---------------------------------------------------------------------------
# MOD-008c: checkpoint save / from_checkpoint round trip and a committed reference
# ---------------------------------------------------------------------------


def test_isla_save_from_checkpoint_round_trip(device, tmp_path):
    model = _instantiate_model(seed=3, **CONFIGS["query_independent"]).to(device)
    inputs = _make_inputs(device, query=True)
    path = tmp_path / "isla_qi.mdlus"
    model.save(path)
    loaded = Module.from_checkpoint(str(path)).to(device).eval()
    assert isinstance(loaded, ISLA)
    for name in (
        "n_slices",
        "frame_mode",
        "scale_mode",
        "geo_kernel",
        "query_independent",
        "n_global_vectors",
    ):
        assert getattr(loaded, name) == getattr(model, name), name
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(**inputs), model(**inputs), atol=0.0, rtol=0.0
        )


def test_isla_reference_checkpoint_matches_golden(device):
    """The committed default-configuration checkpoint keeps loading and keeps its outputs."""
    DATA_DIR.mkdir(exist_ok=True)
    ckpt = DATA_DIR / "isla_default_v1.mdlus"
    if not ckpt.exists():
        _instantiate_model(seed=7, **CONFIGS["default"]).save(ckpt)
        raise IOError(
            f"Reference checkpoint {ckpt} was missing and has been created; commit it and re-run."
        )
    model = Module.from_checkpoint(str(ckpt)).to(device).eval()
    assert isinstance(model, ISLA)
    assert (
        model.n_slices == 8
        and model.scale_mode == "rms_distance"
        and model.geo_kernel == "eager"
    )
    inputs = _make_inputs(device)
    with torch.no_grad():
        out = model(**inputs)
    ref = _load_or_create_reference("isla_default_checkpoint_output_v1.pth", out)
    tol = 1e-5 if device == "cpu" else 1e-4
    torch.testing.assert_close(out, ref, atol=tol, rtol=tol)


# ---------------------------------------------------------------------------
# Checkpoints written by the research class under its former name
# ---------------------------------------------------------------------------


def _rewrite_checkpoint(
    src: Path, dst: Path, *, name=None, module=None, extra_args=None, version=None
):
    """Copy a .mdlus archive, editing its args.json (class name, module, constructor args) and metadata version."""
    with zipfile.ZipFile(src, "r") as zin:
        members = {m: zin.read(m) for m in zin.namelist()}
    args = json.loads(members["args.json"])
    if name is not None:
        args["__name__"] = name
    if module is not None:
        args["__module__"] = module
    if extra_args:
        args["__args__"].update(extra_args)
    members["args.json"] = json.dumps(args).encode()
    if version is not None:
        meta = json.loads(members["metadata.json"])
        meta["mdlus_file_version"] = version
        members["metadata.json"] = json.dumps(meta).encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zout:
        for m, data in members.items():
            zout.writestr(m, data)
    dst.write_bytes(buf.getvalue())


def test_isla_loads_legacy_named_checkpoint_with_former_default_options(
    device, tmp_path
):
    """A checkpoint saved as MeshTransformer2 with retired research options at their former
    defaults describes this very network and loads through ISLA.from_checkpoint."""
    model = _instantiate_model(seed=5, **CONFIGS["default"]).to(device)
    inputs = _make_inputs(device)
    src, dst = tmp_path / "current.mdlus", tmp_path / "legacy.mdlus"
    model.save(src)
    _rewrite_checkpoint(
        src,
        dst,
        name="MeshTransformer2",
        module="physicsnemo.experimental.nn.mt2.model",
        extra_args={
            "odd_head": False,
            "vector_basis": "globe7",
            "local_radii": [0.01, 0.03],
        },
        version="0.1.0",
    )
    with pytest.warns(UserWarning):
        loaded = ISLA.from_checkpoint(str(dst)).to(device).eval()
    assert isinstance(loaded, ISLA)
    assert "odd_head" not in loaded._args["__args__"]
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(**inputs), model(**inputs), atol=0.0, rtol=0.0
        )


def test_isla_refuses_legacy_checkpoint_with_non_default_option(device, tmp_path):
    """A retired option at a non-default value cannot be reproduced and must raise, naming the option."""
    model = _instantiate_model(seed=5, **CONFIGS["default"]).to(device)
    src, dst = tmp_path / "current.mdlus", tmp_path / "legacy_bad.mdlus"
    model.save(src)
    _rewrite_checkpoint(src, dst, extra_args={"odd_head": True}, version="0.1.0")
    with pytest.raises(ValueError, match="odd_head"), pytest.warns(UserWarning):
        ISLA.from_checkpoint(str(dst))


def test_isla_unknown_constructor_argument_is_a_type_error():
    with pytest.raises(TypeError, match="never_an_option"):
        ISLA(hidden=32, n_layers=1, n_slices=8, never_an_option=1)
