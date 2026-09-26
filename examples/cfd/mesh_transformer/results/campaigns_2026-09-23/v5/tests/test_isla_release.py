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
non-regression against golden outputs stored next to this file; checkpoint save and
``Module.from_checkpoint`` round trip; a committed reference ``.mdlus``; and the
pre-refactor backbone golden.

**Regenerating the goldens** (the rank-spec API of 2026-09-22 produced the current
``isla_*_v1.*`` files): delete ``data/isla_default_v1.pth``,
``data/isla_query_independent_v1.pth``, ``data/isla_default_v1.mdlus`` and
``data/isla_default_checkpoint_output_v1.pth`` and run

    uv run --no-sync python -m pytest test/experimental/models/isla/test_isla_release.py

twice: the first run recreates each missing file from the current code (and fails on
purpose), the second run must pass. ``data/isla_backbone_v1.pth`` is NOT regenerated:
it was recorded from the pre-refactor code (a forward hook on ``norm_out`` under
``CONFIGS`` / ``_make_inputs`` below) and pins the backbone bitwise.
"""

from pathlib import Path

import pytest
import torch

from physicsnemo.core.module import Module
from physicsnemo.experimental.models import ISLA

DATA_DIR = Path(__file__).parent / "data"

#: Every configuration declares one scalar and one vector output and one global direction.
IO = dict(outputs={"p": 0, "v": 1}, global_inputs={"g": 1})

#: Small configurations that exercise the two forward paths users see: the
#: boundary-to-boundary model and the query-independent (interior) decoder.
CONFIGS = {
    "default": dict(hidden=32, n_layers=2, n_slices=8, geo_kernel="eager", **IO),
    "query_independent": dict(
        hidden=32,
        n_layers=2,
        n_slices=8,
        query_independent=True,
        n_decoder_layers=2,
        geo_kernel="eager",
        interior_geometry="provided",
        **IO,
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
    inputs = dict(points=pts, normals=nrm, measure_weights=w)
    if query:
        inputs["query_points"] = torch.randn(b, 40, 3, generator=gen) * 2.0
        # the goldens predate the internal geometry: the interior configuration is
        # interior_geometry="provided" and the queries carry a direction
        inputs["query_normals"] = torch.nn.functional.normalize(
            torch.randn(b, 40, 3, generator=gen), dim=-1
        )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["global_inputs"] = {"g": g.to(device)}
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
        assert list(m.output_specs) == ["p", "v"]
        assert (m.output_specs["p"].rank, m.output_specs["v"].rank) == (0, 1)
        assert m.n_head_coeffs == 1 + 5  # a scalar and a true vector over m = 5
        assert m.n_slices == 256
        assert len(m.blocks) == 12
        assert m.norm_out.normalized_shape == (384,)  # the reference width
        assert m.scale_mode == "rms_distance"
        assert m.geo_kernel == "auto"
        assert m.use_measure_weights is True
        assert m.n_global_vectors == 1 and m.n_global_scalars == 0
        assert m.boundary_specs == {} and m.query_specs == {}
        assert m.query_independent is False
        assert m.query_tokens is False and m.support_tokens is False
        assert m.interior_geometry == "nearest_boundary"
        assert m.eps == 1e-12
    else:
        m = ISLA(
            outputs={
                "p": 0,
                "q": 0,
                "nut": 0,
                "u": 1,
                "tau": 1,
                "stress": {"rank": 2, "symmetric": True},
            },
            global_inputs={"g1": 1, "g2": 1, "re": 0},
            boundary_inputs={"trace": {"rank": 0, "channels": 2}, "traction": 1},
            query_inputs={"sdf": 0},
            hidden=64,
            n_layers=3,
            n_slices=16,
            scale_mode="total_measure",
            geo_kernel="eager",
            query_independent=True,
            n_decoder_layers=2,
            local_readout_rho=0.05,
            interior_geometry="provided",
            eps=1e-8,
        )
        assert len(m.output_specs) == 6 and m.output_specs["stress"].symmetric
        assert m.n_slices == 16
        assert len(m.blocks) == 3 and len(m.read_blocks) == 2
        assert m.scale_mode == "total_measure"
        assert m.geo_kernel == "eager"
        assert m.n_global_vectors == 2 and m.n_global_scalars == 1
        assert list(m.boundary_specs) == ["trace", "traction"]
        assert m.boundary_specs["trace"].channels == 2
        assert list(m.query_specs) == ["sdf"]
        assert m.query_independent is True
        assert m.local_readout_rho == 0.05
        assert m.interior_geometry == "provided"
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
    assert out["p"].shape == (2, n_out) and out["v"].shape == (2, n_out, 3)
    ref = _load_or_create_reference(f"isla_{config}_v1.pth", out)
    tol = 1e-5 if device == "cpu" else 1e-4
    _assert_dict_close(out, ref, atol=tol, rtol=tol)


@pytest.mark.parametrize("config", list(CONFIGS))
def test_isla_backbone_is_the_pre_refactor_backbone(config):
    """The rank-spec API (2026-09-22) changed the heads and the input grammar only:
    with no per-token inputs, the pre-head token state (``norm_out`` output) is
    bitwise the one the pre-refactor code produced for the same parameters and
    inputs. ``data/isla_backbone_v1.pth`` was recorded from that code with a forward
    hook on ``norm_out`` (seed 0 parameters, seed 1234 inputs, as here)."""
    ref = torch.load(DATA_DIR / "isla_backbone_v1.pth", weights_only=True)[config]
    model = _instantiate_model(**CONFIGS[config])
    captured = {}
    model.norm_out.register_forward_hook(
        lambda mod, i, o: captured.setdefault("x", o.detach().clone())
    )
    with torch.no_grad():
        model(**_make_inputs("cpu", query=(config == "query_independent")))
    assert torch.equal(captured["x"], ref["norm_out"])


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
        "scale_mode",
        "geo_kernel",
        "query_independent",
        "n_global_vectors",
        "output_specs",
        "global_specs",
    ):
        assert getattr(loaded, name) == getattr(model, name), name
    with torch.no_grad():
        _assert_dict_close(loaded(**inputs), model(**inputs), atol=0.0, rtol=0.0)


def test_isla_reference_checkpoint_matches_golden(device):
    """The committed reference checkpoint keeps loading and keeps its outputs."""
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
        and list(model.output_specs) == ["p", "v"]
    )
    inputs = _make_inputs(device)
    with torch.no_grad():
        out = model(**inputs)
    ref = _load_or_create_reference("isla_default_checkpoint_output_v1.pth", out)
    tol = 1e-5 if device == "cpu" else 1e-4
    _assert_dict_close(out, ref, atol=tol, rtol=tol)


def test_isla_unknown_constructor_argument_is_a_type_error():
    with pytest.raises(TypeError, match="never_an_option"):
        ISLA(hidden=32, n_layers=1, n_slices=8, never_an_option=1, **IO)
