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

"""Dot-collapsed nested field names (2026-09-23).

A field name in ``outputs``, ``boundary_inputs``, ``query_inputs`` or
``global_inputs`` may contain ``"."`` to denote nesting, the project's TensorDict
convention (``physicsnemo.datapipes.keys``). Every value of those mappings is a leaf
rank spec, so a mapping value is never a group. ``forward`` returns a TensorDict of
batch size ``(B, N_out)`` whose dotted names are unflattened, and accepts its named
inputs flat with dotted names, nested, or as TensorDicts.
"""

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.experimental.models import ISLA

D = torch.float64


def _cloud(n=80, seed=0, batch=2):
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn(batch, n, 3, dtype=D, generator=g)
    nrm = torch.nn.functional.normalize(
        torch.randn(batch, n, 3, dtype=D, generator=g), dim=-1
    )
    w = torch.rand(batch, n, dtype=D, generator=g) + 0.5
    drv = torch.randn(batch, 3, dtype=D, generator=g)
    return pts, nrm, w, drv


def _model(**kw):
    torch.manual_seed(0)
    base = dict(
        outputs={
            "surface.pressure": 0,
            "surface.wall_shear": 1,
            "solid.stress": {"rank": 2, "symmetric": True},
            "nut": 0,
        },
        global_inputs={"conditions.freestream": 1, "conditions.re": 0},
        boundary_inputs={"bc.trace": 0, "bc.traction": 1},
        hidden=32,
        n_layers=1,
        n_slices=8,
        geo_kernel="eager",
    )
    base.update(kw)
    return ISLA(**base).double().eval()


def test_outputs_are_a_nested_tensordict_over_batch_and_tokens():
    m = _model()
    pts, nrm, w, drv = _cloud()
    b, n = pts.shape[:2]
    bi = {
        "bc.trace": torch.rand(b, n, dtype=D),
        "bc.traction": torch.randn(b, n, 3, dtype=D),
    }
    with torch.no_grad():
        out = m(
            points=pts,
            normals=nrm,
            measure_weights=w,
            global_inputs={
                "conditions.freestream": drv,
                "conditions.re": torch.ones(b, dtype=D),
            },
            boundary_inputs=bi,
        )
    assert isinstance(out, TensorDict) and out.batch_size == torch.Size([b, n])
    assert set(out.keys()) == {"surface", "solid", "nut"}
    assert out["surface"]["pressure"].shape == (b, n)
    assert out["surface", "wall_shear"].shape == (b, n, 3)
    assert out["solid", "stress"].shape == (b, n, 3, 3)
    assert out["nut"].shape == (b, n)
    assert sorted(
        ".".join(k) if isinstance(k, tuple) else k
        for k in out.keys(include_nested=True, leaves_only=True)
    ) == [
        "nut",
        "solid.stress",
        "surface.pressure",
        "surface.wall_shear",
    ]
    ### the flat view carries the declared names back
    assert set(out.flatten_keys(".").keys()) == set(m.output_specs)
    ### indexing the batch axis gives one case
    assert out[0]["surface", "pressure"].shape == (n,)


def test_flat_dotted_nested_and_tensordict_inputs_are_the_same_call():
    m = _model()
    pts, nrm, w, drv = _cloud()
    b, n = pts.shape[:2]
    re = torch.ones(b, dtype=D) * 0.3
    trace, trac = torch.rand(b, n, dtype=D), torch.randn(b, n, 3, dtype=D)
    forms = [
        (
            {"conditions.freestream": drv, "conditions.re": re},
            {"bc.trace": trace, "bc.traction": trac},
        ),
        (
            {"conditions": {"freestream": drv, "re": re}},
            {"bc": {"trace": trace, "traction": trac}},
        ),
        (
            TensorDict({"conditions": {"freestream": drv, "re": re}}, batch_size=[b]),
            TensorDict({"bc": {"trace": trace, "traction": trac}}, batch_size=[b, n]),
        ),
    ]
    outs = []
    for gi, bi in forms:
        with torch.no_grad():
            outs.append(
                m(
                    points=pts,
                    normals=nrm,
                    measure_weights=w,
                    global_inputs=gi,
                    boundary_inputs=bi,
                )
            )
    for o in outs[1:]:
        for k in outs[0].keys(include_nested=True, leaves_only=True):
            assert torch.equal(o[k], outs[0][k]), k


def test_mixed_flat_and_nested_names_and_the_interior_paths():
    pts, nrm, w, drv = _cloud(batch=1)
    q = torch.randn(1, 12, 3, dtype=D) * 2.0
    for kw in (
        dict(query_tokens=True),
        dict(query_independent=True, n_decoder_layers=1),
    ):
        m = _model(
            outputs={"pressure": 0, "flow.velocity": 1},
            global_inputs={"freestream": 1},
            boundary_inputs=None,
            query_inputs={"geometry.sdf": 0, "geometry.grad": 1},
            **kw,
        )
        qi = {
            "geometry": {
                "sdf": torch.rand(1, 12, dtype=D),
                "grad": torch.randn(1, 12, 3, dtype=D),
            }
        }
        with torch.no_grad():
            out = m(
                points=pts,
                normals=nrm,
                measure_weights=w,
                global_inputs={"freestream": drv},
                query_points=q,
                query_inputs=qi,
            )
        assert out["pressure"].shape == (1, 12) and out["flow", "velocity"].shape == (
            1,
            12,
            3,
        )


def test_covariance_holds_on_the_nested_values():
    m = _model()
    pts, nrm, w, drv = _cloud(batch=1)
    b, n = pts.shape[:2]
    bi = {
        "bc.trace": torch.rand(b, n, dtype=D),
        "bc.traction": torch.randn(b, n, 3, dtype=D),
    }
    R, _ = torch.linalg.qr(
        torch.randn(3, 3, dtype=D, generator=torch.Generator().manual_seed(1))
    )
    t = torch.tensor([1.0, 2.0, -3.0], dtype=D)

    def call(R, t):
        with torch.no_grad():
            return m(
                points=pts @ R.T + t,
                normals=nrm @ R.T,
                measure_weights=w,
                global_inputs={
                    "conditions.freestream": drv @ R.T,
                    "conditions.re": torch.ones(b, dtype=D),
                },
                boundary_inputs={
                    "bc.trace": bi["bc.trace"],
                    "bc.traction": bi["bc.traction"] @ R.T,
                },
            )

    base, out = call(torch.eye(3, dtype=D), torch.zeros(3, dtype=D)), call(R, t)
    assert torch.allclose(
        out["surface", "pressure"], base["surface", "pressure"], atol=1e-12
    )
    assert torch.allclose(out["nut"], base["nut"], atol=1e-12)
    assert torch.allclose(
        out["surface", "wall_shear"], base["surface", "wall_shear"] @ R.T, atol=1e-12
    )
    assert torch.allclose(
        out["solid", "stress"], R @ base["solid", "stress"] @ R.T, atol=1e-12
    )


def test_name_grammar_is_validated():
    base = dict(hidden=32, n_layers=1, n_slices=8)
    with pytest.raises(ValueError, match="is a group of"):
        ISLA(outputs={"surface": 0, "surface.pressure": 0}, **base)
    with pytest.raises(ValueError, match="non-empty"):
        ISLA(outputs={"surface..pressure": 0}, **base)
    with pytest.raises(ValueError, match="non-empty"):
        ISLA(outputs={"surface.": 0}, **base)
    ### a mapping value is always a rank spec, never a nested group
    with pytest.raises(ValueError, match="unknown rank-spec keys"):
        ISLA(outputs={"surface": {"pressure": 0}}, **base)
    ### forward-time names must match the declared dotted names
    m = ISLA(outputs={"p": 0}, boundary_inputs={"bc.trace": 0}, **base).double()
    pts, nrm, w, _ = _cloud(batch=1)
    with pytest.raises(ValueError, match="missing \\['bc.trace'\\]"):
        m(points=pts, normals=nrm, measure_weights=w, boundary_inputs={"trace": w})
    with pytest.raises(TypeError, match="mapping"):
        m(points=pts, normals=nrm, measure_weights=w, boundary_inputs=w)


def test_checkpoint_round_trip_keeps_dotted_names(tmp_path):
    from physicsnemo.core.module import Module

    m = _model()
    m.save(tmp_path / "m.mdlus")
    m2 = Module.from_checkpoint(str(tmp_path / "m.mdlus")).double().eval()
    assert list(m2.output_specs) == list(m.output_specs)
    assert list(m2.boundary_specs) == ["bc.trace", "bc.traction"]
