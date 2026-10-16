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

"""Whole-model torch.compile of ISLA: ``fullgraph=True`` with dynamic shapes.

The compiled model (Mesh inputs and output included) must match the eager model,
outputs and gradients, and a case with other boundary and query counts must reuse
the graph rather than recompile. The surface task has one boundary mesh; the
interior task has two boundary meshes and prediction points, so its token axis is
a sum of three independent counts. The support task adds support points (a fourth
count) and a rank-2 boundary input; the passive task's prediction points are
passive readers (``query_independent``) with a rank-1 query input.
"""

import copy

import pytest
import torch

from physicsnemo.experimental.models import ISLA
from test.experimental.models.isla._meshes import cloud, outputs, points_mesh, soup

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
    ),
]
PV = {"p": {"rank": 0}, "v": {"rank": 1}}

#: Per-case token counts, all distinct (and distinct from every other input
#: dimension) so that duck sizing ties no two of them together.
SIZES = [({"inlet": 23, "wall": 41}, 57), ({"inlet": 31, "wall": 37}, 45)]
#: Support points per case (the support task).
N_SUPPORT = [19, 29]


def _model(task: str) -> ISLA:
    torch.manual_seed(0)
    if task == "surface":
        kw = {"global_schema": {"g": {"rank": 1}}}
    elif task == "interior":
        kw = {
            "boundary_schema": {"bc": {"rank": 0}, "u_bc": {"rank": 1}},
            "global_schema": {"g": {"rank": 1}, "re": {"rank": 0}},
            "interior": True,
        }
    elif task == "support":
        kw = {
            "boundary_schema": {"bc": {"rank": 0}, "T": {"rank": 2}},
            "global_schema": {"g": {"rank": 1}},
            "query_schema": {"sdf": {"rank": 0}},
            "interior": True,
        }
    else:  # passive
        kw = {
            "global_schema": {"g": {"rank": 1}},
            "query_schema": {"m": {"rank": 1}},
            "interior": True,
            "query_independent": True,
        }
    return ISLA(
        output_schema=PV, hidden=32, n_layers=2, n_slices=8, geo_checkpoint=True, **kw
    )


def _case(
    task: str, sizes: dict, nq: int, device, seed: int, n_support: int = 0
) -> dict:
    g = torch.Generator().manual_seed(seed)
    if task == "surface":
        sizes = {"wall": sum(sizes.values())}
    meshes = {}
    for k, (name, n) in enumerate(sorted(sizes.items())):
        pts, nrm, w, _ = cloud(n, seed=seed + k, dtype=torch.float32)
        data = {}
        if task == "interior":
            data = {
                "bc": torch.rand(n, generator=g),
                "u_bc": torch.randn(n, 3, generator=g),
            }
        elif task == "support":
            data = {
                "bc": torch.rand(n, generator=g),
                "T": torch.randn(n, 3, 3, generator=g),
            }
        meshes[name] = soup(pts, nrm, w, cell_data=data).to(device)
    drv = torch.nn.functional.normalize(torch.randn(3, generator=g), dim=0)
    case = {"boundary_meshes": meshes, "global_data": {"g": drv.to(device)}}

    def points(n):
        x = 2 * torch.randn(n, 3, generator=g)
        if task == "support":
            return points_mesh(x, {"sdf": torch.randn(n, generator=g)}).to(device)
        if task == "passive":
            return points_mesh(x, {"m": torch.randn(n, 3, generator=g)}).to(device)
        return x.to(device)

    if task == "interior":
        case["global_data"]["re"] = torch.tensor(0.3, device=device)
    if task != "surface":
        case["prediction_points"] = points(nq)
    if task == "support":
        case["support_points"] = points(n_support)
    return case


def _step(fn, model, case, autocast: bool):
    """One forward and backward: the outputs and every parameter's gradient."""
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        out = {k: t.float() for k, t in outputs(fn(**case)).items()}
    sum(t.square().mean() for t in out.values()).backward()
    return out, {
        k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None
    }


def _assert_close(a, b, device, rel=3e-2):
    if device == "cpu":  # float32: Inductor's reordered reductions only
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
    else:  # bfloat16 autocast, as in training: to its roundoff, in norm
        assert (a - b).norm() <= rel * b.norm() + 1e-6


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("task", ["surface", "interior", "support", "passive"])
def test_whole_model_compiles_fullgraph_with_dynamic_shapes_and_matches_eager(
    task, device
):
    """One training step's outputs and gradients, compiled against eager, at two
    sets of token counts with one compilation. On CUDA the step runs as in
    training, under bf16 autocast with the fused geometry kernel, where the
    interior task's token axis (a sum of three counts) needs the model's
    divisibility hints for Inductor to tile it."""
    torch._dynamo.reset()
    model = _model(task).to(device)
    eager = copy.deepcopy(model)  # never called under compile: no shared module state
    compiled = torch.compile(model, fullgraph=True, dynamic=True)
    autocast = device == "cuda"
    for i, (sizes, nq) in enumerate(SIZES):
        case = _case(task, sizes, nq, device, seed=i, n_support=N_SUPPORT[i])
        ### The second case has other boundary and query counts: no recompile.
        with torch._dynamo.config.patch(error_on_recompile=i > 0):
            out, grads = _step(compiled, model, case, autocast)
        ref, ref_grads = _step(eager, eager, case, autocast)
        assert out.keys() == ref.keys()
        for k in ref:
            _assert_close(out[k], ref[k], device)
        assert grads.keys() == ref_grads.keys() and grads
        if device == "cpu":
            for k in ref_grads:
                _assert_close(grads[k], ref_grads[k], device)
        else:  # small gradients carry bf16's relative roundoff: compare them all at once
            flat = lambda g: torch.cat([g[k].flatten() for k in sorted(g)])  # noqa: E731
            _assert_close(flat(grads), flat(ref_grads), device, rel=5e-2)
