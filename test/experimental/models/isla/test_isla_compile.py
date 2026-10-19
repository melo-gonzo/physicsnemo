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
a sum of three independent counts. The inputs task is an interior task with a
rank-2 boundary input and rank-0 and rank-1 query inputs. ISLA-DTT
(``query_independent``) is not here: its boundary read sizes its pairs from the data
and compiles per block (``compile_blocks``) instead.
"""

import copy

import pytest
import torch
from tensordict import TensorDict

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
#: The conditions of the interior and inputs tasks, each with its own fields (the
#: wall of the interior task with none).
SCHEMAS = {
    "interior": {"inlet": {"bc": {"rank": 0}, "u_bc": {"rank": 1}}, "wall": {}},
    "inputs": {"inlet": {"T": {"rank": 2}}, "wall": {"bc": {"rank": 0}}},
}


def _model(task: str) -> ISLA:
    torch.manual_seed(0)
    if task == "surface":
        kw = {"boundary_schemas": {"wall": {}}, "global_schema": {"g": {"rank": 1}}}
    elif task == "interior":
        kw = {
            "boundary_schemas": SCHEMAS["interior"],
            "global_schema": {"g": {"rank": 1}, "re": {"rank": 0}},
            "interior": True,
        }
    else:  # inputs
        kw = {
            "boundary_schemas": SCHEMAS["inputs"],
            "global_schema": {"g": {"rank": 1}},
            "query_schema": {"sdf": {"rank": 0}, "m": {"rank": 1}},
            "interior": True,
        }
    return ISLA(
        output_schema=PV, hidden=32, n_layers=2, n_slices=8, **kw
    )


def _case(task: str, sizes: dict, nq: int, device, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    if task == "surface":
        sizes = {"wall": sum(sizes.values())}
    meshes = {}
    for k, (name, n) in enumerate(sorted(sizes.items())):
        pts, nrm, w, _ = cloud(n, seed=seed + k, dtype=torch.float32)
        schema = SCHEMAS[task][name] if task in SCHEMAS else {}
        data = {f: torch.randn(n, *(3,) * spec["rank"], generator=g) for f, spec in schema.items()}
        meshes[name] = soup(pts, nrm, w, cell_data=data).to(device)
    drv = torch.nn.functional.normalize(torch.randn(3, generator=g), dim=0)
    case = {"boundary_meshes": meshes, "global_data": {"g": drv.to(device)}}

    if task == "interior":
        case["global_data"]["re"] = torch.tensor(0.3, device=device)
        case["prediction_points"] = (2 * torch.randn(nq, 3, generator=g)).to(device)
    elif task == "inputs":
        x = 2 * torch.randn(nq, 3, generator=g)
        fields = {"sdf": torch.randn(nq, generator=g), "m": torch.randn(nq, 3, generator=g)}
        case["prediction_points"] = points_mesh(x, fields).to(device)
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
@pytest.mark.parametrize("task", ["surface", "interior", "inputs"])
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
        case = _case(task, sizes, nq, device, seed=i)
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


def _reversed_keys(td) -> None:
    """Re-insert the leaves of ``td`` in reverse key order, in place."""
    items = {k: td[k] for k in td.keys()}
    for k in items:
        del td[k]
    for k in reversed(list(items)):
        td[k] = items[k]


def test_whole_model_does_not_guard_on_the_key_order_of_its_inputs():
    """Datasets add fields at different stages, so a field's position among the
    keys varies by case (AeroZoo's boundaries get the effective measure before or
    after their other fields): boundary cell_data and global_data with their keys
    in another order must reuse the graph."""
    torch._dynamo.reset()
    compiled = torch.compile(
        _model("interior"), backend="eager", fullgraph=True, dynamic=True
    )
    for i, (sizes, nq) in enumerate(SIZES):
        case = _case("interior", sizes, nq, "cpu", seed=i)
        case["global_data"] = TensorDict(case["global_data"])
        if i > 0:
            for mesh in case["boundary_meshes"].values():
                _reversed_keys(mesh.cell_data)
            _reversed_keys(case["global_data"])
        with torch._dynamo.config.patch(error_on_recompile=i > 0):
            compiled(**case)


def test_the_in_model_boundary_draw_compiles_once_across_the_budget():
    """An interior model whose boundary_tokens (60) lies between the cases' cell
    counts (64, 111 and 50 cells over two conditions): the compiled model, in
    evaluation (the fixed start), gives the eager outputs with one graph, whether
    the cells are sampled down to the budget or all kept. In training (a start from
    the global RNG inside the graph) it compiles once more and then reuses that
    graph.

    The token count is ``Min(budget, cells)``. Inductor's FX-graph cache, replaying
    a cached graph whose guards hold that ``Min`` (here the CPU code generator's
    accumulator test, token count at most 4,096), splits it into a guard on one side
    of the budget, so the test compiles without that cache."""
    torch._dynamo.reset()
    model = ISLA(
        output_schema=PV,
        boundary_schemas=SCHEMAS["interior"],
        global_schema={"g": {"rank": 1}, "re": {"rank": 0}},
        interior=True,
        boundary_tokens=60,
        hidden=32,
        n_layers=2,
        n_slices=8,
    ).eval()
    eager = copy.deepcopy(model)  # never called under compile: no shared module state
    compiled = torch.compile(model, fullgraph=True, dynamic=True)
    sizes = [({"inlet": 23, "wall": 41}, 57), ({"inlet": 31, "wall": 80}, 45), ({"inlet": 20, "wall": 30}, 40)]
    no_cache = torch._inductor.config.patch(fx_graph_cache=False)
    for i, (s, nq) in enumerate(sizes):
        case = _case("interior", s, nq, "cpu", seed=i)
        with torch.no_grad(), no_cache, torch._dynamo.config.patch(error_on_recompile=i > 0):
            out = outputs(compiled(**case))
        with torch.no_grad():
            ref = outputs(eager(**case))
        for k in ref:
            _assert_close(out[k], ref[k], "cpu")
    model.train()
    for i, (s, nq) in enumerate(sizes):
        case = _case("interior", s, nq, "cpu", seed=i)
        with no_cache, torch._dynamo.config.patch(error_on_recompile=i > 0):
            out = outputs(compiled(**case))
        assert all(torch.isfinite(v).all() for v in out.values())
