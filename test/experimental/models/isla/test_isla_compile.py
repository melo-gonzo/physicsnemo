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
outputs and gradients, and a case with other boundary counts must reuse the graph
rather than recompile. Every model here is a surface task: the surface task has one
boundary mesh; the inputs and scales tasks have two boundary meshes, so their token
axis is a sum of two independent counts, and predict on one of them. The inputs
task has a rank-2 boundary input and a rank-0 one. The interior task (ISLA-DTT) is
not here: its boundary read sizes its pairs from the data and compiles per block
(``compile_blocks``) instead.
"""

import copy

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.experimental.models import ISLA
from test.experimental.models.isla._meshes import cloud, outputs, soup

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
    ),
]
PV = {"p": {"rank": 0}, "v": {"rank": 1}}

#: Per-case token counts, all distinct (and distinct from every other input
#: dimension) so that duck sizing ties no two of them together.
SIZES = [{"inlet": 23, "wall": 41}, {"inlet": 31, "wall": 37}]
#: The conditions of the inputs and scales tasks, each with its own fields.
SCHEMAS = {
    "inputs": {"inlet": {"T": {"rank": 2}}, "wall": {"bc": {"rank": 0}}},
    "scales": {
        "inlet": {
            "bc": {"rank": 0, "units": {"U": 2}},
            "u_bc": {"rank": 1, "units": {"U": 1}, "velocity_scale": True},
        },
        "wall": {},
    },
}


def _model(task: str) -> ISLA:
    torch.manual_seed(0)
    if task == "surface":
        kw = {"boundary_schemas": {"wall": {}}, "global_schema": {"g": {"rank": 1}}}
    elif task == "inputs":
        kw = {
            "boundary_schemas": SCHEMAS["inputs"],
            "global_schema": {"g": {"rank": 1}},
        }
    elif (
        task == "global_source"
    ):  # ISLA's own scales from a global source (AirfRANS's U_inf)
        return ISLA(
            output_schema={
                "p": {"rank": 0},
                "v": {"rank": 1},
                "nut": {"rank": 0, "units": {"U": 1, "L": 1}, "scale": 1e-4},
            },
            boundary_schemas={"wall": {}},
            global_schema={
                "g": {"rank": 1, "units": {"U": 1}, "velocity_scale": True},
                "nu": {"rank": 0, "units": {"U": 1, "L": 1}, "role": "viscosity"},
            },
            reynolds={"mean": 6.0, "std": 0.2},
            output_stats={"nut": {"mean": 1e-3, "std": 2e-3}},
            hidden=32,
            n_layers=2,
            n_slices=8,
        )
    else:  # scales: ISLA's own, with the derived direction, Re and output_stats
        return ISLA(
            output_schema={
                "p": {"rank": 0, "units": {"U": 2, "rho": 1}},
                "v": {"rank": 1, "units": {"U": 1}},
            },
            boundary_schemas=SCHEMAS["scales"],
            global_schema={
                "rho": {"rank": 0, "role": "density"},
                "nu": {"rank": 0, "units": {"U": 1, "L": 1}, "role": "viscosity"},
            },
            reynolds={"mean": 5.0, "std": 1.0},
            output_stats={
                "p": {"mean": 1.0, "std": 30.0},
                "v": {"mean": 0.0, "std": 4.0},
            },
            hidden=32,
            n_layers=2,
            n_slices=8,
        )
    return ISLA(output_schema=PV, hidden=32, n_layers=2, n_slices=8, **kw)


def _case(task: str, sizes: dict, device, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    if task in ("surface", "global_source"):
        sizes = {"wall": sum(sizes.values())}
    meshes = {}
    for k, (name, n) in enumerate(sorted(sizes.items())):
        pts, nrm, w, _ = cloud(n, seed=seed + k, dtype=torch.float32)
        schema = SCHEMAS[task][name] if task in SCHEMAS else {}
        data = {
            f: torch.randn(n, *(3,) * spec["rank"], generator=g)
            for f, spec in schema.items()
        }
        meshes[name] = soup(pts, nrm, w, cell_data=data).to(device)
    drv = torch.nn.functional.normalize(torch.randn(3, generator=g), dim=0)
    case = {"boundary_meshes": meshes, "global_data": {"g": drv.to(device)}}

    if task in ("scales", "global_source"):
        case["global_data"]["rho"] = torch.tensor(1.2, device=device)
        case["global_data"]["nu"] = torch.tensor(1.5e-5, device=device)
    case["prediction_boundaries"] = ["wall"]
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
@pytest.mark.parametrize("task", ["surface", "inputs", "scales", "global_source"])
def test_whole_model_compiles_fullgraph_with_dynamic_shapes_and_matches_eager(
    task, device
):
    """One training step's outputs and gradients, compiled against eager, at two
    sets of token counts with one compilation. On CUDA the step runs as in
    training, under bf16 autocast with the fused geometry kernel, where a token
    axis of several boundary meshes (a sum of counts) needs the model's
    divisibility hints for Inductor to tile it."""
    torch._dynamo.reset()
    model = _model(task).to(device)
    eager = copy.deepcopy(model)  # never called under compile: no shared module state
    compiled = torch.compile(model, fullgraph=True, dynamic=True)
    autocast = device == "cuda"
    for i, sizes in enumerate(SIZES):
        case = _case(task, sizes, device, seed=i)
        ### The second case has other boundary counts: no recompile.
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
        _model("scales"), backend="eager", fullgraph=True, dynamic=True
    )
    for i, sizes in enumerate(SIZES):
        case = _case("scales", sizes, "cpu", seed=i)
        case["global_data"] = TensorDict(case["global_data"])
        if i > 0:
            for mesh in case["boundary_meshes"].values():
                _reversed_keys(mesh.cell_data)
            _reversed_keys(case["global_data"])
        with torch._dynamo.config.patch(error_on_recompile=i > 0):
            compiled(**case)
