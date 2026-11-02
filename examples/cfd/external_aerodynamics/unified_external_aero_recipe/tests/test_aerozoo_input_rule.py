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

"""AeroZoo v1.2, the input rule at the model's output: for every AeroZoo model config (each model on each arm, ISLA
included), the prediction is bitwise the same with every label perturbed or deleted. A mesh model (GLOBE, ISLA) is
handed whole meshes and global_data and picks the leaves its schema declares, so only its prediction can show that
no label reaches it. Changing the viscosity changes what every model reads (control): a tensor model's inputs, a
mesh model's prediction."""

import hydra
import pytest
import torch
from datasets import load_dataset_config
from omegaconf import OmegaConf
from output_normalize import (
    model_dataset_constants,
    model_output_stats,
    normalize_output_to_tensordict,
)
from test_aerozoo_casescale import _batch
from test_aerozoo_configs import (
    _RECIPE_ROOT,
    TARGETS,
    _chain,
    _compose,
    _labelled_case,
    _relabel,
)
from test_aerozoo_flowaxis import _oriented
from test_aerozoo_geometry_configs import (
    BC_CONFIGS,
    GEOMETRY_CONFIGS,
    _scale_viscosity,
    _tensors,
)

from physicsnemo.datapipes.transforms.mesh import NormalizeMeshFields

#: Every AeroZoo model config: a new one fails here until it has CPU-test knobs below.
MODELS = sorted(
    p.stem for p in (_RECIPE_ROOT / "conf" / "model").glob("*aerozoo*.yaml")
)
#: The knobs that shrink each base config for a CPU test without changing its interface.
KNOBS = {
    **GEOMETRY_CONFIGS,
    **BC_CONFIGS,
    "geotransolver_volume_aerozoo": {
        "n_layers": 2,
        "n_hidden": 32,
        "slice_num": 16,
        "n_head": 2,
    },
    "isla_volume_aerozoo": {
        "hidden": 16,
        "n_layers": 1,
        "n_slices": 4,
        "compile_blocks": False,
        "n_boundary_tokens": 64,
    },
}
#: The dataset each arm's configs train on; ISLA computes its scales itself, on the main file.
ARMS = {
    "": "aerozoo_uvolume",
    "_casescale": "aerozoo_uvolume_casescale",
    "_flowaxis": "aerozoo_uvolume_flowaxis",
}


def _arm(model: str) -> tuple[str, str]:
    """(base config, dataset file) of an AeroZoo model config."""
    for suffix in ("_casescale", "_flowaxis"):
        if model.endswith(suffix):
            base = model.removesuffix(suffix)
            return base, "aerozoo_uvolume" if base.startswith("isla") else ARMS[suffix]
    return model, ARMS[""]


def _predict(model: str):
    """A function from a case to the config's (inputs, prediction) tensors, with its model built at seed 0 with the
    CPU knobs; a mesh model's inputs are left out (its prediction shows what it reads)."""
    base, dataset = _arm(model)
    cfg = _compose(model, dataset=dataset)
    chain = _chain(dataset, cfg)
    normalizer = next(t for t in chain if isinstance(t, NormalizeMeshFields))
    ds = load_dataset_config(_RECIPE_ROOT / "datasets" / f"{dataset}.yaml")
    torch.manual_seed(0)
    net = hydra.utils.instantiate(
        OmegaConf.merge(cfg.model, KNOBS[base]),
        _convert_="partial",
        **model_output_stats(cfg.model, normalizer),
        **model_dataset_constants(cfg.model, ds),
    ).eval()

    def predict(case):
        batch = _batch(cfg, case, dataset)
        torch.manual_seed(0)
        with torch.no_grad():
            out = net(**batch["forward_kwargs"])
        inputs = (
            _tensors(batch["forward_kwargs"]) if cfg.input_type == "tensors" else {}
        )
        return inputs, _tensors(
            normalize_output_to_tensordict(out, TARGETS, cfg.output_type)
        )

    return predict


def test_every_aerozoo_config_is_covered():
    """The parametrization below reaches every arm of every model, and ISLA's two configs."""
    assert len(MODELS) == 18 and {_arm(m)[0] for m in MODELS} <= set(KNOBS)
    assert {"isla_volume_aerozoo", "isla_volume_aerozoo_flowaxis"} <= set(MODELS)


@pytest.mark.parametrize("delete", [False, True], ids=["perturbed", "deleted"])
@pytest.mark.parametrize("model", MODELS)
def test_the_prediction_is_blind_to_every_label(model, delete):
    """Every label perturbed (reference.*, identity.*, diagnostics.*, solution_definition.*, the closure id, the body
    ids) or deleted leaves the prediction bitwise equal; a viscosity ten times larger changes what the model reads."""
    predict = _predict(model)
    case = _oriented(_labelled_case(True))
    inputs, ours = predict(case)
    theirs = predict(_relabel(case, delete))[1]
    assert ours.keys() == theirs.keys() == {f"/{k}" for k in TARGETS}
    assert all(torch.equal(ours[k], theirs[k]) for k in ours), model
    read, prediction = predict(_scale_viscosity(case))
    seen, changed = (inputs, read) if inputs else (ours, prediction)
    assert any(not torch.equal(seen[k], changed[k]) for k in seen), model
