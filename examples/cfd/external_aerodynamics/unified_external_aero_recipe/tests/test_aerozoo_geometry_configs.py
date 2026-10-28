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

"""AeroZoo v1.2, interior task from geometry only: each geometry-only model config runs on the synthetic
release-shaped case through the AeroZoo chain, collate, forward and the protocol loss, predicts all seven mean
fields, and reads no prescribed boundary value (every one scaled leaves every model input bitwise equal)."""

import hydra
import pytest
import torch
from collate import build_collate_fn
from loss import LossCalculator
from omegaconf import OmegaConf
from output_normalize import normalize_output_to_tensordict
from test_aerozoo_configs import (
    DATASET,
    TARGETS,
    _chain,
    _compose,
    _labelled_case,
    _run,
)

from physicsnemo.mesh import DomainMesh, Mesh

#: The geometry-only configs and the knobs that shrink them for a CPU test without changing their interface.
GEOMETRY_CONFIGS = {
    "transolver_volume_aerozoo": {
        "n_layers": 2,
        "n_hidden": 32,
        "slice_num": 16,
        "n_head": 2,
    },
    "geotransolver_volume_aerozoo_geometry": {
        "n_layers": 2,
        "n_hidden": 32,
        "slice_num": 16,
        "n_head": 2,
        "include_local_features": False,
    },
    "smart_volume_aerozoo_geometry": {
        "hidden_dim": 32,
        "num_heads": 2,
        "num_layers": 1,
        "num_latents": 16,
        "geometry_subsample": 64,
        "eval_chunk_size": 64,
    },
    "globe_volume_aerozoo_geometry": {
        "hidden_layer_sizes": [16],
        "n_communication_hyperlayers": 1,
    },
}


def _scale_prescribed(d: DomainMesh) -> DomainMesh:
    """Every prescribed boundary value times 3 plus 1."""
    boundaries = {}
    for name in d.boundary_names:
        b = d.boundaries[name]
        cell_data = b.cell_data.clone()
        if "prescribed" in cell_data.keys():
            cell_data["prescribed"] = cell_data["prescribed"].apply(lambda x: 3 * x + 1)
        boundaries[name] = Mesh(points=b.points, cells=b.cells, cell_data=cell_data)
    return DomainMesh(
        interior=d.interior, boundaries=boundaries, global_data=d.global_data
    )


def _tensors(value, prefix="") -> dict[str, torch.Tensor]:
    """The tensors of collated forward kwargs (tensors, dicts, TensorDicts or meshes), keyed by their path."""
    if isinstance(value, torch.Tensor):
        return {prefix: value}
    if isinstance(value, Mesh):
        out = {f"{prefix}/points": value.points, f"{prefix}/cells": value.cells}
        for group in ("point_data", "cell_data"):
            out |= _tensors(getattr(value, group), f"{prefix}/{group}")
        return out
    if isinstance(value, dict):
        return {
            p: t
            for k, v in value.items()
            for p, t in _tensors(v, f"{prefix}/{k}").items()
        }
    if hasattr(value, "keys"):  # TensorDict
        return {
            f"{prefix}/{k}": value[k]
            for k in value.keys(include_nested=True, leaves_only=True)
        }
    return {}


def _batch(cfg, case: DomainMesh) -> dict:
    """Run the chain (with the config's appends) and collate; the seeds SMART's reader adds are set here, since
    the chain leaves the reader out."""
    spec = OmegaConf.to_container(cfg.forward_kwargs, resolve=True)
    if "geometry_seed" in spec:
        gd = case.global_data.clone()
        gd["smart_query_seed"], gd["smart_geometry_seed"] = (
            torch.tensor(1),
            torch.tensor(2),
        )
        case = DomainMesh(
            interior=case.interior, boundaries=case.boundaries, global_data=gd
        )
    d = _run(_chain(DATASET, cfg), case)
    return build_collate_fn(cfg.input_type, spec, TARGETS)([(d, {})])


@pytest.mark.parametrize("model", sorted(GEOMETRY_CONFIGS))
def test_geometry_config_predicts_the_seven_fields(model):
    """The config composes with the protocol loss, runs collate -> forward, and predicts every target."""
    cfg = _compose(model, dataset=DATASET)
    assert cfg.training.loss_type == "case_relative_l2"
    assert OmegaConf.to_container(cfg.training.field_weights) == {"velocity": 3}
    batch = _batch(cfg, _labelled_case(False))
    net = hydra.utils.instantiate(
        OmegaConf.merge(cfg.model, GEOMETRY_CONFIGS[model]), _convert_="partial"
    )
    with torch.no_grad():
        out = net(**batch["forward_kwargs"])
    pred = normalize_output_to_tensordict(out, TARGETS, cfg.output_type)
    assert set(pred.keys()) == set(TARGETS)
    total, _ = LossCalculator(
        TARGETS,
        loss_type=cfg.training.loss_type,
        field_weights=OmegaConf.to_container(cfg.training.field_weights),
    )(pred, batch["targets"])
    assert torch.isfinite(total)


@pytest.mark.parametrize("model", sorted(GEOMETRY_CONFIGS))
def test_geometry_config_reads_no_prescribed_value(model):
    """Every prescribed value scaled leaves the prediction bitwise equal. A mesh model (GLOBE) is handed the
    boundary mesh with its cell_data but reads only the leaves its schema declares, so the check is on the
    output; a tensor model's inputs are compared as well."""
    cfg = _compose(model, dataset=DATASET)
    case = _labelled_case(True)
    batches = [_batch(cfg, x) for x in (case, _scale_prescribed(case))]
    torch.manual_seed(0)
    net = hydra.utils.instantiate(
        OmegaConf.merge(cfg.model, GEOMETRY_CONFIGS[model]), _convert_="partial"
    )
    with torch.no_grad():
        a, b = (
            _tensors(
                normalize_output_to_tensordict(
                    net(**x["forward_kwargs"]), TARGETS, cfg.output_type
                )
            )
            for x in batches
        )
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    if cfg.input_type == "tensors":
        a, b = (_tensors(x["forward_kwargs"]) for x in batches)
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_the_prescribed_scaling_reaches_a_boundary_condition_model():
    """Control: the same scaling changes the inputs of the boundary-condition GeoTransolver."""
    cfg = _compose("geotransolver_volume_aerozoo", dataset=DATASET)
    case = _labelled_case(True)
    a, b = (
        _tensors(_batch(cfg, x)["forward_kwargs"])
        for x in (case, _scale_prescribed(case))
    )
    assert not torch.equal(a["/geometry"], b["/geometry"])
