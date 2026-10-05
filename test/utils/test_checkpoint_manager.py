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

import math

import pytest
import torch

from physicsnemo.distributed import DistributedManager
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.utils import CheckpointManager, load_checkpoint

LOSSES = [0.9, 0.5, 0.7, 0.3, 0.8, 0.4, 0.6, 0.2]


@pytest.fixture(autouse=True)
def dist():
    DistributedManager.initialize()
    yield
    DistributedManager.cleanup()


@pytest.fixture
def model():
    return FullyConnected(in_features=4, out_features=4, num_layers=1)


def _epochs(path):
    return sorted(
        {
            int(f.name.split(".")[-2])
            for f in path.iterdir()
            if f.suffix in (".mdlus", ".pt")
        }
    )


def test_latest_and_best_k(tmp_path, model):
    optimizer = torch.optim.Adam(model.parameters())
    manager = CheckpointManager(tmp_path, keep_best=3)
    for epoch, loss in enumerate(LOSSES):
        manager.save(epoch, metric=loss, models=model, optimizer=optimizer)

    assert _epochs(tmp_path) == [7]
    assert manager.best == [(0.2, 7), (0.3, 3), (0.4, 5)]
    assert manager.best_epoch == 7
    assert _epochs(tmp_path / "best") == [3, 5, 7]
    # best_weights_only drops optimizer state
    best_state = torch.load(tmp_path / "best" / "checkpoint.0.7.pt")
    assert "optimizer_state_dict" not in best_state
    assert _epochs(tmp_path / "top_model") == [7]
    assert load_checkpoint(str(tmp_path), models=model, optimizer=optimizer) == 7


def test_save_every_and_max_mode(tmp_path, model):
    manager = CheckpointManager(
        tmp_path, keep_best=2, save_every=3, mode="max", best_weights_only=False
    )
    for epoch, acc in enumerate(LOSSES):
        manager.save(epoch, metric=torch.tensor(acc, dtype=torch.float64), models=model)

    assert _epochs(tmp_path) == [0, 3, 6, 7]
    assert manager.best == [(0.9, 0), (0.8, 4)]
    assert _epochs(tmp_path / "best") == [0, 4]
    assert _epochs(tmp_path / "top_model") == [0]


def test_resume_and_skip(tmp_path, model):
    manager = CheckpointManager(tmp_path, keep_best=2)
    for epoch, loss in enumerate(LOSSES[:4]):
        manager.save(epoch, metric=loss, models=model)
    assert not manager.save(4, metric=math.nan, models=model)
    assert not manager.save(5, models=model)

    resumed = CheckpointManager(tmp_path, keep_best=2)
    assert resumed.best == [(0.3, 3), (0.5, 1)]
    assert resumed.save(6, metric=0.1, models=model)
    assert resumed.best == [(0.1, 6), (0.3, 3)]
    assert _epochs(tmp_path / "best") == [3, 6]
    assert _epochs(tmp_path / "top_model") == [6]
    assert _epochs(tmp_path) == [6]


def test_invalid_args(tmp_path):
    with pytest.raises(ValueError):
        CheckpointManager(tmp_path, keep_best=-1)
    with pytest.raises(ValueError):
        CheckpointManager(tmp_path, save_every=0)
    with pytest.raises(ValueError):
        CheckpointManager(tmp_path, mode="best")
