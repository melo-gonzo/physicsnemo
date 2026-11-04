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

import json
import math

import pytest
import torch

from physicsnemo.utils import EarlyStopping


def run(stopper, losses, start=1, step=1):
    """Feed ``losses`` at epochs ``start, start + step, ...``; return (epoch, reason) of the stop or None."""
    for i, loss in enumerate(losses):
        epoch = start + i * step
        reason = stopper.step(epoch, loss)
        if reason:
            return epoch, reason
    return None


def test_patience_counts_epochs_not_calls():
    losses = [0.9, 0.5, 0.6, 0.55, 0.52, 0.51]
    assert run(EarlyStopping(patience=4), losses) == (
        6,
        "no improvement for 4 epochs: best 0.5 at epoch 2",
    )
    # The same curve validated every 5 epochs stops after the same number of epochs.
    assert run(EarlyStopping(patience=20), losses, start=5, step=5) == (
        30,
        "no improvement for 20 epochs: best 0.5 at epoch 10",
    )


def test_min_delta_is_relative_and_a_new_best_resets_the_count():
    stopper = EarlyStopping(patience=2, min_delta=0.01)
    assert stopper.step(1, 1.0) is None
    assert stopper.step(2, 0.995) is None  # 0.5 %: not a new best
    assert stopper.best_epoch == 1
    assert stopper.step(3, 0.98) is None  # 2 %: new best, count reset
    assert stopper.best_epoch == 3
    assert stopper.step(4, 0.98) is None
    assert stopper.step(5, 0.98) == "no improvement for 2 epochs: best 0.98 at epoch 3"


def test_min_epochs_holds_patience_and_never_learned_off():
    flat = [0.86, 0.855, 0.853, 0.853, 0.853, 0.853]
    assert run(EarlyStopping(patience=2, min_epochs=5), flat) == (
        5,
        "no improvement for 2 epochs: best 0.853 at epoch 3",
    )
    stopper = EarlyStopping(patience=10, min_epochs=4, min_gain=0.05)
    assert run(stopper, flat) == (
        4,
        "never learned: best 0.853 at epoch 3 improves on the first validation 0.86 "
        "by 0.8%, under 5%",
    )
    learned = EarlyStopping(patience=10, min_epochs=4, min_gain=0.05)
    assert run(learned, [0.86, 0.7, 0.6, 0.55, 0.5]) is None


def test_divergence_needs_consecutive_bad_validations():
    stopper = EarlyStopping(patience=100, divergence_factor=3.0, divergence_patience=2)
    assert stopper.step(1, 0.5) is None
    assert stopper.step(2, 2.0) is None  # one bad validation
    assert stopper.step(3, 0.6) is None  # back under 3x: count reset
    assert stopper.step(4, 1.6) is None
    assert stopper.step(5, 9.0) == (
        "diverged: validation metric 9 at epoch 5 is over 3x the best 0.5 (epoch 1) "
        "for 2 validation(s) in a row"
    )
    assert stopper.step(6, 0.4) == stopper.stopped  # a stop is final


def test_a_non_finite_metric_stops_at_once():
    stopper = EarlyStopping(patience=100, min_epochs=50)
    assert stopper.step(1, 0.5) is None
    assert stopper.step(2, torch.tensor(float("nan"))) == (
        "diverged: non-finite validation metric nan at epoch 2"
    )


def test_max_mode():
    stopper = EarlyStopping(
        patience=2, mode="max", min_delta=0.1, divergence_factor=2.0
    )
    assert stopper.step(1, 0.5) is None
    assert stopper.step(2, 0.52) is None  # 4 %: not a new best
    assert stopper.step(3, 0.9) is None
    assert stopper.best == 0.9 and stopper.best_epoch == 3
    assert stopper.step(4, -0.5) == (
        "diverged: validation metric -0.5 at epoch 4 is over 2x the best 0.9 (epoch 3) "
        "for 1 validation(s) in a row"
    )


def test_the_state_survives_a_restart(tmp_path):
    path = tmp_path / "state" / "early_stopping.json"
    first = EarlyStopping(patience=3, path=path)
    assert first.step(5, 0.7) is None and first.step(10, 0.6) is None
    assert json.loads(path.read_text()) == {
        "best": 0.6,
        "best_epoch": 10,
        "first": 0.7,
        "worse": 0,
        "stopped": None,
    }
    second = EarlyStopping(patience=3, path=path)
    assert (second.best, second.best_epoch, second.first) == (0.6, 10, 0.7)
    assert second.step(15, 0.65) == "no improvement for 5 epochs: best 0.6 at epoch 10"
    assert json.loads(path.read_text())["stopped"].startswith("no improvement")
    third = EarlyStopping(patience=3, path=path)
    assert third.step(20, 0.1) == second.stopped


def test_state_dict_round_trip():
    stopper = EarlyStopping(patience=2)
    stopper.step(1, 0.4)
    other = EarlyStopping(patience=2)
    other.load_state_dict(stopper.state_dict())
    assert other.state_dict() == stopper.state_dict()
    assert math.isclose(other.best, 0.4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"patience": 0},
        {"patience": 1, "min_delta": 1.0},
        {"patience": 1, "min_delta": -0.1},
        {"patience": 1, "mode": "best"},
        {"patience": 1, "min_epochs": -1},
        {"patience": 1, "min_gain": 0.0},
        {"patience": 1, "divergence_factor": 1.0},
        {"patience": 1, "divergence_patience": 0},
    ],
)
def test_invalid_args(kwargs):
    with pytest.raises(ValueError):
        EarlyStopping(**kwargs)
