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

"""Nonconformity-score and difficulty-field unit tests."""

import pytest
import torch

from physicsnemo.experimental.uq.conformal import (
    AbsoluteErrorScore,
    AuxDifficulty,
    NormalizedErrorScore,
    QuantileRegressionScore,
)


def test_absolute_error_values(device):
    score = AbsoluteErrorScore()
    pred = torch.tensor([1.0, -2.0, 0.5], device=device)
    target = torch.tensor([1.5, -1.0, 0.5], device=device)
    torch.testing.assert_close(
        score.score(pred, target), torch.tensor([0.5, 1.0, 0.0], device=device)
    )
    lo, hi = score.interval(pred, torch.tensor(0.25, device=device))
    torch.testing.assert_close(lo, pred - 0.25)
    torch.testing.assert_close(hi, pred + 0.25)


def test_normalized_error_values_and_eps(device):
    score = NormalizedErrorScore(eps=1e-3)
    pred = torch.tensor([0.0, 0.0], device=device)
    target = torch.tensor([1.0, 2.0], device=device)
    aux = {"sigma": torch.tensor([0.5, 0.0], device=device)}
    # Second sigma clamps to eps.
    torch.testing.assert_close(
        score.score(pred, target, aux), torch.tensor([2.0, 2000.0], device=device)
    )
    lo, hi = score.interval(pred, torch.tensor(1.0, device=device), aux)
    torch.testing.assert_close(hi, torch.tensor([0.5, 1e-3], device=device))
    torch.testing.assert_close(lo, -hi)


def test_quantile_regression_asymmetry(device):
    score = QuantileRegressionScore()
    aux = {"lo": torch.zeros(3, device=device), "hi": torch.ones(3, device=device)}
    pred = torch.zeros(3, device=device)
    # Below lo, inside, above hi.
    target = torch.tensor([-0.5, 0.5, 1.75], device=device)
    torch.testing.assert_close(
        score.score(pred, target, aux), torch.tensor([0.5, -0.5, 0.75], device=device)
    )
    lo, hi = score.interval(pred, torch.tensor(0.1, device=device), aux)
    torch.testing.assert_close(lo, aux["lo"] - 0.1)
    torch.testing.assert_close(hi, aux["hi"] + 0.1)


@pytest.mark.parametrize(
    "score,aux",
    [
        (AbsoluteErrorScore(), None),
        (NormalizedErrorScore(), {"sigma": torch.full((5,), 0.7)}),
        (QuantileRegressionScore(), {"lo": -torch.ones(5), "hi": torch.ones(5)}),
    ],
)
def test_interval_endpoints_invert_score(score, aux):
    """score(pred, endpoint) == threshold at both interval endpoints."""
    pred = torch.randn(5)
    threshold = torch.tensor(0.42)
    lo, hi = score.interval(pred, threshold, aux)
    for endpoint in (lo, hi):
        torch.testing.assert_close(
            score.score(pred, endpoint, aux), threshold.expand(5), atol=1e-6, rtol=1e-5
        )


def test_aux_difficulty_channel_max_clamp_and_trailing_reduction():
    difficulty = AuxDifficulty(key="sigma", eps=torch.tensor(1e-2))
    assert type(difficulty.eps) is float  # configuration is stored as primitives
    assert type(NormalizedErrorScore(torch.tensor(1e-3)).eps) is float
    sigma = torch.tensor([[0.5, 1.5], [0.0, 0.0]])
    torch.testing.assert_close(
        difficulty(aux={"sigma": sigma}), torch.tensor([1.5, 1e-2])
    )
    # A (points, time, channels) aux reduces to one scale per point.
    assert difficulty(None, {"sigma": torch.rand(2, 3, 4) + 0.5}).shape == (2,)


_D = AuxDifficulty(key="sigma")
_S = NormalizedErrorScore()
_Z2, _O2 = torch.zeros(2), torch.ones(2)
# fmt: off
STRATEGY_REJECTIONS = [  # (id, thunk, error, match)
    ("difficulty-missing-aux-key", lambda: _D(torch.rand(2, 2)), ValueError, "sigma"),
    ("difficulty-aux-not-mapping", lambda: _D(aux=[_O2]), ValueError, "requires aux entry"),
    ("score-missing-sigma", lambda: _S.score(_Z2, _O2), ValueError, "sigma"),
    ("score-aux-not-mapping", lambda: _S.score(_Z2, _O2, aux=[_O2]), ValueError, "requires aux entries"),
    ("key-int", lambda: AuxDifficulty(key=7), TypeError, "key must be a string"),
    ("key-empty", lambda: AuxDifficulty(key=""), ValueError, "non-empty"),
    ("score-eps-none", lambda: NormalizedErrorScore(eps=None), ValueError, "positive finite value"),
    ("difficulty-eps-string", lambda: AuxDifficulty(eps="not a number"), ValueError, "positive finite value"),
]
# fmt: on


@pytest.mark.parametrize(
    "thunk,error,match",
    [pytest.param(*row[1:], id=row[0]) for row in STRATEGY_REJECTIONS],
)
def test_strategy_inputs_and_configuration_are_validated(thunk, error, match):
    with pytest.raises(error, match=match):
        thunk()
