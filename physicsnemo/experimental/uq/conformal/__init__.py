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

r"""Post-hoc conformal prediction for spatio-temporal fields.

Three calibrators provide per-output-element marginal coverage on a fixed
discretization
(:class:`~physicsnemo.experimental.uq.conformal.CellwiseCalibrator`),
simultaneous functional bands
(:class:`~physicsnemo.experimental.uq.conformal.FunctionalBandCalibrator`),
or expected point-risk control by conformal risk control (CRC)
(:class:`~physicsnemo.experimental.uq.conformal.RiskControlCalibrator`). All
accept tensors and ``TensorDict`` field containers. A calibrator fits a
threshold, the conformal quantile of the nonconformity scores at rank
:math:`k = \lceil (n_{cal} + 1)(1 - \alpha) \rceil` over :math:`n_{cal}`
calibration samples; the functional and risk-control tiers may multiply it
by a per-point difficulty field :math:`s(x)` before the score inverts it into
an interval.

Typical two-phase usage::

    calibrator = RiskControlCalibrator(AbsoluteErrorScore(), alpha=0.1)
    for pred, target in calibration_set:
        calibrator.update_sample(pred, target)
    predictor = calibrator.finalize()

    lo, hi = predictor.predict_interval(model_output)

``points=`` (mesh coordinates) is required on every call for
:class:`~physicsnemo.experimental.uq.conformal.CellwiseCalibrator`, whose
guarantee is tied to one fixed discretization, and otherwise only when a
difficulty field consumes it.

Fitted predictors round-trip through portable, ``weights_only``-safe
artifacts::

    predictor.save("predictor.pt", provenance={"dataset": "holdout-v1"})
    predictor = ConformalPredictor.load("predictor.pt")

Guarantees assume calibration and prediction samples are exchangeable. See
the calibrator docstrings for each tier's exact statement.
"""

from .calibrators import (
    CellwiseCalibrator,
    FunctionalBandCalibrator,
    RiskControlCalibrator,
)
from .difficulty import AuxDifficulty
from .predictors import ConformalPredictor
from .scores import (
    AbsoluteErrorScore,
    NormalizedErrorScore,
    QuantileRegressionScore,
)

__all__ = [
    "AbsoluteErrorScore",
    "AuxDifficulty",
    "CellwiseCalibrator",
    "ConformalPredictor",
    "FunctionalBandCalibrator",
    "NormalizedErrorScore",
    "QuantileRegressionScore",
    "RiskControlCalibrator",
]
