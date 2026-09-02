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

r"""Difficulty fields: positive scale functions :math:`s(x)` for query inputs.

The functional-band and conformal risk control (CRC) tiers multiply a fitted
scalar threshold by :math:`s(x)`. Because :math:`s` is evaluated from the
query location or per-point auxiliary data, calibration samples and queries
may have different point sets. The guarantee still concerns exchangeable
whole samples and the statistic defined over their observed points; it does
not claim invariance to arbitrary changes in discretization. A well-chosen
:math:`s` (large where the model errs) yields tight bands; no difficulty
field (:math:`s = 1`) is valid but conservative.

.. warning::
    The difficulty field must be fixed independently of the calibration
    samples: fit it on training residuals or a split disjoint from
    calibration. Fitting :math:`s` on the calibration set voids the coverage
    guarantee (the scores are no longer exchangeable with test scores).
"""

import copy
from collections.abc import Mapping

from jaxtyping import Float
from torch import Tensor

from ._validation import clamp_min_floor, positive_finite_float
from .scores import _NonconformityScore

__all__ = ["AuxDifficulty"]


class AuxDifficulty:
    r"""Per-point difficulty :math:`s(x)` read from the ``aux`` mapping.

    Use it with
    :class:`~physicsnemo.experimental.uq.conformal.FunctionalBandCalibrator` or
    :class:`~physicsnemo.experimental.uq.conformal.RiskControlCalibrator` when
    the model emits a per-point uncertainty proxy (a predicted sigma,
    MC-dropout or ensemble spread) that should widen the band where the
    model is least confident. Multi-channel inputs of shape
    :math:`(n_{\text{points}}, C)` are reduced by ``max`` over the trailing
    dimensions so one scale per point dominates every channel.

    Parameters
    ----------
    key : str, optional
        Aux key to read. Default is ``"sigma"``.
    eps : float, optional
        Lower clamp keeping :math:`s` positive. Default is ``1e-8``.

    Notes
    -----
    Pairing this field with a score that already divides by the same aux key
    (:class:`~physicsnemo.experimental.uq.conformal.NormalizedErrorScore` on
    ``"sigma"``) would scale every interval twice; calibrators and predictors
    raise ``ValueError`` on that combination. The aux entry must be present
    at every calibration and prediction call.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.uq.conformal import AuxDifficulty
    >>> _ = torch.manual_seed(0)
    >>> difficulty = AuxDifficulty("sigma")
    >>> sigma = torch.rand(50, 3) + 0.1
    >>> difficulty(aux={"sigma": sigma}).shape
    torch.Size([50])
    """

    def __init__(self, key: str = "sigma", eps: float = 1e-8) -> None:
        if not isinstance(key, str):
            raise TypeError(f"key must be a string, got {type(key).__name__}.")
        if not key:
            raise ValueError("key must be a non-empty string.")
        self.key = key
        self.eps = positive_finite_float(eps, "eps")

    def __call__(
        self,
        points: Float[Tensor, "n_points n_spatial_dims"] | None = None,
        aux: Mapping[str, Tensor] | None = None,
    ) -> Float[Tensor, " n_points"]:
        r"""Evaluate :math:`s` from the per-point ``aux`` entry ``key``.

        Parameters
        ----------
        points : torch.Tensor, optional
            Mesh coordinates of shape
            :math:`(n_{\text{points}}, n_{\text{spatial\_dims}})`. Accepted
            for API uniformity and unused. Default is ``None``.
        aux : Mapping[str, torch.Tensor], optional
            Must contain ``key`` with a tensor of shape
            :math:`(n_{\text{points}}, *\text{dims})`. Default is ``None``.

        Returns
        -------
        torch.Tensor
            Positive difficulty values of shape :math:`(n_{\text{points}},)`,
            reduced by ``max`` over any trailing dimensions and clamped below
            by ``eps``.
        """
        if not isinstance(aux, Mapping) or self.key not in aux:
            raise ValueError(
                f"AuxDifficulty requires aux entry '{self.key}' at every call; "
                "pass aux={key: tensor}."
            )
        s = aux[self.key]
        if s.ndim >= 2:
            # Reduce EVERY trailing (non-point) dimension: the contract is
            # one positive scale per leading point, matching the
            # risk-control loss's point risk unit (max over components).
            s = s.amax(dim=tuple(range(1, s.ndim)))
        return clamp_min_floor(s, self.eps)


def _check_no_double_scale(
    score: _NonconformityScore, difficulty: AuxDifficulty | None
) -> None:
    """Reject the known ergonomic footgun of dividing by the same aux twice.

    A score that divides its residual by an aux key (advertised via
    ``score.scale_aux_keys``, e.g.
    :class:`~physicsnemo.experimental.uq.conformal.NormalizedErrorScore` on
    ``"sigma"``) paired with an :class:`AuxDifficulty` reading the same key
    scales every interval by that value twice. This is an ergonomics guard,
    not a conformal-validity requirement: it catches the one structural
    built-in pairing (``AuxDifficulty`` on a divisor key). Reading a key
    additively (CQR reads ``lo``/``hi``) is not dividing by it, so those
    pairings are allowed.

    Run from calibrator and predictor construction so the rule is enforced
    consistently.
    """
    if isinstance(difficulty, AuxDifficulty):
        if difficulty.key in score.scale_aux_keys:
            raise ValueError(
                f"Double-scaling: score {type(score).__name__} already divides the "
                f"residual by aux '{difficulty.key}', and AuxDifficulty(key="
                f"'{difficulty.key}') would divide by it again. Pair AuxDifficulty("
                f"'{difficulty.key}') with a score that does not scale by it (e.g. "
                "AbsoluteErrorScore), or use the scaling score with no difficulty "
                "field."
            )


_DIFFICULTY_REGISTRY: dict[str, type[AuxDifficulty]] = {
    "aux": AuxDifficulty,
}
"""Private identifiers for the exact built-in difficulty field types."""


def _difficulty_kind(difficulty: AuxDifficulty) -> str | None:
    """Identifier for an exact built-in difficulty type, otherwise ``None``."""
    for kind, cls in _DIFFICULTY_REGISTRY.items():
        if type(difficulty) is cls:
            return kind
    return None


def _snapshot_difficulty(difficulty: AuxDifficulty | None) -> AuxDifficulty | None:
    """Require and snapshot one of the shipped difficulty strategies.

    ``None`` means no difficulty field (``s = 1``) and passes through; the
    single validation shared by calibrators and fitted predictors.
    """
    if difficulty is None:
        return None
    if _difficulty_kind(difficulty) is None:
        names = ", ".join(sorted(cls.__name__ for cls in _DIFFICULTY_REGISTRY.values()))
        raise TypeError(
            f"difficulty must be None or one of the shipped strategies ({names}); "
            f"got {type(difficulty).__name__}."
        )
    return copy.deepcopy(difficulty)
