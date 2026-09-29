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

r"""Nonconformity scores for conformal prediction.

A nonconformity score quantifies, elementwise, how badly a prediction
disagrees with an observed target. Every score also knows how to invert a
calibrated threshold into a prediction interval, so calibrators and fitted
predictors are score-agnostic.

That invertibility is what separates a nonconformity score from an error
metric: a metric (MAE, RMSE) is a scalar read after the fact, whereas a
score is an elementwise residual. Its returned interval conservatively
encloses the sublevel set :math:`\{y : \text{score}(\hat y, y) \le t\}`.
The ``...Score`` suffix marks that distinction; these classes are not error
metrics and are not interchangeable with ``physicsnemo`` metrics.

Scores operate on plain tensors; field-container (``TensorDict``) iteration
is handled by the calibrators. Optional ``aux`` inputs carry the built-in
score data: predicted standard deviations (``"sigma"``) or quantile heads
(``"lo"``/``"hi"``).

:class:`AuxDifficulty` is the one shipped difficulty field: a positive
per-point scale :math:`s(x)` that the functional-band and conformal risk
control (CRC) tiers multiply into the fitted threshold. Because :math:`s` is
evaluated per query point, calibration samples and queries may have
different point sets. No difficulty field (:math:`s = 1`) is valid but
conservative.

.. warning::
    The difficulty field must be fixed independently of the calibration
    samples: fit it on training residuals or a split disjoint from
    calibration. Fitting :math:`s` on the calibration set voids the coverage
    guarantee (the scores are no longer exchangeable with test scores).
"""

import copy
from collections.abc import Mapping

import torch
from jaxtyping import Float
from torch import Tensor

from ._utils import cast_directed, check_real, clamp_min_floor, positive_finite_float

__all__ = [
    "AbsoluteErrorScore",
    "AuxDifficulty",
    "NormalizedErrorScore",
    "QuantileRegressionScore",
]


def _coarsest_finfo(*tensors: Tensor | None) -> torch.finfo:
    """``finfo`` of the least-precise floating dtype among the inputs (aux included)."""
    infos = [
        torch.finfo(t.dtype)
        for t in tensors
        if t is not None and torch.is_tensor(t) and t.is_floating_point()
    ]
    if not infos:
        return torch.finfo(torch.float64)
    return max(infos, key=lambda fi: fi.eps)


def _slack_threshold(threshold: Tensor, *dtype_sources: Tensor | None) -> Tensor:
    """Float64 threshold inflated by ``4 * eps * |t| + 4 * tiny`` of the coarsest dtype.

    The slack dominates score rounding, so no score-admitted target is excluded.
    """
    t = threshold.to(torch.float64)
    fi = _coarsest_finfo(*dtype_sources)
    return t + (t.abs() * (4.0 * fi.eps) + 4.0 * fi.tiny)


def _outward_interval(
    prediction: Tensor, lo64: Tensor, hi64: Tensor
) -> tuple[Tensor, Tensor]:
    """Round float64 endpoints outward: ``score <= threshold`` implies ``lo <= y <= hi``."""
    return (
        cast_directed(lo64, prediction.dtype, up=False),
        cast_directed(hi64, prediction.dtype, up=True),
    )


def _require_aux(
    aux: Mapping[str, Tensor] | None, keys: tuple[str, ...], score_name: str
) -> None:
    """Require the aux keys a score reads; entry types are checked by ``check_aux``."""
    missing = [k for k in keys if not isinstance(aux, Mapping) or k not in aux]
    if missing:
        raise ValueError(
            f"{score_name} requires aux entries {list(keys)}; missing {missing}. "
            "Pass aux={key: tensor} (or, for TensorDict inputs, "
            "aux={field: {key: tensor}})."
        )


class _NonconformityScore:
    r"""Internal base shared by the shipped score strategies.

    Only the shipped subclasses below are accepted by calibrators and
    predictors (they are the serializable strategies). :meth:`score`
    (calibration time) and :meth:`interval` (prediction time) are related by
    the property that the prediction set
    :math:`\{y : \text{score}(\hat y, y) \le t\}` lies within
    :math:`[\text{lo}, \text{hi}] = \text{interval}(\hat y, t)` elementwise.
    Finite-precision inversion gives a conservative enclosure of this
    sublevel set.

    ``aux_keys`` lists the aux keys a score reads; ``scale_aux_keys`` is the
    subset it divides the residual by. Intervals must never round inward, or
    the finite-sample coverage is lost.
    """

    aux_keys: tuple[str, ...] = ()
    scale_aux_keys: tuple[str, ...] = ()

    def score(
        self,
        prediction: Float[Tensor, "*dims"],
        target: Float[Tensor, "*dims"],
        aux: Mapping[str, Tensor] | None = None,
    ) -> Float[Tensor, "*dims"]:
        r"""Elementwise nonconformity of ``target`` given ``prediction``.

        Parameters
        ----------
        prediction : torch.Tensor
            Model output of shape :math:`(*\text{dims})`.
        target : torch.Tensor
            Observed values, same shape as ``prediction``.
        aux : Mapping[str, torch.Tensor], optional
            Finite real floating-point tensors read by the score
            (``aux_keys``), each of the same shape as ``prediction``.
            Default is ``None``, which means no aux mapping; only scores
            without required aux keys allow omission. Required entries must
            be tensors, not ``None``.

        Returns
        -------
        torch.Tensor
            Nonconformity scores of shape :math:`(*\text{dims})`.
        """
        raise NotImplementedError

    def interval(
        self,
        prediction: Float[Tensor, "*dims"],
        threshold: Float[Tensor, "*dims"] | Float[Tensor, ""],
        aux: Mapping[str, Tensor] | None = None,
    ) -> tuple[Float[Tensor, "*dims"], Float[Tensor, "*dims"]]:
        r"""Invert a calibrated ``threshold`` into an interval ``(lo, hi)``.

        Parameters
        ----------
        prediction : torch.Tensor
            Model output of shape :math:`(*\text{dims})`.
        threshold : torch.Tensor
            Fitted conformal quantile broadcastable against ``prediction``:
            an elementwise tensor of shape :math:`(*\text{dims})` for the
            cellwise tier, or a scalar (already multiplied by the difficulty
            field, when one is configured) for the functional and
            risk-control tiers.
        aux : Mapping[str, torch.Tensor], optional
            Finite real floating-point tensors read by the score
            (``aux_keys``), each of the same shape as ``prediction``.
            Default is ``None``, which means no aux mapping; only scores
            without required aux keys allow omission. Required entries must
            be tensors, not ``None``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Lower and upper bounds ``(lo, hi)``, each of shape
            :math:`(*\text{dims})` and in the dtype of ``prediction``.
        """
        raise NotImplementedError


class AbsoluteErrorScore(_NonconformityScore):
    r"""Absolute error residual :math:`|y - \hat y|`.

    The default score for deterministic models: no architectural
    requirements and no aux inputs. A calibrated threshold :math:`t` inverts
    to the interval :math:`[\hat y - t, \hat y + t]`. This is the split
    conformal score of `Distribution-Free Predictive Inference for Regression
    <https://arxiv.org/abs/1604.04173>`_ (Lei et al., 2018).

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.uq.conformal import AbsoluteErrorScore
    >>> _ = torch.manual_seed(0)
    >>> score = AbsoluteErrorScore()
    >>> prediction, target = torch.randn(50, 2), torch.randn(50, 2)
    >>> score.score(prediction, target).shape
    torch.Size([50, 2])
    >>> threshold = torch.tensor(0.5, dtype=torch.float64)
    >>> lo, hi = score.interval(prediction, threshold)
    >>> lo.shape, bool((hi > lo).all())
    (torch.Size([50, 2]), True)
    """

    def score(
        self,
        prediction: Float[Tensor, "*dims"],
        target: Float[Tensor, "*dims"],
        aux: Mapping[str, Tensor] | None = None,
    ) -> Float[Tensor, "*dims"]:
        return (target - prediction).abs()

    def interval(
        self,
        prediction: Float[Tensor, "*dims"],
        threshold: Float[Tensor, "*dims"] | Float[Tensor, ""],
        aux: Mapping[str, Tensor] | None = None,
    ) -> tuple[Float[Tensor, "*dims"], Float[Tensor, "*dims"]]:
        p = prediction.to(torch.float64)
        t = _slack_threshold(threshold, prediction)
        return _outward_interval(prediction, p - t, p + t)


class NormalizedErrorScore(_NonconformityScore):
    r"""Sigma-normalized residual :math:`|y - \mu| / \max(\sigma, \epsilon)`.

    For probabilistic models emitting a mean :math:`\mu` and standard
    deviation :math:`\sigma` (NLL heads, MC-dropout or ensemble spread). The
    resulting intervals scale with the model's own uncertainty, giving
    input-dependent widths. A calibrated threshold :math:`t` inverts to
    :math:`[\mu - t\sigma, \mu + t\sigma]`. The prediction is :math:`\mu`;
    :math:`\sigma` is read from ``aux["sigma"]``.

    Parameters
    ----------
    eps : float, optional
        Lower clamp :math:`\epsilon` on ``sigma`` to avoid division blow-up.
        Default is ``1e-8``.

    Notes
    -----
    The effective floor is the larger of ``eps`` and the smallest positive
    normal value of the ``sigma`` dtype, so the clamp cannot underflow to a
    no-op in low-precision dtypes. Raw sigma values must be finite before
    clamping; finite nonpositive values use the floor. Calling :meth:`score`
    or :meth:`interval` without ``aux["sigma"]`` raises ``ValueError``.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.uq.conformal import NormalizedErrorScore
    >>> _ = torch.manual_seed(0)
    >>> score = NormalizedErrorScore()
    >>> prediction, target = torch.randn(50, 2), torch.randn(50, 2)
    >>> aux = {"sigma": torch.rand(50, 2) + 0.1}
    >>> score.score(prediction, target, aux).shape
    torch.Size([50, 2])
    >>> threshold = torch.tensor(2.0, dtype=torch.float64)
    >>> lo, hi = score.interval(prediction, threshold, aux)
    >>> hi.shape
    torch.Size([50, 2])
    """

    aux_keys = ("sigma",)
    scale_aux_keys = ("sigma",)  # the residual is divided by sigma

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = positive_finite_float(eps, "eps")

    def score(
        self,
        prediction: Float[Tensor, "*dims"],
        target: Float[Tensor, "*dims"],
        aux: Mapping[str, Tensor] | None = None,
    ) -> Float[Tensor, "*dims"]:
        _require_aux(aux, self.aux_keys, type(self).__name__)
        sigma = clamp_min_floor(aux["sigma"], self.eps)
        # Divide in float64, then round up so thresholds can only grow.
        s64 = (target.to(torch.float64) - prediction.to(torch.float64)).abs()
        s64 = s64 / sigma.to(torch.float64)
        out_dtype = torch.result_type(prediction, target)
        return cast_directed(s64, out_dtype, up=True)

    def interval(
        self,
        prediction: Float[Tensor, "*dims"],
        threshold: Float[Tensor, "*dims"] | Float[Tensor, ""],
        aux: Mapping[str, Tensor] | None = None,
    ) -> tuple[Float[Tensor, "*dims"], Float[Tensor, "*dims"]]:
        _require_aux(aux, self.aux_keys, type(self).__name__)
        # The exact clamp used by score(), then upcast.
        sigma = clamp_min_floor(aux["sigma"], self.eps).to(torch.float64)
        p = prediction.to(torch.float64)
        # Sigma embeds exactly in float64, so its dtype does not round the division.
        half = _slack_threshold(threshold, prediction) * sigma
        return _outward_interval(prediction, p - half, p + half)


class QuantileRegressionScore(_NonconformityScore):
    r"""Conformalized quantile regression (CQR) score.

    For models with quantile-regression heads emitting lower and upper
    quantile estimates :math:`q_{lo}` and :math:`q_{hi}` (read from
    ``aux["lo"]`` and ``aux["hi"]``), the score
    :math:`\max(q_{lo} - y,\; y - q_{hi})` is the signed distance to the
    nearest violated bound, so a calibrated threshold :math:`t` inverts to
    :math:`[q_{lo} - t, q_{hi} + t]`; :math:`t` may be negative when the base
    band is already conservative. Introduced in `Conformalized Quantile
    Regression <https://arxiv.org/abs/1905.03222>`_ (Romano, Patterson and
    Candes, 2019).

    Notes
    -----
    ``prediction`` is unused by the score itself (the heads carry the
    information) but is threaded through for API uniformity; its dtype sets
    the dtype of the returned interval. Calling :meth:`score` or
    :meth:`interval` without both ``aux["lo"]`` and ``aux["hi"]`` raises
    ``ValueError``.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.uq.conformal import QuantileRegressionScore
    >>> _ = torch.manual_seed(0)
    >>> score = QuantileRegressionScore()
    >>> prediction, target = torch.randn(50, 2), torch.randn(50, 2)
    >>> aux = {"lo": prediction - 1.0, "hi": prediction + 1.0}
    >>> score.score(prediction, target, aux).shape
    torch.Size([50, 2])
    >>> threshold = torch.tensor(0.25, dtype=torch.float64)
    >>> lo, hi = score.interval(prediction, threshold, aux)
    >>> lo.shape
    torch.Size([50, 2])
    """

    aux_keys = ("lo", "hi")

    def score(
        self,
        prediction: Float[Tensor, "*dims"],
        target: Float[Tensor, "*dims"],
        aux: Mapping[str, Tensor] | None = None,
    ) -> Float[Tensor, "*dims"]:
        _require_aux(aux, self.aux_keys, type(self).__name__)
        return torch.maximum(aux["lo"] - target, target - aux["hi"])

    def interval(
        self,
        prediction: Float[Tensor, "*dims"],
        threshold: Float[Tensor, "*dims"] | Float[Tensor, ""],
        aux: Mapping[str, Tensor] | None = None,
    ) -> tuple[Float[Tensor, "*dims"], Float[Tensor, "*dims"]]:
        _require_aux(aux, self.aux_keys, type(self).__name__)
        t = _slack_threshold(threshold, prediction, aux["lo"], aux["hi"])
        lo64 = aux["lo"].to(torch.float64) - t
        hi64 = aux["hi"].to(torch.float64) + t
        return _outward_interval(prediction, lo64, hi64)


_Score = AbsoluteErrorScore | NormalizedErrorScore | QuantileRegressionScore
"""The shipped (and serializable) score strategies accepted by the public API."""

_SCORE_REGISTRY: dict[str, type[_NonconformityScore]] = {
    "absolute_error": AbsoluteErrorScore,
    "normalized_error": NormalizedErrorScore,
    "quantile_regression": QuantileRegressionScore,
}
"""Private identifiers for the exact built-in score types."""


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
    raise ``ValueError`` on that combination. The aux entry must be a finite
    real floating-point tensor at every calibration and prediction call.
    Validation precedes reduction and clamping, so neither operation can
    hide NaN or infinity. Finite nonpositive values use the positive floor.
    ``difficulty=None`` on a calibrator or predictor means :math:`s = 1`;
    ``aux=None`` or an entry of ``None`` cannot supply an ``AuxDifficulty``.

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
            Must contain ``key`` with a finite real floating-point tensor of
            shape :math:`(n_{\text{points}}, *\text{dims})`. The default
            ``None`` means no aux mapping and raises ``ValueError`` here;
            an entry of ``None`` raises ``TypeError``.

        Returns
        -------
        torch.Tensor
            Positive difficulty values of shape :math:`(n_{\text{points}},)`,
            reduced by ``max`` over any trailing dimensions and clamped below
            by the larger of ``eps`` and the dtype's smallest positive
            normal value.
        """
        if not isinstance(aux, Mapping) or self.key not in aux:
            raise ValueError(
                f"AuxDifficulty requires aux entry '{self.key}' at every call; "
                "pass aux={key: tensor}."
            )
        s = aux[self.key]
        if not isinstance(s, Tensor):
            raise TypeError(
                f"AuxDifficulty aux '{self.key}' must be a torch.Tensor, got "
                f"{type(s).__name__}."
            )
        check_real(self.key, "difficulty aux", s)
        if s.ndim >= 2:
            # One scale per leading point, matching the CRC point-risk unit.
            s = s.amax(dim=tuple(range(1, s.ndim)))
        return clamp_min_floor(s, self.eps)


def _check_no_double_scale(
    score: _NonconformityScore, difficulty: AuxDifficulty | None
) -> None:
    """Reject an ``AuxDifficulty`` on a key in ``score.scale_aux_keys`` (double scaling)."""
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


def _strategy_kind(strategy: object, registry: Mapping[str, type]) -> str | None:
    """Registry identifier for an exact built-in strategy type, otherwise ``None``."""
    return {cls: kind for kind, cls in registry.items()}.get(type(strategy))


def _snapshot_strategy(strategy: object, registry: Mapping[str, type], what: str):
    """Require and snapshot one of the shipped strategies in ``registry``."""
    if _strategy_kind(strategy, registry) is None:
        names = ", ".join(sorted(cls.__name__ for cls in registry.values()))
        raise TypeError(
            f"{what} must be one of the shipped strategies ({names}); got "
            f"{type(strategy).__name__}."
        )
    return copy.deepcopy(strategy)
