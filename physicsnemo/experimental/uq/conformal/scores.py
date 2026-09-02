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
score is an elementwise residual whose sublevel set
:math:`\{y : \text{score}(\hat y, y) \le t\}` is exactly the interval the
calibrated threshold :math:`t` certifies. The ``...Score`` suffix marks that
distinction; these classes are not error metrics and are not interchangeable
with ``physicsnemo`` metrics.

Scores operate on plain tensors; field-container (``TensorDict``) iteration
is handled by the calibrators. Optional ``aux`` inputs carry the built-in
score data: predicted standard deviations (``"sigma"``) or quantile heads
(``"lo"``/``"hi"``).
"""

import copy
from collections.abc import Mapping

import torch
from jaxtyping import Float
from torch import Tensor

from ._quantile import cast_directed
from ._validation import clamp_min_floor, positive_finite_float

__all__ = [
    "AbsoluteErrorScore",
    "NormalizedErrorScore",
    "QuantileRegressionScore",
]


def _coarsest_finfo(*tensors: Tensor | None) -> torch.finfo:
    """``finfo`` of the least-precise floating dtype among the inputs.

    The slack that protects the inversion must be sized by the dtype the
    score arithmetic actually rounds in, which is driven by the aux tensors
    as much as by the prediction (float16 CQR heads with a float64
    prediction round in float16, not float64).
    """
    infos = [
        torch.finfo(t.dtype)
        for t in tensors
        if t is not None and torch.is_tensor(t) and t.is_floating_point()
    ]
    if not infos:
        return torch.finfo(torch.float64)
    return max(infos, key=lambda fi: fi.eps)


def _slack_threshold(threshold: Tensor, *dtype_sources: Tensor | None) -> Tensor:
    """Threshold in float64 with a conservative inflation for the working dtypes.

    A target whose finite-precision score equals the threshold can sit up
    to 0.5 ulp (of the threshold, in the working dtype) beyond the exact
    inverse, plus rounding from the score's own arithmetic (division,
    normalization). Inflating the threshold by ``4 * eps * |threshold| + 4 *
    tiny``, with ``eps``/``tiny`` taken from the coarsest dtype involved in
    the score arithmetic, dominates that envelope, so the outward-rounded
    interval never excludes a score-admitted target (see
    :func:`_outward_interval`).
    """
    t = threshold.to(torch.float64)
    fi = _coarsest_finfo(*dtype_sources)
    return t + (t.abs() * (4.0 * fi.eps) + 4.0 * fi.tiny)


def _outward_interval(
    prediction: Tensor, lo64: Tensor, hi64: Tensor
) -> tuple[Tensor, Tensor]:
    """Round float64 endpoints outward into the prediction's dtype.

    Together with :func:`_slack_threshold` this implements the package's
    theorem-preserving inversion policy: ``score(prediction, y, aux) <=
    threshold`` (evaluated in the working dtype) implies ``lo <= y <= hi``.
    Intervals are valid, at most a few ulps wider than the exact inverse.
    """
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
    :math:`\{y : \text{score}(\hat y, y) \le t\}` equals
    :math:`[\text{lo}, \text{hi}] = \text{interval}(\hat y, t)` elementwise.

    Notes
    -----
    ``aux_keys`` lists the keys a score reads from the ``aux`` mapping;
    ``scale_aux_keys`` is the subset it divides the residual by (a
    multiplicative scale). The latter is empty for scores that only read aux
    additively (quantile bounds) and is used to detect a double-scaling
    pairing with an
    :class:`~physicsnemo.experimental.uq.conformal.AuxDifficulty` on the same
    key: reading a key is not dividing by it, so a CQR score that reads
    ``lo``/``hi`` remains coherent with an ``AuxDifficulty`` scale.

    Exact finite-sample coverage is only preserved if the interval never
    rounds inward: a target whose finite-precision score is admitted by the
    threshold must land inside the constructed bounds. Interval endpoints
    are therefore computed in float64 with a conservative threshold inflation
    and rounded outward into the working dtype (:func:`_slack_threshold` and
    :func:`_outward_interval`); the result is valid and at most a few ulps
    wider than the exact inverse. This float64 hot path is load-bearing for
    the guarantee and is markedly slower than working-dtype math on
    accelerators. A future optimization may compute the bulk residual in the
    working dtype and reserve float64 for the directed-rounding correction,
    but must re-establish the containment property first.
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
            Auxiliary tensors read by the score (``aux_keys``), each of the
            same shape as ``prediction``. Default is ``None``.

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
            Auxiliary tensors read by the score (``aux_keys``), each of the
            same shape as ``prediction``. Default is ``None``.

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
    no-op in low-precision dtypes. Calling :meth:`score` or :meth:`interval`
    without ``aux["sigma"]`` raises ``ValueError``.

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
        # Divide in float64 (input dtypes embed exactly), then round the
        # result *up* into the callers' dtype: calibration thresholds built
        # from up-rounded scores can only grow, never shrink.
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
        half = _slack_threshold(threshold, prediction, aux["sigma"]) * sigma
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


def _score_kind(score: _NonconformityScore) -> str | None:
    """Identifier for an exact built-in score type, otherwise ``None``."""
    for kind, cls in _SCORE_REGISTRY.items():
        if type(score) is cls:
            return kind
    return None


def _snapshot_score(score: _Score) -> _NonconformityScore:
    """Require and snapshot one of the shipped score strategies.

    The single validation used by calibrators and fitted predictors alike.
    """
    if _score_kind(score) is None:
        names = ", ".join(sorted(cls.__name__ for cls in _SCORE_REGISTRY.values()))
        raise TypeError(
            f"score must be one of the shipped strategies ({names}); got "
            f"{type(score).__name__}."
        )
    return copy.deepcopy(score)
