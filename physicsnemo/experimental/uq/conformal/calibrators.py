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

r"""Single-rank split-conformal calibrators for field predictions.

Each calibrator collects one exchangeable sample per ``update_sample`` call
and returns a fitted
:class:`~physicsnemo.experimental.uq.conformal.ConformalPredictor` from
``finalize``. Inputs may be plain tensors or ``TensorDict`` field
containers. The first accepted sample freezes the container mode and field
schema; every update validates exact prediction/target shapes and finite
theorem-bearing values before committing any state.

The split conformal construction follows `Distribution-Free Predictive
Inference for Regression <https://arxiv.org/abs/1604.04173>`_ (Lei et al.,
2018): with :math:`n_{cal}` calibration scores the fitted threshold is the
:math:`k`-th smallest score, :math:`k = \lceil (n_{cal} + 1)(1 - \alpha)
\rceil`. The conformal risk control (CRC) tier instead follows `Conformal
Risk Control <https://arxiv.org/abs/2208.02814>`_ (Angelopoulos et al.,
2022).
"""

import copy
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction

import torch
from jaxtyping import Float
from tensordict import TensorDict
from torch import Tensor

from ._containers import field_items, pack_fields, slice_aux
from ._quantile import (
    alpha_as_fraction,
    conformal_quantile_index,
    kth_smallest_of_samples,
    require_feasible_alpha,
    validate_alpha,
)
from ._validation import (
    broadcast_difficulty,
    check_aux,
    check_difficulty,
    check_exact_shape,
    check_finite,
    check_point_alignment,
    check_points,
    check_real,
    multi_rank_active,
    normalize_keys,
    points_fingerprint,
    require_matching_keys,
)
from .difficulty import (
    AuxDifficulty,
    _check_no_double_scale,
    _snapshot_difficulty,
)
from .predictors import ConformalPredictor
from .scores import _NonconformityScore, _Score, _snapshot_score

__all__ = [
    "CellwiseCalibrator",
    "FunctionalBandCalibrator",
    "RiskControlCalibrator",
]

# One field's staged record: (key, prediction, target, aux) -> stored value.
_Stage = Callable[[str, Tensor, Tensor, Mapping[str, Tensor] | None], object]


def _normalized_scores(raw: Tensor, difficulty: Tensor | None, key: str) -> Tensor:
    """Compute the theorem-bearing score/difficulty quotient in float64."""
    raw64 = raw.to(torch.float64)
    if difficulty is None:
        return raw64
    difficulty64 = broadcast_difficulty(difficulty, raw, key).to(torch.float64)
    normalized = raw64 / difficulty64
    check_finite(key, "the normalized scores (score / difficulty)", normalized)
    underflow = (normalized == 0) & (raw64 != 0)
    if bool(underflow.any()):
        raise ValueError(
            f"Field '{key}': {int(underflow.sum())} normalized score(s) "
            "underflowed to zero in float64. Rescale the difficulty field "
            "or score before calibration."
        )
    return normalized


def _reject_multi_rank() -> None:
    """Fail closed instead of fitting a rank-local calibration result."""
    if multi_rank_active():
        raise NotImplementedError(
            "Conformal calibration is single-rank only. Gather the exact, "
            "deduplicated calibration samples onto one rank before finalize()."
        )


class _SplitCalibratorBase:
    """Shared strategy snapshots, schema checks, and conformal ranks."""

    def __init__(
        self,
        score: _Score,
        alpha: float,
        *,
        keys: Sequence[str] | None = None,
    ) -> None:
        self._score = _snapshot_score(score)
        self._alpha = validate_alpha(alpha)
        self._keys = normalize_keys(keys)
        self._n = 0
        self._tensor_mode: bool | None = None
        self._schema: tuple[str, ...] | None = None

    @property
    def score(self) -> _NonconformityScore:
        r"""Defensive copy of the fixed calibration score."""
        return copy.deepcopy(self._score)

    @property
    def alpha(self) -> float:
        r"""Target miscoverage or risk level."""
        return self._alpha

    @property
    def keys(self) -> list[str] | None:
        r"""Configured field restriction, or ``None`` for all fields."""
        return list(self._keys) if self._keys is not None else None

    @property
    def n_cal(self) -> int:
        r"""Number of accepted calibration samples."""
        return self._n

    def _validated_fields(
        self,
        prediction: Tensor | TensorDict,
        target: Tensor | TensorDict,
        aux: Mapping | None,
    ) -> tuple[bool, tuple[str, ...], list[tuple[str, Tensor, Tensor, Mapping | None]]]:
        """Validate one sample against the frozen schema without mutating state."""
        prediction_items = field_items(prediction, self._keys)
        target_items = dict(field_items(target, self._keys))
        _tensor_mode = isinstance(prediction, Tensor)
        if _tensor_mode != isinstance(target, Tensor):
            raise TypeError(
                "prediction and target must both be plain tensors or both be "
                "TensorDict field containers."
            )
        if self._tensor_mode is not None and self._tensor_mode != _tensor_mode:
            raise TypeError(
                "Cannot mix plain-tensor and TensorDict updates in one calibrator."
            )
        prediction_keys = tuple(key for key, _ in prediction_items)
        require_matching_keys(
            prediction_keys, target_items, "prediction/target field mismatch"
        )
        if self._schema is not None and prediction_keys != self._schema:
            raise KeyError(
                "Field schema changed across updates: first update had "
                f"{list(self._schema)}, this update has {list(prediction_keys)}."
            )

        fields = []
        for key, prediction_field in prediction_items:
            target_field = target_items[key]
            if prediction_field.numel() == 0:
                raise ValueError(
                    f"Field '{key}': empty sample; every calibration sample "
                    "must contain at least one value."
                )
            check_exact_shape(
                key, "prediction", prediction_field, "target", target_field
            )
            check_real(key, "prediction", prediction_field)
            check_real(key, "target", target_field)
            aux_field = slice_aux(aux, key)
            check_aux(key, self._score, prediction_field, aux_field)
            fields.append((key, prediction_field, target_field, aux_field))
        return _tensor_mode, prediction_keys, fields

    def _collect(
        self,
        prediction: Tensor | TensorDict,
        target: Tensor | TensorDict,
        aux: Mapping | None,
        points: Tensor | None,
        store: dict[str, list],
        stage: _Stage,
    ) -> None:
        """Validate, stage every field, then commit one sample transactionally.

        Nothing is written to ``store`` or the schema until every field has
        passed validation and ``stage``, so a rejected sample leaves the
        calibrator exactly as it was. A supplied ``points`` tensor must
        satisfy the coordinate contract and align with every field's leading
        (point) axis so a contradictory coordinate tensor cannot be silently
        accepted.
        """
        with torch.no_grad():
            if points is not None:
                check_points(points)
            _tensor_mode, schema, fields = self._validated_fields(
                prediction, target, aux
            )
            staged: dict[str, object] = {}
            for key, prediction_field, target_field, aux_field in fields:
                if points is not None:
                    check_point_alignment(key, prediction_field, points, "prediction")
                staged[key] = stage(key, prediction_field, target_field, aux_field)
            self._tensor_mode, self._schema = _tensor_mode, schema
            for key, record in staged.items():
                store.setdefault(key, []).append(record)
            self._n += 1

    def _require_finalizable(self) -> None:
        _reject_multi_rank()
        if self._n == 0:
            raise RuntimeError("No calibration samples collected.")

    def _conformal_thresholds(
        self, store: Mapping[str, list[Tensor]]
    ) -> Tensor | TensorDict:
        k = conformal_quantile_index(self._n, self._alpha)
        return pack_fields(
            {
                key: kth_smallest_of_samples(per_sample, k)
                for key, per_sample in store.items()
            }
        )

    def _build_predictor(
        self, tier: str, thresholds: Tensor | TensorDict, **state
    ) -> ConformalPredictor:
        return ConformalPredictor(
            tier=tier,
            score=self._score,
            alpha=self._alpha,
            n_cal=self._n,
            thresholds=thresholds,
            **state,
        )


class CellwiseCalibrator(_SplitCalibratorBase):
    r"""Calibrate one exact split-conformal threshold per output element.

    Use this tier when every calibration and deployment sample lives on the
    same fixed discretization (the same mesh or grid) and per-element
    intervals are wanted, as in the field-level conformal prediction of
    `Gopakumar et al., 2024 <https://arxiv.org/abs/2408.09881>`_. The
    coordinate tensor is mandatory on every sample; its dtype, shape, values,
    and ordering are fingerprinted exactly, so equal point counts alone do
    not establish mesh identity.

    Guarantee: on that fixed discretization and under exchangeability of
    calibration and test samples, every calibrated output element has
    marginal coverage
    :math:`\mathbb{P}(\text{lo} \le y \le \text{hi}) \ge 1 - \alpha`. This is
    not a simultaneous whole-field guarantee.

    Parameters
    ----------
    score : AbsoluteErrorScore | NormalizedErrorScore | QuantileRegressionScore
        One of the shipped score strategies, snapshotted at construction.
    alpha : float
        Target miscoverage level in :math:`(0, 1)`.
    keys : Sequence[str], optional
        Restrict calibration to this subset of ``TensorDict`` fields. The
        fitted :class:`~physicsnemo.experimental.uq.conformal.ConformalPredictor`
        carries the restriction: its ``predict_interval`` selects these
        fields automatically from a superset container. By default every
        field is calibrated.

    Notes
    -----
    Every element's threshold is the :math:`k`-th smallest of its
    :math:`n_{cal}` calibration scores,
    :math:`k = \lceil (n_{cal} + 1)(1 - \alpha) \rceil`, computed exactly
    with ``torch.kthvalue``. Per-sample scores are retained on the CPU until
    :meth:`finalize`, so memory grows as :math:`n_{cal}` times the field
    size. Calibration is single-rank: :meth:`finalize` raises
    ``NotImplementedError`` when an initialized ``torch.distributed`` group
    spans several ranks, and ``ValueError`` when :math:`\alpha < 1 / (n_{cal}
    + 1)`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.uq.conformal import (
    ...     AbsoluteErrorScore, CellwiseCalibrator,
    ... )
    >>> _ = torch.manual_seed(0)
    >>> points = torch.rand(50, 3)
    >>> calibrator = CellwiseCalibrator(AbsoluteErrorScore(), alpha=0.1)
    >>> for _ in range(20):
    ...     prediction = torch.randn(50, 2)
    ...     target = prediction + 0.1 * torch.randn(50, 2)
    ...     calibrator.update_sample(prediction, target, points=points)
    >>> predictor = calibrator.finalize()
    >>> lo, hi = predictor.predict_interval(torch.randn(50, 2), points=points)
    >>> lo.shape
    torch.Size([50, 2])
    """

    def __init__(
        self,
        score: _Score,
        alpha: float,
        *,
        keys: Sequence[str] | None = None,
    ) -> None:
        super().__init__(score, alpha, keys=keys)
        self._scores: dict[str, list[Tensor]] = {}
        self._shapes: dict[str, tuple[int, ...]] = {}
        self._mesh_fingerprint: str | None = None

    @property
    def mesh_fingerprint(self) -> str | None:
        r"""Exact calibration-mesh fingerprint after the first accepted sample."""
        return self._mesh_fingerprint

    def update_sample(
        self,
        prediction: Float[Tensor, "*dims"] | TensorDict,
        target: Float[Tensor, "*dims"] | TensorDict,
        *,
        aux: Mapping[str, Float[Tensor, "*dims"]] | Mapping[str, Mapping] | None = None,
        points: Float[Tensor, "n_points n_spatial_dims"] | None = None,
    ) -> None:
        r"""Collect one sample on the fixed calibration mesh.

        Parameters
        ----------
        prediction : torch.Tensor | TensorDict
            Model output of shape :math:`(n_{\text{points}}, *\text{dims})`,
            or a field container of such tensors.
        target : torch.Tensor | TensorDict
            Observed values, same shape and container type as ``prediction``.
        aux : Mapping[str, torch.Tensor] | Mapping[str, Mapping], optional
            Auxiliary tensors read by the score, each of the same shape as
            ``prediction``. For ``TensorDict`` inputs, nest the mapping by
            field name. Default is ``None``.
        points : torch.Tensor, optional
            Mesh coordinates of shape
            :math:`(n_{\text{points}}, n_{\text{spatial\_dims}})`. Required
            for this tier and must be identical on every call. Default is
            ``None``.

        Returns
        -------
        None
            The sample's scores are committed to the calibrator.

        Notes
        -----
        Nothing is committed unless every field validates, so a rejected
        sample leaves the calibrator unchanged. Raises ``ValueError`` when
        ``points`` is missing or differs from the first accepted mesh, on a
        shape mismatch, or on non-finite values.
        """
        if points is None:
            raise ValueError(
                "CellwiseCalibrator requires points= on every calibration "
                "sample to verify exact mesh identity."
            )
        fingerprint = points_fingerprint(points)
        if self._mesh_fingerprint is not None and fingerprint != self._mesh_fingerprint:
            raise ValueError(
                "Cellwise calibration requires the same mesh coordinates, "
                "dtype, and ordering for every sample; this sample differs "
                "from the first accepted calibration mesh."
            )

        def stage(key, prediction_field, target_field, aux_field):
            check_point_alignment(key, target_field, points, "target")
            score = check_finite(
                key,
                "the nonconformity scores",
                self._score.score(prediction_field, target_field, aux_field),
            )
            shape = tuple(score.shape)
            if key in self._shapes and shape != self._shapes[key]:
                raise ValueError(
                    f"Field '{key}': score shape {shape} differs from the "
                    f"first sample's {self._shapes[key]}. Cellwise "
                    "calibration requires an identical output layout."
                )
            return score.detach().cpu()

        self._collect(prediction, target, aux, points, self._scores, stage)
        for key, scores in self._scores.items():
            self._shapes.setdefault(key, tuple(scores[-1].shape))
        self._mesh_fingerprint = fingerprint

    def finalize(self) -> ConformalPredictor:
        r"""Fit the per-element thresholds and build the predictor.

        Returns
        -------
        ConformalPredictor
            A cellwise
            :class:`~physicsnemo.experimental.uq.conformal.ConformalPredictor`
            whose thresholds have the calibrated field shape and which
            carries the calibration-mesh fingerprint.
        """
        self._require_finalizable()
        return self._build_predictor(
            "cellwise",
            self._conformal_thresholds(self._scores),
            mesh_fingerprint=self._mesh_fingerprint,
        )


class _ScaledCalibratorBase(_SplitCalibratorBase):
    """Shared difficulty handling for the tiers that permit varying point sets."""

    def __init__(
        self,
        score: _Score,
        alpha: float,
        *,
        difficulty: AuxDifficulty | None = None,
        keys: Sequence[str] | None = None,
    ) -> None:
        super().__init__(score, alpha, keys=keys)
        self._difficulty = _snapshot_difficulty(difficulty)
        _check_no_double_scale(self._score, self._difficulty)

    @property
    def difficulty(self) -> AuxDifficulty | None:
        r"""Defensive copy of the fixed difficulty field, or ``None``."""
        return copy.deepcopy(self._difficulty)

    def _normalized_scores_stage(self, points: Tensor | None) -> _Stage:
        """Stage function producing each field's float64 normalized scores."""

        def stage(key, prediction_field, target_field, aux_field):
            difficulty = (
                None
                if self._difficulty is None
                else check_difficulty(self._difficulty(points, aux_field))
            )
            raw = check_finite(
                key,
                "the nonconformity scores",
                self._score.score(prediction_field, target_field, aux_field),
            )
            return _normalized_scores(raw, difficulty, key)

        return stage


class FunctionalBandCalibrator(_ScaledCalibratorBase):
    r"""Calibrate a scalar simultaneous whole-field band per output field.

    Use this tier when a single band must contain an entire field at once,
    for instance to certify a worst-case error over a mesh, and the point
    sets may differ between calibration and deployment. Each sample
    contributes the supremum of its difficulty-normalized scores,
    :math:`\max_x \text{score}(x) / s(x)`, following the sup-norm band
    construction of `Conformal prediction bands for multivariate functional
    data <https://arxiv.org/abs/2106.01792>`_ (Diquigiovanni, Fontana and
    Vantini, 2021).

    Guarantee: under exchangeability of whole field samples, the band
    :math:`\text{threshold} \cdot s(x)` contains every observed point of a
    fresh field simultaneously with probability at least :math:`1 - \alpha`.

    Parameters
    ----------
    score : AbsoluteErrorScore | NormalizedErrorScore | QuantileRegressionScore
        One of the shipped score strategies, snapshotted at construction.
    alpha : float
        Target miscoverage level in :math:`(0, 1)`.
    difficulty : AuxDifficulty, optional
        Per-point positive scale field :math:`s(x)` multiplying the fitted
        threshold. Default is ``None`` (no scaling, :math:`s = 1`).
    keys : Sequence[str], optional
        Restrict calibration to this subset of ``TensorDict`` fields (see
        :class:`~physicsnemo.experimental.uq.conformal.CellwiseCalibrator`).
        By default every field is calibrated.

    Notes
    -----
    The threshold is the :math:`k`-th smallest of the :math:`n_{cal}`
    per-sample suprema, :math:`k = \lceil (n_{cal} + 1)(1 - \alpha) \rceil`.
    Normalized scores are computed in float64. Calibration is single-rank:
    :meth:`finalize` raises ``NotImplementedError`` when an initialized
    ``torch.distributed`` group spans several ranks, and ``ValueError`` when
    :math:`\alpha < 1 / (n_{cal} + 1)`. Pairing a score that divides by an
    aux key with an ``AuxDifficulty`` on the same key raises ``ValueError``
    at construction.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.uq.conformal import (
    ...     AbsoluteErrorScore, FunctionalBandCalibrator,
    ... )
    >>> _ = torch.manual_seed(0)
    >>> calibrator = FunctionalBandCalibrator(AbsoluteErrorScore(), alpha=0.1)
    >>> for n_points in range(40, 60):
    ...     prediction = torch.randn(n_points, 2)
    ...     target = prediction + 0.1 * torch.randn(n_points, 2)
    ...     calibrator.update_sample(prediction, target)
    >>> predictor = calibrator.finalize()
    >>> lo, hi = predictor.predict_interval(torch.randn(80, 2))
    >>> hi.shape
    torch.Size([80, 2])
    """

    def __init__(
        self,
        score: _Score,
        alpha: float,
        *,
        difficulty: AuxDifficulty | None = None,
        keys: Sequence[str] | None = None,
    ) -> None:
        super().__init__(score, alpha, difficulty=difficulty, keys=keys)
        self._sup_scores: dict[str, list[Tensor]] = {}

    def update_sample(
        self,
        prediction: Float[Tensor, "*dims"] | TensorDict,
        target: Float[Tensor, "*dims"] | TensorDict,
        *,
        aux: Mapping[str, Float[Tensor, "*dims"]] | Mapping[str, Mapping] | None = None,
        points: Float[Tensor, "n_points n_spatial_dims"] | None = None,
    ) -> None:
        r"""Collect one sample's supremum of difficulty-normalized scores.

        Parameters
        ----------
        prediction : torch.Tensor | TensorDict
            Model output of shape :math:`(n_{\text{points}}, *\text{dims})`
            (any shape :math:`(*\text{dims})` when neither ``points`` nor a
            difficulty field is used), or a field container of such tensors.
        target : torch.Tensor | TensorDict
            Observed values, same shape and container type as ``prediction``.
        aux : Mapping[str, torch.Tensor] | Mapping[str, Mapping], optional
            Auxiliary tensors read by the score or the difficulty field, each
            of the same shape as ``prediction``. For ``TensorDict`` inputs,
            nest the mapping by field name. Default is ``None``.
        points : torch.Tensor, optional
            Mesh coordinates of shape
            :math:`(n_{\text{points}}, n_{\text{spatial\_dims}})`; may differ
            between samples. Default is ``None``.

        Returns
        -------
        None
            The sample's supremum score is committed to the calibrator.

        Notes
        -----
        Nothing is committed unless every field validates, so a rejected
        sample leaves the calibrator unchanged. Raises ``ValueError`` on a
        shape mismatch, non-finite values, or a non-positive difficulty.
        """
        normalized = self._normalized_scores_stage(points)
        self._collect(
            prediction,
            target,
            aux,
            points,
            self._sup_scores,
            lambda *field: normalized(*field).amax().detach().cpu(),
        )

    def finalize(self) -> ConformalPredictor:
        r"""Fit the scalar band threshold and build the predictor.

        Returns
        -------
        ConformalPredictor
            A functional
            :class:`~physicsnemo.experimental.uq.conformal.ConformalPredictor`
            with one float64 scalar threshold per field and the configured
            difficulty field.
        """
        self._require_finalizable()
        return self._build_predictor(
            "functional",
            self._conformal_thresholds(self._sup_scores),
            difficulty=self._difficulty,
        )


def _crc_threshold(
    sorted_scores: list[Tensor], lengths: list[int], alpha: float
) -> float:
    r"""Return the smallest observed threshold satisfying the exact CRC bound.

    Conformal risk control (CRC) selects the smallest :math:`\lambda` with
    :math:`(\hat R(\lambda) + 1)/(n + 1) \le \alpha`, where :math:`\hat R`
    is the summed per-sample point-miscoverage fraction over the :math:`n`
    calibration samples (`Conformal Risk Control
    <https://arxiv.org/abs/2208.02814>`_, Angelopoulos et al., 2022).
    Feasibility is evaluated with rational arithmetic over integer exceedance
    counts and the user's declared decimal ``alpha``. Candidate selection is
    an exact binary search over observed float64 scores, not value-space
    bisection.
    """
    n = len(sorted_scores)

    alpha_exact = alpha_as_fraction(alpha)
    require_feasible_alpha(n, alpha)

    scores64 = [scores.to(torch.float64) for scores in sorted_scores]

    def corrected_risk(candidate: Tensor) -> Fraction:
        total_loss = Fraction()
        for scores, length in zip(scores64, lengths):
            probe = candidate.to(device=scores.device)
            exceed = scores.numel() - int(torch.searchsorted(scores, probe, right=True))
            total_loss += Fraction(exceed, length)
        return (total_loss + 1) / (n + 1)

    # Duplicate candidates are harmless: corrected_risk is a function of the
    # probe value alone and monotone non-increasing in it, so a sorted list
    # with repeats selects the same smallest feasible threshold that a
    # deduplicated one would, without torch.unique's extra full-corpus
    # workspace on top of the pooled buffer.
    candidates = torch.cat(scores64).sort().values
    if corrected_risk(candidates[-1]) > alpha_exact:  # pragma: no cover
        raise RuntimeError("CRC bound is infeasible even at the maximal score.")

    lo = -1
    hi = candidates.numel() - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if corrected_risk(candidates[mid]) <= alpha_exact:
            hi = mid
        else:
            lo = mid
    threshold = candidates[hi]
    if corrected_risk(threshold) > alpha_exact:  # pragma: no cover
        raise RuntimeError("CRC threshold selection violated its risk bound.")
    return float(threshold)


def _sorted_point_scores(normalized: Tensor) -> tuple[Tensor, int]:
    """Reduce trailing dims to point events; sorted CPU scores plus count."""
    if normalized.ndim > 1:
        normalized = normalized.amax(dim=tuple(range(1, normalized.ndim)))
    point_scores = normalized.reshape(-1)
    return point_scores.detach().cpu().sort().values, point_scores.numel()


class RiskControlCalibrator(_ScaledCalibratorBase):
    r"""Calibrate expected miscovered-point risk across varying point sets.

    Use this tier when point sets vary between samples and the quantity to
    control is the expected fraction of miscovered points rather than
    whole-field containment; it gives much tighter bands than
    :class:`~physicsnemo.experimental.uq.conformal.FunctionalBandCalibrator`
    at the cost of a weaker, risk-based statement. A point is miscovered
    when any of its trailing components is outside the band. Each sample
    contributes its own miscovered-point fraction, and samples receive equal
    weight regardless of point count. The threshold is chosen by conformal
    risk control (CRC), `Conformal Risk Control
    <https://arxiv.org/abs/2208.02814>`_ (Angelopoulos, Bates, Fisch, Lei
    and Schuster, 2022).

    Guarantee: under exchangeability of whole samples, the expected
    miscovered-point fraction of a fresh sample is at most :math:`\alpha`.
    The fitted threshold :math:`\lambda` is the smallest observed score with
    :math:`(\hat R(\lambda) + 1)/(n_{cal} + 1) \le \alpha`, where
    :math:`\hat R` sums the per-sample miscovered fractions; this requires
    :math:`\alpha \ge 1 / (n_{cal} + 1)`.

    Parameters
    ----------
    score : AbsoluteErrorScore | NormalizedErrorScore | QuantileRegressionScore
        One of the shipped score strategies, snapshotted at construction.
    alpha : float
        Target expected point-miscoverage risk in :math:`(0, 1)`.
    difficulty : AuxDifficulty, optional
        Per-point positive scale field :math:`s(x)` multiplying the fitted
        threshold. Default is ``None`` (no scaling, :math:`s = 1`).
    keys : Sequence[str], optional
        Restrict calibration to this subset of ``TensorDict`` fields (see
        :class:`~physicsnemo.experimental.uq.conformal.CellwiseCalibrator`).
        By default every field is calibrated.

    Notes
    -----
    Each sample's sorted float64 point scores are retained on the CPU until
    :meth:`finalize`, which pools them to search the threshold. The risk
    bound is evaluated with exact rational arithmetic on the declared
    ``alpha``. Calibration is single-rank: :meth:`finalize` raises
    ``NotImplementedError`` when an initialized ``torch.distributed`` group
    spans several ranks, and ``ValueError`` when
    :math:`\alpha < 1 / (n_{cal} + 1)`. Pairing a score that divides by an
    aux key with an ``AuxDifficulty`` on the same key raises ``ValueError``
    at construction.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.uq.conformal import (
    ...     AbsoluteErrorScore, AuxDifficulty, RiskControlCalibrator,
    ... )
    >>> _ = torch.manual_seed(0)
    >>> calibrator = RiskControlCalibrator(
    ...     AbsoluteErrorScore(), alpha=0.1, difficulty=AuxDifficulty("sigma")
    ... )
    >>> for n_points in range(40, 60):
    ...     prediction = torch.randn(n_points, 2)
    ...     sigma = torch.rand(n_points, 2) + 0.5
    ...     target = prediction + sigma * torch.randn(n_points, 2)
    ...     calibrator.update_sample(prediction, target, aux={"sigma": sigma})
    >>> predictor = calibrator.finalize()
    >>> sigma = torch.rand(30, 2) + 0.5
    >>> lo, hi = predictor.predict_interval(torch.randn(30, 2), aux={"sigma": sigma})
    >>> lo.shape
    torch.Size([30, 2])
    """

    def __init__(
        self,
        score: _Score,
        alpha: float,
        *,
        difficulty: AuxDifficulty | None = None,
        keys: Sequence[str] | None = None,
    ) -> None:
        super().__init__(score, alpha, difficulty=difficulty, keys=keys)
        self._samples: dict[str, list[tuple[Tensor, int]]] = {}

    def update_sample(
        self,
        prediction: Float[Tensor, "*dims"] | TensorDict,
        target: Float[Tensor, "*dims"] | TensorDict,
        *,
        aux: Mapping[str, Float[Tensor, "*dims"]] | Mapping[str, Mapping] | None = None,
        points: Float[Tensor, "n_points n_spatial_dims"] | None = None,
    ) -> None:
        r"""Collect one sample's sorted difficulty-normalized point scores.

        Parameters
        ----------
        prediction : torch.Tensor | TensorDict
            Model output of shape :math:`(n_{\text{points}}, *\text{dims})`,
            or a field container of such tensors. Trailing dimensions are
            reduced by ``max`` into one score per point.
        target : torch.Tensor | TensorDict
            Observed values, same shape and container type as ``prediction``.
        aux : Mapping[str, torch.Tensor] | Mapping[str, Mapping], optional
            Auxiliary tensors read by the score or the difficulty field, each
            of the same shape as ``prediction``. For ``TensorDict`` inputs,
            nest the mapping by field name. Default is ``None``.
        points : torch.Tensor, optional
            Mesh coordinates of shape
            :math:`(n_{\text{points}}, n_{\text{spatial\_dims}})`; may differ
            between samples. Default is ``None``.

        Returns
        -------
        None
            The sample's sorted point scores are committed to the calibrator.

        Notes
        -----
        Nothing is committed unless every field validates, so a rejected
        sample leaves the calibrator unchanged. Raises ``ValueError`` on a
        shape mismatch, non-finite values, or a non-positive difficulty.
        """
        normalized = self._normalized_scores_stage(points)
        self._collect(
            prediction,
            target,
            aux,
            points,
            self._samples,
            lambda *field: _sorted_point_scores(normalized(*field)),
        )

    def finalize(self) -> ConformalPredictor:
        r"""Fit the CRC threshold and build the predictor.

        Returns
        -------
        ConformalPredictor
            A risk-control
            :class:`~physicsnemo.experimental.uq.conformal.ConformalPredictor`
            with one float64 scalar threshold per field and the configured
            difficulty field.
        """
        self._require_finalizable()
        thresholds = {
            key: torch.tensor(
                _crc_threshold(
                    [scores for scores, _ in samples],
                    [length for _, length in samples],
                    self._alpha,
                ),
                dtype=torch.float64,
            )
            for key, samples in self._samples.items()
        }
        return self._build_predictor(
            "risk_control", pack_fields(thresholds), difficulty=self._difficulty
        )
