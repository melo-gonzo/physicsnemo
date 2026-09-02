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

r"""The fitted in-memory conformal predictor value type."""

import copy
from collections.abc import Mapping
from pathlib import Path

import torch
from jaxtyping import Float
from tensordict import TensorDict
from torch import Tensor

from ._containers import TENSOR_KEY, field_items, pack_fields, slice_aux
from ._quantile import conformal_quantile_index, validate_alpha, validate_n_cal
from ._validation import (
    TIERS,
    Tier,
    broadcast_difficulty,
    check_aux,
    check_difficulty,
    check_finite,
    check_floating,
    check_point_alignment,
    check_points,
    check_real,
    points_fingerprint,
    require_matching_keys,
    validate_provenance,
)
from .difficulty import (
    AuxDifficulty,
    _check_no_double_scale,
    _snapshot_difficulty,
)
from .scores import (
    _NonconformityScore,
    _Score,
    _score_kind,
    _snapshot_score,
)

__all__ = ["ConformalPredictor"]

_SIGNED_THRESHOLD_KINDS = ("quantile_regression",)


def _validate_alpha_n_cal(alpha, n_cal) -> tuple[float, int]:
    """Validate the fitted sample count and target error level."""
    alpha = validate_alpha(alpha)
    n_cal = validate_n_cal(n_cal)
    # A fitted predictor must describe an order statistic that its claimed
    # calibration population could actually produce. This exact-rational
    # check is shared with split calibration and direct predictor construction.
    conformal_quantile_index(n_cal, alpha)
    return alpha, n_cal


def _validate_mesh_fingerprint(value: object) -> str:
    """Validate the SHA-256 hex digest produced by ``points_fingerprint``."""
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            "mesh_fingerprint must be a 64-character lowercase SHA-256 hex digest."
        )
    return value


def _validate_thresholds(
    tier: str,
    thresholds: Tensor | TensorDict,
    score_kind: str,
) -> dict[str, Tensor]:
    """Validate, detach, and clone fitted thresholds."""
    out: dict[str, Tensor] = {}
    for key, value in field_items(thresholds):
        check_floating(key, "threshold", value)
        if value.numel() == 0:
            raise ValueError(f"Field '{key}': empty threshold tensor.")
        check_finite(key, "threshold", value)
        if tier == "cellwise":
            if value.ndim == 0:
                raise ValueError(
                    f"Field '{key}': cellwise thresholds must have at least one "
                    "dimension, got a scalar."
                )
            threshold = value
        else:
            if value.ndim != 0:
                raise ValueError(
                    f"Field '{key}': {tier} thresholds must be scalars, got "
                    f"shape {tuple(value.shape)}."
                )
            # Scalar fitted thresholds always live in float64. This prevents
            # deployment-time scalar promotion from shrinking a calibrated rank.
            threshold = value.to(torch.float64)
        if score_kind not in _SIGNED_THRESHOLD_KINDS and bool((threshold < 0).any()):
            raise ValueError(
                f"Field '{key}': negative threshold for nonnegative score "
                f"kind {score_kind!r}."
            )
        out[key] = threshold.detach().clone()
    return out


class ConformalPredictor:
    r"""A fitted conformal interval rule for exactly one guarantee tier.

    Instances are produced by a calibrator's ``finalize()`` or by
    :meth:`load`; the constructor is public so artifacts and tests can build
    one directly. A predictor holds the fitted threshold (the conformal
    quantile of the calibration scores) for each field and turns a new
    prediction into lower and upper bounds through
    :meth:`predict_interval`. Cellwise predictors carry the exact
    calibration-mesh fingerprint; functional and conformal risk control
    (CRC) predictors may instead carry a difficulty field :math:`s(x)`.

    Parameters
    ----------
    tier : {"cellwise", "functional", "risk_control"}
        Guarantee tier the thresholds were fitted for.
    score : AbsoluteErrorScore | NormalizedErrorScore | QuantileRegressionScore
        Nonconformity score used at calibration, snapshotted at construction.
    alpha : float
        Target miscoverage (or risk) level in :math:`(0, 1)`.
    n_cal : int
        Number of calibration samples. Must satisfy
        :math:`\alpha \ge 1 / (n_{cal} + 1)` so the conformal rank
        :math:`k = \lceil (n_{cal} + 1)(1 - \alpha) \rceil` exists.
    thresholds : torch.Tensor | TensorDict
        Fitted thresholds: one tensor of shape :math:`(*\text{dims})` per
        field for the cellwise tier, one scalar per field otherwise. Scalars
        are stored in float64.
    difficulty : AuxDifficulty, optional
        Difficulty field multiplying the scalar threshold at prediction time
        (functional and risk-control tiers only). Default is ``None``.
    mesh_fingerprint : str, optional
        SHA-256 hex digest of the calibration coordinates (cellwise tier
        only). Default is ``None``.
    provenance : Mapping, optional
        Strict-JSON metadata stored alongside the artifact. Default is
        ``None``.

    Notes
    -----
    Construction raises ``ValueError`` when the tier-dependent state is
    inconsistent (a cellwise predictor without ``mesh_fingerprint`` or with a
    difficulty field, a scalar-threshold tier given a non-scalar threshold, a
    negative threshold for a nonnegative score, or an infeasible
    ``alpha``/``n_cal`` pair). Predictors round-trip through
    ``weights_only``-safe artifacts via :meth:`save` and :meth:`load`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.uq.conformal import (
    ...     AbsoluteErrorScore, ConformalPredictor,
    ... )
    >>> _ = torch.manual_seed(0)
    >>> predictor = ConformalPredictor(
    ...     tier="functional", score=AbsoluteErrorScore(), alpha=0.1, n_cal=20,
    ...     thresholds=torch.tensor(0.3, dtype=torch.float64),
    ... )
    >>> lo, hi = predictor.predict_interval(torch.randn(50, 2))
    >>> lo.shape
    torch.Size([50, 2])
    """

    def __init__(
        self,
        *,
        tier: Tier,
        score: _Score,
        alpha: float,
        n_cal: int,
        thresholds: Float[Tensor, "*dims"] | Float[Tensor, ""] | TensorDict,
        difficulty: AuxDifficulty | None = None,
        mesh_fingerprint: str | None = None,
        provenance: Mapping | None = None,
    ) -> None:
        if tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}, got {tier!r}.")
        alpha, n_cal = _validate_alpha_n_cal(alpha, n_cal)
        score_snapshot = _snapshot_score(score)
        score_kind = _score_kind(score_snapshot)

        if tier == "cellwise":
            if difficulty is not None:
                raise ValueError(
                    "A cellwise predictor must not have a difficulty field."
                )
            if mesh_fingerprint is None:
                raise ValueError(
                    "A cellwise predictor requires mesh_fingerprint from the "
                    "calibration coordinates."
                )
            difficulty_snapshot = None
            mesh_snapshot = _validate_mesh_fingerprint(mesh_fingerprint)
        else:
            if mesh_fingerprint is not None:
                raise ValueError(
                    f"A {tier} predictor must not carry mesh_fingerprint; its "
                    "calibration statistic permits varying point sets."
                )
            difficulty_snapshot = _snapshot_difficulty(difficulty)
            _check_no_double_scale(score_snapshot, difficulty_snapshot)
            mesh_snapshot = None

        self._tier = tier
        self._score = score_snapshot
        self._alpha = alpha
        self._n_cal = n_cal
        self._thresholds_by_key = _validate_thresholds(tier, thresholds, score_kind)
        self._difficulty = difficulty_snapshot
        self._mesh_fingerprint = mesh_snapshot
        self._provenance = {} if provenance is None else validate_provenance(provenance)

    @property
    def tier(self) -> Tier:
        r"""The fitted guarantee tier."""
        return self._tier

    @property
    def alpha(self) -> float:
        r"""The fitted target miscoverage or risk level."""
        return self._alpha

    @property
    def n_cal(self) -> int:
        r"""The number of exchangeable calibration samples."""
        return self._n_cal

    @property
    def score(self) -> _NonconformityScore:
        r"""A defensive copy of the fitted built-in score strategy."""
        return copy.deepcopy(self._score)

    @property
    def difficulty(self) -> AuxDifficulty | None:
        r"""A defensive copy of the fitted difficulty field, or ``None``."""
        return copy.deepcopy(self._difficulty)

    @property
    def mesh_fingerprint(self) -> str | None:
        r"""The exact calibration-mesh digest for a cellwise predictor."""
        return self._mesh_fingerprint

    @property
    def provenance(self) -> dict:
        r"""A defensive copy of strict-JSON artifact provenance."""
        return copy.deepcopy(self._provenance)

    @property
    def _tensor_mode(self) -> bool:
        """Whether calibration used a plain tensor rather than a ``TensorDict``."""
        return set(self._thresholds_by_key) == {TENSOR_KEY}

    @property
    def keys(self) -> list[str] | None:
        r"""Sorted calibrated field names, or ``None`` for plain-tensor mode."""
        return None if self._tensor_mode else sorted(self._thresholds_by_key)

    @property
    def thresholds(self) -> Tensor | TensorDict:
        r"""A defensive clone of the fitted thresholds in their public container."""
        return pack_fields(
            {
                key: value.detach().clone()
                for key, value in self._thresholds_by_key.items()
            }
        )

    def to(self, device: torch.device | str) -> "ConformalPredictor":
        r"""Move the fitted thresholds to ``device`` in place.

        Follows the ``torch.nn.Module.to`` convention. Only threshold storage
        moves, so the per-call device transfer in :meth:`predict_interval`
        becomes a no-op for predictions on that device.

        Parameters
        ----------
        device : torch.device | str
            Target device for the threshold tensors.

        Returns
        -------
        ConformalPredictor
            ``self``, for chaining.
        """
        self._thresholds_by_key = {
            key: value.to(device=device)
            for key, value in self._thresholds_by_key.items()
        }
        return self

    def _threshold_for(
        self,
        key: str,
        prediction: Tensor,
        aux: Mapping[str, Tensor] | None,
        points: Tensor | None,
    ) -> Tensor:
        threshold = self._thresholds_by_key[key]
        if self._tier == "cellwise":
            if threshold.shape != prediction.shape:
                raise ValueError(
                    f"Field '{key}': prediction shape {tuple(prediction.shape)} "
                    f"differs from calibrated shape {tuple(threshold.shape)}."
                )
            # A no-op when the predictor was moved with .to(device) first.
            return threshold.to(device=prediction.device)

        scalar = threshold.to(device=prediction.device)
        if self._difficulty is None:
            return scalar
        difficulty = check_difficulty(self._difficulty(points, aux))
        difficulty = difficulty.to(device=prediction.device, dtype=torch.float64)
        return scalar * broadcast_difficulty(difficulty, prediction, key)

    def predict_interval(
        self,
        prediction: Float[Tensor, "*dims"] | TensorDict,
        *,
        aux: Mapping[str, Float[Tensor, "*dims"]] | Mapping[str, Mapping] | None = None,
        points: Float[Tensor, "n_points n_spatial_dims"] | None = None,
    ) -> tuple[
        Float[Tensor, "*dims"] | TensorDict, Float[Tensor, "*dims"] | TensorDict
    ]:
        r"""Construct lower and upper bounds for one prediction sample.

        A ``TensorDict`` prediction may carry a superset of the calibrated
        fields (e.g. when calibration restricted ``keys=``): the fitted
        fields are selected automatically and the returned containers hold
        exactly the calibrated fields.

        Parameters
        ----------
        prediction : torch.Tensor | TensorDict
            Model output of shape :math:`(*\text{dims})`, or a field container
            of such tensors. For the cellwise tier the shape must equal the
            calibrated shape; when ``points`` is supplied the leading
            dimension must be :math:`n_{\text{points}}`.
        aux : Mapping[str, torch.Tensor] | Mapping[str, Mapping], optional
            Auxiliary tensors read by the score or the difficulty field, each
            of shape :math:`(*\text{dims})`. For ``TensorDict`` inputs, nest
            the mapping by field name. Default is ``None``.
        points : torch.Tensor, optional
            Mesh coordinates of shape
            :math:`(n_{\text{points}}, n_{\text{spatial\_dims}})`. Required
            for the cellwise tier, where they must reproduce the calibration
            mesh fingerprint exactly. Default is ``None``.

        Returns
        -------
        tuple[torch.Tensor | TensorDict, torch.Tensor | TensorDict]
            Lower and upper bounds ``(lo, hi)`` in the container type and
            dtype of ``prediction``, each of shape :math:`(*\text{dims})`.

        Notes
        -----
        Input validation forces host-device synchronizations on every call:
        finiteness checks on the prediction, aux, and difficulty values, and
        for the cellwise tier a CPU SHA-256 digest of ``points``. This method
        is therefore not intended for use inside ``torch.compile`` regions.
        Calling :meth:`to` with the prediction device beforehand avoids the
        per-call threshold transfer. Raises ``ValueError`` on a shape, mesh,
        or finiteness violation and ``KeyError`` when the prediction fields
        do not match the fitted fields.
        """
        selection = None if self._tensor_mode else list(self._thresholds_by_key)
        items = field_items(prediction, selection)
        require_matching_keys(
            (key for key, _ in items),
            self._thresholds_by_key,
            "Prediction fields must exactly match the fitted predictor",
        )
        if self._tier == "cellwise":
            if points is None:
                raise ValueError(
                    "Cellwise conformal prediction requires points= to verify "
                    "the calibration mesh."
                )
            fingerprint = points_fingerprint(points)
            if fingerprint != self._mesh_fingerprint:
                raise ValueError(
                    "Cellwise conformal prediction requires the exact calibration "
                    "mesh coordinates, dtype, and ordering; this mesh differs. "
                    "Calibrate with FunctionalBandCalibrator or "
                    "RiskControlCalibrator when point sets vary."
                )
        elif points is not None:
            check_points(points)

        lo_out: dict[str, Tensor] = {}
        hi_out: dict[str, Tensor] = {}
        for key, prediction_field in items:
            if points is not None:
                check_point_alignment(key, prediction_field, points, "prediction")
            aux_field = slice_aux(aux, key)
            check_real(key, "prediction", prediction_field)
            check_aux(key, self._score, prediction_field, aux_field)
            threshold = self._threshold_for(key, prediction_field, aux_field, points)
            lo_field, hi_field = self._score.interval(
                prediction_field, threshold, aux_field
            )
            lo_out[key] = lo_field
            hi_out[key] = hi_field

        if isinstance(prediction, TensorDict):
            lo = prediction.empty()
            hi = prediction.empty()
            lo.update(lo_out)
            hi.update(hi_out)
            return lo, hi
        return pack_fields(lo_out), pack_fields(hi_out)

    def save(self, path: Path | str, *, provenance: Mapping | None = None) -> None:
        r"""Atomically write a portable, ``weights_only``-safe artifact.

        Parameters
        ----------
        path : Path | str
            Destination file. Parent directories are created; the file is
            written to a temporary sibling, read back for validation, then
            renamed into place.
        provenance : Mapping, optional
            Strict-JSON metadata to store instead of the predictor's own
            provenance. Default is ``None``.

        Returns
        -------
        None
            The artifact is written to ``path``.
        """
        from .artifacts import _save_predictor  # artifacts depends on this module

        _save_predictor(self, path, provenance=provenance)

    @classmethod
    def load(
        cls, path: Path | str, map_location: str | torch.device = "cpu"
    ) -> "ConformalPredictor":
        r"""Load and semantically validate a fitted predictor artifact.


        Parameters
        ----------
        path : Path | str
            Artifact written by :meth:`save`.
        map_location : str | torch.device, optional
            Device for the loaded thresholds. Default is ``"cpu"``.

        Returns
        -------
        ConformalPredictor
            The reconstructed predictor.
        """
        from .artifacts import _load_predictor  # artifacts depends on this module

        return _load_predictor(path, map_location=map_location)
