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

r"""Single-rank, tier-aligned empirical diagnostics for conformal intervals."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from jaxtyping import Float
from tensordict import TensorDict
from torch import Tensor

from ._utils import (
    TENSOR_KEY,
    TIERS,
    Tier,
    _field_label,
    alpha_as_fraction,
    check_exact_shape,
    check_real,
    field_items,
    normalize_keys,
    pack_fields,
    require_single_rank,
    validate_alpha,
    validate_n_cal,
)

__all__ = ["CoverageAccumulator"]


@dataclass
class _FieldCounters:
    """Minimal streaming state for one field."""

    coverage_sum: float = 0.0
    width_sum: float = 0.0
    width_count: int = 0
    n_samples: int = 0


def _minimum_hits_at_target(n_samples: int, alpha: float) -> int:
    """Exact count required for empirical coverage of at least ``1 - alpha``."""
    return math.ceil(n_samples * (1 - alpha_as_fraction(alpha)))


class CoverageAccumulator:
    r"""Accumulate empirical evaluation statistics aligned to a predictor tier.

    Create instances through ``coverage_accumulator()`` on a fitted
    :class:`~physicsnemo.experimental.uq.conformal.ConformalPredictor`,
    which fixes the aggregation to the predictor's guarantee tier so a report
    cannot silently describe a different empirical event than the fitted
    guarantee. Feed held-out ``(lo, hi, target)`` triples through
    :meth:`update` and read the report from :meth:`finalize`. The empirical
    event per tier is: element covered (cellwise, per-element marginal
    coverage), whole field covered simultaneously (functional), and fraction
    of miscovered points per sample (conformal risk control (CRC)).

    Parameters
    ----------
    tier : {"cellwise", "functional", "risk_control"}
        Guarantee tier whose empirical event is accumulated.
    alpha : float
        Target miscoverage (or risk) level in :math:`(0, 1)`, reported as
        metadata and used for the cellwise at-target fraction.
    n_cal : int
        Number of calibration samples of the predictor, reported as metadata.
    keys : Sequence[str], optional
        Calibrated field names for ``TensorDict`` inputs; ``None`` selects
        plain-tensor mode. Default is ``None``.

    Notes
    -----
    Diagnostics are single-rank: :meth:`finalize` and
    :attr:`empirical_coverage_map` raise ``NotImplementedError`` when an
    initialized ``torch.distributed`` group spans several ranks. Gather
    evaluation samples onto one rank first.

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
    >>> accumulator = predictor.coverage_accumulator()
    >>> for _ in range(5):
    ...     prediction = torch.randn(50, 2)
    ...     lo, hi = predictor.predict_interval(prediction)
    ...     accumulator.update(lo, hi, prediction + 0.1 * torch.randn(50, 2))
    >>> report = accumulator.finalize()
    >>> sorted(report)
    ['_meta', 'value']
    >>> report["value"]["n_samples"]
    5
    """

    def __init__(
        self,
        *,
        tier: Tier,
        alpha: float,
        n_cal: int,
        keys: Sequence[str] | None = None,
    ) -> None:
        if tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}, got {tier!r}.")
        self._tier = tier
        self._alpha = validate_alpha(alpha)
        self._n_cal = validate_n_cal(n_cal)
        self._keys = normalize_keys(keys)
        self._tensor_mode = keys is None
        field_keys = (TENSOR_KEY,) if self._tensor_mode else self._keys
        self._counters = {key: _FieldCounters() for key in field_keys}
        self._element_hits: dict[str, Tensor] = {}

    def update(
        self,
        lo: Float[Tensor, "*dims"] | TensorDict,
        hi: Float[Tensor, "*dims"] | TensorDict,
        target: Float[Tensor, "*dims"] | TensorDict,
    ) -> None:
        r"""Validate and accumulate one interval/target sample transactionally.

        Field containers may carry a superset of the fitted fields (e.g. the
        model's natural full output as ``target``): the fitted keys are
        selected automatically, mirroring ``predict_interval`` on
        :class:`~physicsnemo.experimental.uq.conformal.ConformalPredictor`.

        Parameters
        ----------
        lo : torch.Tensor | TensorDict
            Lower bounds of shape :math:`(*\text{dims})`, or a field container
            of such tensors.
        hi : torch.Tensor | TensorDict
            Upper bounds, same shape and container type as ``lo``.
        target : torch.Tensor | TensorDict
            Observed values, same shape and container type as ``lo``.

        Returns
        -------
        None
            The sample's statistics are committed to the accumulator.

        Notes
        -----
        Nothing is committed unless every field validates, so a rejected
        sample leaves the accumulator unchanged. Raises ``ValueError`` on a
        shape mismatch, non-finite value, width or width-sum overflow, or
        (cellwise tier) a sample shape that differs from earlier updates,
        and ``TypeError`` when the container type does not match the fitted
        mode.

        Diagnostics require finite bounds and targets. Conservative prediction
        bounds may become infinite when they overflow the prediction's dtype.
        Upcast the prediction (e.g. to float32) before calling
        ``predict_interval`` and recompute the bounds; upcasting infinite
        bounds afterward does not repair them. If widths or their accumulated
        sum overflow float64, rescale interval bounds and targets consistently
        and restart diagnostics.
        """
        # field_items(..., self._keys) yields exactly the keys of self._counters.
        inputs = (("lo", lo), ("hi", hi), ("target", target))
        containers = {
            name: dict(field_items(value, self._keys)) for name, value in inputs
        }
        for name, value in inputs:
            if isinstance(value, Tensor) != self._tensor_mode:
                expected = "plain tensors" if self._tensor_mode else "TensorDicts"
                raise TypeError(f"This accumulator expects {expected}; got {name}.")

        staged: list[tuple[str, float | Tensor, float, int]] = []
        for key in self._counters:
            lo_field = containers["lo"][key]
            hi_field = containers["hi"][key]
            target_field = containers["target"][key]
            if target_field.numel() == 0:
                raise ValueError(f"{_field_label(key)}: empty target tensor.")
            check_exact_shape(key, "lo", lo_field, "target", target_field)
            check_exact_shape(key, "hi", hi_field, "target", target_field)
            check_real(key, "lo", lo_field)
            check_real(key, "hi", hi_field)
            check_real(key, "target", target_field)

            element_covered = (target_field >= lo_field) & (target_field <= hi_field)
            widths = hi_field.to(torch.float64) - lo_field.to(torch.float64)
            if not bool(torch.isfinite(widths).all()):
                raise ValueError(
                    f"{_field_label(key)}: interval width overflows even though both "
                    "endpoints are finite. Rescale interval bounds and targets "
                    "consistently, and restart diagnostics."
                )
            # Quantile regression may produce an empty prediction set when a
            # valid negative calibrated threshold contracts a narrower base band.
            widths = widths.clamp_min(0.0)
            # Nonnegative widths make this catch both sample and total overflow.
            width_total = self._counters[key].width_sum + float(widths.sum())
            if not math.isfinite(width_total):
                raise ValueError(
                    f"{_field_label(key)}: interval width sum overflows float64 within "
                    "this sample or across updates. Rescale interval bounds and "
                    "targets consistently, and restart diagnostics."
                )

            match self._tier:
                case "cellwise":
                    # Hit counts stay on device until finalize().
                    coverage = element_covered.to(torch.int64)
                    previous = self._element_hits.get(key)
                    if previous is not None and previous.shape != coverage.shape:
                        raise ValueError(
                            f"{_field_label(key)}: elementwise coverage requires a fixed "
                            "sample shape across updates."
                        )
                case "functional":
                    coverage = float(element_covered.all())
                case "risk_control":
                    points = torch.atleast_1d(element_covered)
                    point_covered = points.reshape(points.shape[0], -1).all(dim=1)
                    coverage = float(point_covered.to(torch.float64).mean())

            staged.append((key, coverage, width_total, widths.numel()))

        for key, coverage, width_total, width_count in staged:
            counters = self._counters[key]
            counters.width_sum = width_total
            counters.width_count += width_count
            counters.n_samples += 1
            if isinstance(coverage, float):
                counters.coverage_sum += coverage
            elif key in self._element_hits:
                previous = self._element_hits[key]
                previous += coverage.to(device=previous.device)
            else:
                self._element_hits[key] = coverage

    @property
    def empirical_coverage_map(self) -> Float[Tensor, "*dims"] | TensorDict:
        r"""Per-element empirical coverage for a cellwise predictor.

        Returns
        -------
        torch.Tensor | TensorDict
            Fraction of accumulated samples in which each element was
            covered, of the calibrated shape :math:`(*\text{dims})` per field,
            in float64. Raises ``RuntimeError`` for non-cellwise tiers or
            before the first update.
        """
        require_single_rank("diagnostics")
        if self._tier != "cellwise":
            raise RuntimeError(
                "empirical_coverage_map is available only for cellwise predictors."
            )
        if not self._element_hits:
            raise RuntimeError("No diagnostic samples collected.")
        return pack_fields(
            {
                key: hits.to(torch.float64) / self._counters[key].n_samples
                for key, hits in self._element_hits.items()
            }
        )

    def finalize(self) -> dict:
        r"""Return a strict-JSON tier-aligned empirical diagnostic report.

        Returns
        -------
        dict
            ``{"_meta": {...}, <field>: {...}}`` where ``_meta`` records the
            tier, ``alpha``, ``n_cal``, and the target coverage or risk, and
            each field entry (``"value"`` in plain-tensor mode) records
            ``n_samples``, the element-weighted mean interval width, and the
            tier's empirical statistic: per-element coverage summaries
            (cellwise), ``whole_field_coverage`` (functional), or
            ``empirical_mean_risk`` (risk control). Statistics are ``None``
            before the first update.
        """
        require_single_rank("diagnostics")
        metadata = {
            "tier": self._tier,
            "alpha": self._alpha,
            "n_cal": self._n_cal,
        }
        if self._tier == "risk_control":
            metadata["target_risk"] = self._alpha
        else:
            metadata["target_coverage"] = 1.0 - self._alpha
        report: dict = {"_meta": metadata}
        for key, counters in self._counters.items():
            n = counters.n_samples
            entry: dict = {
                "n_samples": n,
                "element_weighted_mean_interval_width": (
                    counters.width_sum / counters.width_count
                    if counters.width_count
                    else None
                ),
            }
            match self._tier:
                case "cellwise" if n:
                    hits = self._element_hits[key]
                    at_target = hits >= _minimum_hits_at_target(n, self._alpha)
                    entry.update(
                        mean_element_coverage=float(hits.to(torch.float64).mean()) / n,
                        minimum_element_coverage=float(hits.min()) / n,
                        fraction_elements_at_or_above_target=float(
                            at_target.to(torch.float64).mean()
                        ),
                    )
                case "cellwise":
                    entry.update(
                        mean_element_coverage=None,
                        minimum_element_coverage=None,
                        fraction_elements_at_or_above_target=None,
                    )
                case "functional":
                    entry["whole_field_coverage"] = (
                        counters.coverage_sum / n if n else None
                    )
                case "risk_control":
                    entry["empirical_mean_risk"] = (
                        1.0 - counters.coverage_sum / n if n else None
                    )
            report["value" if key == TENSOR_KEY else key] = entry
        return report
