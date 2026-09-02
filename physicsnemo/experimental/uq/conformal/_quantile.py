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

r"""Private exact-rank, order-statistic, and directed-rounding utilities.

The conformal quantile at miscoverage level :math:`\alpha` is the empirical
quantile of the :math:`n` calibration scores at level
:math:`\lceil (n + 1)(1 - \alpha) \rceil / n` with "higher" interpolation,
which is exactly the :math:`k`-th smallest score with
:math:`k = \lceil (n + 1)(1 - \alpha) \rceil`. All quantiles in this package
are therefore computed with :func:`torch.kthvalue` rather than
:func:`torch.quantile`: this is both exact (no interpolation ambiguity) and
avoids the ``2**24``-element input limit of ``torch.quantile``, which
elementwise calibration stacks routinely exceed.

Rank and risk arithmetic is exact rational arithmetic on the declared
``alpha`` (:func:`alpha_as_fraction`), never float arithmetic.

Threshold dtype policy: the calibrated threshold must never round below the
selecting order statistic, or the finite-sample guarantee is void. Quantiles
are computed in the scores' promoted dtype (no silent down-casts), and
:func:`cast_directed` rounds in a chosen direction whenever precision must
be reduced.
"""

import math
from collections.abc import Sequence
from fractions import Fraction
from numbers import Real

import torch
from jaxtyping import Float
from torch import Tensor

__all__ = [
    "alpha_as_fraction",
    "cast_directed",
    "conformal_quantile_index",
    "kth_smallest_of_samples",
    "require_feasible_alpha",
    "validate_alpha",
    "validate_n_cal",
]


def validate_alpha(alpha: object) -> float:
    """Return a finite float ``alpha`` strictly between zero and one.

    Only values exactly representable as a Python float are accepted: all
    rank and risk arithmetic in this package is exact on the declared value
    (see :func:`alpha_as_fraction`), so silently flooring an exact rational
    (``Fraction(1, 3)``) to float would compute a different rank than the
    caller asked for, and could mask an infeasible level as feasible.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise TypeError(f"alpha must be a real number, got {alpha!r}.")
    as_float = float(alpha)
    if not isinstance(alpha, float) and alpha != as_float:
        raise TypeError(
            f"alpha={alpha!r} is not exactly representable as a float; the "
            "conformal rank arithmetic is exact on the declared value, so a "
            "silent float conversion would change the requested level. Pass "
            "a float."
        )
    if not math.isfinite(as_float) or not 0.0 < as_float < 1.0:
        raise ValueError(f"alpha must be a finite value in (0, 1), got {as_float}.")
    return as_float


def validate_n_cal(n_cal: object) -> int:
    """Return ``n_cal`` as a positive ``int`` (``bool`` is not an int here)."""
    if isinstance(n_cal, bool) or not isinstance(n_cal, int):
        raise TypeError(f"n_cal must be an integer, got {n_cal!r}.")
    if n_cal < 1:
        raise ValueError(f"n_cal must be >= 1, got {n_cal}.")
    return n_cal


def alpha_as_fraction(alpha: float) -> Fraction:
    """The user's declared ``alpha`` as an exact rational.

    ``str(alpha)`` recovers the shortest decimal that round-trips to the
    given float, i.e. the number the user actually typed (``0.18``, not
    ``0.17999999999999999...``). That typed decimal is the ``alpha``
    contract for the whole package (:func:`validate_alpha` rejects values a
    float cannot represent exactly, so nothing is silently floored first).
    All rank/parameter arithmetic goes through exact rational arithmetic
    on that declared value: a floating tolerance can neither be small
    enough to respect a genuinely requested offset nor large enough to
    absorb representation error (``alpha=0.49999999995`` must round the
    rank up, while ``150 * 0.82`` must stay exactly ``123``).
    """
    return Fraction(str(validate_alpha(alpha)))


def require_feasible_alpha(n_cal: int, alpha: float) -> None:
    r"""Require ``alpha >= 1 / (n_cal + 1)``, the shared finite-sample floor.

    Below the floor the conformal quantile rank exceeds ``n_cal``
    (:math:`\lceil (n + 1)(1 - \alpha) \rceil > n \iff \alpha < 1/(n + 1)`,
    so the prediction set would be infinite) and the conformal risk control
    (CRC) bound :math:`(\hat R(\lambda) + 1)/(n + 1) \le \alpha` is
    unsatisfiable even at zero risk. Both are the same algebraic constraint,
    so both boundaries share this one check.

    Notes
    -----
    Raises ``ValueError`` if ``n_cal < 1`` or ``alpha`` is infeasible for
    ``n_cal``.
    """
    alpha_exact = alpha_as_fraction(alpha)
    n_cal = validate_n_cal(n_cal)
    if alpha_exact < Fraction(1, n_cal + 1):
        min_n = math.ceil((1 - alpha_exact) / alpha_exact)
        raise ValueError(
            f"Insufficient calibration samples for alpha={alpha}: the "
            f"conformal guarantee requires alpha >= 1/(n_cal + 1), i.e. "
            f"n_cal >= {min_n} (got {n_cal}); this level is infeasible with "
            "the collected samples. Collect more calibration data or "
            "increase alpha."
        )


def conformal_quantile_index(n_cal: int, alpha: float) -> int:
    r"""Return the order-statistic index of the conformal quantile.

    Parameters
    ----------
    n_cal : int
        Number of calibration samples.
    alpha : float
        Target miscoverage level in :math:`(0, 1)`.

    Returns
    -------
    int
        :math:`k = \lceil (n_{cal} + 1)(1 - \alpha) \rceil` computed with
        exact rational arithmetic on the declared ``alpha``, such that the
        :math:`k`-th smallest calibration score is the conformal quantile.
        Always satisfies :math:`1 \le k \le n_{cal}`.

    Notes
    -----
    Raises ``ValueError`` if ``alpha`` is outside :math:`(0, 1)`,
    ``n_cal < 1``, or ``n_cal`` is too small for the requested level (see
    :func:`require_feasible_alpha`).
    """
    require_feasible_alpha(n_cal, alpha)
    # 1 <= k <= n_cal holds whenever alpha is feasible (checked above).
    return math.ceil((n_cal + 1) * (1 - alpha_as_fraction(alpha)))


def kth_smallest_of_samples(
    per_sample: Sequence[Float[Tensor, "*dims"]],
    k: int,
    *,
    chunk_numel: int = 2**26,
) -> Float[Tensor, "*dims"]:
    r"""``k``-th smallest across per-sample tensors, without stacking the corpus.

    Equivalent to ``torch.kthvalue(torch.stack(per_sample), k, dim=0)``, but
    only one :math:`(n_{cal}, \text{cells\_per\_chunk})` block is ever
    materialized beside the retained per-sample list, so peak memory at
    large field sizes stays ``list + one block`` instead of ``2 x list``.

    Computed in the promoted common dtype of the samples (exactly what a
    full ``torch.stack`` would produce), never a down-cast of any sample's
    order statistic (see the module docstring's threshold dtype policy).

    Parameters
    ----------
    per_sample : Sequence[torch.Tensor]
        One calibration-score tensor per sample, all of one shape
        :math:`(*\text{dims})` (``torch.stack`` rejects mismatched shapes).
    k : int
        Order-statistic index in ``[1, len(per_sample)]`` (as produced by
        :func:`conformal_quantile_index`; ``torch.kthvalue`` rejects others).
    chunk_numel : int, optional
        Upper bound on the number of elements stacked per
        :func:`torch.kthvalue` call, to bound peak memory. Default is
        ``2**26``.

    Returns
    -------
    torch.Tensor
        The per-element ``k``-th smallest scores, of shape
        :math:`(*\text{dims})`.
    """
    n = len(per_sample)
    first = per_sample[0]
    cell_shape = first.shape
    dtype = first.dtype
    for t in per_sample[1:]:
        dtype = torch.promote_types(dtype, t.dtype)
    flats = [t.reshape(-1) for t in per_sample]
    n_cells = flats[0].numel()
    cells_per_chunk = max(1, chunk_numel // max(n, 1))
    out = torch.empty(n_cells, dtype=dtype, device=first.device)
    for start in range(0, n_cells, cells_per_chunk):
        stop = min(start + cells_per_chunk, n_cells)
        block = torch.stack([flat[start:stop] for flat in flats], dim=0)
        out[start:stop] = torch.kthvalue(block, k, dim=0).values
    return out.reshape(cell_shape)


def cast_directed(t: Tensor, dtype: torch.dtype, *, up: bool) -> Tensor:
    """Cast to ``dtype`` with directed rounding.

    Round-to-nearest can land on either side of the true value; this bumps
    every element that landed on the wrong side to the next representable
    value toward ``+inf`` (``up=True``) or ``-inf`` (``up=False``). The
    wrong-side test compares in float64, which represents every supported
    source dtype exactly. Total bit width is not a precision order (bfloat16
    and float16 are both 16 bits with different mantissas), so no dtype-pair
    shortcut is taken.
    """
    cast = t.to(dtype)
    if dtype == t.dtype or not (t.is_floating_point() and cast.is_floating_point()):
        return cast
    exact = t.to(torch.float64)
    got = cast.to(torch.float64)
    wrong_side = got < exact if up else got > exact
    toward = torch.inf if up else -torch.inf
    bumped = torch.nextafter(cast, torch.full_like(cast, toward))
    return torch.where(wrong_side, bumped, cast)
