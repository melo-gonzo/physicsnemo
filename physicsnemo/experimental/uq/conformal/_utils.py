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

r"""Private helpers shared by the conformal package.

Field-container dispatch (a plain tensor or a ``TensorDict``), exact
conformal rank and order-statistic arithmetic, and the input-contract
validators. Each rule lives in one function so the calibrate and predict
boundaries cannot drift apart.

Threshold dtype policy: the calibrated threshold must never round below the
selecting order statistic, or the finite-sample guarantee is void.
Quantiles are computed in the scores' promoted dtype (no silent down-casts),
and :func:`cast_directed` rounds in a chosen direction whenever precision
must be reduced.
"""

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal, get_args

import torch
from jaxtyping import Float
from tensordict import TensorDict
from torch import Tensor

if TYPE_CHECKING:
    from .scores import _NonconformityScore

TENSOR_KEY = "__tensor__"
"""Reserved field key used when the inputs are plain tensors."""


def field_items(
    x: Tensor | TensorDict,
    keys: Sequence[str] | None = None,
) -> list[tuple[str, Tensor]]:
    """Normalize a tensor or ``TensorDict`` into sorted ``(key, tensor)`` pairs.

    A plain tensor maps to one pair keyed by :data:`TENSOR_KEY`.
    """
    if isinstance(x, Tensor):
        if keys is not None:
            raise TypeError(
                "Named-field access requires a field container: keys="
                f"{list(keys)} were requested but the input is a plain tensor. "
                "Pass a TensorDict keyed by field name, or drop keys."
            )
        return [(TENSOR_KEY, x)]
    if not isinstance(x, TensorDict):
        raise TypeError(
            "Conformal tensor containers must be a torch.Tensor or TensorDict, "
            f"got {type(x).__name__}."
        )
    available = sorted(x.keys())
    if not available:
        raise ValueError("TensorDict conformal inputs must contain at least one field.")
    if TENSOR_KEY in available:
        raise ValueError(
            f"{TENSOR_KEY!r} is reserved for internal plain-tensor bookkeeping "
            "and cannot be used as a field-container key. Rename the field."
        )
    if "_meta" in available:
        raise ValueError(
            "'_meta' is reserved for conformal report metadata and cannot be "
            "used as a field-container key. Rename the field."
        )
    if keys is not None:
        missing = sorted(set(keys) - set(available))
        if missing:
            raise KeyError(
                f"Requested fields {missing} not present; available: {available}."
            )
        available = [k for k in available if k in set(keys)]
    items = [(k, x[k]) for k in available]
    for key, value in items:
        if not isinstance(value, Tensor):
            raise TypeError(
                f"Field {key!r} must contain a torch.Tensor, got "
                f"{type(value).__name__}."
            )
    return items


def pack_fields(
    items: Mapping[str, Tensor],
) -> Tensor | TensorDict:
    """Inverse of :func:`field_items`."""
    if set(items.keys()) == {TENSOR_KEY}:
        return items[TENSOR_KEY]
    return TensorDict(dict(items), batch_size=[])


def slice_aux(
    aux: Mapping[str, Tensor] | Mapping[str, Mapping[str, Tensor]] | None,
    field: str,
) -> Mapping[str, Tensor] | None:
    """Select one field's aux mapping (nested by field for ``TensorDict`` inputs)."""
    if aux is None:
        return None
    if not isinstance(aux, Mapping):
        raise TypeError(f"aux must be a mapping, got {type(aux).__name__}.")
    if field == TENSOR_KEY:
        return aux  # type: ignore[return-value]
    entry = aux.get(field)
    if entry is None and aux and all(isinstance(v, Tensor) for v in aux.values()):
        raise TypeError(
            "aux for TensorDict inputs must be nested by field name, got "
            f"top-level tensor entries {sorted(aux)}; pass "
            f"aux={{'{field}': {{key: tensor}}, ...}}."
        )
    if entry is not None and not isinstance(entry, Mapping):
        raise TypeError(
            f"Field '{field}': aux entry must be a mapping, got {type(entry).__name__}."
        )
    return entry  # type: ignore[return-value]


def validate_alpha(alpha: object) -> float:
    """Return ``alpha`` as a float in ``(0, 1)``, rejecting non-float-exact values."""
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
    """The declared ``alpha`` as an exact rational (the shortest round-trip decimal).

    All rank arithmetic is exact on the typed decimal: no float tolerance both
    keeps ``150 * 0.82 == 123`` and rounds ``alpha=0.49999999995`` up.
    """
    return Fraction(str(validate_alpha(alpha)))


def require_feasible_alpha(n_cal: int, alpha: float) -> None:
    """Require ``alpha >= 1 / (n_cal + 1)``.

    Below it the quantile rank exceeds ``n_cal`` and the CRC bound is
    unsatisfiable even at zero risk; both tiers share this one check.
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
    """Return ``k = ceil((n_cal + 1)(1 - alpha))`` exactly, ``1 <= k <= n_cal``."""
    require_feasible_alpha(n_cal, alpha)
    return math.ceil((n_cal + 1) * (1 - alpha_as_fraction(alpha)))


def kth_smallest_of_samples(
    per_sample: Sequence[Float[Tensor, "*dims"]],
    k: int,
    *,
    chunk_numel: int = 2**26,
) -> Float[Tensor, "*dims"]:
    """``torch.kthvalue(torch.stack(per_sample), k, dim=0)`` in ``chunk_numel`` chunks.

    Uses ``kthvalue`` rather than ``torch.quantile``: exact, and free of the
    ``2**24``-element input limit. Computed in the samples' promoted dtype.
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
    """Cast to ``dtype``, rounding toward ``+inf`` (``up=True``) or ``-inf``.

    Outward rounding keeps a threshold from landing below its order statistic.
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


Tier = Literal["cellwise", "functional", "risk_control"]
"""The one shared spelling of the guarantee-tier vocabulary."""

TIERS: tuple[str, ...] = get_args(Tier)


def _field_label(key: str) -> str:
    """Name a field in user-facing messages without leaking the tensor sentinel."""
    return "Plain tensor" if key == TENSOR_KEY else f"Field '{key}'"


def check_points(points: Tensor) -> Tensor:
    """Require finite floating ``points`` of shape ``(n_points, n_spatial_dims)``."""
    if not isinstance(points, Tensor):
        raise TypeError(f"points must be a torch.Tensor, got {type(points).__name__}.")
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError(
            "points must have non-empty shape (n_points, n_spatial_dims), "
            f"got {tuple(points.shape)}."
        )
    if not points.is_floating_point():
        raise TypeError(f"points must have a floating dtype, got {points.dtype}.")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("points contains non-finite coordinate value(s).")
    return points


def points_fingerprint(
    points: Tensor,
) -> str:
    """Return an order-sensitive exact SHA-256 fingerprint of point coordinates.

    Not memoized: ``.data`` or ``.numpy()`` writes skip the version counter.
    """
    check_points(points)
    tensor = points.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def check_point_alignment(
    key: str, tensor: Tensor, points: Tensor, name: str = "prediction"
) -> Tensor:
    """Require one leading tensor entry per supplied mesh point."""
    if tensor.ndim == 0 or tensor.shape[0] != points.shape[0]:
        raise ValueError(
            f"{_field_label(key)} ({name}): when points= is supplied it must have one "
            f"leading entry per point: got shape {tuple(tensor.shape)} for "
            f"points.shape={tuple(points.shape)}."
        )
    return tensor


def require_single_rank(what: str) -> None:
    """Fail closed when any initialized ``torch.distributed`` group spans ranks."""
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    ):
        raise NotImplementedError(
            f"Conformal {what} supports single-rank execution only. Gather the "
            "exact, deduplicated samples onto one rank first."
        )


def clamp_min_floor(t: Tensor, eps: float) -> Tensor:
    """Clamp ``t`` to at least ``max(eps, finfo(t.dtype).tiny)``.

    ``eps`` alone can underflow to zero in float16; every scale floor shares this.
    """
    if not math.isfinite(eps):
        raise ValueError(f"eps must be finite, got {eps}.")
    floor = eps
    if t.is_floating_point():
        info = torch.finfo(t.dtype)
        if eps > info.max:
            raise ValueError(
                f"eps={eps} exceeds the largest finite {t.dtype} value "
                f"({info.max}); use a smaller eps or a wider dtype."
            )
        floor = max(eps, float(info.tiny))
    return t.clamp_min(floor)


def broadcast_difficulty(t: Tensor, ref: Tensor, key: str) -> Tensor:
    """Right-pad per-point difficulty with singleton dims to match ``ref``."""
    if t.ndim == 0:
        return t
    if ref.shape[0] != t.shape[0]:
        raise ValueError(
            f"{_field_label(key)}: difficulty has {t.shape[0]} points but the "
            f"leading dimension is {ref.shape[0]}; per-point difficulty "
            "must align with the leading (point) dimension."
        )
    return t.reshape(t.shape[0], *([1] * (ref.ndim - 1)))


def normalize_keys(keys: Sequence[str] | None) -> tuple[str, ...] | None:
    """Normalize ``keys`` to a deduplicated tuple or ``None``; reject a bare string."""
    if keys is None:
        return None
    if isinstance(keys, str):
        raise TypeError(
            f"keys must be a sequence of field names, not a bare string "
            f"{keys!r} (which would iterate into single characters). Pass "
            f"[{keys!r}]."
        )
    return tuple(dict.fromkeys(keys))


def positive_finite_float(value: object, name: str) -> float:
    """Coerce a strategy scalar (``eps``, ``value``) to a positive finite float."""
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a positive finite value, got {value!r}."
        ) from exc
    if not (value > 0 and math.isfinite(value)):
        raise ValueError(f"{name} must be a positive finite value, got {value}.")
    return value


def require_matching_keys(
    present: Iterable[str], expected: Iterable[str], subject: str
) -> None:
    """Require two field-key sets to match exactly."""
    present = set(present)
    expected = set(expected)
    if present != expected:
        differing = sorted(present.symmetric_difference(expected))
        raise KeyError(f"{subject}; differing keys: {differing}.")


def check_exact_shape(
    key: str, name: str, tensor: Tensor, reference_name: str, reference: Tensor
) -> Tensor:
    """Require ``tensor.shape == reference.shape`` exactly."""
    if tensor.shape != reference.shape:
        raise ValueError(
            f"{_field_label(key)}: {name} shape {tuple(tensor.shape)} != "
            f"{reference_name} shape {tuple(reference.shape)}. Shapes must "
            "match exactly (silent broadcasting would produce a different "
            "statistic than the calibrated one)."
        )
    return tensor


def check_aux(
    key: str,
    score: "_NonconformityScore",
    prediction: Tensor,
    aux: Mapping[str, Tensor] | None,
) -> None:
    """Require the aux entries a score reads to be finite and prediction-shaped.

    Shared by calibrate and predict: ``+inf`` sigma would otherwise zero a score.
    """
    if aux is None:
        return
    present = [aux_key for aux_key in score.aux_keys if aux_key in aux]
    for aux_key in present:
        if not isinstance(aux[aux_key], Tensor):
            raise TypeError(
                f"{_field_label(key)}: aux '{aux_key}' must be a torch.Tensor, got "
                f"{type(aux[aux_key]).__name__}."
            )
        check_exact_shape(
            key, f"aux '{aux_key}'", aux[aux_key], "prediction", prediction
        )
    for aux_key in present:
        check_real(key, f"aux '{aux_key}'", aux[aux_key])


def check_floating(key: str, name: str, tensor: Tensor) -> Tensor:
    """Require a floating dtype: integer data would truncate interval endpoints."""
    if not tensor.is_floating_point():
        raise TypeError(
            f"{_field_label(key)}: {name} must use a floating-point dtype, got "
            f"{tensor.dtype}. Integer/bool conformal data would truncate "
            "scores or interval endpoints."
        )
    return tensor


def check_finite(key: str, name: str, tensor: Tensor) -> Tensor:
    """Reject NaN/inf values with an actionable per-field message."""
    if not torch.isfinite(tensor).all():
        n_bad = int((~torch.isfinite(tensor)).sum())
        raise ValueError(
            f"{_field_label(key)}: {n_bad} non-finite value(s) (NaN/inf) in {name}. "
            "Non-finite inputs would silently corrupt the calibrated "
            "threshold or the reported statistic; clean or mask them first."
        )
    return tensor


def check_real(key: str, name: str, tensor: Tensor) -> Tensor:
    """Require finite floating-point data."""
    return check_finite(key, name, check_floating(key, name, tensor))


def _strict_json_snapshot(value: Any, name: str) -> Any:
    """Validate and detach a strict-JSON value without implicit coercion."""
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain only finite JSON numbers.")
        return value
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(
                    f"{name} keys must be strings, got {type(key).__name__} ({key!r})."
                )
            out[key] = _strict_json_snapshot(item, f"{name}.{key}")
        return out
    if type(value) is list:
        return [
            _strict_json_snapshot(item, f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{name} must contain only strict-JSON values; got {type(value).__name__}."
    )


def validate_provenance(value: object) -> dict:
    """Snapshot a predictor's strict-JSON provenance mapping."""
    if not isinstance(value, Mapping):
        raise TypeError(f"provenance must be a mapping, got {type(value).__name__}.")
    snapshot = _strict_json_snapshot(value, "provenance")
    if "mesh_fingerprint" in snapshot:
        raise ValueError(
            "provenance must not contain 'mesh_fingerprint'; it is fitted "
            "predictor state."
        )
    return snapshot
