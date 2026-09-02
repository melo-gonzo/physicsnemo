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

r"""Shared input-contract validators for the conformal package.

Calibration and interval construction must enforce the same contract:
exact shapes (silent broadcasting computes a different statistic than the
guaranteed one), finite values, strictly positive difficulty. Keeping the
checks in one module guarantees the boundaries cannot drift apart.
"""

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, get_args

import torch
from torch import Tensor

from ._containers import TENSOR_KEY

if TYPE_CHECKING:
    from .scores import _NonconformityScore

__all__ = [
    "TIERS",
    "Tier",
    "broadcast_difficulty",
    "clamp_min_floor",
    "multi_rank_active",
    "normalize_keys",
    "positive_finite_float",
    "require_matching_keys",
    "validate_provenance",
    "check_aux",
    "check_difficulty",
    "check_exact_shape",
    "check_finite",
    "check_floating",
    "check_point_alignment",
    "check_points",
    "check_real",
    "points_fingerprint",
]

Tier = Literal["cellwise", "functional", "risk_control"]
"""The one shared spelling of the guarantee-tier vocabulary."""

TIERS: tuple[str, ...] = get_args(Tier)


def _field_label(key: str) -> str:
    """Name a field in user-facing messages without leaking the tensor sentinel."""
    return "Plain tensor" if key == TENSOR_KEY else f"Field '{key}'"


def check_points(points: Tensor) -> Tensor:
    """Validate the coordinate-tensor contract: dense, 2-D, floating, finite.

    The contract is one row per mesh point, ``(n_points, n_spatial_dims)``;
    every boundary that accepts ``points=`` must enforce it so a
    contradictory coordinate tensor cannot be silently accepted.
    """
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
    """Return an order-sensitive exact fingerprint of point coordinates.

    Deliberately recomputed on every call: memoizing per tensor object via
    the autograd version counter is unsound for an exactness gate, because
    in-place writes through ``.data`` or ``.numpy()`` aliasing (e.g. a
    reused staging buffer refilled by a data reader) never bump the version
    counter and would let a mutated mesh pass with a stale digest. A trusted
    fast path must be an explicit opt-in API, not a silent cache.
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


def multi_rank_active() -> bool:
    """True when an initialized ``torch.distributed`` group spans ranks.

    Probes ``torch.distributed`` directly rather than
    :class:`~physicsnemo.distributed.DistributedManager`: the manager is a
    singleton the caller must initialize first, while this guard has to be
    correct for any process group, including ones set up outside physicsnemo.
    """
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    )


def clamp_min_floor(t: Tensor, eps: float) -> Tensor:
    """Clamp ``t`` with a dtype-aware positive floor.

    ``eps`` alone can underflow to zero in low-precision dtypes (``1e-8``
    is unrepresentable in float16), turning the clamp into a no-op and the
    subsequent division into inf/NaN. The effective floor is the larger
    of ``eps`` and the dtype's smallest positive normal value; every
    boundary that floors a scale (score, interval, difficulty) must share
    this exact function so calibration and prediction normalize
    identically.
    """
    floor = eps
    if t.is_floating_point():
        floor = max(eps, float(torch.finfo(t.dtype).tiny))
    return t.clamp_min(floor)


def broadcast_difficulty(t: Tensor, ref: Tensor, key: str) -> Tensor:
    """Right-pad per-point difficulty with singleton dims to match ``ref``."""
    if t.ndim == 0:
        return t
    if ref.shape[0] != t.shape[0]:
        raise ValueError(
            f"Field '{key}': difficulty has {t.shape[0]} points but the "
            f"leading dimension is {ref.shape[0]}; per-point difficulty "
            "must align with the leading (point) dimension."
        )
    return t.reshape(t.shape[0], *([1] * (ref.ndim - 1)))


def normalize_keys(keys: Sequence[str] | None) -> tuple[str, ...] | None:
    """Normalize a ``keys`` argument to a deduplicated tuple or ``None``.

    Rejects a bare string (``"pressure"`` would iterate to a character
    list); unknown or non-string names surface from ``field_items`` as
    missing fields.
    """
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
    """Require two field-key sets to match exactly.

    Used where a container's keys are not already restricted to the
    reference set (calibration prediction/target, and a plain-tensor
    predictor handed a ``TensorDict``), so both boundaries report the
    identical sorted symmetric difference.
    """
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
    """Require the aux entries a score reads to match the prediction shape.

    Entries must also be finite. One function enforces the identical aux
    contract at the calibrate and predict boundaries so they cannot drift
    apart: a ``NormalizedErrorScore`` with ``+inf`` sigma would otherwise
    divide a finite residual to a zero score that passes the
    score-finiteness check and drags the threshold toward zero, yet the
    identical aux is rejected at predict time. Two passes (all shapes first,
    then finiteness) preserve the error precedence across keys.
    """
    if aux is None:
        return
    present = [aux_key for aux_key in score.aux_keys if aux_key in aux]
    for aux_key in present:
        if not isinstance(aux[aux_key], Tensor):
            raise TypeError(
                f"Field '{key}': aux '{aux_key}' must be a torch.Tensor, got "
                f"{type(aux[aux_key]).__name__}."
            )
        check_exact_shape(
            key, f"aux '{aux_key}'", aux[aux_key], "prediction", prediction
        )
    for aux_key in present:
        check_real(key, f"aux '{aux_key}'", aux[aux_key])


def check_floating(key: str, name: str, tensor: Tensor) -> Tensor:
    """Require real floating-point data at a conformal API boundary.

    Integer predictions make score inverses cast fractional endpoints back
    to integers, silently collapsing an otherwise valid interval. Targets
    and score aux follow the same contract so calibration and prediction
    cannot use different arithmetic domains.
    """
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
    """Require finite floating-point data.

    Runs :func:`check_floating` then :func:`check_finite`.
    """
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


def check_difficulty(s: Tensor) -> Tensor:
    """Difficulty values must be finite and strictly positive."""
    if not s.is_floating_point():
        raise TypeError(
            "Difficulty values must use a floating-point dtype; got "
            f"{s.dtype}. Integer difficulty would truncate interval scaling."
        )
    if not torch.isfinite(s).all() or not (s > 0).all():
        raise ValueError(
            "Difficulty values must be finite and strictly positive; got "
            f"min={float(s.min()) if s.numel() else 'empty'}, "
            f"finite={bool(torch.isfinite(s).all())}."
        )
    return s
