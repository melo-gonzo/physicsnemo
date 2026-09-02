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

r"""Portable, ``weights_only``-safe artifacts for fitted conformal predictors."""

import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

import torch
from tensordict import TensorDict
from torch import Tensor

from ._containers import TENSOR_KEY, pack_fields
from ._validation import validate_provenance
from .difficulty import _DIFFICULTY_REGISTRY, _difficulty_kind
from .predictors import ConformalPredictor
from .scores import _SCORE_REGISTRY, _score_kind

__all__: list[str] = []

_ARTIFACT_FORMAT = "physicsnemo.uq.conformal"
_ARTIFACT_VERSION = 1
_SCHEMA_KEYS = {
    "format",
    "version",
    "tier",
    "score",
    "alpha",
    "n_cal",
    "thresholds",
    "difficulty",
    "mesh_fingerprint",
    "provenance",
}
# The single authoritative kwarg table per strategy kind: the encoder reads
# these attributes off the live strategy, and the decoder type-checks the
# same names: one list, so the two directions cannot drift apart.
_SCORE_KWARG_TYPES = {
    "absolute_error": {},
    "normalized_error": {"eps": float},
    "quantile_regression": {},
}
_DIFFICULTY_KWARG_TYPES = {
    "aux": {"key": str, "eps": float},
}


def _check_exact_keys(value: object, expected: set[str], where: str) -> Mapping:
    """Require one exact mapping schema."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a mapping, got {type(value).__name__}.")
    keys = set(value)
    missing = expected - keys
    unexpected = keys - expected
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)!r}")
        if unexpected:
            details.append(f"unexpected {sorted(map(repr, unexpected))!r}")
        raise ValueError(f"{where} schema is invalid: {', '.join(details)}.")
    return value


def _strategy_spec(
    strategy: object, kind_of: Callable[[object], str | None], kwarg_types: Mapping
) -> dict:
    """Encode one exact built-in strategy from the shared kwarg table."""
    kind = kind_of(strategy)
    return {
        "kind": kind,
        "kwargs": {name: getattr(strategy, name) for name in kwarg_types[kind]},
    }


def _resolve_strategy(
    spec: object,
    registry: Mapping,
    kwarg_types: Mapping[str, Mapping[str, type]],
    what: str,
):
    """Reconstruct one exact built-in strategy spec."""
    spec = _check_exact_keys(spec, {"kind", "kwargs"}, f"Artifact {what} spec")
    kind = spec["kind"]
    if type(kind) is not str or kind not in registry:
        raise ValueError(
            f"Unknown built-in {what} kind {kind!r}; expected one of "
            f"{sorted(registry)!r}."
        )
    kwargs = spec["kwargs"]
    if not isinstance(kwargs, Mapping):
        raise ValueError(
            f"Artifact {what} kwargs must be a mapping, got {type(kwargs).__name__}."
        )
    expected_types = kwarg_types[kind]
    kwargs = _check_exact_keys(
        kwargs, set(expected_types), f"Artifact {what} {kind!r} kwargs"
    )
    for name, expected_type in expected_types.items():
        value = kwargs[name]
        if type(value) is not expected_type:
            raise TypeError(
                f"Artifact {what} {kind!r} kwarg {name!r} must have exact type "
                f"{expected_type.__name__}, got {type(value).__name__}."
            )
    try:
        return registry[kind](**dict(kwargs))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid constructor arguments for built-in {what} {kind!r}: {exc}"
        ) from exc


def _wire_thresholds(value: object) -> Tensor | TensorDict:
    """Validate the tensor-only threshold mapping stored in an artifact."""
    if not isinstance(value, Mapping) or not value:
        raise ValueError("Artifact thresholds must be a non-empty mapping.")
    if TENSOR_KEY in value and len(value) != 1:
        raise ValueError(
            f"Artifact thresholds may use reserved key {TENSOR_KEY!r} only as "
            "the sole key."
        )
    for key, threshold in value.items():
        if not isinstance(threshold, Tensor):
            raise TypeError(
                f"Artifact threshold {key!r} must be a torch.Tensor, got "
                f"{type(threshold).__name__}."
            )
    return pack_fields(dict(value))


def _parse_artifact(payload: object) -> ConformalPredictor:
    """Validate the one supported conformal artifact schema."""
    marker = payload.get("format") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or type(marker) is not str
        or marker != _ARTIFACT_FORMAT
    ):
        marker = (
            payload.get("format") if isinstance(payload, Mapping) else type(payload)
        )
        raise ValueError(
            "Not a conformal predictor artifact (format marker missing or "
            f"unknown: {marker!r})."
        )
    version = payload.get("version")
    if type(version) is not int or version != _ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported conformal artifact version {version!r}. Re-run calibration "
            "with this build."
        )
    payload = _check_exact_keys(payload, _SCHEMA_KEYS, "Conformal artifact")
    score = _resolve_strategy(
        payload["score"], _SCORE_REGISTRY, _SCORE_KWARG_TYPES, "score"
    )
    difficulty = (
        None
        if payload["difficulty"] is None
        else _resolve_strategy(
            payload["difficulty"],
            _DIFFICULTY_REGISTRY,
            _DIFFICULTY_KWARG_TYPES,
            "difficulty",
        )
    )
    return ConformalPredictor(
        tier=payload["tier"],
        score=score,
        alpha=payload["alpha"],
        n_cal=payload["n_cal"],
        thresholds=_wire_thresholds(payload["thresholds"]),
        difficulty=difficulty,
        mesh_fingerprint=payload["mesh_fingerprint"],
        provenance=payload["provenance"],
    )


def _artifact_payload(
    predictor: ConformalPredictor, provenance: Mapping | None
) -> dict:
    """Encode public predictor state.

    The threshold entries are serialized as CPU views of the predictor's own
    (already validated, standalone) storage; ``torch.save`` only reads them,
    so no second full-size copy is materialized here. Semantic validation
    happens once, on the read-back in :func:`_save_predictor`.
    """
    chosen_provenance = predictor.provenance if provenance is None else provenance
    difficulty = predictor.difficulty
    return {
        "format": _ARTIFACT_FORMAT,
        "version": _ARTIFACT_VERSION,
        "tier": predictor.tier,
        "score": _strategy_spec(predictor.score, _score_kind, _SCORE_KWARG_TYPES),
        "alpha": predictor.alpha,
        "n_cal": predictor.n_cal,
        "thresholds": {
            key: value.detach().cpu()
            for key, value in predictor._thresholds_by_key.items()
        },
        "difficulty": (
            None
            if difficulty is None
            else _strategy_spec(difficulty, _difficulty_kind, _DIFFICULTY_KWARG_TYPES)
        ),
        "mesh_fingerprint": predictor.mesh_fingerprint,
        "provenance": validate_provenance(chosen_provenance),
    }


def _save_predictor(
    predictor: ConformalPredictor,
    path: Path | str,
    *,
    provenance: Mapping | None = None,
) -> None:
    """Atomically write a weights-only-safe conformal artifact."""
    payload = _artifact_payload(predictor, provenance)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
        # Release the payload before the integrity read-back so at most one
        # extra full-size threshold copy is ever live beside the predictor.
        del payload
        _load_predictor(temporary, map_location="cpu")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _load_predictor(
    path: Path | str, map_location: str | torch.device = "cpu"
) -> ConformalPredictor:
    r"""Load and semantically validate a fitted predictor artifact.

    :meth:`~physicsnemo.experimental.uq.conformal.ConformalPredictor.load` is
    the equivalent class-level entry point.

    Parameters
    ----------
    path : Path | str
        Artifact written by
        :meth:`~physicsnemo.experimental.uq.conformal.ConformalPredictor.save`.
    map_location : str | torch.device, optional
        Device for the loaded thresholds. Default is ``"cpu"``.

    Returns
    -------
    ConformalPredictor
        The reconstructed
        :class:`~physicsnemo.experimental.uq.conformal.ConformalPredictor`.

    Notes
    -----
    The file is read with ``torch.load(weights_only=True)``. Raises
    ``ValueError`` when the payload is not a conformal artifact, has an
    unsupported version, or fails the schema and predictor validation.
    """
    payload = torch.load(path, map_location=map_location, weights_only=True)
    try:
        return _parse_artifact(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid conformal artifact at {path}: {exc}") from exc
