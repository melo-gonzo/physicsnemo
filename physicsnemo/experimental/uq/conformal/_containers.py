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

r"""Private Tensor / ``TensorDict`` field-container dispatch.

The package accepts exactly two prediction containers: a plain tensor (one
anonymous field, keyed internally by :data:`TENSOR_KEY`) or a ``TensorDict``
of named fields. Every boundary normalizes through :func:`field_items` and
packs results back with :func:`pack_fields`, so the two representations
cannot drift apart.
"""

from collections.abc import Mapping, Sequence

from tensordict import TensorDict
from torch import Tensor

__all__ = ["TENSOR_KEY", "field_items", "pack_fields", "slice_aux"]

TENSOR_KEY = "__tensor__"
"""Reserved field key used when the inputs are plain tensors."""


def field_items(
    x: Tensor | TensorDict,
    keys: Sequence[str] | None = None,
) -> list[tuple[str, Tensor]]:
    """Normalize a tensor or field container into ``(key, tensor)`` pairs.

    Plain tensors map to a single pair keyed by the reserved internal
    :data:`TENSOR_KEY`; :class:`TensorDict` inputs map to their (optionally
    restricted) items in sorted key order. Arbitrary mappings are rejected:
    the fitted/calibration container contract has exactly two supported
    representations.
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
    """Inverse of :func:`field_items`.

    A single :data:`TENSOR_KEY` entry unpacks to a plain tensor; anything
    else packs into a :class:`~tensordict.TensorDict` with an empty batch
    size.
    """
    if set(items.keys()) == {TENSOR_KEY}:
        return items[TENSOR_KEY]
    return TensorDict(dict(items), batch_size=[])


def slice_aux(
    aux: Mapping[str, Tensor] | Mapping[str, Mapping[str, Tensor]] | None,
    field: str,
) -> Mapping[str, Tensor] | None:
    """Select the aux entries for one field.

    In plain-tensor mode (``field == TENSOR_KEY``) the aux mapping is used
    as-is. In field-container mode, ``aux`` is keyed by field name, each
    value being the aux mapping for that field (e.g. ``{"pressure":
    {"sigma": ...}}``). Missing fields resolve to ``None``.
    """
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
