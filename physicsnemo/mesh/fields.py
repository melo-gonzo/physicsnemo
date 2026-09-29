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

r"""Semantic tensor-rank field specifications.

Mesh models commonly need to turn named physical fields into dense channel
tensors.  A plain concatenation is not sufficient: a scalar field and a
Cartesian vector field transform differently under a change of frame.  This
module gives every named field a declared transformation law that models and
data pipelines can validate against.

:class:`RankSpec` is the per-field declaration: the tensor ``rank`` (0 scalar,
1 vector, 2 rank-2 tensor, ...), whether a rank >= 2 tensor is fully
``symmetric`` under index permutation, and its ``parity`` (``"even"`` for a
true tensor, ``"odd"`` for a pseudotensor that flips sign under an improper
rotation).  Multiple channels are represented by multiple named leaves, never
by an extra unnamed feature axis.

``RankSpecDict`` maps field names to leaf specs.  A leaf is written as a bare
integer rank, a :class:`RankSpec`, or a mapping with a ``"rank"`` key; a
mapping without a ``"rank"`` key is a nested group of fields.  A flat name may
also contain the separator (``"surface.pressure"``) to denote nesting, so the
nested and dotted forms describe the same field.  A name cannot be both a
field and a group of fields.

A rank-1 leaf is not an arbitrary feature axis of length ``D``: it is
explicitly promised to transform as a vector, and a rank-0 leaf has shape
``(N,)`` with no feature axis.  Axial vectors and higher-order tensors are
declared through :class:`RankSpec` rather than silently treated as polar
vectors.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, Union

from tensordict import TensorDict


@dataclass(frozen=True)
class RankSpec:
    r"""Transformation law of one named tensor field.

    Parameters
    ----------
    rank : int
        Number of spatial indices: 0 for a scalar, 1 for a vector, 2 for a
        rank-2 tensor, and so on.
    symmetric : bool, default=False
        Whether the field is fully symmetric under any permutation of its
        indices.  Only meaningful for ``rank >= 2``; rejected below that.
    parity : {"even", "odd"}, default="even"
        ``"even"`` for a true tensor and ``"odd"`` for a pseudotensor, which
        additionally flips sign under an improper rotation.

    Examples
    --------
    >>> RankSpec.parse(1)
    RankSpec(rank=1, symmetric=False, parity='even')
    >>> RankSpec.parse({"rank": 2, "symmetric": True}).shape(3)
    (3, 3)
    """

    rank: int
    symmetric: bool = False
    parity: Literal["even", "odd"] = "even"

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise TypeError(f"rank must be an integer, got {self.rank!r}")
        if self.rank < 0:
            raise ValueError(f"rank must be non-negative, got {self.rank}")
        if not isinstance(self.symmetric, bool):
            raise TypeError(f"symmetric must be a bool, got {self.symmetric!r}")
        if self.symmetric and self.rank < 2:
            raise ValueError(
                f"symmetric=True is only meaningful for rank >= 2, got rank {self.rank}"
            )
        if self.parity not in ("even", "odd"):
            raise ValueError(f"parity must be 'even' or 'odd', got {self.parity!r}")

    def size(self, dim: int) -> int:
        r"""Number of components per point in ``dim`` spatial dimensions."""
        return dim**self.rank

    def shape(self, dim: int) -> tuple[int, ...]:
        r"""Trailing component shape: ``()`` at rank 0, ``(dim,) * rank`` above."""
        return () if self.rank == 0 else (dim,) * self.rank

    @classmethod
    def parse(cls, spec: object, *, source_label: str = "rank spec") -> RankSpec:
        r"""Normalize a leaf spec to a :class:`RankSpec`.

        Parameters
        ----------
        spec : int, RankSpec, or Mapping
            A bare integer rank, an existing :class:`RankSpec`, or a mapping
            with the required key ``"rank"`` and the optional keys
            ``"symmetric"`` and ``"parity"``.
        source_label : str, default="rank spec"
            Human-readable name used in error messages.

        Raises
        ------
        TypeError
            If ``spec`` is a bool, is none of the accepted forms, or a field
            has the wrong type.
        ValueError
            If the mapping has unknown keys or lacks ``"rank"``, or a field
            value is invalid (negative rank, bad parity, symmetric below
            rank 2).
        """
        if isinstance(spec, cls):
            return spec
        if isinstance(spec, bool):
            raise TypeError(
                f"{source_label} must be an integer rank, a RankSpec, or a "
                f"mapping with a 'rank' key; got a bool"
            )
        if isinstance(spec, int):
            fields: dict[str, Any] = {"rank": spec}
        elif isinstance(spec, Mapping):
            unknown = set(spec) - {"rank", "symmetric", "parity"}
            if unknown:
                raise ValueError(
                    f"{source_label} has unknown keys {sorted(map(str, unknown))!r}; "
                    f"allowed keys are ['parity', 'rank', 'symmetric']"
                )
            if "rank" not in spec:
                raise ValueError(f"{source_label} mapping must contain a 'rank' key")
            fields = dict(spec)
        else:
            raise TypeError(
                f"{source_label} must be an integer rank, a RankSpec, or a "
                f"mapping with a 'rank' key; got {type(spec).__name__}"
            )
        try:
            return cls(**fields)
        except (TypeError, ValueError) as error:
            raise type(error)(f"{source_label}: {error}") from None


# TODO: replace with ``type RankSpecDict = ...`` after Python 3.11 support is
# dropped (PEP 695).
RankSpecDict: TypeAlias = dict[
    str, Union[int, RankSpec, Mapping[str, Any], "RankSpecDict"]
]


def is_leaf_spec(value: object) -> bool:
    r"""Decide whether a ``RankSpecDict`` value is a field leaf or a group.

    A mapping without a ``"rank"`` key is a nested group of fields; every
    other value (an integer, a :class:`RankSpec`, a mapping with ``"rank"``,
    or anything invalid that :meth:`RankSpec.parse` will reject) is a leaf.
    """
    return not isinstance(value, Mapping) or "rank" in value


def _split_name(name: str, sep: str, source_label: str) -> tuple[str, ...]:
    """Split a field name on ``sep`` into non-empty path components."""
    if not isinstance(name, str):
        raise TypeError(f"All keys in {source_label} must be strings; got {name!r}")
    path = tuple(name.split(sep))
    if not all(path):
        raise ValueError(
            f"Field name {name!r} in {source_label} has an empty path component"
        )
    return path


def _leaf_entries(
    rank_spec: RankSpecDict,
    *,
    sep: str,
    source_label: str,
    allowed_ranks: Collection[int] | None = None,
) -> list[tuple[tuple[str, ...], RankSpec]]:
    """Walk ``rank_spec`` and return validated ``(path, RankSpec)`` leaves.

    Nested groups and separator-containing names are both expanded to path
    tuples.  Duplicate names and names that are simultaneously a field and a
    group of fields are refused.
    """
    if not isinstance(rank_spec, Mapping):
        raise TypeError(
            f"{source_label} must be a mapping, got {type(rank_spec).__name__}"
        )
    allowed = None if allowed_ranks is None else frozenset(allowed_ranks)
    entries: list[tuple[tuple[str, ...], RankSpec]] = []

    def _walk(spec: Mapping[str, Any], prefix: tuple[str, ...]) -> None:
        for key, value in spec.items():
            path = (*prefix, *_split_name(key, sep, source_label))
            if not is_leaf_spec(value):
                _walk(value, path)
                continue
            name = sep.join(path)
            leaf = RankSpec.parse(value, source_label=f"{source_label}[{name!r}]")
            if allowed is not None and leaf.rank not in allowed:
                raise ValueError(
                    f"Rank for {name!r} in {source_label} must be one of "
                    f"{sorted(allowed)}, got {leaf.rank}"
                )
            entries.append((path, leaf))

    _walk(rank_spec, ())

    seen: set[tuple[str, ...]] = set()
    for path, _ in entries:
        if path in seen:
            raise ValueError(
                f"{source_label} declares the field {sep.join(path)!r} more than once"
            )
        seen.add(path)
    for path in seen:
        for depth in range(1, len(path)):
            if path[:depth] in seen:
                raise ValueError(
                    f"{source_label}: {sep.join(path[:depth])!r} is a group of "
                    f"{sep.join(path)!r}; a name cannot be both a field and a "
                    f"group of fields"
                )
    return entries


def flatten_rank_specs(rank_spec: RankSpecDict, sep: str = ".") -> dict[str, RankSpec]:
    r"""Flatten a rank specification to separator-joined names and leaf specs.

    Nested groups and separator-containing flat names are treated alike, so
    ``{"surface": {"pressure": 0}}`` and ``{"surface.pressure": 0}`` flatten
    to the same result.  The insertion order of ``rank_spec`` is retained.

    Parameters
    ----------
    rank_spec : RankSpecDict
        Mapping from field names to leaf specs (integer ranks,
        :class:`RankSpec` instances, or mappings with a ``"rank"`` key) or to
        nested groups.
    sep : str, default="."
        Separator used to join nested path components and to split flat names.

    Returns
    -------
    dict[str, RankSpec]
        Flat field-name to :class:`RankSpec` mapping.

    Raises
    ------
    TypeError
        If a key is not a string or a leaf is not an accepted spec form.
    ValueError
        If a leaf spec is invalid, a name is declared twice, or a name is
        both a field and a group of fields.

    Examples
    --------
    >>> flatten_rank_specs({"fluid": {"pressure": 0}, "stress": {"rank": 2}})
    {'fluid.pressure': RankSpec(rank=0, symmetric=False, parity='even'), 'stress': RankSpec(rank=2, symmetric=False, parity='even')}
    """
    return {
        sep.join(path): leaf
        for path, leaf in _leaf_entries(rank_spec, sep=sep, source_label="rank_spec")
    }


def flatten_rank_spec(rank_spec: RankSpecDict, sep: str = ".") -> dict[str, int]:
    r"""Flatten a nested rank specification to separator-joined names and ranks.

    This is :func:`flatten_rank_specs` reduced to the integer rank of every
    leaf; symmetry and parity are dropped.  The insertion order of
    ``rank_spec`` is retained for compatibility.  Code that assigns positional
    channels should sort the returned names, as :class:`FieldLayout` does.

    Parameters
    ----------
    rank_spec : RankSpecDict
        Mapping from field names to leaf specs or nested mappings.
    sep : str, default="."
        Separator used to join nested path components and to split flat names.

    Returns
    -------
    dict[str, int]
        Flat field-name to semantic-rank mapping.

    Examples
    --------
    >>> flatten_rank_spec({"pressure": 0, "velocity": 1})
    {'pressure': 0, 'velocity': 1}
    >>> flatten_rank_spec({"fluid": {"pressure": 0, "velocity": 1}})
    {'fluid.pressure': 0, 'fluid.velocity': 1}
    """
    return {
        name: leaf.rank for name, leaf in flatten_rank_specs(rank_spec, sep=sep).items()
    }


def validate_rank_spec(
    rank_spec: RankSpecDict,
    *,
    allowed_ranks: Collection[int] | None = None,
    source_label: str = "rank_spec",
) -> None:
    r"""Validate the structure and leaves of a rank specification.

    Every leaf is validated through :meth:`RankSpec.parse`, so integer ranks,
    :class:`RankSpec` instances, and mappings with a ``"rank"`` key are all
    accepted.

    Parameters
    ----------
    rank_spec : RankSpecDict
        Rank specification to validate.
    allowed_ranks : Collection[int] or None, optional
        If supplied, every leaf rank must belong to this collection.
        Otherwise all non-negative integer ranks are accepted.
    source_label : str, default="rank_spec"
        Human-readable name used in error messages.

    Raises
    ------
    TypeError
        If a key is not a string or a leaf is not an accepted spec form.
    ValueError
        If a leaf spec is invalid, a rank is not in ``allowed_ranks``, or a
        name is declared twice or is both a field and a group of fields.
    """
    _leaf_entries(
        rank_spec, sep=".", source_label=source_label, allowed_ranks=allowed_ranks
    )


def rank_counts(rank_spec: RankSpecDict) -> Counter[int]:
    r"""Count leaves of each semantic rank in ``rank_spec``."""
    return Counter(flatten_rank_spec(rank_spec).values())


def ranks_from_tensordict(td: TensorDict) -> RankSpecDict:
    r"""Derive semantic-rank-shaped metadata from TensorDict leaf shapes.

    A leaf rank is its number of non-batch dimensions. For a point-field
    TensorDict with batch size ``(N,)``, ``(N,)`` is therefore rank 0 and
    ``(N, D)`` is rank 1.
    """
    result: RankSpecDict = {}
    for key in td.keys():
        value = td[key]
        if isinstance(value, TensorDict):
            result[key] = ranks_from_tensordict(value)  # ty: ignore[invalid-assignment]
        else:
            result[key] = value.ndim - td.batch_dims  # ty: ignore[invalid-assignment]
    return result


def validate_data_contains_ranks(
    *,
    data: TensorDict,
    declared_ranks: RankSpecDict,
    source_label: str,
) -> None:
    r"""Validate that ``data`` contains every declared leaf at its stated rank.

    Additional leaves in ``data`` are allowed. Missing leaves and rank
    mismatches are reported together to make schema errors easier to diagnose.

    Parameters
    ----------
    data : TensorDict
        Data TensorDict to validate.
    declared_ranks : RankSpecDict
        Rank specification that ``data`` must contain as a subset.
    source_label : str
        Human-readable description used in the error message.

    Raises
    ------
    ValueError
        If a declared field is missing or has a different rank.
    """
    validate_rank_spec(declared_ranks, source_label="declared_ranks")
    declared = flatten_rank_spec(declared_ranks)
    actual = flatten_rank_spec(ranks_from_tensordict(data))

    lines = [
        f"  - missing leaf {key!r} (declared rank {declared[key]})"
        for key in sorted(declared.keys() - actual.keys())
    ]
    lines.extend(
        [
            f"  - rank mismatch for {key!r}: declared {declared[key]}, "
            f"got {actual[key]}"
            for key in sorted(declared.keys() & actual.keys())
            if declared[key] != actual[key]
        ]
    )
    if lines:
        raise ValueError(
            f"{source_label} does not contain its declared rank spec:\n"
            + "\n".join(lines)
        )


__all__ = [
    "RankSpec",
    "RankSpecDict",
    "flatten_rank_spec",
    "flatten_rank_specs",
    "is_leaf_spec",
    "rank_counts",
    "ranks_from_tensordict",
    "validate_data_contains_ranks",
    "validate_rank_spec",
]
