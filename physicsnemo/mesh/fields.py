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

r"""Semantic tensor-rank field specifications and scalar/vector layouts.

Mesh models commonly need to turn named physical fields into dense channel
tensors.  A plain concatenation is not sufficient: a scalar field and a
Cartesian vector field transform differently under a change of frame.  This
module gives every named field a declared transformation law and a stable,
name-based channel order.

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

The currently supported :class:`FieldLayout` leaves are

* rank 0: one scalar per point, with shape ``(N,)``;
* rank 1: one polar (parity ``"even"``) vector per point, with shape ``(N, D)``.

Consequently, a rank-1 leaf is not an arbitrary feature axis of length ``D``:
it is explicitly promised to transform as a polar vector.  Axial vectors and
higher-order tensors are declared through :class:`RankSpec` and are not
silently treated as polar vectors here.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, TypeAlias, Union

import torch
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


class ScalarVectorFields(NamedTuple):
    r"""Dense scalar and polar-vector channels for a collection of points.

    ``scalars`` has shape ``(N, C_s)`` and ``vectors`` has shape
    ``(N, C_v, D)``.  Shape validation belongs to the :class:`FieldLayout`
    that defines ``C_s``, ``C_v``, and ``D``.
    """

    scalars: torch.Tensor
    vectors: torch.Tensor


def _td_key(path: tuple[str, ...]) -> str | tuple[str, ...]:
    """Use a string for top-level TensorDict keys and tuples for nested keys."""
    return path[0] if len(path) == 1 else path


class FieldLayout:
    r"""Deterministic packing layout for named scalar and polar-vector fields.

    Channel order is lexicographic by the flattened field name and therefore
    does not depend on dictionary or TensorDict construction order.  Scalars
    and vectors are packed separately so that equivariant code cannot
    accidentally mix their transformation laws.

    Parameters
    ----------
    rank_spec : RankSpecDict
        Nested or flat field schema.  Only rank-0 and polar (parity
        ``"even"``) rank-1 leaves are supported; leaves may be given as
        integers, :class:`RankSpec` instances, or mappings with a ``"rank"``
        key.
    spatial_dim : int
        Cartesian vector dimension ``D``.
    sep : str, default="."
        Separator used for public flattened names and sorting, and to split
        flat names into nested TensorDict paths.

    Notes
    -----
    Each rank-0 leaf must have shape ``(N,)`` and each rank-1 leaf must have
    shape ``(N, D)``.  Multiple channels must be represented by multiple named
    leaves rather than an extra unnamed feature dimension.
    """

    def __init__(
        self,
        rank_spec: RankSpecDict,
        spatial_dim: int,
        *,
        sep: str = ".",
    ) -> None:
        if isinstance(spatial_dim, bool) or not isinstance(spatial_dim, int):
            raise TypeError(f"spatial_dim must be an integer, got {spatial_dim!r}")
        if spatial_dim < 1:
            raise ValueError(f"spatial_dim must be positive, got {spatial_dim}")
        if not isinstance(sep, str):
            raise TypeError(f"sep must be a string, got {type(sep).__name__}")
        if not sep:
            raise ValueError("sep must not be empty")

        leaves = _leaf_entries(
            rank_spec, sep=sep, source_label="rank_spec", allowed_ranks=(0, 1)
        )
        if not leaves:
            raise ValueError("rank_spec must contain at least one field leaf")

        entries = []
        for path, leaf in leaves:
            name = sep.join(path)
            if leaf.parity != "even":
                raise ValueError(
                    f"Field {name!r} in rank_spec has parity {leaf.parity!r}; "
                    f"FieldLayout packs scalars and polar vectors only"
                )
            entries.append((name, path, leaf.rank))

        entries.sort(key=lambda entry: entry[0])
        self._rank_spec = deepcopy(rank_spec)
        self.spatial_dim = spatial_dim
        self.sep = sep
        self._entries = tuple(entries)
        self._scalar_entries = tuple(entry for entry in entries if entry[2] == 0)
        self._vector_entries = tuple(entry for entry in entries if entry[2] == 1)

    @property
    def rank_spec(self) -> RankSpecDict:
        """A defensive copy of the named-field schema."""
        return deepcopy(self._rank_spec)

    @property
    def flat_rank_spec(self) -> dict[str, int]:
        """Field ranks in deterministic packed-name order."""
        return {name: rank for name, _, rank in self._entries}

    @property
    def scalar_names(self) -> tuple[str, ...]:
        """Names of scalar channels in packed order."""
        return tuple(name for name, _, _ in self._scalar_entries)

    @property
    def vector_names(self) -> tuple[str, ...]:
        """Names of polar-vector channels in packed order."""
        return tuple(name for name, _, _ in self._vector_entries)

    @property
    def n_scalars(self) -> int:
        """Number of packed scalar channels."""
        return len(self._scalar_entries)

    @property
    def n_vectors(self) -> int:
        """Number of packed polar-vector channels."""
        return len(self._vector_entries)

    def pack(self, data: TensorDict) -> ScalarVectorFields:
        r"""Pack named point fields into deterministic scalar/vector channels.

        Extra leaves in ``data`` are ignored.  All selected leaves must share
        dtype and device, in addition to satisfying the semantic shapes in this
        layout.
        """
        if data.batch_dims != 1:
            raise ValueError(
                "FieldLayout expects a point TensorDict with one batch dimension "
                f"(N,), got batch_size={tuple(data.batch_size)}"
            )
        validate_data_contains_ranks(
            data=data,
            declared_ranks=self._rank_spec,
            source_label="data",
        )

        n_points = data.batch_size[0]
        selected: list[torch.Tensor] = []
        scalar_tensors: list[torch.Tensor] = []
        vector_tensors: list[torch.Tensor] = []

        for name, path, rank in self._entries:
            tensor = data[_td_key(path)]
            expected = (n_points,) if rank == 0 else (n_points, self.spatial_dim)
            if tuple(tensor.shape) != expected:
                kind = "scalar" if rank == 0 else "polar vector"
                raise ValueError(
                    f"Field {name!r} is declared as a {kind} and must have "
                    f"shape {expected}, got {tuple(tensor.shape)}"
                )
            selected.append(tensor)
            if rank == 0:
                scalar_tensors.append(tensor)
            else:
                vector_tensors.append(tensor)

        reference = selected[0]
        for name, tensor in zip(
            (name for name, _, _ in self._entries), selected, strict=True
        ):
            if tensor.device != reference.device:
                raise ValueError(
                    f"All packed fields must share a device; field {name!r} is "
                    f"on {tensor.device}, expected {reference.device}"
                )
            if tensor.dtype != reference.dtype:
                raise ValueError(
                    f"All packed fields must share a dtype; field {name!r} has "
                    f"{tensor.dtype}, expected {reference.dtype}"
                )

        scalars = (
            torch.stack(scalar_tensors, dim=-1)
            if scalar_tensors
            else reference.new_empty((n_points, 0))
        )
        vectors = (
            torch.stack(vector_tensors, dim=-2)
            if vector_tensors
            else reference.new_empty((n_points, 0, self.spatial_dim))
        )
        return ScalarVectorFields(scalars=scalars, vectors=vectors)

    def unpack(self, fields: ScalarVectorFields) -> TensorDict:
        r"""Unpack scalar/vector channels to a TensorDict matching ``rank_spec``."""
        scalars, vectors = fields
        if scalars.ndim != 2:
            raise ValueError(
                f"scalars must have shape (N, {self.n_scalars}), got "
                f"{tuple(scalars.shape)}"
            )
        if vectors.ndim != 3:
            raise ValueError(
                f"vectors must have shape (N, {self.n_vectors}, "
                f"{self.spatial_dim}), got {tuple(vectors.shape)}"
            )

        n_points = scalars.shape[0]
        expected_scalars = (n_points, self.n_scalars)
        expected_vectors = (n_points, self.n_vectors, self.spatial_dim)
        if tuple(scalars.shape) != expected_scalars:
            raise ValueError(
                f"scalars must have shape {expected_scalars}, got "
                f"{tuple(scalars.shape)}"
            )
        if tuple(vectors.shape) != expected_vectors:
            raise ValueError(
                f"vectors must have shape {expected_vectors}, got "
                f"{tuple(vectors.shape)}"
            )
        if vectors.shape[0] != n_points:
            raise ValueError(
                "scalars and vectors must contain the same number of points"
            )
        if vectors.device != scalars.device:
            raise ValueError("scalars and vectors must be on the same device")
        if vectors.dtype != scalars.dtype:
            raise ValueError("scalars and vectors must have the same dtype")

        nested: dict[str, object] = {}

        def _assign(path: tuple[str, ...], tensor: torch.Tensor) -> None:
            target = nested
            for key in path[:-1]:
                child = target.setdefault(key, {})
                # The schema validation guarantees that a path component is
                # either always a group or always a leaf.
                target = child  # type: ignore[assignment]
            target[path[-1]] = tensor

        for index, (_, path, _) in enumerate(self._scalar_entries):
            _assign(path, scalars[:, index])
        for index, (_, path, _) in enumerate(self._vector_entries):
            _assign(path, vectors[:, index, :])

        return TensorDict(
            nested,
            batch_size=torch.Size([n_points]),
            device=scalars.device,
        )


__all__ = [
    "FieldLayout",
    "RankSpec",
    "RankSpecDict",
    "ScalarVectorFields",
    "flatten_rank_spec",
    "flatten_rank_specs",
    "is_leaf_spec",
    "rank_counts",
    "ranks_from_tensordict",
    "validate_data_contains_ranks",
    "validate_rank_spec",
]
