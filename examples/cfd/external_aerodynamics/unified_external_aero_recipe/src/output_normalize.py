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

"""Adapt model outputs into a per-field `TensorDict` keyed by target name.

The recipe's loss / metrics consume `TensorDict[name -> tensor]` predictions
keyed by the field names declared in the dataset YAML's ``targets:`` block.
Different model output conventions need different unpacking:

- **Mesh output** (e.g. GLOBE): ``output.point_data.select(*target_config)``
  yields a TensorDict with batch_size ``[N]`` matching the mesh's points.
- **Tensor output** (e.g. GeoTransolver / Transolver / FLARE): a ``(B, N, C)``
  tensor whose channels are concatenated in ``target_config`` order; we
  slice by per-field dim count and squeeze the trailing 1 for scalars.
- **Fields output** (ISLA): a ``{name: tensor}`` mapping keyed by the target
  names, each tensor of its declared rank -- ``(B, N)`` for a scalar,
  ``(B, N, 3)`` for a vector, ``(B, N, 3, 3)`` for a symmetric tensor, of which
  the listed laboratory components are read off.

Kept in its own module (separate from ``train.py``) so tests can exercise
the slicing logic without paying the cost of importing tensorboard.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypeAlias

import torch
from jaxtyping import Float
from omegaconf import DictConfig
from tensordict import TensorDict
from utils import (
    FieldType,
    field_dim,
    field_kind,
    select_tensor_components,
    tensor_components,
)

from physicsnemo.datapipes.keys import as_nested_key, format_leaf_keys
from physicsnemo.mesh import Mesh

### Recipe-wide I/O contract literal: every model declares whether its
### `forward()` consumes / returns Mesh-like objects, plain `(B, N, C)`
### tensors, or (output side only) a mapping of named fields of declared
### rank. The collate, output normalizer, and forward-pass dispatch all key
### off this same enum.
IOType: TypeAlias = Literal["mesh", "tensors", "fields"]


def require_output_type(cfg: DictConfig) -> IOType:
    """Return the model's declared ``output_type``, or raise if missing/invalid.

    Both entry points (``train.py`` / ``infer.py``) require the model YAML
    to declare ``output_type`` (``"mesh"``, ``"tensors"`` or ``"fields"``) so
    the forward output can be unpacked; this is the shared validation.
    """
    output_type = cfg.get("output_type", None)
    if output_type not in ("mesh", "tensors", "fields"):
        raise ValueError(
            f"Model YAML must declare `output_type` as one of 'mesh', "
            f"'tensors', 'fields'; got {output_type!r}."
        )
    return output_type


def split_concat_by_target(
    tensor: Float[torch.Tensor, "B N C"],
    target_config: dict[str, FieldType],
    n_spatial_dims: int = 3,
) -> TensorDict:
    """Slice a ``(B, N, C)`` tensor by ``target_config`` into a per-field TensorDict.

    Channels are consumed in the iteration order of ``target_config``: a
    ``"scalar"`` field eats one channel and the trailing dim is squeezed
    (so the leaf is shape ``(B, N)``); a ``"vector"`` field eats
    ``n_spatial_dims`` channels and keeps its trailing dim (leaf shape
    ``(B, N, n_spatial_dims)``). A symmetric-tensor target has no flat layout
    and raises (``field_dim``): it needs a ``"fields"`` model.

    Args:
        tensor: Concatenated output tensor, shape ``(B, N, C)`` where
            ``C == sum(field_dim(t, n_spatial_dims) for t in target_config.values())``.
        target_config: Ordered ``{name: "scalar" | "vector"}`` mapping.
        n_spatial_dims: Vector-field dimensionality. Default 3.

    Returns:
        TensorDict with one leaf per target field and ``batch_size == tensor.shape[:2]``.

    Raises:
        ValueError: If the channel count does not match the expected total.
    """
    expected_channels = sum(
        field_dim(t, n_spatial_dims) for t in target_config.values()
    )
    if tensor.shape[-1] != expected_channels:
        raise ValueError(
            f"Output channel dim {tensor.shape[-1]} does not match the "
            f"expected total channels {expected_channels} for "
            f"target_config={target_config!r}."
        )

    ### Build the TD leaf by leaf; ``set`` with a tuple key creates the
    ### nesting, so a ``"solution.p"`` target lands at the same place
    ### ``extract_targets`` reads it from.
    out = TensorDict({}, batch_size=tensor.shape[:2], device=tensor.device)
    idx = 0
    for name, ftype in target_config.items():
        dim = field_dim(ftype, n_spatial_dims)
        slice_ = tensor[..., idx : idx + dim]
        if field_kind(ftype) == "scalar":
            slice_ = slice_.squeeze(-1)
        out.set(as_nested_key(name), slice_)
        idx += dim

    return out


def explode_tensor_targets(
    td: TensorDict, target_config: dict[str, FieldType]
) -> TensorDict:
    """Replace each symmetric-tensor leaf by one leaf per listed component field.

    ``stress`` with ``components: {sig11: xx, sig22: yy, sig12: xy}`` becomes
    ``sig11``, ``sig22``, ``sig12`` (the dataset's own field names), so the
    dataset normalizer's inverse and the saved ``pred_<name>`` files read
    exactly as they do for lanes that predicted the components as scalars.
    A config without tensor targets is returned unchanged.
    """
    if not any(field_kind(t) == "symmetric_tensor" for t in target_config.values()):
        return td
    out = TensorDict({}, batch_size=td.batch_size, device=td.device)
    for name, ftype in target_config.items():
        key = as_nested_key(name)
        if field_kind(ftype) == "symmetric_tensor":
            leaf = td[key]
            for i, field in enumerate(tensor_components(ftype)):
                out.set(as_nested_key(field), leaf[..., i])
        else:
            out.set(key, td[key])
    return out


def select_fields_by_target(
    output: Mapping[str, torch.Tensor], target_config: dict[str, FieldType]
) -> TensorDict:
    """Adapt a ``{name: tensor}`` model output into the per-field TensorDict.

    The model's output names are the target names (the model YAML declares its
    ``outputs:`` with the dataset's target names). A ``"scalar"`` target takes
    the ``(B, N)`` field; a ``"vector"`` target the ``(B, N, 3)`` field; a
    symmetric-tensor target reads its listed laboratory components off the
    ``(B, N, 3, 3)`` field, in the YAML's component order (leaf shape
    ``(B, N, n_listed)``, matching :func:`forward_kwargs.extract_targets`).

    Raises:
        KeyError: If a target name is missing from the output.
        ValueError: If a field's shape does not match its declared type.
    """
    missing = [name for name in target_config if name not in output]
    if missing:
        raise KeyError(
            f"Model output is missing target fields {missing!r}; "
            f"available: {sorted(output)!r}."
        )
    any_field = output[next(iter(target_config))]
    out = TensorDict({}, batch_size=any_field.shape[:2], device=any_field.device)
    for name, ftype in target_config.items():
        field = output[name]
        kind = field_kind(ftype)
        expected = {"scalar": 2, "vector": 3, "symmetric_tensor": 4}[kind]
        if field.ndim != expected:
            raise ValueError(
                f"Output field {name!r} is declared {kind!r} and must have "
                f"{expected} dims ((B, N{', 3' * (expected - 2)})), got shape "
                f"{tuple(field.shape)}."
            )
        if kind == "symmetric_tensor":
            field = select_tensor_components(field, ftype)
        out.set(as_nested_key(name), field)
    return out


def normalize_output_to_tensordict(
    output: Any,
    target_config: dict[str, FieldType],
    output_type: IOType,
    n_spatial_dims: int = 3,
) -> TensorDict:
    """Adapt a model output into a `TensorDict` keyed by target name.

    For ``output_type == "mesh"``, the output is expected to be a `Mesh`
    whose `.point_data` contains one tensor per target name (e.g. GLOBE);
    we return ``output.point_data.select(*target_config)`` so the result
    inherits the mesh's batch_size (``[N]``) and device.

    For ``output_type == "tensors"``, the output is expected to be a
    ``(B, N, C)`` tensor whose channels are concatenated in
    ``target_config`` order (e.g. GeoTransolver, Transolver, FLARE,
    DoMINO); we slice it via :func:`split_concat_by_target`. DoMINO
    returns a ``(vol, surf)`` tuple; we take the non-None element
    automatically.

    For ``output_type == "fields"``, the output is a ``{name: tensor}``
    mapping keyed by the target names (ISLA); we select the targets via
    :func:`select_fields_by_target`.

    Args:
        output: Raw model output. ``Mesh`` for ``output_type='mesh'``;
            ``Tensor`` (or DoMINO ``(vol, surf)`` tuple) for
            ``output_type='tensors'``; a mapping for ``output_type='fields'``.
        target_config: Ordered ``{name: "scalar" | "vector" | tensor spec}`` mapping.
        output_type: ``"mesh"``, ``"tensors"`` or ``"fields"``.
        n_spatial_dims: Vector-field dimensionality. Default 3.

    Returns:
        Per-field TensorDict ready to feed into the loss / metric calculators.
    """
    if output_type == "mesh":
        if not isinstance(output, Mesh):
            raise TypeError(
                f"output_type='mesh' but model returned {type(output).__name__}"
            )
        keys = [as_nested_key(name) for name in target_config]
        missing = [
            name
            for name, key in zip(target_config, keys)
            if key not in output.point_data
        ]
        if missing:
            raise KeyError(
                f"Mesh output is missing target fields {missing!r}; "
                f"available: {format_leaf_keys(output.point_data)!r}"
            )
        return output.point_data.select(*keys)

    if output_type == "tensors":
        if isinstance(output, tuple):
            ### DoMINO returns ``(vol, surf)`` with the unused branch as
            ### ``None``; pick the first non-None entry. A tuple of all
            ### Nones means the model was misconfigured for both modes,
            ### which the bare ``next(...)`` would surface as a cryptic
            ### ``StopIteration``; raise an explicit, actionable error here
            ### instead.
            non_none = [o for o in output if o is not None]
            if not non_none:
                raise ValueError(
                    f"output_type='tensors' got a tuple of all-None values; "
                    f"expected at least one non-None tensor in the tuple. "
                    f"Got {output!r}."
                )
            output = non_none[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"output_type='tensors' but model returned {type(output).__name__}"
            )
        ### A 2-D output (B, N) is almost certainly a model that dropped the
        ### channel dim for a single-scalar target. Diagnose that explicitly
        ### before the channel-count check, otherwise the user sees
        ### "expected 1, got N" which mistakes the per-element axis for the
        ### channel axis.
        if output.ndim < 3:
            raise ValueError(
                f"output_type='tensors' expects a (B, N, C) tensor; got "
                f"shape {tuple(output.shape)} (ndim={output.ndim}). If your "
                f"model returns (B, N) for a single-scalar target, add a "
                f"trailing channel dim (e.g. ``out.unsqueeze(-1)``)."
            )
        return split_concat_by_target(output, target_config, n_spatial_dims)

    if output_type == "fields":
        if not isinstance(output, Mapping):
            raise TypeError(
                f"output_type='fields' but model returned {type(output).__name__}"
            )
        return select_fields_by_target(output, target_config)

    raise ValueError(f"Unknown output_type {output_type!r}")
