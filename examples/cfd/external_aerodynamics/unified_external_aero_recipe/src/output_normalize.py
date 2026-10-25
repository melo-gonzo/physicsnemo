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

- **Mesh output** (GLOBE, ISLA): a point-cloud ``Mesh`` whose ``point_data``
  holds one leaf per target name (a dotted target name is a nested leaf),
  each at its natural rank -- ``(N,)`` for a scalar, ``(N, 3)`` for a vector,
  ``(N, 3, 3)`` for a symmetric tensor, of which the listed laboratory
  components are read off. The result has batch_size ``[N]`` matching the
  mesh's points.
- **Tensor output** (e.g. GeoTransolver / Transolver / FLARE): a ``(B, N, C)``
  tensor whose channels are concatenated in ``target_config`` order; we
  slice by per-field dim count and squeeze the trailing 1 for scalars.

Kept in its own module (separate from ``train.py``) so tests can exercise
the slicing logic without paying the cost of importing tensorboard.
"""

from __future__ import annotations

import math
from typing import Any, Literal, TypeAlias

import torch
from jaxtyping import Float
from omegaconf import DictConfig, OmegaConf
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
### `forward()` consumes / returns Mesh-like objects or plain `(B, N, C)`
### tensors. The collate, output normalizer, and forward-pass dispatch all
### key off this same enum.
IOType: TypeAlias = Literal["mesh", "tensors"]


def require_output_type(cfg: DictConfig) -> IOType:
    """Return the model's declared ``output_type``, or raise if missing/invalid.

    Both entry points (``train.py`` / ``infer.py``) require the model YAML
    to declare ``output_type`` (``"mesh"`` or ``"tensors"``) so the forward
    output can be unpacked; this is the shared validation.
    """
    output_type = cfg.get("output_type", None)
    if output_type not in ("mesh", "tensors"):
        raise ValueError(
            f"Model YAML must declare `output_type` as one of 'mesh', "
            f"'tensors'; got {output_type!r}."
        )
    return output_type


def model_output_stats(model_cfg: DictConfig, normalizer: Any | None) -> dict:
    """The ``output_stats`` keyword of a model config that declares the key (ISLA's own
    scales: the model standardizes its physical outputs itself) left null: the
    dataset's ``NormalizeMeshFields`` stats as plain numbers, to inject; ``{}`` for a
    model config without the key or with its own stats, which must be the dataset's
    (:func:`check_model_output_stats`)."""
    if "output_stats" not in model_cfg or model_cfg.get("output_stats") is not None:
        return {}
    if normalizer is None:
        raise ValueError(
            "the model config declares output_stats, but the dataset pipeline has no NormalizeMeshFields"
        )
    return {
        "output_stats": {
            name: {"mean": torch.as_tensor(s["mean"]).tolist(), "std": float(s["std"])}
            for name, s in normalizer.stats.items()
        }
    }


def model_dataset_constants(model_cfg: DictConfig, dataset_cfg: DictConfig) -> dict:
    """The constants a model config leaves to the dataset (ISLA's own scales: an output's or a
    boundary input's ``scale: null``), from the dataset file's ``isla_constants``
    block, which is fitted per split on its training side like the targets' stats;
    ``{}`` for a model config that leaves none. A config that leaves one to a dataset
    file without it is refused."""

    def nulls(schema):
        return [
            n
            for n, f in schema.items()
            if isinstance(f, dict) and "scale" in f and f["scale"] is None
        ]

    outputs = (
        OmegaConf.to_container(model_cfg.output_schema)
        if "output_schema" in model_cfg
        else {}
    )
    conditions = (
        OmegaConf.to_container(model_cfg.boundary_schemas)
        if "boundary_schemas" in model_cfg
        else {}
    )
    null_out = nulls(outputs)
    null_in = {c: nulls(f or {}) for c, f in conditions.items() if nulls(f or {})}
    if not (null_out or null_in):
        return {}
    block = OmegaConf.select(dataset_cfg, "isla_constants", default=None)
    block = OmegaConf.to_container(block) if block is not None else {}
    given_in = block.get("input_scale", {})
    missing = [
        f"output_scale.{n}" for n in null_out if n not in block.get("output_scale", {})
    ] + [
        f"input_scale.{c}.{n}"
        for c, names in null_in.items()
        for n in names
        if n not in given_in.get(c, {})
    ]
    if missing:
        raise ValueError(
            f"the model config leaves {missing!r} to the dataset, whose file has no such isla_constants entry"
        )
    out = {}
    if null_out:
        for n in null_out:
            outputs[n]["scale"] = block["output_scale"][n]
        out["output_schema"] = outputs
    if null_in:
        for c, names in null_in.items():
            for n in names:
                conditions[c][n]["scale"] = given_in[c][n]
        out["boundary_schemas"] = conditions
    return out


def check_model_output_stats(model: torch.nn.Module, normalizer: Any | None) -> None:
    """Refuse a model whose ``output_stats`` are not the dataset normalizer's: the
    model's standardized outputs and the targets must be in one space."""
    model = getattr(model, "module", model)
    stats = getattr(model, "output_stats", None)
    if stats is None:
        return
    live = normalizer.stats if normalizer is not None else {}
    same = stats.keys() == live.keys() and all(
        torch.allclose(
            torch.as_tensor(stats[k]["mean"], dtype=torch.float64),
            torch.as_tensor(live[k]["mean"]).double(),
        )
        and math.isclose(stats[k]["std"], float(live[k]["std"]), rel_tol=1e-6)
        for k in stats
    )
    if not same:
        raise ValueError(
            f"the model's output_stats {stats!r} are not the dataset's NormalizeMeshFields stats {live!r}"
        )


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
    and raises (``field_dim``): it needs a mesh-output model.

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


def select_mesh_fields_by_target(
    output: Mesh, target_config: dict[str, FieldType]
) -> TensorDict:
    """Adapt a mesh-output model's ``Mesh`` into the per-field TensorDict.

    The output Mesh's ``point_data`` holds one leaf per target name (the model
    YAML declares its outputs with the dataset's target names, dotted where the
    dataset nests them). A ``"scalar"`` target takes the ``(N,)`` leaf; a
    ``"vector"`` target the ``(N, 3)`` leaf; a symmetric-tensor target reads its
    listed laboratory components off the ``(N, 3, 3)`` leaf, in the YAML's
    component order (leaf shape ``(N, n_listed)``, matching
    :func:`forward_kwargs.extract_targets`). The result keeps the mesh's
    batch_size ``[N]`` and device.

    Raises:
        KeyError: If a target name is missing from the output's ``point_data``.
        ValueError: If a leaf's shape does not match its declared type.
    """
    pd = output.point_data
    keys = [as_nested_key(name) for name in target_config]
    missing = [name for name, key in zip(target_config, keys) if key not in pd]
    if missing:
        raise KeyError(
            f"Mesh output is missing target fields {missing!r}; "
            f"available: {format_leaf_keys(pd)!r}"
        )
    out = TensorDict({}, batch_size=pd.batch_size, device=pd.device)
    for (name, ftype), key in zip(target_config.items(), keys):
        field = pd[key]
        kind = field_kind(ftype)
        expected = {"scalar": 1, "vector": 2, "symmetric_tensor": 3}[kind]
        if field.ndim != expected:
            raise ValueError(
                f"Output field {name!r} is declared {kind!r} and must have "
                f"{expected} dims ((N{', 3' * (expected - 1)})), got shape "
                f"{tuple(field.shape)}."
            )
        if kind == "symmetric_tensor":
            field = select_tensor_components(field, ftype)
        out.set(key, field)
    return out


def normalize_output_to_tensordict(
    output: Any,
    target_config: dict[str, FieldType],
    output_type: IOType,
    n_spatial_dims: int = 3,
) -> TensorDict:
    """Adapt a model output into a `TensorDict` keyed by target name.

    For ``output_type == "mesh"``, the output is expected to be a `Mesh`
    whose `.point_data` contains one leaf per target name at its natural
    rank (GLOBE, ISLA); we select them via
    :func:`select_mesh_fields_by_target`, so the result inherits the mesh's
    batch_size (``[N]``) and device and a symmetric-tensor target is reduced
    to its listed components.

    For ``output_type == "tensors"``, the output is expected to be a
    ``(B, N, C)`` tensor whose channels are concatenated in
    ``target_config`` order (e.g. GeoTransolver, Transolver, FLARE,
    DoMINO); we slice it via :func:`split_concat_by_target`. DoMINO
    returns a ``(vol, surf)`` tuple; we take the non-None element
    automatically.

    Args:
        output: Raw model output. ``Mesh`` for ``output_type='mesh'``;
            ``Tensor`` (or DoMINO ``(vol, surf)`` tuple) for
            ``output_type='tensors'``.
        target_config: Ordered ``{name: "scalar" | "vector" | tensor spec}`` mapping.
        output_type: ``"mesh"`` or ``"tensors"``.
        n_spatial_dims: Vector-field dimensionality. Default 3.

    Returns:
        Per-field TensorDict ready to feed into the loss / metric calculators.
    """
    if output_type == "mesh":
        if not isinstance(output, Mesh):
            raise TypeError(
                f"output_type='mesh' but model returned {type(output).__name__}"
            )
        return select_mesh_fields_by_target(output, target_config)

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

    raise ValueError(f"Unknown output_type {output_type!r}")
