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

"""Shared utilities for the unified training recipe."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypeAlias

import hydra.utils
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torch.amp import autocast

from physicsnemo.datapipes.keys import as_nested_key
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.optim import CombinedOptimizer, Muon

### Recipe-wide type aliases. Re-exported for use in loss.py, metrics.py,
### output_normalize.py, forward_kwargs.py, collate.py, train.py, infer.py,
### and the tests so that ``target_config`` values share a single source of
### truth.
### A target is ``"scalar"``, ``"vector"`` or (2026-09-22, TENSOR-HEAD) a symmetric
### rank-2 tensor spelled as a mapping whose ``components`` name the dataset fields
### that hold its laboratory-frame components::
###
###     stress: {type: symmetric_tensor, components: {sig11: xx, sig22: yy, sig12: xy}}
###
### A tensor target needs a mesh-output model (ISLA), whose output Mesh carries the
### whole ``(N, 3, 3)`` tensor in its point_data; the recipe reads the listed
### laboratory components off it, and the loss and the metrics keep the
### per-component metric names (``sig11_l2``), so results stay comparable with
### lanes that predicted the same components as separate scalars. Normalization is
### the dataset's per-component z-score, applied component by component: a
### diagonal affine map per lab-frame component, consistent with the loss and
### chosen to keep the loss comparable with the scalar-head lanes.
FieldType: TypeAlias = Literal["scalar", "vector"] | dict[str, Any]
FieldKind: TypeAlias = Literal["scalar", "vector", "symmetric_tensor"]

### Laboratory-frame index pairs of the six independent components of a symmetric
### rank-2 tensor ``T[..., i, j]``.
SYMMETRIC_TENSOR_COMPONENTS: dict[str, tuple[int, int]] = {
    "xx": (0, 0),
    "yy": (1, 1),
    "zz": (2, 2),
    "xy": (0, 1),
    "xz": (0, 2),
    "yz": (1, 2),
}


def field_kind(field_type: FieldType) -> FieldKind:
    """Classify a ``targets:`` value as ``scalar``, ``vector`` or ``symmetric_tensor``.

    Raises:
        ValueError: On an unknown type tag or a malformed tensor spec.
    """
    if isinstance(field_type, dict):
        kind = field_type.get("type")
        if kind != "symmetric_tensor":
            raise ValueError(
                f"Unknown mapping field type {field_type!r}. Expected "
                f"{{type: symmetric_tensor, components: {{<field>: <xx|yy|zz|xy|xz|yz>}}}}."
            )
        tensor_components(field_type)
        return "symmetric_tensor"
    if field_type in ("scalar", "vector", "symmetric_tensor"):
        ### Idempotent on an already-classified kind (``align_target_measure``
        ### is called with kinds as well as raw config values).
        return field_type
    raise ValueError(
        f"Unknown field type {field_type!r}. Expected 'scalar', 'vector' or a "
        f"symmetric_tensor mapping."
    )


def tensor_components(field_type: FieldType) -> dict[str, str]:
    """``{dataset field name: component label}`` of a symmetric-tensor target, in YAML order.

    Raises:
        ValueError: If the spec has no components, repeats a component, or names
            a label outside :data:`SYMMETRIC_TENSOR_COMPONENTS`.
    """
    comps = field_type.get("components") if isinstance(field_type, dict) else None
    if not comps or not isinstance(comps, dict):
        raise ValueError(
            f"symmetric_tensor target needs a non-empty `components` mapping "
            f"{{<field>: <xx|yy|zz|xy|xz|yz>}}; got {field_type!r}."
        )
    labels = list(comps.values())
    bad = [c for c in labels if c not in SYMMETRIC_TENSOR_COMPONENTS]
    if bad or len(set(labels)) != len(labels):
        raise ValueError(
            f"symmetric_tensor components must be distinct labels from "
            f"{tuple(SYMMETRIC_TENSOR_COMPONENTS)!r}; got {comps!r}."
        )
    return {str(k): str(v) for k, v in comps.items()}


def select_tensor_components(
    tensor: torch.Tensor, field_type: FieldType
) -> torch.Tensor:
    """The listed laboratory components of a ``(..., d, d)`` tensor, in YAML order, as ``(..., n_listed)``.

    ``d`` is 3, or 2 for a planar problem restricted to its plane (``RestrictToPlane`` + ``ISLA(n_spatial_dims=2)``,
    2026-10-03): then only the in-plane labels ``xx``, ``yy``, ``xy`` can be listed.
    """
    d = tensor.shape[-1]
    if tensor.dim() < 2 or tensor.shape[-2] != d or d not in (2, 3):
        raise ValueError(
            f"a symmetric_tensor prediction must have shape (..., d, d) with d in (2, 3), got {tuple(tensor.shape)}"
        )
    pairs = [
        SYMMETRIC_TENSOR_COMPONENTS[c] for c in tensor_components(field_type).values()
    ]
    out_of_plane = [
        c
        for c, (i, j) in zip(tensor_components(field_type).values(), pairs)
        if max(i, j) >= d
    ]
    if out_of_plane:
        raise ValueError(
            f"symmetric_tensor components {out_of_plane!r} index a coordinate the {d}-dimensional prediction does not have"
        )
    return torch.stack([tensor[..., i, j] for i, j in pairs], dim=-1)


def target_field_names(target_config: dict[str, FieldType]) -> list[str]:
    """Dataset field names a ``targets:`` block reads: tensor targets expand to their components."""
    names: list[str] = []
    for name, ftype in target_config.items():
        if field_kind(ftype) == "symmetric_tensor":
            names.extend(tensor_components(ftype))
        else:
            names.append(name)
    return names


def field_loss_channels(field_type: FieldType, n_spatial_dims: int = 3) -> int:
    """Number of components a field contributes to the loss / metrics.

    Equal to :func:`field_dim` for scalars and vectors; for a symmetric tensor it is
    the number of *listed* components (the model predicts the whole tensor, the loss
    compares the listed components), so a tensor target weighs exactly as much in the
    channel-normalized total as the same components declared as separate scalars.
    """
    if field_kind(field_type) == "symmetric_tensor":
        return len(tensor_components(field_type))
    return field_dim(field_type, n_spatial_dims)


### Allowed mixed-precision modes for the autocast context. ``"float8"`` is
### intentionally absent; `get_autocast_context` rejects it at runtime (see
### its error message for the padding rationale).
Precision: TypeAlias = Literal["float32", "float16", "bfloat16"]

### Canonical ``phase`` tags for each ``metrics.jsonl`` record, shared by
### train.py and infer.py so both entry points emit one vocabulary. Values are
### ``{split}_{granularity}`` (or a one-shot metadata tag): ``config`` /
### ``dataset`` are run metadata; ``*_step`` rows are per-unit (one per step /
### sample, as the recipe runs ``batch_size == 1``); ``*_summary`` rows are the
### reduced per-pass aggregates (``infer_forces_summary`` is surface-only).
Phase: TypeAlias = Literal[
    "config",
    "dataset",
    "train_step",
    "val_step",
    "infer_step",
    "train_summary",
    "val_summary",
    "infer_summary",
    "infer_forces_summary",
]


def set_seed(seed: int | None, rank: int = 0) -> None:
    """Pin all RNG states for reproducible training.

    When *seed* is not None, seeds Python, NumPy, and PyTorch (CPU + all
    CUDA devices) with ``seed + rank`` so that different ranks diverge
    deterministically.  When *seed* is None this function is a no-op,
    preserving the current (non-deterministic) behaviour.
    """
    if seed is None:
        return
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed % (1 << 31))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compile_model(model: torch.nn.Module, cfg: DictConfig) -> torch.nn.Module:
    """``torch.compile(model, **cfg.compile_options)`` when ``cfg.compile`` is true.

    ``compile_options`` holds the keyword arguments of ``torch.compile`` (e.g.
    ``fullgraph``, ``dynamic``, ``mode``); absent or empty means PyTorch's
    defaults. Model templates may set them (the ISLA templates do). With
    ``fullgraph``, a DistributedDataParallel model's wrapped module is compiled
    in place instead, without DDPOptimizer's graph splitting: Dynamo cannot
    trace DDP's own forward in one graph.
    """
    if not cfg.get("compile", False):
        return model
    options = cfg.get("compile_options", {})
    if options.get("fullgraph", False) and isinstance(
        model, torch.nn.parallel.DistributedDataParallel
    ):
        ### Dynamo cannot trace DDP._pre_forward under fullgraph, and DDPOptimizer's
        ### bucket splitting is itself a graph break: compile the module DDP wraps,
        ### as one graph.
        torch._dynamo.config.optimize_ddp = False
        model.module.compile(**options)
        return model
    return torch.compile(model, **options)


def build_optimizer(
    model: torch.nn.Module, cfg: DictConfig, *, compile_optimizer: bool = False
) -> torch.optim.Optimizer:
    """Build an explicit Hydra optimizer, or retain the existing Muon default.

    Explicit optimizers receive all model parameters. Their configuration must
    contain only arguments accepted by that optimizer, including any parameter
    group choices; the recipe does not silently reinterpret optimizer options.
    """
    if "_target_" not in cfg.training.optimizer:
        return build_muon_optimizer(model, cfg, compile_optimizer=compile_optimizer)
    optimizer = hydra.utils.instantiate(
        cfg.training.optimizer, params=model.parameters()
    )
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("training.optimizer must instantiate a torch optimizer")
    if compile_optimizer:
        optimizer.step = torch.compile(optimizer.step)
    return optimizer


def build_muon_optimizer(
    model: torch.nn.Module, cfg: DictConfig, *, compile_optimizer: bool = False
) -> torch.optim.Optimizer:
    """Build Muon + AdamW combined optimizer.

    Muon handles 2-D parameters (linear/attention weight matrices) while AdamW
    handles everything else (biases, layer-norm, embeddings, etc.).

    Args:
        model: The model (may be DDP-wrapped).
        cfg: Full Hydra config. Reads ``cfg.training.optimizer.*`` for lr,
            weight_decay, betas, eps, and muon_cuda_graph.
        compile_optimizer: If True, compile the optimizer step functions
            with ``torch.compile``.
    """
    base_model = model.module if hasattr(model, "module") else model
    muon_params = [p for p in base_model.parameters() if p.ndim == 2]
    other_params = [p for p in base_model.parameters() if p.ndim != 2]

    opt_cfg = cfg.training.optimizer
    lr = opt_cfg.lr
    weight_decay = opt_cfg.get("weight_decay", 1e-4)
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
    eps = opt_cfg.get("eps", 1e-8)
    cuda_graph = opt_cfg.get("muon_cuda_graph", False)

    compile_kwargs = {} if compile_optimizer else None

    if muon_params and other_params:
        return CombinedOptimizer(
            [
                Muon(
                    muon_params,
                    lr=lr,
                    weight_decay=weight_decay,
                    adjust_lr_fn="match_rms_adamw",
                    cuda_graph=cuda_graph,
                ),
                torch.optim.AdamW(
                    other_params,
                    lr=lr,
                    weight_decay=weight_decay,
                    betas=betas,
                    eps=eps,
                ),
            ],
            torch_compile_kwargs=compile_kwargs,
        )
    elif muon_params:
        opt = Muon(
            muon_params,
            lr=lr,
            weight_decay=weight_decay,
            adjust_lr_fn="match_rms_adamw",
            cuda_graph=cuda_graph,
        )
        if compile_optimizer:
            opt.step = torch.compile(opt.step)
        return opt
    else:
        opt = torch.optim.AdamW(
            other_params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps
        )
        if compile_optimizer:
            opt.step = torch.compile(opt.step)
        return opt


# ---------------------------------------------------------------------------
# Field type helpers for target configurations
# ---------------------------------------------------------------------------


def field_dim(field_type: FieldType, n_spatial_dims: int = 3) -> int:
    """Number of channels a field occupies in a flat ``(B, N, C)`` model output.

    The type tag is always lowercase by contract -- the recipe normalises
    YAML inputs at the LossCalculator / MetricCalculator boundary. Pass
    pre-lowercased strings here.

    Args:
        field_type: ``"scalar"`` or ``"vector"``. A symmetric-tensor target has
            no flat layout (it is predicted as a whole tensor by a mesh-output
            model) and raises.
        n_spatial_dims: Dimensionality of vector fields. Default 3.

    Raises:
        ValueError: If ``field_type`` is not a recognized type or is a tensor.
    """
    kind = field_kind(field_type)
    if kind == "scalar":
        return 1
    if kind == "vector":
        return n_spatial_dims
    raise ValueError(
        "a symmetric_tensor target has no flat channel layout; it needs a mesh-output "
        "model (output_type: mesh, e.g. ISLA), which returns the tensor itself"
    )


def align_scalar_shapes(
    p: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align a ``(...)`` / ``(..., 1)`` shape mismatch by squeezing one side.

    Used in scalar-field loss / metric paths where the prediction may
    arrive as ``(B, N, 1)`` (sliced from a concatenated ``(B, N, C)``
    tensor before squeeze) while the target is ``(B, N)`` (per-element
    scalar from a TensorDict), or vice versa. After alignment both
    tensors share the same shape (or were already equal-shape).
    """
    if p.ndim > t.ndim and p.shape[-1] == 1:
        p = p.squeeze(-1)
    elif t.ndim > p.ndim and t.shape[-1] == 1:
        t = t.squeeze(-1)
    return p, t


def align_target_measure(
    target_measure: torch.Tensor | None,
    reference: torch.Tensor,
    field_type: FieldType,
) -> torch.Tensor | None:
    """Validate and cast a per-query measure for one target field.

    Scalar fields have shape ``(..., N)`` and vector fields
    ``(..., N, D)``; in both cases the quadrature measure must have exactly
    one value per query, shape ``(..., N)``. Requiring an exact shape prevents
    a stale or reordered measure from silently broadcasting across a field.
    The weights must be finite and nonnegative, with a positive total per
    sample: a zero weight (a degenerate, zero-area cell) is a valid quadrature
    weight that leaves its query out of every weighted reduction.
    """
    if target_measure is None:
        return None
    field_type = field_kind(field_type)
    expected_shape = reference.shape if field_type == "scalar" else reference.shape[:-1]
    if target_measure.shape != expected_shape:
        raise ValueError(
            "target_measure must have one value per target query: expected "
            f"shape {tuple(expected_shape)} for a {field_type} field with "
            f"shape {tuple(reference.shape)}, got {tuple(target_measure.shape)}."
        )
    aligned = target_measure.to(device=reference.device, dtype=reference.dtype)
    if not torch.compiler.is_compiling():
        is_valid = aligned.numel() > 0 and bool(
            (
                (torch.isfinite(aligned) & (aligned >= 0)).all()
                & (aligned.sum(dim=-1) > 0).all()
            ).item()
        )
        if not is_valid:
            raise ValueError(
                "target_measure must contain finite, nonnegative integration "
                "weights, one per target query, with a positive total per sample"
            )
    return aligned


def normalize_target_measure(
    target_measure: torch.Tensor,
) -> torch.Tensor:
    """Remove the arbitrary uniform scale of per-query measure weights.

    Relative-error kernels apply a fixed epsilon to their reduced target
    energy.  Passing raw quadrature weights would therefore make that floor,
    and hence the reported relative error for a nearly-zero field, depend on
    the units or a uniform Horvitz--Thompson factor.  Mean-one weights retain
    the intended nonuniform measure shape while giving the epsilon the same
    interpretation as the unweighted path.
    """
    mean_measure = target_measure.mean(dim=-1, keepdim=True)
    safe_mean = torch.where(
        mean_measure > 0,
        mean_measure,
        torch.ones_like(mean_measure),
    )
    return target_measure / safe_mean


def validate_field_coverage(
    target_config: dict[str, FieldType],
    pred: TensorDict,
    target: TensorDict,
) -> None:
    """Raise ``KeyError`` if *pred* or *target* is missing any field in *target_config*.

    Shared precondition check at the top of :class:`loss.LossCalculator` and
    :class:`metrics.MetricCalculator`. The error message identifies which
    side (``pred`` vs ``target``) is missing fields so config bugs surface
    against the right tensor.
    """
    for label, source in (("pred", pred), ("target", target)):
        ### ``key in td`` resolves nested keys; ``set(td.keys())`` would not.
        missing = [name for name in target_config if as_nested_key(name) not in source]
        if missing:
            raise KeyError(f"{label} is missing target fields {sorted(missing)!r}")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def resolve_dict(cfg: DictConfig, path: str) -> dict[str, Any] | None:
    """Resolve `cfg.<path>` to a plain dict, or ``None`` if missing/empty.

    Wraps the OmegaConf incantation
    ``OmegaConf.to_container(OmegaConf.select(cfg, path, default=...), resolve=True) or None``
    that would otherwise repeat at every read site.
    """
    selected = OmegaConf.select(cfg, path, default=OmegaConf.create({}))
    container = OmegaConf.to_container(selected, resolve=True)
    return container or None


# ---------------------------------------------------------------------------
# Mixed-precision autocast
# ---------------------------------------------------------------------------


def get_autocast_context(precision: Precision):
    """Return an autocast context manager for the given precision.

    Args:
        precision: One of ``"float32"``, ``"float16"``, or ``"bfloat16"``.
            ``"float32"`` yields a no-op ``nullcontext``.

    Returns:
        An autocast context manager for the requested precision, or a
        no-op ``nullcontext`` when no casting is needed.

    Raises:
        NotImplementedError: For ``"float8"`` -- intentionally scoped out
            of this recipe; the raised message carries the padding
            rationale.
        ValueError: For any other unrecognized value (e.g. a YAML typo
            like ``"bf16"``), rather than silently running in fp32.
    """
    if precision == "float32":
        return nullcontext()
    elif precision == "float16":
        return autocast("cuda", dtype=torch.float16)
    elif precision == "bfloat16":
        return autocast("cuda", dtype=torch.bfloat16)
    elif precision == "float8":
        raise NotImplementedError(
            "precision='float8' is not supported in this recipe: TE fp8 needs "
            "every GEMM dimension (the per-sample point count and the target "
            "out_dim, e.g. 4) divisible by 16, which is not padded here. Use "
            "float32 / float16 / bfloat16. For an fp8 reference see "
            "examples/cfd/external_aerodynamics/transformer_models "
            "(update_model_params_for_fp8 / pad_input_for_fp8 / "
            "unpad_output_for_fp8) and TE's Fp8Padding / Fp8Unpadding modules."
        )
    else:
        raise ValueError(
            f"Unknown precision {precision!r}; expected one of "
            f"'float32', 'float16', 'bfloat16'."
        )


# ---------------------------------------------------------------------------
# Recursive tensor / mesh device movement
# ---------------------------------------------------------------------------

### Callable types for the recursive walker. ``LeafFn`` is the per-Tensor
### transform (mandatory); ``ContainerFn`` is the optional override
### applied to tensor-aware containers (Mesh / DomainMesh / TensorDict)
### when the default ``container.apply(leaf_fn)`` semantics aren't enough.
LeafFn = Callable[[torch.Tensor], torch.Tensor]
ContainerFn = Callable[[Any], Any]


def _recursive_apply(
    obj: Any,
    leaf_fn: LeafFn,
    *,
    container_fn: ContainerFn | None = None,
) -> Any:
    """Walk a nested structure, applying ``leaf_fn`` to every Tensor leaf.

    Tensor-aware containers (Mesh, DomainMesh, TensorDict) are routed
    through ``container_fn``. By default, ``container_fn`` delegates to
    ``container.apply(leaf_fn)``, which walks every tensor leaf in
    lock-step but does NOT touch container-level metadata
    (``TensorDict.device`` in particular stays at whatever it was).
    Override ``container_fn`` (e.g. ``lambda c: c.to(device)``) when the
    metadata change matters -- ``TensorDict`` treats ``device is None``
    as "leaves may be on any device", so device moves must go through
    ``.to(device)`` to be observable on the container.

    Plain dicts / lists / tuples are walked recursively. Note that
    ``TensorDict`` is matched in the container branch above, so it does
    NOT fall into the ``isinstance(obj, dict)`` branch (it isn't a
    ``dict`` subclass, but the explicit container check is what makes
    this work). Anything else passes through unchanged.
    """
    if container_fn is None:
        container_fn = lambda c: c.apply(leaf_fn)  # noqa: E731
    if isinstance(obj, torch.Tensor):
        return leaf_fn(obj)
    if isinstance(obj, (Mesh, DomainMesh, TensorDict)):
        return container_fn(obj)
    if isinstance(obj, dict):
        return {
            k: _recursive_apply(v, leaf_fn, container_fn=container_fn)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return type(obj)(
            _recursive_apply(v, leaf_fn, container_fn=container_fn) for v in obj
        )
    return obj


def recursive_to_device(obj: Any, device: torch.device | str) -> Any:
    """Move every tensor / Mesh / DomainMesh / TensorDict in a nested value to *device*.

    Containers go through ``.to(device)`` (not ``.apply(...)``) so that
    ``TensorDict.device`` is updated alongside the leaves; otherwise a
    later consumer reading ``td.device`` would still see ``None`` even
    though the underlying tensors have already moved.
    """
    return _recursive_apply(
        obj,
        lambda t: t.to(device),
        container_fn=lambda c: c.to(device),
    )


# ---------------------------------------------------------------------------
# JSONL logging
# ---------------------------------------------------------------------------


def make_jsonl_logger(path: str | Path) -> Callable[[dict], None]:
    """Return a logger that appends timestamped JSON records to *path*.

    Each call serialises one ``record`` dict as a JSON line, stamping it
    with a UTC ``ts``. Shared by the trainer and the inference companion
    for their per-run ``metrics.jsonl``.
    """

    def log_jsonl(record: dict) -> None:
        record["ts"] = datetime.now(timezone.utc).isoformat()
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    return log_jsonl
