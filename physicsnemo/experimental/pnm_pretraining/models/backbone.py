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

"""GeoPT-style pretraining wrapper around :class:`physicsnemo.models.transolver.Transolver`.

This module ports the GeoPT-General pretraining model contract onto
PhysicsNeMo's `Transolver` backbone. The wrapper composes a stock
`Transolver` (no source-side modifications) with a small `TrajectoryHead`
MLP that emits a flat per-point trajectory feature vector
:math:`(B, N, n_{\\text{steps}} \\cdot 3)` matching the
``walks_supervise`` target produced by
:class:`physicsnemo.experimental.pnm_pretraining.data.transforms.WalkSampler`.

**Data contract.** Per
``geopt-datagen-round1-plan.md`` §A Convention 1 the supervision target
is *surface-pointing* (``v(p) = c(p) - p`` rather than GeoPT's
``v_geopt = p - c(p)``); the wrapper does not re-sign the target. The
trajectory feature vector is the flat concatenation of the
``n_steps`` surface-pointing offsets, one per integration step of the
constrained walk.

**Per-walk schema.** Per ``pr2-recipe-extension-plan.md`` Q5 resolution,
each ``__getitem__`` emits *one* walk's worth of supervision: the
trajectory head's output dimension is therefore ``n_steps * 3`` (not
``n_walks * n_steps * 3``). The walk index is sampled inside
``WalkSampler``.

**Direct subclassing, no forward hook.** Per
``pr2-recipe-extension-plan.md`` D2, this wrapper does *not* use a
``forward_pre_hook`` to capture pre-output activations. Instead it
**replaces** the Transolver's output projection
(``transolver.blocks[-1].ln_mlp2``, see
``physicsnemo/models/transolver/transolver.py:271-280``) with an
``nn.Identity`` and attaches its own ``trajectory_head`` at the wrapper
level. This keeps the wrapper a clean ``physicsnemo.Module`` subclass
whose param names are::

    transolver.preprocess.*         # backbone — load on fine-tune
    transolver.blocks.{i}.*         # backbone — load on fine-tune
    trajectory_head.*               # head — drop on fine-tune

PR 2.5's
:func:`physicsnemo.utils.checkpoint.load_pretrained_backbone` consumes
this layout via ``key_remap={"transolver.": ""}`` (to strip the wrapper
prefix when targeting a plain :class:`Transolver`) plus
``exclude_layers=["trajectory_head"]`` (to drop the head). The
fine-tune run reconstructs a stock ``Transolver`` with its own ``out_dim``
and inherits only the matched-by-name backbone weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.models.transolver import Transolver

# ---------------------------------------------------------------------
# Activation registry. Mirrors the small dispatch in
# physicsnemo/models/transolver/transolver.py so the head's activation
# string accepts the same names as the backbone's `act` arg.
# ---------------------------------------------------------------------
_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


class TrajectoryHead(nn.Module):
    r"""MLP head producing a flat per-point trajectory feature vector.

    Replaces the Transolver's per-block output projection
    (``LayerNorm + Linear``) with a small multi-layer perceptron whose
    final linear projection emits a fixed feature width per point. Used
    as the supervised regression head during GeoPT-style pretraining; in
    the fine-tune stage this head is dropped and the backbone is loaded
    into a plain :class:`Transolver` with its own ``out_dim``-sized
    head (see :func:`physicsnemo.utils.checkpoint.load_pretrained_backbone`
    with ``exclude_layers=["trajectory_head"]``).

    Parameters
    ----------
    in_dim : int
        Input feature dimension (the Transolver's hidden width,
        typically ``n_hidden``).
    out_dim : int
        Output feature dimension. Pretraining uses ``n_steps * 3``
        — the flat per-point trajectory.
    hidden_dim : int, optional
        Hidden layer width. Defaults to ``256``.
    n_layers : int, optional
        Number of hidden layers. ``n_layers=2`` produces
        ``Linear → Act → Linear → Act → Linear``; the final layer is
        the output projection. Must be ``>= 1``. Defaults to ``2``.
    activation : str, optional
        Activation name. One of ``{"gelu", "relu", "silu", "tanh"}``.
        Defaults to ``"gelu"``.

    Notes
    -----
    No layer normalization or dropout are applied: the head's input is
    the post-residual hidden of the last Transolver block (which itself
    follows a LayerNormMLP residual), and pretraining doesn't benefit
    from extra regularization on this tiny head.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"TrajectoryHead requires n_layers >= 1, got {n_layers}")
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"TrajectoryHead activation must be one of {sorted(_ACTIVATIONS)}, "
                f"got {activation!r}"
            )
        act_cls = _ACTIVATIONS[activation]

        layers: list[nn.Module] = []
        current = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(current, hidden_dim))
            layers.append(act_cls())
            current = hidden_dim
        layers.append(nn.Linear(current, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Project per-point hidden features to the trajectory output.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, N, in_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, N, out_dim)``.
        """
        return self.layers(x)


@dataclass
class _MetaData(ModelMetaData):
    r"""Metadata for the GeoPT pretraining wrapper.

    Mirrors the Transolver MetaData
    (``physicsnemo/models/transolver/transolver.py:313-327``); the
    wrapper inherits the same optimization / inference / physics-informed
    flags because everything heavy-weight lives inside the wrapped
    :class:`Transolver`.
    """

    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = True
    # Inference
    onnx_cpu: bool = False
    onnx_gpu: bool = True
    onnx_runtime: bool = True
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class TransolverPretrainBackbone(Module):
    r"""GeoPT-style pretraining wrapper around :class:`Transolver`.

    Composes a stock :class:`Transolver` backbone with a
    :class:`TrajectoryHead` MLP. The Transolver's last-block output
    projection (``ln_mlp2`` in ``TransolverBlock``) is replaced with
    ``nn.Identity`` so the backbone emits its hidden state ``(B, N,
    n_hidden)``; the head then projects to ``(B, N, n_steps * 3)``.

    Subclasses :class:`physicsnemo.core.module.Module` (not plain
    :class:`torch.nn.Module`) so the wrapper inherits the
    ``.mdlus``-archive serialization machinery and is loadable by PR
    2.5's :func:`physicsnemo.utils.checkpoint.load_pretrained_backbone`.

    Parameters
    ----------
    functional_dim : int
        Forwarded to :class:`Transolver`. Width of the per-point
        functional input ``fx`` (i.e. the conditioning vector). For
        GeoPT pretraining this is ``4`` — direction (3) plus step
        length (1) — produced by the ``WalkSampler`` transform.
    embedding_dim : int
        Forwarded to :class:`Transolver`. Width of the per-point
        embedding input.  For the recipe's ``transolver_pretrain``
        template this is ``7`` — points (3) + sdf (1) +
        normals_face_barycentric (3).
    n_steps : int, optional
        Number of integration steps in the supervised constrained
        walk. The trajectory head emits ``n_steps * 3`` features per
        point. Defaults to ``3`` (matches GeoPT's released config and
        :mod:`physicsnemo.experimental.pnm_pretraining.data.builder`'s
        default).
    head_hidden_dim : int, optional
        Width of the trajectory head's hidden layers.
        Defaults to ``256``.
    head_n_layers : int, optional
        Number of hidden layers in the trajectory head.
        Defaults to ``2``.
    n_layers : int, optional
        Forwarded to :class:`Transolver`. Number of transformer blocks.
        Defaults to ``4``.
    n_hidden : int, optional
        Forwarded to :class:`Transolver`. Hidden dimension of the
        backbone. Defaults to ``256``.
    dropout : float, optional
        Forwarded to :class:`Transolver`. Defaults to ``0.0``.
    n_head : int, optional
        Forwarded to :class:`Transolver`. Number of attention heads.
        Must divide ``n_hidden``. Defaults to ``8``.
    act : str, optional
        Forwarded to :class:`Transolver`. Activation name for the
        backbone (separate from ``head_activation``). Defaults to
        ``"gelu"``.
    mlp_ratio : int, optional
        Forwarded to :class:`Transolver`. Defaults to ``4``.
    slice_num : int, optional
        Forwarded to :class:`Transolver`. Number of physics slices.
        Defaults to ``32``.
    unified_pos : bool, optional
        Forwarded to :class:`Transolver`. Defaults to ``False``.
    ref : int, optional
        Forwarded to :class:`Transolver`. Reference grid size for
        unified position encoding. Defaults to ``8``.
    structured_shape : tuple[int, ...] | None, optional
        Forwarded to :class:`Transolver`. ``None`` for unstructured
        meshes (the GeoPT default). Defaults to ``None``.
    use_te : bool, optional
        Forwarded to :class:`Transolver`. Defaults to ``False``.
    time_input : bool, optional
        Forwarded to :class:`Transolver`. Defaults to ``False``.
    plus : bool, optional
        Forwarded to :class:`Transolver`. Defaults to ``False``.
    head_activation : str, optional
        Activation name for the trajectory head's hidden layers. One of
        ``{"gelu", "relu", "silu", "tanh"}``. Defaults to ``"gelu"``.

    Forward
    -------
    embedding : torch.Tensor
        Shape ``(B, N, embedding_dim)``. Per-point geometry features
        (typically ``[points, sdf, normals_face_barycentric]``).
    fx : torch.Tensor
        Shape ``(B, N, functional_dim)``. Per-point conditioning
        vector (typically ``[directions, step_lengths]``).

    Returns
    -------
    torch.Tensor
        Shape ``(B, N, n_steps * 3)``. The flat per-point trajectory
        prediction.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.pnm_pretraining.models import (
    ...     TransolverPretrainBackbone,
    ... )
    >>> model = TransolverPretrainBackbone(
    ...     embedding_dim=7,
    ...     functional_dim=4,
    ...     n_steps=3,
    ...     n_layers=2,
    ...     n_hidden=32,
    ...     n_head=4,
    ...     slice_num=16,
    ...     use_te=False,
    ... )
    >>> embedding = torch.randn(2, 64, 7)
    >>> fx = torch.randn(2, 64, 4)
    >>> out = model(embedding=embedding, fx=fx)
    >>> out.shape
    torch.Size([2, 64, 9])

    See Also
    --------
    physicsnemo.utils.checkpoint.load_pretrained_backbone
        The PR 2.5 helper that consumes a wrapper checkpoint and loads
        only the backbone weights into a fine-tune model.
    physicsnemo.experimental.pnm_pretraining.data.transforms.WalkSampler
        The dataset-side transform that produces the per-walk
        ``directions`` / ``step_lengths`` / ``supervise`` arrays this
        wrapper consumes.
    """

    # Per `physicsnemo.Module.from_checkpoint` semantics, only args in
    # `_overridable_args` may be changed when loading a checkpoint.
    # Allow head-shape and `n_steps` overrides so a checkpoint can be
    # loaded into a wrapper with a different head if needed; backbone
    # shape must match.
    _overridable_args = {"n_steps", "head_hidden_dim", "head_n_layers"}

    def __init__(
        self,
        functional_dim: int,
        embedding_dim: int,
        n_steps: int = 3,
        head_hidden_dim: int = 256,
        head_n_layers: int = 2,
        n_layers: int = 4,
        n_hidden: int = 256,
        dropout: float = 0.0,
        n_head: int = 8,
        act: str = "gelu",
        mlp_ratio: int = 4,
        slice_num: int = 32,
        unified_pos: bool = False,
        ref: int = 8,
        structured_shape: tuple[int, ...] | None = None,
        use_te: bool = False,
        time_input: bool = False,
        plus: bool = False,
        head_activation: str = "gelu",
        # Tolerated but ignored: the recipe machinery wires `out_dim`
        # from the dataset's `targets:` block. The wrapper computes its
        # own head out_dim from `n_steps`. Accept and ignore so a model
        # YAML written without the wrapper-aware overrides still
        # composes; raise only if the user passed a non-None override
        # that disagrees with `n_steps * 3`.
        out_dim: int | None = None,
    ) -> None:
        super().__init__(meta=_MetaData())

        if n_steps < 1:
            raise ValueError(
                f"TransolverPretrainBackbone requires n_steps >= 1, got {n_steps}"
            )

        head_out_dim = n_steps * 3
        if out_dim is not None and out_dim != head_out_dim:
            raise ValueError(
                "TransolverPretrainBackbone: explicit `out_dim` "
                f"({out_dim}) disagrees with n_steps * 3 ({head_out_dim}). "
                "Either omit `out_dim` (the wrapper computes it from "
                "`n_steps`) or pass a matching value."
            )

        self.n_steps = n_steps
        self.head_out_dim = head_out_dim

        # Build the stock Transolver backbone. We pass `out_dim=n_hidden`
        # so the last-block projection lays out as
        # `LayerNorm(n_hidden) + Linear(n_hidden, n_hidden)`; we then
        # replace it with `nn.Identity` to expose the post-residual
        # hidden state at `(B, N, n_hidden)` for the trajectory head.
        # The exact `out_dim` here is irrelevant after the swap, but we
        # set it to `n_hidden` so the param shapes that *do* exist on
        # the unused linear layer match (they're not in the state_dict
        # after the swap, but using `n_hidden` is the least surprising
        # default for any future inspector).
        self.transolver = Transolver(
            functional_dim=functional_dim,
            out_dim=n_hidden,
            embedding_dim=embedding_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout=dropout,
            n_head=n_head,
            act=act,
            mlp_ratio=mlp_ratio,
            slice_num=slice_num,
            unified_pos=unified_pos,
            ref=ref,
            structured_shape=structured_shape,
            use_te=use_te,
            time_input=time_input,
            plus=plus,
        )

        # Replace the last-block output projection with `nn.Identity`.
        # Per `physicsnemo/models/transolver/transolver.py:271-280`,
        # `TransolverBlock.ln_mlp2` is `Sequential(LayerNorm, Linear)`
        # and is invoked unconditionally when `last_layer=True` (line
        # 306-307): `if self.last_layer: return self.ln_mlp2(fx)`. By
        # setting it to `nn.Identity()` the block returns `fx` (the
        # post-MLP residual) directly; that hidden state is what the
        # trajectory head consumes.
        self.transolver.blocks[-1].ln_mlp2 = nn.Identity()

        # Trajectory head at the wrapper level. Param names are
        # `trajectory_head.layers.{2*i}.weight` etc., which the PR 2.5
        # `exclude_layers=["trajectory_head"]` filter strips cleanly.
        self.trajectory_head = TrajectoryHead(
            in_dim=n_hidden,
            out_dim=head_out_dim,
            hidden_dim=head_hidden_dim,
            n_layers=head_n_layers,
            activation=head_activation,
        )

    def forward(
        self,
        embedding: torch.Tensor,
        fx: torch.Tensor,
    ) -> torch.Tensor:
        r"""Forward pass: backbone → trajectory head.

        Parameters
        ----------
        embedding : torch.Tensor
            Shape ``(B, N, embedding_dim)``.
        fx : torch.Tensor
            Shape ``(B, N, functional_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, N, n_steps * 3)``.
        """
        # Transolver.forward signature is `(fx, embedding=None, time=None)`
        # (see physicsnemo/models/transolver/transolver.py:643-648). The
        # last-block `ln_mlp2 = Identity` makes its output a hidden
        # tensor of shape `(B, N, n_hidden)` instead of `(B, N, out_dim)`.
        hidden = self.transolver(fx, embedding=embedding)
        return self.trajectory_head(hidden)
