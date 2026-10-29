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

r"""Exact chunked inference for Transolver and GeoTransolver on point clouds too
large for one forward pass.

Every step that mixes points in these models is either a softmax-weighted sum
over points (the slice tokens of each block and of the context tokenizers) or a
ball query against a cloud that is fixed for the pass. A forward pass over
:math:`N` points therefore streams exactly: per block, one pass over point
chunks accumulates the slice sums, the slice attention runs once on the
finished tokens, and a second pass applies it chunk by chunk. Ball queries run
against the whole cloud, chunked over their queries only, so every query sees
the neighbors a single pass gives it. The result equals one forward pass over
all :math:`N` points up to floating-point summation order (the slice sums
accumulate in float32).

The per-point hidden states of all :math:`N` points stay resident
(:math:`N \times` hidden values); the working memory of a chunk is that of a
forward pass over ``chunk_points`` points.

Inference only. Supported: irregular point clouds, ``GALE`` attention for
GeoTransolver, ``plus=False``, no concrete dropout, one input stream, and a
model in eval mode. Anything else raises.
"""

from __future__ import annotations

import torch
from torch import nn

from physicsnemo.models.geotransolver.context_projector import ContextProjector
from physicsnemo.models.geotransolver.geotransolver import GeoTransolver
from physicsnemo.models.transolver.transolver import Transolver, TransolverBlock
from physicsnemo.nn.module.gale import GALE, _mix_self_and_cross
from physicsnemo.nn.module.physics_attention import PhysicsAttentionIrregularMesh


def _ranges(n: int, chunk: int) -> list[slice]:
    return [slice(s, min(s + chunk, n)) for s in range(0, n, chunk)]


def _weights_and_features(module: nn.Module, x: torch.Tensor):
    r"""Slice weights :math:`(B, n, H, S)` and features :math:`(B, n, H, D)` of
    a chunk, as ``_compute_slices_from_projections`` forms them (``plus=False``)."""
    x_mid, fx_mid = module.project_input_onto_slices(x)
    projections = module.in_project_slice(x_mid)
    temperature = torch.clamp(module.temperature, min=0.5, max=5).to(projections.dtype)
    weights = nn.functional.softmax(projections / temperature, dim=-1)
    return weights.to(projections.dtype), fx_mid


class _SliceSums:
    r"""Running :math:`\sum_n w_n x_n` and :math:`\sum_n w_n` of one slice
    aggregation, in float32."""

    def __init__(self) -> None:
        self.num = self.den = None
        self.dtype = None

    def add(self, weights: torch.Tensor, features: torch.Tensor) -> None:
        w = weights.float()
        ### Autocast would run the einsum, and so the sums, in reduced precision.
        with torch.autocast(device_type=w.device.type, enabled=False):
            num = torch.einsum("bnhs,bnhd->bhsd", w, features.float())
            den = w.sum(1)
        if self.num is None:
            self.num, self.den = num, den
        else:
            self.num += num
            self.den += den
        self.dtype = features.dtype

    def tokens(self) -> torch.Tensor:
        return (self.num / (self.den + 1e-2)[..., None]).to(self.dtype)


def _tokenize(tokenizer: nn.Module, chunks) -> torch.Tensor:
    r"""A ``ContextProjector``'s slice tokens over a cloud given chunk by chunk."""
    sums = _SliceSums()
    for x in chunks:
        sums.add(*_weights_and_features(tokenizer, x))
    return sums.tokens()


def _stream_block(
    block: nn.Module,
    h: torch.Tensor,
    ranges: list[slice],
    context: torch.Tensor | None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    r"""One ``GALEBlock`` or ``TransolverBlock`` over the hidden states ``h``,
    updated in place. A last ``TransolverBlock`` writes its head into ``out``."""
    attn = block.Attn
    sums = _SliceSums()
    for r in ranges:
        sums.add(*_weights_and_features(attn, block.ln_1(h[:, r])))
    tokens = sums.tokens()
    if attn.use_te:
        out_tokens = attn._compute_slice_attention_te(tokens)
    else:
        out_tokens = attn._compute_slice_attention_sdpa(tokens)
    if context is not None:
        cross = attn.compute_slice_attention_cross([tokens], context)[0]
        out_tokens = _mix_self_and_cross(
            out_tokens,
            cross,
            attn.state_mixing_mode,
            state_mixing=getattr(attn, "state_mixing", None),
            concat_project=getattr(attn, "concat_project", None),
        )
    head = isinstance(block, TransolverBlock) and block.last_layer
    for r in ranges:
        x = h[:, r]
        weights, _ = _weights_and_features(attn, block.ln_1(x))
        y = attn._project_attention_outputs(out_tokens, weights) + x
        y = block.ln_mlp1(y) + y
        if head:
            y = block.ln_mlp2(y)
            if out is None:
                out = y.new_empty(h.shape[0], h.shape[1], y.shape[-1])
            out[:, r] = y
        else:
            h[:, r] = y
    return out


def _single_stream(x, name: str) -> torch.Tensor | None:
    if x is None or isinstance(x, torch.Tensor):
        return x
    if len(x) != 1:
        raise ValueError(f"streamed inference supports one {name} stream, got {len(x)}")
    return x[0]


def _require(condition: bool, what: str) -> None:
    if not condition:
        raise NotImplementedError(f"streamed inference does not support {what}")


def _check_attention(attn: nn.Module, kind: type) -> None:
    _require(type(attn) is kind, f"attention {type(attn).__name__}")
    _require(not attn.plus, "plus=True")


def _geotransolver(
    model: GeoTransolver,
    chunk_points: int,
    local_embedding,
    local_positions=None,
    global_embedding: torch.Tensor | None = None,
    geometry: torch.Tensor | None = None,
    time: torch.Tensor | None = None,
) -> torch.Tensor:
    _require(time is None, "a time input")
    _require(model.structured_shape is None, "structured inputs")
    for block in model.blocks:
        _check_attention(block.Attn, GALE)
        _require(
            block.attn_dropout is None and block.ffn_dropout is None,
            "concrete dropout",
        )
    local_embedding = _single_stream(local_embedding, "local_embedding")
    local_positions = _single_stream(local_positions, "local_positions")
    builder = model.context_builder
    tokenizers = [builder.geometry_tokenizer, builder.global_tokenizer]
    if builder.local_extractors is not None:
        _require(len(builder.local_extractors) == 1, "several input streams")
        tokenizers += list(builder.local_extractors[0].tokenizers)
    for tokenizer in tokenizers:
        if tokenizer is not None:
            _check_attention(tokenizer, ContextProjector)
            _require(tokenizer.output_dropout is None, "concrete dropout")

    batch, n = local_embedding.shape[:2]
    ranges = _ranges(n, chunk_points)
    use_local = builder.local_extractors is not None and geometry is not None
    if use_local and local_positions is None:
        raise ValueError("Local positions are required if local features are enabled.")

    # Context, in build_context's order: the ball-query tokens of each scale,
    # then the geometry tokens, then the global tokens.
    context_parts = []
    if use_local:
        ball_geometry = (
            geometry
            if builder.local_geometry_dims is None
            else geometry[..., : builder.local_geometry_dims]
        )
        geometry_ranges = _ranges(geometry.shape[1], chunk_points)
        extractor = builder.local_extractors[0]
        for processor, tokenizer in zip(extractor.processors, extractor.tokenizers):
            # Each geometry point's neighbors among all the query points.
            context_parts.append(
                _tokenize(
                    tokenizer,
                    (
                        processor(local_positions, ball_geometry[:, r])
                        for r in geometry_ranges
                    ),
                )
            )
    if builder.geometry_tokenizer is not None and geometry is not None:
        context_parts.append(
            _tokenize(
                builder.geometry_tokenizer,
                (geometry[:, r] for r in _ranges(geometry.shape[1], chunk_points)),
            )
        )
    if builder.global_tokenizer is not None and global_embedding is not None:
        context_parts.append(builder.global_tokenizer(global_embedding))
    context = torch.cat(context_parts, dim=-1) if context_parts else None

    h = None
    for r in ranges:
        x = model.preprocess[0](local_embedding[:, r])
        if model.include_local_features and use_local:
            # Each query point's neighbors among all the geometry points.
            local = extractor.extract_local_features(
                local_positions[:, r], ball_geometry
            )
            x = torch.cat([x, local], dim=-1)
        if h is None:
            h = x.new_empty(batch, n, x.shape[-1])
        h[:, r] = x

    for block in model.blocks:
        _stream_block(block, h, ranges, context)

    out = None
    for r in ranges:
        y = model.ln_mlp_out[0](h[:, r])
        if out is None:
            out = y.new_empty(batch, n, y.shape[-1])
        out[:, r] = y
    return out


def _transolver(
    model: Transolver,
    chunk_points: int,
    fx: torch.Tensor,
    embedding: torch.Tensor | None = None,
    time: torch.Tensor | None = None,
) -> torch.Tensor:
    _require(time is None, "a time input")
    _require(model.structured_shape is None, "structured inputs")
    _require(not model.unified_pos, "unified_pos=True")
    if embedding is None:
        raise ValueError("Embedding is required for unstructured data")
    for block in model.blocks:
        _check_attention(block.Attn, PhysicsAttentionIrregularMesh)

    batch, n = fx.shape[:2]
    ranges = _ranges(n, chunk_points)
    h = None
    for r in ranges:
        x = model.preprocess(torch.cat((embedding[:, r], fx[:, r]), -1))
        if h is None:
            h = x.new_empty(batch, n, x.shape[-1])
        h[:, r] = x
    out = None
    for block in model.blocks:
        out = _stream_block(block, h, ranges, None, out)
    return h if out is None else out


def streamed_forward(
    model: nn.Module, chunk_points: int, **forward_kwargs
) -> torch.Tensor:
    r"""The model's forward pass over every point, computed ``chunk_points`` at a time.

    Parameters
    ----------
    model : GeoTransolver | Transolver
        A model in eval mode.
    chunk_points : int
        Points per chunk. Peak memory is the resident hidden states of all
        points plus a forward pass over this many.
    **forward_kwargs
        The model's forward arguments, as for ``model(**forward_kwargs)``.

    Returns
    -------
    torch.Tensor
        The output for every point, :math:`(B, N, C_{out})`.
    """
    if model.training:
        raise ValueError("streamed inference needs the model in eval mode")
    if chunk_points < 1:
        raise ValueError(f"chunk_points must be positive, got {chunk_points}")
    with torch.no_grad():
        if isinstance(model, GeoTransolver):
            return _geotransolver(model, chunk_points, **forward_kwargs)
        if isinstance(model, Transolver):
            return _transolver(model, chunk_points, **forward_kwargs)
    raise TypeError(
        f"streamed inference supports GeoTransolver and Transolver, not {type(model).__name__}"
    )
