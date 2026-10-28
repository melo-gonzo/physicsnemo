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

"""
ISLA-DTT's training step in chunks of prediction points.

:func:`chunked_forward_backward` runs the forward and backward of an ISLA-DTT
training step (an ISLA with ``interior=True``) one chunk of prediction points at
a time (``ISLA.query_chunks``), with the fewest chunks whose predicted peak
memory fits the device (:func:`_chunk_spans`). ``train.py`` takes it in place of
its ``forward_pass`` and backward for a batch where :func:`_chunked_step` says so.
"""

import logging
import math
from typing import Any

import torch
import torch.distributed as dist
from jaxtyping import Float
from loss import LossCalculator
from metrics import MetricCalculator
from output_normalize import normalize_output_to_tensordict
from tensordict import TensorDict
from utils import FieldType, Precision, get_autocast_context

### Loss types whose per-field value is a sum over the points of terms that vanish
### where the prediction equals the target, normalized by the targets alone: a
### chunk's share of the loss and of its gradient is the loss with every other
### point's prediction set to its target. The others (a root over all points:
### relative_l2, case_relative_l2) take the two-pass form.
_POINTWISE_LOSSES = ("huber", "mse", "relative_mse")

### ISLA-DTT's step over a chunk of prediction points, by precision: its peak
### reserved memory above the memory allocated once the boundary side and the walk
### have run, in GiB per million of the chunk's walk pairs and per 100,000 of its
### points, and the margin. Least squares on GB300 over DrivAerML Volume (the
### vehicle alone, the whole boundary) and AeroZoo at 200,000 tokens with the fused
### read and its fused backward, each chunk's points walked when it runs: one chunk
### of every point and chunks of a half to a quarter of them, in one pass (huber)
### and in two (relative_l2, case_relative_l2, within the one-pass rows'
### residuals), the largest point count of each dataset held out. bfloat16: 20 rows
### at 100k-600k points and 7 held out at 500k-800k, the margin the intercept (4.05
### GiB) plus the largest held-out residual (3.47 above the fit); float32 without an
### intercept, which holds out three times closer than with one: 10 rows at
### 50k-200k and 6 held out at 150k-250k, the margin the largest residual of any row
### (3.93 above the fit). Every row's peak is within its prediction. Each peak was
### measured over a few steps after empty_cache, so fragmentation over a long run
### is not in it. Refit (fit_rule.py of DTT-FAST RULE, on cost_rule.py's rows) when
### the read's or the blocks' memory changes.
_CHUNK_GIB: dict[str, tuple[float, float, float]] = {
    "bfloat16": (0.136, 23.25, 7.6),
    "float32": (0.231, 44.97, 4.0),
}

_LOGGER = logging.getLogger("training.step")
_chunks_logged: tuple[int, bool] | None = None


def _isla_dtt(model: torch.nn.Module) -> bool:
    """Whether the model (or the module DDP wraps) is ISLA-DTT: it has
    ``query_chunks`` and predicts on the interior (``interior``)."""
    core = (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )
    return hasattr(core, "query_chunks") and getattr(core, "interior", False) is True


def _chunked_step(model: torch.nn.Module, batch: dict[str, Any]) -> bool:
    """Whether this batch takes :func:`chunked_forward_backward`: the model is
    ISLA-DTT (:func:`_isla_dtt`) and the batch has prediction points."""
    if not _isla_dtt(model):
        return False
    return batch["forward_kwargs"].get("prediction_points") is not None


def _chunk_peak_gib(
    n_pairs: int, n_points: int, allocated_gib: float, precision: Precision
) -> float:
    """The predicted peak memory in GiB of ISLA-DTT's step over a chunk of
    ``n_points`` prediction points with ``n_pairs`` walk pairs, from the memory
    allocated once the boundary side has run, with the fit's margin."""
    per_m_pairs, per_100k_points, margin = _CHUNK_GIB[precision]
    return (
        allocated_gib
        + per_m_pairs * n_pairs / 1e6
        + per_100k_points * n_points / 1e5
        + margin
    )


def _chunk_spans(
    chunks: Any,
    device: torch.device,
    ddp: bool,
    pointwise: bool,
    precision: Precision,
    capacity_gib: float | None,
) -> tuple[list[tuple[int, int]], bool]:
    """The ranges of prediction points of ``chunks`` (``ISLA.query_chunks``) that
    :func:`chunked_forward_backward` runs, and whether it takes two passes over
    them: the fewest chunks of one size (the last one shorter) whose predicted peak
    (:func:`_chunk_peak_gib`) is within ``capacity_gib``, by default the device's
    free memory and the memory the caching allocator holds (on CPU, no limit). The
    chunk size comes from each chunk's share of the walk's pairs; under DDP the
    smallest over the ranks bounds every rank's chunk size and decides the passes,
    so every rank makes the same one-or-two-pass choice. Then each rank adds chunks
    while one of them, by its exact pair count, is predicted not to fit. A
    ``pointwise`` loss takes one pass; any other takes one pass when every rank's
    points fit in one chunk, else two. Raises ``torch.OutOfMemoryError`` when not
    even a chunk of one point fits. The decision is logged at its first call and
    when it changes."""
    global _chunks_logged
    n = chunks.n_points
    cum = torch.cat([chunks.pair_counts.new_zeros(1), chunks.pair_counts.cumsum(0)])
    n_pairs = int(cum[-1])
    allocated = 0.0
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 2**30
        if capacity_gib is None:
            free, _ = torch.cuda.mem_get_info(device)
            capacity_gib = (free + torch.cuda.memory_reserved(device)) / 2**30
    if capacity_gib is None:
        capacity_gib = math.inf
    empty = _chunk_peak_gib(0, 0, allocated, precision)
    per_point = (_chunk_peak_gib(n_pairs, n, allocated, precision) - empty) / n
    fit, most = (capacity_gib - empty) / per_point, n
    if ddp:
        reduced = torch.tensor([fit, -n], dtype=torch.float64, device=device)
        dist.all_reduce(reduced, op=dist.ReduceOp.MIN)
        fit, most = float(reduced[0]), -int(reduced[1])
    if fit < 1:
        raise torch.OutOfMemoryError(
            "ISLA-DTT training step: a chunk of one prediction point is predicted to"
            f" peak at {empty + per_point:.1f} GiB of {capacity_gib:.1f}"
            + (f"; {fit:.2f} points fit on the tightest rank" if ddp else "")
        )
    size = most if fit >= most else math.floor(fit)
    two = not pointwise and most > size
    ### One chunk holds every pair, as the size counts them, so only a split can
    ### add chunks here, and then the passes are already two.
    k = -(-n // size)
    while True:
        step = -(-n // k)
        spans = [(a, min(a + step, n)) for a in range(0, n, step)]
        bounds = [a for a, _ in spans] + [n]
        pairs = cum[bounds].diff()
        peak = max(
            _chunk_peak_gib(p, b - a, allocated, precision)
            for p, (a, b) in zip(pairs.tolist(), spans)
        )
        if len(spans) == 1 or peak <= capacity_gib:
            break
        if step == 1:
            raise torch.OutOfMemoryError(
                "ISLA-DTT training step: a chunk of one prediction point is"
                f" predicted to peak at {peak:.1f} GiB of {capacity_gib:.1f}"
            )
        k += 1
    if (len(spans), two) != _chunks_logged:
        _chunks_logged = (len(spans), two)
        if not ddp or dist.get_rank() == 0:
            _LOGGER.info(
                f"ISLA-DTT training step: {n} prediction points in {len(spans)} chunk"
                + ("s" if len(spans) > 1 else "")
                + (", two passes" if two else "")
                + f" (a chunk's predicted peak {peak:.1f} GiB of {capacity_gib:.1f};"
                + f" the largest chunk that fits {fit:,.0f} points"
                + (", the smallest over the ranks)" if ddp else ")")
            )
    return spans, two


def _average_gradients_(grads: list[torch.Tensor]) -> None:
    """Average ``grads`` over the ranks in place: one flat buffer, one ``all_reduce``.

    Bitwise ``g.copy_(r / world_size)`` for ``r`` in ``fused_all_reduce(grads)``,
    without its per-tensor divisions: the buffer sums in the gradients' promoted
    dtype, at least float32, and is divided once when every gradient has its
    dtype; gradients of other dtypes are divided after the cast back, as there.
    """
    flat = torch.cat([g.reshape(-1) for g in grads])
    flat = flat.to(torch.promote_types(flat.dtype, torch.float32))
    dist.all_reduce(flat)
    world_size = dist.get_world_size()
    one_dtype = all(g.dtype == flat.dtype for g in grads)
    if one_dtype:
        flat.div_(world_size)
    views = flat.split([g.numel() for g in grads])
    torch._foreach_copy_(grads, [v.view_as(g) for v, g in zip(views, grads)])
    if not one_dtype:
        for g in grads:
            g.div_(world_size)


def chunked_forward_backward(
    batch: dict[str, Any],
    model: torch.nn.Module,
    precision: Precision,
    loss_calculator: LossCalculator,
    metric_calculator: MetricCalculator,
    *,
    target_config: dict[str, FieldType],
    capacity_gib: float | None = None,
) -> tuple[Float[torch.Tensor, ""], TensorDict, TensorDict]:
    """ISLA-DTT's training step in chunks of prediction points
    (``ISLA.query_chunks``): :func:`forward_pass` and the backward of its loss, with
    the activations of one chunk of points held at a time. The loss, its fields, the
    metrics and every parameter's gradient (averaged over the ranks, as DDP
    averages them) are the unchunked step's up to summation order. The chunks are
    the fewest that fit the device (:func:`_chunk_spans`, decided after the
    boundary side and the walk, with ``capacity_gib`` its capacity, for tests). A
    loss in ``_POINTWISE_LOSSES`` takes one pass over the chunks; any other takes
    one pass over one chunk of every point, else two: the outputs without a graph,
    the loss's gradient in them, then per chunk a recompute and its backward with
    that gradient. Returns ``(loss, loss_td, metric_td)``, detached, with the
    gradients in the parameters.
    """
    ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
    core = model.module if ddp else model
    targets: TensorDict = batch["targets"].float()
    if targets.batch_dims != 1:
        raise ValueError(
            "chunked_forward_backward takes one case's targets over its prediction"
            f" points, batch size (n_points,); got {tuple(targets.batch_size)}"
        )
    measure = batch.get("target_measure")
    measure = None if measure is None else measure.float()
    with get_autocast_context(precision):
        chunks = core.query_chunks(**batch["forward_kwargs"])

    def predict(a: int, b: int) -> TensorDict:
        with get_autocast_context(precision):
            out = chunks(a, b)
        pred = normalize_output_to_tensordict(out, target_config, "mesh").float()
        return pred.apply(lambda p, t: p.reshape(t.shape), targets[a:b])

    def one_pass(spans: list[tuple[int, int]]) -> TensorDict:
        parts = []
        for a, b in spans:
            pred = predict(a, b)
            rest = targets.select(*pred.keys(include_nested=True, leaves_only=True))
            share, _ = loss_calculator(
                torch.cat([rest[:a], pred, rest[b:]]), targets, measure
            )
            share.backward()
            parts.append(pred.detach())
        return torch.cat(parts)

    def two_pass(spans: list[tuple[int, int]]) -> TensorDict:
        with torch.no_grad():
            pred = torch.cat([predict(a, b) for a, b in spans])
        pred = pred.apply(lambda t: t.requires_grad_())
        loss, _ = loss_calculator(pred, targets, measure)
        loss.backward()
        keys = list(pred.keys(include_nested=True, leaves_only=True))
        for a, b in spans:
            part = predict(a, b)
            torch.autograd.backward(
                [part[k] for k in keys], [pred[k].grad[a:b] for k in keys]
            )
        return pred.detach()

    spans, two = _chunk_spans(
        chunks,
        next(core.parameters()).device,
        ddp,
        loss_calculator.loss_type in _POINTWISE_LOSSES,
        precision,
        capacity_gib,
    )
    pred = two_pass(spans) if two else one_pass(spans)
    with torch.no_grad():
        loss, loss_td = loss_calculator(pred, targets, measure)
    chunks.backward()
    if ddp:
        ### Paired across the ranks by position: every rank reduces the same list.
        params = [p for p in core.parameters() if p.requires_grad]
        if any(p.grad is None for p in params):
            raise RuntimeError(
                "chunked_forward_backward: a trainable parameter has no gradient"
            )
        _average_gradients_([p.grad for p in params])
    with torch.no_grad():
        metric_td = metric_calculator(pred, targets, measure)
    return loss.detach(), loss_td.detach(), metric_td.detach()
