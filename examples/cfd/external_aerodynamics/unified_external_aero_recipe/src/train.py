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
Unified External Aerodynamics Training Script

Trains a point-cloud model (GeoTransolver, Transolver, etc.) on surface
or volume fields using the mesh datapipe infrastructure.

Usage::

    # Single-GPU
    python src/train.py

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=N src/train.py

    # I/O benchmark: iterate dataloaders without model logic
    python src/train.py benchmark_io=true profile=true
    python src/train.py benchmark_io=true +training.benchmark_max_steps=20
"""

import os
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Any, Literal, NamedTuple

import hydra
import torch
import torch.distributed as dist
from datasets import build_dataloaders, load_dataset_config
from isla_chunked import (
    _POINTWISE_LOSSES,
    _chunked_step,
    _isla_dtt,
    chunked_forward_backward,
)
from jaxtyping import Float
from loss import LossCalculator
from metrics import MetricCalculator, resolve_metrics
from omegaconf import DictConfig, OmegaConf
from output_normalize import (
    IOType,
    check_model_output_stats,
    model_dataset_constants,
    model_output_stats,
    normalize_output_to_tensordict,
    require_output_type,
)
from tabulate import tabulate
from tensordict import TensorDict
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from training_state import ModelEMA, check_grad_clip_norm, optimizer_update
from utils import (
    FieldType,
    Phase,
    Precision,
    build_optimizer,
    compile_model,
    get_autocast_context,
    make_jsonl_logger,
    recursive_to_device,
    resolve_dict,
    set_seed,
)

from physicsnemo import datapipes  # noqa: F401 - registers ${dp:...} resolver
from physicsnemo.datapipes import DataLoader
from physicsnemo.distributed import DistributedManager, fused_all_reduce
from physicsnemo.mesh import MESH_FIELD_ASSOCIATIONS, DomainMesh, Mesh
from physicsnemo.utils import CheckpointManager, load_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils.profiling import Profiler, profile

### When `cfg.profile` is set, every train / val epoch breaks out of its
### batch loop after this many steps. Keeps profiling traces short enough
### to be useful without changing the rest of the training contract.
_PROFILE_MAX_STEPS = 10


### ---------------------------------------------------------------------------
### Config
### ---------------------------------------------------------------------------


def _flatten_config(
    d: dict[str, Any], parent: str = "", sep: str = "."
) -> dict[str, str]:
    """Recursively flatten a nested dict into dot-separated key/value pairs."""
    items: dict[str, str] = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            items.update(_flatten_config(v, key, sep))
        else:
            items[key] = str(v)
    return items


### ---------------------------------------------------------------------------
### Aggregation
### ---------------------------------------------------------------------------


def _reduce_and_average(
    loss_sum: Float[torch.Tensor, ""],
    losses_td: TensorDict | None,
    metrics_td: TensorDict | None,
    n_samples: int,
    *,
    device: torch.device | str,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Collapse rank-local loss/metric *sums* into a global mean-of-means.

    Under DDP each rank only sees its own shard, so the numbers we log are
    meaningless until reduced across ranks. This divides the rank-local sums
    (``loss_sum`` plus the 0-D ``losses_td`` / ``metrics_td`` leaves) by the
    local sample count, then averages those per-rank means across ranks with
    one fused ``AVG`` ``all_reduce``
    (:func:`physicsnemo.distributed.fused_all_reduce`). Under even shards
    (train ``drop_last=True``, val pads) that equals the true global mean and
    keeps the integer count out of the float reduction buffer. It is
    granularity-neutral, mirrors the inference-side ``infer._allreduce_sums``,
    and is called once per epoch, with ``n_samples == n_local`` and the running
    epoch sums, for the dataset-wide summary (per-step values go through
    :func:`_start_step_readback`).

    Args:
        loss_sum: Rank-local sum of scalar losses over the ``n_samples`` being
            collapsed, as a 0-D on-device tensor (the running epoch loss),
            not a mean.
        losses_td: Rank-local per-field loss sum: a 0-D (``batch_size=[]``)
            ``TensorDict`` whose leaves are summed scalar losses, one per loss
            term. ``None`` is the "zero samples" sentinel (see Returns).
        metrics_td: The matching per-field metric-sum accumulator, with the
            same ``None`` sentinel. Seeded in lock-step with ``losses_td``, so
            the two are ``None`` together or populated together.
        n_samples: Number of samples this rank contributed to ``loss_sum`` and
            the accumulators (``n_local``, equal to the step count because
            the recipe runs ``batch_size == 1``).
        device: The rank's collective/compute device (``dist_manager.device``),
            where the ``all_reduce`` runs.

    Returns:
        A ``(avg_loss, avg_losses, avg_metrics)`` tuple of Python floats:
        ``avg_loss`` is the global mean loss, ``avg_losses`` is
        ``{loss_name: mean}``, and ``avg_metrics`` is ``{metric_name: mean}``.
        The dict keys and their order are taken from ``losses_td`` /
        ``metrics_td``. On the ``None`` sentinel it returns
        ``(loss_sum.item() / max(n_samples, 1), {}, {})`` without entering the
        collective.

    Notes:
        Every rank packs the same leaves in the same order (all ranks share
        one ``target_config``), so the collective matches up across ranks.
        Single-process skips the reduction, leaving single-GPU logs unchanged.
    """
    if losses_td is None or metrics_td is None:
        return loss_sum.item() / max(n_samples, 1), {}, {}
    ### Divide by the local sample count first, then AVG across ranks: a
    ### mean-of-means equal to the global mean under even shards, with no
    ### integer count entering the float reduction buffer.
    n = max(n_samples, 1)
    bundle = TensorDict(
        {
            "loss": loss_sum / n,
            "losses": losses_td / n,
            "metrics": metrics_td / n,
        },
    )
    ### Pull the reduced bundle host-side once; .item() off the CPU copy is then
    ### a free index, with no per-leaf device sync (AVG is a no-op single-process).
    reduced = fused_all_reduce(bundle, op=dist.ReduceOp.AVG, device=device).cpu()
    return (
        reduced["loss"].item(),
        {key: value.item() for key, value in reduced["losses"].items()},
        {key: value.item() for key, value in reduced["metrics"].items()},
    )


def _start_step_readback(
    loss: Float[torch.Tensor, ""],
    losses_td: TensorDict,
    metrics_td: TensorDict,
    *,
    device: torch.device | str,
) -> tuple[TensorDict, torch.cuda.Event | None]:
    """Start the all-rank mean of one step's values and its copy to the host.

    The copy is ``non_blocking``, so no step waits for the device. Read the
    returned host ``TensorDict`` only once the returned event (``None`` off
    CUDA, where the copy is synchronous) has completed.
    """
    bundle = TensorDict({"loss": loss, "losses": losses_td, "metrics": metrics_td})
    reduced = fused_all_reduce(bundle, op=dist.ReduceOp.AVG, device=device)
    if torch.device(device).type != "cuda":
        return reduced, None
    reduced, event = reduced.to("cpu", non_blocking=True), torch.cuda.Event()
    event.record()
    return reduced, event


class _PendingStep(NamedTuple):
    """One step's record, logged once its host copy has landed. ``span`` is
    the step's iteration time in seconds (off CUDA), or the pair of CUDA
    timing events around it, recorded before the step's readback starts."""

    i: int
    step_dt: float
    mem_gb: float
    lr: float | None
    span: float | tuple[torch.cuda.Event, torch.cuda.Event]
    host_values: TensorDict
    landed: torch.cuda.Event | None


### ---------------------------------------------------------------------------
### Logging
### ---------------------------------------------------------------------------


def _log_to_tensorboard(
    writer: SummaryWriter | None,
    values: Mapping[str, float | Float[torch.Tensor, ""]],
    tag_prefix: str,
    global_step: int,
) -> None:
    """Write a flat ``{name: scalar}`` mapping to TensorBoard under ``tag_prefix/<name>``.

    No-op when *writer* is ``None``. The caller chooses *tag_prefix* to
    namespace the entries (e.g. ``"epoch"`` vs ``"iteration/metrics"``).
    """
    if writer is None:
        return
    for k, v in values.items():
        writer.add_scalar(f"{tag_prefix}/{k}", v, global_step=global_step)


### ---------------------------------------------------------------------------
### Forward pass
### ---------------------------------------------------------------------------


def forward_pass(
    batch: dict[str, Any],
    model: torch.nn.Module,
    precision: Precision,
    loss_calculator: LossCalculator,
    metric_calculator: MetricCalculator,
    *,
    output_type: IOType,
    target_config: dict[str, FieldType],
) -> tuple[Float[torch.Tensor, ""], TensorDict, TensorDict]:
    """Run a forward pass + loss + metrics on one collated batch.

    Args:
        batch: Collated ``"forward_kwargs"`` and ``"targets"``, plus an
            optional ``"target_measure"``. ``"targets"`` has batch_size
            ``[N]`` (mesh-input mode) or ``[1, N]`` (tensor-input mode).
        model: Model whose ``forward`` accepts the resolved
            ``forward_kwargs`` as keyword arguments.
        precision: One of ``"float32"``, ``"float16"``, or ``"bfloat16"``.
            Wraps the forward call in the matching ``torch.autocast``
            context; inputs keep their native dtype.
        loss_calculator: Returns ``(loss, loss_td)`` from
            ``(pred, target)`` TensorDicts.
        metric_calculator: Returns a per-field metrics ``TensorDict``.
        output_type: ``"mesh"`` or ``"tensors"``; controls how the model
            output is unpacked into a TensorDict.
        target_config: ``{name: "scalar"|"vector"}``; used to split tensor
            outputs and validate Mesh outputs.

    Returns:
        ``(loss, loss_td, metric_td)``. The two TensorDicts are kept
        separate so callers can route them to different log namespaces
        without textual key inspection. Per-field values are returned
        as **detached, on-device 0-D tensors** (no ``.item()`` sync
        here): the caller decides when to sync, so the loss kernels can
        overlap with backward instead of being serialised by an
        in-line D2H transfer.
    """
    forward_kwargs = batch["forward_kwargs"]
    targets: TensorDict = batch["targets"]
    target_measure: torch.Tensor | None = batch.get("target_measure")

    ### Inputs keep their native dtype; autocast handles model-internal precision.
    with get_autocast_context(precision):
        output = model(**forward_kwargs)

    pred_td = normalize_output_to_tensordict(output, target_config, output_type)

    ### Loss runs in float32 to avoid bf16 precision loss in the reduction.
    pred_f32 = pred_td.float()
    target_f32 = targets.float()
    target_measure_f32 = None if target_measure is None else target_measure.float()

    loss, loss_td = loss_calculator(pred_f32, target_f32, target_measure_f32)
    with torch.no_grad():
        metric_td = metric_calculator(pred_f32, target_f32, target_measure_f32)
    ### Detach (don't sync) the per-field TDs so the caller controls when
    ### a D2H copy happens; running ``.item()`` here would serialise the
    ### forward kernels against the host. ``TensorDict.detach()`` walks
    ### every leaf in one fast-apply pass.
    return loss, loss_td.detach(), metric_td.detach()


### ---------------------------------------------------------------------------
### Epoch loops
### ---------------------------------------------------------------------------


def _run_epoch(
    dataloader: DataLoader,
    model: torch.nn.Module,
    loss_calculator: LossCalculator,
    metric_calculator: MetricCalculator,
    logger: Any,
    epoch: int,
    cfg: DictConfig,
    dist_manager: DistributedManager,
    *,
    mode: Literal["train", "val"],
    output_type: IOType,
    target_config: dict[str, FieldType],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: GradScaler | None = None,
    ema: ModelEMA | None = None,
    writer: SummaryWriter | None = None,
    log_jsonl: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[float, dict[str, float]]:
    """Run one training-or-validation epoch.

    Train and val share the same per-batch loop (``forward_pass`` +
    metric accumulation + per-step console log + per-epoch summary).
    Train mode additionally runs the backward / optimizer / scheduler
    step and emits per-step TensorBoard + JSONL entries (``phase: "step"``);
    val mode wraps the loop in ``torch.no_grad()``, skips TensorBoard
    per-step logging, and emits a lighter-weight JSONL record per step
    (``phase: "val_step"``) carrying ``epoch``, ``val_step``, ``loss``,
    ``step_time_s`` and ``iter_time_s``. Epoch summaries carry
    ``epoch_seconds`` and, on GPU, ``peak_memory_gib``: the epoch's peak
    allocated memory, the maximum over ranks.

    The two per-step times (console ``Step:`` and ``Iter:``):

    - ``step_time_s``: host time from the end of the previous step to just
      before this step's readback starts; it misses the device work still
      queued and the wait for the other ranks.
    - ``iter_time_s``: the true time per iteration, between CUDA events
      recorded at that point of consecutive steps (the first from the start
      of the loop), read once the step's readback has landed, so it adds no
      host sync. The events time the device's stream, so the iteration times
      sum to the loop's time and include any wait for the loader. Off CUDA,
      where every step's readback is synchronous, it is host time.

    Args:
        mode: ``"train"`` or ``"val"``. ``"train"`` requires *optimizer*
            and *scheduler*; ``"val"`` ignores them.
        scaler: GradScaler for fp16 (train mode only).
        writer: TensorBoard writer for the matching split. Per-epoch
            metrics are written to it on rank 0; per-step metrics are
            written only in train mode.
        log_jsonl: Optional ``record -> None`` callback for JSONL logs.
            See ``forward_pass`` and ``main`` docstrings for the rest of
            the parameters.
    """
    is_train = mode == "train"
    if is_train and (optimizer is None or scheduler is None):
        raise ValueError("train mode requires both optimizer and scheduler")
    if is_train:
        model.train()
    else:
        model.eval()

    grad_ctx = nullcontext() if is_train else torch.no_grad()
    log_prefix = "Epoch" if is_train else "Val Epoch"
    is_rank0 = dist_manager.rank == 0

    ### All three accumulators live on-device, so epoch sums build up with no
    ### per-step host sync (per-step values are read back without waiting, below).
    ### total_loss is float64 for accumulation precision, downcast to float32 at
    ### the epoch reduce. None = not yet seeded; n_local is this rank's local
    ### sample count.
    total_loss = torch.zeros((), dtype=torch.float64, device=dist_manager.device)
    total_losses_td: TensorDict | None = None
    total_metrics_td: TensorDict | None = None
    precision = getattr(cfg, "precision", "float32")
    n_local = 0
    num_steps = len(dataloader)
    ### One entry per step, logged in order once its readback has landed;
    ### the epoch's end logs the rest (see _start_step_readback).
    pending: deque[_PendingStep] = deque()

    def log_step(step: _PendingStep) -> None:
        """Log one step's landed record, whose values are global all-rank means."""
        i, step_dt, mem_gb, lr, span, host_values, _ = step
        ### The span's events precede the readback's on the stream, so they
        ### are complete once it has landed and the wait below never runs.
        if isinstance(span, float):
            iter_dt = span
        else:
            if not span[1].query():
                span[1].synchronize()
            iter_dt = span[0].elapsed_time(span[1]) / 1e3
        ### Landed host copies, so .item() here does not wait for the GPU.
        step_loss = host_values["loss"].item()
        step_losses = {k: v.item() for k, v in host_values["losses"].items()}
        step_metrics = {k: v.item() for k, v in host_values["metrics"].items()}
        ### Train mode includes Mem in the per-step line; val drops it
        ### because the no_grad path is the lowest-noise place to look.
        mem_str = f" Mem: {mem_gb:.2f}GB" if is_train else ""
        logger.info(
            f"{log_prefix} {epoch} [{i + 1}/{num_steps}] "
            f"Loss: {step_loss:.6f} "
            f"Step: {step_dt:.3f}s "
            f"Iter: {iter_dt:.3f}s"
            f"{mem_str}"
        )

        ### Per-step TensorBoard is train-only (val_writer is epoch-only);
        ### per-step JSONL is written in both modes so downstream tooling gets
        ### val step-times directly. The logged values are global all-rank
        ### means (rank 0 is only the writer).
        if is_rank0:
            if is_train:
                global_step = epoch * num_steps + i
                if writer is not None:
                    ### Tags: iteration/loss/... and iteration/metrics/...
                    _log_to_tensorboard(writer, step_losses, "iteration", global_step)
                    _log_to_tensorboard(
                        writer, step_metrics, "iteration/metrics", global_step
                    )
                    for tag, value in (
                        ("lr", lr),
                        ("performance/mem_gb", mem_gb),
                        ("performance/step_time_s", step_dt),
                        ("performance/iter_time_s", iter_dt),
                    ):
                        writer.add_scalar(f"iteration/{tag}", value, global_step)
                if log_jsonl is not None:
                    log_jsonl(
                        {
                            "phase": "train_step",
                            "global_step": global_step,
                            "loss": step_loss,
                            "mem_gb": mem_gb,
                            "step_time_s": step_dt,
                            "iter_time_s": iter_dt,
                            **step_losses,
                            **step_metrics,
                        }
                    )
            elif log_jsonl is not None:
                ### Val records name their epoch and omit mem_gb (lowest-noise path).
                log_jsonl(
                    {
                        "phase": "val_step",
                        "epoch": epoch,
                        "val_step": i,
                        "loss": step_loss,
                        "step_time_s": step_dt,
                        "iter_time_s": iter_dt,
                        **step_losses,
                        **step_metrics,
                    }
                )

    ### Each summary's peak memory measures only its own epoch.
    on_cuda = torch.device(dist_manager.device).type == "cuda"
    if on_cuda:
        torch.cuda.reset_peak_memory_stats(dist_manager.device)

    def iteration_mark():
        """A point on the device's timeline (a CUDA timing event, recorded),
        or off CUDA the host clock."""
        if not on_cuda:
            return time.perf_counter()
        mark = torch.cuda.Event(enable_timing=True)
        mark.record()
        return mark

    def log_landed() -> None:
        """Log, in order, every pending step whose copy has landed."""
        while pending and (pending[0].landed is None or pending[0].landed.query()):
            log_step(pending.popleft())

    epoch_t0 = time.perf_counter()
    with grad_ctx, ExitStack() as on_exit:
        ### If a step raises, log the steps whose copies have landed, without waiting.
        on_exit.callback(log_landed)
        step_t0 = time.perf_counter()
        iter_mark = iteration_mark()
        for i, batch in enumerate(dataloader):
            batch = recursive_to_device(batch, dist_manager.device)
            chunked = is_train and scaler is None and _chunked_step(model, batch)
            if chunked:
                optimizer.zero_grad()
                loss, losses, metrics = chunked_forward_backward(
                    batch,
                    model,
                    precision,
                    loss_calculator,
                    metric_calculator,
                    target_config=target_config,
                )
            else:
                loss, losses, metrics = forward_pass(
                    batch,
                    model,
                    precision,
                    loss_calculator,
                    metric_calculator,
                    output_type=output_type,
                    target_config=target_config,
                )

            if is_train:
                if chunked:
                    ### chunked_forward_backward already accumulated the gradients.
                    grad_clip_norm = check_grad_clip_norm(
                        cfg.training.get("grad_clip_norm")
                    )
                    if grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), grad_clip_norm, error_if_nonfinite=True
                        )
                    optimizer.step()
                    updated = True
                    if ema is not None:
                        ema.update()
                else:
                    updated = optimizer_update(
                        loss,
                        optimizer,
                        model.parameters(),
                        scaler=scaler if precision == "float16" else None,
                        grad_clip_norm=cfg.training.get("grad_clip_norm"),
                        ema=ema,
                    )
                if (
                    updated
                    and cfg.training.get("scheduler_update_mode", "epoch") == "step"
                ):
                    scheduler.step()

            ### Accumulate on-device with no sync. First iteration clones
            ### so subsequent in-place ``add_`` calls don't alias the
            ### per-step TDs; both accumulators are seeded in lock-step
            ### (the joint ``is None`` check exists to satisfy the type
            ### checker, which can't see the invariant from per-variable
            ### narrowing).
            if total_losses_td is None or total_metrics_td is None:
                total_losses_td = losses.clone()
                total_metrics_td = metrics.clone()
            else:
                total_losses_td.add_(losses)
                total_metrics_td.add_(metrics)
            n_local += 1

            ### Detached scalar loss: accumulate the epoch sum on-device (no
            ### host sync) and keep it for this step's readback below.
            loss_det = loss.detach()
            total_loss += loss_det

            step_dt = time.perf_counter() - step_t0
            mem_gb = (
                torch.cuda.memory_reserved() / 1024**3
                if torch.cuda.is_available()
                else 0
            )

            ### Start this step's all-rank readback without waiting, then log
            ### every step whose copy has landed, in order. The readback is a
            ### collective, so every rank calls it, outside the rank-0 logging
            ### gate; equal per-rank step counts keep it deadlock-free.
            lr = scheduler.get_last_lr()[0] if is_train else None
            mark = iteration_mark()
            span = mark - iter_mark if not on_cuda else (iter_mark, mark)
            iter_mark = mark
            readback = _start_step_readback(
                loss_det, losses, metrics, device=dist_manager.device
            )
            pending.append(_PendingStep(i, step_dt, mem_gb, lr, span, *readback))
            log_landed()

            if cfg.profile and i >= _PROFILE_MAX_STEPS:
                break
            step_t0 = time.perf_counter()

        ### The epoch's end waits for the copies still in flight.
        while pending:
            step = pending.popleft()
            if step.landed is not None:
                step.landed.synchronize()
            log_step(step)

    epoch_dt = time.perf_counter() - epoch_t0
    n = max(n_local, 1)
    ### Reduce the epoch sums + sample count across ranks once so logged
    ### loss/metrics are global averages (not rank-0's shard) under DDP; `n`
    ### above stays local for the per-rank step-rate line. Downcast the float64
    ### loss accumulator to float32 so every reduced leaf shares one dtype.
    avg_loss, avg_losses, avg_metrics = _reduce_and_average(
        total_loss.float(),
        total_losses_td,
        total_metrics_td,
        n_local,
        device=dist_manager.device,
    )

    logger.info(
        f"Epoch {epoch} {mode} done in {epoch_dt:.1f}s "
        f"({n_local} steps, {epoch_dt / n:.3f}s/step avg)"
    )

    ### Peak device memory of the epoch, the maximum over ranks.
    peak_gib = None
    if on_cuda:
        peak = torch.tensor(
            [torch.cuda.max_memory_allocated(dist_manager.device) / 2**30],
            device=dist_manager.device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        peak_gib = float(peak.item())

    if is_rank0:
        _log_to_tensorboard(writer, avg_losses, "epoch", epoch)
        _log_to_tensorboard(writer, avg_metrics, "epoch/metrics", epoch)
        if log_jsonl is not None:
            summary_phase: Phase = "train_summary" if is_train else "val_summary"
            log_jsonl(
                {
                    "phase": summary_phase,
                    "epoch": epoch,
                    "loss": avg_loss,
                    **avg_losses,
                    **avg_metrics,
                    "epoch_seconds": epoch_dt,
                    **({"peak_memory_gib": peak_gib} if peak_gib is not None else {}),
                }
            )

    return avg_loss, {**avg_losses, **avg_metrics}


@profile
def train_epoch(
    dataloader: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_calculator: LossCalculator,
    metric_calculator: MetricCalculator,
    logger: Any,
    epoch: int,
    cfg: DictConfig,
    dist_manager: DistributedManager,
    scaler: GradScaler | None = None,
    *,
    output_type: IOType,
    target_config: dict[str, FieldType],
    train_writer: SummaryWriter | None = None,
    ema: ModelEMA | None = None,
    log_jsonl: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[float, dict[str, float]]:
    """Run one training epoch (delegates to :func:`_run_epoch` in train mode)."""
    return _run_epoch(
        dataloader,
        model,
        loss_calculator,
        metric_calculator,
        logger,
        epoch,
        cfg,
        dist_manager,
        mode="train",
        output_type=output_type,
        target_config=target_config,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        ema=ema,
        writer=train_writer,
        log_jsonl=log_jsonl,
    )


@profile
def val_epoch(
    dataloader: DataLoader,
    model: torch.nn.Module,
    loss_calculator: LossCalculator,
    metric_calculator: MetricCalculator,
    logger: Any,
    epoch: int,
    cfg: DictConfig,
    dist_manager: DistributedManager,
    *,
    output_type: IOType,
    target_config: dict[str, FieldType],
    val_writer: SummaryWriter | None = None,
    log_jsonl: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[float, dict[str, float]]:
    """Run one validation epoch (delegates to :func:`_run_epoch` in val mode)."""
    return _run_epoch(
        dataloader,
        model,
        loss_calculator,
        metric_calculator,
        logger,
        epoch,
        cfg,
        dist_manager,
        mode="val",
        output_type=output_type,
        target_config=target_config,
        writer=val_writer,
        log_jsonl=log_jsonl,
    )


### ---------------------------------------------------------------------------
### I/O benchmarking
### ---------------------------------------------------------------------------


def _walk_batch_for_logging(
    value: Any, prefix: str = ""
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(dotted_name, Tensor)`` pairs from a batch (nested dicts / TensorDicts of tensors / Mesh).

    The TensorDict branch delegates the recursion to ``TD.flatten_keys('.')``
    rather than driving it from Python via ``.items()`` -- a TD's own
    flattening produces dotted leaf paths in one call. The plain ``dict``
    branch keeps the manual visitor because dicts may contain mixed
    Tensor / Mesh / nested-dict values that need the full recursion.
    """
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, TensorDict):
        for key, leaf in value.flatten_keys(".").items():
            sub = f"{prefix}.{key}" if prefix else key
            yield sub, leaf
    elif isinstance(value, dict):
        for k, v in value.items():
            sub = f"{prefix}.{k}" if prefix else str(k)
            yield from _walk_batch_for_logging(v, sub)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            sub = f"{prefix}[{i}]" if prefix else f"[{i}]"
            yield from _walk_batch_for_logging(v, sub)
    elif isinstance(value, DomainMesh):
        ### Recurse into interior, boundaries, and domain-level global_data
        ### so I/O benchmarks see every leaf the model would actually
        ### consume (point_data targets, boundary cell_data inputs, etc).
        yield from _walk_batch_for_logging(value.interior, f"{prefix}.interior")
        for bname in value.boundary_names:
            yield from _walk_batch_for_logging(
                value.boundaries[bname], f"{prefix}.boundaries.{bname}"
            )
        if value.global_data.keys():
            yield from _walk_batch_for_logging(
                value.global_data, f"{prefix}.global_data"
            )
    elif isinstance(value, Mesh):
        ### Mesh-level inputs: emit geometry tensors + every per-element /
        ### per-vertex / per-sample field. Each *_data attribute is itself
        ### a TensorDict, so the TD branch above handles dotted leaf paths.
        yield (f"{prefix}.points", value.points)
        if value.n_cells > 0:
            yield (f"{prefix}.cells", value.cells)
        for section in MESH_FIELD_ASSOCIATIONS:
            td = getattr(value, section)
            if td.keys():
                yield from _walk_batch_for_logging(td, f"{prefix}.{section}")


@profile
def benchmark_io_epoch(
    dataloader: DataLoader,
    label: str,
    logger: Any,
    max_steps: int | None = None,
) -> None:
    """Iterate a dataloader without any model logic and report I/O timing.

    Args:
        dataloader: Dataloader to benchmark.
        label: Human-readable label for logging (e.g. ``"train"`` or
            ``"val"``).
        logger: Logger for console output.
        max_steps: Stop after this many batches. ``None`` means exhaust
            the loader.
    """
    import statistics

    num_steps = len(dataloader)
    times: list[float] = []

    step_t0 = time.perf_counter()
    for i, batch in enumerate(dataloader):
        dt = time.perf_counter() - step_t0
        times.append(dt)

        mem_gb = (
            torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
        )

        named_tensors = list(_walk_batch_for_logging(batch))
        shapes = "  ".join(f"{name}:{tuple(t.shape)}" for name, t in named_tensors)
        logger.info(
            f"  [{label}] [{i + 1}/{num_steps}] "
            f"dt={dt:.4f}s  Mem={mem_gb:.2f}GB  {shapes}"
        )
        for name, t in named_tensors:
            v_flat = t.float() if t.is_floating_point() else t.to(torch.float32)
            logger.info(
                f"    {name:30s}  "
                f"min={v_flat.min().item(): .6e}  "
                f"mean={v_flat.mean().item(): .6e}  "
                f"std={v_flat.std().item(): .6e}  "
                f"max={v_flat.max().item(): .6e}"
            )

        if max_steps is not None and i + 1 >= max_steps:
            break
        step_t0 = time.perf_counter()

    if not times:
        logger.info(f"  [{label}] empty dataloader")
        return

    total = sum(times)
    mean = statistics.mean(times)
    med = statistics.median(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    p95 = sorted(times)[int(len(times) * 0.95)] if len(times) > 1 else times[0]

    logger.info(
        f"  [{label}] {len(times)} batches in {total:.2f}s  "
        f"mean={mean:.4f}s  median={med:.4f}s  std={std:.4f}s  p95={p95:.4f}s  "
        f"throughput={len(times) / total:.2f} batches/sec"
    )


def _save_atomic(obj: Any, path: str) -> None:
    """``torch.save`` to a temporary name, then rename: a killed process leaves the old file or the new one."""
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


### ---------------------------------------------------------------------------
### Driver
### ---------------------------------------------------------------------------


@profile
def main(cfg: DictConfig) -> None:
    """Run the full training loop, or I/O-only benchmark when ``benchmark_io=true``.

    Orchestrates the complete training workflow:

    1. Initialise distributed training and TensorBoard/JSONL logging.
    2. Build train/val dataloaders and extract pipeline transforms.
    3. If ``cfg.benchmark_io`` is true, iterate dataloaders to measure
       I/O throughput and return early (no model, no optimizer).
    4. Otherwise, instantiate the model, optimizer, and run the normal
       train/val epoch loop with checkpointing.

    Args:
        cfg: Hydra config containing ``model``, ``training``, ``dataset``,
            ``data``, ``output_dir``, ``run_id``, ``precision``,
            ``compile``, ``profile``, ``benchmark_io``, ``logging``, and
            related keys.
    """

    started = time.monotonic()
    DistributedManager.initialize()
    dist_manager = DistributedManager()
    device = dist_manager.device
    is_rank0 = dist_manager.rank == 0
    logger = RankZeroLoggingWrapper(PythonLogger(name="training"), dist_manager)

    seed = cfg.training.get("seed", None)
    set_seed(seed, rank=dist_manager.rank)
    logger.info(f"Random seed: {seed} (rank offset: {dist_manager.rank})")

    checkpoint_dir = getattr(cfg, "checkpoint_dir", None) or cfg.output_dir

    # -- Logging setup (rank 0 only) ----------------------------------------------
    train_writer = None
    val_writer = None
    log_jsonl = None
    run_dir = os.path.join(cfg.output_dir, cfg.run_id)
    if is_rank0:
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        train_writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb", "train"))
        val_writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb", "val"))
        log_jsonl = make_jsonl_logger(os.path.join(run_dir, "metrics.jsonl"))

    train_loader, val_loader, normalizer, dataset_info = build_dataloaders(cfg)
    target_config: dict[str, FieldType] = dataset_info["targets"]
    ### `metrics_list` is derived later from cfg.metrics (recipe-side);
    ### build_dataloaders no longer ships a "metrics" key in dataset_info.

    ### Log the resolved config AFTER build_dataloaders() because that's
    ### where `cfg.out_dim` is auto-derived from the chosen dataset's
    ### `targets:` block; resolving earlier would fail on the model
    ### template's `out_dim: ${out_dim}` interpolation.
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    logger.info(f"Train samples: {len(train_loader.sampler)}")
    logger.info(f"Val samples: {len(val_loader.sampler)}")
    logger.info(f"Targets (from dataset YAML): {target_config}")

    # -- Log dataset metadata (rank 0) --------------------------------------------
    if is_rank0 and log_jsonl is not None:
        ### Use len(sampler) so manifest mode (where train and val share
        ### one underlying dataset) reports the actual per-split count,
        ### not the always-identical len(dataset). PyTorch always assigns
        ### a sampler (a default SequentialSampler when none is passed),
        ### so len(loader.sampler) is always defined.
        log_jsonl(
            {
                "phase": "dataset",
                "train_samples": len(train_loader.sampler),
                "val_samples": len(val_loader.sampler),
                "dataset_size": len(train_loader.dataset),
                "targets": target_config,
            }
        )

    # -- I/O benchmark mode: iterate dataloaders, skip model entirely -----------
    if cfg.get("benchmark_io", False):
        num_epochs = cfg.training.num_epochs
        max_steps = cfg.training.get("benchmark_max_steps", None)
        logger.info(
            f"benchmark_io=True  — benchmarking dataloader I/O only "
            f"({num_epochs} epoch(s), max_steps={max_steps})"
        )
        with torch.no_grad(), Profiler():
            for epoch in range(num_epochs):
                logger.info(f"--- Epoch {epoch + 1}/{num_epochs} ---")
                train_loader.set_epoch(epoch)
                benchmark_io_epoch(train_loader, "train", logger, max_steps=max_steps)
                benchmark_io_epoch(val_loader, "val", logger, max_steps=max_steps)
        logger.info("benchmark_io complete!")
        if is_rank0:
            if train_writer is not None:
                train_writer.close()
            if val_writer is not None:
                val_writer.close()
        return

    # -- Normal training path ---------------------------------------------------
    ### A model that standardizes its own outputs (ISLA's scales: output_stats) takes the
    ### dataset's NormalizeMeshFields stats, so its outputs and the targets are one space,
    ### and the constants it leaves to the dataset (its split's isla_constants).
    ds_cfg = load_dataset_config(
        Path(__file__).resolve().parent.parent / "datasets" / f"{cfg.dataset}.yaml"
    )
    model = hydra.utils.instantiate(
        cfg.model,
        _convert_="partial",
        **model_output_stats(cfg.model, normalizer),
        **model_dataset_constants(cfg.model, ds_cfg),
    )
    check_model_output_stats(model, normalizer)
    logger.info(f"Model: {model.__class__.__name__}")
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {num_params:,}")

    model.to(device)
    ema_model = model  # Keep stable parameter names before DDP / compile wrappers.

    if dist_manager.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist_manager.local_rank],
            output_device=device,
        )

    if normalizer is not None:
        norm_summary = ", ".join(
            f"{k}({v['type']})" for k, v in normalizer.stats.items()
        )
        logger.info(f"Normalization: {norm_summary}")

    optimizer = build_optimizer(model, cfg, compile_optimizer=cfg.compile)
    logger.info(f"Optimizer: {optimizer}")
    scheduler = hydra.utils.instantiate(cfg.training.scheduler, optimizer=optimizer)

    precision = cfg.precision
    scaler = GradScaler() if precision == "float16" else None

    # -- Log full config + model params (rank 0) ---------------------------------
    if is_rank0:
        flat_cfg = _flatten_config(
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
        )
        if log_jsonl is not None:
            log_jsonl(
                {
                    "phase": "config",
                    "model": model.__class__.__name__,
                    "num_parameters": num_params,
                    "params": flat_cfg,
                }
            )

        # Save the full resolved config
        resolved_yaml = OmegaConf.to_yaml(cfg, resolve=True)
        config_artifact_path = os.path.join(run_dir, "resolved_config.yaml")
        with open(config_artifact_path, "w") as f:
            f.write(resolved_yaml)

    ### `target_config` was loaded from the chosen dataset YAML's
    ### `targets:` block by `build_dataloaders`.  The metrics list is
    ### now recipe-side: train.yaml's `cfg.metrics` is the canonical
    ### source. Falls back to the recipe's default set when unset.
    metrics_list = resolve_metrics(cfg)

    field_weights = resolve_dict(cfg, "training.field_weights")

    metric_calculator = MetricCalculator(
        target_config=target_config,
        metrics=metrics_list,
    )
    loss_type = cfg.training.get("loss_type", "huber")
    ### case_relative_l2 measures each case's error against its targets in the dataset's units: it reads
    ### the standardization's mean / std (a field without stats has offset 0).
    target_offsets = (
        {
            name: torch.as_tensor(s["mean"]) / torch.as_tensor(s["std"])
            for name, s in normalizer.stats.items()
            if name in target_config
        }
        if loss_type == "case_relative_l2" and normalizer is not None
        else None
    )
    loss_calculator = LossCalculator(
        target_config=target_config,
        loss_type=loss_type,
        field_weights=field_weights,
        target_offsets=target_offsets,
    )
    output_type = require_output_type(cfg)
    logger.info(f"Loss: {loss_calculator}")
    logger.info(f"Metrics: {metric_calculator}")
    if _isla_dtt(model):
        logger.info(
            "ISLA-DTT training step: the fewest chunks of prediction points whose"
            " predicted peak memory fits the device on every rank"
            + (
                ""
                if loss_type in _POINTWISE_LOSSES
                else f", in two passes when more than one ({loss_type} is not"
                " pointwise)"
            )
        )
        if cfg.compile:
            logger.info(
                "ISLA-DTT training step: compile=true does not compile the chunked"
                " step (query_chunks runs eagerly; compile_blocks still applies)"
            )
    logger.info(
        f"Model contract: input_type={cfg.input_type}, output_type={output_type}"
    )

    ckpt_args = {
        "path": os.path.join(checkpoint_dir, cfg.run_id, "checkpoints"),
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": scaler,
        "models": model,
    }
    ### Retention: the latest checkpoint (the one a resume loads), every save_interval-th epoch, and the
    ### keep_best checkpoints with the lowest validation loss in best/ (the best also in top_model/).
    ckpt_manager = CheckpointManager(
        ckpt_args["path"],
        keep_best=int(cfg.training.keep_best),
        save_every=int(cfg.training.save_interval),
        mode="min",
    )
    checkpoint_metadata: dict[str, Any] = {}
    loaded_epoch = load_checkpoint(
        device=device, metadata_dict=checkpoint_metadata, **ckpt_args
    )
    ema_decay = cfg.training.get("ema_decay")
    ema = ModelEMA(ema_model, ema_decay) if ema_decay is not None else None
    if loaded_epoch and ema is not None:
        if "training_ema" not in checkpoint_metadata:
            raise ValueError(
                "Cannot resume EMA training from a checkpoint without EMA state"
            )
        ema.load_state_dict(checkpoint_metadata["training_ema"])
    elif loaded_epoch and "training_ema" in checkpoint_metadata:
        raise ValueError(
            "Checkpoint contains EMA state; restore training.ema_decay to resume"
        )

    ### EMA weights rotate like the run's own checkpoints: the latest and every save_interval-th.
    ema_manager = (
        CheckpointManager(
            os.path.join(ckpt_args["path"], "ema"),
            keep_best=0,
            save_every=int(cfg.training.save_interval),
        )
        if ema is not None
        else None
    )

    model = compile_model(model, cfg)

    num_epochs = cfg.training.num_epochs
    ### The job's time budget: after an epoch's saves, stop cleanly when the time used plus 1.1 times the longest
    ### epoch so far would pass it, so a time-limited job ends between epochs rather than in a save.
    budget = cfg.training.get("stop_after_seconds")
    if budget is not None and not float(budget) > 0:
        raise ValueError(
            f"training.stop_after_seconds must be a positive number of seconds or null, got {budget!r}"
        )
    longest_epoch = 0.0
    stopped = False
    logger.info(f"Starting training for {num_epochs} epochs...")

    # Unless profiling is enabled, this is a null context:
    with Profiler():
        for epoch in range(loaded_epoch, num_epochs):
            epoch_started = time.monotonic()
            logger.info(f"--- Epoch {epoch + 1}/{num_epochs} ---")
            train_loader.set_epoch(epoch)

            train_loss, train_metrics = train_epoch(
                train_loader,
                model,
                optimizer,
                scheduler,
                loss_calculator,
                metric_calculator,
                logger,
                epoch,
                cfg,
                dist_manager,
                scaler,
                output_type=output_type,
                target_config=target_config,
                train_writer=train_writer,
                ema=ema,
                log_jsonl=log_jsonl,
            )

            # Validation cadence (default every epoch, and always the last
            # epoch). Training and the scheduler do not depend on it, so a
            # sparser cadence leaves training bit-identical. The best-k
            # ranking and top_model/ rank only the validated epochs, so the
            # cadence decides which epochs can be the "best" checkpoint.
            val_interval = int(cfg.training.get("val_interval", 1))
            run_val = (
                val_interval <= 1
                or epoch % val_interval == 0
                or epoch == num_epochs - 1
            )
            if run_val:
                val_loss, val_metrics = val_epoch(
                    val_loader,
                    model,
                    loss_calculator,
                    metric_calculator,
                    logger,
                    epoch,
                    cfg,
                    dist_manager,
                    output_type=output_type,
                    target_config=target_config,
                    val_writer=val_writer,
                    log_jsonl=log_jsonl,
                )
            else:
                val_loss, val_metrics = float("nan"), {}

            if ema is not None and run_val:
                logger.info(
                    "Evaluating EMA weights (raw validation remains separately logged)"
                )

                def log_ema(record):
                    if log_jsonl is not None:
                        log_jsonl({**record, "phase": "ema_" + record["phase"]})

                with ema.average_parameters():
                    val_epoch(
                        val_loader,
                        model,
                        loss_calculator,
                        metric_calculator,
                        logger,
                        epoch,
                        cfg,
                        dist_manager,
                        output_type=output_type,
                        target_config=target_config,
                        log_jsonl=log_ema,
                    )

            if is_rank0 and run_val:
                all_keys = list(dict.fromkeys(list(train_metrics) + list(val_metrics)))

                rows = [
                    [
                        k,
                        f"{train_metrics.get(k, float('nan')):.6f}",
                        f"{val_metrics.get(k, float('nan')):.6f}",
                    ]
                    for k in all_keys
                ]

                table = tabulate(
                    rows, headers=["Metric", "Train", "Val"], tablefmt="pretty"
                )
                logger.info(
                    f"\nEpoch [{epoch}/{cfg.training.num_epochs}] "
                    f"Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}\n"
                    f"{table}\n"
                )

            ### Step first so a checkpoint holds the schedule its resume uses.
            if cfg.training.get("scheduler_update_mode", "epoch") == "epoch":
                scheduler.step()

            ### Every epoch: the latest checkpoint replaces the previous one, so a crash loses at most the
            ### epoch in flight; validated epochs also enter the best-k ranking by validation loss.
            if is_rank0:
                metadata = (
                    {"training_ema": ema.state_dict()} if ema is not None else None
                )
                ckpt_manager.save(
                    epoch + 1,
                    metric=val_loss if run_val else None,
                    **{k: v for k, v in ckpt_args.items() if k != "path"},
                    metadata=metadata,
                )
                if ema_manager is not None:
                    with ema.average_parameters():
                        ema_manager.save(epoch + 1, models=ema_model)
                    if normalizer is not None:
                        _save_atomic(
                            normalizer.stats,
                            os.path.join(ema_manager.path, "norm_stats.pt"),
                        )
                if normalizer is not None:
                    _save_atomic(
                        normalizer.stats,
                        os.path.join(ckpt_args["path"], "norm_stats.pt"),
                    )

            if budget is not None and epoch + 1 < num_epochs:
                now = time.monotonic()
                longest_epoch = max(longest_epoch, now - epoch_started)
                stop = torch.tensor(
                    float(now - started + 1.1 * longest_epoch > float(budget)),
                    device=device,
                )
                ### Rank 0's clock decides (it also does the saves), so every rank stops at the same epoch.
                if dist_manager.world_size > 1:
                    dist.broadcast(stop, src=0)
                if bool(stop):
                    logger.info(
                        f"Time used {now - started:.0f} s, longest epoch {longest_epoch:.0f} s, "
                        f"budget {float(budget):.0f} s (training.stop_after_seconds)"
                    )
                    logger.info(f"STOP time budget after epoch {epoch + 1}")
                    stopped = True
                    break

    if is_rank0:
        if train_writer is not None:
            train_writer.close()
        if val_writer is not None:
            val_writer.close()

    if not stopped:
        logger.info("Training completed!")


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="train",
)
def launch(cfg: DictConfig) -> None:
    """Hydra entry point: configure profiling and delegate to :func:`main`.

    Args:
        cfg: Hydra-composed config (override with ``--config-name``).
            When ``cfg.profile`` is truthy, torch profiling is enabled.
    """
    profiler = Profiler()
    if cfg.profile:
        profiler.enable("torch")
    profiler.initialize()
    main(cfg)
    profiler.finalize()


if __name__ == "__main__":
    launch()
