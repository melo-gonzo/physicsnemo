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

import json
import math
from pathlib import Path
from typing import Any, Literal

import fsspec
import fsspec.utils
import torch

from physicsnemo.distributed import DistributedManager


class EarlyStopping:
    r"""Decide when a training run has nothing left to gain.

    Call :meth:`step` after every validation with the epoch count and the
    validation metric. It returns ``None`` while training should go on, and a
    plain-English reason once it should stop. Four rules end a run:

    1. The metric is not finite: ``"diverged: non-finite validation metric"``.
    2. The metric is worse than ``divergence_factor`` times the best for
       ``divergence_patience`` validations in a row: ``"diverged: ..."``.
    3. By ``min_epochs``, the best metric improves on the first validation by
       less than ``min_gain``: ``"never learned: ..."``. Such a run predicts
       the mean of its targets and will not recover.
    4. No validation improved on the best by more than ``min_delta`` for
       ``patience`` epochs: ``"no improvement for N epochs: ..."``.

    Patience counts epochs, not calls, so sparse validation (every few epochs)
    works with the same numbers. A new best resets the patience and the
    divergence count. Rules 3 and 4 wait until ``min_epochs``; rules 1 and 2
    do not. Improvements are relative: a metric improves on the best when
    ``(best - metric) / |best|`` exceeds ``min_delta`` (the sign flips for
    ``mode="max"``).

    With ``path`` the state is written as JSON after every call and read back
    by a new instance on the same path, so a run that restarts between epochs
    keeps its counters. Only rank 0 writes; the ``metric`` passed to
    :meth:`step` must agree across ranks, as it does for a reduced validation
    loss. Pair it with :class:`~physicsnemo.utils.CheckpointManager`, which
    keeps the best checkpoints that the stop leaves behind.

    Parameters
    ----------
    patience : int
        Epochs without an improvement on the best after which to stop.
    min_delta : float, optional
        Relative improvement a validation needs to count as a new best, as a
        fraction of the best. By default 0.0.
    mode : {"min", "max"}, optional
        Whether a lower or higher metric is better. By default ``"min"``.
    min_epochs : int, optional
        Epoch before which neither patience nor ``min_gain`` can stop the run.
        By default 0.
    min_gain : float | None, optional
        Relative improvement over the first validation the best must reach by
        ``min_epochs``, as a fraction of the first metric; the run stops as
        "never learned" otherwise. By default ``None`` (rule off).
    divergence_factor : float | None, optional
        A validation this many times worse than the best counts toward a
        divergence. Must exceed 1. By default ``None`` (rule off).
    divergence_patience : int, optional
        Consecutive divergent validations that stop the run. By default 1.
    path : Path | str | None, optional
        JSON file (local path or ``fsspec`` URI) that holds the state across
        restarts. By default ``None`` (state in memory only).

    Examples
    --------
    >>> from physicsnemo.utils import EarlyStopping
    >>> stopper = EarlyStopping(patience=4, min_delta=0.01)
    >>> for epoch, val_loss in enumerate([0.9, 0.5, 0.498, 0.6, 0.5, 0.5], start=1):
    ...     reason = stopper.step(epoch, val_loss)
    ...     if reason:
    ...         break
    >>> epoch, stopper.best_epoch, reason
    (6, 2, 'no improvement for 4 epochs: best 0.5 at epoch 2')
    """

    def __init__(
        self,
        patience: int,
        *,
        min_delta: float = 0.0,
        mode: Literal["min", "max"] = "min",
        min_epochs: int = 0,
        min_gain: float | None = None,
        divergence_factor: float | None = None,
        divergence_patience: int = 1,
        path: Path | str | None = None,
    ):
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if not 0.0 <= min_delta < 1.0:
            raise ValueError(f"min_delta must be in [0, 1), got {min_delta}")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        if min_epochs < 0:
            raise ValueError(f"min_epochs must be >= 0, got {min_epochs}")
        if min_gain is not None and not 0.0 < min_gain < 1.0:
            raise ValueError(f"min_gain must be in (0, 1) or None, got {min_gain}")
        if divergence_factor is not None and not divergence_factor > 1.0:
            raise ValueError(
                f"divergence_factor must be > 1 or None, got {divergence_factor}"
            )
        if divergence_patience < 1:
            raise ValueError(
                f"divergence_patience must be >= 1, got {divergence_patience}"
            )

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.min_epochs = min_epochs
        self.min_gain = min_gain
        self.divergence_factor = divergence_factor
        self.divergence_patience = divergence_patience
        self.path = str(path) if path is not None else None

        self.best: float | None = None
        self.best_epoch: int | None = None
        self.first: float | None = None
        self.worse = 0
        self.stopped: str | None = None
        if self.path is not None:
            self._fs = fsspec.filesystem(fsspec.utils.get_protocol(self.path))
            if self._fs.exists(self.path):
                with self._fs.open(self.path, "r") as f:
                    self.load_state_dict(json.load(f))

    def state_dict(self) -> dict[str, Any]:
        """The counters: ``best``, ``best_epoch``, ``first``, ``worse`` and ``stopped``."""
        return {
            "best": self.best,
            "best_epoch": self.best_epoch,
            "first": self.first,
            "worse": self.worse,
            "stopped": self.stopped,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the counters saved by :meth:`state_dict`."""
        self.best = None if state["best"] is None else float(state["best"])
        self.best_epoch = (
            None if state["best_epoch"] is None else int(state["best_epoch"])
        )
        self.first = None if state["first"] is None else float(state["first"])
        self.worse = int(state["worse"])
        self.stopped = state["stopped"]

    def _gain(self, old: float, new: float) -> float:
        """How much ``new`` improves on ``old``, as a fraction of ``old``; negative when worse."""
        sign = 1.0 if self.mode == "min" else -1.0
        return sign * (old - new) / (abs(old) or 1.0)

    def step(self, epoch: int, metric: float | torch.Tensor) -> str | None:
        """Record a validation and say whether to stop.

        Parameters
        ----------
        epoch : int
            Epochs trained so far, counted from 1; the number a checkpoint
            saved after this validation carries.
        metric : float | torch.Tensor
            The validation metric, the same on every rank.

        Returns
        -------
        str | None
            Why training should stop, or ``None`` to go on. Once a reason is
            returned every later call returns it again.
        """
        if self.stopped is not None:
            return self.stopped
        value = float(metric)
        reason = None
        if not math.isfinite(value):
            reason = f"diverged: non-finite validation metric {value} at epoch {epoch}"
        else:
            if self.first is None:
                self.first = value
            if self.best is None or self._gain(self.best, value) > self.min_delta:
                self.best, self.best_epoch, self.worse = value, epoch, 0
            elif (
                self.divergence_factor is not None
                and -self._gain(self.best, value) > self.divergence_factor - 1.0
            ):
                self.worse += 1
            else:
                self.worse = 0
            if (
                self.divergence_factor is not None
                and self.worse >= self.divergence_patience
            ):
                reason = (
                    f"diverged: validation metric {value:.4g} at epoch {epoch} is over "
                    f"{self.divergence_factor:g}x the best {self.best:.4g} (epoch "
                    f"{self.best_epoch}) for {self.worse} validation(s) in a row"
                )
            elif epoch >= self.min_epochs:
                gain = self._gain(self.first, self.best)
                if self.min_gain is not None and gain < self.min_gain:
                    reason = (
                        f"never learned: best {self.best:.4g} at epoch {self.best_epoch} "
                        f"improves on the first validation {self.first:.4g} by "
                        f"{gain * 100:.1f}%, under {self.min_gain * 100:g}%"
                    )
                elif epoch - self.best_epoch >= self.patience:
                    reason = (
                        f"no improvement for {epoch - self.best_epoch} epochs: "
                        f"best {self.best:.4g} at epoch {self.best_epoch}"
                    )
        self.stopped = reason
        self._save()
        return reason

    def _save(self) -> None:
        if self.path is None:
            return
        if DistributedManager.is_initialized() and DistributedManager().rank != 0:
            return
        parent = self.path.rsplit("/", 1)[0] if "/" in self.path else ""
        if parent:
            self._fs.makedirs(parent, exist_ok=True)
        tmp = f"{self.path}.tmp"
        with self._fs.open(tmp, "w") as f:
            json.dump(self.state_dict(), f)
        self._fs.mv(tmp, self.path)
