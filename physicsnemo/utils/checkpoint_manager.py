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
import re
from pathlib import Path
from typing import Any, Literal

import fsspec
import fsspec.utils
import torch

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.checkpoint import save_checkpoint

# Matches files written by ``save_checkpoint``: ``{name}.{mp_rank}.{epoch}.{ext}``
_CHECKPOINT_FILE = re.compile(r"^.+\.\d+\.(\d+)\.(?:mdlus|pt)$")


class CheckpointManager:
    r"""Bounded checkpoint retention on top of :func:`save_checkpoint`.

    Each call to :meth:`save` writes the *latest* checkpoint to ``path`` and
    deletes the previous one, so ``load_checkpoint(path)`` always resumes from
    the most recent epoch. Checkpoints whose ``metric`` ranks in the top
    ``keep_best`` are additionally written to ``path/best``; evicted ones are
    deleted. The single best checkpoint is also copied to ``path/top_model``,
    so ``load_checkpoint(path/top_model)`` loads the best model so far.
    Optionally, every ``save_every``-th epoch is kept permanently in ``path``.

    Layout::

        path/
            {Model}.0.{epoch}.mdlus, checkpoint.0.{epoch}.pt   # latest + every-N
            best/
                {Model}.0.{epoch}.mdlus, ...                   # top-k by metric
                best.json                                      # [[metric, epoch], ...]
            top_model/
                {Model}.0.{epoch}.mdlus, ...                   # copy of best[0]

    The ranking is persisted in ``best/best.json`` (best first), so a new
    manager on the same ``path`` resumes the ranking. Files are deleted only
    by rank 0; the ``metric`` passed to :meth:`save` must agree across ranks.

    Parameters
    ----------
    path : Path | str
        Checkpoint root directory (local path or ``fsspec`` URI).
    keep_best : int, optional
        Number of best checkpoints to keep. ``0`` disables best tracking.
        By default 5.
    save_every : int | None, optional
        Keep every checkpoint whose ``epoch % save_every == 0`` in ``path``
        instead of rotating it out. By default ``None`` (keep only latest).
    mode : {"min", "max"}, optional
        Whether a lower or higher ``metric`` is better. By default ``"min"``.
    best_weights_only : bool, optional
        When ``True``, best checkpoints store model weights and metadata only,
        omitting optimizer, scheduler, and scaler state. By default ``True``.

    Examples
    --------
    >>> import tempfile, os
    >>> from physicsnemo.models.mlp import FullyConnected
    >>> from physicsnemo.utils import CheckpointManager
    >>> model = FullyConnected(in_features=4, out_features=4)
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     manager = CheckpointManager(tmpdir, keep_best=2)
    ...     for epoch, val_loss in enumerate([0.5, 0.3, 0.4, 0.6]):
    ...         _ = manager.save(epoch, metric=val_loss, models=model)
    ...     manager.best, sorted(os.listdir(tmpdir))
    ([(0.3, 1), (0.4, 2)], ['FullyConnected.0.3.mdlus', 'best', 'checkpoint.0.3.pt', 'top_model'])
    """

    def __init__(
        self,
        path: Path | str,
        keep_best: int = 5,
        save_every: int | None = None,
        mode: Literal["min", "max"] = "min",
        best_weights_only: bool = True,
    ):
        if keep_best < 0:
            raise ValueError(f"keep_best must be >= 0, got {keep_best}")
        if save_every is not None and save_every < 1:
            raise ValueError(f"save_every must be >= 1, got {save_every}")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

        self.path = str(path).rstrip("/")
        self.best_path = f"{self.path}/best"
        self.top_path = f"{self.path}/top_model"
        self.keep_best = keep_best
        self.save_every = save_every
        self.mode = mode
        self.best_weights_only = best_weights_only
        self._fs = fsspec.filesystem(fsspec.utils.get_protocol(self.path))
        self._index = f"{self.best_path}/best.json"

        self.best: list[tuple[float, int]] = []
        if self._fs.exists(self._index):
            with self._fs.open(self._index, "r") as f:
                self.best = [(float(m), int(e)) for m, e in json.load(f)]

    @property
    def best_epoch(self) -> int | None:
        """Epoch of the best checkpoint so far, or ``None``."""
        return self.best[0][1] if self.best else None

    def save(
        self,
        epoch: int,
        metric: float | torch.Tensor | None = None,
        models: torch.nn.Module | list[torch.nn.Module] | None = None,
        **kwargs: Any,
    ) -> bool:
        r"""Save the latest checkpoint and update the best-k set.

        Parameters
        ----------
        epoch : int
            Epoch index, used in filenames.
        metric : float | torch.Tensor | None, optional
            Validation metric for ranking. ``None`` or non-finite values skip
            best tracking for this epoch.
        models : torch.nn.Module | list[torch.nn.Module] | None, optional
            Model(s) to save.
        **kwargs : Any
            Forwarded to :func:`save_checkpoint` (``optimizer``,
            ``scheduler``, ``scaler``, ``metadata``, ``optimizer_model``).

        Returns
        -------
        bool
            ``True`` if this epoch entered the best-k set.
        """
        save_checkpoint(self.path, models=models, epoch=epoch, **kwargs)
        is_rank0 = DistributedManager().rank == 0
        if is_rank0:
            for e, files in self._epoch_files(self.path).items():
                if e != epoch and not (self.save_every and e % self.save_every == 0):
                    self._fs.rm(files)

        if metric is None or self.keep_best == 0:
            return False
        metric = float(metric)
        if not math.isfinite(metric):
            return False
        sign = 1.0 if self.mode == "min" else -1.0
        if (
            len(self.best) == self.keep_best
            and sign * metric >= sign * self.best[-1][0]
        ):
            return False

        if self.best_weights_only:
            kwargs = {
                k: kwargs[k] for k in ("metadata", "optimizer_model") if k in kwargs
            }
        save_checkpoint(self.best_path, models=models, epoch=epoch, **kwargs)
        self.best = sorted(
            [(m, e) for m, e in self.best if e != epoch] + [(metric, epoch)],
            key=lambda x: sign * x[0],
        )
        evicted, self.best = self.best[self.keep_best :], self.best[: self.keep_best]
        if is_rank0:
            files = self._epoch_files(self.best_path)
            for _, e in evicted:
                if e in files:
                    self._fs.rm(files[e])
            with self._fs.open(self._index, "w") as f:
                json.dump(self.best, f)
            if self.best_epoch == epoch:
                if self._fs.exists(self.top_path):
                    self._fs.rm(self.top_path, recursive=True)
                self._fs.makedirs(self.top_path, exist_ok=True)
                for f in files[epoch]:
                    self._fs.copy(f, f"{self.top_path}/{f.rsplit('/', 1)[-1]}")
        return True

    def _epoch_files(self, path: str) -> dict[int, list[str]]:
        """Group checkpoint files directly under ``path`` by epoch."""
        out: dict[int, list[str]] = {}
        if not self._fs.exists(path):
            return out
        for f in self._fs.ls(path, detail=False):
            match = _CHECKPOINT_FILE.match(f.rsplit("/", 1)[-1])
            if match:
                out.setdefault(int(match.group(1)), []).append(f)
        return out
