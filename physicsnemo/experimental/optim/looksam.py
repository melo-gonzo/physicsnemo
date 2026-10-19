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

"""LookSAM and LookLayerSAM optimizers for PhysicsNeMo (experimental).

Implements the efficient SAM approximation from:
    "Towards Efficient and Scalable Sharpness-Aware Minimization"
    Liu et al., CVPR 2022.  arXiv:2203.02714
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from itertools import chain
from typing import Any, Callable, Iterable

import torch
import torch.distributed as dist
from torch.optim import Optimizer

__all__ = ["LookSAM", "LookLayerSAM"]

# Reduction key of a bucket: ``None`` for plain tensors, else the device mesh and
# the mesh dimensions over which local shards hold disjoint pieces of the tensor.
_ReduceKey = tuple[Any, tuple[int, ...]] | None

# Group keys under which state_dict() stores the step counters.
_COUNTERS = ("looksam_global_step", "looksam_sam_step")


def _local(t: torch.Tensor) -> torch.Tensor:
    """Local storage of a ``DTensor`` / ``ShardTensor``, or the tensor itself."""
    return getattr(t, "_local_tensor", t)


def _reduce_key(t: torch.Tensor) -> _ReduceKey:
    """Mesh dims to all-reduce over for global reductions of distributed ``t``.

    Local pieces are disjoint along ``Shard`` mesh dims and identical along
    ``Replicate`` dims, so global sums reduce over the ``Shard`` dims only.
    """
    placements = getattr(t, "placements", None)
    if placements is None or not hasattr(t, "_local_tensor"):
        return None
    if any(p.is_partial() for p in placements):
        raise ValueError(
            "LookSAM requires reduced gradients; got a Partial placement "
            f"{tuple(placements)}. Reduce gradients (e.g. via DDP / FSDP) first."
        )
    return t.device_mesh, tuple(i for i, p in enumerate(placements) if p.is_shard())


def _all_reduce_shards(t: torch.Tensor, key: _ReduceKey) -> None:
    """Sum ``t`` in place over the ``Shard`` mesh dims of ``key``."""
    if key is not None:
        mesh, dims = key
        for d in dims:
            dist.all_reduce(t, group=mesh.get_group(d))


def _acc(dtype: torch.dtype) -> torch.dtype:
    """Accumulation dtype for ``dtype``: at least float32."""
    return torch.promote_types(dtype, torch.float32)


@dataclass
class _Bucket:
    """Parameters sharing one (device, dtype, reduce key) and their g_v.

    ``flat_v`` holds g_v for all parameters in one buffer and ``v_views`` alias
    it in each parameter's local shape. ``chunks`` splits the buffer into
    ranges of at most :attr:`LookSAM._CHUNK` elements (see ``_chunk_plan``),
    which bound the temporaries of the chunked updates.
    """

    params: list[torch.nn.Parameter]
    key: _ReduceKey
    flat_v: torch.Tensor
    v_views: list[torch.Tensor]
    chunks: list[tuple[int, int, list[tuple[int, int, int, int]]]]


class LookSAM(Optimizer):
    r"""Efficient Sharpness-Aware Minimization (LookSAM) optimizer wrapper.

    Implements LookSAM from Liu et al., *"Towards Efficient and Scalable
    Sharpness-Aware Minimization"* (CVPR 2022, arXiv:2203.02714). LookSAM
    speeds up Sharpness-Aware Minimization (SAM) by computing the ascent
    gradient only every ``k`` steps and reusing the cached component ``g_v`` of
    the SAM gradient in between. The paper reports generalization close to
    SAM's at :math:`1/k` of SAM's extra compute over the base optimizer.

    On a full-SAM step (every ``k``-th step), with :math:`g = \nabla L(w)`:

    .. math::

        g_s = \nabla L\left(w + \rho \frac{g}{\|g\|}\right), \qquad
        g_v = g_s - \frac{g \cdot g_s}{\|g\|^2}\, g,

    and the base optimizer consumes :math:`g_s`. On the other steps it consumes

    .. math::

        g + \alpha \frac{\|g\|}{\|g_v\|}\, g_v.

    The wrapped ``base_optimizer`` (``Adam``, ``SGD``, ...) performs the
    update with the gradient LookSAM hands it.

    Parameters
    ----------
    params : iterable or None
        The parameters (or parameter groups) of ``base_optimizer``, in the
        standard :class:`torch.optim.Optimizer` convention. They must be the
        same parameters as ``base_optimizer``'s; pass ``None`` to skip the
        check.
    base_optimizer : torch.optim.Optimizer
        Inner optimizer that performs the actual parameter update.
    rho : float, default=0.05
        Radius of the L2 ascent ball (:math:`\rho` in the paper).
    alpha : float, default=0.7
        Weight of the cached ``g_v`` on fast steps. The paper reports the best
        generalization at :math:`\alpha = 0.7`.
    k : int, default=5
        Steps between full SAM updates; the first step is always one.
        ``k=1`` reduces to vanilla SAM.
    eps : float, default=1e-12
        Stabilizer added to norm denominators.

    Raises
    ------
    TypeError
        If ``base_optimizer`` is not a :class:`torch.optim.Optimizer` instance.
    ValueError
        If ``rho`` is not positive, ``alpha`` or ``eps`` is negative, ``k`` is
        not an integer of at least 1, or ``params`` differs from the
        parameters of ``base_optimizer``.

    Notes
    -----
    * **Closure-based step.** :meth:`step` takes a closure that zeroes
      gradients, computes the loss, calls ``loss.backward()``, and returns the
      loss. It is called twice on full-SAM steps and once on fast steps.
      Gradient clipping, if any, belongs inside the closure after
      ``backward()``.
    * **Hyperparameters.** ``rho``, ``alpha``, ``k``, and ``eps`` are
      attributes of the wrapper and may be changed between steps. They are
      not stored in the parameter groups, whose keys belong to the base
      optimizer (``Adam`` has its own ``eps``, ``RMSprop`` its own ``alpha``).
    * **Exact weight restore.** The weights are copied back after the ascent
      pass, also when the closure raises, so low-precision weights do not
      drift.
    * **Memory.** One parameter-sized buffer (``g_v``) between steps and two
      during a full-SAM step, in the parameters' dtype and per local shard.
      Element-wise updates of low-precision buffers run in float32 over chunks
      of at most :attr:`_CHUNK` elements, so their temporaries do not grow with
      the model.
    * **Mixed precision.** ``float64``, ``float32``, ``bfloat16``, and
      ``float16`` parameters (also mixed) and autocast are supported. A
      ``float16`` :class:`~torch.amp.GradScaler` is not: the ascent direction
      would be computed from scaled gradients.
    * **Data parallel (DDP).** Both backward passes on a full-SAM step must be
      all-reduced. Unlike vanilla SAM, where the ascent gradient is discarded
      and its sync can be skipped with ``no_sync()``, LookSAM reuses the ascent
      gradient :math:`g` inside :math:`g_v`. An unsynced ascent pass would give
      each rank a different :math:`g_v` and the ranks would diverge on the
      fast steps.
    * **Sharded parameters (FSDP2, DTensor, ShardTensor).** Parameters and
      gradients may be ``DTensor`` or
      :class:`~physicsnemo.domain_parallel.ShardTensor` instances with
      ``Shard`` and ``Replicate`` placements. Element-wise updates run on the
      local shards; global norms and dot products are all-reduced over the
      ``Shard`` mesh dimensions. Gradients with a ``Partial`` placement are
      rejected.
    * **Param groups and state.** ``self.param_groups`` is the base
      optimizer's list, so LR schedulers can take the LookSAM instance, and
      :meth:`add_param_group` adds the group to the base optimizer.
      ``self.state[p]`` holds only ``g_v``; the base optimizer's state stays
      in ``base_optimizer.state``.
    * **Checkpoints.** When the base optimizer's state dict has the standard
      ``state`` / ``param_groups`` layout, :meth:`state_dict` keeps it and adds
      ``g_v`` to each parameter's state and the step counters to each group,
      so the ``torch.distributed.checkpoint`` helpers can reshard it. Other
      base optimizers (e.g. ``CombinedOptimizer``) are nested under
      ``"base_optimizer"``. :meth:`load_state_dict` reads both layouts.
    * **Step counters.** :attr:`global_step` advances once per :meth:`step`
      call with a closure. :attr:`sam_step` counts only full-SAM updates.

    Examples
    --------
    >>> import torch
    >>> from torch.optim import Adam
    >>> from physicsnemo.experimental.optim import LookSAM
    >>> _ = torch.manual_seed(0)
    >>> model = torch.nn.Linear(4, 1)
    >>> base = Adam(model.parameters(), lr=1e-3)
    >>> opt = LookSAM(model.parameters(), base_optimizer=base, rho=0.05, k=5)
    >>> loss_fn = torch.nn.MSELoss()
    >>> x, y = torch.randn(8, 4), torch.randn(8, 1)
    >>> def closure():
    ...     opt.zero_grad()
    ...     loss = loss_fn(model(x), y)
    ...     loss.backward()
    ...     return loss
    >>> _ = opt.step(closure)
    >>> opt.global_step
    1
    """

    # Elements per chunk of the low-precision element-wise updates.
    _CHUNK = 1 << 24

    def __init__(
        self,
        params: Iterable[Any] | None,
        base_optimizer: Optimizer,
        rho: float = 0.05,
        alpha: float = 0.7,
        k: int = 5,
        eps: float = 1e-12,
    ) -> None:
        if not isinstance(base_optimizer, Optimizer):
            raise TypeError(
                "`base_optimizer` must be a torch.optim.Optimizer instance, e.g. "
                "LookSAM(params, base_optimizer=torch.optim.Adam(params, lr=1e-3)); "
                f"got {base_optimizer!r}."
            )
        self.rho, self.alpha, self.k, self.eps = rho, alpha, k, eps
        self._check_hparams()
        if params is not None:
            self._check_params(params, base_optimizer)

        # Optimizer.__init__ validates the groups; the list itself is then
        # replaced by the base optimizer's, so both share one list of groups.
        self._initializing = True
        try:
            super().__init__(base_optimizer.param_groups, defaults={})
        finally:
            self._initializing = False
        self.param_groups = base_optimizer.param_groups

        self.base_optimizer = base_optimizer
        self._global_step: int = 0
        self._sam_step: int = 0
        # Flat g_v buffers and the cached global ‖g_v‖ used by the fast steps.
        self._buckets: list[_Bucket] = []
        self._g_v_norm: torch.Tensor | None = None

    def _check_hparams(self) -> None:
        if not (math.isfinite(self.rho) and self.rho > 0.0):
            raise ValueError(f"`rho` must be positive and finite, got {self.rho}.")
        if not (math.isfinite(self.alpha) and self.alpha >= 0.0):
            raise ValueError(
                f"`alpha` must be non-negative and finite, got {self.alpha}."
            )
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k < 1:
            raise ValueError(f"`k` must be an integer of at least 1, got {self.k!r}.")
        if not (math.isfinite(self.eps) and self.eps >= 0.0):
            raise ValueError(f"`eps` must be non-negative and finite, got {self.eps}.")

    @staticmethod
    def _check_params(params: Iterable[Any], base_optimizer: Optimizer) -> None:
        given = []
        for item in params:
            if isinstance(item, dict):
                ps = item["params"]
                given.extend([ps] if isinstance(ps, torch.Tensor) else ps)
            else:
                given.append(item)
        base = [p for g in base_optimizer.param_groups for p in g["params"]]
        if {id(p) for p in given} != {id(p) for p in base}:
            raise ValueError(
                "`params` must be the parameters of `base_optimizer` "
                f"({len(base)} tensors); got {len(given)} tensors that differ. "
                "Pass None to skip this check."
            )

    @property
    def global_step(self) -> int:
        """Number of :meth:`step` calls with a closure since construction."""
        return self._global_step

    @property
    def sam_step(self) -> int:
        """Number of full-SAM updates performed."""
        return self._sam_step

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Add a parameter group to the base optimizer (and so to this wrapper)."""
        if getattr(self, "_initializing", False):
            super().add_param_group(param_group)
        else:
            self.base_optimizer.add_param_group(param_group)

    @torch.no_grad()
    def step(
        self, closure: Callable[[], torch.Tensor] | None = None
    ) -> torch.Tensor | None:
        """Perform one LookSAM step.

        Parameters
        ----------
        closure : Callable[[], torch.Tensor] or None
            Zeroes grads, evaluates the loss, calls ``loss.backward()``, and
            returns the loss. Invoked twice on SAM steps, once on fast steps.
            Without a closure only the base optimizer steps, on the current
            gradients, and a warning is raised: this exists for checkpoint
            utilities that initialize optimizer state with a zero-gradient
            ``step()``, and is not a training mode.

        Returns
        -------
        torch.Tensor or None
            Loss returned by the first closure call.

        Raises
        ------
        ValueError
            If a hyperparameter attribute was set to an invalid value, a
            gradient is sparse, or a gradient has a ``Partial`` placement.
        """
        self._check_hparams()
        if closure is None:
            warnings.warn(
                "LookSAM.step() was called without a closure, so only the base "
                "optimizer stepped. Pass a closure to train with LookSAM.",
                stacklevel=2,
            )
            if not self._buckets:
                self._zero_g_v()
            return self.base_optimizer.step()

        closure = torch.enable_grad()(closure)
        loss = closure()
        if self._global_step % self.k == 0:
            self._sam_update(closure)
            self._sam_step += 1
        elif self._buckets:
            self._fast_update()

        self.base_optimizer.step()
        self._global_step += 1
        return loss

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Return the base optimizer's state dict extended with g_v and counters."""
        base_sd = self.base_optimizer.state_dict()
        params = list(chain.from_iterable(g["params"] for g in self.param_groups))
        g_v = [self._export(p, self.state.get(p, {}).get("g_v")) for p in params]
        if not ("state" in base_sd and "param_groups" in base_sd):
            return {
                "base_optimizer": base_sd,
                "g_v": g_v,
                "global_step": self._global_step,
                "sam_step": self._sam_step,
            }
        state = {i: dict(s) for i, s in base_sd["state"].items()}
        for i, (p, v) in enumerate(zip(params, g_v)):
            # Keys other than g_v in self.state are placeholders written by
            # checkpoint loaders that pre-fill optimizer.state; they serve as
            # the template until load_state_dict hands them to the base.
            extra = {k: t for k, t in self.state.get(p, {}).items() if k != "g_v"}
            if extra or v is not None:
                s = state.setdefault(i, {})
                s.update({k: t for k, t in extra.items() if k not in s})
                if v is not None:
                    s["g_v"] = v
        counters = dict(zip(_COUNTERS, (self._global_step, self._sam_step)))
        groups = [{**g, **counters} for g in base_sd["param_groups"]]
        return {**base_sd, "state": state, "param_groups": groups}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state from a dict produced by :meth:`state_dict`.

        Also reads the nested layout written by earlier versions. Collective
        under distributed training: the cached ``‖g_v‖`` is recomputed with an
        all-reduce over sharded parameters.

        Raises
        ------
        ValueError
            If the saved ``g_v`` list does not match the number of parameters.
        """
        if "base_optimizer" in state_dict:
            base_sd = state_dict["base_optimizer"]
            g_v = state_dict["g_v"]
            if g_v and isinstance(g_v[0], list):  # one list per param group
                g_v = list(chain.from_iterable(g_v))
            steps = state_dict["global_step"], state_dict["sam_step"]
        else:
            # State keys are the ids listed in the groups' "params" (indices, or
            # parameter names from torch.distributed.checkpoint).
            state, by_id = {}, {}
            for i, s in state_dict["state"].items():
                s = dict(s)
                if (v := s.pop("g_v", None)) is not None:
                    by_id[i] = v
                state[i] = s
            groups = [
                {k: v for k, v in g.items() if k not in _COUNTERS}
                for g in state_dict["param_groups"]
            ]
            first = state_dict["param_groups"][0]
            steps = tuple(first.get(c, 0) for c in _COUNTERS)
            base_sd = {**state_dict, "state": state, "param_groups": groups}
            g_v = [by_id.get(i) for g in groups for i in g["params"]]

        self.base_optimizer.load_state_dict(base_sd)
        # The base optimizer rebuilt its groups; share its list again.
        self.param_groups = self.base_optimizer.param_groups
        self._global_step, self._sam_step = (int(s) for s in steps)

        params = list(chain.from_iterable(g["params"] for g in self.param_groups))
        if len(g_v) != len(params):
            raise ValueError(
                f"The state dict holds g_v for {len(g_v)} parameters, but this "
                f"optimizer has {len(params)}."
            )
        self._clear_g_v()
        self.state.clear()
        kept = [(p, v) for p, v in zip(params, g_v) if v is not None]
        if kept:
            self._buckets = self._buckets_from(
                [p for p, _ in kept],
                [_local(v).to(_local(p).device, _local(p).dtype) for p, v in kept],
                [_reduce_key(p) for p, _ in kept],
            )
            self._cache_g_v()

    def __getstate__(self) -> dict[str, Any]:
        # Optimizer.__getstate__ keeps only defaults, state, and param_groups.
        return self.__dict__.copy()

    # ------------------------------------------------------------------
    # Ascent hook (overridden by LookLayerSAM)
    # ------------------------------------------------------------------

    def _ascend(
        self,
        buckets: list[_Bucket],
        grads: list[tuple[torch.Tensor, list[torch.Tensor]]],
        g_sq: torch.Tensor,
    ) -> None:
        """Move w to w + rho g / ‖g‖ (global norm)."""
        scale = self.rho / (g_sq.sqrt() + self.eps)
        for b, (flat_g, _) in zip(buckets, grads):
            self._add_scaled([_local(p) for p in b.params], flat_g, b, scale)

    # ------------------------------------------------------------------
    # Update rules
    # ------------------------------------------------------------------

    def _sam_update(self, closure: Callable[[], torch.Tensor]) -> None:
        """Ascend to w + e_w, take g_s there, restore w, cache g_v, hand g_s on."""
        self._clear_g_v()
        params = [
            p
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not params:
            closure()
            return
        self._check_dense(params)

        # g in one flat buffer per bucket; the buckets' flat_v snapshots w and
        # later holds g_s and then g_v.
        buckets = self._buckets_from(
            params,
            [_local(p) for p in params],
            [_reduce_key(p.grad) for p in params],
        )
        grads = [self._flatten([_local(p.grad) for p in b.params]) for b in buckets]
        g_sq = self._global_sum(
            [
                (torch.linalg.vector_norm(g, dtype=_acc(g.dtype)).square(), b.key)
                for b, (g, _) in zip(buckets, grads)
            ]
        )

        try:
            self._ascend(buckets, grads, g_sq)
            closure()
        finally:
            for b in buckets:
                torch._foreach_copy_([_local(p) for p in b.params], b.v_views)

        # g_v = g_s - (g·g_s / ‖g‖²) g.
        dots = []
        for b, (flat_g, _) in zip(buckets, grads):
            present = [i for i, p in enumerate(b.params) if p.grad is not None]
            if len(present) < len(b.params):
                b.flat_v.zero_()
            torch._foreach_copy_(
                [b.v_views[i] for i in present],
                [_local(b.params[i].grad) for i in present],
            )
            dots.append((self._dot(b.flat_v, flat_g), b.key))
        coef = self._global_sum(dots) / (g_sq + self.eps)
        for b, (flat_g, _) in zip(buckets, grads):
            self._axpy(b.flat_v, flat_g, -coef)
        del grads

        self._buckets = buckets
        self._cache_g_v()

    def _fast_update(self) -> None:
        """g ← g + α (‖g‖ / ‖g_v‖) g_v."""
        for b in self._buckets:
            p = _local(b.params[0])
            if (p.dtype, p.device) != (b.flat_v.dtype, b.flat_v.device):
                # The model moved to another dtype or device since the last
                # SAM step; drop g_v until the next one.
                self._clear_g_v()
                return
        grads = [
            [_local(p.grad) if p.grad is not None else None for p in b.params]
            for b in self._buckets
        ]
        self._check_dense([p for b in self._buckets for p in b.params])
        partials = [
            (
                torch.stack(torch._foreach_norm(present, 2, dtype=_acc(b.flat_v.dtype)))
                .square()
                .sum(),
                b.key,
            )
            for b, gs in zip(self._buckets, grads)
            if (present := [g for g in gs if g is not None])
        ]
        if not partials:
            return
        g_sq = self._global_sum(partials)
        blend = self.alpha * g_sq.sqrt() / (self._g_v_norm + self.eps)
        for b, gs in zip(self._buckets, grads):
            self._add_scaled(gs, b.flat_v, b, blend)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cache_g_v(self) -> None:
        """Expose the g_v views as per-parameter state and cache the global ‖g_v‖."""
        for b in self._buckets:
            for p, v in zip(b.params, b.v_views):
                self.state[p]["g_v"] = v
        g_v_sq = self._global_sum(
            [
                (
                    torch.linalg.vector_norm(
                        b.flat_v, dtype=_acc(b.flat_v.dtype)
                    ).square(),
                    b.key,
                )
                for b in self._buckets
            ]
        )
        self._g_v_norm = g_v_sq.sqrt()

    def _clear_g_v(self) -> None:
        for b in self._buckets:
            for p in b.params:
                self.state[p].pop("g_v", None)
        self._buckets, self._g_v_norm = [], None

    def _zero_g_v(self) -> None:
        """Zero g_v for parameters with gradients, as a template for checkpoints."""
        params = [
            p for g in self.param_groups for p in g["params"] if p.grad is not None
        ]
        if params:
            self._buckets = self._buckets_from(
                params,
                [torch.zeros_like(_local(p)) for p in params],
                [_reduce_key(p.grad) for p in params],
            )
            self._cache_g_v()

    @staticmethod
    def _check_dense(params: list[torch.nn.Parameter]) -> None:
        if any(p.grad is not None and p.grad.is_sparse for p in params):
            raise ValueError("LookSAM does not support sparse gradients.")

    @classmethod
    def _buckets_from(cls, params, tensors, keys) -> list[_Bucket]:
        """Copy ``tensors`` into one flat buffer per (device, dtype, reduce key)."""
        groups: dict[Any, tuple[list, list]] = {}
        for p, t, key in zip(params, tensors, keys):
            ps, ts = groups.setdefault((t.device, t.dtype, key), ([], []))
            ps.append(p)
            ts.append(t)
        buckets = []
        for (_, _, key), (ps, ts) in groups.items():
            flat, views = cls._flatten(ts)
            buckets.append(_Bucket(ps, key, flat, views, cls._chunk_plan(views)))
        return buckets

    @classmethod
    def _chunk_plan(cls, views: list[torch.Tensor]) -> list[tuple[int, int, list]]:
        """Split the flat range into chunks of at most _CHUNK elements.

        Each chunk is ``(start, end, segments)`` with one segment
        ``(view index, offset in the view, length, offset in the chunk)`` per
        view it covers; a view larger than a chunk spans several chunks.
        """
        plan, segs, start, pos = [], [], 0, 0
        for i, v in enumerate(views):
            n, off = v.numel(), 0
            while off < n:
                take = min(n - off, cls._CHUNK - (pos - start))
                segs.append((i, off, take, pos - start))
                off, pos = off + take, pos + take
                if pos - start == cls._CHUNK:
                    plan.append((start, pos, segs))
                    segs, start = [], pos
        if segs:
            plan.append((start, pos, segs))
        return plan

    def _add_scaled(
        self,
        targets: list[torch.Tensor | None],
        flat: torch.Tensor,
        b: _Bucket,
        scale: torch.Tensor,
    ) -> None:
        """targets[i] += scale * view i of ``flat``, in at least float32, chunked.

        ``flat`` is laid out like ``b.flat_v``; ``None`` targets are skipped.
        Non-contiguous targets (e.g. channels-last gradients) are updated whole.
        """
        acc = _acc(flat.dtype)
        scale = scale.to(flat.device, acc)
        whole = {
            i for i, t in enumerate(targets) if t is not None and not t.is_contiguous()
        }
        if whole:
            views = self._views(flat, b.v_views)
            for i in whole:
                targets[i].add_(views[i].to(acc, copy=True).mul_(scale).to(flat.dtype))
        for start, end, segs in b.chunks:
            segs = [s for s in segs if targets[s[0]] is not None and s[0] not in whole]
            if not segs:
                continue
            tmp = flat[start:end].to(acc, copy=True).mul_(scale).to(flat.dtype)
            torch._foreach_add_(
                [targets[i].view(-1)[o : o + n] for i, o, n, _ in segs],
                [tmp[c : c + n] for _, _, n, c in segs],
            )
            del tmp  # free before the next chunk is allocated

    def _axpy(self, y: torch.Tensor, x: torch.Tensor, c: torch.Tensor) -> None:
        """y += c * x in place, in at least float32, chunked for low precision."""
        acc = _acc(y.dtype)
        c = c.to(y.device, acc)
        if y.dtype == acc:
            y.addcmul_(x, c)
            return
        for s in range(0, y.numel(), self._CHUNK):
            ys = y[s : s + self._CHUNK]
            ys.copy_(ys.to(acc).addcmul_(x[s : s + self._CHUNK].to(acc), c))

    @classmethod
    def _dot(cls, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """a·b in at least float32, upcasting low precision _CHUNK elements at a time."""
        acc = _acc(a.dtype)
        if a.dtype == acc:
            return torch.dot(a, b)
        total = torch.zeros((), device=a.device, dtype=acc)
        for s in range(0, a.numel(), cls._CHUNK):
            e = s + cls._CHUNK
            total += torch.dot(a[s:e].to(acc), b[s:e].to(acc))
        return total

    @classmethod
    def _flatten(
        cls, tensors: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Copy ``tensors`` into one flat buffer; return it and per-tensor views."""
        flat = torch.empty(
            sum(t.numel() for t in tensors),
            dtype=tensors[0].dtype,
            device=tensors[0].device,
        )
        views = cls._views(flat, tensors)
        torch._foreach_copy_(views, tensors)
        return flat, views

    @staticmethod
    def _views(flat: torch.Tensor, like: list[torch.Tensor]) -> list[torch.Tensor]:
        sizes = [t.numel() for t in like]
        return [v.view(t.shape) for v, t in zip(flat.split(sizes), like)]

    @staticmethod
    def _global_sum(partials: list[tuple[torch.Tensor, _ReduceKey]]) -> torch.Tensor:
        """Sum per-bucket local partial sums into one global value.

        Partials that share a reduce key are added first, so each distinct
        mesh costs one all-reduce per sharded mesh dimension.
        """
        home = partials[0][0].device
        dtype = partials[0][0].dtype
        for t, _ in partials[1:]:
            dtype = torch.promote_types(dtype, t.dtype)
        by_key: dict[Any, torch.Tensor] = {}
        for t, key in partials:
            t = t.to(home, dtype)
            by_key[key] = by_key[key] + t if key in by_key else t
        total = torch.zeros((), device=home, dtype=dtype)
        for key, value in by_key.items():
            _all_reduce_shards(value, key)
            total = total + value
        return total

    @staticmethod
    def _export(p: torch.nn.Parameter, buf: torch.Tensor | None):
        """Detached copy of a g_v buffer in the parameter's distributed layout."""
        if buf is None:
            return None
        buf = _local(buf).detach().clone()
        if not hasattr(p, "_local_tensor"):
            return buf
        from torch.distributed.tensor import DTensor

        if isinstance(p, DTensor):
            return DTensor.from_local(
                buf,
                p.device_mesh,
                p.placements,
                run_check=False,
                shape=p.shape,
                stride=p.stride(),
            )
        from physicsnemo.domain_parallel import ShardTensor

        if isinstance(p, ShardTensor):
            shapes = p._spec._sharding_shapes
            return ShardTensor.from_local(
                buf,
                p._spec.mesh,
                p._spec.placements,
                sharding_shapes="infer" if shapes is None else shapes,
            )
        return buf


class LookLayerSAM(LookSAM):
    r"""Look-LayerSAM: LookSAM with layer-wise scaled ascent steps.

    Implements Look-LayerSAM (Liu et al., Eq. 8 and Algorithm 3). Each
    parameter tensor :math:`\ell` is perturbed by

    .. math::

        \epsilon_\ell = \rho\, \frac{\|w_\ell\|}{\|g_\ell\|}\,
        \frac{g_\ell}{\|g\|},

    a LARS/LAMB-style trust ratio :math:`\|w_\ell\| / \|g_\ell\|` times the
    globally normalized gradient, so each tensor's perturbation is set by its
    own weight scale rather than by the share of the gradient norm it carries.
    A weight and its bias are separate tensors. Parameters with zero weights
    are not perturbed.

    All other behavior (``g_v`` caching, fast steps, distributed support) is
    inherited from :class:`LookSAM`.
    """

    def _ascend(self, buckets, grads, g_sq) -> None:
        g_norm = g_sq.sqrt()
        for b, (_, g_views) in zip(buckets, grads):
            acc = _acc(b.flat_v.dtype)
            sq = torch.stack(
                torch._foreach_norm(g_views, 2, dtype=acc)
                + torch._foreach_norm(b.v_views, 2, dtype=acc)
            ).square()
            _all_reduce_shards(sq, b.key)
            g_l, w_l = sq.sqrt().chunk(2)
            scale = (
                self.rho * w_l / ((g_l + self.eps) * (g_norm.to(sq.device) + self.eps))
            )
            # Expanded float32 scales keep the product in float32 before the
            # in-place add rounds it to the parameter dtype.
            torch._foreach_addcmul_(
                [_local(p) for p in b.params],
                g_views,
                [c.expand_as(v) for c, v in zip(scale.unbind(), g_views)],
            )
