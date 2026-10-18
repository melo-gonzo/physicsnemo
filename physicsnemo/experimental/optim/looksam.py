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

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.optim import Optimizer

__all__ = ["LookSAM", "LookLayerSAM"]

# Reduction key of a bucket: ``None`` for plain tensors, else the device mesh and
# the mesh dimensions over which local shards hold disjoint pieces of the tensor.
_ReduceKey = tuple[Any, tuple[int, ...]] | None


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


@dataclass
class _Bucket:
    """Parameters sharing one (device, dtype, reduce key) with flat buffers.

    During a SAM step ``flat_g`` holds the ascent gradient g (or the ascent
    step e_w, a multiple of g) and ``flat_w`` the weights w; ``flat_w`` then
    becomes ``flat_v``, which carries g_s and finally g_v. The ``*_views``
    lists alias the flat buffers in each parameter's local shape.
    """

    params: list[torch.nn.Parameter]
    key: _ReduceKey
    flat_v: torch.Tensor
    v_views: list[torch.Tensor]
    flat_g: torch.Tensor | None = None
    g_views: list[torch.Tensor] | None = None
    flat_w: torch.Tensor | None = None
    w_views: list[torch.Tensor] | None = None


class LookSAM(Optimizer):
    r"""Efficient Sharpness-Aware Minimization (LookSAM) optimizer wrapper.

    Implements LookSAM from Liu et al., *"Towards Efficient and Scalable
    Sharpness-Aware Minimization"* (CVPR 2022, arXiv:2203.02714). LookSAM
    accelerates Sharpness-Aware Minimization (SAM) by computing the expensive
    inner ascent gradient only every ``k`` steps and reusing the cached
    "flat-region" component ``g_v`` on intermediate steps. This recovers most
    of SAM's generalization benefit at roughly :math:`1/k` the additional
    compute over the base optimizer.

    On a full-SAM step (every ``k``-th step), with :math:`g = \nabla L(w)`:

    .. math::

        g_s = \nabla L\left(w + \rho \frac{g}{\|g\|}\right), \qquad
        g_v = g_s - \frac{g \cdot g_s}{\|g\|^2}\, g,

    and the base optimizer consumes :math:`g_s`. On the other steps it consumes

    .. math::

        g + \alpha \frac{\|g\|}{\|g_v\|}\, g_v.

    LookSAM wraps an existing PyTorch optimizer (``Adam``, ``SGD``, ...).
    The base optimizer consumes the LookSAM-modified gradient and performs the
    actual parameter update.

    Parameters
    ----------
    params : iterable
        Iterable of parameters or parameter groups, in the standard
        :class:`torch.optim.Optimizer` convention. Ignored: the parameters are
        taken from ``base_optimizer``.
    base_optimizer : torch.optim.Optimizer
        Inner optimizer that performs the actual parameter update.
    rho : float, default=0.05
        Radius of the L2 ascent ball (:math:`\rho` in the paper).
    alpha : float, default=0.7
        Reuse weight for the cached flat-region component on non-SAM steps.
        The paper reports best generalization at :math:`\alpha = 0.7`.
    k : int, default=5
        Frequency of full SAM updates. ``k=1`` reduces to vanilla SAM.
    eps : float, default=1e-12
        Numerical stabilizer for gradient-norm denominators.

    Notes
    -----
    * **Closure-based step.** :meth:`step` must be called with a closure that
      zeroes gradients, computes the loss, calls ``loss.backward()``, and
      returns the loss. On full-SAM steps the closure is invoked twice; on
      fast steps it is invoked once. Gradient clipping, if any, belongs inside
      the closure after ``backward()``.
    * **Exact weight restore.** The weights are copied back after the ascent
      pass, not recovered by subtracting the perturbation, so low-precision
      (e.g. ``bfloat16``) weights do not drift.
    * **Mixed precision.** Use ``bfloat16`` autocast. A ``float16``
      :class:`~torch.amp.GradScaler` is not supported: the ascent direction
      would be computed from scaled gradients.
    * **Hyperparameters.** ``rho``, ``alpha``, ``k``, and ``eps`` are read from
      the first parameter group, so they apply to all parameters.
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
    * **Param-group sharing.** ``self.param_groups`` aliases
      ``base_optimizer.param_groups`` (also after :meth:`load_state_dict`), so
      LR schedulers wire transparently.
    * **Step counters.** :attr:`global_step` advances once per :meth:`step`
      call. :attr:`sam_step` counts only full-SAM updates.

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

    _HPARAMS = ("rho", "alpha", "k", "eps")

    def __init__(
        self,
        params: Any,
        base_optimizer: Optimizer,
        rho: float = 0.05,
        alpha: float = 0.7,
        k: int = 5,
        eps: float = 1e-12,
    ) -> None:
        if not isinstance(base_optimizer, Optimizer):
            raise ValueError(
                "`base_optimizer` must be a torch.optim.Optimizer instance."
            )
        if rho <= 0.0:
            raise ValueError(f"`rho` must be positive, got {rho}.")
        if alpha < 0.0:
            raise ValueError(f"`alpha` must be non-negative, got {alpha}.")
        if k < 1:
            raise ValueError(f"`k` must be at least 1, got {k}.")

        # `params` mirrors the torch.optim.Optimizer signature for API compatibility;
        # actual parameter groups are owned by base_optimizer and shared via reference.
        del params
        # Only real hyperparameters go into `defaults`; they are copied into every
        # param group by Optimizer.__init__.
        defaults = {"rho": rho, "alpha": alpha, "k": k, "eps": eps}
        super().__init__(base_optimizer.param_groups, defaults)

        self.base_optimizer = base_optimizer
        self._global_step: int = 0
        self._sam_step: int = 0
        # Flat g_v buffers and the cached global ‖g_v‖ used by the fast steps.
        self._buckets: list[_Bucket] = []
        self._g_v_norm: torch.Tensor | None = None

    @property
    def global_step(self) -> int:
        """Number of :meth:`step` calls since construction."""
        return self._global_step

    @property
    def sam_step(self) -> int:
        """Number of full-SAM updates performed."""
        return self._sam_step

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:  # type: ignore[override]
        """Perform one LookSAM step.

        Parameters
        ----------
        closure : Callable[[], torch.Tensor]
            Zeroes grads, evaluates the loss, calls ``loss.backward()``, and
            returns the loss. Invoked twice on SAM steps, once on fast steps.

        Returns
        -------
        torch.Tensor
            Loss returned by the first closure call.
        """
        if closure is None:
            raise ValueError("LookSAM.step() requires a closure.")
        closure = torch.enable_grad()(closure)
        group = self.param_groups[0]
        rho, alpha, k, eps = (group.get(h, self.defaults[h]) for h in self._HPARAMS)

        loss = closure()
        if self._global_step % k == 0:
            self._sam_step += 1
            self._sam_update(closure, rho, eps)
        elif self._buckets:
            self._fast_update(alpha, eps)

        self.base_optimizer.step()
        self._global_step += 1
        return loss

    def state_dict(self) -> dict[str, Any]:
        """Return serializable LookSAM state (base optimizer + g_v + counters)."""
        g_v = [
            [self._export(p, self.state.get(p, {}).get("g_v")) for p in group["params"]]
            for group in self.param_groups
        ]
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "g_v": g_v,
            "global_step": self._global_step,
            "sam_step": self._sam_step,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore LookSAM state from a dict produced by :meth:`state_dict`.

        Collective under distributed training: the cached ``‖g_v‖`` is
        recomputed with an all-reduce over sharded parameters.
        """
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        # The base optimizer rebuilt its param-group dicts; re-alias them so LR
        # schedulers stepping this wrapper still reach the update rule.
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            for h in self._HPARAMS:
                group.setdefault(h, self.defaults[h])
        self._global_step = int(state_dict["global_step"])
        self._sam_step = int(state_dict["sam_step"])

        params, bufs = [], []
        for group, row in zip(self.param_groups, state_dict["g_v"]):
            for p, buf in zip(group["params"], row):
                self.state.get(p, {}).pop("g_v", None)
                if buf is not None:
                    params.append(p)
                    bufs.append(_local(buf).to(_local(p).device))
        self._buckets = []
        self._g_v_norm = None
        if params:
            self._buckets = [
                self._bucket(ps, key, bs)
                for key, ps, bs in self._group(
                    params, bufs, [_reduce_key(p) for p in params]
                )
            ]
            self._cache_g_v()

    # ------------------------------------------------------------------
    # Ascent hook (overridden by LookLayerSAM)
    # ------------------------------------------------------------------

    def _ascend(
        self, buckets: list[_Bucket], g_norm: torch.Tensor, rho: float, eps: float
    ) -> torch.Tensor:
        """Move w to w + e_w with e_w = rho g / ‖g‖ (global norm).

        Scales ``flat_g`` into e_w in place and returns the factor s with
        ``flat_g = s * g``, so no separate copy of g is kept.
        """
        scale = rho / (g_norm + eps)
        for b in buckets:
            b.flat_g.mul_(scale.to(b.flat_g.device, b.flat_g.dtype))
            torch._foreach_add_([_local(p) for p in b.params], b.g_views)
        return scale

    # ------------------------------------------------------------------
    # Update rules
    # ------------------------------------------------------------------

    def _sam_update(
        self, closure: Callable[[], torch.Tensor], rho: float, eps: float
    ) -> None:
        """Ascend to w + e_w, take g_s there, restore w, cache g_v, hand g_s on."""
        for b in self._buckets:
            for p in b.params:
                self.state[p].pop("g_v", None)
        self._buckets, self._g_v_norm = [], None
        params = [
            p
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not params:
            closure()
            return

        grads = [_local(p.grad) for p in params]
        buckets = []
        for key, ps, gs in self._group(
            params, grads, [_reduce_key(p.grad) for p in params]
        ):
            b = self._bucket(ps, key, gs)
            # One flat snapshot each of g and w.
            b.flat_g, b.g_views = b.flat_v, b.v_views
            b.flat_w, b.w_views = self._flatten([_local(p) for p in ps])
            buckets.append(b)

        g_sq = self._global_sum(
            [
                (torch.linalg.vector_norm(b.flat_g, dtype=self._acc(b)).square(), b)
                for b in buckets
            ]
        )
        s = self._ascend(buckets, g_sq.sqrt(), rho, eps)

        closure()

        # Exact restore of w; flat_w is then reused for g_s and turned into
        # g_v = g_s - (g·g_s / ‖g‖²) g = g_s - (flat_g·g_s / (s² (‖g‖² + eps))) flat_g.
        dots = []
        for b in buckets:
            torch._foreach_copy_([_local(p) for p in b.params], b.w_views)
            b.flat_v, b.v_views = b.flat_w, b.w_views
            b.flat_w = b.w_views = None
            present = [i for i, p in enumerate(b.params) if p.grad is not None]
            if len(present) < len(b.params):
                b.flat_v.zero_()
            torch._foreach_copy_(
                [b.v_views[i] for i in present],
                [_local(b.params[i].grad) for i in present],
            )
            acc = self._acc(b)
            dots.append((torch.dot(b.flat_v.to(acc), b.flat_g.to(acc)), b))
        coef = self._global_sum(dots) / (s * s * (g_sq + eps))
        for b in buckets:
            b.flat_v.addcmul_(b.flat_g, -coef.to(b.flat_v.device, b.flat_v.dtype))
            b.flat_g = b.g_views = None

        self._buckets = buckets
        self._cache_g_v()

    def _fast_update(self, alpha: float, eps: float) -> None:
        """g ← g + α (‖g‖ / ‖g_v‖) g_v."""
        grads = [
            (b, [_local(p.grad) if p.grad is not None else None for p in b.params])
            for b in self._buckets
        ]
        partials = [
            (
                torch.stack(torch._foreach_norm(present, 2, dtype=self._acc(b)))
                .square()
                .sum(),
                b,
            )
            for b, gs in grads
            if (present := [g for g in gs if g is not None])
        ]
        if not partials:
            return
        g_sq = self._global_sum(partials)
        blend = alpha * g_sq.sqrt() / (self._g_v_norm + eps)
        for b, gs in grads:
            delta = b.flat_v * blend.to(b.flat_v.device, b.flat_v.dtype)
            views = self._views(delta, b.v_views)
            pairs = [(g, v) for g, v in zip(gs, views) if g is not None]
            if pairs:
                torch._foreach_add_([g for g, _ in pairs], [v for _, v in pairs])

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
                (torch.linalg.vector_norm(b.flat_v, dtype=self._acc(b)).square(), b)
                for b in self._buckets
            ]
        )
        self._g_v_norm = g_v_sq.sqrt()

    @staticmethod
    def _group(params, tensors, keys):
        """Group aligned (param, tensor) pairs by device, dtype, and reduce key."""
        groups: dict[Any, tuple[list, list]] = {}
        for p, t, key in zip(params, tensors, keys):
            ps, ts = groups.setdefault((t.device, t.dtype, key), ([], []))
            ps.append(p)
            ts.append(t)
        return [(key[2], ps, ts) for key, (ps, ts) in groups.items()]

    @classmethod
    def _bucket(cls, params, key, tensors) -> _Bucket:
        flat, views = cls._flatten(tensors)
        return _Bucket(params, key, flat, views)

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
    def _acc(b: _Bucket) -> torch.dtype:
        """Accumulation dtype for reductions: at least float32."""
        return torch.promote_types(b.flat_v.dtype, torch.float32)

    @staticmethod
    def _global_sum(partials: list[tuple[torch.Tensor, _Bucket]]) -> torch.Tensor:
        """Sum per-bucket local partial sums into one global value.

        Partials of buckets that share a reduce key are added first, so each
        distinct mesh costs one all-reduce per sharded mesh dimension.
        """
        home = partials[0][0].device
        dtype = partials[0][0].dtype
        for t, _ in partials[1:]:
            dtype = torch.promote_types(dtype, t.dtype)
        by_key: dict[Any, torch.Tensor] = {}
        for t, b in partials:
            t = t.to(home, dtype)
            by_key[b.key] = by_key[b.key] + t if b.key in by_key else t
        total = torch.zeros((), device=home, dtype=dtype)
        for key, value in by_key.items():
            if key is not None:
                mesh, dims = key
                for d in dims:
                    dist.all_reduce(value, group=mesh.get_group(d))
            total = total + value
        return total

    @staticmethod
    def _export(p: torch.nn.Parameter, buf: torch.Tensor | None):
        """Detached copy of a g_v buffer in the parameter's distributed layout."""
        if buf is None:
            return None
        buf = buf.detach().clone()
        if hasattr(p, "_local_tensor") and hasattr(p, "placements"):
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
        return buf


class LookLayerSAM(LookSAM):
    r"""Layer-wise variant of LookSAM.

    Replaces the global-norm perturbation with per-layer perturbations
    (analogous to LARS/LAMB layer-wise scaling). Each layer gets its own
    :math:`\epsilon_\ell = \rho \cdot g_\ell / \|g_\ell\|_2`, avoiding
    gradient-magnitude imbalances across layers at large batch sizes.

    All other behavior (``g_v`` caching, fast steps, distributed support) is
    inherited from :class:`LookSAM`.
    """

    def _ascend(
        self, buckets: list[_Bucket], g_norm: torch.Tensor, rho: float, eps: float
    ) -> torch.Tensor:
        """Move each parameter (layer) l by rho g_l / ‖g_l‖; ``flat_g`` keeps g."""
        for b in buckets:
            acc = self._acc(b)
            sq = torch.stack(torch._foreach_norm(b.g_views, 2, dtype=acc)).square()
            if b.key is not None:
                mesh, dims = b.key
                for d in dims:
                    dist.all_reduce(sq, group=mesh.get_group(d))
            sizes = torch.tensor([v.numel() for v in b.g_views], device=sq.device)
            step = (rho / (sq.sqrt() + eps)).to(b.flat_g.dtype)
            step = step.repeat_interleave(sizes, output_size=b.flat_g.numel())
            step.mul_(b.flat_g)
            torch._foreach_add_(
                [_local(p) for p in b.params], self._views(step, b.g_views)
            )
        return torch.ones((), device=g_norm.device, dtype=g_norm.dtype)
