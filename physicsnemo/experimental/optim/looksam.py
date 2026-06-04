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

from typing import Any, Callable

import torch
from torch.optim import Optimizer

__all__ = ["LookSAM", "LookLayerSAM"]


class LookSAM(Optimizer):
    r"""Efficient Sharpness-Aware Minimization (LookSAM) optimizer wrapper.

    Implements LookSAM from Liu et al., *"Towards Efficient and Scalable
    Sharpness-Aware Minimization"* (CVPR 2022, arXiv:2203.02714). LookSAM
    accelerates Sharpness-Aware Minimization (SAM) by computing the expensive
    inner ascent gradient only every ``k`` steps and reusing the cached
    "flat-region" component ``g_v`` on intermediate steps, recovering most of
    SAM's generalization benefit at roughly :math:`1/k` the additional compute.

    LookSAM wraps an existing PyTorch optimizer (``Adam``, ``SGD``, ...).
    The base optimizer consumes the LookSAM-modified gradient and performs the
    actual parameter update.

    Parameters
    ----------
    params : iterable
        Iterable of parameters or parameter groups, in the standard
        :class:`torch.optim.Optimizer` convention. Must reference the same
        parameters as ``base_optimizer``.
    base_optimizer : torch.optim.Optimizer
        Inner optimizer that performs the actual parameter update. Its
        ``param_groups`` must reference the same parameters as ``params``.
    rho : float, default=0.05
        Radius of the L2 ascent ball (:math:`\rho` in the paper).
        Typical range 0.01–0.2.
    alpha : float, default=0.7
        Reuse weight for the cached flat-region component on non-SAM steps.
        The paper reports best generalization at :math:`\alpha = 0.7` (Table 6).
    k : int, default=5
        Frequency of full SAM updates. ``k=1`` reduces to vanilla SAM.
        Larger ``k`` reduces extra compute at the cost of fidelity.
    eps : float, default=1e-12
        Numerical stabilizer for gradient-norm denominators.
    Notes
    -----
    * **Closure-based step.** :meth:`step` must be called with a closure that
      zeroes gradients, computes the loss, calls ``loss.backward()``, and
      returns the loss. On full-SAM steps the closure is invoked twice; on
      fast steps it is invoked once.
    * **AMP.** Use ``bfloat16`` autocast — ``fp16`` produces NaN gradients on
      sm_87 via CC 8.0 PTX fallback. ``bfloat16`` needs no
      ``GradScaler`` and is recommended for all Ampere+ GPUs.
    * **Distributed (DDP).** Both backward passes on a full-SAM step are
      AllReduced normally. Unlike vanilla SAM — where the ascent gradient is
      discarded after perturbing the weights and its sync can be skipped via
      ``no_sync()`` — LookSAM *reuses* the ascent gradient :math:`g` inside the
      cached flat component :math:`g_v` that drives subsequent fast steps. If
      the ascent pass were not synced, each rank would cache a different
      :math:`g_v` and the fast steps would diverge across ranks. The redundant
      ascent AllReduce is therefore the price of LookSAM's gradient reuse and
      cannot be skipped; just wrap the model in ``DistributedDataParallel`` as
      usual.
    * **Param-group sharing.** ``self.param_groups`` aliases
      ``base_optimizer.param_groups`` so LR schedulers wire transparently.
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
            raise ValueError("`base_optimizer` must be a torch.optim.Optimizer instance.")
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

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def global_step(self) -> int:
        """Number of :meth:`step` calls since construction."""
        return self._global_step

    @property
    def sam_step(self) -> int:
        """Number of full-SAM updates performed."""
        return self._sam_step

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        d = self._defaults()
        rho, alpha, k, eps = d["rho"], d["alpha"], d["k"], d["eps"]
        params = self._params()
        is_sam_step = (self._global_step % k) == 0

        if is_sam_step:
            self._sam_step += 1

            # ---- First forward+backward: g = ∇L(w) ----
            # Under DDP this ascent gradient is AllReduced like any other; it
            # must stay synced because LookSAM *reuses* it inside the cached
            # g_v (see Notes, Distributed), so an un-synced g would make every
            # subsequent fast step diverge across ranks.
            loss = closure()

            g_snap = [p.grad.detach().clone() if p.grad is not None else None for p in params]
            self._ascend(params, g_snap, rho, eps)

            # ---- Second forward+backward at perturbed weights: g_s = ∇L(w+e) ----
            closure()

            # Restore weights and compute g_v.
            self._restore_and_cache_gv(params, g_snap, eps)

        else:
            # ---- Fast step: single (descent) forward+backward, synced ----
            loss = closure()

            # g ← g + α·(‖g‖/‖g_v‖)·g_v
            g_norm = self._global_norm(params, [p.grad for p in params])
            g_v_norm = self._buffer_norm(params, "g_v")
            blend = alpha * g_norm / (g_v_norm + eps)
            for p in params:
                if p.grad is None:
                    continue
                g_v = self.state.get(p, {}).get("g_v")
                if g_v is not None:
                    p.grad.detach().add_(g_v, alpha=float(blend))

        self.base_optimizer.step()
        self._global_step += 1
        return loss

    def state_dict(self) -> dict[str, Any]:
        """Return serializable LookSAM state (base optimizer + g_v + counters)."""
        g_v: list[list[torch.Tensor | None]] = []
        for group in self.param_groups:
            row: list[torch.Tensor | None] = []
            for p in group["params"]:
                buf = self.state.get(p, {}).get("g_v")
                row.append(buf.detach().clone() if buf is not None else None)
            g_v.append(row)
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "g_v": g_v,
            "global_step": self._global_step,
            "sam_step": self._sam_step,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore LookSAM state from a dict produced by :meth:`state_dict`."""
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self._global_step = int(state_dict["global_step"])
        self._sam_step = int(state_dict["sam_step"])
        for group, row in zip(self.param_groups, state_dict["g_v"]):
            for p, buf in zip(group["params"], row):
                if buf is None:
                    self.state.get(p, {}).pop("g_v", None)
                else:
                    self.state[p]["g_v"] = buf.detach().clone().to(p.device)

    # ------------------------------------------------------------------
    # Ascent hook (overridden by LookLayerSAM)
    # ------------------------------------------------------------------

    def _ascend(
        self,
        params: list[torch.nn.Parameter],
        g_snap: list[torch.Tensor | None],
        rho: float,
        eps: float,
    ) -> None:
        """Perturb parameters by rho * g / ‖g‖ (global norm)."""
        g_norm = self._global_norm(params, g_snap)
        scale = rho / (g_norm + eps)
        for p, g in zip(params, g_snap):
            if g is None:
                continue
            e_w = g * scale
            self.state[p]["e_w"] = e_w
            p.add_(e_w)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _restore_and_cache_gv(
        self,
        params: list[torch.nn.Parameter],
        g_snap: list[torch.Tensor | None],
        eps: float,
    ) -> None:
        """Restore weights and cache g_v = g_s − (g·g_s/‖g‖²) g."""
        g_norm = self._global_norm(params, g_snap)

        # Global dot product g · g_s
        device = g_norm.device
        dot = torch.zeros((), device=device)
        for p, g in zip(params, g_snap):
            if g is None or p.grad is None:
                continue
            dot = dot + (g * p.grad.detach()).sum().to(dot.dtype)

        proj_coef = dot / (g_norm * g_norm + eps)

        for p, g in zip(params, g_snap):
            e_w = self.state[p].pop("e_w", None)
            if e_w is not None:
                p.sub_(e_w)
            if g is None or p.grad is None:
                continue
            self.state[p]["g_v"] = p.grad.detach() - proj_coef * g

    def _params(self) -> list[torch.nn.Parameter]:
        return [p for group in self.param_groups for p in group["params"]]

    def _defaults(self) -> dict[str, Any]:
        first = self.param_groups[0]
        return {k: first.get(k, self.defaults[k]) for k in ("rho", "alpha", "k", "eps")}

    @staticmethod
    def _global_norm(
        params: list[torch.nn.Parameter],
        grads: list[torch.Tensor | None],
    ) -> torch.Tensor:
        device = next((p.device for p in params if p is not None), torch.device("cpu"))
        sq = torch.zeros((), device=device)
        for g in grads:
            if g is not None:
                sq = sq + g.detach().to(device).pow(2).sum()
        return sq.sqrt()

    def _buffer_norm(self, params: list[torch.nn.Parameter], key: str) -> torch.Tensor:
        device = next((p.device for p in params if p is not None), torch.device("cpu"))
        sq = torch.zeros((), device=device)
        for p in params:
            buf = self.state.get(p, {}).get(key)
            if buf is not None:
                sq = sq + buf.to(device).pow(2).sum()
        return sq.sqrt()


class LookLayerSAM(LookSAM):
    r"""Layer-wise variant of LookSAM (Look-LayerSAM in the paper).

    Replaces the global-norm perturbation with per-layer perturbations
    (analogous to LARS/LAMB layer-wise scaling). Each layer gets its own
    :math:`\epsilon_\ell = \rho \cdot g_\ell / \|g_\ell\|_2`, avoiding
    gradient-magnitude imbalances across layers at large batch sizes.

    All other behaviour — ``decompose_grad``, ``g_v`` caching, fast steps —
    is inherited from :class:`LookSAM`.

    The paper (Section 4.3) reports that Look-LayerSAM outperforms LookSAM
    at large batch sizes (16k–32k) on ImageNet with ViT-B/16.
    """

    def _ascend(
        self,
        params: list[torch.nn.Parameter],
        g_snap: list[torch.Tensor | None],
        rho: float,
        eps: float,
    ) -> None:
        """Perturb each parameter by rho * g_l / ‖g_l‖ (per-layer norm)."""
        for p, g in zip(params, g_snap):
            if g is None:
                continue
            layer_norm = g.norm(p=2)
            e_w = g * (rho / (layer_norm + eps))
            self.state[p]["e_w"] = e_w
            p.add_(e_w)
