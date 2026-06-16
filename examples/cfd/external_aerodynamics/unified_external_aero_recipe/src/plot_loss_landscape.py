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

r"""Filter-normalized loss-landscape visualization.

Reproduces the loss-surface plots used in the Sharpness-Aware Minimization
paper (Foret et al., ICLR 2021, Figure 1), which were generated following the
method of Li et al., *"Visualizing the Loss Landscape of Neural Nets"*
(NeurIPS 2018, arXiv:1712.09913).

The recipe is:

1. Take trained weights :math:`\theta^*`.
2. Draw two random direction tensors :math:`\delta, \eta` shaped like
   :math:`\theta^*`.
3. **Filter-normalize** each direction: for every weight tensor (filter /
   layer), rescale the direction so its per-filter norm matches the
   per-filter norm of :math:`\theta^*`. This removes the scale invariance
   that otherwise makes any network look spuriously flat.
4. Sweep a 2D grid and evaluate the loss at
   :math:`\theta^* + \alpha\,\delta + \beta\,\eta` for
   :math:`\alpha, \beta \in [-\text{span}, \text{span}]`.
5. Render the grid as a 3D surface and a 2D contour.

The "sharp vs. wide" contrast of SAM Figure 1 is simply this same procedure
run on two checkpoints of the *same* architecture (e.g. Adam vs. LookSAM) and
rendered with a shared z-scale via :func:`compare_landscapes`.

This module is split into two layers:

* A **model-agnostic core** (:func:`filter_normalized_directions`,
  :func:`evaluate_landscape`, :func:`plot_landscape`,
  :func:`compare_landscapes`) that depends only on ``torch``, ``numpy`` and
  ``matplotlib``. It works for any ``nn.Module`` and any scalar ``loss_fn``.
* A **recipe glue** layer (the ``hydra``-driven :func:`main`) that wires the
  core to the unified external-aero recipe so it can sweep a GeoTransolver /
  DrivaerML checkpoint over one fixed mini-batch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Model-agnostic core
# ---------------------------------------------------------------------------


@torch.no_grad()
def filter_normalized_directions(
    model: torch.nn.Module,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    r"""Build two filter-normalized random directions shaped like the weights.

    For each parameter tensor ``w`` the random direction ``d`` is rescaled
    filter-wise so that ``norm(d_filter) == norm(w_filter)`` (Li et al. 2018).
    "Filter" means the leading dimension of a weight tensor (output channels /
    rows); the normalization is applied per leading-dim slice. Parameters with
    fewer than two dimensions (biases, norm scales) carry no meaningful
    curvature direction and are zeroed so the sweep does not move them.

    Parameters
    ----------
    model : torch.nn.Module
        Model holding the trained weights :math:`\theta^*`.
    seed : int, default=0
        Seed for the random directions (reproducible surfaces).
    device : torch.device | str, optional
        Device for the generated directions. Defaults to each parameter's
        own device.

    Returns
    -------
    (dir_a, dir_b) : tuple[list[Tensor], list[Tensor]]
        Two lists of tensors, parallel to ``list(model.parameters())``.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def _one_direction() -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        for p in model.parameters():
            dev = device if device is not None else p.device
            # Generate on CPU for cross-device-reproducible seeds, then move.
            d = torch.randn(p.shape, generator=gen, dtype=p.dtype).to(dev)
            if p.dim() < 2:
                # Bias / norm parameters: no filter structure, leave fixed.
                d.zero_()
            else:
                # Per-filter (leading-dim) normalization to the weight norm.
                w = p.detach()
                flat_d = d.reshape(d.shape[0], -1)
                flat_w = w.reshape(w.shape[0], -1)
                d_norm = flat_d.norm(dim=1, keepdim=True)
                w_norm = flat_w.norm(dim=1, keepdim=True)
                flat_d.mul_(w_norm / (d_norm + 1e-10))
                d = flat_d.reshape(p.shape)
            out.append(d)
        return out

    return _one_direction(), _one_direction()


@torch.no_grad()
def evaluate_landscape(
    model: torch.nn.Module,
    loss_fn: Callable[[], float],
    dir_a: Sequence[torch.Tensor],
    dir_b: Sequence[torch.Tensor],
    alphas: np.ndarray,
    betas: np.ndarray,
    progress: bool = True,
) -> np.ndarray:
    r"""Evaluate ``loss_fn`` over the 2D grid of perturbed weights.

    For each ``(alpha, beta)`` the parameters are set to
    :math:`\theta^* + \alpha\,\delta + \beta\,\eta` in place, ``loss_fn()`` is
    called, and the original weights are restored **exactly** (the saved
    reference is added back, not recomputed). Runs entirely under
    ``torch.no_grad()``; ``loss_fn`` should likewise avoid building a graph.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose parameters are perturbed. Restored to :math:`\theta^*` on
        return (including if ``loss_fn`` raises).
    loss_fn : Callable[[], float]
        Zero-argument closure returning a scalar loss for the *current*
        weights. Typically wraps a forward pass over one fixed batch.
    dir_a, dir_b : Sequence[Tensor]
        Directions from :func:`filter_normalized_directions`.
    alphas, betas : np.ndarray
        1D coordinate arrays for the two axes.
    progress : bool, default=True
        Print a per-row progress line to stdout.

    Returns
    -------
    np.ndarray
        ``(len(betas), len(alphas))`` array of loss values. Row index is
        ``beta`` (y), column index is ``alpha`` (x), so it plots directly
        with ``meshgrid(alphas, betas)``.
    """
    params = list(model.parameters())
    theta_star = [p.detach().clone() for p in params]

    grid = np.empty((len(betas), len(alphas)), dtype=np.float64)
    try:
        for i, beta in enumerate(betas):
            for j, alpha in enumerate(alphas):
                for p, w0, da, db in zip(params, theta_star, dir_a, dir_b):
                    p.copy_(w0 + alpha * da + beta * db)
                grid[i, j] = float(loss_fn())
            if progress:
                print(f"  landscape row {i + 1}/{len(betas)} done", flush=True)
    finally:
        # Restore exactly, even on error.
        for p, w0 in zip(params, theta_star):
            p.copy_(w0)
    return grid


def plot_landscape(
    alphas: np.ndarray,
    betas: np.ndarray,
    grid: np.ndarray,
    out_path: str | Path,
    title: str = "Loss landscape",
    log_z: bool = False,
) -> Path:
    """Render a single landscape as a 3D surface + 2D contour PNG.

    Parameters
    ----------
    alphas, betas : np.ndarray
        Axis coordinates used for the sweep.
    grid : np.ndarray
        ``(len(betas), len(alphas))`` loss values from
        :func:`evaluate_landscape`.
    out_path : str | Path
        PNG path. The raw grid is also saved next to it as ``.npz``.
    title : str
        Figure title.
    log_z : bool, default=False
        Plot ``log10(loss)`` instead of raw loss (useful when the surface
        spans several orders of magnitude).

    Returns
    -------
    Path
        The written PNG path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    z = np.log10(grid) if log_z else grid
    zlabel = "log10(loss)" if log_z else "loss"
    A, B = np.meshgrid(alphas, betas)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(A, B, z, cmap="coolwarm", linewidth=0, antialiased=True)
    ax1.set_xlabel(r"$\alpha$")
    ax1.set_ylabel(r"$\beta$")
    ax1.set_zlabel(zlabel)
    ax1.set_title(f"{title} (surface)")

    ax2 = fig.add_subplot(1, 2, 2)
    cs = ax2.contourf(A, B, z, levels=30, cmap="coolwarm")
    ax2.contour(A, B, z, levels=15, colors="k", linewidths=0.3, alpha=0.5)
    fig.colorbar(cs, ax=ax2, label=zlabel)
    ax2.set_xlabel(r"$\alpha$")
    ax2.set_ylabel(r"$\beta$")
    ax2.set_title(f"{title} (contour)")
    ax2.set_aspect("equal")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    np.savez(
        out_path.with_suffix(".npz"), alphas=alphas, betas=betas, grid=grid
    )
    return out_path


def compare_landscapes(
    alphas: np.ndarray,
    betas: np.ndarray,
    grid_a: np.ndarray,
    grid_b: np.ndarray,
    out_path: str | Path,
    titles: tuple[str, str] = ("A", "B"),
    log_z: bool = False,
    shared_scale: bool = False,
) -> Path:
    """Render two landscapes side-by-side: the SAM Figure 1 contrast.

    Parameters
    ----------
    shared_scale : bool, default=False
        If True, both surfaces use a common z/color range so their heights
        are directly comparable. This is the most honest comparison, but when
        one minimum is far sharper than the other the flatter surface can
        render as a near-flat sheet. If False (default, matching how Li et al.
        2018 and SAM Fig 1 present the surfaces) each surface is scaled to its
        own range so its shape stays visible; the contour/relative curvature
        is still the takeaway.

    Returns
    -------
    Path
        The written PNG path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    za = np.log10(grid_a) if log_z else grid_a
    zb = np.log10(grid_b) if log_z else grid_b
    zlabel = "log10(loss)" if log_z else "loss"
    A, B = np.meshgrid(alphas, betas)

    if shared_scale:
        lo = float(min(za.min(), zb.min()))
        hi = float(max(za.max(), zb.max()))
        limits = [(lo, hi), (lo, hi)]
    else:
        limits = [(float(za.min()), float(za.max())),
                  (float(zb.min()), float(zb.max()))]

    fig = plt.figure(figsize=(12, 5))
    for k, (z, name, (zmin, zmax)) in enumerate(
        [(za, titles[0], limits[0]), (zb, titles[1], limits[1])]
    ):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.plot_surface(
            A, B, z, cmap="coolwarm", vmin=zmin, vmax=zmax,
            linewidth=0, antialiased=True,
        )
        if zmax > zmin:
            ax.set_zlim(zmin, zmax)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$\beta$")
        ax.set_zlabel(zlabel)
        ax.set_title(name)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    np.savez(
        out_path.with_suffix(".npz"),
        alphas=alphas,
        betas=betas,
        grid_a=grid_a,
        grid_b=grid_b,
    )
    return out_path


def compute_landscape(
    model: torch.nn.Module,
    loss_fn: Callable[[], float],
    resolution: int = 21,
    span: float = 1.0,
    seed: int = 0,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience wrapper: directions + symmetric grid + evaluation.

    Returns ``(alphas, betas, grid)``.
    """
    alphas = np.linspace(-span, span, resolution)
    betas = np.linspace(-span, span, resolution)
    dir_a, dir_b = filter_normalized_directions(model, seed=seed)
    grid = evaluate_landscape(
        model, loss_fn, dir_a, dir_b, alphas, betas, progress=progress
    )
    return alphas, betas, grid


# ---------------------------------------------------------------------------
# Recipe glue (unified external aero recipe)
# ---------------------------------------------------------------------------


def _build_recipe_loss_fn(cfg):  # pragma: no cover - needs GPU + checkpoint
    """Build ``(model, loss_fn)`` from the unified external-aero recipe.

    Reuses the recipe's own model construction, loss calculator, checkpoint
    loading and a single fixed mini-batch. ``loss_fn`` evaluates that batch's
    scalar loss for the model's current weights, under ``no_grad`` + autocast.
    """
    import os

    import hydra

    from datasets import build_dataloaders
    from loss import LossCalculator
    from metrics import MetricCalculator
    from output_normalize import require_output_type
    from train import forward_pass
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint

    DistributedManager.initialize()
    dist_manager = DistributedManager()
    device = dist_manager.device

    train_loader, _val_loader, _normalizer, dataset_info = build_dataloaders(cfg)
    target_config = dataset_info["targets"]
    output_type = require_output_type(cfg)

    model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(device)
    model.eval()

    loss_calculator = LossCalculator(
        target_config=target_config, **cfg.training.loss
    )
    metric_calculator = MetricCalculator(target_config=target_config)

    ckpt_args = {
        "path": os.path.join(
            cfg.output, cfg.run_id, "checkpoints"
        ),
        "models": model,
    }
    load_checkpoint(device=device, **ckpt_args)

    # One fixed batch — cached once, reused for every grid point.
    batch = next(iter(train_loader))

    def loss_fn() -> float:
        loss, _loss_td, _metric_td = forward_pass(
            batch,
            model,
            cfg.training.precision,
            loss_calculator,
            metric_calculator,
            output_type=output_type,
            target_config=target_config,
        )
        return float(loss.detach())

    return model, loss_fn


def main():  # pragma: no cover - needs GPU + checkpoint
    """CLI entry point for the recipe path.

    Examples
    --------
    Single checkpoint::

        python plot_loss_landscape.py --config-name train \
            run_id=my_run resolution=21 span=1.0 out=landscape.png

    The model-agnostic core (everything above this section) is what the unit
    tests exercise; this entry point requires a GPU and a trained checkpoint.
    """
    import argparse

    import hydra
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", default="../conf")
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--resolution", type=int, default=21)
    parser.add_argument("--span", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-z", action="store_true")
    parser.add_argument("--out", default="loss_landscape.png")
    parser.add_argument(
        "overrides", nargs="*", help="Hydra config overrides (key=value)."
    )
    args = parser.parse_args()

    with hydra.initialize(config_path=args.config_path, version_base=None):
        cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)
    print(OmegaConf.to_yaml(cfg, resolve=False))

    model, loss_fn = _build_recipe_loss_fn(cfg)
    alphas, betas, grid = compute_landscape(
        model,
        loss_fn,
        resolution=args.resolution,
        span=args.span,
        seed=args.seed,
    )
    out = plot_landscape(
        alphas, betas, grid, args.out, title=f"Loss landscape ({args.out})",
        log_z=args.log_z,
    )
    print(f"wrote {out} and {out.with_suffix('.npz')}")


if __name__ == "__main__":  # pragma: no cover
    main()
