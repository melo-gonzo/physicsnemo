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

r"""ISLA's boundary-token sample: systematic sampling proportional to size.

An interior model reads its boundary as a fixed budget of tokens drawn inside
the forward pass from the full-resolution surfaces (:func:`sample_boundary_cells`):

- **Budget.** The total ``budget`` is split over the named conditions by water
  filling (:func:`water_fill`): equal shares, a condition with fewer cells
  than its share kept whole and its unused share given equally to the others,
  until stable. The shares are real numbers summing to the budget.
- **Inclusion probabilities.** Within a condition of share :math:`b` the cell
  :math:`i` of measure :math:`a_i` is drawn with probability
  :math:`\pi_i = \min(1, c\,a_i)`, :math:`c` the one constant with
  :math:`\sum_i \pi_i = b` (the capped fixed point of the iterative adjustment,
  found exactly from the sorted measures: :func:`pps_inclusion_probabilities`);
  a condition kept whole has every :math:`\pi_i = 1`.
- **Draw.** Systematic sampling: the cells are laid along a line, each
  condition's cells in the Morton (Z-order) order of their centroids in the
  boundary's canonical frame (``tree_read.canonical_frame`` over every cell), each
  cell an interval of length :math:`\pi_i`, and the cells hit by the points
  :math:`u + m`, :math:`m = 0, \dots, k - 1`, are kept. The intervals are
  half-open, so a hit exactly on a boundary takes the later cell. The count is exactly
  :math:`k = \min(\sum_c N_c, \text{budget})`, a function of the cell counts
  only (static shapes); every cell is kept with probability
  :math:`\pi_i` over a uniform start :math:`u \in [0, 1)` (up to the line's
  grid, below), a cell with :math:`\pi_i = 1` exactly once; and the kept cells
  are spread along the space-filling curve, a spatially stratified sample.
- **Weights.** A kept cell stands for :math:`a_i / \pi_i` of the boundary (the
  Horvitz-Thompson weight), so every measure-weighted sum over the tokens is
  unbiased for the sum over the full surface, and exact on a condition kept
  whole.
- **Precision.** The measures and centroids the draw reads are computed in
  float64 from the mesh's coordinates as given (:func:`draw_inputs`), whatever
  the model's precision, so the draw is a function of the input mesh only: a
  float32 and a float64 run on the same inputs keep the same cells. The tail
  sums of the measures are a fixed-order float64 scan (:func:`_cumsum`), the
  same bits on every device. The line is exact int64 arithmetic in units of
  :math:`2^{-Q}`, :math:`Q = 62 - b`, :math:`b` the bit length of the budget (a
  Python int), each cell an interval of :math:`\operatorname{round}(\pi_i 2^Q)`
  units; a cell with :math:`\pi_i = 1` is exactly :math:`2^Q` units, and the
  start is clamped by the line's rounding residual, so such a cell holds exactly
  one hit. The drawn inclusion probability is that length over :math:`2^Q`,
  within :math:`2^{-(Q+1)}` of :math:`\pi_i` (about 3e-14 at a budget of
  200,000). So the draw is bitwise the same run to run, and on CPU and CUDA
  given the same float64 inputs. (The Morton order reads the canonical frame,
  which devices compute to roundoff, so a centroid on a grid line could still
  order differently.)

Everything is tensor arithmetic without a data-dependent shape, so the draw
traces into one graph under ``torch.compile(fullgraph=True, dynamic=True)``.
It synchronizes with the host only in the canonical frame's 3 x 3 eigensolve
(:func:`~physicsnemo.experimental.models.isla.tree_read.canonical_frame`).
"""

import torch
from jaxtyping import Float, Int

from physicsnemo.experimental.models.isla.tree_read import (
    canonical_frame,
    frame_coordinates,
)
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import EFFECTIVE_MEASURE_KEY
from physicsnemo.mesh.geometry._cell_areas import compute_cell_areas
from physicsnemo.mesh.spatial.bvh import _compute_morton_codes
from physicsnemo.mesh.utilities._gather import gather_rows


@torch.library.custom_op("physicsnemo::isla_cumsum", mutates_args=())
def _cumsum(x: torch.Tensor) -> torch.Tensor:
    r"""``x.cumsum(0)``, run under torch.compile as in eager: ATen's for integers, a
    doubling (Kogge-Stone) scan of elementwise adds for floats.

    The draw's sums run over every full-resolution cell, an axis long enough for
    Inductor to emit a split scan; under the whole-model compile with
    ``mode="max-autotune-no-cudagraphs"`` that kernel failed on GB300 (an illegal
    memory access, or "operation not supported on global/shared address space";
    torch 2.12 and 2.14.1). As an opaque op the scan runs eagerly, so the compiled
    draw is the eager one, bitwise. Integer sums are exact in any order. A float sum
    depends on its order, and ATen's CUDA float scan is not run-to-run deterministic;
    in the doubling scan every partial sum is a fixed binary tree of correctly rounded
    adds, so its bits are the same run to run and on every device, within about
    :math:`\log_2 n` roundings of the exact sum for nonnegative ``x``."""
    if not x.is_floating_point():
        return x.cumsum(0)
    n = x.shape[0]
    y, z = x.clone(), torch.empty_like(x)
    s = 1
    while s < n:
        z[:s] = y[:s]
        torch.add(y[s:], y[: n - s], out=z[s:])
        y, z = z, y
        s *= 2
    return y


@_cumsum.register_fake
def _(x):
    return torch.empty_like(x)


def _systematic_positions(
    pi: Float[torch.Tensor, " n"], k: int, start: Float[torch.Tensor, ""], budget: int
) -> Int[torch.Tensor, " k"]:
    r"""The interval hit by each point :math:`u + m`, :math:`m < k`, of the line of
    intervals of lengths ``pi`` laid end to end, in exact int64 fixed point.

    The line is in units of :math:`2^{-Q}`, :math:`Q = 62 -` ``budget.bit_length()``,
    ``budget`` a Python int at least the sum of ``pi`` (so the line fits in int64):
    each interval :math:`\operatorname{round}(\pi_i 2^Q)` units, hit :math:`m` the
    integer :math:`U + m 2^Q`, :math:`U = \lfloor u 2^Q \rfloor`. The intervals are
    half-open: a hit on a boundary takes the later one. The line's end :math:`T`
    differs from :math:`k 2^Q` by the rounding residual :math:`D`; :math:`U` is
    clamped to :math:`[\max(0, D), \min(2^Q, 2^Q + D) - 1]`, so every hit lies on the
    line and every window of :math:`2^Q` units on it (an interval with
    :math:`\pi_i = 1`) holds exactly one hit, whenever :math:`|D| < 2^Q`.
    """
    Q = 62 - budget.bit_length()
    line = _cumsum((pi * 2.0**Q).round().to(torch.int64))
    ### D = T - k 2**Q in shifts by Q and small integers only: Inductor folds an
    ### int64 constant times an index or size into an int32 Triton expression, which
    ### Triton rejects.
    T = line[-1]
    D = (((T >> Q) - k) << Q) + (T - ((T >> Q) << Q))
    U = torch.floor(start * 2.0**Q).to(torch.int64)
    U = torch.maximum(torch.minimum(U, D.clamp_max(0) + ((1 << Q) - 1)), D.clamp_min(0))
    ### line_j > U + m 2**Q  <=>  (line_j - U - 1) >> Q >= m, so the first interval
    ### past hit m is a search for m among these hit counts (nondecreasing), without
    ### the product m 2**Q.
    count = (line - U - 1) >> Q
    hit = torch.arange(k, device=pi.device, dtype=torch.int64)
    return torch.searchsorted(count, hit).clamp_max(pi.shape[0] - 1)


def water_fill(
    counts: Float[torch.Tensor, " conditions"], budget: int
) -> Float[torch.Tensor, " conditions"]:
    """The budget's share of each condition, ``min(counts, level)``.

    ``level`` is the water level of the budget over the conditions: equal shares,
    a condition with at most its share of cells kept whole and its unused share
    split equally over the rest, iterated to the fixed point (at most one
    iteration per condition). The shares sum to ``budget`` when the conditions
    have more cells than that in total; otherwise every condition is whole.
    """
    n = counts.shape[0]
    level = counts.new_full((), budget / n)
    for _ in range(n):
        whole = counts <= level
        rest = (~whole).sum()
        level = torch.where(
            rest > 0,
            (budget - torch.where(whole, counts, 0).sum()) / rest.clamp_min(1),
            level,
        )
    return torch.minimum(counts, level)


def pps_inclusion_probabilities(
    measures: Float[torch.Tensor, " cells"], share: Float[torch.Tensor, ""]
) -> Float[torch.Tensor, " cells"]:
    r"""Inclusion probabilities proportional to size, capped at one, summing to ``share``.

    :math:`\pi_i = 1` for the :math:`m` largest measures and :math:`\pi_i = c\,a_i`
    for the rest, with :math:`c = (b - m) / \sum_{\text{rest}} a_i`; :math:`m` is the
    smallest count with :math:`c\,a_{(m+1)} < 1` (measures sorted in decreasing
    order), which is the fixed point of the iterative adjustment (cap, rescale the
    rest, repeat). ``share`` at least the cell count gives every :math:`\pi_i = 1`.
    In float64. The sort is stable, so ties resolve the same on every device, and the
    tail sums of the sorted measures are a fixed-order scan (:func:`_cumsum`), so the
    probabilities are bitwise the same on every device.
    """
    n = measures.shape[0]
    a, order = measures.sort(descending=True, stable=True)
    tail = _cumsum(a.flip(0)).flip(0)  # sum of the measures from rank m on
    rank = torch.arange(n, device=a.device, dtype=a.dtype)
    uncapped = (share - rank) * a < tail
    m = torch.where(uncapped.any(), uncapped.to(torch.int8).argmax(), n)
    ### (a gather, not tail[m]: indexing by a 0-d tensor would read it on the host)
    tail_m = tail.gather(0, m.clamp_max(n - 1).reshape(1))[0]
    c = (share - m) / tail_m.clamp_min(torch.finfo(a.dtype).tiny)
    pi = torch.where(rank < m, 1.0, (c * a).clamp(max=1.0))
    ### share >= n keeps every cell: float tails can leave one a rounding below 1.
    pi = torch.where(share >= n, 1.0, pi)
    return torch.empty_like(pi).scatter_(0, order, pi)


def draw_inputs(mesh: Mesh) -> tuple[Float[torch.Tensor, " cells"], Float[torch.Tensor, "cells dim"]]:
    """A condition's cell measures and centroids for :func:`sample_boundary_cells`, in
    float64 from the mesh's points as given: its effective measures when stored, else
    the cells' geometric measures, and the vertex means. Never the mesh's cached
    geometry, which is in the mesh's precision."""
    with torch.no_grad():
        verts = gather_rows(mesh.points.detach().double(), mesh.cells)
        measure = mesh.cell_data.get(EFFECTIVE_MEASURE_KEY, None)
        if measure is None:
            measure = compute_cell_areas(verts[:, 1:] - verts[:, :1])
        return measure.detach().double(), verts.mean(dim=1)


def sample_boundary_cells(
    measures: list[Float[torch.Tensor, " cells"]],
    centroids: list[Float[torch.Tensor, "cells dim"]],
    budget: int,
    start: Float[torch.Tensor, ""],
) -> tuple[Int[torch.Tensor, " tokens"], Float[torch.Tensor, " tokens"]]:
    r"""The boundary token sample of one case: kept cells and their weights.

    ``measures`` and ``centroids`` hold each named condition's cells, in the
    order their tokens are concatenated, in float64 (:func:`draw_inputs`);
    ``budget`` is the total token budget, a Python int (it fixes the line's grid), and
    ``start`` the systematic draw's start :math:`u \in [0, 1)` (a float64 scalar).
    Returns the indices of the kept cells into the concatenation of the
    conditions' cells, in increasing order, ``(k,)`` with :math:`k = \min(\sum_c N_c,` ``budget``
    :math:`)`, and their Horvitz-Thompson measure weights :math:`a_i / \pi_i`
    ``(k,)``, in the dtype of the measures, with the float64 :math:`\pi_i`. The tail
    sums are a fixed-order float64 scan and the line and its hits exact int64 fixed
    point (:func:`_systematic_positions`), so the draw is bitwise the same run to run
    and on every device for the same order.
    See the module docstring.
    """
    with torch.no_grad():
        counts = torch.stack([m.new_full((), m.shape[0], dtype=torch.float64) for m in measures])
        shares = water_fill(counts, budget)
        ### The Morton grid lies in the boundary's own frame over all its cells, so a
        ### rigid motion of the case leaves the line, and the draw, unchanged (up to
        ### roundoff and the frame's ties).
        frame = canonical_frame(torch.cat(centroids).detach(), torch.cat(measures).detach())
        pis, orders, offset = [], [], 0
        for i, (a, x) in enumerate(zip(measures, centroids)):
            n_i = a.shape[0]
            if n_i == 0:
                continue
            pis.append(pps_inclusion_probabilities(a.detach().double(), shares[i]))
            ### Morton order of the centroids on cubic cells, ties broken by index.
            code = _compute_morton_codes(frame_coordinates(x.detach(), frame), isotropic=True)
            orders.append(code.argsort(stable=True) + offset)
            offset = offset + n_i
        pi = torch.cat(pis)
        order = torch.cat(orders)
        total = sum(m.shape[0] for m in measures)
        ### k = min(total, budget) in integer division: 1 // (total // budget + 1)
        ### is 1 when every cell is kept, else 0. A symbolic Min would be fixed to
        ### one side of the budget when Inductor's FX-graph cache replays the guards
        ### that hold it (evaluated with the builtin min), a recompile on the other.
        k = budget + (total - budget) * (1 // (total // budget + 1))
        position = _systematic_positions(pi[order], k, start, budget)
        ### In the cells' own order, so a condition kept whole is its mesh as given.
        kept = order[position].sort().values
        pi_kept = pi[kept]
    measure = torch.cat(measures)
    return kept, measure[kept] / pi_kept.to(measure.dtype)
