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
  found exactly from the sorted measures: the library's
  ``_pps_inclusion_probabilities`` in
  :mod:`physicsnemo.mesh.sampling.systematic_sampling`); a
  condition kept whole has every :math:`\pi_i = 1`. A share past the count of
  positive measures keeps every positive cell, and the cells of zero measure fill
  the rest of the share equally (as
  :func:`~physicsnemo.mesh.sampling.sample_cells_systematic` does), at
  Horvitz-Thompson weight 0.
- **Draw.** Systematic sampling
  (:func:`~physicsnemo.mesh.sampling.systematic_sampling._systematic_sample`): the
  cells are laid along a line, each condition's cells in the Morton (Z-order) order
  of their centroids in the boundary's canonical frame
  (:func:`~physicsnemo.mesh.spatial._frame.canonical_frame` over every cell), each
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
  sums of the measures are a fixed-order float64 scan (the library's
  ``physicsnemo::mesh_cumsum``), the same bits on every device. The line is exact
  int64 arithmetic in units of :math:`2^{-Q}`, :math:`Q = 62 - b`, :math:`b` the
  bit length of the budget (a Python int), each cell an interval of
  :math:`\operatorname{round}(\pi_i 2^Q)` units; a cell with :math:`\pi_i = 1` is
  exactly :math:`2^Q` units, and the start is clamped by the line's rounding
  residual, so such a cell holds exactly one hit. The drawn inclusion probability
  is that length over :math:`2^Q`, within :math:`2^{-(Q+1)}` of :math:`\pi_i`
  (about 3e-14 at a budget of 200,000). So the draw is bitwise the same run to
  run, and on CPU and CUDA given the same float64 inputs. (The Morton order reads
  the canonical frame, which devices compute to roundoff, so a centroid on a grid
  line could still order differently.)

Everything is tensor arithmetic without a data-dependent shape, so the draw
traces into one graph under ``torch.compile(fullgraph=True, dynamic=True)``.
It synchronizes with the host only in the canonical frame's 3 x 3 eigensolve
(:func:`~physicsnemo.mesh.spatial._frame.canonical_frame`).
"""

import torch
from jaxtyping import Float, Int

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import EFFECTIVE_MEASURE_KEY
from physicsnemo.mesh.geometry._cell_areas import compute_cell_areas
from physicsnemo.mesh.sampling.systematic_sampling import _pps_inclusion_probabilities, _systematic_sample
from physicsnemo.mesh.spatial._frame import canonical_frame, frame_coordinates
from physicsnemo.mesh.spatial.bvh import _compute_morton_codes
from physicsnemo.mesh.utilities._gather import gather_rows


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
    point (:func:`~physicsnemo.mesh.sampling.systematic_sampling._systematic_sample`),
    so the draw is bitwise the same run to run and on every device for the same order.
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
            a = a.detach().double()
            pi_i = _pps_inclusion_probabilities(a, shares[i])
            ### A share past the positive measures keeps every positive cell, and the
            ### zero-measure cells fill the rest of it equally, at weight 0 (as
            ### sample_cells_systematic does); without a zero measure pi_i is as is.
            zero = a == 0
            n_pos = n_i - zero.sum()
            pis.append(
                torch.where(
                    zero, ((shares[i] - n_pos) / (n_i - n_pos).clamp_min(1)).clamp_min(0), pi_i
                )
            )
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
        ### In the cells' own order, so a condition kept whole is its mesh as given.
        kept = _systematic_sample(pi, k, start, order=order, max_n_samples=budget)
        pi_kept = pi[kept]
    measure = torch.cat(measures)
    return kept, measure[kept] / pi_kept.to(measure.dtype)
