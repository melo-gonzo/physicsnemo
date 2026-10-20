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

"""The boundary-token sample of the interior task (``isla.sampling``).

The budget is water-filled over the named conditions, each condition's cells are
drawn proportional to their measure (capped at one) by systematic sampling along
the Morton order of their centroids, and the kept cells carry Horvitz-Thompson
weights. These tests pin the exact counts, the inclusion probabilities, the
unbiasedness of the measure sums, the conditions kept whole, the spatial spread,
and one compiled graph for every cell count.
"""

import pytest
import torch

from physicsnemo.experimental.models.isla.sampling import (
    pps_inclusion_probabilities,
    sample_boundary_cells,
    water_fill,
)

D = torch.float64


def _conditions(sizes, seed=0):
    g = torch.Generator().manual_seed(seed)
    measures = [torch.rand(n, generator=g, dtype=D) ** 3 + 1e-3 for n in sizes]
    centroids = [torch.randn(n, 3, generator=g, dtype=D) for n in sizes]
    return measures, centroids


def _per_condition(kept, weights, sizes):
    bounds = torch.tensor(sizes).cumsum(0)[:-1]
    cid = torch.bucketize(kept, bounds, right=True)
    counts = torch.zeros(len(sizes), dtype=torch.long).index_add(0, cid, torch.ones_like(kept))
    sums = torch.zeros(len(sizes), dtype=weights.dtype).index_add(0, cid, weights)
    return counts, sums


def test_water_fill_shares_the_budget():
    """Equal shares; a condition below its share is whole and its unused share goes
    to the others; the shares sum to the budget, or every condition is whole."""
    c = lambda *v: torch.tensor(v, dtype=D)  # noqa: E731
    assert torch.equal(water_fill(c(10, 1000, 30, 5000), 400), c(10, 180, 30, 180))
    assert torch.equal(water_fill(c(10**6), 200_000), c(200_000))
    assert torch.equal(water_fill(c(10**6, 10**6, 10**6, 10**6), 200_000), c(*[50_000] * 4))
    assert torch.equal(water_fill(c(23_380, 54_853, 291_030, 54_621), 200_000), c(23_380, 54_853, 67_146, 54_621))
    assert torch.equal(water_fill(c(10, 20), 400), c(10, 20))
    assert torch.equal(water_fill(c(0, 500), 100), c(0, 100))
    shares = water_fill(c(3, 7, 400, 401, 9_000), 1_000)
    assert float(shares.sum()) == pytest.approx(1_000) and torch.equal(shares[:2], c(3, 7))


def test_inclusion_probabilities_are_capped_proportional_to_size():
    """pi = min(1, c a) with sum pi equal to the share: the largest cells are capped
    at one, the rest proportional to their measure; a share of at least the count
    keeps every cell."""
    g = torch.Generator().manual_seed(0)
    a = torch.rand(1000, generator=g, dtype=D) ** 4
    a[:5] = 50.0
    pi = pps_inclusion_probabilities(a, torch.tensor(100.0, dtype=D))
    assert float(pi.sum()) == pytest.approx(100.0, rel=1e-12)
    capped = pi == 1
    assert bool(capped[:5].all()) and int(capped.sum()) == 5
    ratio = pi[~capped] / a[~capped]
    assert torch.allclose(ratio, ratio[0].expand_as(ratio), rtol=1e-12)
    assert bool((ratio[0] * a[capped] >= 1).all())
    assert torch.equal(pps_inclusion_probabilities(a, torch.tensor(1000.0, dtype=D)), torch.ones_like(a))


@pytest.mark.parametrize("budget", [1, 77, 300, 3539, 3540, 10**6])
def test_the_count_is_exact(budget):
    """Exactly min(total cells, budget) distinct cells, in increasing order."""
    sizes = [500, 40, 3000]
    measures, centroids = _conditions(sizes)
    for u in (0.0, 0.37, 0.999):
        kept, w = sample_boundary_cells(measures, centroids, budget, torch.tensor(u, dtype=D))
        assert kept.shape == w.shape == (min(sum(sizes), budget),)
        assert torch.equal(kept, kept.unique())


def test_horvitz_thompson_measure_sums_are_unbiased():
    """Over random starts each condition's sum of token weights is unbiased for its
    total measure (within four standard errors over 2,000 draws), the condition kept
    whole (40 cells under a share of 100) exactly; every cell's inclusion frequency
    matches its probability."""
    sizes = [500, 40, 3000]
    measures, centroids = _conditions(sizes)
    budget, draws = 300, 2_000
    shares = water_fill(torch.tensor(sizes, dtype=D), budget)
    pi = torch.cat([pps_inclusion_probabilities(a, s) for a, s in zip(measures, shares)])
    total = torch.stack([a.sum() for a in measures])
    torch.manual_seed(0)
    sums, hits = [], torch.zeros(sum(sizes), dtype=D)
    for _ in range(draws):
        kept, w = sample_boundary_cells(measures, centroids, budget, torch.rand((), dtype=D))
        counts, s = _per_condition(kept, w, sizes)
        assert int(counts[1]) == 40
        assert float(s[1]) == pytest.approx(float(total[1]), rel=1e-12)
        sums.append(s)
        hits[kept] += 1
    sums = torch.stack(sums)
    se = sums.std(0) / draws**0.5
    assert bool(((sums.mean(0) - total).abs() <= 4 * se + 1e-9 * total).all())
    ### Inclusion frequencies: a binomial count per cell, within five of its standard
    ### deviations plus two counts (the rare cells' counts are small integers).
    sd = (draws * pi * (1 - pi)).sqrt()
    assert bool(((hits - draws * pi).abs() <= 5 * sd + 2).all())


def test_the_sample_is_spread_over_the_surface():
    """Along the Morton order the kept cells are evenly spaced in cumulative
    probability, so a uniform measure on a grid gives tokens in every octant of the
    box in proportion to its cells (to one token), unlike an unstratified draw that
    could leave one empty."""
    g = torch.arange(16, dtype=D)
    xyz = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    a = torch.ones(xyz.shape[0], dtype=D)
    for u in (0.1, 0.5, 0.9):
        kept, _ = sample_boundary_cells([a], [xyz], 64, torch.tensor(u, dtype=D))
        octant = ((xyz[kept] >= 8).long() * torch.tensor([4, 2, 1])).sum(-1)
        assert torch.equal(torch.bincount(octant, minlength=8), torch.full((8,), 8))


def test_an_empty_condition_takes_no_tokens():
    """A condition without cells (a case without slip walls) is skipped."""
    measures, centroids = _conditions([0, 50, 80])
    kept, w = sample_boundary_cells(measures, centroids, 60, torch.tensor(0.5, dtype=D))
    assert kept.shape == (60,) and bool((kept < 130).all())
    counts, _ = _per_condition(kept, w, [0, 50, 80])
    assert counts.tolist() == [0, 30, 30]


def test_one_compiled_graph_for_every_cell_count():
    """Under torch.compile(fullgraph=True, dynamic=True) the draw is one graph whether
    the cells exceed the budget or not, and matches eager."""

    def draw(a1, a2, x1, x2, h1, h2, u):
        kept, w = sample_boundary_cells([a1, a2], [x1, x2], 300, u)
        return (torch.cat([h1, h2])[kept] * w[:, None]).sum(0), kept

    torch._dynamo.reset()
    compiled = torch.compile(draw, fullgraph=True, dynamic=True)
    for i, (n1, n2) in enumerate([(500, 40), (700, 90), (100, 50), (2000, 3), (150, 140)]):
        g = torch.Generator().manual_seed(i)
        args = (
            torch.rand(n1, generator=g), torch.rand(n2, generator=g),
            torch.randn(n1, 3, generator=g), torch.randn(n2, 3, generator=g),
            torch.randn(n1, 4, generator=g), torch.randn(n2, 4, generator=g),
            torch.tensor(0.5, dtype=D),
        )
        with torch._dynamo.config.patch(error_on_recompile=i > 0):
            out, kept = compiled(*args)
        ref, ref_kept = draw(*args)
        assert torch.equal(kept, ref_kept)
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_the_length_unit_is_the_full_boundary_s_for_every_draw():
    """An interior model without declared scales takes its length unit from every
    cell of the full-resolution boundary before the draw: two training draws of 60
    of 300 cells (different starts, different tokens) read the same unit, the
    measure-weighted RMS pairwise distance of all the cells."""
    from physicsnemo.experimental.models import ISLA
    from physicsnemo.experimental.models.isla.geometry import rms_pairwise_distance
    from test.experimental.models.isla._meshes import cloud, soup

    torch.manual_seed(0)
    m = ISLA(
        output_schema={"p": {"rank": 0}},
        boundary_schemas={"a": {}, "b": {}},
        interior=True,
        boundary_tokens=60,
        hidden=16,
        n_layers=1,
        n_slices=4,
    ).double()
    assert not m._scaled
    seen, forward_tokens = [], m._forward_tokens

    def spy(boundary, *a, **k):
        seen.append((k["gauge"], boundary["points"].clone()))
        return forward_tokens(boundary, *a, **k)

    m._forward_tokens = spy
    meshes = {k: soup(*cloud(150, seed=i)[:3]) for i, k in enumerate("ab")}
    q = torch.randn(10, 3, dtype=D, generator=torch.Generator().manual_seed(5))
    for seed in (0, 1):
        torch.manual_seed(seed)
        with torch.no_grad():
            m(boundary_meshes=meshes, prediction_points=q)
    full = rms_pairwise_distance(
        torch.cat([meshes[k].cell_centroids for k in "ab"])[None],
        torch.cat([meshes[k].cell_data["_effective_measure"] for k in "ab"])[None],
    )
    (g0, p0), (g1, p1) = seen
    assert p0.shape[1] == 60 and not torch.equal(p0, p1)  # two different draws
    assert torch.equal(g0, g1)
    torch.testing.assert_close(g0, full, atol=0, rtol=1e-14)
