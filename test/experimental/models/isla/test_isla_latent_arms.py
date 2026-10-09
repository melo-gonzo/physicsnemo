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

"""Contract tests for two diagnostic read-only arms (the latent-capacity probes of the
SMART comparison, 2026-10-09): the global read over a Morton-strided subsample of the
boundary cells (``kernel_read_cells``) and the control in which every query block reads
the final boundary state instead of the state at its own depth (``read_final_boundary``).
Both are read-only arms, so query independence must hold exactly; the subsample read
over every cell must be the global read; every parameter must receive a gradient; the
subsample in training mode moves its stride offset between forward passes.
"""

import pytest
import torch

from physicsnemo.experimental.models import ISLA
from test.experimental.models.isla._meshes import cloud, outputs, points_mesh, same, soup

PV = {"p": {"rank": 0}, "v": {"rank": 1}}
G = {"g": {"rank": 1}}
QS = {"sdf": {"rank": 0}, "sdf_normals": {"rank": 1}}
D = torch.float64


def _model(theta=None, cells=None, final=False, read=False, seed=0, **kw):
    """A small read-only model; with ``read`` (or a theta or a cell count) it carries
    the global read, dense (``theta=None``) or through the tree (``theta=0``)."""
    torch.manual_seed(seed)
    read = (
        dict(kernel_read="mean", kernel_read_radius=None, kernel_read_theta=theta, kernel_read_cells=cells)
        if read or theta is not None or cells is not None
        else {}
    )
    m = ISLA(
        output_schema=PV,
        global_schema=G,
        query_schema=QS,
        interior=True,
        query_independent=True,
        read_final_boundary=final,
        hidden=32,
        n_layers=2,
        n_slices=8,
        dim=3,
        **read,
        **kw,
    )
    if m.kernel_reads is not None:
        with torch.no_grad():
            for p in [*m.kernel_reads.parameters(), *m.kernel_basis.parameters()]:
                p.normal_(0.0, 0.3)
    return m.double().eval()


def _case(n=300, nq=120, seed=0):
    pts, nrm, w, drv = cloud(n, d=3, seed=seed)
    g = torch.Generator().manual_seed(seed + 1)
    q = torch.cat([pts[: nq - 10] + 0.15 * torch.randn(nq - 10, 3, generator=g, dtype=D), 40.0 + torch.randn(10, 3, generator=g, dtype=D)])
    sdf = torch.randn(nq, generator=g, dtype=D) * 0.3
    sn = torch.nn.functional.normalize(torch.randn(nq, 3, generator=g, dtype=D), dim=-1)
    return soup(pts, nrm, w), drv, q, sdf, sn


def _run(m, mesh, drv, q, sdf, sn, grad=False):
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        return m(
            boundary_meshes={"wall": mesh},
            prediction_points=points_mesh(q, {"sdf": sdf, "sdf_normals": sn}),
            global_data={"g": drv},
        )


def _rows(out, idx):
    return {k: v[idx] for k, v in out.items()}


def _independent(m, mesh, drv, q, sdf, sn, atol=1e-11):
    full = outputs(_run(m, mesh, drv, q, sdf, sn))
    idx = torch.tensor([3, 50, 117])
    assert same(outputs(_run(m, mesh, drv, q[idx], sdf[idx], sn[idx])), _rows(full, idx), atol=atol)
    perm = torch.randperm(q.shape[0], generator=torch.Generator().manual_seed(5))
    assert same(outputs(_run(m, mesh, drv, q[perm], sdf[perm], sn[perm])), _rows(full, perm), atol=atol)
    half = q.shape[0] // 2
    assert same(outputs(_run(m, mesh, drv, q[:half], sdf[:half], sn[:half])), _rows(full, torch.arange(half)), atol=atol)
    return full


@pytest.mark.parametrize("theta", [None, 0.0])
def test_subsample_read_over_every_cell_is_the_global_read(theta):
    """With as many cells as the boundary has, the subsample is the whole boundary in
    Morton order and the read equals the dense global read (a softmax over the same
    set of cells) to roundoff, in both the dense and the tree evaluation."""
    dense = _model(read=True)
    sub = _model(theta=theta, cells=300)
    sub.load_state_dict(dense.state_dict())
    mesh, drv, q, sdf, sn = _case()
    a = outputs(_run(dense, mesh, drv, q, sdf, sn))
    b = outputs(_run(sub, mesh, drv, q, sdf, sn))
    assert same(b, a, atol=1e-10)
    assert sub.kernel_read_pairs == dense.kernel_read_pairs == 120 * 300


def test_subsample_read_is_exactly_query_independent_and_reads_the_subsample_only():
    """A point's prediction depends on its point and the boundary only; the read holds
    exactly ``kernel_read_cells`` cells per point."""
    m = _model(theta=0.0, cells=64)
    mesh, drv, q, sdf, sn = _case()
    _independent(m, mesh, drv, q, sdf, sn)
    _run(m, mesh, drv, q, sdf, sn)
    assert m.kernel_read_pairs == 120 * 64
    assert m.kernel_read_cells == 64


def test_subsample_read_moves_its_offset_in_training_and_is_fixed_in_evaluation():
    """In training the stride offset is drawn per forward pass (eight passes at a stride
    of four are not all equal); in evaluation it is zero (two passes are identical)."""
    m = _model(theta=0.0, cells=75)
    mesh, drv, q, sdf, sn = _case()
    e1 = outputs(_run(m, mesh, drv, q, sdf, sn))
    e2 = outputs(_run(m, mesh, drv, q, sdf, sn))
    assert same(e2, e1, atol=0.0)
    m.train()
    torch.manual_seed(0)
    outs = [outputs(_run(m, mesh, drv, q, sdf, sn))["p"] for _ in range(8)]
    assert any(not torch.equal(o, outs[0]) for o in outs[1:])
    m.eval()
    assert same(outputs(_run(m, mesh, drv, q, sdf, sn)), e1, atol=0.0)


def test_subsample_read_refuses_more_cells_than_the_boundary_has():
    m = _model(theta=0.0, cells=301)
    mesh, drv, q, sdf, sn = _case()
    with pytest.raises(ValueError, match="kernel_read_cells"):
        _run(m, mesh, drv, q, sdf, sn)


def test_read_final_boundary_is_exactly_query_independent_and_differs_from_depth_matched():
    """The control shares every parameter with the depth-matched model, is exactly
    query independent, and gives different predictions (the query blocks read the
    final boundary state's slices rather than the state at their own depth)."""
    ref = _model()
    ctl = _model(final=True)
    assert not ctl.load_state_dict(ref.state_dict(), strict=True).missing_keys
    mesh, drv, q, sdf, sn = _case()
    full = _independent(ctl, mesh, drv, q, sdf, sn)
    base = outputs(_run(ref, mesh, drv, q, sdf, sn))
    assert not same(full, base, atol=1e-6)


def test_read_final_boundary_with_a_read_reads_the_final_boundary_state():
    """With a kernel read as well, the control still runs and stays query independent."""
    m = _model(theta=0.0, cells=64, final=True)
    mesh, drv, q, sdf, sn = _case()
    _independent(m, mesh, drv, q, sdf, sn)


@pytest.mark.parametrize("arm", ["cells", "final"])
def test_every_parameter_receives_a_gradient(arm):
    m = (_model(theta=0.0, cells=64) if arm == "cells" else _model(final=True)).train()
    mesh, drv, q, sdf, sn = _case()
    out = _run(m, mesh, drv, q, sdf, sn, grad=True)
    loss = sum(v.square().sum() for v in outputs(out).values())
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


def test_arms_are_validated_at_construction():
    with pytest.raises(ValueError, match="kernel_read_cells"):
        ISLA(output_schema=PV, global_schema=G, interior=True, query_independent=True, kernel_read_cells=64, hidden=32, n_layers=1, n_slices=8)
    with pytest.raises(ValueError, match="kernel_read_cells"):
        ISLA(output_schema=PV, global_schema=G, interior=True, query_independent=True, kernel_read="mean", kernel_read_radius=None, kernel_read_cells=0, hidden=32, n_layers=1, n_slices=8)
    with pytest.raises(ValueError, match="read_final_boundary"):
        ISLA(output_schema=PV, global_schema=G, interior=True, read_final_boundary=True, hidden=32, n_layers=1, n_slices=8)
