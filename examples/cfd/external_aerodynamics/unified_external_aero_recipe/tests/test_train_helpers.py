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

"""Unit tests for `src/train.py`'s private TensorDict-aware helpers and for `src/output_normalize.py`.

``TensorDict`` is not a ``dict`` subclass, so the bare
``isinstance(obj, dict)`` branches in the recipe's recursive helpers
must be paired with explicit ``isinstance(obj, TensorDict)`` branches
for TD inputs to be walked at all. These tests pin that explicit
handling for:

- :func:`train._walk_batch_for_logging`: must yield ``(name, tensor)``
  pairs from TensorDict leaves -- including correctly producing dotted
  paths for nested TDs via ``TD.flatten_keys('.')``.
- :func:`output_normalize.normalize_output_to_tensordict`: routes a
  model output (``Mesh`` or ``(B, N, C)`` tensor) to a per-target
  TensorDict, with clear error messages on shape / channel-count
  mismatches.
- :func:`train._reduce_and_average`: averages rank-local loss / metric
  sums over the global sample count (used per epoch); its
  single-process path must equal plain ``total_loss / n`` + per-leaf
  ``sum / n`` averaging.
- :func:`train._run_epoch`: each step's values are read back without a
  host sync (:func:`train._start_step_readback`), and every step is logged
  once, in order, with the values the blocking per-step readback gave.

(The analogous tests for the shared, tensorboard-free
:func:`utils.recursive_to_device` live in ``test_utils.py``, outside
this module's tensorboard skip guard.)
"""

from __future__ import annotations

import logging
import os
import socket
import time
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from tensordict import TensorDict

### `train.py` imports `torch.utils.tensorboard.SummaryWriter` at module
### load, which transitively requires the `tensorboard` package. That
### dep is not declared in pyproject.toml; CI / training environments
### have it installed, but bare dev sandboxes might not. Skip cleanly.
### `output_normalize` itself is tensorboard-free, so we import it
### directly (no skip).
pytest.importorskip("tensorboard")

import isla_chunked  # noqa: E402  -- monkeypatch module globals in focused tests
import train  # noqa: E402  -- monkeypatch module globals in focused tests
from output_normalize import normalize_output_to_tensordict  # noqa: E402
from train import (  # noqa: E402  -- after the skip guard
    _reduce_and_average,
    _run_epoch,
    _walk_batch_for_logging,
)

from physicsnemo.mesh import Mesh  # noqa: E402  -- after the importorskip guard


def test_forward_pass_preserves_query_measure_in_loss_metrics_and_gradients():
    """Training weights each query by its measure through output normalization."""
    predictions = torch.tensor([[[1.0], [3.0]]], requires_grad=True)
    target_config = {"pressure": "scalar"}
    batch = {
        "forward_kwargs": {"input": predictions},
        "targets": TensorDict({"pressure": torch.ones(1, 2)}, batch_size=[1, 2]),
        "target_measure": torch.tensor([[3.0, 1.0]]),
    }

    loss, losses, metrics = train.forward_pass(
        batch,
        torch.nn.Identity(),
        "float32",
        train.LossCalculator(target_config, loss_type="mse"),
        train.MetricCalculator(target_config, metrics=["mae"]),
        output_type="tensors",
        target_config=target_config,
    )

    # The erroneous query occupies one quarter of the measure: MSE=1, MAE=0.5.
    assert loss.item() == pytest.approx(1.0)
    assert losses["loss/pressure"].item() == pytest.approx(1.0)
    assert metrics["pressure_mae"].item() == pytest.approx(0.5)
    loss.backward()
    torch.testing.assert_close(predictions.grad, torch.tensor([[[0.0], [1.0]]]))


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
def test_isla_training_on_thin_triangles(device):
    """Thin faces keep positive measures through ISLA's mesh forward and the weighted
    loss; the boundary Mesh carries the corrected measure ISLA reads."""
    from collate import build_collate_fn

    from physicsnemo.datapipes.transforms.mesh import MeshToDomainMesh
    from physicsnemo.experimental.models.isla import ISLA
    from physicsnemo.mesh.calculus import cell_measures, scale_measures

    mesh = Mesh(
        points=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1e-4, 0.0],
                [5.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [5.0, 1.0, 0.0],
            ],
            device=device,
        ),
        cells=torch.arange(6, device=device).reshape(2, 3),
        cell_data={"pressure": torch.tensor([1.0, 3.0], device=device)},
        global_data={"U_inf_dir": torch.tensor([1.0, 0.0, 0.0], device=device)},
    )
    scale_measures(mesh, 3.0)
    domain = MeshToDomainMesh(cell_data_targets=["pressure"])(mesh)
    targets = {"pressure": "scalar"}
    batch = build_collate_fn(
        "mesh",
        {
            "boundary_meshes": {"vehicle": "boundaries.vehicle"},
            "prediction_boundaries": {"literal": ["vehicle"]},
            "global_data": "global_data",
        },
        targets,
    )([(domain, {})])
    measure = batch["target_measure"]
    torch.testing.assert_close(measure, measure.new_tensor([0.00015, 1.5]))
    ### The boundary Mesh keeps the same corrected measure in its cell_data, which
    ### is what ISLA reads as the boundary tokens' quadrature measure.
    torch.testing.assert_close(
        cell_measures(batch["forward_kwargs"]["boundary_meshes"]["vehicle"]), measure
    )
    model = ISLA(
        output_schema={"pressure": {"rank": 0}},
        boundary_schemas={"vehicle": {}},
        global_schema={"U_inf_dir": {"rank": 1}},
        hidden=16,
        n_layers=1,
        n_slices=4,
    ).to(device)
    with torch.no_grad():
        prediction = model(**batch["forward_kwargs"]).point_data["pressure"]
        error = prediction - batch["targets"]["pressure"]
        expected = (measure * error.square()).sum() / measure.sum()
    loss, _, _ = train.forward_pass(
        batch,
        model,
        "float32",
        train.LossCalculator(targets, loss_type="mse"),
        train.MetricCalculator(targets, metrics=["mae"]),
        output_type="mesh",
        target_config=targets,
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any((g != 0).any() for g in gradients)


@pytest.mark.parametrize(
    "loss_type,measured",
    [
        ("mse", True),
        ("huber", False),
        ("relative_l2", True),
        ("case_relative_l2", True),
    ],
)
def test_chunked_forward_backward_equals_the_unchunked_step(
    loss_type, measured, monkeypatch
):
    """ISLA-DTT's chunked step in the fewest chunks that fit (one pass for a
    pointwise loss; otherwise one pass over one chunk of every point when it fits,
    else two passes): the loss, its fields, the metrics and every parameter's
    gradient of forward_pass and its backward, from the same token draw."""
    from physicsnemo.experimental.models.isla import model as isla_model

    model, batch, target_config = _isla_dtt_case(50, measured)
    n = 50
    lc = train.LossCalculator(target_config, loss_type=loss_type)
    mc = train.MetricCalculator(target_config, metrics=["mae"])
    torch.manual_seed(1)
    ref = train.forward_pass(
        batch, model, "float32", lc, mc, output_type="mesh", target_config=target_config
    )
    ref[0].backward()
    grads = {k: p.grad.clone() for k, p in model.named_parameters()}
    scale = max(float(v.abs().max()) for v in grads.values())
    call = isla_model._QueryChunks.__call__
    sizes = []

    def recorded(self, start, stop):
        sizes.append(stop - start)
        return call(self, start, stop)

    monkeypatch.setattr(isla_model._QueryChunks, "__call__", recorded)
    ### One GiB per point and no margin: the capacity is the most points that fit.
    monkeypatch.setitem(isla_chunked._CHUNK_GIB, "float32", (0.0, 1e5, 0.0))

    def check(capacity_gib=None):
        sizes.clear()
        model.zero_grad(set_to_none=True)
        torch.manual_seed(1)
        out = isla_chunked.chunked_forward_backward(
            batch,
            model,
            "float32",
            lc,
            mc,
            target_config=target_config,
            capacity_gib=capacity_gib,
        )
        for a, b in zip(out, ref):
            torch.testing.assert_close(a, b.detach(), rtol=1e-5, atol=1e-7)
        for k, p in model.named_parameters():
            torch.testing.assert_close(
                p.grad, grads[k], rtol=1e-4, atol=1e-6 * scale, msg=k
            )
        return list(sizes)

    assert check() == check(n) == [n]
    passes = 1 if loss_type in isla_chunked._POINTWISE_LOSSES else 2
    assert check(20) == [17, 17, 16] * passes
    assert check(1.0) == [1] * n * passes
    with pytest.raises(torch.OutOfMemoryError, match="one prediction point"):
        check(0.5)


def _isla_dtt_case(n: int, measured: bool = True, seed: int = 0):
    """A small float64 ISLA-DTT, its batch of ``n`` prediction points (drawn from
    ``seed``) and the target config."""
    from physicsnemo.experimental.models.isla import ISLA

    g = torch.Generator().manual_seed(seed)
    tri = torch.randn(60, 3, 3, generator=g, dtype=torch.float64)
    tri = tri[:, :1] * 2.0 + 0.3 * tri  # small triangles spread over a blob
    wall = Mesh(points=tri.reshape(-1, 3), cells=torch.arange(180).reshape(60, 3))
    targets = TensorDict(
        {
            "pressure": torch.randn(n, generator=g),
            "velocity": torch.randn(n, 3, generator=g),
        },
        batch_size=[n],
    )
    batch = {
        "forward_kwargs": {
            "boundary_meshes": {"vehicle": wall},
            "prediction_points": Mesh(
                points=3.0 * torch.randn(n, 3, generator=g, dtype=torch.float64)
            ),
            "global_data": {
                "U_inf_dir": torch.tensor([1.0, 0.2, 0.0], dtype=torch.float64)
            },
        },
        "targets": targets,
    }
    if measured:
        batch["target_measure"] = torch.rand(n, generator=g) + 0.5
    target_config = {"pressure": "scalar", "velocity": "vector"}
    torch.manual_seed(0)
    model = ISLA(
        output_schema={"pressure": {"rank": 0}, "velocity": {"rank": 1}},
        boundary_schemas={"vehicle": {}},
        global_schema={"U_inf_dir": {"rank": 1}},
        interior=True,
        n_boundary_tokens=40,
        hidden=16,
        n_layers=2,
        n_slices=4,
    ).double()
    with torch.no_grad():
        for p in model.kernel_reads.parameters():
            p.normal_(0.0, 0.3)
    return model, batch, target_config


def test_chunk_rule_decides_from_sizes(monkeypatch, caplog):
    """The predicted peak of ISLA-DTT's step over a chunk of points, from its walk
    pairs, its points and the memory allocated after the boundary side; the fewest
    chunks that fit the capacity, each with its share of the pairs, and the passes
    by loss, logged at the first decision and on change."""
    ### Held-out steps on a GB300 (275.0 GiB free and reserved; DTT-FAST integration
    ### job 370118): a chunk's walk pairs and points, GiB allocated at the decision
    ### and the step's peak reserved GiB, which the prediction covers within the
    ### device.
    for pairs, points, allocated, reserved, precision in (
        (209_094_835, 300_000, 73.06, 178.38, "bfloat16"),  # vehicle, 2 chunks
        (425_319_791, 500_000, 73.06, 251.19, "bfloat16"),  # whole boundary
        (85_131_592, 400_000, 71.76, 182.52, "bfloat16"),  # AeroZoo 800k, two passes
        (138_943_574, 200_000, 110.12, 235.67, "float32"),  # DrivAerML vehicle
        (126_352_902, 150_000, 110.13, 210.54, "float32"),  # whole boundary
        (52_608_716, 250_000, 108.78, 231.31, "float32"),  # AeroZoo
    ):
        peak = isla_chunked._chunk_peak_gib(pairs, points, allocated, precision)
        assert reserved <= peak < 275.0
    ### 20 pairs per point.
    chunks = SimpleNamespace(pair_counts=torch.full((50,), 20), n_points=50)
    one = isla_chunked._chunk_peak_gib(1000, 50, 0.0, "float32")
    third = isla_chunked._chunk_peak_gib(340, 17, 0.0, "float32")
    cpu = torch.device("cpu")

    def spans(pointwise, capacity):
        return isla_chunked._chunk_spans(
            chunks, cpu, False, pointwise, "float32", capacity
        )

    monkeypatch.setattr(isla_chunked, "_chunks_logged", None)
    with caplog.at_level(logging.INFO, logger="training.step"):
        assert spans(True, None) == spans(False, one + 1e-9) == ([(0, 50)], False)
        assert spans(True, one - 1e-6) == ([(0, 25), (25, 50)], False)
        assert spans(False, one - 1e-6) == ([(0, 25), (25, 50)], True)
        assert spans(True, third + 1e-9) == ([(0, 17), (17, 34), (34, 50)], False)
        assert spans(True, third - 1e-6) == (
            [(0, 13), (13, 26), (26, 39), (39, 50)],
            False,
        )
    logged = [r.getMessage() for r in caplog.records if r.name == "training.step"]
    assert len(logged) == 5, logged
    assert "in 1 chunk (" in logged[0] and "2 chunks (" in logged[1]
    assert "2 chunks, two passes" in logged[2]
    assert "3 chunks (" in logged[3] and "4 chunks (" in logged[4]


def test_chunk_rule_counts_each_chunks_pairs(monkeypatch):
    """Points stored in spatial order: the first ten hold 10 pairs each, the rest
    one, so a chunk at the start holds more than its share of the walk's pairs.
    The rule adds chunks until each chunk's exact pairs fit, and raises when not
    even one point fits."""
    ### One GiB per pair, nothing per point, no margin: the capacity is in pairs.
    monkeypatch.setitem(isla_chunked._CHUNK_GIB, "float32", (1e6, 0.0, 0.0))
    counts = torch.tensor([10] * 10 + [1] * 40)
    chunks = SimpleNamespace(pair_counts=counts, n_points=50)
    cpu = torch.device("cpu")

    def spans(pointwise, capacity):
        return isla_chunked._chunk_spans(
            chunks, cpu, False, pointwise, "float32", capacity
        )

    ### By their share (2.8 pairs per point) 21 points fit 60 pairs, so 3 chunks of
    ### 17; the first holds 107 pairs. Chunks of 6 hold at most 60.
    sixes = [(a, min(a + 6, 50)) for a in range(0, 50, 6)]
    assert spans(True, 60.0) == (sixes, False)
    assert spans(False, 60.0) == (sixes, True)
    assert spans(False, 141.0) == ([(0, 50)], False)
    ### Below the first point's 10 pairs; below a point's share.
    for capacity in (9.0, 2.0):
        with pytest.raises(torch.OutOfMemoryError, match="one prediction point"):
            spans(True, capacity)


def _two_ranks_one_over_the_limit(rank: int, store: str, loss_type: str) -> None:
    """Rank ``rank`` of :func:`test_chunked_forward_backward_agrees_across_ranks`."""
    import torch.distributed as dist

    from physicsnemo.experimental.models.isla import model as isla_model

    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=2
    )
    try:
        n = (50, 30)[rank]
        model, batch, target_config = _isla_dtt_case(n, seed=rank)
        lc = train.LossCalculator(target_config, loss_type=loss_type)
        mc = train.MetricCalculator(target_config, metrics=["mae"])
        torch.manual_seed(1)
        ref = train.forward_pass(
            batch,
            model,
            "float32",
            lc,
            mc,
            output_type="mesh",
            target_config=target_config,
        )
        ref[0].backward()
        ref_grads = [p.grad / 2 for p in model.parameters()]
        for g in ref_grads:
            dist.all_reduce(g)
        model.zero_grad(set_to_none=True)
        ### One GiB per point and no margin: room for 100 points on rank 0 and 20 on
        ### rank 1, so chunks of at most 20 on both.
        isla_chunked._CHUNK_GIB["float32"] = (0.0, 1e5, 0.0)
        sizes = []
        call = isla_model._QueryChunks.__call__
        isla_model._QueryChunks.__call__ = lambda self, a, b: (
            sizes.append(b - a) or call(self, a, b)
        )
        torch.manual_seed(1)
        isla_chunked.chunked_forward_backward(
            batch,
            torch.nn.parallel.DistributedDataParallel(model),
            "float32",
            lc,
            mc,
            target_config=target_config,
            capacity_gib=(100.0, 20.0)[rank],
        )
        passes = 1 if loss_type in isla_chunked._POINTWISE_LOSSES else 2
        assert sizes == ([17, 17, 16] if rank == 0 else [15, 15]) * passes, sizes
        for p, g in zip(model.parameters(), ref_grads):
            torch.testing.assert_close(p.grad, g, rtol=1e-4, atol=1e-9)
        torch.optim.SGD(model.parameters(), lr=0.1).step()
        flat = torch.cat(
            [p.grad.reshape(-1) for p in model.parameters()]
            + [p.detach().reshape(-1) for p in model.parameters()]
        )
        both = [torch.empty_like(flat) for _ in range(2)]
        dist.all_gather(both, flat)
        assert torch.equal(both[0], both[1])
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="requires Gloo")
@pytest.mark.parametrize("loss_type", ["mse", "relative_l2"])
def test_chunked_forward_backward_agrees_across_ranks(tmp_path, loss_type):
    """Two DDP ranks (gloo) with different point counts, where every point in one
    chunk fits on rank 0 and not on rank 1: both take chunks of rank 1's largest
    (in two passes under a root loss), with gradients equal to the mean of the
    ranks' unchunked ones and gradients and parameters bitwise equal across the
    ranks."""
    torch.multiprocessing.spawn(
        _two_ranks_one_over_the_limit,
        args=(str(tmp_path / "store"), loss_type),
        nprocs=2,
    )


def _average_gradients_rank(rank: int, store: str, world_size: int) -> None:
    """Rank ``rank`` of :func:`test_average_gradients_equals_the_fused_reduction`."""
    import torch.distributed as dist

    from physicsnemo.distributed import fused_all_reduce

    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=world_size
    )
    try:
        torch.manual_seed(rank)
        shapes = [(), (1,), (7,), (3, 5), (4, 1, 6), (2, 3, 4, 5)]
        lists = [
            [torch.randn(s) for s in shapes],
            [
                torch.randn(s, dtype=d)
                for s in shapes
                for d in (torch.float32, torch.bfloat16)
            ],
            [torch.randn(s, dtype=torch.bfloat16) for s in shapes],
        ]
        for grads in lists:
            expected = [
                r / world_size for r in fused_all_reduce([g.clone() for g in grads])
            ]
            isla_chunked._average_gradients_(grads)
            for g, e in zip(grads, expected):
                assert g.dtype == e.dtype and torch.equal(g, e)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="requires Gloo")
@pytest.mark.parametrize("world_size", [1, 2, 3])
def test_average_gradients_equals_the_fused_reduction(tmp_path, world_size):
    """The chunked step's flat-buffer gradient average is bitwise the per-tensor
    ``r / world_size`` of ``fused_all_reduce`` on every rank, for float32,
    bfloat16 and mixed gradients of many shapes; one rank (fused_all_reduce's
    no-op path) and three (a world size whose division is inexact) included."""
    torch.multiprocessing.spawn(
        _average_gradients_rank,
        args=(str(tmp_path / "store"), world_size),
        nprocs=world_size,
    )


def test_chunked_step_chunks_isla_dtt():
    """ISLA-DTT, the interior task (``interior=True``), takes the chunked step when
    the batch has prediction points; a surface ISLA never chunks."""
    from physicsnemo.experimental.models.isla import ISLA

    def isla(interior):
        return ISLA(
            output_schema={"pressure": {"rank": 0}},
            boundary_schemas={"vehicle": {}},
            interior=interior,
            hidden=16,
            n_layers=1,
            n_slices=4,
        )

    def batch(n):
        return {"forward_kwargs": {"prediction_points": Mesh(points=torch.zeros(n, 3))}}

    assert isla_chunked._chunked_step(isla(True), batch(50))
    assert not isla_chunked._chunked_step(
        isla(True), {"forward_kwargs": {"prediction_boundaries": ["vehicle"]}}
    )
    assert not isla_chunked._chunked_step(isla(False), batch(50))


def test_chunked_forward_backward_refuses_batched_targets():
    """The chunked step slices the targets by point: targets over more than the
    points' axis (a batch of cases) are refused before the model runs."""
    model, batch, target_config = _isla_dtt_case(20)
    batch["targets"] = batch["targets"].unsqueeze(0)
    lc = train.LossCalculator(target_config, loss_type="mse")
    mc = train.MetricCalculator(target_config, metrics=["mae"])
    with pytest.raises(ValueError, match="one case's targets"):
        isla_chunked.chunked_forward_backward(
            batch, model, "float32", lc, mc, target_config=target_config
        )


### ---------------------------------------------------------------------------
### _walk_batch_for_logging
### ---------------------------------------------------------------------------


class TestWalkBatchForLogging:
    """Tests for `_walk_batch_for_logging`."""

    def test_yields_from_tensordict_leaves(self):
        """Bare TD input yields one entry per leaf with the leaf path."""
        td = TensorDict(
            {"pressure": torch.zeros(5), "wss": torch.zeros(5, 3)},
            batch_size=[5],
        )

        items = dict(_walk_batch_for_logging(td))
        assert set(items) == {"pressure", "wss"}
        assert items["pressure"].shape == torch.Size([5])
        assert items["wss"].shape == torch.Size([5, 3])

    def test_dict_containing_tensordict_yields_dotted_keys(self):
        """Nested dict -> TD -> leaves: keys come back dot-joined."""
        batch = {
            "targets": TensorDict(
                {"pressure": torch.zeros(5), "wss": torch.zeros(5, 3)},
                batch_size=[5],
            ),
        }

        items = dict(_walk_batch_for_logging(batch))
        ### Without the TD branch in the walker, neither `targets.pressure`
        ### nor `targets.wss` would appear in the output.
        assert set(items) == {"targets.pressure", "targets.wss"}
        assert items["targets.pressure"].shape == torch.Size([5])

    def test_walk_handles_nested_tensordict_via_flatten_keys(self):
        """A TD nested under another TD: ``flatten_keys`` produces dotted paths.

        This exercises the idiomatic-TD path: ``flatten_keys('.')`` on a
        nested TD returns a flat TD whose keys are dotted leaf paths.
        Without that delegation, a manual ``.items()`` walk would still
        work for flat TDs but would silently mishandle nested ones.
        """
        td = TensorDict(
            {
                "scalar": torch.zeros(3),
                "nested": TensorDict({"x": torch.zeros(3)}, batch_size=[3]),
            },
            batch_size=[3],
        )
        items = dict(_walk_batch_for_logging(td))
        assert set(items) == {"scalar", "nested.x"}
        ### And under a plain dict prefix, paths cascade correctly:
        items_with_prefix = dict(_walk_batch_for_logging({"targets": td}))
        assert set(items_with_prefix) == {"targets.scalar", "targets.nested.x"}


### ---------------------------------------------------------------------------
### normalize_output_to_tensordict
### ---------------------------------------------------------------------------


class TestNormalizeOutputToTensordict:
    """Tests for `normalize_output_to_tensordict`."""

    def test_tensors_output_three_dim_splits_correctly(self):
        """Standard (B, N, total_C) output splits into per-field leaves."""
        target_config = {"pressure": "scalar", "wss": "vector"}
        out = torch.randn(1, 50, 4)  # 1 scalar + 1 vector(3) = 4 channels
        td = normalize_output_to_tensordict(out, target_config, "tensors")
        assert tuple(td["pressure"].shape) == (1, 50)  # squeezed scalar
        assert tuple(td["wss"].shape) == (1, 50, 3)
        assert td.batch_size == torch.Size([1, 50])

    def test_tensors_output_two_dim_raises_clearly(self):
        """Two-D output (missing channel dim) raises a clear shape error.

        A ``(B, N)`` output for a single-scalar target is a config bug:
        without the explicit ``ndim < 3`` guard the per-element axis ``N``
        gets compared to the expected channel count ``C``, yielding a
        confusing "channel dim ``N`` does not match expected ``1``" error.
        The guard surfaces the actual problem (missing trailing channel
        dimension) directly.
        """
        target_config = {"pressure": "scalar"}
        out = torch.randn(1, 50)
        with pytest.raises(ValueError, match=r"expects a \(B, N, C\) tensor"):
            normalize_output_to_tensordict(out, target_config, "tensors")

    def test_tensors_output_channel_mismatch_still_raises(self):
        """Three-D output with wrong channel count still raises the channel error."""
        target_config = {"pressure": "scalar"}
        out = torch.randn(1, 50, 3)  # expected 1 channel
        with pytest.raises(ValueError, match="does not match the expected"):
            normalize_output_to_tensordict(out, target_config, "tensors")

    def test_mesh_output_extracts_target_fields(self):
        """Mesh output: ``point_data.select(*target_config)`` keeps batch_size [N]."""
        target_config = {"pressure": "scalar", "wss": "vector"}
        mesh = Mesh(
            points=torch.randn(7, 3),
            point_data={
                "pressure": torch.randn(7),
                "wss": torch.randn(7, 3),
                ### A non-target field that must NOT appear in the result.
                "extra": torch.randn(7),
            },
        )
        td = normalize_output_to_tensordict(mesh, target_config, "mesh")
        assert set(td.keys()) == {"pressure", "wss"}
        assert td.batch_size == torch.Size([7])

    def test_mesh_output_missing_target_raises(self):
        """Missing target field on a Mesh output is reported clearly."""
        target_config = {"pressure": "scalar"}
        mesh = Mesh(points=torch.randn(7, 3), point_data={"other": torch.randn(7)})
        with pytest.raises(KeyError, match="missing target fields"):
            normalize_output_to_tensordict(mesh, target_config, "mesh")


### ---------------------------------------------------------------------------
### _reduce_and_average
### ---------------------------------------------------------------------------


class TestReduceAndAverage:
    """Tests for `_reduce_and_average` (single-process path).

    The distributed branch is gated on an initialized process group with
    ``world_size > 1``; with no group initialized these tests exercise the
    pure-local path, which must stay equivalent to the previous
    ``total_loss / n`` + per-leaf ``sum / n`` averaging it replaced. The
    collective branch mirrors the already-shipped ``infer._allreduce_sums``
    and is validated by inspection.
    """

    @staticmethod
    def _epoch_sums() -> tuple[TensorDict, TensorDict]:
        """A representative pair of 0-D (epoch-accumulated) sum TensorDicts."""
        losses_td = TensorDict(
            {"pressure": torch.tensor(6.0), "wss": torch.tensor(9.0)},
        )
        metrics_td = TensorDict(
            {"pressure_l2": torch.tensor(3.0), "wss_mae": torch.tensor(12.0)},
        )
        return losses_td, metrics_td

    def test_single_process_divides_sums_by_local_count(self):
        """No process group: global average == local sum / n_local.

        ``loss_sum`` is passed as a 0-D tensor (matching the on-device epoch
        accumulator); the reducer returns Python floats.
        """
        losses_td, metrics_td = self._epoch_sums()
        avg_loss, avg_losses, avg_metrics = _reduce_and_average(
            torch.tensor(15.0), losses_td, metrics_td, 3, device="cpu"
        )
        assert avg_loss == pytest.approx(5.0)
        assert avg_losses == pytest.approx({"pressure": 2.0, "wss": 3.0})
        assert avg_metrics == pytest.approx({"pressure_l2": 1.0, "wss_mae": 4.0})

    def test_none_sentinel_returns_loss_only(self):
        """The "no steps seeded" sentinel (either TD ``None``) yields (loss / n, {}, {})."""
        loss, losses, metrics = _reduce_and_average(
            torch.tensor(8.0), None, None, 2, device="cpu"
        )
        assert loss == pytest.approx(4.0)
        assert losses == {} and metrics == {}
        ### A single ``None`` is enough to trip the sentinel.
        losses_td, _ = self._epoch_sums()
        loss, losses, metrics = _reduce_and_average(
            torch.tensor(8.0), losses_td, None, 2, device="cpu"
        )
        assert loss == pytest.approx(4.0)
        assert losses == {} and metrics == {}

    def test_zero_local_count_avoids_zero_division(self):
        """``n_local == 0`` (a step-less epoch) divides by 1, not 0."""
        loss, losses, metrics = _reduce_and_average(
            torch.tensor(7.0), None, None, 0, device="cpu"
        )
        assert loss == pytest.approx(7.0)
        assert losses == {} and metrics == {}


### ---------------------------------------------------------------------------
### _run_epoch: per-step readback
### ---------------------------------------------------------------------------

_DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA not available"
        ),
    ),
]


class _InFlight:
    """A readback event that lands only when ``landed()`` says so (by default
    never before the epoch's end, so every record goes through the epoch-end
    drain); counts its waits. It wraps the real event, so a record's host copy
    has always landed by the time it is read."""

    def __init__(self, waits, landed, event):
        self.waits = waits
        self.landed = landed
        self.event = event

    def query(self):
        if not self.landed():
            return False
        if self.event is not None:
            self.event.synchronize()
        return True

    def synchronize(self):
        self.waits.append(self)
        if self.event is not None:
            self.event.synchronize()


def _hold_readbacks(monkeypatch, landed=lambda step: False):
    """Patch :func:`train._start_step_readback` so a record lands before the
    epoch's end only when ``landed(step)`` says so; returns the list of
    drain waits, one per record."""
    waits, steps = [], []
    start = train._start_step_readback

    def held(*args, **kwargs):
        reduced, event = start(*args, **kwargs)
        steps.append(step := len(steps))
        return reduced, _InFlight(waits, lambda: landed(step), event)

    monkeypatch.setattr(train, "_start_step_readback", held)
    return waits


def _run_test_epoch(
    monkeypatch, loader, device, *, mode, profile=False, reference=True, writer=None
):
    """Run one :func:`train._run_epoch` with a one-weight model and SGD.

    ``train.forward_pass`` is patched to return step-dependent loss and
    metric values; returns ``(jsonl records, expected per-step records)``,
    where the expected values come from the per-step blocking readback
    (``_reduce_and_average`` with ``n_samples=1``, main's per-step path).
    ``reference=False`` skips that readback, which syncs.
    """
    model = torch.nn.Linear(1, 1, bias=False).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    expected = []

    def forward(batch, *args, **kwargs):
        loss = (model.weight.sum() - batch["k"]) ** 2 / 3
        values = TensorDict({"loss/test": loss.detach() * 7})
        metrics = TensorDict({"test_l2": loss.detach().sqrt()})
        if reference:
            expected.append(
                _reduce_and_average(loss.detach(), values, metrics, 1, device=device)
            )
        return loss, values, metrics

    monkeypatch.setattr(train, "forward_pass", forward)
    records = []
    cfg = OmegaConf.create(
        {
            "precision": "float32",
            "profile": profile,
            "training": {"scheduler_update_mode": "step"},
        }
    )
    _run_epoch(
        loader,
        model,
        None,
        None,
        SimpleNamespace(info=lambda *args, **kwargs: None),
        0,
        cfg,
        SimpleNamespace(rank=0, world_size=1, device=torch.device(device)),
        mode=mode,
        output_type="tensors",
        target_config={"pressure": "scalar"},
        optimizer=optimizer,
        scheduler=scheduler,
        writer=writer,
        log_jsonl=records.append,
    )
    return records, expected


def _check_steps(records, expected, mode, n_steps):
    """Every step is logged once, in order, bitwise equal to ``expected``."""
    steps = [r for r in records if r["phase"] == f"{mode}_step"]
    key = "global_step" if mode == "train" else "val_step"
    assert [r[key] for r in steps] == list(range(n_steps))
    for r, (loss, losses, metrics) in zip(steps, expected, strict=True):
        assert r["loss"] == loss
        assert {k: r[k] for k in losses} == losses
        assert {k: r[k] for k in metrics} == metrics
    assert records[-1]["phase"] == f"{mode}_summary"


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("mode", ["train", "val"])
def test_every_step_logged_once_in_order_with_blocking_values(
    monkeypatch, device, mode
):
    """Each of 10 steps is logged once, in order, with the per-step blocking
    readback's loss, losses and metrics, bitwise (single process, so the
    all-reduce is the identity on both paths). On CUDA the records log as
    their copies land, and the epoch's end drains the rest."""
    loader = [{"k": float(k)} for k in range(10)]
    records, expected = _run_test_epoch(monkeypatch, loader, device, mode=mode)
    _check_steps(records, expected, mode, 10)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("mode", ["train", "val"])
def test_epoch_end_drains_records_still_in_flight(monkeypatch, device, mode):
    """With no copy landed before the loop ends, nothing is logged during the
    steps and the epoch's end waits for and logs every record, in order."""
    waits = _hold_readbacks(monkeypatch)
    loader = [{"k": float(k)} for k in range(7)]
    records, expected = _run_test_epoch(monkeypatch, loader, device, mode=mode)
    _check_steps(records, expected, mode, 7)
    assert len(waits) == 7


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("mode", ["train", "val"])
def test_iter_time_is_the_whole_iteration(monkeypatch, device, mode):
    """Every per-step record carries ``iter_time_s``, the whole iteration: on
    CUDA it includes device work the host does not wait for (a 50M-cycle GPU
    sleep queued by the loader, which ``step_time_s`` misses), off CUDA the
    loader's host time. The iteration times add up to the loop's time."""

    class Loader:
        """Five batches, each after a GPU sleep (CUDA) or a host sleep."""

        def __len__(self):
            return 5

        def __iter__(self):
            for k in range(5):
                if device == "cuda":
                    torch.cuda._sleep(50_000_000)
                else:
                    time.sleep(0.02)
                yield {"k": float(k)}

    records, _ = _run_test_epoch(
        monkeypatch, Loader(), device, mode=mode, reference=False
    )
    steps = [r for r in records if r["phase"] == f"{mode}_step"]
    (summary,) = [r for r in records if r["phase"] == f"{mode}_summary"]
    assert len(steps) == 5
    assert all(r["iter_time_s"] > 0.005 for r in steps), steps
    if device == "cuda":
        assert all(r["step_time_s"] < r["iter_time_s"] for r in steps[1:]), steps
    total = sum(r["iter_time_s"] for r in steps)
    assert 0.8 * summary["epoch_seconds"] < total <= summary["epoch_seconds"]


@pytest.mark.parametrize("device", _DEVICES)
def test_iter_time_survives_the_epoch_end_drain(monkeypatch, device):
    """Records logged by the epoch-end drain carry ``iter_time_s`` too."""
    _hold_readbacks(monkeypatch)
    loader = [{"k": float(k)} for k in range(4)]
    records, _ = _run_test_epoch(monkeypatch, loader, device, mode="train")
    steps = [r for r in records if r["phase"] == "train_step"]
    assert len(steps) == 4 and all(r["iter_time_s"] > 0 for r in steps)


@pytest.mark.parametrize("device", _DEVICES)
def test_summary_records_epoch_cost(monkeypatch, device):
    """The epoch summary carries ``epoch_seconds``, and ``peak_memory_gib``
    on CUDA only, measured from the start of the epoch."""
    if device == "cuda":
        ### A 256 MiB tensor, freed at once, before the epoch must not count.
        torch.empty(2**28, dtype=torch.uint8, device=device)
    loader = [{"k": float(k)} for k in range(2)]
    records, _ = _run_test_epoch(monkeypatch, loader, device, mode="train")
    (summary,) = [r for r in records if r["phase"] == "train_summary"]
    assert summary["epoch_seconds"] > 0
    assert ("peak_memory_gib" in summary) == (device == "cuda")
    if device == "cuda":
        assert 0 < summary["peak_memory_gib"] < 0.25


def test_profile_stop_drains_its_records(monkeypatch):
    """Under ``cfg.profile`` the loop stops after step ``_PROFILE_MAX_STEPS``
    and the epoch's end still logs every step it ran."""
    _hold_readbacks(monkeypatch)
    loader = [{"k": float(k)} for k in range(train._PROFILE_MAX_STEPS + 5)]
    records, _ = _run_test_epoch(monkeypatch, loader, "cpu", mode="train", profile=True)
    steps = [r["global_step"] for r in records if r["phase"] == "train_step"]
    assert steps == list(range(train._PROFILE_MAX_STEPS + 1))


def test_error_logs_only_the_landed_records(monkeypatch):
    """When a step raises, the records whose copies have landed are logged,
    in order up to the first one still in flight, with no wait; nothing
    after them is logged, and the error propagates."""
    raised = False
    waits = _hold_readbacks(monkeypatch, landed=lambda step: raised and step != 3)
    scalars = []
    writer = SimpleNamespace(
        add_scalar=lambda tag, value, global_step: scalars.append((tag, global_step))
    )

    class Loader:
        """Five batches, then an error, at which steps 0-2 and 4 have landed."""

        def __len__(self):
            return 7

        def __iter__(self):
            nonlocal raised
            yield from ({"k": float(k)} for k in range(5))
            raised = True
            raise RuntimeError("step failed")

    with pytest.raises(RuntimeError, match="step failed"):
        _run_test_epoch(monkeypatch, Loader(), "cpu", mode="train", writer=writer)
    assert [step for tag, step in scalars if tag == "iteration/lr"] == [0, 1, 2]
    assert waits == []


@pytest.mark.parametrize("device", _DEVICES)
def test_error_path_records_carry_iter_time(monkeypatch, device):
    """The landed records that a raising step logs carry ``iter_time_s``,
    with no wait for the records still in flight."""
    raised = False
    waits = _hold_readbacks(monkeypatch, landed=lambda step: raised and step < 2)
    scalars = []
    writer = SimpleNamespace(
        add_scalar=lambda tag, value, global_step: scalars.append(
            (tag, value, global_step)
        )
    )

    class Loader:
        """Three batches, then an error, at which steps 0 and 1 have landed."""

        def __len__(self):
            return 4

        def __iter__(self):
            nonlocal raised
            yield from ({"k": float(k)} for k in range(3))
            raised = True
            raise RuntimeError("step failed")

    with pytest.raises(RuntimeError, match="step failed"):
        _run_test_epoch(monkeypatch, Loader(), device, mode="train", writer=writer)
    iter_times = [
        (step, value)
        for tag, value, step in scalars
        if tag == "iteration/performance/iter_time_s"
    ]
    assert [step for step, _ in iter_times] == [0, 1]
    assert all(value > 0 for _, value in iter_times)
    assert waits == []


def test_lr_is_read_at_its_own_step(monkeypatch):
    """A record logged late carries the lr right after its own step's
    scheduler step, not the lr when it is logged."""
    _hold_readbacks(monkeypatch)
    scalars = []
    writer = SimpleNamespace(
        add_scalar=lambda tag, value, global_step: scalars.append(
            (tag, value, global_step)
        )
    )
    loader = [{"k": float(k)} for k in range(7)]
    _run_test_epoch(monkeypatch, loader, "cpu", mode="train", writer=writer)
    lrs = [(step, value) for tag, value, step in scalars if tag == "iteration/lr"]
    assert lrs == [(i, 0.1 * 0.5 ** (i + 1)) for i in range(7)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_train_steps_do_not_sync_with_the_device(monkeypatch):
    """From the first batch to the loader's end, no step syncs the host with
    the device: torch's sync debug mode raises on any flagged sync, and
    explicit event, stream and device waits (which it does not flag) are
    counted. Only the epoch-end drain may wait."""
    waits = []
    in_steps = False
    for owner, name in (
        (torch.cuda.Event, "synchronize"),
        (torch.cuda.Stream, "synchronize"),
        (torch.cuda, "synchronize"),
    ):
        original = getattr(owner, name)

        def counted(*args, _original=original, _name=f"{owner}.{name}", **kwargs):
            if in_steps:
                waits.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(owner, name, counted)

    class Loader:
        """Twelve batches, with the sync checks on while the loop runs."""

        def __len__(self):
            return 12

        def __iter__(self):
            nonlocal in_steps
            in_steps = True
            previous = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                for k in range(12):
                    yield {"k": float(k)}
            finally:
                torch.cuda.set_sync_debug_mode(previous)
                in_steps = False

    records, _ = _run_test_epoch(
        monkeypatch, Loader(), "cuda", mode="train", reference=False
    )
    assert waits == []
    steps = [r["global_step"] for r in records if r["phase"] == "train_step"]
    assert steps == list(range(12))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_step_readback_lands_in_pinned_memory():
    """The ``non_blocking`` copy targets pinned host memory; a pageable
    target would make it synchronous."""
    loss = torch.ones((), device="cuda")
    reduced, event = train._start_step_readback(
        loss,
        TensorDict({"loss/test": loss * 2}),
        TensorDict({"test_l2": loss * 3}),
        device="cuda",
    )
    event.synchronize()
    leaves = reduced.values(include_nested=True, leaves_only=True)
    assert all(leaf.device.type == "cpu" and leaf.is_pinned() for leaf in leaves)


def _readback_rank(rank, port):
    """One rank of :func:`test_step_readback_is_the_all_rank_mean`."""
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        loss = torch.tensor(1.37 * (rank + 1))
        losses = TensorDict({"loss/test": loss * 7})
        metrics = TensorDict({"test_l2": loss.sqrt()})
        host_values, _ = train._start_step_readback(loss, losses, metrics, device="cpu")
        expected = _reduce_and_average(loss, losses, metrics, 1, device="cpu")
        assert host_values["loss"].item() == expected[0]
        assert {k: v.item() for k, v in host_values["losses"].items()} == expected[1]
        assert {k: v.item() for k, v in host_values["metrics"].items()} == expected[2]
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available()
    or not dist.is_gloo_available()
    or "spawn" not in mp.get_all_start_methods(),
    reason="needs gloo and the spawn start method",
)
def test_step_readback_is_the_all_rank_mean():
    """Across two CPU ranks with different values, each step's readback is
    the all-rank mean, equal to the blocking per-step readback
    (``_reduce_and_average`` with ``n_samples=1``)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(_readback_rank, args=(port,), nprocs=2)


def test_float32_matmuls_stay_full_precision():
    """``precision=float32`` runs without autocast, and nothing the trainer or the baselines import turns
    TF32 on for matmuls (PyTorch's default: ``allow_tf32`` False, precision ``highest``), so an fp32 arm of a
    bf16-vs-fp32 comparison is true fp32 (2026-10-19)."""
    import physicsnemo.models.geotransolver  # noqa: F401
    import physicsnemo.models.transolver  # noqa: F401

    assert torch.get_float32_matmul_precision() == "highest"
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert isinstance(train.get_autocast_context("float32"), nullcontext)


@pytest.mark.parametrize("scheduler_mode", ["epoch", "step"])
@pytest.mark.parametrize("precision", ["float32", "float16"])
def test_training_resume_matches_uninterrupted_schedule(
    tmp_path, monkeypatch, scheduler_mode, precision
):
    """Resume at a decay boundary with identical weights, LR and AMP state.

    Exercise the actual training driver and checkpoint utilities with tiny CPU
    updates. Stubbing only the data/epoch work keeps this independent of GPUs
    and checks checkpoint timing, final-epoch saving and scaler persistence.
    """
    from contextlib import nullcontext
    from types import SimpleNamespace
    from unittest.mock import Mock

    import train
    from omegaconf import OmegaConf

    manager = SimpleNamespace(device="cpu", rank=0, world_size=1)
    monkeypatch.setattr(train, "DistributedManager", Mock(return_value=manager))
    monkeypatch.setattr(train, "RankZeroLoggingWrapper", Mock(return_value=Mock()))
    monkeypatch.setattr(train, "SummaryWriter", Mock())
    monkeypatch.setattr(train, "Profiler", nullcontext)
    monkeypatch.setattr(
        train, "GradScaler", lambda: torch.amp.GradScaler("cpu", growth_interval=2)
    )
    monkeypatch.setattr(
        train,
        "build_muon_optimizer",
        lambda model, cfg, **kwargs: torch.optim.SGD(
            model.parameters(), lr=0.2, momentum=0.9
        ),
    )
    loader = SimpleNamespace(sampler=[0], dataset=[0], set_epoch=Mock())
    monkeypatch.setattr(
        train,
        "build_dataloaders",
        lambda cfg: (loader, loader, None, {"targets": {"pressure": "scalar"}}),
    )
    monkeypatch.setattr(train, "val_epoch", lambda *args, **kwargs: (0.0, {}))
    monkeypatch.setattr(train, "load_dataset_config", lambda path: OmegaConf.create({}))
    trajectories = {}

    def update(
        dataloader,
        model,
        optimizer,
        scheduler,
        loss_calculator,
        metric_calculator,
        logger,
        epoch,
        cfg,
        dist_manager,
        scaler,
        **kwargs,
    ):
        optimizer.zero_grad()
        loss = model(torch.ones(1, 1)).square().sum()
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if scheduler_mode == "step":
            scheduler.step()
        trajectories.setdefault(cfg.run_id, []).append(
            (
                epoch,
                optimizer.param_groups[0]["lr"],
                torch.cat([p.detach().flatten().clone() for p in model.parameters()]),
                scaler.state_dict() if scaler is not None else None,
            )
        )
        return loss.item(), {}

    monkeypatch.setattr(train, "train_epoch", update)

    def run(run_id, epochs):
        cfg = OmegaConf.create(
            {
                "model": {
                    "_target_": "torch.nn.Linear",
                    "in_features": 1,
                    "out_features": 1,
                },
                "input_type": "tensors",
                "output_type": "tensors",
                "dataset": "stub",
                "output_dir": str(tmp_path),
                "run_id": run_id,
                "precision": precision,
                "compile": False,
                "training": {
                    "seed": 42,
                    "num_epochs": epochs,
                    "save_interval": 2,
                    "scheduler_update_mode": scheduler_mode,
                    "scheduler": {
                        "_target_": "torch.optim.lr_scheduler.StepLR",
                        "step_size": 2,
                        "gamma": 0.1,
                    },
                },
            }
        )
        train.main(cfg)

    run("uninterrupted", 4)
    run("resumed", 2)
    run("resumed", 4)
    expected = trajectories["uninterrupted"]
    actual = trajectories["resumed"]
    assert len(actual) == len(expected) == 4
    for (epoch, lr, weights, scaler), reference in zip(actual, expected):
        assert epoch == reference[0]
        assert lr == reference[1]
        torch.testing.assert_close(weights, reference[2], rtol=0, atol=0)
        assert scaler == reference[3]
    # Both invocations must also persist the requested final epoch, even
    # when it falls outside the existing periodic checkpoint cadence.
    assert train.load_checkpoint(path=tmp_path / "resumed" / "checkpoints") == 4
