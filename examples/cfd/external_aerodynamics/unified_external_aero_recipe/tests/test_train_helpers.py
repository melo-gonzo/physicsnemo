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
  sums over the global sample count (used per step and per epoch); its
  single-process path must equal plain ``total_loss / n`` + per-leaf
  ``sum / n`` averaging.
- :func:`train._run_epoch`: a train step fetches the next batch after its
  backward and before its optimizer step.

(The analogous tests for the shared, tensorboard-free
:func:`utils.recursive_to_device` live in ``test_utils.py``, outside
this module's tensorboard skip guard.)
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

### `train.py` imports `torch.utils.tensorboard.SummaryWriter` at module
### load, which transitively requires the `tensorboard` package. That
### dep is not declared in pyproject.toml; CI / training environments
### have it installed, but bare dev sandboxes might not. Skip cleanly.
### `output_normalize` itself is tensorboard-free, so we import it
### directly (no skip).
pytest.importorskip("tensorboard")

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
    monkeypatch.setitem(train._CHUNK_GIB, "float32", (0.0, 1e5, 0.0))

    def check(capacity_gib=None):
        sizes.clear()
        model.zero_grad(set_to_none=True)
        torch.manual_seed(1)
        out = train.chunked_forward_backward(
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
    passes = 1 if loss_type in train._POINTWISE_LOSSES else 2
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
        query_independent=True,
        boundary_tokens=40,
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
    ### Held-out steps on a GB300 (275.0 GiB free and reserved; DTT-FAST RULE jobs
    ### 368672, 368728): a chunk's walk pairs and points, GiB allocated at the
    ### decision and the step's peak reserved GiB, which the prediction covers
    ### within the device.
    for pairs, points, allocated, reserved, precision in (
        (417_882_241, 600_000, 79.28, 263.28, "bfloat16"),  # DrivAerML vehicle
        (209_094_835, 300_000, 79.28, 173.33, "bfloat16"),  # the same in 2 chunks
        (425_319_791, 500_000, 79.36, 238.91, "bfloat16"),  # whole boundary
        (85_131_592, 400_000, 74.26, 182.65, "bfloat16"),  # AeroZoo 800k in 2 chunks
        (138_943_574, 200_000, 112.19, 221.38, "float32"),  # DrivAerML vehicle
        (126_352_902, 150_000, 112.01, 204.76, "float32"),  # whole boundary
    ):
        peak = train._chunk_peak_gib(pairs, points, allocated, precision)
        assert reserved <= peak < 275.0
    ### 20 pairs per point over two trees, each list sorted by point.
    points = torch.arange(50)
    chunks = SimpleNamespace(
        walk=SimpleNamespace(
            lists=[
                (points.repeat_interleave(12), None, 0),
                (points.repeat_interleave(8), None, 0),
            ]
        ),
        n_points=50,
    )
    one = train._chunk_peak_gib(1000, 50, 0.0, "float32")
    third = train._chunk_peak_gib(340, 17, 0.0, "float32")
    cpu = torch.device("cpu")

    def spans(pointwise, capacity):
        return train._chunk_spans(chunks, cpu, False, pointwise, "float32", capacity)

    monkeypatch.setattr(train, "_chunks_logged", None)
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
    monkeypatch.setitem(train._CHUNK_GIB, "float32", (1e6, 0.0, 0.0))
    counts = torch.tensor([10] * 10 + [1] * 40)
    chunks = SimpleNamespace(
        walk=SimpleNamespace(
            lists=[(torch.arange(50).repeat_interleave(counts), None, 0)]
        ),
        n_points=50,
    )
    cpu = torch.device("cpu")

    def spans(pointwise, capacity):
        return train._chunk_spans(chunks, cpu, False, pointwise, "float32", capacity)

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
        train._CHUNK_GIB["float32"] = (0.0, 1e5, 0.0)
        sizes = []
        call = isla_model._QueryChunks.__call__
        isla_model._QueryChunks.__call__ = lambda self, a, b: (
            sizes.append(b - a) or call(self, a, b)
        )
        torch.manual_seed(1)
        train.chunked_forward_backward(
            batch,
            torch.nn.parallel.DistributedDataParallel(model),
            "float32",
            lc,
            mc,
            target_config=target_config,
            capacity_gib=(100.0, 20.0)[rank],
        )
        passes = 1 if loss_type in train._POINTWISE_LOSSES else 2
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


def test_chunked_step_chunks_isla_dtt():
    """ISLA-DTT takes the chunked step; an interacting ISLA never chunks."""
    from physicsnemo.experimental.models.isla import ISLA

    def isla(query_independent):
        return ISLA(
            output_schema={"pressure": {"rank": 0}},
            boundary_schemas={"vehicle": {}},
            interior=True,
            query_independent=query_independent,
            hidden=16,
            n_layers=1,
            n_slices=4,
        )

    def batch(n):
        return {"forward_kwargs": {"prediction_points": Mesh(points=torch.zeros(n, 3))}}

    assert train._chunked_step(isla(True), batch(50))
    assert not train._chunked_step(isla(False), batch(50))


def test_chunked_forward_backward_refuses_batched_targets():
    """The chunked step slices the targets by point: targets over more than the
    points' axis (a batch of cases) are refused before the model runs."""
    model, batch, target_config = _isla_dtt_case(20)
    batch["targets"] = batch["targets"].unsqueeze(0)
    lc = train.LossCalculator(target_config, loss_type="mse")
    mc = train.MetricCalculator(target_config, metrics=["mae"])
    with pytest.raises(ValueError, match="one case's targets"):
        train.chunked_forward_backward(
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
### Epoch loop
### ---------------------------------------------------------------------------


def _single_rank() -> SimpleNamespace:
    """Return a single-process stand-in for ``DistributedManager``."""
    return SimpleNamespace(rank=0, world_size=1, device=torch.device("cpu"))


def _run_test_epoch(
    loader,
    model,
    *,
    mode,
    dist_manager,
    loss_calculator=None,
    metric_calculator=None,
):
    """Run one :func:`train._run_epoch` on a one-field tensor model with SGD."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = OmegaConf.create(
        {
            "precision": "float32",
            "profile": False,
            "training": {"scheduler_update_mode": "epoch"},
        }
    )
    return _run_epoch(
        loader,
        model,
        loss_calculator,
        metric_calculator,
        SimpleNamespace(
            info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None
        ),
        0,
        cfg,
        dist_manager,
        mode=mode,
        output_type="tensors",
        target_config={"pressure": "scalar"},
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1),
    )


def test_float32_matmuls_stay_full_precision():
    """``precision=float32`` runs without autocast, and nothing the trainer or the baselines import turns
    TF32 on for matmuls (PyTorch's default: ``allow_tf32`` False, precision ``highest``), so an fp32 arm of a
    bf16-vs-fp32 comparison is true fp32 (2026-10-19)."""
    import physicsnemo.models.geotransolver  # noqa: F401
    import physicsnemo.models.transolver  # noqa: F401

    assert torch.get_float32_matmul_precision() == "highest"
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert isinstance(train.get_autocast_context("float32"), nullcontext)


def test_train_loads_the_next_batch_after_backward(monkeypatch):
    """Train mode fetches batch i + 1 after step i's backward and before its
    optimizer step; every batch reaches the forward once, in loader order."""
    events = []
    model = torch.nn.Linear(1, 1, bias=False)

    class Loader:
        """Yield three numbered batches, recording each load."""

        def __len__(self):
            return 3

        def __iter__(self):
            for k in range(3):
                events.append(f"load {k}")
                yield {"k": k}

    def forward(batch, *args, **kwargs):
        events.append(f"forward {batch['k']}")
        loss = model.weight.sum()
        loss.register_hook(lambda g: events.append("backward"))
        values = TensorDict({"loss/test": loss.detach()})
        return loss, values, values.clone()

    step = torch.optim.SGD.step
    monkeypatch.setattr(train, "forward_pass", forward)
    monkeypatch.setattr(
        torch.optim.SGD,
        "step",
        lambda self, *a, **k: (events.append("step"), step(self, *a, **k))[1],
    )
    _run_test_epoch(Loader(), model, mode="train", dist_manager=_single_rank())
    assert events == [
        "load 0", "forward 0", "backward", "load 1", "step",
        "forward 1", "backward", "load 2", "step",
        "forward 2", "backward", "step",
    ]  # fmt: skip
