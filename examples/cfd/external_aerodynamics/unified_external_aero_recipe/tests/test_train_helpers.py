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
    """ISLA-DTT's chunked step (one pass for a pointwise loss; otherwise one pass
    over one chunk of every point when it fits, else two passes): the loss, its
    fields, the metrics and every parameter's gradient of forward_pass and its
    backward, from the same token draw."""
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

    def check(chunk, capacity_gib=None):
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
            chunk=chunk,
            capacity_gib=capacity_gib,
        )
        for a, b in zip(out, ref):
            torch.testing.assert_close(a, b.detach(), rtol=1e-5, atol=1e-7)
        for k, p in model.named_parameters():
            torch.testing.assert_close(
                p.grad, grads[k], rtol=1e-4, atol=1e-6 * scale, msg=k
            )
        return list(sizes)

    if loss_type in train._POINTWISE_LOSSES:
        assert check(17) == check(17, capacity_gib=0.0) == [17, 17, 16]
    else:
        assert check(17) == [n]
        assert check(17, capacity_gib=0.0) == [17, 17, 16] * 2
    assert check(n) == [n]


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


def test_one_chunk_rule_decides_from_sizes(monkeypatch, caplog):
    """The predicted peak of ISLA-DTT's step with every point in one chunk, from
    the walk's pairs, the points and the memory allocated after the boundary side;
    the decision against the capacity, logged at its first call and on change."""
    ### AeroZoo in bfloat16 on a GB300 (275.0 GiB free and reserved; jobs 365629,
    ### 366092): 400k points ran in one chunk (197.7 GiB reserved), 800k did not fit.
    assert train._one_chunk_peak_gib(83_886_094, 400_000, 73.9, "bfloat16") < 275.0
    assert train._one_chunk_peak_gib(169_149_486, 800_000, 75.1, "bfloat16") > 275.0
    ### DrivAerML Volume in float32 (job 366197, 366092): the whole boundary at 150k
    ### points ran in one chunk (263.8 GiB reserved), the vehicle at 200k did not fit.
    assert train._one_chunk_peak_gib(126_390_000, 150_000, 112.9, "float32") < 275.0
    assert train._one_chunk_peak_gib(138_883_949, 200_000, 113.0, "float32") > 275.0
    chunks = SimpleNamespace(
        walk=SimpleNamespace(
            lists=[(torch.zeros(600), None, 0), (torch.zeros(400), None, 0)]
        ),
        n_points=50,
    )
    peak = train._one_chunk_peak_gib(1000, 50, 0.0, "float32")
    cpu = torch.device("cpu")
    monkeypatch.setattr(train, "_one_chunk_logged", None)
    with caplog.at_level(logging.INFO, logger="training.step"):
        assert train._one_chunk_fits(chunks, cpu, False, "float32", peak)
        assert train._one_chunk_fits(chunks, cpu, False, "float32", None)
        assert not train._one_chunk_fits(chunks, cpu, False, "float32", peak - 1e-6)
    logged = [r.getMessage() for r in caplog.records if r.name == "training.step"]
    assert len(logged) == 2, logged
    assert "in one chunk" in logged[0] and "two passes" in logged[1]


def _two_ranks_one_over_the_limit(rank: int, store: str) -> None:
    """Rank ``rank`` of :func:`test_chunked_forward_backward_agrees_across_ranks`."""
    import torch.distributed as dist

    from physicsnemo.experimental.models.isla import model as isla_model

    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=2
    )
    try:
        n = (50, 30)[rank]
        model, batch, target_config = _isla_dtt_case(n, seed=rank)
        lc = train.LossCalculator(target_config, loss_type="relative_l2")
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
        ### Room for one chunk of 40 points: rank 1's 30 fit, rank 0's 50 do not.
        capacity = train._one_chunk_peak_gib(0, 40, 0.0, "float32")
        sizes, peaks = [], []
        call, predict = isla_model._QueryChunks.__call__, train._one_chunk_peak_gib
        isla_model._QueryChunks.__call__ = lambda self, a, b: (
            sizes.append(b - a) or call(self, a, b)
        )
        train._one_chunk_peak_gib = lambda *a: peaks.append(predict(*a)) or peaks[-1]
        torch.manual_seed(1)
        train.chunked_forward_backward(
            batch,
            torch.nn.parallel.DistributedDataParallel(model),
            "float32",
            lc,
            mc,
            target_config=target_config,
            chunk=17,
            capacity_gib=capacity,
        )
        assert (peaks[0] <= capacity) == (rank == 1), (rank, peaks, capacity)
        assert sizes == ([17, 17, 16] if rank == 0 else [17, 13]) * 2, (rank, sizes)
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
def test_chunked_forward_backward_agrees_across_ranks(tmp_path):
    """Two DDP ranks (gloo) under a root loss, with different point counts, where
    every point in one chunk fits on rank 1 and not on rank 0: both take the
    two-pass form, with gradients equal to the mean of the ranks' unchunked ones
    and gradients and parameters bitwise equal across the ranks."""
    torch.multiprocessing.spawn(
        _two_ranks_one_over_the_limit, args=(str(tmp_path / "store"),), nprocs=2
    )


def test_query_chunk_chunks_isla_dtt(monkeypatch):
    """ISLA-DTT takes chunks of _QUERY_CHUNK points; an interacting ISLA never
    chunks."""
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

    monkeypatch.setattr(train, "_QUERY_CHUNK", 17)
    assert train._query_chunk(isla(True), batch(50)) == 17
    assert train._query_chunk(isla(False), batch(50)) is None


def test_chunked_forward_backward_refuses_batched_targets():
    """The chunked step slices the targets by point: targets over more than the
    points' axis (a batch of cases) are refused before the model runs."""
    model, batch, target_config = _isla_dtt_case(20)
    batch["targets"] = batch["targets"].unsqueeze(0)
    lc = train.LossCalculator(target_config, loss_type="mse")
    mc = train.MetricCalculator(target_config, metrics=["mae"])
    with pytest.raises(ValueError, match="one case's targets"):
        train.chunked_forward_backward(
            batch, model, "float32", lc, mc, target_config=target_config, chunk=7
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
