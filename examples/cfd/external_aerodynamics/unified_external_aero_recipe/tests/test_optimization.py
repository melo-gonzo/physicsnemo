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

"""Hand-derived updates and interrupted training checks for optimization."""

import copy
import math
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from optimization import Lion, WarmupCosine

_RECIPE_ROOT = Path(__file__).resolve().parents[1]


def test_lion_hand_computed_updates():
    """Match two hand-calculated Lion parameter and momentum updates."""
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
    opt = Lion([p], lr=0.1, weight_decay=0.2)
    p.grad = torch.tensor([2.0, -3.0], dtype=p.dtype)
    opt.step()
    torch.testing.assert_close(p, torch.tensor([0.88, 2.06], dtype=p.dtype))
    p.grad = torch.tensor([-0.01, 0.01], dtype=p.dtype)
    opt.step()
    torch.testing.assert_close(p, torch.tensor([0.7624, 2.1188], dtype=p.dtype))
    torch.testing.assert_close(
        opt.state[p]["momentum"], torch.tensor([0.0197, -0.0296], dtype=p.dtype)
    )


@pytest.mark.parametrize("warmup", [0, 4])
def test_schedule_endpoints_and_resume(warmup):
    """Preserve scheduled learning rates and updates across checkpoint resume."""
    p = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    opt = Lion([p], lr=0.1)
    scheduler = WarmupCosine(opt, total_steps=12, warmup_steps=warmup, min_lr=0.001)
    rates = []
    for step in range(12):
        rates.append(opt.param_groups[0]["lr"])
        p.grad = torch.tensor([(-1.0) ** step, 0.3])
        opt.step()
        scheduler.step()
        if step == 5:
            checkpoint = (
                p.detach().clone(),
                copy.deepcopy(opt.state_dict()),
                copy.deepcopy(scheduler.state_dict()),
            )
    assert rates[0] == pytest.approx(0.1 / warmup if warmup else 0.1)
    assert rates[warmup - 1 if warmup else 0] == pytest.approx(0.1)
    assert rates[-1] == pytest.approx(0.001)
    resumed = torch.nn.Parameter(checkpoint[0])
    opt2 = Lion([resumed], lr=0.1)
    scheduler2 = WarmupCosine(opt2, total_steps=12, warmup_steps=warmup, min_lr=0.001)
    opt2.load_state_dict(checkpoint[1])
    scheduler2.load_state_dict(checkpoint[2])
    for step in range(6, 12):
        assert opt2.param_groups[0]["lr"] == rates[step]
        resumed.grad = torch.tensor([(-1.0) ** step, 0.3])
        opt2.step()
        scheduler2.step()
    torch.testing.assert_close(p, resumed, rtol=0, atol=0)
    torch.testing.assert_close(
        opt.state[p]["momentum"], opt2.state[resumed]["momentum"], rtol=0, atol=0
    )


@pytest.mark.parametrize("warmup", [0, 4])
def test_every_rate_is_the_closed_form(warmup):
    """Every update's rate is the closed form conf/train.yaml states: peak (e + 1) / w for warmup update e, then
    min_lr + (peak - min_lr) (1 + cos(pi t)) / 2 with t = (e + 1 - w) / (T - w), or t = e / (T - 1) without warmup.
    The first update after warmup is already below the peak."""
    total, peak, low = 12, 0.1, 0.001
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=peak)
    scheduler = WarmupCosine(opt, total_steps=total, warmup_steps=warmup, min_lr=low)
    rates = []
    for _ in range(total):
        rates.append(opt.param_groups[0]["lr"])
        opt.step()
        scheduler.step()
    for e, rate in enumerate(rates):
        if e < warmup:
            expected = peak * (e + 1) / warmup
        else:
            t = (e + 1 - warmup) / (total - warmup) if warmup else e / (total - 1)
            expected = low + (peak - low) * (1 + math.cos(math.pi * t)) / 2
        assert rate == pytest.approx(expected, rel=1e-12, abs=0), e
    if warmup:
        assert rates[warmup] < rates[warmup - 1]


def test_the_default_schedule_spans_the_run():
    """conf/train.yaml's default: WarmupCosine stepped once per epoch over training.num_epochs, five warmup epochs,
    down to 1e-6, so a run with other epochs anneals over them."""
    with initialize_config_dir(
        config_dir=str(_RECIPE_ROOT / "conf"), version_base=None
    ):
        cfg = compose(config_name="train")
        short = compose(config_name="train", overrides=["training.num_epochs=250"])
    assert OmegaConf.to_container(cfg.training.scheduler, resolve=True) == {
        "_target_": "optimization.WarmupCosine",
        "total_steps": cfg.training.num_epochs,
        "warmup_steps": 5,
        "min_lr": 1.0e-6,
    }
    assert cfg.training.scheduler_update_mode == "epoch"
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    scheduler = instantiate(short.training.scheduler, optimizer=opt)
    assert isinstance(scheduler, WarmupCosine)
    assert (scheduler.total_steps, scheduler.warmup_steps) == (250, 5)


def test_a_resume_with_another_schedule_is_refused():
    """A WarmupCosine state loads only into the schedule it was planned with. A resume with other total_steps (another
    training.num_epochs), warmup_steps or min_lr raises instead of continuing the old schedule."""

    def schedule(**kwargs):
        p = torch.nn.Parameter(torch.zeros(1))
        return WarmupCosine(torch.optim.SGD([p], lr=0.1), **kwargs)

    planned = {"total_steps": 250, "warmup_steps": 5, "min_lr": 1e-6}
    saved = schedule(**planned)
    for _ in range(100):
        saved.optimizer.step()
        saved.step()
    resumed = schedule(**planned)
    resumed.load_state_dict(saved.state_dict())
    assert resumed.last_epoch == 100
    for change in ({"total_steps": 500}, {"warmup_steps": 10}, {"min_lr": 1e-5}):
        with pytest.raises(ValueError, match="differs from the configured"):
            schedule(**(planned | change)).load_state_dict(saved.state_dict())


def test_optimizer_factory_explicit_and_legacy():
    """Select an explicit optimizer while preserving the legacy default."""
    from omegaconf import OmegaConf
    from utils import build_optimizer

    model = torch.nn.Linear(2, 1)
    cfg = OmegaConf.create(
        {"training": {"optimizer": {"_target_": "optimization.Lion", "lr": 5e-5}}}
    )
    assert isinstance(build_optimizer(model, cfg), Lion)
    cfg.training.optimizer = {"lr": 1e-4}
    legacy = build_optimizer(model, cfg)
    assert not isinstance(legacy, Lion)
