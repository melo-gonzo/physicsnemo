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

"""Tests for the transfer program's campaign-C hook: weight-only
initialization from another run's checkpoint."""

import torch
from utils import initialize_from_checkpoint

from physicsnemo.experimental.models.isla import ISLA
from physicsnemo.utils.checkpoint import save_checkpoint


def test_initialize_from_checkpoint_loads_weights_only(tmp_path):
    """init_from copies a checkpoint's weights into a freshly built model of the same architecture."""
    torch.manual_seed(0)
    src = ISLA(
        output_schema={"p": {"rank": 0}, "v": {"rank": 1}},
        boundary_schemas={"wall": {}},
        global_schema={"g": {"rank": 1}},
        hidden=32,
        n_layers=2,
        n_slices=8,
    )
    save_checkpoint(str(tmp_path), models=src, epoch=7)
    saved = sorted(tmp_path.glob("*.mdlus"))
    assert saved, "save_checkpoint wrote no .mdlus file"
    torch.manual_seed(1)
    dst = ISLA(
        output_schema={"p": {"rank": 0}, "v": {"rank": 1}},
        boundary_schemas={"wall": {}},
        global_schema={"g": {"rank": 1}},
        hidden=32,
        n_layers=2,
        n_slices=8,
    )
    before = {k: v.clone() for k, v in dst.state_dict().items()}
    assert any(not torch.equal(before[k], v) for k, v in src.state_dict().items())
    report = initialize_from_checkpoint(dst, str(tmp_path))
    assert report["epoch"] == 7
    for k, v in src.state_dict().items():
        assert torch.equal(dst.state_dict()[k], v)


def test_initialize_from_checkpoint_rejects_architecture_mismatch(tmp_path):
    """init_from refuses a checkpoint whose architecture differs from the model's."""
    src = ISLA(
        output_schema={"p": {"rank": 0}, "v": {"rank": 1}},
        boundary_schemas={"wall": {}},
        global_schema={"g": {"rank": 1}},
        hidden=32,
        n_layers=2,
        n_slices=8,
    )
    save_checkpoint(str(tmp_path), models=src, epoch=1)
    other = ISLA(
        output_schema={"p": {"rank": 0}, "v": {"rank": 1}},
        boundary_schemas={"wall": {}},
        global_schema={"g": {"rank": 1}},
        hidden=48,
        n_layers=2,
        n_slices=8,
    )
    try:
        initialize_from_checkpoint(other, str(tmp_path))
    except Exception:
        return
    raise AssertionError("mismatched architecture loaded without error")
