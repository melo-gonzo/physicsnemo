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

"""FixedDrawDomainMeshReader: the subsample of a case is identical across epochs, while the base reader's changes."""

import torch
from tensordict import TensorDict

from physicsnemo.datapipes.readers.mesh import DomainMeshReader
from physicsnemo.mesh import DomainMesh, Mesh

from domain_transforms import FixedDrawDomainMeshReader


def _write_case(tmp_path, n_int=500):
    g = torch.Generator().manual_seed(0)
    pts = torch.rand(n_int, 3, generator=g)
    pts[:, 2] = 0
    interior = Mesh(
        points=pts,
        cells=torch.zeros(0, 1, dtype=torch.long),
        point_data=TensorDict(
            {"u": torch.arange(n_int, dtype=torch.float32)[:, None]}, batch_size=[n_int]
        ),
    )
    bpts = torch.tensor([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    bnd = Mesh(points=bpts, cells=torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]]))
    d = tmp_path / "case_000"
    d.mkdir()
    DomainMesh(interior=interior, boundaries={"outer": bnd}).save(d / "case_000.pdmsh")
    return tmp_path


def _draw(reader, epoch):
    reader.set_generator(torch.Generator().manual_seed(123))
    reader.set_epoch(epoch)
    out = reader[0]
    dm = out[0] if isinstance(out, tuple) else out
    return dm.interior.point_data["u"].flatten()


def test_fixed_draw_is_epoch_independent_and_base_reader_is_not(tmp_path):
    """The fixed-draw reader returns the same subsample at every epoch; the base reader does not."""
    root = _write_case(tmp_path)
    kw = dict(
        path=str(root),
        pattern="*/*.pdmsh",
        subsample_n_points=50,
        drop_interior_cells=True,
    )
    fixed = FixedDrawDomainMeshReader(**kw)
    a, b = _draw(fixed, 0), _draw(fixed, 7)
    torch.testing.assert_close(a, b)
    assert a.numel() == 50
    base = DomainMeshReader(**kw)
    c, d = _draw(base, 0), _draw(base, 7)
    torch.testing.assert_close(a, c)  # epoch 0 draw is the base reader's
    assert not torch.equal(c, d)
