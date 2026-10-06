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

"""QI-GLOBAL model-level step time and profile on a synthetic 2-D case (protocol width, bf16, compiled blocks).
usage: python model_bench.py <points> <ro|global> <all|blocks> [prof]"""

import sys
import time

import torch

from physicsnemo.experimental.models import ISLA
from test.experimental.models.isla._meshes import cloud, outputs, points_mesh, soup

dev = "cuda"
Q = int(sys.argv[1])
mode = sys.argv[2]
blocks = None if sys.argv[3] == "all" else int(sys.argv[3])
pts, nrm, w, drv = cloud(1000, d=2, seed=0, dtype=torch.float32)
g = torch.Generator().manual_seed(1)
q = torch.randn(Q, 2, generator=g) * 2
sdf = torch.randn(Q, generator=g)
sn = torch.nn.functional.normalize(torch.randn(Q, 2, generator=g), dim=-1)
kw = dict(
    output_schema={"p": {"rank": 0}, "v": {"rank": 1}},
    global_schema={"g": {"rank": 1}},
    query_schema={"sdf": {"rank": 0}, "sdf_normals": {"rank": 1}},
    interior=True,
    query_independent=True,
    dim=2,
    compile_blocks=True,
    geo_checkpoint=True,
)
if mode == "global":
    kw.update(kernel_read="mean", kernel_read_radius=None, kernel_read_blocks=blocks)
m = ISLA(**kw).to(dev)
bm = {"wall": soup(pts, nrm, w).to(dev)}
pp = points_mesh(q.to(dev), {"sdf": sdf.to(dev), "sdf_normals": sn.to(dev)})
gd = {"g": drv.to(dev)}
opt = torch.optim.AdamW(m.parameters(), lr=1e-5)
ts = []
for s in range(8):
    torch.cuda.synchronize()
    t = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = outputs(m(boundary_meshes=bm, prediction_points=pp, global_data=gd))
    sum(o.float().square().mean() for o in out.values()).backward()
    opt.step()
    opt.zero_grad()
    torch.cuda.synchronize()
    ts.append(time.time() - t)
print(
    mode,
    blocks,
    Q,
    [f"{1000 * x:.0f}" for x in ts],
    f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB",
)
if len(sys.argv) > 4:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as p:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = outputs(m(boundary_meshes=bm, prediction_points=pp, global_data=gd))
        sum(o.float().square().mean() for o in out.values()).backward()
        torch.cuda.synchronize()
    tot = sum(e.self_device_time_total for e in p.key_averages())
    print("total", tot / 1000)
    for e in sorted(p.key_averages(), key=lambda e: -e.self_device_time_total)[:25]:
        print(f"{e.self_device_time_total / 1000:8.2f}ms n={e.count:4d} {e.key[:110]}")
