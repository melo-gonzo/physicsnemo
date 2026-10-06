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

"""QI-GLOBAL kernel timings at the protocol size (synthetic 2-D case: 100,000 points, 1,000 cells): the fused basis
and the fused read under block-size variants, then a CUDA profile of one forward + backward of each."""

import itertools
import time

import torch
from torch.profiler import ProfilerActivity, profile

import physicsnemo.experimental.models.isla.global_read_kernels as grk
import physicsnemo.experimental.models.isla.kernel_read as kr

dev = "cuda"
Q, N, d = 100000, 1000, 2
g = torch.Generator(device=dev).manual_seed(0)


def mk(*s, dtype=torch.float32):
    """A random tensor on the device."""
    return torch.randn(*s, generator=g, device=dev, dtype=dtype)


def nrm(x):
    """Unit vectors along the last axis."""
    return torch.nn.functional.normalize(x, dim=-1)


geo = dict(
    query_r=mk(Q, d),
    query_n=nrm(mk(Q, d)),
    query_wall=mk(Q).abs() + 0.01,
    query_vecs=nrm(mk(Q, 1, d)),
    boundary_r=mk(N, d),
    boundary_n=nrm(mk(N, d)),
    boundary_vecs=None,
    g_unit=nrm(mk(1, d)),
    eps=1e-12,
)
basis = kr._KernelBasis(kr.global_geo_width(1, 1, 0)).to(dev)


def timeit(fn, n=5):
    """Median wall time (ms) of fn over n calls after one warm-up."""
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t = time.time()
        fn()
        torch.cuda.synchronize()
        ts.append(time.time() - t)
    return 1000 * sorted(ts)[len(ts) // 2]


def basis_fb():
    """Fused basis forward and backward."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        phi = grk.fused_kernel_basis(basis, geo)
    phi.backward(torch.ones_like(phi))


def basis_f():
    """Fused basis forward, no gradient."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        grk.fused_kernel_basis(basis, geo)


for bp, wp in itertools.product((32, 64, 128), (4, 8)):
    grk.BLOCK_P, grk.BASIS_WARPS = bp, wp
    try:
        print(
            f"basis BP={bp} warps={wp}: fwd {timeit(basis_f):.1f} ms, fwd+bwd {timeit(basis_fb):.1f} ms",
            flush=True,
        )
    except Exception as e:
        print("basis", bp, wp, "failed", repr(e)[:200], flush=True)
grk.BLOCK_P, grk.BASIS_WARPS = 64, 8
H = 8
dt = torch.bfloat16
args = [
    mk(16, Q * N, dtype=dt),
    mk(H, 16) * 0.5,
    mk(H, Q, 4, dtype=dt),
    mk(H, N, 4, dtype=dt),
    mk(H, N, 8, dtype=dt),
    mk(N),
]
args = [a.requires_grad_() for a in args[:5]] + [args[5]]


def read_fb():
    """Fused read forward and backward."""
    o = grk.global_softmax_read(*args, 0.5)
    o.backward(torch.ones_like(o))


def read_f():
    """Fused read forward, no gradient."""
    with torch.no_grad():
        grk.global_softmax_read(*args, 0.5)


for bq, bn, bkv in [
    (64, 64, 32),
    (128, 64, 32),
    (64, 128, 32),
    (64, 64, 16),
    (128, 128, 16),
    (32, 64, 32),
]:
    grk.BLOCK_Q, grk.BLOCK_N, grk.BLOCK_N_KV = bq, bn, bkv
    try:
        print(
            f"read BQ={bq} BN={bn} BNKV={bkv}: fwd {timeit(read_f):.1f} ms, fwd+bwd {timeit(read_fb):.1f} ms",
            flush=True,
        )
    except Exception as e:
        print("read", bq, bn, bkv, "failed", repr(e)[:200], flush=True)
grk.BLOCK_Q, grk.BLOCK_N, grk.BLOCK_N_KV = 64, 64, 32

with profile(activities=[ProfilerActivity.CUDA]) as p:
    read_fb()
    basis_fb()
    torch.cuda.synchronize()
for e in sorted(p.key_averages(), key=lambda e: -e.self_device_time_total)[:14]:
    print(f"{e.self_device_time_total / 1000:8.2f}ms n={e.count:4d} {e.key[:100]}")
