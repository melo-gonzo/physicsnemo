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

"""Training-step cost of the kernel read on one GPU at the protocol's request.

Reads held-out cases exactly as ``infer.py`` does, builds the protocol model (bf16 autocast, compiled blocks, fused
geometry kernel) with the given ``++model.*`` overrides, and times forward + backward + AdamW steps on alternating
cases; reports the median step time, the peak memory and the kernel read's pair count.

usage (cwd the recipe): python bench_step.py --model M --dataset D --cells N --extra "<overrides>" --label X --out O
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch


def main():
    """Run the measurement described in the module docstring."""
    p = argparse.ArgumentParser()
    for k in ("model", "dataset", "label", "out"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--cells", type=int, required=True)
    p.add_argument("--extra", default="")
    p.add_argument("--steps", type=int, default=12)
    a = p.parse_args()
    for k, v in (
        ("RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("LOCAL_RANK", "0"),
        ("MASTER_ADDR", "127.0.0.1"),
        ("MASTER_PORT", str(25000 + os.getpid() % 20000)),
    ):
        os.environ.setdefault(k, v)
    import hydra
    from datasets import build_dataloaders
    from hydra import compose, initialize_config_dir
    from output_normalize import normalize_output_to_tensordict, require_output_type
    from utils import recursive_to_device

    from physicsnemo import datapipes  # noqa: F401
    from physicsnemo.distributed import DistributedManager

    overrides = [
        f"model={a.model}",
        *a.extra.split(),
        f"dataset={a.dataset}",
        "run_id=bench",
        "infer_split=val",
        f"sampling_resolution={a.cells}",
        "precision=bfloat16",
        "+save_predictions=false",
        "output_dir=/tmp/bench",
    ]
    overrides = [o for o in overrides if "training." not in o]
    with initialize_config_dir(config_dir=str(Path.cwd() / "conf"), version_base=None):
        cfg = compose(config_name="infer", overrides=overrides)
    DistributedManager.initialize()
    dev = DistributedManager().device
    _, loader, _, info = build_dataloaders(cfg)
    ds, coll = loader.dataset, loader.collate_fn
    batches = [
        recursive_to_device(coll([ds[i]]), dev) for i in list(loader.sampler)[:2]
    ]
    torch.manual_seed(0)
    model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    times, pairs = [], []
    torch.cuda.reset_peak_memory_stats()
    for s in range(a.steps):
        b = batches[s % 2]
        torch.cuda.synchronize()
        t = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**b["forward_kwargs"])
        pred = normalize_output_to_tensordict(
            out, info["targets"], require_output_type(cfg)
        )
        loss = sum(
            (pred[k].float() - b["targets"][k].float()).square().mean()
            for k in info["targets"]
        )
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.time() - t)
        pairs.append(int(getattr(model, "kernel_read_pairs", 0)))
        print(
            f"[{a.label}] step {s}: {times[-1] * 1000:.0f} ms loss {float(loss):.4f} pairs {pairs[-1]:,}",
            flush=True,
        )
    rec = {
        "label": a.label,
        "extra": a.extra,
        "params": n_params,
        "step_ms_median": 1000 * statistics.median(times[4:]),
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "pairs": pairs[-2:],
        "points": int(batches[0]["forward_kwargs"]["prediction_points"].n_points),
    }
    print(json.dumps(rec), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
