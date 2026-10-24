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

"""COST-090 (RENDER-090 lane, 2026-10-22): renderings_2026-10-18/scripts/bench_step.py with one change, the model build of
recipe_r090's train.py: where the recipe has output_normalize.model_output_stats (recipe_r090), the model is instantiated
with the dataset normalizer's output stats and the dataset constants (model_dataset_constants), as train.py does for ISLA
0.9.0, whose outputs carry its own scales; elsewhere (recipe_v8, the baselines) as before. Everything else is the original:

COST (FIG lane, 2026-10-18): training-step time and peak memory against token count for the reference configurations,
each built and stepped as its own recipe snapshot's train.py builds and steps it.

One process per arm, one GPU. The Hydra train config is composed from the arm's recipe with the lane's overrides
(model, dataset, the lane's extra, sampling_resolution = N), then, as train.py does:
  * the model is ``hydra.utils.instantiate(cfg.model)``; compiled with ``utils.compile_model(model, cfg)`` where the recipe
    has it (recipe_a2r: ``torch.compile(model, **cfg.compile_options)``, i.e. fullgraph + dynamic +
    max-autotune-no-cudagraphs for ISLA), else ``torch.compile(model)`` when ``cfg.compile`` (recipe_v8, the baselines);
  * the optimizer is ``utils.build_muon_optimizer(model, cfg, compile_optimizer=cfg.compile)`` (Muon on matrices,
    AdamW on the rest), the loss ``LossCalculator`` with the cfg's loss type and field weights, the step
    ``train.forward_pass`` (autocast at cfg.precision = bfloat16; loss in float32) + zero_grad + backward +
    optimizer.step, exactly the body of train.py's ``_run_epoch`` for bfloat16 (the finite-gradient check included).
  * the batch is one real training sample of the dataset at N cells (its train loader's first batch, augmentations on),
    moved to the GPU once and reused for every step; data loading is not timed.
Per N: ``--warmup`` untimed steps (compilation and autotuning happen here), then ``--steps`` timed steps, each bracketed
by torch.cuda.synchronize(); peak = torch.cuda.max_memory_allocated over the timed steps (reset after warm-up),
reported together with the resident (parameters + optimizer state) bytes before the step. An out-of-memory error is a
result and ends the arm's sweep. The model and optimizer persist across N (as in training, where one compiled model
sees every sample).
usage: python bench_step.py --label L --recipe-dir D --model M --dataset DS [--extra "k=v ..."] --tokens 25000,50000
       --out out.json [--warmup 8 --steps 25]
"""

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback

import torch


def env_info():
    """The GPU, driver, software and host of this process, for the output json."""
    p = torch.cuda.get_device_properties(0)
    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,clocks.max.sm",
                "--format=csv,noheader",
                "-i",
                os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        smi = repr(e)
    return dict(
        gpu=torch.cuda.get_device_name(0),
        nvidia_smi=smi,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        python=platform.python_version(),
        hostname=socket.gethostname(),
        total_memory=p.total_memory,
        sm_count=p.multi_processor_count,
        capability=f"{p.major}.{p.minor}",
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        pythonpath=os.environ.get("PYTHONPATH", ""),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
    )


def main():
    """Sweep the token counts for one arm and write its json (module docstring)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--recipe-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--extra", default="")
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--steps", type=int, default=25)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.recipe_dir, "src"))
    import hydra
    from hydra import compose, initialize_config_dir

    import train as T  # the recipe's train.py (forward_pass)
    import utils as U
    from datasets import build_dataloaders
    from loss import LossCalculator
    from metrics import MetricCalculator, resolve_metrics
    from output_normalize import require_output_type
    from physicsnemo.distributed import DistributedManager

    DistributedManager.initialize()
    device = DistributedManager().device
    tokens = [int(t) for t in args.tokens.split(",")]
    extra = [
        t for t in args.extra.split() if t and not t.startswith("training.num_epochs")
    ]
    res = dict(
        label=args.label,
        env=env_info(),
        recipe_dir=args.recipe_dir,
        model=args.model,
        dataset=args.dataset,
        extra=extra,
        warmup=args.warmup,
        steps=args.steps,
        runs=[],
    )
    model = opt = None
    for n in tokens:
        overrides = [
            f"model={args.model}",
            f"dataset={args.dataset}",
            f"sampling_resolution={n}",
            "run_id=bench",
            "output_dir=/tmp/fig_bench",
            *extra,
        ]
        with initialize_config_dir(
            config_dir=os.path.join(args.recipe_dir, "conf"), version_base=None
        ):
            cfg = compose(config_name="train", overrides=overrides)
        rec = dict(tokens_requested=n, overrides=overrides)
        try:
            tl, _vl, norm, info = build_dataloaders(cfg)
            batch = U.recursive_to_device(next(iter(tl)), device)
            target_config = info["targets"]
            rec["n_targets"] = (
                int(batch["targets"].batch_size[-1])
                if hasattr(batch["targets"], "batch_size")
                else None
            )
            fk = batch["forward_kwargs"]
            shapes = {}
            for k, v in fk.items():
                try:
                    if hasattr(v, "points"):
                        shapes[k] = list(v.points.shape)
                    elif isinstance(v, dict) or hasattr(v, "items"):
                        for kk, vv in v.items():
                            if hasattr(vv, "points"):
                                shapes[f"{k}.{kk}"] = list(vv.points.shape)
                                if hasattr(vv, "cells") and vv.cells is not None:
                                    shapes[f"{k}.{kk}.cells"] = list(vv.cells.shape)
                    elif torch.is_tensor(v):
                        shapes[k] = list(v.shape)
                except Exception:  # noqa: BLE001
                    pass
            rec["input_shapes"] = shapes
            if model is None:
                try:
                    from datasets import load_dataset_config
                    from output_normalize import (
                        check_model_output_stats,
                        model_dataset_constants,
                        model_output_stats,
                    )
                except ImportError:  # recipe_v8: no model-side output stats
                    model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(
                        device
                    )
                    res["model_build"] = "hydra.utils.instantiate(cfg.model)"
                else:
                    ds_cfg = load_dataset_config(
                        os.path.join(args.recipe_dir, "datasets", f"{cfg.dataset}.yaml")
                    )
                    model = hydra.utils.instantiate(
                        cfg.model,
                        _convert_="partial",
                        **model_output_stats(cfg.model, norm),
                        **model_dataset_constants(cfg.model, ds_cfg),
                    ).to(device)
                    check_model_output_stats(model, norm)
                    res["model_build"] = (
                        "train.py: instantiate(cfg.model, **model_output_stats, **model_dataset_constants)"
                    )
                res["n_params"] = sum(p.numel() for p in model.parameters())
                opt = U.build_muon_optimizer(model, cfg, compile_optimizer=cfg.compile)
                if hasattr(U, "compile_model"):
                    model = U.compile_model(model, cfg)
                    res["compile"] = dict(
                        how="utils.compile_model",
                        compile=bool(cfg.compile),
                        options=dict(cfg.get("compile_options", {}) or {}),
                    )
                elif cfg.compile:
                    model = torch.compile(model)
                    res["compile"] = dict(
                        how="torch.compile(model)", compile=True, options={}
                    )
                else:
                    res["compile"] = dict(how="none", compile=False)
                res["precision"] = cfg.precision
                res["lr"] = float(cfg.training.optimizer.lr)
                res["model_cfg"] = {
                    k: (
                        v
                        if isinstance(v, (int, float, str, bool, type(None)))
                        else str(v)
                    )
                    for k, v in dict(cfg.model).items()
                }
                loss_calc = LossCalculator(
                    target_config=target_config,
                    loss_type=cfg.training.get("loss_type", "huber"),
                    field_weights=U.resolve_dict(cfg, "training.field_weights"),
                )
                metric_calc = MetricCalculator(
                    target_config=target_config, metrics=resolve_metrics(cfg)
                )
                output_type = require_output_type(cfg)
            model.train()

            def step():
                loss, _l, _m = T.forward_pass(
                    batch,
                    model,
                    cfg.precision,
                    loss_calc,
                    metric_calc,
                    output_type=output_type,
                    target_config=target_config,
                )
                opt.zero_grad()
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                if torch.nn.utils.get_total_norm(grads).isfinite():
                    opt.step()
                return loss

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.warmup):
                step()
            torch.cuda.synchronize()
            rec["warmup_s"] = time.perf_counter() - t0
            torch.cuda.reset_peak_memory_stats()
            rec["resident_bytes"] = torch.cuda.memory_allocated()
            ms = []
            for _ in range(args.steps):
                torch.cuda.synchronize()
                t = time.perf_counter()
                loss = step()
                torch.cuda.synchronize()
                ms.append(1e3 * (time.perf_counter() - t))
            srt = sorted(ms)
            rec.update(
                status="ok",
                step_ms=ms,
                step_ms_median=srt[len(srt) // 2]
                if len(srt) % 2
                else 0.5 * (srt[len(srt) // 2 - 1] + srt[len(srt) // 2]),
                step_ms_p10=srt[max(0, int(round(0.1 * (len(srt) - 1))))],
                step_ms_p90=srt[int(round(0.9 * (len(srt) - 1)))],
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                reserved_bytes=torch.cuda.memory_reserved(),
                last_loss=float(loss),
            )
        except torch.OutOfMemoryError as e:
            rec.update(status="oom", error=str(e)[-400:])
        except Exception as e:  # noqa: BLE001
            rec.update(status="error", error=traceback.format_exc()[-2000:])
        res["runs"].append(rec)
        print(
            f"{args.label} N={n}: {rec['status']} median {rec.get('step_ms_median', float('nan')):.1f} ms "
            f"p10 {rec.get('step_ms_p10', float('nan')):.1f} peak {rec.get('peak_allocated_gib', float('nan')):.2f} GiB "
            f"shapes {rec.get('input_shapes')} {rec.get('error', '')[-300:]}",
            flush=True,
        )
        json.dump(res, open(args.out, "w"), indent=1)
        if rec["status"] != "ok":
            break
        del batch
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
