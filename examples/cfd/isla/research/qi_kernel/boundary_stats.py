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

"""Boundary-sample statistics that fix the kernel read's radius and candidate count.

For a few held-out cases read exactly as ``infer.py`` reads them (same Hydra composition and pipeline, the protocol's
sample size), report the length unit L (the model's measure-weighted RMS pairwise distance), the boundary measure in
units of L^(d-1), the |stored sdf| / L distribution of the prediction points, and, for radii R (in units of L), the
number of boundary cells within R of each prediction point (mean, quantiles, max) and the measure inside it relative to
the flat-wall disc (pi R^2 L^2 in 3-D, 2 R L in 2-D). Also times the library kNN (k candidates) at the protocol sizes.

usage (cwd the recipe): python boundary_stats.py --model M --dataset D --cells N --extra "<overrides>" --out O.json
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

RADII = (0.005, 0.01, 0.02, 0.03, 0.05, 0.1)
KS = (32, 64, 128, 256)


def main():
    """Run the measurement described in the module docstring."""
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--cells", type=int, required=True)
    p.add_argument("--extra", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--cases", type=int, default=3)
    a = p.parse_args()
    for k, v in (
        ("RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("LOCAL_RANK", "0"),
        ("MASTER_ADDR", "127.0.0.1"),
        ("MASTER_PORT", str(25000 + os.getpid() % 20000)),
    ):
        os.environ.setdefault(k, v)
    from datasets import build_dataloaders
    from hydra import compose, initialize_config_dir
    from utils import recursive_to_device

    from physicsnemo import datapipes  # noqa: F401
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.experimental.models.isla.geometry import rms_pairwise_distance
    from physicsnemo.mesh.calculus.measure import cell_measures
    from physicsnemo.nn.functional.neighbors.knn.knn import knn

    overrides = [
        f"model={a.model}",
        *a.extra.split(),
        f"dataset={a.dataset}",
        "run_id=stats",
        "infer_split=val",
        f"sampling_resolution={a.cells}",
        "precision=float32",
        "+save_predictions=false",
        "output_dir=/tmp/stats",
    ]
    overrides = [o for o in overrides if "training." not in o]
    with initialize_config_dir(config_dir=str(Path.cwd() / "conf"), version_base=None):
        cfg = compose(config_name="infer", overrides=overrides)
    DistributedManager.initialize()
    dev = DistributedManager().device
    _, loader, _, _ = build_dataloaders(cfg)
    ds, coll = loader.dataset, loader.collate_fn
    recs = []
    for ci, idx in enumerate(list(loader.sampler)[: a.cases]):
        b = recursive_to_device(coll([ds[idx]]), dev)
        fk = b["forward_kwargs"]
        meshes = list(fk["boundary_meshes"].values())
        y = torch.cat([m.cell_centroids for m in meshes]).float()
        w = torch.cat([cell_measures(m) for m in meshes]).float()
        x = fk["prediction_points"].points.float()
        d = y.shape[-1]
        L = float(rms_pairwise_distance(y[None], w[None]).reshape(()))
        sdf = fk["prediction_points"].point_data["sdf"].reshape(-1).abs().float() / L
        rec = {
            "case": ci,
            "n_cells": y.shape[0],
            "n_points": x.shape[0],
            "dim": d,
            "L": L,
            "measure_over_L^(d-1)": float(w.sum() / L ** (d - 1)),
            "cell_measure_over_L^(d-1)_quantiles": [
                float(v)
                for v in torch.quantile(
                    w[torch.randperm(w.numel(), device=dev)[:100000]] / L ** (d - 1),
                    torch.tensor([0.01, 0.5, 0.99, 1.0], device=dev),
                )
            ],
            "sdf_over_L_quantiles": {
                str(q): float(torch.quantile(sdf[:100000], q))
                for q in (0.1, 0.25, 0.5, 0.72, 0.9, 0.98)
            },
        }
        ### counts within R: exact chunked cdist (float32 on the GPU)
        cnt = {R: [] for R in RADII}
        mass = {R: [] for R in RADII}
        nn_dist = []
        for xc in x.split(4096):
            dist = torch.cdist(xc, y) / L
            nn_dist.append(dist.min(dim=1).values)
            for R in RADII:
                inside = dist < R
                cnt[R].append(inside.sum(1))
                disc = math.pi * R**2 if d == 3 else 2 * R
                mass[R].append(
                    (inside.float() * w[None]).sum(1) / (disc * L ** (d - 1))
                )
        nn_dist = torch.cat(nn_dist)
        rec["nearest_cell_over_L_quantiles"] = {
            str(q): float(torch.quantile(nn_dist[:100000], q))
            for q in (0.01, 0.5, 0.9, 0.99)
        }
        rec["radius"] = {}
        for R in RADII:
            c = torch.cat(cnt[R]).float()
            m = torch.cat(mass[R])
            qs = torch.tensor([0.5, 0.9, 0.99, 0.999], device=dev)
            rec["radius"][str(R)] = {
                "frac_nonempty": float((c > 0).float().mean()),
                "mean": float(c.mean()),
                "q50_90_99_999": [float(v) for v in torch.quantile(c[:200000], qs)],
                "max": float(c.max()),
                "mass_over_disc_q50_90_99_max": [
                    float(v) for v in torch.quantile(m[:200000], qs[:3])
                ]
                + [float(m.max())],
            }
        ### kNN timings (the library search the model uses)
        rec["knn_ms"] = {}
        for k in KS:
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(3):
                knn(y.contiguous(), x.contiguous(), k)
            torch.cuda.synchronize()
            rec["knn_ms"][str(k)] = 1000 * (time.time() - t) / 3
        print(json.dumps(rec), flush=True)
        recs.append(rec)
    Path(a.out).write_text(json.dumps(recs, indent=1))


if __name__ == "__main__":
    main()
