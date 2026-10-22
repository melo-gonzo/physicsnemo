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

"""COST-090 reduction (RENDER-090 lane, 2026-10-22): renderings_2026-10-18/scripts/cost_reduce.py for the current standing's
models: merge the six per-arm bench_step_090.py outputs of LHR job 336005 (cost.sbatch, one GB300 node, each arm on its own
GPU, the four GPU streams concurrent) into ../cost_sweep_ref_090_gb300_bf16.json.

Per arm and token count: median, p10 and p90 of 25 timed training steps (forward + loss + backward + optimizer step,
bf16 autocast, after 8 untimed warm-up steps that include compilation), the peak allocated memory over the timed steps,
the resident bytes (parameters + optimizer state + the batch) before them, and the model's input shapes. "tokens" is the
DrivAerML sampling_resolution: surface cells of the vehicle (ISLA: boundary_meshes.vehicle.cells; GeoTransolver and
Transolver: geometry / embedding points). For the volume arms it is the number of prediction points; ISLA 0.9.0 reads the
vehicle surface whole (drivaer_ml_volume_reference.yaml has no cell cap) and draws min(cells, boundary_tokens = 200,000)
boundary tokens inside the model, so its boundary token count no longer follows N (the A2 sweep had N boundary cells plus
N prediction points); the input shapes give the vehicle's full cell count.
ISLA 0.9.0 arms: code_cmpaf_fix (code_r090, 92d9d7800, with per-block compile at Inductor's mix-order reduction off, as
psharpe/isla compiles every block since b62a93f2c), whole-model compile off and compile_blocks on, as the r090 lanes'
resolved_config.yaml records; isla_surface_090_codeR090 is the same ISLA on code_r090, the snapshot the r090 DrivAerML lanes
trained on (mix-order reduction at Inductor's default). ISLA-DTT: query_independent=true, tree read at theta = 1
(TREE_READ_THETA). The baselines read U_inf_dir, as the standing's fair-drive arms do (the same shapes as U_inf).
The learning rate does not enter the step time.
Source files: scratch/isla-render090-2026-10-22/cost/<arm>.json (copies of LHR $T/book_figs_2026-10-22/cost/<arm>.json).
usage: python cost_reduce.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "/home/psharpe/gh/physicsnemo/scratch/isla-render090-2026-10-22/cost"
ARMS = {
    "isla_surface_090": dict(
        name="ISLA",
        task="DrivAerML Surface",
        code="code_cmpaf_fix (92d9d7800 + mix-order reduction off)",
        recipe="recipe_r090",
    ),
    "geotransolver_surface_256": dict(
        name="GeoTransolver 256",
        task="DrivAerML Surface",
        code="code_v20 (7ef7c34fa)",
        recipe="recipe_v8 (343b284eb)",
    ),
    "transolver_surface_512x16": dict(
        name="Transolver 512 x 16",
        task="DrivAerML Surface",
        code="code_v20 (7ef7c34fa)",
        recipe="recipe_v8 (343b284eb)",
    ),
    "isla_volume_090": dict(
        name="ISLA",
        task="DrivAerML Volume",
        code="code_cmpaf_fix (92d9d7800 + mix-order reduction off)",
        recipe="recipe_r090",
    ),
    "isla_dtt_volume_090": dict(
        name="ISLA-DTT",
        task="DrivAerML Volume",
        code="code_cmpaf_fix (92d9d7800 + mix-order reduction off)",
        recipe="recipe_r090",
    ),
    "isla_surface_090_codeR090": dict(
        name="ISLA (code_r090: mix-order reduction at Inductor's default)",
        task="DrivAerML Surface",
        code="code_r090 (92d9d7800)",
        recipe="recipe_r090",
    ),
}

out = {"protocol": __doc__, "job": "LHR 336005", "arms": {}}
for arm, meta in ARMS.items():
    d = json.load(open(f"{SRC}/{arm}.json"))
    out.setdefault("env", d["env"])
    rows = []
    for r in d["runs"]:
        row = {
            "tokens": r["tokens_requested"],
            "status": r["status"],
            "input_shapes": r.get("input_shapes"),
        }
        if r["status"] == "ok":
            row.update(
                {
                    k: r[k]
                    for k in (
                        "step_ms_median",
                        "step_ms_p10",
                        "step_ms_p90",
                        "peak_allocated_gib",
                        "peak_allocated_bytes",
                        "resident_bytes",
                        "warmup_s",
                        "step_ms",
                    )
                }
            )
        else:
            row["error"] = r.get("error", "")[-300:]
        rows.append(row)
    out["arms"][arm] = {
        **meta,
        "model_yaml": d["model"],
        "dataset_yaml": d["dataset"],
        "extra": d["extra"],
        "n_params": d.get("n_params"),
        "model_build": d.get("model_build"),
        "compile": d.get("compile"),
        "precision": d.get("precision"),
        "warmup_steps": d["warmup"],
        "timed_steps": d["steps"],
        "gpu_env": d["env"],
        "runs": rows,
    }
    print(
        f"{arm:28s} "
        + "  ".join(
            f"{r['tokens'] // 1000}k:"
            + (
                f"{r['step_ms_median']:.0f}ms/{r['peak_allocated_gib']:.0f}G"
                if r["status"] == "ok"
                else r["status"]
            )
            for r in rows
        )
    )
json.dump(
    out,
    open(
        os.path.join(os.path.dirname(HERE), "cost_sweep_ref_090_gb300_bf16.json"), "w"
    ),
    indent=1,
)
