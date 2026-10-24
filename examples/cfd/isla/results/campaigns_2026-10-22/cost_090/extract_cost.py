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

"""Compact cost extract of ISLA 0.9.0 and baseline training logs on LHR, streamed over ssh (read only; prints one JSON line per run).

Per run under $T/runs/<run>: the md5 and size of metrics.jsonl; its last training config row; per epoch the last train_summary row's
epoch_seconds and peak_memory_gib (peak allocated device memory, the trainer's maximum over ranks); trainer step-time statistics over the
train_step rows with step_time_s < 10 s (the AZ-DECIDE rule: leaves out the compile steps of the first epoch and of each restarted link);
the CUDA devices train.log names (the ranks), its nodes and code snapshots; and the last config row of the run's native-mesh scorer output (num_parameters).
usage: ssh lhr2 "python - run [run ...]" < extract_cost.py > extract.jsonl
"""

import glob
import hashlib
import json
import os
import re
import statistics
import sys

T = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"
SCORER = (
    "evals_*/{r}/native/{r}/metrics.jsonl",
    "p200k_fp32_test_azc/{r}/{r}/metrics.jsonl",
)


def last_config(path):
    """The last config row of a metrics.jsonl, or None."""
    cfg = None
    with open(path, "rb") as fh:
        for line in fh:
            if b'"phase": "config"' in line:
                cfg = json.loads(line)
    return cfg


for run in sys.argv[1:]:
    d = f"{T}/runs/{run}"
    p = f"{d}/metrics.jsonl"
    if not os.path.exists(p):
        print(json.dumps({"run": run, "missing": p}), flush=True)
        continue
    md5, cfg, train, steps, n_slow = hashlib.md5(), None, {}, [], 0
    with open(p, "rb") as fh:
        for line in fh:
            md5.update(line)
            if b'"train_step"' in line:
                s = json.loads(line).get("step_time_s")
                if s is not None:
                    if s < 10:
                        steps.append(s)
                    else:
                        n_slow += 1
            elif b'"phase": "config"' in line:
                cfg = json.loads(line)
            elif b'"phase": "train_summary"' in line:
                r = json.loads(line)
                train[r["epoch"]] = [r.get("epoch_seconds"), r.get("peak_memory_gib")]
    log = open(f"{d}/train.log", errors="replace").read()
    scorer = [
        f for pat in SCORER for f in sorted(glob.glob(f"{T}/{pat.format(r=run)}"))
    ]
    sc = last_config(scorer[0]) if scorer else None
    rec = {
        "run": run,
        "source": p,
        "md5": md5.hexdigest(),
        "bytes": os.path.getsize(p),
        "config": {k: v for k, v in (cfg or {}).items() if k != "params"},
        "config_params": {
            k: v
            for k, v in ((cfg or {}).get("params") or {}).items()
            if k.startswith(
                (
                    "precision",
                    "compile",
                    "model._target_",
                    "model.compile",
                    "training.seed",
                    "training.num_epochs",
                )
            )
        },
        "devices": sorted(set(re.findall(r"cuda:\d", log))),
        "nodes": sorted(set(re.findall(r"node=(\S+)", log))),
        "code": sorted({os.path.basename(c) for c in re.findall(r" code=(\S+)", log)}),
        "completed": "Training completed" in log,
        "train": train,
        "step": {
            "n": len(steps),
            "n_slow": n_slow,
            "median": statistics.median(steps) if steps else None,
            "mean": statistics.fmean(steps) if steps else None,
        },
        "scorer_source": scorer[0] if scorer else None,
        "scorer_config": {k: v for k, v in (sc or {}).items() if k != "params"},
    }
    print(json.dumps(rec), flush=True)
