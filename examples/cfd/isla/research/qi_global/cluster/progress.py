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

"""QI-GLOBAL progress: per lane the last epoch and the trainer's validation relative L2 (p / U / nut), and for G and T
the ratios to I and P of the same seed at matched epochs (training-log values; no verdict rests on them).
usage: python progress.py [epochs...]"""

import re
import sys

AG = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents"
T = f"{AG}/2026-10-05-isla-qi-global"
FIELDS = ("pressure_l2", "velocity_l2", "nut_l2")
RUNS = {
    "I": {
        42: f"{T}/refs/runs/qi_af_int_seed42",
        43: f"{T}/refs/runs/qi_af_int_seed43",
        44: f"{T}/runs/gl_af_int_seed44",
        45: f"{T}/runs/gl_af_int_seed45",
    },
    "P": {
        42: f"{T}/refs/runs/qi_af_pas_seed42",
        43: f"{T}/refs/runs/qi_af_pas_seed43",
        44: f"{T}/runs/gl_af_ro_seed44",
        45: f"{T}/runs/gl_af_ro_seed45",
    },
    "T": {
        42: f"{T}/refs/runs/kq_af_sum4_seed42",
        43: f"{T}/refs/runs/kq_af_sum4_seed43",
        44: f"{T}/runs/gl_af_trunc_seed44",
        45: f"{T}/runs/gl_af_trunc_seed45",
    },
    "G": {s: f"{T}/runs/gl_af_global_seed{s}" for s in (42, 43, 44, 45)},
}


def parse(path):
    """Per-epoch validation fields of a run's train.log and its last epoch."""
    out, ep = {}, None
    try:
        lines = open(f"{path}/train.log").read().splitlines()
    except OSError:
        return out, None
    last = None
    for line in lines:
        m = re.match(r"^Epoch \[(\d+)/\d+\]", line)
        if m:
            ep = int(m.group(1))
            last = ep
            continue
        m = re.match(r"^\|\s*(\S+)\s*\|\s*([0-9.e-]+)\s*\|\s*([0-9.e-]+)\s*\|", line)
        if m and ep is not None and m.group(1) in FIELDS:
            out.setdefault(ep, {})
            if m.group(1) not in out[ep]:
                out[ep][m.group(1)] = float(m.group(3))
    return out, last


logs = {a: {s: parse(p) for s, p in r.items()} for a, r in RUNS.items()}
for a in ("I", "P", "T", "G"):
    for s in (44, 45) if a != "G" else (42, 43, 44, 45):
        v, last = logs[a][s]
        full = [e for e in v if len(v[e]) == 3]
        e = max(full, default=None)
        print(
            f"{a} seed {s}: epoch {last}; val@{e}: "
            + (" ".join(f"{v[e][f]:.4f}" for f in FIELDS) if e is not None else "-")
        )
epochs = [int(e) for e in sys.argv[1:]]
for a in ("G", "T"):
    for s in (42, 43, 44, 45):
        v = logs[a][s][0]
        if not v:
            continue
        es = epochs or sorted(e for e in v if len(v[e]) == 3)[-3:]
        for e in es:
            row = [
                f"{a}/{b} "
                + " ".join(f"{v[e][f] / logs[b][s][0][e][f]:.3f}" for f in FIELDS)
                for b in ("I", "P")
                if e in v
                and e in logs[b][s][0]
                and len(logs[b][s][0][e]) == 3
                and len(v[e]) == 3
            ]
            if row:
                print(f"  seed {s} epoch {e}: " + " | ".join(row))
