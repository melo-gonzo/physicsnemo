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

"""AF-EXTRAP clamp diagnostic (PREREG section 13, post hoc) on one common case set: the reynolds_test cases below the
training band (Re < 3e6), every arm averaged over the same cases.

clamp.json (clamp_reduce.py) averages each arm over its own cases outside its clamped range: for ISLA and ISLA-DTT that
leaves out the cases whose standardized log Re lies inside the training range, so its below-band means are over 241 cases
against the baselines' 254. The second read (scratch/isla-review-2026-10-21/second-read-2026-10-22/afx.md, item 6) recomputed
them on the 254 common cases; this script makes that number a file. A case inside an arm's clamped range is unchanged by the
clamp, so it enters with its native error in both columns.

Sources: the per-case scorer rows of scratch/isla-af-extrap-2026-10-21/scores/evals_afx/<run>/native/<run>/metrics.jsonl (native)
and scores/evals_afx_clamp/<run>/native/<run>/metrics.jsonl (clamped), copied by that lane from LHR $T/evals_afx and
$T/evals_afx_clamp. A case's Reynolds number is its inlet speed (from the case name) times the 1 m chord over the kinematic
viscosity 1.56e-5 m^2/s.

Writes afx_clamp_common.json beside this script:
  {"n_cases": n, "arms": {arm: {"native": {field: seed mean}, "clamped": {field: seed mean}, "ratio": {field: clamped / native},
                                "per_seed": {seed: {"native": {...}, "clamped": {...}}}}}}

usage: python afx_clamp_common_2026-10-22.py [scratch root, default ~/gh/physicsnemo/scratch]
"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(
    sys.argv[1] if len(sys.argv) > 1 else Path.home() / "gh/physicsnemo/scratch"
)
SCORES = ROOT / "isla-af-extrap-2026-10-21" / "scores"
ARMS = {
    "ISLA": "afxf_reynolds_isla",
    "ISLA-DTT": "afxf_reynolds_dtt",
    "GeoTransolver": "afx_reynolds_gt",
    "Transolver": "afx_reynolds_t",
}
SEEDS = ("42", "43")
FIELDS = ("pressure_l2", "velocity_l2", "nut_l2")
NU, CHORD, BAND_LO = 1.56e-5, 1.0, 3e6


def per_case(path: Path) -> dict:
    """{case name: metrics} from a scorer's metrics.jsonl (the last infer_step row per case)."""
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("phase") == "infer_step":
            sid = r["sample_id"].split("_", 1)[1]
            out[sid[: len(sid) // 2]] = r["metrics"]
    return out


def reynolds(case: str) -> float:
    """The Reynolds number of a case: its inlet speed (from the case name) times the chord over the viscosity."""
    return float(case.split("_")[2]) * CHORD / NU


rows = {}
for arm, stem in ARMS.items():
    for s in SEEDS:
        run = f"{stem}_seed{s}"
        rows[arm, s, "native"] = per_case(
            SCORES / "evals_afx" / run / "native" / run / "metrics.jsonl"
        )
        rows[arm, s, "clamped"] = per_case(
            SCORES / "evals_afx_clamp" / run / "native" / run / "metrics.jsonl"
        )
case_sets = {frozenset(v) for v in rows.values()}
assert len(case_sets) == 1, "every run must score the same cases"
below = sorted(c for c in next(iter(case_sets)) if reynolds(c) < BAND_LO)

out = {"n_cases": len(below), "arms": {}}
for arm in ARMS:
    per_seed = {
        s: {
            k: {
                f: statistics.mean(rows[arm, s, k][c][f] for c in below) for f in FIELDS
            }
            for k in ("native", "clamped")
        }
        for s in SEEDS
    }
    m = {
        k: {f: statistics.mean(per_seed[s][k][f] for s in SEEDS) for f in FIELDS}
        for k in ("native", "clamped")
    }
    out["arms"][arm] = {
        **m,
        "ratio": {f: m["clamped"][f] / m["native"][f] for f in FIELDS},
        "per_seed": per_seed,
    }
json.dump(out, open(Path(__file__).with_name("afx_clamp_common.json"), "w"), indent=1)
for arm, v in out["arms"].items():
    print(
        arm,
        len(below),
        " ".join(
            f"{f} {v['native'][f]:.4f} -> {v['clamped'][f]:.4f} ({v['ratio'][f]:.3f})"
            for f in FIELDS
        ),
    )
