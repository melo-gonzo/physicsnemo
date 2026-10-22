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

"""SHIFT-SUV across families (2026-10-22): the estate-trained standing checkpoints (ISLA 0.9.0 r090_suv_isla, GeoTransolver
256 clud_suv_gt, Transolver 512 x 16 clud_suv_t; seeds 42 and 43, epoch 500, float32, ISLA's 200,000-cell set) scored
zero-shot on unseen fastbacks (fastback test 99 cases, fastback val 99) and in family on estate test (100); estate val is the
standing's own split. Registered rules: scratch/isla-suv-xfam-2026-10-22/PREREG.md (md5 c0d0caef).

Sources, copied unmodified into results/campaigns_2026-10-22/suv_xfam/ from scratch/isla-suv-xfam-2026-10-22/: reduce_out.json
(reduce.py, the registered rules, run on LHR over $T/evals_suvx/<run>/<split>/<run>/metrics.jsonl: E = the infer_summary row,
the mean over cases of the per-case relative L2) and percase_out.json (percase.py, beside the rules: case bootstrap).

Writes results/char/shift_suv_xfam_090.json:
  arms[arm][split][field]  {mean, per_seed, n_cases, n_seeds}: seed means of E
  R1[split][cell]          r = baseline / ISLA (seed means), its band b1, verdict as registered ('baseline leads' / 'trails' / 'level')
  R2[split]                penalties P = E(fastback) / E(estate test) per arm and field (P_geo: geometric seed mean, P_seeds),
                           cells D = ln(P_B / P_ISLA) with SE, z and verdict, and the summary as registered
  B_trigger, predictions   as registered
  case_level               percase_out.json (fraction of cases ISLA is lower on, case-bootstrap intervals; no verdict)
  part_b                   {status: in_flight, in_flight: {what, replacement}}: the fastback-trained comparison
Part A is final; the file's own status is final, with part B's status inside part_b.
usage: python shift_suv_xfam_090_2026-10-22.py"""

import json
import os
import statistics as st

R = os.path.dirname(os.path.abspath(__file__))
SRC = f"{os.path.dirname(R)}/campaigns_2026-10-22/suv_xfam"
ARMS = {
    "isla": "ISLA 512 with 512 slices",
    "gt": "GeoTransolver 256",
    "t": "Transolver 512 x 16 at 100k cells",
}
BASE = {"gt": "GeoTransolver", "t": "Transolver"}
SPLITS = {
    "fb_test": ("fastback test", 99),
    "fb_val": ("fastback val", 99),
    "est_test": ("estate test", 100),
    "est_val": ("estate val", 99),
}
FIELDS = {"pressure_l2": "pressure", "wss_l2": "wall shear"}
SEEDS = ("42", "43")
PART_B = {
    "status": "in_flight",
    "in_flight": {
        "what": "models trained on estates only, so the penalty mixes the transfer gap with how hard fastbacks are in themselves",
        "replacement": "The reverse direction, ISLA and Transolver trained on fastbacks and scored on estates,",
    },
}


def main():
    """Write results/char/shift_suv_xfam_090.json."""
    red = json.load(open(f"{SRC}/reduce_out.json"))
    assert red["checks_all_pass"] and all(
        all(v for k, v in c.items() if k != "run") for c in red["checks"]
    )
    E = red["E"]
    arms = {
        label: {
            sp: {
                f: {
                    "mean": st.mean(E[f"{a}/{s}/{sp}/{f}"] for s in SEEDS),
                    "per_seed": {s: E[f"{a}/{s}/{sp}/{f}"] for s in SEEDS},
                    "n_cases": n,
                    "n_seeds": len(SEEDS),
                }
                for f in FIELDS
            }
            for sp, (_, n) in SPLITS.items()
        }
        for a, label in ARMS.items()
    }
    rename = lambda cells: {
        f"{BASE[k.split('/')[0]]}/{k.split('/')[1]}": v for k, v in cells.items()
    }  # noqa: E731
    out = {
        "protocol": __doc__,
        "status": "final",
        "in_flight": None,
        "splits": {k: v[0] for k, v in SPLITS.items()},
        "fields": FIELDS,
        "isla_arm": ARMS["isla"],
        "arms": arms,
        "R1": {sp: rename(c) for sp, c in red["R1"].items()},
        "R2": {
            sp: {
                "penalties": {
                    f"{ARMS[k.split('/')[0]]}/{k.split('/')[1]}": v
                    for k, v in r["penalties"].items()
                },
                "cells": rename(r["cells"]),
                "summary": r["summary"],
            }
            for sp, r in red["R2"].items()
        },
        "B_trigger": red["B_trigger"],
        "predictions": red["predictions"],
        "case_level": json.load(open(f"{SRC}/percase_out.json")),
        "part_b": PART_B,
    }
    json.dump(out, open(f"{R}/shift_suv_xfam_090.json", "w"), indent=1)
    for label, a in arms.items():
        print(
            f"{label:34s}",
            " ".join(
                f"{sp}:" + "/".join(f"{a[sp][f]['mean']:.4f}" for f in FIELDS)
                for sp in SPLITS
            ),
        )
    print(
        "R2 fb_test:",
        out["R2"]["fb_test"]["summary"],
        {k: v["verdict"] for k, v in out["R2"]["fb_test"]["cells"].items()},
    )


if __name__ == "__main__":
    main()
