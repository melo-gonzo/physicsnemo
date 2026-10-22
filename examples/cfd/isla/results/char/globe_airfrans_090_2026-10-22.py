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

"""GLOBE on the AirfRANS interior task (2026-10-22): GLOBE at its published configuration (globe_af_seed42/43, 500 epochs,
trained on 4,096 points per step), scored in float32 on the final checkpoint on the 200 full_test cases, native mesh / uniform
point cloud / rotated + translated, on the same 100,000-point draw as every AirfRANS arm. Registered rules:
scratch/isla-globe-2026-10-19/PREREG.md (no adoption rule: the result adds a GLOBE column to the AirfRANS standing).

Source: results/campaigns_2026-10-22/globe/globe_af_scores.json, copied unmodified from scratch/isla-globe-2026-10-19/scores_af/
(reduce_af.py: infer_summary rows checked against the mean of the per-case rows, epoch 500 and 200 cases asserted). The
comparators are the arms of results/char/airfrans_090.json (written by r090_protocol_artifacts_2026-10-22.py; rerun this
script after it).

Writes results/char/globe_airfrans_090.json:
  arms["GLOBE"][protocol][field]   {mean, per_seed, n_cases, n_seeds}
  posed_over_native, area_over_native, posed_per_case_max_rel_dev   per field and seed
  over_baselines[arm][protocol][field]   GLOBE / baseline (seed means; per_seed seed-matched): final
  over_isla       GLOBE / ISLA's arm of airfrans_090.json, with that file's status and in-flight text: the AirfRANS standing
                  of ISLA is in flight, so this ratio is
  drivaerml       GLOBE on the DrivAerML Surface task: in flight, no numbers
usage: python globe_airfrans_090_2026-10-22.py"""

import json
import os

R = os.path.dirname(os.path.abspath(__file__))
SRC = f"{os.path.dirname(R)}/campaigns_2026-10-22/globe/globe_af_scores.json"
SEEDS = ("42", "43")
DRIVAERML = {
    "status": "in_flight",
    "in_flight": {
        "what": "GLOBE on the AirfRANS interior task alone",
        "replacement": "GLOBE on the DrivAerML Surface task",
    },
}


def over(a, b, protocols, fields):
    """{protocol: {field: {mean, per_seed}}} of arm a over arm b: seed means, and seed-matched per seed."""
    return {
        p: {
            f: {
                "mean": a[p][f]["mean"] / b[p][f]["mean"],
                "per_seed": {
                    s: a[p][f]["per_seed"][s] / b[p][f]["per_seed"][s] for s in SEEDS
                },
            }
            for f in fields
        }
        for p in protocols
    }


def main():
    """Write results/char/globe_airfrans_090.json."""
    g = json.load(open(SRC))
    af = json.load(open(f"{R}/airfrans_090.json"))
    fields, protocols = af["fields"], af["protocols"]
    globe = {
        p: {
            f: {
                **g["globe"][p][f],
                "n_cases": 200,
                "n_seeds": len(g["globe"][p][f]["per_seed"]),
            }
            for f in fields
        }
        for p in protocols
    }
    isla = af["arms"][af["isla_arm"]]
    out = {
        "protocol": __doc__,
        "status": "final",
        "in_flight": None,
        "protocols": protocols,
        "fields": fields,
        "arms": {"GLOBE": globe},
        **{
            k: g[k]
            for k in (
                "posed_over_native",
                "area_over_native",
                "posed_per_case_max_rel_dev",
            )
        },
        "over_baselines": {
            k: over(globe, v, protocols, fields)
            for k, v in af["arms"].items()
            if not k.startswith("ISLA")
        },
        "over_isla": {
            "arm": af["isla_arm"],
            "status": af["status"],
            "in_flight": af["in_flight"],
            **over(globe, isla, protocols, fields),
        },
        "drivaerml": DRIVAERML,
    }
    json.dump(out, open(f"{R}/globe_airfrans_090.json", "w"), indent=1)
    for p in protocols:
        print(f"GLOBE {p:7s}", " / ".join(f"{globe[p][f]['mean']:.5f}" for f in fields))
    for k, v in out["over_baselines"].items():
        print(
            f"GLOBE / {k:22s} native",
            " / ".join(f"{v['native'][f]['mean']:.3f}" for f in fields),
        )
    print(
        f"GLOBE / ISLA ({af['status']})       native",
        " / ".join(f"{out['over_isla']['native'][f]['mean']:.3f}" for f in fields),
    )


if __name__ == "__main__":
    main()
