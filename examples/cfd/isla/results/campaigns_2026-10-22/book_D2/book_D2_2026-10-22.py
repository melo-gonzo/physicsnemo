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

"""Reduce three cluster outputs that the Tensile2d and lug chapters (15, 16) read, and that no record holds in reduced form,
into book_D2.json beside this file.

``lug_closest_feature_ties``: ATTRIB-090's geometry probe (job 334176, launcher geo_flip_probe.py; RESULTS.md section 4 of
results/campaigns_2026-10-22/attrib090/). On 60 lug test cases it computes ISLA 0.9.0's exact query geometry (signed distance
s = d / L and direction nu = sign(d) u from the closest surface feature) twice, on the stored surface and on the surface with
every triangle's vertex order reversed and its normal negated, which negates the geometry exactly where the closest feature is
unique. Its nodes.jsonl has one row per case and one total row. Kept here: the totals, the per-case range of interior nodes
whose direction changes, and the largest deviations.

``tensile2d_eager_validation``: the in-training validation rows (``"phase": "val_summary"``, every 10 epochs and the last) of
the eager-trained Tensile2d standing, r090eag_t2dz_isla seeds 42-45: per seed the epoch and value of the lowest displacement
error, the largest value after it, and the last value.

``tensile2d_never_yield``: the validation specimens (manifest split val_100) whose plastic strain q is identically zero, from
the converter's per-case record (largest q and the sum and sum of squares of q over the nodes).

Sources (copied unmodified from LHR, $T = /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/
2026-08-09-mt2-stage0, into scratch/isla-book-current-2026-10-22/book_D2_src/):

- $T/kernel/att2/geoflip/nodes.jsonl -> geoflip_nodes.jsonl (md5 0a543a07601449987c84362508ebb67f)
- the val_summary rows of $T/runs/r090eag_t2dz_isla_seed{42,43,44,45}/metrics.jsonl (md5 8494725f66591846a3bb7b9aae6b898c,
  01fde3c4d6f12a9534ca943714c38658, 6930accf4ae524969e1715ceee2f52bb, 55f5a97dc09c0be1101fc3765d523f81)
  -> r090eag_t2dz_isla_seed<s>_val_summary.jsonl
- /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/datasets/plaid_tensile2d_pdmsh/convert_results.json
  and manifest.json there -> plaid_tensile2d_convert_results.json (md5 9865e100b88fe99ea53cf581a049c712),
  plaid_tensile2d_manifest.json (md5 7abdf9af380590a2a4e778d80f07e12e)

usage: python book_D2_2026-10-22.py [source folder, default ~/gh/physicsnemo/scratch/isla-book-current-2026-10-22/book_D2_src]
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path.home()
    / "gh"
    / "physicsnemo"
    / "scratch"
    / "isla-book-current-2026-10-22"
    / "book_D2_src"
)


def lug_ties() -> dict:
    """The geometry probe's totals over its 60 lugs, checked against its own total row."""
    rows = [json.loads(ln) for ln in open(SRC / "geoflip_nodes.jsonl")]
    cases = [r for r in rows if r.get("phase") == "case"]
    total = next(r for r in rows if r.get("phase") == "total")
    n_nodes = sum(c["n_nodes"] for c in cases)
    n_surface = sum(c["n_surface"] for c in cases)
    bad = [c["nu_bad_interior"] for c in cases]
    assert (n_nodes, n_surface, sum(bad)) == (
        total["nodes"],
        total["surf"],
        total["int_bad"],
    )
    return {
        "source": "$T/kernel/att2/geoflip/nodes.jsonl (ATTRIB-090 job 334176)",
        "n_cases": len(cases),
        "n_nodes": n_nodes,
        "n_surface": n_surface,
        "n_interior": n_nodes - n_surface,
        "interior_direction_changed": sum(bad),
        "interior_direction_changed_per_case": [min(bad), max(bad)],
        "surface_direction_changed": sum(c["nu_bad_surface"] for c in cases),
        "distance_mismatched": sum(c["d_bad"] for c in cases),
        "surface_zero_in_flipped": sum(c["surface_zero_in_flipped"] for c in cases),
        "max_distance_deviation": max(c["max_dd"] for c in cases),
        "max_angle_interior_rad": max(c["max_angle_interior"] for c in cases),
        "max_angle_surface_rad": max(c["max_angle_surface"] for c in cases),
        "max_normal_flip_deviation": max(c["normals_flip_max"] for c in cases),
        "max_length_unit_rel_change": max(
            abs(c["L_f"] - c["L_n"]) / c["L_n"] for c in cases
        ),
    }


def t2d_validation() -> dict:
    """Per seed: the lowest in-training validation displacement error, the largest after it, and the last."""
    out = {}
    for s in (42, 43, 44, 45):
        rows = [
            json.loads(ln)
            for ln in open(SRC / f"r090eag_t2dz_isla_seed{s}_val_summary.jsonl")
        ]
        e = [(r["epoch"], r["displacement_l2"]) for r in rows]
        best = min(e, key=lambda t: t[1])
        after = max((t for t in e if t[0] > best[0]), key=lambda t: t[1])
        out[str(s)] = {
            "best_epoch": best[0],
            "best": best[1],
            "max_after_best_epoch": after[0],
            "max_after_best": after[1],
            "last_epoch": e[-1][0],
            "last": e[-1][1],
        }
    return {
        "source": "$T/runs/r090eag_t2dz_isla_seed{42,43,44,45}/metrics.jsonl, val_summary rows (in-training validation, "
        "displacement_l2)",
        "seeds": out,
    }


def t2d_never_yield() -> dict:
    """The val_100 specimens on which q is zero at every node."""
    val = json.load(open(SRC / "plaid_tensile2d_manifest.json"))["val_100"]
    conv = {
        r["case"]: r
        for r in json.load(open(SRC / "plaid_tensile2d_convert_results.json"))
    }
    zero = [c for c in val if conv[c]["scalars"]["max_q"] == 0]
    assert zero == [c for c in val if conv[c]["sums"]["q"][:2] == [0, 0]]
    return {
        "source": "plaid_tensile2d_pdmsh/convert_results.json (scalars.max_q, sums.q) over manifest.json val_100",
        "n_val": len(val),
        "n_q_zero": len(zero),
    }


if __name__ == "__main__":
    res = {
        "lug_closest_feature_ties": lug_ties(),
        "tensile2d_eager_validation": t2d_validation(),
        "tensile2d_never_yield": t2d_never_yield(),
    }
    (HERE / "book_D2.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
