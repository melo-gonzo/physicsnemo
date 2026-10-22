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

"""Numbers of the geometry-precision and discretization limits read by 03-isla, 04-invariances and 05-forward-pass that
no copied record holds as a table (2026-10-22).

Sources:
  theta      ISLA-DTT's tree-read opening angle against exact refinement of the AirfRANS airfoil, ISLA-DTT 0.9.0 weights
             trained at theta = 1 under the compiled-path fault (r090_af_dtt_seed42/43), theta patched at inference, eager
             float32, 200 test cases (PROBES-090, post hoc): scratch/isla-probes090-2026-10-21/results/remesh_theta/summary.md
  stl_offset the distance between the watertight STL and the CFD wall of DrivAerML Volume, 48 validation cars, in the
             length unit L (PROBES-090 diagnostic): scratch/isla-probes090-2026-10-21/diag_stl/STL_OFFSET.md
  lug_ties   closest-feature ties of the exact geometry on the GINOT lug under a winding flip, 60 test cases, geometry only
             (ATTRIB-090 section 4): results/campaigns_2026-10-22/attrib090/RESULTS.md

Writes book_a.json beside this script:
  theta[theta][variant]   {"pressure", "velocity", "nut"}: seed-mean error ratio refined / native; "native_s42": the
                          native-mesh error of seed 42 at that theta, {"pressure", "velocity", "nut"}
  stl_offset              {"median_L": median over cases of the per-case median CFD-wall-to-STL distance, "max_L": largest
                          sampled distance in either direction, "frac_above_1e-2_L": the area fraction farther than 0.01 L}
  lug_ties                {"nodes", "interior_fraction", "all_fraction", "per_case_min", "per_case_max", "n_cases"}
usage: python book_a_2026-10-22.py [scratch root, default ~/gh/physicsnemo/scratch]"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = (
    sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/gh/physicsnemo/scratch")
)
PROBES = os.path.join(SCRATCH, "isla-probes090-2026-10-21")
THETA = os.path.join(PROBES, "results", "remesh_theta", "summary.md")
STL = os.path.join(PROBES, "diag_stl", "STL_OFFSET.md")
ATTRIB = os.path.join(HERE, "..", "attrib090", "RESULTS.md")


def rows(path, first_cell):
    """The cells of every markdown table row of a file whose first cell starts with ``first_cell``."""
    out = []
    for ln in open(path).read().splitlines():
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if ln.lstrip().startswith("|") and cells[0].startswith(first_cell):
            out.append(cells)
    return out


def first(cell):
    """The first number in a cell."""
    return float(re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cell)[0])


theta = {}
for cells in rows(THETA, "")[2:]:
    t = str(first(cells[0]))
    native = [float(x) for x in cells[6].split("/")]
    theta.setdefault(
        t, {"native_s42": dict(zip(("pressure", "velocity", "nut"), native))}
    )
    theta[t][cells[1]] = {
        "pressure": first(cells[2]),
        "velocity": first(cells[3]),
        "nut": first(cells[4]),
    }
assert set(theta) == {"1.0", "0.5", "0.25", "0.0"}, theta.keys()

mid = lambda cell: float(cell.split("/")[1])  # "min / median / max over cases"
hi = lambda cell: float(cell.split("/")[2])
median = rows(STL, "median")[0]
max_samples = rows(STL, "max (samples)")[0]
frac = rows(STL, "fraction > 1e-2 L")[0]
stl_offset = {
    "median_L": mid(median[1]),
    "max_L": max(hi(max_samples[1]), hi(max_samples[2])),
    "frac_above_1e-2_L": max(hi(frac[1]), hi(frac[2])),
}

text = re.sub(r"\s+", " ", open(ATTRIB).read())
m = re.search(
    r"\*\*([\d,]+) interior nodes \(([\d.]+) % of the interior, ([\d.]+) % of all nodes; (\d+)-(\d+) per case\)\*\*",
    text,
)
n = re.search(r"geometry only, (\d+) cases, ([\d,]+) nodes", text)
lug_ties = {
    "nodes": int(m.group(1).replace(",", "")),
    "interior_fraction": float(m.group(2)) / 100,
    "all_fraction": float(m.group(3)) / 100,
    "per_case_min": int(m.group(4)),
    "per_case_max": int(m.group(5)),
    "n_cases": int(n.group(1)),
}

out = {
    "sources": {
        "theta": "scratch/" + os.path.relpath(THETA, SCRATCH),
        "stl_offset": "scratch/" + os.path.relpath(STL, SCRATCH),
        "lug_ties": "examples/cfd/isla/results/campaigns_2026-10-22/attrib090/RESULTS.md",
    },
    "theta": theta,
    "stl_offset": stl_offset,
    "lug_ties": lug_ties,
}
json.dump(out, open(os.path.join(HERE, "book_a.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
