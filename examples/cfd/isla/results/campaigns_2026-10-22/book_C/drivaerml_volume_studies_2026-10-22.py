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

"""Two study reductions for the DrivAerML Volume chapter (09), written to drivaerml_volume_studies.json beside this script.

1. ``stl_offset``: how far the watertight STL of each car lies from its CFD vehicle wall (PROBES-090's STL-OFFSET
   diagnostic, 48 validation cars, 4 M area samples per surface, exact point-to-triangle distances, in the length unit L),
   and the share of interior nodes that sit on the CFD wall. Sources: scratch/isla-probes090-2026-10-21/diag_stl/data/
   stl_offset_rows.jsonl and interior_context_rows.jsonl (jobs 312063, 312116; STL_OFFSET.md there).
2. ``dtt_instrument``: ISLA against ISLA-DTT on the Volume task at the study configuration of research/dtt/REPORT.md
   (512 wide, 512 slices, 3.75e-4, raw slice reads): per-seed native errors of the interacting reference (I), the read-only
   points without a boundary read (P), the exact read of 256 and 1,024 cells, and ISLA-DTT (the forest in the boundary's own
   frame, theta 1), and ISLA-DTT's request probe. Parsed from the tables and sentences of research/dtt/REPORT.md; the seed
   means are the tables' own seed-mean column, and the mean of the two seeds for the 256- and 1,024-cell reads.

usage: python drivaerml_volume_studies_2026-10-22.py [scratch root, default ~/gh/physicsnemo/scratch]
"""

import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ISLA = HERE.parents[2]
SCRATCH = (
    Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "gh/physicsnemo/scratch"
)
STL = SCRATCH / "isla-probes090-2026-10-21/diag_stl/data"
REPORT = ISLA / "research/dtt/REPORT.md"
FIELDS = ("pressure_l2", "velocity_l2", "nut_l2")


def rows(p):
    """The rows of a JSON-lines file."""
    return [json.loads(ln) for ln in open(p) if ln.strip()]


def triple(s):
    """'0.0454 / 0.0847 / 0.1745' -> {field: value}."""
    v = [float(x) for x in re.findall(r"\d+\.\d+", s)]
    assert len(v) == 3, s
    return dict(zip(FIELDS, v))


def stl_offset():
    """The STL-to-CFD-wall separation over the 48 cars and the share of interior nodes on the CFD wall."""
    so = rows(STL / "stl_offset_rows.jsonl")
    ic = rows(STL / "interior_context_rows.jsonl")
    assert len(so) == 48 and len(ic) == 48
    med = [r["cfd_to_stl"]["q0.5_L"] for r in so]
    return {
        "n_cases": len(so),
        "median_offset_L": statistics.median(med),
        "median_offset_L_range": [min(med), max(med)],
        "max_cfd_to_stl_L": max(r["cfd_to_stl"]["max_L"] for r in so),
        "max_stl_to_cfd_L": max(r["stl_to_cfd"]["max_L"] for r in so),
        "max_frac_gt_0.01L": max(
            max(r["cfd_to_stl"]["frac_gt_0.01L"], r["stl_to_cfd"]["frac_gt_0.01L"])
            for r in so
        ),
        "median_frac_cfd_inside_stl": statistics.median(
            r["cfd_to_stl_signed"]["frac_inside_stl"] for r in so
        ),
        "n_cells_stl_range": [
            min(r["n_cells_stl"] for r in so),
            max(r["n_cells_stl"] for r in so),
        ],
        "n_cells_cfd_range": [
            min(r["n_cells_cfd"] for r in so),
            max(r["n_cells_cfd"] for r in so),
        ],
        "median_frac_nodes_on_wall": statistics.median(
            r["band_lt_1e-06L"]["frac_points"] for r in ic
        ),
        "source": "scratch/isla-probes090-2026-10-21/diag_stl/data/{stl_offset_rows,interior_context_rows}.jsonl",
    }


def dtt_instrument():
    """Per-seed native errors of the ISLA-DTT study arms and ISLA-DTT's request probe, from research/dtt/REPORT.md."""
    rep = REPORT.read_text()

    def table_row(after, label):
        sec = rep[rep.index(after) :]
        ln = next(x for x in sec.splitlines() if x.startswith(f"| {label}"))
        return [c.strip() for c in ln.strip().strip("|").split("|")]

    def seeds(cells):
        return {
            "42": triple(cells[1]),
            "43": triple(cells[2]),
            "mean": triple(cells[3]),
        }

    arms = {
        "I": seeds(
            table_row("### E6 interim on the car", "I, 512 slices (AGA reference)")
        ),
        "P": seeds(table_row("### E6 interim on the car", "P, 512 slices (E3)")),
        "DTT": seeds(
            table_row(
                "### Pose-free parity on the car", "forest, own frame (`code_dtt16`)"
            )
        ),
    }
    for key, head in (
        ("K256", "First rung on the car"),
        ("K1024", "Last rung, the car at 1,024 cells"),
    ):
        sec = " ".join(rep[rep.index(head) :].split())
        m = re.search(r"\(seeds ([0-9. /]+) and ([0-9. /]+)\)", sec)
        arms[key] = {"42": triple(m.group(1)), "43": triple(m.group(2))}
    for a in arms.values():
        a.setdefault("mean", {f: (a["42"][f] + a["43"][f]) / 2 for f in FIELDS})
    pf = " ".join(
        rep[rep.index("### Pose-free parity on the car") :].split("\n### ")[0].split()
    )
    probe = {
        "ratio": float(
            re.search(r"error ratios ([0-9.]+) on every field", pf).group(1)
        ),
        "largest_departure": float(
            re.search(r"largest departure ([0-9.e-]+)\)", pf).group(1)
        ),
        "largest_change": float(
            re.search(r"largest prediction change on the shared points ([0-9.e-]+)", pf)
            .group(1)
            .rstrip(".")
        ),
    }
    return {
        "arms": arms,
        "request_probe": probe,
        "source": "research/dtt/REPORT.md: 'E6 interim on the car' (I, P), 'Pose-free parity on the car' (ISLA-DTT and its request "
        "probe), 'First rung on the car' and 'Last rung, the car at 1,024 cells' (the 256- and 1,024-cell reads)",
    }


if __name__ == "__main__":
    out = {"stl_offset": stl_offset(), "dtt_instrument": dtt_instrument()}
    (HERE / "drivaerml_volume_studies.json").write_text(
        json.dumps(out, indent=1) + "\n"
    )
    print(json.dumps(out, indent=1))
