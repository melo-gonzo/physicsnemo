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

"""Copy the study records the book's chapters read into results/campaigns_2026-10-23/<topic>/, unmodified, and write
RECORDS.md: one row per file with its source path under the program's scratch folder and its md5.

The records are the lanes' own outputs (reductions, tables, READMEs); nothing here recomputes a number. A chapter reads
them through ``book/records.py``. Rerun after a lane updates a record: the copy and the index are rewritten together.

usage: python copy_records_2026-10-23.py [scratch root, default ~/gh/physicsnemo/scratch]
"""

import hashlib
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path.home() / "gh" / "physicsnemo" / "scratch"
)

#: topic -> (scratch folder, what it measures, files relative to that folder)
RECORDS = {
    "attrib090": (
        "isla-attrib090-2026-10-21",
        "ATTRIB-090: which change of ISLA 0.9.0 moves Tensile2d, AirfRANS and the lug",
        [
            "RESULTS.md",
            "results_native.json",
            "results2_native.json",
            "results2_posed.json",
            "results_native_eager.json",
            "results_posed_eager.json",
            "relaunch_native.md",
            "relaunch_posed.md",
            "arch_lug.md",
            "arch_tensile2d.md",
            "arch_airfrans.md",
            "t2d_eager_native.md",
            "t2d_eager_posed.md",
        ],
    ),
    "ref090": (
        "isla-ref090-2026-10-20",
        "REF-090: ISLA 0.9.0 against the nearest-token form, the training-path control",
        [
            "r090_final_tables.md",
            "r090_final.json",
            "t2d_control.md",
            "t2d_control.json",
            "t2d_fourseed.json",
            "t2d_fourseed.md",
            "PREREG.md",
            "RESULTS.md",
            "eag_lug.md",
            "eag_lug.json",
            "eag_af_isla.md",
            "eag_af_isla.json",
        ],
    ),
    "af_extrap": (
        "isla-af-extrap-2026-10-21",
        "AF-EXTRAP: AirfRANS extrapolation in Reynolds number and angle of attack",
        [
            "RESULTS.md",
            "POSTHOC.md",
            "afx_tables.md",
            "afx_results.json",
            "posthoc.json",
            "clamp.json",
            "stats/afx_fits.json",
            "stats/clamp_bounds.json",
            "LRE.md",
            "lre.json",
        ],
    ),
    "suv_xfam_b": (
        "isla-suv-xfam-2026-10-22",
        "SUV-XFAM part B: SHIFT-SUV models trained on fastbacks, against the estate-trained ones (stages 1 and 2)",
        [
            "PREREG_B.md",
            "RESULTS_B.md",
            "JOBS.md",
            "reduce_b_out.json",
            "identity_b_out.json",
            "reduce_b2_out.json",
        ],
    ),
    "probes090": (
        "isla-probes090-2026-10-21",
        "PROBES-090: query-population dependence and re-meshing invariance of ISLA 0.9.0",
        [
            "RESULTS.md",
            "results/qpop/reduced.json",
            "results/qpop_afix/reduced.json",
            "results/qpop_afeag/reduced.json",
            "results/qpop_az/reduced.json",
            "results/remesh/reduced_full.json",
            "results/remesh_afix/reduced_full.json",
            "results/remesh_afix_dtt/reduced_full.json",
            "results/remesh_afeag/reduced_full.json",
            "results/afeag_book_numbers.md",
            "remesh/facts_dv.json",
            "remesh/README.md",
            "qpop/qpop_rows.tsv",
        ],
    ),
    "cap_audit": (
        "isla-cap-audit-2026-10-20",
        "CAP-AUDIT: capacity floors of the three architectures, phases 1, 1b and 2",
        [
            "RESULTS.md",
            "HANDOFF.md",
            "tables/*.md",
            "tables/*.json",
            "tables/*.csv",
            "tables/probes/*.json",
        ],
    ),
    "floor_drv": (
        "isla-floor-drv-2026-10-23",
        "FLOOR-DRV: ISLA on one DrivAerML Volume car with the domain's four boundary meshes, the pre-registration",
        ["PREREG.md"],
    ),
    "az_compliant": (
        "isla-az-compliant-2026-10-19",
        "AeroZoo under the benchmark protocol: the registered rule for ISLA against ISLA-DTT (PREREG_v2 sections 12(f) and 13)",
        ["PREREG_v2.md"],
    ),
    "boundary_audit": (
        "isla-boundary-audit-2026-10-23",
        "BOUNDARY-AUDIT: what each dataset stores of its domain's boundary, and the whole boundary's cost on DrivAerML Volume",
        [
            "REPORT.md",
            "BASELINE_PARITY.md",
            "cost_table_isla.md",
            "cost_table_dtt.md",
            "cost_seed_table.md",
        ],
    ),
    "drv_ground": (
        "isla-drv-ground-2026-10-23",
        "DRV-GROUND: the DrivAerML domain meshes' ground offset on all 484 files, and the recipe transform that places it",
        [
            "classify_summary.txt",
            "check_summary.txt",
            "offset_runs.txt",
        ],
    ),
    "setup_research": (
        "isla-setup-research-2026-10-23",
        "SETUP-RESEARCH: the CFD domains of DrivAerML, SHIFT-SUV and HiLiftAeroML, and where their boundary pieces come from",
        ["SETUP.md"],
    ),
    "af_whole": (
        "isla-af-whole-2026-10-23",
        "AF-WHOLE: ISLA and ISLA-DTT reading the AirfRANS far field, the pre-registration and its facts before training",
        ["PREREG.md", "FACTS.md"],
    ),
    "surface_bc": (
        "isla-surface-bc-2026-10-23",
        "SURFACE-BC: a surface task that reads the whole boundary and predicts at the boundaries it names, built and timed",
        ["REPORT.md", "DESIGN.md", "cost_table.md"],
    ),
    "surfqpop": (
        "isla-surf-probe-2026-10-23",
        "SURF-QPOP: how a DrivAerML Surface prediction depends on the other requested cells",
        [
            "RESULTS.md",
            "PREREG.md",
            "results/out/reduced.json",
            "results/out/tables.md",
        ],
    ),
    "slices_af": (
        "isla-slices-2026-10-23",
        "SLICES-AF: ISLA with 128 slices against the 512-slice AirfRANS standing",
        [
            "RESULTS.md",
            "PREREG.md",
            "reading.md",
            "reading.json",
            "nearwall/nearwall_table.md",
        ],
    ),
}


def md5(p: Path) -> str:
    """The file's md5, as RECORDS.md lists it."""
    return hashlib.md5(p.read_bytes()).hexdigest()


topics = ["| topic | what it measures | source folder |", "| --- | --- | --- |"] + [
    f"| `{topic}/` | {what} | `scratch/{folder}/` |"
    for topic, (folder, what, _) in RECORDS.items()
]
rows = ["| topic | file | source | md5 |", "| --- | --- | --- | --- |"]
for topic, (folder, what, files) in RECORDS.items():
    src = SCRATCH / folder
    for pattern in files:
        for s in sorted(src.glob(pattern)):
            rel = s.relative_to(src)
            d = HERE / topic / rel
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
            rows.append(f"| {topic} | {rel} | scratch/{folder}/{rel} | {md5(d)} |")
head = [
    "# Study records read by the book",
    "",
    f"Written by `{Path(__file__).name}`; every file is an unmodified copy of its",
    "source. Do not edit by hand.",
    "",
]
(HERE / "RECORDS.md").write_text("\n".join(head + topics + [""] + rows) + "\n")
print(f"{len(rows) - 2} files indexed in RECORDS.md")
