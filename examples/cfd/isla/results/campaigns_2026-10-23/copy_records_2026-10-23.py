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
    "tie_exp": (
        "isla-tie-2026-10-22",
        "TIE-EXP: ISLA with the least-norm tie rules trained on the lug and Tensile2d, read on flip symmetry, native accuracy and pose",
        [
            "RESULTS.md",
            "PREREG.md",
            "JOBS.md",
            "read/results.md",
            "read/results.json",
        ],
    ),
    "fold_sign": (
        "isla-fold-sign-2026-10-22",
        "FOLD-SIGN: the wall distance signed by the generalized winding number, the registered f32floor rerun on AeroZoo",
        ["PREREG.md", "f32floor_tables.md"],
    ),
    "token_draw": (
        "isla-token-draw-2026-10-22",
        "TOKEN-DRAW: the boundary-token draw in float64 in the boundary's canonical frame, measured under precision, pose and renumbering",
        ["DESIGN.md"],
    ),
    "bundle_merge": (
        "isla-bundle-merge-2026-10-24",
        "BUNDLE-MERGE: the geometry robustness bundle merged as ISLA 0.11.0, its version change and tests",
        ["REPORT.md", "FINAL_REPORT.md"],
    ),
    "geo_fast": (
        "isla-geo-fast-2026-10-23",
        "GEO-FAST: the exact geometry's fast path, its bitwise checks and its GB300 timing, and the zero-area-cell defect",
        ["RESULTS.md", "FINAL_REPORT.md", "DESIGN.md", "lhr/final2_table.md"],
    ),
    "align8": (
        "isla-align8-2026-10-23",
        "ALIGN8: the step cost of token counts that are not a multiple of 8 or 32, and the padding fix",
        [
            "FINAL_REPORT.md",
            "results/repro_table.md",
            "exposure/exposure_gpuh.md",
            "exposure/exposure_classes.tsv",
        ],
    ),
    "bc_sidecar": (
        "isla-bc-sidecar-2026-10-23",
        "BC-SIDECAR: the SHIFT-SUV box and the HiLiftAeroML far field and symmetry plane, built and checked",
        ["REPORT.md", "FINAL_REPORT.md"],
    ),
    "geo_degen": (
        "isla-geo-degen-2026-10-24",
        "GEO-DEGEN: cells of zero area read as the segment or point their vertices span, its checks and cost, and the limits left",
        ["FINAL_REPORT.md", "RESULTS.md", "DESIGN.md"],
    ),
    "fast_merge": (
        "isla-fast-merge-2026-10-24",
        "FAST-MERGE: the exact geometry's fast path merged with the zero-area fix, its bitwise checks and its GB300 timing",
        ["REPORT.md", "FINAL_REPORT.md", "lhr/timing_table.md", "lhr/book_table.md"],
    ),
    "dtt_knobs": (
        "isla-dtt-knobs-2026-10-23",
        "DTT-KNOBS: ISLA-DTT's boundary read, its knobs and its accuracy gap (stage 1, inference only) and the stage-2 registration",
        [
            "STAGE1.md",
            "STAGE1_REVIEW.md",
            "PREREG.md",
            "PREREG_REVIEW.md",
            "CODE_FACTS.md",
            "NOTEBOOK_DRAFT_STAGE1.md",
            "NOTEBOOK_REVIEW.md",
            "NOTEBOOK_REVISION_NOTE.md",
            "sens/REPORT.md",
            "sens/sens_table.md",
            "sens/sens_regions.md",
            "cost/REPORT.md",
            "cost/cost_table.md",
            "cap/REPORT.md",
            "cap/cap_table.md",
            "order/REPORT.md",
            "order/order_table.md",
            "gap/REPORT.md",
            "gap/gap_table.md",
            "gap/gap_table_bf16.md",
            "gap/rough.md",
            "treeprec/REPORT.md",
            "treeprec/treeprec_table.md",
            "treeprec/signtest.md",
        ],
    ),
    "dtt_fast_compile": (
        "isla-dtt-fast-2026-10-23/compile",
        "DTT-FAST, piece COMPILE: ISLA-DTT's compiled blocks certified against eager, and the bf16 casts Inductor leaves out",
        ["RESULTS.md", "REPORT.md"],
    ),
    "dtt_fast_verify_fusedread": (
        "isla-dtt-fast-2026-10-23/verify_fusedread",
        "DTT-FAST, piece FUSEDREAD verified independently: pair lists, precision and the GB300 step",
        ["VERIFY.md"],
    ),
    "dtt_fast_integration": (
        "isla-dtt-fast-2026-10-23/integration",
        "DTT-FAST, integration pass 1: the tree walk, tree build and fused read on ISLA 0.11.0 behind one execution attribute",
        ["RESULTS.md"],
    ),
    "boundary_merge": (
        "isla-merge-prep-2026-10-24",
        "MERGE-PREP: the token padding and the whole boundary with the surface task merged into ISLA 0.11.0, checked and timed",
        [
            "REPORT.md",
            "FINAL_REPORT.md",
            "book/BOOK_CHANGES.md",
            "results/TIMING.md",
            "results/book_diff_7213_vs_17c9df5c0.txt",
        ],
    ),
    "dtt_pose": (
        "isla-dtt-pose-2026-10-24",
        "DTT-POSE: why ISLA-DTT's forest depends on pose and precision, and the design that removes it",
        ["DESIGN.md", "MECHANISMS.md"],
    ),
    "dtt_pose_exp": (
        "isla-dtt-pose-exp-2026-10-24",
        "DTT-POSE-EXP: ISLA-DTT retrained with DTT-POSE's forest on the lug and AirfRANS, the pre-registration",
        ["PREREG.md"],
    ),
    "geo_thin": (
        "isla-geo-thin-2026-10-24",
        "GEO-THIN: thin cells of nonzero area in the exact geometry, the census of the stored meshes and the design",
        ["DESIGN.md", "RESULTS.md"],
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
