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

"""The book's one reader of the study records under ``results/campaigns_<date>/<topic>/``.

Those files are unmodified copies of the lanes' records (index and md5s: ``RECORDS.md`` beside them, written by
``copy_records_<date>.py``). A topic is read from the newest folder that holds it (``FOLDER``). A chapter reads a number from them through this module, so a figure or a sentence
updates when the record does::

    import records as R
    afx = R.load("af_extrap", "afx_results.json")
    rows = R.md_table("az_diag", "f32floor/README.md", "Sign flips")  # the first table under that heading

Captions name the file with ``R.source(topic, name)``.
"""

import csv
import json
import re
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

#: topics copied again into a newer folder -> that folder; every other topic is in campaigns_2026-10-22
FOLDER = {
    "attrib090": "campaigns_2026-10-23",
    "ref090": "campaigns_2026-10-23",
    "af_extrap": "campaigns_2026-10-23",
    "suv_xfam_b": "campaigns_2026-10-23",
    "probes090": "campaigns_2026-10-23",
    "cap_audit": "campaigns_2026-10-23",
    "floor_drv": "campaigns_2026-10-23",
    "az_compliant": "campaigns_2026-10-23",
    "boundary_audit": "campaigns_2026-10-23",
    "drv_ground": "campaigns_2026-10-23",
    "setup_research": "campaigns_2026-10-23",
    "af_whole": "campaigns_2026-10-23",
    "surface_bc": "campaigns_2026-10-23",
    "surfqpop": "campaigns_2026-10-23",
    "slices_af": "campaigns_2026-10-23",
    "tie_exp": "campaigns_2026-10-23",
    "fold_sign": "campaigns_2026-10-23",
    "token_draw": "campaigns_2026-10-23",
    "bundle_merge": "campaigns_2026-10-23",
    "geo_fast": "campaigns_2026-10-23",
    "align8": "campaigns_2026-10-23",
    "bc_sidecar": "campaigns_2026-10-23",
    "geo_degen": "campaigns_2026-10-23",
    "fast_merge": "campaigns_2026-10-23",
    "dtt_knobs": "campaigns_2026-10-23",
    "dtt_fast_compile": "campaigns_2026-10-23",
    "dtt_fast_verify_fusedread": "campaigns_2026-10-23",
    "dtt_fast_integration": "campaigns_2026-10-23",
    "boundary_merge": "campaigns_2026-10-23",
    "dtt_pose": "campaigns_2026-10-23",
    "dtt_pose_exp": "campaigns_2026-10-23",
    "geo_thin": "campaigns_2026-10-23",
    "hl_dup": "campaigns_2026-10-23",
    "dtt_fast_final": "campaigns_2026-10-23",
    "nearwall": "campaigns_2026-10-23",
    "geo_thin_final": "campaigns_2026-10-23",
    "nearwall_final": "campaigns_2026-10-23",
    "dtt_fast_bf16diag": "campaigns_2026-10-23",
    "az_decide_final": "campaigns_2026-10-23",
    "e1_az": "campaigns_2026-10-23",
    "lre_az": "campaigns_2026-10-23",
    "geo_band_phase0": "campaigns_2026-10-23",
    "kernel_tests": "campaigns_2026-10-23",
    "slices_drv": "campaigns_2026-10-23",
    "freeze_plan": "campaigns_2026-10-23",
    "bc_supply": "campaigns_2026-10-23",
    "geo_band_final": "campaigns_2026-10-23",
    "band_merge": "campaigns_2026-10-23",
    "default_dtt": "campaigns_2026-10-23",
    "dtt_default_int": "campaigns_2026-10-23",
    "geo_thinfast": "campaigns_2026-10-23",
    "dtt_fast_done": "campaigns_2026-10-23",
    "dtt_fast_final_table_p5": "campaigns_2026-10-23",
}


def path(topic: str, name: str) -> Path:
    """The record's file; raises if it is missing."""
    p = RESULTS / FOLDER.get(topic, "campaigns_2026-10-22") / topic / name
    if not p.exists():
        raise FileNotFoundError(f"no study record at {p}")
    return p


def source(topic: str, name: str) -> str:
    """The record's path as captions print it."""
    return f"results/{FOLDER.get(topic, 'campaigns_2026-10-22')}/{topic}/{name}"


def load(topic: str, name: str):
    """A JSON record parsed; a TSV or CSV record as a list of dicts (one per row, keyed by the header)."""
    p = path(topic, name)
    if p.suffix == ".json":
        return json.loads(p.read_text())
    if p.suffix in (".tsv", ".csv"):
        with open(p, newline="") as f:
            return list(
                csv.DictReader(f, delimiter="\t" if p.suffix == ".tsv" else ",")
            )
    return p.read_text()


def md_table(
    topic: str, name: str, heading: str | None = None, index: int = 0
) -> list[dict]:
    """The ``index``-th markdown table of a record (after the first heading containing ``heading``, if given), as a list
    of dicts keyed by its header cells. Cell text is kept as written (bold marks stripped); convert with ``num``."""
    lines = path(topic, name).read_text().splitlines()
    if heading is not None:
        start = next(
            i
            for i, ln in enumerate(lines)
            if ln.lstrip().startswith("#") and heading in ln
        )
        lines = lines[start + 1 :]
    tables, cur = [], []
    for ln in lines + [""]:
        if ln.strip().startswith("|"):
            cur.append(ln)
        elif cur:
            tables.append(cur)
            cur = []
    rows = [
        [c.strip().replace("**", "") for c in ln.strip().strip("|").split("|")]
        for ln in tables[index]
    ]
    head, body = rows[0], [r for r in rows[2:]]
    return [dict(zip(head, r)) for r in body]


def num(cell: str, k: int = 0) -> float:
    """The ``k``-th number in a table cell: ``num('0.0389 (0.0391, 0.0387)', 1) == 0.0391``."""
    return float(
        re.findall(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?", cell)[k].replace(",", "")
    )
