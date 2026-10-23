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
