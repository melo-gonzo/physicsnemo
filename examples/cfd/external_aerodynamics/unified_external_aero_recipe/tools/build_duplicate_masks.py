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

"""Build, adopt or check the cached duplicate-cell masks of ``drop_duplicate_cells`` (see ``src/dedup.py``).

usage:
  python tools/build_duplicate_masks.py build <root> <out_dir> <case> ...    # write <case>.npy and <case>.json
  python tools/build_duplicate_masks.py adopt <root> <out_dir> <case> ...    # write <case>.json for an existing <case>.npy
  python tools/build_duplicate_masks.py check <root> <out_dir> <case> ...    # stored mask == a fresh exact computation
  python tools/build_duplicate_masks.py manifest <out_dir>                   # MANIFEST.tsv of every <case>.json

``<root>/<case>/*.pdmsh/_tensordict/boundaries/<boundary>`` is the mesh (``--boundary``, default ``vehicle``). ``adopt``
verifies the existing mask's length and records the fingerprint and the connectivity md5; ``check`` recomputes the mask
with ``duplicate_cells_reference`` (an independent exact lexsort, about 35 GB for a HiLift wing) and the connectivity md5,
and fails on any difference. One JSON line per case.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dedup import (  # noqa: E402
    CHUNK,
    connectivity_fingerprint,
    duplicate_cells_reference,
    write_mask,
)


def load_cells(root: Path, case: str, boundary: str):
    """The case's boundary connectivity as a read-only memmap, and its directory."""
    (pd,) = sorted((root / case).glob("*.pdmsh"))
    td = pd / "_tensordict" / "boundaries" / boundary / "_tensordict"
    meta = json.loads((td / "meta.json").read_text())["cells"]
    if meta["dtype"] != "torch.int64":
        raise ValueError(f"{case}: cells are {meta['dtype']}, expected torch.int64")
    return np.memmap(
        td / "cells.memmap", dtype=np.int64, mode="r", shape=tuple(meta["shape"])
    ), str(td)


def connectivity_md5(cells) -> str:
    """The md5 of the connectivity's int64 bytes, read in chunks."""
    md5 = hashlib.md5()
    for s in range(0, len(cells), CHUNK):
        md5.update(np.ascontiguousarray(cells[s : s + CHUNK], dtype="<i8").tobytes())
    return md5.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["build", "adopt", "check", "manifest"])
ap.add_argument("args", nargs="+")
ap.add_argument("--boundary", default="vehicle")
a = ap.parse_args()
if a.mode == "manifest":
    out = Path(a.args[0])
    rows = [json.loads(p.read_text()) for p in sorted(out.glob("*.json"))]
    cols = [
        "case",
        "n_cells",
        "n_extra",
        "fingerprint",
        "connectivity_md5",
        "mask_md5",
        "builder",
        "source",
    ]
    (out / "MANIFEST.tsv").write_text(
        "\t".join(cols)
        + "\n"
        + "".join("\t".join(str(r[c]) for c in cols) + "\n" for r in rows)
    )
    print(f"{len(rows)} masks")
    sys.exit(0)
root, out, cases = Path(a.args[0]), Path(a.args[1]), a.args[2:]
failed = 0
for case in cases:
    t0 = time.time()
    cells, src = load_cells(root, case, a.boundary)
    if a.mode == "build":
        rec = write_mask(out, case, cells, src)
    elif a.mode == "adopt":
        packed = np.load(out / f"{case}.npy")
        if len(packed) != (len(cells) + 7) // 8:
            raise ValueError(
                f"{case}: mask has {len(packed)} bytes for {len(cells)} cells"
            )
        rec = {
            "case": case,
            "source": src,
            "n_cells": int(len(cells)),
            "n_extra": int(np.unpackbits(packed, count=len(cells)).sum()),
            "fingerprint": connectivity_fingerprint(cells),
            "connectivity_md5": connectivity_md5(cells),
            "mask_md5": hashlib.md5((out / f"{case}.npy").read_bytes()).hexdigest(),
            "builder": "HL-DUP stage-1 census (scratch/isla-hl-dup-2026-10-24/lhr/hl_census.py, the same exact rule)",
        }
        (out / f"{case}.json").write_text(json.dumps(rec, indent=1) + "\n")
    else:
        meta = json.loads((out / f"{case}.json").read_text())
        stored = np.unpackbits(np.load(out / f"{case}.npy"), count=len(cells)).astype(
            bool
        )
        fresh = duplicate_cells_reference(cells)
        rec = {
            "case": case,
            "mask_equal": bool(np.array_equal(stored, fresh)),
            "n_extra_fresh": int(fresh.sum()),
            "n_extra_stored": int(stored.sum()),
            "fingerprint_equal": meta["fingerprint"] == connectivity_fingerprint(cells),
            "connectivity_md5_equal": meta["connectivity_md5"]
            == connectivity_md5(cells),
            "mask_md5_equal": meta["mask_md5"]
            == hashlib.md5((out / f"{case}.npy").read_bytes()).hexdigest(),
        }
        failed += not all(v for k, v in rec.items() if k.endswith("_equal"))
    rec["s"] = round(time.time() - t0, 1)
    print(json.dumps(rec), flush=True)
sys.exit(1 if failed else 0)
