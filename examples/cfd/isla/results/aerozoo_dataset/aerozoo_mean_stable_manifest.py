#!/usr/bin/env python
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

"""The split parts of AeroZoo's canonical benchmark: every part's mean_stable samples (2026-10-26).

AeroZoo's canonical benchmark (docs/book/06-splits.qmd @sec-canonical, aerozoo ba5b38e8) trains, validates and tests
on the ``mean_stable`` samples of every split part only. This writes, from the row-shuffled copy's ``manifest.json``
(``<split>.<part>`` -> complete case ids), the same keys with each list restricted to the cases whose stored label
``global_data/diagnostics/solve/solution_quality_id`` is 4 (``mean_stable``; 5 is ``mean_unstable``, aerozoo
docs/book/B-reference.qmd). The dataset files select the same samples at load (their ``sample_filter``); this file
is the input of the constants fits (``aerozoo_pstats.py reduce --manifest``, ``isla_constants_fit.py aerozoo_fit``
and ``aerozoo_rms_fit``). With ``--release``, every case's stored label must equal the release manifest's
``solution_quality`` (``release-v1.2.json``, the spec's source), and every case must be ``complete``::

    python aerozoo_mean_stable_manifest.py <shuffled root> manifest_mean_stable.json --release release-v1.2.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from aerozoo_pstats import TD, _leaf

LABEL = ("diagnostics", "solve", "solution_quality_id")
LABELS = {4: "mean_stable", 5: "mean_unstable"}
KEEP = 4


def case_label(case_dir: Path) -> int:
    """A case's stored stability label id, read from its global_data."""
    group = case_dir / "domain.pdmsh" / TD / "global_data" / Path(*LABEL[:-1])
    return int(_leaf(group, LABEL[-1]))


def main(argv=None) -> None:
    """Write the mean_stable manifest and check it against the release manifest."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root")
    parser.add_argument("out")
    parser.add_argument("--release", default=None)
    args = parser.parse_args(argv)
    root = Path(args.root)
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    parts = {k: v for k, v in manifest.items() if isinstance(v, list)}
    cases = sorted(set().union(*parts.values()))
    dirs = {p.name: p for p in root.glob("shard-*/case-*")}
    labels = {c: case_label(dirs[c]) for c in cases}
    unknown = {c: v for c, v in labels.items() if v not in LABELS}
    if unknown:
        raise SystemExit(f"labels outside {LABELS}: {list(unknown.items())[:5]}")
    if args.release:
        release = json.loads(Path(args.release).read_text())["cases"]
        bad = [
            c
            for c in cases
            if release[c]["status"] != "complete"
            or release[c]["solution_quality"] != LABELS[labels[c]]
        ]
        if bad:
            raise SystemExit(
                f"{len(bad)} cases disagree with {args.release}: {bad[:5]}"
            )
    out = {k: [c for c in v if labels[c] == KEEP] for k, v in sorted(parts.items())}
    out["_meta"] = {
        "label": "/".join(LABEL),
        "keep": {KEEP: LABELS[KEEP]},
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "n_cases": len(cases),
        "n_kept": sum(labels[c] == KEEP for c in cases),
        "release_checked": args.release,
        "sizes": {k: [len(out[k]), len(v)] for k, v in sorted(parts.items())},
    }
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out["_meta"][k] for k in ("n_cases", "n_kept")}))


if __name__ == "__main__":
    main()
