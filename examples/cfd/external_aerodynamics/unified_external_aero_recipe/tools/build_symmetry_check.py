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

"""Build the record of HiLift's symmetry-plane check (``check_root`` of ``src/symmetry.py``), one per dataset root.

usage:
  python tools/build_symmetry_check.py --root R --cap-md5 M [M ...] --cases all --shard 0/8 --part P0.jsonl [--workers 2]
  python tools/build_symmetry_check.py --root R --cap-md5 M [M ...] --cases all --merge P0.jsonl ... --out record.json

``--cases`` is ``all`` (every ``<root>/geo_*`` folder with a stored wing) or a file of case names, one per
line. A shard ``k/n`` runs the exact check on the sorted cases whose index is ``k`` mod ``n``, ``--workers``
at a time in separate processes, and appends one JSON line per case to its part (a case the part already
holds is skipped, so a rerun resumes): the verdict, both maxima, the counts, the worst vertex, the wing's
connectivity and points fingerprints, the cap's key and tolerance, and the seconds it took. The merge
writes one record with the cap and its tolerance and every case's line, refusing duplicate, missing or
unknown cases and parts made with another cap or tolerance. The record holds no geometry.
"""

import argparse
import datetime
import json
import multiprocessing
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dedup import connectivity_fingerprint, points_fingerprint  # noqa: E402
from symmetry import check_root, load_root_cap, read_memmap_mesh  # noqa: E402

FORMAT = "hl-symmetry-check/1"
#: The stored wing below a case folder.
WING = "*.pdmsh/_tensordict/boundaries/vehicle"
#: The part lines' keys that describe the cap, not the case (kept out of the record's cases).
CAP_KEYS = ("case", "cap_md5", "tau", "y_eps")


def case_names(root: Path, cases: str) -> list[str]:
    """The sorted case names: every ``geo_*`` folder with a stored wing, or those listed in a file."""
    if cases == "all":
        return sorted(
            {p.relative_to(root).parts[0] for p in root.glob(f"geo_*/{WING}")}
        )
    lines = (s.strip() for s in Path(cases).read_text().splitlines())
    return sorted({s for s in lines if s and not s.startswith("#")})


def check_case(job: tuple[str, str, tuple[str, ...], float]) -> dict:
    """One case's part line."""
    root, case, md5s, y_eps = job
    t0 = time.time()
    cap = load_root_cap(root, md5s, y_eps)
    (wing,) = sorted((Path(root) / case).glob(WING))
    points, cells = read_memmap_mesh(wing)
    rec = {"case": case, **check_root(points, cells, cap, y_eps=y_eps)}
    rec["connectivity_fingerprint"] = connectivity_fingerprint(cells)
    rec["points_fingerprint"] = points_fingerprint(points)
    rec.update(cap_md5=cap.md5, tau=cap.tau, y_eps=y_eps, s=round(time.time() - t0, 1))
    return rec


def tool_commit() -> str | None:
    """The recipe's git commit, with ``+dirty`` when its src or tools differ from it; None outside git."""
    here = Path(__file__).resolve().parent.parent
    try:
        head = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(here), "status", "--porcelain", "--", "src", "tools"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return head + ("+dirty" if dirty else "")


def run_shard(a, root: Path) -> None:
    """Check the shard's cases not yet in its part and append their lines."""
    k, n = (int(x) for x in a.shard.split("/"))
    if not 0 <= k < n:
        raise ValueError(f"--shard {a.shard}: need 0 <= k < n")
    part = Path(a.part)
    done = set()
    if part.exists():
        done = {json.loads(s)["case"] for s in part.read_text().splitlines() if s}
    todo = [
        c
        for i, c in enumerate(case_names(root, a.cases))
        if i % n == k and c not in done
    ]
    md5s = tuple(a.cap_md5)
    load_root_cap(root, md5s, a.y_eps)  # refuse a wrong cap before any case runs
    jobs = [(str(root), c, md5s, a.y_eps) for c in todo]
    print(
        f"shard {a.shard}: {len(todo)} cases to check, {len(done)} already in {part}",
        flush=True,
    )
    with part.open("a") as f:
        if a.workers > 1:
            with multiprocessing.get_context("fork").Pool(a.workers) as pool:
                results = pool.imap_unordered(check_case, jobs)
                for rec in results:
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    print(json.dumps(rec), flush=True)
        else:
            for job in jobs:
                rec = check_case(job)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(json.dumps(rec), flush=True)


def merge(a, root: Path) -> None:
    """Merge the parts into one record, refusing duplicate, missing or unknown cases and other caps."""
    cap = load_root_cap(root, tuple(a.cap_md5), a.y_eps)
    expected = case_names(root, a.cases)
    cases: dict[str, dict] = {}
    for part in a.merge:
        for s in Path(part).read_text().splitlines():
            if not s:
                continue
            rec = json.loads(s)
            case = rec["case"]
            if case in cases:
                raise ValueError(f"{part}: case {case!r} appears twice among the parts")
            got = (rec["cap_md5"], rec["tau"], rec["y_eps"])
            if got != (cap.md5, cap.tau, a.y_eps):
                raise ValueError(
                    f"{part}: case {case!r} was checked with cap {got}, not this root's "
                    f"{(cap.md5, cap.tau, a.y_eps)}"
                )
            cases[case] = {k: v for k, v in rec.items() if k not in CAP_KEYS}
    missing = sorted(set(expected) - set(cases))
    unknown = sorted(set(cases) - set(expected))
    if missing or unknown:
        raise ValueError(
            f"the parts miss {len(missing)} cases (e.g. {missing[:5]}) and hold {len(unknown)} "
            f"unknown ones (e.g. {unknown[:5]})"
        )
    record = {
        "format": FORMAT,
        "dataset_root": str(root),
        "cap": {
            "file": cap.file,
            "md5": cap.md5,
            "n_cells": cap.n_cells,
            "n_outline_vertices": len(cap.outline),
            "y_eps": cap.y_eps,
        },
        "tolerance": cap.tolerance,
        "tool_commit": tool_commit(),
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "cases": {c: cases[c] for c in sorted(cases)},
    }
    Path(a.out).write_text(json.dumps(record, indent=1) + "\n")
    n_pass = sum(r["passed"] for r in cases.values())
    print(f"{a.out}: {len(cases)} cases, {n_pass} pass, tau {cap.tau}")


def main() -> None:
    """Parse the command line and run a shard or the merge."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="the dataset root")
    ap.add_argument(
        "--cap-md5",
        nargs="+",
        required=True,
        help="the accepted cap keys (AddSymmetryPlane's cap_md5 in datasets/highlift_surface.yaml)",
    )
    ap.add_argument("--cases", required=True, help='"all" or a file of case names')
    ap.add_argument(
        "--y-eps", type=float, default=1e-3, help="on-plane tolerance (curator's)"
    )
    ap.add_argument("--shard", help="k/n")
    ap.add_argument("--part", help="the shard's JSON-lines output")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--merge", nargs="+", help="the parts to merge")
    ap.add_argument("--out", help="the merged record")
    a = ap.parse_args()
    root = Path(a.root)
    if a.merge:
        if not a.out or a.shard or a.part:
            ap.error("--merge needs --out, and no --shard or --part")
        merge(a, root)
    else:
        if not a.shard or not a.part or a.out:
            ap.error("a shard needs --shard and --part (and no --out)")
        run_shard(a, root)


if __name__ == "__main__":
    main()
