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

"""Compare two records of :mod:`.record` (the A/B harness: base A, head B).

    python -m test.experimental.models.isla.goldens.compare A B [--all]

Exit 0 (PASS), 1 (FAIL) or 2 (INCONCLUSIVE). The reading rules:

- **CPU: bitwise.** Every input, output and gradient has the same SHA-256 on both sides.
- **CUDA (kernels auto): bitwise**, except
  - the draw's kept cells are compared first, in every repeat of both sides: where any
    two differ, the case is INCONCLUSIVE (the draw's float64 ``cumsum`` is not
    run-to-run deterministic on CUDA, so a kept cell can flip; rerun, never pass). On
    CPU a kept difference is a FAIL (the CPU draw is deterministic). DRAW-EXACT;
  - the draw's Horvitz-Thompson weights compare at 1e-15 relative to ``max|A|``, every
    pair of repeats within a side or across;
  - the outputs a case names as its ``family`` (float atomics, measured) compare
    against the run-to-run family: the median over every pair of repeats across the
    sides of ``max|B_i - A_j|`` must not exceed the largest ``max|X_i - X_j|`` within
    either side times ``FAMILY_SLACK`` (one code, one distribution: a cross pair is
    then one more draw of the within-side distances; a family that repeats bitwise
    demands bitwise), and that spread must stay under ``FAMILY_RTOL[dtype] * max|A|``.
- Any tensor not named a family that differs between the repeats of one side is a
  FAIL (non-determinism the case does not declare).
- Cases or tensors present on one side only are a FAIL, and so is a comparison of no
  case at all.
"""

import argparse
import fnmatch
import statistics
import sys
from pathlib import Path

import torch

from test.experimental.models.isla.goldens import record as R

PASS, FAIL, INCONCLUSIVE = 0, 1, 2
STATUS = {PASS: "PASS", FAIL: "FAIL", INCONCLUSIVE: "INCONCLUSIVE"}
#: DRAW-EXACT: the draw's weights on CUDA: their float64 sums wobble by a few ulps run
#: to run (and a kept cell can flip, read INCONCLUSIVE below); DRAW-EXACT makes both
#: bitwise.
WEIGHTS_RTOL = 1e-15
#: A family passes when the median cross-side distance is at most this many times the
#: largest within-side distance: five runs a side estimate the family's spread from
#: below (cross/within ratios up to 1.2 between two records of the same code, job
#: 379129, query_chunks' reference read).
FAMILY_SLACK = 2.0
#: The largest within-side spread a family may have, relative to ``max|A|`` by dtype,
#: so that a head whose own repeats are noisy cannot widen its own tolerance (a case
#: may state its own for named tensors, ``Case.family_rtol``). Measured on GB300 (jobs
#: 377864, 379129; the largest spread of every family over 5 repeats): float32
#: 1.9e-5 (the boundary reads' weight gradients), float64 4.6e-13, bfloat16 0.
FAMILY_RTOL = {"float32": 1e-4, "float64": 1e-11, "bfloat16": 2**-6}


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape or a.dtype != b.dtype:
        return float("inf")
    if not a.is_floating_point():
        return float((a.long() - b.long()).abs().max()) if a.numel() else 0.0
    a, b = a.double(), b.double()
    same = (a == b) | (a.isnan() & b.isnan())
    return float(torch.where(same, 0.0, (a - b).abs()).max()) if a.numel() else 0.0


def _runs(rec: dict, key: str, name: str) -> list:
    """Every repeat of a tensor (the record stores the first and those that differ)."""
    hashes = rec["cases"][key]["tensors"][name]["sha256"]
    stored = iter(rec["tensors"][key][name][1:])
    first = rec["tensors"][key][name][0]
    return [first if h == hashes[0] else next(stored) for h in hashes]


def _family(a: dict, b: dict, key: str, name: str) -> tuple[float, float]:
    """The median cross-side distance and the largest within-side distance."""
    ra, rb = _runs(a, key, name), _runs(b, key, name)
    within = [
        _maxdiff(x, y) for r in (ra, rb) for i, x in enumerate(r) for y in r[i + 1 :]
    ]
    cross = [_maxdiff(x, y) for x in ra for y in rb]
    return statistics.median(cross), max(within, default=0.0)


def _all_pairs(a: dict, b: dict, key: str, name: str) -> float:
    """The largest distance between any two repeats, within a side or across."""
    runs = _runs(a, key, name) + _runs(b, key, name)
    return max(
        (_maxdiff(x, y) for i, x in enumerate(runs) for y in runs[i + 1 :]),
        default=0.0,
    )


def compare_case(a: dict, b: dict, key: str) -> tuple[int, list]:
    """Status and rows ``(tensor, verdict, detail)`` of one case variant."""
    ca, cb = a["cases"][key], b["cases"][key]
    ta, tb = ca["tensors"], cb["tensors"]
    rows = []
    status = PASS
    if ta.keys() != tb.keys():
        only = sorted(set(ta) ^ set(tb))
        return FAIL, [(n, "FAIL", "on one side only") for n in only]
    cuda = ca["device"] == "cuda"
    ### DRAW-EXACT: the kept cells first, in every repeat of both sides.
    for n in ca["kept"]:
        if len(set(ta[n]["sha256"]) | set(tb[n]["sha256"])) > 1:
            d = _all_pairs(a, b, key, n)
            verdict = INCONCLUSIVE if cuda else FAIL
            rows.append(
                (n, STATUS[verdict], f"kept cells differ (max index difference {d:g})")
            )
            status = max(status, verdict)
    if status != PASS:
        return status, rows
    for n in ta:
        ha, hb = ta[n]["sha256"], tb[n]["sha256"]
        if ta[n]["dtype"] != tb[n]["dtype"] or ta[n]["shape"] != tb[n]["shape"]:
            rows.append(
                (
                    n,
                    "FAIL",
                    f"{ta[n]['dtype']}{ta[n]['shape']} against {tb[n]['dtype']}{tb[n]['shape']}",
                )
            )
            status = FAIL
            continue
        family = n in ca["family"] and cuda
        weights = n in ca["weights"] and cuda
        if not (family or weights) and (len(set(ha)) > 1 or len(set(hb)) > 1):
            rows.append(
                (
                    n,
                    "FAIL",
                    "differs between repeats of one side, and is not a declared family",
                )
            )
            status = FAIL
            continue
        if len(set(ha) | set(hb)) == 1:
            rows.append((n, "PASS", "bitwise"))
            continue
        scale = float(a["tensors"][key][n][0].double().abs().max())
        if weights:
            ### DRAW-EXACT: every pair of repeats, within a side or across, at 1e-15.
            d = _all_pairs(a, b, key, n)
            bound = WEIGHTS_RTOL * scale
            ok = d <= bound
            rows.append(
                (
                    n,
                    "PASS" if ok else "FAIL",
                    f"weights: max diff over every pair of repeats {d:.3e}, bound {bound:.3e} (1e-15 relative)",
                )
            )
        elif family:
            d, spread = _family(a, b, key, n)
            rtol = ca.get("family_rtol", {}).get(
                n, FAMILY_RTOL.get(ta[n]["dtype"], 0.0)
            )
            ceiling = rtol * scale
            ok = d <= FAMILY_SLACK * spread and spread <= ceiling
            rows.append(
                (
                    n,
                    "PASS" if ok else "FAIL",
                    f"family: median cross-side max diff {d:.3e}, within-side spread {spread:.3e} "
                    f"(ceiling {ceiling:.3e})",
                )
            )
        else:
            ok = False
            d = _maxdiff(a["tensors"][key][n][0], b["tensors"][key][n][0])
            rows.append((n, "FAIL", f"not bitwise: max diff {d:.3e}"))
        status = max(status, PASS if ok else FAIL)
    return status, rows


def compare(a: dict, b: dict, log=print, show_all: bool = False, only=None) -> int:
    """Compare two records; print a per-tensor table (the differing rows, or every row
    with ``show_all``) and a summary; return PASS, FAIL or INCONCLUSIVE."""
    fa, fb = a["fingerprint"], b["fingerprint"]
    for k in sorted(set(fa) | set(fb)):
        if fa.get(k) != fb.get(k):
            log(
                f"WARNING: platform differs in {k}: {fa.get(k)!r} against {fb.get(k)!r} (an A/B comparison needs one machine)"
            )
    keys = sorted(set(a["cases"]) | set(b["cases"]))
    if only:
        keys = [k for k in keys if any(fnmatch.fnmatchcase(k, p) for p in only)]
    worst, counts = PASS, {s: 0 for s in STATUS.values()}
    log(f"{'case variant':<62} {'tensor':<34} verdict  detail")
    for key in sorted(set(a.get("errors", {})) | set(b.get("errors", {}))):
        side = "A" if key in a.get("errors", {}) else "B"
        log(
            f"{key:<62} {'-':<34} FAIL     raised on side {side} (traceback in its record.json)"
        )
        counts["FAIL"] += 1
        worst = FAIL
    raised = set(a.get("errors", {})) | set(b.get("errors", {}))
    for key in keys:
        if key in raised:
            continue  # counted above
        if key not in a["cases"] or key not in b["cases"]:
            status, rows = (
                FAIL,
                [
                    (
                        "-",
                        "FAIL",
                        f"recorded on side {'A' if key in a['cases'] else 'B'} only",
                    )
                ],
            )
        else:
            status, rows = compare_case(a, b, key)
        counts[STATUS[status]] += 1
        worst = max(worst, status)
        for n, verdict, detail in rows:
            if show_all or verdict != "PASS" or detail != "bitwise":
                log(f"{key:<62} {n:<34} {verdict:<8} {detail}")
    for side, rec in (("A", a), ("B", b)):
        for cid, why in rec.get("skipped", {}).items():
            log(f"skipped on side {side}: {cid}: {why}")
    if not keys:
        log("nothing compared: no case variant is in either record (or matches --only)")
        worst = FAIL
    log(
        f"{len(keys)} case variants: "
        + ", ".join(f"{v} {k}" for k, v in counts.items())
    )
    log(f"RESULT {STATUS[worst]}")
    return worst


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("a", type=Path, help="record directory of the base")
    p.add_argument("b", type=Path, help="record directory of the head")
    p.add_argument(
        "--all",
        action="store_true",
        help="print every tensor's row, not only the non-bitwise ones",
    )
    p.add_argument(
        "--only",
        nargs="*",
        help="fnmatch patterns of the case variants to compare (default all)",
    )
    args = p.parse_args(argv)
    a, b = R.load(args.a), R.load(args.b)
    R.print_versions(a, "A ")
    R.print_versions(b, "B ")
    return compare(a, b, show_all=args.all, only=args.only)


if __name__ == "__main__":
    sys.exit(main())
