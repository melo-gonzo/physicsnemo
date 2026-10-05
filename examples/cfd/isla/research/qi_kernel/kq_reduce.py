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

"""QI-KERNEL reduction: the kernel-read lanes against the QI references, with the brief's bars.

Reads ``infer.py`` scorer rows (``<score dir>/<run>/metrics.jsonl``) for the kernel-read lanes and for the QI
campaign's interacting and read-only references, and (when present) ``qi_probe.py`` band anatomies. Prints and
writes per field: seed means, the ratio (read-only + kernel) / interacting-256 with the paired per-seed ratios, the
same against read-only-256, the verdict by the brief's bars (<= 1.03 every field and no paired seed above 1.06:
equal or better; >= 1.10 on any field: falsified; between otherwise), and the error by |sdf|/L band on DrivAerML.

usage: python kq_reduce.py --kernel-scores DIR --ref-scores DIR [--kernel-probes DIR --ref-probes DIR] --out O.json
"""

import argparse
import json
from pathlib import Path

FIELDS = ("pressure", "velocity", "nut")
SEEDS = (42, 43)
DATASETS = {"drvol": "DrivAerML Volume", "af": "AirfRANS"}


def summary_of(path):
    """The scorer's summary, or the mean of its per-case rows when infer.py wrote none (2-D AirfRANS)."""
    rows = [json.loads(line) for line in open(path) if line.strip()]
    s = [r for r in rows if r.get("phase") == "infer_summary"]
    cfg = next(r for r in rows if r.get("phase") == "config")
    if s:
        return (
            {f: s[-1]["metrics"][f"{f}_l2"] for f in FIELDS},
            cfg["epoch"],
            s[-1]["num_samples"],
        )
    steps = [r["metrics"] for r in rows if r.get("phase") == "infer_step"]
    return (
        {f: sum(m[f"{f}_l2"] for m in steps) / len(steps) for f in FIELDS},
        cfg["epoch"],
        len(steps),
    )


def bands_of(path):
    """Relative L2 per |sdf|/L band, pooled over the held-out cases of one probe JSON."""
    d = json.loads(Path(path).read_text())
    cases = d["cases"] if isinstance(d, dict) and "cases" in d else d
    acc = {}
    for c in cases:
        for band, v in c.get("bands", {}).items():
            a = acc.setdefault(band, {"n": 0, **{f: [0.0, 0.0] for f in FIELDS}})
            a["n"] += v["n"]
            for f in FIELDS:
                a[f][0] += v[f][0]
                a[f][1] += v[f][1]
    return {
        b: {
            "n": a["n"],
            **{f: (a[f][0] / a[f][1]) ** 0.5 if a[f][1] else None for f in FIELDS},
        }
        for b, a in acc.items()
    }


def verdict(ratios):
    """The brief's bars on {field: (seed-mean ratio, [paired ratios])}."""
    if any(m >= 1.10 for m, _ in ratios.values()):
        return "falsified"
    if all(m <= 1.03 and max(p) <= 1.06 for m, p in ratios.values()):
        return "equal or better"
    return (
        "between (above 1.03: "
        + ", ".join(f for f, (m, p) in ratios.items() if m > 1.03 or max(p) > 1.06)
        + ")"
    )


def main():
    """Reduce the scorer rows and probes named on the command line."""
    a = argparse.ArgumentParser()
    a.add_argument("--kernel-scores", required=True)
    a.add_argument("--ref-scores", required=True)
    a.add_argument("--kernel-probes", default=None)
    a.add_argument("--ref-probes", default=None)
    a.add_argument("--kernel-arm", default="sum4")
    a.add_argument("--out", required=True)
    o = a.parse_args()
    res = {
        "bars": "<= 1.03 every field (seed means) and no paired seed > 1.06: equal or better; >= 1.10: falsified",
        "datasets": {},
    }
    for ds, label in DATASETS.items():
        runs = {
            "kernel": {
                s: Path(o.kernel_scores)
                / f"kq_{ds}_{o.kernel_arm}_seed{s}"
                / "metrics.jsonl"
                for s in SEEDS
            },
            "int": {
                s: Path(o.ref_scores) / f"qi_{ds}_int_seed{s}" / "metrics.jsonl"
                for s in SEEDS
            },
            "ro": {
                s: Path(o.ref_scores) / f"qi_{ds}_pas_seed{s}" / "metrics.jsonl"
                for s in SEEDS
            },
        }
        vals, missing = {}, []
        for arm, paths in runs.items():
            for s, p in paths.items():
                if p.exists():
                    vals.setdefault(arm, {})[s] = summary_of(p)
                else:
                    missing.append(str(p))
        out = {"label": label, "missing": missing, "arms": {}}
        for arm, by_seed in vals.items():
            out["arms"][arm] = {
                "per_seed": {s: v[0] for s, v in by_seed.items()},
                "epochs": {s: v[1] for s, v in by_seed.items()},
                "cases": {s: v[2] for s, v in by_seed.items()},
                "mean": {
                    f: sum(v[0][f] for v in by_seed.values()) / len(by_seed)
                    for f in FIELDS
                },
            }
        for ref in ("int", "ro"):
            if (
                "kernel" in vals
                and ref in vals
                and set(vals["kernel"]) == set(vals[ref]) == set(SEEDS)
            ):
                k, r = out["arms"]["kernel"], out["arms"][ref]
                ratios = {
                    f: (
                        k["mean"][f] / r["mean"][f],
                        [k["per_seed"][s][f] / r["per_seed"][s][f] for s in SEEDS],
                    )
                    for f in FIELDS
                }
                out[f"kernel_over_{ref}"] = ratios
                if ref == "int":
                    out["verdict"] = verdict(ratios)
        if o.kernel_probes and o.ref_probes and ds == "drvol":
            bands = {}
            for arm, d, name in (
                ("kernel", o.kernel_probes, f"kq_{ds}_{o.kernel_arm}_seed{{}}"),
                ("int", o.ref_probes, f"qi_{ds}_int_seed{{}}"),
                ("ro", o.ref_probes, f"qi_{ds}_pas_seed{{}}"),
            ):
                for s in SEEDS:
                    p = Path(d) / (name.format(s) + ".json")
                    if p.exists():
                        bands.setdefault(arm, {})[s] = bands_of(p)
            out["bands"] = bands
        res["datasets"][ds] = out
    Path(o.out).write_text(json.dumps(res, indent=1))
    for ds, out in res["datasets"].items():
        print(f"== {out['label']}  missing: {len(out['missing'])}")
        for arm, v in out["arms"].items():
            cells = []
            for f in FIELDS:
                per = "/".join(
                    "%.4f" % v["per_seed"][s][f] for s in sorted(v["per_seed"])
                )
                cells.append("%s %.4f (%s)" % (f, v["mean"][f], per))
            print(f"  {arm:6s} " + "  ".join(cells))
        for ref in ("int", "ro"):
            if f"kernel_over_{ref}" in out:
                r = out[f"kernel_over_{ref}"]
                cells = [
                    "%s %.3f (%s)" % (f, m, " / ".join("%.3f" % x for x in p))
                    for f, (m, p) in r.items()
                ]
                print(f"  kernel / {ref}: " + "  ".join(cells))
        if "verdict" in out:
            print(f"  verdict: {out['verdict']}")


if __name__ == "__main__":
    main()
