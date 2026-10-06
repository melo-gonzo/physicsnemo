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

"""QI-GLOBAL reduction: the global read (G) against interacting (I), read-only (P) and
the truncated read (T) on AirfRANS, four seeds each, with the preregistered bars.

Reads ``infer.py`` scorer rows (``<dir>/<run>/metrics.jsonl``, searched in every
``--scores`` directory in order) and ``qi_probe.py`` band anatomies (``<dir>/<run>.json``
in the ``--probes`` directories). Prints and writes: per arm and field the seed
values, mean and standard error; the ratios G/I, G/P, G/T, T/I, P/I of the seed means
with the paired per-seed ratios; the band anatomy (pooled relative L2 per |sdf|/L
band, seed mean); and the bars (REPORT.md). Missing runs are reported and left out.

usage: python qg_reduce.py --scores D1 [D2 ...] [--probes P1 ...] --out O.json
"""

import argparse
import json
import math
from pathlib import Path

FIELDS = ("pressure", "velocity", "nut")
SEEDS = (42, 43, 44, 45)
ARMS = {
    "G": {s: f"gl_af_global_seed{s}" for s in SEEDS},
    "I": {
        42: "qi_af_int_seed42",
        43: "qi_af_int_seed43",
        44: "gl_af_int_seed44",
        45: "gl_af_int_seed45",
    },
    "P": {
        42: "qi_af_pas_seed42",
        43: "qi_af_pas_seed43",
        44: "gl_af_ro_seed44",
        45: "gl_af_ro_seed45",
    },
    "T": {
        42: "kq_af_sum4_seed42",
        43: "kq_af_sum4_seed43",
        44: "gl_af_trunc_seed44",
        45: "gl_af_trunc_seed45",
    },
}
WAKE = "1.0-inf"


def summary_of(path):
    """The case mean of the scorer's per-case rows (infer.py writes no 2-D summary)."""
    rows = [json.loads(line) for line in open(path) if line.strip()]
    cfg = next(r for r in rows if r.get("phase") == "config")
    steps = [r["metrics"] for r in rows if r.get("phase") == "infer_step"]
    if not steps:
        return None
    return {
        "epoch": cfg["epoch"],
        "n": len(steps),
        **{f: sum(m[f"{f}_l2"] for m in steps) / len(steps) for f in FIELDS},
    }


def find(dirs, rel):
    """The first of dirs that holds rel."""
    for d in dirs:
        p = Path(d) / rel
        if p.exists():
            return p
    return None


def bands_of(path):
    """Pooled relative L2 per band over the cases of one probe JSON."""
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
        b: {"n": a["n"], **{f: math.sqrt(a[f][0] / a[f][1]) for f in FIELDS}}
        for b, a in acc.items()
    }


def stats(values):
    """Mean and standard error of the non-missing values."""
    v = [x for x in values if x is not None]
    if not v:
        return None, None
    m = sum(v) / len(v)
    se = (
        math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1) / len(v))
        if len(v) > 1
        else None
    )
    return m, se


def main():
    """Run the reduction described in the module docstring."""
    p = argparse.ArgumentParser()
    p.add_argument("--scores", nargs="+", required=True)
    p.add_argument("--probes", nargs="*", default=[])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = {"arms": {}, "ratios": {}, "bands": {}, "missing": []}
    scores = {}
    for arm, runs in ARMS.items():
        scores[arm] = {}
        for s, run in runs.items():
            path = find(a.scores, f"{run}/metrics.jsonl")
            row = summary_of(path) if path else None
            if row is None:
                out["missing"].append(run)
            scores[arm][s] = row
        out["arms"][arm] = {
            f: {
                "seeds": {s: (r or {}).get(f) for s, r in scores[arm].items()},
                "mean": stats([(r or {}).get(f) for r in scores[arm].values()])[0],
                "se": stats([(r or {}).get(f) for r in scores[arm].values()])[1],
            }
            for f in FIELDS
        }
        out["arms"][arm]["epochs"] = {
            s: (r or {}).get("epoch") for s, r in scores[arm].items()
        }
    for num, den in (("G", "I"), ("G", "P"), ("G", "T"), ("T", "I"), ("P", "I")):
        out["ratios"][f"{num}/{den}"] = {}
        for f in FIELDS:
            mn, md = out["arms"][num][f]["mean"], out["arms"][den][f]["mean"]
            paired = {
                s: scores[num][s][f] / scores[den][s][f]
                for s in SEEDS
                if scores[num][s] and scores[den][s]
            }
            out["ratios"][f"{num}/{den}"][f] = {
                "ratio": mn / md if mn and md else None,
                "paired": paired,
            }
    for arm, runs in ARMS.items():
        per_seed = {}
        for s, run in runs.items():
            path = find(a.probes, f"{run}.json")
            if path:
                per_seed[s] = bands_of(path)
        if not per_seed:
            continue
        bands = next(iter(per_seed.values())).keys()
        out["bands"][arm] = {
            b: {
                "n": next(iter(per_seed.values()))[b]["n"],
                "seeds": sorted(per_seed),
                **{f: stats([v[b][f] for v in per_seed.values()])[0] for f in FIELDS},
            }
            for b in bands
        }
    ### The bars (REPORT.md), on the float32 scores.
    r = out["ratios"]
    g_i = {f: r["G/I"][f]["ratio"] for f in FIELDS}
    t_i = r["T/I"]["nut"]["ratio"]
    bars = {}
    if all(v is not None for v in g_i.values()):
        wake = None
        if WAKE in out["bands"].get("G", {}) and WAKE in out["bands"].get("I", {}):
            wake = out["bands"]["G"][WAKE]["nut"] / out["bands"]["I"][WAKE]["nut"]
        bars["wake_nut_G/I"] = wake
        bars["closes_the_gap"] = bool(
            g_i["pressure"] <= 1.03
            and g_i["velocity"] <= 1.03
            and g_i["nut"] <= 1.05
            and wake is not None
            and wake <= 1.10
        )
        if t_i is not None:
            wake_gt = None
            if WAKE in out["bands"].get("T", {}) and WAKE in out["bands"].get("G", {}):
                wake_gt = (
                    out["bands"]["G"][WAKE]["nut"] / out["bands"]["T"][WAKE]["nut"]
                )
            bars["wake_nut_G/T"] = wake_gt
            bars["helps_the_wake"] = bool(
                (g_i["nut"] - 1.0) <= 0.5 * (t_i - 1.0)
                and wake_gt is not None
                and wake_gt < 1.0
            )
            ag, at = out["arms"]["G"]["nut"], out["arms"]["T"]["nut"]
            if ag["se"] is not None and at["se"] is not None:
                se = math.hypot(ag["se"], at["se"])
                bars["nut_G-T_over_se"] = (ag["mean"] - at["mean"]) / se
                bars["no_wake_effect"] = abs(ag["mean"] - at["mean"]) <= se
    out["bars"] = bars
    Path(a.out).write_text(json.dumps(out, indent=1))
    ### Print.
    print("arm  " + "  ".join(f"{f:>24s}" for f in FIELDS))
    for arm in ARMS:
        row = []
        for f in FIELDS:
            v = out["arms"][arm][f]
            se = f"{v['se']:.4f}" if v["se"] is not None else "-"
            row.append(f"{v['mean']:.4f} +- {se}" if v["mean"] else "-")
        print(f"{arm:4s} " + "  ".join(f"{x:>24s}" for x in row))
        print(
            "     seeds "
            + " | ".join(
                f"{f}: "
                + "/".join(
                    f"{x:.4f}" if x else "-"
                    for x in out["arms"][arm][f]["seeds"].values()
                )
                for f in FIELDS
            )
        )
    for k, v in out["ratios"].items():
        print(
            f"{k:5s} "
            + "  ".join(
                f"{f} {v[f]['ratio']:.3f} ("
                + "/".join(f"{x:.3f}" for x in v[f]["paired"].values())
                + ")"
                if v[f]["ratio"]
                else f"{f} -"
                for f in FIELDS
            )
        )
    for arm, bands in out["bands"].items():
        print(
            f"bands {arm}: "
            + "; ".join(
                f"{b} (n={v['n']}): " + "/".join(f"{v[f]:.4f}" for f in FIELDS)
                for b, v in bands.items()
            )
        )
    print("bars:", json.dumps(bars))
    if out["missing"]:
        print("missing:", " ".join(out["missing"]))


if __name__ == "__main__":
    main()
