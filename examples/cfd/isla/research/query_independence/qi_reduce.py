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

"""Reduce the QI campaign: readout 1 from the scorer's metrics.jsonl, readouts 2-5 and the anatomy from the probe
JSONs, graded against the bars of PREREG.md. Plain Python (json only), run on a login node.

usage: python3 qi_reduce.py <task_dir> <out.json>
"""

import json
import sys
from pathlib import Path

FIELDS = ("pressure", "velocity", "nut")
DATASETS = {"drvol": "DrivAerML Volume (48 validation cars, 200,000 cells + 200,000 queries)",
            "af": "AirfRANS interior (200 test airfoils, 100,000 queries)"}
SEEDS = (42, 43)


def summary_of(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    s = [r for r in rows if r.get("phase") == "infer_summary"]
    cfg = next(r for r in rows if r.get("phase") == "config")
    return {f: s[-1]["metrics"][f"{f}_l2"] for f in FIELDS}, cfg["epoch"], s[-1]["num_samples"]


def main():
    task, out = Path(sys.argv[1]), Path(sys.argv[2])
    res = {"prereg": "examples/cfd/isla/research/query_independence/PREREG.md", "datasets": DATASETS, "readout1": {}, "probe": {}, "missing": []}
    for ds in DATASETS:
        arms = {}
        for arm in ("int", "pas"):
            seeds = {}
            for seed in SEEDS:
                run = f"qi_{ds}_{arm}_seed{seed}"
                p = task / "eval" / "score" / run / "metrics.jsonl"
                if not p.exists():
                    res["missing"].append(str(p))
                    continue
                m, epoch, n = summary_of(p)
                seeds[seed] = {"fields": m, "epoch": epoch, "n_cases": n}
                pj = task / "eval" / "probe" / f"{run}.json"
                if pj.exists():
                    res["probe"][run] = json.load(open(pj))["summary"]
                else:
                    res["missing"].append(str(pj))
            arms[arm] = seeds
        if all(len(arms[a]) == len(SEEDS) for a in arms):
            mean = {a: {f: sum(arms[a][s]["fields"][f] for s in SEEDS) / len(SEEDS) for f in FIELDS} for a in arms}
            ratio = {f: mean["pas"][f] / mean["int"][f] for f in FIELDS}
            paired = {f: [arms["pas"][s]["fields"][f] / arms["int"][s]["fields"][f] for s in SEEDS] for f in FIELDS}
            res["readout1"][ds] = {"arms": arms, "mean": mean, "ratio_pas_over_int": ratio, "paired": paired}
    r1 = res["readout1"]
    if len(r1) == len(DATASETS):
        ratios = [(ds, f, r1[ds]["ratio_pas_over_int"][f], r1[ds]["paired"][f]) for ds in r1 for f in FIELDS]
        if all(r <= 1.03 and max(p) <= 1.06 for _, _, r, p in ratios):
            verdict = "SUPPORTED (equal or better)"
        elif any(r >= 1.10 for _, _, r, _ in ratios):
            verdict = "FALSIFIED"
        else:
            verdict = "BETWEEN"
        res["verdict_accuracy"] = verdict
        res["better_fields"] = [f"{ds}:{f}" for ds, f, r, p in ratios if r <= 0.97 and max(p) < 1]
        res["fields_above_1.03"] = [f"{ds}:{f}" for ds, f, r, p in ratios if r > 1.03 or max(p) > 1.06]
    out.write_text(json.dumps(res, indent=1))
    ### human-readable summary
    for ds, d in r1.items():
        print(f"\n## {DATASETS[ds]}")
        print("| field | interacting (42 / 43) | passive (42 / 43) | passive / interacting | paired |")
        print("|---|---|---|---|---|")
        for f in FIELDS:
            i = [d["arms"]["int"][s]["fields"][f] for s in SEEDS]
            p = [d["arms"]["pas"][s]["fields"][f] for s in SEEDS]
            print(f"| {f} | {d['mean']['int'][f]:.4f} ({i[0]:.4f} / {i[1]:.4f}) | {d['mean']['pas'][f]:.4f} ({p[0]:.4f} / {p[1]:.4f}) "
                  f"| {d['ratio_pas_over_int'][f]:.3f} | {' / '.join(f'{x:.3f}' for x in d['paired'][f])} |")
    if "verdict_accuracy" in res:
        print(f"\nAccuracy verdict: {res['verdict_accuracy']}; better: {res['better_fields']}; above 1.03: {res['fields_above_1.03']}")
    for run, s in sorted(res["probe"].items()):
        print(f"\n{run}: native {fmt(s['native'])}")
        print(f"  rear alone / inside native: {fmt({f: s['rear_alone'][f] / s['rear_inside_native'][f] for f in FIELDS}, 3)}  max diff {fmt(s['rear_maxdiff_max'], e=True)}")
        if "sentinels" in s:
            print("  sentinels " + "; ".join(f"{k}: {fmt(v)}" for k, v in s["sentinels"].items()))
            print("  sentinel max diff " + "; ".join(f"{k}: {fmt(v, e=True)}" for k, v in s["sentinel_maxdiff_max"].items()))
            print(f"  chunked max diff {fmt(s['chunked_maxdiff_max'], e=True)}; posed (float32) {fmt(s['posed_max'], e=True)}")
        if "fp64_case0" in s:
            print("  float64: " + "; ".join(f"{k} {fmt(v, e=True)}" for k, v in s["fp64_case0"].items()))
        print("  bands " + "; ".join(f"{b}: n={v['n']} p={v['pressure']:.4f} u={v['velocity']:.4f} nut={v['nut']:.4f}" for b, v in s["bands_pooled_rel_l2"].items()))
    if res["missing"]:
        print(f"\nmissing ({len(res['missing'])}): " + ", ".join(res["missing"]))


def fmt(d, nd=4, e=False):
    return " ".join(f"{k}={v:.2e}" if e else f"{k}={v:.{nd}f}" for k, v in d.items())


if __name__ == "__main__":
    main()
