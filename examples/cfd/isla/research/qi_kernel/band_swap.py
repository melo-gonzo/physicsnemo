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

"""Where an arm's excess error over a reference lives, by |sdf|/L band (no training, probe JSONs only).

``qi_probe.py`` stores, per held-out case and |sdf|/L band, the summed squared error and the summed squared target
of each field on the native request. Every arm is probed on the same points (same cases, same request), so a case's
relative L2 is ``sqrt(sum_b err_b / sum_b tgt_b)`` and swapping one band's error for the reference's error on the same
points is an exact counterfactual. Per field and band this prints: the share of points, the arm's share of the summed
squared error (pooled over cases), the pooled relative L2 of arm and reference in that band, and the seed-mean ratio
arm / reference of the case-mean relative L2 (the scorer's statistic) after swapping that band's error for the
reference's ("swap"), next to the unswapped ratio ("all").

usage: python band_swap.py --arm 'DIR/kq_drvol_sum4_seed{}.json' --ref 'QI/qi_drvol_int_seed{}.json' [--seeds 42 43]
"""

import argparse
import json
from pathlib import Path

FIELDS = ("pressure", "velocity", "nut")


def cases_of(pattern, seed):
    """Per-case band tables of one probe JSON, keyed by sample index."""
    d = json.loads(Path(pattern.format(seed)).read_text())
    return {c["sample_index"]: c["bands"] for c in d["cases"]}


def case_mean(cases, field, swap=None, ref=None):
    """Case-mean relative L2 of one field, optionally with band ``swap``'s error taken from ``ref``."""
    tot = 0.0
    for k, bands in cases.items():
        err = sum((ref[k] if b == swap else bands)[b][field][0] for b in bands)
        tgt = sum(v[field][1] for v in bands.values())
        tot += (err / tgt) ** 0.5
    return tot / len(cases)


def main():
    """Print the band anatomy of arm against reference."""
    a = argparse.ArgumentParser()
    a.add_argument("--arm", required=True)
    a.add_argument("--ref", required=True)
    a.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    a.add_argument("--out", default=None)
    o = a.parse_args()
    arm = {s: cases_of(o.arm, s) for s in o.seeds}
    ref = {s: cases_of(o.ref, s) for s in o.seeds}
    for s in o.seeds:
        if set(arm[s]) != set(ref[s]):
            raise ValueError(f"seed {s}: different cases")
        for k in arm[s]:
            for b, v in arm[s][k].items():
                if v["n"] != ref[s][k][b]["n"]:
                    raise ValueError(f"seed {s} case {k}: different points")
    bands = list(next(iter(arm[o.seeds[0]].values())))
    n_tot = sum(v["n"] for c in arm[o.seeds[0]].values() for v in c.values())
    res = {"arm": o.arm, "ref": o.ref, "seeds": o.seeds, "fields": {}}
    print(f"arm {o.arm}\nref {o.ref}\nseeds {o.seeds}")
    for f in FIELDS:
        r_all = sum(case_mean(arm[s], f) for s in o.seeds) / sum(
            case_mean(ref[s], f) for s in o.seeds
        )
        rows = {}
        e_tot = sum(
            v[f][0] for s in o.seeds for c in arm[s].values() for v in c.values()
        )
        for b in bands:
            n = sum(c[b]["n"] for c in arm[o.seeds[0]].values())

            def pooled(x, b=b):
                e = sum(c[b][f][0] for s in o.seeds for c in x[s].values())
                t = sum(c[b][f][1] for s in o.seeds for c in x[s].values())
                return (e / t) ** 0.5, e

            pa, ea = pooled(arm)
            pr, _ = pooled(ref)
            r_swap = sum(
                case_mean(arm[s], f, swap=b, ref=ref[s]) for s in o.seeds
            ) / sum(case_mean(ref[s], f) for s in o.seeds)
            rows[b] = {
                "points": n / n_tot,
                "err_share": ea / e_tot,
                "arm_rel_l2": pa,
                "ref_rel_l2": pr,
                "ratio_after_swap": r_swap,
            }
        res["fields"][f] = {"ratio_all": r_all, "bands": rows}
        print(f"== {f}: case-mean ratio arm / ref {r_all:.3f}")
        print("  band        points  err share  arm L2   ref L2   ratio after swap")
        for b, v in rows.items():
            print(
                f"  {b:10s} {v['points']:6.1%}  {v['err_share']:8.1%}  {v['arm_rel_l2']:.4f}  "
                f"{v['ref_rel_l2']:.4f}   {v['ratio_after_swap']:.3f}"
            )
    if o.out:
        Path(o.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
