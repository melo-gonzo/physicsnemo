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

"""QI-GLOBAL weighted-error readout: G, I, P, T on AirfRANS under count and volume
(exact dual-area) weighting of the same predictions (an added readout; the verdict stays on the
protocol scores).

Reads ``we_probe.py`` JSONs (the weighted-error lane's probe: every mesh point of the
200 held-out airfoils, exact dual areas) from the ``--probes`` directories, uses
``we_reduce.case_errors`` for the two weightings (case means, then seed means) and
prints per arm the errors and the ratios G/I, G/P, G/T, P/I, T/I of the seed means
with the paired seed ratios, and per |sdf|/L band the volume-weighted RMS error ratio.
The interacting model I (chunk-trained) is read on its own tiles (law ``tiles``);
the read-only arms need no writers (law ``none``).

usage: python qg_we_reduce.py --probes D1 [D2 ...] --we-reduce DIR_OF_we_reduce.py --out O.json
"""

import argparse
import json
import sys
from pathlib import Path

FIELDS = ("pressure", "velocity", "nut")
WEIGHTINGS = ("count", "volume")
SEEDS = (42, 43, 44, 45)
ARMS = {
    "G": ({s: f"gl_af_global_seed{s}" for s in SEEDS}, "none"),
    "I": (
        {
            42: "qi_af_int_seed42",
            43: "qi_af_int_seed43",
            44: "gl_af_int_seed44",
            45: "gl_af_int_seed45",
        },
        "tiles",
    ),
    "P": (
        {
            42: "qi_af_pas_seed42",
            43: "qi_af_pas_seed43",
            44: "gl_af_ro_seed44",
            45: "gl_af_ro_seed45",
        },
        "none",
    ),
    "T": (
        {
            42: "kq_af_sum4_seed42",
            43: "kq_af_sum4_seed43",
            44: "gl_af_trunc_seed44",
            45: "gl_af_trunc_seed45",
        },
        "none",
    ),
}


def mean(xs):
    """Field-wise mean of the non-missing dicts."""
    xs = [x for x in xs if x is not None]
    return {f: sum(x[f] for x in xs) / len(xs) for f in FIELDS} if xs else None


def main():
    """Run the reduction described in the module docstring."""
    p = argparse.ArgumentParser()
    p.add_argument("--probes", nargs="+", required=True)
    p.add_argument("--we-reduce", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    sys.path.insert(0, a.we_reduce)
    from we_reduce import case_errors

    res, missing = {}, []
    for arm, (runs, law) in ARMS.items():
        res[arm] = {}
        for s, run in runs.items():
            path = next(
                (
                    Path(d) / f"{run}.json"
                    for d in a.probes
                    if (Path(d) / f"{run}.json").exists()
                ),
                None,
            )
            if path is None:
                missing.append(run)
                continue
            d = json.loads(path.read_text())
            ce = [case_errors(c["laws"][law]) for c in d["cases"]]
            res[arm][s] = {
                **{w: mean([c[w] for c in ce]) for w in WEIGHTINGS},
                "band_rms": {
                    k: mean([c["band_rms"][k] for c in ce]) for k in ce[0]["band_rms"]
                },
                "n_cases": len(ce),
                "epoch": d.get("epoch"),
            }
    out = {"arms": {}, "ratios": {}, "missing": missing}
    for arm, by_seed in res.items():
        out["arms"][arm] = {
            w: {
                "mean": mean([v[w] for v in by_seed.values()]),
                "seeds": {s: v[w] for s, v in by_seed.items()},
            }
            for w in WEIGHTINGS
        }
        if by_seed:
            bands = next(iter(by_seed.values()))["band_rms"]
            out["arms"][arm]["band_rms"] = {
                k: mean([v["band_rms"][k] for v in by_seed.values()]) for k in bands
            }
        out["arms"][arm]["n_cases"] = {s: v["n_cases"] for s, v in by_seed.items()}
    for num, den in (("G", "I"), ("G", "P"), ("G", "T"), ("P", "I"), ("T", "I")):
        r = {}
        for w in WEIGHTINGS:
            mn, md = out["arms"][num][w]["mean"], out["arms"][den][w]["mean"]
            r[w] = {
                "ratio": None
                if not mn or not md
                else {f: mn[f] / md[f] for f in FIELDS},
                "paired": {
                    s: {f: res[num][s][w][f] / res[den][s][w][f] for f in FIELDS}
                    for s in SEEDS
                    if s in res[num] and s in res[den]
                },
            }
        bn, bd = out["arms"][num].get("band_rms"), out["arms"][den].get("band_rms")
        r["band_rms"] = (
            {k: {f: bn[k][f] / bd[k][f] for f in FIELDS} for k in bn}
            if bn and bd
            else None
        )
        out["ratios"][f"{num}/{den}"] = r
    Path(a.out).write_text(json.dumps(out, indent=1))

    def fmt(x, nd=3):
        return "/".join(f"{x[f]:.{nd}f}" for f in FIELDS) if x else "-"

    print(
        "relative L2 (training space), seed means p/U/nut; I on tiles, read-only arms alone"
    )
    for arm in ARMS:
        print(
            f"  {arm}: "
            + "  ".join(
                f"{w} {fmt(out['arms'][arm][w]['mean'], 4)}" for w in WEIGHTINGS
            )
            + f"  cases {out['arms'][arm]['n_cases']}"
        )
    for k, r in out["ratios"].items():
        print(
            f"{k}: "
            + "  ".join(
                f"{w} {fmt(r[w]['ratio'])} ["
                + " ".join(fmt(x, 2) for x in r[w]["paired"].values())
                + "]"
                for w in WEIGHTINGS
            )
        )
        if r["band_rms"]:
            print(
                "   band vol-RMS ratio: "
                + "; ".join(f"{b} {fmt(v, 2)}" for b, v in r["band_rms"].items())
            )
    if missing:
        print("missing:", " ".join(missing))


if __name__ == "__main__":
    main()
