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

"""ISLA 0.9.0 standings (2026-10-22): the results/char/<dataset>_a2.json record of each dataset with ISLA's arm replaced by
ISLA 0.9.0 at its reference configuration (width 512, 12 layers, 512 slices, 3.75e-4), scored in float32 on the final
checkpoint by the protocol scorer: native mesh / uniform point cloud / rotated + translated. The baseline arms (GeoTransolver
256, Transolver 512 x 16, both reading the unit freestream direction) and the keys that document them (fair_input, scored_set,
raw_drive_<arch>, transolver_100k_scoring, unlabelled_geotransolver, char_baselines) are the _a2 file's, unchanged.

ISLA's arm is built from the scorer's rows, copied unmodified into results/campaigns_2026-10-22/standing_090/scores/
<evals>/<run>/<protocol>/<run>/metrics.jsonl from the record folders (scratch/isla-ref090-2026-10-20/scores/,
scratch/isla-hilift090-2026-10-20/scores/; originals on LHR under $T/<evals>/): per seed and protocol the mean over cases of
the per-case relative L2 (the last infer_step row per sample id, after the last config row), then the mean over seeds. Each
config row must name the final epoch and each protocol must hold the expected case count.

STANDING below is the one table a later update edits: per dataset the run stem, the evals folder, the seeds, the final epoch,
the case count and the status. A status of "in_flight" writes the in-flight text beside the numbers ("in_flight": {"what",
"replacement"}, read by book/standing.py as "These numbers come from <what>. <replacement> is being measured ...").
"Fix-trained" rows are trained with the compiled training path after its fault (pytorch/pytorch#195845) was fixed;
"eager-trained" rows are trained without compilation. When the eager-trained AirfRANS and lug rows land: copy their score
files into the scores folder, set the stem, evals folder and status "final" here, and rerun.

DEGENERATE marks a protocol that changes no arm's input: its cells keep the scorer's numbers under "scored" and carry
"mean": null with the reason, and the file's "degenerate" key says why. On the DrivAerML Volume task ISLA draws its boundary
tokens by area inside the model, and the baselines read only the interior points and the signed distance, so the uniform point
cloud gives every arm its native input.

HiLift adds "registered" (R-HL090 of scratch/isla-hilift090-2026-10-20/PREREG.md: ISLA 0.9.0 against the normalized-read
nearest-token arm of hilift_surface_a2.json, as registered) and "case_level" (the per-case readouts of hilift_a2_2026-10-20.py
against the baselines' per-case rows in results/campaigns_2026-10-20/hilift/scores/). AirfRANS adds the arm
"ISLA-DTT 512 with 512 slices" (query_independent=True, the same configuration).
usage: python r090_protocol_artifacts_2026-10-22.py"""

import json
import os
import re
import statistics as st

R = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(R)
SCORES = f"{RES}/campaigns_2026-10-22/standing_090/scores"
ISLA_ARM, DTT_ARM = "ISLA 512 with 512 slices", "ISLA-DTT 512 with 512 slices"
S2, S4 = ("42", "43"), ("42", "43", "44", "45")

# dataset stem -> (run stem, evals folder, seeds, final epoch, cases, status, in-flight text or None)
STANDING = {
    "drivaerml_surface": ("r090_dr_isla", "evals_r090", S2, 500, 48, "final", None),
    "drivaerml_volume": ("r090_vol_isla", "evals_r090", S2, 500, 48, "final", None),
    "shift_suv_surface": ("r090_suv_isla", "evals_r090", S2, 500, 99, "final", None),
    "hilift_surface": ("h090_hl_isla", "evals_h090", S2, 500, 180, "final", None),
    "deepjeb_surface": ("r090_jb_isla", "evals_r090", S2, 40, 5540, "final", None),
    "tensile2d": ("r090eag_t2dz_isla", "evals_r090eag", S4, 500, 100, "final", None),
    "airfrans": ("r090eag_af_isla", "evals_r090eag", S2, 500, 200, "final", None),
    "lug": ("r090eag_lug_isla", "evals_r090eag", S2, 500, 600, "final", None),
}
# dataset stem -> (ISLA-DTT run stem, evals folder); seeds, epoch and cases as the ISLA arm's
DTT = {"airfrans": ("r090fix_af_dtt", "evals_r090fix")}
# dataset stem -> {arm: the training path of its runs}, written as isla_runs["training_path"] where the arms' paths differ
TRAINING_PATH = {
    "airfrans": {
        ISLA_ARM: "eager-trained: trained without compilation",
        DTT_ARM: "fix-trained: trained with the compiled training path after its fault was fixed",
    }
}
# dataset stem -> {arm: {"what", "replacement"}} for an arm that a running experiment will replace in a file whose status is
# "final", written as the file's "arm_in_flight" (book/standing.py: Standing.arm_in_flight)
ARM_FLIGHT = {
    "airfrans": {
        DTT_ARM: {
            "what": "ISLA-DTT trained with the compiled training path after its fault was fixed, beside ISLA trained "
            "without compilation",
            "replacement": "ISLA-DTT trained without compilation",
        }
    }
}
# dataset stem -> {protocol: why it changes no arm's input}
DEGENERATE = {
    "drivaerml_volume": {
        "area": "ISLA draws its boundary tokens by area inside the model, and GeoTransolver and Transolver read only the "
        "interior points and the signed distance, so the uniform point cloud gives every arm the same input as the native mesh"
    }
}
# keys of the _a2 file that hold the nearest-token-form ISLA or its registered outcomes, not carried over
DROP = ("raw_read_isla", "replication", "registered", "case_level")


def rows(evals, run, protocol, epoch, n_cases):
    """{sample id: metrics} of one scoring (last row per id after the last config row), with its epoch and case count checked."""
    path = f"{SCORES}/{evals}/{run}/{protocol}/{run}/metrics.jsonl"
    cfg, by = None, {}
    for line in open(path):
        r = json.loads(line)
        if r.get("phase") == "config":
            cfg, by = r, {}
        elif r.get("phase") == "infer_step":
            by[r["sample_id"]] = r["metrics"]
    assert cfg and cfg.get("epoch") == epoch, (
        f"{path}: epoch {cfg and cfg.get('epoch')} != {epoch}"
    )
    assert len(by) == n_cases, f"{path}: {len(by)} cases != {n_cases}"
    return by


def arm(evals, stem, seeds, epoch, n_cases, fields, protocols):
    """{protocol: {field: {mean, per_seed, n_cases, n_seeds}}} of one ISLA arm from its per-case rows."""
    out = {}
    for p in protocols:
        per = {s: rows(evals, f"{stem}_seed{s}", p, epoch, n_cases) for s in seeds}
        out[p] = {
            f: {
                "mean": st.mean(st.mean(m[f] for m in per[s].values()) for s in seeds),
                "per_seed": {s: st.mean(m[f] for m in per[s].values()) for s in seeds},
                "n_cases": n_cases,
                "n_seeds": len(seeds),
            }
            for f in fields
        }
    return out


def case_means(path_of, seeds, f):
    """{case: seed-mean error} from per-case rows, case ids without the dataset-index prefix."""
    by = {}
    for s in seeds:
        rows_ = {}
        for line in open(path_of(s)):
            r = json.loads(line)
            if r.get("phase") == "infer_step":
                rows_[re.sub(r"^\d+_", "", r["sample_id"])] = r["metrics"][f]
        by[s] = rows_
    ids = set.intersection(*(set(v) for v in by.values()))
    return {i: st.mean(by[s][i] for s in seeds) for i in ids}


def hilift_extras(out, a2):
    """R-HL090 as registered and the per-case readouts against the baselines, on the native mesh."""
    I, A = out["arms"][ISLA_ARM]["native"], a2["arms"][a2["isla_arm"]]["native"]
    G = {f: I[f]["mean"] / A[f]["mean"] for f in out["fields"]}
    r = {
        f: {s: I[f]["per_seed"][s] / A[f]["per_seed"][s] for s in S2}
        for f in out["fields"]
    }
    worse = any(G[f] >= 1.05 and all(v >= 1.05 for v in r[f].values()) for f in G)
    better = all(G[f] <= 0.95 for f in G)
    level = all(0.95 < G[f] < 1.05 for f in G)
    out["registered"] = {
        "R-HL090": {
            "source": "scratch/isla-hilift090-2026-10-20/PREREG.md (md5 a057d998): G = ISLA 0.9.0 seed mean / the normalized-read "
            "nearest-token arm's seed mean (results/char/hilift_surface_a2.json), r_s seed-matched; checked in the order worse "
            "(G >= 1.05 with both seeds >= 1.05 on a field), better (G <= 0.95 on both fields), level (G within 0.95-1.05 on both)",
            "G": G,
            "r": r,
            "verdict": "worse"
            if worse
            else "better"
            if better
            else "level"
            if level
            else "none",
        }
    }
    sc = f"{RES}/campaigns_2026-10-20/hilift/scores"
    CI = {
        f: case_means(
            lambda s: (
                f"{SCORES}/evals_h090/h090_hl_isla_seed{s}/native/h090_hl_isla_seed{s}/metrics.jsonl"
            ),
            S2,
            f,
        )
        for f in out["fields"]
    }
    CB = {
        b: {
            f: case_means(
                lambda s: f"{sc}/{run}_seed{s}/native/{run}_seed{s}/metrics.jsonl",
                S2,
                f,
            )
            for f in out["fields"]
        }
        for b, run in (("GeoTransolver", "udrv_hl_gt"), ("Transolver", "udrv_hl_t"))
    }
    ids = sorted(CI["pressure_l2"])
    assert len(ids) == 180 and all(
        set(CB[b][f]) == set(ids) for b in CB for f in out["fields"]
    )
    aoa = lambda c: int(re.search(r"AoA_(\d+)", c).group(1))  # noqa: E731
    cl = {"n_cases": len(ids), "per_baseline": {}}
    for b in CB:
        cl["per_baseline"][b] = {
            f: {
                "lower_on": sum(CB[b][f][i] < CI[f][i] for i in ids),
                "median_ratio": st.median(CB[b][f][i] / CI[f][i] for i in ids),
            }
            for f in out["fields"]
        }
    lead = {
        i: CI["pressure_l2"][i] - CB["GeoTransolver"]["pressure_l2"][i] for i in ids
    }
    worst = sorted(ids, key=lambda i: -CI["pressure_l2"][i])[: len(ids) // 10]
    cl["geotransolver_lead_in_isla_worst"] = {
        "field": "pressure_l2",
        "n_worst": len(worst),
        "share_of_mean_lead": sum(lead[i] for i in worst) / sum(lead.values()),
        "aoa_deg": sorted(aoa(i) for i in worst),
        "isla_mean": st.mean(CI["pressure_l2"][i] for i in worst),
        "geotransolver_mean": st.mean(
            CB["GeoTransolver"]["pressure_l2"][i] for i in worst
        ),
    }
    cl["by_aoa"] = {
        str(a): {
            "n": len(g),
            "ISLA": st.mean(CI["pressure_l2"][i] for i in g),
            **{b: st.mean(CB[b]["pressure_l2"][i] for i in g) for b in CB},
        }
        for a in sorted({aoa(i) for i in ids})
        for g in [[i for i in ids if aoa(i) == a]]
    }
    out["case_level"] = cl


def main():
    """Write results/char/<dataset>_090.json for every dataset of STANDING."""
    for ds, (stem, evals, seeds, epoch, n_cases, status, flight) in STANDING.items():
        a2 = json.load(open(f"{R}/{ds}_a2.json"))
        fields, protocols = a2["fields"], a2["protocols"]
        isla = arm(evals, stem, seeds, epoch, n_cases, fields, protocols)
        complete = a2["complete"] and all(
            c["n_seeds"] == len(seeds) for p in isla.values() for c in p.values()
        )
        arms = {ISLA_ARM: isla}
        runs = {ISLA_ARM: f"{evals}/{stem}_seed{{{','.join(seeds)}}}"}
        if ds in DTT:
            d_stem, d_evals = DTT[ds]
            arms[DTT_ARM] = arm(
                d_evals, d_stem, seeds, epoch, n_cases, fields, protocols
            )
            runs[DTT_ARM] = f"{d_evals}/{d_stem}_seed{{{','.join(seeds)}}}"
        arms.update({k: v for k, v in a2["arms"].items() if k != a2["isla_arm"]})
        for p, why in DEGENERATE.get(ds, {}).items():
            for a in arms.values():
                for f, c in a[p].items():
                    a[p][f] = {
                        "mean": None,
                        "per_seed": {},
                        "n_cases": c["n_cases"],
                        "n_seeds": 0,
                        "degenerate": why,
                        "scored": {"mean": c["mean"], "per_seed": c["per_seed"]},
                    }
        out = {
            "protocol": __doc__,
            "protocols": protocols,
            "fields": fields,
            "isla_arm": ISLA_ARM,
            "status": status,
            "in_flight": flight if status == "in_flight" else None,
            **({"arm_in_flight": ARM_FLIGHT[ds]} if ds in ARM_FLIGHT else {}),
            "arms": arms,
            "complete": complete,
            "degenerate": DEGENERATE.get(ds, {}),
            "isla_runs": {
                "scores": "results/campaigns_2026-10-22/standing_090/scores",
                "epoch": epoch,
                **runs,
                **({"training_path": TRAINING_PATH[ds]} if ds in TRAINING_PATH else {}),
            },
            **{
                k: v
                for k, v in a2.items()
                if k
                not in (
                    "protocol",
                    "protocols",
                    "fields",
                    "isla_arm",
                    "arms",
                    "complete",
                    *DROP,
                )
            },
        }
        if ds == "hilift_surface":
            hilift_extras(out, a2)
        json.dump(out, open(f"{R}/{ds}_090.json", "w"), indent=1)
        for name in (ISLA_ARM, DTT_ARM):
            if name in arms:
                print(
                    f"{ds:18s} {name:30s}",
                    " ".join(
                        f"{p}:"
                        + "/".join(
                            "n/a"
                            if arms[name][p][f]["mean"] is None
                            else f"{arms[name][p][f]['mean']:.5g}"
                            for f in fields
                        )
                        for p in protocols
                    ),
                )


if __name__ == "__main__":
    main()
