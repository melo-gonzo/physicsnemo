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

"""AeroZoo standing under the compliant protocol (2026-10-22): ISLA 0.9.0 at its reference configuration on the seven splits,
GeoTransolver 256 on the seven splits, Transolver 512 x 16 on `full`; test side, final checkpoint (epoch 250), float32, 200,000
interior points, seeds 42 and 43 (scratch/isla-az-compliant-2026-10-19/PREREG_v2.md, sections 7 and 11; reduction as
scratch/isla-az-decide-2026-10-21/reduce_azdecide.py). The relative L2 is the recipe's physical one per field and case, so
predicting zero scores 1.0 on every field and case.

Scores: results/campaigns_2026-10-22/aerozoo/scores/p200k_fp32_test_azc/<run>/<run>/metrics.jsonl (with .done and
SNAPSHOT_USED), copied unmodified from scratch/isla-az-decide-2026-10-21/scores/ (originals on LHR, $T/p200k_fp32_test_azc/).
Per split, arm and field: the seed mean of the per-seed means over cases (the rows after the last config row, checked equal to
the infer_summary row), the per-seed values, the median over cases of the per-case seed mean, and the case count. Checks: epoch
250, the test side, the snapshot (code_r090 for ISLA, code_azv2 for the baselines), the manifest's test-side case count.

ISLA-DTT (query_independent=True, the same configuration) is an in-flight arm with no numbers. "diagnostics" holds the
machine-readable readouts of the AZ-DECIDE diagnostics: the velocity speed ratio and direction on profile_crossflow and nonbox
(preddiag/predictions_diag.tsv: medians over the 30 diagnosed cases per run), and GeoTransolver's velocity loss over its last
epochs (basecheck/raw/curves/*.json: the training loss/velocity with its weight of 3 divided out, and the validation
velocity_l2), both copied into results/campaigns_2026-10-22/aerozoo/.
usage: python aerozoo_090_2026-10-22.py  ->  results/char/aerozoo_090.json"""

import csv
import json
import os
import statistics as st

R = os.path.dirname(os.path.abspath(__file__))
SRC = f"{os.path.dirname(R)}/campaigns_2026-10-22/aerozoo"
SPLITS = (
    "full",
    "nonbox",
    "reynolds",
    "blobs",
    "bank_zero_to_cad",
    "profile_crossflow",
    "profile_dual_inlet",
)
FIELDS = {
    "velocity_l2": "velocity",
    "pressure_l2": "pressure",
    "nut_l2": "eddy viscosity",
    "tke_l2": "turbulent kinetic energy",
    "omega_l2": "specific dissipation rate",
}
SEEDS = ("42", "43")
N_CASES = {
    "full": 716,
    "nonbox": 1074,
    "reynolds": 1385,
    "blobs": 1367,
    "bank_zero_to_cad": 333,
    "profile_crossflow": 578,
    "profile_dual_inlet": 516,
}
# arm label -> (run tag, splits, snapshot, note)
ARMS = {
    "ISLA 512 with 512 slices": ("isla", SPLITS, "code_r090", ""),
    "GeoTransolver 256": (
        "gt",
        SPLITS,
        "code_azv2",
        "about a zero predictor on pressure: its output row for pressure (and for eddy viscosity, k and omega) is "
        "about zero at the final epoch, so its median per-case pressure error is about 1.0, the zero predictor's; "
        "its velocity is learned in part",
    ),
    "Transolver 512 x 16": (
        "t",
        ("full",),
        "code_azv2",
        "about a zero predictor on every field: its predictions are constants, about zero, on every case",
    ),
}
DTT = {
    "label": "ISLA-DTT 512 with 512 slices",
    "status": "in_flight",
    "in_flight": {
        "what": "interacting ISLA, GeoTransolver and Transolver, without ISLA-DTT",
        "replacement": "ISLA-DTT on the same seven splits",
    },
}


def score(run, snap):
    """{case: metrics} of one scoring after its last config row, with its checks."""
    d = f"{SRC}/scores/p200k_fp32_test_azc/{run}"
    cfg, cases, summ = None, {}, None
    for line in open(f"{d}/{run}/metrics.jsonl"):
        r = json.loads(line)
        if r.get("phase") == "config":
            cfg, cases, summ = r, {}, None
        elif r.get("phase") == "infer_step":
            cases[r["sample_id"].split("_", 1)[1]] = r["metrics"]
        elif r.get("phase") == "infer_summary":
            summ = r
    split = run[4:].rsplit("_", 2)[0]
    assert cfg["epoch"] == 250 and cfg["infer_split"] == "test", run
    assert open(f"{d}/SNAPSHOT_USED").read().strip().endswith(snap), run
    assert not open(f"{d}/.done").read().strip(), run
    assert len(cases) == N_CASES[split] == summ["num_samples"], run
    for f in FIELDS:
        m = st.fmean(c[f] for c in cases.values())
        assert abs(m / summ["metrics"][f] - 1) < 1e-6, (run, f)
    return cases


def diagnostics():
    """Medians of the velocity diagnostics and GeoTransolver's velocity loss over its last epochs."""
    rows = list(
        csv.DictReader(open(f"{SRC}/preddiag/predictions_diag.tsv"), delimiter="\t")
    )
    cols = {
        "speed_ratio_rms": "RMS predicted speed / RMS true speed",
        "cos_global": "global cosine of the predicted and true velocity fields",
        "angle_g_pred_mean": "angle between the inlet drive direction and the mean predicted velocity [deg]",
        "angle_true_pred_means": "angle between the mean true and the mean predicted velocity [deg]",
        "l2_speed_matched": "velocity relative L2 after rescaling the prediction to the true RMS speed",
    }
    vel = {"columns": cols, "n_cases": {}, "runs": {}}
    for run in sorted({r["run"] for r in rows}):
        sel = [r for r in rows if r["run"] == run]
        vel["n_cases"][run] = len(sel)
        vel["runs"][run] = {c: st.median(float(r[c]) for r in sel) for c in cols}
    gt = {}
    for sp in SPLITS:
        for s in SEEDS:
            run = f"azc_{sp}_gt_seed{s}"
            d = json.load(open(f"{SRC}/basecheck_curves/{run}.json"))
            tr = {int(e): v["loss/velocity"] / 3.0 for e, v in d["train"].items()}
            last = max(tr)
            gt[run] = {
                "train_velocity": {str(e): tr[e] for e in (200, last - 10, last)},
                "train_velocity_slope_last_10_epochs": (tr[last] - tr[last - 10]) / 10,
                "train_velocity_slope_200_to_last": (tr[last] - tr[200]) / (last - 200),
                "val_velocity_l2": {
                    e: d["val"][e]["velocity_l2"] for e in ("240", str(last))
                },
            }
    return {
        "velocity_direction_and_speed": vel,
        "geotransolver_velocity_last_epochs": gt,
    }


def main():
    """Write results/char/aerozoo_090.json."""
    arms = {}
    for label, (tag, splits, snap, note) in ARMS.items():
        a = {"note": note, "splits": {}}
        for sp in splits:
            per = {s: score(f"azc_{sp}_{tag}_seed{s}", snap) for s in SEEDS}
            ids = set(per["42"]) & set(per["43"])
            assert len(ids) == N_CASES[sp]
            a["splits"][sp] = {
                f: {
                    "mean": st.fmean(
                        st.fmean(c[f] for c in per[s].values()) for s in SEEDS
                    ),
                    "per_seed": {
                        s: st.fmean(c[f] for c in per[s].values()) for s in SEEDS
                    },
                    "median": st.median(
                        st.fmean(per[s][i][f] for s in SEEDS) for i in ids
                    ),
                    "n_cases": len(ids),
                    "n_seeds": len(SEEDS),
                }
                for f in FIELDS
            }
        arms[label] = a
    arms[DTT["label"]] = {
        "note": "",
        "status": DTT["status"],
        "in_flight": DTT["in_flight"],
        "splits": {},
    }
    out = {
        "protocol": __doc__,
        "status": "in_flight",
        "in_flight": DTT["in_flight"],
        "splits": list(SPLITS),
        "held_out": list(SPLITS[1:]),
        "fields": FIELDS,
        "n_cases": N_CASES,
        "zero_predictor": 1.0,
        "isla_arm": "ISLA 512 with 512 slices",
        "arms": arms,
        "diagnostics": diagnostics(),
    }
    json.dump(out, open(f"{R}/aerozoo_090.json", "w"), indent=1)
    for label, a in arms.items():
        for sp, c in a["splits"].items():
            print(
                f"{label:28s} {sp:20s}",
                " ".join(
                    f"{f}:{c[f]['mean']:.4g}[{c[f]['median']:.4g}]" for f in FIELDS
                ),
            )


if __name__ == "__main__":
    main()
