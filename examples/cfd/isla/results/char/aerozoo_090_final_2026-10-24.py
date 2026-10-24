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

"""AeroZoo standing under the compliant protocol with its ISLA-DTT arm (2026-10-24): results/char/aerozoo_090.json, whose
ISLA, GeoTransolver and Transolver arms and diagnostics are kept as they are, plus ISLA-DTT 0.9.0 (query_independent=True,
the same configuration as ISLA) on the seven splits, scored as the other arms: test side, final checkpoint (epoch 250),
float32, 200,000 interior points, seeds 42 and 43 (scratch/isla-az-compliant-2026-10-19/PREREG_v2.md, sections 7, 11 and
12(f); the read is scratch/isla-az-decide-2026-10-21/RESULTS.md, copied to results/campaigns_2026-10-23/az_decide_final/).

Scores: results/campaigns_2026-10-24/aerozoo_dtt/scores/p200k_fp32_test_azc/<run>/<run>/metrics.jsonl (the .done and
SNAPSHOT_USED markers in markers.json beside them), copied unmodified from scratch/isla-az-decide-2026-10-21/scores/
(originals on LHR, $T/p200k_fp32_test_azc/). Per split and field: the seed mean of the per-seed means over cases, the
per-seed values, the median over cases of the per-case seed mean, and the case count, with the same checks as
aerozoo_090_2026-10-22.py (epoch 250, the test side, snapshot code_r090, an empty .done, the manifest's case count, the
mean over cases equal to the infer_summary row).
usage: python aerozoo_090_final_2026-10-24.py  ->  results/char/aerozoo_090_final.json"""

import json
import os
import statistics as st

R = os.path.dirname(os.path.abspath(__file__))
SRC = (
    f"{os.path.dirname(R)}/campaigns_2026-10-24/aerozoo_dtt/scores/p200k_fp32_test_azc"
)
SEEDS = ("42", "43")
DTT = "ISLA-DTT 512 with 512 slices"
MARKERS = json.load(open(f"{SRC}/markers.json"))["runs"]


def score(run, n_cases, fields):
    """{case: metrics} of one scoring after its last config row, with its checks."""
    cfg, cases, summ = None, {}, None
    for line in open(f"{SRC}/{run}/{run}/metrics.jsonl"):
        r = json.loads(line)
        if r.get("phase") == "config":
            cfg, cases, summ = r, {}, None
        elif r.get("phase") == "infer_step":
            cases[r["sample_id"].split("_", 1)[1]] = r["metrics"]
        elif r.get("phase") == "infer_summary":
            summ = r
    assert cfg["epoch"] == 250 and cfg["infer_split"] == "test", run
    assert MARKERS[run]["SNAPSHOT_USED"].endswith("code_r090"), run
    assert not MARKERS[run][".done"].strip(), run
    assert len(cases) == n_cases == summ["num_samples"], run
    for f in fields:
        m = st.fmean(c[f] for c in cases.values())
        assert abs(m / summ["metrics"][f] - 1) < 1e-6, (run, f)
    return cases


def main():
    """Write results/char/aerozoo_090_final.json."""
    out = json.load(open(f"{R}/aerozoo_090.json"))
    arm = {"note": "", "splits": {}}
    for sp in out["splits"]:
        per = {
            s: score(f"azc_{sp}_dtt_seed{s}", out["n_cases"][sp], out["fields"])
            for s in SEEDS
        }
        ids = set(per["42"]) & set(per["43"])
        assert len(ids) == out["n_cases"][sp]
        arm["splits"][sp] = {
            f: {
                "mean": st.fmean(
                    st.fmean(c[f] for c in per[s].values()) for s in SEEDS
                ),
                "per_seed": {s: st.fmean(c[f] for c in per[s].values()) for s in SEEDS},
                "median": st.median(st.fmean(per[s][i][f] for s in SEEDS) for i in ids),
                "n_cases": len(ids),
                "n_seeds": len(SEEDS),
            }
            for f in out["fields"]
        }
    out["arms"][DTT] = arm
    out["protocol"] = __doc__ + "\n\nThe other arms:\n" + out["protocol"]
    out["status"] = "final"
    out.pop("in_flight")
    json.dump(out, open(f"{R}/aerozoo_090_final.json", "w"), indent=1)
    for sp, c in arm["splits"].items():
        print(
            f"{DTT:28s} {sp:20s}",
            " ".join(
                f"{f}:{c[f]['mean']:.4g}[{c[f]['median']:.4g}]" for f in out["fields"]
            ),
        )


if __name__ == "__main__":
    main()
