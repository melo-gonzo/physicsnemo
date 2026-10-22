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

"""The numbers of the limits chapter (20) that no single record holds: seed means of the PROBES-090 probes, the float32
ranges of the AeroZoo geometry diagnostic, and the GPU-hours of one training lane of each reference configuration.

Sources, all unmodified copies under results/campaigns_2026-10-22/ (RECORDS.md there names each original and its md5):

* probes090/results/qpop_afix/reduced.json: QUERY-POP on the AirfRANS ISLA and ISLA-DTT pairs trained on the compiled
  path after its fault was fixed (r090fix_af_*), PROBES-090 section 7;
* probes090/results/qpop/reduced.json: QUERY-POP on DrivAerML Volume (r090_vol_isla) and on the AirfRANS pairs trained
  under the fault (r090_af_*, ISLA-DTT's roundoff check only);
* probes090/results/qpop_az/reduced.json: QUERY-POP on AeroZoo `full` (azc_full_isla), every fourth test case;
* probes090/results/remesh_afix/reduced_full.json and remesh_afix_dtt/reduced_full.json: REMESH on the fix-trained
  AirfRANS pairs; probes090/results/remesh/reduced_full.json: REMESH on DrivAerML Volume;
* probes090/RESULTS.md: the STL ratios by distance to the wall (section 4a, its table), the separation of the two
  surfaces (section 4a and the erratum) and the ISLA-DTT refinement ratios against the opening angle (section 4b);
* az_diag/f32floor/README.md: the sign flips and the draw changes between float32 and float64 (section 3, its table);
* cost_090/extract.jsonl: each lane's per-epoch wall time (`train`, epoch -> [epoch_seconds, peak GiB]), its devices
  and its configured epochs.

A probe ratio is the error under the probe over the error of the same checkpoint on its native request (QUERY-POP) or
boundary (REMESH); a seed mean is the mean of the two seeds' ratios, as PROBES-090 reports them. A lane's GPU-hours are
its mean epoch wall time times its configured epochs times its devices, the mean over its seeds; a lane in flight is
priced at its configured epochs from the epochs it has logged.

Writes limits_090.json beside this script.

usage: python limits_090_2026-10-22.py
"""

import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
REC = HERE.parent
P = REC / "probes090"
FIELDS3 = ("pressure_l2", "velocity_l2", "nut_l2")
QUERY_MODES = (
    "half",
    "chunks4",
    "same",
    "far",
    "wall",
    "sub1000",
    "sub100",
    "sub10",
    "sub1",
)

#: (DATASETS key, architecture) -> the run-name prefix of the lanes priced from cost_090/extract.jsonl.
LANES = {
    ("drivaerml_surface", "ISLA"): "r090_dr_isla",
    ("drivaerml_volume", "ISLA"): "r090_vol_isla",
    ("shift_suv", "ISLA"): "r090_suv_isla",
    ("shift_suv", "GeoTransolver"): "clud_suv_gt",
    ("shift_suv", "Transolver"): "clud_suv_t",
    ("deepjeb", "ISLA"): "r090_jb_isla",
    ("hilift", "ISLA"): "h090_hl_isla",
    ("airfrans", "ISLA"): "r090fix_af_isla",
    ("airfrans", "ISLA-DTT"): "r090fix_af_dtt",
    ("lug", "ISLA"): "r090fix_lug_isla",
    ("lug", "ISLA-DTT"): "r090fix_lug_dtt",
    ("tensile2d", "ISLA"): "r090fix_t2dz_isla",
    ("aerozoo", "ISLA"): "azc_full_isla",
    ("aerozoo", "ISLA-DTT"): "azc_full_dtt",
    ("aerozoo", "GeoTransolver"): "azc_full_gt",
    ("aerozoo", "Transolver"): "azc_full_t",
    ("drivaerml_surface", "GeoTransolver"): "r80ud_dr_gt_h256",
    ("drivaerml_surface", "Transolver"): "r80ud_dr_t_h512_l16_c100k",
    ("airfrans", "GeoTransolver"): "afu_ud_gt",
    ("airfrans", "Transolver"): "afu_ud_t",
    ("lug", "GeoTransolver"): "lugbc_gt",
}


#: (DATASETS key, architecture) whose standing lane has no training log: priced from cost_090.json's median step.
SMOKE_PRICED = (("tensile2d", "GeoTransolver"), ("tensile2d", "Transolver"))


def load(path: Path):
    """A JSON record."""
    return json.loads(path.read_text())


def seed_mean(runs: dict, prefix: str, block: str, mode: str, fields) -> dict:
    """{field: mean over the prefix's seeds of runs[run][block][mode][field]}."""
    rs = [r for k, r in runs.items() if k.startswith(prefix + "_seed")]
    return {f: statistics.mean(r[block][mode][f] for r in rs) for f in fields}


def max_dev(runs: dict, prefix: str, block: str, modes) -> float:
    """The largest |r - 1| over the prefix's seeds, the modes and every field."""
    return max(
        abs(v - 1)
        for k, r in runs.items()
        if k.startswith(prefix + "_seed")
        for m in modes
        for v in r[block][m].values()
    )


def query_population() -> dict:
    """QUERY-POP seed means per mode and field, and ISLA-DTT's largest deviation from one."""
    afix = load(P / "results/qpop_afix/reduced.json")
    qpop = load(P / "results/qpop/reduced.json")
    qaz = load(P / "results/qpop_az/reduced.json")
    az_fields = tuple(qaz["azc_full_isla_seed42"]["ratio_to_native_h"]["half"])
    blk = "ratio_to_native_h"
    return {
        "airfrans_fix": {
            m: seed_mean(afix, "r090fix_af_isla", blk, m, FIELDS3) for m in QUERY_MODES
        },
        "drivaerml_volume": {
            m: seed_mean(qpop, "r090_vol_isla", blk, m, FIELDS3) for m in QUERY_MODES
        },
        "aerozoo_full": {
            m: seed_mean(qaz, "azc_full_isla", blk, m, az_fields) for m in QUERY_MODES
        },
        "aerozoo_cases": [
            qaz["azc_full_isla_seed42"]["coverage"][k]
            for k in ("n_cases", "n_cases_split")
        ],
        "dtt_max_abs_dev": max(
            max_dev(afix, "r090fix_af_dtt", blk, QUERY_MODES),
            max_dev(qpop, "r090_af_dtt", blk, QUERY_MODES),
        ),
        "n_modes": len(QUERY_MODES),
    }


def remesh() -> dict:
    """REMESH seed means per variant and field."""
    af = load(P / "results/remesh_afix/reduced_full.json")["seed_mean_ratio"]
    afd = load(P / "results/remesh_afix_dtt/reduced_full.json")["seed_mean_ratio"]
    vol = load(P / "results/remesh/reduced_full.json")
    af_runs = load(P / "results/remesh_afix/reduced_full.json")["runs"]
    d_refine = max(
        r["d"][v][f]["median"]
        for r in af_runs.values()
        for v in ("refine2", "refine4", "refine31")
        for f in r["d"][v]
    )
    return {
        "airfrans_fix_isla": af["r090fix_af_isla"],
        "airfrans_fix_dtt": afd["r090fix_af_dtt"],
        "airfrans_fix_isla_refine_d_max_median": d_refine,
        "drivaerml_volume": vol["seed_mean_ratio"]["r090_vol_isla"],
    }


def stl() -> dict:
    """The STL ratios per band of native wall distance (RESULTS.md section 4a), the median offset of the two surfaces and
    their largest separation (the erratum's)."""
    text = (P / "RESULTS.md").read_text()
    sec = text[text.index("**(a) Volume STL") : text.index("**How far apart")]
    rows = [ln for ln in sec.splitlines() if ln.startswith("|")][2:]
    bands = {}
    for ln in rows:
        c = [x.strip() for x in ln.strip("|").split("|")]
        bands[c[0]] = {
            f: [float(x) for x in c[i].split("/")] for i, f in zip((2, 3, 4), FIELDS3)
        }
    median = float(re.search(r"offset everywhere by a median ([0-9.e-]+) L", text)[1])
    largest = float(
        re.search(r"largest STL-to-CFD-wall separation is ([0-9.e-]+) L", text)[1]
    )
    return {"bands": bands, "offset_median_L": median, "separation_max_L": largest}


def theta() -> dict:
    """ISLA-DTT's pressure ratio under refine2 / refine4 / refine31 at each opening angle, and its native errors (pressure /
    velocity / eddy viscosity, seed 42) at the trained angle and at zero (RESULTS.md section 4b)."""
    text = (P / "RESULTS.md").read_text()
    sec = text[text.index("**(b) ISLA-DTT refinement") :]
    sweep = sec[sec.index("refine31:") : sec.index("Velocity and nut")]
    out = {}
    for th, nums in re.findall(
        r"θ = ([0-9.]+)[^;]*? ([0-9.]+ / [0-9.]+ / [0-9.]+)[;.]", sweep
    ):
        out.setdefault(th, [float(x) for x in nums.split(" / ")])
    native = re.search(
        r"native error rises from ([0-9. /]+) \(θ = 1\) to ([0-9. /]+) \(θ = 0\)",
        " ".join(sec.split()),
    )
    return {
        "pressure_ratio": out,
        "native_theta1": [float(x) for x in native[1].split(" / ")],
        "native_theta0": [float(x) for x in native[2].split(" / ")],
    }


def f32floor() -> dict:
    """The ranges of wall-distance sign flips and of changed drawn cells between float32 and float64 (section 3)."""
    text = (REC / "az_diag/f32floor/README.md").read_text()
    sec = text[text.index("## 3. Results") : text.index("## 4.")]
    lines = [ln for ln in sec.splitlines() if ln.startswith("|")]
    head = [c.strip() for c in lines[0].strip("|").split("|")]
    i_cells, i_flip = head.index("cells D32≠D64"), head.index("sign flips (CPU)")
    cells, flips = [], []
    for ln in lines[2:]:
        c = [x.strip() for x in ln.strip("|").split("|")]
        cells += [int(x) for x in re.findall(r"\d+", c[i_cells])]
        flips += [int(x) for x in re.findall(r"\d+", c[i_flip]) if int(x) > 0]
    return {
        "sign_flips": [min(flips), max(flips)],
        "draw_changes": [min(cells), max(cells)],
    }


def lane_gpuh() -> dict:
    """{key: {arch: {"gpuh", "epochs", "logged_epochs", "runs"}}} from the per-epoch wall times of each lane."""
    rows = [
        json.loads(ln)
        for ln in (REC / "cost_090/extract.jsonl").read_text().splitlines()
    ]
    out = {}
    for (key, arch), prefix in LANES.items():
        rs = [r for r in rows if r["run"].startswith(prefix + "_seed")]
        per = []
        for r in rs:
            ep = [v[0] for v in r["train"].values()]
            planned = int(r["config_params"]["training.num_epochs"])
            per.append(
                (
                    statistics.mean(ep) * planned * len(r["devices"]) / 3600,
                    planned,
                    len(ep),
                )
            )
        out.setdefault(key, {})[arch] = {
            "gpuh": statistics.mean(p[0] for p in per),
            "epochs": per[0][1],
            "logged_epochs": min(p[2] for p in per),
            "steps_per_epoch": statistics.mean(
                r["step"]["n"] / len(r["train"]) for r in rs
            ),
            "runs": sorted(r["run"] for r in rs),
        }
    # Lanes with no training log left: the median step of results/char/cost_090.json (its smoke-lane price) times the ISLA
    # lane's steps per epoch on the same dataset, its epochs and its devices.
    cost = json.loads((REC.parent / "char" / "cost_090.json").read_text())["cost"]
    for key, arch in SMOKE_PRICED:
        isla = out[key]["ISLA"]
        c = cost[key][arch]
        out[key][arch] = {
            "gpuh": c["step_seconds"]
            * isla["steps_per_epoch"]
            * isla["epochs"]
            * c["gpus"]
            / 3600,
            "epochs": isla["epochs"],
            "priced_from": "results/char/cost_090.json step_seconds x the ISLA lane's steps per epoch",
        }
    return out


def main():
    """Write limits_090.json."""
    out = {
        "sources": {
            "query_population": "probes090/results/{qpop_afix,qpop,qpop_az}/reduced.json",
            "remesh": "probes090/results/{remesh_afix,remesh_afix_dtt,remesh}/reduced_full.json",
            "stl": "probes090/RESULTS.md (section 4a and the erratum)",
            "theta": "probes090/RESULTS.md (section 4b)",
            "f32floor": "az_diag/f32floor/README.md (section 3)",
            "lane_gpuh": "cost_090/extract.jsonl",
        },
        "query_population": query_population(),
        "remesh": remesh(),
        "stl": stl(),
        "theta": theta(),
        "f32floor": f32floor(),
        "lane_gpuh": lane_gpuh(),
    }
    (HERE / "limits_090.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
