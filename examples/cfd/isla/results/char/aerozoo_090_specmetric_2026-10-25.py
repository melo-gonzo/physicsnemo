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

"""AeroZoo standing under AeroZoo's own interior metrics (2026-10-25): the 48 checkpoints of
results/char/aerozoo_090_final.json, E1-AZ and LRE-AZ rescored by AZ-RESCORE with the release's reference functions.

eps_U and eps_p are the root-mean-square velocity and pressure errors over every interior cell, each cell weighted by its
volume, in units of the case's reference speed and dynamic pressure; eps_U_near and eps_p_near are the same over the cells
within one reference length of a body face; computed in float64 from chunked full-mesh inference in float32 (aerozoo
docs/book/06-splits.qmd @ 96bd2b62, lines 282-298 and 429-472). Per split and case set ("all" test cases, or the
"stable" ones, AeroZoo's mean_stable): ``mean`` is the seed mean of the per-seed mean over cases, ``median`` the seed
mean of the per-seed median over cases (AeroZoo's convention), ``median_of_case_seed_mean`` the median over cases of the
per-case seed mean (the convention of aerozoo_090_final.json). ``rel_*`` is the full-mesh unweighted relative L2 in
float64, and ``reg_rel_*`` the registered 200,000-cell float32 relative L2 of aerozoo_090_final.json.

Source: results/campaigns_2026-10-23/az_rescore/AEROZOO_SPECMETRIC.json (an unmodified copy of
scratch/isla-az-rescore-2026-10-24/AEROZOO_SPECMETRIC.json, md5 c8fb1c82; LHR jobs 376802, 376803, 376804, 376818,
378817, 378820). This script renames the arms to the book's names, puts the splits in the standing's order, adds the
release's identity from the lane's record, and checks the file against the registered standing: the same case counts,
the registered relative L2 equal to aerozoo_090_final.json's case means, two seeds everywhere, and the registered readings
of E1-AZ, LRE-AZ and section 12(f) recomputed from the arms equal to READINGS.md to its printed digits.
usage: python aerozoo_090_specmetric_2026-10-25.py  ->  results/char/aerozoo_090_specmetric.json"""

import json
import math
import os
import re

R = os.path.dirname(os.path.abspath(__file__))
REC = f"{os.path.dirname(R)}/campaigns_2026-10-23/az_rescore"
ARMS = {
    "ISLA": "ISLA (interacting)",
    "ISLA-DTT": "ISLA-DTT",
    "GeoTransolver": "GeoTransolver",
    "Transolver": "Transolver",
    "E1-AZ": "E1-AZ HEADNORM",
    "LRE-AZ": "LRE-AZ B",
}
FIELDS = {
    "eps_U": "$\\varepsilon_U$",
    "eps_p": "$\\varepsilon_p$",
    "eps_U_near": "$\\varepsilon^N_U$",
    "eps_p_near": "$\\varepsilon^N_p$",
    "eps_nut": "$\\varepsilon_{\\nu_t}$",
    "eps_k": "$\\varepsilon_k$",
    "eps_omega": "$\\varepsilon_\\omega$",
    "rel_velocity": "relative L2, velocity, full mesh",
    "rel_pressure": "relative L2, pressure, full mesh",
    "rel_nut": "relative L2, eddy viscosity, full mesh",
    "rel_tke": "relative L2, turbulent kinetic energy, full mesh",
    "rel_omega": "relative L2, specific dissipation rate, full mesh",
    "reg_rel_velocity": "relative L2, velocity, registered",
    "reg_rel_pressure": "relative L2, pressure, registered",
}
REG = {"reg_rel_velocity": "velocity_l2", "reg_rel_pressure": "pressure_l2"}


def ratio(a, b, f, sp, cs="all", stat="mean"):
    """Seed-mean ratio a ÷ b and its seed-matched ratios, of case means (or of case medians)."""
    x, y = a["splits"][sp][f][cs], b["splits"][sp][f][cs]
    k = "per_seed" if stat == "mean" else "per_seed_median"
    return x[stat] / y[stat], {s: x[k][s] / y[k][s] for s in ("42", "43")}


def readings_md():
    """{(section, metric, statistic): (velocity ratio, pressure ratio)} of READINGS.md's E1-AZ table."""
    out = {}
    for line in open(f"{REC}/READINGS.md"):
        m = re.match(
            r"\| (registered relL2 \(200k, float32\)|spec eps_U / eps_p|spec eps\^N_U / eps\^N_p) \| (.+?) \| ([\d.]+) \(.*?\) \| ([\d.]+) \(",
            line,
        )
        if m:
            out[(m.group(1), m.group(2))] = (float(m.group(3)), float(m.group(4)))
    return out


def main():
    """Write results/char/aerozoo_090_specmetric.json."""
    src = json.load(open(f"{REC}/AEROZOO_SPECMETRIC.json"))
    std = json.load(open(f"{R}/aerozoo_090_final.json"))
    sha = re.search(
        r"`index\.json` sha256\s+([0-9a-f]{64})",
        " ".join(open(f"{REC}/NOTEBOOK_DRAFT.md").read().split()),
    )
    sha = sha.group(1)
    assert (
        sha.startswith("c8e9508e")
        and "c8e9508e" in open(f"{REC}/../az_spec_audit/AUDIT.md").read()
    )
    arms = {
        k: {
            "name": v,
            "splits": {
                sp: src["arms"][v]["splits"][sp]
                for sp in std["splits"]
                if sp in src["arms"][v]["splits"]
            },
        }
        for k, v in ARMS.items()
    }
    std_arm = {n.split()[0]: a for n, a in std["arms"].items()}
    for k, a in arms.items():
        for sp, c in a["splits"].items():
            for f in FIELDS:
                for cs in ("all", "stable"):
                    assert c[f][cs]["n_seeds"] == 2, (k, sp, f, cs)
            assert c["eps_U"]["all"]["n_cases"] == std["n_cases"][sp], (k, sp)
            if k in std_arm:
                for f, g in REG.items():
                    assert (
                        abs(
                            c[f]["all"]["mean"] / std_arm[k]["splits"][sp][g]["mean"]
                            - 1
                        )
                        < 1e-9
                    ), (k, sp, f)
    # E1-AZ: HEADNORM ÷ CONTROL (interacting ISLA on family_non_box), the rule's form and beside it
    e1 = {}
    for lab, (fv, fp) in {
        "registered relL2 (200k, float32)": ("reg_rel_velocity", "reg_rel_pressure"),
        "spec eps_U / eps_p": ("eps_U", "eps_p"),
        "spec eps^N_U / eps^N_p": ("eps_U_near", "eps_p_near"),
    }.items():
        for stat_lab, cs, stat in (
            ("all mean (the rule’s form)", "all", "mean"),
            ("all median", "all", "median"),
            ("mean_stable mean", "stable", "mean"),
            ("mean_stable median", "stable", "median"),
        ):
            v = ratio(arms["E1-AZ"], arms["ISLA"], fv, "nonbox", cs, stat)
            p = ratio(arms["E1-AZ"], arms["ISLA"], fp, "nonbox", cs, stat)
            fires = any(r >= 1.10 and min(s.values()) >= 1.10 for r, s in (v, p))
            g14 = (0.7814 * v[0] * p[0]) ** (1 / 14)
            e1[f"{lab} | {stat_lab}"] = {
                "velocity": v[0],
                "velocity_per_seed": v[1],
                "pressure": p[0],
                "pressure_per_seed": p[1],
                "branch1_fires": fires,
                "G14": g14,
            }
    md = readings_md()
    assert len(md) == 12, md
    for (lab, stat_lab), (rv, rp) in md.items():
        x = e1[f"{lab} | {stat_lab}"]
        assert (
            f"{x['velocity']:.3f}" == f"{rv:.3f}"
            and f"{x['pressure']:.3f}" == f"{rp:.3f}"
        ), (lab, stat_lab, x)
    # LRE-AZ: B ÷ A on reynolds (A = interacting ISLA), case means; D and 2 SE as registered (R1)
    lre = {}
    for f in (
        "eps_U",
        "eps_p",
        "eps_U_near",
        "eps_p_near",
        "reg_rel_velocity",
        "reg_rel_pressure",
    ):
        b, a = (
            arms["LRE-AZ"]["splits"]["reynolds"][f]["all"],
            arms["ISLA"]["splits"]["reynolds"][f]["all"],
        )
        lb = [math.log(b["per_seed"][s]) for s in ("42", "43")]
        la = [math.log(a["per_seed"][s]) for s in ("42", "43")]
        sd = lambda xs: abs(xs[0] - xs[1]) / math.sqrt(2)
        d = math.log(b["mean"] / a["mean"])
        se2 = 2 * math.sqrt((sd(la) ** 2 + sd(lb) ** 2) / 2)
        label = (
            "harms"
            if d > math.log(1.10) and abs(d) > se2
            else "helps"
            if d < math.log(0.90) and abs(d) > se2
            else "no change"
        )
        lre[f] = {
            "ratio": math.exp(d),
            "D": d,
            "two_se": se2,
            "label": label,
            "per_seed": {s: b["per_seed"][s] / a["per_seed"][s] for s in ("42", "43")},
        }
    LT = open(f"{REC}/READINGS.md").read()
    for f, pat in (
        (
            "eps_U",
            r"spec eps \(case means\) \| velocity \| ([\d.]+) \|.*?\| (\w[\w ]*) \|",
        ),
        (
            "eps_p",
            r"spec eps \(case means\) \| pressure \| ([\d.]+) \|.*?\| (\w[\w ]*) \|",
        ),
        (
            "reg_rel_velocity",
            r"registered relL2 \| velocity \| ([\d.]+) \|.*?\| (\w[\w ]*) \|",
        ),
    ):
        m = re.search(pat, LT)
        assert f"{lre[f]['ratio']:.3f}" == m.group(1) and lre[f]["label"] == m.group(
            2
        ), (f, lre[f], m.groups())
    out = {
        "protocol": __doc__ + "\n\nThe lane's record:\n" + src["protocol"],
        "status": "final",
        "release": {"version": "v1.2", "index_sha256": sha},
        "splits": std["splits"],
        "held_out": std["held_out"],
        "fields": FIELDS,
        "n_cases": {
            sp: {
                cs: arms["ISLA"]["splits"][sp]["eps_U"][cs]["n_cases"]
                for cs in ("all", "stable")
            }
            for sp in std["splits"]
        },
        "arms": arms,
        "references": {
            r: {sp: src["references"][r][sp] for sp in std["splits"]}
            for r in ("zero", "freestream")
        },
        "readings": {"e1_az": e1, "lre_az": lre},
    }
    json.dump(out, open(f"{R}/aerozoo_090_specmetric.json", "w"), indent=1)
    for k in ("ISLA", "ISLA-DTT", "GeoTransolver"):
        print(
            k,
            " ".join(
                f"{sp}:{arms[k]['splits'][sp]['eps_U']['all']['mean']:.3g}/{arms[k]['splits'][sp]['eps_p']['all']['mean']:.3g}"
                for sp in std["splits"]
            ),
        )
    print(
        "E1-AZ eps case means",
        {
            x: round(e1["spec eps_U / eps_p | all mean (the rule’s form)"][x], 3)
            for x in ("velocity", "pressure")
        },
    )
    print("LRE-AZ eps_U", round(lre["eps_U"]["ratio"], 3), lre["eps_U"]["label"])


if __name__ == "__main__":
    main()
