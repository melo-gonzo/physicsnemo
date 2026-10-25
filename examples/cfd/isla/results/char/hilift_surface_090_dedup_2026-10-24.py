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

"""HiLiftAeroML standing on the deduplicated wing (HL-DUP stage 2, 2026-10-24): ISLA 0.9.0 at its reference
configuration (width 512, 12 layers, 512 slices, 3.75e-4, 200,000 cells), GeoTransolver 256 (200,000 cells) and
Transolver 512 x 16 (100,000 cells), both baselines reading the unit freestream direction U_inf_dir, each trained 500
epochs on the 510-case rung with the recipe option drop_duplicate_cells (the stored wing's extra copies of its root cells
dropped at load, the dataset unchanged), two seeds (42, 43), and scored in float32 on the final checkpoint on the 180
validation cases: 200,000 cells drawn per case and the extra copies then dropped (about 186,000 kept), so the metric runs
over distinct cells. Native mesh and rotated + translated; the uniform point cloud was not scored on this data.

Inputs, all copied unmodified into results/campaigns_2026-10-23/hl_dup_stage2_scores/ (RECORDS.md lists their md5s):
  scores_stage2/<run>/<protocol>/<run>/metrics.jsonl   the scorer's per-case rows of the six runs hldupx_hl_{isla,gt,t}_seed{42,43}
  score_summary.json                                   stage 1's scores of the stored-wing standing checkpoints: (a) as the
                                                       standing scored them, (b) the same predictions with the copies left out
                                                       of the metric, (c) the copies left out of the input too; per-case pressure
and results/char/hilift_surface_090.json (immutable) for the unit-drive control below.

Reduction as every standing: per seed and protocol the mean over cases of the per-case relative L2 (the last infer_step row
per sample id after the last config row), then the mean over seeds. Each config row must name epoch 500 and the dataset
highlift_surface_medium_dedup (its _posed variant for the posed scoring), and each protocol must hold 180 cases.

Writes results/char/hilift_surface_090_dedup.json in the standing schema of book/standing.py (protocols, fields, isla_arm,
status, in_flight, arms[arm][protocol][field] = {mean, per_seed, n_cases, n_seeds}, degenerate, complete) plus:
  isla_runs           the score folder, epoch and run stems
  fair_input          what each baseline reads
  stored_wing_b       stage 1's (b): each arm's stored-wing checkpoints scored over the same distinct cells, the like-for-like
                      figure the old standing gives; and (a), the old standing's own metric, for the record
  registered          the rule of scratch/isla-hl-dup-2026-10-24/PREREG.md (md5 072b6b1c), as registered: G against the
                      registered (b) references, per cell and seed, and its verdict
  case_level          per-case readouts of the native scores (the schema of hilift_surface_090.json's case_level), plus the
                      angle-of-attack bands of the registration (4-12 and 14-22 degrees): seed means, seed-matched ratios
                      and the cases on which ISLA's seed-mean pressure is lower, and the same count under (b)
  unit_drive_control  the input-scale control on the stored wing, from hilift_surface_090.json: the unit-direction baselines,
                      the same configurations reading the physical vector U_inf (raw_drive_<arch>), and the standing's
                      stored-wing ISLA, native and uniform point cloud; no deduplicated physical-vector arm exists

usage: python hilift_surface_090_dedup_2026-10-24.py
"""

import json
import re
import statistics as st
from pathlib import Path

R = Path(__file__).resolve().parent
SRC = R.parent / "campaigns_2026-10-23" / "hl_dup_stage2_scores"
OLD = json.load(open(R / "hilift_surface_090.json"))
S1 = json.load(open(SRC / "score_summary.json"))

SEEDS = ("42", "43")
EPOCH, N_CASES, DATASET = 500, 180, "highlift_surface_medium_dedup"
FIELDS = {"pressure_l2": "pressure", "tau_wall_l2": "wall shear"}
PROTOS = {"native": "native mesh", "posed": "rotated + translated"}
ISLA_ARM = "ISLA 512 with 512 slices"
GT_ARM, T_ARM = "GeoTransolver 256", "Transolver 512 x 16, trained at 100k cells"
RUN = {ISLA_ARM: "hldupx_hl_isla", GT_ARM: "hldupx_hl_gt", T_ARM: "hldupx_hl_t"}
ARCH = {ISLA_ARM: "ISLA", GT_ARM: "GeoTransolver", T_ARM: "Transolver"}
#: the registered (b) references of the rule (PREREG.md, "Rule"): stage 1's seed-mean baseline / ISLA under (b), to 3 decimals
REF = {
    ("GeoTransolver", "pressure_l2"): 0.917,
    ("GeoTransolver", "tau_wall_l2"): 0.922,
    ("Transolver", "pressure_l2"): 0.924,
    ("Transolver", "tau_wall_l2"): 0.909,
}
BAND = 0.03  # the rule's band on G - reference
AOA_BANDS = {"4-12": (4, 12), "14-22": (14, 22)}


def rows(run, protocol):
    """{case id: metrics} of one scoring (last row per id after the last config row), with its epoch, dataset and case count
    checked; case ids without the dataset-index prefix and the domain suffix."""
    path = SRC / "scores_stage2" / run / protocol / run / "metrics.jsonl"
    cfg, by = None, {}
    for line in open(path):
        r = json.loads(line)
        if r.get("phase") == "config":
            cfg, by = r, {}
        elif r.get("phase") == "infer_step":
            by[r["sample_id"].split("_", 1)[1].rsplit("_domain", 1)[0]] = r["metrics"]
    want = DATASET + ("_posed" if protocol == "posed" else "")
    assert cfg and cfg["epoch"] == EPOCH and cfg["dataset"] == want, (path, cfg)
    assert len(by) == N_CASES, (path, len(by))
    return by


PER = {
    (a, s, p): rows(f"{RUN[a]}_seed{s}", p) for a in RUN for s in SEEDS for p in PROTOS
}


def cell(a, p, f):
    """{mean, per_seed, n_cases, n_seeds} of one arm on one field under one protocol."""
    per = {s: st.mean(m[f] for m in PER[(a, s, p)].values()) for s in SEEDS}
    return {
        "mean": st.mean(per.values()),
        "per_seed": per,
        "n_cases": N_CASES,
        "n_seeds": len(SEEDS),
    }


arms = {a: {p: {f: cell(a, p, f) for f in FIELDS} for p in PROTOS} for a in RUN}
out = {
    "protocol": __doc__,
    "protocols": PROTOS,
    "fields": FIELDS,
    "isla_arm": ISLA_ARM,
    "status": "final",
    "in_flight": None,
    "arms": arms,
    "complete": all(
        c["n_seeds"] == 2 for a in arms.values() for p in a.values() for c in p.values()
    ),
    "degenerate": {},
    "isla_runs": {
        "scores": "results/campaigns_2026-10-23/hl_dup_stage2_scores/scores_stage2",
        "epoch": EPOCH,
        "dataset": DATASET,
        **{a: f"{RUN[a]}_seed{{42,43}}" for a in RUN},
    },
    "fair_input": {
        GT_ARM: "reads U_inf_dir as its global embedding (global_embedding: global_data.U_inf_dir in the saved resolved config)",
        T_ARM: "reads U_inf_dir (fx.source: global_data.U_inf_dir in the saved resolved config), trained at 100,000 cells and "
        "scored on ISLA's draw",
    },
}

# --- stage 1's (a) and (b) of the stored-wing checkpoints --------------------------------------------------------------
out["stored_wing_b"] = {
    "what": "the stored-wing standing's checkpoints (hilift_surface_090.json) scored by stage 1 on the same draw: (b) leaves the "
    "copies out of the metric only, so it runs over the same distinct cells as this file's arms; (a) is the old standing's "
    "own metric, with the copies",
    **{
        ARCH[a]: {
            m: {
                f: {
                    "mean": st.mean(
                        S1["means"][f"{ARCH[a]}_seed{s}"][m][f] for s in SEEDS
                    ),
                    "per_seed": {
                        s: S1["means"][f"{ARCH[a]}_seed{s}"][m][f] for s in SEEDS
                    },
                }
                for f in FIELDS
            }
            for m in ("a", "b")
        }
        for a in RUN
    },
}
B = out["stored_wing_b"]

# --- the registered rule -------------------------------------------------------------------------------------------------
cells = {}
for (b, f), ref in REF.items():
    bb = [a for a in RUN if ARCH[a] == b][0]
    G = arms[bb]["native"][f]["mean"] / arms[ISLA_ARM]["native"][f]["mean"]
    r = {
        s: arms[bb]["native"][f]["per_seed"][s]
        / arms[ISLA_ARM]["native"][f]["per_seed"][s]
        for s in SEEDS
    }
    b_ratio = B[b]["b"][f]["mean"] / B["ISLA"]["b"][f]["mean"]
    assert round(b_ratio, 3) == ref, (b, f, b_ratio, ref)
    cells[f"{b}/{f}"] = {
        "reference": ref,
        "G": G,
        "change": G - ref,
        "per_seed": r,
        "narrows": G - ref >= BAND and all(v > ref for v in r.values()),
        "widens": G - ref <= -BAND and all(v < ref for v in r.values()),
        "within_band": abs(G - ref) <= BAND,
    }
verdict = (
    "gap narrows"
    if all(c["narrows"] for c in cells.values())
    else "gap widens"
    if all(c["widens"] for c in cells.values())
    else "unchanged"
    if all(c["within_band"] for c in cells.values())
    else "mixed"
)
out["registered"] = {
    "HL-DUP stage 2": {
        "source": "scratch/isla-hl-dup-2026-10-24/PREREG.md (md5 072b6b1c), 'Rule': G = seed-mean baseline / seed-mean ISLA, both "
        "trained on the deduplicated wing, native, pressure and wall shear, against the stage-1 (b) ratio; narrows: every cell "
        "rises by at least 0.03 with both seed ratios above the reference; widens: every cell falls by at least 0.03 with both "
        "below; unchanged: every cell within 0.03; else mixed, reported per cell",
        "cells": cells,
        "verdict": verdict,
    }
}

# --- case level ----------------------------------------------------------------------------------------------------------
aoa = lambda case: int(re.search(r"AoA_(\d+)", case).group(1))  # noqa: E731
seed_case = {
    (ARCH[a], s, f): {c: m[f] for c, m in PER[(a, s, "native")].items()}
    for a in RUN
    for s in SEEDS
    for f in FIELDS
}
CM = {
    (ARCH[a], f): {
        c: st.mean(seed_case[(ARCH[a], s, f)][c] for s in SEEDS)
        for c in PER[(a, SEEDS[0], "native")]
    }
    for a in RUN
    for f in FIELDS
}
ids = sorted(CM[("ISLA", "pressure_l2")])
BASE = ("GeoTransolver", "Transolver")
assert all(set(CM[(b, f)]) == set(ids) for b in BASE for f in FIELDS)
case_level = {
    "n_cases": N_CASES,
    "per_baseline": {
        b: {
            f: {
                "lower_on": sum(CM[(b, f)][i] < CM[("ISLA", f)][i] for i in ids),
                "median_ratio": st.median(
                    CM[(b, f)][i] / CM[("ISLA", f)][i] for i in ids
                ),
            }
            for f in FIELDS
        }
        for b in BASE
    },
}
lead = {
    i: CM[("ISLA", "pressure_l2")][i] - CM[("GeoTransolver", "pressure_l2")][i]
    for i in ids
}
n_worst = N_CASES // 10
worst = sorted(ids, key=lambda i: -CM[("ISLA", "pressure_l2")][i])[:n_worst]
case_level["geotransolver_lead_in_isla_worst"] = {
    "field": "pressure_l2",
    "n_worst": n_worst,
    "share_of_mean_lead": sum(lead[i] for i in worst) / sum(lead.values()),
    "aoa_deg": sorted(aoa(i) for i in worst),
    "isla_mean": st.mean(CM[("ISLA", "pressure_l2")][i] for i in worst),
    "geotransolver_mean": st.mean(
        CM[("GeoTransolver", "pressure_l2")][i] for i in worst
    ),
}
case_level["by_aoa"] = {
    str(a): {
        "n": len(g),
        **{x: st.mean(CM[(x, "pressure_l2")][i] for i in g) for x in ("ISLA",) + BASE},
    }
    for a in sorted({aoa(i) for i in ids})
    for g in [[i for i in ids if aoa(i) == a]]
}
# stage 1's per-case pressure of the stored-wing checkpoints under (b), seed means, on the same cases
CB = {
    x: {
        c: st.mean(S1["per_case_pressure"][f"{x}_seed{s}"][c]["b"] for s in SEEDS)
        for c in ids
    }
    for x in ("ISLA",) + BASE
}


def band(lo, hi):
    """Pressure on the cases with angle of attack in [lo, hi]: seed means, per seed, ratios and case counts."""
    g = [i for i in ids if lo <= aoa(i) <= hi]
    per = {
        (x, s): st.mean(seed_case[(x, s, "pressure_l2")][i] for i in g)
        for x in ("ISLA",) + BASE
        for s in SEEDS
    }
    return {
        "n": len(g),
        **{x: st.mean(per[(x, s)] for s in SEEDS) for x in ("ISLA",) + BASE},
        "per_seed": {x: {s: per[(x, s)] for s in SEEDS} for x in ("ISLA",) + BASE},
        "ratio": {
            b: st.mean(per[(b, s)] for s in SEEDS)
            / st.mean(per[("ISLA", s)] for s in SEEDS)
            for b in BASE
        },
        "ratio_per_seed": {
            b: {s: per[(b, s)] / per[("ISLA", s)] for s in SEEDS} for b in BASE
        },
        "isla_lower_on": {
            b: sum(
                CM[("ISLA", "pressure_l2")][i] < CM[(b, "pressure_l2")][i] for i in g
            )
            for b in BASE
        },
        "isla_lower_on_stored_wing_b": {
            b: sum(CB["ISLA"][i] < CB[b][i] for i in g) for b in BASE
        },
        "isla_stored_wing_b": st.mean(CB["ISLA"][i] for i in g),
    }


case_level["bands"] = {k: band(*v) for k, v in AOA_BANDS.items()}
case_level["bands"]["all"] = band(min(aoa(i) for i in ids), max(aoa(i) for i in ids))
out["case_level"] = case_level

# --- the input-scale control, on the stored wing -------------------------------------------------------------------------
OA = {a.split()[0]: v for a, v in OLD["arms"].items()}
keep = lambda arm: {p: arm[p] for p in ("native", "area")}  # noqa: E731
out["unit_drive_control"] = {
    "what": "the wing meshes as stored, with their duplicated root cells, 200,000 cells per case, the 180 validation cases: the "
    "stored-wing standing's unit-direction baselines and ISLA (results/char/hilift_surface_090.json arms) and the same baseline "
    "configurations reading the physical freestream vector U_inf in inch/s (its raw_drive_<arch>)",
    "ISLA": keep(OA["ISLA"]),
    **{
        b: {
            "unit": keep(OA[b]),
            "physical": keep(OLD[f"raw_drive_{b.lower()}"]),
        }
        for b in BASE
    },
}

json.dump(out, open(R / "hilift_surface_090_dedup.json", "w"), indent=1)
for a in RUN:
    print(
        f"{ARCH[a]:14s}",
        " ".join(
            f"{p}: " + " / ".join(f"{arms[a][p][f]['mean']:.5f}" for f in FIELDS)
            for p in PROTOS
        ),
    )
print("rule:", verdict, {k: round(c["G"], 3) for k, c in cells.items()})
print(
    "bands:", {k: (v["n"], v["isla_lower_on"]) for k, v in case_level["bands"].items()}
)
