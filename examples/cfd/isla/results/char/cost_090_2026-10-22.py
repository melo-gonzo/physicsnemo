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

"""Training cost of the reference configurations at ISLA 0.9.0 (2026-10-22): per dataset (a book/standing.py DATASETS key) and
architecture (ISLA, ISLA-DTT, GeoTransolver 256, Transolver 512 x 16), the lanes behind the standings, each priced from its own
training log as the seed mean of

* step_seconds: the median training step, over the train_step rows with step_time_s < 10 s, which leaves out the compile steps of
  the first epoch and of each restarted link (the AZ-DECIDE rule, scratch/isla-az-decide-2026-10-21/reduce_azdecide.py);
* peak_gib: the peak allocated device memory, the maximum over epochs of the train_summary peak_memory_gib (the trainer's maximum
  over ranks);

with n_parameters from the scorer's config row (checked equal to the training config row), the GPUs and ranks (the CUDA devices
the training log names), the precision of the training config row, and a note where a lane differs from the configuration that
trains ISLA.

Inputs: results/campaigns_2026-10-22/cost_090/extract.jsonl, one line per run made on LHR by extract_cost.py beside it (README.md
there names each source and its md5). Five standing baseline lanes have no training log left on LHR (cl_r_t2dz_gt, cl_r_t2dz_t,
cl_r_jeb_gt, cl_r_jeb_t, cl_r_lug_t); each is priced from the one-epoch smoke lane of the same configuration
(results/campaigns_2026-10-02/char/cs/cs_r_<task>_<arch>/metrics.jsonl, seed 42), its parameter count from the standing lane's
scorer rows (results/campaigns_2026-10-02/char/protocols/<run>/native/metrics.jsonl). The lanes trained before the compile fix
(r090_af_*, r090_lug_*, r090_t2dz_isla) are kept under compiled_before_fix.

Writes results/char/cost_090.json: cost[key][arch] = {peak_gib, step_seconds, n_parameters, gpus, ranks, precision, source, note,
seeds, per_seed, code, completed}.
usage: python cost_090_2026-10-22.py
"""

import json
import os
import statistics as st

R = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(R)
EXTRACT = "campaigns_2026-10-22/cost_090/extract.jsonl"
CS = "campaigns_2026-10-02/char/cs"
PROTOCOLS = "campaigns_2026-10-02/char/protocols"
EX = {d["run"]: d for d in map(json.loads, open(f"{RES}/{EXTRACT}"))}
S2 = ("42", "43")
FIXED = "code_cmpaf_fix"
# The cs group's lanes ran on four GB300 ranks (char_lanes_cost_2026-10-04.py); their logs name no devices.
CS_RANKS = 4

# dataset key -> {architecture: run stem}
LANES = {
    "drivaerml_surface": {
        "ISLA": "r090_dr_isla",
        "GeoTransolver": "r80ud_dr_gt_h256",
        "Transolver": "r80ud_dr_t_h512_l16_c100k",
    },
    "drivaerml_volume": {
        "ISLA": "r090_vol_isla",
        "GeoTransolver": "clud_vol_gt",
        "Transolver": "clud_vol_t",
    },
    "hilift": {
        "ISLA": "h090_hl_isla",
        "GeoTransolver": "udrv_hl_gt",
        "Transolver": "udrv_hl_t",
    },
    "shift_suv": {
        "ISLA": "r090_suv_isla",
        "GeoTransolver": "clud_suv_gt",
        "Transolver": "clud_suv_t",
    },
    "airfrans": {
        "ISLA": "r090fix_af_isla",
        "ISLA-DTT": "r090fix_af_dtt",
        "GeoTransolver": "afu_ud_gt",
        "Transolver": "afu_ud_t",
    },
    "aerozoo": {
        "ISLA": "azc_full_isla",
        "ISLA-DTT": "azc_full_dtt",
        "GeoTransolver": "azc_full_gt",
        "Transolver": "azc_full_t",
    },
    "tensile2d": {"ISLA": "r090fix_t2dz_isla"},
    "lug": {
        "ISLA": "r090fix_lug_isla",
        "ISLA-DTT": "r090fix_lug_dtt",
        "GeoTransolver": "lugbc_gt",
    },
    "deepjeb": {"ISLA": "r090_jb_isla"},
}
SEEDS = {"r090fix_t2dz_isla": ("42", "43", "44", "45")}
# (dataset key, architecture) -> (standing run stem, one-epoch smoke lane of the same configuration)
SMOKE = {
    ("tensile2d", "GeoTransolver"): ("cl_r_t2dz_gt", "cs_r_t2dz_gt"),
    ("tensile2d", "Transolver"): ("cl_r_t2dz_t", "cs_r_t2dz_t"),
    ("lug", "Transolver"): ("cl_r_lug_t", "cs_r_lug_t"),
    ("deepjeb", "GeoTransolver"): ("cl_r_jeb_gt", "cs_r_jeb_gt"),
    ("deepjeb", "Transolver"): ("cl_r_jeb_t", "cs_r_jeb_t"),
}
BEFORE_FIX = {
    "airfrans": {"ISLA": "r090_af_isla", "ISLA-DTT": "r090_af_dtt"},
    "lug": {"ISLA": "r090_lug_isla", "ISLA-DTT": "r090_lug_dtt"},
    "tensile2d": {"ISLA": "r090_t2dz_isla"},
}
NOTE = {
    (
        "tensile2d",
        "ISLA",
    ): "the standing's ISLA checkpoints trained eager (r090eag_t2dz_isla) for the training-path control, at about "
    "twice this step time; this is the compiled lane of the same configuration",
}


def one(values, what, stem):
    """The one value the seeds of a lane share."""
    v = set(values)
    assert len(v) == 1, f"{stem}: {what} differs across seeds: {v}"
    return v.pop()


def lane(key, arch, stem):
    """The cost record of one lane from its extract lines."""
    seeds = SEEDS.get(stem, S2)
    rs = {s: EX[f"{stem}_seed{s}"] for s in seeds}
    for r in rs.values():
        assert "missing" not in r, r
        assert r["step"]["n"] > 0 and r["train"], r["run"]
        sc = r["scorer_config"].get("num_parameters")
        assert sc in (None, r["config"]["num_parameters"]), (
            f"{r['run']}: scorer {sc} != training {r['config']['num_parameters']}"
        )
    per = {
        s: {
            "step_seconds": r["step"]["median"],
            "peak_gib": max(e[1] for e in r["train"].values() if e[1] is not None),
        }
        for s, r in rs.items()
    }
    scored = all(r["scorer_config"].get("num_parameters") for r in rs.values())
    ranks = one((len(r["devices"]) for r in rs.values()), "ranks", stem)
    code = sorted({c for r in rs.values() for c in r["code"]})
    notes = [NOTE[(key, arch)]] if (key, arch) in NOTE else []
    if arch.startswith("ISLA") and code != [FIXED]:
        notes.append(
            f"trained on {', '.join(code)}, before the compile fix (the compiled blocks ran Inductor's mix-order reduction)"
        )
    if not arch.startswith("ISLA"):
        notes.append("uncompiled, as released")
    if not all(r["completed"] for r in rs.values()):
        n = one(
            (r["config_params"]["training.num_epochs"] for r in rs.values()),
            "num_epochs",
            stem,
        )
        notes.append(
            f"in flight: {' and '.join(str(len(r['train'])) for r in rs.values())} of {n} epochs logged"
        )
    if not scored:
        notes.append("parameter count from the training config row (not yet scored)")
    return {
        "peak_gib": st.fmean(p["peak_gib"] for p in per.values()),
        "step_seconds": st.fmean(p["step_seconds"] for p in per.values()),
        "n_parameters": one(
            (r["config"]["num_parameters"] for r in rs.values()), "num_parameters", stem
        ),
        "gpus": ranks,
        "ranks": ranks,
        "precision": one(
            (r["config_params"]["precision"] for r in rs.values()), "precision", stem
        ),
        "source": f"{EXTRACT}: {stem}_seed{{{','.join(seeds)}}}",
        "note": "; ".join(notes) or None,
        "seeds": list(seeds),
        "per_seed": per,
        "code": code,
        "completed": all(r["completed"] for r in rs.values()),
    }


def smoke(stem, cs):
    """The cost record of a standing lane whose log is gone, from the one-epoch smoke lane of its configuration."""
    rows = [json.loads(line) for line in open(f"{RES}/{CS}/{cs}/metrics.jsonl")]
    cfg = [r for r in rows if r.get("phase") == "config"][-1]
    steps = [
        r["step_time_s"]
        for r in rows
        if r.get("phase") == "train_step" and r.get("step_time_s") is not None
    ]
    n_par = one(
        (
            next(
                json.loads(line)
                for line in open(
                    f"{RES}/{PROTOCOLS}/{stem}_seed{s}/native/metrics.jsonl"
                )
                if '"phase": "config"' in line
            )["num_parameters"]
            for s in S2
        ),
        "num_parameters",
        stem,
    )
    assert n_par == cfg["num_parameters"], (
        f"{stem}: scorer {n_par} != smoke lane {cfg['num_parameters']}"
    )
    return {
        "peak_gib": max(
            r["peak_memory_gib"] for r in rows if r.get("phase") == "train_summary"
        ),
        "step_seconds": st.median(s for s in steps if s < 10),
        "n_parameters": n_par,
        "gpus": CS_RANKS,
        "ranks": CS_RANKS,
        "precision": cfg["params"]["precision"],
        "source": f"{CS}/{cs}/metrics.jsonl; parameters: {PROTOCOLS}/{stem}_seed{{42,43}}/native/metrics.jsonl",
        "note": f"uncompiled, as released; the standing lane's training log ({stem}) is gone, so this is the one-epoch smoke lane of "
        f"the same configuration, one seed ({cfg['params']['training.seed']}), {sum(s < 10 for s in steps)} steps",
        "seeds": [cfg["params"]["training.seed"]],
        "per_seed": None,
        "code": None,
        "completed": True,
    }


out = {
    "protocol": __doc__,
    "note": "The cost of the configuration that trains ISLA: GB300 GPUs, four ranks, bfloat16 autocast, the blocks compiled one by "
    "one with Inductor's mix-order reduction off; the baselines train uncompiled, as released. The eager-trained lanes (r090eag_*) "
    "are not priced: they trained eager only for the training-path control, at about twice the step time.",
    "cost": {},
    "compiled_before_fix": {},
}
for key, arches in LANES.items():
    out["cost"][key] = {a: lane(key, a, stem) for a, stem in arches.items()}
for (key, arch), (stem, cs) in SMOKE.items():
    out["cost"][key][arch] = smoke(stem, cs)
for key, arches in BEFORE_FIX.items():
    out["compiled_before_fix"][key] = {
        a: lane(key, a, stem) for a, stem in arches.items()
    }
    for a, b in out["compiled_before_fix"][key].items():
        c = out["cost"][key][a]
        fix = f"the compile fix moves the median step from {b['step_seconds']:.3f} s to {c['step_seconds']:.3f} s (compiled_before_fix)"
        c["note"] = "; ".join(filter(None, (c["note"], fix)))
json.dump(out, open(f"{R}/cost_090.json", "w"), indent=1)
for key, d in out["cost"].items():
    print(
        f"== {key}: "
        + "  ".join(
            f"{a} {v['step_seconds']:.3f} s / {v['peak_gib']:.1f} GiB / {v['n_parameters'] / 1e6:.2f}M {v['precision']}"
            for a, v in d.items()
        )
    )
