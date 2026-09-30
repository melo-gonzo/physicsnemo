"""ROUTE-NORM at the reporting protocol (notebook #sec-nb-routenorm-p200k-prereg): the wing half and the deep car pair.

Lanes (group rn, `kernel/routenorm_aga.sbatch` rows 36 to 43; bf16 training, 500 epochs, float32 scoring through the
fleet pump under the `code_routenorm` snapshot, two seeds):
  35-wing rung (HiLiftAeroML super_scarce split), 200,000 cells: cosine 192 at 1e-3, cosine 384 at 1e-3, linear 384 at 1e-3.
  DrivAerML 435 cars, 200,000 cells: cosine 384 wide, 24 layers, at 5e-4.
Also read: the 510-wing rung (medium split) lanes of the original design at 100,000 cells, for their validation
trajectories only (they are in flight; their scores belong to the SCALE-HL reduction).

The collapse clause is evaluated on the training logs' validation-loss trajectories. The preregistered threshold
("rises above 0.01 after epoch 20") was written in the cars' loss units; on the wings every lane, cosine included, sits
above 0.01 through epoch 500 (the 35-wing floor is 0.020), so the threshold as written cannot discriminate there. The
reduction therefore states the clause in the units ROUTING-SAT measured it in: the collapse signature is a validation loss
that rises after epoch 20 and plateaus at more than three times the cosine lane's value at the same epoch (the cars'
plateau was 0.75 against 0.043, seventeen times). Both readings are written to the artifact.

usage: python routenorm_wings_reduce_2026-09-22.py <fetched p200k_fp32 dir> <valloss.tsv> <fleet.json> <extension.json> <out.json>
"""

from __future__ import annotations

import glob, json, statistics as st, sys
from collections import defaultdict
from pathlib import Path

root, val_path, fleet_path, ext_path, out_path = (Path(a) for a in sys.argv[1:6])
FLEET = json.load(open(fleet_path))["datasets"]
EXT = json.load(open(ext_path))["configurations"]

LANES = {
    "wings35|cos|w192|lr1e-3|c200k": "rn_cos_w192_lr1e_3_ss_c200k_seed{}",
    "wings35|cos|w384|lr1e-3|c200k": "rn_cos_w384_lr1e_3_ss_c200k_seed{}",
    "wings35|lin|w384|lr1e-3|c200k": "rn_lin_w384_lr1e_3_ss_c200k_seed{}",
    "cars|cos|w384|d24|lr5e-4|c200k": "rn_cos_w384_d24_lr5e_4_p200k_c200k_seed{}",
}
TRAJ = {
    "wings35": {
        "cos192": "rn_cos_w192_lr1e_3_ss_c200k_seed{}",
        "cos384": "rn_cos_w384_lr1e_3_ss_c200k_seed{}",
        "lin384": "rn_lin_w384_lr1e_3_ss_c200k_seed{}",
    },
    "wings510": {
        "cos192": "rn_cos_w192_lr1e_3_med_c100k_seed{}",
        "cos384": "rn_cos_w384_lr1e_3_med_c100k_seed{}",
        "lin384": "rn_lin_w384_lr1e_3_med_c100k_seed{}",
    },
}
SEEDS = (42, 43)


def score(run):
    """Float32 surface-pressure L2 of one run from its metrics.jsonl, or None if unscored."""
    ps = glob.glob(str(root / run / "**" / "metrics.jsonl"), recursive=True)
    if not ps:
        return None
    rows = [json.loads(l) for l in open(ps[0]) if l.strip()]
    s = [r for r in rows if r.get("phase") == "infer_summary"]
    return s[0]["metrics"].get("pressure_l2") if s else None


conf = {}
for key, pat in LANES.items():
    runs = {pat.format(s): score(pat.format(s)) for s in SEEDS}
    got = [v for v in runs.values() if v is not None]
    c = {
        "runs": runs,
        "n_scored": len(got),
        "status": "complete" if len(got) == 2 else ("partial" if got else "pending"),
    }
    if got:
        c["pressure_l2_seeds"] = got
        c["pressure_l2_mean"] = st.mean(got)
        if len(got) == 2:
            c["seed_spread"] = max(got) / min(got) - 1
    conf[key] = c

# validation trajectories
traj = defaultdict(dict)
for line in open(val_path):
    p = line.split()
    if len(p) != 3:
        continue
    try:
        traj[p[0]][int(p[1])] = float(p[2])
    except ValueError:
        traj[p[0]][int(p[1])] = float("nan")

collapse = {}
for rung, arms in TRAJ.items():
    r = {"arms": {}}
    for arm, pat in arms.items():
        for s in SEEDS:
            run = pat.format(s)
            t = traj.get(run, {})
            if not t:
                continue
            last = max(t)
            after20 = {e: v for e, v in t.items() if e > 20}
            r["arms"][run] = {
                "last_epoch": last,
                "val_at_last": t[last],
                "val_at_25": t.get(25),
                "max_after_20": max(after20.values()) if after20 else None,
                "n_readings": len(t),
            }
    # ratio of the linear control to the cosine lane at matched epochs (same width, 384), per seed
    ratios = {}
    for s in SEEDS:
        lin = traj.get(arms["lin384"].format(s), {})
        cos = traj.get(arms["cos384"].format(s), {})
        common = sorted(e for e in lin if e in cos and e > 20)
        if common:
            ratios[f"seed{s}"] = {
                "epochs": common,
                "lin_over_cos": [lin[e] / cos[e] for e in common],
            }
    r["lin384_over_cos384_after_epoch_20"] = ratios
    plateaued = [s for s, x in ratios.items() if max(x["lin_over_cos"]) > 3]
    cos_below_001 = all(
        v < 0.01
        for arm in ("cos192", "cos384")
        for s in SEEDS
        for e, v in traj.get(arms[arm].format(s), {}).items()
        if e > 20
    )
    lin_above_001 = any(
        v > 0.01
        for s in SEEDS
        for e, v in traj.get(arms["lin384"].format(s), {}).items()
        if e > 20
    )
    r["clause_as_written"] = {
        "linear_above_0.01_after_20": lin_above_001,
        "cosine_below_0.01_after_20": cos_below_001,
        "fires": lin_above_001 and cos_below_001,
        "note": "threshold written in the cars' loss units; not discriminating on the 35-wing rung, where every lane sits above 0.01",
    }
    r["clause_in_signature_units"] = {
        "seeds_with_linear_plateau_above_3x_cosine": plateaued,
        "fires": bool(plateaued),
        "definition": "linear 384 validation loss more than three times cosine 384 at a matched epoch after epoch 20",
    }
    r["status"] = (
        "final (epoch 500)"
        if all(a["last_epoch"] >= 475 for a in r["arms"].values())
        else "interim (in flight)"
    )
    collapse[rung] = r

# references
ss = FLEET["highlift_surface_super_scarce"]["arms"]
lin192 = ss["isla"]["pressure_l2_mean"]
gt = ss["gt_unit"]["pressure_l2_mean"]
tr = ss["transolver_unit"]["pressure_l2_mean"]
d24_100k = EXT["isla|cos|w384|d24|lr5e-4"]["pressure_l2_mean"]

bars = {}
c = conf["wings35|cos|w192|lr1e-3|c200k"]
if c["status"] != "pending":
    r = c["pressure_l2_mean"] / lin192
    bars["second_dataset_accuracy_null"] = {
        "cosine_192": c["pressure_l2_mean"],
        "linear_192_fleet": lin192,
        "cosine_over_linear": r,
        "n_scored": c["n_scored"],
        "verdict": (
            "null (within 2%)"
            if abs(1 - r) <= 0.02
            else (
                "cosine better by more than 2%: a finding"
                if r < 0.98
                else "cosine worse by more than 2%: a cost"
            )
        )
        + ("" if c["status"] == "complete" else " [partial: one seed]"),
        "gt_over_cosine": gt / c["pressure_l2_mean"],
        "transolver_over_cosine": tr / c["pressure_l2_mean"],
        "gt_over_linear_fleet": gt / lin192,
        "transolver_over_linear_fleet": tr / lin192,
    }
c2 = conf["wings35|cos|w384|lr1e-3|c200k"]
if c["status"] != "pending" and c2["status"] != "pending":
    r = c2["pressure_l2_mean"] / c["pressure_l2_mean"]
    bars["width_on_the_wings"] = {
        "cosine_384": c2["pressure_l2_mean"],
        "cosine_192": c["pressure_l2_mean"],
        "ratio_384_over_192": r,
        "verdict": (
            "width carries (at least 3% below 192)"
            if r <= 0.97
            else "within 3%: the wing deficit is not capacity"
        )
        + ("" if c["status"] == c2["status"] == "complete" else " [partial]"),
        "gt_over_cosine_384": gt / c2["pressure_l2_mean"],
        "transolver_over_cosine_384": tr / c2["pressure_l2_mean"],
    }
c3 = conf["wings35|lin|w384|lr1e-3|c200k"]
if c3["status"] != "pending" and c2["status"] != "pending":
    bars["linear_384_control_accuracy"] = {
        "linear_384": c3["pressure_l2_mean"],
        "cosine_384": c2["pressure_l2_mean"],
        "linear_over_cosine": c3["pressure_l2_mean"] / c2["pressure_l2_mean"],
        "note": "the control trained cleanly on this rung; its score is reported, not a bar",
    }
c4 = conf["cars|cos|w384|d24|lr5e-4|c200k"]
if c4["status"] != "pending":
    v = c4["pressure_l2_mean"]
    bars["best_configuration_at_the_protocol"] = {
        "cosine_384x24_200k": v,
        "cosine_384x24_100k": d24_100k,
        "ratio_200k_over_100k": v / d24_100k,
        "cells_flat": abs(v / d24_100k - 1) <= 0.02,
        "gt_best_mark": 0.0406,
        "at_or_below_gt_best": v <= 0.0406,
        "relative_to_gt_best": v / 0.0406 - 1,
    }
bars["collapse_clause"] = {
    rung: {
        "as_written": r["clause_as_written"]["fires"],
        "in_signature_units": r["clause_in_signature_units"]["fires"],
        "status": r["status"],
    }
    for rung, r in collapse.items()
}

out = {
    "study": "ROUTE-NORM at the reporting protocol: cosine routing on the 35-wing rung at 200,000 cells (cosine 192 and 384 at 1e-3, linear 384 control) and cosine 384 x 24 layers on the 435 cars at 200,000 cells; float32 scoring, two seeds; 510-wing trajectories read for the collapse clause only",
    "configurations": conf,
    "collapse_trajectories": collapse,
    "references": {
        "wings35_linear_isla_192_fleet": lin192,
        "wings35_geotransolver": gt,
        "wings35_transolver": tr,
        "cars_cosine_384x24_100k": d24_100k,
        "sources": [
            "results/p200k_fleet_reduction_2026-09-22.json",
            "results/routenorm_extension_2026-09-21.json",
        ],
    },
    "bars": bars,
}
out_path.write_text(json.dumps(out, indent=2))
for k, c in conf.items():
    print(k, c["status"], c.get("pressure_l2_mean"), c.get("pressure_l2_seeds"))
for rung, r in collapse.items():
    print(
        rung,
        r["status"],
        "as written:",
        r["clause_as_written"]["fires"],
        "| signature units:",
        r["clause_in_signature_units"],
    )
    for s, x in r["lin384_over_cos384_after_epoch_20"].items():
        print("   ", s, "lin/cos", [round(v, 2) for v in x["lin_over_cos"]])
print(json.dumps(bars, indent=1))
