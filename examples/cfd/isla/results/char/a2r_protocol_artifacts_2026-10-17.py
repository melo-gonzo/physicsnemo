"""A2-RERUN artifacts (2026-10-17, #sec-nb-a2-rerun-jeb-t2d, #sec-nb-a2-rerun-drivaerml,
#sec-nb-a2-rerun-suv-lug): the CHAR-PROTOCOLS record of each
rerun dataset with ISLA's arm replaced by the A2 reruns (the reference configuration with the normalized slice reads, seeds 42
and 43, final checkpoints, the CHAR scorer: float32, native mesh / uniform point cloud / rotated + translated). The A2 scores
are the reductions committed in results/campaigns_2026-10-17/a2r/<stem>_a2_scores.json (copied unmodified from
scratch/isla-a2-rerun-2026-10-17/, written there by char_a2_table.py and deepjeb_a2_table.py); the baselines' arms are the CHAR
record's, unchanged (GeoTransolver and Transolver are not rerun). Writes results/char/<dataset>_a2.json in the schema of
results/char/<dataset>.json (protocols, fields, isla_arm, arms[arm][protocol][field] = {mean, per_seed, n_cases, n_seeds},
complete), so the chapters' protocol_table() and char_arms() read it unchanged, plus the key "raw_read_isla", the CHAR record's
ISLA arm (raw slice reads), which protocol_table ignores. n_cases is the A2 record's where it has one (DeepJEB++), else the CHAR
ISLA arm's: the A2 lanes are scored on the same validation split by the same scorer. Two lists replace a baseline arm: FAIR_INPUT
(retrained on fair inputs) and SCORESET (the same checkpoints re-scored on ISLA's scored set); each keeps the replaced arm under its
own key.
usage: python a2r_protocol_artifacts_2026-10-17.py"""
import json, os

R = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(R), "campaigns_2026-10-17", "a2r")
ISLA_ARM = "ISLA 512 with 512 slices"
# CHAR dataset stem -> (A2 score file stem, the arm of that file holding A2, or None for the char_a2_table.py layout)
DATASETS = {"drivaerml_surface": ("drivaerml_surface", None), "drivaerml_volume": ("drivaerml_volume", None),
            "tensile2d": ("tensile2d", None), "deepjeb_surface": ("deepjeb", "ISLA A2 at 3.75e-4 (jbn2)"),
            "shift_suv_surface": ("shift_suv_surface", None), "lug": ("lug", None)}
# Baseline arms replaced by a fair-input rerun of the same configuration, scored by the same protocol:
# dataset -> {arm label: (reduced score file under results/, run stem without _seedNN, the key the replaced arm is kept under,
# what the rerun reads)}. The score file is reduce_drivefix.py's layout ({"<stem>_seed<s>/<protocol>": {metrics, n, epoch, source}}).
# DRIVE-FIX (2026-10-19, campaigns_2026-10-19/drive_fix/README.md): on the lug GeoTransolver's geometry context becomes the surface
# triangles with their normals and the four boundary-condition labels ISLA reads (bc_fixed, bc_hole, bc_loaded, bc_pressure), where the
# CHAR arm's context is the interior nodes and reads no label.
FAIR_INPUT = {
    "lug": {"GeoTransolver 256": ("campaigns_2026-10-19/drive_fix/drivefix_scores_2026-10-19.json", "evals_lugbc/lugbc_gt",
                                  "unlabelled_geotransolver", "the surface triangles with their normals and the four boundary-condition "
                                  "labels as its geometry context (lugbc_gt)")},
}
# Baseline arms re-scored on ISLA's scored set: dataset -> {arm label: (score file under results/, run stem without _seedNN, the key
# the replaced arm is kept under, the arm's new label, what changed)}. The score file is reduce_tset200k.py's layout
# ({"files": {"<evals>/<stem>_seed<s>/<protocol>": {metrics, n, source}}}). T-SCORESET (2026-10-19,
# campaigns_2026-10-19/drive_fix2/README.md): the DrivAerML Surface Transolver arm, trained at 100,000 cells, was scored on its own
# 100,000-cell draw; re-scored on the 200,000 cells of ISLA's set (hash-identical per car), same checkpoints and scorer.
SCORESET = {
    "drivaerml_surface": {"Transolver 512 x 16 at 100k cells": (
        "campaigns_2026-10-19/drive_fix2/scoreset/tset200k_scores.json", "evals_tset200k/r80_dr_t_h512_l16_c100k",
        "transolver_100k_scoring", "Transolver 512 x 16, trained at 100k cells",
        "re-scored on ISLA's 200,000-cell set (r80_dr_t_h512_l16_c100k, the same checkpoints)")},
}


def rescored_arm(path, stem, fields, protocols, seeds=("42", "43")):
    """{protocol: {field: {mean, per_seed, n_cases, n_seeds}}} of a re-scored arm, from a reduce_tset200k.py score file."""
    rec = json.load(open(f"{os.path.dirname(R)}/{path}"))["files"]
    runs = {s: {pk: rec[f"{stem}_seed{s}/{pk}"] for pk in protocols} for s in seeds}
    assert all(r["source"] == "summary" for v in runs.values() for r in v.values())
    return {pk: {f: {"mean": sum(runs[s][pk]["metrics"][f] for s in seeds) / len(seeds),
                     "per_seed": {s: runs[s][pk]["metrics"][f] for s in seeds},
                     "n_cases": min(runs[s][pk]["n"] for s in seeds), "n_seeds": len(seeds)} for f in fields} for pk in protocols}


def rerun_arm(path, stem, fields, protocols, seeds=("42", "43")):
    """{protocol: {field: {mean, per_seed, n_cases, n_seeds}}} of a fair-input rerun, from a reduce_drivefix.py score file."""
    rec = json.load(open(f"{os.path.dirname(R)}/{path}"))
    runs = {s: {pk: rec[f"{stem}_seed{s}/{pk}"] for pk in protocols} for s in seeds}
    assert all(r["epoch"] == 500 and r["source"] == "summary" for v in runs.values() for r in v.values())
    return {pk: {f: {"mean": sum(runs[s][pk]["metrics"][f] for s in seeds) / len(seeds),
                     "per_seed": {s: runs[s][pk]["metrics"][f] for s in seeds},
                     "n_cases": min(runs[s][pk]["n"] for s in seeds), "n_seeds": len(seeds)} for f in fields} for pk in protocols}


for ds, (stem, a2_arm) in DATASETS.items():
    C = json.load(open(f"{R}/{ds}.json")); old = C["arms"][C["isla_arm"]]
    S = json.load(open(f"{SRC}/{stem}_a2_scores.json"))
    a2 = S["arms"][a2_arm] if a2_arm else S["a2"]
    arm = {pk: {f: {"mean": a2[pk][f]["mean"], "per_seed": a2[pk][f]["per_seed"],
                    "n_cases": a2[pk][f].get("n_cases", old[pk][f]["n_cases"]), "n_seeds": len(a2[pk][f]["per_seed"])}
                for f in C["fields"]} for pk in C["protocols"]}
    out = {"protocol": __doc__, "protocols": C["protocols"], "fields": C["fields"], "isla_arm": ISLA_ARM,
           "arms": {ISLA_ARM: arm, **{k: v for k, v in C["arms"].items() if k != C["isla_arm"]}},
           "complete": all(c["n_seeds"] == 2 for p in arm.values() for c in p.values()),
           "raw_read_isla": {"label": C["isla_arm"], **old}}
    for label, (path, stem, key, reads) in FAIR_INPUT.get(ds, {}).items():
        out[key] = {"label": f"{label}, the CHAR record's arm", **out["arms"][label]}
        out["arms"][label] = rerun_arm(path, stem, C["fields"], C["protocols"])
        out.setdefault("fair_input", {})[label] = f"retrained reading {reads}; scores: {path}; the replaced arm is kept under {key}"
    for label, (path, stem, key, new, what) in SCORESET.get(ds, {}).items():
        out[key] = {"label": f"{label}, the CHAR record's arm", **out["arms"].pop(label)}
        out["arms"][new] = rescored_arm(path, stem, C["fields"], C["protocols"])
        out.setdefault("scored_set", {})[new] = f"{what}; scores: {path}; the arm scored on its own draw is kept under {key}"
    json.dump(out, open(f"{R}/{ds}_a2.json", "w"), indent=1)
    print(ds, " ".join(f"{pk}:" + "/".join(f"{arm[pk][f]['mean']:.4f}" for f in C["fields"]) for pk in C["protocols"]))
