"""A2-RERUN artifacts (2026-10-17, #sec-nb-a2-rerun-jeb-t2d, #sec-nb-a2-rerun-drivaerml): the CHAR-PROTOCOLS record of each
rerun dataset with ISLA's arm replaced by the A2 reruns (the reference configuration with the normalized slice reads, seeds 42
and 43, final checkpoints, the CHAR scorer: float32, native mesh / uniform point cloud / rotated + translated). The A2 scores
are the reductions committed in results/campaigns_2026-10-17/a2r/<stem>_a2_scores.json (copied unmodified from
scratch/isla-a2-rerun-2026-10-17/, written there by char_a2_table.py and deepjeb_a2_table.py); the baselines' arms are the CHAR
record's, unchanged (GeoTransolver and Transolver are not rerun). Writes results/char/<dataset>_a2.json in the schema of
results/char/<dataset>.json (protocols, fields, isla_arm, arms[arm][protocol][field] = {mean, per_seed, n_cases, n_seeds},
complete), so the chapters' protocol_table() and char_arms() read it unchanged, plus the key "raw_read_isla", the CHAR record's
ISLA arm (raw slice reads), which protocol_table ignores. n_cases is the A2 record's where it has one (DeepJEB++), else the CHAR
ISLA arm's: the A2 lanes are scored on the same validation split by the same scorer.
usage: python a2r_protocol_artifacts_2026-10-17.py"""
import json, os

R = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(R), "campaigns_2026-10-17", "a2r")
ISLA_ARM = "ISLA 512 with 512 slices"
# CHAR dataset stem -> (A2 score file stem, the arm of that file holding A2, or None for the char_a2_table.py layout)
DATASETS = {"drivaerml_surface": ("drivaerml_surface", None), "drivaerml_volume": ("drivaerml_volume", None),
            "tensile2d": ("tensile2d", None), "deepjeb_surface": ("deepjeb", "ISLA A2 at 3.75e-4 (jbn2)")}

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
    json.dump(out, open(f"{R}/{ds}_a2.json", "w"), indent=1)
    print(ds, " ".join(f"{pk}:" + "/".join(f"{arm[pk][f]['mean']:.4f}" for f in C["fields"]) for pk in C["protocols"]))
