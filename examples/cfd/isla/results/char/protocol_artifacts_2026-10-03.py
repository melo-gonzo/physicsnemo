"""CHAR-PROTOCOLS artifacts (2026-10-03, #sec-nb-char-prereg, #sec-nb-char-protocols-instruments): for every campaign dataset, the
three reference configurations (ISLA, GeoTransolver, Transolver) under the three scorings of one checkpoint: the native validation
mesh, a uniform point cloud at the same budget (area-proportional Poisson draw with exact Horvitz-Thompson weights), and every
validation case rotated and translated (keyed SO(3) or in-plane rotation plus one RMS radius). Float32, seed means over the two
seeds, per-case relative L2 averaged over the validation cases. Reads campaigns_2026-10-02/char/protocols/<run>/<protocol>/metrics.jsonl
(fetched from $T/evals_char/<run>/<protocol>/<run>/metrics.jsonl) and writes results/char/<dataset>.json in the schema the dataset
chapters' protocol_table() reads: protocols, fields, arms[arm][protocol][field] = {mean, per_seed, n_cases}, isla_arm.
usage: python protocol_artifacts_2026-10-03.py"""
import glob, json, os, statistics as st

R = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(R)
SRC = f"{RES}/campaigns_2026-10-02/char/protocols"
PROTOCOLS = {"native": "native mesh", "area": "uniform point cloud", "posed": "rotated + translated"}
SEEDS = (42, 43)
# dataset artifact -> (fields {metric key: label}, arms {label: run stem without _seedNN}, isla arm label)
DATASETS = {
    "drivaerml_surface.json": ({"pressure_l2": "pressure", "wss_l2": "wall shear"},
                               {"ISLA 512 with 512 slices": "r80_dr_isla_w512_s512",
                                "GeoTransolver 256": "r80_dr_gt_h256",
                                "Transolver 512 x 16 at 100k cells": "r80_dr_t_h512_l16_c100k"}, "ISLA 512 with 512 slices"),
    "hilift_surface.json": ({"pressure_l2": "pressure", "tau_wall_l2": "wall shear"},
                            {"ISLA 512 with 512 slices": "r80_hl_isla_w512_s512",
                             "GeoTransolver 256": "r80_hl_gt_h256",
                             "Transolver 512 x 16 at 100k cells": "r80_hl_t_h512_l16_c100k"}, "ISLA 512 with 512 slices"),
    "drivaerml_volume.json": ({"pressure_l2": "pressure", "velocity_l2": "velocity", "nut_l2": "eddy viscosity"},
                              {"ISLA 512 with 512 slices": "cl_r_vol_isla", "GeoTransolver 256": "cl_r_vol_gt",
                               "Transolver 512 x 16 at 100k cells": "cl_r_vol_t"}, "ISLA 512 with 512 slices"),
    "airfrans.json": ({"pressure_l2": "pressure", "velocity_l2": "velocity", "nut_l2": "eddy viscosity"},
                      {"ISLA 512 with 512 slices": "cl_r_af_isla", "GeoTransolver 256": "cl_r_af_gt", "Transolver 512 x 16": "cl_r_af_t"}, "ISLA 512 with 512 slices"),
    "shift_suv_surface.json": ({"pressure_l2": "pressure", "wss_l2": "wall shear"},
                               {"ISLA 512 with 512 slices": "cl_r_suv_isla", "GeoTransolver 256": "cl_r_suv_gt",
                                "Transolver 512 x 16 at 100k cells": "cl_r_suv_t"}, "ISLA 512 with 512 slices"),
    "tensile2d.json": ({"stress_l2": "stress (Frobenius)", "displacement_l2": "displacement", "q_l2": "plastic strain"},
                       {"ISLA 512 with 512 slices": "cl_r_t2dz_isla", "GeoTransolver 256": "cl_r_t2dz_gt", "Transolver 512 x 16": "cl_r_t2dz_t"}, "ISLA 512 with 512 slices"),
    "lug.json": ({"von_mises_l2": "von Mises"},
                 {"ISLA 512 with 512 slices": "cl_r_lug_isla", "GeoTransolver 256": "cl_r_lug_gt", "Transolver 512 x 16": "cl_r_lug_t"}, "ISLA 512 with 512 slices"),
    "deepjeb_surface.json": ({"von_mises_l2": "von Mises", "displacement_l2": "displacement"},
                             {"ISLA 512 with 512 slices": "cl_r_jeb_isla", "GeoTransolver 256": "cl_r_jeb_gt", "Transolver 512 x 16": "cl_r_jeb_t"}, "ISLA 512 with 512 slices"),
}
# First-pass lanes (DrivAerML-half references, 2026-10-03) used while a dataset's final-reference lanes are not yet scored.
FIRST_PASS = {
    "deepjeb_surface.json": ({"ISLA 768 at 100k cells": "cl_jeb_isla", "GeoTransolver 1,024 at 100k cells": "cl_jeb_gt",
                              "Transolver 1,024 at 100k cells": "cl_jeb_t"}, "ISLA 768 at 100k cells"),
}


def summary(path):
    """infer_summary metrics, else the mean of the per-sample rows (last row per sample id)."""
    if not os.path.exists(path):
        return None
    by_sample = {}
    for line in open(path):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("phase") == "infer_summary":
            return {"metrics": rec["metrics"], "n_cases": rec.get("num_samples")}
        if rec.get("phase") == "infer_step":
            by_sample[rec.get("sample_id", len(by_sample))] = rec["metrics"]
    if not by_sample:
        return None
    steps = list(by_sample.values()); keys = sorted({k for m in steps for k in m})
    return {"metrics": {k: st.mean(m[k] for m in steps if k in m) for k in keys}, "n_cases": len(steps)}


def scored(arms):
    return all(os.path.exists(f"{SRC}/{stem}_seed{s}/{pk}/metrics.jsonl") for stem in arms.values() for s in SEEDS for pk in PROTOCOLS)


written = []
for art, (fields, arms, isla_arm) in DATASETS.items():
    if art in FIRST_PASS and not scored(arms) and scored(FIRST_PASS[art][0]):
        arms, isla_arm = FIRST_PASS[art]
        print(f"== {art}: final-reference lanes not scored; using the first-pass lanes {list(arms.values())}")
    out = {"protocol": __doc__, "protocols": PROTOCOLS, "fields": fields, "isla_arm": isla_arm, "arms": {}}
    complete = True
    for label, stem in arms.items():
        d = {}
        for pk in PROTOCOLS:
            per_seed = {}
            for s in SEEDS:
                rec = summary(f"{SRC}/{stem}_seed{s}/{pk}/metrics.jsonl")
                if rec:
                    per_seed[s] = rec
            if not per_seed:
                complete = False; break
            d[pk] = {f: {"mean": st.mean(r["metrics"][f] for r in per_seed.values() if f in r["metrics"]),
                         "per_seed": {s: r["metrics"].get(f) for s, r in per_seed.items()},
                         "n_cases": min(r["n_cases"] or 0 for r in per_seed.values()), "n_seeds": len(per_seed)}
                     for f in fields if all(f in r["metrics"] for r in per_seed.values())}
            if len(d[pk]) < len(fields):
                missing = [f for f in fields if f not in d[pk]]
                print(f"   {art} {label} {pk}: fields missing {missing}")
                complete = False
        if len(d) == len(PROTOCOLS):
            out["arms"][label] = d
        else:
            complete = False
    if isla_arm in out["arms"] and len(out["arms"]) >= 1:
        out["complete"] = complete
        prev = json.load(open(f"{R}/{art}")) if os.path.exists(f"{R}/{art}") else None
        if not complete and prev and prev.get("complete"):
            print(f"== {art}: partial ({len(out['arms'])} arms); the complete artifact on disk (arms {list(prev['arms'])}) is kept")
            continue
        json.dump(out, open(f"{R}/{art}", "w"), indent=1); written.append(art)
        print(f"== {art} ({'complete' if complete else 'partial'})")
        for label, d in out["arms"].items():
            print("   " + f"{label:36s}" + "  ".join(f"{pk}: " + " / ".join(f"{d[pk][f]['mean']:.4f}" for f in fields if f in d[pk]) for pk in PROTOCOLS))
    else:
        print(f"== {art}: not scored yet")
print("written:", written)
