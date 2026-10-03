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
                               {"ISLA 768 at 100k cells": "r80_dr_isla_w768_c100k",
                                "GeoTransolver 1,024 at 100k cells": "r80_dr_gt_h1024_c100k",
                                "Transolver 1,024 at 100k cells": "r80_dr_t_h1024_c100k"}, "ISLA 768 at 100k cells"),
    "hilift_surface.json": ({"pressure_l2": "pressure", "tau_wall_l2": "wall shear"},
                            {"ISLA 768 at 100k cells": "r80_hl_isla_w768_c100k",
                             "GeoTransolver 1,024 at 100k cells": "r80_hl_gt_h1024_c100k",
                             "Transolver 1,024 at 100k cells": "r80_hl_t_h1024_c100k"}, "ISLA 768 at 100k cells"),
    "drivaerml_volume.json": ({"pressure_l2": "pressure", "velocity_l2": "velocity", "nut_l2": "eddy viscosity"},
                              {"ISLA 768 at 100k cells": "cl_vol_isla", "GeoTransolver 1,024 at 100k cells": "cl_vol_gt",
                               "Transolver 1,024 at 100k cells": "cl_vol_t"}, "ISLA 768 at 100k cells"),
    "airfrans.json": ({"pressure_l2": "pressure", "velocity_l2": "velocity", "nut_l2": "eddy viscosity"},
                      {"ISLA 768": "cl_af_isla", "GeoTransolver 1,024": "cl_af_gt", "Transolver 1,024": "cl_af_t"}, "ISLA 768"),
    "shift_suv_surface.json": ({"pressure_l2": "pressure", "wss_l2": "wall shear"},
                               {"ISLA 768 at 100k cells": "cl_suv_isla", "GeoTransolver 1,024 at 100k cells": "cl_suv_gt",
                                "Transolver 1,024 at 100k cells": "cl_suv_t"}, "ISLA 768 at 100k cells"),
    "tensile2d.json": ({"sig11_l2": "σ11", "sig22_l2": "σ22", "sig12_l2": "σ12", "displacement_l2": "displacement"},
                       {"ISLA 768": "cl_t2d_isla", "GeoTransolver 1,024": "cl_t2d_gt", "Transolver 1,024": "cl_t2d_t"}, "ISLA 768"),
    "lug.json": ({"von_mises_l2": "von Mises"},
                 {"ISLA 768": "cl_lug_isla", "GeoTransolver 1,024": "cl_lug_gt", "Transolver 1,024": "cl_lug_t"}, "ISLA 768"),
    "deepjeb_surface.json": ({"von_mises_l2": "von Mises", "displacement_l2": "displacement"},
                             {"ISLA 768": "cl_jeb_isla", "GeoTransolver 1,024": "cl_jeb_gt", "Transolver 1,024": "cl_jeb_t"}, "ISLA 768"),
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


written = []
for art, (fields, arms, isla_arm) in DATASETS.items():
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
        json.dump(out, open(f"{R}/{art}", "w"), indent=1); written.append(art)
        print(f"== {art} ({'complete' if complete else 'partial'})")
        for label, d in out["arms"].items():
            print("   " + f"{label:36s}" + "  ".join(f"{pk}: " + " / ".join(f"{d[pk][f]['mean']:.4f}" for f in fields if f in d[pk]) for pk in PROTOCOLS))
    else:
        print(f"== {art}: not scored yet")
print("written:", written)
