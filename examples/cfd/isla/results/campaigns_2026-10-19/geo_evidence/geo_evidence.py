"""The book's reduction of the exact-geometry evidence (GEO1 and GEO-CONT) into one file, geo_evidence.json.

Every number is a seed mean (seeds 42, 43) of per-case relative L2 averaged over the scored cases (the infer_summary row of
each metrics.jsonl, else the mean of the per-sample rows), float32, final checkpoints, the CHAR scorer. Arms:

* geo1 (results/campaigns_2026-10-14/afu/, AirfRANS uniform interior, 200 full_test cases): afu_geo1_isla (the query geometry
  computed exactly against the full-resolution airfoil, encoding [s, sign(s) log(|s| + 1e-12)], snapshot code_geo1) against
  afu_a2s_isla (the standing's ISLA arm: nearest-boundary-token query geometry, snapshot code_a2s). Both: normalized slice
  reads, width 512, 12 layers, 512 slices, 3.75e-4, 500 epochs, recipe_v11. Protocols native / posed from protocols/, and the
  400-segment reduced-budget protocol (ca2u: a uniform draw of 400 airfoil segments; ca2a: an area-law draw of 400; ca2v: a
  second uniform draw) from ca2_400/. Pre-registration: campaigns_2026-10-17/geo1_airfrans/PREREG.md.
* geo_cont (results/campaigns_2026-10-18/geo_cont/scores/): CONTROL (gcont_ctl_*: exact geometry, encoding [s, sign(s)
  log(|s| + 1e-12)], direction u) against CONT (gcont_cont_*: exact geometry, encoding [s, asinh(s / 1e-6)], direction
  sign(d) u, sign(0) = +1) on AirfRANS uniform (af), the GINOT lug (lug) and PLAID Tensile2d (t2dz); native / area / posed.
  Pre-registration: campaigns_2026-10-18/geo_cont/PREREG.md.
* a2: the standing's ISLA arm in results/char/{airfrans,lug,tensile2d}_a2.json (nearest-token query geometry; on AirfRANS,
  Tensile2d and the lug the stored SDF read as a query input), for the ratios CONT / A2 and CONTROL / A2.

usage: python geo_evidence.py   (writes geo_evidence.json next to this file)"""
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(os.path.dirname(HERE))
SEEDS = (42, 43)
AF = ("pressure_l2", "velocity_l2", "nut_l2")
FIELDS = {"af": AF, "lug": ("von_mises_l2",), "t2dz": ("stress_l2", "displacement_l2", "q_l2")}
A2_FILE = {"af": "airfrans_a2.json", "lug": "lug_a2.json", "t2dz": "tensile2d_a2.json"}


def summary(path):
    by_sample = {}
    for line in open(path):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("phase") == "infer_summary":
            return rec["metrics"], rec.get("num_samples")
        if rec.get("phase") == "infer_step":
            by_sample[rec.get("sample_id", len(by_sample))] = rec["metrics"]
    rows = list(by_sample.values())
    return {k: st.mean(r[k] for r in rows) for k in rows[0]}, len(rows)


def arm(path_of, fields):
    """{field: {mean, per_seed, n_cases}} for one arm and protocol; path_of(seed) -> metrics.jsonl."""
    per = {s: summary(path_of(s)) for s in SEEDS}
    return {f: {"mean": st.mean(per[s][0][f] for s in SEEDS), "per_seed": {s: per[s][0][f] for s in SEEDS},
                "n_cases": min(per[s][1] for s in SEEDS)} for f in fields}


def ratio(a, b):
    return {f: a[f]["mean"] / b[f]["mean"] for f in a}


def main():
    out = {"doc": __doc__.split("usage:")[0].strip()}
    # GEO1
    afu = f"{RES}/campaigns_2026-10-14/afu"
    g1 = {}
    for stem, key in (("afu_geo1_isla", "geo1"), ("afu_a2s_isla", "a2s")):
        g1[key] = {p: arm(lambda s, p=p: f"{afu}/protocols/{stem}_seed{s}/{p}/metrics.jsonl", AF) for p in ("native", "posed")}
        g1[key].update({p: arm(lambda s, p=p: f"{afu}/ca2_400/{stem}_seed{s}/{p}/metrics.jsonl", AF) for p in ("ca2u", "ca2a", "ca2v")})
    g1["geo1_over_a2s"] = {p: ratio(g1["geo1"][p], g1["a2s"][p]) for p in ("native", "posed")}
    g1["geo1_over_a2s_per_seed"] = {f: {s: g1["geo1"]["native"][f]["per_seed"][s] / g1["a2s"]["native"][f]["per_seed"][s]
                                        for s in SEEDS} for f in AF}
    for key in ("geo1", "a2s"):
        d = g1[key]
        g1[f"{key}_effects"] = {
            "law_area_over_uniform": {f: {s: d["ca2a"][f]["per_seed"][s] / d["ca2u"][f]["per_seed"][s] for s in SEEDS} for f in AF},
            "count_uniform_over_native": {f: {s: d["ca2u"][f]["per_seed"][s] / d["native"][f]["per_seed"][s] for s in SEEDS} for f in AF},
            "area_over_native": {f: {s: d["ca2a"][f]["per_seed"][s] / d["native"][f]["per_seed"][s] for s in SEEDS} for f in AF},
            "noise_abs_v_over_u_minus_1": {f: {s: abs(d["ca2v"][f]["per_seed"][s] / d["ca2u"][f]["per_seed"][s] - 1) for s in SEEDS} for f in AF},
        }
    out["geo1_airfrans"] = g1
    # GEO-CONT
    sc = f"{RES}/campaigns_2026-10-18/geo_cont/scores"
    gc = {}
    for ds, fields in FIELDS.items():
        def path(arm_key, p):
            stem = f"gcont_{arm_key}_{ds}_isla"
            base = sc if ds == "t2dz" else f"{sc}/evals_gcont"
            return lambda s: f"{base}/{stem}_seed{s}/{p}/{stem}_seed{s}/metrics.jsonl"
        d = {a: {p: arm(path(a, p), fields) for p in ("native", "area", "posed")} for a in ("ctl", "cont")}
        a2j = json.load(open(f"{RES}/char/{A2_FILE[ds]}"))
        a2 = a2j["arms"][a2j["isla_arm"]]
        d["a2"] = {p: {f: {"mean": a2[p][f]["mean"], "per_seed": {int(k): v for k, v in a2[p][f]["per_seed"].items()}}
                       for f in fields} for p in ("native", "posed")}
        d["cont_over_ctl"] = {p: ratio(d["cont"][p], d["ctl"][p]) for p in ("native", "posed")}
        d["cont_over_a2"] = {p: ratio(d["cont"][p], d["a2"][p]) for p in ("native", "posed")}
        d["ctl_over_a2"] = {p: ratio(d["ctl"][p], d["a2"][p]) for p in ("native", "posed")}
        d["posed_over_native_per_seed"] = {
            a: {f: {s: d[a]["posed"][f]["per_seed"][s] / d[a]["native"][f]["per_seed"][s] for s in SEEDS} for f in fields}
            for a in ("ctl", "cont", "a2")}
        gc[ds] = d
    out["geo_cont"] = gc
    json.dump(out, open(f"{HERE}/geo_evidence.json", "w"), indent=1)
    # print the headline numbers
    print("GEO1 native geo1/a2s", {f: round(v, 3) for f, v in g1["geo1_over_a2s"]["native"].items()})
    for k in ("geo1", "a2s"):
        e = g1[f"{k}_effects"]
        print(k, "law", {f: [round(x, 2) for x in e["law_area_over_uniform"][f].values()] for f in AF[:2]},
              "count", {f: [round(x, 1) for x in e["count_uniform_over_native"][f].values()] for f in AF[:2]})
    for ds in FIELDS:
        print(ds, "cont/ctl", {f: round(v, 3) for f, v in gc[ds]["cont_over_ctl"]["native"].items()},
              "cont/a2", {f: round(v, 3) for f, v in gc[ds]["cont_over_a2"]["native"].items()},
              "posed/native", {a: {f: [round(x, 4) for x in v.values()] for f, v in gc[ds]["posed_over_native_per_seed"][a].items()} for a in ("ctl", "cont", "a2")})


if __name__ == "__main__":
    main()
