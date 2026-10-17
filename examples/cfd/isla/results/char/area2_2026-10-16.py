"""CHAR-AREA2 reduction: per run x protocol, the mean over cases of the per-case relative L2 per field; r = ca2a / ca2u,
noise floor d = |ca2v / ca2u - 1|, count effect c = ca2u / native (native = the committed CHAR / AF-SCORE record, last
pass). Writes results.json and RESULTS_TABLE.md; prints the verdicts against PREREG.md (amendment).

Run from scratch/isla-char-area2-2026-10-16/ (evals_lhr/ holds the copied LHR metrics); its results.json is
committed here unchanged as area2_2026-10-16.json."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EV = f"{HERE}/evals_lhr"
RES = "/home/psharpe/gh/physicsnemo/examples/cfd/isla/results"
NATIVE = {"afu": f"{RES}/campaigns_2026-10-14/afu/protocols", "char": f"{RES}/campaigns_2026-10-02/char/protocols"}
F_AF, F_T2D, F_LUG, F_JEB = ["pressure_l2", "velocity_l2", "nut_l2"], ["stress_l2", "displacement_l2", "q_l2"], ["von_mises_l2"], ["von_mises_l2", "displacement_l2"]
LANES = [  # dataset, arch, stem, fields, native root
    ("airfrans", "ISLA", "afu_r_isla", F_AF, "afu"), ("airfrans", "ISLA-DTT", "afu_dtt", F_AF, "afu"),
    ("tensile2d", "ISLA", "cl_r_t2dz_isla", F_T2D, "char"), ("lug", "ISLA", "cl_r_lug_isla", F_LUG, "char"),
    ("deepjeb", "ISLA", "cl_jd_rate_e40", F_JEB, "char"), ("deepjeb", "GeoTransolver", "cl_r_jeb_gt", F_JEB, "char"),
    ("deepjeb", "Transolver", "cl_r_jeb_t", F_JEB, "char"),
]


def last_pass(path):
    passes, cur, cfg = [], {}, None
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("phase") == "config":
            if cur:
                passes.append(cur)
            cur, cfg = {}, r
        elif r.get("phase") == "infer_step":
            cur[r["sample_id"]] = r["metrics"]
    if cur:
        passes.append(cur)
    return (passes[-1] if passes else {}), cfg


out, missing = {}, []
for ds, arch, stem, fields, nat in LANES:
    for s in (42, 43):
        run = f"{stem}_seed{s}"
        p, cfgs = {}, {}
        for proto in ("ca2a", "ca2u", "ca2v"):
            f = f"{EV}/{run}/{proto}/{run}/metrics.jsonl"
            if os.path.exists(f):
                p[proto], cfgs[proto] = last_pass(f)
        nf = f"{NATIVE[nat]}/{run}/native/metrics.jsonl"
        if os.path.exists(nf):
            p["native"], cfgs["native"] = last_pass(nf)
        if not all(k in p and p[k] for k in ("ca2a", "ca2u", "ca2v")):
            missing.append(run)
            continue
        keys = sorted(set(p["ca2a"]) & set(p["ca2u"]) & set(p["ca2v"]))
        rec = {"dataset": ds, "arch": arch, "seed": s, "n_cases": len(keys),
               "n_cases_by_protocol": {k: len(v) for k, v in p.items()},
               "datasets_used": {k: (v or {}).get("dataset") for k, v in cfgs.items()}, "fields": {}}
        for fld in fields:
            m = {k: sum(p[k][c][fld] for c in keys) / len(keys) for k in ("ca2a", "ca2u", "ca2v")}
            if "native" in p:
                kn = [c for c in keys if c in p["native"]]
                m["native"] = sum(p["native"][c][fld] for c in kn) / len(kn) if kn else float("nan")
                m["native_on_ca2u_cases"] = sum(p["ca2u"][c][fld] for c in kn) / len(kn) if kn else float("nan")
            r = m["ca2a"] / m["ca2u"]
            d = abs(m["ca2v"] / m["ca2u"] - 1)
            c = m["native_on_ca2u_cases"] / m["native"] if "native" in m else float("nan")
            rec["fields"][fld] = {"mean": m, "r": r, "d": d, "count_effect": c}
        out[run] = rec

json.dump({"runs": out, "missing": missing}, open(f"{HERE}/results.json", "w"), indent=1)

lines = ["| dataset | arch | seed | field | native | uniform (ca2u) | area (ca2a) | uniform-B (ca2v) | r = a/u | d = abs(v/u - 1) | count u/native |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for run, rec in out.items():
    for fld, v in rec["fields"].items():
        m = v["mean"]
        lines.append(f"| {rec['dataset']} | {rec['arch']} | {rec['seed']} | {fld[:-3]} | {m.get('native', float('nan')):.4g} | {m['ca2u']:.4g} | "
                     f"{m['ca2a']:.4g} | {m['ca2v']:.4g} | {v['r']:.4f} | {v['d']:.4f} | {v['count_effect']:.3f} |")
open(f"{HERE}/RESULTS_TABLE.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("missing:", missing)


def get(stem, s):
    return out.get(f"{stem}_seed{s}")


def verdict(name, ok, why):
    print(f"{name}: {'HOLDS' if ok else 'FAILS'} -- {why}")


S = (42, 43)
if all(get(x, s) for x in ("cl_jd_rate_e40", "cl_r_jeb_gt", "cl_r_jeb_t") for s in S):
    isla_ok = all(0.95 <= get("cl_jd_rate_e40", s)["fields"][f]["r"] <= 1.05 for s in S for f in F_JEB)
    base_ok = {}
    for b in ("cl_r_jeb_gt", "cl_r_jeb_t"):
        base_ok[b] = any(all(abs(get(b, s)["fields"][f]["r"] - 1) > abs(get("cl_jd_rate_e40", s)["fields"][f]["r"] - 1)
                             and abs(get(b, s)["fields"][f]["r"] - 1) > get(b, s)["fields"][f]["d"] for s in S) for f in F_JEB)
    verdict("(S) DeepJEB++", isla_ok and all(base_ok.values()), f"ISLA within [0.95,1.05] every field/seed: {isla_ok}; baselines: {base_ok}")
if all(get("afu_r_isla", s) for s in S):
    ok = all(get("afu_r_isla", s)["fields"][f]["r"] > max(1.05, 1 + get("afu_r_isla", s)["fields"][f]["d"]) for s in S for f in F_AF)
    verdict("(B-AF) AirfRANS afu ISLA", ok, "r > 1.05 and r > 1 + d on p, velocity, nut, both seeds")
if all(get("cl_r_lug_isla", s) for s in S):
    ok = all(abs(get("cl_r_lug_isla", s)["fields"]["von_mises_l2"]["r"] - 1) <= max(2 * get("cl_r_lug_isla", s)["fields"]["von_mises_l2"]["d"], 0.05) for s in S)
    verdict("(B-lug) lug ISLA null", ok, "|r - 1| <= max(2d, 0.05) on von_mises, both seeds")
if all(get("cl_r_t2dz_isla", s) for s in S):
    ok = all(0.95 <= get("cl_r_t2dz_isla", s)["fields"][f]["r"] <= 1.05 for s in S for f in F_T2D)
    verdict("(N-T2D) Tensile2d ISLA null", ok, "r within [0.95, 1.05] on stress, displacement, q, both seeds")
