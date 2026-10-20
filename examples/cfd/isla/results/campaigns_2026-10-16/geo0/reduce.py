"""GEO0 reduction: per run x protocol, the mean over cases of the per-case relative L2 per field; r = g0a / g0u,
d = |g0v / g0u - 1|, c = g0u / native (rescored here), R = g0a / native, plus the ENV check of the rescored native against
the committed 2026-09-23 records. Writes results.json and RESULTS_TABLE.md and prints the verdicts against PREREG.md.
usage: python reduce.py   (reads evals_lhr/<run>/<protocol>/<run>/metrics.jsonl fetched from $T/kernel/geo0/evals)"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EV = f"{HERE}/evals_lhr"
COMMITTED = "/home/psharpe/gh/physicsnemo/examples/cfd/isla/results/campaigns_2026-09-23/sg/metrics"
F = ["pressure_l2", "velocity_l2", "nut_l2"]
PROTOS = ("native", "g0a", "g0u", "g0v")
RUNS = [(f"{stem}_seed{s}", ds, geo) for ds, stems in (("airfrans", ("sg_af_provided", "sg_af_internal")),
                                                       ("drivaerml", ("sg_drvol_provided", "sg_drvol_internal")))
        for stem in stems for s in (42, 43) for geo in [("provided" if "provided" in stem else "internal")]]


def last_pass(path):
    passes, cur = [], {}
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("phase") == "config":
            if cur:
                passes.append(cur)
            cur = {}
        elif r.get("phase") == "infer_step":
            cur[r["sample_id"]] = r["metrics"]
    if cur:
        passes.append(cur)
    return passes[-1] if passes else {}


def mean(p, keys, f):
    return sum(p[c][f] for c in keys) / len(keys)


out, missing = {}, []
for run, ds, geo in RUNS:
    ### DrivAerML redraws ran as g1a / g1u / g1v (PREREG.md implementation note); g0* there stopped in the guard.
    dirs = {k: (k.replace("g0", "g1") if ds == "drivaerml" else k) for k in PROTOS}
    p = {k: last_pass(f) for k in PROTOS if os.path.exists(f := f"{EV}/{run}/{dirs[k]}/{run}/metrics.jsonl")}
    if not all(p.get(k) for k in PROTOS):
        missing.append(run)
        continue
    keys = sorted(set.intersection(*(set(p[k]) for k in PROTOS)))
    com = last_pass(f"{COMMITTED}/{run}/{run}/metrics.jsonl")
    kc = [c for c in keys if c in com]
    rec = {"dataset": ds, "geometry": geo, "seed": int(run[-2:]), "n_cases": len(keys),
           "n_cases_by_protocol": {k: len(v) for k, v in p.items()}, "n_cases_committed_native": len(com), "fields": {}}
    for f in F:
        m = {k: mean(p[k], keys, f) for k in PROTOS}
        m["committed_native"] = mean(com, kc, f) if kc else float("nan")
        m["native_on_committed_cases"] = mean(p["native"], kc, f) if kc else float("nan")
        rec["fields"][f] = {"mean": m, "r": m["g0a"] / m["g0u"], "d": abs(m["g0v"] / m["g0u"] - 1),
                            "c": m["g0u"] / m["native"], "R": m["g0a"] / m["native"], "cv": m["g0v"] / m["native"],
                            "env": m["native_on_committed_cases"] / m["committed_native"] - 1}
    out[run] = rec

json.dump({"runs": out, "missing": missing}, open(f"{HERE}/results.json", "w"), indent=1)
L = ["| dataset | geometry | seed | field | committed native | native (rescored) | uniform g0u | area g0a | uniform-B g0v | r = a/u | d = abs(v/u - 1) | c = u/native | R = a/native | ENV rescored/committed - 1 |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for run, rec in out.items():
    for f, v in rec["fields"].items():
        m = v["mean"]
        L.append(f"| {rec['dataset']} | {rec['geometry']} | {rec['seed']} | {f[:-3]} | {m['committed_native']:.4g} | {m['native']:.4g} | {m['g0u']:.4g} | "
                 f"{m['g0a']:.4g} | {m['g0v']:.4g} | {v['r']:.4f} | {v['d']:.4f} | {v['c']:.3f} | {v['R']:.4f} | {v['env']:+.4%} |")
open(f"{HERE}/RESULTS_TABLE.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
print("missing:", missing)


def g(stem, s):
    return out.get(f"{stem}_seed{s}")


def verdict(name, ok, why):
    print(f"{name}: {'HOLDS' if ok else 'FAILS'} -- {why}")


S = (42, 43)
if not missing:
    env = max(abs(v["env"]) for rec in out.values() for v in rec["fields"].values())
    verdict("(ENV)", env <= 0.01, f"max |rescored native / committed - 1| = {env:.4%}")
    P, I = "sg_af_provided", "sg_af_internal"
    ok_r = all(g(P, s)["fields"][f]["r"] <= 1.5 for s in S for f in F[:2])
    ok_c = all(g(P, s)["fields"][f]["c"] <= 4 for s in S for f in F)
    verdict("(G0-AF) r <= 1.5 on p, v", ok_r, str({s: [round(g(P, s)["fields"][f]["r"], 3) for f in F[:2]] for s in S}))
    verdict("(G0-AF) c <= 4 on every field", ok_c, str({s: [round(g(P, s)["fields"][f]["c"], 2) for f in F] for s in S}))
    ok = all(g(I, s)["fields"][f]["r"] > 1 + max(0.05, g(I, s)["fields"][f]["d"]) for s in S for f in F[:2])
    verdict("(M-AF) internal r > 1 + max(0.05, d) on p, v", ok, str({s: [(round(g(I, s)["fields"][f]["r"], 3), round(g(I, s)["fields"][f]["d"], 3)) for f in F[:2]] for s in S}))
    ok = all(g(P, s)["fields"][f]["r"] - 1 < g(I, s)["fields"][f]["r"] - 1 for s in S for f in F[:2])
    verdict("(M-AF) provided r - 1 < internal r - 1 on p, v", ok, "")
    P, I = "sg_drvol_provided", "sg_drvol_internal"
    ok = all(0.95 <= g(P, s)["fields"][f]["R"] <= 1.05 for s in S for f in F)
    verdict("(G0-DV) provided R within [0.95, 1.05]", ok, str({s: [round(g(P, s)["fields"][f]["R"], 4) for f in F] for s in S}))
    ok = all(abs(g(P, s)["fields"][f]["R"] - 1) <= max(2 * g(P, s)["fields"][f]["d"], 0.01) for s in S for f in F)
    verdict("(recorded) DV provided R within 1 + max(2d, 0.01)", ok, str({s: [round(g(P, s)["fields"][f]["d"], 4) for f in F] for s in S}))
    ok = all(g(I, s)["fields"][f]["R"] >= 2 for s in S for f in F)
    verdict("(M-DV) internal R >= 2", ok, str({s: [round(g(I, s)["fields"][f]["R"], 3) for f in F] for s in S}))
