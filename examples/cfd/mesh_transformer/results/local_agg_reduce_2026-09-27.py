"""LOCAL-AGG reduction (2026-09-27, #sec-nb-local-agg-prereg): does a contract-preserving local relational aggregation among
tokens (ISLA local_rho / local_k, k-nearest form with k = 32) close the fixed-mesh OLBC Poisson deficit that the boundary-value
extension leaves? Arms: la_poisson_ext_local05_n512, la_poisson_ext_local05_n2048, la_poisson_ext_local15_n2048 (two seeds each,
extension + local aggregation, code_v7). References: ISLA R1 (K = 0) and GeoTransolver-512 per rung from results/olbc_2026-09-22.json;
the extension-only arm from campaign om (om_poisson_isla_r1_ext_n{512,2048}). Readout: float32 per-case relative L2 on u over the
test cases; fractions of the log-gap ln(base/arm)/ln(base/gt512) on the two-seed means. Bars: closes >= 0.95; pays >= 0.88; null
within 0.03 of the extension's fraction; between otherwise; hurts if below the extension by more than 0.03. Lanes not yet scored
are skipped and reported. usage: python local_agg_reduce_2026-09-27.py  (writes local_agg_2026-09-27.json beside this script)."""
import glob, json, math, os, statistics as st

R = os.path.dirname(os.path.abspath(__file__))


def per_case(run, camp, field="u_l2"):
    fs = glob.glob(f"{R}/campaigns_{camp}/*/metrics/{run}/**/metrics.jsonl", recursive=True)
    if not fs:
        return None
    v = [json.loads(l)["metrics"][field] for l in open(sorted(fs)[0]) if '"infer_step"' in l]
    return st.mean(v), len(v)


ol = json.load(open(f"{R}/olbc_2026-09-22.json"))["lanes"]
out = {"protocol": __doc__, "rungs": {}}
for n in (512, 2048):
    base = {s: ol[f"ol_poisson_isla_r1_n{n}_seed{s}"]["fields"]["u"]["mean"] for s in (42, 43)}
    gt = {s: ol[f"ol_poisson_gt512_n{n}_seed{s}"]["fields"]["u"]["mean"] for s in (42, 43)}
    b, g = st.mean(base.values()), st.mean(gt.values())
    frac = lambda u: math.log(b / u) / math.log(b / g)
    rung = {"base": b, "base_per_seed": base, "gt512": g, "gt512_per_seed": gt, "arms": {}}
    arms = {"ext": f"om_poisson_isla_r1_ext_n{n}_seed{{s}}"}
    for rho in ("05", "15"):
        if n == 512 and rho == "15":
            continue
        arms[f"ext_local{rho}"] = f"la_poisson_ext_local{rho}_n{n}_seed{{s}}"
    for arm, pat in arms.items():
        ps, cases = {}, None
        for s in (42, 43):
            camp = "2026-09-22" if arm == "ext" else "2026-09-27"
            r = per_case(pat.format(s=s), camp)
            if r:
                ps[s], cases = r
        if not ps:
            rung["arms"][arm] = {"status": "not scored"}
            continue
        u = st.mean(ps.values())
        rung["arms"][arm] = {"u": u, "per_seed": ps, "n_seeds": len(ps), "n_cases": cases, "frac": frac(u), "gt512_over": g / u,
                             "frac_per_seed": {s: frac(v) for s, v in ps.items()}}
    e = rung["arms"]["ext"].get("frac")
    for arm, a in rung["arms"].items():
        if arm == "ext" or "frac" not in a or e is None:
            continue
        f = a["frac"]
        a["ratio_to_ext"] = a["u"] / rung["arms"]["ext"]["u"]
        a["verdict"] = ("closes" if f >= 0.95 else "pays" if f >= 0.88 else "null" if abs(f - e) <= 0.03
                        else "hurts" if f < e - 0.03 else "between")
        a["interim"] = a["n_seeds"] < 2
    out["rungs"][str(n)] = rung
json.dump(out, open(f"{R}/local_agg_2026-09-27.json", "w"), indent=1)
for n, rung in out["rungs"].items():
    print(f"== {n} cases: base {rung['base']:.4f} gt512 {rung['gt512']:.4f}")
    for arm, a in rung["arms"].items():
        if "u" in a:
            print(f"  {arm:14s} u {a['u']:.4f} per-seed {[round(v, 4) for v in a['per_seed'].values()]} frac {a['frac']:.3f} "
                  f"gt/isla {a['gt512_over']:.3f} {a.get('verdict', '')}{' (one seed)' if a.get('interim') else ''}")
        else:
            print(f"  {arm:14s} {a['status']}")
