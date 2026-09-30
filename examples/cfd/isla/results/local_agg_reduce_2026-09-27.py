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
    if n == 512:
        arms["ext_ctrl_v7"] = f"la_poisson_ext_ctrl_n{n}_seed{{s}}"
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
# --- elasticity twin (added 2026-09-27 14:30): the RANK-IO arm (rank-1 boundary inputs + rank-2 stress head, campaign v2)
# is the best ISLA arm and plays the extension's role; la_elasticity_tensor_ctrl (code_v7, 2,048 only) is the same-snapshot
# control; fields displacement / sig11 / sig22 / sig12; the verdict reads the mean over the four field fractions.
EF = ("displacement", "sig11", "sig22", "sig12")
out["elasticity"] = {}
for n in (512, 2048):
    base = {f: st.mean(ol[f"ol_elasticity_isla_r1_n{n}_seed{s}"]["fields"][f]["mean"] for s in (42, 43)) for f in EF}
    gt = {f: st.mean(ol[f"ol_elasticity_gt512_n{n}_seed{s}"]["fields"][f]["mean"] for s in (42, 43)) for f in EF}
    rung = {"base": base, "gt512": gt, "arms": {}}
    arms = {"rankio": (f"v2_elasticity_isla_r1_rankio_n{n}_seed{{s}}", "2026-09-22"),
            "rankio_local05": (f"la_elasticity_tensor_local05_n{n}_seed{{s}}", "2026-09-27")}
    if n == 2048:
        arms["rankio_ctrl_v7"] = (f"la_elasticity_tensor_ctrl_n{n}_seed{{s}}", "2026-09-27")
    for arm, (pat, camp) in arms.items():
        ps = {}
        for s in (42, 43):
            vals = {}
            for f in EF:
                r = per_case(pat.format(s=s), camp, f"{f}_l2")
                if r:
                    vals[f] = r[0]
            if vals:
                ps[s] = vals
        if not ps:
            rung["arms"][arm] = {"status": "not scored"}
            continue
        m = {f: st.mean(ps[s][f] for s in ps) for f in EF}
        fr = {f: math.log(base[f] / m[f]) / math.log(base[f] / gt[f]) for f in EF}
        rung["arms"][arm] = {"means": m, "per_seed": ps, "n_seeds": len(ps), "frac": fr, "frac_mean": st.mean(fr.values()),
                             "gt512_over": {f: gt[f] / m[f] for f in EF}}
    ref = rung["arms"]["rankio"].get("frac_mean")
    for arm, a in rung["arms"].items():
        if arm == "rankio" or "frac_mean" not in a or ref is None:
            continue
        f = a["frac_mean"]
        a["ratio_to_rankio"] = {k: a["means"][k] / rung["arms"]["rankio"]["means"][k] for k in EF}
        a["verdict"] = ("closes" if f >= 0.95 else "pays" if f >= 0.88 else "null" if abs(f - ref) <= 0.03
                        else "hurts" if f < ref - 0.03 else "between")
        a["interim"] = a["n_seeds"] < 2
    out["elasticity"][str(n)] = rung
# AirfRANS twin (la rows 16-17): the frame-free interior reference (isla_volume_airfrans_ref_provided) + local aggregation
# (rho 0.05, k 32) against the sg_af_provided pair; ratios twin / reference on pressure / velocity / eddy viscosity; pays if any
# <= 0.95 with none > 1.03 (both seeds on the paying field), hurts if any > 1.05, null within 3%, between otherwise.
F = ("pressure_l2", "velocity_l2", "nut_l2")
ref = {s: {f: per_case(f"sg_af_provided_seed{s}", "2026-09-23", f)[0] for f in F} for s in (42, 43) if per_case(f"sg_af_provided_seed{s}", "2026-09-23", F[0])}
tw = {s: {f: r[0] for f in F if (r := per_case(f"la_af_provided_local05_seed{s}", "2026-09-27", f))} for s in (42, 43)}
tw = {s: v for s, v in tw.items() if v}
if ref and tw:
    rm = {f: st.mean(ref[s][f] for s in ref) for f in F}; tm = {f: st.mean(tw[s][f] for s in tw) for f in F}
    ratio = {f: tm[f] / rm[f] for f in F}
    ps = {s: {f: tw[s][f] / ref[s][f] for f in F} for s in tw if s in ref}
    v = ("pays" if any(r <= 0.95 and all(ps[s][f] <= 0.95 for s in ps) for f, r in ratio.items()) and all(r <= 1.03 for r in ratio.values())
         else "hurts" if any(r > 1.05 for r in ratio.values()) else "null" if all(abs(r - 1) <= 0.03 for r in ratio.values()) else "between")
    out["airfrans"] = {"reference": rm, "twin": tm, "ratio": ratio, "per_seed": {"reference": ref, "twin": tw, "ratio": ps}, "n_seeds": len(tw),
                       "verdict": v, "interim": len(tw) < 2}
else:
    out["airfrans"] = {"status": "not scored", "ref_seeds": len(ref), "twin_seeds": len(tw)}
# LOCAL-AGG-STENCIL (la rows 18-21, #sec-nb-local-agg-stencil-prereg): aggregation vs no-aggregation control, both on one fixed query draw
# per airfoil (FixedDrawDomainMeshReader). Supported if aggregation / control <= 1.03 on every field under the fixed draw; refuted if
# >= 1.10 on any field; between otherwise. Also read: fixed-draw control / resampled reference (the augmentation price).
fref = {s: {f: r[0] for f in F if (r := per_case(f"la_af_fixeddraw_ref_seed{s}", "2026-09-27", f))} for s in (42, 43)}; fref = {s: v for s, v in fref.items() if v}
fagg = {s: {f: r[0] for f in F if (r := per_case(f"la_af_fixeddraw_local05_seed{s}", "2026-09-27", f))} for s in (42, 43)}; fagg = {s: v for s, v in fagg.items() if v}
if fref and fagg:
    frm = {f: st.mean(fref[s][f] for s in fref) for f in F}; fam = {f: st.mean(fagg[s][f] for s in fagg) for f in F}
    ratio = {f: fam[f] / frm[f] for f in F}; ps = {s: {f: fagg[s][f] / fref[s][f] for f in F} for s in fagg if s in fref}
    v = "supported" if all(r <= 1.03 for r in ratio.values()) else "refuted" if any(r >= 1.10 for r in ratio.values()) else "between"
    aug = {f: frm[f] / rm[f] for f in F} if ref else None
    resampled = {f: tm[f] / rm[f] for f in F} if (ref and tw) else None
    out["stencil"] = {"fixed_reference": frm, "fixed_aggregation": fam, "ratio": ratio, "per_seed": {"reference": fref, "aggregation": fagg, "ratio": ps},
                      "fixed_control_over_resampled_reference": aug, "resampled_aggregation_over_reference": resampled, "n_seeds": min(len(fref), len(fagg)), "verdict": v}
    print("== STENCIL (fixed draw): agg/ctrl", {f: round(x, 3) for f, x in ratio.items()}, "per seed", {s: {f: round(x, 3) for f, x in r.items()} for s, r in ps.items()}, "->", v)
    print("   fixed ctrl / resampled ref", {f: round(x, 3) for f, x in aug.items()} if aug else None, " resampled agg / ref", {f: round(x, 3) for f, x in resampled.items()} if resampled else None)
else:
    out["stencil"] = {"status": "not scored", "ref_seeds": len(fref), "agg_seeds": len(fagg)}
json.dump(out, open(f"{R}/local_agg_2026-09-27.json", "w"), indent=1)
print("== AirfRANS twin:", {k: ({f: round(x, 3) for f, x in v.items()} if isinstance(v, dict) and k == "ratio" else v) for k, v in out["airfrans"].items() if k in ("ratio", "verdict", "status", "n_seeds")})
for n, rung in out["elasticity"].items():
    print(f"== elasticity {n} cases: base {[round(v, 3) for v in rung['base'].values()]} gt512 {[round(v, 3) for v in rung['gt512'].values()]}")
    for arm, a in rung["arms"].items():
        if "means" in a:
            print(f"  {arm:16s} {[round(v, 4) for v in a['means'].values()]} frac {[round(v, 2) for v in a['frac'].values()]} "
                  f"mean {a['frac_mean']:.3f} {a.get('verdict', '')}{' (one seed)' if a.get('interim') else ''}")
        else:
            print(f"  {arm:16s} {a['status']}")
for n, rung in out["rungs"].items():
    print(f"== {n} cases: base {rung['base']:.4f} gt512 {rung['gt512']:.4f}")
    for arm, a in rung["arms"].items():
        if "u" in a:
            print(f"  {arm:14s} u {a['u']:.4f} per-seed {[round(v, 4) for v in a['per_seed'].values()]} frac {a['frac']:.3f} "
                  f"gt/isla {a['gt512_over']:.3f} {a.get('verdict', '')}{' (one seed)' if a.get('interim') else ''}")
        else:
            print(f"  {arm:14s} {a['status']}")
