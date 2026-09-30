"""HARMONIC + EXT-ABL reduction (2026-09-27, #sec-nb-harmonic-prereg). OLBC Poisson at 512 and 2,048 cases: float32 per-case
relative L2 on u over the 256 test cases; fractions of the log-gap ln(base/arm)/ln(base/gt512) with the frame-free ISLA reference
and GeoTransolver-512 from results/olbc_2026-09-22.json; the inverse-square extension arm from campaign om. Arms (campaign hm,
code_v7/recipe_v4): hm_poisson_hm_n{512,2048} (six harmonic fields), hm_poisson_ext_valonly_n{...} (three value averages),
hm_poisson_ext_distonly_n{...} (three per-type distances); two seeds each. Bars: HARMONIC kernel pays if f_hm >= f_ext + 0.05 on
both rungs, kernel-insensitive within 0.05 on both, hurts below f_ext - 0.05 on either; EXT-ABL values carry (valonly >= 0.75 f_ext
and distonly <= 0.35 f_ext), distances carry (reverse), joint (both <= 0.5 f_ext), either (both >= 0.75 f_ext). AirfRANS half:
hm_af_provided_hm_seed{42,43} against the sg_af_provided pair (campaign 2026-09-23/sg metrics), ratios on pressure / velocity /
eddy viscosity; pays if any <= 0.95 with none > 1.03, null within 3%, hurts if any > 1.05. Unscored lanes are reported as such.
usage: python harmonic_reduce_2026-09-27.py  (writes harmonic_2026-09-27.json beside this script)."""
import glob, json, math, os, statistics as st

R = os.path.dirname(os.path.abspath(__file__))


def per_case(run, field, camps=("2026-09-28", "2026-09-27", "2026-09-22", "2026-09-23")):
    for camp in camps:
        fs = glob.glob(f"{R}/campaigns_{camp}/*/metrics/{run}/**/metrics.jsonl", recursive=True)
        if fs:
            v = [json.loads(l)["metrics"][field] for l in open(sorted(fs)[0]) if '"infer_step"' in l]
            return st.mean(v), len(v)
    return None


ol = json.load(open(f"{R}/olbc_2026-09-22.json"))["lanes"]
out = {"protocol": __doc__, "poisson": {}, "airfrans": {}}
for n in (512, 2048):
    b = st.mean(ol[f"ol_poisson_isla_r1_n{n}_seed{s}"]["fields"]["u"]["mean"] for s in (42, 43))
    g = st.mean(ol[f"ol_poisson_gt512_n{n}_seed{s}"]["fields"]["u"]["mean"] for s in (42, 43))
    frac = lambda u: math.log(b / u) / math.log(b / g)
    rung = {"base": b, "gt512": g, "arms": {}}
    for arm, pat in {"ext": f"om_poisson_isla_r1_ext_n{n}_seed{{s}}", "hm": f"hm_poisson_hm_n{n}_seed{{s}}",
                     "ext_valonly": f"hm_poisson_ext_valonly_n{n}_seed{{s}}", "ext_distonly": f"hm_poisson_ext_distonly_n{n}_seed{{s}}",
                     "hm_measures": f"hm_poisson_hm_measures_n{n}_seed{{s}}", "hm_plus_dist": f"hm_poisson_hm_plus_dist_n{n}_seed{{s}}",
                     "ext_ctrl_v7": f"la_poisson_ext_ctrl_n{n}_seed{{s}}", "ext_local05": f"la_poisson_ext_local05_n{n}_seed{{s}}",
                     "hm_measures_local05": f"hm_poisson_hm_measures_local05_n{n}_seed{{s}}", "ext_distonly_local05": f"hm_poisson_ext_distonly_local05_n{n}_seed{{s}}",
                     "nd_distonly_2d": f"nd_poisson_ext_distonly_2d_n{n}_seed{{s}}",
                     "lpr_distonly_local05": f"lw_poisson_ext_distonly_local05_lpr_n{n}_seed{{s}}"}.items():
        ps = {s: r[0] for s in (42, 43) if (r := per_case(pat.format(s=s), "u_l2"))}
        if not ps:
            rung["arms"][arm] = {"status": "not scored"}
            continue
        u = st.mean(ps.values())
        rung["arms"][arm] = {"u": u, "per_seed": ps, "n_seeds": len(ps), "frac": frac(u), "gt512_over": g / u,
                             "frac_per_seed": {s: frac(v) for s, v in ps.items()}}
    e = rung["arms"]["ext"].get("frac")
    if e is not None:
        for arm in ("hm", "ext_valonly", "ext_distonly", "hm_measures", "hm_plus_dist", "ext_ctrl_v7", "ext_local05"):
            a = rung["arms"][arm]
            if "frac" in a:
                a["frac_minus_ext"] = a["frac"] - e
                a["share_of_ext"] = a["frac"] / e
                a["interim"] = a["n_seeds"] < 2
        h = rung["arms"]["hm"]
        if "frac" in h:
            d = h["frac"] - e
            h["verdict"] = "kernel pays" if d >= 0.05 else "hurts" if d <= -0.05 else "kernel-insensitive"
        v, dd = rung["arms"]["ext_valonly"], rung["arms"]["ext_distonly"]
        if "frac" in v and "frac" in dd:
            sv, sd = v["frac"] / e, dd["frac"] / e
            rung["ext_abl_verdict"] = ("values carry it" if sv >= 0.75 and sd <= 0.35 else "distances carry it" if sd >= 0.75 and sv <= 0.35
                                       else "joint" if sv <= 0.5 and sd <= 0.5 else "either suffices" if sv >= 0.75 and sd >= 0.75 else "between")
    d0 = rung["arms"]["ext_distonly"]
    if "u" in d0:
        for arm in ("hm_measures", "hm_plus_dist", "hm", "hm_measures_local05", "ext_distonly_local05", "nd_distonly_2d"):
            a = rung["arms"][arm]
            if "u" in a:
                a["ratio_to_distonly"] = a["u"] / d0["u"]
                a["ratio_to_distonly_per_seed"] = {s_: a["per_seed"][s_] / d0["per_seed"][s_] for s_ in a["per_seed"] if s_ in d0["per_seed"]}
        # COMBINED (hm rows 22-29): per-type channel + local aggregation; pays if <= 0.95 of the channel-only arm on both seeds
        comb = {}
        for arm, ref in (("ext_distonly_local05", "ext_distonly"), ("hm_measures_local05", "hm_measures")):
            a, r_ = rung["arms"][arm], rung["arms"][ref]
            if "u" in a and "u" in r_:
                ratio = a["u"] / r_["u"]; ps_ = {s_: a["per_seed"][s_] / r_["per_seed"][s_] for s_ in a["per_seed"] if s_ in r_["per_seed"]}
                comb[arm] = {"ratio_to_channel_only": ratio, "per_seed": ps_, "vs_local05": a["u"] / rung["arms"]["ext_local05"]["u"] if "u" in rung["arms"]["ext_local05"] else None,
                             "verdict": "pays" if ratio <= 0.95 and all(v <= 0.95 for v in ps_.values()) else "hurts" if ratio >= 1.05 else "null within 5%"}
        rung["combined"] = comb
        # ND-GENERIC pass 2: native 2-D vs the embedded 3-D distances-only arm; within 0.03 = validated
        # LOCAL-LPR fixed-mesh regression check: lpr weights vs Gaussian weights, both with the per-type distances
        lp, ga = rung["arms"]["lpr_distonly_local05"], rung["arms"]["ext_distonly_local05"]
        if "u" in lp and "u" in ga:
            r_ = lp["u"] / ga["u"]
            lp["ratio_to_gaussian"] = r_
            lp["ratio_to_gaussian_per_seed"] = {s_: lp["per_seed"][s_] / ga["per_seed"][s_] for s_ in lp["per_seed"] if s_ in ga["per_seed"]}
            lp["lpr_verdict"] = "within 0.05 (no fixed-mesh cost)" if abs(r_ - 1) <= 0.05 else "pays on the fixed mesh" if r_ < 0.95 else "fixed-mesh regression"
        nd = rung["arms"]["nd_distonly_2d"]
        if "u" in nd:
            r_ = nd["ratio_to_distonly"]
            nd["nd_verdict"] = "validated (within 0.03)" if abs(r_ - 1) <= 0.03 else "embedding costs" if r_ < 0.97 else "defect or capacity effect"
        m, pd_, h6 = rung["arms"]["hm_measures"], rung["arms"]["hm_plus_dist"], rung["arms"]["hm"]
        fu = {}
        if "u" in m and "u" in h6:
            fu["values_hurt"] = m["u"] <= 0.95 * h6["u"]
        if "u" in m and "u" in pd_:
            fu["measures_redundant"] = abs(m["u"] / d0["u"] - 1) <= 0.03 and abs(pd_["u"] / d0["u"] - 1) <= 0.03
            fu["measures_add"] = pd_["u"] <= 0.95 * d0["u"] and all(pd_["per_seed"][s_] <= 0.95 * d0["per_seed"][s_] for s_ in pd_["per_seed"] if s_ in d0["per_seed"])
        rung["followup"] = fu
    out["poisson"][str(n)] = rung
# HARMONIC-EL (2026-09-28, #sec-nb-harmonic-elasticity-prereg): the per-component geometry channel on OLBC elasticity on top of the
# RANK-IO arm. Bases: 512 = the code_v2 RANK-IO record, 2048 = the code_v7 control (both in local_agg_2026-09-27.json); GT-512 per
# field from the same artifact. Ratios arm / base per field; stress mean over sig11/sig22/sig12; bars: pays <= 0.95 on every stress
# on both seeds, null within 3%, hurts > 1.05 on any stress; harmonic / distances <= 0.95 on the stress mean at 2048 = edge reproduces.
EF = ("displacement", "sig11", "sig22", "sig12")
la = json.load(open(f"{R}/local_agg_2026-09-27.json"))["elasticity"]
out["elasticity"] = {}
def ctrl_v7(n):
    """Same-snapshot (code_v7) control pair hm_elasticity_tensor_ctrl_n{n}; None until both seeds are scored."""
    ps = {}
    for sd in (42, 43):
        vals = {f: r[0] for f in EF if (r := per_case(f"hm_elasticity_tensor_ctrl_n{n}_seed{sd}", f"{f}_l2"))}
        if len(vals) == len(EF):
            ps[str(sd)] = vals
    return {"means": {f: st.mean(v[f] for v in ps.values()) for f in EF}, "per_seed": ps} if len(ps) == 2 else None


for n, base_arm in ((512, "rankio"), (2048, "rankio_ctrl_v7")):
    base = la[str(n)]["arms"][base_arm]; gt = la[str(n)]["gt512"]
    if n == 512 and (c := ctrl_v7(512)):   # 2026-09-29: the 512 control (hm rows 40-41) replaces the cross-snapshot code_v2 record
        base, base_arm = c, "ctrl_v7_hm40"
    rung = {"base_arm": base_arm, "base": base["means"], "base_per_seed": base["per_seed"], "gt512": gt, "arms": {}}
    for arm, pat in {"hm_measures": f"hm_elasticity_tensor_hm_measures_n{n}_seed{{s}}", "distonly": f"hm_elasticity_tensor_distonly_n{n}_seed{{s}}",
                     "hm_measures_local05": f"hm_elasticity_tensor_hm_measures_local05_n{n}_seed{{s}}"}.items():
        ps = {}
        for sd in (42, 43):
            vals = {f: r[0] for f in EF if (r := per_case(pat.format(s=sd), f"{f}_l2"))}
            if len(vals) == len(EF):
                ps[str(sd)] = vals
        if not ps:
            rung["arms"][arm] = {"status": "not scored"}; continue
        means = {f: st.mean(v[f] for v in ps.values()) for f in EF}
        ratio = {f: means[f] / base["means"][f] for f in EF}
        ratio_ps = {sd: {f: ps[sd][f] / base["per_seed"][sd][f] for f in EF} for sd in ps if sd in base["per_seed"]}
        stress = [f for f in EF if f != "displacement"]
        smean = st.mean(ratio[f] for f in stress)
        verdict = ("pays" if all(ratio[f] <= 0.95 for f in stress) and all(ratio_ps[sd][f] <= 0.95 for sd in ratio_ps for f in stress)
                   else "hurts" if any(ratio[f] > 1.05 for f in stress) else "null" if all(abs(ratio[f] - 1) <= 0.03 for f in stress) else "between")
        rung["arms"][arm] = {"means": means, "per_seed": ps, "n_seeds": len(ps), "ratio_to_base": ratio, "ratio_per_seed": ratio_ps,
                             "stress_mean_ratio": smean, "gt512_over": {f: gt[f] / means[f] for f in EF}, "verdict": verdict, "interim": len(ps) < 2}
    h, d0 = rung["arms"]["hm_measures"], rung["arms"]["distonly"]
    if "means" in h and "means" in d0:
        rung["harmonic_over_distances_stress"] = st.mean(h["means"][f] / d0["means"][f] for f in ("sig11", "sig22", "sig12"))
    out["elasticity"][str(n)] = rung
F = ("pressure_l2", "velocity_l2", "nut_l2")
ref = {s: {f: per_case(f"sg_af_provided_seed{s}", f)[0] for f in F} for s in (42, 43) if per_case(f"sg_af_provided_seed{s}", F[0])}
hm = {s: {f: r[0] for f in F if (r := per_case(f"hm_af_provided_hm_seed{s}", f))} for s in (42, 43)}
hm = {s: v for s, v in hm.items() if v}
if ref and hm:
    rm = {f: st.mean(ref[s][f] for s in ref) for f in F}
    hmm = {f: st.mean(hm[s][f] for s in hm) for f in F}
    ratio = {f: hmm[f] / rm[f] for f in F}
    v = ("pays" if any(r <= 0.95 for r in ratio.values()) and all(r <= 1.03 for r in ratio.values())
         else "hurts" if any(r > 1.05 for r in ratio.values()) else "null" if all(abs(r - 1) <= 0.03 for r in ratio.values()) else "between")
    out["airfrans"] = {"provided": rm, "hm": hmm, "ratio": ratio, "per_seed": {"provided": ref, "hm": hm}, "n_seeds": len(hm), "verdict": v,
                       "interim": len(hm) < 2}
else:
    out["airfrans"] = {"status": "not scored", "ref_seeds": len(ref), "hm_seeds": len(hm)}
json.dump(out, open(f"{R}/harmonic_2026-09-27.json", "w"), indent=1)
for n, rung in out["poisson"].items():
    print(f"== Poisson {n}: base {rung['base']:.4f} gt512 {rung['gt512']:.4f}" + (f"  EXT-ABL: {rung['ext_abl_verdict']}" if "ext_abl_verdict" in rung else ""))
    for arm, a in rung["arms"].items():
        print(f"  {arm:13s} " + (f"u {a['u']:.4f} seeds {[round(x, 4) for x in a['per_seed'].values()]} frac {a['frac']:.3f} {a.get('verdict', '')}"
                                 f"{' (one seed)' if a.get('interim') else ''}" if "u" in a else a["status"]))
for n, rung in out["elasticity"].items():
    print(f"== Elasticity {n} (base {rung['base_arm']}): base {[round(rung['base'][f],4) for f in EF]} gt512 {[round(rung['gt512'][f],4) for f in EF]}")
    for arm, a in rung["arms"].items():
        print(f"  {arm:20s} " + (f"{[round(a['means'][f],4) for f in EF]} ratio {[round(a['ratio_to_base'][f],3) for f in EF]} stress {a['stress_mean_ratio']:.3f} {a['verdict']}{' (one seed)' if a['interim'] else ''}" if "means" in a else a["status"]))
    if "harmonic_over_distances_stress" in rung: print(f"  harmonic/distances stress {rung['harmonic_over_distances_stress']:.3f}")
print("== AirfRANS:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["airfrans"].items() if k in ("ratio", "verdict", "status", "n_seeds")})
