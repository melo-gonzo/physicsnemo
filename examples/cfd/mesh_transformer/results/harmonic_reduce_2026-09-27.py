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


def per_case(run, field, camps=("2026-09-27", "2026-09-22", "2026-09-23")):
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
                     "ext_valonly": f"hm_poisson_ext_valonly_n{n}_seed{{s}}", "ext_distonly": f"hm_poisson_ext_distonly_n{n}_seed{{s}}"}.items():
        ps = {s: r[0] for s in (42, 43) if (r := per_case(pat.format(s=s), "u_l2"))}
        if not ps:
            rung["arms"][arm] = {"status": "not scored"}
            continue
        u = st.mean(ps.values())
        rung["arms"][arm] = {"u": u, "per_seed": ps, "n_seeds": len(ps), "frac": frac(u), "gt512_over": g / u,
                             "frac_per_seed": {s: frac(v) for s, v in ps.items()}}
    e = rung["arms"]["ext"].get("frac")
    if e is not None:
        for arm in ("hm", "ext_valonly", "ext_distonly"):
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
    out["poisson"][str(n)] = rung
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
print("== AirfRANS:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["airfrans"].items() if k in ("ratio", "verdict", "status", "n_seeds")})
