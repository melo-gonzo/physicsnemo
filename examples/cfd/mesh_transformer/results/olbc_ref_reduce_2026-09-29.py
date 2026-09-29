"""OLBC re-referencing (2026-09-29, #sec-nb-olbc-reference-correction, #sec-nb-olbc-ref-lanes): every OLBC number in chapter 10 against
current-snapshot ISLA references. Poisson: reference pairs pc_poisson_ref_n{512,2048,8448} (code_v11); GeoTransolver-512 physics attention
(ol_poisson_gt512_n*, olbc_2026-09-22.json), GeoTransolver-512 FLARE++ (gf_poisson_gt512_fpp_n*), GeoTransolver-256 at 8,448; ablation arms on
code_v11 (pc_poisson_{distonly,hm_measures,distonly_local05}_n*) as ratios to the same-snapshot reference; the code_v7 arms are listed for the
record only. Elasticity: reference pairs pc_elasticity_ref_n{512,2048,8448} (code_v11) when scored, else the code_v7 controls
(hm_elasticity_tensor_ctrl_n512, la_elasticity_tensor_ctrl_n2048) marked interim; GeoTransolver-512 physics attention and FLARE++
(gf_elasticity_gt512_fpp_n*); arm pc_elasticity_hm_measures_n* (code_v11) and hm_elasticity_tensor_hm_measures_n* (code_v7). The headline
GeoTransolver-512 baseline per rung is the STRONGER backend at that rung (@sec-nb-gt-fpp-poisson-512). Float32, 256 test cases, two seeds;
seed means with per-seed values. usage: python olbc_ref_reduce_2026-09-29.py (writes olbc_ref_2026-09-29.json)."""
import glob, json, os, statistics as st

R = os.path.dirname(os.path.abspath(__file__))
PF = ("u_l2",); EF = ("displacement_l2", "sig11_l2", "sig22_l2", "sig12_l2")


def per_case(run, fields):
    fs = glob.glob(f"{R}/campaigns_*/*/metrics/{run}/**/metrics.jsonl", recursive=True)
    if not fs:
        return None
    rows = [json.loads(l)["metrics"] for l in open(sorted(fs)[0]) if '"infer_step"' in l]
    return {f: st.mean(r[f] for r in rows) for f in fields if f in rows[0]}


def pair(pat, fields):
    d = {str(s): per_case(pat.format(s=s), fields) for s in (42, 43)}
    d = {s: v for s, v in d.items() if v}
    if not d:
        return None
    return {"means": {f: st.mean(v[f] for v in d.values()) for f in fields}, "per_seed": d, "n_seeds": len(d)}


ol = json.load(open(f"{R}/olbc_2026-09-22.json"))["lanes"]


def ol_pair(prefix, n, fields_map):
    d = {}
    for s in (42, 43):
        k = f"{prefix}_n{n}_seed{s}"
        if k in ol:
            d[str(s)] = {f: ol[k]["fields"][fm]["mean"] for f, fm in fields_map.items()}
    return {"means": {f: st.mean(v[f] for v in d.values()) for f in d["42"]}, "per_seed": d, "n_seeds": len(d)} if d else None


def ratio(a, b, fields):
    r = {f: a["means"][f] / b["means"][f] for f in fields}
    ps = {s: {f: a["per_seed"][s][f] / b["per_seed"][s][f] for f in fields} for s in a["per_seed"] if s in b["per_seed"]}
    return {"ratio": r, "per_seed": ps}


out = {"protocol": __doc__, "poisson": {}, "elasticity": {}}
for n in (512, 2048, 8448):
    ref = pair(f"pc_poisson_ref_n{n}_seed{{s}}", PF)
    gt = ol_pair("ol_poisson_gt512", n, {"u_l2": "u"}); gt256 = ol_pair("ol_poisson_gt256", n, {"u_l2": "u"})
    fpp = pair(f"gf_poisson_gt512_fpp_n{n}_seed{{s}}", PF)
    rung = {"reference": ref, "gt512_physics": gt, "gt512_flarepp": fpp, "gt256_physics": gt256, "arms": {}}
    cands = {k: v for k, v in (("physics", gt), ("flarepp", fpp)) if v}
    if cands:
        best = min(cands, key=lambda k: cands[k]["means"]["u_l2"]); rung["stronger_gt512"] = best
        if ref:
            rung["gt512_over_isla"] = {k: ratio(v, ref, PF) for k, v in cands.items()}
            rung["gt512_over_isla"]["stronger"] = rung["gt512_over_isla"][best]
    if ref:
        for arm, pat in {"distances": f"pc_poisson_distonly_n{n}_seed{{s}}", "harmonic_measures": f"pc_poisson_hm_measures_n{n}_seed{{s}}",
                         "distances_local05": f"pc_poisson_distonly_local05_n{n}_seed{{s}}"}.items():
            a = pair(pat, PF); rung["arms"][arm] = ({**a, **ratio(a, ref, PF)} if a else {"status": "not scored"})
        for arm, pat in {"v7_distances": f"hm_poisson_ext_distonly_n{n}_seed{{s}}", "v7_harmonic_measures": f"hm_poisson_hm_measures_n{n}_seed{{s}}",
                         "v7_distances_local05": f"hm_poisson_ext_distonly_local05_n{n}_seed{{s}}", "v7_extension": f"la_poisson_ext_ctrl_n{n}_seed{{s}}"}.items():
            a = pair(pat, PF); rung["arms"][arm] = ({**a, **ratio(a, ref, PF), "cross_snapshot": True} if a else {"status": "not scored"})
    out["poisson"][str(n)] = rung
for n in (512, 2048, 8448):
    ref = pair(f"pc_elasticity_ref_n{n}_seed{{s}}", EF); interim = False
    if not ref and n == 512:
        ref = pair("hm_elasticity_tensor_ctrl_n512_seed{s}", EF); interim = True
    if not ref and n == 2048:
        ref = pair("la_elasticity_tensor_ctrl_n2048_seed{s}", EF); interim = True
    gt = ol_pair("ol_elasticity_gt512", n, {"displacement_l2": "displacement", "sig11_l2": "sig11", "sig22_l2": "sig22", "sig12_l2": "sig12"})
    fpp = pair(f"gf_elasticity_gt512_fpp_n{n}_seed{{s}}", EF)
    rung = {"reference": ref, "reference_interim_code_v7": interim, "gt512_physics": gt, "gt512_flarepp": fpp, "arms": {}}
    cands = {k: v for k, v in (("physics", gt), ("flarepp", fpp)) if v}
    if cands:
        best = min(cands, key=lambda k: st.mean(cands[k]["means"][f] for f in EF[1:])); rung["stronger_gt512"] = best
        if ref:
            rung["gt512_over_isla"] = {k: ratio(v, ref, EF) for k, v in cands.items()}
            rung["gt512_over_isla"]["stronger"] = rung["gt512_over_isla"][best]
    if ref:
        for arm, pat, cross in (("harmonic_measures", f"pc_elasticity_hm_measures_n{n}_seed{{s}}", False), ("v7_harmonic_measures", f"hm_elasticity_tensor_hm_measures_n{n}_seed{{s}}", True),
                                ("v7_distances", f"hm_elasticity_tensor_distonly_n{n}_seed{{s}}", True)):
            a = pair(pat, EF); rung["arms"][arm] = ({**a, **ratio(a, ref, EF), "cross_snapshot": cross} if a else {"status": "not scored"})
    out["elasticity"][str(n)] = rung
json.dump(out, open(f"{R}/olbc_ref_2026-09-29.json", "w"), indent=1)
for set_ in ("poisson", "elasticity"):
    for n, r in out[set_].items():
        ref = r["reference"]; print(f"== {set_} {n}: ref", None if not ref else {f[:-3]: round(v, 4) for f, v in ref["means"].items()}, "interim" if r.get("reference_interim_code_v7") else "",
              "| stronger GT:", r.get("stronger_gt512"), "| GT/ISLA:", None if "gt512_over_isla" not in r else {k: {f[:-3]: round(v, 2) for f, v in r["gt512_over_isla"][k]["ratio"].items()} for k in r["gt512_over_isla"] if k != "stronger"})
        for a, v in r["arms"].items():
            if "ratio" in v: print(f"   {a:24s}", {f[:-3]: round(x, 3) for f, x in v["ratio"].items()}, "cross-snapshot" if v.get("cross_snapshot") else "")
