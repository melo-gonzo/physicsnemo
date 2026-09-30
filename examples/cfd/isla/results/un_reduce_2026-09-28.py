"""UNIVERSAL-AGG reduction (2026-09-28, #sec-nb-universal-prereg): the local aggregation switched on everywhere (rho 0.05, k 32) against
the same-snapshot reference configuration (code_v10), on the three datasets the aggregation's rule was not fitted on: Tensile2d at 400
cases (un_t2d_{ref,local05}_n400), the lug at 2,400 cases (un_lug_{ref,local05}_n2400) and DrivAerML Volume (un_drvol_{ref,local05}).
Readout: float32 per-case relative L2 per field (test split), two seeds; ratio aggregation / reference per field on seed means and per
seed. Bars: default-on candidate if <= 1.03 on every field of every dataset; not part of the model if > 1.05 on any field of any
dataset; between otherwise. usage: python un_reduce_2026-09-28.py  (writes universal_agg_2026-09-28.json)."""
import glob, json, os, statistics as st

R = os.path.dirname(os.path.abspath(__file__))
SETS = {"t2d": ("un_t2d_{arm}_n400_seed{s}", None), "lug": ("un_lug_{arm}_n2400_seed{s}", None), "drvol": ("un_drvol_{arm}_seed{s}", None)}


def per_case(run):
    fs = glob.glob(f"{R}/campaigns_2026-09-28/un/metrics/{run}/**/metrics.jsonl", recursive=True)
    if not fs:
        return None
    rows = [json.loads(l)["metrics"] for l in open(sorted(fs)[0]) if '"infer_step"' in l]
    keys = [k for k in rows[0] if k.endswith("_l2")]
    return {k: st.mean(r[k] for r in rows) for k in keys}


out = {"protocol": __doc__, "datasets": {}}
worst = 0.0; any_scored = False; complete = True
for name, (pat, _) in SETS.items():
    ref = {s: per_case(pat.format(arm="ref", s=s)) for s in (42, 43)}; ref = {s: v for s, v in ref.items() if v}
    agg = {s: per_case(pat.format(arm="local05", s=s)) for s in (42, 43)}; agg = {s: v for s, v in agg.items() if v}
    if not ref or not agg:
        out["datasets"][name] = {"status": "not scored", "ref_seeds": len(ref), "agg_seeds": len(agg)}; complete = False; continue
    fields = sorted(set.intersection(*(set(v) for v in list(ref.values()) + list(agg.values()))))
    # primary fields only: drop per-component copies (_x/_y/_z), the aggregate stress_l2 (its components are scored) and zero fields (planar z)
    fields = [f for f in fields if not f.endswith(("_x_l2", "_y_l2", "_z_l2")) and f != "stress_l2" and st.mean(ref[s][f] for s in ref) > 0]
    rm = {f: st.mean(ref[s][f] for s in ref) for f in fields}; am = {f: st.mean(agg[s][f] for s in agg) for f in fields}
    ratio = {f: am[f] / rm[f] for f in fields}
    ps = {s: {f: agg[s][f] / ref[s][f] for f in fields} for s in agg if s in ref}
    v = "default-on candidate" if all(r <= 1.03 for r in ratio.values()) else "not part of the model" if any(r > 1.05 for r in ratio.values()) else "between"
    out["datasets"][name] = {"fields": fields, "reference": rm, "aggregation": am, "ratio": ratio, "per_seed": {"reference": ref, "aggregation": agg, "ratio": ps},
                             "n_seeds": {"reference": len(ref), "aggregation": len(agg)}, "verdict": v, "interim": min(len(ref), len(agg)) < 2}
    worst = max(worst, max(ratio.values())); any_scored = True
    if min(len(ref), len(agg)) < 2: complete = False
if any_scored:
    out["overall"] = {"worst_ratio": worst, "complete": complete,
                      "verdict": ("not part of the model" if worst > 1.05 else ("default-on candidate" if worst <= 1.03 and complete else "between" if complete else "pending"))}
json.dump(out, open(f"{R}/universal_agg_2026-09-28.json", "w"), indent=1)
for name, d in out["datasets"].items():
    if "ratio" in d:
        print(f"== {name}: " + "  ".join(f"{f} ref {d['reference'][f]:.4f} agg {d['aggregation'][f]:.4f} ratio {d['ratio'][f]:.3f}" for f in d["fields"]) + f"  -> {d['verdict']}{' (interim)' if d['interim'] else ''}")
        for s, r in d["per_seed"]["ratio"].items(): print(f"   seed {s}: " + " ".join(f"{f} {x:.3f}" for f, x in r.items()))
    else:
        print(f"== {name}: {d}")
print("overall:", out.get("overall"))
