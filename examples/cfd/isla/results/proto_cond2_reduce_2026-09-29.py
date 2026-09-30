"""PROTO-COND-2 reduction (2026-09-29, #sec-nb-proto-cond-2-prereg): centered, scale-bounded conditioned slice prototypes
(ISLA slice_prototypes='centered', campaign pc2 on code_v12; arms pc2_*_cent_* against pc2_*_ref_*) against the static reference on the same snapshot (code_v11), campaign pc: OLBC Poisson 512 and
2,048 (pc_poisson_{ref,cond}_n{512,2048}), DrivAerML Surface 435 at 200k (pc_dr_{ref,cond}), HiLift 510 medium at 200k (pc_hl_{ref,cond}_medium),
AirfRANS interior (pc_af_{ref,cond}); two seeds each. Readout: float32 per-case relative L2 per primary field; ratio conditioned / reference
on seed means and per seed. Bars (two-seed means): reference candidate = <= 0.97 on >= 1 dataset and <= 1.03 on every field of every
dataset; deleted = > 1.05 on any field of any dataset, or no dataset <= 0.97; between otherwise. usage: python proto_cond_reduce_2026-09-29.py"""
import glob, json, os, statistics as st

R = os.path.dirname(os.path.abspath(__file__))
SETS = {"poisson_512": "pc2_poisson_{arm}_n512_seed{s}", "poisson_2048": "pc2_poisson_{arm}_n2048_seed{s}", "drivaer_435": "pc2_dr_{arm}_seed{s}",
        "hilift_510": "pc2_hl_{arm}_medium_seed{s}", "airfrans": "pc2_af_{arm}_seed{s}"}


def per_case(run):
    fs = glob.glob(f"{R}/campaigns_2026-09-29/pc2/metrics/{run}/**/metrics.jsonl", recursive=True)
    if not fs:
        return None
    rows = [json.loads(l)["metrics"] for l in open(sorted(fs)[0]) if '"infer_step"' in l]
    return {k: st.mean(r[k] for r in rows) for k in rows[0] if k.endswith("_l2")}


out = {"protocol": __doc__, "datasets": {}}
worst, best, complete, any_scored = 0.0, 9.0, True, False
for name, pat in SETS.items():
    ref = {s: per_case(pat.format(arm="ref", s=s)) for s in (42, 43)}; ref = {s: v for s, v in ref.items() if v}
    cond = {s: per_case(pat.format(arm="cent", s=s)) for s in (42, 43)}; cond = {s: v for s, v in cond.items() if v}
    if not ref or not cond:
        out["datasets"][name] = {"status": "not scored", "ref_seeds": len(ref), "cond_seeds": len(cond)}; complete = False; continue
    fields = sorted(set.intersection(*(set(v) for v in list(ref.values()) + list(cond.values()))))
    fields = [f for f in fields if not f.endswith(("_x_l2", "_y_l2", "_z_l2")) and f != "stress_l2" and st.mean(ref[s][f] for s in ref) > 0]
    rm = {f: st.mean(ref[s][f] for s in ref) for f in fields}; cm = {f: st.mean(cond[s][f] for s in cond) for f in fields}
    ratio = {f: cm[f] / rm[f] for f in fields}
    ps = {s: {f: cond[s][f] / ref[s][f] for f in fields} for s in cond if s in ref}
    d = {"fields": fields, "reference": rm, "conditioned": cm, "ratio": ratio, "per_seed": {"reference": ref, "conditioned": cond, "ratio": ps},
         "n_seeds": {"reference": len(ref), "conditioned": len(cond)}, "interim": min(len(ref), len(cond)) < 2,
         "reading": "pays" if min(ratio.values()) <= 0.97 and max(ratio.values()) <= 1.03 else "hurts" if max(ratio.values()) > 1.05 else "neutral" if max(ratio.values()) <= 1.03 else "between"}
    out["datasets"][name] = d; any_scored = True
    worst = max(worst, max(ratio.values())); best = min(best, min(ratio.values()))
    if d["interim"]: complete = False
if any_scored:
    out["overall"] = {"worst_ratio": worst, "best_ratio": best, "complete": complete,
                      "verdict": "deleted" if worst > 1.05 else ("reference candidate" if (complete and worst <= 1.03 and best <= 0.97) else "pending" if not complete else "deleted (no dataset <= 0.97)" if best > 0.97 else "between")}
json.dump(out, open(f"{R}/proto_cond2_2026-09-29.json", "w"), indent=1)
for name, d in out["datasets"].items():
    if "ratio" in d:
        print(f"== {name}: " + "  ".join(f"{f} ref {d['reference'][f]:.4f} cond {d['conditioned'][f]:.4f} r {d['ratio'][f]:.3f}" for f in d["fields"]) + f"  -> {d['reading']}{' (interim)' if d['interim'] else ''}")
        for s, r in d["per_seed"]["ratio"].items(): print(f"   seed {s}: " + " ".join(f"{f} {x:.3f}" for f, x in r.items()))
    else:
        print(f"== {name}: {d}")
print("overall:", out.get("overall"))
