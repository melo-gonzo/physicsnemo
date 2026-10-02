"""code_v18 reference lanes (PROTO-COND-2 adopted, 2026-10-02, #sec-nb-proto-cond-2-adopted): the mesh-native ISLA reference
configuration with sample-conditioned centered prototypes in every block and in the head frame (code_v18 = 7ef7c34fa, static
prototypes deleted) against the last static-prototype lanes on the same recipe (recipe_v8) and the same per-block compile
(group pf, code_v16 `pf_*_cb_*`, PERF-3 neutrality pairs), two seeds each, DrivAerML Surface (435 cars, 200,000 cells) and
DrivAerML Volume (200,000 boundary cells, 200,000 interior queries), HiLift 210-wing medium rung and AirfRANS when scored.
Readout: float32 per-case relative L2 per primary field, seed means; ratio v18 / static. Reads
campaigns_2026-10-02/v18/metrics/<run>/**/metrics.jsonl and campaigns_2026-10-01/pf/metrics/<run>/**/metrics.jsonl.
usage: python v18_reference_reduce_2026-10-02.py"""
import glob, json, os, statistics as st

R = os.path.dirname(os.path.abspath(__file__))
PAIRS = {
    "drivaer_surface": ("campaigns_2026-10-02/v18/metrics/v18_dr_ref_seed{s}", "campaigns_2026-10-01/pf/metrics/pf_dr_cb_seed{s}"),
    "drivaer_volume": ("campaigns_2026-10-02/v18/metrics/v18_vol_ref_seed{s}", "campaigns_2026-10-01/pf/metrics/pf_vol_cb_seed{s}"),
}
SOLO = {"hilift_210": "campaigns_2026-10-02/v18/metrics/v18_hl_med_seed{s}", "airfrans": "campaigns_2026-10-02/v18/metrics/v18_af_ref_seed{s}"}


def per_case(rel):
    fs = glob.glob(f"{R}/{rel}/**/metrics.jsonl", recursive=True)
    if not fs:
        return None
    rows = [json.loads(l)["metrics"] for l in open(sorted(fs)[0]) if '"infer_step"' in l]
    return {k: st.mean(r[k] for r in rows) for k in rows[0] if k.endswith("_l2")} | {"n_cases": len(rows)}


def fields_of(d):
    return sorted(k for k in d if k.endswith("_l2") and not k.endswith(("_x_l2", "_y_l2", "_z_l2")) and k != "stress_l2" and d[k] > 0)


out = {"protocol": __doc__, "datasets": {}}
for name, (new_pat, old_pat) in PAIRS.items():
    new = {s: per_case(new_pat.format(s=s)) for s in (42, 43)}; new = {s: v for s, v in new.items() if v}
    old = {s: per_case(old_pat.format(s=s)) for s in (42, 43)}; old = {s: v for s, v in old.items() if v}
    if not new or not old:
        out["datasets"][name] = {"status": "not scored", "v18_seeds": len(new), "static_seeds": len(old)}; continue
    fields = [f for f in fields_of(next(iter(new.values()))) if all(f in v for v in list(new.values()) + list(old.values()))]
    nm = {f: st.mean(new[s][f] for s in new) for f in fields}; om = {f: st.mean(old[s][f] for s in old) for f in fields}
    out["datasets"][name] = {"fields": fields, "v18": nm, "static": om, "ratio": {f: nm[f] / om[f] for f in fields},
                             "per_seed": {"v18": new, "static": old}, "n_seeds": {"v18": len(new), "static": len(old)},
                             "interim": min(len(new), len(old)) < 2}
for name, pat in SOLO.items():
    new = {s: per_case(pat.format(s=s)) for s in (42, 43)}; new = {s: v for s, v in new.items() if v}
    if not new:
        out["datasets"][name] = {"status": "not scored"}; continue
    fields = fields_of(next(iter(new.values())))
    out["datasets"][name] = {"fields": fields, "v18": {f: st.mean(new[s][f] for s in new) for f in fields}, "per_seed": {"v18": new},
                             "n_seeds": {"v18": len(new)}, "interim": len(new) < 2}
json.dump(out, open(f"{R}/v18_reference_2026-10-02.json", "w"), indent=1)
for name, d in out["datasets"].items():
    if "ratio" in d:
        print(f"== {name}: " + "  ".join(f"{f} v18 {d['v18'][f]:.4f} static {d['static'][f]:.4f} r {d['ratio'][f]:.3f}" for f in d["fields"]) + (" (interim)" if d["interim"] else ""))
        for s in d["per_seed"]["v18"]:
            if s in d["per_seed"]["static"]:
                print(f"   seed {s}: " + " ".join(f"{f} {d['per_seed']['v18'][s][f] / d['per_seed']['static'][s][f]:.3f}" for f in d["fields"]))
    elif "v18" in d:
        print(f"== {name}: " + "  ".join(f"{f} v18 {d['v18'][f]:.4f}" for f in d["fields"]) + (" (interim)" if d["interim"] else ""))
    else:
        print(f"== {name}: {d}")
