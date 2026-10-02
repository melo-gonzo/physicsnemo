"""PERF-3 neutrality reduction (2026-10-01, #sec-nb-perf-3-verdict): the reference ISLA configuration trained with per-block
torch.compile (compile_blocks=true, the recipe default since 343b284e) against the same snapshot trained eagerly
(compile_blocks=false), on DrivAerML Surface (pf_dr_{cb,nocb}) and DrivAerML Volume (pf_vol_{cb,nocb}), 200,000 cells, two seeds
each, code_v16 + recipe_v8. Readout: float32 per-case relative L2 per field on the test split, averaged over cases, then over seeds.
Ratio = compiled / eager per field (below 1 favors compiled). Bar (the KERNEL-NEUTRAL rule): neutral if every field on both datasets
is within 3% of 1 on the seed means; otherwise the compiled default is withdrawn. A compile is a numerics change only (operator
fusion, different accumulation order), so the prediction is a null: no field moves beyond seed noise.
usage: python perf3_reduce_2026-10-01.py  (writes perf3_neutrality_2026-10-01.json)."""
import glob, json, os, statistics as st

R = os.path.dirname(os.path.abspath(__file__))
SETS = {"drivaerml_surface": "pf_dr_{arm}_seed{s}", "drivaerml_volume": "pf_vol_{arm}_seed{s}"}
BAR = 0.03


def per_case(run):
    fs = glob.glob(f"{R}/campaigns_2026-10-01/pf/metrics/{run}/**/metrics.jsonl", recursive=True)
    if not fs:
        return None, 0
    rows = [json.loads(l)["metrics"] for l in open(sorted(fs)[0]) if '"infer_step"' in l]
    keys = [k for k in rows[0] if k.endswith("_l2")]
    return {k: st.mean(r[k] for r in rows) for k in keys}, len(rows)


# Chain wall-clock per lane (hours), the sum of ElapsedRaw over the lane's chain links from `sacct --name=isla.pf.<lane>` on AGA
# (2026-10-02); each pair has the same link count (Surface 2, Volume 3), so startup and the wall-killed tail of a link enter both arms.
WALL_H = {"drivaerml_surface": {"cb": {42: 2.28, 43: 2.36}, "nocb": {42: 3.39, 43: 3.41}},
          "drivaerml_volume": {"cb": {42: 5.20, 43: 5.11}, "nocb": {42: 7.42, 43: 7.35}}}

out = {"protocol": __doc__, "bar": BAR, "datasets": {}}
worst = 0.0; complete = True
for name, pat in SETS.items():
    eager = {}; comp = {}; n_cases = {}
    for s in (42, 43):
        e, ne = per_case(pat.format(arm="nocb", s=s)); c, nc = per_case(pat.format(arm="cb", s=s))
        if e: eager[s] = e; n_cases[f"nocb_seed{s}"] = ne
        if c: comp[s] = c; n_cases[f"cb_seed{s}"] = nc
    if not eager or not comp:
        out["datasets"][name] = {"status": "not scored"}; complete = False; continue
    fields = sorted(set.intersection(*(set(v) for v in list(eager.values()) + list(comp.values()))))
    fields = [f for f in fields if not f.endswith(("_x_l2", "_y_l2", "_z_l2")) and st.mean(eager[s][f] for s in eager) > 0]
    em = {f: st.mean(eager[s][f] for s in eager) for f in fields}; cm = {f: st.mean(comp[s][f] for s in comp) for f in fields}
    ratio = {f: cm[f] / em[f] for f in fields}
    ps = {s: {f: comp[s][f] / eager[s][f] for f in fields} for s in comp if s in eager}
    dev = max(abs(r - 1) for r in ratio.values())
    out["datasets"][name] = {"fields": fields, "eager": em, "compiled": cm, "ratio": ratio, "max_abs_deviation": dev,
                             "per_seed": {"eager": eager, "compiled": comp, "ratio": ps}, "n_cases": n_cases,
                             "n_seeds": {"eager": len(eager), "compiled": len(comp)}, "within_bar": dev <= BAR,
                             "interim": min(len(eager), len(comp)) < 2}
    w = WALL_H[name]; out["datasets"][name]["wall_clock_hours"] = {"compiled": w["cb"], "eager": w["nocb"],
        "ratio_compiled_over_eager": st.mean(w["cb"].values()) / st.mean(w["nocb"].values()), "links": 2 if "surface" in name else 3}
    worst = max(worst, dev)
    if min(len(eager), len(comp)) < 2: complete = False
out["overall"] = {"worst_abs_deviation": worst, "complete": complete,
                  "verdict": "neutral (compiled default stands)" if worst <= BAR and complete else "withdrawn" if worst > BAR else "pending"}
json.dump(out, open(f"{R}/perf3_neutrality_2026-10-01.json", "w"), indent=1)
for name, d in out["datasets"].items():
    if "ratio" in d:
        print(f"== {name} (cases {d['n_cases']}): " + "  ".join(f"{f} eager {d['eager'][f]:.5f} compiled {d['compiled'][f]:.5f} ratio {d['ratio'][f]:.3f}" for f in d["fields"]) + f"  -> within 3%: {d['within_bar']}; wall-clock compiled/eager {d['wall_clock_hours']['ratio_compiled_over_eager']:.2f}")
        for s, r in d["per_seed"]["ratio"].items(): print(f"   seed {s}: " + " ".join(f"{f} {x:.3f}" for f, x in r.items()))
    else:
        print(f"== {name}: {d}")
print("overall:", out["overall"])
