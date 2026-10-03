"""REF-80 selection (CHAR, 2026-10-03, #sec-nb-char-prereg): for every candidate configuration of ISLA, GeoTransolver and Transolver
that trains under 80 GiB per GPU (results/char/ref80_frontier_2026-10-02.json), the float32 validation error of its 500-epoch
lanes (group r80, two seeds) on DrivAerML Surface (48 validation cars, scored at 200,000 cells, the reporting protocol, for
every candidate; the `_c100k` candidates are additionally scored at their training count of 100,000 cells, kept under
metrics_train_count/ as a secondary reading) and on the HiLift 210-wing medium rung when scored. Headline = area-weighted relative L2 of
surface pressure, seed mean. Rule (preregistered): per architecture and dataset, the candidate with the lowest seed-mean headline;
candidates within 3% of the best are ties and the cheaper one (lower training peak memory) wins. The reference configuration is
the candidate that wins on both datasets; if the two datasets disagree, both winners are listed and the decision is written in the
notebook. Reads campaigns_2026-10-02/char/r80/metrics/<run>/**/metrics.jsonl.
usage: python ref80_select_2026-10-03.py"""
import glob, json, os, statistics as st

R = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(R)
FRONT = json.load(open(f"{R}/ref80_frontier_2026-10-02.json"))
ARCH = {"isla": "ISLA", "gt": "GeoTransolver", "t": "Transolver"}
DATASETS = {"drivaer_surface": "dr", "hilift_210": "hl"}
TIE = 0.03


def per_case(run, sub="metrics"):
    fs = glob.glob(f"{RES}/campaigns_2026-10-02/char/r80/{sub}/{run}/**/metrics.jsonl", recursive=True)
    if not fs:
        return None
    rows = [json.loads(l)["metrics"] for l in open(sorted(fs)[0]) if '"infer_step"' in l]
    if not rows:
        return None
    return {k: st.mean(r[k] for r in rows) for k in rows[0] if k.endswith("_l2")} | {"n_cases": len(rows)}


def peak(tag):
    c = FRONT["candidates"].get(tag) or FRONT["candidates"].get(tag + "_nc")
    return c["train_peak_gib"] if c else None


out = {"protocol": __doc__, "tie_fraction": TIE, "datasets": {}}
for ds, pfx in DATASETS.items():
    out["datasets"][ds] = {}
    for a, name in ARCH.items():
        cands = {}
        for tag in FRONT["frontier"][name]:
            if tag.endswith("_nc"):
                continue
            seeds = {s: per_case(f"r80_{pfx}_{tag}_seed{s}") for s in (42, 43)}
            seeds = {s: v for s, v in seeds.items() if v}
            if not seeds:
                continue
            cands[tag] = {"pressure_l2": st.mean(v["pressure_l2"] for v in seeds.values()),
                          "per_seed": {s: v["pressure_l2"] for s, v in seeds.items()},
                          "fields": {k: st.mean(v[k] for v in seeds.values()) for k in next(iter(seeds.values())) if k.endswith("_l2")},
                          "n_seeds": len(seeds), "n_cases": min(v["n_cases"] for v in seeds.values()), "peak_memory_gib": peak(tag)}
            tc = {s: per_case(f"r80_{pfx}_{tag}_seed{s}", "metrics_train_count") for s in (42, 43)}; tc = {s: v for s, v in tc.items() if v}
            if tc:
                cands[tag]["pressure_l2_at_training_count"] = st.mean(v["pressure_l2"] for v in tc.values())
        if not cands:
            out["datasets"][ds][name] = {"status": "not scored"}; continue
        best = min(c["pressure_l2"] for c in cands.values())
        ties = [t for t, c in cands.items() if c["pressure_l2"] <= best * (1 + TIE)]
        winner = min(ties, key=lambda t: (cands[t]["peak_memory_gib"] or 1e9))
        complete = all(c["n_seeds"] == 2 for c in cands.values()) and len(cands) == len([t for t in FRONT["frontier"][name] if not t.endswith("_nc")])
        out["datasets"][ds][name] = {"candidates": cands, "best": best, "ties_within_3pct": ties, "winner": winner, "complete": complete}
out["reference"] = {}
for name in ARCH.values():
    w = {ds: d[name].get("winner") for ds, d in out["datasets"].items() if "winner" in d.get(name, {})}
    out["reference"][name] = {"winners": w, "agreed": len(set(w.values())) == 1 and len(w) == len(DATASETS)}
json.dump(out, open(f"{R}/ref80_select_2026-10-03.json", "w"), indent=1)
for ds, d in out["datasets"].items():
    print(f"== {ds}")
    for name, r in d.items():
        if "candidates" not in r:
            print(f"   {name}: {r}"); continue
        for t, c in sorted(r["candidates"].items(), key=lambda kv: kv[1]["pressure_l2"]):
            mark = "WIN" if t == r["winner"] else ("tie" if t in r["ties_within_3pct"] else "")
            seeds = " / ".join(f"{c['per_seed'][s]:.4f}" for s in sorted(c["per_seed"]))
            tc = f" [at 100k: {c['pressure_l2_at_training_count']:.4f}]" if "pressure_l2_at_training_count" in c else ""
            print(f"   {name:14s} {t:22s} p {c['pressure_l2']:.4f} ({seeds}) n={c['n_seeds']} cases={c['n_cases']} peak {c['peak_memory_gib']:.1f} GiB {mark}{tc}")
print("reference:", json.dumps(out["reference"]))
