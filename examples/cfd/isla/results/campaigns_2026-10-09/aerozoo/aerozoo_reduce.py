#!/usr/bin/env python
"""AEROZOO reduction (2026-10-10; pre-registration #sec-nb-aerozoo-prereg): ISLA, ISLA-DTT and GeoTransolver at their reference
configurations on AeroZoo v1.2's interior task, in distribution (`full`) and with an enclosure family (`family_non_box`), a Reynolds
range (`reynolds`) or the blob bank (`bank_blobs`) held out. Reads the scoring records fetched by fetch_scores.sh into scores/<run>/
{val,test}/metrics.jsonl (float32, the training draw, per-case relative L2 per field; the last record per case counts), the per-case
labels of case_labels.json (stability population, enclosure, bank, Reynolds number) and the lanes' training records (validation
trajectory every ten epochs, step time and memory of the final epoch). Writes aerozoo_reduction.json:
  fields                     metric key -> label
  splits[split]              label, held-out family, the run stem
  arms[arm][split][side]     fields[f] = {mean, per_seed, n_cases, n_seeds}; by_quality[q][f]; by_label[kind][value][f] (test sides)
  ratios[split][side][arm]   arm ÷ ISLA per field: seed-mean ratio, per-seed ratios, both_seeds_agree, counts (outside 1 ± 0.05)
  penalties[split][arm]      test error of the split's own model ÷ the same arm's `full` test error, per field (P2, P3)
  predictions                P1, P2, P4 verdicts against the pre-registered bars
  cost[arm][split]           median step seconds and peak memory of the final epoch, parameters, epochs completed, GPU-hours
  trajectories[run]          [{epoch, loss, <field>_l2 ...}] from val_summary rows
A seed mean is over the seeds present; a block is `complete` only when both seeds are scored.
usage: python aerozoo_reduce.py [--out aerozoo_reduction.json]"""
import argparse, json, math, os, re, statistics as st

R = os.path.dirname(os.path.abspath(__file__))
SCORES = f"{R}/scores"
FIELDS = {"velocity_l2": "velocity", "pressure_l2": "pressure", "nut_l2": "eddy viscosity",
          "tke_l2": "turbulent kinetic energy", "omega_l2": "specific dissipation rate"}
SEEDS = (42, 43)
SIDES = ("val", "test")
SPLITS = {
    "full": {"label": "in distribution", "held_out": None, "stem": "full", "n_train": 2713, "n_val": 386, "n_test": 716},
    "family_non_box": {"label": "enclosure family held out (trained on external boxes)", "held_out": "enclosure", "stem": "nonbox",
                       "n_train": 2390, "n_val": 351, "n_test": 1074},
    "reynolds": {"label": "Reynolds range held out (trained below 10^6)", "held_out": "log10_re_decade", "stem": "reynolds",
                 "n_train": 2119, "n_val": 311, "n_test": 1385},
    "bank_blobs": {"label": "blob bank held out", "held_out": "bank", "stem": "blobs", "n_train": 2156, "n_val": 292, "n_test": 1367},
}
ARMS = {"ISLA": "isla", "GeoTransolver": "gt", "ISLA-DTT": "dtt"}
ISLA = "ISLA"
BAR_COUNTS = 0.05       # a difference counts only if both seeds order the same way and the seed-mean ratio is outside 1 ± 0.05
P1_VEL, P1_PRES = 1.2, 0.9          # supported: GT ÷ ISLA >= 1.2 on velocity and <= 0.9 on pressure (full test side)
P1_VEL_F, P1_PRES_F = 1.1, 1.0      # falsified: velocity ratio < 1.1 or pressure ratio > 1.0
P2_FRACTION = 0.8                   # supported: ISLA's penalty <= 0.8 x GeoTransolver's; falsified: not smaller
P4_TOL = 0.05                       # ISLA-DTT within 5% of ISLA on every field (full test side)
LABELS = json.load(open(f"{R}/case_labels.json")) if os.path.exists(f"{R}/case_labels.json") else {}
LABELS.pop("_meta", None)


def run_id(split, arm, seed):
    return f"az_{SPLITS[split]['stem']}_{ARMS[arm]}_seed{seed}"


def per_case(path):
    """{sample_id: metrics} from an infer metrics.jsonl (the last infer_step record per sample id)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    by = {}
    for line in open(path):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("phase") == "infer_step" and "metrics" in rec:
            by[str(rec.get("sample_id", len(by)))] = rec["metrics"]
    return by or None


def label_of(sample_id):
    if sample_id in LABELS:
        return LABELS[sample_id]
    m = re.search(r"case-\d+-[0-9a-f]+", sample_id)
    return LABELS.get(m.group(0)) if m else None


def fmean(rows, f):
    v = [m[f] for m in rows if f in m and m[f] is not None and not (isinstance(m[f], float) and math.isnan(m[f]))]
    return (st.mean(v), len(v)) if v else (None, 0)


def block(per_seed_cases, with_labels):
    """fields / by_quality / by_label from {seed: {sample_id: metrics}}."""
    out = {"fields": {}, "n_seeds": len(per_seed_cases), "complete": set(per_seed_cases) >= set(SEEDS)}
    for f in FIELDS:
        ps = {s: fmean(list(c.values()), f) for s, c in per_seed_cases.items()}
        ps = {s: v for s, v in ps.items() if v[0] is not None}
        if ps:
            out["fields"][f] = {"mean": st.mean(v[0] for v in ps.values()), "per_seed": {s: v[0] for s, v in ps.items()},
                                "n_cases": min(v[1] for v in ps.values()), "n_seeds": len(ps)}
    if not with_labels or not LABELS:
        return out
    def strat(kind):
        groups = {}
        for s, cases in per_seed_cases.items():
            for sid, m in cases.items():
                lab = label_of(sid)
                if lab is None:
                    key = "unlabeled"
                elif kind == "log10_re_decade":
                    key = f"1e{int(math.floor(lab['log10_re']))}" if lab.get("log10_re") is not None else "unlabeled"
                else:
                    key = str(lab.get(kind))
                groups.setdefault(key, {}).setdefault(s, []).append(m)
        res = {}
        for key, by_seed in sorted(groups.items()):
            res[key] = {"n_cases": min(len(v) for v in by_seed.values())}
            for f in FIELDS:
                ps = {s: fmean(rows, f)[0] for s, rows in by_seed.items()}
                ps = {s: v for s, v in ps.items() if v is not None}
                if ps:
                    res[key][f] = {"mean": st.mean(ps.values()), "per_seed": ps}
        return res
    out["by_quality"] = strat("quality")
    out["by_label"] = {kind: strat(kind) for kind in ("enclosure", "scenario", "bank", "log10_re_decade", "body_layout")}
    return out


def ratio(num, den, f):
    """arm ÷ ISLA on one field: seed-mean ratio, per-seed ratios (paired by seed), agreement and the counts rule."""
    if f not in num["fields"] or f not in den["fields"]:
        return None
    a, b = num["fields"][f], den["fields"][f]
    r = a["mean"] / b["mean"]
    per = {s: a["per_seed"][s] / b["per_seed"][s] for s in a["per_seed"] if s in b["per_seed"]}
    agree = len(per) >= 2 and all((v > 1) == (r > 1) for v in per.values())
    return {"mean_ratio": r, "per_seed_ratio": per, "both_seeds_agree": agree,
            "counts": agree and abs(r - 1) > BAR_COUNTS and a["n_seeds"] >= 2 and b["n_seeds"] >= 2}


def trajectory(run):
    p = f"{SCORES}/{run}/train_metrics.jsonl"
    if not os.path.exists(p):
        return None, None, None
    rows, cfg, ds = [], {}, {}
    for line in open(p):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("phase") == "val_summary":
            rows.append({"epoch": rec.get("epoch"), "loss": rec.get("loss"), **{f: rec.get(f) for f in FIELDS if f in rec}})
        elif rec.get("phase") == "config":
            cfg = rec
        elif rec.get("phase") == "dataset":
            ds = rec
    return rows, cfg, ds


def cost(run, cfg, ds, epochs_target=250):
    p = f"{SCORES}/{run}/train_tail.log"
    if not os.path.exists(p):
        return None
    txt = open(p, errors="ignore").read()
    steps = [float(x) for x in re.findall(r"Step: ([0-9.]+)s", txt)]
    mems = [float(x) for x in re.findall(r"Mem: ([0-9.]+)GB", txt)]
    epochs = [int(x) for x in re.findall(r"Epoch (\d+) \[", txt)]
    steps_per_epoch = ds.get("train_samples")
    med = st.median(steps) if steps else None
    done = "Training completed" in txt
    last_epoch = max(epochs) if epochs else None
    gpu_h = (epochs_target * steps_per_epoch * med * 4 / 3600) if (done and med and steps_per_epoch) else None
    return {"step_seconds_median": med, "mem_gb_max": max(mems) if mems else None, "n_parameters": cfg.get("num_parameters"),
            "steps_per_epoch": steps_per_epoch, "last_epoch_seen": last_epoch, "trained": done,
            "gpu_hours_est": gpu_h, "links": txt.count("=== link start")}


def main(out_path):
    out = {"protocol": __doc__, "fields": FIELDS, "splits": SPLITS, "seeds": list(SEEDS), "arms": {}, "ratios": {}, "penalties": {},
           "predictions": {}, "cost": {}, "trajectories": {}, "runs": {}}
    for arm in ARMS:
        out["arms"][arm], out["cost"][arm] = {}, {}
        for split in SPLITS:
            sides = {}
            for side in SIDES:
                cases = {s: c for s in SEEDS if (c := per_case(f"{SCORES}/{run_id(split, arm, s)}/{side}/metrics.jsonl"))}
                if cases:
                    sides[side] = block(cases, with_labels=(side == "test" or split == "full"))
            if sides:
                out["arms"][arm][split] = sides
            for s in SEEDS:
                run = run_id(split, arm, s)
                rows, cfg, ds = trajectory(run)
                if rows is not None:
                    out["trajectories"][run] = rows
                    out["runs"][run] = {"arm": arm, "split": split, "seed": s}
                    c = cost(run, cfg, ds)
                    if c:
                        out["cost"][arm].setdefault(split, {})[str(s)] = c
    # best-validation checkpoints (a labeled diagnostic, never the standing): eval_bestval.sbatch scores one chosen checkpoint per
    # lane on the validation side exactly like the protocol; fetch_scores.sh keeps the record under scores/<run>/bestval/ with the
    # checkpoint file it scored in CHECKPOINT_USED. Same block shape as the arms, plus the chosen epoch per seed.
    for arm in ARMS:
        for split in SPLITS:
            cases, epochs = {}, {}
            for s in SEEDS:
                run = run_id(split, arm, s)
                c = per_case(f"{SCORES}/{run}/bestval/metrics.jsonl")
                if not c:
                    continue
                cases[s] = c
                ck = f"{SCORES}/{run}/bestval/CHECKPOINT_USED"
                m = re.search(r"checkpoint\.0\.(\d+)\.pt", open(ck).read()) if os.path.exists(ck) else None
                epochs[str(s)] = int(m.group(1)) if m else None
            if cases:
                out.setdefault("bestval", {}).setdefault(arm, {})[split] = {**block(cases, with_labels=False), "epochs": epochs}
    # ratios arm ÷ ISLA
    for split in SPLITS:
        for side in SIDES:
            isla = out["arms"].get(ISLA, {}).get(split, {}).get(side)
            if not isla:
                continue
            for arm in ARMS:
                if arm == ISLA:
                    continue
                b = out["arms"].get(arm, {}).get(split, {}).get(side)
                if b:
                    rs = {f: r for f in FIELDS if (r := ratio(b, isla, f))}
                    if rs:
                        out["ratios"].setdefault(split, {}).setdefault(side, {})[arm] = rs
    # penalties: the split's own model on its test side ÷ the arm's full model on the full test side
    for split in SPLITS:
        if split == "full":
            continue
        for arm in ARMS:
            held = out["arms"].get(arm, {}).get(split, {}).get("test"); base = out["arms"].get(arm, {}).get("full", {}).get("test")
            if held and base:
                pen = {f: r for f in FIELDS if (r := ratio(held, base, f))}
                if pen:
                    out["penalties"].setdefault(split, {})[arm] = pen
    # predictions
    P = out["predictions"]
    r1 = out["ratios"].get("full", {}).get("test", {}).get("GeoTransolver")
    if r1 and "velocity_l2" in r1 and "pressure_l2" in r1:
        v, p = r1["velocity_l2"]["mean_ratio"], r1["pressure_l2"]["mean_ratio"]
        verdict = ("supported" if (v >= P1_VEL and p <= P1_PRES) else "falsified" if (v < P1_VEL_F or p > P1_PRES_F) else "between")
        P["P1"] = {"statement": f"full test side: GeoTransolver ÷ ISLA >= {P1_VEL} on velocity and <= {P1_PRES} on pressure; falsified "
                                f"below {P1_VEL_F} on velocity or above {P1_PRES_F} on pressure",
                   "velocity_ratio": v, "pressure_ratio": p, "verdict": verdict,
                   "complete": all(out["arms"][a]["full"]["test"]["complete"] for a in (ISLA, "GeoTransolver"))}
    pen = out["penalties"].get("family_non_box", {})
    if ISLA in pen and "GeoTransolver" in pen:
        per_field = {}
        for f in FIELDS:
            if f in pen[ISLA] and f in pen["GeoTransolver"]:
                a, g = pen[ISLA][f]["mean_ratio"], pen["GeoTransolver"][f]["mean_ratio"]
                per_field[f] = {"isla_penalty": a, "gt_penalty": g, "fraction": a / g,
                                "verdict": "supported" if a <= P2_FRACTION * g else "falsified" if a >= g else "between"}
        vs = [d["verdict"] for d in per_field.values()]
        P["P2"] = {"statement": f"ISLA's held-out-enclosure penalty (family_non_box test ÷ full test, same arm) at least "
                                f"{(1 - P2_FRACTION) * 100:.0f}% smaller than GeoTransolver's; falsified if not smaller",
                   "per_field": per_field,
                   "verdict": ("supported" if vs and all(v == "supported" for v in vs) else "falsified" if vs and all(v == "falsified" for v in vs)
                               else "mixed" if vs else "pending"),
                   "n_supported": vs.count("supported"), "n_falsified": vs.count("falsified"), "n_fields": len(vs)}
    for split in ("reynolds", "bank_blobs"):
        pen = out["penalties"].get(split, {})
        if pen:
            P[f"P3_{split}"] = {"statement": "no directional prediction; recorded",
                                "penalties": {arm: {f: pen[arm][f]["mean_ratio"] for f in pen[arm]} for arm in pen}}
    r4 = out["ratios"].get("full", {}).get("test", {}).get("ISLA-DTT")
    if r4:
        devs = {f: abs(r4[f]["mean_ratio"] - 1) for f in r4}
        P["P4"] = {"statement": f"ISLA-DTT's full test error within {P4_TOL * 100:.0f}% of ISLA's on every field",
                   "ratios": {f: r4[f]["mean_ratio"] for f in r4}, "max_deviation": max(devs.values()),
                   "verdict": "supported" if max(devs.values()) <= P4_TOL else "falsified"}
    json.dump(out, open(out_path, "w"), indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    # console
    for split in SPLITS:
        for side in SIDES:
            rows = [(arm, out["arms"][arm][split][side]) for arm in ARMS if side in out["arms"].get(arm, {}).get(split, {})]
            if not rows:
                continue
            print(f"== {split} / {side}")
            for arm, b in rows:
                print(f"   {arm:14s}" + "  ".join(f"{FIELDS[f][:8]:>8s} {b['fields'][f]['mean']:.4f}" for f in FIELDS if f in b["fields"])
                      + f"   n={min((d['n_cases'] for d in b['fields'].values()), default=0)} seeds={b['n_seeds']}")
            for arm, rs in out["ratios"].get(split, {}).get(side, {}).items():
                print(f"   {arm + ' ÷ ISLA':14s}" + "  ".join(f"{FIELDS[f][:8]:>8s} {rs[f]['mean_ratio']:.3f}{'*' if rs[f]['counts'] else ' '}" for f in FIELDS if f in rs))
    for split, d in out["penalties"].items():
        print(f"== penalty {split} (test ÷ full test):  " + "  ".join(f"{arm} " + "/".join(f"{d[arm][f]['mean_ratio']:.2f}" for f in FIELDS if f in d[arm]) for arm in d))
    for k, v in P.items():
        print(f"== {k}: {v.get('verdict', '')}")
    print("wrote", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{R}/aerozoo_reduction.json")
    main(ap.parse_args().out)
