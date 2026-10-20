"""UDRV-HL readout: reduce the HiLift scores of the unit-drive baselines (udrv_hl_{gt,t}_seed42/43), their physical-drive
REF-80 predecessors (r80_hl_gt_h256_seed42/43, r80_hl_t_h512_l16_c100k_seed42/43) and the A2 ISLA comparator
(a2r_hl_isla_seed42/43), every arm on ISLA's 200,000-cell set, exactly as char_a2_table.py does (infer_summary, else the
mean of the per-sample infer_step rows, last row per sample id). Extended 2026-10-20 (readout) from the 2026-10-18
version: physical-drive arms, five fields, unit / physical and baseline / ISLA ratios for every field and protocol,
per-case paired readouts, and the PREREG P1-P3 verdicts on like-for-like (200,000-cell) scores.

Sources (all fetched from LHR $T unless noted):
  scores/udrv_hl_{gt,t}_seed{s}/<p>/<run>/metrics.jsonl      LHR job 299041 (kernel/udrv_hl/score.sbatch)
  scores/a2r_hl_isla_seed{s}/<p>/<run>/metrics.jsonl          LHR jobs 293808/294162 (evals_a2r, identical rows)
  scores/r80_hl_t_h512_l16_c100k_seed{s}_s200k/<p>/<run>/...  LHR job 299159 (kernel/udrv_hl/phys_s200k, CHAR scorer copy)
  scores/r80_hl_gt_h256_seed{s}_s200k/<p>/<run>/...           same job, LHR re-score (reproduction check of CHAR)
  CHAR protocol files (examples/cfd/isla/results/campaigns_2026-10-02/char/protocols/<run>/<p>/metrics.jsonl): AGA,
  physical-drive GeoTransolver at 200k (used for the GT physical arm when GT_PHYS_SOURCE == "char") and Transolver at
  100k (record only; not like-for-like).
usage: python udrv_table.py"""
import json, os, re, statistics as st
H = os.path.dirname(os.path.abspath(__file__))
CHARP = f"{H}/../../examples/cfd/isla/results/campaigns_2026-10-02/char/protocols"
F = {"pressure_l2": "pressure", "tau_wall_l2": "wall shear", "velocity_l2": "velocity", "density_l2": "density",
     "temperature_l2": "temperature"}
P = ("native", "area", "posed"); S = ("42", "43")
### Which physical-drive GeoTransolver scores enter the table: "char" (AGA CHAR record, 200k) if its set equals ISLA's
### 200k set, else "lhr" (the LHR re-score through the same CHAR scorer copy). Default "lhr": the CHAR record was scored on
### the AGA full dataset, where each case has a different dataset index, and the per-sample subsample generator is seeded
### by the index, so its 200k subsample is not ISLA's (RESULTS.md).
GT_PHYS_SOURCE = os.environ.get("GT_PHYS_SOURCE", "lhr")


def path(kind, run, p):
    if kind == "lhr":
        return f"{H}/scores/{run}/{p}/{run}/metrics.jsonl"
    if kind == "lhr_s200k":
        return f"{H}/scores/{run}_s200k/{p}/{run}/metrics.jsonl"
    return f"{CHARP}/{run}/{p}/metrics.jsonl"  # char


ARMS = {  # key: (label, kind, run stem)
    "isla": ("ISLA A2 (a2r_hl_isla)", "lhr", "a2r_hl_isla"),
    "gt_unit": ("GeoTransolver 256, U_inf_dir (udrv_hl_gt)", "lhr", "udrv_hl_gt"),
    "gt_phys": ("GeoTransolver 256, U_inf (r80_hl_gt_h256)", "char" if GT_PHYS_SOURCE == "char" else "lhr_s200k", "r80_hl_gt_h256"),
    "t_unit": ("Transolver 512 x 16, U_inf_dir (udrv_hl_t)", "lhr", "udrv_hl_t"),
    "t_phys": ("Transolver 512 x 16, U_inf (r80_hl_t_h512_l16_c100k) @200k", "lhr_s200k", "r80_hl_t_h512_l16_c100k"),
}
EXTRA = {  # checks and record rows, not part of the verdicts
    "gt_phys_lhr": ("GeoTransolver 256, U_inf, LHR re-score @200k [check]", "lhr_s200k", "r80_hl_gt_h256"),
    "gt_phys_char": ("GeoTransolver 256, U_inf, CHAR record (AGA) @200k", "char", "r80_hl_gt_h256"),
    "t_phys_c100k": ("Transolver 512 x 16, U_inf, CHAR record (AGA) @100k [not like-for-like]", "char", "r80_hl_t_h512_l16_c100k"),
}


def rows(fp):
    by, summ, cfg = {}, None, None
    for line in open(fp):
        r = json.loads(line)
        if r.get("phase") == "config": cfg = r
        if r.get("phase") == "infer_summary": summ = (r["metrics"], r.get("num_samples"))
        ### case key without the dataset-index prefix ("00002_geo_..."): the AGA full dataset and the LHR 690-case subset
        ### number the same case differently (AGA 00003_geo_LHC001_AoA_16, LHR 00002_geo_LHC001_AoA_16)
        if r.get("phase") == "infer_step": by[re.sub(r"^\d+_", "", str(r.get("sample_id", len(by))))] = r["metrics"]
    return summ, by, cfg


def load_arm(kind, stem):
    res, cases, meta = {}, {}, {}
    for p in P:
        res[p], cases[p] = {}, {}
        got = {}
        for s in S:
            fp = path(kind, f"{stem}_seed{s}", p)
            if not os.path.exists(fp): return None
            summ, by, cfg = rows(fp)
            m = summ[0] if summ else {k: st.mean(x[k] for x in by.values()) for k in F}
            got[s] = (m, by)
            meta[(p, s)] = {"epoch": cfg.get("epoch"), "num_samples": summ[1] if summ else len(by), "file": os.path.relpath(fp, H)}
        for f in F:
            per = {s: got[s][0][f] for s in S}
            res[p][f] = {"mean": st.mean(per.values()), "per_seed": per}
        ids = set.intersection(*(set(got[s][1]) for s in S))
        cases[p] = {f: {i: st.mean(got[s][1][i][f] for s in S) for i in ids} for f in F}
    return {"res": res, "cases": cases, "meta": meta}


data = {k: load_arm(kind, stem) for k, (lab, kind, stem) in {**ARMS, **EXTRA}.items()}
missing = [k for k in ARMS if data[k] is None]
if missing: raise SystemExit(f"missing scores for {missing}")
lab = {k: v[0] for k, v in {**ARMS, **EXTRA}.items()}
fmt = lambda d: f"{d['mean']:.4f} ({d['per_seed']['42']:.4f}, {d['per_seed']['43']:.4f})"
out = ["## UDRV-HL readout: HiLift (highlift_surface_medium, 180 val cases, float32, epoch-500 checkpoints, ISLA's 200,000-cell set)",
       "", f"Physical-drive GeoTransolver source: {GT_PHYS_SOURCE}. Seed mean (seed 42, seed 43); relative L2.", "",
       "| arm | protocol | " + " | ".join(F.values()) + " |", "|---|---|" + "---|" * len(F)]
for k in [*ARMS, *EXTRA]:
    if data[k] is None: continue
    for p in P:
        out.append(f"| {lab[k]} | {p} | " + " | ".join(fmt(data[k]["res"][p][f]) for f in F) + " |")


def ratio_rows(title, pairs):
    o = ["", f"### {title} (seed means)", "", "| ratio | protocol | " + " | ".join(F.values()) + " |", "|---|---|" + "---|" * len(F)]
    for a, b, name in pairs:
        for p in P:
            o.append(f"| {name} | {p} | " + " | ".join(f"{data[a]['res'][p][f]['mean'] / data[b]['res'][p][f]['mean']:.3f}" for f in F) + " |")
    return o


out += ratio_rows("Unit drive ÷ physical drive, per arm, like-for-like at 200k",
                  [("gt_unit", "gt_phys", "GeoTransolver"), ("t_unit", "t_phys", "Transolver")])
out += ratio_rows("Baseline ÷ ISLA A2", [(k, "isla", lab[k]) for k in ("gt_unit", "t_unit", "gt_phys", "t_phys")])


def paired(a, b, p="native", f="pressure_l2"):
    ca, cb = data[a]["cases"][p][f], data[b]["cases"][p][f]
    ids = sorted(set(ca) & set(cb))
    lower = sum(ca[i] < cb[i] for i in ids)
    med = st.median(ca[i] / cb[i] for i in ids)
    return lower, len(ids), med


verd = {}
out += ["", "### P1: unit drive against the physical-drive self, native pressure, like-for-like (200k)", ""]
for a, b, name, rule in (("gt_unit", "gt_phys", "GeoTransolver", 0.90), ("t_unit", "t_phys", "Transolver", 1.00)):
    r = data[a]["res"]["native"]["pressure_l2"]["mean"] / data[b]["res"]["native"]["pressure_l2"]["mean"]
    lower, n, med = paired(a, b)
    ok = (r <= rule if rule < 1 else r < 1) and lower > n / 2
    verd[f"P1 {name}"] = ok
    out.append(f"- {name}: unit {data[a]['res']['native']['pressure_l2']['mean']:.4f} vs physical {data[b]['res']['native']['pressure_l2']['mean']:.4f}; "
               f"ratio {r:.3f} (rule {'<= 0.90' if rule < 1 else '< 1.00'}); unit lower on {lower} of {n} cases (seed-mean pairs), "
               f"median per-case factor {med:.3f} -> {'HOLDS' if ok else 'FAILS'}")
    for f in ("tau_wall_l2",):
        lw, n2, md = paired(a, b, f=f)
        out.append(f"  - wall shear: ratio {data[a]['res']['native'][f]['mean'] / data[b]['res']['native'][f]['mean']:.3f}, lower on {lw} of {n2}, median per-case {md:.3f}")
if data["t_phys_c100k"] is not None:
    r = data["t_unit"]["res"]["native"]["pressure_l2"]["mean"] / data["t_phys_c100k"]["res"]["native"]["pressure_l2"]["mean"]
    out.append(f"- Record only (not like-for-like): Transolver unit @200k ÷ CHAR physical @100k, native pressure {r:.3f}")

out += ["", "### P2: baseline ÷ ISLA A2, native, seed means; cases lower paired on seed-mean values", ""]
for k in ("gt_unit", "t_unit", "gt_phys", "t_phys"):
    r = {f: data[k]["res"]["native"][f]["mean"] / data["isla"]["res"]["native"][f]["mean"] for f in F}
    lower, n, med = paired(k, "isla")
    lw, _, mdw = paired(k, "isla", f="tau_wall_l2")
    rp = r["pressure_l2"]
    band = ("baseline ahead" if rp <= 0.95 and lower >= 2 * n / 3 else "ISLA ahead" if rp >= 1.05
            else "parity" if abs(rp - 1) <= 0.05 else "between (ratio <= 0.95 but lower on < 2/3 of cases)")
    if k in ("gt_unit", "t_unit"): verd[f"P2 {lab[k]}"] = band
    out.append(f"- {lab[k]}: " + ", ".join(f"{F[f]} {r[f]:.3f}" for f in F) +
               f"; pressure lower on {lower} of {n} cases (median per-case {med:.3f}); wall shear lower on {lw} of {n}; band: {band}")

out += ["", "### P3: area ÷ native and posed ÷ native (seed means; ISLA also per seed, R-POSE |posed/native - 1| <= 0.002)", ""]
for k in ARMS:
    out.append(f"- {lab[k]}: " + "; ".join(f"{F[f]} area {data[k]['res']['area'][f]['mean'] / data[k]['res']['native'][f]['mean']:.3f}, "
                                           f"posed {data[k]['res']['posed'][f]['mean'] / data[k]['res']['native'][f]['mean']:.4f}" for f in F))
dev = {(f, s): data["isla"]["res"]["posed"][f]["per_seed"][s] / data["isla"]["res"]["native"][f]["per_seed"][s] - 1 for f in F for s in S}
out.append("- ISLA |posed/native - 1| per field and seed: " + ", ".join(f"{F[f]} {s} {abs(v):.4f}" for (f, s), v in dev.items()))

out += ["", "### Score metadata", "", "| arm | protocol | seed | epoch | num_samples | file |", "|---|---|---|---|---|---|"]
for k in [*ARMS, *EXTRA]:
    if data[k] is None: continue
    for (p, s), m in data[k]["meta"].items():
        out.append(f"| {k} | {p} | {s} | {m['epoch']} | {m['num_samples']} | {m['file']} |")
json.dump({"protocol": __doc__, "gt_phys_source": GT_PHYS_SOURCE, "fields": F, "labels": lab,
           "arms": {k: v["res"] for k, v in data.items() if v is not None}, "verdict_inputs": {k: str(v) for k, v in verd.items()}},
          open(f"{H}/udrv_hl_scores.json", "w"), indent=1)
open(f"{H}/udrv_hl_table.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
