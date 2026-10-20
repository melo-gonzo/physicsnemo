"""HiLiftAeroML standing with the normalized slice reads and unit-direction baselines (2026-10-20): the CHAR-PROTOCOLS record
of HiLift (results/char/hilift_surface.json) with ISLA's arm replaced by the A2 reruns (a2r_hl_isla_seed42/43, the reference
configuration with the normalized slice reads) and both baseline arms replaced by the same reference configurations retrained
reading the unit freestream direction U_inf_dir in place of the physical velocity U_inf in inch/s (UDRV-HL: udrv_hl_gt_seed42/43,
GeoTransolver 256 at 200,000 cells; udrv_hl_t_seed42/43, Transolver 512 x 16 at 100,000 cells). Every arm is scored on ISLA's
200,000-cell set of the 180 validation cases, float32, epoch-500 checkpoints, native mesh / uniform point cloud / rotated +
translated. Scores: results/campaigns_2026-10-20/hilift/udrv_hl_scores.json (copied unmodified from
scratch/isla-udrv-hl-2026-10-18/, written there by udrv_table.py); per-case rows: the scores/ tree beside it.
Writes results/char/hilift_surface_a2.json in the schema of results/char/<dataset>_a2.json (protocols, fields, isla_arm,
arms[arm][protocol][field] = {mean, per_seed, n_cases, n_seeds}, complete) plus:
  raw_read_isla            the CHAR record's ISLA arm (raw slice reads, scored on AGA's copy of the dataset)
  raw_drive_geotransolver  the physical-drive GeoTransolver 256 (r80_hl_gt_h256), re-scored on ISLA's 200,000-cell set
  raw_drive_transolver     the physical-drive Transolver 512 x 16 (r80_hl_t_h512_l16_c100k), re-scored on ISLA's set
  char_baselines           the CHAR record's two baseline arms as scored there (AGA; Transolver at its own 100,000 cells)
  fair_input               what each replaced arm reads now, and where the replaced arm is kept
  registered               the pre-registered outcomes: A2-RERUN R-A2 on HiLift; UDRV-HL P1, P2 (with its band), P3
  case_level               per-case readouts of the native scores: cases on which each baseline is lower than ISLA, the median
                           per-case ratio, the share of GeoTransolver's mean lead carried by ISLA's worst tenth of cases and their
                           angles of attack, and the case means per angle of attack
usage: python hilift_a2_2026-10-20.py"""
import json, os, re, statistics as st

R = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(R), "campaigns_2026-10-20", "hilift")
ISLA_ARM = "ISLA 512 with 512 slices"
GT_ARM, T_ARM = "GeoTransolver 256", "Transolver 512 x 16, trained at 100k cells"
SEEDS = ("42", "43")

C = json.load(open(f"{R}/hilift_surface.json"))
U = json.load(open(f"{SRC}/udrv_hl_scores.json"))
FIELDS, PROTOS = C["fields"], C["protocols"]
N_CASES = 180


def arm(key):
    """{protocol: {field: {mean, per_seed, n_cases, n_seeds}}} of one UDRV-HL arm on the standing's fields."""
    a = U["arms"][key]
    return {p: {f: {"mean": a[p][f]["mean"], "per_seed": a[p][f]["per_seed"], "n_cases": N_CASES, "n_seeds": len(a[p][f]["per_seed"])}
                for f in FIELDS} for p in PROTOS}


def cases(run, p="native", f="pressure_l2"):
    """{case: seed-mean error} from the scorer's per-case rows (last row per case), case ids without the dataset-index prefix."""
    by = {}
    for s in SEEDS:
        rows = {}
        for line in open(f"{SRC}/scores/{run}_seed{s}/{p}/{run}_seed{s}/metrics.jsonl"):
            r = json.loads(line)
            if r.get("phase") == "infer_step":
                rows[re.sub(r"^\d+_", "", r["sample_id"])] = r["metrics"][f]
        by[s] = rows
    ids = set.intersection(*(set(v) for v in by.values()))
    return {i: st.mean(by[s][i] for s in SEEDS) for i in ids}


aoa = lambda case: int(re.search(r"AoA_(\d+)", case).group(1))
out = {"protocol": __doc__, "protocols": PROTOS, "fields": FIELDS, "isla_arm": ISLA_ARM,
       "arms": {ISLA_ARM: arm("isla"), GT_ARM: arm("gt_unit"), T_ARM: arm("t_unit")}}
out["complete"] = all(c["n_seeds"] == 2 for a in out["arms"].values() for p in a.values() for c in p.values())
out["raw_read_isla"] = {"label": C["isla_arm"], **C["arms"][C["isla_arm"]]}
out["raw_drive_geotransolver"] = {"label": "GeoTransolver 256 reading U_inf (r80_hl_gt_h256), re-scored on ISLA's 200,000-cell set",
                                  **arm("gt_phys")}
out["raw_drive_transolver"] = {"label": "Transolver 512 x 16 reading U_inf (r80_hl_t_h512_l16_c100k), re-scored on ISLA's 200,000-cell set",
                               **arm("t_phys")}
out["char_baselines"] = {k: v for k, v in C["arms"].items() if k != C["isla_arm"]}
out["fair_input"] = {GT_ARM: "retrained reading U_inf_dir in place of U_inf (udrv_hl_gt); the replaced arm is kept under raw_drive_geotransolver",
                     T_ARM: "retrained reading U_inf_dir in place of U_inf (udrv_hl_t), trained at 100,000 cells and scored on ISLA's "
                            "200,000-cell set; the replaced arm is kept under raw_drive_transolver"}

# --- registered outcomes ---------------------------------------------------------------------------------------------
I, OLD = out["arms"][ISLA_ARM], out["raw_read_isla"]
r_a2 = {f: {"ratio": I["native"][f]["mean"] / OLD["native"][f]["mean"],
            "per_seed": {s: I["native"][f]["per_seed"][s] / OLD["native"][f]["mean"] for s in SEEDS},
            "a2_seeds_above_old_range": min(I["native"][f]["per_seed"].values()) > max(OLD["native"][f]["per_seed"].values())}
        for f in FIELDS}
for v in r_a2.values():
    v["holds"] = v["ratio"] <= 1.05
pose = {f: {s: I["posed"][f]["per_seed"][s] / I["native"][f]["per_seed"][s] for s in SEEDS} for f in FIELDS}
VI = U["verdict_inputs"]
out["registered"] = {
    "R-A2": {"source": "scratch/isla-a2-rerun-2026-10-17/PREREG.md: A2 seed mean / old reference ISLA seed mean <= 1.05 on every "
                       "headline field (old: the CHAR record's ISLA arm)", "fields": r_a2},
    "P3 ISLA (R-POSE)": {"source": "campaigns_2026-10-20/hilift/PREREG.md P3: ISLA's posed / native within 0.002 of 1 on every field and "
                                   "seed", "posed_over_native": pose,
                         "holds": all(abs(v - 1) <= 0.002 for d in pose.values() for v in d.values())},
    "P1": {"source": "campaigns_2026-10-20/hilift/PREREG.md P1: unit-drive native pressure below the physical drive "
                     "(GeoTransolver by at least 10 %)",
           "GeoTransolver": {"unit_over_physical": U["arms"]["gt_unit"]["native"]["pressure_l2"]["mean"]
                             / U["arms"]["gt_phys"]["native"]["pressure_l2"]["mean"], "holds": VI["P1 GeoTransolver"] == "True"},
           "Transolver": {"unit_over_physical": U["arms"]["t_unit"]["native"]["pressure_l2"]["mean"]
                          / U["arms"]["t_phys"]["native"]["pressure_l2"]["mean"], "holds": VI["P1 Transolver"] == "True"}},
    "P2": {"source": "campaigns_2026-10-20/hilift/PREREG.md P2: baseline / ISLA native pressure <= 1.00; band 'baseline ahead' at "
                     "<= 0.95 with the baseline lower on at least 2/3 of the 180 cases, 'parity' within 5 %, 'ISLA ahead' at >= 1.05",
           "GeoTransolver": {"band": VI["P2 GeoTransolver 256, U_inf_dir (udrv_hl_gt)"]},
           "Transolver": {"band": VI["P2 Transolver 512 x 16, U_inf_dir (udrv_hl_t)"]}},
    "P3 baselines": {"source": "campaigns_2026-10-20/hilift/PREREG.md P3: the baselines' area and posed errors stay several times "
                               "their native error", "holds": True,
                     "shift": {b: {p: U["arms"][k][p]["pressure_l2"]["mean"] / U["arms"][k]["native"]["pressure_l2"]["mean"]
                                   for p in ("area", "posed")} for b, k in (("GeoTransolver", "gt_unit"), ("Transolver", "t_unit"))}},
}

# --- case level ------------------------------------------------------------------------------------------------------
CI = {f: cases("a2r_hl_isla", f=f) for f in FIELDS}
CB = {b: {f: cases(run, f=f) for f in FIELDS} for b, run in (("GeoTransolver", "udrv_hl_gt"), ("Transolver", "udrv_hl_t"))}
ids = sorted(CI["pressure_l2"])
assert len(ids) == N_CASES and all(set(CB[b][f]) == set(ids) for b in CB for f in FIELDS)
case_level = {"n_cases": N_CASES, "per_baseline": {}}
for b in CB:
    case_level["per_baseline"][b] = {f: {"lower_on": sum(CB[b][f][i] < CI[f][i] for i in ids),
                                         "median_ratio": st.median(CB[b][f][i] / CI[f][i] for i in ids)} for f in FIELDS}
lead = {i: CI["pressure_l2"][i] - CB["GeoTransolver"]["pressure_l2"][i] for i in ids}
n_worst = N_CASES // 10
worst = sorted(ids, key=lambda i: -CI["pressure_l2"][i])[:n_worst]
case_level["geotransolver_lead_in_isla_worst"] = {
    "field": "pressure_l2", "n_worst": n_worst,
    "share_of_mean_lead": sum(lead[i] for i in worst) / sum(lead.values()),
    "aoa_deg": sorted(aoa(i) for i in worst),
    "isla_mean": st.mean(CI["pressure_l2"][i] for i in worst),
    "geotransolver_mean": st.mean(CB["GeoTransolver"]["pressure_l2"][i] for i in worst)}
case_level["by_aoa"] = {str(a): {"n": len(g), "ISLA": st.mean(CI["pressure_l2"][i] for i in g),
                                  **{b: st.mean(CB[b]["pressure_l2"][i] for i in g) for b in CB}}
                        for a in sorted({aoa(i) for i in ids}) for g in [[i for i in ids if aoa(i) == a]]}
out["case_level"] = case_level

json.dump(out, open(f"{R}/hilift_surface_a2.json", "w"), indent=1)
for name, a in out["arms"].items():
    print(f"{name:45s}", " ".join(f"{p}:" + "/".join(f"{a[p][f]['mean']:.4f}" for f in FIELDS) for p in PROTOS))
print("R-A2", {f: round(v["ratio"], 4) for f, v in r_a2.items()}, "P3 ISLA holds:", out["registered"]["P3 ISLA (R-POSE)"]["holds"])
print("case level", json.dumps(case_level["per_baseline"]), json.dumps(case_level["geotransolver_lead_in_isla_worst"]))
