"""Deflection sealed test, 200,000-cell twin (P200K-FLEET group defl; prereg @sec-nb-deflection-prereg, instrument as
amended @sec-nb-deflection-amend). Arms: ISLA in the reference configuration (rb_hl_isla_deflection_seed42/43: 384 wide,
cosine routing, lr 5e-4, REF-BOARD rows 20-21, code_ref) and unit-direction GeoTransolver (p200k_hl_gt_unit_deflection_seed42/43,
lr 1e-3), trained on the 144 milder geometries at 200,000 cells, 500 epochs; evaluated once in float32 on the sealed 360-case
high-deflection test list at 200,000 cells (highlift/hl_defl_test_eval_p200k_aga.sbatch -> hl_test_evals/<run>) and on the
180-case in-distribution validation list (p200k_fp32/<run>). The first ISLA twin (p200k_hl_isla_deflection_seed42/43, 192 wide,
linear routing, lr 1e-3) is NOT an arm: both seeds left their minimum (val loss 0.0088 / 0.0101 at epoch 26) for the mean-field
plateau by epoch 51-76 and ended at 0.143 / 0.157 (@sec-nb-p200k-deflection-plateau); recorded under "superseded". Same reduction as results/deflection_sealed_test_2026-09-15.json (the 10,000-cell arms):
per-field seed means on test and val, the degradation d = test / val, ISLA over GeoTransolver on both splits; plus
the 10k-to-200k comparison. Bars (10k prereg, applied unchanged): carried = ISLA <= 0.7x GeoTransolver on sealed
pressure with d <= 4; falsified = ISLA >= 0.9x or both d >= 8."""
import glob
import json
import statistics as st

R = "/home/psharpe/gh/physicsnemo/examples/cfd/mesh_transformer/results"
D = f"{R}/campaigns_2026-09-19/defl"
ARMS = {"isla": "rb_hl_isla_deflection_seed{s}", "gt": "p200k_hl_gt_unit_deflection_seed{s}"}
VAL_DIR = {"isla": f"{R}/campaigns_2026-09-23/rb/metrics", "gt": f"{D}/metrics"}
SEEDS = ("42", "43")


def per_case(path_glob):
    fs = glob.glob(path_glob, recursive=True)
    if not fs:
        return None
    rows = [json.loads(l)["metrics"] for l in open(sorted(fs)[0]) if '"infer_step"' in l]
    keys = sorted(k for k in rows[0] if k.endswith("_l2"))
    return {k: st.mean(r[k] for r in rows if k in r) for k in keys}, len(rows)


per_run = {"test": {}, "val": {}}
n_cases = {}
for arm, pat in ARMS.items():
    for s in SEEDS:
        run = pat.format(s=s)
        t = per_case(f"{D}/test/{run}/**/metrics.jsonl")
        v = per_case(f"{VAL_DIR[arm]}/{run}/**/metrics.jsonl")
        if t:
            per_run["test"][run], n_cases[("test", run)] = t
        if v:
            per_run["val"][run], n_cases[("val", run)] = v

keys = sorted(set().union(*(set(m) for m in per_run["test"].values()), *(set(m) for m in per_run["val"].values())))
arms = {}
for arm, pat in ARMS.items():
    arms[arm] = {}
    for k in keys:
        ts = [per_run["test"][pat.format(s=s)][k] for s in SEEDS if pat.format(s=s) in per_run["test"]]
        vs = [per_run["val"][pat.format(s=s)][k] for s in SEEDS if pat.format(s=s) in per_run["val"]]
        e = {"test_seeds": ts, "val_seeds": vs}
        if ts:
            e["test_mean"] = st.mean(ts)
        if vs:
            e["val_mean"] = st.mean(vs)
        if ts and vs:
            e["degradation_d"] = e["test_mean"] / e["val_mean"]
        arms[arm][k] = e

readout = {}
for k in keys:
    a, g = arms["isla"][k], arms["gt"][k]
    r = {}
    if "test_mean" in a and "test_mean" in g:
        r["isla_over_gt_test"] = a["test_mean"] / g["test_mean"]
    if "degradation_d" in a:
        r["d_isla"] = a["degradation_d"]
    if "degradation_d" in g:
        r["d_gt"] = g["degradation_d"]
    if "val_mean" in a and "val_mean" in g:
        r["isla_over_gt_val"] = a["val_mean"] / g["val_mean"]
    readout[k] = r

p = readout.get("pressure_l2", {})
complete = all(len(arms[a][k].get("test_seeds", [])) == 2 and len(arms[a][k].get("val_seeds", [])) == 2 for a in ARMS for k in ("pressure_l2",))
verdict = {"pressure_readout": p, "complete": complete}
if complete:
    verdict["carried"] = p["isla_over_gt_test"] <= 0.7 and p["d_isla"] <= 4
    verdict["falsified"] = p["isla_over_gt_test"] >= 0.9 or (p["d_isla"] >= 8 and p["d_gt"] >= 8)
    verdict["neither"] = not (verdict["carried"] or verdict["falsified"])

ten_k = json.load(open(f"{R}/deflection_sealed_test_2026-09-15.json"))
compare = {}
for k in keys:
    if k in ten_k["readout"] and "isla_over_gt_test" in readout.get(k, {}):
        compare[k] = {
            "isla_over_gt_test_10k": ten_k["readout"][k]["isla_over_gt_test"],
            "isla_over_gt_test_200k": readout[k]["isla_over_gt_test"],
            "d_isla_10k": ten_k["readout"][k]["d_isla"], "d_isla_200k": readout[k].get("d_isla"),
            "d_gt_10k": ten_k["readout"][k]["d_gt"], "d_gt_200k": readout[k].get("d_gt"),
            "isla_test_200k_over_10k": arms["isla"][k]["test_mean"] / ten_k["arms"]["isla"][k]["test_mean"],
            "gt_test_200k_over_10k": arms["gt"][k]["test_mean"] / ten_k["arms"]["gt"][k]["test_mean"],
        }

superseded = {}
for s_ in SEEDS:
    run = f"p200k_hl_isla_deflection_seed{s_}"
    v = per_case(f"{D}/metrics/{run}/**/metrics.jsonl")
    if v:
        superseded[run] = {"val": v[0], "reason": "mean-field plateau from epoch 51-76 (val loss 0.143 / 0.157 at 500); linear routing, 192 wide"}
out = {"protocol": __doc__, "superseded": superseded, "per_run": per_run, "n_cases": {f"{s}:{r}": n for (s, r), n in n_cases.items()}, "keys": keys,
       "arms": arms, "readout": readout, "verdict": verdict, "compare_10k": compare}
json.dump(out, open(f"{R}/deflection_p200k_2026-09-29.json", "w"), indent=1)

print(f"{'field':<16} {'ISLA test':>9} {'GT test':>9} {'ISLA/GT':>8} {'d_isla':>7} {'d_gt':>6} {'ISLA/GT val':>11} | {'10k ISLA/GT':>11}")
for k in ("pressure_l2", "velocity_l2", "tau_wall_l2", "temperature_l2", "density_l2"):
    if k not in readout:
        continue
    a, g, r = arms["isla"][k], arms["gt"][k], readout[k]
    print(f"{k:<16} {a.get('test_mean', float('nan')):>9.4f} {g.get('test_mean', float('nan')):>9.4f} {r.get('isla_over_gt_test', float('nan')):>8.3f} "
          f"{r.get('d_isla', float('nan')):>7.3f} {r.get('d_gt', float('nan')):>6.3f} {r.get('isla_over_gt_val', float('nan')):>11.3f} | "
          f"{ten_k['readout'].get(k, {}).get('isla_over_gt_test', float('nan')):>11.3f}")
print("per seed pressure (test / val): ISLA", arms["isla"]["pressure_l2"], "\n  GT", arms["gt"]["pressure_l2"])
print("verdict:", json.dumps(verdict))
