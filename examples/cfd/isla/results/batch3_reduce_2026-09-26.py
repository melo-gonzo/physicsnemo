"""Third batch 2026-09-26: OLBC-MECH combined channel at 2,048 Poisson cases (om rows 11-12) and FRAME-3D-R2 (fa rows 6-7)."""
import json, glob, math, statistics as st, os
R = os.path.dirname(os.path.abspath(__file__))
def per_case(run, camp, field):
    fs = glob.glob(f"{R}/campaigns_{camp}/*/metrics/{run}/**/metrics.jsonl", recursive=True) + glob.glob(f"{R}/campaigns_{camp}/*/metrics/{run}/metrics.jsonl")
    p = sorted(set(fs))[0]
    v = [json.loads(l)["metrics"][field] for l in open(p) if '"infer_step"' in l]
    return st.mean(v), len(v)
out = {"protocol": "2026-09-26 third batch. OLBC Poisson at 2,048 cases: float32 per-case relative L2 on u over the 256 test cases; ISLA R1 (inlet-free, K = 0) baseline and GeoTransolver-512 from results/olbc_2026-09-22.json; single-channel arms and the combined frame + extension arm from campaign om (rows 11-12 are the 2,048 combined pair). Fractions are of the log-gap ln(base/arm)/ln(base/gt512). FRAME-3D-R2: DrivAerML Surface, 435 cars, 200,000 cells, 48 validation cars, ISLA R2 (384x24) with inlet + body x, y, z at lr 5e-4; R1 references from results/frame3d_2026-09-26.json.", "groups": {}}
ol = json.load(open(f"{R}/olbc_2026-09-22.json"))["lanes"]
base = {s: ol[f"ol_poisson_isla_r1_n2048_seed{s}"]["fields"]["u"]["mean"] for s in (42, 43)}
gt = {s: ol[f"ol_poisson_gt512_n2048_seed{s}"]["fields"]["u"]["mean"] for s in (42, 43)}
b, g = st.mean(base.values()), st.mean(gt.values())
grp = {"base": b, "base_per_seed": base, "gt512": g, "gt512_per_seed": gt}
def frac(u): return math.log(b / u) / math.log(b / g)
for arm in ["ext", "frame2", "leak", "s1024", "frame_ext"]:
    ps = {}
    for s in (42, 43):
        try: ps[s] = per_case(f"om_poisson_isla_r1_{arm}_n2048_seed{s}", "2026-09-22", "u_l2")[0]
        except IndexError: pass
    u = st.mean(ps.values())
    grp[arm] = {"u": u, "per_seed": ps, "n_seeds": len(ps), "frac": frac(u), "gt_over": g / u}
f, e = grp["frame2"]["frac"], grp["ext"]["frac"]
grp["additive_threshold"] = min(1.0, f + e - 0.1)
grp["redundant_band"] = [max(f, e) - 0.1, max(f, e) + 0.1]
c = grp["frame_ext"]["frac"]
grp["verdict"] = "additive" if c >= grp["additive_threshold"] else ("redundant" if abs(c - max(f, e)) <= 0.1 else "between")
out["groups"]["olbc_combined_2048"] = grp
f3 = json.load(open(f"{R}/frame3d_2026-09-26.json"))
r1 = f3["seed_means"]["isla_r1"]; k4 = f3["seed_means"]["isla_r1_k4"]; r2 = f3["seed_means"]["isla_r2"]; g5 = f3["seed_means"]["gt512"]
ps = {s: {fld: per_case(f"fa_dr435_frame4_r2_seed{s}", "2026-09-23", fld)[0] for fld in ("pressure_l2", "wss_l2")} for s in (42, 43)}
m = {fld: st.mean(ps[s][fld] for s in ps) for fld in ("pressure_l2", "wss_l2")}
grp2 = {"per_seed": ps, "seed_means": m, "ratio_to_r1": {k: m[k] / r1[k] for k in m}, "ratio_to_k4": {k: m[k] / k4[k] for k in m}, "ratio_to_r2": {k: m[k] / r2[k] for k in m}, "gt512_over": {k: g5[k] / m[k] for k in m}, "product_of_gains": {k: (k4[k] / r1[k]) * (r2[k] / r1[k]) for k in m}}
p = grp2["ratio_to_r1"]["pressure_l2"]
grp2["verdict"] = "additive" if p <= 0.93 else ("redundant" if abs(p - k4["pressure_l2"] / r1["pressure_l2"]) <= 0.01 else "between")
out["groups"]["frame3d_r2"] = grp2
json.dump(out, open(f"{R}/batch3_reductions_2026-09-26.json", "w"), indent=1)
for k, v in out["groups"].items():
    print("==", k); print(json.dumps({a: b for a, b in v.items() if a not in ("per_seed",)}, indent=0))
