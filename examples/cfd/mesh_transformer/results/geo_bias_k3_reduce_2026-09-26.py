"""GEO-BIAS-K3 reduction (prereg @sec-nb-geo-bias-k3-prereg): inlet + x + y (K = 3) WITH the MLP routing bias on the local
160-airfoil AirfRANS rung (192 wide, 120 training airfoils, 20,000 queries, 300 epochs, lr 1e-3, eager region), two seeds,
float32 per-case relative L2 on the 40 test airfoils, cases matched by name (the manifest index prefix shifted between
splits). Comparators from the same local ladder: the K = 3 linear arm (frameaf_frame), the K = 1 MLP arm (frameaf_mlpbias)
and the inlet-only linear base (frameaf_base). Bars: one mechanism on velocity = K3-MLP within 3% of K3-linear; additive
velocity = >= 8% better; additive pressure = >= 6% better; not additive pressure = within 3%.
usage: python geo_bias_k3_reduce_2026-09-26.py [out.json]"""
import json, re, statistics as st, sys, os
ROOT = "/home/psharpe/data/runs_local/infer"; RES = os.path.dirname(os.path.abspath(__file__))
F = ["pressure_l2", "velocity_l2", "nut_l2"]
def per_case(run):
    out = {}
    for l in open(f"{ROOT}/{run}/{run}/metrics.jsonl"):
        r = json.loads(l)
        if r.get("phase") == "infer_step": out[re.sub(r"^\d+_", "", r["sample_id"])] = r["metrics"]
    return out
ARMS = {"base": "frameaf_base_seed{}", "mlpbias_k1": "frameaf_mlpbias_seed{}", "frame_k3": "frameaf_frame_seed{}", "framemlp_k3": "frameaf_framemlp_seed{}"}
data = {a: {s: per_case(p.format(s)) for s in (42, 43)} for a, p in ARMS.items()}
ids = sorted(set.intersection(*(set(d) for arm in data.values() for d in arm.values())))
out = {"protocol": __doc__, "n_cases": len(ids), "per_run": {}, "seed_means": {}}
for a in ARMS:
    out["per_run"][a] = {str(s): {f: st.fmean(data[a][s][c][f] for c in ids) for f in F} for s in (42, 43)}
    out["seed_means"][a] = {f: st.fmean(out["per_run"][a][str(s)][f] for s in (42, 43)) for f in F}
sm = out["seed_means"]
out["ratios"] = {"framemlp_over_frame": {f: sm["framemlp_k3"][f] / sm["frame_k3"][f] for f in F}, "framemlp_over_mlpbias": {f: sm["framemlp_k3"][f] / sm["mlpbias_k1"][f] for f in F},
                 "framemlp_over_base": {f: sm["framemlp_k3"][f] / sm["base"][f] for f in F}, "frame_over_base": {f: sm["frame_k3"][f] / sm["base"][f] for f in F}, "mlpbias_over_base": {f: sm["mlpbias_k1"][f] / sm["base"][f] for f in F}}
v, p = out["ratios"]["framemlp_over_frame"]["velocity_l2"], out["ratios"]["framemlp_over_frame"]["pressure_l2"]
out["verdict"] = {"velocity": "one mechanism" if abs(v - 1) <= 0.03 else ("additive" if v <= 0.92 else "between"),
                  "pressure": "additive" if p <= 0.94 else ("not additive" if abs(p - 1) <= 0.03 else "between")}
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else f"{RES}/geo_bias_k3_2026-09-26.json", "w"), indent=1)
print("cases", len(ids)); 
for a in ARMS: print(f"{a:12s}", {f: round(sm[a][f], 4) for f in F}, "seeds p:", [round(out['per_run'][a][str(s)]['pressure_l2'], 4) for s in (42, 43)])
for k, r in out["ratios"].items(): print(f"{k:22s}", {f: round(x, 3) for f, x in r.items()})
print(out["verdict"])
