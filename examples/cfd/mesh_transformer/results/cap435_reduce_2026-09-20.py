"""CAP-435 reduction (prereg #sec-nb-cap435-prereg): width, depth and slice ladders at 435 DrivAerML cars, 100,000 cells,
float32 at the 200,000-cell protocol, two seeds; joined with the ISO-COST grid's 100,000-cell points (ISLA 192/384/768,
GeoTransolver 256/512, Transolver 256) so the ladders are complete.

usage: python reduce_cap435.py <fetched p200k_fp32 dir> <isocost_reduction.json> <out.json>
"""
import glob, json, re, statistics as st, sys
from pathlib import Path

root, iso_path, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
NAME = re.compile(r"cap_dr_(isla|gt|transolver)_w(\d+)(?:_d(\d+))?(?:_s(\d+))?(?:_lr([0-9.e_]+))?_c100k_seed(\d+)")
PARAMS = {("isla", 512, 12, 256): 60298076, ("isla", 1024, 12, 256): 237508956, ("isla", 1536, 12, 256): 531636060,
          ("isla", 384, 6, 256): 17257266, ("isla", 384, 24, 256): 68276016, ("isla", 384, 48, 256): 136301016,
          ("isla", 384, 12, 1024): 38107356, ("isla", 384, 12, 256): 34263516, ("isla", 768, 12, 256): 134288988,
          ("isla", 192, 12, 256): 8912796, ("gt", 1024, 12, 256): 142734464, ("gt", 512, 24, 256): 71120620,
          ("gt", 512, 12, 256): 35848576, ("gt", 256, 12, 256): 9046016, ("transolver", 512, 8, 256): 23887428,
          ("transolver", 256, 8, 256): 6021956}
RULE_LR = {("isla", 192): "1e-3", ("isla", 384): "5e-4", ("isla", 512): "3.75e-4", ("isla", 768): "2.5e-4",
           ("isla", 1024): "1.875e-4", ("isla", 1536): "1.25e-4", ("gt", 256): "3e-3", ("gt", 512): "3e-3", ("gt", 1024): "3e-3",
           ("transolver", 256): "3e-3", ("transolver", 512): "3e-3"}


def val(run):
    ps = glob.glob(str(root / run / "**" / "metrics.jsonl"), recursive=True)
    if not ps:
        return None
    rows = [json.loads(l) for l in open(ps[0]) if l.strip()]
    s = [r for r in rows if r.get("phase") == "infer_summary"]
    return {k: s[0]["metrics"][k] for k in ("pressure_l2", "wss_l2")} if s else None


cfgs = {}
for d in sorted(glob.glob(str(root / "cap_dr_*"))):
    m = NAME.fullmatch(Path(d).name)
    if not m:
        continue
    model, w, depth, slices, lr, seed = m.groups()
    w, depth, slices = int(w), int(depth or (8 if model == "transolver" else 12)), int(slices or 256)
    lr = (lr.replace("_", "-") if lr else RULE_LR[(model, w)])
    key = f"{model}|w{w}|d{depth}|s{slices}|lr{lr}"
    c = cfgs.setdefault(key, dict(model=model, width=w, depth=depth, slices=slices, lr=lr, params=PARAMS.get((model, w, depth, slices)), runs={}))
    c["runs"][Path(d).name] = val(Path(d).name)
# join the ISO-COST 100k points
iso = json.load(open(iso_path))["configurations"]
for k, c in iso.items():
    if c["cells"] != 100000 or c["epochs"] != 500 or c["status"] != "complete":
        continue
    model = {"isla": "isla", "gt_unit": "gt", "transolver_unit": "transolver"}[c["model"]]
    depth = 8 if model == "transolver" else 12
    key = f"{model}|w{c['width']}|d{depth}|s256|lr{c['lr']}"
    cfgs[key] = dict(model=model, width=c["width"], depth=depth, slices=256, lr=c["lr"], params=c["params"],
                     runs={r: {"pressure_l2": v} for r, v in zip(c["runs"], c["pressure_l2_seeds"])}, source="ISO-COST")
for c in cfgs.values():
    got = [v for v in c["runs"].values() if v]
    c["n_evaluated"], c["n_runs"] = len(got), len(c["runs"])
    c["status"] = "complete" if got and len(got) == len(c["runs"]) else "pending"
    if c["status"] == "complete":
        c["pressure_l2_seeds"] = [v["pressure_l2"] for v in got]
        c["pressure_l2_mean"] = st.mean(c["pressure_l2_seeds"])
        c["seed_spread"] = max(c["pressure_l2_seeds"]) / min(c["pressure_l2_seeds"]) - 1

done = {k: c for k, c in cfgs.items() if c["status"] == "complete"}


def best(model, w, depth=None, slices=256):
    """Best complete configuration at (model, width[, depth]) over learning rates (GeoTransolver keeps its better rate)."""
    depth = depth or (8 if model == "transolver" else 12)
    cands = [c for c in done.values() if c["model"] == model and c["width"] == w and c["depth"] == depth and c["slices"] == slices]
    return min(cands, key=lambda c: c["pressure_l2_mean"]) if cands else None


out = {"study": "CAP-435: width, depth and slice ladders at 435 DrivAerML cars, 100,000 cells, 500 epochs, float32 at 200,000 cells, two seeds; ISO-COST 100k points joined",
       "configurations": cfgs, "ladders": {}, "bars": {}}
for model, widths in (("isla", [192, 384, 512, 768, 1024, 1536]), ("gt", [256, 512, 1024]), ("transolver", [256, 512])):
    lad = []
    for w in widths:
        b = best(model, w)
        lad.append({"width": w, "params": b["params"] if b else PARAMS.get((model, w, 8 if model == "transolver" else 12, 256)),
                    "pressure_l2": b["pressure_l2_mean"] if b else None, "seeds": b["pressure_l2_seeds"] if b else None,
                    "lr": b["lr"] if b else None, "status": "complete" if b else "pending"})
    out["ladders"][f"{model}_width"] = lad
    steps = [(a, bb) for a, bb in zip(lad, lad[1:]) if a["pressure_l2"] and bb["pressure_l2"]]
    gains = [{"from": a["width"], "to": bb["width"], "gain": 1 - bb["pressure_l2"] / a["pressure_l2"]} for a, bb in steps]
    turn = next((g for g in gains if g["gain"] < 0.01), None)
    out["bars"][f"{model}_turnover"] = {"gains": gains, "first_step_under_1pct": turn}
out["ladders"]["isla_depth_w384"] = [{"depth": d, **({"pressure_l2": b["pressure_l2_mean"], "seeds": b["pressure_l2_seeds"], "lr": b["lr"]} if (b := best("isla", 384, d)) else {"pressure_l2": None})} for d in (6, 12, 24, 48)]
d12, d24, d6 = best("isla", 384, 12), best("isla", 384, 24), best("isla", 384, 6)
if d12 and d24:
    r = d24["pressure_l2_mean"] / d12["pressure_l2_mean"]
    out["bars"]["isla_depth_24_vs_12"] = {"ratio": r, "verdict": "depth carries (>= 3%)" if r <= 0.97 else ("depth hurts (> 3%)" if r >= 1.03 else "null (within 3%)")}
if d12 and d6:
    r = d6["pressure_l2_mean"] / d12["pressure_l2_mean"]
    out["bars"]["isla_depth_6_vs_12"] = {"ratio": r, "verdict": "depth not binding (6 within 3% of 12)" if r <= 1.03 else "12 layers needed"}
g12, g24 = best("gt", 512, 12), best("gt", 512, 24)
if g12 and g24:
    r = g24["pressure_l2_mean"] / g12["pressure_l2_mean"]
    out["bars"]["gt_depth_24_vs_12"] = {"ratio": r, "verdict": "depth carries (>= 3%)" if r <= 0.97 else ("depth hurts" if r >= 1.03 else "within 3%")}
s256, s1024 = best("isla", 384, 12, 256), best("isla", 384, 12, 1024)
if s256 and s1024:
    r = s1024["pressure_l2_mean"] / s256["pressure_l2_mean"]
    out["bars"]["isla_slices_1024_vs_256"] = {"ratio": r, "verdict": "slices null (within 2%)" if abs(1 - r) <= 0.02 else ("slices help" if r < 0.98 else "slices hurt")}
for a, b_, label in ((("isla", 768), ("gt", 1024), "isla768_vs_gt1024"), (("isla", 384), ("gt", 512), "isla384_vs_gt512")):
    A, B = best(*a), best(*b_)
    if A and B:
        r = B["pressure_l2_mean"] / A["pressure_l2_mean"]
        out["bars"][f"iso_param_{label}"] = {"gt_over_isla": r, "params": [A["params"], B["params"]],
                                             "verdict": "parity (within 3%)" if abs(1 - r) <= 0.03 else ("ISLA leads" if r > 1.03 else "GeoTransolver leads")}
# rate controls at a glance
out["rate_controls"] = {k: {"pressure_l2": c.get("pressure_l2_mean"), "seeds": c.get("pressure_l2_seeds"), "status": c["status"]}
                        for k, c in cfgs.items() if c["model"] == "isla" and c["lr"] != RULE_LR.get((c["model"], c["width"]))}
out_path.write_text(json.dumps(out, indent=2))
print(f"{len(done)}/{len(cfgs)} configurations complete")
for k, c in sorted(cfgs.items(), key=lambda kv: (kv[1]['model'], kv[1]['width'], kv[1]['depth'], kv[1]['slices'], kv[1]['lr'])):
    tag = f"{c['pressure_l2_mean']:.4f} (seeds {' / '.join(f'{v:.4f}' for v in c['pressure_l2_seeds'])})" if c["status"] == "complete" else f"pending {c['n_evaluated']}/{c['n_runs']}"
    print(f"  {k:34s} params {(c['params'] or 0)/1e6:6.1f}M  {tag}")
print(json.dumps(out["bars"], indent=1)[:2500])
