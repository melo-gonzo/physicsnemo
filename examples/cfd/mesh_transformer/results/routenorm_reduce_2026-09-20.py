"""ROUTE-NORM reduction (prereg #sec-nb-routenorm-prereg): cosine (bounded) routing against linear routing at the rule
rates and at the rates that collapse linear routing, on DrivAerML (435 cars) and the 510-wing HiLift rung, 100,000 cells,
float32 at the 200,000-cell protocol, two seeds.

Linear references come from the ISO-COST and CAP-435 artifacts (DrivAerML) and from the SCALE-HL lanes (wings, when
scored). Bars: collapse removed (cosine lanes at collapse rates complete and finite with error within the clean range,
i.e. below 0.06); accuracy at rule rates within 1% of linear = neutral; width recovers if cosine 768 at 5e-4 or 1e-3 is
>= 2% below linear 384 (0.0411) with the two seeds within 1%.

usage: python reduce_routenorm.py <fetched p200k_fp32 dir> <isocost.json> <cap435.json> <out.json>
"""
import glob, json, re, statistics as st, sys
from pathlib import Path

root, iso_path, cap_path, out_path = (Path(a) for a in sys.argv[1:5])
NAME = re.compile(r"rn_(cos|lin)_w(\d+)_lr([0-9.e_]+?)(_med)?_c100k_seed(\d+)")


def val(run):
    ps = glob.glob(str(root / run / "**" / "metrics.jsonl"), recursive=True)
    if not ps:
        return None
    rows = [json.loads(l) for l in open(ps[0]) if l.strip()]
    s = [r for r in rows if r.get("phase") == "infer_summary"]
    return s[0]["metrics"]["pressure_l2"] if s else None


cfgs = {}
for d in sorted(glob.glob(str(root / "rn_*"))):
    m = NAME.fullmatch(Path(d).name)
    if not m:
        continue
    routing, w, lr, med, seed = m.groups()
    key = f"{'wings' if med else 'cars'}|{routing}|w{w}|lr{lr.replace('_', '-')}"
    c = cfgs.setdefault(key, dict(dataset="wings" if med else "cars", routing="cosine" if routing == "cos" else "linear", width=int(w), lr=lr.replace("_", "-"), runs={}))
    c["runs"][Path(d).name] = val(Path(d).name)
for c in cfgs.values():
    got = [v for v in c["runs"].values() if v is not None]
    c["status"] = "complete" if got and len(got) == len(c["runs"]) else "pending"
    if c["status"] == "complete":
        c["seeds"] = got; c["mean"] = st.mean(got); c["seed_spread"] = max(got) / min(got) - 1

# linear references on the cars
iso = json.load(open(iso_path))["configurations"]; cap = json.load(open(cap_path))["configurations"]
lin = {}
for c in iso.values():
    if c["model"] == "isla" and c["cells"] == 100000 and c["epochs"] == 500 and c["status"] == "complete":
        lin[(c["width"], c["lr"])] = c["pressure_l2_mean"]
for c in cap.values():
    if c["model"] == "isla" and c["depth"] == 12 and c["slices"] == 256 and c["status"] == "complete":
        lin[(c["width"], c["lr"])] = c["pressure_l2_mean"]

out = {"study": "ROUTE-NORM: cosine routing (T = 10) against linear routing at 100,000 cells, two seeds, float32 at 200,000 cells",
       "configurations": cfgs, "linear_references_cars": {f"w{w}|lr{lr}": v for (w, lr), v in lin.items()}, "bars": {}}
RULE = {192: "1e-3", 384: "5e-4", 768: "2.5e-4"}
COLLAPSE = [(384, "1e-3"), (768, "5e-4"), (768, "1e-3")]
cc = {}
for w, lr in COLLAPSE:
    c = cfgs.get(f"cars|cos|w{w}|lr{lr}")
    if c and c["status"] == "complete":
        cc[f"w{w}|lr{lr}"] = {"mean": c["mean"], "seeds": c["seeds"], "clean": all(v < 0.06 for v in c["seeds"]),
                              "linear_reference": lin.get((w, lr))}
if cc:
    out["bars"]["collapse_removed_cars"] = {"lanes": cc, "verdict": "removed (all clean)" if all(x["clean"] for x in cc.values()) else "not removed at every rate"}
neutral = {}
for w, lr in RULE.items():
    c = cfgs.get(f"cars|cos|w{w}|lr{lr}")
    if c and c["status"] == "complete" and (w, lr) in lin:
        r = c["mean"] / lin[(w, lr)]
        neutral[f"w{w}|lr{lr}"] = {"cosine": c["mean"], "linear": lin[(w, lr)], "cosine_over_linear": r,
                                   "verdict": "neutral (within 1%)" if abs(1 - r) <= 0.01 else ("cosine worse (> 2%)" if r > 1.02 else ("cosine better (> 2%)" if r < 0.98 else "within 2%"))}
out["bars"]["accuracy_at_rule_rates"] = neutral
best768 = [c for k, c in cfgs.items() if k.startswith("cars|cos|w768") and c["status"] == "complete"]
if best768 and (384, "5e-4") in lin:
    b = min(best768, key=lambda c: c["mean"])
    out["bars"]["width_recovers"] = {"best_cosine_768": {"lr": b["lr"], "mean": b["mean"], "seeds": b["seeds"], "seed_spread": b["seed_spread"]},
                                     "linear_384": lin[(384, "5e-4")], "ratio": b["mean"] / lin[(384, "5e-4")],
                                     "verdict": "width recovers (>= 2% below 384, seeds within 1%)" if b["mean"] <= 0.98 * lin[(384, "5e-4")] and b["seed_spread"] <= 0.01 else "not recovered"}
wings = {k: c for k, c in cfgs.items() if c["dataset"] == "wings"}
if wings:
    out["bars"]["wings"] = {k: {"status": c["status"], "mean": c.get("mean"), "seeds": c.get("seeds")} for k, c in wings.items()}
out_path.write_text(json.dumps(out, indent=2))
for k, c in sorted(cfgs.items()):
    tag = f"{c['mean']:.4f} (seeds {' / '.join(f'{v:.4f}' for v in c['seeds'])})" if c["status"] == "complete" else "pending"
    ref = lin.get((c["width"], c["lr"])) if c["dataset"] == "cars" else None
    print(f"  {k:28s} {tag:34s}" + (f" linear {ref:.4f}" if ref else ""))
print(json.dumps(out["bars"], indent=1)[:2500])
