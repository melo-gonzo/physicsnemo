"""SCALE-LAW reduction (notebook #sec-nb-campaigns-2026-09-21-prereg): surface-pressure error against training-set size n
and parameter count P for cosine ISLA (192/384/768 wide) and GeoTransolver (256/512/1,024 wide) on DrivAerML
(27/54/109/218 cars) and HiLiftAeroML (35/210/510 wings), float32 at 100,000 cells, two seeds.

Fits, per model and dataset, the separable law  eps(n, P) = a n^-alpha + b P^-beta + c  by least squares on the seed
means (scipy curve_fit, log-parametrized positive coefficients), and the marginal power laws eps(n) at each width and
eps(P) at each n. Configurations with fewer than two scored seeds are listed as pending; the fit runs on what is complete
and says how many points it used.

usage: python scalelaw_reduce.py <fetched fp32 dir> <lanes.tsv> <cap435.json> <out.json>
"""
from __future__ import annotations

import csv, glob, json, statistics as st, sys
from pathlib import Path

import numpy as np

root, lanes, cap_path, out_path = (Path(a) for a in sys.argv[1:5])
CAP = json.load(open(cap_path))["configurations"]
PARAMS = {("isla", 192): CAP["isla|w192|d12|s256|lr1e-3"]["params"], ("isla", 384): CAP["isla|w384|d12|s256|lr1e-3"]["params"],
          ("isla", 768): CAP["isla|w768|d12|s256|lr5e-4"]["params"], ("gt", 256): CAP["gt|w256|d12|s256|lr3e-3"]["params"],
          ("gt", 512): CAP["gt|w512|d12|s256|lr3e-3"]["params"], ("gt", 1024): CAP["gt|w1024|d12|s256|lr3e3"]["params"]}
N_OF = {"n27": 27, "n54": 54, "n109": 109, "n218": 218, "super_scarce": 35, "scarce": 210, "medium": 510}


def score(run):
    ps = glob.glob(str(root / run / "**" / "metrics.jsonl"), recursive=True)
    if not ps:
        return None
    rows = [json.loads(l) for l in open(ps[0]) if l.strip()]
    s = [r for r in rows if r.get("phase") == "infer_summary"]
    return s[0]["metrics"].get("pressure_l2") if s else None


rows = list(csv.DictReader(open(lanes), delimiter="\t"))
conf: dict = {}
for r in rows:
    rid = r["run_id"]; parts = rid.split("_")
    ds = "drivaerml" if parts[1] == "dr" else "hilift"
    model = "isla" if parts[2] == "isla" else "gt"
    w = int([p for p in parts if p.startswith("w")][0][1:])
    rung = [p for p in N_OF if f"_{p}_" in rid][0]
    key = f"{ds}|{model}|w{w}|{rung}"
    conf.setdefault(key, {"dataset": ds, "model": model, "width": w, "n": N_OF[rung], "params": PARAMS[(model, w)], "runs": {}})
    conf[key]["runs"][rid] = score(rid)
for c in conf.values():
    vals = [v for v in c["runs"].values() if v is not None]
    c["n_scored"] = len(vals); c["pressure_l2_seeds"] = vals; c["pressure_l2_mean"] = st.mean(vals) if vals else None
    c["status"] = "complete" if len(vals) >= 2 else ("partial" if vals else "pending")

fits: dict = {}
try:
    from scipy.optimize import curve_fit

    def law(X, la, lalpha, lb, lbeta, lc):
        n, P = X
        return np.exp(la) * n ** (-np.exp(lalpha)) + np.exp(lb) * P ** (-np.exp(lbeta)) + np.exp(lc)

    for ds in ("drivaerml", "hilift"):
        for model in ("isla", "gt"):
            pts = [c for c in conf.values() if c["dataset"] == ds and c["model"] == model and c["pressure_l2_mean"] is not None]
            f = {"n_points": len(pts)}
            distinct_n, distinct_P = len({c["n"] for c in pts}), len({c["params"] for c in pts})
            f["distinct_n"], f["distinct_P"] = distinct_n, distinct_P
            if distinct_n >= 3 and distinct_P >= 3:   # the separable law has five coefficients; two rungs cannot pin an exponent
                n = np.array([c["n"] for c in pts], float); P = np.array([c["params"] for c in pts], float); y = np.array([c["pressure_l2_mean"] for c in pts])
                try:
                    p0 = [np.log(0.3), np.log(0.5), np.log(0.5), np.log(0.3), np.log(0.02)]
                    popt, _ = curve_fit(law, (n, P), y, p0=p0, maxfev=200000)
                    pred = law((n, P), *popt)
                    f.update({"a": float(np.exp(popt[0])), "alpha": float(np.exp(popt[1])), "b": float(np.exp(popt[2])), "beta": float(np.exp(popt[3])),
                              "c": float(np.exp(popt[4])), "rms_rel_residual": float(np.sqrt(np.mean((pred / y - 1) ** 2)))})
                except Exception as e:  # noqa: BLE001
                    f["error"] = repr(e)
            # marginal exponents: log-log slope of eps against n at each width, and against P at each n
            f["alpha_by_width"] = {}; f["beta_by_n"] = {}
            for w in sorted({c["width"] for c in pts}):
                q = sorted([c for c in pts if c["width"] == w], key=lambda c: c["n"])
                if len(q) >= 2:
                    s_ = np.polyfit(np.log([c["n"] for c in q]), np.log([c["pressure_l2_mean"] for c in q]), 1)[0]; f["alpha_by_width"][w] = float(-s_)
            for nn in sorted({c["n"] for c in pts}):
                q = sorted([c for c in pts if c["n"] == nn], key=lambda c: c["params"])
                if len(q) >= 2:
                    s_ = np.polyfit(np.log([c["params"] for c in q]), np.log([c["pressure_l2_mean"] for c in q]), 1)[0]; f["beta_by_n"][nn] = float(-s_)
            fits[f"{ds}|{model}"] = f
except ImportError:
    fits["error"] = "scipy missing"

out = {"study": "SCALE-LAW: surface-pressure error against training cases and parameters, cosine ISLA and GeoTransolver, DrivAerML and HiLiftAeroML, float32 at 100,000 cells, two seeds",
       "law": "eps(n, P) = a n^-alpha + b P^-beta + c (least squares on seed means; positive coefficients)", "configurations": conf, "fits": fits}
out_path.write_text(json.dumps(out, indent=2))
for ds in ("drivaerml", "hilift"):
    print(f"== {ds}")
    for key, c in sorted(conf.items(), key=lambda kv: (kv[1]["dataset"], kv[1]["model"], kv[1]["width"], kv[1]["n"])):
        if c["dataset"] != ds:
            continue
        tag = f"{c['pressure_l2_mean']:.4f} ({' / '.join(f'{v:.4f}' for v in c['pressure_l2_seeds'])})" if c["pressure_l2_mean"] else "pending"
        print(f"  {c['model']:4s} w{c['width']:<5d} n={c['n']:<4d} P={c['params']/1e6:6.1f}M  {tag}  [{c['status']}]")
for k, f in fits.items():
    print(k, json.dumps(f)[:400])
