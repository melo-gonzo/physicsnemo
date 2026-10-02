"""Tensile2D geometry variation (CHAR, 2026-10-02): for every converted PLAID Tensile2d case, the boundary polyline's node count,
the interior node count, the specimen half-width at five heights (x-extent of boundary nodes within a band around each height,
coordinates normalized to [-1, 1]) and the waist width (minimum of the profile). Decides whether the set is fixed-mesh or
geometry-varying. Output: one JSON with per-case rows and summary quantiles."""
import glob, json, os, sys
import numpy as np
D = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/datasets/plaid_tensile2d_pdmsh"
rows = []
for c in sorted(glob.glob(f"{D}/sample_*/sample_*.pdmsh")):
    b = f"{c}/_tensordict/boundaries/outer/_tensordict"; it = f"{c}/_tensordict/interior/_tensordict"
    try:
        mb = json.load(open(f"{b}/meta.json")); mi = json.load(open(f"{it}/meta.json"))
        pts = np.memmap(f"{b}/points.memmap", dtype=np.float32, mode="r").reshape(mb["points"]["shape"])[:, :2]
        n_int = int(mi["points"]["shape"][0])
    except Exception as e:
        rows.append({"case": os.path.basename(os.path.dirname(c)), "error": str(e)}); continue
    lo, hi = pts.min(0), pts.max(0); H = hi[1] - lo[1]
    ys = np.linspace(lo[1] + 0.1 * H, hi[1] - 0.1 * H, 5); prof = []
    for y in ys:
        sel = np.abs(pts[:, 1] - y) < 0.03 * H
        prof.append(float(np.ptp(pts[sel, 0])) if sel.sum() > 1 else None)
    rows.append({"case": os.path.basename(os.path.dirname(c)), "n_boundary": int(pts.shape[0]), "n_interior": n_int,
                 "bbox": [float(v) for v in (*lo, *hi)], "width_profile": prof, "waist_width": min(p for p in prof if p is not None)})
ok = [r for r in rows if "waist_width" in r]
w = np.array([r["waist_width"] for r in ok]); nb = np.array([r["n_boundary"] for r in ok]); ni = np.array([r["n_interior"] for r in ok])
out = {"protocol": __doc__, "n_cases": len(rows), "n_ok": len(ok),
       "waist_width": {"min": float(w.min()), "q25": float(np.quantile(w, .25)), "median": float(np.median(w)), "q75": float(np.quantile(w, .75)), "max": float(w.max()), "max_over_min": float(w.max() / w.min())},
       "n_boundary": {"min": int(nb.min()), "max": int(nb.max()), "n_distinct": int(len(set(nb.tolist())))},
       "n_interior": {"min": int(ni.min()), "max": int(ni.max()), "n_distinct": int(len(set(ni.tolist())))},
       "distinct_interior_node_counts_fraction": float(len(set(ni.tolist())) / len(ok)), "cases": rows}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "cases"}, indent=1)); print("T2D-GEOM-DONE")
