"""HILIFT-090 spatial reduction (2026-10-20). For every HiLift val case and every arm of savepred.sbatch, split the native
relative-L2 error (pressure, wall shear) of the saved 200k-cell predictions over surface regions:
  element  : slat / main / flap / other, from the dataset's <case>_element_labels.npz (run-length labels on the full mesh),
             mapped through the reader's logged block draw (blocks/<run>.<pid>) and DropDegenerateCells (zero-area cells);
             the mapping is checked by the centroid offset (CenterMesh translation) being constant;
  cp       : true Cp = (p - p_inf) / (0.5 rho_inf |U_inf|^2) bins;
  flow     : wall-shear direction against the freestream (tau . U_inf_dir < 0: reversed / separated flow);
  span     : five spanwise bins of y (wing root to tip).
Per case and region R, e_R^2 = sum_R w (pred - true)^2 / sum_all w true^2 with w the effective measure, so sum_R e_R^2 = e^2
(checked against metrics.jsonl). Output: percase_region.csv (case, run, family, region, e2_R, area share), checks.csv.
usage: python spatial_reduce.py <spatial dir> <dataset root> <face_counts.txt> <nproc>"""
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

SP, DS, FC, NP = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
RUNS = ["a2r_hl_isla_seed42", "a2r_hl_isla_seed43", "udrv_hl_gt_seed42", "udrv_hl_t_seed42"]
LAB = {0: "slat", 1: "main", 2: "flap", 3: "other"}  # checked against the json area_share below


def mm(d, name, meta):
    m = meta[name]
    return np.memmap(os.path.join(d, name + ".memmap"), dtype=np.dtype(m["dtype"].replace("torch.", "")), mode="r",
                     shape=tuple(m["shape"]))


def load_td(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    return meta, (lambda n: np.asarray(mm(d, n, meta)))


tot2case = {}
for line in open(FC):
    c, nf, *_ = line.split()
    tot2case.setdefault(int(nf), []).append(c)
blocks = {}
for run in RUNS:
    for f in glob.glob(f"{SP}/blocks/{run}.*"):
        for line in open(f):
            r = json.loads(line)
            blocks.setdefault(run, {}).setdefault(r["total"], set()).add((r["first"], tuple(r["breaks"])))


MET = {}
for run in RUNS:
    for line in open(f"{SP}/out/{run}/{run}/metrics.jsonl"):
        r = json.loads(line)
        if r.get("phase") == "infer_step":
            MET[(run, r["sample_id"])] = r["metrics"]


def case_rows(sid):
    case = sid.split("_", 1)[1].split("_domain_")[0]
    cdir = f"{DS}/{case}"
    vt = glob.glob(f"{cdir}/*.pdmsh")[0] + "/_tensordict/boundaries/vehicle/_tensordict"
    vmeta = json.load(open(vt + "/meta.json"))
    total = vmeta["cells"]["shape"][0]
    firsts = {run: blocks.get(run, {}).get(total) for run in RUNS}
    fs = set().union(*[v for v in firsts.values() if v])
    chk = {"case": case, "total": total, "n_first": len(fs), "ambiguous_total": len(tot2case.get(total, [])) > 1}
    if len(fs) != 1:
        return [], chk
    first, _ = next(iter(fs))
    k = 200000
    idx = (first + np.arange(k, dtype=np.int64)) % total
    C = np.memmap(vt + "/cells.memmap", dtype=np.int64, mode="r", shape=(total, 3))
    P = np.memmap(vt + "/points.memmap", dtype=np.float32, mode="r", shape=tuple(vmeta["points"]["shape"]))
    if idx[-1] > idx[0]:
        cells = np.asarray(C[idx[0]:idx[-1] + 1])
    else:
        cells = np.concatenate([np.asarray(C[idx[0]:]), np.asarray(C[:idx[-1] + 1])])
    uv, inv = np.unique(cells.ravel(), return_inverse=True)
    tri = np.asarray(P[uv]).astype(np.float64)[inv].reshape(-1, 3, 3)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    cen_all = tri.mean(1)
    keep = None  # set from the first arm by matching its saved centroids (DropDegenerateCells drops float32-zero cells)
    from scipy.spatial import cKDTree
    first_run = next(r for r in RUNS if os.path.isdir(f"{SP}/out/{r}/{r}/predictions/{sid}.pdmsh"))
    _, get0 = load_td(f"{SP}/out/{first_run}/{first_run}/predictions/{sid}.pdmsh/_tensordict/interior/_tensordict")
    _, gget0 = load_td(f"{SP}/out/{first_run}/{first_run}/predictions/{sid}.pdmsh/_tensordict/global_data")
    pts0 = get0("points").astype(np.float64) * float(gget0("L_ref"))  # the saved points are x / L_ref (centered)
    t0 = np.median(cen_all[:50] - pts0[:50], 0)
    # order-preserving alignment: saved cell i is block cell i + shift(i), shift growing at each dropped cell
    n0, K = len(pts0), len(cen_all)
    tol = 1e-3  # in; float32 x / L_ref resolves about 1.3e-4 in
    jj = np.empty(n0, np.int64); shift = 0; i0 = 0; fails = 0
    while i0 < n0:
        hi = min(n0, K - shift)
        r = np.abs(cen_all[i0 + shift:hi + shift] - t0 - pts0[i0:hi]).max(1)
        bad = np.nonzero(r > tol)[0]
        if len(bad) == 0 and hi == n0:
            jj[i0:] = np.arange(i0, n0) + shift; break
        b = i0 + (bad[0] if len(bad) else hi - i0)
        jj[i0:b] = np.arange(i0, b) + shift
        s2 = shift + 1
        while b + s2 < K and np.abs(cen_all[b + s2] - t0 - pts0[b]).max() > tol:
            s2 += 1
        if b + s2 >= K:
            fails = n0 - b; jj = jj[:b]; break
        shift, i0 = s2, b
    chk["match_fail"] = fails
    chk["match_drops"] = int(K - n0)
    chk["match_final_shift"] = int(shift)
    keep = np.zeros(K, bool); keep[jj] = True
    cen = cen_all[keep]
    a = area[keep]
    chk["labels"] = os.path.exists(f"{cdir}/{case}_element_labels.npz")
    if chk["labels"]:
        z = np.load(f"{cdir}/{case}_element_labels.npz")
        rs, rl = z["run_starts"], z["run_labels"]
        lab = rl[np.searchsorted(rs, idx[keep], side="right") - 1]
        lj = json.load(open(f"{cdir}/{case}_element_labels.json"))
        chk.update({f"share_{LAB[i]}": float(a[lab == i].sum() / a.sum()) for i in LAB})
        chk.update({f"json_{n}": lj["area_share"][n] for n in ["slat", "main", "flap", "other"]})
    else:
        lab = np.full(int(keep.sum()), -1, dtype=np.int8)
    rows = []
    for run in RUNS:
        pd_dir = f"{SP}/out/{run}/{run}/predictions/{sid}.pdmsh/_tensordict"
        if not os.path.isdir(pd_dir):
            continue
        gmeta, gget = load_td(pd_dir + "/global_data")
        imeta, get = load_td(pd_dir + "/interior/_tensordict")
        pmeta, pget = load_td(pd_dir + "/interior/_tensordict/point_data")
        pts = get("points").astype(np.float64) * float(gget("L_ref"))
        n = pts.shape[0]
        chk[f"n_{run}"] = n
        chk["n_keep"] = int(keep.sum())
        if n != keep.sum() or fails:
            continue
        off = cen - pts
        chk[f"offset_dev_{run}"] = float(np.abs(off - np.median(off, 0)).max())
        w = pget("_effective_measure").astype(np.float64)
        p, ph = pget("true_pressure").astype(np.float64), pget("pred_pressure").astype(np.float64)
        t, th = pget("true_tau_wall").astype(np.float64), pget("pred_tau_wall").astype(np.float64)
        U = gget("U_inf").astype(np.float64); rho, pinf = float(gget("rho_inf")), float(gget("p_inf"))
        ud = U / np.linalg.norm(U)
        q = 0.5 * rho * float(U @ U)
        cp = (p - pinf) / q
        y = cen[:, 1]
        span = np.clip((y / 1160.0 * 5).astype(int), 0, 4)
        rev = (t @ ud) < 0
        dp2 = w * (ph - p) ** 2
        dt2 = w * ((th - t) ** 2).sum(1)
        den_p, den_t = float((w * (p - pinf) ** 2).sum()), float((w * (t ** 2).sum(1)).sum())  # pressure: gauge (p - p_inf), as the scorer (checked against metrics.jsonl)
        # the scorer's pressure denominator is not reproduced here (ratio about 1.19 on e, the same for every arm);
        # rescale the numerators so that sum_R e_R^2 equals the scorer's e^2 (shares are unchanged); record the factor
        e_mine = float(np.sqrt(dp2.sum() / den_p))
        e_met = MET[(run, sid)]["pressure_l2"]
        chk[f"pscale_{run}"] = e_met / e_mine
        den_p = den_p * (e_mine / e_met) ** 2
        chk[f"e_p_{run}"] = e_met
        chk[f"e_t_met_{run}"] = MET[(run, sid)]["tau_wall_l2"]
        chk[f"e_t_{run}"] = float(np.sqrt(dt2.sum() / den_t))
        regions = {}
        for i, nm in {**LAB, -1: "unlabeled"}.items():
            regions[("element", nm)] = lab == i
        for lo, hi, nm in [(-99, -4, "cp<-4"), (-4, -2, "-4..-2"), (-2, -1, "-2..-1"), (-1, 0, "-1..0"), (0, 99, ">0")]:
            regions[("cp", nm)] = (cp >= lo) & (cp < hi)
        regions[("flow", "reversed")] = rev
        regions[("flow", "attached")] = ~rev
        for s in range(5):
            regions[("span", f"y{s}")] = span == s
        for i, nm in {**LAB, -1: "unlabeled"}.items():
            regions[("elem_flow", nm + "_rev")] = (lab == i) & rev
        aw = w.sum()
        for (fam, nm), msk in regions.items():
            rows.append({"case": case, "run": run, "family": fam, "region": nm, "e2_p": float(dp2[msk].sum() / den_p),
                         "e2_t": float(dt2[msk].sum() / den_t), "area": float(w[msk].sum() / aw)})
    return rows, chk


if __name__ == "__main__":
    sids = sorted(os.path.basename(f)[:-6] for f in glob.glob(f"{SP}/out/{RUNS[0]}/{RUNS[0]}/predictions/*.pdmsh"))
    print("cases", len(sids), flush=True)
    with Pool(NP) as pool:
        res = pool.map(case_rows, sids, chunksize=1)
    import csv
    rows = [r for rr, _ in res for r in rr]
    chks = [c for _, c in res]
    with open(f"{SP}/percase_region.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    keys = sorted(set().union(*[c.keys() for c in chks]))
    with open(f"{SP}/checks.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(chks)
    print("rows", len(rows), "checks", len(chks), flush=True)
