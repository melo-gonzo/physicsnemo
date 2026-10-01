"""P200K-FLEET HiLift group (`hl`): the complete wing ladder at the reporting protocol (200,000 cells, float32, 180
validation wings, two seeds), the 200k-against-10k comparison the fleet preregistration promised, and the wing half
of SCALE-LAW's case exponents for the three models.

Inputs: results/campaigns_2026-09-19/hl/metrics/<run>/<run>/metrics.jsonl for the lanes
p200k_hl_{isla,gt_unit,transolver_unit}_<rung>_seed{42,43} and p200k_hl_gt_unit_<rung>_lr3e3_seed{42,43}
(ISLA 192 wide, linear routing, lr 1e-3, the configuration of 2026-09-19; unit-direction GeoTransolver-256 at 1e-3 and
3e-3; Transolver-256 at 3e-3). One scoring block per file: the block whose config row carries the highest epoch.
REF-BOARD values of the reference configuration (384 wide, cosine routing, 5e-4) are read beside from
results/ref_surface_2026-09-23.json (35 and 210 wings) and results/campaigns_2026-09-22/ref/metrics (510 wings).
10,000-cell predecessors: results/p200k_fleet_reduction_2026-09-19.json (tenk_reference blocks, the 35- and 4-geometry
rungs), results/frame_reduction_2026-09-11.json (relative-frame ISLA at 10k on the 210-, 21-geometry and fixed-angle
rungs) and results/fp32_reeval/fp32_shift_2026-09-09.json (unit-direction GeoTransolver and Transolver at 10k on the
fixed-angle rung). The 510- and 1,260-wing rungs have no 10,000-cell twin of the same configuration.

Plateau exclusion: a lane whose train.log shows the mean-field signature (validation loss descending to about 0.01 and
then jumping to 0.12 to 0.16 and staying flat, training loss at the same value) is excluded from every mean, ratio and
fit and recorded under "excluded". The signature was read on the cluster train logs on 2026-09-30 (runs/<run>/train.log,
val_interval 25) and is confirmed here by the scored pressure error above 0.4 (the mean-field level is 0.6 to 0.7).

usage: python p200k_hl_ladder_2026-09-30.py   (writes p200k_hl_ladder_2026-09-30.json and .png beside itself)
"""
import json, math, os, statistics as st, sys

R = os.path.dirname(os.path.abspath(__file__))
M = f"{R}/campaigns_2026-09-19/hl/metrics"
FIELDS = ["pressure_l2", "velocity_l2", "tau_wall_l2", "temperature_l2", "density_l2"]
SEEDS = (42, 43)
PLATEAU_LEVEL = 0.4  # scored pressure relative L2 above this is the mean-field output, not a trained model

# rung key -> (training cases, label, validation cases)
RUNGS = {
    "super_scarce": (35, "35 wings (random split)", 180),
    "scarce": (210, "210 wings (random split)", 180),
    "medium": (510, "510 wings (random split)", 180),
    "full": (1260, "1,260 wings (random split)", 180),
    "geometry_super_scarce": (40, "4 geometries at all angles (held-out geometries)", 180),
    "geometry_scarce": (210, "21 geometries at all angles (held-out geometries)", 180),
    "single_aoa_12": (126, "126 geometries at 12 degrees (fixed angle)", 18),
    "super_scarce_area": (35, "35 wings, area-law training draw", 180),
}
CASE_LADDER = ["super_scarce", "scarce", "medium", "full"]
# model arm -> run pattern; GeoTransolver's rate per rung is read from the fleet manifest copy
ARMS = {
    "isla": "p200k_hl_isla_{rung}_seed{s}",
    "gt_unit": "p200k_hl_gt_unit_{rung}_seed{s}",
    "gt_unit_lr3e3": "p200k_hl_gt_unit_{rung}_lr3e3_seed{s}",
    "transolver_unit": "p200k_hl_transolver_unit_{rung}_seed{s}",
}
RATES = {k: v["lr"] for k, v in json.load(open(f"{R}/p200k_fleet_rates_2026-09-22.json"))["runs"].items()}
# validation-loss evidence from the cluster train logs (epoch: value), for the lanes the plateau rule excludes
PLATEAU_LOG = {
    "p200k_hl_isla_full_seed42": {"25": 0.0225, "50": 0.1432, "75": 0.1632, "100": 0.1512, "499": 0.1517, "train_499": 0.1542},
    "p200k_hl_isla_full_seed43": {"25": 0.0103, "50": 0.1549, "75": 0.1506, "100": 0.1545, "499": 0.1483, "train_499": 0.1510},
    "p200k_hl_isla_medium_seed42": {"25": 0.0161, "50": 0.0112, "75": 0.0245, "100": 0.1534, "499": 0.1212, "train_499": 0.1223},
}


def lane(path):
    """Mean per-case metrics of the highest-epoch scoring block in one metrics.jsonl; None if absent."""
    if not os.path.exists(path):
        return None
    blocks, cur = [], None
    for l in open(path):
        d = json.loads(l)
        if d.get("phase") == "config":
            cur = {"epoch": d.get("epoch"), "rows": []}
            blocks.append(cur)
        elif d.get("phase") == "infer_step" and cur is not None:
            cur["rows"].append(d)
    blocks = [b for b in blocks if b["rows"]]
    if not blocks:
        return None
    top = max(b["epoch"] or 0 for b in blocks)
    # a scoring job that was interrupted and restarted leaves two config rows at the same epoch (the 35-wing files:
    # 96 cases, then a restart that re-scores the list); one row per validation case, the last written wins
    rows = {}
    for b in blocks:
        if (b["epoch"] or 0) == top:
            for r in b["rows"]:
                rows[r["sample_id"]] = r["metrics"]
    rows = list(rows.values())
    keys = [k for k in FIELDS if k in rows[0]]
    return {k: st.mean(r[k] for r in rows) for k in keys} | {"n_cases": len(rows), "epoch": top}


out = {"protocol": __doc__.strip(), "rungs": {}, "excluded": {}, "ladder": {}, "tenk": {}, "refboard": {}}

# ---- per rung and model --------------------------------------------------------------------------------------------
for rung, (n, label, n_val) in RUNGS.items():
    row = {"n_train": n, "label": label, "n_val": n_val, "arms": {}}
    for arm, pat in ARMS.items():
        runs, excl = {}, {}
        for s in SEEDS:
            run = pat.format(rung=rung, s=s)
            v = lane(f"{M}/{run}/{run}/metrics.jsonl")
            if v is None:
                continue
            if v["pressure_l2"] > PLATEAU_LEVEL:
                excl[run] = v | {"val_loss_log": PLATEAU_LOG.get(run), "reason": "mean-field plateau (validation loss flat at 0.12 to 0.16 from before epoch 100; scored pressure error at the field-mean level)"}
                continue
            runs[run] = v
        if excl:
            out["excluded"].update(excl)
        if not runs:
            continue
        keys = [k for k in FIELDS if k in next(iter(runs.values()))]
        a = {"runs": runs, "n_seeds": len(runs), "lr": RATES.get(next(iter(runs))),
             "seed_means": {k: st.mean(r[k] for r in runs.values()) for k in keys},
             "pressure_l2_seeds": [runs[r]["pressure_l2"] for r in sorted(runs)],
             "excluded_runs": sorted(excl)}
        row["arms"][arm] = a
    A = row["arms"]
    # GeoTransolver's better rate on this rung (by seed-mean pressure)
    gts = [k for k in ("gt_unit", "gt_unit_lr3e3") if k in A]
    if gts:
        best = min(gts, key=lambda k: A[k]["seed_means"]["pressure_l2"])
        row["gt_better"] = {"arm": best, "lr": A[best]["lr"],
                            "other": {k: A[k]["seed_means"]["pressure_l2"] for k in gts if k != best}}
    if "isla" in A:
        P = lambda k: A[k]["seed_means"]["pressure_l2"]
        row["ratios_over_isla"] = {}
        if gts:
            row["ratios_over_isla"]["gt_better_rate"] = {f: A[best]["seed_means"][f] / A["isla"]["seed_means"][f] for f in A["isla"]["seed_means"] if f in A[best]["seed_means"]}
            for k in gts:
                row["ratios_over_isla"][k] = {"pressure_l2": P(k) / P("isla")}
        if "transolver_unit" in A:
            row["ratios_over_isla"]["transolver_unit"] = {f: A["transolver_unit"]["seed_means"][f] / A["isla"]["seed_means"][f] for f in A["isla"]["seed_means"] if f in A["transolver_unit"]["seed_means"]}
    out["rungs"][rung] = row

# ---- 10,000-cell predecessors ----------------------------------------------------------------------------------------
fleet19 = json.load(open(f"{R}/p200k_fleet_reduction_2026-09-19.json"))["datasets"]
frame11 = json.load(open(f"{R}/frame_reduction_2026-09-11.json"))["hilift_val"]
fp32 = json.load(open(f"{R}/fp32_reeval/fp32_shift_2026-09-09.json"))["runs"]
tenk = {}
t = fleet19["highlift_surface_super_scarce"]["tenk_reference"]
tenk["super_scarce"] = {"isla": t["isla"], "gt_unit_lr3e3": t["gt_lr3e3"], "transolver_unit": t["transolver_lr3e3"], "source": t["source"]}
t = fleet19["highlift_surface_geometry_super_scarce"]["tenk_reference"]
tenk["geometry_super_scarce"] = {"isla": t["isla"], "gt_unit": t["gt_unit"], "transolver_unit": t["transolver_lr3e3"], "source": t["source"]}
tenk["single_aoa_12"] = {
    "isla": [frame11[f"rf_hl_mt2_meas_single_aoa_12_seed{s}"]["pressure_l2"] for s in (42, 43, 44)],
    "gt_unit": [fp32[f"udrv_hl_gt_single_aoa_12_seed{s}"]["fields"]["pressure_l2"]["fp32_mean"] for s in SEEDS],
    "transolver_unit": [fp32[f"udrv_hl_transolver_single_aoa_12_lr3e3_seed{s}"]["fields"]["pressure_l2"]["fp32_mean"] for s in SEEDS],
    "source": "results/frame_reduction_2026-09-11.json (rf_hl_mt2_meas_single_aoa_12, three seeds); results/fp32_reeval/fp32_shift_2026-09-09.json (udrv_hl_* at 1e-3 / 3e-3)"}
tenk["scarce"] = {"isla": [frame11[f"rf_hl_mt2_meas_scarce_seed{s}"]["pressure_l2"] for s in SEEDS],
                  "source": "results/frame_reduction_2026-09-11.json (rf_hl_mt2_meas_scarce); no unit-direction baseline twin at 10k on this rung"}
tenk["geometry_scarce"] = {"isla": [frame11[f"rf_hl_mt2_meas_geometry_scarce_seed{s}"]["pressure_l2"] for s in SEEDS],
                           "source": "results/frame_reduction_2026-09-11.json (rf_hl_mt2_meas_geometry_scarce); no unit-direction baseline twin at 10k on this rung"}
for rung, tk in tenk.items():
    A = out["rungs"][rung]["arms"]
    comp = {}
    gt10 = "gt_unit_lr3e3" if "gt_unit_lr3e3" in tk else ("gt_unit" if "gt_unit" in tk else None)
    gt200 = out["rungs"][rung].get("gt_better", {}).get("arm")
    # 10k GeoTransolver at the rate it ran there (3e-3 on the 35-wing rung, 1e-3 elsewhere) against the 200k better rate
    pairs = {"isla": ("isla", "isla"), "gt": (gt10, gt200), "transolver": ("transolver_unit", "transolver_unit")}
    for name, (k10, k200) in pairs.items():
        if k10 and k10 in tk and k200 in A:
            m10 = st.mean(tk[k10]); m200 = A[k200]["seed_means"]["pressure_l2"]
            comp[name] = {"tenk_mean": m10, "tenk_seeds": tk[k10], "tenk_arm": k10, "p200k_arm": k200, "p200k_lr": A[k200]["lr"],
                          "p200k_mean": m200, "p200k_over_tenk": m200 / m10}
    if gt10 and "isla" in tk:
        comp["gt_over_isla_tenk"] = st.mean(tk[gt10]) / st.mean(tk["isla"])
    if "transolver_unit" in tk and "isla" in tk:
        comp["transolver_over_isla_tenk"] = st.mean(tk["transolver_unit"]) / st.mean(tk["isla"])
    comp["source"] = tk["source"]
    out["tenk"][rung] = comp

# ---- REF-BOARD (reference configuration) beside --------------------------------------------------------------------
ref = json.load(open(f"{R}/ref_surface_2026-09-23.json"))["summary"]
rb = {"super_scarce": ref["isla_r1|hl35"]["fields"]["pressure_l2"]["seeds"], "scarce": ref["isla_r1|hl210"]["fields"]["pressure_l2"]["seeds"]}
r510 = [lane(f"{R}/campaigns_2026-09-22/ref/metrics/ref_isla_r1_hl510_seed{s}/ref_isla_r1_hl510_seed{s}/metrics.jsonl") for s in SEEDS]
rb["medium"] = [v["pressure_l2"] for v in r510 if v]
gt512 = {"super_scarce": ref["gt_w512|hl35"]["fields"]["pressure_l2"]["seeds"]}
g210 = [lane(f"{R}/campaigns_2026-09-22/ref/metrics/ref_gt_w512_hl210_seed{s}/ref_gt_w512_hl210_seed{s}/metrics.jsonl") for s in SEEDS]
if all(g210):
    gt512["scarce"] = [v["pressure_l2"] for v in g210]
for rung, seeds in rb.items():
    out["refboard"][rung] = {"isla_reference_pressure_l2_seeds": seeds, "isla_reference_pressure_l2_mean": st.mean(seeds),
                             "earlier_protocol_isla_mean": out["rungs"][rung]["arms"]["isla"]["seed_means"]["pressure_l2"],
                             "reference_over_earlier": st.mean(seeds) / out["rungs"][rung]["arms"]["isla"]["seed_means"]["pressure_l2"]}
    g = out["rungs"][rung].get("gt_better")
    if g:
        out["refboard"][rung]["gt256_better_rate_over_isla_reference"] = out["rungs"][rung]["arms"][g["arm"]]["seed_means"]["pressure_l2"] / st.mean(seeds)
    if rung in gt512:
        out["refboard"][rung]["gt512_pressure_l2_seeds"] = gt512[rung]
        out["refboard"][rung]["gt512_over_isla_reference"] = st.mean(gt512[rung]) / st.mean(seeds)
out["refboard"]["source"] = "results/ref_surface_2026-09-23.json (hl35, hl210; GeoTransolver-512 at 3e-3 on the same rungs); results/campaigns_2026-09-22/ref/metrics/ref_isla_r1_hl510_seed*/ (hl510); @sec-nb-ref-hl510-verdict quotes GeoTransolver-512 ÷ reference 0.95 at 510 wings from the 100,000-cell SCALE-LAW pair"


# ---- learning-curve exponents over the case ladder -----------------------------------------------------------------
def slope(pts):
    """Least-squares slope of log(error) on log(cases); returns -slope (the exponent alpha in error ~ n^-alpha)."""
    if len(pts) < 2:
        return None
    xs = [math.log(n) for n, _ in pts]; ys = [math.log(e) for _, e in pts]
    mx, my = st.mean(xs), st.mean(ys)
    return -sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def fit(pts):
    """Power law error = a * n^-alpha through the seed means (log-log least squares)."""
    al = slope(pts)
    xs = [math.log(n) for n, _ in pts]; ys = [math.log(e) for _, e in pts]
    lna = st.mean(ys) + al * st.mean(xs)
    return {"alpha": al, "a": math.exp(lna), "rungs": [n for n, _ in pts]}


ladder = {}
for model in ("isla", "gt_better", "transolver_unit"):
    pts, per_seed, local = [], {s: [] for s in SEEDS}, {}
    for rung in CASE_LADDER:
        row = out["rungs"][rung]
        arm = row["gt_better"]["arm"] if model == "gt_better" else model
        if arm not in row["arms"]:
            continue
        a = row["arms"][arm]; n = row["n_train"]
        pts.append((n, a["seed_means"]["pressure_l2"]))
        for run, v in a["runs"].items():
            per_seed[int(run[-2:])].append((n, v["pressure_l2"]))
    for (n0, e0), (n1, e1) in zip(pts, pts[1:]):
        local[f"{n0}-{n1}"] = -math.log(e1 / e0) / math.log(n1 / n0)
    f = fit(pts)
    f["per_seed"] = {s: {"alpha": slope(p), "rungs": [n for n, _ in p]} for s, p in per_seed.items()}
    f["alpha_range_over_seeds"] = sorted(v["alpha"] for v in f["per_seed"].values() if v["alpha"] is not None)
    f["local_exponents"] = local
    f["points"] = pts
    ladder[model] = f
# the reference configuration's three-rung curve beside
rpts = [(RUNGS[r][0], st.mean(rb[r])) for r in ("super_scarce", "scarce", "medium") if rb[r]]
fr = fit(rpts); fr["local_exponents"] = {f"{a[0]}-{b[0]}": -math.log(b[1] / a[1]) / math.log(b[0] / a[0]) for a, b in zip(rpts, rpts[1:])}; fr["points"] = rpts
fr["per_seed"] = {s: {"alpha": slope([(RUNGS[r][0], rb[r][i]) for r in ("super_scarce", "scarce", "medium") if len(rb[r]) > i]), "rungs": [35, 210, 510]} for i, s in enumerate(SEEDS)}
fr["alpha_range_over_seeds"] = sorted(v["alpha"] for v in fr["per_seed"].values())
ladder["isla_reference_configuration"] = fr
# crossover of the fitted laws (where GeoTransolver's and ISLA's fitted curves meet), if the exponents differ the right way
for k in ("gt_better", "transolver_unit"):
    g, i = ladder[k], ladder["isla"]
    d = i["alpha"] - g["alpha"]
    # a n^-alpha_i = b n^-alpha_g  =>  n = (a / b)^(1 / (alpha_i - alpha_g)); only meaningful if ISLA's curve is the steeper
    ladder[k]["crossover_cases_vs_isla"] = math.exp(math.log(i["a"] / g["a"]) / d) if d > 0 else None
    gi = ladder["isla_reference_configuration"]; d2 = gi["alpha"] - g["alpha"]
    ladder[k]["crossover_cases_vs_isla_reference"] = math.exp(math.log(gi["a"] / g["a"]) / d2) if d2 > 0 else None
    # the same three rungs as the ISLA curves (35 / 210 / 510), so the exponents are compared on a matched ladder
    ladder[k]["alpha_35_to_510"] = slope([p for p in g["points"] if p[0] <= 510])
out["ladder"] = ladder
# a convenience table: per rung, the three standings and the ratios
out["ladder_table"] = {rung: {"n_train": out["rungs"][rung]["n_train"],
                              "isla": out["rungs"][rung]["arms"].get("isla", {}).get("seed_means", {}).get("pressure_l2"),
                              "isla_seeds": out["rungs"][rung]["arms"].get("isla", {}).get("pressure_l2_seeds"),
                              "gt_better_arm": out["rungs"][rung].get("gt_better", {}).get("lr"),
                              "gt_better": out["rungs"][rung]["arms"][out["rungs"][rung]["gt_better"]["arm"]]["seed_means"]["pressure_l2"],
                              "gt_better_seeds": out["rungs"][rung]["arms"][out["rungs"][rung]["gt_better"]["arm"]]["pressure_l2_seeds"],
                              "transolver": out["rungs"][rung]["arms"].get("transolver_unit", {}).get("seed_means", {}).get("pressure_l2"),
                              "transolver_seeds": out["rungs"][rung]["arms"].get("transolver_unit", {}).get("pressure_l2_seeds"),
                              "gt_over_isla": out["rungs"][rung].get("ratios_over_isla", {}).get("gt_better_rate", {}).get("pressure_l2"),
                              "transolver_over_isla": out["rungs"][rung].get("ratios_over_isla", {}).get("transolver_unit", {}).get("pressure_l2")}
                       for rung in RUNGS}

json.dump(out, open(f"{R}/p200k_hl_ladder_2026-09-30.json", "w"), indent=1)

# ---- figure: pressure error against training cases, the fitted exponents -------------------------------------------
sys.path.insert(0, f"{R}/../book")
import palette as P  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(7.2, 4.6))
ns = [35, 50, 100, 200, 500, 1000, 1260]
xx = [35 * (1260 / 35) ** (k / 200) for k in range(201)]
series = {"isla": ("isla_ref", "isla"), "gt_better": ("gt", "gt_unit"), "transolver_unit": ("t", "transolver_unit")}
for model, (key, _) in series.items():
    f = ladder[model]
    kw = P.series_kwargs(key)
    ax.plot(xx, [f["a"] * x ** -f["alpha"] for x in xx], color=kw["color"], ls=kw["ls"], lw=1.6, alpha=0.9, marker=None)
    ax.plot([n for n, _ in f["points"]], [e for _, e in f["points"]], **P.series_kwargs(key, ls="none", ms=8))
    for rung in CASE_LADDER:
        row = out["rungs"][rung]; arm = row["gt_better"]["arm"] if model == "gt_better" else model
        if arm in row["arms"]:
            ax.plot([row["n_train"]] * len(row["arms"][arm]["pressure_l2_seeds"]), row["arms"][arm]["pressure_l2_seeds"],
                    ls="none", marker=".", color=kw["color"], ms=5, alpha=0.9)
fr = ladder["isla_reference_configuration"]
ax.plot([n for n, _ in fr["points"]], [e for _, e in fr["points"]], **P.series_kwargs("isla_ref", ls="--", lw=1.4, mfc="white", ms=8))
for rung in ("super_scarce", "scarce", "medium"):
    ax.plot([RUNGS[rung][0]] * len(rb[rung]), rb[rung], ls="none", marker=".", color=P.BLUE, ms=5, alpha=0.9)
lo = {"isla": ladder["isla"]["alpha_range_over_seeds"], "gt_better": ladder["gt_better"]["alpha_range_over_seeds"], "transolver_unit": ladder["transolver_unit"]["alpha_range_over_seeds"]}
P.label_line(ax, 1260, 0.0205, f"GeoTransolver-256 at its better rate, 35 to 1,260 wings\nexponent {ladder['gt_better']['alpha']:.2f} (seeds {lo['gt_better'][0]:.2f} to {lo['gt_better'][-1]:.2f})", P.RED, ha="right", va="top")
P.label_line(ax, 1260, 0.074, f"Transolver-256 (triangles, dashed), 35 to 1,260 wings\nexponent {ladder['transolver_unit']['alpha']:.2f} (seeds {lo['transolver_unit'][0]:.2f} to {lo['transolver_unit'][-1]:.2f})", P.AMBER, ha="right", va="bottom")
P.label_line(ax, 35, 0.158, f"ISLA, earlier protocol (192 wide, linear routing, 1e-3): filled squares, solid\nexponent {ladder['isla']['alpha']:.2f} over 35 to 510 wings (seeds {lo['isla'][0]:.2f} to {lo['isla'][-1]:.2f}); its 1,260-wing lanes plateaued", P.BLUE, ha="left", va="bottom")
P.label_line(ax, 36, 0.034, f"ISLA, reference configuration (384 wide, cosine routing, 5e-4): open squares, dashed\nexponent {fr['alpha']:.2f} over 35 to 510 wings (seeds {fr['alpha_range_over_seeds'][0]:.2f} to {fr['alpha_range_over_seeds'][-1]:.2f}); no 1,260-wing lane yet", P.BLUE, ha="left", va="top")
ns = [35, 50, 100, 200, 500, 1260]
ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([f"{n:,}" for n in ns]); ax.minorticks_off()
ax.set_xlim(30, 1500); ax.set_ylim(0, 0.18)
ax.set_xlabel("training wings [–]"); ax.set_ylabel("surface-pressure error, area-weighted relative L2 [–]")
ax.set_title("HiLiftAeroML case ladder at 200,000 cells per case, float32, 180 validation wings\nseed means with seeds as dots; lines are log-log power-law fits through the seed means")
fig.tight_layout(); fig.savefig(f"{R}/p200k_hl_ladder_2026-09-30.png", dpi=200); plt.close(fig)

# ---- console summary ------------------------------------------------------------------------------------------------
r4 = lambda v: None if v is None else (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v])
for rung, t in out["ladder_table"].items():
    print(f"{rung:24s} n={t['n_train']:5d} isla={r4(t['isla'])} {r4(t['isla_seeds'])} gt({t['gt_better_arm']})={r4(t['gt_better'])} {r4(t['gt_better_seeds'])} "
          f"t={r4(t['transolver'])} {r4(t['transolver_seeds'])} gt/isla={r4(t['gt_over_isla'])} t/isla={r4(t['transolver_over_isla'])}")
for m, f in ladder.items():
    print(m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items() if k in ("alpha", "alpha_range_over_seeds", "local_exponents", "crossover_cases_vs_isla", "crossover_cases_vs_isla_reference", "rungs")})
print("excluded:", {k: round(v["pressure_l2"], 3) for k, v in out["excluded"].items()})
for rung, c in out["tenk"].items():
    print("tenk", rung, {k: round(v["p200k_over_tenk"], 3) for k, v in c.items() if isinstance(v, dict)}, {k: round(v, 3) for k, v in c.items() if isinstance(v, float)})
print("refboard", {k: {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, float)} for k, v in out["refboard"].items() if isinstance(v, dict)})
