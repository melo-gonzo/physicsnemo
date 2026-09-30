"""FORCEREF reducer (2026-09-26): per-car lift and drag on the 200,000-cell DrivAerML Surface checkpoints.

Input: results/campaigns_2026-09-26/forceref/forceref_cases.json (written on the cluster by
kernel/forceref/forceref_integrate.py; per run and per validation car the predicted and the sampled-cell truth
coefficients on the 200,000 evaluated cells with Horvitz-Thompson weights, plus the full-surface truth per car).
Output: results/forceref_2026-09-26.json.

Coefficients use A_ref = 1 and q_inf = 0.5 rho |U_inf|^2 (the recipe's infer.yaml convention); a common positive
factor cancels in every relative error and correlation below. Errors are relative to the full-surface truth unless
the key says `_vs_sampled`.
"""

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "campaigns_2026-09-26" / "forceref" / "forceref_cases.json"
OUT = HERE / "forceref_2026-09-26.json"

MODELS = {
    "ISLA": ["ref_isla_r1_dr435_seed42", "ref_isla_r1_dr435_seed43"],
    "GeoTransolver-512": ["ref_gt_w512_dr435_seed42", "ref_gt_w512_dr435_seed43"],
}
COEFFS = ["CD", "CL", "CS"]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def rel_l1(pred, true):
    return mean(abs(p - t) for p, t in zip(pred, true)) / mean(abs(t) for t in true)


def rel_l2(pred, true):
    return math.sqrt(sum((p - t) ** 2 for p, t in zip(pred, true))) / math.sqrt(sum(t * t for t in true))


def pearson(x, y):
    mx, my = mean(x), mean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy)


def ranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    for rank, i in enumerate(order):
        r[i] = float(rank)
    return r


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def main():
    data = json.loads(SRC.read_text())
    runs = data["runs"]
    full = data["truth_full"]
    cases = sorted(full, key=lambda s: int(s.split("_")[1]))
    all_runs = [r for rs in MODELS.values() for r in rs]
    for r in all_runs:
        assert set(runs[r]) == set(cases), (r, set(cases) ^ set(runs[r]))

    ### The sampled cells are drawn by the reader with a per-index generator, so the sampled truth should agree
    ### across runs to float roundoff; record the spread so the artifact says whether it does.
    sampled_spread = {
        c: max(abs(runs[r][case]["truth_sampled"][c] - runs[all_runs[0]][case]["truth_sampled"][c]) for r in all_runs for case in cases)
        for c in COEFFS
    }
    ### Cross-check of the integrator against the recipe's own ForceContext numbers (metrics.jsonl `forces`).
    recipe_dev = {
        c: max(abs(runs[r][case]["recipe_forces"]["true"][c] - runs[r][case]["truth_sampled"][c]) for r in all_runs for case in cases)
        for c in COEFFS
    } | {
        f"{c}_pred": max(abs(runs[r][case]["recipe_forces"]["pred"][c] - runs[r][case]["pred"][c]) for r in all_runs for case in cases)
        for c in COEFFS
    }

    ref_run = all_runs[0]
    per_car = []
    for case in cases:
        row = {"case": case, "n_cells_full": full[case]["n_cells"], "U_inf": full[case]["U_inf"]}
        for c in COEFFS:
            row[f"{c}_truth_full"] = full[case][c]
            row[f"{c}_truth_sampled"] = runs[ref_run][case]["truth_sampled"][c]
            row[f"{c}_sampling_rel_err"] = abs(row[f"{c}_truth_sampled"] - row[f"{c}_truth_full"]) / abs(row[f"{c}_truth_full"])
            for model, rs in MODELS.items():
                for r in rs:
                    row[f"{c}_pred_{model}_{r[-6:]}"] = runs[r][case]["pred"][c]
        row["area_full"] = full[case]["area_geometric"]
        row["area_sampled_represented"] = runs[ref_run][case]["truth_sampled"]["area_represented"]
        row["area_sampled_geometric"] = runs[ref_run][case]["truth_sampled"]["area_geometric"]
        per_car.append(row)

    summary = {"n_cases": len(cases), "sampling": {}, "models": {}}
    for c in COEFFS:
        tf = [full[case][c] for case in cases]
        ts = [runs[ref_run][case]["truth_sampled"][c] for case in cases]
        errs = [abs(s - f) / abs(f) for s, f in zip(ts, tf)]
        summary["sampling"][c] = {
            "truth_full_mean": mean(tf),
            "truth_sampled_mean": mean(ts),
            "rel_l1_sampled_vs_full": rel_l1(ts, tf),
            "rel_l2_sampled_vs_full": rel_l2(ts, tf),
            "per_car_rel_err_mean": mean(errs),
            "per_car_rel_err_median": sorted(errs)[len(errs) // 2],
            "per_car_rel_err_max": max(errs),
            "pearson_sampled_vs_full": pearson(ts, tf),
            "spearman_sampled_vs_full": spearman(ts, tf),
        }
    summary["sampling"]["area_ratio_represented_over_full_mean"] = mean(
        r["area_sampled_represented"] / r["area_full"] for r in per_car
    )
    summary["sampling"]["area_fraction_geometric_mean"] = mean(r["area_sampled_geometric"] / r["area_full"] for r in per_car)
    summary["sampling"]["truth_sampled_spread_across_runs_max"] = sampled_spread
    summary["integrator_vs_recipe_forcecontext_max_abs_dev"] = recipe_dev

    for model, rs in MODELS.items():
        m = {}
        for c in COEFFS:
            tf = [full[case][c] for case in cases]
            ts = [runs[ref_run][case]["truth_sampled"][c] for case in cases]
            per_seed = []
            for r in rs:
                pr = [runs[r][case]["pred"][c] for case in cases]
                tsr = [runs[r][case]["truth_sampled"][c] for case in cases]
                per_seed.append({
                    "run": r,
                    "rel_l1_vs_full": rel_l1(pr, tf), "rel_l2_vs_full": rel_l2(pr, tf),
                    "rel_l1_vs_sampled": rel_l1(pr, tsr), "rel_l2_vs_sampled": rel_l2(pr, tsr),
                    "per_car_rel_err_vs_full_mean": mean(abs(p - t) / abs(t) for p, t in zip(pr, tf)),
                    "per_car_rel_err_vs_sampled_mean": mean(abs(p - t) / abs(t) for p, t in zip(pr, tsr)),
                    "per_car_rel_err_vs_full_max": max(abs(p - t) / abs(t) for p, t in zip(pr, tf)),
                    "pearson_vs_full": pearson(pr, tf), "spearman_vs_full": spearman(pr, tf),
                    "pearson_vs_sampled": pearson(pr, tsr), "spearman_vs_sampled": spearman(pr, tsr),
                    "pred_mean": mean(pr),
                })
            keys = [k for k in per_seed[0] if k != "run"]
            m[c] = {"seed_mean": {k: mean(s[k] for s in per_seed) for k in keys}, "per_seed": per_seed}
        summary["models"][model] = m

    ### Ratios GeoTransolver-512 / ISLA of the seed-mean errors, the book's convention.
    summary["ratios_GeoTransolver-512_over_ISLA"] = {
        c: {k: summary["models"]["GeoTransolver-512"][c]["seed_mean"][k] / summary["models"]["ISLA"][c]["seed_mean"][k]
            for k in ("rel_l1_vs_full", "rel_l2_vs_full", "rel_l1_vs_sampled", "rel_l2_vs_sampled")}
        for c in COEFFS
    }

    out = {
        "node": "FORCEREF",
        "date": "2026-09-26",
        "protocol": (
            "DrivAerML Surface, 435 training cars, 48 validation cars; four checkpoints (ISLA reference seeds 42/43, "
            "GeoTransolver-512 seeds 42/43, 500 epochs, 200,000 cells), float32 inference at 200,000 cells via "
            "recipe_globin/src/infer.py under the code_ref snapshot, mirroring kernel/p200k_fleet/p200k_eval_aga.sbatch "
            "with +save_predictions=true rescale_geometry=true (kernel/forceref/forceref_aga.sbatch, job 885134). "
            "Force coefficient vector = sum over cells of (-Cp n + Cf) A_c w_c with A_ref = 1, Cp = (p - p_inf)/q_inf, "
            "Cf = tau/q_inf, q_inf = 0.5 rho |U_inf|^2, n = Mesh.cell_normals (cell winding), w_c = Horvitz-Thompson "
            "measure weight (N/k) on the sampled cells and 1 on the full surface; CD along U_inf, CL along z "
            "orthogonalized against U_inf, CS completing the triad (forces.build_axis_frame). truth_full: ground truth on "
            "all vehicle cells of the full validation mesh; truth_sampled: ground truth on the 200,000 evaluated cells; "
            "pred: checkpoint prediction on those cells. Relative errors are |pred - truth| / |truth| per car and the "
            "population relative L1/L2 over the 48 cars; correlations are Pearson and Spearman over the 48 cars."
        ),
        "source": str(SRC.relative_to(HERE.parent)),
        "summary": summary,
        "per_car": per_car,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)
    s = summary["sampling"]
    print(f"sampling: CD per-car rel err mean {s['CD']['per_car_rel_err_mean']:.4f} max {s['CD']['per_car_rel_err_max']:.4f};"
          f" CL mean {s['CL']['per_car_rel_err_mean']:.4f} max {s['CL']['per_car_rel_err_max']:.4f}")
    for model in MODELS:
        for c in COEFFS:
            sm = summary["models"][model][c]["seed_mean"]
            print(f"{model:18s} {c}: relL1 full {sm['rel_l1_vs_full']:.4f} sampled {sm['rel_l1_vs_sampled']:.4f} | "
                  f"per-car mean full {sm['per_car_rel_err_vs_full_mean']:.4f} sampled {sm['per_car_rel_err_vs_sampled_mean']:.4f} | "
                  f"pearson full {sm['pearson_vs_full']:.4f} sampled {sm['pearson_vs_sampled']:.4f}")
    print("ratios GT/ISLA:", json.dumps(summary["ratios_GeoTransolver-512_over_ISLA"], indent=None))
    print("integrator vs recipe:", recipe_dev)
    print("sampled truth spread across runs:", sampled_spread)


if __name__ == "__main__":
    main()
