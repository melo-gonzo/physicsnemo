"""EXACT-SDF (prereg @sec-nb-exact-sdf-prereg): the interior token's wall distance and direction from the exact point-to-triangle
signed distance of the boundary mesh (physicsnemo.mesh.spatial.sdf, wall_geometry="exact") against the nearest-sample form
(wall_geometry="sampled"), DrivAerML Volume 435 cars, 200,000 boundary cells x 100,000 queries, reference configuration, two seeds,
code_v13 + recipe_v5 (campaign ex). Float32 scoring on the held-out cars. Bars: exact / sampled at or below 0.90 on velocity and
0.97 on pressure adopts (pending the 2-D form on AirfRANS); above 1.03 on any field refutes; otherwise the sampled form stays."""
import glob, json, statistics as st
R = "/home/psharpe/gh/physicsnemo/examples/cfd/isla/results"
def per_case(run):
    fs = glob.glob(f"{R}/campaigns_2026-09-29/ex/metrics/{run}/**/metrics.jsonl", recursive=True)
    if not fs: return None
    rows = [json.loads(l)["metrics"] for l in open(sorted(fs)[0]) if '"infer_step"' in l]
    return {k: st.mean(r[k] for r in rows) for k in ("pressure_l2", "velocity_l2", "nut_l2") if k in rows[0]}
arms = {a: {s: per_case(f"ex_drvol_{a}_seed{s}") for s in ("42", "43")} for a in ("exact", "sampled")}
arms = {a: {s: v for s, v in d.items() if v} for a, d in arms.items()}
fields = ["pressure_l2", "velocity_l2", "nut_l2"]
means = {a: {f: st.mean(arms[a][s][f] for s in arms[a]) for f in fields} for a in arms}
ratio = {f: means["exact"][f] / means["sampled"][f] for f in fields}
per_seed = {s: {f: arms["exact"][s][f] / arms["sampled"][s][f] for f in fields} for s in ("42", "43") if s in arms["exact"] and s in arms["sampled"]}
adopt = ratio["velocity_l2"] <= 0.90 and ratio["pressure_l2"] <= 0.97
refuted = any(r > 1.03 for r in ratio.values())
verdict = {"adopt_pending_2d": adopt, "refuted": refuted, "neither_sampled_stays": not adopt and not refuted, "n_seeds": min(len(arms["exact"]), len(arms["sampled"]))}
out = {"protocol": __doc__, "per_seed_values": arms, "means": means, "ratio_exact_over_sampled": ratio, "per_seed_ratio": per_seed, "verdict": verdict}
json.dump(out, open(f"{R}/exact_sdf_2026-09-29.json", "w"), indent=1)
for f in fields:
    print(f"{f:<12} sampled {means['sampled'][f]:.4f} exact {means['exact'][f]:.4f} ratio {ratio[f]:.3f} per seed", {s: round(per_seed[s][f], 3) for s in per_seed})
print("verdict:", verdict)
