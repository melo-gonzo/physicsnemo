"""Equal-epoch companion at the largest OLBC Poisson rung (ruling 3 of 2026-09-30, #sec-nb-rulings-2026-09-30): the reference
ISLA and GeoTransolver-512 both trained for 500 epochs on 8,448 cases (campaign pc rows 44 to 47, code_v11 + recipe_v4, lanes
pc_poisson_{ref,gt512}_n8448_e500_seed{42,43}), float32 relative L2 of u on the 256 held-out cases, averaged over cases then seeds.
Ratio = GeoTransolver-512 / ISLA (below 1 favors GeoTransolver). Companion to the equal-step rung of olbc_ref_2026-09-29.json
(120 epochs at 8,448 so that the step count matches the 512 and 2,048 rungs). usage: python poisson8448_equal_epoch_reduce_2026-10-02.py"""
import glob, json, os, statistics as st

R = os.path.dirname(os.path.abspath(__file__))


def per_case(run):
    fs = glob.glob(f"{R}/campaigns_2026-09-29/pc/metrics/{run}/**/metrics.jsonl", recursive=True)
    rows = [json.loads(l)["metrics"] for l in open(sorted(fs)[0]) if '"infer_step"' in l]
    return st.mean(r["u_l2"] for r in rows), len(rows)


out = {"protocol": __doc__, "lanes": {}}
for arm in ("ref", "gt512"):
    for s in (42, 43):
        m, n = per_case(f"pc_poisson_{arm}_n8448_e500_seed{s}")
        out["lanes"][f"{arm}_seed{s}"] = {"u_l2": m, "n_cases": n}
isla = {s: out["lanes"][f"ref_seed{s}"]["u_l2"] for s in (42, 43)}
gt = {s: out["lanes"][f"gt512_seed{s}"]["u_l2"] for s in (42, 43)}
out["isla_u_l2"] = isla; out["gt512_u_l2"] = gt
out["ratio_gt512_over_isla"] = {"seed_means": st.mean(gt.values()) / st.mean(isla.values()), "per_seed": {s: gt[s] / isla[s] for s in (42, 43)}}
out["isla_seed_spread"] = max(isla.values()) / min(isla.values())
json.dump(out, open(f"{R}/poisson8448_equal_epoch_2026-10-02.json", "w"), indent=1)
print(f"ISLA u {isla[42]:.4f} / {isla[43]:.4f}  GT-512 u {gt[42]:.4f} / {gt[43]:.4f}  GT/ISLA {out['ratio_gt512_over_isla']['seed_means']:.3f} "
      f"(per seed {out['ratio_gt512_over_isla']['per_seed'][42]:.3f} / {out['ratio_gt512_over_isla']['per_seed'][43]:.3f}); ISLA seed spread {out['isla_seed_spread']:.2f}x")
