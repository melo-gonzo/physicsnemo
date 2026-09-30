"""WING-ANATOMY-SURFACE reducer and figure (2026-09-30, @sec-nb-wing-anatomy-surface).

Reads wing_anatomy_surface_2026-09-30.json, written on the cluster by
$T/highlift/wing_anatomy_surface.py from the per-cell float32 predictions of the HiLiftAeroML
210-wing rung (ISLA reference configuration against GeoTransolver-512, 180 validation wings) and
the fixed-angle 12-degree rung (ISLA against GeoTransolver-256, 18 validation wings), 200,000
cells, seeds 42 and 43 paired. Per class the artifact holds, per lane, the case mean of the
area-weighted RMS pressure error of the class divided by the area-weighted standard deviation of
the case's true pressure, and the ratio ISLA / GeoTransolver per seed pair. Classes on every
wing: curvature bins of the normal turning rate kappa (top 10% of area, next 40%, bottom 50%).
Classes on the five STL-bearing wings only (pilot): slat, main, flap, the gap band, element x
high curvature.

Draws one figure: per rung, the class ratio ISLA / GeoTransolver, seeds as dots and the seed
mean as the filled square, against the whole-surface ratio (solid line) and the preregistered
bar (whole-surface ratio + 0.15, dashed). Prints the verdict table. Output figure beside the
json (results/figures does not exist)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
RES = Path(__file__).resolve().parent
sys.path.insert(0, str(RES.parent / "book"))
import palette as P  # noqa: E402

data = json.loads((RES / "wing_anatomy_surface_2026-09-30.json").read_text())
BAR = data["bar"]
LABEL = {"all": "whole surface", "high_curvature": "high curvature (top 10% of area by κ)", "mid_curvature": "medium curvature (next 40%)",
         "low_curvature": "low curvature (bottom 50%)", "slat": "slat (pilot)", "main": "main wing and body (pilot)", "flap": "flap (pilot)",
         "gap_band": "gap band, within 0.02 L_ref of another element (pilot)", "slat_high_curvature": "slat, high curvature (pilot)",
         "main_high_curvature": "main, high curvature (pilot)", "flap_high_curvature": "flap, high curvature (pilot)"}
ORDER = ["high_curvature", "mid_curvature", "low_curvature", "slat", "flap", "gap_band", "main", "slat_high_curvature", "flap_high_curvature", "main_high_curvature"]
TITLE = {"hl210": "210-wing rung: ISLA against GeoTransolver-512", "single_aoa_12": "fixed-angle rung (12 degrees): ISLA against GeoTransolver-256"}

rungs = [r for r in ("hl210", "single_aoa_12") if r in data["rungs"]]
fig, axes = plt.subplots(len(rungs), 1, figsize=(8.4, 2.4 + 3.0 * len(rungs)), sharex=True, squeeze=False)
for ax, rung in zip(axes[:, 0], rungs):
    R = data["rungs"][rung]
    S = dict(R["summary"]); S.update({k: v for k, v in R.get("pilot_element_summary", {}).items() if k not in S})  # pilot adds the element classes only
    whole = S["all"]["ratio_isla_over_gt"]; wm = sum(whole.values()) / len(whole)
    names = [c for c in ORDER if c in S and S[c]["ratio_isla_over_gt"]]
    y = list(range(len(names)))[::-1]
    print(f"== {rung} ({R['n_cases']} wings, {R['n_cases_with_stl']} with an STL): whole-surface ISLA/GT seed 42 {whole['42']:.3f}, seed 43 {whole['43']:.3f}; "
          f"kappa bin edges {R['kappa_thresholds_per_inch'][0]:.4f} and {R['kappa_thresholds_per_inch'][1]:.4f} 1/in")
    for yy, c in zip(y, names):
        r = S[c]["ratio_isla_over_gt"]; vals = [r[s] for s in sorted(r)]
        pilot = c not in R["summary"]
        ax.plot(vals, [yy] * len(vals), ls="none", marker="o", ms=4, color=P.INK2, alpha=0.6, zorder=3)
        ax.plot([sum(vals) / len(vals)], [yy], ls="none", marker="s", ms=7, color=P.BLUE, mfc=("white" if pilot else P.BLUE), mec=P.BLUE, mew=1.8, zorder=4)
        ex = S[c]["excess_over_whole_surface"]
        share = {k.rsplit("_seed", 1)[0][:4] + k[-2:]: round(v["case_mean_error_share"], 3) for k, v in S[c]["per_run"].items()}
        area = list(S[c]["per_run"].values())[0]["case_mean_area_share"]
        print(f"   {c:22s} ratio 42 {r['42']:.3f}  43 {r['43']:.3f}   excess {ex['42']:+.3f} {ex['43']:+.3f}   carries={S[c]['carries_excess']}   "
              f"area share {area:.3f}   error share {share}")
    ax.axvline(wm, color=P.MUTED, lw=1.5, zorder=2); ax.axvline(wm + BAR, color=P.MUTED, lw=1.2, ls="--", zorder=2)
    P.label_line(ax, wm, y[0] + 0.6, f"whole surface {wm:.2f}", P.MUTED, dx=-0.01, ha="right", fontsize=8.5)
    P.label_line(ax, wm + BAR, y[0] + 0.6, f"bar {wm + BAR:.2f}", P.MUTED, dx=0.01, fontsize=8.5)
    ax.set_yticks(y); ax.set_yticklabels([LABEL[c] for c in names], fontsize=8.5); ax.set_ylim(-0.7, y[0] + 1.1)
    ax.set_title(f"{TITLE[rung]} ({R['n_cases']} validation wings, 200,000 cells, float32)")
    ax.axvline(1.0, color=P.BASELINE, lw=1.0, zorder=1)
axes[-1, 0].set_xlabel("ISLA ÷ GeoTransolver, case-mean area-weighted RMS pressure error of the class [–]")
axes[-1, 0].set_xlim(left=0.6)
fig.text(0.01, -0.01, "filled square: seed mean over all validation wings; open square: pilot classes on the wings with an STL geometry (4 of 180 at 210 wings, 1 of 18 on the fixed-angle rung); dots: seeds 42 and 43",
         fontsize=8, color=P.INK2)
fig.tight_layout()
out = RES / "wing_anatomy_surface_2026-09-30.png"
fig.savefig(out, bbox_inches="tight")
print("figure:", out)
