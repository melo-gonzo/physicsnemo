"""WING-ELEMENT-ANATOMY reducer and figure (2026-09-30, @sec-nb-wing-element-anatomy).

Reads wing_element_anatomy_2026-09-30.json, written on the cluster by
$T/highlift/wing_element_anatomy.py from the per-cell float32 predictions of the HiLiftAeroML
210-wing rung (ISLA reference configuration against GeoTransolver-512, 180 validation wings) and
the fixed-angle 12-degree rung (ISLA against GeoTransolver-256, 18 validation wings), 200,000
cells per case, seeds 42 and 43 paired, with every sampled cell labelled by high-lift element
from the connected components of the case's full boundary mesh ($T/highlift/wing_element_labels.py).
Per class the artifact holds, per lane, the case mean of the area-weighted RMS pressure error of
the class divided by the area-weighted standard deviation of the case's true pressure, the ratio
ISLA / GeoTransolver per seed pair, and each model's error share against the class's area share.

Draws one figure: per rung, the class ratio ISLA / GeoTransolver, seeds as dots and the seed mean
as the filled square, against the whole-surface ratio (solid line) and the preregistered bar
(whole-surface ratio + 0.15, dashed). Prints the verdict table. Output figure beside the json."""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
RES = Path(__file__).resolve().parent
sys.path.insert(0, str(RES.parent / "book"))
import palette as P  # noqa: E402

data = json.loads((RES / "wing_element_anatomy_2026-09-30.json").read_text())
BAR = data["bar"]
LABEL = {"all": "whole surface", "slat": "slat", "main": "main wing and body", "flap": "flap", "other": "other bodies",
         "gap_band": "gap band (within 0.02 L_ref of another element)", "slat_core": "slat outside the gap band",
         "main_core": "main outside the gap band", "flap_core": "flap outside the gap band"}
ORDER = ["slat", "flap", "gap_band", "main", "other", "slat_core", "flap_core", "main_core"]
TITLE = {"hl210": "210-wing rung: ISLA against GeoTransolver-512", "single_aoa_12": "fixed-angle rung (12 degrees): ISLA against GeoTransolver-256"}

rungs = [r for r in ("hl210", "single_aoa_12") if r in data["rungs"] and data["rungs"][r]["summary"]]
fig, axes = plt.subplots(len(rungs), 1, figsize=(8.4, 2.2 + 2.8 * len(rungs)), sharex=True, squeeze=False)
for ax, rung in zip(axes[:, 0], rungs):
    R = data["rungs"][rung]; S = R["summary"]
    whole = S["all"]["ratio_isla_over_gt"]; wm = sum(whole.values()) / len(whole)
    names = [c for c in ORDER if c in S and len(S[c]["ratio_isla_over_gt"]) == 2]
    y = list(range(len(names)))[::-1]
    print(f"== {rung} ({R['n_cases_labelled']} of {R['n_cases_shared']} validation wings labelled): whole-surface ISLA/GT seed 42 {whole['42']:.3f}, seed 43 {whole['43']:.3f}")
    for yy, c in zip(y, names):
        r = S[c]["ratio_isla_over_gt"]; vals = [r[s] for s in sorted(r)]
        ax.plot(vals, [yy] * len(vals), ls="none", marker="o", ms=4, color=P.INK2, alpha=0.6, zorder=3)
        ax.plot([sum(vals) / len(vals)], [yy], ls="none", marker="s", ms=7, color=P.BLUE, zorder=4)
        ex = S[c]["excess_over_whole_surface"]
        share = {k.rsplit("_seed", 1)[0][:4] + k[-2:]: round(v["case_mean_error_share"], 3) for k, v in S[c]["per_run"].items()}
        area = list(S[c]["per_run"].values())[0]["case_mean_area_share"]
        print(f"   {c:12s} ratio 42 {r['42']:.3f}  43 {r['43']:.3f}   excess {ex['42']:+.3f} {ex['43']:+.3f}   carries={S[c]['carries_excess']}   "
              f"area share {area:.3f}   error share {share}")
    ax.axvline(wm, color=P.MUTED, lw=1.5, zorder=2); ax.axvline(wm + BAR, color=P.MUTED, lw=1.2, ls="--", zorder=2)
    P.label_line(ax, wm, y[0] + 0.6, f"whole surface {wm:.2f}", P.MUTED, dx=-0.01, ha="right", fontsize=8.5)
    P.label_line(ax, wm + BAR, y[0] + 0.6, f"bar {wm + BAR:.2f}", P.MUTED, dx=0.01, fontsize=8.5)
    ax.set_yticks(y); ax.set_yticklabels([LABEL[c] for c in names], fontsize=8.5); ax.set_ylim(-0.7, y[0] + 1.1)
    ax.set_title(f"{TITLE[rung]} ({R['n_cases_labelled']} validation wings, 200,000 cells, float32)")
    ax.axvline(1.0, color=P.BASELINE, lw=1.0, zorder=1)
axes[-1, 0].set_xlabel("ISLA ÷ GeoTransolver, case-mean area-weighted RMS pressure error of the class [–]")
axes[-1, 0].set_xlim(left=0.6)
fig.text(0.01, -0.01, "filled square: seed mean over the labelled validation wings; dots: seeds 42 and 43; element labels from the connected components of each wing's full boundary mesh",
         fontsize=8, color=P.INK2)
fig.tight_layout()
out = RES / "wing_element_anatomy_2026-09-30.png"
fig.savefig(out, bbox_inches="tight")
print("figure:", out)
