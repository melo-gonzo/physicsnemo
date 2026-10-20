"""HILIFT-090: summarize percase_region.csv / checks.csv (from cluster/spatial/spatial_reduce.py, LHR jobs savepred +
spred) into SPATIAL.md. ISLA = mean of A2 seeds 42/43; GT / T = the unit-drive seed-42 lanes.
Squared relative error e^2 = sum over regions of e_R^2 per case; groups average e_R^2 over their cases."""
import csv
import collections
import math
import re
import sys

D = sys.argv[1] if len(sys.argv) > 1 else "."
rows = list(csv.DictReader(open(f"{D}/percase_region.csv")))
chk = {r["case"]: r for r in csv.DictReader(open(f"{D}/checks.csv"))}
aoa = {c: int(re.search(r"AoA_(\d+)", c).group(1)) for c in chk}
E = collections.defaultdict(dict)  # (case, family, region) -> run -> (e2p, e2t, area)
for r in rows:
    E[(r["case"], r["family"], r["region"])][r["run"]] = (float(r["e2_p"]), float(r["e2_t"]), float(r["area"]))
ARMS = {"ISLA": ["a2r_hl_isla_seed42", "a2r_hl_isla_seed43"], "GT": ["udrv_hl_gt_seed42"], "T": ["udrv_hl_t_seed42"]}


def arm(v, a, j):
    xs = [v[r][j] for r in ARMS[a] if r in v]
    return sum(xs) / len(xs) if xs else float("nan")


cases = sorted(c for c in chk if all(f"e_p_{r}" in chk[c] and chk[c][f"e_p_{r}"] for r in sum(ARMS.values(), [])))
ep = {c: {a: sum(float(chk[c][f"e_p_{r}"]) for r in ARMS[a]) / len(ARMS[a]) for a in ARMS} for c in cases}
worst = sorted(cases, key=lambda c: -ep[c]["ISLA"])[:18]
groups = {"all": cases, "ISLA worst 18": worst, "AoA>=16": [c for c in cases if aoa[c] >= 16], "AoA<16": [c for c in cases if aoa[c] < 16]}
out = []
P = out.append
P("# HILIFT-090 spatial split of HiLift native error (A2 ISLA vs unit-drive baselines)\n")
P(f"Cases with all four arms: {len(cases)}. ISLA = mean of A2 seeds 42/43; GT, T = unit-drive seed 42. Squared relative L2 "
  "error per case split over surface regions (sum over a family's regions = the case's e^2), averaged over the group's cases.\n")
P("## Checks\n")
offs = [float(chk[c][k]) for c in cases for k in chk[c] if k.startswith("offset_dev_") and chk[c][k]]
P(f"- block mapping: max centroid-offset deviation over cases/arms {max(offs):.2e} in (constant CenterMesh translation => exact cell match)")
P(f"- ISLA worst 18 AoA: {collections.Counter(aoa[c] for c in worst)}")
for nm in ["slat", "main", "flap", "other"]:
    d = [abs(float(chk[c][f'share_{nm}']) - float(chk[c][f'json_{nm}'])) for c in cases if chk[c]['labels'] == 'True']
    P(f"- element area share vs the labels json, {nm}: max |diff| {max(d):.3f}")
P("")
for fam in ["element", "cp", "flow", "span", "elem_flow"]:
    for j, fld in [(0, "pressure"), (1, "wall shear")]:
        P(f"## {fld} by {fam}\n")
        regs = sorted({k[2] for k in E if k[1] == fam and not k[2].startswith("unlabeled")})
        P("| group | region | area | ISLA e² ×1e4 | GT | T | ISLA−GT | share of ISLA−GT | ISLA−T | share of ISLA−T |")
        P("|---|---|---|---|---|---|---|---|---|---|")
        for g, cs in groups.items():
            if fam in ("element", "elem_flow"):
                cs = [c for c in cs if chk[c]["labels"] == "True"]
                g = f"{g} (labeled, n={len(cs)})"
            tot = {a: sum(arm(E[(c, fam, rg)], a, j) for c in cs for rg in (regs if fam != "elem_flow" else sorted({k[2] for k in E if k[1] == "element" and k[2] != "unlabeled"}))) / len(cs) for a in ARMS}
            if fam == "elem_flow":
                tot = {a: sum(arm(E[(c, "element", rg)], a, j) for c in cs for rg in ["slat", "main", "flap", "other"]) / len(cs) for a in ARMS}
            for rg in regs:
                vals = {a: sum(arm(E[(c, fam, rg)], a, j) for c in cs) / len(cs) for a in ARMS}
                area = sum(arm(E[(c, fam, rg)], "ISLA", 2) for c in cs) / len(cs)
                dg, dt = vals["ISLA"] - vals["GT"], vals["ISLA"] - vals["T"]
                P(f"| {g} | {rg} | {area:.3f} | {1e4*vals['ISLA']:.2f} | {1e4*vals['GT']:.2f} | {1e4*vals['T']:.2f} | {1e4*dg:+.2f} | "
                  f"{dg/(tot['ISLA']-tot['GT']):+.2f} | {1e4*dt:+.2f} | {dt/(tot['ISLA']-tot['T']):+.2f} |")
            P(f"| {g} | **total** | 1 | {1e4*tot['ISLA']:.2f} | {1e4*tot['GT']:.2f} | {1e4*tot['T']:.2f} | {1e4*(tot['ISLA']-tot['GT']):+.2f} | 1 | {1e4*(tot['ISLA']-tot['T']):+.2f} | 1 |")
        P("")
P("## Per-case pressure error, ISLA worst 18 (this re-run)\n")
P("| case | AoA | ISLA | GT | T | ISLA reversed-flow e² share | GT reversed-flow e² share |")
P("|---|---|---|---|---|---|---|")
for c in worst:
    rv = E[(c, "flow", "reversed")]
    si = arm(rv, "ISLA", 0) / ep[c]["ISLA"] ** 2 if ep[c]["ISLA"] else float("nan")
    sg = arm(rv, "GT", 0) / ep[c]["GT"] ** 2
    P(f"| {c} | {aoa[c]} | {ep[c]['ISLA']:.4f} | {ep[c]['GT']:.4f} | {ep[c]['T']:.4f} | {si:.2f} | {sg:.2f} |")
open(f"{D}/SPATIAL.md", "w").write("\n".join(out) + "\n")
print("\n".join(out[:12]))
