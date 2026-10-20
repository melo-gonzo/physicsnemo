"""E3b ANCHOR readout: seed means, ratios to A, the registered rules and predictions (PREREG.md).

Inputs beside this file: scores.json (scores.py), cost_partial.json + cost_mesh42.json (cost.py),
qpop_f32_reduced.json and qpop_f64_reduced.json (kernel/anchor_af/qpop/reduce.py). Writes tables.md and verdict.json.
"""
import json, math
from pathlib import Path

H = Path(__file__).parent
S = json.loads((H / "scores.json").read_text())
C = {**json.loads((H / "cost_partial.json").read_text()), **json.loads((H / "cost_mesh42.json").read_text())}
Q32 = json.loads((H / "qpop_f32_reduced.json").read_text())["runs"]
Q64 = json.loads((H / "qpop_f64_reduced.json").read_text())["runs"]
ARMS = {"A": "isla", "B": "passive", "C": "anchor", "D": "dtt", "E": "mesh"}
F = {"p": "pressure_l2", "v": "velocity_l2", "nut": "nut_l2"}
SEEDS = (42, 43)
SW = ("half", "far", "wall", "chunks")


def val(arm, seed, proto, f):
    return S[f"anc_{ARMS[arm]}_seed{seed}/{proto}"][F[f]]


def mean(arm, proto, f):
    return sum(val(arm, s, proto, f) for s in SEEDS) / 2


md, V = [], {}
for proto in ("native", "posed"):
    md += [f"## {proto}: mean rel-L2 over 200 full_test cases (seed mean; seed 42 / 43)", "",
           "| arm | p | v | nut |", "|---|---|---|---|"]
    for a in ARMS:
        md.append(f"| {a} {ARMS[a]} | " + " | ".join(
            f"{mean(a, proto, f):.5f} ({val(a, 42, proto, f):.5f} / {val(a, 43, proto, f):.5f})" for f in F) + " |")
    md += ["", f"### {proto}: ratios of seed means (per-seed ratios, paired by seed)", "",
           "| ratio | p | v | nut | geomean(p, v) |", "|---|---|---|---|---|"]
    for num, den in (("C", "A"), ("B", "A"), ("D", "A"), ("E", "A"), ("C", "D"), ("C", "B"), ("C", "E")):
        r = {f: mean(num, proto, f) / mean(den, proto, f) for f in F}
        ps = {f: [val(num, s, proto, f) / val(den, s, proto, f) for s in SEEDS] for f in F}
        g = math.sqrt(r["p"] * r["v"])
        V[f"{proto}/{num}/{den}"] = dict(seedmean=r, per_seed=ps, geomean_pv=g)
        md.append(f"| {num}/{den} | " + " | ".join(f"{r[f]:.3f} ({ps[f][0]:.3f} / {ps[f][1]:.3f})" for f in F)
                  + f" | {g:.3f} |")
    md.append("")

md += ["## posed / native per run", "", "| run | p | v | nut |", "|---|---|---|---|"]
p9 = 0.0
for a in ARMS:
    for s in SEEDS:
        r = {f: val(a, s, "posed", f) / val(a, s, "native", f) for f in F}
        p9 = max(p9, *(abs(x - 1) for x in r.values()))
        V[f"posed_over_native/{a}{s}"] = r
        md.append(f"| {a} seed{s} | " + " | ".join(f"{r[f]:.4f}" for f in F) + " |")
md.append("")

# Validity (PREREG, as clarified and confirmed): float32 |ratio - 1| <= 1e-4 in every swapped condition and field,
# and float64 largest change <= 1e-9 of the RMS target.
md += ["## QUERY-POP (float32: 200 cases; float64: first 20 cases, eager blocks)", "",
       "| run | f32 max abs(ratio-1) | f32 max change/RMS | f32 repeat floor | f64 max abs(ratio-1) | f64 max change/RMS | f32 far ratio p / v / nut | validity |",
       "|---|---|---|---|---|---|---|---|"]
for a in ARMS:
    for s in SEEDS:
        run = f"anc_{ARMS[a]}_seed{s}"
        c32, c64 = Q32[run]["conditions"], Q64[run]["conditions"]
        r32 = max(abs(c32[c]["ratio"][f] - 1) for c in SW for f in F.values())
        ch32 = max(c32[c]["max_change_over_rms"][f] for c in SW for f in F.values())
        rep = max(c32["repeat"]["max_change_over_rms"].values())
        r64 = max(abs(c64[c]["ratio"][f] - 1) for c in SW for f in F.values())
        ch64 = max(c64[c]["max_change_over_rms"][f] for c in SW for f in F.values())
        ok = r32 <= 1e-4 and ch64 <= 1e-9
        far = " / ".join(f"{c32['far']['ratio'][f]:.3g}" for f in F.values())
        V[f"qpop/{run}"] = dict(f32_ratio_dev=r32, f32_change=ch32, f32_repeat=rep, f64_ratio_dev=r64, f64_change=ch64,
                                valid=ok, f32_full_vs_infer=Q32[run]["validity"]["max_rel_diff"])
        md.append(f"| {run} | {r32:.1e} | {ch32:.1e} | {rep:.1e} | {r64:.1e} | {ch64:.1e} | {far} | {'pass' if ok else 'FAIL'} |")
md.append("")

md += ["## Cost (train.log, rank 0)", "",
       "| run | median step, epoch 0 steps 10-200 (s) | × A | median step, epoch 250 (s) | × A | median all steps (s) | peak Mem (GB) |",
       "|---|---|---|---|---|---|---|"]
a0 = sum(C[f"anc_isla_seed{s}"]["epoch0_median_s"] for s in SEEDS) / 2
a250 = sum(C[f"anc_isla_seed{s}"]["epoch250_median_s"] for s in SEEDS) / 2
for a in ARMS:
    for s in SEEDS:
        c = C[f"anc_{ARMS[a]}_seed{s}"]
        md.append(f"| {a} seed{s} | {c['epoch0_median_s']:.3f} | {c['epoch0_median_s'] / a0:.2f} | {c['epoch250_median_s']:.3f} | "
                  f"{c['epoch250_median_s'] / a250:.2f} | {c['all_median_s']:.3f} | {c['peak_mem_gb']:.1f} |")
md.append("")

# Rules.
ca = V["native/C/A"]
valid_c = V["qpop/anc_anchor_seed42"]["valid"] and V["qpop/anc_anchor_seed43"]["valid"]
proceed = ca["geomean_pv"] <= 1.05 and ca["seedmean"]["nut"] <= 1.10
dnr = any(all(x >= 1.10 for x in ca["per_seed"][f]) for f in ("p", "v"))
verdict = "INVALID" if not valid_c else ("PROCEED" if proceed else ("DOES NOT REPLACE" if dnr else "PETER DECIDES"))
V["verdict"] = verdict
md.insert(0, f"# E3b ANCHOR readout tables (verdict: {verdict})\n")
(H / "tables.md").write_text("\n".join(md) + "\n")
(H / "verdict.json").write_text(json.dumps(V, indent=1))
print("\n".join(md))
print("P9 max |posed/native - 1|:", p9)
