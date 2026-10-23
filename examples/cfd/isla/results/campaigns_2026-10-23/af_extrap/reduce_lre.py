"""AF-EXTRAP LRE (PREREG §14) and LIN (§14b) readout (written 2026-10-23 before reading their scores): readings (a) below-band
pressure ratio X ÷ afxf ISLA per seed, (b) the in-band fit guard on reynolds_train, beside (B) the near-band quartiles, (c) R1 and R2
with X in ISLA's place against both baselines (reference r090fix_af_isla; σ pooled as §6 over 8 pairs with X's test pair in place
of ISLA's), (d) the clamp test on X; then the §14b joint reading. Writes lre.json and LRE.md. usage: python reduce_lre.py"""
import json, math, os, statistics as st

H = os.path.dirname(os.path.abspath(__file__))
exec(open(f"{H}/posthoc.py").read().split("ID = {a:")[0])  # helpers: per_case, CASES, MAN, F, SEEDS, REF, run, slope, spearman
SC, IB, CL = f"{H}/scores/evals_afx", f"{H}/scores/evals_afx_inband", f"{H}/scores/evals_afx_clamp"
X = {"afxl": ("LRE", "airfrans_uvolume_2d_reynolds_lre", (0.09565967965056732, 0.3349428385888533), lambda c: CASES[c]["log10_re"] - 6.0),
     "afxlin": ("LIN", "airfrans_uvolume_2d_reynolds_lin", (-1.8569697241416367, 1.9819439540137824),
                lambda c: (10 ** CASES[c]["log10_re"] - 1689509.7116) / 238616.3216)}
test = sorted(MAN["reynolds_test"]); train = sorted(MAN["reynolds_train"])
below = [c for c in test if CASES[c]["Re"] < 3e6]; above = [c for c in test if CASES[c]["Re"] > 5e6]
assert len(below) == 254 and len(above) == 242


def load(tree, r, n_cases):
    p = f"{tree}/{r}/native/{r}/metrics.jsonl"
    ep = next(json.loads(l)["epoch"] for l in open(p) if '"phase": "config"' in l)
    assert ep == 500, (p, ep)
    pc = per_case(p)
    assert len(pc) == n_cases, (p, len(pc))
    return pc


def dist(c):
    re = CASES[c]["Re"]
    return math.log10(3e6 / re) if re < 3e6 else math.log10(re / 5e6)


q1 = {s: sorted(cs, key=dist)[: len(cs) // 4] for s, cs in (("below", below), ("above", above))}
cm = lambda pc, cs, f: st.mean(pc[c][f] for c in cs)
base = {s: load(SC, f"afxf_reynolds_isla_seed{s}", 496) for s in SEEDS}
base_in = {s: load(IB, f"afxf_reynolds_isla_seed{s}", 504) for s in SEEDS}
BL = {b: {s: load(SC, f"afx_reynolds_{b}_seed{s}", 496) for s in SEEDS} for b in ("gt", "t")}
REFI = {s: per_case(f"{REF['isla']}_seed{s}/native/r090fix_af_isla_seed{s}/metrics.jsonl") for s in SEEDS}
REFB = {b: {s: per_case(f"{REF[b]}_seed{s}/native/{os.path.basename(REF[b])}_seed{s}/metrics.jsonl") for s in SEEDS} for b in ("gt", "t")}
REFD = {s: per_case(f"{REF['dtt']}_seed{s}/native/r090fix_af_dtt_seed{s}/metrics.jsonl") for s in SEEDS}
DTT = {s: load(SC, f"afxf_reynolds_dtt_seed{s}", 496) for s in SEEDS}
sig = lambda a, b: abs(math.log(a / b)) / math.sqrt(2)
res, L, helps = {}, ["# LRE and LIN readout (PREREG §14, §14b)", "", "ISLA family fix-trained (code_cmpaf_fix / code_afxlin); reference for R2 r090fix_af_isla.", ""], {}
for k, (name, ds, (zl, zh), zf) in X.items():
    pc = {s: load(SC, f"{k}_reynolds_isla_seed{s}", 496) for s in SEEDS}
    pin = {s: load(IB, f"{k}_reynolds_isla_seed{s}", 504) for s in SEEDS}
    assert set(pin["42"]) == set(train) and set(base_in["42"]) == set(train)
    T = {}
    # (a)
    r = {s: cm(pc[s], below, "pressure_l2") / cm(base[s], below, "pressure_l2") for s in SEEDS}
    va = ("the fixed scale helps" if k == "afxl" else "helps") if all(v <= 0.8 for v in r.values()) else \
         ("no effect" if all(0.95 <= v <= 1.05 for v in r.values()) else "mixed")
    T["a"] = {"r": r, "verdict": va}
    helps[name] = all(v <= 0.8 for v in r.values())
    # (b)
    g = {f: st.mean(cm(pin[s], train, f) for s in SEEDS) / st.mean(cm(base_in[s], train, f) for s in SEEDS) for f in F}
    T["b"] = {"ratio": g, "per_seed": {s: {f: cm(pin[s], train, f) / cm(base_in[s], train, f) for f in F} for s in SEEDS},
              "verdict": "passes" if all(v <= 1.05 for v in g.values()) else "the fixed scale costs in-band fit" if k == "afxl" else "costs in-band fit"}
    # (B)
    T["B"] = {side: {f: st.mean(cm(pc[s], cs, f) for s in SEEDS) / st.mean(cm(base[s], cs, f) for s in SEEDS) for f in F} for side, cs in q1.items()}
    # (c)
    E = {s: {f: cm(pc[s], test, f) for f in F} for s in SEEDS}
    I = {s: {f: st.mean(REFI[s][c][f] for c in REFI[s]) for f in F} for s in SEEDS}
    sigma = {f: math.sqrt(st.mean([sig(E["42"][f], E["43"][f]) ** 2] + [sig(cm(BL[b]["42"], test, f), cm(BL[b]["43"], test, f)) ** 2 for b in BL]
                                  + [sig(cm(DTT["42"], test, f), cm(DTT["43"], test, f)) ** 2]
                                  + [sig(I["42"][f], I["43"][f]) ** 2]
                                  + [sig(*(st.mean(R[s][c][f] for c in R[s]) for s in SEEDS)) ** 2 for R in (REFB["gt"], REFB["t"], REFD)])) for f in F}
    T["sigma"] = sigma
    T["c"] = {}
    for b in BL:
        EB = {s: {f: cm(BL[b][s], test, f) for f in F} for s in SEEDS}
        IBb = {s: {f: st.mean(REFB[b][s][c][f] for c in REFB[b][s]) for f in F} for s in SEEDS}
        for f in F:
            m = lambda d: st.mean(d[s][f] for s in SEEDS)
            ratio = m(EB) / m(E); band1 = max(0.03, 2 * sigma[f])
            sep_up = min(EB[s][f] for s in SEEDS) > max(E[s][f] for s in SEEDS); sep_dn = max(EB[s][f] for s in SEEDS) < min(E[s][f] for s in SEEDS)
            v1 = ("leads" if sep_up else "level (straddle)") if math.log(ratio) >= math.log(1 + band1) else \
                 ("trails" if sep_dn else "level (straddle)") if math.log(ratio) <= -math.log(1 + band1) else "level"
            rho = (m(EB) / m(IBb)) / (m(E) / m(I)); pairs = {s: (EB[s][f] / IBb[s][f]) / (E[s][f] / I[s][f]) for s in SEEDS}
            band2 = max(math.log(1.10), 2 * math.sqrt(2) * sigma[f])
            v2 = "extrapolates better" if math.log(rho) >= band2 and all(p > 1 for p in pairs.values()) else \
                 "extrapolates worse" if math.log(rho) <= -band2 and all(p < 1 for p in pairs.values()) else "no difference"
            T["c"][f"{b}/{f}"] = {"R1_ratio": ratio, "R1": v1, "rho": rho, "pairs": pairs, "R2": f"{name} {v2}", "band2": math.exp(band2)}
    T["E"] = {f: st.mean(E[s][f] for s in SEEDS) for f in F}
    T["g"] = {f: st.mean(E[s][f] for s in SEEDS) / st.mean(I[s][f] for s in SEEDS) for f in F}
    # (d)
    try:
        cl = {s: load(CL, f"{k}_reynolds_isla_seed{s}", 496) for s in SEEDS}
        inside = [c for c in test if zl <= zf(c) <= zh]
        worst = max((abs(cl[s][c][f] / pc[s][c][f] - 1) for s in SEEDS for c in inside for f in F), default=0.0)
        T["d"] = {"inside": len(inside), "worst_inside": worst,
                  "changed_outside": sum(any(abs(cl[s][c][f] / pc[s][c][f] - 1) > 1e-5 for f in F) for s in SEEDS for c in test if c not in inside),
                  **{side: {f: st.mean(cm(cl[s], cs, f) for s in SEEDS) / st.mean(cm(pc[s], cs, f) for s in SEEDS) for f in F}
                     for side, cs in (("below", below), ("above", above))},
                  "below_per_seed_p": {s: cm(cl[s], below, "pressure_l2") / cm(pc[s], below, "pressure_l2") for s in SEEDS},
                  "below_clamped_p": st.mean(cm(cl[s], below, "pressure_l2") for s in SEEDS)}
    except FileNotFoundError:
        T["d"] = None
    T["slope_below"] = {f: slope([dist(c) for c in below], [st.mean(pc[s][c][f] for s in SEEDS) for c in below]) for f in F}
    res[name] = T
    L += [f"## {name}", "",
          f"- (a) below-band pressure, {name} ÷ afxf ISLA: s42 {r['42']:.3f}, s43 {r['43']:.3f} → **{va}**",
          f"- (b) in-band fit guard (504 training cases; fit inside the band, not generalization): " + " / ".join(f"{g[f]:.3f}" for f in F)
          + f" → **{T['b']['verdict']}**",
          "- (B) near-band held-out (overlaps the population of (a)): below " + " / ".join(f"{T['B']['below'][f]:.3f}" for f in F)
          + ", above " + " / ".join(f"{T['B']['above'][f]:.3f}" for f in F),
          f"- test error p / v / nut " + " / ".join(f"{T['E'][f]:.4f}" for f in F) + "; g against r090fix_af_isla " + " / ".join(f"{T['g'][f]:.2f}" for f in F)
          + "; below-band slope per decade " + " / ".join(f"{T['slope_below'][f]:.3f}" for f in F),
          "- (c) " + "; ".join(f"{b} {F[f]}: R1 {v['R1_ratio']:.3f} {v['R1']}, R2 ρ {v['rho']:.3f} ({v['pairs']['42']:.3f}, {v['pairs']['43']:.3f}) [×{v['band2']:.3f}] {v['R2']}"
                               for (bf, v) in T["c"].items() for b, f in [bf.split("/")]), ]
    if T["d"]:
        d = T["d"]
        L.append(f"- (d) clamp, clamped ÷ native: below " + " / ".join(f"{d['below'][f]:.3f}" for f in F) + f" (pressure s42 {d['below_per_seed_p']['42']:.3f}, s43 {d['below_per_seed_p']['43']:.3f}; clamped below-band pressure {d['below_clamped_p']:.4f}), above "
                 + " / ".join(f"{d['above'][f]:.3f}" for f in F) + f"; non-degeneracy: {d['inside']} cases inside unchanged (max |Δ| {d['worst_inside']:.1e}), {d['changed_outside']} outside case-seeds changed")
    L.append("")
joint = {(True, False): "LIN helps and LRE does not: the log coordinate's reach drives the penalty",
         (False, True): "LRE helps and LIN does not: training dynamics with a narrow input spread",
         (True, True): "both help: both",
         (False, False): "neither helps: the learned Re dependence extrapolates poorly in any coordinate"}[(helps["LIN"], helps["LRE"])]
res["joint"] = joint
L += [f"**Joint reading (§14b): {joint}.** (LRE (a) {res['LRE']['a']['verdict']}; LIN (a) {res['LIN']['a']['verdict']}.)", ""]
json.dump(res, open(f"{H}/lre.json", "w"), indent=1)
open(f"{H}/LRE.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
