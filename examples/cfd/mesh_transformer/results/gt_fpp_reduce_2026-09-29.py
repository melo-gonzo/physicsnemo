"""GT-FPP reduction (2026-09-29, #sec-nb-proto-cond-prereg): GeoTransolver with the released FLARE++ backend (attention_type=GALE_FPP,
campaign gf on code_v11) against the physics-attention GeoTransolver references of the book, per flagship benchmark: DrivAerML Surface
435 cars at 200k (gf_dr_gt_fpp_lr3e3 vs p200k_dr_gt_unit_lr3e3; gf_dr_gt_fpp_lr1e3 vs p200k_dr_gt_unit; results/p200k_trio_reduction_2026-09-19.json),
HiLift 510 wings width 512 at 100k (gf_hl_gt512_fpp_medium_c100k vs shl_med_gt_w512_c100k; results/scalehl_2026-09-26.json), AirfRANS interior
(gf_af_gt_vol_fpp vs af_gt_vol_lr1e3; results/airfrans_reduction_2026-09-22.json), OLBC Poisson 2,048 width 512 (gf_poisson_gt512_fpp_n2048 vs
ol_poisson_gt512_n2048; results/olbc_2026-09-22.json). Readout: float32 per-case relative L2 per field, two seeds; ratio FLARE++ /
physics-attention on seed means and per seed. Reading: FLARE++ better by more than 3% on a dataset's headline field replaces the
GeoTransolver baseline there (the book quotes both). usage: python gt_fpp_reduce_2026-09-29.py"""
import glob, json, os, statistics as st

R = os.path.dirname(os.path.abspath(__file__))


def per_case(run):
    fs = glob.glob(f"{R}/campaigns_2026-09-29/gf/metrics/{run}/**/metrics.jsonl", recursive=True)
    if not fs:
        return None
    rows = [json.loads(l)["metrics"] for l in open(sorted(fs)[0]) if '"infer_step"' in l]
    return {k: st.mean(r[k] for r in rows) for k in rows[0] if k.endswith("_l2")}


trio = json.load(open(f"{R}/p200k_trio_reduction_2026-09-19.json"))["arms"]
shl = json.load(open(f"{R}/scalehl_2026-09-26.json"))["arms"]["gt_512_c100k"]["runs"]
afr = json.load(open(f"{R}/airfrans_reduction_2026-09-22.json"))["arms"]["gt_volume_1e-3"]["runs"]
ol = json.load(open(f"{R}/olbc_2026-09-22.json"))["lanes"]
REFS = {
    "drivaer_435_lr3e3": ({s: trio["gt_unit_lr3e3"]["per_run"][f"p200k_dr_gt_unit_lr3e3_seed{s}"] for s in ("42", "43")}, "gf_dr_gt_fpp_lr3e3_seed{s}", "pressure_l2"),
    "drivaer_435_lr1e3": ({s: trio["gt_unit_lr1e3"]["per_run"][f"p200k_dr_gt_unit_seed{s}"] for s in ("42", "43")}, "gf_dr_gt_fpp_lr1e3_seed{s}", "pressure_l2"),
    "hilift_510_w512": ({s: {k: v for k, v in shl[f"shl_med_gt_w512_c100k_seed{s}"].items() if k.endswith("_l2")} for s in ("42", "43")}, "gf_hl_gt512_fpp_medium_c100k_seed{s}", "pressure_l2"),
    "airfrans_interior": ({s: afr[f"af_gt_vol_lr1e3_seed{s}"] for s in ("42", "43")}, "gf_af_gt_vol_fpp_seed{s}", "pressure_l2"),
    "poisson_2048_w512": ({s: {"u_l2": ol[f"ol_poisson_gt512_n2048_seed{s}"]["fields"]["u"]["mean"]} for s in ("42", "43")}, "gf_poisson_gt512_fpp_n2048_seed{s}", "u_l2"),
}
out = {"protocol": __doc__, "datasets": {}}
for name, (ref, pat, head) in REFS.items():
    fpp = {s: per_case(pat.format(s=s)) for s in ("42", "43")}; fpp = {s: v for s, v in fpp.items() if v}
    if not fpp:
        out["datasets"][name] = {"status": "not scored", "reference": {f: st.mean(ref[s][f] for s in ref) for f in ref["42"]}}; continue
    fields = [f for f in ref["42"] if all(f in v for v in fpp.values())]
    rm = {f: st.mean(ref[s][f] for s in ref) for f in fields}; fm = {f: st.mean(fpp[s][f] for s in fpp) for f in fields}
    ratio = {f: fm[f] / rm[f] for f in fields}
    ps = {s: {f: fpp[s][f] / ref[s][f] for f in fields} for s in fpp if s in ref}
    r = ratio[head]
    out["datasets"][name] = {"fields": fields, "headline": head, "physics_attention": rm, "flare_pp": fm, "ratio": ratio, "per_seed": {"physics_attention": ref, "flare_pp": fpp, "ratio": ps},
                             "n_seeds": len(fpp), "interim": len(fpp) < 2,
                             "reading": "FLARE++ stronger (replaces baseline)" if r < 0.97 else "physics attention stronger" if r > 1.03 else "within 3%"}
json.dump(out, open(f"{R}/gt_fpp_2026-09-29.json", "w"), indent=1)
for name, d in out["datasets"].items():
    if "ratio" in d:
        print(f"== {name}: " + "  ".join(f"{f} phys {d['physics_attention'][f]:.4f} fpp {d['flare_pp'][f]:.4f} r {d['ratio'][f]:.3f}" for f in d["fields"]) + f"  -> {d['reading']}{' (interim)' if d['interim'] else ''}")
    else:
        print(f"== {name}: not scored; reference {{{', '.join(f'{k}: {v:.4f}' for k, v in d['reference'].items())}}}")
