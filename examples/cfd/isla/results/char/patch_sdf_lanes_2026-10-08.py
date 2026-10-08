"""PATCH-SDF lanes: the native / posed scores and the AirfRANS resampling probe of the two `sp_af_patch_*` lanes
(query_geometry="patch", code_v25), against the nearest-form reference lanes and the kernel-form lanes of SOFT-GEO
(notebook #sec-nb-patch-sdf-prereg; verdict entry follows).

Inputs (fetched from LHR into <fetched_dir>):
  protocols/<run>/<protocol>/metrics.jsonl      the CHAR protocol scorer's float32 metrics (native, posed), infer_summary rows
  af_geo/<run>.json                              kernel/char/af_geo_sp/<run>/summary.json: the resampling probe (subu, suba, refu, reff
                                                 and their *_natgeo swaps; fraction_of_log_excess_removed_by_native_geometry)
  af_geo_le/<run>.json                           kernel/char/af_geo_sp_le/<run>/summary.json: the chordwise split of the swap
References read from the repo: results/char/airfrans.json (the CHAR ISLA reference arm = nearest form, native and posed per seed)
and results/char/soft_geo_lanes_2026-10-07.json (the relaunched kernel lanes' shares).

Bars (the SOFT-GEO gate as pre-registered): (i) posed / native within 1.005 on every field; (ii) the built-in channel's share of the
log excess under the 400-segment coarsening below 0.1 (both laws); (iii) native error within the nearest reference's seed range or
better; re-panelization (refu, reff) within 2 percent.

usage: python patch_sdf_lanes_2026-10-08.py <fetched_dir> <out.json>
"""

from __future__ import annotations

import glob
import json
import statistics as st
import sys
from pathlib import Path

fetched, out_path = Path(sys.argv[1]), Path(sys.argv[2])
RES = Path(__file__).resolve().parent
RUNS = ("sp_af_patch_seed42", "sp_af_patch_seed43")
FIELDS = ("pressure_l2", "velocity_l2", "nut_l2")


def protocol_score(run: str, proto: str) -> dict | None:
    ps = glob.glob(str(fetched / "protocols" / run / proto / "**" / "metrics.jsonl"), recursive=True)
    if not ps:
        return None
    rows = [json.loads(l) for l in open(ps[0]) if l.strip()]
    s = [r for r in rows if r.get("phase") == "infer_summary"]
    return {f: s[0]["metrics"][f] for f in FIELDS} if s else None


native_posed = {}
for run in RUNS:
    nat, pos = protocol_score(run, "native"), protocol_score(run, "posed")
    native_posed[run] = {
        "native": nat,
        "posed": pos,
        "posed_over_native": {f: pos[f] / nat[f] for f in FIELDS} if nat and pos else None,
    }

af_geo, af_geo_le = {}, {}
for run in RUNS:
    p = fetched / "af_geo" / f"{run}.json"
    if p.exists():
        S = json.load(open(p))
        af_geo[run] = {
            "native_means": {f: S["means"]["native"][f] for f in FIELDS},
            "ratio_to_native": {k: {f: v[f] for f in FIELDS} for k, v in S["ratio_to_native"].items()},
            "geometry_share_of_log_excess": S["fraction_of_log_excess_removed_by_native_geometry"],
            "tokens": {k: S["shift_means_all"][k] for k in ("n_tokens_native", "subu_n_tokens", "subu_frac_nearest_token_changed") if k in S["shift_means_all"]},
        }
    p = fetched / "af_geo_le" / f"{run}.json"
    if p.exists():
        S = json.load(open(p))
        af_geo_le[run] = {
            "ratio_to_native": {k: {f: v[f] for f in FIELDS} for k, v in S["ratio_to_native"].items()},
            "fraction_of_log_excess_removed_by_native_geometry": S["fraction_of_log_excess_removed_by_native_geometry"],
        }

### references
AF = json.load(open(RES / "airfrans.json"))
isla_arm = AF["arms"][AF["isla_arm"]]
nearest_ref = {pk: {f: {"mean": isla_arm[pk][f]["mean"], "per_seed": isla_arm[pk][f]["per_seed"]} for f in FIELDS} for pk in ("native", "posed")}
SG = json.load(open(RES / "soft_geo_lanes_2026-10-07.json"))["relaunch_code_v23"]
kernel_shares = {run: SG["af_geo"][run]["geometry_share_of_log_excess"] for run in SG["af_geo"]}
nearest_shares = {run: v["geometry_share_of_log_excess"] for run, v in SG.get("nearest_af_geo", {}).items()} if "nearest_af_geo" in SG else None

### bars
bars = {}
pon = [native_posed[r]["posed_over_native"] for r in RUNS if native_posed[r]["posed_over_native"]]
if pon:
    worst = max(abs(v[f] - 1) for v in pon for f in FIELDS)
    bars["i_posed_within_1.005"] = {"pass": worst <= 0.005, "max_abs_deviation": worst,
                                    "per_run": {r: native_posed[r]["posed_over_native"] for r in RUNS}}
if af_geo:
    shares = {r: {law: af_geo[r]["geometry_share_of_log_excess"][law] for law in ("subu", "suba")} for r in af_geo}
    flat = [v[f] for r in shares for law in shares[r] for v in [shares[r][law]] for f in FIELDS]
    bars["ii_coarsening_geometry_share_below_0.1"] = {"pass": max(flat) < 0.1, "range": [min(flat), max(flat)], "per_run": shares,
                                                        "kernel_lanes_for_comparison": kernel_shares}
    rep = [af_geo[r]["ratio_to_native"][k][f] for r in af_geo for k in ("refu", "reff") for f in FIELDS]
    bars["re_panelization_within_2pct"] = {"pass": max(abs(x - 1) for x in rep) <= 0.02, "range": [min(rep), max(rep)]}
nat = [native_posed[r]["native"] for r in RUNS if native_posed[r]["native"]]
if nat:
    verdict = {}
    for f in FIELDS:
        lo, hi = min(nearest_ref["native"][f]["per_seed"].values()), max(nearest_ref["native"][f]["per_seed"].values())
        m = st.mean(v[f] for v in nat)
        verdict[f] = {"patch_mean": m, "patch_seeds": [v[f] for v in nat], "nearest_seed_range": [lo, hi],
                      "pass": m <= hi, "ratio_to_nearest_mean": m / nearest_ref["native"][f]["mean"]}
    bars["iii_native_within_nearest_reference_seed_range"] = {"pass": all(v["pass"] for v in verdict.values()), "per_field": verdict}

out = {
    "protocol": __doc__,
    "runs": list(RUNS),
    "native_posed": native_posed,
    "nearest_reference": nearest_ref,
    "af_geo": af_geo,
    "af_geo_le": af_geo_le,
    "bars": bars,
}
json.dump(out, open(out_path, "w"), indent=1)
print(json.dumps(bars, indent=1, default=str)[:4000])
