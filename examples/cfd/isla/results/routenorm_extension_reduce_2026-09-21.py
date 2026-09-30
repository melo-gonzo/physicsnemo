"""ROUTE-NORM extension reduction (notebook #sec-nb-routenorm-prereg addenda, #sec-nb-routenorm-depth-prereg): the
DrivAerML lanes beyond the original twelve — cosine ISLA at 1,024 and 1,536 wide (5e-4 and 1e-3), cosine ISLA at 384 wide
with 24 layers (5e-4) and 48 layers (2.5e-4), GeoTransolver at 2,048 wide (1e-3 and 3e-4), and the cosine interior pair at
1e-3 (200,000 cells and queries). Float32 at the 200,000-cell protocol, 48 validation cars, two seeds. Parameter counts are
copied from the CAP-435 reduction for configurations it measured; GeoTransolver-2,048 is estimated from the measured
256/512/1,024 ladder (parameters grow as width squared: 142.7M x 4 = 571M) and flagged.

usage: python routenorm_extension_reduce_2026-09-21.py <fetched p200k_fp32 dir> <cap435.json> <out.json>
"""
import glob, json, statistics as st, sys
from pathlib import Path

root, cap_path, out_path = (Path(a) for a in sys.argv[1:4])
CAP = json.load(open(cap_path))["configurations"]
FIELDS = ("pressure_l2", "velocity_l2", "nut_l2")
LANES = {
    "isla|cos|w1024|d12|lr5e-4": ("rn_cos_w1024_lr5e_4_c100k_seed{}", CAP["isla|w1024|d12|s256|lr1.875e-4"]["params"], False),
    "isla|cos|w1024|d12|lr1e-3": ("rn_cos_w1024_lr1e_3_c100k_seed{}", CAP["isla|w1024|d12|s256|lr1.875e-4"]["params"], False),
    "isla|cos|w1536|d12|lr5e-4": ("rn_cos_w1536_lr5e_4_c100k_seed{}", CAP["isla|w1536|d12|s256|lr1.25e-4"]["params"], False),
    "isla|cos|w1536|d12|lr1e-3": ("rn_cos_w1536_lr1e_3_c100k_seed{}", CAP["isla|w1536|d12|s256|lr1.25e-4"]["params"], False),
    "isla|cos|w384|d24|lr5e-4": ("rn_cos_w384_d24_lr5e_4_c100k_seed{}", CAP["isla|w384|d24|s256|lr5e-4"]["params"], False),
    "isla|cos|w384|d48|lr2.5e-4": ("rn_cos_w384_d48_lr2.5e_4_c100k_seed{}", CAP["isla|w384|d48|s256|lr5e-4"]["params"], False),
    "gt|w2048|d12|lr1e-3": ("rn_gt_w2048_lr1e_3_c100k_seed{}", 4 * CAP["gt|w1024|d12|s256|lr1e3"]["params"], True),
    "gt|w2048|d12|lr3e-4": ("rn_gt_w2048_lr3e_4_c100k_seed{}", 4 * CAP["gt|w1024|d12|s256|lr1e3"]["params"], True),
    "isla|cos|interior|w192|lr1e-3": ("rn_cos_vol_lr1e_3_c200k_seed{}", None, False),
}


def summary(run):
    ps = glob.glob(str(root / run / "**" / "metrics.jsonl"), recursive=True)
    if not ps:
        return None
    rows = [json.loads(l) for l in open(ps[0]) if l.strip()]
    s = [r for r in rows if r.get("phase") == "infer_summary"]
    return {f: s[0]["metrics"][f] for f in FIELDS if f in s[0]["metrics"]} if s else None


out = {"study": "ROUTE-NORM extension lanes on DrivAerML: cosine width 1,024/1,536, cosine depth 24/48 at width 384, GeoTransolver 2,048, cosine interior at 1e-3; float32 at 200,000 cells, 48 validation cars, two seeds",
       "configurations": {}}
for key, (pat, params, est) in LANES.items():
    runs = {pat.format(s): summary(pat.format(s)) for s in (42, 43)}
    c = {"runs": runs, "params": params, "params_estimated": est, "status": "complete" if all(runs.values()) else "pending"}
    if c["status"] == "complete":
        for f in FIELDS:
            vals = [r[f] for r in runs.values() if f in r]
            if vals:
                c[f + "_seeds"] = vals; c[f + "_mean"] = st.mean(vals)
    out["configurations"][key] = c
out_path.write_text(json.dumps(out, indent=2))
for k, c in out["configurations"].items():
    tag = f"{c['pressure_l2_mean']:.4f} ({' / '.join(f'{v:.4f}' for v in c['pressure_l2_seeds'])})" if c["status"] == "complete" else "pending"
    print(f"  {k:34s} params {str(c['params']):>11s}{'*' if c['params_estimated'] else ' '}  {tag}")
