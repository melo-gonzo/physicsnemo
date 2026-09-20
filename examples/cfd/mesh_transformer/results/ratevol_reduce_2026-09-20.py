"""RATE-VOL reduction (prereg #sec-nb-ratevol-prereg): the DrivAerML interior row at the 200,000-cell protocol.

ISLA's interior configuration (200,000 boundary cells and 200,000 interior queries per case, 435 training cars) rose to
the mean-field loss plateau at learning rate 1e-3 on both seeds; the same configuration was retrained at 5e-4 and 3e-4.
GeoTransolver-volume (unit direction, 200,000 cells and queries) is the comparator. All values float32 on the 48
validation cars at 200,000 interior queries, two seeds.

Bars: a rate is clean if neither seed's validation loss visits the plateau (below 0.01 after epoch 20); the row is written
with ISLA at the higher clean rate; pressure and velocity ratios (GeoTransolver-volume / ISLA) within 5% of their
10,000-query values (0.93 and 1.37) mean the interior standing survives the protocol; a larger swing is reported as the
budget's effect.

usage: python ratevol_reduce_2026-09-20.py <fetched p200k dir> <out.json>
"""
import glob, json, statistics as st, sys
from pathlib import Path

root, out_path = Path(sys.argv[1]), Path(sys.argv[2])
FIELDS = ("pressure_l2", "velocity_l2", "nut_l2")
ARMS = {"isla_lr1e3": "p200k_dr_isla_vol_seed{}", "isla_lr5e4": "p200k_dr_isla_vol_lr5e4_seed{}",
        "isla_lr3e4": "p200k_dr_isla_vol_lr3e4_seed{}", "gt_vol_unit": "p200k_dr_gt_unit_vol_seed{}"}
TENK = {"gt_vol_unit": {"pressure_l2": 0.0513, "velocity_l2": 0.107, "nut_l2": 0.0879},  # results/relint2_reduction_2026-09-14.json
        "isla": {"pressure_l2": 0.055, "velocity_l2": 0.0782, "nut_l2": 0.1005}}


def summary(run):
    ps = glob.glob(str(root / run / "**" / "metrics.jsonl"), recursive=True)
    if not ps:
        return None
    rows = [json.loads(l) for l in open(ps[0]) if l.strip()]
    s = [r for r in rows if r.get("phase") == "infer_summary"]
    return {f: s[0]["metrics"][f] for f in FIELDS} if s else None


arms = {}
for arm, pat in ARMS.items():
    runs = {pat.format(s): summary(pat.format(s)) for s in (42, 43)}
    a = {"runs": runs, "status": "complete" if all(runs.values()) else "pending"}
    if a["status"] == "complete":
        for f in FIELDS:
            vals = [runs[r][f] for r in runs]
            a[f] = {"seeds": vals, "mean": st.mean(vals)}
    arms[arm] = a

out = {"study": "RATE-VOL: DrivAerML interior at the 200,000-cell protocol; ISLA at 5e-4 (row) and 3e-4 (rate control) against the 1e-3 pair on the mean-field plateau; GeoTransolver-volume comparator; float32, 48 validation cars, 200,000 interior queries, two seeds",
       "arms": arms, "row_rate": "isla_lr5e4", "rate_control": "isla_lr3e4", "ratios_gt_vol_over_isla": {}, "tenk_ratios": {}, "bars": {}}
for f in FIELDS:
    out["tenk_ratios"][f] = TENK["gt_vol_unit"][f] / TENK["isla"][f]
    for arm in ("isla_lr5e4", "isla_lr3e4", "isla_lr1e3"):
        if arms[arm]["status"] == "complete":
            out["ratios_gt_vol_over_isla"].setdefault(arm, {})[f] = arms["gt_vol_unit"][f]["mean"] / arms[arm][f]["mean"]
row = out["ratios_gt_vol_over_isla"].get("isla_lr5e4", {})
for f in ("pressure_l2", "velocity_l2"):
    if f in row:
        d = row[f] / out["tenk_ratios"][f] - 1
        out["bars"][f] = {"ratio_200k": row[f], "ratio_10k": out["tenk_ratios"][f], "relative_change": d,
                          "verdict": "standing survives (within 5%)" if abs(d) <= 0.05 else "budget effect (swing above 5%)"}
if arms["isla_lr5e4"]["status"] == arms["isla_lr3e4"]["status"] == "complete":
    out["bars"]["rate_control"] = {f: arms["isla_lr3e4"][f]["mean"] / arms["isla_lr5e4"][f]["mean"] for f in FIELDS}
out_path.write_text(json.dumps(out, indent=2))
for arm, a in arms.items():
    tag = "  ".join(f"{f[:-3]} {a[f]['mean']:.4f} ({' / '.join(f'{v:.4f}' for v in a[f]['seeds'])})" for f in FIELDS) if a["status"] == "complete" else "pending"
    print(f"  {arm:12s} {tag}")
print(json.dumps({k: out[k] for k in ("ratios_gt_vol_over_isla", "tenk_ratios", "bars")}, indent=1))
