"""Reduce the HILIFT-090 records to the numbers the book's HiLiftAeroML chapter reads (where ISLA trails).

Inputs, all in this folder, copied unmodified from scratch/isla-hilift090-2026-10-20/ of the main checkout:
  partA/percase_w512s512.csv     one row per validation case: seed-mean native relative L2 of every arm (reduce_a_w512s512.py)
  partA/tables_a_w512s512.md     the same reduction's tables, including the per-seed values of the ten largest ISLA-GT gaps
  partA/hl_medium_manifest.json  the train / val case lists of highlift_surface_medium
  partA_spatial/SPATIAL.md       the squared error split over surface regions (summarize_spatial.py)
Output: hilift_deficit_2026-10-20.json. Run: python3 reduce_deficit.py
"""
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

H = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(H / "partA" / "percase_w512s512.csv")))
for r in rows:
    r["aoa"] = int(r["aoa"]); r["post_stall"] = r["post_stall"] == "1"
    for k in ("isla_p", "gt_p", "t_p", "old_p", "isla_wss", "gt_wss", "t_wss", "old_wss"):
        r[k] = float(r[k])
n = len(rows)
mean = lambda k, rs=rows: sum(r[k] for r in rs) / len(rs)
ratio = lambda a, b, rs=rows: mean(a, rs) / mean(b, rs)
out = {"source": "HILIFT-090 (scratch/isla-hilift090-2026-10-20): highlift_surface_medium validation split, 180 cases, ISLA's "
                 "200,000-cell set, float32, epoch-500 checkpoints, native mesh; per-case relative L2, seed means over 42 and 43. "
                 "ISLA = a2r_hl_isla (the standing's checkpoints), GeoTransolver = udrv_hl_gt, Transolver = udrv_hl_t, raw-read "
                 "variant = r80_hl_isla_w512_s512 (the CHAR record's ISLA, raw slice reads, scored on another 200,000-cell draw).",
       "n_cases": n,
       "mean": {k: mean(k) for k in ("isla_p", "gt_p", "t_p", "old_p", "isla_wss", "gt_wss", "t_wss", "old_wss")}}
out["isla_over_gt"] = {"pressure": ratio("isla_p", "gt_p"), "wall_shear": ratio("isla_wss", "gt_wss")}
out["isla_over_t"] = {"pressure": ratio("isla_p", "t_p"), "wall_shear": ratio("isla_wss", "t_wss")}

# the ISLA - GeoTransolver pressure gap, case by case
gap = sorted(rows, key=lambda r: -(r["isla_p"] - r["gt_p"]))
tot = sum(r["isla_p"] - r["gt_p"] for r in rows)
top = gap[:10]; rest = gap[10:]
out["gt_gap"] = {
    "total_gap_mean": tot / n,
    "top10": [{"case": r["case"], "aoa": r["aoa"], "post_stall": r["post_stall"], "isla": r["isla_p"], "gt": r["gt_p"],
               "share": (r["isla_p"] - r["gt_p"]) / tot} for r in top],
    "top10_share": sum(r["isla_p"] - r["gt_p"] for r in top) / tot,
    "isla_over_gt_without_top10": {"pressure": ratio("isla_p", "gt_p", rest), "wall_shear": ratio("isla_wss", "gt_wss", rest)},
}
# per-case ISLA ÷ Transolver
out["t_offset"] = {"isla_worse_on": sum(r["isla_p"] > r["t_p"] for r in rows),
                   "by_aoa": {a: {"n": len(rs), "isla_over_t": ratio("isla_p", "t_p", rs), "isla_over_gt": ratio("isla_p", "gt_p", rs)}
                              for a, rs in sorted(((a, [r for r in rows if r["aoa"] == a]) for a in {r["aoa"] for r in rows}))}}
# ISLA ÷ its raw-read variant, by angle and by case
og = sorted(rows, key=lambda r: -(r["isla_p"] - r["old_p"]))
otot = sum(r["isla_p"] - r["old_p"] for r in rows)
by = defaultdict(list)
for r in rows:
    by[r["aoa"]].append(r)
out["raw_read"] = {
    "isla_over_raw": {"pressure": ratio("isla_p", "old_p"), "wall_shear": ratio("isla_wss", "old_wss")},
    "by_aoa": {a: {"n": len(rs), "isla_over_raw": ratio("isla_p", "old_p", rs), "share": sum(r["isla_p"] - r["old_p"] for r in rs) / otot}
               for a, rs in sorted(by.items())},
    "top3": [{"case": r["case"], "aoa": r["aoa"], "share": (r["isla_p"] - r["old_p"]) / otot} for r in og[:3]],
}

# per-seed values of the ten largest gaps (the table of tables_a_w512s512.md)
T = open(H / "partA" / "tables_a_w512s512.md").read()
seeds = {}
for line in T.split("top 10 ISLA-GT pressure-gap cases, per seed")[1].splitlines():
    m = re.match(r"\| (geo_\S+) \| (.+) \|$", line)
    if m:
        c = m.group(2).split(" | ")
        pair = lambda s: [float(x) for x in s.split("/")]
        seeds[m.group(1)] = {"ISLA": pair(c[0]), "raw_read": pair(c[1]), "GeoTransolver": pair(c[2]), "Transolver": pair(c[3])}
out["top10_per_seed"] = seeds

# within AoA >= 16: Spearman of the per-case ISLA ÷ GT pressure ratio with each case feature (the table of tables_a_w512s512.md)
rho = {}
for line in T.split("Spearman correlations within AoA>=16")[1].split("within AoA>=16, split at median")[0].splitlines():
    c = [x.strip() for x in line.strip().strip("|").split("|")]
    if len(c) >= 6 and c[0] not in ("feature",) and not c[0].startswith("---"):
        rho["|".join(c[:-5]).strip()] = float(c[-4])   # a feature name may hold "|" (|dAoA|)
out["spearman_isla_over_gt_aoa_ge_16"] = rho

# training coverage
M = json.load(open(H / "partA" / "hl_medium_manifest.json"))
aoa = lambda c: int(re.search(r"AoA_(\d+)", c).group(1)); geo = lambda c: re.search(r"(LHC\d+)", c).group(1)
tr_geo = {geo(c) for c in M["train"]}; va_geo = {geo(c) for c in M["val"]}
out["training"] = {"n_train": len(M["train"]), "n_val": len(M["val"]),
                   "train_by_aoa": dict(sorted(Counter(aoa(c) for c in M["train"]).items())),
                   "n_train_aoa_ge_16": sum(aoa(c) >= 16 for c in M["train"]),
                   "n_val_geometries": len(va_geo), "n_val_geometries_in_train": len(va_geo & tr_geo)}

# where on the surface the excess sits (SPATIAL.md: ISLA seed mean, GeoTransolver and Transolver seed 42)
S = open(H / "partA_spatial" / "SPATIAL.md").read()
sp = {}
for block in re.split(r"\n## ", S)[1:]:
    title, body = block.split("\n", 1)
    for line in body.splitlines():
        c = [x.strip() for x in line.strip("|").split("|")]
        if len(c) == 10 and c[0].startswith("all") and not c[1].startswith("**") and c[2] != "nan" and c[0] != "group":
            sp.setdefault(title, {})[c[1]] = {"area": float(c[2]), "isla_e2": float(c[3]) * 1e-4, "gt_e2": float(c[4]) * 1e-4,
                                              "t_e2": float(c[5]) * 1e-4, "share_of_isla_minus_gt": float(c[7]),
                                              "share_of_isla_minus_t": float(c[9])}
    m = re.search(r"all \(labeled, n=(\d+)\)", body)
    if m:
        sp.setdefault("_n_labeled", int(m.group(1)))
sp["_note"] = ("squared relative L2 per case split over surface regions, averaged over the cases; ISLA the mean of its two seeds, "
               "GeoTransolver and Transolver seed 42; cp bins by the true pressure coefficient; span bins y0..y4 are fifths of the "
               "semispan from the root; elements from the per-cell labels of the labeled validation cases")
out["spatial"] = sp
json.dump(out, open(H / "hilift_deficit_2026-10-20.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("isla_over_gt", "isla_over_t", "training")}, indent=1))
print(json.dumps(out["gt_gap"], indent=1)[:2500])
print(json.dumps(out["raw_read"], indent=1)[:1500])
print(sorted(sp))
