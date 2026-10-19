"""T-SCORESET (2026-10-19): per-case comparison of scoredset_check.py outputs (Transolver arm vs ISLA A2 arm).
Counts cases whose scored points + targets match, and cases whose boundary hashes also match. Writes scoredset_compare.json."""
import json, pathlib
d = pathlib.Path(__file__).parent / "scoredset"
pairs = [("DrivAerML Surface native", "dr_t_c200k", "dr_isla_a2r"), ("DrivAerML Surface native, U_inf_dir arm", "dr_tud_c200k", "dr_isla_a2r"),
         ("DrivAerML Surface area", "dra_t_c200k", "dra_isla_a2r"), ("DrivAerML Surface area, U_inf_dir arm", "dra_tud_c200k", "dra_isla_a2r"),
         ("DrivAerML Surface posed", "drp_t_c200k", "drp_isla_a2r"), ("DrivAerML Surface posed, U_inf_dir arm", "drp_tud_c200k", "drp_isla_a2r"),
         ("DrivAerML Surface native, Transolver at 100k (CHAR)", "dr_t_r80_c100k", "dr_isla_a2r"),
         ("SHIFT-SUV native", "suv_t_c200k", "suv_isla_a2r"), ("SHIFT-SUV area", "suva_t_c200k", "suva_isla_a2r"), ("SHIFT-SUV posed", "suvp_t_c200k", "suvp_isla_a2r"),
         ("DrivAerML Volume native", "vol_t_c200k", "vol_isla_a2r"), ("DrivAerML Volume posed", "volp_t_c200k", "volp_isla_a2r")]
def load(n):
    rows, summ = [], None
    for l in (d / f"{n}.jsonl").read_text().splitlines():
        r = json.loads(l)
        if r.get("phase") == "summary": summ = r
        elif "i" in r: rows.append(r)
    return rows, summ
out = []
for label, a, b in pairs:
    ra, sa = load(a); rb, sb = load(b)
    n = min(len(ra), len(rb))
    st = sum(ra[i]["idx"] == rb[i]["idx"] and ra[i]["points"] == rb[i]["points"] and ra[i]["targets"] == rb[i]["targets"] for i in range(n))
    bk = lambda r: {k: v for k, v in r.items() if k.startswith("bnd.")}
    bnd = sum(bk(ra[i]) == bk(rb[i]) for i in range(n))
    rec = {"set": label, "transolver": a, "isla_a2": b, "cases": n, "digest_t": sa["digest"], "digest_isla": sb["digest"],
           "points_targets_match": st, "boundaries_match": bnd,
           "bnd_keys_t": sorted(bk(ra[0])), "bnd_keys_isla": sorted(bk(rb[0])), "n_points": ra[0]["n_points"]}
    out.append(rec)
    print(f"{label:52s} n={n:3d} pts={rec['n_points']} digest {sa['digest'][:8]} vs {sb['digest'][:8]}  points+targets {st}/{n}  boundaries {bnd}/{n}  bnd_t={rec['bnd_keys_t']} bnd_isla={rec['bnd_keys_isla']}")
json.dump(out, open(d.parent / "scoredset_compare.json", "w"), indent=1)
