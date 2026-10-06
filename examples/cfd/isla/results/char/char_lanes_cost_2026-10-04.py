"""CHAR-LANES cost (2026-10-04, #sec-nb-char-lanes-launched): per dataset task and architecture at the reference configurations, the
training peak device memory (GiB, maximum over the four GB300 ranks), the epoch wall time (s) and the parameter count, read from the
one-epoch smoke lanes of the final reference configurations (group cs, runs/cs_r_<task>_<arch>/metrics.jsonl, REF-80 verdict
2026-10-04: ISLA width 512 with 512 slices, GeoTransolver width 256, Transolver width 512 x 16 at 100k; the recipe's train_summary /
config rows) and, for DrivAerML Surface, from the one-epoch frontier lanes of results/char/ref80_frontier_2026-10-02.json. The
first-pass smoke lanes (cs_<task>_<arch>, DrivAerML-half references) are kept under cost_first_pass. Writes
results/char/char_lanes_cost_2026-10-04.json: cost[task][arch] = {peak_gib, epoch_seconds, n_parameters, source}.
usage: python char_lanes_cost_2026-10-04.py"""
import glob, json, os

R = os.path.dirname(os.path.abspath(__file__)); RES = os.path.dirname(R)
ARCH = {"isla": "ISLA", "gt": "GeoTransolver", "t": "Transolver"}
TASKS = {"vol": "DrivAerML Volume", "af": "AirfRANS", "suv": "Shift-SUV", "t2dz": "Tensile2D", "lug": "lug", "jeb": "DeepJEB++"}
FRONT = json.load(open(f"{R}/ref80_frontier_2026-10-02.json"))["candidates"]
REF = {"ISLA": "isla_w512_s512", "GeoTransolver": "gt_h256", "Transolver": "t_h512_l16_c100k"}
def read(fs):
    rows = [json.loads(l) for l in open(fs[0])]
    cfg = next((r for r in rows if r.get("phase") == "config"), {})
    ts = [r for r in rows if r.get("phase") in ("train_summary", "epoch") and ("peak_memory_gib" in r or "peak_memory_gib" in r.get("metrics", {}))]
    if not ts:
        return None
    last = ts[-1]; m = last if "peak_memory_gib" in last else last["metrics"]
    return {"peak_gib": m.get("peak_memory_gib"), "epoch_seconds": m.get("epoch_seconds"),
            "n_parameters": cfg.get("num_parameters"), "source": os.path.relpath(fs[0], RES)}


out = {"protocol": __doc__, "cost": {}, "cost_first_pass": {}}
for key, task in TASKS.items():
    out["cost"][task] = {}; out["cost_first_pass"][task] = {}
    for a, name in ARCH.items():
        fs = glob.glob(f"{RES}/campaigns_2026-10-02/char/cs/cs_r_{key}_{a}/metrics.jsonl")
        if fs and (rec := read(fs)):
            out["cost"][task][name] = rec
        fs = glob.glob(f"{RES}/campaigns_2026-10-02/char/cs/cs_{key}_{a}/metrics.jsonl") or glob.glob(f"{RES}/campaigns_2026-10-02/char/cs/cs_{'t2d' if key=='t2dz' else key}_{a}/metrics.jsonl")
        if fs and (rec := read(fs)):
            out["cost_first_pass"][task][name] = rec
out["cost"]["DrivAerML Surface"] = {n: {"peak_gib": FRONT[t]["train_peak_gib"], "epoch_seconds": FRONT[t]["epoch_seconds"],
                                        "n_parameters": FRONT[t]["n_parameters"], "source": "results/char/ref80_frontier_2026-10-02.json"} for n, t in REF.items()}
json.dump(out, open(f"{R}/char_lanes_cost_2026-10-04.json", "w"), indent=1)
for task, d in out["cost"].items():
    print(f"== {task}: " + "  ".join(f"{n} {v['peak_gib']:.1f} GiB / {v['epoch_seconds']:.0f} s / {v['n_parameters']/1e6:.0f}M" for n, v in d.items() if v["peak_gib"] is not None))
