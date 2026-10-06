"""REF-80 memory frontier (CHAR, 2026-10-02, #sec-nb-char-prereg): one real epoch of every candidate configuration of ISLA,
GeoTransolver and Transolver on DrivAerML Surface at the reporting protocol (lanes rf1_*, code_v18 + recipe_v8, four GB300 ranks,
batch one, bfloat16), reading the recipe's epoch summary: peak device memory of the training epoch (maximum over ranks, GiB), the
epoch's wall time, the parameter count. A candidate is on the frontier when its training peak is at most 80 GiB. The `_nc` lanes
run with the recipe's whole-model torch.compile off (`compile=false`) so that ISLA's per-block compile engages; the others carry
the recipe default. Reads campaigns_2026-10-02/char/rf1/<lane>/metrics.jsonl. usage: python ref80_frontier_reduce_2026-10-02.py"""
import glob, json, os

R = os.path.dirname(os.path.abspath(__file__))
BUDGET_GIB = 80.0
ARCH = {"isla": "ISLA", "gt": "GeoTransolver", "t": "Transolver"}
out = {"protocol": __doc__, "budget_gib": BUDGET_GIB, "candidates": {}}
for f in sorted(glob.glob(f"{R}/../campaigns_2026-10-02/char/rf1/rf1_*/metrics.jsonl")):
    lane = os.path.basename(os.path.dirname(f)); tag = lane[4:]
    rows = [json.loads(l) for l in open(f)]
    cfg = next((r for r in rows if r.get("phase") == "config"), {})
    tr = [r for r in rows if r.get("phase") == "train_summary"]; va = [r for r in rows if r.get("phase") == "val_summary"]
    if not tr: continue
    t = tr[-1]; v = va[-1] if va else {}
    arch = ARCH[tag.split("_")[0]]
    params = cfg.get("params", {}); overrides = {k: params[k] for k in params if k.startswith("model.") or k == "sampling_resolution"}
    out["candidates"][tag] = {"architecture": arch, "lane": lane, "n_parameters": cfg.get("num_parameters"),
        "train_peak_gib": t.get("peak_memory_gib"), "epoch_seconds": t.get("epoch_seconds"), "val_peak_gib": v.get("peak_memory_gib"),
        "compile_whole_model": not tag.endswith("_nc"), "on_frontier": (t.get("peak_memory_gib") or 1e9) <= BUDGET_GIB}
out["frontier"] = {a: sorted([k for k, c in out["candidates"].items() if c["architecture"] == a and c["on_frontier"]],
                             key=lambda k: -out["candidates"][k]["train_peak_gib"]) for a in ARCH.values()}
json.dump(out, open(f"{R}/ref80_frontier_2026-10-02.json", "w"), indent=1)
for a in ARCH.values():
    print(f"== {a}")
    for k, c in sorted(out["candidates"].items(), key=lambda kv: kv[1]["train_peak_gib"] or 0):
        if c["architecture"] != a: continue
        print(f"   {k:20s} {c['n_parameters']/1e6:7.1f}M  peak {c['train_peak_gib']:6.1f} GiB  epoch {c['epoch_seconds']:5.0f} s  {'FRONTIER' if c['on_frontier'] else 'over'}")
