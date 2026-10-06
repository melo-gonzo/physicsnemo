"""JEB-PLATEAU (2026-10-05, #sec-nb-char-deepjeb-plateau): per-epoch training loss and validation von Mises / displacement relative L2
(bfloat16 training, every epoch for the diagnostic lanes, every second epoch for the campaign lanes) of the ISLA lanes on
DeepJEB++, from runs/<run>/metrics.jsonl (train_summary / val_summary rows; copied under campaigns_2026-10-02/char/jd/).
Diagnostic lanes (5 epochs, one GPU node, seed 42 unless named): cs_jd_rate (width 512, 512 slices, lr 1.875e-4, code_v20),
cs_jd_r1 (384 / 256, 5e-4, code_v20), cs_jd_v17 (512 / 512, 3.75e-4, code_v17 = static prototypes), cs_jd_v17r1 (384 / 256, 5e-4,
code_v17); second-seed twins cs_jd_*_s43 and the code_v20 reference control cs_jd_v20ref_s43 when present. Campaign lanes: cl_jeb_isla
(768, 3.75e-4, code_v20), cl_r_jeb_isla (512 / 512, 3.75e-4, code_v20), cl_jeb_gt / cl_jeb_t (baselines). Writes
results/char/jd_plateau_2026-10-05.json: lanes[run] = {label, code, config, lr, train_loss[(epoch, loss)], val[(epoch, vm, disp)],
plateau: bool (final validation von Mises > 0.8)}.
usage: python jd_plateau_2026-10-05.py"""
import glob, json, os

R = os.path.dirname(os.path.abspath(__file__)); RES = os.path.dirname(R)
SRC = f"{RES}/campaigns_2026-10-02/char/jd"
LANES = {
    "cs_jd_rate": ("ISLA 512 / 512 slices, 1.875e-4, sample-conditioned prototypes (code_v20)", "code_v20", "512/512", 1.875e-4),
    "cs_jd_rate_s43": ("same, seed 43", "code_v20", "512/512", 1.875e-4),
    "cs_jd_r1": ("ISLA 384 / 256 slices, 5e-4, sample-conditioned prototypes (code_v20)", "code_v20", "384/256", 5e-4),
    "cs_jd_r1_s43": ("same, seed 43", "code_v20", "384/256", 5e-4),
    "cs_jd_v20ref_s43": ("ISLA 512 / 512 slices, 3.75e-4, sample-conditioned prototypes (code_v20), seed 43", "code_v20", "512/512", 3.75e-4),
    "cs_jd_v17": ("ISLA 512 / 512 slices, 3.75e-4, static prototypes (code_v17)", "code_v17", "512/512", 3.75e-4),
    "cs_jd_v17_s43": ("same, seed 43", "code_v17", "512/512", 3.75e-4),
    "cs_jd_v17r1": ("ISLA 384 / 256 slices, 5e-4, static prototypes (code_v17)", "code_v17", "384/256", 5e-4),
    "cl_jeb_isla_seed42": ("campaign first pass: ISLA 768 / 256 slices, 3.75e-4 (code_v20)", "code_v20", "768/256", 3.75e-4),
    "cl_jeb_isla_seed43": ("same, seed 43", "code_v20", "768/256", 3.75e-4),
    "cl_r_jeb_isla_seed42": ("campaign final reference: ISLA 512 / 512 slices, 3.75e-4 (code_v20)", "code_v20", "512/512", 3.75e-4),
    "cl_r_jeb_isla_seed43": ("same, seed 43", "code_v20", "512/512", 3.75e-4),
    "cl_jeb_gt_seed42": ("GeoTransolver 1,024, 1e-3 (baseline control)", "code_v20", "gt1024", 1e-3),
    "cl_jeb_t_seed42": ("Transolver 1,024, 1e-3 (baseline control)", "code_v20", "t1024", 1e-3),
}
out = {"protocol": __doc__, "lanes": {}}
for run, (label, code, cfg, lr) in LANES.items():
    fs = glob.glob(f"{SRC}/{run}/metrics.jsonl") + glob.glob(f"{SRC}/{run}/summaries.jsonl")
    if not fs:
        continue
    rows = [json.loads(l) for l in open(fs[0]) if l.strip()]
    tr = [(r["epoch"], r["loss"]) for r in rows if r.get("phase") == "train_summary"]
    va = [(r["epoch"], r["von_mises_l2"], r["displacement_l2"]) for r in rows if r.get("phase") == "val_summary"]
    if not va:
        continue
    out["lanes"][run] = {"label": label, "code": code, "config": cfg, "lr": lr, "train_loss": tr, "val": va,
                         "plateau": va[-1][1] > 0.8, "epochs": len(tr)}
    print(f"{run:22s} {label[:58]:58s} val vm " + " ".join(f"e{e}:{v:.2f}" for e, v, _ in va[:3] + va[-2:]) + f"  train e0 {tr[0][1]:.3f} -> last {tr[-1][1]:.3f}  {'PLATEAU' if va[-1][1] > 0.8 else 'trains'}")
json.dump(out, open(f"{R}/jd_plateau_2026-10-05.json", "w"), indent=1)
