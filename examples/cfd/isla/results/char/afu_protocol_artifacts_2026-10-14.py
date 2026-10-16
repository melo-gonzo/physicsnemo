"""AF-SCORE reduction (2026-10-14, #sec-nb-airfrans-uniform-prereg): the AF-UNIFORM reruns of chapter 9's AirfRANS reference
lanes (afu_r_{isla,gt,t}_seed42/43, uniform interior draw, recipe_v8 + code_v20) and of ISLA-DTT (afu_dtt_seed42/43, code_dtt16)
reduced exactly as protocol_artifacts_2026-10-03.py reduced the block-window lanes cl_r_af_* into results/char/airfrans.json:
float32, the three scorings of one checkpoint (native mesh, uniform point cloud = <dataset>_area, rotated + translated =
<dataset>_posed), per-case relative L2 averaged over the 200 full_test cases (infer_summary, else the per-sample rows), seed
means over seeds 42 and 43. Reads campaigns_2026-10-14/afu/protocols/<run>/<protocol>/metrics.jsonl, fetched from the LHR site's
$T/evals_afu/<run>/<protocol>/<run>/metrics.jsonl (kernel/airfrans_u/score.sbatch), e.g.
  for r in afu_r_isla afu_r_gt afu_r_t afu_dtt; do for s in 42 43; do for p in native area posed; do d=campaigns_2026-10-14/afu/protocols/${r}_seed$s/$p
    mkdir -p $d && scp lhr2:$T/evals_afu/${r}_seed$s/$p/${r}_seed$s/metrics.jsonl $d/; done; done; done
The old checkpoints scored on the same uniform twins (cl_r_af_*_seed42/43 from $T/runs_from_aga, af_tg1pf_seed42/43; LHR
$T/evals_afu_oldckpt/<run>/<protocol>/<run>/metrics.jsonl, same scorer) are fetched the same way into
campaigns_2026-10-14/afu/protocols_oldckpt/<run>/<protocol>/metrics.jsonl.
Writes results/char/airfrans_uniform.json in the schema of airfrans.json (protocols, fields, isla_arm, arms[arm][protocol][field]
= {mean, per_seed, n_cases, n_seeds}, complete), so the chapter's protocol_table("airfrans_uniform.json", ...) prints it, plus
keys protocol_table ignores: "controls" (ISLA-DTT, the read-only control, in the arm schema), "old_checkpoint_uniform" (the old
checkpoints on uniform draws, arm schema, ISLA-DTT included) and "decomposition" (per arm, protocol and field: old_block = the old
checkpoint on block windows, from airfrans.json or for ISLA-DTT af_tg1pf's records in campaigns_2026-10-14/afu/protocols_block;
old_uniform; new_uniform; scoring_effect = old_uniform / old_block; training_effect = new_uniform / old_uniform; null until scored).
Prints the standing as protocol_table does, the old checkpoints on uniform draws in the same layout, and the decomposition.
usage: python afu_protocol_artifacts_2026-10-14.py [--src DIR] [--oldckpt-src DIR] [--block-src DIR] [--out PATH]"""
import argparse, json, os, statistics as st

R = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(R)
PROTOCOLS = {"native": "native mesh", "area": "uniform point cloud", "posed": "rotated + translated"}
FIELDS = {"pressure_l2": "pressure", "velocity_l2": "velocity", "nut_l2": "eddy viscosity"}
SEEDS = (42, 43)
# arm label (airfrans.json's) -> run stem without _seedNN
ARMS = {"ISLA 512 with 512 slices": "afu_r_isla", "GeoTransolver 256": "afu_r_gt", "Transolver 512 x 16": "afu_r_t"}
ISLA_ARM = "ISLA 512 with 512 slices"
CONTROLS = {"ISLA-DTT (forest read at theta = 1)": ("afu_dtt", "af_tg1pf")}
# the old (block-window-trained) reference lane of each arm
OLD_STEMS = {"ISLA 512 with 512 slices": "cl_r_af_isla", "GeoTransolver 256": "cl_r_af_gt", "Transolver 512 x 16": "cl_r_af_t"}


def summary(path):
    """infer_summary metrics, else the mean of the per-sample rows (last row per sample id). Verbatim protocol_artifacts_2026-10-03.py."""
    if not os.path.exists(path):
        return None
    by_sample = {}
    for line in open(path):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("phase") == "infer_summary":
            return {"metrics": rec["metrics"], "n_cases": rec.get("num_samples")}
        if rec.get("phase") == "infer_step":
            by_sample[rec.get("sample_id", len(by_sample))] = rec["metrics"]
    if not by_sample:
        return None
    steps = list(by_sample.values()); keys = sorted({k for m in steps for k in m})
    return {"metrics": {k: st.mean(m[k] for m in steps if k in m) for k in keys}, "n_cases": len(steps)}


def arm(src, stem):
    """{protocol: {field: {mean, per_seed, n_cases, n_seeds}}} over the seeds on disk, and whether every seed and field is there."""
    d, complete = {}, True
    for pk in PROTOCOLS:
        per_seed = {s: r for s in SEEDS if (r := summary(f"{src}/{stem}_seed{s}/{pk}/metrics.jsonl"))}
        if not per_seed:
            complete = False; continue
        complete &= len(per_seed) == len(SEEDS)
        d[pk] = {f: {"mean": st.mean(r["metrics"][f] for r in per_seed.values()),
                     "per_seed": {s: r["metrics"].get(f) for s, r in per_seed.items()},
                     "n_cases": min(r["n_cases"] or 0 for r in per_seed.values()), "n_seeds": len(per_seed)}
                 for f in FIELDS if all(f in r["metrics"] for r in per_seed.values())}
        if len(d[pk]) < len(FIELDS):
            print(f"   {stem} {pk}: fields missing {[f for f in FIELDS if f not in d[pk]]}"); complete = False
    return d, complete and len(d) == len(PROTOCOLS)


def table(arms, isla):
    """The rows protocol_table() of book/09-airfrans.qmd prints (a ratio is baseline / ISLA)."""
    flab = " / ".join(FIELDS.values())
    out = ["| configuration | " + " | ".join(f"{pl}: {flab}" for pl in PROTOCOLS.values()) + " |", "|---|" + "---|" * len(PROTOCOLS)]
    for a, d in arms.items():
        one = any(d[pk][f]["n_seeds"] < 2 for pk in PROTOCOLS for f in FIELDS)
        out.append(f"| {a}{' (one seed)' if one else ''} | " + " | ".join(" / ".join(f"{d[pk][f]['mean']:.4f}" for f in FIELDS) for pk in PROTOCOLS) + " |")
        if a != ISLA_ARM:
            out.append(f"| {a} ÷ ISLA | " + " | ".join(" / ".join(f"{d[pk][f]['mean'] / isla[pk][f]['mean']:.2f}" for f in FIELDS) for pk in PROTOCOLS) + " |")
    return "\n".join(out)


def full(d):
    return len(d) == len(PROTOCOLS) and all(len(v) == len(FIELDS) for v in d.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=f"{RES}/campaigns_2026-10-14/afu/protocols")
    ap.add_argument("--oldckpt-src", default=f"{RES}/campaigns_2026-10-14/afu/protocols_oldckpt")
    ap.add_argument("--block-src", default=f"{RES}/campaigns_2026-10-14/afu/protocols_block")
    ap.add_argument("--out", default=f"{R}/airfrans_uniform.json")
    a = ap.parse_args()
    old = json.load(open(f"{R}/airfrans.json"))
    out = {"protocol": __doc__, "protocols": PROTOCOLS, "fields": FIELDS, "isla_arm": ISLA_ARM, "arms": {}, "controls": {},
           "old_checkpoint_uniform": {}, "decomposition": {}}
    complete = True
    ### new checkpoint, uniform draws: the standing (arms) and the ISLA-DTT control
    for label, stem in ARMS.items():
        d, ok = arm(a.src, stem); complete &= ok
        if full(d):
            out["arms"][label] = d
    for label, (stem, _) in CONTROLS.items():
        d, ok = arm(a.src, stem); complete &= ok
        if full(d):
            out["controls"][label] = d
    ### old checkpoint: block windows (the reference records) and uniform draws (the old checkpoints on the uniform twins)
    old_block, old_stems = {}, {**{label: OLD_STEMS[label] for label in ARMS}, **{label: bs for label, (_, bs) in CONTROLS.items()}}
    for label in ARMS:
        old_block[label] = old["arms"][label]
    for label, (_, block_stem) in CONTROLS.items():
        b, _ = arm(a.block_src, block_stem)
        if full(b):
            old_block[label] = b
    for label, stem in old_stems.items():
        d, ok = arm(a.oldckpt_src, stem); complete &= ok
        if full(d):
            out["old_checkpoint_uniform"][label] = d
    new_all = {**out["arms"], **out["controls"]}
    for label in old_stems:
        cols = {"old_block": old_block.get(label), "old_uniform": out["old_checkpoint_uniform"].get(label), "new_uniform": new_all.get(label)}
        rec = {}
        for pk in PROTOCOLS:
            rec[pk] = {}
            for f in FIELDS:
                v = {k: (c[pk][f]["mean"] if c else None) for k, c in cols.items()}
                v["scoring_effect"] = v["old_uniform"] / v["old_block"] if v["old_uniform"] is not None and v["old_block"] is not None else None
                v["training_effect"] = v["new_uniform"] / v["old_uniform"] if v["new_uniform"] is not None and v["old_uniform"] is not None else None
                rec[pk][f] = v
        out["decomposition"][label] = rec
    if not out["arms"] and not out["old_checkpoint_uniform"]:
        print("== airfrans_uniform: nothing scored yet; nothing written"); return
    out["complete"] = complete
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"== {a.out} ({'complete' if complete else 'partial'}); arms {list(out['arms'])}, controls {list(out['controls'])}, "
          f"old checkpoints on uniform draws {list(out['old_checkpoint_uniform'])}")
    if ISLA_ARM in out["arms"]:
        print("\nThe standing, uniform interior (the rows protocol_table prints):\n")
        print(table(out["arms"], out["arms"][ISLA_ARM]))
        if out["controls"]:
            print("\n" + table({ISLA_ARM: out["arms"][ISLA_ARM], **out["controls"]}, out["arms"][ISLA_ARM]))
    if ISLA_ARM in out["old_checkpoint_uniform"]:
        print("\nThe old checkpoints (block-window training) on uniform draws, in the standing's layout:\n")
        print(table(out["old_checkpoint_uniform"], out["old_checkpoint_uniform"][ISLA_ARM]))
    fmt = lambda x, p: "–" if x is None else f"{x:.{p}f}"
    print("\nScoring and training effects, seed means, " + " / ".join(FIELDS.values()) + "; scoring effect = old checkpoint uniform ÷ "
          "block, training effect = new checkpoint uniform ÷ old checkpoint uniform:\n")
    print("| configuration | protocol | old ckpt, block | old ckpt, uniform | new ckpt, uniform | scoring effect | training effect |\n|---|---|---|---|---|---|---|")
    for label, rec in out["decomposition"].items():
        for pk, fv in rec.items():
            print(f"| {label} | {PROTOCOLS[pk]} | " + " | ".join(" / ".join(fmt(fv[f][k], p) for f in FIELDS) for k, p in
                  (("old_block", 4), ("old_uniform", 4), ("new_uniform", 4), ("scoring_effect", 2), ("training_effect", 2))) + " |")


if __name__ == "__main__":
    main()
