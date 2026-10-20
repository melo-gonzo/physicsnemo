"""STREAM-AF, DeepJEB++ part (2026-10-19): the H-RESID residual stream on DeepJEB++ surface, in the schema of
results/renderings_2026-10-18/stream_norms.json, as one compact json.

Source (H-RESID Instrument 1, `hresid_probe.py` md5 514f3c6b68380b1e4a0a5bcea36a18a1, run as `hresid_probe_af2.py` md5
8347c0a13b22dd7151d865a6eb4b1f53: recipe, manifest and checkpoint root from the environment, `model.compile_blocks=` dropped
from the lane extra; nothing else changed), one json per checkpoint:
  * LHR job 289830, 2026-10-19, `stream_jeb.sbatch`, output $T/stream_af_2026-10-19/out_jeb/<run>/e<N>.json.
    Local copy scratch/isla-stream-af-2026-10-19/out_jeb/<run>/e<N>.json.
Lanes: raw reads cl_r_jeb_isla_seed42/43 (AGA jobs 1017653/1017654, kernel/char/cl_lanes.tsv rows 72-73, code_v20 +
recipe_v8; pruned on AGA to epochs 39 and 40, copied read-only to LHR, md5 checked) and A2 jbn2_isla_seed42/43 (LHR,
kernel/deepjeb_bn/lanes.tsv, code_v20a2 = code_v20 plus the A2 reads, recipe_v8; the cl_r_jeb rows with only the code
changed). Both: DeepJEB++ surface `train_clean`, 40 epochs, lr 3.75e-4, 12 blocks, width 512, 512 slices, .venv-recipe; the
same four test_clean cases (sampler order) with cached draws (200,000 cells, the training count), bfloat16 forward.
The raw lanes have only the last two checkpoints, so the raw series is two points; A2 is probed at 1, 9, 19, 29, 39, 40.
Every number is the mean over the four cases' tokens (surface model: every token is a boundary token).

Per checkpoint the json keeps:
  stream_out      mean token norm of h after the last block (blocks[11].norm_out) -- the "mean token norm of h"
  embed_out       mean token norm of h entering block 0 (blocks[0].norm_in)
  norm_in[b]      mean token norm of h entering block b
  amp[b]          block b's amplification: mean over tokens of |h_out| / |h_in|
  amp_bcast[b]    the same for the slice (broadcast) sublayer step alone; amp_mlp[b] for the MLP step
  log10_prod_amp  sum over blocks of log10 amp[b]
  val_rel_l2      the probe's per-field relative L2 on the 4 cases (normalized space; a sanity read, not a score)
usage: python stream_reduce_jeb.py   (writes ../stream_norms_deepjebpp.json)
"""
import glob
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCR = "/home/psharpe/gh/physicsnemo/scratch"
SRC = {
    "job289830": f"{SCR}/isla-stream-af-2026-10-19/out_jeb",
}
R = "raw slice reads (pooling and broadcast read h)"
A2 = "pooling and broadcast read norm_assign(h) (the adopted reads)"
RUNS = {
    "cl_r_jeb_isla_seed42": dict(arm="raw", reads=R, lr=3.75e-4, protocol="surface", seed=42, code="code_v20"),
    "cl_r_jeb_isla_seed43": dict(arm="raw", reads=R, lr=3.75e-4, protocol="surface", seed=43, code="code_v20"),
    "jbn2_isla_seed42": dict(arm="A2", reads=A2, lr=3.75e-4, protocol="surface", seed=42, code="code_v20a2"),
    "jbn2_isla_seed43": dict(arm="A2", reads=A2, lr=3.75e-4, protocol="surface", seed=43, code="code_v20a2"),
}


def main():
    out = {"protocol": __doc__, "sources": SRC, "runs": {}}
    for run, meta in RUNS.items():
        series = {}
        for tag, root in SRC.items():
            for p in glob.glob(f"{root}/{run}/e*.json"):
                j = json.load(open(p))
                e = int(j["epoch"])
                assert j["run"] == run and j["hidden"] == 512 and j["precision"] == "bfloat16", p
                B = j["blocks"]
                nb = len(B)
                rec = {
                    "epoch": e, "source_job": tag, "source_file": os.path.relpath(p, SCR), "code": j["code"],
                    "case_ids": j["case_ids"], "n_tokens": j["n_tokens"],
                    "stream_out": B[str(nb - 1)]["norm_out"], "embed_out": B["0"]["norm_in"],
                    "norm_in": [B[str(b)]["norm_in"] for b in range(nb)],
                    "amp": [B[str(b)]["amp"] for b in range(nb)],
                    "amp_bcast": [B[str(b)]["amp_bcast"] for b in range(nb)],
                    "amp_mlp": [B[str(b)]["amp_mlp"] for b in range(nb)],
                    "log10_prod_amp": sum(math.log10(B[str(b)]["amp"]) for b in range(nb)),
                    "val_rel_l2": {k: sum(c[k] for c in j["rel_l2_normalized"]) / len(j["rel_l2_normalized"])
                                   for k in j["rel_l2_normalized"][0]},
                }
                assert e not in series or abs(series[e]["stream_out"] / rec["stream_out"] - 1) < 1e-6, (run, e)
                series.setdefault(e, rec)
        assert {r["code"] for r in series.values()} == {meta["code"]}, (run, {r["code"] for r in series.values()})
        out["runs"][run] = {**meta, "epochs": [series[e] for e in sorted(series)]}
        s = out["runs"][run]["epochs"]
        print(f"{run:26s} {meta['arm']:22s} epochs {[r['epoch'] for r in s]}")
        print("    stream_out " + " ".join(f"{r['epoch']}:{r['stream_out']:.3g}" for r in s))
    json.dump(out, open(os.path.join(os.path.dirname(HERE), "stream_norms_deepjebpp.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
