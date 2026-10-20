"""STREAM-AF (2026-10-19): the residual-stream growth behind H-RESID re-measured on AirfRANS, in the schema of
results/renderings_2026-10-18/stream_norms.json (AeroZoo, being retired), as one compact json.

Source (H-RESID Instrument 1, `hresid_probe.py` md5 514f3c6b68380b1e4a0a5bcea36a18a1, run as `hresid_probe_af.py` md5
8d7cfc701c58ecb7194e80717c8d5766: the recipe and lane manifest come from the environment, and `model.compile_blocks=` is
dropped from the lane's extra so the probe runs eager as it did on AeroZoo; nothing else changed), one json per checkpoint:
  * LHR jobs 288480 (afu_r_isla_seed42/43, afu_a2s_isla_seed42/43) and 288481 (afu_bn2_isla_seed42/43), 2026-10-19,
    `stream_af.sbatch`, output $T/stream_af_2026-10-19/out/<run>/e<N>.json.
    Local copy scratch/isla-stream-af-2026-10-19/out/<run>/e<N>.json.
Each lane on its own snapshots (its train.log and kernel/airfrans_u/lanes.tsv): raw code_v20 + recipe_v8; A2 (bn2) code_v20a2
(code_v20 plus the A2 reads) + recipe_v8; A2 shipped (a2s) code_a2s + recipe_v11 (trained with compile_blocks=true, probed
eager). Training venv .venv-recipe. All runs: AirfRANS uvolume 2-D (`full_train`, 500 epochs, lr 3.75e-4), 12 blocks,
width 512, 512 slices, the same four full_test cases (sampler indices 9, 10, 12, 14) with cached draws (100,000 cells, the
lanes' training count), bfloat16 forward as in training. Epochs 1-249 as the AeroZoo A2 series, extended to 499 over the
500-epoch schedule. Every number is the mean over the four cases' tokens.

Per checkpoint the json keeps:
  stream_out      mean token norm of h after the last block (blocks[11].norm_out) -- the "mean token norm of h"
  embed_out       mean token norm of h entering block 0 (blocks[0].norm_in)
  norm_in[b]      mean token norm of h entering block b
  amp[b]          block b's amplification: mean over tokens of |h_out| / |h_in|
  amp_bcast[b]    the same for the slice (broadcast) sublayer step alone; amp_mlp[b] for the MLP step
  log10_prod_amp  sum over blocks of log10 amp[b]
  val_rel_l2      the probe's per-field relative L2 on the 4 cases (normalized space; a sanity read, not a score)
usage: python stream_reduce_af.py   (writes ../stream_norms_airfrans.json)
"""
import glob
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCR = "/home/psharpe/gh/physicsnemo/scratch"
SRC = {
    "job288480_288481": f"{SCR}/isla-stream-af-2026-10-19/out",
}
R = "raw slice reads (pooling and broadcast read h)"
A2 = "pooling and broadcast read norm_assign(h) (the adopted reads)"
RUNS = {
    "afu_r_isla_seed42": dict(arm="raw", reads=R, lr=3.75e-4, protocol="uniform", seed=42, code="code_v20"),
    "afu_r_isla_seed43": dict(arm="raw", reads=R, lr=3.75e-4, protocol="uniform", seed=43, code="code_v20"),
    "afu_bn2_isla_seed42": dict(arm="A2", reads=A2, lr=3.75e-4, protocol="uniform", seed=42, code="code_v20a2"),
    "afu_bn2_isla_seed43": dict(arm="A2", reads=A2, lr=3.75e-4, protocol="uniform", seed=43, code="code_v20a2"),
    "afu_a2s_isla_seed42": dict(arm="A2_shipped", reads=A2 + "; shipped code, recipe_v11", lr=3.75e-4, protocol="uniform", seed=42, code="code_a2s"),
    "afu_a2s_isla_seed43": dict(arm="A2_shipped", reads=A2 + "; shipped code, recipe_v11", lr=3.75e-4, protocol="uniform", seed=43, code="code_a2s"),
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
    json.dump(out, open(os.path.join(os.path.dirname(HERE), "stream_norms_airfrans.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
