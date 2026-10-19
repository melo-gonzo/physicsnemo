"""STREAM (FIG lane, 2026-10-18): the residual-stream growth behind H-RESID (#sec-nb-aerozoo-hlogit-verdict,
#sec-nb-aerozoo-hresid-probe, #sec-nb-aerozoo-hresid-p3-interim, #sec-nb-aerozoo-hresid-p3-final) as one compact json.

Sources (H-RESID Instrument 1, `hresid_probe.py`, md5 514f3c6b68380b1e4a0a5bcea36a18a1, one json per checkpoint):
  * LHR job 224693 (2026-10-15): raw reads, code_v26: azu_full_isla_seed42 (3.75e-4, uniform), azu_full_islahr_seed42
    (1.875e-4, uniform), az_full_islahr_seed42 (1.875e-4, windowed boundary protocol).
    Local copy scratch/isla-hresid-2026-10-15/out/<run>/e<N>.json.
  * LHR jobs 240803 (arm A, code_v29) and 240804 (arm A2, code_v30) (2026-10-16): azu_full_islabn_seed42/43 and
    azu_full_islabn2_seed42/43, epochs 1-69. Local copy scratch/isla-hresid-2026-10-15/arm_probes/<run>/e<N>.json.
  * LHR job 270135 (2026-10-18, this lane, stream_ext.sbatch): the same probe, cases and draws, extended to epochs 99, 149,
    199 and 249 for A, A2 and raw seed 42, and the raw seed-43 series. Local copy
    scratch/isla-book-overhaul-2026-10-18/fig/stream_out/<run>/e<N>.json.
All runs: AeroZoo `full`, 12 blocks, width 512, the same four full.val cases (sampler indices 0, 5, 8, 11) with cached draws
(200,000 cells), bfloat16 forward as in training, recipe_v9. Every number is the mean over the four cases' tokens.

Per checkpoint the json keeps:
  stream_out      mean token norm of h after the last block (blocks[11].norm_out) -- the "mean token norm of h"
  embed_out       mean token norm of h entering block 0 (blocks[0].norm_in)
  norm_in[b]      mean token norm of h entering block b
  amp[b]          block b's amplification: mean over tokens of |h_out| / |h_in|
  amp_bcast[b]    the same for the slice (broadcast) sublayer step alone; amp_mlp[b] for the MLP step
  log10_prod_amp  sum over blocks of log10 amp[b]
  val_rel_l2      the probe's per-field relative L2 on the 4 cases (normalized space; a sanity read, not a score)
usage: python stream_reduce.py   (writes ../stream_norms.json)
"""
import glob
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCR = "/home/psharpe/gh/physicsnemo/scratch"
SRC = {
    "job224693": f"{SCR}/isla-hresid-2026-10-15/out",
    "job240803_240804": f"{SCR}/isla-hresid-2026-10-15/arm_probes",
    "job270135": f"{SCR}/isla-book-overhaul-2026-10-18/fig/stream_out",
}
RUNS = {
    "azu_full_isla_seed42": dict(arm="raw", reads="raw slice reads (pooling and broadcast read h)", lr=3.75e-4, protocol="uniform", seed=42, code="code_v26"),
    "azu_full_isla_seed43": dict(arm="raw", reads="raw slice reads (pooling and broadcast read h)", lr=3.75e-4, protocol="uniform", seed=43, code="code_v26"),
    "azu_full_islahr_seed42": dict(arm="raw_halfrate", reads="raw slice reads", lr=1.875e-4, protocol="uniform", seed=42, code="code_v26"),
    "az_full_islahr_seed42": dict(arm="raw_halfrate_windowed", reads="raw slice reads", lr=1.875e-4, protocol="windowed boundary", seed=42, code="code_v26"),
    "azu_full_islabn_seed42": dict(arm="A", reads="broadcast reads norm_assign(h); pooling reads h", lr=3.75e-4, protocol="uniform", seed=42, code="code_v29"),
    "azu_full_islabn_seed43": dict(arm="A", reads="broadcast reads norm_assign(h); pooling reads h", lr=3.75e-4, protocol="uniform", seed=43, code="code_v29"),
    "azu_full_islabn2_seed42": dict(arm="A2", reads="pooling and broadcast read norm_assign(h) (the adopted reads)", lr=3.75e-4, protocol="uniform", seed=42, code="code_v30"),
    "azu_full_islabn2_seed43": dict(arm="A2", reads="pooling and broadcast read norm_assign(h) (the adopted reads)", lr=3.75e-4, protocol="uniform", seed=43, code="code_v30"),
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
    json.dump(out, open(os.path.join(os.path.dirname(HERE), "stream_norms.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
