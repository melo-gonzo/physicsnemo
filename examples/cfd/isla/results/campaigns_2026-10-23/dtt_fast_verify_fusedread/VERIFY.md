# VERIFY-FUSEDREAD: independent verification of DTT-FAST FUSEDREAD at 6aada61cd

2026-10-24 04:15 EDT. Branch `origin/psharpe/isla-dttfast-fusedread` at **6aada61cd**: unmoved, 9 commits on
f13eb5ce0.

**Result: the piece passes. I would merge it**, resolving the text conflict with `psharpe/isla-dtt-fast` by keeping
both sides (REVIEW.md, "Merge note"). No central claim is refuted. One claim is unverified: that 300k fits, which I did
not rerun.

## What ran

- **Code.**
  - Reference: `$T/code_dvcost2a`. I checked it by md5: its isla model files equal f13eb5ce0's, except
    `segment_softmax.py`, which differs only in how Triton is imported.
  - Candidate: `$T/code_dttfast_fusedread_t4`, a git archive of 6aada61cd. All 32 isla model and test files are
    md5-equal to `git archive 6aada61cd`. This is not the author's overlay snapshot `s2`.
  - Merged probe snapshot: `$T/code_dttfast_verify_fusedread_merged`, which is `t4` plus the four files of
    `git merge-tree b9ccb44e6 6aada61cd`, its conflicts resolved keeping both sides (`lhr/overlay_merged.md5`).
  - Each process printed its version and the md5s of the files it loaded.
- **Probes.** All mine, in `lhr/`; I did not reuse the author's `fr_probe.py`.
  - `vf_probe.py` uses its own parameter draw: every read's output projection N(0, 0.03) and kernel weights N(0, 0.3),
    seed 2027, on the seed-42 model. It uses its own point subsample and counts which read and which node means ran.
  - `vf_compare.py`, `vf_sink.py`, `vf_b2.py`.
- **Jobs.**
  - 365645: GPU, one node, 12:37 (0.84 GPU-h).
  - 365794: GPU, rerun of two probes I had fixed, 1:53 (0.13 GPU-h).
  - 365646: the whole ISLA test directory on CPU, `t4`, cpu-short.
  - Total 0.97 of the 3 GPU-h cap (JOBS.md).

## Claims

| claim (author) | result | evidence |
|---|---|---|
| Pair lists and node counts identical | confirmed | All 15 runs (5 dumps x f64/f32/bf16): `torch.equal` of the lexsorted (qi, bj) keys. The 200k bf16 step: equal sorted-key md5, 135,747,178 pairs and 129,363,137 node pairs on both sides (`out_365645/tables.md`). |
| float64 at roundoff | confirmed, but vacuous for the fused kernels | Cand vs ref at most 1.6e-14 (kernel_basis gradient, drv_body). Both sides run the reference read in float64: the counters show `segment_softmax_read` and `node_means` on both (REVIEW.md, nit 4). |
| float32: within 2x of the reference's own distance to float64 | confirmed | Per-group ratio 0.98 to 1.03 over all 5 dumps; worst single tensor 1.74 (AirfRANS, `blocks.5.slice_mlp.0.weight`, 3.0e-7 against 1.7e-7). Cand vs ref is at the reference's own run-to-run spread (for example drv_body outputs 1.7e-6 against 1.6e-6). |
| bf16: same distance to float64 as the reference's bf16 run | confirmed | Per-group ratio 0.94 to 1.00. Worst single tensor 2.15: AirfRANS `blocks.2.geo_logit.weight`, 5.3e-3 against 2.5e-3, a bf16-noise-sized tensor. |
| bf16 step at 200k, DrivAerML body | confirmed | Distance to a float32 200k step is 1.00x the reference's in every group (loss 8.55e-3 against 8.56e-3, outputs 3.34e-2 against 3.34e-2). The float32 step is the candidate's, because the reference's float32 step runs out of memory on a 276.6 GiB GB300. Cand vs ref outputs 1.5e-2, against the reference's own run-to-run spread of 3.1e-3: bf16 rounding flips that leave the distance to float32 unchanged. |
| P1 fixed (no stale basis gradient after a partial backward) | confirmed | Whole model on CUDA, float32 and bf16, checkpoint off/read/block: every basis and weight gradient within 0 to 1.2e-7 of a full backward; the sink is empty after every sequence (`out_365794/sink_*.log`; REVIEW.md). |
| Frozen basis, no_grad | confirmed | Frozen: no sink, no basis gradient, other gradients agree with the reference read to 3e-7 (float32). no_grad outputs 7e-7 (float32), bitwise in bf16. |
| Node means: backward bitwise, forward to summation order | confirmed | Real 200k forest: backward `torch.equal` in bf16 and float32. Forward float32 fused vs ref 2.15e-7, equal to ref vs ref; every leaf with 2 or fewer cells bitwise. |
| Reference path unchanged; auto selection; not in the state dict | confirmed | Diff (REVIEW.md); counters (auto resolves to fused in f32/bf16 on CUDA, reference in f64); the candidate loads the reference-written state dict strictly. |
| B > 1, 2-D, empty points | confirmed | `vf_b2.py`: two-case batch in 3-D and 2-D, pairs identical, both cases populated. Distances to CPU f64, fused against reference: float32 0.85 to 1.77, bf16 1.00. Empty rows zero in both reads (`out_365794/b2_cand.log`). |
| int32 limits | partly confirmed | At 200k the basis offsets pass 2^31 elements (2.17e9) and the step matches. More than 2^31 pairs: code reading only. |
| CUDA tests 16 passed | confirmed | `pytest_cand.log`: 16 passed. Merged snapshot: 27 passed, including the 3 checkpoint tests. |
| Whole ISLA directory on CPU, 325 passed | confirmed | 365646 on `t4`: 325 passed, 60 skipped, 0 failed (`tests/tests_isla.dttfast.vfusedread.tests_365646.log`). |
| Cost at DrivAerML body 200k bf16: -15% time, -29 GiB peak | confirmed | One sample, 3 steps each, my parameters. Reference 2135 / 2131 ms, peak 219.2 GiB; candidate 1809 / 1811 ms, peak 190.2 GiB. That is -15.3% and -29.0 GiB (author: 2218 to 1877 ms, 217.2 to 188.0 GiB). |
| 300k fits for the candidate and not the reference | unverified | Not rerun; the author's table is the record. Related: at 200k in float32 the candidate fits (8.2 s, 251.5 GiB) and the reference does not. |
| Timings on AeroZoo, AirfRANS, whole boundary | unverified | Not rerun, as the brief allows. |

## Equivalence summary

The distance ratio is cand to exact over ref to exact, per tensor group, worst group shown. Exact is the reference's
float64 step (the 200k row uses the float32 step). Full tables are in `lhr/out_365645/tables.md`.

| case | pairs (f32) | f64 cand vs ref | f32 ratio (worst group) | bf16 ratio (worst group) | f32 cand vs ref, outputs | f32 ref vs ref2, outputs |
|---|---|---|---|---|---|---|
| DrivAerML body | 1,379,115 | 1.6e-14 | 1.00 | 1.00 | 1.7e-6 | 1.6e-6 |
| DrivAerML whole | 1,664,337 | 2.7e-15 | 1.00 | 1.00 | 1.1e-6 | 1.4e-6 |
| AeroZoo | 555,440 | 2.4e-15 | 1.03 | 1.00 | 8.7e-7 | 1.1e-6 |
| AirfRANS (2-D) | 43,038 | 2.2e-15 | 1.02 | 1.00 | 1.0e-6 | 9.8e-7 |
| lug | 241,420 | 7.9e-15 | 1.00 | 1.00 | 1.9e-6 | 1.5e-6 |
| DrivAerML body 200k (bf16) | 135,747,178 | n/a | n/a | 1.00 | n/a | n/a |

## Left on disk

- Local: this folder. It holds the scripts, the copied logs and tables, and the merge overlay. It is small; the raw
  `.pt` files stay on LHR only until deleted (below).
- LHR:
  - `$T/code_dttfast_verify_fusedread_merged`: 57 MB, read-only. Kept so the parent can rerun the merged checks.
  - Launchers and logs under `$T/kernel/dttfast_verify_fusedread/`.
  - The raw `.pt` outputs of 365645 (31 GB) are deleted after this report.
