# DTT-FAST SPIKE: does the compiled bf16 gradient blow-up show in training? (2026-10-24)

## For the decision-maker

- **Cause (probable, quantitatively supported):** not `1/dist`, but the normalization of a slice's mean normal,
  `m_s / |m_s|.clamp_min(1e-12)` (`blocks.py`, `_SliceBlock.forward`). A slice spread over a closed boundary has a raw
  mean normal near zero (the normals of a closed curve sum to zero). The blown-up row is block 9's slice 446, the
  block's smallest |m| (9.9e-4). Its bf16 value is mostly rounding noise. When that noise rounds it to exactly zero,
  the clamp takes over and the gradient is amplified by |m| / 1e-12 = 1e9. The observed amplification is 1.5e9.
- **How often:** 1 blow-up in 557 distinct bf16 one-ulp samples (234 of them compiled, 323 eager); that one draw
  appeared twice (FLAG 368799 and its rerun brepro). Compiled only, never eager.
- **Training:** no FLAG-size spike in any scanned training run. Finite routing-path spikes of up to about 3e5 times
  the running RMS do occur, in compiled and eager runs alike. They leave at most a one-epoch loss blip of about 15%
  and no lasting damage (Muon bounds the update). The training logs alone cannot show any of this; the optimizer state
  in the checkpoints can.
- **Certification:** the planned compiled-region certification catches the blow-up only by chance: 95% needs about
  60 sample-processes at this setting's rate and about 700 at the campaign-wide rate. It cannot settle the question,
  because the weakness is in the reference architecture's conditioning, not in compile. A deterministic test is
  better (below).

## 1. How often, where (FLAG, BF16-DIAG, integration pass 3 tables)

`count_samples.py` parses every comparison table in `../flag/out`, `../bf16diag/out`, `../integration/out/p3`
(`out_count.txt`). A run shared by two tables is counted once.

- **557 distinct bf16 one-ulp sample runs:** 234 compiled (FLAG, integration g1: `DIAG_COMPILE=1`), 323 eager
  (BF16-DIAG, integration f1). FLAG's follow-up processes outside the tables add about 27 more (brepro, bblk, bblke,
  bcomp, beager).
- **One blow-up** (a group ratio above 1e3): ISLA-DTT's unchunked reference, AirfRANS, 2,000 points, huber,
  compile_blocks, flag on, sample k = 2, job 368799 (node nvl72d039-T01). Its cold rerun 369180 brepro drew the same
  kernels and repeated it (6.68e5); 369325 bblk and 369180 bcomp did not. At that exact setting that is 1 blow-up in
  about 17 distinct samples over 3 distinct compile draws, 0 of 7 with the flag off, and 0 eager.
- **Next largest ratios, all far smaller:**
  - 12.4: the chunked head, AirfRANS whole boundary at 100k, flag off, k = 4, compiled (FLAG offB).
  - 8.2: the DrivAerML whole-boundary reference at 2k, k = 4, compiled, both flags bitwise.
  - 6.4-6.8: BF16-DIAG's eager chunked AirfRANS 2k family. This is systematic (present at ulp 0), the known chunked
    tail.
  - 5.4: the blow-up's own process at k = 4, in `blocks.2.assign.weight` (50 times the truth).
  FLAG's tail tensors are `blocks.{1,2,3,6,7}.assign.weight`, the same parameter family.
- **Graph:** block 9's compiled slice block.
  - The forward is normal: the loss and outputs are normal and nothing is non-finite. A zero `m_s` only zeroes two
    invariants of one slice.
  - The excess arises in block 9's backward. It is rank one and in one row: row 446 of `blocks.9.assign.weight`, the
    slice seeds; the other rows are off by about 12 through the seeds' rms coupling (`out_rows.txt`,
    `out_rows_slice.txt`).
  - It runs upstream through `norm_assign` and `h` into every earlier block (229 of 427 tensors above 1e3 times their
    truth). Blocks 10-11 and block 9's own geo/MLP weights are untouched.

## 2. Training logs (LHR `$T/runs`, 333 run dirs, 267 ISLA / ISLA-DTT, all bf16)

**What a run records** (recipe `train.py`):
- per step, the all-rank mean loss and its components, step time and memory (`metrics.jsonl` train_step,
  TensorBoard);
- per epoch, train and val summaries;
- a warning `non-finite gradient norm; skipped the optimizer step` when the total norm is not finite.

It logs **no gradient norm**: `get_total_norm` is computed every step for the finiteness test, then dropped. There is
**no clipping**. The optimizer is **Muon** for every 2-D parameter (momentum 0.95, Nesterov, Newton–Schulz
orthogonalized update, `match_rms_adamw`) and **AdamW** for the rest (`utils.build_muon_optimizer`). A finite spike with
a normal loss is therefore invisible in the logs except through later losses.

**Log scan** (`scan_runs.py`, `scan_runs.tsv`; 227 compiled-blocks, 40 eager):
- 0 skipped steps and 0 non-finite losses in any run.
- Loss jumps were measured three ways (definitions in the script): the 20-step median over the median of the 200
  steps before it; the epoch mean over the best of the 3 epochs before it; and the median of 5 epochs over the best of
  the 3 before. The three measures are distributed alike in compiled and eager runs:
  - ISLA-DTT AirfRANS, 20-step jump above 3.5: 8 of 36 compiled, 5 of 16 eager.
  - Steps whose 20-step jump exceeds 1.5: 153 per 1e4 steps compiled, 337 eager.
- The largest jumps (5x in the 10-step median, recovering in about 150 steps) occur in both kinds of run
  (`att2_af_dttqemb_seed43` compiled, `dttknobs_s2_exact_seed42` eager).
- Matched compiled/eager pairs (`out_pairs_final.txt`): final train losses agree within 2% for AirfRANS ISLA-DTT,
  AirfRANS ISLA, the lug and tensile2d.
  - Aside, not this lane's question: the compiled AirfRANS ISLA-DTT validation loss is higher on both seeds
    (2.71e-4 / 3.04e-4 against 1.97e-4 / 2.15e-4), while pressure and velocity L2 agree within 5%.

**Optimizer state as a spike record** (cpu-short 369919: `lhr/ckscan.py`, `ckana.py`; `out_ckana_summary.md`,
`out_ckana_30.txt`, `out_ckana.txt`; 12 runs, 2,878 checkpoints, every 2 epochs):
- How a spike shows in AdamW: a gradient rho times the running RMS raises sqrt(exp_avg_sq) by about 0.03 rho. 400 steps
  later 0.82 of that is left, so every step is covered by the next checkpoint.
- FLAG's blow-up would have shown as jumps of 3e2 to 2e7 in its AdamW tensors (`assign.gain` rho 6e8, `norm_assign`
  2e6, the embed biases).
- **Largest AdamW jump per run:** 3-22 in 8 runs, and 112 to 1.0e4 in 4 runs. The 1.0e4 is `afu_dtt_seed43`
  (compiled, lr 5e-4), at epoch 27 in `blocks.0.assign.gain`, a gradient about 3e5 times its RMS. So **no step in
  these runs had a FLAG-size spike** in any AdamW tensor.
- **Moderate spikes** (jumps of 30-330, rho about 1e3-1e4) occur in `afu_dtt` seeds 42/43 and `anc_anchor_seed42`
  (compiled), and in `r090eag_af_isla_seed42` (eager).
  - All are in the routing path of blocks 0-2: `assign.gain`, `norm_assign`, the assign projections in Muon.
  - Epoch losses around them show at most a one-epoch blip of about 15% (for example `afu_dtt_seed42`: 9.8e-5 at
    epoch 396, about 8.5e-5 on either side).
- The r090fix pairs (ISLA-DTT seeds 42/43, ISLA seed 42, lug seed 42) are as quiet as their eager pairs.
- Muon momentum norms jump 34-900x at most between checkpoints in every run, eager included (8.7e4 in
  `afu_dtt_seed43`), so they are no detector at this spacing.

## 3. Mechanism

- **Code (`blocks.py` at 5e82a9b98).** Both anchor products run under autocast, so both are bf16:
  `z_pos = einsum("bns,bnc->bsc", a, r)` and `m_s = einsum("bns,bnc->bsc", a, n_hat)`. Then
  `m_s = m_s / m_s.norm().clamp_min(eps)` with `eps = EPS = 1e-12`.
  - The region's comment says "the geometry must stay in the input precision", but the anchors themselves are bf16.
  - `_relational_invariants` then computes `rel = r - z` (float32), `dist = |rel|.clamp_min(1e-12)`, `rel / dist` and
    `log(dist)`.
- **`1/dist`: refuted on this input.**
  - Probe: `slice_probe.py`, local CPU, eager, FLAG's AirfRANS dump and state, samples 0/2/4, float64, float32 and CPU
    bf16 (`out_slice_probe.txt`, `out_slice_summary.txt`).
  - Every slice is diffuse: the largest point weight is at most 0.005 and the median effective count is 490 of 1,017
    tokens. Every anchor is at least 0.12 (gauge units) from the nearest token.
  - At the clamp, `1/dist` is 1e12. But the anchor is bf16-rounded, about 2^-9 |z|, roughly 1e-3, so `dist` falls below
    about 1e-3 only if a slice collapses onto one token. That never happens here.
- **`1/|m|`: fits.**
  - 139 of 6,144 slices have |m| < 3e-3, and the smallest is 1.5e-4. Block 9's slice 446 has the block's smallest |m|
    (9.9e-4) and is exactly the blown-up row.
  - CPU bf16 moves |m| by a median 3e-4 (up to 5e-3). For the small slices, |m_bf16| / |m_64| runs from 0.19 to 5.6.
  - In the float64 truth, row 446 is already 15 times the median row (`out_row446.txt`): 1/|m| conditioning even
    without rounding.
  - If bf16 rounds `m_raw` to (0, 0), the clamp gives `d m_hat / d m = I / 1e-12` in place of
    `(I - m_hat m_hat^T) / |m|`, a gain of |m| / 1e-12 = 1.0e9. Observed: row 446's norm is **1.47e9** times the truth's,
    with cosine 0.79 (the missing projection).
- **Why compile and the flag matter (inferred, not measured on GPU).**
  - Not the elided casts: `m_s` is a bf16 GEMM output in eager and compiled alike, and its norm is float32 in both.
  - What changes is which GEMM kernel each compile draw picks for the [512, N] x [N, 2] anchor products. FLAG saw these
    24 products run as cuBLASLt split-K with **bf16 partial sums** when the flag is on, and the kernel choice vary
    between cold processes (pad_mm, autotuning).
  - Exact zero needs coarse partials: bf16 partials of about 0.5 lie on a grid of about 1e-3, as large as |m| itself,
    so they can cancel exactly. A float32 sum rounded once to bf16 essentially never gives an exact zero.
  - Eager's kernel for this product was not profiled. Eager never blew up in 323 samples, but nothing in the code
    protects it.
  - Hypothesis, untested: at training's 100k points, the ~100k passive tokens follow the 1,017 boundary tokens on the
    K axis, so the boundary tokens may fall into one split-K partial and never cancel across partials.

## 4. Would the final rerun's compiled-region certification catch it?

- Per evaluation, a blow-up fails the gate by about 1e8, so it is caught if it is drawn at all. The draw depends on the
  process (the compile draw) and on the sample.
- The rate p is about 1/21 at FLAG's setting (1 of 7 samples, in 1 of 3 processes) and about 1/234 over all compiled
  samples.
- Catch probability is 1 - (1 - p)^n:

| evaluations n | p = 1/21 | p = 1/234 |
|---|---|---|
| 7 (one family) | 29% | 3% |
| 21 (3 processes x 7) | 64% | 9% |
| 60 | 95% | 23% |
| 700 | 100% | 95% |

- Samples inside one process share its kernels (bblk: 0 of 7 in another draw), so spreading over cold processes
  counts more than adding samples.
- Even a pass would not show safety: the conditioning is the reference's, and eager is not provably immune.
- **Deterministic check instead:** a closed boundary with uniform routing (equal logits and measures on a regular
  polygon) has m = 0 exactly. Its m-gradient is then 1e12 times larger, in float64 eager too. That makes a one-line
  test of whatever floor DTT-KNOBS chooses.

## Ideas for DTT-KNOBS / the program (not implemented: they change arithmetic or the architecture)

1. **A smooth floor on the mean-normal norm.** For example `m / sqrt(|m|^2 + delta^2)`, with delta at the bf16
   resolution of the sum (about 1e-2), or keeping `m` unnormalized. This removes the 1/|m| conditioning. It is an
   architecture change and needs retraining.
2. **Anchor products in float32** (`z_pos`, `m_s`), as the region's comment intends. This takes |m|'s noise from about
   3e-4 down to about 1e-7, so exact zeros vanish, but the 1/|m| conditioning stays. It moves a rounding point, so it
   is out of DTT-FAST's scope.
3. **Log the gradient norm.** `train.py` already computes `get_total_norm` every step; writing it to `metrics.jsonl`
   would make spikes visible without a checkpoint forensic. This is a recipe change.

## Files

- Scripts: `count_samples.py`, `rows.py`, `row446.py`, `scan_runs.py`, `pairs_final.py`, `slice_probe.py`,
  `slice_summary.py`, `lhr/ckscan.py`, `lhr/ckscan.sbatch`, `ckana.py`.
- Outputs: `out_*.txt`, `out_ckana_summary.md`, `scan_runs.tsv`, `ck/` (12 per-run records and logs, 6.6 MB).
- Probe records: `probe/slices.pt`.
- Jobs: `JOBS.md`.
