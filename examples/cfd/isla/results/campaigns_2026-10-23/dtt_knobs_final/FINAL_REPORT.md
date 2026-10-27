Status: DONE

# DTT-KNOBS: final report (coordinator, 2026-10-26)

**Outcome.** Every pre-registered stage-2 arm was trained and read on AirfRANS (eager, two seeds each).
- Retrained at theta 2, 1, 0.5 and 0, ISLA-DTT's tree read is a **pure cost knob, free to theta 2** (R2). No rung of the
  ladder moves any field past the 10 % bar, so R2's "converges" label does not apply; its registered gloss is that the
  approximation costs nothing at theta <= 1. R2 does not test the book's knob-rule credit for pose invariance (the
  read's error at fixed weights vanishing as theta -> 0); that question is flagged for the supervisor.
- A finer read **narrows the far-field gap** to interacting ISLA (Q3): theta 0.5 narrows wake velocity, and the exact
  read narrows nut beyond one chord.
- **K-B1** (the boundary read after every second block) reads no harm and raises no flag. It is a scope-freeze candidate,
  pending Peter's word and stage 3. K-BASIS8 harms nut. ISLA-DTT uses ISLA's scales on pressure, where interacting
  ISLA does not.
- Stage 1 (inference only) showed that every knob is a trained constant, and found two code facts that went to
  DTT-POSE.

**What ran.**
- Stage 1: 21.76 GPU-h of 30, in seven children: sens, cost, cap, gap, order, treeprec, and design (read-only). Each
  child's report was second-read and revised. Job ids are in `JOBS.md` and each child's JOBS.md.
- Stage 2: smokes 366116 / 366117; lanes isla.dttknobs.s2.<arm>-seed4x (first links 366334-366486); scorers 376686,
  382281, 400809; regional jobs 382386, 382387, 400920; cpu-short cohcount jobs 366603 / 366610 / 366622.
  1,311.4 GPU-h of the 1,450 cap: arms 1-4 and 6 used 993.0 of 1,050, K-EXACT 318.4 of 400.
- Campaign total: about 1,333 GPU-h.
- Commit: `psharpe/isla-dttknobs-b1` a1d34dd39, pushed. It is a record of K-B1, not for merge, with the whole ISLA test
  directory passing (325 passed).

**Results** (arm ÷ control, pressure / in-plane velocity / nut, ratios of arithmetic seed means; `RESULTS.md`):

| arm | ratio | R1 | flag | GPU-h, 2 seeds | step |
|---|---|---|---|---|---|
| control r090eag_af_dtt | 1 | - | - | 220.8 | 0.934 s |
| K-THETA2 | 1.071 / 1.040 / 0.980 | no change | up (p) | 216.4 | 0.935 s |
| K-THETA05 | 0.936 / 0.950 / 0.932 | no change | down (p, v, nut) | 217.5 | 0.939 s |
| K-B1 | 1.013 / 0.988 / 0.991 | no change | - | 204.2 | 0.887 s |
| K-BASIS8 | 1.054 / 1.027 / **1.287** | harms nut | up (nut) | 216.9 | 0.934 s |
| K-EXACT | 0.957 / 0.976 / 0.879 | no change | down (p) | 317.6 | 1.354 s |
| DTT-NOSC-STD ÷ B (fix-trained) | 1.208 / 1.203 / 1.177 | the scales carry it (R5) | v flagged | 133.3 | 0.572 s |

R4 (each arm ÷ the control, in the far-field cells):
- Wake velocity: narrowed by K-THETA05 (D_R -0.122 against 2 SE 0.112).
- Nut beyond one chord: narrowed by K-EXACT (-0.248 against 0.213).
- Everything else reads level.

Stage 1's numbers are in `STAGE1.md`, and the code facts in `CODE_FACTS.md`.

**What it means, and how sure.**
- On AirfRANS, theta 1 is not a costly approximation. Pressure has a resolved slope of 6-7 % per rung from theta 2 to
  0.5 (best at 0.5), and the other fields move inside seed noise. High confidence, two seeds.
- The far-field gap does respond to a finer read, cell by cell, but by margins of only 10-16 % over 2 SE. Medium
  confidence.
- K-B1 halves the reads at no measurable cost on AirfRANS. Its car-scale saving is unknown, which is why stage 3 exists.
- The arms ran on 0.9.0 code (code_cmpaf_fix), before DTT-FAST and before ISLA-DTT became the default interior form.
  With DTT-FAST, 400k whole-boundary car points already fit at theta 1, so theta 2's case rests on step time alone.

**Rulings relied on.**
- The supervisor's approval of the PREREG with its amendment: K-EXACT eager with a 400 GPU-h gate, the caps, 12 lanes.
- Peter's knob rule, Peter's whole-boundary ruling, ISLA-DTT as default (10-24), the devin model rule (10-24 13:49).
- My 17:35 clarification before results (arithmetic seed means; pooled e_R).

**For project memory (suggested).**
- A theta-1 / leaf-4-trained ISLA-DTT is read best at its trained setting.
- The coherence mask is vacuous at theta >= 0.919 (3D) / 0.765 (2D).
- The bf16 forest under bf16 training (now DTT-POSE's).
- Retrained theta is a pure cost knob on AirfRANS.
- Basis 8 harms nut after retraining, though inference ablation suggested slack.
- `sbatch --export=ALL,X='a,b'` truncates at the first comma.

**Left.**
- STAGE3_PROPOSAL.md: K-B1 on DrivAerML Volume, about 520 GPU-h. It needs the supervisor's approval and Peter's word.
- A pending question to the supervisor: prune the stage-2 runs' 1.6 TB of intermediate checkpoints.
- The notebook drafts, each reviewed and revised, with their chapter changes, for the supervisor to apply:
  NOTEBOOK_DRAFT_STAGE1.md (md5 293fbd0b) and NOTEBOOK_DRAFT_STAGE2.md (md5 d1ff8d1a; ten changes, one In-flight callout
  removed at 20:1240).
- A question for the supervisor, not decided here: whether the book's knob-rule credit for ISLA-DTT's tree read still
  holds. Every chapter credits the read's error at fixed weights vanishing as theta -> 0, which stage 1 measured; only
  20:1228-1231 tied it to a retrained ladder, and R2 reads "pure cost knob" there (NOTEBOOK2_REVIEW.md, knob-rule note).
- Nothing is running. The registry row was removed at 06:28.

**On disk.**
- This folder: about 90 MB.
- LHR: `$T/runs/dttknobs_s2_*`, 1.6 TB (pending the prune question); `$T/kernel/dttknobs_*`, about 0.4 GB (compiler caches deleted); `$T/code_dttk_*`, 284 MB; `$T/evals_dttknobs_s2`, 2 MB.
- No worktree; the K-B1 worktree was removed.
- DK-GAP's 5.4 GiB of predictions were deleted.
