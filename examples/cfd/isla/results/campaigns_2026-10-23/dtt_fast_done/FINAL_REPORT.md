Status: DONE (2026-10-24 18:35 EDT; three days before the deadline)

**Outcome.** ISLA-DTT's training step on GB300 is now 1.6-1.8x faster than the reference on DrivAerML and AeroZoo and uses
0.7-0.8x its peak memory. It also trains point counts the reference cannot: 400k, and up to 12.8M on DrivAerML. It computes
the same function: pair lists are identical, outputs and gradients agree to roundoff, and the version and state dict are
unchanged. Two branches are ready for your merge, both on psharpe/isla b4b10d5a9 and pushed:
- **psharpe/isla-dtt-fast 4d281a702** (ISLA 0.12.0): this is the final rerun's head if the current forest stays.
- **psharpe/isla-dttfast-pose 38a7915cc** (ISLA 0.13.0, the version you approved at 15:22): the same code on DTT-POSE's
  forest. It waits on DTT-POSE-EXP's registered read.

**What is in it.** Each piece was verified or reviewed independently, then integrated in 5 passes; the records are in PLAN.md.
- A Warp forest build: every field bitwise equal; 107 to 11-18 ms; 187 host syncs down to 5.
- A Warp walk: lists equal in the reference's order; 122 to 12 ms; 183 syncs down to 1.
- A fused Triton read, which recomputes the kernel term in backward. Its backward is deterministic and 1.41x faster.
- A fused basis MLP with its invariants; its backward is 1.66x faster.
- PairPlan built by searchsorted.
- Query-chunked training, which is the training path with no switch. A size rule picks the fewest chunks that fit, is
  all-reduced across ranks, and is refit on the final code.
- One execution attribute, `tree_read_kernel` ("auto" or "reference"). The reference path is unchanged and is the CPU and
  float64 fallback.

Dropped, with their reasons recorded: read checkpointing (chunking makes it redundant), the compiled read projections
(part P, which missed certification by 1 of 1,440 groups), and per-chunk padding (no gain).

**Equivalence** of the head against psharpe/isla, each version in its own process:
- Pair lists are raw-equal on 5 real dumps.
- float64 differs by at most 3.1e-13; the float32 ratio is at most 1.14 (CHILD_RULES bound 2).
- bf16 lies inside the reference's one-ulp family, apart from single-run maxima, which I accepted.
- Whole ISLA directory on a GB300 with every CUDA test: 709 passed, 0 failed (pose head: 728, 0 failed).

Tables: `integration/RESULTS.md` (passes 1-5), `pose/RESULTS.md`, and the piece verifications `verify_*/VERIFY.md`.

**Step time** (`final2/TABLE.md`; `final/TABLE.md` has the f13eb5ce0 and 7213c70c5 columns). The recipe's own bf16 step
with per-block compile:
- DrivAerML whole boundary at 200k: 4,112 ms at f13eb5ce0, 2,105 at 7213c70c5, 1,209-1,231 at 4d281a702, 1,174 on the
  pose head.
- DrivAerML at 400k: out of memory before; about 1.8-2.0 s now.
- AeroZoo at 200k: 1,250 to 691-718 ms.
- Against interacting ISLA, ISLA-DTT now costs 1.27-1.48x its step on DrivAerML (2.5x at the start), 1.15x on AeroZoo and
  1.1-1.3x on AirfRANS, and less memory from 200k up.
- For FREEZE-PLAN: steps are 5-20% faster than the pass-2 table, but peaks are 1.3-1.9x higher (143 GiB at DrivAerML
  200k), because the size rule fills the card.
- Surface task in ISLA-DTT form, DrivAerML whole boundary at 200k: 3,996 to 1,188 ms. That is still about 4x option A.
  HiLift still runs out of memory in the geometry call.

**Ownership note** (your 06:40 note, received at 18:30): the fast pieces also hook into tree_read.py. There is a build= branch in
`BoundaryForest._build`, and on the pose branch `_point_pairs` is split into `frame_points` (DTT-POSE's formula, verbatim) and
`frame_pairs`. The reference semantics are unchanged, and two reviews checked that against 0f06e4065.

**Your open decisions:**
- The bf16 reduced-precision flag. It is free in time but not the cause, and I recommend keeping the default (`flag/`).
- Part P (`integration/RESULTS.md` pass 3). Read by your 10:04 rule (a one-ulp family of at least 7 samples), it is
  inside eager's family on every group: the one near-miss group's excess is 3.45e-7, against draws of 2.59-5.18e-7
  (`integration/out/p3/p_cert2.md`). It stays out per your 12:24 message. To restore it, run `git revert 023c0f3d6`, then
  the GB300 suite on both heads (about 1.5 GPU-h). I recommend applying it after the final rerun.
- `compile_blocks` elides bf16 casts. You ruled to keep it; the evidence is in `compile/RESULTS.md` and
  `block_precision_casts.patch`.
- The pose merge after DTT-POSE-EXP's read.

**The 6.6e5 gradient blow-up** (`spike/RESULTS.md`): 1 in 557 bf16 samples, and only in compiled AirfRANS steps. The
suspected cause is `m_s/|m_s|.clamp_min(1e-12)`. It was not seen in 267 training runs, and certification catches it only by
chance, so the fix belongs to DTT-KNOBS.

**For project memory:**
- Judge bf16 equivalence against a one-ulp family of samples spread across cold compile processes.
- Time steps in training's own point-count state.
- The recipe logs no gradient norm.

**GPU-h:** about 81 of 88. **Running:** nothing; the registry row is removed.

**On disk:**
- Local: worktrees `physicsnemo-isla-dtt-fast` and `physicsnemo-isla-dttfast-pose` (about 1 GB each; keep them until
  you merge), and this folder (69 MB).
- LHR `$T/kernel/dttfast_*` (about 37 GB):
  - `dttfast_surface/cases`, 16 GB: the harness inputs.
  - `dttfast_integ` parameter states, 4.8 GB.
  - `dttfast_plan` outputs, 5.3 GB.
  - `dttfast_fusedread` plan dump, 7 GB.
  - `dttfast_flag/keep_blowup`, 1.1 GB.
- LHR: 90 read-only snapshots of about 58 MB each.
- I deleted about 205 GB of raw outputs whose tables are written.
