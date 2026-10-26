Status: DONE

The registered condition to adopt DTT-POSE's forest is met. No field of either dataset reads cost under N, the lug reads no
paired cost, and both checks pass: P (pose) and Q (precision). Below the rule's resolution, though, the lug is probably
about 10 % worse natively, and AirfRANS is 6-7 % worse, seeds separated, on pressure and nut. The supervisor may want to
take that to Peter before adopting.

**What ran (LHR):**
- Snapshot `$T/code_dttposex`: code_cmpaf_fix plus DTT-POSE's 0.9.0 port (patch 5ab4f654; tree_read 00d83cb9, model
  fc8d0fa9), read-only. Overlay proof in `snap/`.
- Checks and smokes: 367263, 367264, 367265 and 367376, all passed (CHECKS.md).
  - Check (b) failed its bound. The posed residual with random weights is 0.9.0's geometry: theta 1 and theta 0 agree to
    a ratio of 1.000. Amendment A1.2's fallback applies.
  - Item G: the compiled graphs are identical, so no certification was needed.
- Lanes: lug 367422 / 367423 and their chains, AirfRANS 367454 / 367455 and their chains. Two lug links were lost to the
  NFS Inductor-cache race; the chains restarted them.
- Scoring: 383802 (AirfRANS) and 401023 (lug). Q probes: 383803 and 401024. Regions: 383935.
- No commits; no branch was written.

**Results** (`RESULTS.md`; tables in `results/read/read_all.md`). Arm / control, seed means:

| dataset / field | arm / control | D | 2 SE | seed-matched | N | sub-bar flag |
|---|---|---|---|---|---|---|
| lug von Mises | 1.113 | +0.107 | 0.234 | 1.086 / 1.146 | no change | none |
| AirfRANS pressure | 1.058 | +0.057 | 0.051 | 1.030 / 1.087 | no change | up |
| AirfRANS velocity | 1.016 | +0.016 | 0.060 | 0.999 / 1.033 | no change | none |
| AirfRANS nut | 1.074 | +0.071 | 0.070 | 1.105 / 1.042 | no change | up |

- The lug paired reading gives no label: seed 42's ratio, 1.086, is under the 1.10 threshold.
- P passes: lug posed / native is 1.0001 / 1.0014, against the control's 1.528 / 1.977.
- Q passes: the bf16 forest is bitwise the float32 one on all 600 lug and 200 AirfRANS cases, native and posed.

**Beside the verdicts:**
- Lug: the arm beats the control in only 19 % of cases, and the case median ratio is 1.109. Arm / interacting ISLA is
  1.552, against the control's 1.394.
- AirfRANS arm / ISLA: 1.052 / 1.146 / 1.440, against the control's 0.994 / 1.129 / 1.342.
- DK-GAP regions: wake velocity level (0.999); nut worse near the wall (1.21), in the wake (1.12) and beyond one chord
  (1.067).
- Pairs per point are 3 % lower; step time is the same as the control's (lug 0.420 against 0.425 s).

**What it means.** The fix does what it was built for: trained models are pose-invariant (posed error 0.64 of the
control's on the lug), and the training forest is the scored forest. The accuracy signal is consistent in direction on
both datasets and both seeds, so I would call a small native cost likely. The two-seed rule cannot label it; my
confidence that it is real is moderate. A cause to test, which this lane did not: 6 face trees in place of 8 octant
trees, together with the banded box test, give coarser clusters.

**Rulings relied on:**
- the supervisor's approval of PREREG 80bae24a and amendments A1.1-A1.5 (PREREG md5 2f46a10c, cap 650 GPU-h);
- the A1.2 fallback for check (b);
- A1.3, the lug paired reading.

New rule for project memory, received mid-run: subagents launch with model `devin/claude-opus-5.5:high`. This lane
launched none.

**Cost and what stays on disk.**
- About 508 GPU-h (sacct), under the 650 cap.
- Nothing is running, and the registry row is removed.
- On LHR, `$T/runs/dttposex_{lug,af}_dtt_seed4{2,3}` hold about 130 GB each, with checkpoints every 2 epochs. Only the
  final checkpoints are needed; deleting the rest is your call.
- `$T/kernel/dttposex` (5.1 GB, mostly the Inductor cache) holds check (a)'s captures in `check/cap/`. Keep them for the
  pre-merge equality check (A1.4).
- `$T/code_dttposex` (57 MB) is read-only.
- Locally, the scratch folder is 4.6 MB.
