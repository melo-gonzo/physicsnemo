# DRAW-EXACT design: a bitwise-deterministic token draw on GPU (2026-10-25)

Scope: `physicsnemo/experimental/models/isla/sampling.py` at fc6ed19d0. The PPS fixed point
(`pi_i = min(1, c a_i)`, `sum pi = share`) and the Horvitz-Thompson weights `a_i / pi_i` keep their meaning in every
option. Only the two float64 prefix sums change: the tail sums of the sorted measures (which set `c`) and the
systematic line `cumsum(pi[order])` (which sets the kept cells).

## What the probes measured

Jobs **377433** (`isla.drawexact.probe`, 3 min 18 s on 4 GB300s, 0.22 GPU-h) and **377520** (`isla.drawexact.debug`,
about 2 min, 0.13 GPU-h), code `$T/code_drawexact_base` (fc6ed19d0 + td0152), torch 2.12.0+cu130, Triton 3.7.0.
The probe is `probes/draw_probe.py` (md5 67991964 for 377433); `probes/compile_debug.py` (8c218a4d) is the debug probe.
Raw results are in `results/probe_377433/` and `results/debug_377520.log`; the per-case table is
`results/summary_377433.md`.
Inputs: the real boundaries of 16 AeroZoo cases and DrivAerML run_1 and run_2. AeroZoo includes reynolds-test
cases 0 and 1, the cases REYNOLDS-OUT ran. DrivAerML is the whole boundary: five pieces, 17.7M and 16.4M cells. Each
is read raw from its `.pdmsh`, with float32 points and float32 cell areas as the stored effective measure, as TOKEN-DRAW
read them. The budget is 200,000. Each variant runs `prepare` + `finish` in `draw_probe.py`, which is
`sample_boundary_cells` with only the scans and the measure sort swapped. The probe's `today` variant equals the
module's own function bitwise, on CPU, in all 18 cases.

**Why the draw is not deterministic: exact ties at the evaluation start.**
- In evaluation the start is 0.5 (`EVAL_START`). AeroZoo's water fill often gives half-integer shares. Case 0 has
  shares 31250, 58315.5, 52119 and 58315.5, so its cumulative shares are 89565.5 and 141684.5. Eval hits `0.5 + m` then
  land *exactly* on condition boundaries: 2 ties in case 0, 3 in case 050, 2 in case 087.
- Worse, a condition kept whole (every `pi = 1`) that starts at a half-integer offset puts *every* one of its cell
  boundaries exactly on a hit. Case 0's outlet is such a condition, with 52,119 cells.
- Only the rounding of the scan decides which side of each tie a hit falls on. CUDA's float scan rounds differently
  from run to run, so the kept cells change: up to 4 cells over 6 runs in case 0, 9 in case 050 and 2 in case 087.
  This is the mechanism behind REYNOLDS-OUT's AeroZoo case 0. It is now shown on that case's own boundary meshes, read
  raw rather than through the recipe.
- It is also a **correctness bug**. Today's GPU draw keeps *duplicate* cells and drops `pi = 1` cells (377520). In 6
  runs, case 0 had 0, 0, 4, 1, 2 and 3 duplicates; case 050 had 12, 5, 1, 3, 12 and 4, with the same number of
  whole-condition cells dropped. The module's promise "a cell with pi = 1 exactly once" fails on GPU in evaluation.
- Away from exact ties, float rounding still moves cells. Even CPU's sequential sum drifts from the exact line,
  because the summands are identical. Today's CPU and GPU draws differ by one kept cell in 5 of 18 cases at the eval
  start, and by 27 cells over 544 random-start draws.
- The HT weights differ by ulps between runs only where the float64 tail sums are inexact: in case 022, and in
  TEST-AUDIT's float64-measure cases. Float32-derived measures usually sum exactly in float64, in any order.

## Options

"Function change" is the number of kept cells that differ from today's **CPU** result, out of 200,000 tokens per draw.
Times are the median of 10 whole draws on GB300. Peak memory is 2,699 MiB on run_1 for every variant, so it is not
a column.

| option | determinism, 6 runs (kept & weights bitwise), 18 cases | CPU = GPU bitwise, same float64 inputs | function change: eval start (cases) / 544 random draws | weights vs today-CPU | DrivAerML run_1 / run_2 draw (ms) | AeroZoo median (ms) | sync-free | fullgraph + dynamic compile |
|---|---|---|---|---|---|---|---|---|
| today: ATen float64 cumsum | **14/18** (runs differ by up to 9 cells; duplicates) | 12/18 | 0 (reference) | — | 19.4 / 18.2 | 6.6 | yes | yes |
| **A: exact int64 fixed point** | **18/18** | **18/18** | 5 cells (5 cases) / 27 | ≤ 1.0e-15 rel. (float64) | 21.1 / 19.2 | 7.4 | **yes** | CPU yes (bitwise with eager). CUDA: the first form failed in Triton (below); the rewritten whole draw is **unverified on CUDA** until the final GB300 job |
| B: float64 cumsum on the host | 18/18 | 18/18 | **0 / 0** | 0 | 45.6 / 43.2 (**+26 ms**) | 8.0 | **no**: a host round trip per condition (tail sums) plus one for the line | yes, as an opaque op; no CUDA graphs |
| C: fixed-order Triton scan | 18/18 | 13/18 | 5 (5) / 27 | 0 | 20.4 / 19.2 | 6.9 | yes | yes |
| D: ATen deterministic cumsum, global flag toggled around it | 18/18 | 13/18 | 5 (5) / 27 | 0 | 19.8 / 18.4 | 6.5 | yes | yes |

In detail:

- **A. Exact int64 fixed point.**
  - *Tail sums:* the sorted measures go on the grid `2**-shift`, with `shift = 62 - e(max a) - e(n)`, where `e` is the
    frexp exponent. That makes every integer `< 2**62 / n`, so the int64 sums are exact in any order. They convert back
    to float64 with one rounding.
  - *Line:* each cell spans `q_i = round(pi_i 2**Q)` units, `Q = 62 - budget.bit_length()` (44 at 200,000). The line
    is the int64 cumsum. The start is `U = floor(u 2**Q)`, and hit `m` is `U + m 2**Q`.
  - *Exact properties:* a `pi = 1` cell spans exactly `2**Q`, so it holds exactly one hit. No cell spans more than
    `2**Q`, so no cell is drawn twice. A hit on a boundary takes the later cell (half-open intervals), on every device.
    The order of summation cannot matter, so the result is the same on CPU, on GPU, and with any scan algorithm or
    compiler.
  - *Inclusion probability:* the drawn probability is `q_i / 2**Q`, within `2**-45` (3e-14) of `pi_i`. The weights stay
    `a_i / pi_i`.
  - *Function change:* the 5 changed cells at the eval start are where today's CPU float line was itself off. In 4 of
    the 5 cases (186, 087, run_1, run_2) today's own GPU result differs from today's CPU result by the same one cell;
    in cases 0 and 050 today's GPU result varies from run to run, so the comparison does not apply there. In case 0 a
    tie now resolves by exact arithmetic, and the GPU duplicates go to 0 in all 6 runs of cases 0 and 050.
  - *Cost:* +1 to +1.6 ms at 17.7M cells (+5 to 8 %), about +0.8 ms on AeroZoo, no extra memory, and no host sync.
  - *Compile:* Inductor folds `arange(k) * 2**44` into an int32 Triton index expression. Triton rejects it: "Scalar
    17592186044416 is out of range for type int32" (377520, forms f1 to f5). The search is therefore rewritten without
    that product: `count = (line - U - 1) >> Q`, then `searchsorted(count, arange(k))`. This is exactly equivalent:
    `line_j > U + m 2**Q` holds if and only if `(line_j - U - 1) >> Q >= m`. It was checked bitwise against the direct
    search on CPU, and it compiles on CPU, bitwise with eager. That Inductor bug is a candidate PyTorch issue; the draft
    is for Peter.
- **B. A CPU cumsum at a host sync.**
  - It reproduces today's CPU function exactly, so the function change is 0.
  - It costs 2.3 times the draw at DrivAerML size (+26 ms per forward) and adds a device-to-host and host-to-device
    copy with a sync per scan.
  - It keeps the float line's own defects: arbitrary tie sides, and drift. On CPU the offset error at an exact tie
    can change sign at a binade crossing (inferred, not observed here), which would give a duplicate.
  - It is a poor fit for the library move (LIB-MOVE row 9): a generic `systematic_sample` should not round-trip to
    the host.
- **C. A fixed-order GPU scan (Triton).**
  - It is deterministic on one GPU and one Triton version, and speed is unchanged.
  - Its in-block order comes from `tl.cumsum`'s implementation, so its result is defined only by a Triton version. It
    matches the CPU in only 13 of 18 cases.
  - It keeps the float line's tie and drift defects (duplicates at exact ties inferred, not measured for C), and it
    adds a kernel to maintain. No well-defined result: rejected. Its 13/18 CPU = GPU count partly reflects the
    sequential CPU sum's own drift, not an error of C.
  - Warp's `array_scan` calls `cub::DeviceScan` on CUDA (`warp/native/scan.cu`), the same family as ATen's, so it
    was not built.
- **D. The deterministic ATen scan.**
  - Deterministic, but it toggles process-wide state around one op, which is not thread-safe.
  - CPU differs from GPU, and the float defects stay. Rejected. The global switch was already rejected by TEST-AUDIT.
- **Considered, not built:**
  - **(E) A plus exact condition offsets.** An integer water fill would make the line total exactly `k 2**Q`, so a tie
    at a condition boundary would follow the half-open rule exactly instead of the sign of a rounding residual of
    about 1e-10 tokens. A already makes that residual deterministic and the same on every device, and either side of
    an exact tie is a valid systematic sample. E adds an integer water fill for no measurable gain.
  - **(F) An evaluation start other than 0.5**, such as `(sqrt 5 - 1)/2`. It would avoid exact ties at half-integer
    shares, but it changes every evaluation draw, and the float scan would still be non-deterministic at near-ties.
    It is not needed with A.

## Recommendation: A

A is the only option that is exact: deterministic run to run, the same on CPU and GPU given the same float64 inputs,
the same with any scan or compiler, sync-free, and with the draw's invariants (each `pi = 1` cell exactly once, no
duplicates) holding by construction. It costs about 1 ms per forward at DrivAerML size.

Its function change against today's CPU is one kept cell in 5 of 18 evaluation draws and 0.05 cells per random draw.
That is no more than today's own CPU-to-GPU difference. Weights move by at most 1e-15 relative in float64, which
vanishes in the float32 token measures. In outputs, one changed token of 200,000 moved AeroZoo case 0's outputs by
2.8e-6 to 8.8e-6 max abs (REYNOLDS-OUT 376855), far below the two-seed spread.

B is the only zero-change option, but it pays +26 ms and a sync per forward to keep a float line that is itself wrong
at exact ties.

Not settled by this design: whether the draw's *inputs* (`draw_inputs`, the canonical frame) are bitwise the same on
CPU and GPU. They are not: computed on each device, the float64 draw inputs and the frame differ by ulps
(`draw_inputs_bitwise_cpu_vs_dev` and `frame_bitwise_cpu_vs_dev` false in all 18 cases; the probe does not split
measures from centroids). The Morton order was nevertheless equal
on all 18 cases. So "CPU = GPU" holds for the draw given the same float64 inputs, not end to end from the mesh.

## Amendments after the design review (REVIEW_DESIGN.md, lane-reviewer, 00:47 EDT 10-25), before the build

The review found one blocking flaw in A as first written and two exactness gaps. The writer was steered at about
00:55 EDT, before writing the corresponding code; the numbers above are unchanged (they came from the probe version
with `round`, on float32-derived measures, where the flaw does not show).

1. **Blocking, fixed: floor, not round, for the quantized measures.** With `round`, full-precision float64 measures
   of a condition kept whole (`share >= n`) could give the smallest cell `pi < 1` (its rounded tail above its own
   measure); the existing test `test_inclusion_probabilities_are_capped_proportional_to_size` fails, and a draw with
   every cell kept then duplicates one cell and drops another. With `floor` the integer tail at rank r is at most
   `(n - r) a_r 2**s`, and int-to-float rounding and power-of-two scaling preserve order, so every `pi = 1` when
   `share >= n`, as today (reviewer: 320 CPU cases, no failure).
2. **Tail-grid precision: two passes.** A single grid `2**-(62 - e(max) - e(n))` has relative error up to
   `n (max/mean) 2**-62` (reviewer: `sum pi - share` = 6.9e-8 on lognormal measures, against 2.9e-11 in float). The
   build uses a second pass whose grid comes from the exact integer sum of the first (`shift = s1 + 61 - e(S1 + n)`),
   so the sums stay below `2**61` and the precision is about `n 2**-60` relative whatever the skew. So "≤ 1.0e-15
   rel." above holds for the float32-derived inputs measured; for float64 inputs the bound is about `n 2**-60`.
3. **Exponent range.** `2**shift` is built in two exact steps, so measures below about 1e-290 no longer overflow the
   bit-built power of two (relevant for the generic utils version; ISLA's measures are never that small).
4. **The start clamp, so "each `pi = 1` cell exactly once" is exact.** The integer line total `T` differs from
   `k 2**Q` by a residual (rounding of each `q_i`, float error of `sum pi`). Clamping `U` into
   `[max(0, T - k 2**Q), min(2**Q - 1, T - 1 - (k-1) 2**Q)]` puts every hit in `[0, T)` and one hit in every window of
   length `2**Q` inside it whenever the residual is below `2**Q` (reviewer's construction: sizes [5000, 40], budget 300,
   `u = 9.6e-13` kept 39 of 40 whole cells before, 40 of 40 after). Today's code has the same defect at starts within
   about 1e-9 of 0 or 1. It covers the line-total part of option E without an integer water fill.
5. **Corrections to the text** (applied above): the CUDA compile of the rewritten draw is unverified until the final
   GB300 job; the 5-case function-change comparison holds for 4 cases; B makes one round trip per condition plus one;
   C's and D's duplicates at ties are inferred.
6. **A′, not listed before:** round `pi` and the start to a dyadic float64 grid (`2**-35` at this budget) so every
   partial sum is exact in float64 and ATen's float scan is exact in any order; hits stay `start + arange(k)`. Simpler,
   but 2**9 coarser on the line, and the tail sums lose the same 9 bits (worse on skewed measures). A keeps the extra
   bits; the review agrees A is a reasonable pick.
7. **Notes for the generic version (LIB-MOVE row 9):** `Q` is fixed from the budget, so `budget` must be a Python
   int (a SymInt `num_samples` would be specialized); inf and NaN measures stay undefined, as today.

## Amendment 2 after the diff review (REVIEW_DIFF.md, 02:21 EDT 10-25): the tail sums in a fixed-order float scan

The diff review of the first build (7a3f91f20) found that fixed point is the wrong tool for the *tail sums*: on a grid
set by the condition's total, positive measures below one unit floor to 0, and with share below n the cap search then
runs past the share (`(share - r) a < 0`), giving negative pi and 18 unique cells of 300 (finding 1); and the tail's
precision is relative to the total, not to the tail at m (finding 3: 19.6 tokens of error in sum pi on a pathological
spike distribution). Both come from fixed point's absolute precision. The line keeps exact int64: its quantity is
absolute by nature (positions on a line of length k), and its invariants need exact arithmetic.

**Option G for the tails:** the `_cumsum` op scans floating dtypes by doubling (Kogge-Stone): `log2 n` passes of
elementwise correctly rounded adds in a fixed tree. Elementwise IEEE adds give the same bits on CPU and CUDA and run to
run, so the tails, and through them `c`, `pi` and the weights, are bitwise reproducible on every device, with float
relative precision (about `log2 n` ulps, better than a sequential sum) and no grid. A CPU check of a quick version:
the reviewer's sliver and spike constructions give exactly the base's values; the branch's tests pass. Cost: about
25 passes over 17.7M float64 values on the GB300 (to be measured in the round-2 job; estimated 2 to 3 ms).

**A defect of today's code found on the way:** float tail sums cannot guarantee the documented "share at least the
cell count gives every pi = 1". With equal measures (uniform cells, common on box faces), the base gives pi = 1 - ulps
in 26 of 244 tested cases (pi down to 1 - 5.5e-12), so a condition kept whole is not exactly whole and its HT weights
are off by ulps. The build adds the explicit rule `pi = 1` when `share >= n`.

The writer was relaunched for round 2 at about 02:40 EDT with BRIEF_WRITER2.md (md5 6e0282d1).

## As built and checked (03:25 EDT 10-25)

Branch `psharpe/isla-draw-exact`, head **5b5e72c2f** (pushed): 7a3f91f20 (A, round 1), 18fdcd7c0 (tails in the
fixed-order float scan, `pi = 1` when `share >= n`), 02d5451fd (merge of psharpe/isla b86c7391d, ISLA 0.13.0, one
docstring conflict), 5b5e72c2f (tests: the float scan's order pinned on CPU). No version bump.
- Tails: `_cumsum` on floats is a doubling (Kogge-Stone) scan of elementwise adds; on integers ATen's exact cumsum.
- Line: exact int64, `Q = 62 - budget.bit_length()`, start clamped by the line's residual, search by `>> Q`.
- Diff review round 2 (REVIEW_DIFF2.md): OK with notes; both notes fixed in 5b5e72c2f, except the pre-existing
  exact-zeros case (a condition with more than n - share cells of measure exactly 0 gets sum pi = n, as on the base),
  left for the utils move.
- GB300 (378892, head 5b5e72c2f): whole ISLA directory 716 passed, 1 skipped, 2 xfailed, 0 failed; det_probe `dtt`
  5/5 forwards bitwise at every stage; head_check on 6 real boundaries: 6 runs bitwise, 0 duplicates, every pi = 1
  cell kept, CPU = GPU bitwise, compiled (fullgraph, dynamic) = eager bitwise with no recompile.
- The new CUDA tests fail on the base (378518): bitwise reproducibility and CPU = CUDA.
