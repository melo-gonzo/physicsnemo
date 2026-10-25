# Where ISLA's eager GPU forward is and is not run-to-run deterministic (TEST-AUDIT, 23:37 EDT 10-24)

Asked by the supervisor at 23:28, after REYNOLDS-OUT found that 0.9.0's eager AeroZoo GPU forward does not always
repeat before block 0 (`scratch/isla-reynolds-out-2026-10-24/bitwise/RESULTS_BITWISE.md`).

## What ran

- Job **377000**, `isla.testaudit.det` (LHR GB300, nvl72d040-T01, 2 min 42 s, 0.18 GPU-h). The cap was raised to
  2.7 GPU-h for this one job by the supervisor's reply at 23:31; the lane's total is 1.95 GPU-h.
- Code: `code_mainmerge_base` (5b70c8686, ISLA 0.12.0), torch 2.12.0+cu130.
- The probe is `probes/det_probe.py` (md5 3c71cfec), launched by `lhr/det.sbatch` (b7dcfb08). It was first tested
  locally on CPU under the lock. Raw output is in `results/det_377000/`.
- Each configuration: one model (hidden 64, 3 layers, 32 slices, read randomized), one synthetic AeroZoo-sized
  input: 250k cells over 4 named conditions, 20k prediction points, a 100k-token draw. Forwards in float32, eager,
  eval mode, `torch.manual_seed(0)` before each, 6 forwards.
- Hooks compare every tensor at every stage bitwise against forward 0:
  - `draw_inputs`;
  - `sample_boundary_cells`: `[0]` kept cells, `[1]` weights;
  - `exact_boundary_geometry`;
  - the boundary tokens;
  - each tree read;
  - each block's input and output;
  - the outputs.

## Results (the first stage that differs, per configuration; 5 repeats each against forward 0)

| config | first differing tensor | what differs | outputs |
|---|---|---|---|
| `dtt`: interior ISLA-DTT, kernels auto (geometry fast, fused read, fused geo region) | `sample_boundary_cells[1]`, the HT weights, in all 5 | weights differ by up to 2.2e-19 (float64, a few ulps); **kept cells bitwise**; tokens bitwise once the weights are cast to float32 | **bitwise** (5/5) |
| `dtt_ref`: every kernel forced to the reference or eager | `sample_boundary_cells[1]` (4 of 5), then `read0.out` | the same weights, plus the **reference tree read** differs by 1.7e-6 to 2.4e-6 from read 0 onward (blocks up to 3.8e-6) | not bitwise: 7.2e-7 to 1.0e-6 |
| `dtt_whole`: every cell kept (budget > cells) | none | — | bitwise (5/5) |
| `surface`: the surface task (no draw, no tree read) | none | — | bitwise (5/5) |
| `dtt_det`: `dtt` under `torch.use_deterministic_algorithms(True, warn_only=True)` | none | — | bitwise (5/5), the weights included |

## Reading

1. **The draw's float64 sums are not run-to-run deterministic on CUDA.** `sample_boundary_cells` builds the
   inclusion probabilities and the systematic line with `_cumsum` (`sampling.py:71-82`, `x.cumsum(0)`, used at
   `:127` and `:194`).
   - By default, CUDA's floating-point `cumsum` is not deterministic. Under `use_deterministic_algorithms` torch
     switches to a deterministic scan, and the draw becomes bitwise. With every cell kept (π = 1, an integer line)
     the draw is also bitwise.
   - The ulp-level wobble of `line` moves a kept cell only when a hit `start + j` lands within the wobble of a
     cell's boundary on the line. That flip changes one token entirely: an O(1) change in block 0's input at one
     row, hidden states far apart there, and a small change in the outputs.
   - That matches REYNOLDS-OUT's AeroZoo case 0 exactly: block 0's input off by 0.65 and 5.45, outputs off by
     3e-6 to 9e-6, intermittent from case to case.
   - **This is the likely cause, inferred, not shown on that case.** In my synthetic case the kept cells never
     flipped in 5 repeats. The cheap confirmation is REYNOLDS-OUT's `reyout_repeat.py` with one more hook on
     `sample_boundary_cells`' `kept` (CPU-tested scripts exist; about 0.2 GPU-h).
2. **The reference (unfused) tree read is not deterministic on CUDA.** It differs at the first read by about 2e-6
   relative. The likely source is `node_means`' float `index_add` (`tree_read.py:515`), whose CUDA atomics have no
   fixed order (inferred: the probe locates the read, not the op).
   - The fused read is bitwise, together with its node means: those are Triton kernels with a fixed order, and
     the fused backward is already tested to be deterministic.
   - REYNOLDS-OUT's AirfRANS DTT floor (2e-6 from block 1 onward) has the same signature. 0.9.0 predates the fused
     node means, or 2-D took the reference path; which of the two is not checked.
   - On CUDA the reference read runs in float64, and wherever the fused read is unsupported: `unsupported_reason`
     lists widths that are not a power of two, fp16 without the autocast cache, and missing Triton.
3. **Everything else measured is bitwise run to run on GB300 eager:** the fast exact geometry, `draw_inputs`, the
   kept cells, the tokens, the fused geo region, the fused read, every block, and the surface task.
4. Not measured: 2-D, bf16 autocast, training mode (the draw's start is random then, so it is seeded), backward,
   `compile_blocks`, and the real AeroZoo and AirfRANS cases.

## Proposed tests (in TEST_PLAN P3-11, GPU tier)

`test_isla_determinism.py`, `@pytest.mark.cuda`:
1. `sample_boundary_cells` on CUDA, run twice on about 300k float64 cells over 4 conditions: `kept` and `weights`
   bitwise.
   - **It fails today** (weights). Mark it `xfail(strict=True)` until the draw is fixed, so that the fix flips it.
   - Also plant hits within 1e-12 of line boundaries: `start` chosen from `line` itself, which forces the
     boundary case.
2. The whole model, run twice in eval on CUDA, kernels auto, draw active, 3-D and 2-D, surface and DTT, float32
   and bf16 autocast: every block output and the outputs bitwise. Record which kernels resolved.
3. Forward and backward run twice, with gradients bitwise for the fused paths.
4. The reference read on CUDA: compare the run-to-run spread over 5 runs against a stated bound (about 1e-5
   relative in float32), until it is made deterministic or its non-determinism is accepted and documented.
5. On CPU (fast tier), one thread: the whole model twice, bitwise. This is the baseline the cleanup's A/B harness
   assumes.

## A deterministic mode, and how to read the goldens

- **No global deterministic mode.** `use_deterministic_algorithms` is process-wide. It errors or warns on other
  ops, slows cuBLAS (`CUBLAS_WORKSPACE_CONFIG`), and would hide non-determinism that users meet by default.
- **Fix the two sources in ISLA instead**, so the default eager GPU forward is bitwise:
  - (a) **The draw.** Take the cumulative sums in exact integer fixed point. Quantize π to multiples of 2^-44
    (line ≤ 2^18 tokens, so the total stays below 2^62), take an int64 `cumsum`, which is exact and so the same in
    any order, and compare against integer hits.
    - This also makes the draw exact rather than float-rounded, and pose-independent of summation order.
    - It changes which cells are kept wherever a hit is within 2^-44 of a boundary. So it is a numbers-changing
      model change, outside any bitwise cleanup: a decision for Peter and the supervisor.
    - Alternative without changing numbers: run the two float64 cumsums on CPU. That costs a host round trip of
      about 2 × 8 MB per case at 1M cells, plus a sync. Or call a deterministic float scan if torch exposes one.
  - (b) **The reference read on CUDA.** Leave it as is: it is the CPU and float64 reference, and the production
    float32 and bf16 path is the fused read, which is deterministic. Document it, and keep test 4.
- **Reading goldens:**
  - **CPU: bitwise** (one thread, same platform: the A/B harness).
  - **GPU, kernels auto: bitwise**, for every stage this probe found bitwise, which today is everything except the
    draw's weights. Hash `kept`, not `weights`, until fix (a) lands; compare `weights` at 1e-15 relative.
  - **GPU, reference read: within the run-to-run family.** Record the stage 5 times in-process and require the
    new run's distance to the reference to lie inside the family's range, as in the bf16-equivalence rule.
  - **On real AeroZoo-like inputs before fix (a):** a kept-cell flip can occur. The A/B harness must compare
    `kept` first. If `kept` differs between base and head but each side flips run to run, the comparison is
    inconclusive and must be rerun, not passed.
