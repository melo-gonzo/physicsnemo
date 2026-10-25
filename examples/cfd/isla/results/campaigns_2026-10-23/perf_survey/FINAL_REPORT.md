Status: DONE (2026-10-25 10:46 EDT)

**What ran.** Eight children (research-lane, devin/claude-opus-5.5:high; `CHILDREN.md`). On LHR GB300 (short QoS, the
five bad nodes excluded): 39 jobs `isla.perfsurvey.*`, **9.24 GPU-h of the 12-h cap** (by sacct). Nothing is running,
and the registry row is removed. One prototype branch, local and not pushed: `psharpe/isla-perfsurvey-fusedpad`
80a58e1e8 (2 commits on b86c7391d). On it the whole ISLA directory passes: CPU 453 passed, 0 failed (382714); GB300
with every CUDA test 735 passed, 0 failed (382761).

**Profile** (`profile/PROFILE.md`, job 382020; bf16, the recipe's configs, one GPU, median of 10 steps)

| case | step ms | peak GiB | biggest regions (share of step) | GPU idle (ms) |
|---|---|---|---|---|
| DrivAerML Volume, whole boundary, 200k | 1,190 | 143 | geometry 31 %, blocks + point reads 26 %, fused read 18 %, basis 8 %; real loader +105 ms exposed | 37 |
| AeroZoo, 200k | 716 | 126 | blocks + point reads 46 %, geometry 13 %, fused read 10 % | 73 |
| AirfRANS with far field, compile_blocks / eager | 530 / 895 | 27 / 33 | point reads 59-63 % (2-D runs the eager geo region), geometry 11 % | 106 / 211 |
| lug, compile_blocks / eager | 234 / 394 | 5 / 7 | host-bound: GPU busy 42 % / 31 % | 135 / 272 |
| DrivAerML Surface, option A | 246 | 66 | slice blocks 75 % (memory-bound; GEMMs are 1/4 of block time) | 26 |
| 4-GPU DDP, DrivAerML | 1,244 (1,360 with the loader) | 143 | all-reduce after backward (0 overlap) + rank waiting in the size rule's all_reduce(MIN) | 77 |

- Geometry holds 92 of DrivAerML's 101 host syncs.
- Every case peaks in the chunk's backward. The MLP activations are 31 % of DrivAerML's peak.
- The optimizer has 3-4 ms of device time but 750-850 launches, and starves the GPU 25-70 ms on the small cases.

**Ranked opportunities.** All 15, with share, gain, exactness class, memory, lines, home, risk and fit, are in
`OPPORTUNITIES.md`. These five are the ones I would bring to Peter:
1. **FUSED-PAD (#1).** 2-D models and models without a global direction (AirfRANS, the lug, Tensile2d) take today's
   fused geo_kernel, with exact zero padding, instead of the eager reference.
   - Measured step: AirfRANS 0.59x compiled and 0.42x eager; lug eager 0.8x; lug compiled neutral; memory a little lower.
   - Certified within rounding for eager blocks. No new kernel; lines about net zero in the model.
   - Why Peter: it closes a generality gap with code he already approved.
2. **int64 offsets in geo_kernel (#2).** A correctness bug: ISLA's `forward` faults once tokens x slices reaches
   2^31, which a full AeroZoo case does. The fix is bitwise below the limit and costs about +1 ms of a 1,190 ms step.
   Why Peter: correctness comes first. The supervisor was told at 09:08.
3. **Geometry G2 + G3 (#4), in physicsnemo.mesh.spatial.**
   - G2, a ball search in one traversal: DrivAerML's call 400 to 347 ms, about -4.5 % of the step.
   - G3, lean buffers: geometry peak -29 %. A boundary the size of HiLift's wing (272M cells) then fits at 202 GiB
     instead of running out of memory.
   - Both bitwise. Why Peter: exact, mostly removals, fixed upstream so GLOBE gains too.
4. **Sync-free FlipBoundaryOrientation (#3), recipe.** Measured -24 ms per step on DrivAerML (-1.9 %), bitwise, one
   line. Dropping the sdf ISLA never reads saves no time, but is a removal.
5. **Interior `forward` through the read path (#5).** Evaluation then computes what training computes, with one
   interior path and fewer lines.
   - Measured per forward at 200k: -8 % DrivAerML, -12.5 % AeroZoo. Memory is flat in points (2M points: OOM
     becomes 21 GiB).
   - Exactness: certified on DrivAerML. On AeroZoo float64 differs by up to 1e-5, from the known octant-tie flips.
- Also exact and small: the chunked step's flat-buffer all-reduce (#6; lug 4-GPU -10.5 % measured), Muon as one CUDA
  graph (#7; lug -9 % measured, 0 on AeroZoo, +0.6-0.9 GiB reserved), the read kernel tweaks and the `log_m` skip
  (#10, about 1 %), and real step-time logging (#11; the logged `Step:` is host time, up to 1.37x short on HiLift).
- In flight:
  - 128 slices adds about -105 ms (-9 %) on DrivAerML and makes #13 moot.
  - Part P: -29 ms and -7 to -9 GiB; Peter's open decision.
  - DTT-KNOBS K-B1: about -130 ms (estimated), a numerics change. K-BASIS8 first needs a `basis width >= 16` guard,
    or the fused read fails to compile.
  - LIB-MOVE is cost-neutral and is the home of #4, #8 and G7-G9.

**Cost to build and certify.**
- #1 + #2: already built and tested.
  - Left: the compiled-regions certification on real AirfRANS, lug and Tensile2d checkpoints (at least 14 samples
    over 2 cold compiles; about 1-1.5 GPU-h, half a day).
  - A merge with the current psharpe/isla. Conflicts are mechanical in the tests, plus `resolve_geo_kernel` against
    psharpe/isla-kernel-bugs. Then a GPU suite run (about 0.5 GPU-h).
  - Not yet checked: whether fused_read, fused_basis and segment_softmax have the same int32 issue.
- #3 and #6: the patches exist (`system/patch`, `system2/patch`). About 1 h each, plus a no-sync GPU test and the
  recipe tests.
- #4: probe code exists (`geo/lhr`). Half a day to build in the library with or after LIB-MOVE's geometry move, plus
  the CPU and GPU suites (about 1 GPU-h).
- #5: half a day plus test churn; float32 certification on DrivAerML.
- #7: +47 library lines, a unit test, and a 4-GPU A/B (about 0.3 GPU-h).

**Needs Peter's decision:**
- (a) Whether FUSED-PAD and the int64 fix go in before the final rerun. FUSED-PAD roughly halves AirfRANS' training
  cost, but changes the kernel AirfRANS, the lug and Tensile2d train on, so it needs the certification first.
- (b) Part P (open).
- (c) #5, given the AeroZoo float64 bound of 1e-5.
- (d) Whether Muon's +0.6-0.9 GiB reserved is acceptable for #7.

**Rulings relied on:** Peter's screen as briefed; LANE_RULES on compiled regions, the bf16 yardstick, execution
choices from sizes, separate processes per version; the boundary-only ruling (no cross-call cache); "fix upstream".

**For project memory:**
- The recipe's `Step:` / `step_time_s` is host time; cost plans should use epoch seconds divided by steps.
- `geo_logit.bias` has an identically zero gradient (#14, cleanup).
- Host-bound timings (the lug) mis-read by up to 1.7x while another process compiles with Inductor on the same node.

**Process note.** Several children's need_decision messages (extra GPU-h) did not reach me while I waited in
blocking commands. FUSEDPAD went over its job-count limit (4 jobs, 1.18 GPU-h, inside its GPU cap), and SYSTEM's
real-loop A/B ran later as SYSTEM2.

**Left:**
- The certification and merge of #1 and #2.
- A 4-GPU DDP A/B of #7.
- What holds DrivAerML's remaining 47 ms of exposed loader time.

**On disk:**
- Worktree `/home/psharpe/gh/physicsnemo-isla-perfsurvey-fusedpad` (1.1 GB; keep until #1 and #2 are decided).
- This folder (11 MB).
- On LHR:
  - `$T/kernel/perfsurvey/` (about 115 MB);
  - read-only snapshots `$T/recipe_perfsurvey_b86c`, `$T/code_perfsurvey_system{,2}` (58 MB each),
    `$T/recipe_perfsurvey_system*`, and `$T/kernel/isla_tests/snap/fp80a58{cpu,gpu}`. These can go once the
    candidates are decided.
