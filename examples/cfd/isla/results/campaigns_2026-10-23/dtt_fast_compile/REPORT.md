Status: DONE

The results, tables and patches are in `scratch/isla-dtt-fast-2026-10-23/compile/`. Read `RESULTS.md` first.

**What ran.** Nine LHR jobs, all named `isla.dttfast.compile.*`, are listed in `JOBS.md`:
- GPU jobs: 363857, 364478, 364792, 365108, 365234, 365355 (failed at an import), 365375 and 365445. They used 4.79
  of the 6 GPU-h cap.
- CPU job: cpu-short 365063 ran the whole ISLA test directory.
- The probe is `lhr/cx.py`, which overlays the extensions at run time and calls the reference functions unchanged. The
  real-step timer is `lhr/timed_cx2.py`, run under the recipe's `train.py`.
- I made no commits, as the brief says. The code change is a patch file.

**Results.**
1. **The map.** `compile_blocks` compiles only the 12 slice blocks. The whole read runs eager: forest, walk, plan,
   pair invariants, basis MLP, `node_means`, LayerNorms with the q/k/v and output projections, the kernel term and
   the segment softmax. So do the geometry call and the head. PROFILE's device times for each region are in section
   1 of `RESULTS.md`.
2. **The current compiled path, certified** on the AeroZoo ISLA-DTT s42/s43 and AirfRANS `r090eag_af_dtt` s42/s43
   checkpoints, block by block at full size, plus gradients at 20k (section 2):
   - In float32, blocks match eager to 1e-5 relative or better.
   - In float64, every output and gradient is at roundoff (2.3e-13 relative or better).
   - In bf16 the ratios to float64 match eager's (1.0 to 1.4) everywhere except two cases. On AirfRANS s42 in
     float32, the current path is at 2.0 to 2.3 and 3.4 on the loss: eager sits unusually close to float64 there,
     so 2 to 3 ulps of difference doubles the distance.
   - The current `compile_blocks` elides bf16 casts (Inductor's default), so it does not keep eager's bf16 rounding
     points. Bf16 block outputs differ from eager by up to 5.8e-2 relative; with `emulate_precision_casts` they
     differ by 5.0e-3.
3. **Extensions.** P (the read's projections), M (`node_means` as three graphs that are reused for any forest depth)
   and B (invariants and basis MLP), compiled with `dynamic=True` and the precision casts kept.
   - Model step against the current path in the same process: DrivAerML vehicle 200k −4.5% (181.5 to 173.3 GiB),
     AeroZoo −6.6% (157.4 to 149.7 GiB), AirfRANS −3.7%, lug −3.1%.
   - Real recipe steps, base and patch side by side: −6% (DrivAerML), −10% to −14% (AeroZoo), −3% to −14%
     (AirfRANS).
   - Over 60 real steps on each dataset, no graph was added after step 10.
   - The certification holds wherever the current path's does (table in `RESULTS.md` section 2). Pair lists are
     equal in every case.
4. **Patches.**
   - `compile_read.patch` (md5 c9aaacba). Snapshot `$T/code_dttfast_compile_p1`: the tree-read tests give 36 passed;
     `patchcheck.py` passes; the whole ISLA test directory gives 321 passed and 46 skipped on CPU.
   - `block_precision_casts.patch` (md5 ee940cef).

**What it means.** Compiling the read's dense regions is a safe, small gain: about 5% of the step and a few GiB.
The large items stay with the other pieces: the segment-softmax backward and the per-pair `ker` (FUSEDREAD), the
forest and walk (FOREST/WALK), and the geometry call (GEO-FAST). I am confident in the in-process numbers, which
agree within 3% across jobs. Real-step numbers vary 10 to 15% from host contention on a shared node.

**Decision for the parent: keep the current compiled blocks or restore eager's bf16 rounding points?** The
rounding-point finding concerns code that already exists. Today's blocks meet the accuracy bound but do not round
where eager does. Restoring the casts is free on 3-D cases but adds 11 to 14% on AirfRANS and 5% on the lug, and it
changes the function that every checkpoint was trained under. I recommend leaving the blocks as they are and
recording this in the program's notes.

**Rulings relied on.** CHILD_RULES: equivalence standard, no worktree or commits for this piece. LANE_RULES:
LHR-only GPU work, the 00:45 rule sending test suites to cpu-short.

**New rule for project memory.** Inductor's default `emulate_precision_casts=False` moves bf16 rounding points. Any
new compiled ISLA region should set it to True, as this patch does.

**What is left.**
- Apply `compile_read.patch` in a worktree.
- Add a CUDA test comparing `compile_blocks` on and off (`lhr/patchcheck.py`'s body).
- Optionally certify on a trained DrivAerML ISLA-DTT checkpoint, which does not exist yet.

**Nothing is running.**

**On disk.** The scratch folder is 3.2 MB. On LHR: `$T/code_dttfast_compile_p1` (57 MB, read-only) and the launcher
directories `$T/kernel/dttfast_compile_*` (under 1 MB each, caches removed).
