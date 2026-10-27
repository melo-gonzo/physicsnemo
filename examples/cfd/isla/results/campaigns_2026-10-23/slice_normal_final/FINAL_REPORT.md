Status: DONE

I read every arm. Neither fix is a candidate for adoption. Option (a) harms DrivAerML Surface (option A), and (a)+(s) harms DrivAerML Surface and the lug. The reference normalization stays. Written at 00:08 EDT on 10-27. Details are in `RESULTS.md`, jobs and md5s in `JOBS.md`, and the book text in `NOTEBOOK_DRAFT.md`.

**PREREG.** `PREREG.md`, md5 aac40afd, was written before any job and sent at about 19:56 on 10-24. It holds the draft verbatim plus A1-A5: LUG joins the core, launch now, AF-WHOLE stays blinded, harms block while flags do not, the cap is 720, and the verdict rule. Part F was added before any S job, giving md5 **4580615d**. It records A6, the (a)+(s) arms with the cap raised to 1,440, and your two rulings. At about 20:21 you ruled that C4 fails for the standing and the arm alike, so the lug trains compiled. At about 20:23 you ruled that DR-A-S may force the eager geometry region, since (s) has no fused kernel.

**Launches (LHR).**
- Smokes 374067-9 and 374632-5 passed. The m_s check was bitwise in 240 of 240 blocks, in bf16 and fp32, in all six.
- 14 lanes:
  - DR-A: 374352, 374386, 374415, 374439
  - AF: 374459, 374496
  - LUG: 374544, 374686
  - DR-A-S: 374955, 374980
  - AF-S: 375015, 375043
  - LUG-S: 375067, 375212
- Scorers: 379296, 401974, 404046, 404777, 416321, 416322.
- I made no commits and wrote nothing to GitHub. The only code change is the one registered line, plus blocks_ad.py and the one F3 line for DR-A-S.

**Reads (native; ratio = arm ÷ control).**
- **DR-A (a):** pressure 1.167 and wall shear 1.121, both **harms**. The arm is worse on 48 of 48 cases.
- **DR-A-S:** 1.422 and 1.297, both **harms**. The marginal effect of (s) over (a) is 1.219 / 1.157. Eager against fused differs by 1-2 % in bf16 and 1e-5 in fp32, so the forced eager region cannot explain this.
- **AF (a):** 1.076 / 1.153 / 1.234, no change. **AF-S:** 1.294 / 1.196 / 1.457, no change, but every field has an up flag. The marginal effect of (s) on pressure is 1.202. The control's seeds are 31-39 % apart, so these AF reads have low resolution.
- **LUG (a):** 1.199, no change, with an up flag. **LUG-S:** 1.263, **harms**. The marginal effect of (s) is 1.054. Beside the rule: (s) doubles the lug's posed ÷ native (4.59 / 3.25 against 2.42 / 1.86 for (a)).
- **Verdicts:** neither (a) nor (a)+(s) is a candidate. Every point estimate of (s) is a cost.

**GPU-h:** 1,379.9 by sacct, against the cap of 1,440 (720 before A6). No job is left.

**Findings for the freeze.**
1. The restated 'Compiled regions' rule fails on the **standing's own** compiled slice blocks on the lug: 86 of 1,200 tensor groups fail, and fp64 compiled-vs-eager reaches 4e-11. Unless the final code certifies, the lug trains eager in the final rerun. With (s), only 14-24 groups fail, and fp64 is at roundoff (RESULTS §2).
2. If (s) were ever adopted, the surface task would lose the fused kernel: the step is 3.28x slower (0.471 → 1.547 s at 205k tokens).
3. Beside the rule, not a like-for-like comparison: the option-A control scores 0.0524 / 0.0803 against the vehicle-only 0.9.0 standing's 0.0412 / 0.0662. Code, training path and boundaries all differ.

**For you to decide.**
- Whether to start a (c) lane: a smooth floor at bf16's rounding level. `C_DESIGN.md` has the derivation, the bound (gain at most 128√d) and a staged test: DR-A-C first at about 65 GPU-h, then AF-C and LUG-C at about 550. Its constant u = 2⁻⁸ is Peter's to accept under the no-magic-constants rule.
- Whether to prune the 14 run dirs on LHR to their final checkpoints. They hold about 1.8 TB (about 130 GB each, 500 checkpoints, as the controls do). I deleted nothing.

**For project memory.** The lug's compile_blocks fails the restated certification on the standing's code. Your 20:21 ruling covers this lane only; the freeze decides the final-rerun path.

**On disk.**
- LHR: snapshots `code_slicenormal{,s}_{fix,surfbc}` (about 57 MB each); `kernel/slicenormalx` (8.9 GB, mostly caches and smoke outputs); `runs/slicenormalx_*` (about 1.8 TB); `evals_slicenormalx`.
- Here: this folder (11 MB).
- No worktree or venv was made.
