Status: DONE (02:49 EDT 10-25)

**What ran.** Branch psharpe/isla-geo-tieorder: commit 0f26e6db7, then merge 3a1a15d48 of psharpe/isla b86c7391d (0.13.0). Both pushed with a plain push, not merged, no version bump. `git merge-tree` is clean against psharpe/isla and isla-api-013. GB300 jobs: 377610 (prototype probes), 377673 and 377915 (timing), 377708 (inference, a child), 378186 (final checks), 378187 (timing), 378687 (suite after the merge). cpu-short: 378185, 378676 (suites) and 378382 (HiLift). Total 6.51 GPU-h, inside the cap the supervisor raised from 6 to 7.5. Details are in JOBS.md, DESIGN.md (v2) and RESULTS.md.

**Mechanism.** Two causes, both in `_closest_candidates_impl` (geometry.py at fc6ed19d0):
- **Order (lines 1951-1967).** Cells that share the closest point compute it bitwise, so they tie exactly in float64. `first` then takes the least cell index, and that index is the input order. The order reaches:
  - the untied on-wall `u = n64[idx]` (lines 2142, 2248);
  - the on-wall label, through `err[first]`;
  - the off-wall G5 rebuild.
- **Precision (line 2050).** `_certified_tied`'s bound scales with `u_in`. In float32 the near-equal normals are not certified, so u is one normal. In float64 they are certified, so u is their midpoint.
- Both causes act in the reference and in both fast paths, since all three share the post-search code. Both also act on 2-D vertices of near-collinear segments: on a 20k-segment ellipse, renumbering moved u at 9,216 vertices by up to 6.3e-3.
- The lane that found the defect guessed it only partly: it was right on order, but precision is a separate cause.

**Rule** (approved 00:50 EDT; no constant):
- **T1.** Exact distance ties are broken by the cells' canonical vertex positions, then parity. This is one `scatter_reduce` per key column, with no host sync.
- **T2.** On the wall the bound is 0: u is the least-norm point of the normals wherever they differ. That point lies within `tau_k + max_j tau_j` of each member and is the certified case's own value. Off the wall, G4 is unchanged.
- Fast and reference stay bitwise equal by construction, and this was measured on every dump.
- Across precisions, float32 values and their float64 copies now give the same u bitwise wherever both hold the query by the same cells. Where they do not, the float32 group strictly contains the float64 group. A float64 mesh and its own float32 rounding are different inputs, so they agree only within `min(sqrt(2h), h + diam)` (DESIGN §3).

**Function change** (float32, base → head; RESULTS §2):
- DrivAerML: on-wall u changes at 3.1k and 3.2k of 200k queries (median 2.8e-4, max 4.5e-3).
- AirfRANS: 70-93 airfoil nodes change (max 1.6e-4).
- Lug: rounding level (6e-8, off the wall), and d changes at 2 queries by 1e-10.
- Tensile2d and AeroZoo: no change.
- No sign or on-wall label changes anywhere.
- Renumbering moves (2.6k per DrivAerML draw in the base) and float32/float64 mismatches (3.1k per DrivAerML draw) drop to 0.
- At every query the base already tied, u is unchanged bitwise.
- Pose: DrivAerML's on-wall u gets steadier (max change 1.4e-2 → 3.1e-3; tie flips 630 → 0). On the lug's planar faces the tie *flag* now flips (0 → about 2,000 per case), but u does not get worse. The flag only decides whether u has a vertex gradient.

**Inference and HiLift, side by side:**
- **Inference check.** No 0.12.0-trained checkpoint exists, so it reused GEO-BAND's set. Largest |log new/old|: DrivAerML 2.7e-4, AirfRANS 6.5e-5, lug 6.7e-7; the worst single case is 1.2e-3. Nothing exceeds 1e-3.
- **HiLift (slab of 2.4M cells, 50k queries per case).** u changes by more than 0.1 at about 2% of queries. These sit at slivers (cells whose normal float32 cannot pin down) and folds of the stored wing. There the base's u was badly order-dependent: renumbering moved it at about 30% of queries; the head moves none. HiLift's surface task (option A) makes no geometry call, so this touches no training run.

**Cost** (fast kernel):
- The geometry call costs 2-8 ms more.
- Training step: DrivAerML +1%. Lug +5% in one job and -1% in the other; its passes differ by up to 9 ms, so the true cost is at most about 5%.

**Suites at the merged head:**
- CPU: 468 passed, 250 skipped.
- GB300, every CUDA test: 723 passed, 1 skipped.
- pre-commit passed. Two reviewers: the design review's P1 findings were fixed in v2; the diff review approved with fixes, and the fixes are applied.

**For you or Peter:**
1. Merge 3a1a15d48. It joins the 0.14.0 version commit.
2. Decide on the on-wall tie-flag flips under pose on planar regions (the review's finding 2). I recommend accepting them.
3. The full-resolution HiLift wing (0.27 billion cells) runs out of memory on a GB300 in the base geometry too, in `_winding_moments`.
4. A residual remains in float64: the winding sign. Its BVH far field depends on Morton ties, which follow the cell order. It flipped one query 1.3e-10 off the wall under renumbering. The root fix is to number vertices by position at the weld (DESIGN §2).
5. `$T/code_drawexact_base/test` stops at `diffusion/`, so suites cannot run on it. DRAW-EXACT should know.

Drafts for the book are in NOTEBOOK_DRAFT.md: chapters 05, 21 and 14, plus a notebook entry. For project memory: T1 and T2 are the on-wall rule, and the GB300 cap was 7.5 GPU-h.

**On disk.** Nothing is running and the registry row is removed.
- Worktree `/home/psharpe/gh/physicsnemo-isla-geo-tieorder` (pushed, clean). Remove it after the merge.
- On LHR under `$K`: HiLift dumps (26 GB; delete?), slabs (150 MB), and checkpoint copies under `infer/ckpt` (1.65 GB, can go).
- Snapshots under `$T`: code_tieorder_proto, code_tieorder_head1, code_tieorder_merge1 and recipe_tieorder_fc6ed, each under 100 MB.
- Scratch folder: 82 MB.
