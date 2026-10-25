Status: DONE

**Head:** psharpe/isla-main-merge-1024 is at **b8b1543c4**, pushed with a plain push as a new branch. I did not merge it into psharpe/isla. The branch holds three commits, all made with `git commit -s` and the hooks on:
- **a682959bc** merges upstream/main 289e57382 (22 commits) into 5b70c8686;
- **15e9063aa** changes the torus tests, on your ruling of 22:46;
- **b8b1543c4** merges origin/psharpe/isla 2e3bfb66b (the symmetry plane and its book batch).

**Conflicts.** Only CHANGELOG.md conflicted, and rerere recorded the resolution rather than replaying an old one.
- Under Added, I kept both entries: main's Strata entry and ISLA's MeshDataset/CenterMesh entry.
- Under Fixed, I dropped ISLA's copy of the #1972 checkpoint entry, because main now carries the same entry lower down. I kept main's three new entries.
- Nine files changed on both sides and merged cleanly. For each one I checked that its +/- lines equal both sides' deltas. train.py comes out byte-identical to ISLA's version, which already held #1972.
- The re-merge of 2e3bfb66b had no conflicts; README.md and highlift_surface.yaml keep both sides' changes. The new recipe code does not conflict with main: symmetry.py and dedup.py use only torch and numpy. `_box_face` re-winds the faces that main's #2148 `structured_grid` change reversed, so its cells come out bitwise identical.

**One finding.** Main's #2148 reversed `torus.load`'s winding. Two ISLA interior-geometry tests take `cells[:, :2]` midpoints, and with the new winding those land on each quad's diagonal. Both tests then failed on x86 CPU and on GB300, and passed at the base.
- The defect is pre-existing, and the same at 5b70c8686 and in both windings. At diagonal midpoints, where the two normals agree to rounding, the on-wall u is not a function of the mesh alone:
  - renumbering moves it at 51 of 1152 points, by up to 1.79e-6;
  - float32 against float64 inputs differ at 5 of 35 points.
- On your ruling I pinned both tests to the base winding (`_flipped`, with a comment citing #2148) and added `test_on_wall_direction_at_quad_diagonals`, a strict xfail with both cases. ISLA's geometry code is unchanged.
- The details, my untested guess at the mechanism (near-equal normals that are not certified distinct), and the probes are in `DEFECT_TORUS.md` and `probes/`.

**Main's effect on ISLA, its recipe and its book** (the child review's full table is in `IMPACT.md`). ISLA's own numbers do not change:
- ISLA calls none of the measure, LSQ, facet or BVH-traversal code that main fixed.
- Only main's primitive changes reach ISLA's test inputs. Apart from the torus, those inputs are bitwise unchanged.
- No old import path is used in scope. `weighted_multinomial` has a shim; `nn.functional.safe_normalize` was removed and is not used.
- **Baselines only:** the degenerate-face SDF repair in #2125 can change `ComputeSDFFromBoundary` values near faces thinner than 1e-4 of their longest edge. Those values are the Transolver and GeoTransolver volume inputs; ISLA does not read them.
- **Tiny:** float64 meshes rotated by `Mesh.rotate` now get float64 trigonometry, which changes R by about 1e-7 relative.
- **Notes:** `from_zarr` is stricter now, but no ISLA dataset is zarr. Six ISLA-only dataset YAMLs lack main's new `validate_args: false`; they work only through a process-wide side effect.

**Checks.**
- pre-commit on every file changed against origin/psharpe/isla, and `import-linter --all-files`: all pass.
- That needs `COLUMNS=250`. Without it, interrogate's narrow table breaks the checker's path parsing and gives false failures; the same happens at c8f684288.

**Suites on LHR.** Base is 2e3bfb66b, whose physicsnemo/ and test/ trees are identical to 5b70c8686's. Logs are in `logs/` and the job records in `JOBS.md`.

| suite | head b8b1543c4 | base |
|---|---|---|
| ISLA directory on GB300, every CUDA test (376777 on the head; 376126 on the base) | 709 passed, 1 skipped, 2 xfailed | 709 passed, 1 skipped |
| ISLA directory on CPU (376778) | 461 passed, 241 skipped, 2 xfailed | 461 passed, 241 skipped |
| recipe tests on CPU | 609 passed, 18 skipped, 2 failed | the same 2 failures |
| GLOBE, test/models/globe (CPU; GB300) | 250 passed, 73 skipped; 315 passed, 8 skipped | the same |
| test/mesh on CPU | 3911 passed, 1503 skipped | 3610 passed, 1388 skipped |

- The two recipe failures are both in `test_campaign_c_hooks`. Their E lines match the base's exactly: a TypeError from a variable left unset under sbatch, which earlier merges also saw. Your local 611 passed equals 609 plus these 2.
- test/mesh has no failures on either side. The extra tests at the head are main's new ones.
- The skip lines match, apart from one line-number shift.
- GPU use was 3.15 of the 4 GPU-h cap.

**Render.** Full book under the lock, with PYTHONPATH set to each tree and CUDA hidden.
- Head b8b1543c4: rc 0. Base 2e3bfb66b (as you asked): rc 0.
- Neither has an unresolved reference. The only warnings are 21 "Kernel is running over TCP" and 7 "Warp CUDA error 100".
- The visible text of all 23 pages is identical (`compare_render.py`, `render_diff.txt`), so no printed value changed.
- One of 78 figures differs: 03's fig-isla-stream-toy, which draws from an unseeded `torch.randn`. That is not a merge effect.
- An earlier render of a682959bc also exited 0.

**For you to decide:**
1. Merge psharpe/isla-main-merge-1024 (b8b1543c4).
2. Launch the geometry lane on `DEFECT_TORUS.md`.
3. Optionally, add `validate_args: false` to the six ISLA YAMLs that pin z with a zero-width Uniform, as #2076 did for main's YAMLs.

**For memory:**
- pre-commit's interrogate needs `COLUMNS=250` (or a terminal that wide) for multi-file runs.
- Main's primitives are wound outward since #2148. A test that takes `cells[:, :k]` as "edges" may now hit quad diagonals.

**Left on disk:**
- Worktree /home/psharpe/gh/physicsnemo-isla-main-merge-1024 (about 1 GB), clean and pushed. Remove it after your merge.
- On LHR, read-only snapshots of about 60 MB each: `$T/code_mainmerge_{head,base,head2,base2}`, `$T/recipe_mainmerge_*`, and `$T/kernel/mainmerge`.
- The registry row has been removed (backups `ACTIVE_EXPERIMENTS.tsv.bak` and `.bak2`). Nothing is running.
