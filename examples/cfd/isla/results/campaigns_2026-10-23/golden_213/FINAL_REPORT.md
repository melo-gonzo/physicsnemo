# GOLDEN-213: the goldens' icosphere from correctly rounded operations (2026-10-26, 07:42 EDT)

Status: READY. Branch `psharpe/isla-golden-213`, pushed, not merged. Worktree `/home/psharpe/gh/physicsnemo-isla-golden-213`.
- e57e5e183: the fix and the regenerated data, with the reason "fixture sphere from correctly rounded ops (torch 2.13 vector_norm moves torch.norm by 1 ulp)".
- c75affde0 (head): `git merge --no-ff --signoff origin/psharpe/isla` at 9eb786e20 (#2172's head 0d968ef46), as the supervisor asked at 06:54.
  - origin/psharpe/isla has since moved to 00ac48041. Those commits touch only examples/cfd/isla (the book and results), so nothing under physicsnemo/ or test/ changed.
- Diff against 9eb786e20: 4 files. cases.py has +19 and -10 lines; goldens_v1.pt, goldens_v1.json and isla_release_v2.pth are regenerated with the tool.

## Change
`goldens/cases.py` gets `_icosphere(radius, subdivisions)`, used at its three sphere sites: `_draw_inputs_build`, `_geo_sphere` and `_sphere_cells`. It builds the sphere as follows:
- It takes the library icosahedron's cells.
- It recovers the icosahedron's vertices before normalization, (0, ±1, ±phi), from the sign and magnitude of the library's points.
- It normalizes them with the existing `_unit` (fixed order x*x + y*y + z*z, then sqrt, then divide).
- It runs the library's linear subdivision, which takes two-point means and so is correctly rounded.
- It projects the result with `_unit` and multiplies by the radius.

Neither `icosahedron_surface.load` nor `sphere_icosahedral.load` is used for the points any more; both call `torch.norm`. The library primitives are unchanged.

Against `sphere_icosahedral.load` locally (torch 2.13, x86; `probe_icosphere.py`):
- The cells are equal.
- The points differ by less than one ulp: 16 of 486 coordinates at 2 subdivisions, 176 of 1,926 at 3, and 0 at 1.

## Golden entries
- **goldens_v1.pt**: 11 of 90 cases changed. These are exactly the sphere-wall cases that failed on 2.13:
  - exact_geometry_sphere_f32/f64 and geometry_fast_parts;
  - model_dtt_f32/f64 and model_dtt_draw_f32;
  - query_chunks_f32/f64;
  - parts_dtt_f32/f64 and parts_dtt_draw_f32.
- **The other 79 cases are bitwise unchanged.** Two checks show this:
  - A CPU A/B record: base fixture against the new fixture, each in its own process, through `goldens.compare` (`ab_base_head.txt`). 99 of 110 case variants PASS bitwise, and only the 11 above fail.
  - An entry-by-entry comparison of the old and new golden files (`data_diff.txt`).
  - draw_inputs also uses the sphere but did not change: adding 40 in float32 rounds the one-ulp difference away.
- **isla_release_v2.pth** (`release_diff.txt`):
  - dtt_f64, dtt_draw_f32 and query_chunks changed. They use the sphere wall.
  - surface_f64 and rich_surface_f32 are bitwise unchanged.
  - dtt_2d_f64 moved by at most 3.1e-16 relative, in 5 of 180 elements. This is not from the fixture: regenerating with the base cases.py gives the same new bits (`release_basecases.pth`). The committed file had been stale since b3f92a932; the geometry commits since then moved it within roundoff. The tool regenerates the whole file, so this entry moves too.
- **The .mdlus files and every other data file are md5-identical.** They are not built from the sphere and were not regenerated.

| file | md5 before | md5 after |
|---|---|---|
| goldens_v1.pt | 816b5a3be297a8cafb2af7c9dd33c3b1 | 398f3b6f0f594c68900b30ed665f4ee2 |
| goldens_v1.json | 84d16831b20298895e927a587c58f136 | 0247347371ea288a8174f06cf1074343 |
| isla_release_v2.pth | 7834931fa6d362d0c5bac58996b4092e | cfb5c5d41b2181da9665795fe6063c7b |
| isla_default_v1.mdlus | 8194b51f7bdf31d6230d23dc626448ef | same |
| isla_dtt_v1.mdlus | 1264468dc8aee97003a0b121388f8768 | same |
| the other 5 .pth files | `md5_before.txt` | same (`md5_after.txt`) |

The goldens were regenerated where goldens_v1 was last recorded: locally on x86 with torch 2.13 and MKL, on one thread.

## Tiers at c75affde0 (JOBS.md)
| torch | GB300 gpu tier | CPU full tier |
|---|---|---|
| 2.12 (.venv-recipe, shared submit.sh, TD=0153) | 401648: 1,304 passed, 1 xfailed | 402034: 758 passed, 530 skipped, 1 xfailed |
| 2.13 ($T/venvs/isla-rerun-t213, `submit_t213.sh`) | 401700: 1,298 passed, 6 skipped, 1 xfailed | 402181: 758 passed, 530 skipped, 1 xfailed |

There are no failures on either torch.
- **The 6 skips on 2.13** are `test_isla_kernels.py:587`'s positive controls for pytorch#195845. They skip because Inductor 2.13 builds no mix-order kernel. RERUN-ENV's 397630 had the same 6 skips; they are not caused by this change.
- **Pre-merge commit e57e5e183 on 2.12**: the CPU tier (401519) gave 758 passed. The GPU job on that commit (401479) was cancelled after 207 s when the merge was requested.
- **Local checks (2.13, x86)**: test_isla_goldens.py and test_isla_release.py gave 199 passed, 155 skipped and 1 xfailed, at e57e5e183 and again at c75affde0.

**How the 2.13 jobs ran**: `submit_t213.sh` is a lane copy of the master submit.sh. It differs in two ways:
- It submits RERUN-ENV's lane sbatch copies in `$T/kernel/rerunenv/isla_tests` (the VENV override).
- It exports VENV.

The shared launcher files are unchanged.

**GPU-h**: (207 + 463 + 503) s × 4 GPUs = 1.30 GPU-h, against a cap of 2.

**Registry**: the `^golden213\.` row was added at launch and removed at the end. A backup is beside this report.

## On disk
- The worktree `/home/psharpe/gh/physicsnemo-isla-golden-213` (1.1 GB) stays until its branch merges.
- On LHR: the snapshots `$T/kernel/isla_tests/snap/t2{12,13}{gpu,cpu}_c75affd` and `t212cpu_e57e5e1`, plus a snapshot of `t212_e57e5e1`, each a few MB.
- This folder: 624 KB. The raw A/B records (136 MB) are deleted.
