Status: DONE (ISLA-GEO-PORT, written 03:10 EDT 10-26)

**Branch** psharpe/isla-geo-port, head **a7a8d133899f02d22a2b6921b9c26f60f595ca9c**, pushed to origin with a plain push
(ls-remote empty before, a7a8d1338 after). Not merged into psharpe/isla; no GitHub activity. Worktree
/home/psharpe/gh/physicsnemo-isla-geo-port (clean). origin/psharpe/isla (bf71190af) is an ancestor of the head.

**BASE and merges.**
- Start: psharpe/isla-lib-sampling fe86a0011. Merge of psharpe/isla-geo-tieorder 3a1a15d48 = **103b546ee** (BASE for
  every port commit). geometry.py merged clean with INT64-OFFSETS 389b5092c, KERNEL-BUGS e44015436 and FUSED-PAD
  9f279664c, and equals PR-P7's reference resolution (src/geometry_port.py of 83325b653) except the frame's new library
  import. One conflict: test_isla_geometry_band.py imports (tests-p4's `_exact_d2` home kept, `_quad_diagonals` from
  test_isla_interior_geometry).
- Supervisor steers: origin/psharpe/isla moved to 2ed83c0cf (tree = fe86a0011) and then bf71190af (ISLA-PREMERGE);
  merged bf71190af (contains 2ed83c0cf) as **6d6409e3f** before F2, `--no-ff --signoff`, clean, no ISLA model/test
  file touched (ca35a1a5c, its merge of 2ed83c0cf, came in with it). BASE content = 103b546ee + bf71190af; the
  geometry of BASE is 103b546ee's (geometry.py md5 20156177), so the cross-version probes use code_103b546ee.

**Commits** (all `-s`, hooks on, author Peter Sharpe, no trailers; each bitwise in eager against its parent except F2's
u gradient, by the writers' 24-run CPU probes and the 602-run set below):

| item (PR-P7 source) | sha | subject | +/- |
|---|---|---|---|
| G2 (9b11ad176) | 540571e35 | search a crowded query's ball in one traversal (both ball kernels, one int64 counter across blocks) | +107/-18 |
| G3 (2e3c18464) | 6e3c29cfa | free the search's buffers as soon as they are read | +24/-5 |
| empty (c6deb9490) | 3c02170b7 | refuse a boundary with no cells and accept zero queries in 2-D (ISLA lacked both) | +33/-2 |
| F1 (cfc588d53) | 8779db494 | half-precision normals off the surface from float64 | +20/-1 |
| F3 (b052a83cb) | 30cad70f3 | bound the least-norm search of a tie with many directions | +106/-30 |
| merge | 6d6409e3f | Merge psharpe/isla (bf71190af) | no ISLA file |
| F2 (ab7e49618) | 6fcb1c6e1 | u's gradient slides along the edge (straight-through); W3-B1 xfails removed; goldens_v1 regenerated | +117/-49 |
| F2 fix | 269b84226 | gather, not take_along_dim: F2 had specialized the compiled `_case_inputs` point count (2 compile tests failed at 6fcb1c6e1); bitwise | +6/-1 |
| G2 test fix | a7a8d1338 | test-only: kernel_refs' second-pass test fills the overflow list (G2 made it pass through the list) | +10/-14 |

**602-run set** (RESULTS.md §1; logs out_gb_397953/, out_cpu_397954/, also at 6fcb1c6e1 in out_gb_397456/, out_cpu_397457/):
both kernels, f32+f64, GB300 346 runs + CPU (1 thread) 256 runs each. Every op output, w, d, u and d's gradients bitwise
in all 1,204 runs. u's gradient differs at every edge-region query row and only there: GB300 690,438 slide rows, 0
outside; CPU 93,331, 0 outside; vertex gradients differ only at slide cells' vertices (0 outside). Unchanged runs:
2-D sets and the float32 HiLift slabs (all queries on the wall). No other change. F1, F3 and the empty cases are reached
by no set and are covered by new tests.

**W3-B1/F2.** `test_direction_passes_gradcheck[3d-edge-queries]` and `[3d-edge-vertices]` lost the strict xfail (and the
vertices case its slow mark) and pass on CPU and GB300; a finite-difference Jacobian test was added for both kernels.
Goldens (D1/D2 answered by the supervisor; GOLDEN_DECISION.md, D2_ANSWER.md): goldens_v1.pt regenerated in 6fcb1c6e1 with
`goldens.record --regenerate goldens_v1.pt` on the merged tree. Final CPU A/B, BASE 6d6409e3f vs head a7a8d1338, each its
own process (goldens_ab/compare.log, rows.log): 104 of 110 variants bitwise; the 6 changed:

| 3-D G-GEO case | keys | edge rows | entries differ (grad.queries + grad.points) | outside edge rows / slide-cell vertices |
|---|---|---|---|---|
| sphere_f64 | grad.* | 36 / 426 | 104 + 181 | 0 |
| sphere_f32 | grad.* | 36 / 426 | 106 + 177 | 0 |
| torus_f64 | grad.* | 106 / 200 | 264 + 285 | 0 |
| sliver_soup_f64 | grad.* | 61 / 152 | 183 + 190 | 0 |
| zero_area_box_f64 | grad.* | 31 / 128 | 53 + 59 | 0 |
| open_cube_f64 | grad.* | 91 / 200 | 160 + 71 | 0 |

The regeneration also refreshed 13 cases to BASE's own outputs (committed file dated from 14518b280; each bitwise
base==head): GEO-TIEORDER (box_ties_2d/3d, naca, thin_cap, d/u of sphere_f64/torus_f64), DRAW-EXACT and library sampling
(G-SAMP pps_inclusion_probabilities, sample_boundary_cells_2d/3d; parts_dtt_draw_f32), library frame and sampling
(model_dtt_f64, model_dtt_2d_f64, query_chunks_f64, parts_dtt_f64, parts_dtt_2d_f64); causes by module, not bisected
(REPORT_WRITER.md). Both .mdlus files md5-identical (8194b51f, 1264468d); goldens_v1.pt 345,087 → 345,535 B, data/
772,305 B. The golden test passes in both tiers. **At the assembly, a binary conflict on goldens_v1.pt is resolved by
regenerating on the merged tree.**

**Suites** (JOBS.md): final head a7a8d1338, TD 0153 — full CPU tier 398071: 758 passed, 530 skipped, 1 xfailed
(Module.load), 0 failed (BASE 395419: 731/526/3); GB300 tier 398092: 1,304 passed, 1 xfailed, 0 failed (compile tests
included). The first head 6fcb1c6e1 failed 4 CPU / 6 GPU tests (397528, 397557), fixed by the last two commits. Recipe tests
on CPU (local, td0153): 631 passed, 24 skipped. pre-commit on the 8 changed files and import-linter --all-files pass.
Compiled path: no CERT run; the only compiled difference seen was F2's recompile, fixed; compile tests pass.

**Timings** (job 397456, one GB300, median of 10 calls, two reps BASE/head; RESULTS.md §2):
- DrivAerML whole domain, fast kernel (the model's default): 404.4 → 347.0 ms, peak 17.44 → 12.40 GiB.
- The same call, reference kernel: 525.1 → 463.0 ms, 17.16 → 13.34 GiB. PR-P7's library measured 521.9 → 397.4 ms
  and 17.16 → 12.40 GiB.
- AeroZoo: time unchanged (fast 110 ms, reference 183 ms); peak 1.44 → 1.03 GiB.
- Lug: unchanged within noise (fast 34 ms, reference 75 ms).

**md5 of changed files at a7a8d1338:** geometry.py 0dcb76f4d860e62c6cbeced8c36006b1; data/goldens_v1.json
84d16831b20298895e927a587c58f136; data/goldens_v1.pt 816b5a3be297a8cafb2af7c9dd33c3b1; test_isla_geometry_fast.py
3d6a85c6c63b739c15db7897e61854e3; test_isla_gradcheck.py 6d625d14c572277f794d27488443757e; test_isla_interior_geometry.py
5286d45ce127d4f4df12803f2b291d77; test_isla_kernel_refs.py 03ca7df6ab62c647a1e55a1b767623e2; test_isla_tree_read.py
ade66c1dcd9e3b2fc4a2b1df393e4a83 (all under physicsnemo/experimental/models/isla/ or test/experimental/models/isla/).

**Book passages that go stale** (not edited):
- 06-using-isla.qmd:1648 (`para-using-tests`): "Five tests are strict expected failures, in {files[:-1]} and {last}".
  `_xf` now finds only test_isla_release.py, so this **renders broken** ("in  and `test_isla_release.py`"). The count is
  now 1 (the quad-diagonal xfails had already gone with GEO-TIEORDER). The tier counts there come from KERNEL-BUGS.
- 20-limits.qmd:629-650 (`para-limits-xfail`): "Five tests ... Two record ... the gradient of the on-wall direction u
  ... is wrong". W3-B1 is fixed, so one test remains (Module.load).
- 06-using-isla.qmd:1281-1395 (`tbl-using-geometry-cost` and its paragraphs): the class's geometry-call times
  (GEO-THINFAST jobs) are high by about 57 ms (fast) and 62 ms (reference) on DrivAerML's whole boundary; peak memory is
  not stated there.
- 05-forward-pass.qmd:1389-1394: "δ_q and u_q are differentiable": could add that u's gradient follows the closest
  point sliding along an edge.
- 05-forward-pass.qmd:1793 lists `_closest_features` live: it shows the new code automatically.

**Relied on:** LANE_RULES; the supervisor's D1 (23:10) and D2 (01:30, merge bf71190af first, use td0153) answers, and its
2ed83c0cf steer. **For memory:** nothing new beyond D1/D2 (golden regeneration may refresh drifted cases with the tool, if
they are bitwise base==head and listed in the commit).

**Left:** nothing running; registry row removed. The supervisor merges and regenerates goldens on conflict, and fixes
06/20 above. GPU use: 1.62 of 4 GPU-h.

**On disk:** worktree /home/psharpe/gh/physicsnemo-isla-geo-port (pushed, clean; remove after the merge); this folder
(1.8 MB: briefs, reports, logs, lhr/ harness); on LHR $T/kernel/geoport (184 MB: code_103b546ee, code_6fcb1c6e1,
code_a7a8d1338 snapshots, logs) and $T/kernel/isla_tests/snap/{base103b,head6fcb*,heada7a8*} (5 × ~60 MB).
