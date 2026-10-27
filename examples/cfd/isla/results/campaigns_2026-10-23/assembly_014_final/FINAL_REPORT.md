# ASSEMBLY-014: canonical ISLA 0.14.0, the staged branch (hand-over, 2026-10-26 11:15 EDT)

Status: DONE (staged; the AirfRANS pieces and the version commit wait on the supervisor's word).

## Branch
psharpe/isla-014, pushed, head **121b78578**, on psharpe/isla e89e77180. Worktree /home/psharpe/gh/physicsnemo-isla-014.

| commit | what | lines (text, binary data aside) |
|---|---|---|
| eca42af08 | ISLA-DTT forest as a function of the boundary alone: DTT-POSE's D1-D5 ported from 38a7915cc, without its version bump | +662 / -125 (tests +435) |
| 1909e1cde | Warp-limit guards: fast build checks its node count (fill pass), fast walk checks the node boxes' n_nodes*d entries | +68 / -16 |
| 83757fb16 | dead geo_logit.bias removed; name-keyed seeded fills (golden harness and release writer); k_proj routing exemption | +89 / -38 |
| a525ba773 | test_isla_nd count 41362; resolution test gates on refined < 3 spread + kept-whole 1e-3, tripwire 0.37 | +17 / -8 |
| 121b78578 | review fixes: param_generator in goldens/cases.py; k_proj compared in float32 at 5e-2 | +25 / -23 |

Notes on item 3 (the guard): in ISLA the build's boxes are torch's, so the binding Warp arrays are the fill pass's
per-node arrays (up to 2N-1 entries) and, in the walk, the flattened node boxes (n_nodes*d, up to 2Nd). Both are now
checked; both new tests fail on eca42af08 and pass after (c3_before_fix.txt). A/B 110/110 bitwise.

## A1.4 (A14-CHECK; a14/RESULTS.md; jobs 403458, 403496, 403567, 403608; 0.90 GPU-h): PASS
- Replay of the 96 check-(a) captures: ported (eca42af08) = DTT-POSE (code_dttpose_c3, 4242962e0, tree_read 99aac192)
  bitwise on 23 fields (pair lists qi/si/n_node, boxes, measures, separation diameters, per-cell face class, frame,
  origin, ...), fast Warp and reference paths, torch 2.12 and 2.13. Both versions differ from the captures' own hashes
  only in frame (<= 1.55e-15, 88/96), as CHECKS.md recorded.
- Current-model inputs on the same cases: lug points/normals/query_r bitwise; omega and normal_tol move 1-2 ulp from
  the draw's float64 measures (2042c50cf/DRAW-EXACT; rebuilt from float32 measures they reproduce check (a) 96/96);
  AirfRANS gains the far field (whole boundary), so its length unit goes from ~0.42 to 305.3 chords and points/query_r
  are the old values / 704-740. GEO-PORT feeds no forest input. On the current model's own 96 captures: bitwise again.

## Golden changes (each regenerated with record.py --regenerate in its commit)
- eca42af08 (forest): goldens_v1.pt 36/90 cases (every forest case: G-FOREST forest_*/node_means_*, G-WALK, G-READ
  tree_read_*, G-MODEL model_dtt_*/query_chunks_*, G-PARTS parts_dtt_*); isla_release_v2 dtt_f64, dtt_2d_f64,
  dtt_draw_f32, query_chunks; isla_dtt_v1.pth, isla_dtt_checkpoint_output_v1.pth. Other 54 variants A/B bitwise.
- 83757fb16 (bias + name-keyed fills): 30/90 cases (all that draw parameters: G-MODEL, G-PARTS, G-BLOCKS slice_block_*,
  routing_logits_*, geo_region_*, G-READ tree_read_*); every release output golden; both .mdlus (lose the 2 biases, new
  values; still labelled 0.13.0). A/B fails exactly those 30, other 80 bitwise. Bias-less vs bias-0.37 same weights:
  rel 1e-16 (f64) / 1e-7 (f32). Lists: data_diff_c1.txt, data_diff_c2.txt, ab/.
- 121b78578: manifest only (no case changes).

## Tiers on 121b78578 (JOBS.md)
| | torch 2.12 | torch 2.13 |
|---|---|---|
| GB300 whole dir | 404758: 1,324 passed, 1 xfailed | 404771: 1,318 passed, 6 skipped (mix-order, by design), 1 xfailed |
| CPU full | 404880: 775 passed, 533 skipped, 1 xfailed | 404923: same |
Recipe tests (local, 2.13, td0153, at 83757fb16; later commits touch only ISLA tests): 631 passed, 24 skipped.

## Smokes (SMOKE-014; smoke/SMOKES.md; jobs 404076, 404077, 404837; 7.22 GPU-h), torch 2.13, head/base iter_time_s
drv 0.961, az 0.988, af 1.052 then 1.004 flipped (1.028 both orders), lug 0.959, ten 1.018 / 1.014, drs 0.988, suv 1.003,
hl 0.999, djb 0.997: **all pass**; compile counts identical, no breaks, no tracebacks. Beside: AeroZoo train peak +2.9 %
(133.6 -> 137.5 GiB) while drv falls 2.7 %; 2-D DTT 1.4-2.8 % slower (inside the bound; not investigated).

## Rulings used / new
- Supervisor 09:05: k_proj joins the routing exemption by measurement; name-keyed seeded fills in harness and release
  writer (test shows 84/84 other params bitwise when one is removed; ordered fill gave 2/84). Supervisor 09:45: the
  resolution test's absolute cap dropped, tripwire 0.37 from the measurement. For memory: seeded test fills are
  name-keyed since 83757fb16 (param_generator in goldens/cases.py), so a parameter change moves only its own values;
  module default inits under a global seed still shift (seen in test_isla_interior_geometry).
- Candidate for the cleanup list: kernel_basis.net.4.bias is also dead in exact arithmetic (_MODEL_ZERO_GRADS).

## Version commit
Needs the supervisor's word; plan in VERSION_COMMIT_PLAN.md (0.14.0 / _version 14, refuse 0.7.0-0.13.0, refusal text,
EARLIER_RELEASES += 0.13.0, regenerate both .mdlus + checkpoint-output goldens; expect 57/57 and 85/85 tensors bitwise).

## Compute, disk
12.00 GPU-h of 40 (18 jobs, sacct). Registry row `^isla\.asm014\.` stays while I wait for the steer. On disk: worktree
(~1.1 GB); LHR $T/kernel/asm014 (~0.5 GB: snapshots, captures, smoke outputs); this folder 34 MB.

## Addendum (2026-10-27 01:25 EDT): the supervisor's 13:44, 22:31 and 23:05 steers
- cb73af61b: kernel_basis.net.4.bias removed (dead). |grad| / largest gradient 3.6e-18 (f64), 9.5e-10 (f32), 1.9e-7
  (bf16 autocast); same weights without it: outputs 2.3e-16 (f64), 1.2e-7 (f32), inside a one-ulp input change; bf16
  per-group distances to f64 inside or below a 9-sample base family (c6probe/). Goldens: 14/90 cases (every case with a
  kernel basis), A/B the other 96 variants bitwise; isla_dtt_v1.mdlus 84/84 other tensors bitwise; surface files untouched.
- 272886d21: merge of psharpe/isla 1a86c09f9 (A/B against cb73af61b: 110/110 bitwise; only sampling.py's import names).
- 72b624ae4: test_isla_nd's interior count without the basis bias (5760 / 18688); the only failure of 415487 / 415527.
- 8097ed00f (head): merge of psharpe/isla 6f1e527f1 (recipe-only).
- Tiers on 72b624ae4: GB300 416459 (2.12) 1,324 passed, 1 xfailed; 416080 (2.13) 1,318 passed, 6 skipped, 1 xfailed;
  CPU 416926 (2.12) and 416217 (2.13) 775 passed, 533 skipped, 1 xfailed. Recipe tests on 8097ed00f: 648 passed, 34
  skipped. No smoke: the merges changed no ISLA model file beyond sampling.py's import names.
- GPU-h: 14.03 of 40. The version commit plan (VERSION_COMMIT_PLAN.md) now expects 57/57 and 84/84 tensors bitwise.
