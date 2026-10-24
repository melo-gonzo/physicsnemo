# TIE-EXP: jobs, snapshots, records (LHR)

`T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0`

## Registration
- PREREG.md written 15:11 EDT 2026-10-22, before any TIE result: md5 **fecaeedde84df28c866180eff687fccc**.
- Supervisor 15:00 EDT: about 200 GPU-h approved for the four arms, eager; cap 220 GPU-h for everything; conditions 1-6
  (recorded in PREREG §§0-7).

## Snapshots (read-only, chmod a-w)
| dir | what | MD5SUMS md5 | SNAPSHOT_HEAD md5 |
|---|---|---|---|
| `$T/code_tie` | `cp -r --preserve=timestamps code_cmpaf_fix` + d3d9da49d geometry.py 2d39efb4, tree_read.py c1d34bc1 | 35397694 (3,248) | 1020f321 |
| `$T/recipe_tieprobe` | byte copy of recipe_att090 + conf/model/isla_tensile2d_tie_flip.yaml (757db3aa); probes only | 98c0a00c | - |

Overlay proof: `evidence/overlay_proof.txt`. Training and scoring use `$T/recipe_r090` unchanged.
(A first `cp -a` attempt failed on NFS permission preservation; the partial copy was removed and rebuilt with
`cp -r --preserve=timestamps`, as recipe_att090 was.)

## Probe files (`$T/kernel/tie/probe`, local copies `launcher/probe/`)
| file | md5 |
|---|---|
| geo_flip_probe_tie.py | 662b7ca6427e31af81ce6b917b81416a |
| flip_probe_tie.py | 75fcb51f3fd2fc25ef348d17155cbead |
| cellcount.py | 8cf5c0c182d31899ad1725a8d0e810d9 |
| pre_cpu.sbatch | 37534deab8a700744dc21af44321ec0f |
| pre_gpu.sbatch | 2d3cdb7d7b3ecd00086de9fb0706422d |

## Pre-steps (submitted 15:12 EDT)
- 337674 `isla.tie.precpu` (cpu-short array 0-3): geometry flip lug base / branch on CPU; cell counts lug / Tensile2d.
- 337675 `isla.tie.pregpu` (short QoS, 1 node): flip probe base / branch on r090fix_lug_isla_seed42 and
  r090eag_t2dz_isla_seed42, then the geometry probe on GPU (lug, Tensile2d), base / branch.
- Outputs `$T/kernel/tie/probe/out/`, logs `$T/kernel/tie/probe/logs/`.

## Pre-step reruns and smoke on code_tie (all **run on the defective snapshot, superseded**; PREREG §9.2)
- 337934 `isla.tie.precpu2`, 338014 `isla.tie.precpu3`: failed before any geometry (process group on a CPU node; then the
  datapipe's pin_memory needs a GPU). 337993 `isla.tie.pregpu2`: flip_probe_tie2.py (head compensation; PREREG §8.1).
- 337910 `isla.tie.smoke` (short QoS, 6.6 min): both units rc 0, Training completed; lug median step 0.238 s, Tensile2d 0.215 s.
- 338161 `isla.tie.pregpu3`: cancelled at 16:02 EDT on the supervisor's steer (it used code_tie).
- Local copies of the outputs: `prestep/` (summary_all.txt, smoke-337910.log). Superseded; not read for the gate.
- PREREG A1 (16:01 EDT) md5 4af20d08; A2 (16:05 EDT) md5 d5f905b7.

## code_tie2 (waiting for the TIE-FIX commit)
- Launcher `$T/kernel/tie2` staged 16:07 EDT (LAUNCH_MD5S.txt; files as `launcher/tie2/`). Nothing submitted.
- Probe md5s: f32_validity.py 57e51fd5, gate2.sbatch 647de537, geo_flip_probe_tie2.py cb98a614, flip_probe_tie2.py 63145842,
  cellcount.py 8cf5c0c1. smoke.sbatch d0bd6902, lanes_smoke.tsv f3b25248, lanes.tsv 99048964, score_rows.tsv b896737a.
- PREREG A3 (19:37 EDT, FOLD-SIGN evidence 338225 / 338281 against d3d9da49d) md5 fb7655e7
- PREREG A4 (19:40 EDT, gate (a) with the exact KKT reference; f32_validity.py v2 ae9ad05f replaces 57e51fd5, never run) md5 ad336712
- PREREG A5 (19:53 EDT: code_tie2 = 19fb0a754, MD5SUMS 49aeb192; gate (a) v3 f32_validity3.py 95c82f1f, items 4-6; gate_a_cpu.sbatch f7b0b7b7, gate_a_tasks.tsv f99f9704; gate2.sbatch 30385f83) md5 61af1998
- 20:15 EDT: the supervisor read partial gate (a) output, logs/f32v3_gpu_lug_code_tie2.log (105 lug case lines). Reported at
  20:16 (message headed 20:20; time corrected by the supervisor at 20:17): 18 of 105 cases each have one on-wall node with |u32 - u64| just above 1e-5 (max 1.124e-5, case 01432),
  |u64 - uref| <= 3e-10, all other counts 0.
- Gate jobs on code_tie2: 341643 isla.tie.gate2 (short), 341644 isla.tie.smoke (short; kernel/tie2), cpu-short gate (a) chunks
  341647, 341686, 341760, 341792, 341864, 341929, 341989, 342048 (tasks 0-30 of gate_a_tasks.tsv), all ended by 20:31 EDT.
- PREREG A6 (20:33 EDT: transfer to 34654e01d via item 4's b_zero_out; the supervisor's decision rule, approved 20:16 EDT) md5 2ec36219
- 20:35 EDT, gate read (local copy `gate2/`, summary `gate2/gate_summary.txt`, `gate2/gate_a_summary.json`): (i), (ii) branch, (b),
  smokes pass; (a)(1)/(2) u32 half fail on the lug only (101 test / 89 train nodes, max 2.07e-5); f32_validity3's instrument
  check fails as registered on the lug (b_check_dev 5.2e-9 > 1e-12); item 4's b_zero_out = 189,092 / 172,836 lug rows, all with
  exact point 0. gate2 ran only the branch half of (ii).
- 20:45 EDT, diagnosis and transfer measurement (not registered verdicts; for the supervisor's rule and A6.1's question):
  f32_diag.py 31915f44, f32_transfer.py f25523b3, min_norm_34654.py 633f07b3 (34654e01d's _min_norm_points, verbatim plus a
  `certified` record), bad_nodes.json 55b50b61, diag_cpu.sbatch 72bcc47a, transfer_gpu.sbatch 70c019ea.
- 20:51 EDT: $T writable again (97 % used). Submitted 342298 isla.tie.diag (cpu-short array 0-3) and 342299 isla.tie.transfer (short).
- 21:01 EDT, read: 342298 (diagnosis) and 342299 (transfer, base (ii)) COMPLETED. Summary `gate2/diag_transfer_summary.txt`,
  per-node table `gate2/DIAG_NODES.md` (190 nodes). Under the supervisor's 20:16 rule: r > 2 on all 190 nodes (max 2,261, median
  about 300), so **no launch**; reported with K, the pairwise angles and |uref| (need_decision). Transfer (A6.1): 0 uncertified rows,
  0 differing rows in 5,143,196 rows of both datasets and both precisions, so code_tie2 = 34654e01d on these data. Base (ii): lug
  d_F median 1.53e-4 (reproduces). The self-check deviation (5.2e-9) is 3 float64 rows with k = 5 near-duplicate directions.
- PREREG A7 (21:10 EDT: supervisor-approved deviation from (a)(1)/(2), override at 21:10 EDT) md5 d3641b98

## Arms (code_tie2 = 19fb0a754; launcher `$T/kernel/tie2`; submitted 21:10 EDT 2026-10-22 after PREREG A7)
| idx | run | job name | first link |
|---|---|---|---|
| 0 | tie_lug_isla_seed42 | isla.tie.lug.s42 | 342513 |
| 1 | tie_lug_isla_seed43 | isla.tie.lug.s43 | 342514 |
| 2 | tie_t2dz_isla_seed42 | isla.tie.t2d.s42 | 342515 |
| 3 | tie_t2dz_isla_seed43 | isla.tie.t2d.s43 | 342516 |
- Launcher files checked against LAUNCH_MD5S.txt before submission: lanes.sbatch 8eecec8f, ckpt_preflight.py e34bd611, site.env
  3ea1da2a, submit.sh b8c33b86, lanes.tsv 99048964, score.sbatch 0c524289, score_rows.tsv b896737a: all OK. (LAUNCH_MD5S.txt's
  other FAILED lines are superseded probe versions, probe/f32_validity.py v1 and gate2.sbatch v1, and six probe files whose lines
  were written from inside probe/.) Submission log: submit_arms.log.
- Registry row `^isla\.tie\.` added to watchers/ACTIVE_EXPERIMENTS.tsv at 21:10 EDT (expected_end 2026-10-25; backup
  ACTIVE_EXPERIMENTS.tsv.bak here, cmp-checked; diff 1 line).
- Monitor: `$T/kernel/tie2/check.sh` (launcher/tie2/check.sh), polled by launcher/tie2/monitor.sh.
- Scoring after "Training completed": `cd $T/kernel/tie2 && sbatch --array=<k> score.sbatch` (k = lane idx; rows 4k..4k+2)
  -> `$T/evals_tie2/<run>/<native|area|posed>/<run>/metrics.jsonl`.
- Monitor: launcher/tie2/monitor.py 3275a049, check.sh 3a1e1c1f (remote copy identical); log monitor.log.
- PREREG A8 (21:44 EDT: transfer to ONWALL 1e2c431cd via the gray count; full ISLA CPU suite on 19fb0a754, 290 passed) md5 fb823a8d
- Acceptance 21:45 EDT: all four first links started about 21:19 EDT and are RUNNING, with their chain-ahead links pending
  (342619, 342620, 342622, 342631). The link-start lines read code_tie2 and recipe_r090; resolved configs show precision
  bfloat16, compile false, compile_blocks false, lr 3.75e-4, 500 epochs, seeds 42/43, hidden 512, n_slices 512, boundary_tokens
  200,000; no traceback; median step lug 0.23-0.24 s, Tensile2d 0.21-0.23 s.
- 21:43 EDT: the first monitor run crashed: check.sh's `grep -c ... || echo 0` printed two lines once train.log existed.
  Fixed: check.sh 22158c5a (remote identical), monitor.py a49de137 (now skips malformed status output).
- 22:30 / 22:34 / 01:00 EDT supervisor notes: B's lug pair was scored by REF-090-CLOSE (343170, read by the supervisor at 22:04
  EDT 10-22). Blinding list (the ref090 and attrib090 records, $T/evals_r090eag score files, the book and results trees, .codex-tmp
  STATUS, the program-context memory topic): **none of these opened by this lane since 22:04 EDT 10-22.** Earlier, about 14:50-15:40
  EDT 10-22, I read ref090's JOBS.md, t2d_fourseed.md, t2d_control.md, t2d_standing.md and attrib090's RESULTS.md (no eager
  lug score existed then).
- Tensile2d arms: Training completed (t2d.s42 by 00:44, t2d.s43 by 01:14 EDT 10-23). Scoring 345741 (kernel/tie2 score.sbatch
  --array=2,3, short QoS 01:59:00; inference command unchanged) -> $T/evals_tie2/. Rule F probe 345742 (fprobe.sbatch e8131213,
  DS=t2d; TIE with code_tie2, B with code_cmpaf_fix beside it).
- 01:25 EDT: Tensile2d TIE scores complete (epoch 500, 100 cases, native / area / posed, SNAPSHOT_USED code_tie2; local copy
  read/evals_tie2/; metrics not read before the registered read). Rule F, Tensile2d: TIE d_F exactly 0 in every field, both
  seeds (d_U 2.5-175); B beside (code_cmpaf_fix): displacement 8.0-8.2e-7, stress 1.4-1.5e-6, q 2.7-3.0e-5 (local read/F/).
- Reducer launcher/read_tie.py de7f5223 (tested on synthetic data), registered rules of PREREG section 4 as amended.
- 03:41 EDT 10-23 supervisor note: a read-only audit (scratch/isla-geo-audit-2026-10-23/AUDIT.md, no scores) reran the counts
  behind A3 and A8 with a check that each version ran its own search; they reproduce exactly. This lane's gate probes ran one
  snapshot per process, so the custom-op replacement flaw does not apply to them.
- 11:23 EDT 10-23: lug arms at epochs 264 / 257 (links 345784 / 345844 running, 351601 / 351653 pending); no anomaly in 10 h.
- PREREG A9 (11:37 EDT 10-23, records only: FOLD-SIGN's repeat 341328 / 341486 against 19fb0a754, GEO-AUDIT rerun; A3 reported 'not met as registered'): md5 before fb823a8dc58b91efdba72aa9cbe89ad8, after f0e70cb27eee686c3877b7f91eb86686
- 11:38 EDT 10-23: monitor re-armed (monitor.py --hours 11.8 --t2d-reported, to about 23:30 EDT) after the supervisor cancelled the previous wait to deliver the 04:03 request; the old process (pid 1612161) was stopped first.
- 23:36 EDT 10-23: lug arms at checkpoints 461 / 455, final links 358193 / 358897 pending (normal, expected start 06:20 EDT); moved both to short QoS, TimeLimit 01:59:00 (about 40 / 45 epochs at about 156 s each; if a short link times out, its chain-ahead normal link finishes). Short total was 7 before.
- 23:36 EDT 10-23: monitor re-armed to about 10:00 EDT 10-24 (--hours 10.4).
- Lug arms: Training completed (lug.s43 by 01:34, lug.s42 by about 01:19 EDT 10-24; final links 358193 / 358897 on short QoS).
  Cancelled the no-op chain-ahead link 362865 (lug.s42, its run already completed). Lug scoring 364216 (score.sbatch --array=0,1,
  short 01:59:00) and rule F probe 364217 (fprobe.sbatch, DS=lug) submitted 01:36 EDT.
- 01:43 EDT 10-24 registered read: read_tie.py on read/ (TIE scores $T/evals_tie2, scorers 345741 / 364216; B's scores
  $T/evals_r090eag, lug from REF-090-CLOSE 343170 / 343179, Tensile2d 322903 / 322904; F probes 345742 / 364217). Results:
  RESULTS.md, read/results.md. F restored; N no cost (lug 1.004, Tensile2d 1.016); P most (e_TIE / e_B 0.004).
- 01:45 EDT: registry row ^isla\.tie\. removed from watchers/ACTIVE_EXPERIMENTS.tsv (backup ACTIVE_EXPERIMENTS.tsv.bak2,
  cmp-checked, 1 line). No isla.tie job left in the queue. GPU-h 206.1 (read/sacct_tie.txt).
