Verdict: CONFIRMED WITH CORRECTIONS

DK-R5-REVIEW, read-only second read of RESULTS.md section R5 (written 2026-10-24 23:01 EDT by `date`). Read: PREREG.md (md5 39d8272f,
s4, s5 R5, s9 incl. the 17:35 clarification), reduce_s2.py (md5 e2b605d2), out_r5/*, the s2data and ctrl trees, ATTRIB RESULTS 2c/3b/3c,
reduce.py/reduce2.py, LHR $T/evals_dttknobs_s2/dttknobs_s2_noscstd_seed4x, score_376686_6.log, and sacct. Only DTT-NOSC-STD scores were read.

Confirmed by my own recomputation from the raw metrics.jsonl (infer_summary, equal to the step means within 4e-8):
- Table L17-19: arithmetic Δ +0.1886 / +0.1848 / +0.1627 (x1.208 / 1.203 / 1.177), Δ_N -0.0461 / -0.0183 / +0.0183 (x0.955 / 0.982 / 1.018).
  σ7 0.0503 / 0.0866 / 0.0744, so 2 SE is 0.1006 / 0.1731 / 0.1489. A is yes on all three fields and N is no, which reads
  "leaves" on each field. Geometric: +0.1869 / +0.1730 / +0.1619; velocity's geometric A fails by 0.0001 (unresolved), so the flag is right.
  The per-seed ratios (L21), the 36 % velocity seed gap (L12), "clears 2 SE by 7 %" (L27), P10 and P12 failed (L33): all correct.
- σ pooling: ATTRIB reduce.py's convention, sqrt(ΣSS of ln / Σdof), over 3c's six arms (B, DTT-CONT, ARM-D, DTT-NOSC, DTT-SDF,
  DTT-QEMB; results2_native.json md5 1c58aa88) plus NOSC-STD gives 7 dof, as PREREG s5 registers. σ6 is 0.0422 / 0.0291 / 0.0637 (3c: 0.042 / 0.029 / 0.064).
  B and DTT-NOSC from the trees equal ATTRIB's copies (relative difference 0). The pool's velocity SS is 0.0051; NOSC-STD's own seeds add 0.0474 (L12-13 is right).
- Section-4 checks, all four rows: "epoch": 500; 200 infer_step rows whose sample ids equal r090eag_af_dtt s42's (same order);
  SNAPSHOT_USED $T/code_att090fix; DATASET_USED airfrans_uvolume_2d / _posed; the log's extra is the s2n lane extra plus
  `compile=false model.compile_blocks=false`; recipe_att090b; precision float32; ISLA.0.500 / checkpoint.0.500 loaded; .done present.
  The fetched metrics.jsonl md5s equal LHR's. Job 376686_6 handled rows 24-27 and COMPLETED in 333 s. Its log has only
  `[: : integer expression expected`, which r090eagaf's and r090fix's score logs also show (benign).
- The labels follow the reading table; the bundle thread uses pressure and nut only (C_f 0.131 / 0.047 / 0.121, velocity below ln 1.05),
  and "sits in the scales, as far as two seeds can show" is the registered wording for "leaves".
- Cost (L43) against sacct: seed 42's five links come to 60,357 s (67.06 GPU-h). Seed 43's six links come to 59,591 s (66.21 GPU-h),
  including 376024 (10 s); 376051 was cancelled while pending (0 s). Total 133.3. The 0.37 GPU-h scorer is left out, as in PREREG s8's split.

Corrections:
1. L22, φ and Δ / D_DTT are the geometric form, not the registered one. The reducer's phi_G is the geometric Δ over the tree's
   geometric G. In the registered form (arithmetic Δ over the registered constants G_f and D_DTT), φ = 0.81 / 0.91 / 1.11 and
   Δ / D_DTT = 1.83 / 1.18 / 6.26. Velocity's φ 0.85 differs from the registered 0.91. Fix: give the registered values
   (with the geometric values in brackets, labelled).
2. L37-39, overclaim. "The forms differ ... ISLA-DTT does [need the scales], on every field" compares two separate readings,
   and no registered rule tests the difference. A plain 2-SE check of Δ_DTT − Δ_ISLA (SE = sqrt(σ7² + σ_2c²), with 2c's 8-dof
   0.063 / 0.045 / 0.059) resolves only pressure: 0.186 against 2 SE 0.161. Velocity gives 0.156 against 0.195 and nut 0.121
   against 0.190. Velocity's own A is also the flagged one: it rests on seed 43 (÷ B 1.022 / 1.382). Fix: "The two forms read
   differently. The difference is clear on pressure; on velocity (flagged) and nut it is within two seeds' noise." Change "on every field" to match.
   (Minor: 2c's ratios are geometric means and L17-19's are arithmetic; at these sizes that changes nothing, but say so.)
3. L30-32, "It is not an interaction with the raw speed" is categorical, but the registered rule only says "leaves" locates the
   cost in the scales "as far as two seeds can show". "No better than DTT-NOSC" is also too strong: pressure's Δ_N is −0.046 (x0.955), which is unresolved.
   Fix: "There is no sign that it is an interaction with the raw speed. With the speed standardized and no scales, ISLA-DTT is
   still about 20 % worse than B, and not measurably better than DTT-NOSC."
4. L13, P2: "exactly" should read "to the registered three decimals" (σ6 0.0422 / 0.0291 / 0.0637; G 0.2345 / 0.2028 / 0.1468, within the reducer's 0.0006 tolerance).
   L43, P2: B's GPU-h recomputed from reduce/ctrl/cost (sacct) is 66.07 / 66.99 (66.1 / 67.0). The quoted 66.0 is DK-S2PLAN's constant.
   L22-23, P2: posed agrees within 0.001 on the ratios (1.207 / 1.202 / 1.177) and gives the same labels. Say that instead of "to the third digit".

None of these changes a registered label, the flag, the bundle-thread reading or a prediction outcome.
