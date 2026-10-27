# Study records read by the book

Written by `copy_records_2026-10-23.py`; every file is an unmodified copy of its
source. Do not edit by hand.

| topic | what it measures | source folder |
| --- | --- | --- |
| `attrib090/` | ATTRIB-090: which change of ISLA 0.9.0 moves Tensile2d, AirfRANS and the lug | `scratch/isla-attrib090-2026-10-21/` |
| `ref090/` | REF-090: ISLA 0.9.0 against the nearest-token form, the training-path control | `scratch/isla-ref090-2026-10-20/` |
| `af_extrap/` | AF-EXTRAP: AirfRANS extrapolation in Reynolds number and angle of attack | `scratch/isla-af-extrap-2026-10-21/` |
| `suv_xfam_b/` | SUV-XFAM part B: SHIFT-SUV models trained on fastbacks, against the estate-trained ones (stages 1 and 2) | `scratch/isla-suv-xfam-2026-10-22/` |
| `probes090/` | PROBES-090: query-population dependence and re-meshing invariance of ISLA 0.9.0 | `scratch/isla-probes090-2026-10-21/` |
| `cap_audit/` | CAP-AUDIT: capacity floors of the three architectures, phases 1, 1b and 2 | `scratch/isla-cap-audit-2026-10-20/` |
| `floor_drv/` | FLOOR-DRV: ISLA on one DrivAerML Volume car with the domain's four boundary meshes, the pre-registration | `scratch/isla-floor-drv-2026-10-23/` |
| `az_compliant/` | AeroZoo under the benchmark protocol: the registered rule for ISLA against ISLA-DTT (PREREG_v2 sections 12(f) and 13) | `scratch/isla-az-compliant-2026-10-19/` |
| `boundary_audit/` | BOUNDARY-AUDIT: what each dataset stores of its domain's boundary, and the whole boundary's cost on DrivAerML Volume | `scratch/isla-boundary-audit-2026-10-23/` |
| `drv_ground/` | DRV-GROUND: the DrivAerML domain meshes' ground offset on all 484 files, and the recipe transform that places it | `scratch/isla-drv-ground-2026-10-23/` |
| `setup_research/` | SETUP-RESEARCH: the CFD domains of DrivAerML, SHIFT-SUV and HiLiftAeroML, and where their boundary pieces come from | `scratch/isla-setup-research-2026-10-23/` |
| `af_whole/` | AF-WHOLE: ISLA and ISLA-DTT reading the AirfRANS far field, the pre-registration and its facts before training | `scratch/isla-af-whole-2026-10-23/` |
| `surface_bc/` | SURFACE-BC: a surface task that reads the whole boundary and predicts at the boundaries it names, built and timed | `scratch/isla-surface-bc-2026-10-23/` |
| `surfqpop/` | SURF-QPOP: how a DrivAerML Surface prediction depends on the other requested cells | `scratch/isla-surf-probe-2026-10-23/` |
| `slices_af/` | SLICES-AF: ISLA with 128 slices against the 512-slice AirfRANS standing | `scratch/isla-slices-2026-10-23/` |
| `tie_exp/` | TIE-EXP: ISLA with the least-norm tie rules trained on the lug and Tensile2d, read on flip symmetry, native accuracy and pose | `scratch/isla-tie-2026-10-22/` |
| `fold_sign/` | FOLD-SIGN: the wall distance signed by the generalized winding number, the registered f32floor rerun on AeroZoo | `scratch/isla-fold-sign-2026-10-22/` |
| `token_draw/` | TOKEN-DRAW: the boundary-token draw in float64 in the boundary's canonical frame, measured under precision, pose and renumbering | `scratch/isla-token-draw-2026-10-22/` |
| `bundle_merge/` | BUNDLE-MERGE: the geometry robustness bundle merged as ISLA 0.11.0, its version change and tests | `scratch/isla-bundle-merge-2026-10-24/` |
| `geo_fast/` | GEO-FAST: the exact geometry's fast path, its bitwise checks and its GB300 timing, and the zero-area-cell defect | `scratch/isla-geo-fast-2026-10-23/` |
| `align8/` | ALIGN8: the step cost of token counts that are not a multiple of 8 or 32, and the padding fix | `scratch/isla-align8-2026-10-23/` |
| `bc_sidecar/` | BC-SIDECAR: the SHIFT-SUV box and the HiLiftAeroML far field and symmetry plane, built and checked | `scratch/isla-bc-sidecar-2026-10-23/` |
| `geo_degen/` | GEO-DEGEN: cells of zero area read as the segment or point their vertices span, its checks and cost, and the limits left | `scratch/isla-geo-degen-2026-10-24/` |
| `fast_merge/` | FAST-MERGE: the exact geometry's fast path merged with the zero-area fix, its bitwise checks and its GB300 timing | `scratch/isla-fast-merge-2026-10-24/` |
| `dtt_knobs/` | DTT-KNOBS: ISLA-DTT's boundary read, its knobs and its accuracy gap (stage 1, inference only) and the stage-2 registration | `scratch/isla-dtt-knobs-2026-10-23/` |
| `dtt_fast_compile/` | DTT-FAST, piece COMPILE: ISLA-DTT's compiled blocks certified against eager, and the bf16 casts Inductor leaves out | `scratch/isla-dtt-fast-2026-10-23/compile/` |
| `dtt_fast_verify_fusedread/` | DTT-FAST, piece FUSEDREAD verified independently: pair lists, precision and the GB300 step | `scratch/isla-dtt-fast-2026-10-23/verify_fusedread/` |
| `dtt_fast_integration/` | DTT-FAST, integration pass 1: the tree walk, tree build and fused read on ISLA 0.11.0 behind one execution attribute | `scratch/isla-dtt-fast-2026-10-23/integration/` |
| `boundary_merge/` | MERGE-PREP: the token padding and the whole boundary with the surface task merged into ISLA 0.11.0, checked and timed | `scratch/isla-merge-prep-2026-10-24/` |
| `dtt_pose/` | DTT-POSE: why ISLA-DTT's forest depends on pose and precision, and the design that removes it | `scratch/isla-dtt-pose-2026-10-24/` |
| `dtt_pose_exp/` | DTT-POSE-EXP: ISLA-DTT retrained with DTT-POSE's forest on the lug and AirfRANS, the pre-registration | `scratch/isla-dtt-pose-exp-2026-10-24/` |
| `geo_thin/` | GEO-THIN: thin cells of nonzero area in the exact geometry, the census of the stored meshes and the design | `scratch/isla-geo-thin-2026-10-24/` |
| `dtt_fast_final/` | DTT-FAST, pass 2: ISLA-DTT's training step before and after on GB300, against interacting ISLA, and its surface form | `scratch/isla-dtt-fast-2026-10-23/final/` |
| `nearwall/` | NEARWALL: ISLA's near-wall distance channel under the whole boundary, its exposure, response and precision, and the design | `scratch/isla-nearwall-2026-10-24/` |
| `hl_dup/` | HL-DUP: HiLiftAeroML's duplicated wing cells, their census and effect on the standing (stage 1, revised after its second read) and the stage-2 registration | `scratch/isla-hl-dup-2026-10-24/` |
| `geo_thin_final/` | GEO-THIN, final: the thin-cell fix as merged, its amended design (what is derived and what is estimated), checks, cost and limits | `scratch/isla-geo-thin-2026-10-24/` |
| `nearwall_final/` | NEARWALL, final: the comment fix as merged, the contingency drafted for AF-WHOLE and the geometry note | `scratch/isla-nearwall-2026-10-24/` |
| `dtt_fast_bf16diag/` | DTT-FAST, BF16-DIAG: where the chunked step's bf16 excess on AirfRANS comes from, its remedy, and the one-ulp yardstick | `scratch/isla-dtt-fast-2026-10-23/bf16diag/` |
| `az_decide_final/` | AZ-DECIDE, final: AeroZoo's compliant standings of four architectures, the registered read of ISLA against ISLA-DTT, and its context | `scratch/isla-az-decide-2026-10-21/` |
| `e1_az/` | E1-AZ: the head normalization against ISLA 0.9.0 on compliant AeroZoo's nonbox split, the registered verdict | `scratch/isla-headnorm-2026-10-18/` |
| `lre_az/` | LRE-AZ: ISLA on AeroZoo with the fixed Reynolds scale against its fitted constants, registered and read | `scratch/isla-lre-az-2026-10-23/` |
| `geo_band_phase0/` | GEO-BAND, phase 0: the whole-boundary geometry call re-timed on GB300 at the merged head | `scratch/isla-geo-band-2026-10-24/` |
| `kernel_tests/` | KERNEL-TESTS: the five CUDA tests that failed after the token padding, their cause and the fix | `scratch/isla-kernel-tests-2026-10-24/` |
| `slices_drv/` | SLICES-DRV: ISLA with 128 slices against the 512-slice DrivAerML Volume standing | `scratch/isla-slices-drv-2026-10-24/` |
| `freeze_plan/` | FREEZE-PLAN: the final reference rerun re-planned with ISLA-DTT as the default interior form, its lanes, cost and calendar | `scratch/isla-freeze-plan-2026-10-24/` |
| `bc_supply/` | BC-SUPPLY: SHIFT-SUV's box and HiLiftAeroML's inlet and outlet built in the recipe, the deduplicated wing, their checks and cost | `scratch/isla-bc-supply-2026-10-24/` |
| `geo_band_final/` | GEO-BAND: the exact geometry's derived tie and on-wall bands in the boundary's centred frame (ISLA 0.12.0), their derivation, its independent re-derivation, the pose and native effects, the trained-model check and the cost | `scratch/isla-geo-band-2026-10-24/` |
| `band_merge/` | BAND-MERGE: the ISLA 0.12.0 version commit for GEO-BAND and its test review | `scratch/isla-band-merge-2026-10-24/` |
| `default_dtt/` | DEFAULT-DTT: ISLA-DTT the default interior form, its tests, smokes and query-independence check | `scratch/isla-default-dtt-2026-10-24/` |
| `dtt_default_int/` | DTT-DEFAULT-INT: the default switch integrated with ISLA 0.12.0, and its fixture test reworked | `scratch/isla-dtt-default-int-2026-10-24/` |
| `geo_thinfast/` | GEO-THINFAST: the thin-cell guard and the accurate closest point fused into the fast path, its equality and timing | `scratch/isla-geo-thinfast-2026-10-24/` |
| `dtt_fast_done/` | DTT-FAST, done: ISLA-DTT's exact speedups as merged, their equivalence, the step on the final heads, the pose port, the compiled bf16 gradient spike and the bf16 reduction flag | `scratch/isla-dtt-fast-2026-10-23/` |
| `hl_dup_stage2/` | HL-DUP, stage 2: ISLA and both unit-drive baselines retrained and scored on the deduplicated HiLift wing, its review, and the failed root-outline rebuild | `scratch/isla-hl-dup-2026-10-24/` |
| `dtt_fast_final_table_p5/` | DTT-FAST, pass 2's table with its pass-3 columns added, as it now stands | `scratch/isla-dtt-fast-2026-10-23/final/` |
| `hl_dup_stage2_scores/` | HL-DUP, stage 2: the scorer's per-case rows of the six deduplicated-wing runs (native and posed), and the stage-1 scores of the stored-wing checkpoints that its rule and its like-for-like figures read | `scratch/isla-hl-dup-2026-10-24/` |
| `hl_symplane/` | HL-SYMPLANE: HiLiftAeroML's symmetry plane built at load from the source STL's root cap, its per-case check and tolerance, the census of the 690 canonical cases, the comparison with the curation pipeline's plane, and the book changes | `scratch/isla-hl-symplane-2026-10-24/` |
| `hl_dup_stage2_area/` | HL-AREA: HL-DUP stage 2's six deduplicated-wing checkpoints scored on the uniform point cloud (per-case rows, reduction and table), and the six lanes' training cost from their own logs (extract_cost.py's lines and the cost report) | `scratch/isla-hl-area-2026-10-24/` |
| `isla_api_013/` | ISLA-API: ISLA 0.13.0 (the interior task is ISLA-DTT with no switch; Mesh's names n_spatial_dims and n_boundary_tokens; no Reynolds argument; the schema key defines), its design and mapping table, the deleted tests, the assertion audit, the suites and the datapipe check of re_z against the in-model z, and the book changes | `scratch/isla-api-013-2026-10-24/` |
| `reynolds_out/` | REYNOLDS-OUT: the Reynolds feature moved out of ISLA's constructor into the recipe (ComputeBoundaryScales and float64 ComputeLogReynolds in ISLA's model configs), its design and evidence, and the bitwise check of z on 160 cases over three checkpoints | `scratch/isla-reynolds-out-2026-10-24/` |
| `az_spec_audit/` | AZ-SPEC-AUDIT: the AeroZoo runs against AeroZoo's task specification (inputs, statistics, splits, targets, metric, loss and reporting), with the deviations M1, T1, I1, I2, L1, S1, S1b, R1 and the hardening notes | `scratch/isla-az-spec-audit-2026-10-24/` |
| `az_rescore/` | AZ-RESCORE: the 48 AeroZoo checkpoints rescored with AeroZoo's own interior metrics (eps_U, eps_p, eps^N; mean and median over all test cases and the mean_stable ones; full-mesh float64, chunked inference), with the zero and free-stream predictors, the orders, the partition probes and the registered readings beside | `scratch/isla-az-rescore-2026-10-24/` |
| `outlet_exp/` | OUTLET-EXP: ISLA trained with AeroZoo's outlet gauge pressure held by construction, read on profile_crossflow and full by its registered rule, its cost and the supervisor's decision not to adopt it | `scratch/isla-outlet-exp-2026-10-22/` |
| `geo_tieorder/` | GEO-TIEORDER: the on-wall direction's dependence on cell order and precision at near-equal normals, its two causes, the rule that removes it (a candidate for ISLA 0.14.0) and its measured function change and cost | `scratch/isla-geo-tieorder-2026-10-25/` |
| `draw_exact/` | DRAW-EXACT: the token draw's run-to-run and CPU-to-GPU dependence at exact ties, the duplicate and dropped cells of today's GPU evaluation draw, and the exact int64 line that removes them (a candidate for ISLA 0.14.0) | `scratch/isla-draw-exact-2026-10-25/` |
| `slice_normal_exp/` | SLICE-NORMAL-EXP: the no-harm retrain check of the slice mean normal, its registration and the DrivAerML Surface read of option (a), the unnormalized mean normal | `scratch/isla-slice-normal-exp-2026-10-24/` |
| `dtt_knobs_r5/` | DTT-KNOBS stage 2, R5: ISLA-DTT without its physical scales on a standardized inlet speed against the AirfRANS standing's pair, and its second read | `scratch/isla-dtt-knobs-2026-10-23/` |
| `main_merge/` | MAIN-MERGE: upstream main merged into psharpe/isla, its effect on ISLA, its recipe and its book, and the on-wall direction's defect at quad diagonals that it found | `scratch/isla-main-merge-2026-10-24/` |
| `lib_tree/` | LIB-TREE: the cluster tree's Warp build and walk and the grouped forest in physicsnemo.mesh.spatial, a library-only branch, with its bitwise proofs against ISLA and GLOBE and its timings | `scratch/lib-tree-2026-10-24/` |
| `af_whole_read/` | AF-WHOLE, the read: ISLA and ISLA-DTT trained with AirfRANS's far field as a named boundary against the eager airfoil-only standings, by field, by wall-distance band and under pose, with the octant-tie analysis | `scratch/isla-af-whole-2026-10-23/` |
| `test_audit/` | TEST-AUDIT: ISLA's test suite audited (coverage on CPU and GB300, MOD-008, determinism on the GPU) and the test plan | `scratch/isla-test-audit-2026-10-24/` |
| `test_build/` | TEST-BUILD: ISLA's test suite built from TEST-AUDIT's plan (goldens, MOD-008, properties, gradcheck and opcheck, kernels against references, determinism, forced-path guards, tiers), its suite counts and the bugs it pinned | `scratch/isla-test-build-2026-10-25/` |
| `kernel_bugs/` | KERNEL-BUGS: four of the bugs TEST-BUILD pinned, fixed (the Triton read's log-measure gradient, the eager read's bf16 gradient sums, float16 autocast on CUDA and on CPU), with its bitwise evidence and suite counts | `scratch/isla-kernel-bugs-2026-10-25/` |
| `af_fix/` | AF-FIX: the mechanisms behind the AirfRANS far-field harm (the octant tie under faces, the near-wall length's candidates, where the slices' and the tree read's mass goes) and the drafted campaign | `scratch/isla-af-fix-2026-10-25/` |
| `af_campaign/` | AF-CAMPAIGN: the frozen pre-registration of ISLA-DTT on AirfRANS with its far field, arms F, FN, FNC and FNOmega | `scratch/isla-af-campaign-2026-10-25/` |
| `pr_2170/` | NVIDIA/physicsnemo#2170 (LIB-TREE's library branch): its CI, the one bot comment and its fix | `scratch/pr-2170-2026-10-25/` |
| `perf_survey/` | PERF-SURVEY: a profile of ISLA 0.13.0's training step on GB300 and the ranked performance opportunities, measured, with its inference child's details (the read path's memory against points, the fused geometry kernel past 2^31) | `scratch/isla-perf-survey-2026-10-25/` |
| `floor_drv_read/` | FLOOR-DRV, the read: ISLA on one DrivAerML Volume car with the domain's four boundary meshes against the same run without them, the registered readings, the floors by field and wall-distance band, and the cost | `scratch/isla-floor-drv-2026-10-23/` |
| `globe_dr/` | GLOBE-SCORE on the DrivAerML Surface task: GLOBE at its authors' configuration under the recipe protocol, scored native, on the uniform point cloud and posed, against ISLA and the baselines | `scratch/isla-globe-2026-10-19/` |
| `fused_pad/` | FUSED-PAD: the fused geometry kernel for 2-D models and models without a global direction, with exact zero padding; its speed, its bitwise 3-D check and its eager and compiled certification | `scratch/isla-fused-pad-2026-10-25/` |
| `int64_offsets/` | INT64-OFFSETS: 64-bit offsets where ISLA's kernels passed 2^31 at full resolution, the audit of every kernel and the tests past the limit | `scratch/isla-int64-offsets-2026-10-25/` |
| `lib_sampling/` | ISLA-LIB-SAMPLING: ISLA's token draw and canonical frame from physicsnemo.mesh, bitwise outside the zero-measure edge, with DRAW-EXACT | `scratch/isla-lib-sampling-2026-10-25/` |
| `premerge/` | ISLA-PREMERGE: upstream main and five open pull requests merged ahead of their merges; every training config's inputs bitwise | `scratch/isla-premerge-2026-10-26/` |
| `rerun_configs/` | RERUN-CONFIGS: every dataset file the final rerun reads committed with its whole boundary, AeroZoo's seven fields, the area draws, the re-pricing and the supervisor's decisions | `scratch/isla-final-rerun-2026-10-26/` |
| `geo_port/` | ISLA-GEO-PORT: the geometry speedups and fixes of the closed library pull request, in ISLA's own geometry; the edge gradient of the on-wall direction fixed | `scratch/isla-geo-port-2026-10-26/` |
| `dtt_knobs_a14/` | DTT-KNOBS stage 2, arms 1-4: ISLA-DTT retrained on AirfRANS at theta 2 and 0.5, with a read after every second block and with a basis of 8, against its standing | `scratch/isla-dtt-knobs-2026-10-23/` |
| `dtt_knobs_exact/` | DTT-KNOBS stage 2, K-EXACT: ISLA-DTT retrained on AirfRANS with the exact read, the ladder (R2) and the far-field gap (Q3) | `scratch/isla-dtt-knobs-2026-10-23/` |
| `dtt_pose_exp_read/` | DTT-POSE-EXP, the read: ISLA-DTT retrained with DTT-POSE's forest on the lug and AirfRANS against the standings' pairs, native, posed and the bf16 forest | `scratch/isla-dtt-pose-exp-2026-10-24/` |
| `dtt_knobs_final/` | DTT-KNOBS, closed: the results of stage 2 with R2's scope against the knob-rule credit, the final report, the stage-3 proposal and the two notebook drafts | `scratch/isla-dtt-knobs-2026-10-23/` |
| `slices_all/` | SLICES-ALL: 128 slices against 512 for ISLA-DTT on AirfRANS and the lug and for interacting ISLA on the DrivAerML Surface task, read for harm, with its cost | `scratch/isla-slices-all-2026-10-24/` |
| `slice_normal_dras/` | SLICE-NORMAL-EXP, the DrivAerML Surface read of (a)+(s), the unnormalized mean normal with the slice-spread distance, against the control and against (a) | `scratch/isla-slice-normal-exp-2026-10-24/` |
| `rerun_env/` | RERUN-ENV: torch 2.13 against 2.12 for the final rerun: tests, smokes, the HiLift fault and its re-smoke after the fix | `scratch/isla-rerun-env-2026-10-26/` |
| `compile_flag/` | COMPILE-FLAG: torch 2.13's process-wide is_compiling() during a compile, every call site the final rerun reaches, and the thread-correct guard | `scratch/compile-flag-2026-10-26/` |
| `golden_213/` | GOLDEN-213: the goldens' icosphere from correctly rounded operations, and ISLA's test tiers on torch 2.12 and 2.13 | `scratch/isla-golden-213-2026-10-26/` |
| `final_rerun_plan/` | The final rerun's plan as decided: the supervisor's decisions with Peter's answers, and the price with AeroZoo's canonical task | `scratch/isla-final-rerun-2026-10-26/` |
| `az_canonical/` | AZ-CANONICAL: AeroZoo's canonical task, the mean_stable cases only, in every model's configuration, with the constants refit on them | `scratch/isla-az-canonical-2026-10-26/` |
| `assembly_014/` | ASSEMBLY-014: ISLA 0.14.0 staged: the boundary-only forest ported, the dead biases removed, name-keyed test fills, its tiers and smokes | `scratch/isla-assembly-014-2026-10-26/` |
| `cert/` | CERT: the compiled slice blocks of ISLA 0.14.0's staged code certified against eager on nine families (Parts 1 and 2), and the compiled and eager steps timed | `scratch/isla-cert-2026-10-27/` |
| `cert_routing/` | CERT-ROUTING: why the compiled routing gradients differ from eager's (the two bf16 roundings Inductor drops), the precision-raising reference family, and Part 3's verdicts with its positive control | `scratch/isla-cert-routing-2026-10-27/` |
| `slice_normal_final/` | SLICE-NORMAL-EXP, final: every arm read (DrivAerML Surface, AirfRANS and the lug, options (a) and (a)+(s)), the lug's block-wise certification, and the design of option (c) | `scratch/isla-slice-normal-exp-2026-10-24/` |
| `af_campaign_read/` | AF-CAMPAIGN, the read: ISLA-DTT on AirfRANS with its far field, arms F, FN, FNC and FNOmega read by the registered rule, with its second read | `scratch/isla-af-campaign-2026-10-25/` |
| `forest_neartie/` | FOREST-NEARTIE: the face forest's pairs at near-tied frames against the octant forest's, the memory and resume risk, the options and the fix (the rotation bounded in proportion to each node's size, capped by a frame-free ball) | `scratch/isla-forest-neartie-2026-10-27/` |
| `rule1_root_cause/` | AF-CONFIRM, rule 1 on AeroZoo: the per-condition frame-rotation bound behind its blow-up, its blast radius and fix V2 | `scratch/isla-af-confirm-2026-10-27/` |
| `version_014/` | VERSION-014: ISLA 0.14.0's version commit, the upstream merge into it, and the near-tie fix staged with its tiers and smokes | `scratch/isla-version-014-2026-10-27/` |
| `assembly_014_final/` | ASSEMBLY-014, final: ISLA 0.14.0 staged, with its addendum (the kernel basis's dead bias removed, the merges and the tiers) | `scratch/isla-assembly-014-2026-10-26/` |
| `main_merge_1026/` | MAIN-MERGE-1026: upstream main and #2171's head merged into psharpe/isla, the event-timed iteration time, and their checks | `scratch/isla-mainmerge-1026/` |
| `main_merge_1027/` | MAIN-MERGE-1027: upstream main with #2174's final form merged into psharpe/isla, and its bitwise checks | `scratch/isla-mainmerge-1027/` |
| `merge_2171_tidy/` | MERGE-2171-TIDY: #2171's tidied head merged into psharpe/isla, and its checks | `scratch/isla-2171tidy-2026-10-26/` |
| `af_confirm_prereg/` | AF-CONFIRM's registration: AF-CAMPAIGN's pre-registration with amendments A3 (arm FC, C without N, on AirfRANS), A4 and A5 (C on AeroZoo, read for harm) | `scratch/isla-af-campaign-2026-10-25/` |
| `rerun_azb/` | RERUN-AZB: the final rerun's 16 AeroZoo baseline lanes launched on JHB, their snapshot, smoke and first epochs | `scratch/isla-rerun-azb-2026-10-27/` |

| topic | file | source | md5 |
| --- | --- | --- | --- |
| attrib090 | RESULTS.md | scratch/isla-attrib090-2026-10-21/RESULTS.md | e01a94de29d14f876636ea92b553bcb5 |
| attrib090 | results_native.json | scratch/isla-attrib090-2026-10-21/results_native.json | 3f92835ce703e13a5f72641fa6cf241e |
| attrib090 | results2_native.json | scratch/isla-attrib090-2026-10-21/results2_native.json | 1c58aa8828c6210a71b4df7836af782b |
| attrib090 | results2_posed.json | scratch/isla-attrib090-2026-10-21/results2_posed.json | 5b55f04bb3089f81fbf89f461cbbb8a5 |
| attrib090 | results_native_eager.json | scratch/isla-attrib090-2026-10-21/results_native_eager.json | 80be6f1a18d5b622ec9ee5c56a6c5e1a |
| attrib090 | results_posed_eager.json | scratch/isla-attrib090-2026-10-21/results_posed_eager.json | 4d5713853c3c7e758c18586f61dd06ff |
| attrib090 | relaunch_native.md | scratch/isla-attrib090-2026-10-21/relaunch_native.md | 69a5d34836fe28e5cde2095a48b2c9ce |
| attrib090 | relaunch_posed.md | scratch/isla-attrib090-2026-10-21/relaunch_posed.md | 0d0d4af949234d07ab56dd763686c8e3 |
| attrib090 | arch_lug.md | scratch/isla-attrib090-2026-10-21/arch_lug.md | ec7191a56c0dace45f1d7921252c4a95 |
| attrib090 | arch_tensile2d.md | scratch/isla-attrib090-2026-10-21/arch_tensile2d.md | cac94bc8495b8073dd30098b83d0a3bf |
| attrib090 | arch_airfrans.md | scratch/isla-attrib090-2026-10-21/arch_airfrans.md | abbf8c160dafe0b13cd5890f7026f73f |
| attrib090 | t2d_eager_native.md | scratch/isla-attrib090-2026-10-21/t2d_eager_native.md | 72138ebbc50aebbeafa8446aaebc7aed |
| attrib090 | t2d_eager_posed.md | scratch/isla-attrib090-2026-10-21/t2d_eager_posed.md | 31e3214261471c9757d03f5a5e430c95 |
| ref090 | r090_final_tables.md | scratch/isla-ref090-2026-10-20/r090_final_tables.md | 5a90be5bef32b14d13d53db2175dfec2 |
| ref090 | r090_final.json | scratch/isla-ref090-2026-10-20/r090_final.json | 5fb0022d17a82b9d1303dd9b4a4f4b1b |
| ref090 | t2d_control.md | scratch/isla-ref090-2026-10-20/t2d_control.md | f2e62620d17a62a4add41f1873b52ba7 |
| ref090 | t2d_control.json | scratch/isla-ref090-2026-10-20/t2d_control.json | 37335a48f8d838dc94452fb88931de25 |
| ref090 | t2d_fourseed.json | scratch/isla-ref090-2026-10-20/t2d_fourseed.json | 0c67dc5548a683fd122a62c9aed0a811 |
| ref090 | t2d_fourseed.md | scratch/isla-ref090-2026-10-20/t2d_fourseed.md | f443a07295a57fd1b4548017f5f1241d |
| ref090 | PREREG.md | scratch/isla-ref090-2026-10-20/PREREG.md | 74ee9c8ff5801eb930e2ae27e0f28146 |
| ref090 | RESULTS.md | scratch/isla-ref090-2026-10-20/RESULTS.md | 0d6f5325a3ba6823aa8c8f7289bfc1d6 |
| ref090 | eag_lug.md | scratch/isla-ref090-2026-10-20/eag_lug.md | 103172e919599832b34e61ebd2c384d5 |
| ref090 | eag_lug.json | scratch/isla-ref090-2026-10-20/eag_lug.json | ed4d5799bcd2d0a67c8f8bc25a17fbed |
| ref090 | eag_af_isla.md | scratch/isla-ref090-2026-10-20/eag_af_isla.md | b4dd12f005fe0f37d40c2037de5b6b16 |
| ref090 | eag_af_isla.json | scratch/isla-ref090-2026-10-20/eag_af_isla.json | 29fd6a422a1ea99b151551d87f4de116 |
| af_extrap | RESULTS.md | scratch/isla-af-extrap-2026-10-21/RESULTS.md | 371927c6c6c04e84d8c114b8c74bc625 |
| af_extrap | POSTHOC.md | scratch/isla-af-extrap-2026-10-21/POSTHOC.md | 230e926b28b02490fcb350134ee007f8 |
| af_extrap | afx_tables.md | scratch/isla-af-extrap-2026-10-21/afx_tables.md | 2387eaa1930aa830492de937e28d7cec |
| af_extrap | afx_results.json | scratch/isla-af-extrap-2026-10-21/afx_results.json | 60dfd526fb60ee7563a60a710f5fa8cf |
| af_extrap | posthoc.json | scratch/isla-af-extrap-2026-10-21/posthoc.json | 11b97a9357bbffdee5b2284d705b0fc0 |
| af_extrap | clamp.json | scratch/isla-af-extrap-2026-10-21/clamp.json | 0e805bcf85fe32e377a40dc7953a2cd5 |
| af_extrap | stats/afx_fits.json | scratch/isla-af-extrap-2026-10-21/stats/afx_fits.json | bc9aa7e5ceb343f7bb4d1e6027e2811c |
| af_extrap | stats/clamp_bounds.json | scratch/isla-af-extrap-2026-10-21/stats/clamp_bounds.json | 5fc5005d16d7ec38475f488382e656f6 |
| af_extrap | LRE.md | scratch/isla-af-extrap-2026-10-21/LRE.md | d4af4d170c1860aa3b0cacc2bf389e01 |
| af_extrap | lre.json | scratch/isla-af-extrap-2026-10-21/lre.json | 493d95be22c44af6a41029e9d6822f54 |
| suv_xfam_b | PREREG_B.md | scratch/isla-suv-xfam-2026-10-22/PREREG_B.md | 118f36a0e86fe690414f8b3feaaa9c3b |
| suv_xfam_b | RESULTS_B.md | scratch/isla-suv-xfam-2026-10-22/RESULTS_B.md | 5458fb411157543ddaea49fc9df1e6ff |
| suv_xfam_b | JOBS.md | scratch/isla-suv-xfam-2026-10-22/JOBS.md | eeccbdce7208570e10bf595682ee8969 |
| suv_xfam_b | reduce_b_out.json | scratch/isla-suv-xfam-2026-10-22/reduce_b_out.json | 91e0cbab2d509fbe8b71ba76c5f436d4 |
| suv_xfam_b | identity_b_out.json | scratch/isla-suv-xfam-2026-10-22/identity_b_out.json | 389255a80b9364a941b2cbbf98826eb6 |
| suv_xfam_b | reduce_b2_out.json | scratch/isla-suv-xfam-2026-10-22/reduce_b2_out.json | 12b150610a9f8aee1dfa4b25bd6ea128 |
| probes090 | RESULTS.md | scratch/isla-probes090-2026-10-21/RESULTS.md | c9d1e9c66e5131cfc76047c3b617a049 |
| probes090 | results/qpop/reduced.json | scratch/isla-probes090-2026-10-21/results/qpop/reduced.json | bfd1167be24751f0094d44dd1de213c6 |
| probes090 | results/qpop_afix/reduced.json | scratch/isla-probes090-2026-10-21/results/qpop_afix/reduced.json | 82913383c1bd164d586a383c19919c9a |
| probes090 | results/qpop_afeag/reduced.json | scratch/isla-probes090-2026-10-21/results/qpop_afeag/reduced.json | 91facdcc78c171b8c9e596ee22d10033 |
| probes090 | results/qpop_az/reduced.json | scratch/isla-probes090-2026-10-21/results/qpop_az/reduced.json | bc85fd58c4cb7bf84846647fede4fec0 |
| probes090 | results/remesh/reduced_full.json | scratch/isla-probes090-2026-10-21/results/remesh/reduced_full.json | bf7d8b98527cead8577b20cfe52cb596 |
| probes090 | results/remesh_afix/reduced_full.json | scratch/isla-probes090-2026-10-21/results/remesh_afix/reduced_full.json | 1726c5c9dd8a920cd58e2ef7f7960f18 |
| probes090 | results/remesh_afix_dtt/reduced_full.json | scratch/isla-probes090-2026-10-21/results/remesh_afix_dtt/reduced_full.json | a5176cfa9bbe4398cb7f5c73535830cc |
| probes090 | results/remesh_afeag/reduced_full.json | scratch/isla-probes090-2026-10-21/results/remesh_afeag/reduced_full.json | 1b533a40e193bd5f80937b4256469b80 |
| probes090 | results/afeag_book_numbers.md | scratch/isla-probes090-2026-10-21/results/afeag_book_numbers.md | a02fc6786a8b386f063e5dd4b09c1212 |
| probes090 | remesh/facts_dv.json | scratch/isla-probes090-2026-10-21/remesh/facts_dv.json | 2d6eeb21e7deac33aaa87e54fab703ec |
| probes090 | remesh/README.md | scratch/isla-probes090-2026-10-21/remesh/README.md | 9ad17033baca19c8d90f313172f764d6 |
| probes090 | qpop/qpop_rows.tsv | scratch/isla-probes090-2026-10-21/qpop/qpop_rows.tsv | 23a0b83bf5fe89ae07221ffa57f7621d |
| cap_audit | RESULTS.md | scratch/isla-cap-audit-2026-10-20/RESULTS.md | f311a64e00b3f9a4c26ed9f5c34e8348 |
| cap_audit | HANDOFF.md | scratch/isla-cap-audit-2026-10-20/HANDOFF.md | 55938e6054614136eca355da7d8d9e0c |
| cap_audit | tables/ctl40k.md | scratch/isla-cap-audit-2026-10-20/tables/ctl40k.md | c44fecb8e16f9670288840f8747d4a11 |
| cap_audit | tables/phase1_floors.md | scratch/isla-cap-audit-2026-10-20/tables/phase1_floors.md | 10be6ce3275e378395da307d9fa7046c |
| cap_audit | tables/phase1b_floors.md | scratch/isla-cap-audit-2026-10-20/tables/phase1b_floors.md | 9e1484de2d0e78a5d856589866ee4606 |
| cap_audit | tables/phase2_floors.md | scratch/isla-cap-audit-2026-10-20/tables/phase2_floors.md | c8c31550f1a241381d4eb00816dabe49 |
| cap_audit | tables/resid_af_ladder.md | scratch/isla-cap-audit-2026-10-20/tables/resid_af_ladder.md | 01c30a3be9056512c56ff28d875baed3 |
| cap_audit | tables/resid_af_rd.md | scratch/isla-cap-audit-2026-10-20/tables/resid_af_rd.md | 0a3a0da34d76decfb42405463027a1f3 |
| cap_audit | tables/resid_deficit_cases.md | scratch/isla-cap-audit-2026-10-20/tables/resid_deficit_cases.md | b44c24507943ea2a0b18a229b719e721 |
| cap_audit | tables/resid_drv_ladder.md | scratch/isla-cap-audit-2026-10-20/tables/resid_drv_ladder.md | 4cd3d5377818dc8f5fd26c3d91e3b5fb |
| cap_audit | tables/resid_drv_p1.md | scratch/isla-cap-audit-2026-10-20/tables/resid_drv_p1.md | 63d6c434172f6e190a089b778b4dd54d |
| cap_audit | tables/resid_drv_partB.md | scratch/isla-cap-audit-2026-10-20/tables/resid_drv_partB.md | eb9cf5a6e519bf7e9de58e9950aa1f5d |
| cap_audit | tables/phase1_floors.json | scratch/isla-cap-audit-2026-10-20/tables/phase1_floors.json | 9b24f3349d96ac842000f1d2ccac5781 |
| cap_audit | tables/phase1b_floors.json | scratch/isla-cap-audit-2026-10-20/tables/phase1b_floors.json | 3ee6eda13263dd1778d5d96f33d8c913 |
| cap_audit | tables/phase2_floors.json | scratch/isla-cap-audit-2026-10-20/tables/phase2_floors.json | 8bede23b62dfccb583aa20cef78d0687 |
| cap_audit | tables/residuals_p1_p1b.csv | scratch/isla-cap-audit-2026-10-20/tables/residuals_p1_p1b.csv | c634b0612745169fa4f3b8d7755278cb |
| cap_audit | tables/residuals_p2.csv | scratch/isla-cap-audit-2026-10-20/tables/residuals_p2.csv | 68cbee1e3771c480d81e2442bf67d5fe |
| cap_audit | tables/residuals_p2b.csv | scratch/isla-cap-audit-2026-10-20/tables/residuals_p2b.csv | f779d65ba91171d9fd68e49f2289c6ea |
| cap_audit | tables/probes/pathcheck_drv_d6.json | scratch/isla-cap-audit-2026-10-20/tables/probes/pathcheck_drv_d6.json | b00e2695edb8393ce96ed802b9eb2f6e |
| cap_audit | tables/probes/pathcheck_drv_ref.json | scratch/isla-cap-audit-2026-10-20/tables/probes/pathcheck_drv_ref.json | 6996f00695c4f6dbe13bbdfa404a9db8 |
| cap_audit | tables/probes/pathcheck_drv_s128.json | scratch/isla-cap-audit-2026-10-20/tables/probes/pathcheck_drv_s128.json | bc080b1e46228c4720dd0cedc053cd2c |
| cap_audit | tables/probes/pathcheck_drv_w256.json | scratch/isla-cap-audit-2026-10-20/tables/probes/pathcheck_drv_w256.json | 3e8a43cfbde117ab1c8240ce807ef801 |
| cap_audit | tables/probes/probe_a.json | scratch/isla-cap-audit-2026-10-20/tables/probes/probe_a.json | e78a4025f867112633ecb90466a4fefc |
| cap_audit | tables/probes/probe_bc_cap_drv_isla_rd_s42.json | scratch/isla-cap-audit-2026-10-20/tables/probes/probe_bc_cap_drv_isla_rd_s42.json | 42c04c41ae5dc2d62d7732f040b12198 |
| cap_audit | tables/probes/probe_bc_cap_drv_isla_rd_s43.json | scratch/isla-cap-audit-2026-10-20/tables/probes/probe_bc_cap_drv_isla_rd_s43.json | 739aa09d766de52e7cc23e9e24b7d354 |
| floor_drv | PREREG.md | scratch/isla-floor-drv-2026-10-23/PREREG.md | d04462aa4944a95c74182a236e745f53 |
| az_compliant | PREREG_v2.md | scratch/isla-az-compliant-2026-10-19/PREREG_v2.md | ebd75524aa08dc4f1b01bacea8ac8cc7 |
| boundary_audit | REPORT.md | scratch/isla-boundary-audit-2026-10-23/REPORT.md | 6453db585bc7db322fea6dd9079466ee |
| boundary_audit | BASELINE_PARITY.md | scratch/isla-boundary-audit-2026-10-23/BASELINE_PARITY.md | 83a080c957808105f150931ef3ba8050 |
| boundary_audit | cost_table_isla.md | scratch/isla-boundary-audit-2026-10-23/cost_table_isla.md | 89412ee864d3716906c908457287f483 |
| boundary_audit | cost_table_dtt.md | scratch/isla-boundary-audit-2026-10-23/cost_table_dtt.md | 4a7a31c361c0aa5ba447934736bfd73c |
| boundary_audit | cost_seed_table.md | scratch/isla-boundary-audit-2026-10-23/cost_seed_table.md | 81fef88b5a9ecbacbe0f6ae4658ea3cb |
| drv_ground | classify_summary.txt | scratch/isla-drv-ground-2026-10-23/classify_summary.txt | 9998d780babbae027c338e74cc92d338 |
| drv_ground | check_summary.txt | scratch/isla-drv-ground-2026-10-23/check_summary.txt | 5294b1ae9cfc5040e72655b3294dcd8f |
| drv_ground | offset_runs.txt | scratch/isla-drv-ground-2026-10-23/offset_runs.txt | 45459f48543ec2ef1bfe09433a81a8e9 |
| setup_research | SETUP.md | scratch/isla-setup-research-2026-10-23/SETUP.md | e715bbf2ad0a52d0c8e84c67d425930f |
| af_whole | PREREG.md | scratch/isla-af-whole-2026-10-23/PREREG.md | 2f67af124c696b3b68273ccbaa6abc6b |
| af_whole | FACTS.md | scratch/isla-af-whole-2026-10-23/FACTS.md | 8984de36791f7efdd709dc9d49e76f57 |
| surface_bc | REPORT.md | scratch/isla-surface-bc-2026-10-23/REPORT.md | 5b4c561932cd417d60061761c0f9535c |
| surface_bc | DESIGN.md | scratch/isla-surface-bc-2026-10-23/DESIGN.md | 77ab73f69eb096b1b3ee8c77b668361b |
| surface_bc | cost_table.md | scratch/isla-surface-bc-2026-10-23/cost_table.md | 95e063d222ba37b050af8e86e04114cf |
| surfqpop | RESULTS.md | scratch/isla-surf-probe-2026-10-23/RESULTS.md | 0f3d21e88e94ad66bf6d0cb3de86abef |
| surfqpop | PREREG.md | scratch/isla-surf-probe-2026-10-23/PREREG.md | aadf31792e8f25b5df3611b5382e00ca |
| surfqpop | results/out/reduced.json | scratch/isla-surf-probe-2026-10-23/results/out/reduced.json | 8ff273b3f13f9fbf6fd47384ee054746 |
| surfqpop | results/out/tables.md | scratch/isla-surf-probe-2026-10-23/results/out/tables.md | 42ab3103ca2b1eed5147a694b99911e6 |
| slices_af | RESULTS.md | scratch/isla-slices-2026-10-23/RESULTS.md | 53b1cdc26aa2e79e2c927fc9b76cf711 |
| slices_af | PREREG.md | scratch/isla-slices-2026-10-23/PREREG.md | 0f5c94ff00aaa5b3ed90728ff05757fb |
| slices_af | reading.md | scratch/isla-slices-2026-10-23/reading.md | 9d73e2c6a8a39d354bb0e38094932849 |
| slices_af | reading.json | scratch/isla-slices-2026-10-23/reading.json | b7c3e76fbd71c97a1619e058e395a88a |
| slices_af | nearwall/nearwall_table.md | scratch/isla-slices-2026-10-23/nearwall/nearwall_table.md | 8d69f64113cd92de5d7c4292020e56e4 |
| tie_exp | RESULTS.md | scratch/isla-tie-2026-10-22/RESULTS.md | 4c2cbd599dfd58bb7e666bce0b9b0ab5 |
| tie_exp | PREREG.md | scratch/isla-tie-2026-10-22/PREREG.md | f0e70cb27eee686c3877b7f91eb86686 |
| tie_exp | JOBS.md | scratch/isla-tie-2026-10-22/JOBS.md | 6c18a17e2044618ec3fd4cefde77b418 |
| tie_exp | read/results.md | scratch/isla-tie-2026-10-22/read/results.md | 1da6c5776d71c1c044093059aa7cee76 |
| tie_exp | read/results.json | scratch/isla-tie-2026-10-22/read/results.json | 43f42f214c59bee626b412bb382cea51 |
| fold_sign | PREREG.md | scratch/isla-fold-sign-2026-10-22/PREREG.md | 3fe4597743bb7af6f23fe1e3d707523f |
| fold_sign | f32floor_tables.md | scratch/isla-fold-sign-2026-10-22/f32floor_tables.md | a7974465df17c0b179041e2d7f410230 |
| token_draw | DESIGN.md | scratch/isla-token-draw-2026-10-22/DESIGN.md | 00038f90939bb2c811ac06531a7e2f3d |
| bundle_merge | REPORT.md | scratch/isla-bundle-merge-2026-10-24/REPORT.md | 764a8dc3f644e89648bc8eb1cb893d02 |
| bundle_merge | FINAL_REPORT.md | scratch/isla-bundle-merge-2026-10-24/FINAL_REPORT.md | 71f404a48d9a50174cf652117be44dfb |
| geo_fast | RESULTS.md | scratch/isla-geo-fast-2026-10-23/RESULTS.md | cd9aae8a28eba697fcbe569f0b05eaea |
| geo_fast | FINAL_REPORT.md | scratch/isla-geo-fast-2026-10-23/FINAL_REPORT.md | ffdbf2d2f9f0a4be90dc19db253ff5a9 |
| geo_fast | DESIGN.md | scratch/isla-geo-fast-2026-10-23/DESIGN.md | 4e38af756908e3ea7850a46c3d8dceda |
| geo_fast | lhr/final2_table.md | scratch/isla-geo-fast-2026-10-23/lhr/final2_table.md | 1abb1a21c5e63ea773e008e96618e467 |
| align8 | FINAL_REPORT.md | scratch/isla-align8-2026-10-23/FINAL_REPORT.md | ff1d1a6667efd962eeab64ab9b7fd7a9 |
| align8 | results/repro_table.md | scratch/isla-align8-2026-10-23/results/repro_table.md | 7881b73b38eccbb7bead66a25420648a |
| align8 | exposure/exposure_gpuh.md | scratch/isla-align8-2026-10-23/exposure/exposure_gpuh.md | 6d8273e26cdc11a62c73fc746d5657ac |
| align8 | exposure/exposure_classes.tsv | scratch/isla-align8-2026-10-23/exposure/exposure_classes.tsv | eed81eda1832a1a4f169b1790d9b0ca9 |
| bc_sidecar | REPORT.md | scratch/isla-bc-sidecar-2026-10-23/REPORT.md | 5954c6c2aa03d130a1897b8f91eb2f58 |
| bc_sidecar | FINAL_REPORT.md | scratch/isla-bc-sidecar-2026-10-23/FINAL_REPORT.md | 1dd633f229de15e97177f773396ab495 |
| geo_degen | FINAL_REPORT.md | scratch/isla-geo-degen-2026-10-24/FINAL_REPORT.md | c7d099c33ef217383f2715a19c304f6e |
| geo_degen | RESULTS.md | scratch/isla-geo-degen-2026-10-24/RESULTS.md | f151d5eaeea4afbf2cb8db47a109d19a |
| geo_degen | DESIGN.md | scratch/isla-geo-degen-2026-10-24/DESIGN.md | 18c7bec9e1a16210dc5f1eced9a00a9e |
| fast_merge | REPORT.md | scratch/isla-fast-merge-2026-10-24/REPORT.md | edff7045b7ead28bf0857d3925db41b0 |
| fast_merge | FINAL_REPORT.md | scratch/isla-fast-merge-2026-10-24/FINAL_REPORT.md | 935cc9d05af15b64fd73f6ccbec7a083 |
| fast_merge | lhr/timing_table.md | scratch/isla-fast-merge-2026-10-24/lhr/timing_table.md | c32ba42894dd77779443c99b243b55b5 |
| fast_merge | lhr/book_table.md | scratch/isla-fast-merge-2026-10-24/lhr/book_table.md | e4e2f81ebd5b897d8c0df164db137727 |
| dtt_knobs | STAGE1.md | scratch/isla-dtt-knobs-2026-10-23/STAGE1.md | 1a28dc7cfb43b4ed6cab5784dc717eb8 |
| dtt_knobs | STAGE1_REVIEW.md | scratch/isla-dtt-knobs-2026-10-23/STAGE1_REVIEW.md | 690b4c99d7129a4281ce6262c7cf91d5 |
| dtt_knobs | PREREG.md | scratch/isla-dtt-knobs-2026-10-23/PREREG.md | e601d4af9ff793a7570af7acd9929f43 |
| dtt_knobs | PREREG_REVIEW.md | scratch/isla-dtt-knobs-2026-10-23/PREREG_REVIEW.md | e1a362c6b3d6337fb42d39a64f043994 |
| dtt_knobs | CODE_FACTS.md | scratch/isla-dtt-knobs-2026-10-23/CODE_FACTS.md | 31bae6f8a91a12a7770da09a3c7b0489 |
| dtt_knobs | NOTEBOOK_DRAFT_STAGE1.md | scratch/isla-dtt-knobs-2026-10-23/NOTEBOOK_DRAFT_STAGE1.md | 293fbd0b12dde4686fadf0f0b30414a6 |
| dtt_knobs | NOTEBOOK_REVIEW.md | scratch/isla-dtt-knobs-2026-10-23/NOTEBOOK_REVIEW.md | 64ee69144f8aa2d997079c81c144b369 |
| dtt_knobs | NOTEBOOK_REVISION_NOTE.md | scratch/isla-dtt-knobs-2026-10-23/NOTEBOOK_REVISION_NOTE.md | 72af8b41a253c05ce89301efe2bb25af |
| dtt_knobs | sens/REPORT.md | scratch/isla-dtt-knobs-2026-10-23/sens/REPORT.md | 1fb1c8086d02a04a343e54849f1b6d30 |
| dtt_knobs | sens/sens_table.md | scratch/isla-dtt-knobs-2026-10-23/sens/sens_table.md | 588f626614251d31541059dfdf75bddd |
| dtt_knobs | sens/sens_regions.md | scratch/isla-dtt-knobs-2026-10-23/sens/sens_regions.md | f3a4281d56819ffe750f9758187e4cd4 |
| dtt_knobs | cost/REPORT.md | scratch/isla-dtt-knobs-2026-10-23/cost/REPORT.md | e8c8ce5d0d6877d34a570b8ae36f4a14 |
| dtt_knobs | cost/cost_table.md | scratch/isla-dtt-knobs-2026-10-23/cost/cost_table.md | 645c6958f1888a17defa09bc12891545 |
| dtt_knobs | cap/REPORT.md | scratch/isla-dtt-knobs-2026-10-23/cap/REPORT.md | 95ba4928bbeb1e3b2f6d16411cc27270 |
| dtt_knobs | cap/cap_table.md | scratch/isla-dtt-knobs-2026-10-23/cap/cap_table.md | ae192e13fa8f22c71aa23b853e91454e |
| dtt_knobs | order/REPORT.md | scratch/isla-dtt-knobs-2026-10-23/order/REPORT.md | 6ccfa746846449fdd421a27ab4160acb |
| dtt_knobs | order/order_table.md | scratch/isla-dtt-knobs-2026-10-23/order/order_table.md | 5473f50155c46d64453f7e22193437ef |
| dtt_knobs | gap/REPORT.md | scratch/isla-dtt-knobs-2026-10-23/gap/REPORT.md | 702bfaea7a18e2edc57770cb6bb8c0d7 |
| dtt_knobs | gap/gap_table.md | scratch/isla-dtt-knobs-2026-10-23/gap/gap_table.md | 55f2df0bee09f03b54d78e064ef5d3d0 |
| dtt_knobs | gap/gap_table_bf16.md | scratch/isla-dtt-knobs-2026-10-23/gap/gap_table_bf16.md | af8fecfdbf703563d91a50aae792b9dc |
| dtt_knobs | gap/rough.md | scratch/isla-dtt-knobs-2026-10-23/gap/rough.md | ee1c7c32869ac1f6a465c2f9bc0caf78 |
| dtt_knobs | treeprec/REPORT.md | scratch/isla-dtt-knobs-2026-10-23/treeprec/REPORT.md | 083b766dbcc22186a67a913cd9e1a0e6 |
| dtt_knobs | treeprec/treeprec_table.md | scratch/isla-dtt-knobs-2026-10-23/treeprec/treeprec_table.md | a7b3e116a34ef990dad6e7f8dcdd32a1 |
| dtt_knobs | treeprec/signtest.md | scratch/isla-dtt-knobs-2026-10-23/treeprec/signtest.md | 2bbe2cce1e8463aa627b08a0c6244e95 |
| dtt_fast_compile | RESULTS.md | scratch/isla-dtt-fast-2026-10-23/compile/RESULTS.md | 884f4b2354d539c8ec1ad582e8c3211c |
| dtt_fast_compile | REPORT.md | scratch/isla-dtt-fast-2026-10-23/compile/REPORT.md | 296118e95eab0cc507c6bc77ad8ab104 |
| dtt_fast_verify_fusedread | VERIFY.md | scratch/isla-dtt-fast-2026-10-23/verify_fusedread/VERIFY.md | 3547c3068a6ef54d514468ea1a7d4364 |
| dtt_fast_integration | RESULTS.md | scratch/isla-dtt-fast-2026-10-23/integration/RESULTS.md | cd508ec6241c605c2464c0f4541ef70b |
| boundary_merge | REPORT.md | scratch/isla-merge-prep-2026-10-24/REPORT.md | 511537a48ee5f289a38f253978a7b050 |
| boundary_merge | FINAL_REPORT.md | scratch/isla-merge-prep-2026-10-24/FINAL_REPORT.md | 1a67141eff23bbe060cf5e50b0bdf536 |
| boundary_merge | book/BOOK_CHANGES.md | scratch/isla-merge-prep-2026-10-24/book/BOOK_CHANGES.md | bf07eb9c2fd38aafec43369c1d2adcfe |
| boundary_merge | results/TIMING.md | scratch/isla-merge-prep-2026-10-24/results/TIMING.md | e0f06a4a592bc54c93a3c5aab23c478f |
| boundary_merge | results/book_diff_7213_vs_17c9df5c0.txt | scratch/isla-merge-prep-2026-10-24/results/book_diff_7213_vs_17c9df5c0.txt | 61c79e81a7ceb411c7ce765f361e7d30 |
| dtt_pose | DESIGN.md | scratch/isla-dtt-pose-2026-10-24/DESIGN.md | 8420670e9d9f67957d4c4d7ab05d3599 |
| dtt_pose | MECHANISMS.md | scratch/isla-dtt-pose-2026-10-24/MECHANISMS.md | 2e89f066f0578021a466e768f5c76bc0 |
| dtt_pose_exp | PREREG.md | scratch/isla-dtt-pose-exp-2026-10-24/PREREG.md | 2f46a10cc7cc9f3f8ba9a39c98cb6d39 |
| geo_thin | DESIGN.md | scratch/isla-geo-thin-2026-10-24/DESIGN.md | f7f8ed6849f68cc8a4584989afb1a27e |
| geo_thin | RESULTS.md | scratch/isla-geo-thin-2026-10-24/RESULTS.md | 6daa28650f27ed75c455c12e89fbd25d |
| dtt_fast_final | TABLE.md | scratch/isla-dtt-fast-2026-10-23/final/TABLE.md | a6f352dfc32a342dc1f95a90bba69c17 |
| nearwall | DESIGN.md | scratch/isla-nearwall-2026-10-24/DESIGN.md | 1ff494b61dd700bb27f6eb9a56eae851 |
| nearwall | RESULTS.md | scratch/isla-nearwall-2026-10-24/RESULTS.md | 43888a105e71484145e06785ce13fb70 |
| nearwall | REVIEW_DESIGN.md | scratch/isla-nearwall-2026-10-24/REVIEW_DESIGN.md | 08b552800c395fd6c49a0c7e26a7f2f1 |
| hl_dup | PREREG.md | scratch/isla-hl-dup-2026-10-24/PREREG.md | 072b6b1c3ac1b59df729eadda9f04e40 |
| hl_dup | RESULTS.md | scratch/isla-hl-dup-2026-10-24/RESULTS.md | e1ac7e097098562bef96b41f5744910a |
| hl_dup | NOTEBOOK_DRAFT.md | scratch/isla-hl-dup-2026-10-24/NOTEBOOK_DRAFT.md | 0c2f575b17dc0036ca525c1c97eefba9 |
| hl_dup | REVIEW_STAGE1.md | scratch/isla-hl-dup-2026-10-24/REVIEW_STAGE1.md | 17357946600875a1a44b10705b1b88b8 |
| hl_dup | REVISION_NOTE.md | scratch/isla-hl-dup-2026-10-24/REVISION_NOTE.md | 7883e89de248a357e1024831e8553c04 |
| hl_dup | HANDOFF_HLDUP.md | scratch/isla-hl-dup-2026-10-24/HANDOFF_HLDUP.md | d8fbfbee0c2401f7a6dca072bd78f9ee |
| hl_dup | FINAL_REPORT.md | scratch/isla-hl-dup-2026-10-24/FINAL_REPORT.md | 74bc30896f511fcd9513fb01eb935e9f |
| geo_thin_final | FINAL_REPORT.md | scratch/isla-geo-thin-2026-10-24/FINAL_REPORT.md | fd0d43090cc369e7dfb3bee51778cdc6 |
| geo_thin_final | DESIGN.md | scratch/isla-geo-thin-2026-10-24/DESIGN.md | be6a283c8a973088cc6348ad8fd706f8 |
| geo_thin_final | RESULTS.md | scratch/isla-geo-thin-2026-10-24/RESULTS.md | 50924a0e8a2c5441117b05cde4d8a4fb |
| nearwall_final | FINAL_REPORT.md | scratch/isla-nearwall-2026-10-24/FINAL_REPORT.md | 163bf7637092883b4ef1ace0d29ba33b |
| nearwall_final | PREREG_CONTINGENCY.md | scratch/isla-nearwall-2026-10-24/PREREG_CONTINGENCY.md | 26856eb98d681596301c37bdb7d3ae0d |
| nearwall_final | GEOMETRY_NOTE.md | scratch/isla-nearwall-2026-10-24/GEOMETRY_NOTE.md | 6318e44c3819cbceb854772ba6e6254e |
| dtt_fast_bf16diag | REPORT.md | scratch/isla-dtt-fast-2026-10-23/bf16diag/REPORT.md | 0cba3be885275a1e86191deec4089ad4 |
| dtt_fast_bf16diag | RESULTS.md | scratch/isla-dtt-fast-2026-10-23/bf16diag/RESULTS.md | 8a8ad8b4de7f050e7ecd0f5db9b70e76 |
| az_decide_final | RESULTS.md | scratch/isla-az-decide-2026-10-21/RESULTS.md | ac55a742cc244126355a787d3f37e3ba |
| az_decide_final | tables_final.md | scratch/isla-az-decide-2026-10-21/tables_final.md | 1f291202615015dbf31c2b84fc7a3c2f |
| az_decide_final | percase_final.md | scratch/isla-az-decide-2026-10-21/percase_final.md | 856f67b2c0f3e27391691a32a15a8e28 |
| az_decide_final | nonbox_scales_final.md | scratch/isla-az-decide-2026-10-21/nonbox_scales_final.md | c838c9d5e975b04d2434035f1f223cbd |
| az_decide_final | QUERYPOP090_FOR_12F.md | scratch/isla-az-decide-2026-10-21/QUERYPOP090_FOR_12F.md | 3acdf6dc08d8fe833207e094dd29f8cd |
| az_decide_final | CROSS_DATASET_INPUTS.md | scratch/isla-az-decide-2026-10-21/CROSS_DATASET_INPUTS.md | e3a31e8ba715c4b8fb0dbe70a4fbb7a9 |
| az_decide_final | reynolds/RESULTS_reynolds.md | scratch/isla-az-decide-2026-10-21/reynolds/RESULTS_reynolds.md | 44d33cea346a6fbabeaf593e98467178 |
| e1_az | RESULTS_AZ_2026-10-22.md | scratch/isla-headnorm-2026-10-18/RESULTS_AZ_2026-10-22.md | 913826319bcc2c985798e4c510e4ffa8 |
| lre_az | RESULTS.md | scratch/isla-lre-az-2026-10-23/RESULTS.md | 0e32ac357ddcc1a9322b2e57b41ae778 |
| lre_az | tables.md | scratch/isla-lre-az-2026-10-23/tables.md | 7ff39e128f00f5543f52141e6df928c5 |
| lre_az | lreaz.json | scratch/isla-lre-az-2026-10-23/lreaz.json | 2ec3cbd05b9dc2b0fd18e9c5c8713738 |
| lre_az | posthoc.md | scratch/isla-lre-az-2026-10-23/posthoc.md | 5dc5f708a411e02b19b8f68f07fd7990 |
| lre_az | PREREG.md | scratch/isla-lre-az-2026-10-23/PREREG.md | e00823cdece3ecce8cabf0c2ec3f5a34 |
| geo_band_phase0 | PHASE0.md | scratch/isla-geo-band-2026-10-24/PHASE0.md | 3cf35c2f331938c0faf2de78eaeba261 |
| geo_band_phase0 | lhr/timing_table.md | scratch/isla-geo-band-2026-10-24/lhr/timing_table.md | a720b5ee3c2a3fc656ba95217abaf33b |
| kernel_tests | FINAL_REPORT.md | scratch/isla-kernel-tests-2026-10-24/FINAL_REPORT.md | f4789cdeeae4d0afdc60643e40102b70 |
| kernel_tests | RESULTS.md | scratch/isla-kernel-tests-2026-10-24/RESULTS.md | 915e4f088d60a4891cbc98c7fb89cb7f |
| slices_drv | RESULTS.md | scratch/isla-slices-drv-2026-10-24/RESULTS.md | b2fd11e2b478914e60f53a4b85035808 |
| slices_drv | reading.md | scratch/isla-slices-drv-2026-10-24/reading.md | a810e1a022da3b172820066fa5d6d685 |
| slices_drv | PREREG.md | scratch/isla-slices-drv-2026-10-24/PREREG.md | 88e36cd7e36232773f080309f7935c77 |
| slices_drv | farfield/farfield_table.md | scratch/isla-slices-drv-2026-10-24/farfield/farfield_table.md | f993d7713536e50a78f5dcf80421ca53 |
| freeze_plan | FINAL_RERUN_PLAN.md | scratch/isla-freeze-plan-2026-10-24/FINAL_RERUN_PLAN.md | 33026a229946ed9ca393cd72225b088f |
| bc_supply | REPORT.md | scratch/isla-bc-supply-2026-10-24/REPORT.md | 5c3bac05e27f68f8e34a73f372126d24 |
| bc_supply | FINAL_REPORT.md | scratch/isla-bc-supply-2026-10-24/FINAL_REPORT.md | e0bfcecce4b125b13dd5577a1304eaf0 |
| bc_supply | book/BOOK_CHANGES.md | scratch/isla-bc-supply-2026-10-24/book/BOOK_CHANGES.md | 509bc87c771038f7ee009923cbb6f96f |
| bc_supply | out/hl_val_368914.jsonl | scratch/isla-bc-supply-2026-10-24/out/hl_val_368914.jsonl | cb5c0f152fa98f934c19db5388714c68 |
| bc_supply | out/move_369235.jsonl | scratch/isla-bc-supply-2026-10-24/out/move_369235.jsonl | 195c322778304c0866e0ba53a08be8d2 |
| bc_supply | gb/smoke_summary.txt | scratch/isla-bc-supply-2026-10-24/gb/smoke_summary.txt | c6a833c753006178ce4beb9a249d707b |
| geo_band_final | FINAL_REPORT.md | scratch/isla-geo-band-2026-10-24/FINAL_REPORT.md | efc3cf9ca39930b8fae8e620f06e0d2b |
| geo_band_final | RESULTS.md | scratch/isla-geo-band-2026-10-24/RESULTS.md | e5cf24a2701e3d511d0b3ac23a4b39ba |
| geo_band_final | DERIVATION.md | scratch/isla-geo-band-2026-10-24/DERIVATION.md | de7e33f0fa47cbb048cdb73db3e65294 |
| geo_band_final | INFER_CHECK.md | scratch/isla-geo-band-2026-10-24/INFER_CHECK.md | 0efe2d6f9a29ab8079f3bb2e4139f470 |
| geo_band_final | REVIEW_DERIVATION.md | scratch/isla-geo-band-2026-10-24/REVIEW_DERIVATION.md | f01d6f974622f24d65967177fca11f80 |
| geo_band_final | DESIGN.md | scratch/isla-geo-band-2026-10-24/DESIGN.md | 15bdc0a3c638b066752f1e96afdc42d8 |
| band_merge | FINAL_REPORT.md | scratch/isla-band-merge-2026-10-24/FINAL_REPORT.md | dae15af49551846dbd1df3a330c5f14b |
| band_merge | REVIEW_TESTS.md | scratch/isla-band-merge-2026-10-24/REVIEW_TESTS.md | dfae97e0f29472178221cee3fea9b34c |
| default_dtt | FINAL_REPORT.md | scratch/isla-default-dtt-2026-10-24/FINAL_REPORT.md | dd59de3344620187d373fbf7c8c4a327 |
| default_dtt | BOOK_CHANGES.md | scratch/isla-default-dtt-2026-10-24/BOOK_CHANGES.md | 13b94d1b279ae7006b5ba3a6eab8485e |
| default_dtt | BOOK_SENTENCES_v2.md | scratch/isla-default-dtt-2026-10-24/BOOK_SENTENCES_v2.md | d28e0fcaff3fb5688185a116520f3046 |
| dtt_default_int | FINAL_REPORT.md | scratch/isla-dtt-default-int-2026-10-24/FINAL_REPORT.md | 3377379b0a7d0322d0d9e951e6c76b7f |
| geo_thinfast | FINAL_REPORT.md | scratch/isla-geo-thinfast-2026-10-24/FINAL_REPORT.md | 24a0972e7e8f42d52aad6582f5d4ffd8 |
| geo_thinfast | RESULTS.md | scratch/isla-geo-thinfast-2026-10-24/RESULTS.md | 137b55986608d7a51fcbdce76576c0a2 |
| geo_thinfast | REVIEW.md | scratch/isla-geo-thinfast-2026-10-24/REVIEW.md | fbe1ab69c0e59019899e1e9c5639d601 |
| geo_thinfast | lhr/timing_table.md | scratch/isla-geo-thinfast-2026-10-24/lhr/timing_table.md | c6326b24b902dc7b703a3e324abccce7 |
| dtt_fast_done | FINAL_REPORT.md | scratch/isla-dtt-fast-2026-10-23/FINAL_REPORT.md | c198248873c77f006cd667dcd416e9dc |
| dtt_fast_done | integration/RESULTS.md | scratch/isla-dtt-fast-2026-10-23/integration/RESULTS.md | 21462acfca849c21c635f470a288db5b |
| dtt_fast_done | final2/TABLE.md | scratch/isla-dtt-fast-2026-10-23/final2/TABLE.md | 1d256f9fbf240a7dc1676b0a001d44db |
| dtt_fast_done | pose/RESULTS.md | scratch/isla-dtt-fast-2026-10-23/pose/RESULTS.md | 4f2daf8f3af4c3db8d79826892ff3f35 |
| dtt_fast_done | spike/RESULTS.md | scratch/isla-dtt-fast-2026-10-23/spike/RESULTS.md | c710b48e47df4b2069b16274a2082a47 |
| dtt_fast_done | flag/RESULTS.md | scratch/isla-dtt-fast-2026-10-23/flag/RESULTS.md | c37e8f62600a678b9be3474728c53635 |
| dtt_fast_done | flag/REPORT.md | scratch/isla-dtt-fast-2026-10-23/flag/REPORT.md | 846bcae76c1ff1ff9e69ae4547b02431 |
| hl_dup_stage2 | RESULTS_STAGE2.md | scratch/isla-hl-dup-2026-10-24/RESULTS_STAGE2.md | 4d9fe204add4b8495e8b549d4063c04f |
| hl_dup_stage2 | stage2_table.md | scratch/isla-hl-dup-2026-10-24/stage2_table.md | d620b6a015d0efea3d8813e1e24bc953 |
| hl_dup_stage2 | REVIEW_STAGE2.md | scratch/isla-hl-dup-2026-10-24/REVIEW_STAGE2.md | ff43023948b2e839396ed908cb90c234 |
| hl_dup_stage2 | REVISION_NOTE_STAGE2.md | scratch/isla-hl-dup-2026-10-24/REVISION_NOTE_STAGE2.md | 91947532fa10cb88768927a7b09da87f |
| hl_dup_stage2 | HOLE_CRUST.md | scratch/isla-hl-dup-2026-10-24/HOLE_CRUST.md | 64cc1c9b050c0498c0f2ae434ed45baa |
| hl_dup_stage2 | FINAL_REPORT.md | scratch/isla-hl-dup-2026-10-24/FINAL_REPORT.md | 5b9c637c63009c38280fb691386f82d0 |
| dtt_fast_final_table_p5 | TABLE.md | scratch/isla-dtt-fast-2026-10-23/final/TABLE.md | 9602478cd5dca559fdd1a5326785be17 |
| hl_dup_stage2_scores | scores_stage2/hldupx_hl_gt_seed42/native/hldupx_hl_gt_seed42/metrics.jsonl | scratch/isla-hl-dup-2026-10-24/scores_stage2/hldupx_hl_gt_seed42/native/hldupx_hl_gt_seed42/metrics.jsonl | 2c8a692b53b6ee5ed5278d9614feb996 |
| hl_dup_stage2_scores | scores_stage2/hldupx_hl_gt_seed42/posed/hldupx_hl_gt_seed42/metrics.jsonl | scratch/isla-hl-dup-2026-10-24/scores_stage2/hldupx_hl_gt_seed42/posed/hldupx_hl_gt_seed42/metrics.jsonl | 3a595a7cfd50dfeaf03ce99af5805459 |
| hl_dup_stage2_scores | scores_stage2/hldupx_hl_gt_seed43/native/hldupx_hl_gt_seed43/metrics.jsonl | scratch/isla-hl-dup-2026-10-24/scores_stage2/hldupx_hl_gt_seed43/native/hldupx_hl_gt_seed43/metrics.jsonl | ca1ea7be8a1de51fcef79aa6ecbefcb7 |
| hl_dup_stage2_scores | scores_stage2/hldupx_hl_gt_seed43/posed/hldupx_hl_gt_seed43/metrics.jsonl | scratch/isla-hl-dup-2026-10-24/scores_stage2/hldupx_hl_gt_seed43/posed/hldupx_hl_gt_seed43/metrics.jsonl | 06f1ffe519c0719ee4e6b28047296493 |
| hl_dup_stage2_scores | scores_stage2/hldupx_hl_isla_seed42/native/hldupx_hl_isla_seed42/metrics.jsonl | scratch/isla-hl-dup-2026-10-24/scores_stage2/hldupx_hl_isla_seed42/native/hldupx_hl_isla_seed42/metrics.jsonl | 6284d609d2101d968773b3ae12e3a3f2 |
| hl_dup_stage2_scores | scores_stage2/hldupx_hl_isla_seed42/posed/hldupx_hl_isla_seed42/metrics.jsonl | scratch/isla-hl-dup-2026-10-24/scores_stage2/hldupx_hl_isla_seed42/posed/hldupx_hl_isla_seed42/metrics.jsonl | 0bd487062dd8346709e2018813579a74 |
| hl_dup_stage2_scores | scores_stage2/hldupx_hl_isla_seed43/native/hldupx_hl_isla_seed43/metrics.jsonl | scratch/isla-hl-dup-2026-10-24/scores_stage2/hldupx_hl_isla_seed43/native/hldupx_hl_isla_seed43/metrics.jsonl | f9cf17fb3cbb7cd310a97d0f3d2c012a |
| hl_dup_stage2_scores | scores_stage2/hldupx_hl_isla_seed43/posed/hldupx_hl_isla_seed43/metrics.jsonl | scratch/isla-hl-dup-2026-10-24/scores_stage2/hldupx_hl_isla_seed43/posed/hldupx_hl_isla_seed43/metrics.jsonl | 8e971238dd5bb3c0bae7759d000a8ee2 |
| hl_dup_stage2_scores | scores_stage2/hldupx_hl_t_seed42/native/hldupx_hl_t_seed42/metrics.jsonl | scratch/isla-hl-dup-2026-10-24/scores_stage2/hldupx_hl_t_seed42/native/hldupx_hl_t_seed42/metrics.jsonl | dde305fc03fcc703aaba9681a964a614 |
| hl_dup_stage2_scores | scores_stage2/hldupx_hl_t_seed42/posed/hldupx_hl_t_seed42/metrics.jsonl | scratch/isla-hl-dup-2026-10-24/scores_stage2/hldupx_hl_t_seed42/posed/hldupx_hl_t_seed42/metrics.jsonl | ae96e10dbbb40d31f198a5ef1fbfa5b1 |
| hl_dup_stage2_scores | scores_stage2/hldupx_hl_t_seed43/native/hldupx_hl_t_seed43/metrics.jsonl | scratch/isla-hl-dup-2026-10-24/scores_stage2/hldupx_hl_t_seed43/native/hldupx_hl_t_seed43/metrics.jsonl | 968204a4a6a1336c67828f14fa8606ab |
| hl_dup_stage2_scores | scores_stage2/hldupx_hl_t_seed43/posed/hldupx_hl_t_seed43/metrics.jsonl | scratch/isla-hl-dup-2026-10-24/scores_stage2/hldupx_hl_t_seed43/posed/hldupx_hl_t_seed43/metrics.jsonl | 6746a1603bb71d32ed24aa8a8c9889fc |
| hl_dup_stage2_scores | scores_stage2/hldupx_hl_gt_seed42/native/DATASET_USED | scratch/isla-hl-dup-2026-10-24/scores_stage2/hldupx_hl_gt_seed42/native/DATASET_USED | 6d936e405bec5d944e3197faf2171466 |
| hl_dup_stage2_scores | scores_stage2/hldupx_hl_gt_seed42/posed/DATASET_USED | scratch/isla-hl-dup-2026-10-24/scores_stage2/hldupx_hl_gt_seed42/posed/DATASET_USED | e3a760d41bc7137514d3c181fb84126f |
| hl_dup_stage2_scores | scores_stage2/hldupx_hl_gt_seed43/native/DATASET_USED | scratch/isla-hl-dup-2026-10-24/scores_stage2/hldupx_hl_gt_seed43/native/DATASET_USED | 6d936e405bec5d944e3197faf2171466 |
| hl_dup_stage2_scores | scores_stage2/hldupx_hl_gt_seed43/posed/DATASET_USED | scratch/isla-hl-dup-2026-10-24/scores_stage2/hldupx_hl_gt_seed43/posed/DATASET_USED | e3a760d41bc7137514d3c181fb84126f |
| hl_dup_stage2_scores | scores_stage2/hldupx_hl_isla_seed42/native/DATASET_USED | scratch/isla-hl-dup-2026-10-24/scores_stage2/hldupx_hl_isla_seed42/native/DATASET_USED | 6d936e405bec5d944e3197faf2171466 |
| hl_dup_stage2_scores | scores_stage2/hldupx_hl_isla_seed42/posed/DATASET_USED | scratch/isla-hl-dup-2026-10-24/scores_stage2/hldupx_hl_isla_seed42/posed/DATASET_USED | e3a760d41bc7137514d3c181fb84126f |
| hl_dup_stage2_scores | scores_stage2/hldupx_hl_isla_seed43/native/DATASET_USED | scratch/isla-hl-dup-2026-10-24/scores_stage2/hldupx_hl_isla_seed43/native/DATASET_USED | 6d936e405bec5d944e3197faf2171466 |
| hl_dup_stage2_scores | scores_stage2/hldupx_hl_isla_seed43/posed/DATASET_USED | scratch/isla-hl-dup-2026-10-24/scores_stage2/hldupx_hl_isla_seed43/posed/DATASET_USED | e3a760d41bc7137514d3c181fb84126f |
| hl_dup_stage2_scores | scores_stage2/hldupx_hl_t_seed42/native/DATASET_USED | scratch/isla-hl-dup-2026-10-24/scores_stage2/hldupx_hl_t_seed42/native/DATASET_USED | 6d936e405bec5d944e3197faf2171466 |
| hl_dup_stage2_scores | scores_stage2/hldupx_hl_t_seed42/posed/DATASET_USED | scratch/isla-hl-dup-2026-10-24/scores_stage2/hldupx_hl_t_seed42/posed/DATASET_USED | e3a760d41bc7137514d3c181fb84126f |
| hl_dup_stage2_scores | scores_stage2/hldupx_hl_t_seed43/native/DATASET_USED | scratch/isla-hl-dup-2026-10-24/scores_stage2/hldupx_hl_t_seed43/native/DATASET_USED | 6d936e405bec5d944e3197faf2171466 |
| hl_dup_stage2_scores | scores_stage2/hldupx_hl_t_seed43/posed/DATASET_USED | scratch/isla-hl-dup-2026-10-24/scores_stage2/hldupx_hl_t_seed43/posed/DATASET_USED | e3a760d41bc7137514d3c181fb84126f |
| hl_dup_stage2_scores | score_summary.json | scratch/isla-hl-dup-2026-10-24/score_summary.json | b49b4a6baa0dcade745dc99c6df0848e |
| hl_dup_stage2_scores | score_table.md | scratch/isla-hl-dup-2026-10-24/score_table.md | 17b91d1106afc2b951494c10a72fba37 |
| hl_dup_stage2_scores | stage2_summary.json | scratch/isla-hl-dup-2026-10-24/stage2_summary.json | f08ac2ef904f1ef561cc67d2d646df28 |
| hl_symplane | FINAL_REPORT.md | scratch/isla-hl-symplane-2026-10-24/FINAL_REPORT.md | 11ee5fb8db9ef973aa63092083ceaecd |
| hl_symplane | RESULTS.md | scratch/isla-hl-symplane-2026-10-24/RESULTS.md | 399ad8e84bad76bbb3e70c2c5af556a7 |
| hl_symplane | BOOK_CHANGES.md | scratch/isla-hl-symplane-2026-10-24/BOOK_CHANGES.md | 4c2a160497d0b3f05023f97e02e6b776 |
| hl_dup_stage2_area | RESULTS.md | scratch/isla-hl-area-2026-10-24/RESULTS.md | 3e2efaff2a2dbaed9f93904ee92723f2 |
| hl_dup_stage2_area | FINAL_REPORT.md | scratch/isla-hl-area-2026-10-24/FINAL_REPORT.md | 49371bd5f75dc0076f9fcd2ebcc54cf6 |
| hl_dup_stage2_area | NOTEBOOK_DRAFT.md | scratch/isla-hl-area-2026-10-24/NOTEBOOK_DRAFT.md | 21c27372d43b2cee5ee77886b5ff4e3d |
| hl_dup_stage2_area | area_table.md | scratch/isla-hl-area-2026-10-24/area_table.md | 784802a14d10f14c59c0e21e8c58c216 |
| hl_dup_stage2_area | area_summary.json | scratch/isla-hl-area-2026-10-24/area_summary.json | feb1aff2f4c133955e0c5e1c51e4be49 |
| hl_dup_stage2_area | COST_REPORT.md | scratch/isla-hl-area-2026-10-24/COST_REPORT.md | 6a0492af470839fd7a0e3d7cad9ba111 |
| hl_dup_stage2_area | cost/cost_hldupx.json | scratch/isla-hl-area-2026-10-24/cost/cost_hldupx.json | 313ddf9b089678b52bb4614f517cd45f |
| hl_dup_stage2_area | cost/extract.jsonl | scratch/isla-hl-area-2026-10-24/cost/extract.jsonl | 3879a90c82b6a1f8d1ecace3d3c6e056 |
| hl_dup_stage2_area | scores_area/hldupx_hl_gt_seed42/area/hldupx_hl_gt_seed42/metrics.jsonl | scratch/isla-hl-area-2026-10-24/scores_area/hldupx_hl_gt_seed42/area/hldupx_hl_gt_seed42/metrics.jsonl | 612ad290b0edcb01b37cbfdc8c413dfc |
| hl_dup_stage2_area | scores_area/hldupx_hl_gt_seed43/area/hldupx_hl_gt_seed43/metrics.jsonl | scratch/isla-hl-area-2026-10-24/scores_area/hldupx_hl_gt_seed43/area/hldupx_hl_gt_seed43/metrics.jsonl | d6dfaa8b73f18fceffbc64d530d4ee53 |
| hl_dup_stage2_area | scores_area/hldupx_hl_isla_seed42/area/hldupx_hl_isla_seed42/metrics.jsonl | scratch/isla-hl-area-2026-10-24/scores_area/hldupx_hl_isla_seed42/area/hldupx_hl_isla_seed42/metrics.jsonl | 8ef1974ef04cd5958f45eedad40d94d2 |
| hl_dup_stage2_area | scores_area/hldupx_hl_isla_seed43/area/hldupx_hl_isla_seed43/metrics.jsonl | scratch/isla-hl-area-2026-10-24/scores_area/hldupx_hl_isla_seed43/area/hldupx_hl_isla_seed43/metrics.jsonl | 3e7bf1a2f67041f0fe3a9731c4a77159 |
| hl_dup_stage2_area | scores_area/hldupx_hl_t_seed42/area/hldupx_hl_t_seed42/metrics.jsonl | scratch/isla-hl-area-2026-10-24/scores_area/hldupx_hl_t_seed42/area/hldupx_hl_t_seed42/metrics.jsonl | 52a6bb4b68451266bf62a961cbfc4b67 |
| hl_dup_stage2_area | scores_area/hldupx_hl_t_seed43/area/hldupx_hl_t_seed43/metrics.jsonl | scratch/isla-hl-area-2026-10-24/scores_area/hldupx_hl_t_seed43/area/hldupx_hl_t_seed43/metrics.jsonl | 2eb33a9b3227e96f8bb83ed2f022ad68 |
| hl_dup_stage2_area | scores_area/hldupx_hl_gt_seed42/area/DATASET_USED | scratch/isla-hl-area-2026-10-24/scores_area/hldupx_hl_gt_seed42/area/DATASET_USED | 9beeb4739f8b4a0b198608ed2dcb4e05 |
| hl_dup_stage2_area | scores_area/hldupx_hl_gt_seed43/area/DATASET_USED | scratch/isla-hl-area-2026-10-24/scores_area/hldupx_hl_gt_seed43/area/DATASET_USED | 9beeb4739f8b4a0b198608ed2dcb4e05 |
| hl_dup_stage2_area | scores_area/hldupx_hl_isla_seed42/area/DATASET_USED | scratch/isla-hl-area-2026-10-24/scores_area/hldupx_hl_isla_seed42/area/DATASET_USED | 9beeb4739f8b4a0b198608ed2dcb4e05 |
| hl_dup_stage2_area | scores_area/hldupx_hl_isla_seed43/area/DATASET_USED | scratch/isla-hl-area-2026-10-24/scores_area/hldupx_hl_isla_seed43/area/DATASET_USED | 9beeb4739f8b4a0b198608ed2dcb4e05 |
| hl_dup_stage2_area | scores_area/hldupx_hl_t_seed42/area/DATASET_USED | scratch/isla-hl-area-2026-10-24/scores_area/hldupx_hl_t_seed42/area/DATASET_USED | 9beeb4739f8b4a0b198608ed2dcb4e05 |
| hl_dup_stage2_area | scores_area/hldupx_hl_t_seed43/area/DATASET_USED | scratch/isla-hl-area-2026-10-24/scores_area/hldupx_hl_t_seed43/area/DATASET_USED | 9beeb4739f8b4a0b198608ed2dcb4e05 |
| isla_api_013 | FINAL_REPORT.md | scratch/isla-api-013-2026-10-24/FINAL_REPORT.md | 00824edd70e9c78705603ac46ca36dd4 |
| isla_api_013 | DESIGN.md | scratch/isla-api-013-2026-10-24/DESIGN.md | 3fa9227c0cc9d24e37523c0e65adeb76 |
| isla_api_013 | BOOK_CHANGES.md | scratch/isla-api-013-2026-10-24/BOOK_CHANGES.md | 471c1a6b15efbc93cd2bf0d52f80383a |
| isla_api_013 | BOOK_PROSE.md | scratch/isla-api-013-2026-10-24/BOOK_PROSE.md | a213ff8efa724cc241f9c204117276d7 |
| isla_api_013 | DELETED_TESTS.md | scratch/isla-api-013-2026-10-24/DELETED_TESTS.md | 1930f91d3f428b329c10a268b3bf7ca6 |
| isla_api_013 | DELETED_TESTS_RE.md | scratch/isla-api-013-2026-10-24/DELETED_TESTS_RE.md | c28917aae2c912f3af9669b039e7e230 |
| isla_api_013 | ASSERTION_AUDIT.md | scratch/isla-api-013-2026-10-24/ASSERTION_AUDIT.md | 1422d98b93f707b47bd416370fd0d043 |
| isla_api_013 | JOBS.md | scratch/isla-api-013-2026-10-24/JOBS.md | 441ac4a4fb8bad9fddbf09b5a6db04e8 |
| isla_api_013 | logs/lhr/gb_377645.log | scratch/isla-api-013-2026-10-24/logs/lhr/gb_377645.log | d15086b3dfe5999f464f96241e0dee7b |
| reynolds_out | DESIGN.md | scratch/isla-reynolds-out-2026-10-24/DESIGN.md | 44f40d13d7577dd42a9327e6837f4543 |
| reynolds_out | FINAL_REPORT.md | scratch/isla-reynolds-out-2026-10-24/FINAL_REPORT.md | 62256c8f033f502a802a338c76131e8b |
| reynolds_out | bitwise/RESULTS_BITWISE.md | scratch/isla-reynolds-out-2026-10-24/bitwise/RESULTS_BITWISE.md | af1c962a0c00e5cf43e5c70a2d16af01 |
| reynolds_out | bitwise/JOBS.md | scratch/isla-reynolds-out-2026-10-24/bitwise/JOBS.md | 5e512c7119ec3e8b488f31669d6849bf |
| reynolds_out | bitwise/z_cases_376561.tsv | scratch/isla-reynolds-out-2026-10-24/bitwise/z_cases_376561.tsv | 6b0fb046be4818b1f4a2e6bf58648ab8 |
| az_spec_audit | AUDIT.md | scratch/isla-az-spec-audit-2026-10-24/AUDIT.md | 448bd6f1b6b8a0a63df73cf3feef717e |
| az_rescore | RESULTS.md | scratch/isla-az-rescore-2026-10-24/RESULTS.md | 20ba0df7eab971666fe4bea93ced9d4a |
| az_rescore | NOTEBOOK_DRAFT.md | scratch/isla-az-rescore-2026-10-24/NOTEBOOK_DRAFT.md | 7fb158a11a34a0754de24999965a7a7d |
| az_rescore | AEROZOO_SPECMETRIC.json | scratch/isla-az-rescore-2026-10-24/AEROZOO_SPECMETRIC.json | c8fb1c822f8cb67afa8ef56d520587f5 |
| az_rescore | TABLES_FULL.md | scratch/isla-az-rescore-2026-10-24/TABLES_FULL.md | 1b71f6c498ac7a6b111c12fdae989e27 |
| az_rescore | ORDERS.md | scratch/isla-az-rescore-2026-10-24/ORDERS.md | a7d0ee73d0033f805737463f434875d5 |
| az_rescore | PARTITION.md | scratch/isla-az-rescore-2026-10-24/PARTITION.md | d25bc4f7836c7b2ddc5881e00dd2eee8 |
| az_rescore | READINGS.md | scratch/isla-az-rescore-2026-10-24/READINGS.md | 52d594623a684dc66e9c684dcabbee7c |
| az_rescore | JOBS.md | scratch/isla-az-rescore-2026-10-24/JOBS.md | 676d29c1a757a38536b5db36a81e8c80 |
| outlet_exp | RESULTS.md | scratch/isla-outlet-exp-2026-10-22/RESULTS.md | 457eba6e05c54bdd5fb88ad1c833f87f |
| outlet_exp | NOTEBOOK_DRAFT_OUTLET.md | scratch/isla-outlet-exp-2026-10-22/NOTEBOOK_DRAFT_OUTLET.md | 7857d3c43c8c03b3eadf8eaeeda71d01 |
| outlet_exp | tables.md | scratch/isla-outlet-exp-2026-10-22/tables.md | 1bdbde00f7a97d52a70b3a9138c06f5b |
| geo_tieorder | FINAL_REPORT.md | scratch/isla-geo-tieorder-2026-10-25/FINAL_REPORT.md | e10e95eb5610555c127b875f5c7d7c66 |
| geo_tieorder | RESULTS.md | scratch/isla-geo-tieorder-2026-10-25/RESULTS.md | e73972e897efcefb287511d5dfe6315a |
| geo_tieorder | NOTEBOOK_DRAFT.md | scratch/isla-geo-tieorder-2026-10-25/NOTEBOOK_DRAFT.md | b2372ba725edcb9e449b89295dacbba9 |
| draw_exact | FINAL_REPORT.md | scratch/isla-draw-exact-2026-10-25/FINAL_REPORT.md | a07009001c0f666602248fb6b4f9d788 |
| draw_exact | DESIGN.md | scratch/isla-draw-exact-2026-10-25/DESIGN.md | 67dace2c118f96f21a49f8a1fceeb917 |
| slice_normal_exp | PREREG.md | scratch/isla-slice-normal-exp-2026-10-24/PREREG.md | 4580615da2b06f2a25bf793b8aba0a30 |
| slice_normal_exp | reading_DRA.md | scratch/isla-slice-normal-exp-2026-10-24/reading_DRA.md | e9badbe9ec671650e6fbedcf45ab1514 |
| slice_normal_exp | reading_DRA.json | scratch/isla-slice-normal-exp-2026-10-24/reading_DRA.json | 3b5ea8dfcce592c427ba00cc679f293c |
| dtt_knobs_r5 | RESULTS.md | scratch/isla-dtt-knobs-2026-10-23/RESULTS.md | 3b462f0df18edf0463d90a5cd9495a5c |
| dtt_knobs_r5 | RESULTS_R5_REVIEW.md | scratch/isla-dtt-knobs-2026-10-23/RESULTS_R5_REVIEW.md | 03323923894f13c36c0af41d44265a97 |
| main_merge | FINAL_REPORT.md | scratch/isla-main-merge-2026-10-24/FINAL_REPORT.md | b984edca2ad8b0c30936be0e225e866b |
| main_merge | IMPACT.md | scratch/isla-main-merge-2026-10-24/IMPACT.md | 741b66a410a99ac78a594ec06a2b912b |
| main_merge | DEFECT_TORUS.md | scratch/isla-main-merge-2026-10-24/DEFECT_TORUS.md | de9f1f468393efe35c788427a00c9042 |
| lib_tree | FINAL_REPORT.md | scratch/lib-tree-2026-10-24/FINAL_REPORT.md | 349e78ded5e3b95f700cb79247bfea8c |
| lib_tree | RESULTS.md | scratch/lib-tree-2026-10-24/RESULTS.md | ebc455c1edf19ada2a77bd6b8bc8e06f |
| af_whole_read | FINAL_REPORT.md | scratch/isla-af-whole-2026-10-23/FINAL_REPORT.md | 114b737caaaf1b3cad50c3cd93062543 |
| af_whole_read | RESULTS.md | scratch/isla-af-whole-2026-10-23/RESULTS.md | c085e7640e1565031d45886bc8d79ecb |
| af_whole_read | read.md | scratch/isla-af-whole-2026-10-23/read.md | 5bd11101d78b39382b13ca4122d56dc7 |
| af_whole_read | FACTS.md | scratch/isla-af-whole-2026-10-23/FACTS.md | 8984de36791f7efdd709dc9d49e76f57 |
| af_whole_read | NOTEBOOK_DRAFT.md | scratch/isla-af-whole-2026-10-23/NOTEBOOK_DRAFT.md | 09bb686aaf4331d04ac8f8348900f345 |
| test_audit | FINAL_REPORT.md | scratch/isla-test-audit-2026-10-24/FINAL_REPORT.md | 8d41b6664272c669c7747072d307f43c |
| test_audit | AUDIT.md | scratch/isla-test-audit-2026-10-24/AUDIT.md | bd76134d1953e0183d4dc9093261e02e |
| test_audit | COVERAGE.md | scratch/isla-test-audit-2026-10-24/COVERAGE.md | 133446aa26804b917f2928a6f01d87d6 |
| test_audit | DETERMINISM.md | scratch/isla-test-audit-2026-10-24/DETERMINISM.md | afbaf878ec109e1d3d44afc1a1737a2a |
| test_audit | TEST_PLAN.md | scratch/isla-test-audit-2026-10-24/TEST_PLAN.md | f275c4931c3e5916d16bebbc452640ec |
| test_build | FINAL_REPORT.md | scratch/isla-test-build-2026-10-25/FINAL_REPORT.md | e6c77eb7ca073ca2cbc9f5948397004e |
| test_build | BUGS.md | scratch/isla-test-build-2026-10-25/BUGS.md | 42546087184e80695acec73c92573812 |
| kernel_bugs | FINAL_REPORT.md | scratch/isla-kernel-bugs-2026-10-25/FINAL_REPORT.md | c467234146d965b4b591dc7c9f823934 |
| af_fix | FINAL_REPORT.md | scratch/isla-af-fix-2026-10-25/FINAL_REPORT.md | 7ba2a1144193b970188c389698661632 |
| af_fix | DESIGN.md | scratch/isla-af-fix-2026-10-25/DESIGN.md | 1ba6290fcdab306657055af5a03c533e |
| af_fix | PRESSURE_PROBE.md | scratch/isla-af-fix-2026-10-25/PRESSURE_PROBE.md | 0377ab20d79fb73307fcc1ca57c76c98 |
| af_fix | PREREG_DRAFT.md | scratch/isla-af-fix-2026-10-25/PREREG_DRAFT.md | 553e4714cb05b9ce45d11989553fc8b3 |
| af_campaign | PREREG.frozen_e0739632.md | scratch/isla-af-campaign-2026-10-25/PREREG.frozen_e0739632.md | e073963259b13c1b8f9ad885e5f73236 |
| pr_2170 | FINAL_REPORT.md | scratch/pr-2170-2026-10-25/FINAL_REPORT.md | f76f0cf34507bb97f060a663487fa050 |
| perf_survey | FINAL_REPORT.md | scratch/isla-perf-survey-2026-10-25/FINAL_REPORT.md | a8d88b8509954a501514c926d0a5e119 |
| perf_survey | OPPORTUNITIES.md | scratch/isla-perf-survey-2026-10-25/OPPORTUNITIES.md | f85d83b754f804ac2feb55c9117be624 |
| perf_survey | infer/RESULTS.md | scratch/isla-perf-survey-2026-10-25/infer/RESULTS.md | e476ea414b2a2fc9caa345af85dab41e |
| floor_drv_read | RESULTS.md | scratch/isla-floor-drv-2026-10-23/RESULTS.md | 539d7c2e6605dc4bd49f5c8b548d7f4d |
| floor_drv_read | NOTEBOOK_DRAFT.md | scratch/isla-floor-drv-2026-10-23/NOTEBOOK_DRAFT.md | ff0c56f1d0a188b1f702f37389c23b9f |
| floor_drv_read | tables/reduce_fd.md | scratch/isla-floor-drv-2026-10-23/tables/reduce_fd.md | eea5db7ce336356a2b6c68dbe47362e1 |
| floor_drv_read | tables/resid_fd.md | scratch/isla-floor-drv-2026-10-23/tables/resid_fd.md | 560eba9810b02404ec681e4ea04b6f0a |
| globe_dr | RESULTS_DR.md | scratch/isla-globe-2026-10-19/RESULTS_DR.md | 869c667061f772fc3a96c27ebfaf30e8 |
| globe_dr | NOTEBOOK_DRAFT_DR.md | scratch/isla-globe-2026-10-19/NOTEBOOK_DRAFT_DR.md | 86248b7c999726c5e2ac164956fa775d |
| globe_dr | scores_dr/globe_dr_scores.json | scratch/isla-globe-2026-10-19/scores_dr/globe_dr_scores.json | 2527dbc87c337ff0dbe4e04da8fbe265 |
| fused_pad | FINAL_REPORT.md | scratch/isla-fused-pad-2026-10-25/FINAL_REPORT.md | 4812137d688504cc25a3379fa6007dd1 |
| fused_pad | RESULTS.md | scratch/isla-fused-pad-2026-10-25/RESULTS.md | 2a8c7a0032c015a9a7a1bc8dcfd37647 |
| int64_offsets | FINAL_REPORT.md | scratch/isla-int64-offsets-2026-10-25/FINAL_REPORT.md | 8bdb261694f967a739a73f5a13bd9481 |
| int64_offsets | AUDIT.md | scratch/isla-int64-offsets-2026-10-25/AUDIT.md | 49cdb18eefab51a7b36cd7fbefa85ed4 |
| lib_sampling | FINAL_REPORT.md | scratch/isla-lib-sampling-2026-10-25/FINAL_REPORT.md | 26fce30261b4538259a625fd663d6f38 |
| lib_sampling | RESULTS.md | scratch/isla-lib-sampling-2026-10-25/RESULTS.md | cb56b8256c575cc60ef17c76e8a65634 |
| lib_sampling | BOOK_LIST.md | scratch/isla-lib-sampling-2026-10-25/BOOK_LIST.md | 0ffd92826743778a13a0d7fe5e00b1b6 |
| premerge | FINAL_REPORT.md | scratch/isla-premerge-2026-10-26/FINAL_REPORT.md | 22e9490f78276322300325eba8e4b2b1 |
| premerge | LEAF_AB.md | scratch/isla-premerge-2026-10-26/LEAF_AB.md | d1fabda1e54d989347f1ec5989a9d7e6 |
| rerun_configs | FINAL_REPORT.md | scratch/isla-final-rerun-2026-10-26/FINAL_REPORT.md | 6eea74cd2265191311ce5a7d6c1921e5 |
| rerun_configs | SUPERVISOR_DECISIONS.md | scratch/isla-final-rerun-2026-10-26/SUPERVISOR_DECISIONS.md | ac7fd5a4719af78c839d464b83d8ef25 |
| rerun_configs | price/PRICE.md | scratch/isla-final-rerun-2026-10-26/price/PRICE.md | adeb2a6b8cc18ff40b3a075ba6f8f716 |
| geo_port | FINAL_REPORT.md | scratch/isla-geo-port-2026-10-26/FINAL_REPORT.md | d4d42d2db60236221441968e9d23029b |
| geo_port | RESULTS.md | scratch/isla-geo-port-2026-10-26/RESULTS.md | ab93c586a31e3630c1b9241f5f135e44 |
| dtt_knobs_a14 | RESULTS.md | scratch/isla-dtt-knobs-2026-10-23/RESULTS.md | 32da0774cb5ac07aa9490b0f97063fce |
| dtt_knobs_a14 | RESULTS_A14_REVIEW.md | scratch/isla-dtt-knobs-2026-10-23/RESULTS_A14_REVIEW.md | 87862fc699cd07b4f033bfc24273ff67 |
| dtt_knobs_a14 | reduce/out_r14/reduce_s2.md | scratch/isla-dtt-knobs-2026-10-23/reduce/out_r14/reduce_s2.md | 3219ad59e0280b7f30a0baef474be0f4 |
| dtt_knobs_a14 | reduce/out_r14/reduce_s2.json | scratch/isla-dtt-knobs-2026-10-23/reduce/out_r14/reduce_s2.json | 914f6c90ee8479b29fe9b4c8e049cd6e |
| dtt_knobs_exact | RESULTS.md | scratch/isla-dtt-knobs-2026-10-23/RESULTS.md | ad1ec340c726ec0891c50640b23c6b3c |
| dtt_knobs_exact | RESULTS_EXACT_REVIEW.md | scratch/isla-dtt-knobs-2026-10-23/RESULTS_EXACT_REVIEW.md | 1621789b02fbb8109eafe3771d528b4a |
| dtt_knobs_exact | reduce/out_final/reduce_s2.md | scratch/isla-dtt-knobs-2026-10-23/reduce/out_final/reduce_s2.md | adfd6f8a4b341c7c2943f2b00e1a252c |
| dtt_knobs_exact | reduce/out_final/reduce_s2.json | scratch/isla-dtt-knobs-2026-10-23/reduce/out_final/reduce_s2.json | e289de96b1de8f69902a1a1bc22cebcd |
| dtt_pose_exp_read | RESULTS.md | scratch/isla-dtt-pose-exp-2026-10-24/RESULTS.md | 6eb97cb89900ac4ac46ad370920ce61f |
| dtt_pose_exp_read | FINAL_REPORT.md | scratch/isla-dtt-pose-exp-2026-10-24/FINAL_REPORT.md | 49b6aa857cb9c0b6b37160797448e36b |
| dtt_pose_exp_read | results/read/read_all.md | scratch/isla-dtt-pose-exp-2026-10-24/results/read/read_all.md | a73345ec2e5ccc2b4aca0a2ccd2e83ca |
| dtt_pose_exp_read | results/read/read_all.json | scratch/isla-dtt-pose-exp-2026-10-24/results/read/read_all.json | 889cac97207be3bc8cdb397c7a70fb21 |
| dtt_knobs_final | RESULTS.md | scratch/isla-dtt-knobs-2026-10-23/RESULTS.md | e47571dad0ff0e4ec8010c3c9543c2e3 |
| dtt_knobs_final | FINAL_REPORT.md | scratch/isla-dtt-knobs-2026-10-23/FINAL_REPORT.md | e5af6dd591762d22eac2f681e32fa60f |
| dtt_knobs_final | STAGE3_PROPOSAL.md | scratch/isla-dtt-knobs-2026-10-23/STAGE3_PROPOSAL.md | 67304a2c8628217eba5139b0abf8e7b1 |
| dtt_knobs_final | NOTEBOOK_DRAFT_STAGE1.md | scratch/isla-dtt-knobs-2026-10-23/NOTEBOOK_DRAFT_STAGE1.md | 293fbd0b12dde4686fadf0f0b30414a6 |
| dtt_knobs_final | NOTEBOOK_DRAFT_STAGE2.md | scratch/isla-dtt-knobs-2026-10-23/NOTEBOOK_DRAFT_STAGE2.md | d1ff8d1a83f44abd4751c27265d69194 |
| slices_all | RESULTS.md | scratch/isla-slices-all-2026-10-24/RESULTS.md | 5ccc668f06b9d245673c1bfc929c9f2e |
| slices_all | FINAL_REPORT.md | scratch/isla-slices-all-2026-10-24/FINAL_REPORT.md | 0a0123d550b96b69bde4e746cbc812a3 |
| slices_all | NOTEBOOK_DRAFT.md | scratch/isla-slices-all-2026-10-24/NOTEBOOK_DRAFT.md | 74d359f113c3cd5b59099da171742b16 |
| slices_all | PREREG.md | scratch/isla-slices-all-2026-10-24/PREREG.md | 982bf4562760b8ad96b0ed05a2c51e84 |
| slices_all | reading_DR.md | scratch/isla-slices-all-2026-10-24/reading_DR.md | be4d50625fabaa1227a65c31ed4e05f9 |
| slices_all | reading_DR.json | scratch/isla-slices-all-2026-10-24/reading_DR.json | 4e002ce77c035002e53771fcd7da42d3 |
| slices_all | reading_AF.md | scratch/isla-slices-all-2026-10-24/reading_AF.md | 5a8d1785d300b105a8b211eec820d8f9 |
| slices_all | reading_AF.json | scratch/isla-slices-all-2026-10-24/reading_AF.json | 35c9fab5b5fdeb1cfe9e86861514c9e4 |
| slices_all | reading_LUG.md | scratch/isla-slices-all-2026-10-24/reading_LUG.md | 1879e2c0061c87d80516c38cdd8e34b5 |
| slices_all | reading_LUG.json | scratch/isla-slices-all-2026-10-24/reading_LUG.json | bfe3e133ebb63ae098fbe7ee6ae7b37d |
| slice_normal_dras | RESULTS.md | scratch/isla-slice-normal-exp-2026-10-24/RESULTS.md | aaa467d4e56a6ffad16cc5fde44e8c90 |
| slice_normal_dras | reading_DRAS.md | scratch/isla-slice-normal-exp-2026-10-24/reading_DRAS.md | 2185f713216572121a436d2efb8053c6 |
| slice_normal_dras | reading_DRAS.json | scratch/isla-slice-normal-exp-2026-10-24/reading_DRAS.json | f298a6a973a609f2dd715cbaadb1f769 |
| slice_normal_dras | reading_MDRA.md | scratch/isla-slice-normal-exp-2026-10-24/reading_MDRA.md | fdedf6ca91dc58a6a4a894fb89366e03 |
| slice_normal_dras | reading_MDRA.json | scratch/isla-slice-normal-exp-2026-10-24/reading_MDRA.json | 78052e0f89d4614f79d5e0d0d27d1db0 |
| rerun_env | FINAL_REPORT.md | scratch/isla-rerun-env-2026-10-26/FINAL_REPORT.md | fa161f7af86c5f66dd24b928a44d22ae |
| rerun_env | HILIFT_213.md | scratch/isla-rerun-env-2026-10-26/HILIFT_213.md | c05321ef974c70e5dfb277c87fbbf2e8 |
| compile_flag | FINDINGS.md | scratch/compile-flag-2026-10-26/FINDINGS.md | ab8c1b707e5e7a6ee9f47fd3c6805d4b |
| golden_213 | FINAL_REPORT.md | scratch/isla-golden-213-2026-10-26/FINAL_REPORT.md | 27728a411f91eb9f79161acf04401217 |
| final_rerun_plan | SUPERVISOR_DECISIONS.md | scratch/isla-final-rerun-2026-10-26/SUPERVISOR_DECISIONS.md | 122e3aab89edba597d3aebd00bd1f88d |
| final_rerun_plan | price/PRICE.md | scratch/isla-final-rerun-2026-10-26/price/PRICE.md | 9685bea96fc48d6d2621d7cec42ac0c8 |
| final_rerun_plan | price/price_canonical.tsv | scratch/isla-final-rerun-2026-10-26/price/price_canonical.tsv | b33cfcf5bfe316c2ae0a2007f904289b |
| az_canonical | FINAL_REPORT.md | scratch/isla-az-canonical-2026-10-26/FINAL_REPORT.md | 84d635987ca88b395bff052850d7f6d0 |
| az_canonical | CONSTANT_CHANGES.md | scratch/isla-az-canonical-2026-10-26/CONSTANT_CHANGES.md | 7c8e70b6980139cc48f941389730f00e |
| assembly_014 | FINAL_REPORT.md | scratch/isla-assembly-014-2026-10-26/FINAL_REPORT.md | 60d7c0e2935740a8062626fc5dc5b645 |
| cert | FINAL_REPORT.md | scratch/isla-cert-2026-10-27/FINAL_REPORT.md | 73b643b186ce1b2b766f266154759250 |
| cert | RESULTS.md | scratch/isla-cert-2026-10-27/RESULTS.md | afaef89592ed8fb0ccabc9339bfc67a5 |
| cert | out/part2/PART2_VERDICTS.md | scratch/isla-cert-2026-10-27/out/part2/PART2_VERDICTS.md | 406e42e7ea5008f8756b469e827700ed |
| cert | out/PRICE_CERT.md | scratch/isla-cert-2026-10-27/out/PRICE_CERT.md | 1277e6250a13c984fb696023abcf76d2 |
| cert_routing | FINAL_REPORT.md | scratch/isla-cert-routing-2026-10-27/FINAL_REPORT.md | 8facd5afd3b1974e72b29d4447f32fac |
| cert_routing | DIAGNOSIS.md | scratch/isla-cert-routing-2026-10-27/DIAGNOSIS.md | f14639ff5503470dd652db3c3fe99631 |
| cert_routing | REFORMULATIONS.md | scratch/isla-cert-routing-2026-10-27/REFORMULATIONS.md | 4cfa48f48d1ddc20648f7210355f024e |
| cert_routing | out/part3/PART3_VERDICTS.md | scratch/isla-cert-routing-2026-10-27/out/part3/PART3_VERDICTS.md | 7ab8b9d6484876c7d3706718e6a7a02e |
| slice_normal_final | FINAL_REPORT.md | scratch/isla-slice-normal-exp-2026-10-24/FINAL_REPORT.md | c651b6c4031d2dc10dfb0af2a41dbbf6 |
| slice_normal_final | RESULTS.md | scratch/isla-slice-normal-exp-2026-10-24/RESULTS.md | 815cb8f4492d8d7a5cfd23e54e111e4d |
| slice_normal_final | C_DESIGN.md | scratch/isla-slice-normal-exp-2026-10-24/C_DESIGN.md | 12f8e4d9fccb6753f30c70e7a5aaedbf |
| af_campaign_read | RESULTS.md | scratch/isla-af-campaign-2026-10-25/RESULTS.md | e024bd8e99ad43407feffd1e65c234d3 |
| af_campaign_read | read.md | scratch/isla-af-campaign-2026-10-25/read.md | 0f2882fc911ef71b54d85379265389d8 |
| af_campaign_read | FINAL_REPORT.md | scratch/isla-af-campaign-2026-10-25/FINAL_REPORT.md | 4ae5f43fc5863acb7601ce1222db85b9 |
| af_campaign_read | REVIEW_SECOND_READ.md | scratch/isla-af-campaign-2026-10-25/REVIEW_SECOND_READ.md | 365845a6e2dfb704874cd18740492ba1 |
| forest_neartie | FINAL_REPORT.md | scratch/isla-forest-neartie-2026-10-27/FINAL_REPORT.md | c18974af33d720ae8faf8bd2b0e324d8 |
| forest_neartie | RISK_TABLE.md | scratch/isla-forest-neartie-2026-10-27/RISK_TABLE.md | 1ad93eabbe6c1281b60cc0ad659cdcf8 |
| forest_neartie | OPTIONS.md | scratch/isla-forest-neartie-2026-10-27/OPTIONS.md | 63ef8b2bebde45e3fde6265778119a20 |
| forest_neartie | POSE_TABLE.md | scratch/isla-forest-neartie-2026-10-27/POSE_TABLE.md | 419741a6d11fa9d6402a2d278cd814ff |
| rule1_root_cause | ROOT_CAUSE_RULE1.md | scratch/isla-af-confirm-2026-10-27/ROOT_CAUSE_RULE1.md | ff610d74af97b4c3affc10e81858340a |
| version_014 | FINAL_REPORT.md | scratch/isla-version-014-2026-10-27/FINAL_REPORT.md | 238b37a3edb181d6d3da5c868befae3e |
| version_014 | PHASE2_NEARTIE.md | scratch/isla-version-014-2026-10-27/PHASE2_NEARTIE.md | e25bb8aade65bd6960d78dfdf88e4275 |
| assembly_014_final | FINAL_REPORT.md | scratch/isla-assembly-014-2026-10-26/FINAL_REPORT.md | 7145d1c94789dbc1240249632cc7eb04 |
| main_merge_1026 | FINAL_REPORT.md | scratch/isla-mainmerge-1026/FINAL_REPORT.md | cdbd7e44f2e15ef9fe14df356ff3a94d |
| main_merge_1027 | FINAL_REPORT.md | scratch/isla-mainmerge-1027/FINAL_REPORT.md | a24170c9897ff08bb280e26e5ba37e78 |
| merge_2171_tidy | FINAL_REPORT.md | scratch/isla-2171tidy-2026-10-26/FINAL_REPORT.md | 8001ee770a932c97cfb2df5a7acf9d10 |
| af_confirm_prereg | PREREG.md | scratch/isla-af-campaign-2026-10-25/PREREG.md | fc03c18112131d8595824df72131095c |
| rerun_azb | FINAL_REPORT.md | scratch/isla-rerun-azb-2026-10-27/FINAL_REPORT.md | 68a484d1779a01d206f8e2b6cbe151cf |
