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
