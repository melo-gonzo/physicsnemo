# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Copy the study records the book's chapters read into results/campaigns_2026-10-23/<topic>/, unmodified, and write
RECORDS.md: one row per file with its source path under the program's scratch folder and its md5.

The records are the lanes' own outputs (reductions, tables, READMEs); nothing here recomputes a number. A chapter reads
them through ``book/records.py``. A committed record is never overwritten: if a destination exists and differs from its
source, the script stops before writing anything and names both md5s. A newer version of a record goes in under a new
topic, and the old topic's file is listed in ``KEPT`` with its committed md5.

usage: python copy_records_2026-10-23.py [scratch root, default ~/gh/physicsnemo/scratch]
"""

import hashlib
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path.home() / "gh" / "physicsnemo" / "scratch"
)

#: topic -> (scratch folder, what it measures, files relative to that folder)
RECORDS = {
    "attrib090": (
        "isla-attrib090-2026-10-21",
        "ATTRIB-090: which change of ISLA 0.9.0 moves Tensile2d, AirfRANS and the lug",
        [
            "RESULTS.md",
            "results_native.json",
            "results2_native.json",
            "results2_posed.json",
            "results_native_eager.json",
            "results_posed_eager.json",
            "relaunch_native.md",
            "relaunch_posed.md",
            "arch_lug.md",
            "arch_tensile2d.md",
            "arch_airfrans.md",
            "t2d_eager_native.md",
            "t2d_eager_posed.md",
        ],
    ),
    "ref090": (
        "isla-ref090-2026-10-20",
        "REF-090: ISLA 0.9.0 against the nearest-token form, the training-path control",
        [
            "r090_final_tables.md",
            "r090_final.json",
            "t2d_control.md",
            "t2d_control.json",
            "t2d_fourseed.json",
            "t2d_fourseed.md",
            "PREREG.md",
            "RESULTS.md",
            "eag_lug.md",
            "eag_lug.json",
            "eag_af_isla.md",
            "eag_af_isla.json",
        ],
    ),
    "af_extrap": (
        "isla-af-extrap-2026-10-21",
        "AF-EXTRAP: AirfRANS extrapolation in Reynolds number and angle of attack",
        [
            "RESULTS.md",
            "POSTHOC.md",
            "afx_tables.md",
            "afx_results.json",
            "posthoc.json",
            "clamp.json",
            "stats/afx_fits.json",
            "stats/clamp_bounds.json",
            "LRE.md",
            "lre.json",
        ],
    ),
    "suv_xfam_b": (
        "isla-suv-xfam-2026-10-22",
        "SUV-XFAM part B: SHIFT-SUV models trained on fastbacks, against the estate-trained ones (stages 1 and 2)",
        [
            "PREREG_B.md",
            "RESULTS_B.md",
            "JOBS.md",
            "reduce_b_out.json",
            "identity_b_out.json",
            "reduce_b2_out.json",
        ],
    ),
    "probes090": (
        "isla-probes090-2026-10-21",
        "PROBES-090: query-population dependence and re-meshing invariance of ISLA 0.9.0",
        [
            "RESULTS.md",
            "results/qpop/reduced.json",
            "results/qpop_afix/reduced.json",
            "results/qpop_afeag/reduced.json",
            "results/qpop_az/reduced.json",
            "results/remesh/reduced_full.json",
            "results/remesh_afix/reduced_full.json",
            "results/remesh_afix_dtt/reduced_full.json",
            "results/remesh_afeag/reduced_full.json",
            "results/afeag_book_numbers.md",
            "remesh/facts_dv.json",
            "remesh/README.md",
            "qpop/qpop_rows.tsv",
        ],
    ),
    "cap_audit": (
        "isla-cap-audit-2026-10-20",
        "CAP-AUDIT: capacity floors of the three architectures, phases 1, 1b and 2",
        [
            "RESULTS.md",
            "HANDOFF.md",
            "tables/*.md",
            "tables/*.json",
            "tables/*.csv",
            "tables/probes/*.json",
        ],
    ),
    "floor_drv": (
        "isla-floor-drv-2026-10-23",
        "FLOOR-DRV: ISLA on one DrivAerML Volume car with the domain's four boundary meshes, the pre-registration",
        ["PREREG.md"],
    ),
    "az_compliant": (
        "isla-az-compliant-2026-10-19",
        "AeroZoo under the benchmark protocol: the registered rule for ISLA against ISLA-DTT (PREREG_v2 sections 12(f) and 13)",
        ["PREREG_v2.md"],
    ),
    "boundary_audit": (
        "isla-boundary-audit-2026-10-23",
        "BOUNDARY-AUDIT: what each dataset stores of its domain's boundary, and the whole boundary's cost on DrivAerML Volume",
        [
            "REPORT.md",
            "BASELINE_PARITY.md",
            "cost_table_isla.md",
            "cost_table_dtt.md",
            "cost_seed_table.md",
        ],
    ),
    "drv_ground": (
        "isla-drv-ground-2026-10-23",
        "DRV-GROUND: the DrivAerML domain meshes' ground offset on all 484 files, and the recipe transform that places it",
        [
            "classify_summary.txt",
            "check_summary.txt",
            "offset_runs.txt",
        ],
    ),
    "setup_research": (
        "isla-setup-research-2026-10-23",
        "SETUP-RESEARCH: the CFD domains of DrivAerML, SHIFT-SUV and HiLiftAeroML, and where their boundary pieces come from",
        ["SETUP.md"],
    ),
    "af_whole": (
        "isla-af-whole-2026-10-23",
        "AF-WHOLE: ISLA and ISLA-DTT reading the AirfRANS far field, the pre-registration and its facts before training",
        ["PREREG.md", "FACTS.md"],
    ),
    "surface_bc": (
        "isla-surface-bc-2026-10-23",
        "SURFACE-BC: a surface task that reads the whole boundary and predicts at the boundaries it names, built and timed",
        ["REPORT.md", "DESIGN.md", "cost_table.md"],
    ),
    "surfqpop": (
        "isla-surf-probe-2026-10-23",
        "SURF-QPOP: how a DrivAerML Surface prediction depends on the other requested cells",
        [
            "RESULTS.md",
            "PREREG.md",
            "results/out/reduced.json",
            "results/out/tables.md",
        ],
    ),
    "slices_af": (
        "isla-slices-2026-10-23",
        "SLICES-AF: ISLA with 128 slices against the 512-slice AirfRANS standing",
        [
            "RESULTS.md",
            "PREREG.md",
            "reading.md",
            "reading.json",
            "nearwall/nearwall_table.md",
        ],
    ),
    "tie_exp": (
        "isla-tie-2026-10-22",
        "TIE-EXP: ISLA with the least-norm tie rules trained on the lug and Tensile2d, read on flip symmetry, native accuracy and pose",
        [
            "RESULTS.md",
            "PREREG.md",
            "JOBS.md",
            "read/results.md",
            "read/results.json",
        ],
    ),
    "fold_sign": (
        "isla-fold-sign-2026-10-22",
        "FOLD-SIGN: the wall distance signed by the generalized winding number, the registered f32floor rerun on AeroZoo",
        ["PREREG.md", "f32floor_tables.md"],
    ),
    "token_draw": (
        "isla-token-draw-2026-10-22",
        "TOKEN-DRAW: the boundary-token draw in float64 in the boundary's canonical frame, measured under precision, pose and renumbering",
        ["DESIGN.md"],
    ),
    "bundle_merge": (
        "isla-bundle-merge-2026-10-24",
        "BUNDLE-MERGE: the geometry robustness bundle merged as ISLA 0.11.0, its version change and tests",
        ["REPORT.md", "FINAL_REPORT.md"],
    ),
    "geo_fast": (
        "isla-geo-fast-2026-10-23",
        "GEO-FAST: the exact geometry's fast path, its bitwise checks and its GB300 timing, and the zero-area-cell defect",
        ["RESULTS.md", "FINAL_REPORT.md", "DESIGN.md", "lhr/final2_table.md"],
    ),
    "align8": (
        "isla-align8-2026-10-23",
        "ALIGN8: the step cost of token counts that are not a multiple of 8 or 32, and the padding fix",
        [
            "FINAL_REPORT.md",
            "results/repro_table.md",
            "exposure/exposure_gpuh.md",
            "exposure/exposure_classes.tsv",
        ],
    ),
    "bc_sidecar": (
        "isla-bc-sidecar-2026-10-23",
        "BC-SIDECAR: the SHIFT-SUV box and the HiLiftAeroML far field and symmetry plane, built and checked",
        ["REPORT.md", "FINAL_REPORT.md"],
    ),
    "geo_degen": (
        "isla-geo-degen-2026-10-24",
        "GEO-DEGEN: cells of zero area read as the segment or point their vertices span, its checks and cost, and the limits left",
        ["FINAL_REPORT.md", "RESULTS.md", "DESIGN.md"],
    ),
    "fast_merge": (
        "isla-fast-merge-2026-10-24",
        "FAST-MERGE: the exact geometry's fast path merged with the zero-area fix, its bitwise checks and its GB300 timing",
        ["REPORT.md", "FINAL_REPORT.md", "lhr/timing_table.md", "lhr/book_table.md"],
    ),
    "dtt_knobs": (
        "isla-dtt-knobs-2026-10-23",
        "DTT-KNOBS: ISLA-DTT's boundary read, its knobs and its accuracy gap (stage 1, inference only) and the stage-2 registration",
        [
            "STAGE1.md",
            "STAGE1_REVIEW.md",
            "PREREG.md",
            "PREREG_REVIEW.md",
            "CODE_FACTS.md",
            "NOTEBOOK_DRAFT_STAGE1.md",
            "NOTEBOOK_REVIEW.md",
            "NOTEBOOK_REVISION_NOTE.md",
            "sens/REPORT.md",
            "sens/sens_table.md",
            "sens/sens_regions.md",
            "cost/REPORT.md",
            "cost/cost_table.md",
            "cap/REPORT.md",
            "cap/cap_table.md",
            "order/REPORT.md",
            "order/order_table.md",
            "gap/REPORT.md",
            "gap/gap_table.md",
            "gap/gap_table_bf16.md",
            "gap/rough.md",
            "treeprec/REPORT.md",
            "treeprec/treeprec_table.md",
            "treeprec/signtest.md",
        ],
    ),
    "dtt_fast_compile": (
        "isla-dtt-fast-2026-10-23/compile",
        "DTT-FAST, piece COMPILE: ISLA-DTT's compiled blocks certified against eager, and the bf16 casts Inductor leaves out",
        ["RESULTS.md", "REPORT.md"],
    ),
    "dtt_fast_verify_fusedread": (
        "isla-dtt-fast-2026-10-23/verify_fusedread",
        "DTT-FAST, piece FUSEDREAD verified independently: pair lists, precision and the GB300 step",
        ["VERIFY.md"],
    ),
    "dtt_fast_integration": (
        "isla-dtt-fast-2026-10-23/integration",
        "DTT-FAST, integration pass 1: the tree walk, tree build and fused read on ISLA 0.11.0 behind one execution attribute",
        ["RESULTS.md"],
    ),
    "boundary_merge": (
        "isla-merge-prep-2026-10-24",
        "MERGE-PREP: the token padding and the whole boundary with the surface task merged into ISLA 0.11.0, checked and timed",
        [
            "REPORT.md",
            "FINAL_REPORT.md",
            "book/BOOK_CHANGES.md",
            "results/TIMING.md",
            "results/book_diff_7213_vs_17c9df5c0.txt",
        ],
    ),
    "dtt_pose": (
        "isla-dtt-pose-2026-10-24",
        "DTT-POSE: why ISLA-DTT's forest depends on pose and precision, and the design that removes it",
        ["DESIGN.md", "MECHANISMS.md"],
    ),
    "dtt_pose_exp": (
        "isla-dtt-pose-exp-2026-10-24",
        "DTT-POSE-EXP: ISLA-DTT retrained with DTT-POSE's forest on the lug and AirfRANS, the pre-registration",
        ["PREREG.md"],
    ),
    "geo_thin": (
        "isla-geo-thin-2026-10-24",
        "GEO-THIN: thin cells of nonzero area in the exact geometry, the census of the stored meshes and the design",
        ["DESIGN.md", "RESULTS.md"],
    ),
    "dtt_fast_final": (
        "isla-dtt-fast-2026-10-23/final",
        "DTT-FAST, pass 2: ISLA-DTT's training step before and after on GB300, against interacting ISLA, and its surface form",
        ["TABLE.md"],
    ),
    "nearwall": (
        "isla-nearwall-2026-10-24",
        "NEARWALL: ISLA's near-wall distance channel under the whole boundary, its exposure, response and precision, and the design",
        ["DESIGN.md", "RESULTS.md", "REVIEW_DESIGN.md"],
    ),
    "hl_dup": (
        "isla-hl-dup-2026-10-24",
        "HL-DUP: HiLiftAeroML's duplicated wing cells, their census and effect on the standing (stage 1, revised after its second read) and the stage-2 registration",
        [
            "PREREG.md",
            "RESULTS.md",
            "NOTEBOOK_DRAFT.md",
            "REVIEW_STAGE1.md",
            "REVISION_NOTE.md",
            "HANDOFF_HLDUP.md",
            "FINAL_REPORT.md",
        ],
    ),
    "geo_thin_final": (
        "isla-geo-thin-2026-10-24",
        "GEO-THIN, final: the thin-cell fix as merged, its amended design (what is derived and what is estimated), checks, cost and limits",
        ["FINAL_REPORT.md", "DESIGN.md", "RESULTS.md"],
    ),
    "nearwall_final": (
        "isla-nearwall-2026-10-24",
        "NEARWALL, final: the comment fix as merged, the contingency drafted for AF-WHOLE and the geometry note",
        ["FINAL_REPORT.md", "PREREG_CONTINGENCY.md", "GEOMETRY_NOTE.md"],
    ),
    "dtt_fast_bf16diag": (
        "isla-dtt-fast-2026-10-23/bf16diag",
        "DTT-FAST, BF16-DIAG: where the chunked step's bf16 excess on AirfRANS comes from, its remedy, and the one-ulp yardstick",
        ["REPORT.md", "RESULTS.md"],
    ),
    "az_decide_final": (
        "isla-az-decide-2026-10-21",
        "AZ-DECIDE, final: AeroZoo's compliant standings of four architectures, the registered read of ISLA against ISLA-DTT, and its context",
        [
            "RESULTS.md",
            "tables_final.md",
            "percase_final.md",
            "nonbox_scales_final.md",
            "QUERYPOP090_FOR_12F.md",
            "CROSS_DATASET_INPUTS.md",
            "reynolds/RESULTS_reynolds.md",
        ],
    ),
    "e1_az": (
        "isla-headnorm-2026-10-18",
        "E1-AZ: the head normalization against ISLA 0.9.0 on compliant AeroZoo's nonbox split, the registered verdict",
        ["RESULTS_AZ_2026-10-22.md"],
    ),
    "lre_az": (
        "isla-lre-az-2026-10-23",
        "LRE-AZ: ISLA on AeroZoo with the fixed Reynolds scale against its fitted constants, registered and read",
        ["RESULTS.md", "tables.md", "lreaz.json", "posthoc.md", "PREREG.md"],
    ),
    "geo_band_phase0": (
        "isla-geo-band-2026-10-24",
        "GEO-BAND, phase 0: the whole-boundary geometry call re-timed on GB300 at the merged head",
        ["PHASE0.md", "lhr/timing_table.md"],
    ),
    "kernel_tests": (
        "isla-kernel-tests-2026-10-24",
        "KERNEL-TESTS: the five CUDA tests that failed after the token padding, their cause and the fix",
        ["FINAL_REPORT.md", "RESULTS.md"],
    ),
    "slices_drv": (
        "isla-slices-drv-2026-10-24",
        "SLICES-DRV: ISLA with 128 slices against the 512-slice DrivAerML Volume standing",
        ["RESULTS.md", "reading.md", "PREREG.md", "farfield/farfield_table.md"],
    ),
    "freeze_plan": (
        "isla-freeze-plan-2026-10-24",
        "FREEZE-PLAN: the final reference rerun re-planned with ISLA-DTT as the default interior form, its lanes, cost and calendar",
        ["FINAL_RERUN_PLAN.md"],
    ),
    "bc_supply": (
        "isla-bc-supply-2026-10-24",
        "BC-SUPPLY: SHIFT-SUV's box and HiLiftAeroML's inlet and outlet built in the recipe, the deduplicated wing, their checks and cost",
        [
            "REPORT.md",
            "FINAL_REPORT.md",
            "book/BOOK_CHANGES.md",
            "out/hl_val_368914.jsonl",
            "out/move_369235.jsonl",
            "gb/smoke_summary.txt",
        ],
    ),
    "geo_band_final": (
        "isla-geo-band-2026-10-24",
        "GEO-BAND: the exact geometry's derived tie and on-wall bands in the boundary's centred frame (ISLA 0.12.0), their derivation, its independent re-derivation, the pose and native effects, the trained-model check and the cost",
        [
            "FINAL_REPORT.md",
            "RESULTS.md",
            "DERIVATION.md",
            "INFER_CHECK.md",
            "REVIEW_DERIVATION.md",
            "DESIGN.md",
        ],
    ),
    "band_merge": (
        "isla-band-merge-2026-10-24",
        "BAND-MERGE: the ISLA 0.12.0 version commit for GEO-BAND and its test review",
        [
            "FINAL_REPORT.md",
            "REVIEW_TESTS.md",
        ],
    ),
    "default_dtt": (
        "isla-default-dtt-2026-10-24",
        "DEFAULT-DTT: ISLA-DTT the default interior form, its tests, smokes and query-independence check",
        [
            "FINAL_REPORT.md",
            "BOOK_CHANGES.md",
            "BOOK_SENTENCES_v2.md",
        ],
    ),
    "dtt_default_int": (
        "isla-dtt-default-int-2026-10-24",
        "DTT-DEFAULT-INT: the default switch integrated with ISLA 0.12.0, and its fixture test reworked",
        [
            "FINAL_REPORT.md",
        ],
    ),
    "geo_thinfast": (
        "isla-geo-thinfast-2026-10-24",
        "GEO-THINFAST: the thin-cell guard and the accurate closest point fused into the fast path, its equality and timing",
        [
            "FINAL_REPORT.md",
            "RESULTS.md",
            "REVIEW.md",
            "lhr/timing_table.md",
        ],
    ),
    "dtt_fast_done": (
        "isla-dtt-fast-2026-10-23",
        "DTT-FAST, done: ISLA-DTT's exact speedups as merged, their equivalence, the step on the final heads, the pose port, "
        "the compiled bf16 gradient spike and the bf16 reduction flag",
        [
            "FINAL_REPORT.md",
            "integration/RESULTS.md",
            "final2/TABLE.md",
            "pose/RESULTS.md",
            "spike/RESULTS.md",
            "flag/RESULTS.md",
            "flag/REPORT.md",
        ],
    ),
    "hl_dup_stage2": (
        "isla-hl-dup-2026-10-24",
        "HL-DUP, stage 2: ISLA and both unit-drive baselines retrained and scored on the deduplicated HiLift wing, its review, and the failed root-outline rebuild",
        [
            "RESULTS_STAGE2.md",
            "stage2_table.md",
            "REVIEW_STAGE2.md",
            "REVISION_NOTE_STAGE2.md",
            "HOLE_CRUST.md",
            "FINAL_REPORT.md",
        ],
    ),
    "dtt_fast_final_table_p5": (
        "isla-dtt-fast-2026-10-23/final",
        "DTT-FAST, pass 2's table with its pass-3 columns added, as it now stands",
        ["TABLE.md"],
    ),
    "hl_dup_stage2_scores": (
        "isla-hl-dup-2026-10-24",
        "HL-DUP, stage 2: the scorer's per-case rows of the six deduplicated-wing runs (native and posed), and the stage-1 "
        "scores of the stored-wing checkpoints that its rule and its like-for-like figures read",
        [
            "scores_stage2/*/*/*/metrics.jsonl",
            "scores_stage2/*/*/DATASET_USED",
            "score_summary.json",
            "score_table.md",
            "stage2_summary.json",
        ],
    ),
    "hl_symplane": (
        "isla-hl-symplane-2026-10-24",
        "HL-SYMPLANE: HiLiftAeroML's symmetry plane built at load from the source STL's root cap, its per-case check and "
        "tolerance, the census of the 690 canonical cases, the comparison with the curation pipeline's plane, and the book changes",
        ["FINAL_REPORT.md", "RESULTS.md", "BOOK_CHANGES.md"],
    ),
    "hl_dup_stage2_area": (
        "isla-hl-area-2026-10-24",
        "HL-AREA: HL-DUP stage 2's six deduplicated-wing checkpoints scored on the uniform point cloud (per-case rows, reduction and "
        "table), and the six lanes' training cost from their own logs (extract_cost.py's lines and the cost report)",
        [
            "RESULTS.md",
            "FINAL_REPORT.md",
            "NOTEBOOK_DRAFT.md",
            "area_table.md",
            "area_summary.json",
            "COST_REPORT.md",
            "cost/cost_hldupx.json",
            "cost/extract.jsonl",
            "scores_area/*/area/*/metrics.jsonl",
            "scores_area/*/area/DATASET_USED",
        ],
    ),
    "isla_api_013": (
        "isla-api-013-2026-10-24",
        "ISLA-API: ISLA 0.13.0 (the interior task is ISLA-DTT with no switch; Mesh's names n_spatial_dims and "
        "n_boundary_tokens; no Reynolds argument; the schema key defines), its design and mapping table, the deleted tests, "
        "the assertion audit, the suites and the datapipe check of re_z against the in-model z, and the book changes",
        [
            "FINAL_REPORT.md",
            "DESIGN.md",
            "BOOK_CHANGES.md",
            "BOOK_PROSE.md",
            "DELETED_TESTS.md",
            "DELETED_TESTS_RE.md",
            "ASSERTION_AUDIT.md",
            "JOBS.md",
            "logs/lhr/gb_377645.log",
        ],
    ),
    "reynolds_out": (
        "isla-reynolds-out-2026-10-24",
        "REYNOLDS-OUT: the Reynolds feature moved out of ISLA's constructor into the recipe (ComputeBoundaryScales and float64 "
        "ComputeLogReynolds in ISLA's model configs), its design and evidence, and the bitwise check of z on 160 cases over "
        "three checkpoints",
        [
            "DESIGN.md",
            "FINAL_REPORT.md",
            "bitwise/RESULTS_BITWISE.md",
            "bitwise/JOBS.md",
            "bitwise/z_cases_376561.tsv",
        ],
    ),
    "az_spec_audit": (
        "isla-az-spec-audit-2026-10-24",
        "AZ-SPEC-AUDIT: the AeroZoo runs against AeroZoo's task specification (inputs, statistics, splits, targets, metric, "
        "loss and reporting), with the deviations M1, T1, I1, I2, L1, S1, S1b, R1 and the hardening notes",
        ["AUDIT.md"],
    ),
    "az_rescore": (
        "isla-az-rescore-2026-10-24",
        "AZ-RESCORE: the 48 AeroZoo checkpoints rescored with AeroZoo's own interior metrics (eps_U, eps_p, eps^N; mean and "
        "median over all test cases and the mean_stable ones; full-mesh float64, chunked inference), with the zero and "
        "free-stream predictors, the orders, the partition probes and the registered readings beside",
        [
            "RESULTS.md",
            "NOTEBOOK_DRAFT.md",
            "AEROZOO_SPECMETRIC.json",
            "TABLES_FULL.md",
            "ORDERS.md",
            "PARTITION.md",
            "READINGS.md",
            "JOBS.md",
        ],
    ),
    "outlet_exp": (
        "isla-outlet-exp-2026-10-22",
        "OUTLET-EXP: ISLA trained with AeroZoo's outlet gauge pressure held by construction, read on profile_crossflow and "
        "full by its registered rule, its cost and the supervisor's decision not to adopt it",
        ["RESULTS.md", "NOTEBOOK_DRAFT_OUTLET.md", "tables.md"],
    ),
    "geo_tieorder": (
        "isla-geo-tieorder-2026-10-25",
        "GEO-TIEORDER: the on-wall direction's dependence on cell order and precision at near-equal normals, its two causes, "
        "the rule that removes it (a candidate for ISLA 0.14.0) and its measured function change and cost",
        ["FINAL_REPORT.md", "RESULTS.md", "NOTEBOOK_DRAFT.md"],
    ),
    "draw_exact": (
        "isla-draw-exact-2026-10-25",
        "DRAW-EXACT: the token draw's run-to-run and CPU-to-GPU dependence at exact ties, the duplicate and dropped cells of "
        "today's GPU evaluation draw, and the exact int64 line that removes them (a candidate for ISLA 0.14.0)",
        ["FINAL_REPORT.md", "DESIGN.md"],
    ),
    "slice_normal_exp": (
        "isla-slice-normal-exp-2026-10-24",
        "SLICE-NORMAL-EXP: the no-harm retrain check of the slice mean normal, its registration and the DrivAerML Surface "
        "read of option (a), the unnormalized mean normal",
        ["PREREG.md", "reading_DRA.md", "reading_DRA.json"],
    ),
    "dtt_knobs_r5": (
        "isla-dtt-knobs-2026-10-23",
        "DTT-KNOBS stage 2, R5: ISLA-DTT without its physical scales on a standardized inlet speed against the AirfRANS "
        "standing's pair, and its second read",
        ["RESULTS.md", "RESULTS_R5_REVIEW.md"],
    ),
    "main_merge": (
        "isla-main-merge-2026-10-24",
        "MAIN-MERGE: upstream main merged into psharpe/isla, its effect on ISLA, its recipe and its book, and the on-wall "
        "direction's defect at quad diagonals that it found",
        ["FINAL_REPORT.md", "IMPACT.md", "DEFECT_TORUS.md"],
    ),
    "lib_tree": (
        "lib-tree-2026-10-24",
        "LIB-TREE: the cluster tree's Warp build and walk and the grouped forest in physicsnemo.mesh.spatial, a library-only "
        "branch, with its bitwise proofs against ISLA and GLOBE and its timings",
        ["FINAL_REPORT.md", "RESULTS.md"],
    ),
    "test_audit": (
        "isla-test-audit-2026-10-24",
        "TEST-AUDIT: ISLA's test suite audited (coverage on CPU and GB300, MOD-008, determinism on the GPU) and the test plan",
        [
            "FINAL_REPORT.md",
            "AUDIT.md",
            "COVERAGE.md",
            "DETERMINISM.md",
            "TEST_PLAN.md",
        ],
    ),
}


#: (topic, file) -> md5 of the committed copy, for records whose source has moved on since it was copied
KEPT = {
    ("dtt_fast_integration", "RESULTS.md"): "cd508ec6241c605c2464c0f4541ef70b",
    ("geo_thin", "DESIGN.md"): "f7f8ed6849f68cc8a4584989afb1a27e",
    ("geo_thin", "RESULTS.md"): "6daa28650f27ed75c455c12e89fbd25d",
    ("dtt_fast_final", "TABLE.md"): "a6f352dfc32a342dc1f95a90bba69c17",
    ("dtt_knobs", "PREREG.md"): "e601d4af9ff793a7570af7acd9929f43",
    ("hl_dup", "FINAL_REPORT.md"): "74bc30896f511fcd9513fb01eb935e9f",
}


def md5(p: Path) -> str:
    """The file's md5, as RECORDS.md lists it."""
    return hashlib.md5(p.read_bytes()).hexdigest()


copies = []
for topic, (folder, what, files) in RECORDS.items():
    src = SCRATCH / folder
    for pattern in files:
        for s in sorted(src.glob(pattern)):
            rel = s.relative_to(src)
            d = HERE / topic / rel
            kept = KEPT.get((topic, str(rel)))
            if kept is not None:
                if md5(d) != kept:
                    sys.exit(f"{d}: md5 {md5(d)}, but KEPT lists {kept}")
            elif d.exists() and md5(d) != md5(s):
                sys.exit(
                    f"{d} exists with md5 {md5(d)}, and its source {s} now has md5 {md5(s)}: "
                    "copy the newer version under a new topic instead of overwriting the record"
                )
            copies.append((topic, folder, rel, s, d))

topics = ["| topic | what it measures | source folder |", "| --- | --- | --- |"] + [
    f"| `{topic}/` | {what} | `scratch/{folder}/` |"
    for topic, (folder, what, _) in RECORDS.items()
]
rows = ["| topic | file | source | md5 |", "| --- | --- | --- | --- |"]
for topic, folder, rel, s, d in copies:
    if not d.exists():
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
    rows.append(f"| {topic} | {rel} | scratch/{folder}/{rel} | {md5(d)} |")
head = [
    "# Study records read by the book",
    "",
    f"Written by `{Path(__file__).name}`; every file is an unmodified copy of its",
    "source. Do not edit by hand.",
    "",
]
(HERE / "RECORDS.md").write_text("\n".join(head + topics + [""] + rows) + "\n")
print(f"{len(rows) - 2} files indexed in RECORDS.md")
