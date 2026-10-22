# Study records read by the book

Written by `copy_records_2026-10-22.py`; every file is an unmodified copy of its
source. Do not edit by hand.

| topic | what it measures | source folder |
| --- | --- | --- |
| `probes090/` | PROBES-090: query-population dependence and re-meshing invariance of ISLA 0.9.0 | `scratch/isla-probes090-2026-10-21/` |
| `attrib090/` | ATTRIB-090: which change of ISLA 0.9.0 moves Tensile2d, AirfRANS and the lug | `scratch/isla-attrib090-2026-10-21/` |
| `af_extrap/` | AF-EXTRAP: AirfRANS extrapolation in Reynolds number and angle of attack | `scratch/isla-af-extrap-2026-10-21/` |
| `cap_audit/` | CAP-AUDIT: capacity floors of the three architectures, phases 1 and 1b | `scratch/isla-cap-audit-2026-10-20/` |
| `stream_audit/` | STREAM-AUDIT: residual-stream growth and the head frame of ISLA 0.9.0 | `scratch/isla-stream-audit-2026-10-21/` |
| `az_diag/` | AeroZoo diagnostics: baselines at the zero predictor, ISLA's held-out failures, float32 floor | `scratch/isla-az-decide-2026-10-21/` |
| `medial_tie/` | MEDIAL-TIE: the design of a tie rule for closest-feature ties | `scratch/isla-medial-tie-2026-10-22/` |
| `ref090/` | REF-090: ISLA 0.9.0 against the nearest-token form, the training-path control | `scratch/isla-ref090-2026-10-20/` |
| `hilift090/` | R-HL090: ISLA 0.9.0 on HiLiftAeroML | `scratch/isla-hilift090-2026-10-20/` |

| topic | file | source | md5 |
| --- | --- | --- | --- |
| probes090 | RESULTS.md | scratch/isla-probes090-2026-10-21/RESULTS.md | 66965974ddc6e4118a4cc784abf500f7 |
| probes090 | results/qpop/reduced.json | scratch/isla-probes090-2026-10-21/results/qpop/reduced.json | bfd1167be24751f0094d44dd1de213c6 |
| probes090 | results/qpop_afix/reduced.json | scratch/isla-probes090-2026-10-21/results/qpop_afix/reduced.json | 82913383c1bd164d586a383c19919c9a |
| probes090 | results/qpop_az/reduced.json | scratch/isla-probes090-2026-10-21/results/qpop_az/reduced.json | bc85fd58c4cb7bf84846647fede4fec0 |
| probes090 | results/remesh/reduced_full.json | scratch/isla-probes090-2026-10-21/results/remesh/reduced_full.json | bf7d8b98527cead8577b20cfe52cb596 |
| probes090 | results/remesh_afix/reduced_full.json | scratch/isla-probes090-2026-10-21/results/remesh_afix/reduced_full.json | 1726c5c9dd8a920cd58e2ef7f7960f18 |
| probes090 | results/remesh_afix_dtt/reduced_full.json | scratch/isla-probes090-2026-10-21/results/remesh_afix_dtt/reduced_full.json | a5176cfa9bbe4398cb7f5c73535830cc |
| probes090 | remesh/facts_dv.json | scratch/isla-probes090-2026-10-21/remesh/facts_dv.json | 2d6eeb21e7deac33aaa87e54fab703ec |
| probes090 | remesh/README.md | scratch/isla-probes090-2026-10-21/remesh/README.md | 9ad17033baca19c8d90f313172f764d6 |
| probes090 | qpop/qpop_rows.tsv | scratch/isla-probes090-2026-10-21/qpop/qpop_rows.tsv | 23a0b83bf5fe89ae07221ffa57f7621d |
| attrib090 | RESULTS.md | scratch/isla-attrib090-2026-10-21/RESULTS.md | 72860814756660f927095afff87b9d76 |
| attrib090 | results_native.json | scratch/isla-attrib090-2026-10-21/results_native.json | 3f92835ce703e13a5f72641fa6cf241e |
| attrib090 | results2_native.json | scratch/isla-attrib090-2026-10-21/results2_native.json | a27e92b15407a9c6c99ebf101a2f99a6 |
| attrib090 | results2_posed.json | scratch/isla-attrib090-2026-10-21/results2_posed.json | c2f539f0922a40a9684dd3a4ac6e7391 |
| attrib090 | results_native_eager.json | scratch/isla-attrib090-2026-10-21/results_native_eager.json | 80be6f1a18d5b622ec9ee5c56a6c5e1a |
| attrib090 | results_posed_eager.json | scratch/isla-attrib090-2026-10-21/results_posed_eager.json | 4d5713853c3c7e758c18586f61dd06ff |
| attrib090 | relaunch_native.md | scratch/isla-attrib090-2026-10-21/relaunch_native.md | 4fcd73c32b2cc50bfa1938cab6d40ad2 |
| attrib090 | relaunch_posed.md | scratch/isla-attrib090-2026-10-21/relaunch_posed.md | c8313f3d157971b2651466b6d5c02405 |
| attrib090 | arch_lug.md | scratch/isla-attrib090-2026-10-21/arch_lug.md | ec7191a56c0dace45f1d7921252c4a95 |
| attrib090 | arch_tensile2d.md | scratch/isla-attrib090-2026-10-21/arch_tensile2d.md | cac94bc8495b8073dd30098b83d0a3bf |
| attrib090 | arch_airfrans.md | scratch/isla-attrib090-2026-10-21/arch_airfrans.md | abbf8c160dafe0b13cd5890f7026f73f |
| attrib090 | t2d_eager_native.md | scratch/isla-attrib090-2026-10-21/t2d_eager_native.md | 72138ebbc50aebbeafa8446aaebc7aed |
| attrib090 | t2d_eager_posed.md | scratch/isla-attrib090-2026-10-21/t2d_eager_posed.md | 31e3214261471c9757d03f5a5e430c95 |
| af_extrap | RESULTS.md | scratch/isla-af-extrap-2026-10-21/RESULTS.md | dc0730dedd3fdf90ee438b303406334e |
| af_extrap | POSTHOC.md | scratch/isla-af-extrap-2026-10-21/POSTHOC.md | a5bf31f68d091a5c43c2169fbe732b4d |
| af_extrap | afx_tables.md | scratch/isla-af-extrap-2026-10-21/afx_tables.md | 2387eaa1930aa830492de937e28d7cec |
| af_extrap | afx_results.json | scratch/isla-af-extrap-2026-10-21/afx_results.json | 60dfd526fb60ee7563a60a710f5fa8cf |
| af_extrap | posthoc.json | scratch/isla-af-extrap-2026-10-21/posthoc.json | 11b97a9357bbffdee5b2284d705b0fc0 |
| af_extrap | clamp.json | scratch/isla-af-extrap-2026-10-21/clamp.json | 0e805bcf85fe32e377a40dc7953a2cd5 |
| af_extrap | stats/afx_fits.json | scratch/isla-af-extrap-2026-10-21/stats/afx_fits.json | bc9aa7e5ceb343f7bb4d1e6027e2811c |
| af_extrap | stats/clamp_bounds.json | scratch/isla-af-extrap-2026-10-21/stats/clamp_bounds.json | 5fc5005d16d7ec38475f488382e656f6 |
| cap_audit | RESULTS.md | scratch/isla-cap-audit-2026-10-20/RESULTS.md | 9a4abb09d49f012dd133bd727213a73e |
| cap_audit | tables/phase1_floors.json | scratch/isla-cap-audit-2026-10-20/tables/phase1_floors.json | 9b24f3349d96ac842000f1d2ccac5781 |
| cap_audit | tables/phase1_floors.md | scratch/isla-cap-audit-2026-10-20/tables/phase1_floors.md | 10be6ce3275e378395da307d9fa7046c |
| cap_audit | tables/phase1b_floors.json | scratch/isla-cap-audit-2026-10-20/tables/phase1b_floors.json | 3ee6eda13263dd1778d5d96f33d8c913 |
| cap_audit | tables/phase1b_floors.md | scratch/isla-cap-audit-2026-10-20/tables/phase1b_floors.md | 9e1484de2d0e78a5d856589866ee4606 |
| cap_audit | tables/resid_af_ladder.md | scratch/isla-cap-audit-2026-10-20/tables/resid_af_ladder.md | 01c30a3be9056512c56ff28d875baed3 |
| cap_audit | tables/resid_af_rd.md | scratch/isla-cap-audit-2026-10-20/tables/resid_af_rd.md | 0a3a0da34d76decfb42405463027a1f3 |
| cap_audit | tables/resid_deficit_cases.md | scratch/isla-cap-audit-2026-10-20/tables/resid_deficit_cases.md | b44c24507943ea2a0b18a229b719e721 |
| cap_audit | tables/resid_drv_p1.md | scratch/isla-cap-audit-2026-10-20/tables/resid_drv_p1.md | 63d6c434172f6e190a089b778b4dd54d |
| cap_audit | tables/resid_drv_partB.md | scratch/isla-cap-audit-2026-10-20/tables/resid_drv_partB.md | eb9cf5a6e519bf7e9de58e9950aa1f5d |
| cap_audit | tables/residuals_p1_p1b.csv | scratch/isla-cap-audit-2026-10-20/tables/residuals_p1_p1b.csv | c634b0612745169fa4f3b8d7755278cb |
| cap_audit | tables/residuals_p2.csv | scratch/isla-cap-audit-2026-10-20/tables/residuals_p2.csv | 68cbee1e3771c480d81e2442bf67d5fe |
| stream_audit | REPORT.md | scratch/isla-stream-audit-2026-10-21/REPORT.md | bcc1ef3a385df918960cac07b57c3ab3 |
| stream_audit | tables/summary.json | scratch/isla-stream-audit-2026-10-21/tables/summary.json | 1a456f7b37c81e4c6085c14854c4e22e |
| stream_audit | tables/growth_table.tsv | scratch/isla-stream-audit-2026-10-21/tables/growth_table.tsv | 97a1aab90f94a98a605d527e6d25bb3d |
| stream_audit | tables/hframe_jb.tsv | scratch/isla-stream-audit-2026-10-21/tables/hframe_jb.tsv | 4faef5a054ddf294db61f8ad2a598390 |
| stream_audit | tables/depth_profile.tsv | scratch/isla-stream-audit-2026-10-21/tables/depth_profile.tsv | e4f28a0c104ee1f7b6da1151f2874ecc |
| stream_audit | tables/harm_table.tsv | scratch/isla-stream-audit-2026-10-21/tables/harm_table.tsv | 616af74cc6af4a317c508d35d15eb16a |
| az_diag | basecheck/README.md | scratch/isla-az-decide-2026-10-21/basecheck/README.md | fbcf2f1f3b2dd2627f77024a581a5339 |
| az_diag | pdiag/README.md | scratch/isla-az-decide-2026-10-21/pdiag/README.md | 64626767139b52be87f3b16dbb5b5d4f |
| az_diag | pdiag/pdiag.tsv | scratch/isla-az-decide-2026-10-21/pdiag/pdiag.tsv | 80f4da0ac15ed7637b5101b9e07b7fa2 |
| az_diag | cmpdiag/README.md | scratch/isla-az-decide-2026-10-21/cmpdiag/README.md | 8b56bc65f98dbe39dbe83a2783fee677 |
| az_diag | gtopt/README.md | scratch/isla-az-decide-2026-10-21/gtopt/README.md | 6693bc8d82b829673b9cbe1c2c31d87d |
| az_diag | gtopt/RESULTS_table.md | scratch/isla-az-decide-2026-10-21/gtopt/RESULTS_table.md | 9cf223f17151c17923c51f1287762e38 |
| az_diag | gtopt/raw/summary.json | scratch/isla-az-decide-2026-10-21/gtopt/raw/summary.json | 502ab716e5329b19b016fd03c9d698dd |
| az_diag | f32floor/README.md | scratch/isla-az-decide-2026-10-21/f32floor/README.md | d588167a0ac19bf97f3c7f91fa722f67 |
| az_diag | f32floor/tables.md | scratch/isla-az-decide-2026-10-21/f32floor/tables.md | 4d28de39d90c3be1df544549e1443c75 |
| az_diag | preddiag/README.md | scratch/isla-az-decide-2026-10-21/preddiag/README.md | 6da99c6dec06b08a35ff7a84b8cdda14 |
| az_diag | scales/README.md | scratch/isla-az-decide-2026-10-21/scales/README.md | 2468c30379e14326dac730f6113f3cc3 |
| az_diag | scales/scales_test_all.tsv | scratch/isla-az-decide-2026-10-21/scales/scales_test_all.tsv | 2559911c32562bebadc38ab3cac1ba7e |
| az_diag | scales/scales_train_full.tsv | scratch/isla-az-decide-2026-10-21/scales/scales_train_full.tsv | ef9c59c8aa5f63a13f9c76a2392a48d4 |
| az_diag | worst/cases.tsv | scratch/isla-az-decide-2026-10-21/worst/cases.tsv | 62420953e4af949504e7da343643ca08 |
| medial_tie | DESIGN.md | scratch/isla-medial-tie-2026-10-22/DESIGN.md | 935d06ab953e9d33d6d3d5b7c0c0648d |
| ref090 | r090_final_tables.md | scratch/isla-ref090-2026-10-20/r090_final_tables.md | 80a9b86a10e67a096d45f605c3265aa2 |
| ref090 | r090_final.json | scratch/isla-ref090-2026-10-20/r090_final.json | 0ed88eb280c664338eacb82e2e032a49 |
| ref090 | t2d_control.md | scratch/isla-ref090-2026-10-20/t2d_control.md | fcc11cc3415b316d554182c84903b746 |
| ref090 | t2d_control.json | scratch/isla-ref090-2026-10-20/t2d_control.json | 37335a48f8d838dc94452fb88931de25 |
| ref090 | t2d_fourseed.json | scratch/isla-ref090-2026-10-20/t2d_fourseed.json | 0c67dc5548a683fd122a62c9aed0a811 |
| ref090 | t2d_fourseed.md | scratch/isla-ref090-2026-10-20/t2d_fourseed.md | f443a07295a57fd1b4548017f5f1241d |
| ref090 | PREREG.md | scratch/isla-ref090-2026-10-20/PREREG.md | 6b746af4ba4e1538dbc9772bcd58fa2a |
| hilift090 | RESULTS_HL090.md | scratch/isla-hilift090-2026-10-20/RESULTS_HL090.md | 82f92ccc40060abcf37c49f79d286ac5 |
