# AZ-RESCORE jobs (LHR; $T = /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0)

Scorer: `$T/kernel/azrescore/specscore.py` (local copy `lhr/specscore.py`), reference functions
`reference_metrics.py.txt` (md5 b08f1f4a, = the 06-splits.qmd listing), stability labels `release-v1.2.json` (md5 72b9868a).
Outputs: `$T/azrescore_out/<phase>/<run_id>/<side>/part-*.jsonl`, references in `_references/`.
Registry row `^isla\.azrescore\.` added 2026-10-24 21:58 EDT (expected_end 2026-10-26).

| job | name | what | submitted (EDT) | state |
|---|---|---|---|---|
| 376168 | isla.azrescore.smoke | smoke: 6 full.test cases per family (GT/T, ISLA/DTT), repro of registered relL2, dumps, pseed-1 probe on 2 cases, DTT 1M chunk, E1-AZ / LRE-AZ start-up, independent check | 2026-10-24 22:07 | submitted |
| 376237 | isla.azrescore.smoke2 | smoke 2: baselines on full.test after the out_dim fix (repro, dumps), GT blobs and ISLA/DTT blobs pricing, DTT val side at 1M chunk, independent check | 2026-10-24 22:12 | submitted |

Notes: smoke 1 GPU0 (baselines) failed at model build (`out_dim` set only on the first lane's cfg); fixed in specscore.py md5 469fb568.
TODO (supervisor 22:12): after priority (i), write BASELINES_FOR_AEROZOO.md (no ISLA numbers) and send its md5 via contact_supervisor.
| 376779 | isla.azrescore.smoke3 | smoke 3: claims/plan path (2 base processes sharing claims, limit 3 per split), per-case seeded face draws, parallel near field vs the verbatim one, ISLA-DTT 2M chunk, E1-AZ via plan | 2026-10-24 23:05 | submitted |

Supervisor 23:02 EDT: cap 220 GPU-h; order (i) baselines, (iv) E1-AZ + LRE-AZ, (ii) ISLA + ISLA-DTT test; (iii) dropped, no PREREG_ADDENDUM; S1 handled as text.
| 376802 (array 0-1) | isla.azrescore.base | (i) baselines: GT 14 + Transolver 2 lanes, all 7 test sides (plan base.tsv md5 9dddd3b4), claims | 2026-10-24 23:09 | submitted |
| 376803 (array 0) | isla.azrescore.hn | (iv) E1-AZ HEADNORM 2 lanes, nonbox test (plan hn.tsv) | 2026-10-24 23:09 | submitted |
| 376804 (array 0-1) | isla.azrescore.lre | (iv) LRE-AZ 2 lanes, reynolds test (plan lre.tsv) | 2026-10-24 23:09 | submitted |

Production: `score.sbatch` md5 eb452e66, specscore.py md5 1edc8f04, OUT `$T/azrescore_out/score`.
Smoke results: INDEP_CHECK (smoke 376168): 6 cases x (GT, Transolver, ISLA, ISLA-DTT seed 42 + zero + free stream), worst relative difference 2.4e-8 (near-field threshold cells, float32 vs float64 face centroids); free stream vs AeroZoo freestream-v1.2.csv worst 4.1e-9 over 12 cases.
| 376818 (array 0-9) | isla.azrescore.isla | (ii) ISLA + ISLA-DTT 28 lanes, test sides, plan isla.tsv (nonbox, reynolds, full, blobs, bz2c, crossflow, dual_inlet), DTT chunk 1M | 2026-10-24 23:10 | submitted |

2026-10-25 02:55 EDT: every job of 376802/376803/376804/376818 ended at its deadline; sacct total 214.3 GPU-h (smokes 9.0, base 28.1, hn 10.6, lre 23.7, isla 142.9). One ISLA process (376818_8 gpu0) died at start (EADDRINUSE on MASTER_PORT; known_anomalies.txt); claims moved its share to the others. Fixed: score.sbatch picks a free port (md5 160efddf); specscore.py checks the verbatim near field only on cases under 5 M cells (md5 4577daff).
Complete: E1-AZ, LRE-AZ; ISLA+DTT nonbox, reynolds, full, blobs; baselines full, nonbox, reynolds, blobs, bz2c. Partial: baselines crossflow 376/578; ISLA bz2c 276/333.
Supervisor 02:58 EDT: cap 265 GPU-h; finish (ii) on all seven splits; stop and ask if the cost exceeds 5.5 GPU-h per G cells.

| 378817 | isla.azrescore.base2 | (i) remainder: baselines crossflow + dual_inlet (short QoS, 01:59) | 2026-10-25 02:57 | submitted |
| 378820 (array 0-2) | isla.azrescore.isla2 | (ii) remainder: ISLA+DTT bz2c, crossflow, dual_inlet (plan isla_rest.tsv) | 2026-10-25 02:58 | submitted |

2026-10-25 06:35 EDT: 378817 and 378820 complete; every lane complete (reduce notes: no missing case, no duplicate). sacct: 378817 4.0 GPU-h, 378820 39.9 GPU-h; total 258.2 GPU-h of the 265 cap (ISLA remainder 4.7 GPU-h per G cells). Registry row removed 06:35 EDT (backup ACTIVE_EXPERIMENTS.tsv.bak2).
