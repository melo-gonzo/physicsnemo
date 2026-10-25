# ISLA-API 013 jobs (LHR; $T = /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0)

Registry row `^isla\.isla-api\.` added 2026-10-24 21:55 EDT (expected_end 2026-10-25).
Snapshots (read-only, MD5SUMS, SNAPSHOT_HEAD; built by lhr/build_snapshot.sh): $T/code_islaapi_092e (092ee6a0f, commits a+e),
$T/code_islaapi_base (5b70c8686, the base, for rez_check's old side).
Logs: $T/kernel/islaapi/tests_<jobname>_<jobid>.log, gb_<jobid>.log, out_gb_<jobid>/.

| job | name | what | code | result |
|---|---|---|---|---|
| 377081 | isla.isla-api.cpu092e-a | cpu-short, ISLA tests: tree_read, interior_geometry, geometry_* | code_islaapi_092e | 229 passed, 199 skipped (CUDA), 71 s |
| 377082 | isla.isla-api.cpu092e-b | cpu-short, ISLA tests: the other 18 files | code_islaapi_092e | 1 failed (test_isla_reference_checkpoint_matches_golden: stored args, expected until the version commit), 217 passed, 40 skipped |
| 377083 | isla.isla-api.cpu092e-r | cpu-short, recipe tests | code_islaapi_092e | 2 failed (test_campaign_c_hooks init_from x2: DistributedManager TypeError int(None) under SLURM env), 595 passed, 15 skipped |
| 377137 | isla.isla-api.cpubase-r | cpu-short, recipe test_campaign_c_hooks.py on the base | code_islaapi_base | the same 2 failures, same TypeError messages: pre-existing (SLURM env in cpu-short) |
| 377414-377416 | isla.isla-api.cpu65cc-{a,b,r} | first try at 65cce1c2e | code_islaapi_65cc (incomplete snapshot: tensordict copy missing) | import errors (venv tensordict 0.12.4); snapshot finished by hand, resubmitted |
| 377452 | isla.isla-api.cpu65cc-a | ISLA tests part a (+test_isla.py) | code_islaapi_65cc (65cce1c2e: a, e, C3, C4, revert, merge of fc6ed19d0) | 280 passed, 199 skipped, 2 xfailed |
| 377453 | isla.isla-api.cpu65cc-b | ISLA tests part b | code_islaapi_65cc | 1 failed (mdlus golden, expected before the version commit), 169 passed, 41 skipped |
| 377454 | isla.isla-api.cpu65cc-r | recipe tests | code_islaapi_65cc | 3 failed: campaign_c_hooks x2 (pre-existing), test_symmetry_plane hemisphere (overlay: old dataset_paths.yaml without highlift_aero_ml_symmetry_check) |
| 377644 | isla.isla-api.cpu7b60-isla | whole ISLA directory, CPU | code_islaapi_7b60 (7b60bdf86; physicsnemo+test identical to 25bea52cb) | **453 passed, 240 skipped, 2 xfailed, 0 failed** (6.5 min) |
| 377645 | isla.isla-api.gb | GB300, 16:03 (1.07 GPU-h): g0 whole ISLA dir with CUDA; g1-3 smokes; rez_check | code_islaapi_7b60 | **g0: 698 passed, 1 skipped (ISLA_AF_POSED_SAMPLE env), 2 xfailed, 0 failed**; smokes OK: aerozoo, airfrans (n_spatial_dims 2), lug, tensile2d (2); volume and surface failed on the overlay's dataset_paths (DrivAerML root ???); rez: AeroZoo 4/4 and AirfRANS 4/4 bitwise, 2 dumps x (3+1) old runs, all bitwise |
| 377706 | isla.isla-api.cpu7b60-recipe | recipe tests | code_islaapi_7b60 | 3 failed: campaign_c_hooks x2 (pre-existing), test_boundary_scales_and_log_reynolds_are_isla_s_bitwise (boundary_tokens keyword; fixed in b64e01981) |
| 378081 | isla.isla-api.cpu25be-recipe | recipe tests | code_islaapi_25be (25bea52cb) | **2 failed (campaign_c_hooks x2, pre-existing, same TypeError on the base), 613 passed, 18 skipped** |
| 378080 | isla.isla-api.gbsmoke2 | GB300, 2:38 (0.18 GPU-h): DrivAerML volume and surface smokes | code_islaapi_25be | **both SMOKE OK** (volume interior DTT 3 steps, peak 147.7 GiB; surface 3 steps, peak 84.6 GiB) |

GPU total 1.25 GPU-h of the 3 GPU-h cap. Registry row removed 2026-10-25 01:41 EDT (no jobs left).
Head 7c88b2e1f adds one test-only commit after 25bea52cb (test_isla_scales.py: 15 passed locally, CPU, under the lock).
