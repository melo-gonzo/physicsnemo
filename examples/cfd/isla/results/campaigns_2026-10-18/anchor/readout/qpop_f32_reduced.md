# AF-QPOP reduction (/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/kernel/anchor_af/qpop/out)

ratio = mean rel-L2 over the cases / native_h's; change = max over cases and scored points of |prediction - native_h| / RMS target.

| run | epoch | cases | condition | ratio p | ratio v | ratio nut | change p | change v | change nut |
|---|---|---|---|---|---|---|---|---|---|
| anc_anchor_seed42 | 500 | 200 | repeat | 1.000000 | 1.000000 | 1.000000 | 6.23e-05 | 2.06e-05 | 1.98e-05 |
| anc_anchor_seed42 | 500 | 200 | half | 1.000009 | 1.000003 | 1.000001 | 8.17e-03 | 2.18e-03 | 1.92e-03 |
| anc_anchor_seed42 | 500 | 200 | far | 0.999992 | 1.000002 | 1.000003 | 3.85e-03 | 1.56e-03 | 3.43e-03 |
| anc_anchor_seed42 | 500 | 200 | wall | 0.999992 | 1.000002 | 1.000003 | 3.85e-03 | 1.56e-03 | 3.43e-03 |
| anc_anchor_seed42 | 500 | 200 | chunks | 1.000009 | 1.000001 | 0.999992 | 3.04e-03 | 8.39e-04 | 3.47e-03 |
| anc_anchor_seed43 | 500 | 200 | repeat | 1.000000 | 1.000000 | 1.000000 | 4.07e-04 | 8.53e-05 | 4.41e-05 |
| anc_anchor_seed43 | 500 | 200 | half | 1.000004 | 1.000005 | 1.000019 | 9.53e-03 | 5.89e-03 | 5.87e-03 |
| anc_anchor_seed43 | 500 | 200 | far | 0.999992 | 0.999991 | 0.999966 | 2.09e-02 | 8.24e-03 | 9.20e-03 |
| anc_anchor_seed43 | 500 | 200 | wall | 0.999992 | 0.999991 | 0.999966 | 2.09e-02 | 8.24e-03 | 9.20e-03 |
| anc_anchor_seed43 | 500 | 200 | chunks | 1.000005 | 1.000008 | 1.000030 | 1.04e-02 | 6.47e-03 | 6.40e-03 |
| anc_dtt_seed42 | 500 | 200 | repeat | 1.000000 | 1.000000 | 1.000000 | 6.06e-06 | 5.09e-06 | 9.81e-06 |
| anc_dtt_seed42 | 500 | 200 | half | 1.000000 | 1.000000 | 1.000001 | 1.40e-03 | 5.57e-05 | 2.87e-04 |
| anc_dtt_seed42 | 500 | 200 | far | 1.000000 | 1.000000 | 0.999999 | 7.16e-04 | 2.70e-05 | 3.92e-04 |
| anc_dtt_seed42 | 500 | 200 | wall | 1.000000 | 1.000000 | 0.999999 | 7.15e-04 | 2.70e-05 | 3.94e-04 |
| anc_dtt_seed42 | 500 | 200 | chunks | 0.999999 | 1.000000 | 0.999997 | 6.67e-04 | 4.82e-05 | 1.04e-03 |
| anc_dtt_seed43 | 500 | 200 | repeat | 1.000000 | 1.000000 | 1.000000 | 5.83e-06 | 4.68e-06 | 1.18e-05 |
| anc_dtt_seed43 | 500 | 200 | half | 1.000001 | 1.000000 | 1.000000 | 2.89e-05 | 2.05e-05 | 4.11e-05 |
| anc_dtt_seed43 | 500 | 200 | far | 1.000001 | 1.000000 | 1.000000 | 5.16e-05 | 2.25e-05 | 5.01e-05 |
| anc_dtt_seed43 | 500 | 200 | wall | 1.000001 | 1.000000 | 1.000000 | 5.13e-05 | 2.22e-05 | 5.06e-05 |
| anc_dtt_seed43 | 500 | 200 | chunks | 1.000000 | 1.000000 | 1.000000 | 6.96e-05 | 3.12e-05 | 5.31e-05 |
| anc_isla_seed42 | 500 | 200 | repeat | 1.000000 | 1.000000 | 1.000000 | 3.71e-06 | 3.35e-06 | 3.21e-05 |
| anc_isla_seed42 | 500 | 200 | half | 1.224044 | 1.084292 | 1.161321 | 3.27e-01 | 5.18e-02 | 9.21e-01 |
| anc_isla_seed42 | 500 | 200 | far | 62.571314 | 31.688007 | 47.664929 | 1.01e+01 | 2.19e+00 | 1.47e+01 |
| anc_isla_seed42 | 500 | 200 | wall | 17.083218 | 18.094378 | 11.599480 | 4.58e+00 | 1.69e+00 | 6.83e+00 |
| anc_isla_seed42 | 500 | 200 | chunks | 1.554712 | 1.249534 | 1.413805 | 7.20e-01 | 1.19e-01 | 1.21e+00 |
| anc_isla_seed43 | 500 | 200 | repeat | 1.000000 | 1.000000 | 1.000000 | 5.56e-06 | 2.22e-06 | 3.51e-05 |
| anc_isla_seed43 | 500 | 200 | half | 1.268156 | 1.085626 | 1.143443 | 4.60e-01 | 1.41e-01 | 1.68e+00 |
| anc_isla_seed43 | 500 | 200 | far | 31.555473 | 25.883062 | 50.043296 | 1.25e+01 | 1.80e+00 | 2.31e+01 |
| anc_isla_seed43 | 500 | 200 | wall | 16.717223 | 15.893898 | 6.952477 | 3.06e+00 | 1.18e+00 | 4.79e+00 |
| anc_isla_seed43 | 500 | 200 | chunks | 1.631172 | 1.252792 | 1.430285 | 8.96e-01 | 1.99e-01 | 4.07e+00 |
| anc_mesh_seed42 | 500 | 200 | repeat | 1.000000 | 1.000000 | 1.000000 | 6.63e-06 | 2.63e-06 | 3.08e-05 |
| anc_mesh_seed42 | 500 | 200 | half | 0.999999 | 0.999999 | 1.000000 | 3.45e-04 | 1.39e-04 | 5.36e-04 |
| anc_mesh_seed42 | 500 | 200 | far | 1.000002 | 1.000001 | 1.000001 | 1.07e-03 | 3.08e-04 | 6.24e-04 |
| anc_mesh_seed42 | 500 | 200 | wall | 1.000002 | 1.000001 | 1.000001 | 1.07e-03 | 3.08e-04 | 6.24e-04 |
| anc_mesh_seed42 | 500 | 200 | chunks | 1.000000 | 0.999999 | 1.000001 | 3.97e-04 | 7.80e-05 | 4.98e-04 |
| anc_mesh_seed43 | 500 | 200 | repeat | 1.000000 | 1.000000 | 1.000000 | 7.01e-06 | 2.14e-06 | 3.05e-05 |
| anc_mesh_seed43 | 500 | 200 | half | 1.000001 | 1.000000 | 1.000002 | 1.84e-03 | 5.21e-04 | 1.64e-03 |
| anc_mesh_seed43 | 500 | 200 | far | 1.000000 | 1.000001 | 0.999999 | 3.53e-03 | 4.30e-04 | 1.80e-03 |
| anc_mesh_seed43 | 500 | 200 | wall | 1.000000 | 1.000001 | 0.999999 | 3.53e-03 | 4.30e-04 | 1.80e-03 |
| anc_mesh_seed43 | 500 | 200 | chunks | 1.000003 | 0.999999 | 1.000004 | 3.87e-03 | 3.56e-04 | 2.57e-03 |
| anc_passive_seed42 | 500 | 200 | repeat | 1.000000 | 1.000000 | 1.000000 | 3.97e-06 | 2.26e-06 | 2.97e-05 |
| anc_passive_seed42 | 500 | 200 | half | 1.000001 | 1.000000 | 1.000002 | 7.68e-04 | 9.77e-05 | 1.59e-04 |
| anc_passive_seed42 | 500 | 200 | far | 1.000002 | 1.000000 | 1.000001 | 7.88e-04 | 6.07e-05 | 1.10e-04 |
| anc_passive_seed42 | 500 | 200 | wall | 1.000002 | 1.000000 | 1.000001 | 7.88e-04 | 6.07e-05 | 1.10e-04 |
| anc_passive_seed42 | 500 | 200 | chunks | 1.000000 | 1.000000 | 1.000003 | 1.93e-03 | 1.51e-04 | 2.29e-04 |
| anc_passive_seed43 | 500 | 200 | repeat | 1.000000 | 1.000000 | 1.000000 | 4.50e-06 | 2.25e-06 | 4.41e-05 |
| anc_passive_seed43 | 500 | 200 | half | 1.000000 | 1.000000 | 1.000001 | 9.37e-04 | 1.08e-04 | 1.91e-04 |
| anc_passive_seed43 | 500 | 200 | far | 1.000000 | 1.000000 | 0.999999 | 6.44e-04 | 8.22e-05 | 1.02e-04 |
| anc_passive_seed43 | 500 | 200 | wall | 1.000000 | 1.000000 | 0.999999 | 6.44e-04 | 8.22e-05 | 1.02e-04 |
| anc_passive_seed43 | 500 | 200 | chunks | 1.000003 | 1.000000 | 0.999999 | 3.71e-03 | 4.26e-04 | 2.54e-04 |

| run | native_h p / v / nut | worst |ratio-1| or change | qi_ok | validity (full vs infer.py) |
|---|---|---|---|---|
| anc_anchor_seed42 | 0.0297 / 0.0165 / 0.0287 | 8.2e-03 | False | FAIL max rel diff 1.7e-04 on 200/200 cases (/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/evals_anc/anc_anchor_seed42/native/anc_anchor_seed42/metrics.jsonl) |
| anc_anchor_seed43 | 0.0224 / 0.0123 / 0.0211 | 2.1e-02 | False | ok max rel diff 3.0e-05 on 200/200 cases (/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/evals_anc/anc_anchor_seed43/native/anc_anchor_seed43/metrics.jsonl) |
| anc_dtt_seed42 | 0.0136 / 0.0087 / 0.0213 | 1.4e-03 | False | ok max rel diff 6.4e-05 on 200/200 cases (/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/evals_anc/anc_dtt_seed42/native/anc_dtt_seed42/metrics.jsonl) |
| anc_dtt_seed43 | 0.0137 / 0.0086 / 0.0232 | 7.0e-05 | True | ok max rel diff 3.2e-05 on 200/200 cases (/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/evals_anc/anc_dtt_seed43/native/anc_dtt_seed43/metrics.jsonl) |
| anc_isla_seed42 | 0.0181 / 0.0094 / 0.0211 | 6.2e+01 | False | ok max rel diff 5.4e-05 on 200/200 cases (/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/evals_anc/anc_isla_seed42/native/anc_isla_seed42/metrics.jsonl) |
| anc_isla_seed43 | 0.0186 / 0.0100 / 0.0240 | 4.9e+01 | False | ok max rel diff 3.7e-05 on 200/200 cases (/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/evals_anc/anc_isla_seed43/native/anc_isla_seed43/metrics.jsonl) |
| anc_mesh_seed42 | 0.0214 / 0.0105 / 0.0239 | 1.1e-03 | False | ok max rel diff 3.4e-05 on 200/200 cases (/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/evals_anc/anc_mesh_seed42/native/anc_mesh_seed42/metrics.jsonl) |
| anc_mesh_seed43 | 0.0199 / 0.0106 / 0.0234 | 3.9e-03 | False | FAIL max rel diff 1.2e-04 on 200/200 cases (/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/evals_anc/anc_mesh_seed43/native/anc_mesh_seed43/metrics.jsonl) |
| anc_passive_seed42 | 0.0169 / 0.0103 / 0.0237 | 1.9e-03 | False | FAIL max rel diff 1.0e-04 on 200/200 cases (/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/evals_anc/anc_passive_seed42/native/anc_passive_seed42/metrics.jsonl) |
| anc_passive_seed43 | 0.0145 / 0.0102 / 0.0246 | 3.7e-03 | False | ok max rel diff 6.5e-05 on 200/200 cases (/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/evals_anc/anc_passive_seed43/native/anc_passive_seed43/metrics.jsonl) |
