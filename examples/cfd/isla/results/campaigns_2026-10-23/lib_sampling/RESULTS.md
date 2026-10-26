# ISLA-LIB-SAMPLING: A/B results (BASE 5c39a6b52 vs HEAD 7f7e1c09a)

HEAD's physicsnemo/ is byte-identical to 3ebf9465b, which the probes ran: changed-file md5s a3569577 (sampling.py),
718d4acc (tree_read.py) and 3f4f88b2 (geometry.py) match the LHR snapshot. Commits after 3ebf9465b touch test/ only.
Probe: lhr/ab_probe.py. It runs the recipe's own train.main setup with a seeded init and the seeded train loader. Per case
it records the draw at the evaluation start and at a seeded training start. For the first case it records the eval
forward and the train forward, loss and every parameter gradient, twice (reps), with no optimizer step. Each arm runs in
its own process. Compare: lhr/ab_compare.py, bitwise on raw bytes. Per-pair JSON in results/ab_<job>/.

## Eager, GB300 (fp32; jobs 393521, 393763)
| family (model / dataset) | cases | draws bitwise | fwd, loss, grads bitwise | each arm's reps bitwise |
|---|---|---|---|---|
| DrivAerML volume (isla_volume / drivaer_ml_volume_reference, 200k tokens) | 6 | 12/12 + 3/3 | 931/931 (393763) | 454/454 |
| AeroZoo (isla_volume_aerozoo / aerozoo_uvolume) | 8 | 16/16 + 3/3 | 968/968 (393763) | 467/467 |
| AirfRANS (isla_volume_airfrans / airfrans_uvolume_2d) | 8 | 17/17 | 923/923 | 450/450 |
| lug (isla_lug / ginot_lug_volume) | 8 | 17/17 | 881/881 | 434/434 |
| Tensile2d (isla_tensile2d / plaid_tensile2d_2d) | 8 | 17/17 | 940/940 | 455/455 |
| DrivAerML surface (isla_surface) | 1 | no draw | 663/663 | 322/322 |
| HiLift surface (isla_surface_highlift, full_train) | 1 | no draw | 715/715 (393763) | 339/339 |
| SHIFT-SUV surface (isla_surface_shift_suv) | 1 | no draw | 665/665 | 323/323 |

In 393521 the DrivAerML and AeroZoo forwards stopped at the probe's fixed chunk capacity of 80 GiB, a predicted peak of
119 GiB. 393763 reran them at 250 GiB. HiLift needed train_split=full_train.

## Eager, CPU (fp32; jobs 393947, 393764): HEAD and commit a (c60621385) each against BASE
| family | cases | draws bitwise (HEAD / a) | fwd, loss, grads (HEAD / a) |
|---|---|---|---|
| DrivAerML volume, draws at the full 200k budget | 3 | 6/6 / 6/6 | no forward: CPU OOM at 180 GB (3 arms) |
| AeroZoo, draws at the full budget | 4 | 8/8 / 8/8 | no forward on CPU |
| AirfRANS | 4 | 9/9 / 9/9 | 923/923 / 923/923 |
| lug | 4 | 9/9 / 9/9 | 881/881 / 881/881 |
| Tensile2d | 4 | 9/9 / 9/9 | 940/940 / 940/940 |
| DrivAerML surface / HiLift / SHIFT-SUV | 1 each | no draw | 663/663, 715/715, 665/665 (both) |

## Compiled, GB300 (bf16, model.compile_blocks=true, the model's draw under torch.compile(fullgraph, dynamic))
Draws: compiled = eager on the same inputs in every record. BASE = HEAD bitwise, cold and warm, in every family: drv 6/6,
az 7/7, af 7/7, lug 7/7, t2d 7/7.

Forward and gradients (tensors bitwise / total):
| pair | af | az | lug | t2d | drvs (no draw) |
|---|---|---|---|---|---|
| BASE cold vs HEAD cold, separate caches (393521) | 369/923 | 504/968 | 471/881 | 390/940 | 1305/1345 |
| BASE warm vs HEAD warm, separate caches (393521) | 269/923 | 506/968 | 465/881 | 390/940 | 1305/1345 |
| HEAD cold vs HEAD warm, one cache (393521) | 923/923 | 942/968 | 875/881 | 940/940 | 1345/1345 |
| BASE cold vs BASE warm, one cache (393763) | 923/923 | 943/968 | | | |
| **BASE warm vs HEAD warm, one shared cache (393763)** | **923/923** | **968/968** | | | |

Reading: the cross-arm differences with separate caches are Inductor's compile draws. The evidence:
- They appear in drvs, which has no draw, so the change cannot reach its forward.
- BASE against itself, cold against warm, also differs on az.
- Given the same compiled artifacts, BASE and HEAD are bitwise.

AeroZoo's compiled bf16 gradients also vary from run to run within each arm (reps 150/467 on both sides), yet BASE and
HEAD agree on all 968 tensors. The DrivAerML compiled forward hit the 80 GiB capacity: its compiled draws are proven, its
compiled forward is not. Per the brief, the 14:13 aggregated certification applies only if the result is not bitwise; I
read "bitwise given one cache" as bitwise, and did not run it.

## Zero-measure edge on real data
In every draw recorded, no condition has a cell of zero measure (n_zero max 0; n_neg max 0). So no condition reaches
the edge (water-fill share >= positive count, with zero-measure cells present): **0 of 38 cases** (drv 6, az 8, af 8,
lug 8, t2d 8), over 321 eager GPU, 162 compiled and 152 CPU condition-draws. The library's zero-measure behaviour
changes no ISLA draw on these cases.

The new test `test_zero_measures_fill_a_condition_past_its_positive_cells` uses 200 cells (150 of zero measure) plus
1,000 cells, budget 300.
- On BASE's code it fails (edge_test_on_base.txt): condition 0 gets 150 tokens, but not every positive cell, and its HT
  measure sum is 32.94 against the true 46.99, at all three starts.
- On HEAD it passes (CPU tier 394351).
