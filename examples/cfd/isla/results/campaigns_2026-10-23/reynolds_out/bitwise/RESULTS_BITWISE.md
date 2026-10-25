# REYNOLDS-OUT bitwise relocation check: results

Written 2026-10-24 23:04 EDT by REYNOLDS-OUT-BITWISE. Brief: `../BRIEF_BITWISE.md`.

**Question.** Model A is the checkpoint as trained, with `reynolds={mean, std}` and the viscosity-role global.
Model B has the same class and the same state dict. Its global_schema drops the viscosity field and adds
`re_z: {rank: 0}`, and it is built with `reynolds=None`. B gets z, computed outside the model by
`proto/recipe_z.py`, as an ordinary global. Do A and B compute the same function, bitwise?

## What ran

- GPU verdict job **376561** (LHR GB300, short QoS, node nvl72d024-T13): 0:03:41 elapsed, **0.245 GPU-h**. Three
  single-GPU processes ran in parallel, float32, eager (`compile=false model.compile_blocks=false`), no autocast,
  eval mode. CPU debug jobs on cpu-short are listed in `JOBS.md`.
- Code: `$T/code_r090` (92d9d7800 + td0152), torch 2.12.0+cu130, venv `.venv-recipe`. Recipe: `$T/recipe_r090`.
  The AirfRANS score rows name `code_cmpaf_fix`. Its isla/model.py differs from code_r090 only in the inductor options
  of `compile_blocks` (checked with diff), and those do not apply in eager mode.
- Scripts: `reyout_probe.py` (md5 922be593), `recipe_z.py` (a byte copy of proto/recipe_z.py, bd0b7d24), and
  `reyout.sbatch` (f54b64b5), here and in `$T/kernel/reynoldsout/`. Raw results: `gpu_376561/<run>/result.json`
  and logs. Per-case z values with float64 hex: `z_cases_376561.tsv`.
- How A was built: as infer.py builds it. Hydra compose from the run's score row (AirfRANS) or lane row (AeroZoo,
  with the test scorer's `val_split=reynolds.test`), then model_output_stats, model_dataset_constants,
  load_checkpoint at the final epoch, and check_model_output_stats. Cases are read in the scorer's order, with
  infer.py's seed and the RNG restored after each case. The sample ids match the scorer's metrics.jsonl on every
  case read, in all three runs.
- How B was built: `type(A)(**edited copy of A._args["__args__"])`, then `load_state_dict(A.state_dict(), strict=True)`.
  All keys matched, and every tensor is bitwise equal. B's args differ from A's only in `global_schema` and `reynolds`.
  A's args equal the checkpoint's stored `args.json`, except `compile_blocks` on AeroZoo, which I set to false.
- Checked in code (asserts): A has `global_scalar_names == []`, `n_global_scalars == 1`, and z appended last.
  B has `global_scalar_names == ["re_z"]`, `n_global_scalars == 1`, and no viscosity role. A and B have the same
  velocity sources, and the same density role on AeroZoo. The recipe_z arguments come from the stored declarations
  (sorted conditions, velocity_scale fields) and are asserted equal to A's `_boundary_sources` / `_global_sources`.
  B's global_data had the viscosity leaf REMOVED and `re_z` added as a float64 tensor of the TensorDict's batch shape.

| run | dataset / split | velocity source | stored `reynolds` (= A's) | B global_schema |
|---|---|---|---|---|
| r090eag_af_dtt_seed42 (epoch 500) | airfrans_uvolume_2d / full_test (200) | global U_inf | {6.205, 0.1347} | U_inf, re_z |
| r090eag_af_isla_seed42 (epoch 500) | same | global U_inf | {6.205, 0.1347} | U_inf, re_z |
| azc_reynolds_isla_seed42 (epoch 250) | aerozoo_uvolume_reynolds / reynolds.test (1,385) | boundary inlet.bc_velocity | {5.062, 0.8985} | physics.density (role density), re_z |

**Caveat on the brief's constants.** The brief gives AirfRANS reynolds as {9.065, 0.1338}. The checkpoints'
stored args and recipe_r090's `datasets/airfrans_uvolume_2d.yaml` both say **{6.205, 0.1347}**: log10(U L / nu)
with L about 0.41 chord is about 6, not 9. Both models used the stored values, so the comparison is unaffected,
but the 9.065 figure should not be carried forward. The AeroZoo constants match the brief.

## (Z) z computed by the model vs by the recipe (GPU, 376561)

z_A comes from A's own path: forward's first lines up to `_case_scales`. Inside the F forwards I also captured
z from `_case_scales`, and it was bitwise equal to the path value. z_B comes from recipe_z.

| run | cases | max abs(z_A - z_B), float64 | float64 bitwise | float32 bitwise |
|---|---|---|---|---|
| r090eag_af_dtt_seed42 | 64 | 0 | 64 / 64 | 64 / 64 |
| r090eag_af_isla_seed42 | 64 | 0 | 64 / 64 | 64 / 64 |
| azc_reynolds_isla_seed42 | 32 | 0 | 32 / 32 | 32 / 32 |

z ranges over [-2.04, -0.41] on the AirfRANS test cases and over [1.34, 4.17] on the AeroZoo high-Re test cases.

## (F) forward outputs, A vs B (GPU, 376561)

Before each forward I called `torch.manual_seed(0)`. I compared every point_data leaf and the points, and also the
outputs after normalize_output_to_tensordict. "Seed scalars" means the g_scalars tensor that `_global_fields`
returns, which is the only place z enters the network.

| run | cases | seed scalars bitwise | outputs bitwise | max abs diff (A vs B) | A-vs-A floor (case 0) |
|---|---|---|---|---|---|
| r090eag_af_isla_seed42 | 4 | 4 / 4 | **4 / 4** | 0 | bitwise |
| r090eag_af_dtt_seed42 | 4 | 4 / 4 | 0 / 4 | 7.4e-6 (cases: 2.1e-6, 2.3e-6, 5.5e-6, 7.4e-6; outputs up to 16) | **NOT bitwise: 2.3e-6** |
| azc_reynolds_isla_seed42 | 2 | 2 / 2 | 1 / 2 | 2.6e-6 on case 0 (velocity, max abs value 4.3e-2); case 1: 0 | bitwise (one repeat) |

The points are bitwise equal everywhere. On the CPU debug runs (376218, 376219: 1 case each, all three runs), A vs B
and A vs A were bitwise for every output. The global_data that the output Mesh carries differs by construction:
A carries nu, B carries re_z.

## Verdict

**z: bitwise on all checked cases.** On 160 cases over three checkpoints, recipe_z gives the same float64 z as
the model, bit for bit, so also the same float32 value that enters the network. In every forward checked, the
seed scalars, the network's only z input, were bitwise equal.
**Outputs: bitwise on all checked cases of AirfRANS ISLA (4/4) and AeroZoo case 1. Equal to float32 roundoff
only on AirfRANS DTT (4/4 cases, at most 7.4e-6), where A run twice on the same input also differs (2.3e-6).**
The DTT read is not run-to-run deterministic on GPU, so bitwise A-vs-B cannot be expected there. The A-vs-B
gaps are the same size as the A-vs-A floor: 2.1e-6 on the floor's own case, against a floor of 2.3e-6.
**AeroZoo case 0 differs by 2.6e-6** even though A-vs-A on it was bitwise in one repeat. A, B and every input are
identical, so this is most likely intermittent GPU nondeterminism, but one repeat cannot prove it. I asked the supervisor for a
second short GPU job (A,B,A,B,A,B with block hooks; `reyout_repeat.py`, tested on CPU in 376712) and got no reply,
so it did not run. Overall: **the relocation reproduces the model's function exactly at the network input. At
the output it is bitwise wherever the GPU forward is itself reproducible, and within run-to-run roundoff (at most
7.4e-6) otherwise. The one unexplained case is AeroZoo case 0, at 2.6e-6.** I am confident in the z result and the
mechanism (bitwise inputs; CPU bitwise end to end). My confidence that the AeroZoo case-0 gap is nondeterminism
rather than a relocation effect is moderate, about 85%.

## Addendum (2026-10-24 23:20 EDT): repeat job 376855, A,B,A,B,A,B with block hooks

The parent lane approved this job at about 23:10. The verdict above stands as written; this section reads beside it.

**What ran.** Job **376855** (LHR GB300, short QoS, nvl72d018-T11): 0:02:25 elapsed, **0.161 GPU-h**. I used
`-t 00:11:00` instead of the approved 00:14:00, so that even a run to the time limit (0.73 GPU-h) would keep the
lane under its 1 GPU-h cap. The lane total is 0.406 GPU-h. The job ran `reyout_repeat.py` (md5 f48030a4) through
`reyout2.sbatch` (ec4ee38a) with `NZ=2 NF=2 REPS=3`, on all three runs, cases 0 and 1, in the scorer's order.
Each case ran the six forwards A1, B1, A2, B2, A3, B3 in one process, with `torch.manual_seed(0)` before each.
Hooks compared all 24 block tensors (each of the 12 blocks' input and output) with A1's, bitwise, and recorded
the max abs difference. All 15 output pairs per case are in `repeat_pairs_376855.md`. Raw files:
`gpu_376855/<run>/result.json` and logs. Z was again bitwise on these cases (2/2 per run, float64).

| run, case | A-vs-A pairs (3) | B-vs-B pairs (3) | A-vs-B pairs (9) | first tensor that differs from A1 |
|---|---|---|---|---|
| AirfRANS ISLA, 0 | 3 bitwise | 3 bitwise | **9 bitwise** | none: every block bitwise in all 5 forwards |
| AirfRANS ISLA, 1 | 3 bitwise | 3 bitwise | **9 bitwise** | none |
| AirfRANS DTT, 0 | 0 bitwise, 1.94e-6 to 2.32e-6 | 0 bitwise, 1.89e-6 to 2.09e-6 | 0 bitwise, 1.76e-6 to 2.53e-6 | block 1 input (just after the first tree read), in A and B forwards alike; block 0 bitwise everywhere |
| AirfRANS DTT, 1 | 0 bitwise, 2.16e-6 to 2.37e-6 | 0 bitwise, 2.23e-6 to 2.28e-6 | 0 bitwise, 2.17e-6 to 2.59e-6 | block 1 input, in A and B alike |
| AeroZoo, 0 | 3 bitwise | **0 bitwise, 2.80e-6 to 8.77e-6** | 3 bitwise (all with B3), 6 at 2.80e-6 to 5.98e-6 | B1 and B2: the **block 0 input**, which differs from A1 by 0.65 and 5.45 (max abs). B3: none, so B3 is bitwise equal to A1, A2 and A3 at every block and every output |
| AeroZoo, 1 | 3 bitwise | 3 bitwise | **9 bitwise** | none |

**Reading.**
- **AirfRANS DTT:** every A-vs-B gap lies inside the run-to-run spread. On case 0, A-vs-B spans 1.76e-6 to 2.53e-6,
  against 1.89e-6 to 2.32e-6 for A-vs-A and B-vs-B. On case 1, A-vs-B spans 2.17e-6 to 2.59e-6, against 2.16e-6 to
  2.37e-6. Every forward of A and of B first departs from A1 at the input to block 1, the first point after the
  Barnes-Hut tree read. The read is therefore not run-to-run deterministic on GPU, and the two models are not
  distinguishable from each other.
- **AeroZoo case 0:** the relocation is not the cause. **B3 reproduces A bit for bit**, at all 24 block tensors and
  every output. B1 and B2 differ from each other (8.8e-6) as well as from A. A deterministic effect of the relocation
  would give B1 = B2 = B3. The seed scalars were bitwise equal in all six forwards. The variation enters before
  block 0, in the token tensor that block 0 receives.
- The size of the AeroZoo pattern points to token order, not token values. Block 0's input differs by O(1), and
  the hidden states at block 11 differ by up to 6.7e3. Yet the outputs differ only by 3e-6 to 9e-6, about 1e-4 of
  the largest velocity, which is 4.3e-2. That is what one would see if the same boundary tokens were assembled in
  a different order: the slice reductions are permutation-invariant up to the order of the float sums. Two causes
  fit this: a GPU step without a fixed order in the boundary-token draw or the token assembly (for example, a sort
  or unique with ties, or a nonzero), or a read of uninitialized memory that depends on the allocator's state. I
  did not test which. That needs a per-step hook (`sample_boundary_cells`, `_condition_tokens`), which is outside
  this brief.
- In this sample the variation hit only B forwards (2 of 3), never A (0 of 3), and on case 1 neither model. With
  six forwards that may be chance. B3 shows that B's code path can and does produce A's exact function. If a
  layout difference made such variation more likely, it would be a property of the pre-block step, not of z.

**Updated verdict, beside the first.** The relocated form computes **the same function as the current model,
bitwise, whenever the GPU forward is itself reproducible**. That holds for AirfRANS ISLA (6 of 6 cases over both
jobs, 9 of 9 A-vs-B pairs per case) and AeroZoo case 1, and on CPU for all three runs. Where the forward is not
reproducible, the A-vs-B gaps lie inside the A-vs-A and B-vs-B spread: AirfRANS DTT (tree read, block 1 onward)
and AeroZoo case 0. On AeroZoo case 0 one B forward matched A exactly, and the variation enters before block 0.
I am now confident, about 95%, that the relocation is exact. The remaining doubt is whether something
layout-dependent in the pre-block step reacts to B. That would be the step's nondeterminism, not a change in z.
**Side finding for the program:** ISLA 0.9.0's eager GPU forward on AeroZoo case 0 does not reproduce from run
to run before block 0 (token order or values), with outputs varying by up to 8.8e-6. The DTT tree read varies by
about 2e-6 on AirfRANS. Neither is related to Reynolds. The first matters for any bitwise A/B test on AeroZoo, and
is worth locating.
