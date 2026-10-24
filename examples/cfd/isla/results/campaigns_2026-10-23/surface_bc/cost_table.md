# SURFACE-BC cost on GB300: the ISLA surface task on DrivAerML Surface, body only against the whole boundary

ISLA 0.9.0 (a surface task makes no geometry call, so 0.9.0's and the bundle's geometry do not enter). bfloat16 autocast,
eager with `compile_blocks=true` (the cost lanes' setting; the recipe's default whole-model compile was not measured),
200,000 vehicle cells (the reader's draw), one GB300 per arm. Body = the vehicle alone (`boundary_schemas: {vehicle: {}}`);
whole = vehicle + inlet, no_slip, outlet, slip (205,054 tokens), predictions at the vehicle in both. Raw records in
`out_cost/` (copied from `$T/kernel/surfbc/out_cost` and `$T/kernel/surfbc/prof_*`).

## 1. One case (run_1), `lhr/surf_cost.py`, job 362195, medians of calls 6-25, two orders (body first, whole first)

| quantity | body | whole | whole / body |
|---|---|---|---|
| training step (forward, measure-weighted MSE, backward, AdamW) | 222.2 / 224.1 ms | 274.7 / 271.3 ms | 1.22x |
| eval forward (no_grad) | 63.0 / 64.5 ms | 74.3 / 72.5 ms | 1.15x |
| peak memory, training | 69.06 GB | 70.71 GB | +1.65 GB (1.02x) |
| peak memory, forward | 3.88 GB | 3.96 GB | 1.02x |
| tokens / outputs | 200,000 / 200,000 | 205,054 / 200,000 | |
| ISLA's length unit L | 2.17 m | 59.50 m | 27.4x |
| losses (25 steps) | finite | finite | |

## 2. The recipe's train.py, 20 steps each (`lhr/steps_train.py`), job 362195, augmentation on, seed 42

Body = the base (c1e07b7fd snapshots code_bdaudit + recipe_bdaudit), whole = the branch (code_surfbc 129c66294 +
recipe_surfbc2 35a460b34). Two passes with the GPUs swapped; medians of steps 6-20 of the training log's `step_time_s`.

| run | body | whole | whole / body |
|---|---|---|---|
| pass 1 / pass 2 | 174.1 / 161.2 ms | 197.4 / 217.5 ms | 1.24x (means) |
| peak memory (log) | 64.95 GB | 67.29 GB | +2.3 GB |
| loss step 1 -> 20 | 0.186 -> 0.057 | 0.121 -> 0.062 | finite every step (the 20-step smoke) |

The whole run's resolved config lists the five boundary meshes and `prediction_boundaries: {literal: [vehicle]}`.

## 3. Where the extra time goes: the token count's alignment, not the tunnel (`lhr/surf_prof.py`, jobs 362252, 362293)

torch.profiler, 3 steps after 10 warm-up steps; training step (ms, steps 6-10 of the warm-up):

| arm | tokens | tokens mod 8 | step |
|---|---|---|---|
| body | 200,000 | 0 | 221.7-225.5 (two jobs) |
| body, all five conditions declared, the vehicle alone given | 200,000 | 0 | 223.1-224.4 |
| body | 205,048 | 0 | 227.7-228.3 |
| whole (vehicle at 199,994 cells) | 205,048 | 0 | 229.4-229.8 |
| body | 199,994 | 2 | 268.2-268.5 |
| body | 205,054 | 6 | 268.5-268.8 |
| whole (reference: vehicle at 200,000 cells) | 205,054 | 6 | 267.9-275.8 |

- At a token count that is a multiple of 8, the tunnel costs 1.5 ms (229.5 against 228.0 at the same count), and the whole
  boundary is 1.03x the 200,000-token body, the ratio of the token counts (1.025).
- Any count that is not a multiple of 8 costs about 45 ms (20 percent) more, tunnel or not. In the profile (`out_cost/prof_362252.txt`)
  `aten::mm` goes from 132 to 258 ms per three steps, and the GEMMs move from the `nvjet_sm103` kernels to
  `cutlass_80_tensorop_bf16` kernels; declaring five conditions costs nothing (body5).
- So the 1.22x of tables 1 and 2 is an artefact of 200,000 + 5,054 = 205,054. It concerns every ISLA run whose token
  count is not a multiple of 8 (any dataset, interior or surface), and it was measured in eager mode only.

GPU use: 362195 4:36, 362252 2:58, 362293 3:04 on one 4-GPU node each (short QoS): 0.71 GPU-h of the 4 GPU-h cap.

## 4. (B) for comparison (BOUNDARY-AUDIT section 4, DrivAerML Volume, the same structure as (B))

An interior task with 200,000 prediction points and 200,000 boundary tokens drawn from the full-resolution five pieces:
2,897 ms per training step and 140.6 GB on 0.9.0 (the Warp seed query takes 2.39 s of it; 513 ms without it; about 0.92 s
projected after GEO-FAST's seed fix). (A) on the same five pieces: 271-275 ms and 70.7 GB (229 ms at an aligned count).
