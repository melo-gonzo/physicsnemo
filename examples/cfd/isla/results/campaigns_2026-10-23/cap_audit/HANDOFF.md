# CAP-AUDIT handoff (research-lane CAP-AUDIT, 2026-10-23 11:52 EDT)

The phase-1 handoff of the earlier lane is in `HANDOFF_P1.md`. Results: `RESULTS.md`. Pre-registration and amendments:
`PREREG.md` sections 1-15.11. Jobs and commands: `JOBS.md`. Notebook drafts: `NOTEBOOK_DRAFT_P1.md` and
`NOTEBOOK_DRAFT.md`.

## State

- **Nothing is running.** Every registered and beside-the-rules run of phases 1, 1b and 2 has finished and been read:
  - phase 1: 60 runs; phase 1b: 20;
  - phase 2: the ladder (14 runs), Part B (6), 15.9 (4), 15.10 (4), 15.11 (4);
  - the path checks, the residual maps and the three far-field probes.
- Outputs on LHR, `T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0`:
  - `$T/cap/runs*/` (logs; only the final models of Part B's six drv runs are kept);
  - `$T/cap/labels/`, `$T/cap/probes/`;
  - the reductions `$T/cap/phase1*_floors.*`, `phase2_floors.*`, `ctl40k.md`, `drvladder.md`, `residuals_*.csv`.
- Local copies of the tables are in `tables/`.
- Registry: rows `^isla\.cap\.` and `^isla\.cap\.p2` in ACTIVE_EXPERIMENTS.tsv can be removed; nothing of this lane is
  queued.

## Decision 1: the slice question (costed; not launched)

The question: is s128's 1.27x lower AirfRANS pressure floor (gm 0.00157 against the ladder reference's 0.00198,
seeds 44 / 45; its near-wall band 0.0017-0.0018 against 0.0024-0.0031) real?

**Noise measured on this case.**
- At 20k: over 7 reference runs (phase-1b eager s42 / s43, ladder eager s44 / s45, compiled s42 / s44 / s45), the sd
  of ln F is 0.135 and the range 1.43x.
- At 40k: two seeds, 0.00096 and 0.00062, give sd 0.31. One degree of freedom, so its 95 % interval is roughly
  0.14 to 10.
- The effect to resolve is ln 1.27 = 0.239.

**Seeds per arm at the registered 2 SE bar** (difference of mean ln F >= 2 SE, SE = sigma sqrt(1/n1 + 1/n2)):

| budget | sigma | n per arm for an expected effect = 2 SE | n per arm for 80 % power at 2 SE | GPU-h (s128 + ref, 80 % power) |
|---|---|---|---|---|
| 20k | 0.135 | 3 | 6 | about 15 more (7 references exist; s128 needs 5 in all, so 3 more at 3.7 GPU-h; one 4-GPU node with seeds 46-49, about 4 h) |
| 40k | 0.135 (if the 20k spread holds) | 3 | 6 | about 115 (6 x (7.5 + 11.6) GPU-h) |
| 40k | 0.31 (two-seed estimate) | 14 | 28 | about 535 |

- At 40k, sigma must first be measured: a 4 + 4 pilot (s128 and the reference, seeds 46-49) costs about 77 GPU-h, on
  two 4-GPU nodes for about 12 h.
- These are single-case runs, so they answer only single-case fitting.

**A full-data AirfRANS pair instead** (s128 against 512 slices, two seeds each, the recipe's protocol):
- The 512-slice arm is REF-090-CLOSE's eager-trained AirfRANS standing, two seeds, so only the s128 arm is new.
- At REF-090's cost (about 110-120 GPU-h per 4-GPU seed lane; s128 steps about 0.65-0.8x as long), that is about
  150-180 GPU-h and 25-30 h of wall time.
- About 400 GPU-h if both arms are retrained on the same path.

**Recommendation: the full-data pair.** It answers the scope-freeze question better.
- Phase 1b shows that ISLA's single-case AirfRANS floors (0.0017) sit 9x below its cross-case test error (0.015). So a
  1.27x change in the single-case floor says little about the score that matters.
- The single-case floor is noisy here: 1.43x at 20k and 1.56x at 40k over a few runs. Resolving the effect at 40k
  could take 28 seeds per arm.
- Full-data scores have the program's 1-16 % two-seed spread, so a two-seed pair resolves an effect of 20 % or more.
- Cheaper alternative, if a single-case answer is wanted first: the 20k test (3 more s128 seeds, about 15 GPU-h,
  about 4 h).

## Decision 2: DrivAerML Volume follow-ups (not launched)

1. **Ground and domain walls as inputs.** This is the probe's lead.
   - Height above the floor explains 38 % of the far-field nu_t variance, against 8 % for the streamwise coordinate,
     on 11 cases.
   - ISLA reads only `vehicle`, while the files carry `no_slip` (the floor strip), `slip`, `inlet` and `outlet` as
     meshes.
   - Test: ISLA drv, Part B's protocol, with `no_slip` (and `slip`) added to `forward_kwargs.boundary_meshes` and
     `model.boundary_schemas` by name. This is a model-config change only, but it is a new input: it needs Peter's
     input ruling checked and its own pre-registration.
   - One fresh-draw run at seed 44: about 18 GPU-h, about 18 h on one GPU of a node; pack 2 seeds plus 2 controls.
   - Readout: the far-field (> 1 m) nu_t e_R, from 0.029-0.042 toward Transolver's 0.010, and velocity's uniform
     factor of 2.
2. **drv up-rungs** (15.10 found depth to be the capacity axis: halving it costs 1.36-1.61x).
   - Memory: the reference takes 232 GiB on a GB300. By the af ratios, w1024 / d24 / s2048 need about 415 / 438 / 417
     GiB, so they do not fit.
   - They need a block-level activation-checkpoint wrapper in the harness (not the model): wrap each `ISLA.blocks[i]`
     forward in `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`. Peak memory is then the stored block
     inputs (about 0.8 GB per block at width 512 and 400k tokens) plus one block's activations. All three up-rungs
     should fit under about 150 GiB; verify with a 100-step smoke first.
   - Cost, at about 1.35x step time for the recompute: d24 about 7.7 s/step, so 45 h per 20k run; w1024 about 9.3
     s/step, 54 h; s2048 about 8 s/step, 46 h.
   - Two seeds of d24 alone (the axis that moved): about 90 GPU-h, about 45 h of wall time plus preemptions.
   - All three at two seeds: about 300 GPU-h, about 55 h.

## Rulings and findings for project memory

- **Held-out sets on per-draw-centered datasets.** On DrivAerML Volume (drivaer_ml_volume_reference), CenterMesh
  centers every draw on its own mean. A held-out set must use its draw whole: never another draw's boundary. ISLA
  computes its wall geometry from the boundary, so a 1.16 mm frame shift raised its error from 0.022 to 0.39 (PREREG
  15.8). AeroZoo, AirfRANS, the lug and Tensile2d do not center.
- **Run-to-run spread of single-case floors.** On the AirfRANS case it is 1.43x at 20k (7 runs) and 1.56x at 40k (2
  runs), well above the program's 1-16 % two-seed assumption. Single-case rungs need several seeds.
- **Compiled-path training costs nothing** on this case: at fixed seed, eager and compiled training agree within 2 %
  (P = -0.018). Report it beside REF-090-CLOSE's full-protocol comparison, never pooled.
- **ISLA on DrivAerML Volume does not resume bit-for-bit** after a checkpoint, with or without the 15.8 flag. It is
  nondeterministic on that case; ISLA on AeroZoo and HiLift resumes exactly.

*Supervisor's note (2026-10-23 13:13 EDT):* on DrivAerML Volume every datapipe length is in units of L_ref = 5 m, so the
far-field band '> 1 m' is beyond 5 m from the car and the 15.8 offset is 5.8 mm (RESULTS.md, erratum). The supervisor's
decisions of 12:00: SLICES-AF (the full-data pair, launched), FLOOR-DRV (launched with all four domain meshes, no_slip,
slip, inlet and outlet; about 120 GPU-h, since they cost ISLA 0.9.0's geometry 1.58x per step), and the d24 rung
deferred until FLOOR-DRV reads.
