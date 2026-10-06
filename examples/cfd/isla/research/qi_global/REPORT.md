# QI-GLOBAL: an untruncated, exact boundary read for read-only ISLA

Lane `isla-qi-global`, branch `psharpe/isla-qi-global` (base
`psharpe/isla-qi-kernel` @ bb877be65). AGA task dir:
`/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-10-05-isla-qi-global/`
(job prefix `isla.global.`). Brief: `BRIEF.md` (this folder).

## Status (2026-10-05 17:00 PDT): started

- Implementation in progress.

## Question

An untruncated boundary read has every prediction point read every boundary cell,
with no length scale anywhere. Does it close read-only ISLA's wake eddy-viscosity
gap on AirfRANS, while keeping exact query independence and every ISLA invariant?

## Preregistered bars (written 2026-10-05 17:30 PDT, before any full lane)

AirfRANS only. Arms: **G** read-only + global read (seeds 42-45); **I** interacting-256
(42, 43 from the QI campaign, 44, 45 new); **P** read-only-256 (same seeds split);
**T** truncated read, `"sum"`, R = 0.01 L, last 4 blocks (42, 43 from the kernel
lane, 44, 45 new). Scores: `infer.py` exactly as the QI scorer (float32, the 200
held-out airfoils, the training request, per-case relative L2 averaged over cases),
four-seed means; standard error over seeds; paired ratios per seed.

- **Closes the gap:** all of
  - G / I <= 1.03 on pressure and on velocity;
  - G / I <= 1.05 on eddy viscosity;
  - in the wake band (\|sdf\| > 1 L, probe request), G's eddy-viscosity error
    within 10% of I's (G / I <= 1.10 there).
- **Helps the wake:** G's eddy-viscosity excess over I (ratio - 1) is at most half
  of T's, and G's wake-band eddy-viscosity error is below T's.
- **No wake effect:** G is within one standard error of T on eddy viscosity
  (four-seed means, the standard errors of the two means combined in quadrature).
- Otherwise: between; reported by field.

Also reported: G / P, G / T, the band anatomy (error by \|sdf\|/L band, probe
request), cost per step and peak memory, and every arm scored in bf16 beside the
float32 verdict (like with like).
