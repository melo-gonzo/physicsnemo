# DTT-KNOBS stage 3 proposal: K-B1 on DrivAerML Volume (for approval; nothing built or submitted)

Coordinator DTT-KNOBS, 2026-10-26 06:30 EDT. This is the stage named in PREREG.md section 7 (md5 39d8272f), triggered by
R3. On AirfRANS, K-B1 (the boundary read after every second block only: 0-based blocks 1, 3, ..., 11) read **no change on
all three fields and raised no flag** (RESULTS.md, arms 1-4). K-B1 is an architecture change, so it also needs Peter's
word. No other arm triggers stage 3: K-THETA2 raised an up flag on pressure, K-BASIS8 harmed nut, and the tighter arms
are not cheaper.

## Why DrivAerML Volume

The read's cost lives on the car. Stage 1 measured the 12 per-block reads at 9 % of the AirfRANS step and 26 % of the
lug's (eager, a lower bound), and K-B1 saved 5.0 % of the AirfRANS step. It gives no car-scale share, so K-B1's saving on
the car is unknown and is part of what stage 3 measures. Since 10-24 ISLA-DTT is ISLA's default interior form (AeroZoo's
close call, Peter's ruling of 10-23), so K-B1 would change the default model.

## Arms (registered in a stage-3 PREREG once approved)

- **Control:** ISLA-DTT at the canonical configuration of the day on DrivAerML Volume: the whole boundary, with the domain
  meshes as named conditions (as FLOOR-DRV built them), current psharpe/isla with DTT-FAST's read, theta 1, leaf 4.
- **Arm:** the same, with the read built and applied after blocks 1, 3, ..., 11 only. That is the three statements of
  branch psharpe/isla-dttknobs-b1 (a1d34dd39), ported to the current code.
- Two seeds each (42, 43), on that dataset's standing protocol: native and posed, float32, its scored set.
- New code goes on a branch psharpe/isla-dttknobs-b1v2 off the then-current psharpe/isla, with the whole ISLA test
  directory run before any job.

## Reading

R1 and the sub-bar flag of PREREG section 5, with that dataset's own control spread. Beside them: the step and memory of
both arms at 100k / 200k / 400k points.

## Cost and time (estimate; a smoke re-prices it)

- About 130 GPU-h per run, about 520 for the four runs (PREREG section 7). A fresh control pair is needed, because no
  standing ISLA-DTT DrivAerML Volume pair exists on the current code. If one appears (for example from the defaults
  switch), the ask falls to about 260.
- Wall time: about 1.5-2 days with chained 4 h links.
- That cannot finish inside this lane's deadline (23:34 EDT today). It is a new lane's work, if approved.

## Not proposed

K-THETA05 or K-EXACT at car scale. Their AirfRANS gains are below the bar, and stage 1's cost model (code before DTT-FAST)
put theta 0.5 at about 7x theta 1's pairs on DrivAerML Volume. A theta-0.5 car run would first need a DTT-FAST-era memory
fit, which nothing has measured.
