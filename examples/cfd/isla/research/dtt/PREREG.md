# ISLA-DTT preregistration (2026-10-06, before any run)

Readout units: relative L2 per field on the held-out split of the recipe's protocol (float32 scoring, 200 AirfRANS
airfoils at 100,000 points; 48 DrivAerML cars at 200,000 points), case means; ratios of seed means with the
per-seed ratios beside them. "Seed separation" of a pair of references is the log gap of their two seeds.

## E1. Exactness at theta = 0 (contract)

The tree read at theta = 0 is the dense global read. Unit tests in float64: the two forms agree to 1e-10 on
random clouds in two and three dimensions, for every contract the dense read passes (query independence alone /
prefix / permuted / chunked; rotation, reflection, translation covariance; similarity gauge; boundary refinement
and measure rescale; a gradient for every parameter; CUDA equals CPU). On a trained G checkpoint
(`gl_af_global_seed42`, float32), the per-case predictions at theta = 0 agree with the dense read to 1e-5 relative
L2 and the protocol error agrees to four digits. Fails if any of these does not hold; nothing else is run until
it does.

## E2. Convergence and accuracy against theta on the trained G checkpoints (evaluation only)

Instrument: the four G checkpoints (seeds 42 to 45, LHR task dir `2026-10-05-isla-qi-global`), scored with the
tree read at theta in {0, 0.25, 0.5, 1.0} (leaf size the implementation constant), float32, the full protocol.
Readouts per theta and seed: (a) deviation of the predictions from theta = 0, relative L2 per field over the
held-out cases; (b) the protocol error per field; (c) pairs per block (cells and nodes separately) and read time
per case.

Predictions. (a) falls monotonically as theta falls and is below 1e-2 at theta = 0.5 on every field. (b) The
error at theta = 0.5 is within 1% of the exact read's on every field and seed (the ratio error(theta) /
error(0) in [0.99, 1.01]); at theta = 1.0 within 3%. If (b) fails at 0.5 but holds at 0.25, the DrivAerML lanes
run at 0.25 and the cost forecast is redone; if it fails at 0.25, the far-field aggregate is the suspect and the
orientation-binned aggregate (one virtual source per normal octant per node) is built before any training lane,
with E2 repeated. (c) On AirfRANS the tree cannot beat the dense read (1,000 cells); (c) is recorded for the
cost model only.

## E3. DrivAerML Volume: accuracy of the tree read against the interacting reference

Arms (LHR, 500 epochs, 200,000 cells and points, the campaign's Volume recipe at the reference configuration
512 / 512 slices, lr 3.75e-4, `query_geometry` as the canonical default at launch): TG = read-only + tree read at
the theta E2 selects, seeds 42 and 43; P = read-only, no read, seeds 42 and 43 (new; the QI program's P was at the
older 384 / 256 configuration); I = the existing interacting references `cl_r_vol_isla_seed42/43`. Scoring: the
protocol (float32, 48 cars).

Amended 2026-10-06 evening, before any TG lane ran. E2 failed at every theta (REPORT.md): the dense-trained
kernel's first-order cluster aggregates of keys and values leave a 2% to 5% deviation at theta 0.25 to 0.5 even
with orientation-coherent clusters, so "the theta E2 selects" does not exist. TG is therefore trained with the
tree read at a fixed theta, chosen by cost under the bar below (step within 6x the read-only floor): theta 1 and
theta 0.5, two seeds each, scored at their training theta (and at theta 0 as a diagnostic of how far the trained
model is from its own dense limit). The bars against I and P are unchanged. The orientation-aware tree
(`boundary_tree.py`, RMS normal spread) is the tree of every lane.

Bars (the pi lane's, unchanged): closes the gap if TG / I <= 1.03 on pressure and on velocity and <= 1.05 on eddy
viscosity, seed means; beats it if TG / I < 1 - seed separation on a field. Also reported: TG / P (what the read
buys), step time and peak memory against I (forecast: 3x to 4x the step, memory within 80 GiB), pairs per block.
A step above 6x the reference or memory above 80 GiB is a cost failure even if the bars hold.

## E4. Query-population independence (the property this program exists for)

Instrument: the QUERY-POP probe (`qpop_swap.py`, copied) on the TG checkpoints: the same 100,000 scored points
under the native 200,000 tokens, the 100,000 alone, plus 200,000 far-field companions, plus 200,000 wall-heavy
companions. Prediction (as first written, 2026-10-06 morning, for the dual-tree form): every ratio to the control
is 1.000 exactly at theta = 0 and within 0.5% at the training theta (the admission was per target node, so
companions could move a point's far-field list). Amended 2026-10-06 evening, before any E3 or E4 run, when the
dual-tree form was replaced by the per-point walk: a point's pair list now depends on the boundary tree and the
point alone, so the prediction is 1.000 at every theta to float32 roundoff (ratios within 1e-4). The interacting
reference's 15x to 24x is the comparison.

## Decision

E1 and E2 gate the lanes. E3 and E4 together decide whether ISLA-DTT is proposed as the canonical volume form:
both bars hold, cost within the limit, and the boundary-law protocol (area draw) of the Volume task reported with
whatever built-in geometry is canonical at the time. Either bar failing is reported as such with the measured
mechanism; the program does not tune towards the bars.
