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
theta 0.5, two seeds each, scored at their training theta. The bars against I and P are unchanged. The
orientation-aware tree (`boundary_tree.py`, RMS normal spread) is the tree of every lane. Second amendment, the
same evening, before any TG lane ran: theta 0.5 (6.6e8 pairs per block) does not fit one GB300 and would be about
three times the time of theta 1, outside the bar, so only theta 1 runs (two seeds); and the "theta 0 diagnostic" is
withdrawn at car scale, where theta 0 is the 4e10-pair dense read this program exists to avoid. Third amendment,
2026-10-09, after E3 and E5 closed: the dedicated tree is replaced by a forest of GLOBE `ClusterTree`s (one per
normal octant, box-diagonal admission plus the same RMS coherence test; PR #2079 adds the per-point walk upstream).
The clusters differ, so the theta 1 arms are retrained on the car and on AirfRANS (`dtt_vol_tg1ct_seed42/43`,
`af_tg1ct_seed42/43`, snapshot `code_dtt13`) and scored under the same protocol; the prediction is parity with the
`code_dtt11`/`code_dtt12` theta 1 arms (seed means within the seed separation on every field), since the forest's
pair count is within 10% of the retired tree's at theta 1 and the read is the same quadrature. The bars against I
and P are unchanged. Fourth amendment, 2026-10-10, after the AirfRANS parity pair scored: native and area were at
parity but the posed protocol was not (1.9x / 1.25x / 3.3x on p / v / nut; REPORT.md), because trees on lab
coordinates cluster a rotated case differently from the training cases. The forest is now built in the
boundary's own frame (principal axes signed by third moments, `canonical_frame`), which is exact in every pose
up to ties, and the theta 1 parity arms are relaunched once more (`dtt_vol_tg1pf_seed42/43`, `af_tg1pf_seed42/43`,
snapshot `code_dtt16`). Predictions: native and area at parity with the retired tree's arms within the seed
separation on every field, and posed within 1% of native on every field (the retired tree reproduced native to
1e-4; the frame's ties can move a few clusters). The `code_dtt13` arms stand as the record of the lab-frame form.

## E6: latent capacity against query consensus (added 2026-10-09, before any arm ran)

Question. On the car the read-only class trails the interacting model (P / I = 1.14 / 1.31 / 1.68; the exact
global read TG / I = 1.11 / 1.11 / 1.47) while on the airfoil it does not. SMART (Hagnberger and Niepert, arXiv
2601.18707) is a read-only model by construction (queries cross-attend into a latent of 2,048 to 4,096 boundary
points refined block by block, no attention among queries) and its error falls monotonically with the latent
size; its decoder reads the encoder's latent at matched depth, and reading the final latent instead raises its
error by 140%. Two explanations of the car gap remain: the capacity of what a passive point can read (the
latent), or the consensus that interacting query tokens reach among themselves and a passive point cannot.
Arms, every one read-only (``query_independent=True``), two seeds each, the Volume recipe on the car (512 / 512,
lr 3.75e-4, 200,000 cells and points, 500 epochs) and the AirfRANS recipe (384 / 256, lr 5e-4, 100,000 cells,
500 epochs), scored under the protocols of E3 and E5:

- K-ladder: the exact global read over a Morton-strided subsample of ``kernel_read_cells`` boundary cells
  (measure fractions renormalized over the subsample; the stride offset random in training, zero in
  evaluation; ``kernel_read_theta=0`` so the read is exact over the subsample). K = 256 and K = 1,024 on the car
  and on the airfoil. The ends of the ladder exist: P (K = 0) and TG (every cell, theta 1).
- S-arm: the slice latent doubled, read-only P at ``n_slices`` 1,024 (car) and 512 (airfoil), and the interacting
  model I at the same slice counts as the matched control (the fused geometry kernel caps slices at 1,024).
- FINAL control: read-only P with ``read_final_boundary=True``: the boundary tokens run through every block
  first, then every query block reads the slices of the final boundary state instead of the state at its own
  depth (the same parameters; SMART's depth-matched ablation).

Predictions, stated before any arm ran. (a) Capacity: the K-ladder error falls monotonically from P through
K = 256 and 1,024 to TG, and P gains more than I from the doubled slices (P-S / P below I-S / I by more than the
seed separation). (b) Consensus: the ladder is flat within the seed separation above K = 256 (K = 256 within 3% of
TG on every field), the doubled slices move P and I alike, and TG / I stays the read-only ceiling. (c) FINAL:
if the control's error exceeds P's by more than 30% on any field, depth-matched reading is load-bearing in ISLA
as in SMART and the architecture chapter states it; within the seed separation, it is not. Cost is reported but is
not a bar: these are diagnostic arms, never candidates for the reference.

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

## E5. The residual's mechanism: a theta ladder where the dense read is affordable (added 2026-10-08 00:10 PDT, after E3)

E3 failed (TG / I = 1.11 / 1.11 / 1.47 at theta 1 on the car) and E4 passed. Two mechanisms can carry the
residual: the theta 1 aggregation (the trained kernel wants a finer far field than the car can pay for) or the
absence of query-to-query interaction (structural to the property). Instrument: AirfRANS (100,000 points, 1,000
cells), one snapshot (`code_dtt12`), the reference AirfRANS configuration (384 / 256, lr 5e-4, 500 epochs, full
splits), two seeds (42, 43) of six arms: I interacting; P read-only; G read-only + dense exact global read; TG at
theta 1, 0.5, 0.25. Scored by the float32 protocol on the 200 held-out airfoils (native, area, posed).

Predictions, written before any lane ran. (a) If the aggregation is not the limiter: TG(theta 1) / G within 1.02
on every field, and the ladder flat in theta; then the car's residual is structural and ISLA-DTT's accuracy is what
a read-only model can reach. (b) If the aggregation is the limiter: TG(theta 1) / G above 1.05 on at least one
field and the ratio falling monotonically towards 1 as theta falls from 1 to 0.25 (convergence of the trained
model towards its dense limit); then the avenue is cost-limited on GB300 hardware, and the car rung at theta 0.6 to
0.8 (if the cost probe allows it, step within 6x the floor) measures how much the affordable refinement buys. A
mixed outcome (short of G, not converging) means the trained tree model is a different optimum, to be reported as
such. Also recorded: G / I and P / I on this snapshot (the pi lane's 0.72 / 0.83 / 0.89 and 1.04 to 1.42 were on
its own snapshot; same-snapshot pairs only).

## Decision

E1 and E2 gate the lanes. E3 and E4 together decide whether ISLA-DTT is proposed as the canonical volume form:
both bars hold, cost within the limit, and the boundary-law protocol (area draw) of the Volume task reported with
whatever built-in geometry is canonical at the time. Either bar failing is reported as such with the measured
mechanism; the program does not tune towards the bars.
