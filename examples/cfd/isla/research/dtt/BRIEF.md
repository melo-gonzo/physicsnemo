# ISLA-DTT: a query-independent volume ISLA with a tree-accelerated exact-in-the-limit boundary read

Branch `psharpe/isla-dtt` (worktree `/home/psharpe/gh/physicsnemo/.claude/worktrees/isla-dtt`), based on the pi
lane `psharpe/isla-qi-global` at 43f647e74, which carries the read-only interior model (`query_independent=True`),
the dense global boundary read (`kernel_read="mean", kernel_read_radius=None`; `kernel_read.py`,
`global_read_kernels.py`) and the canonical ISLA through SOFT-GEO (dc6b6b648). Lead: this session, taking over from
the pi agent (2026-10-06, Peter's decision). Canonical ISLA stays with the main session on `psharpe/isla`.
Compute: LHR task dir `agents/2026-10-06-isla-dtt`; job prefix `isla.dtt.`.

## Heilmeier catechism (written 2026-10-06, before any run)

**1. What are we trying to do?** Make ISLA's interior prediction at a point depend only on the boundary and that
point, so that the choice of other prediction points (how many, where, by what law) cannot change it, while keeping
the accuracy of the interacting model and the cost of a slice model at 200,000 boundary cells. The instrument is
the read-only model's direct read of every boundary cell, which the pi lane showed beats the interacting model on
AirfRANS, evaluated through a Barnes-Hut cluster tree so that its cost is near-linear rather than quadratic.

**2. How is it done today, and where does it break?** The interacting volume model (today's canonical class) lets
prediction points write into the slice states with a learned mass of 1.15 to 1.18 of the whole boundary measure, so
its error on fixed scored points rises 15x to 24x when 200,000 far-field companions join the query set and 3.5x
to 4.3x with wall-heavy ones (QUERY-POP, book notebook 2026-10-06). The read-only model is exactly independent but
4% to 42% less accurate (QI program, 2026-10-03). The read-only model with an exact global read of every cell (arm G
of QI-GLOBAL, four seeds, float32 protocol) is 0.72 / 0.83 / 0.89 of the interacting model's error on AirfRANS
pressure / velocity / eddy viscosity, exactly independent, but costs Q x N pairs per block: 1e8 on AirfRANS (2.1x
the step) and 4e10 on DrivAerML Volume (minutes per step, a 1.3 TB pair basis; months per lane).

**3. What is new, and why will it work?** The same read, evaluated through a cluster tree over the boundary cells
(`physicsnemo.mesh.spatial.cluster_tree`, GLOBE's Barnes-Hut layer, reused for the tree and its aggregates) that
every prediction point walks on its own, as a body walks the tree in an N-body code: the point reads nearby cells
one by one and distant clusters through their measure-weighted aggregates (centroid, mean normal, total measure,
mean key and value), a cluster admitted when its box diagonal is below theta times its gap to the point. The read
is a measure-weighted softmax over the boundary, so a cluster's aggregate is the monopole of the same quadrature,
and the error vanishes as theta goes to zero: theta is a numerical tolerance with a convergence guarantee, not a
model constant; the model is defined as the exact dense read and the tree is how it is computed, as in an N-body
code. Leaf size and pair-chunk sizes are implementation constants. Interaction counts scale like Q (k + log N),
which on DrivAerML Volume is the same order as the dense read's count on AirfRANS, so the step should land at a
few times the interacting reference rather than hundreds.

**4. Who cares?** The volume task then has the surface task's robustness to discretization on both sides: the
boundary sampling law (SOFT-GEO) and the query population (this program, exact by construction). A trained model
can be queried on any point cloud, one point or a million, in any order, and inference on a car is sub-second.

**5. Risks.** (a) The far-field aggregate is a monopole in the pair invariants and in the content term; the
kernel's dependence on the cell normal and the key spread inside a cluster do not shrink with distance, so the
accuracy at a usable theta may fall short of the exact read's (mitigations: orientation-binned aggregates, second
moments of the keys, smaller theta at higher cost). (b) The pair count at the theta the accuracy needs may be
larger than forecast at 200,000 by 200,000 (the first, dual-tree implementation was: 1.6e9 pairs at theta = 0.5,
out of memory, because a leaf of points spanning a large box in a sparse region is never well separated from the
boundary leaves near it; the per-point walk replaced it on 2026-10-06). (c) Query independence is exact at every
theta (a point's walk sees the tree and the point alone); equivariance is exact at theta = 0 and approximate
otherwise (Morton boxes are axis-aligned): accepted by Peter as the N-body standard, measured and reported, not a
bar. (d) The far-field pressure, where the dense read is already
1.19x the interacting model on AirfRANS, may not improve. (e) Training throughput of the first PyTorch
implementation may be several times the fused dense read's per pair; Triton segment kernels for the sparse read
already exist (`pair_kernels.py`) and carry the per-block cost; the pair basis is the item to fuse later.

**6. Cost.** About two weeks to a verified snapshot (tree read with a per-point walk of GLOBE's tree, sparse basis,
per-block aggregates, contract and convergence tests), then lanes: AirfRANS evaluation-only on the four G checkpoints
(minutes), DrivAerML Volume four lanes of one to two days each on LHR (two seeds of the tree read, two of
read-only), with the two existing interacting references as the comparison.

**7. Exams.** Written in `PREREG.md`: E1 exactness (theta = 0 reproduces the dense read), E2 convergence and
accuracy on the trained G checkpoints as a function of theta, E3 DrivAerML Volume accuracy against the interacting
reference with the pi lane's bars, E4 query-population independence under the QUERY-POP probe, with cost and
memory reported beside each.

## What the main session asked (2026-10-06)

Job names under `isla.dtt.`; copy, do not run in place, `kernel/char/qpop/qpop_swap.py` and
`kernel/char/af_geo/af_geo_swap.py` from the AGA task dir `2026-08-09-mt2-stage0`; tell it before anything lands in
canonical; it tells us before any change to `physicsnemo/experimental/models/isla`.

## Standard set by Peter (2026-10-06)

Exact Euclidean equivariance and exact query independence are required in the limit (theta = 0, leaf size 1);
at finite theta small approximation errors are accepted, as a Barnes-Hut gravity code is equivariant in spirit.
Magic constants kept to a bare minimum: theta is the one numerical tolerance; no derived radii, no per-dataset
knobs. No approach that buys accuracy or speed by giving up principled correctness.
