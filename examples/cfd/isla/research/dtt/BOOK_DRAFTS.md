# Book drafts: the read-only interior model and ISLA-DTT

Drafted 2026-10-11 for ISLA Main to place in the book. Prose and tables only; no chapter is edited. Sources:
`research/dtt/REPORT.md` (the status of 2026-10-11 02:30 PDT and the dated sections below it), the class on this
branch (`physicsnemo/experimental/models/isla/{model,kernel_read,tree_read}.py` and their tests), and the book's
own artifacts where a placement names one (`results/char/drivaerml_volume.json`, `results/char/qpop_2026-10-06.json`,
`results/char/char_lanes_cost_2026-10-04.json`, and, in the main checkout only, the AeroZoo reduction).

Conventions used throughout:

- p / v / nut are the interior pressure, velocity and eddy-viscosity relative L2, seed means of seeds 42 and 43.
- A ratio is the row's error ÷ ISLA's (the interacting reference), so above one means less accurate than ISLA,
  unless a column names another base. Ratios follow the book's `fmt_ratio` (two decimals below 2, one to 10, none
  above).
- Where REPORT states a ratio, the draft uses it; every such ratio was recomputed from REPORT's four-digit per-seed
  values and agrees within 0.01. Where REPORT disagrees with an artifact, the artifact wins (listed below).
- The DTT scores live on LHR (`scoring/protocol/<run>/<protocol>/metrics.jsonl`), not in `results/`, so the tables
  are static and cite REPORT. To render them from an artifact as STYLE asks, copy the rows into
  `results/char/` in the char schema (arms keyed "ISLA", "ISLA read-only, no read", "ISLA-DTT", protocols
  native / area / posed, `per_seed` and `n_cases`) and the existing `protocol_table()` helper will print them.

Corrections to the numbers in the brief, from the artifacts:

- ISLA's uniform-point-cloud factors on the DrivAerML Volume task are 7.9x / 6.6x / 6.6x (results/char/drivaerml_volume.json:
  0.311 / 0.417 / 0.583 over 0.0394 / 0.0635 / 0.0876), not 7.9x / 6.9x / 7.7x as in REPORT.
- ISLA's far-field-companion factors are 24x / 15x / 12x (results/char/qpop_2026-10-06.json: 23.7 / 15.0 / 12.3), so
  "15x to 24x" omits eddy viscosity's 12x.
- ISLA-DTT ÷ ISLA on AirfRANS eddy viscosity is 1.03 (0.0325 against 0.0314), not REPORT's "0%".
- The AeroZoo cost in the brief (1.17 s, 159 GB against 0.6 s, 129 GB) is the launch-time reading; the reduction
  now reads 1.24 to 1.28 s and 162 to 166 GB for ISLA-DTT against 0.68 s and 129 to 130 GB for ISLA. Placement 1
  reads it from the artifact.
- The per-step cost is quoted from one instrument (job 135330: 1.76 s and 194.5 GiB against the floor's 0.475 s and
  126 GiB on the same GB300). REPORT's "1.72 s and 194 GB per rank" has no job id and appears to be the training
  lanes' log, so it is not set against the benchmark's floor.

New anchors introduced (none exists in the book today): `sec-arch-global-read`, `sec-arch-tree-read`,
`tbl-arch-read-depth`, `sec-interior-read-only`, `tbl-interior-read-only`, `sec-interior-read-only-mechanism`,
`tbl-interior-read-ladder`, `tbl-interior-query-population`, `sec-airfrans-read-only`, `tbl-airfrans-read-only`.
Every other `@` reference below was checked to exist. The AeroZoo chapter's anchors are not cited because
`20-aerozoo.qmd` is outside the book order in `_quarto.yml`.

---

## Placement 1. `04-architecture.qmd`: the read-only task, its global read and the forest

Target: under `## The two tasks: surface and interior {#sec-arch-modes}`, REPLACES the subsection
`### Query independence {#sec-arch-passive-decoder}` (lines 766 to 777), whose first line is
"**A prediction from the interior task depends on which other points are in". The anchor `sec-arch-passive-decoder`
is kept because 02, 07, 16 and 19 cite it.

````qmd
### The read-only interior task {#sec-arch-passive-decoder}

**With `query_independent=True` the prediction points are passive: at every slice layer they read the slices the
boundary formed at that layer and never write into them, so the prediction at a point is a function of that point
and the boundary alone.** A passive point is seeded and embedded as an interacting one is (the built-in direction
and signed distance, the declared query inputs, the type offset) and runs through the same twelve slice layers with
the same parameters as the boundary tokens. Only its measure bias differs: $\ell_q = -\infty$ in place of the
learned routing mass, so every softmax over tokens gives it zero weight. It takes no part in the routing
prototypes, the anchors $z_s, m_s$ (step c), the routing weights $a_{is}$ (step f), the slice states $Z_s$ (step g)
or the head frame's anchors (stage 4). It still forms its own logits, its relational invariants against the anchors
and its read-back weights $\pi_{qs}$, and it reads $b_q = \sum_s \pi_{qs} Z_s$ (step h). The boundary tokens
therefore compute what they compute in the surface task whatever points are requested, and a request split into
chunks, permuted, or reduced to one point returns the same predictions to $10^{-12}$ in float64
(`test_query_independent_interior_contracts`). This mode has no routing-mass parameter. The interacting task is the
class default and the configuration of every interior standing in the book; its dependence on the request is
measured in @sec-interior-companion.

A passive point reads at matched depth: at layer $\ell$ it reads the slices formed from the boundary states at
layer $\ell$, the depth its own state has reached. That alignment carries much of the read-only model's accuracy. A
control with the same parameters, in which the boundary first runs through all twelve layers and every layer's
points then read slices formed from that final boundary state, has 1.37 to 1.40 times the matched-depth model's
error on the DrivAerML Volume task and 1.56 to 1.72 times on the AirfRANS interior task. Doubling the slice count
changes neither the read-only nor the interacting model beyond seed noise (@tbl-arch-read-depth).

::: {.callout-note collapse="true" title="Matched depth and slice count, measured"}
| model | DrivAerML Volume: p / v / nut | ÷ its base | AirfRANS: p / v / nut | ÷ its base |
|---|---|---|---|---|
| passive points, matched depth (base) | 0.0449 / 0.0834 / 0.147 | 1 | 0.0328 / 0.0180 / 0.0346 | 1 |
| passive points reading the final boundary state | 0.0615 / 0.1147 / 0.206 | 1.37 / 1.38 / 1.40 | 0.0522 / 0.0281 / 0.0596 | 1.59 / 1.56 / 1.72 |
| passive points, twice the slices | 0.0449 / 0.0829 / 0.152 | 1.00 / 0.99 / 1.04 | 0.0370 / 0.0181 / 0.0351 | 1.13 / 1.01 / 1.01 |
| interacting points (base) | 0.0394 / 0.0635 / 0.0876 | 1 | 0.0350 / 0.0170 / 0.0314 | 1 |
| interacting points, twice the slices | 0.0391 / 0.0626 / 0.0864 | 0.99 / 0.99 / 0.99 | 0.0351 / 0.0171 / 0.0313 | 1.00 / 1.00 / 1.00 |

: Matched-depth reading and slice count: relative L2 of interior pressure / velocity / eddy viscosity, float32,
seed means of two seeds (42, 43), 500 epochs, no boundary read in any row. DrivAerML Volume task: 435 training cars,
48 validation cars, 200,000 boundary cells and 200,000 interior points per car, ISLA width 512, 12 layers, 512
slices (1,024 when doubled), learning rate 3.75e-4. AirfRANS interior task: 800 training airfoils, 200 test
airfoils, 100,000 interior points and the whole airfoil polyline, ISLA at 384 wide, 12 layers, 256 slices (512
when doubled), 5e-4 (the earlier ISLA reference). How to read this: each ÷ compares a row with the base row of its
own kind (passive or interacting) on the same dataset, above one meaning less accurate; the second row changes
only which boundary state the points read, the third and fifth only the slice count. The 13% on AirfRANS pressure
in the third row is one seed (0.0392; the other, 0.0347, sits below the base's 0.0357), and the doubled
interacting car lanes ran on a later build of the code than their base pair. Data: research/dtt/REPORT.md,
"E6 interim on AirfRANS" and "E6 interim on the car". {#tbl-arch-read-depth}
:::

#### The global boundary read {#sec-arch-global-read}

**A passive point can also read the boundary cells themselves: with `kernel_read="mean"` and
`kernel_read_radius=None`, after every slice layer each prediction point adds a measure-weighted average of the
cells' states over the whole boundary, weighted by a learned kernel of where each cell sits relative to the point
and by how well their states match.** For a point $q$ and a cell $j$, with $\rho_{qj} \equiv r_q - r_j$, the
kernel sees the pair's invariants $\iota_{qj}$: the distance $\lvert\rho_{qj}\rvert$, its logarithm,
$\log(\lvert\rho_{qj}\rvert / \lvert\delta_q\rvert)$ (the distance in units of the point's own wall distance), the
cosines of $\hat\rho_{qj}$ with $\hat n_j$, $\hat n_q$ and every $\hat g_k$, and $\hat n_q \cdot \hat n_j$ and
$\hat n_j \cdot \hat g_k$ (declared per-token vectors add their own cosines). One MLP shared by every layer maps
$\iota_{qj}$ to 16 basis functions per pair, once per forward pass, and each layer's kernel $\kappa_h$ is a learned
combination of them for each of 8 heads. With $\omega_j \equiv w_j / \sum_i w_i$ the cell's share of the boundary
measure, $\tilde h \equiv \mathrm{LN}(h)$ (one norm for cells, one for points) and $c = 4$ key channels per head,
the read after a layer is

$$
\alpha_{qjh} = \operatorname*{softmax}_j \Big( \kappa_h(\iota_{qj})
  + \tfrac{1}{\sqrt{c}} \big\langle W^{\mathrm q}_h \tilde h_q,\ W^{\mathrm k}_h \tilde h_j \big\rangle
  + \log \omega_j \Big),
\qquad
h_q \leftarrow h_q + W^{\mathrm o} \Big[ \textstyle\sum_j \alpha_{qjh}\, W^{\mathrm v}_h \tilde h_j \Big]_{h=1}^{8} ,
$$

where the brackets concatenate the heads' 8 value channels each and $W^{\mathrm o}$ starts at zero, so training
begins from the model without the read. The log measure fraction makes the read a quadrature
(@sec-isla-quadrature): with the kernel and key terms constant it is the measure-weighted mean of the cells'
values, and splitting a cell into two coincident halves or rescaling every measure changes nothing
(`test_tree_read_similarity_refinement_and_measure_scale_at_theta_zero`). Every input is an invariant, so the read
keeps the contracts of @sec-isla-contracts, and it leaves the boundary tokens and the slices untouched, so it keeps
query independence (`test_global_read_is_exactly_query_independent`). Evaluated pair by pair it costs $Q \times N$
pairs per layer: $10^8$ for 100,000 points around a 1,000-segment airfoil, $4 \times 10^{10}$ for 200,000 points
around a 200,000-cell car.

#### The read through a cluster forest {#sec-arch-tree-read}

**With `kernel_read_theta` set to a number $\theta$, the same read runs through a Barnes-Hut forest: a point reads
nearby cells one by one and each distant cluster of similarly oriented cells as one virtual cell, which takes a
200,000-cell car from $4 \times 10^{10}$ pairs per layer to $1.6 \times 10^8$ at $\theta = 1$, keeps the prediction
exactly query independent and keeps it pose invariant up to ties of the boundary's own frame.** ISLA-DTT is ISLA
with passive prediction points, the global read after every layer and the forest at $\theta = 1$; it is the one
read-only configuration the book scores (@sec-interior-read-only, @sec-airfrans-read-only). The forest is built once
per case, without gradient:

1. The frame is the boundary's own: the principal axes of the cells' measure-weighted position covariance, each
   signed so that the third moment of the positions along it is positive. The forest is built in these
   coordinates, so it moves with the body.
2. The cells are split by the sign octant of their normal in that frame, and one cluster tree is built per octant
   (`physicsnemo.mesh.spatial.ClusterTree`, which bisects the boxes of a Morton space-filling curve down to leaves
   of 4 cells). Opposite normals never share a tree, so no cluster spans both faces of a thin body.
3. Every node carries a virtual cell: its cells' measure-weighted centroid, their renormalized measure-weighted
   mean normal, their mean rank-1 inputs and their total measure, and at every layer the measure-weighted means of
   their keys and values.
4. Each point walks every tree once. A node is read as one virtual cell when its box diagonal is below $\theta$
   times the distance from the point to the box and its normals are coherent, the length of their mean exceeding
   $1 - \theta^2/2$ (their measure-weighted RMS chord about the mean is below $\theta$). A leaf that fails either
   test is read cell by cell, and any other node hands the point to its two children, so every cell is read exactly
   once, alone or inside one virtual cell
   (`test_point_pairs_admit_only_separated_coherent_nodes_and_cover_every_cell`).

The softmax of the global read then runs over the point's list, a virtual cell entering with its total measure
fraction as its log bias and its invariants computed from its centroid and normal. Because the read is a
quadrature, a virtual cell is its cluster's monopole: were the logit constant over the cluster, reading the virtual
cell would equal reading its cells one by one. The softmax of each point runs in a fused Triton kernel that keeps
nothing per pair between the forward and backward passes.

Three properties hold by construction. At $\theta = 0$ no cluster is admitted and the read is the dense read pair
for pair, with the same parameters (`test_tree_read_at_theta_zero_is_the_dense_read`). At every $\theta$ a point's
list depends on the forest and the point alone, so the prediction is query independent
(`test_tree_read_is_exactly_query_independent_at_every_theta`). And a rotation, reflection or translation of the
case gives the same forest and the same lists (`test_forest_and_every_list_are_the_same_in_every_pose_at_finite_theta`),
up to ties of the boundary's frame: two equal principal variances, or an axis without a third moment, whose sign
then follows the laboratory axis it overlaps most. On both interior datasets ISLA-DTT's error on rotated and
translated cases is its native error to the fourth digit on pressure and velocity and within 0.3% on eddy
viscosity. $\theta$ is a constant of the trained model, like a kernel radius: ISLA-DTT trains and is scored at
$\theta = 1$, so its kernel, keys and values are learned as the forest groups the cells.

On one GB300 at 200,000 boundary cells and 200,000 prediction points (width 512, 512 slices, bfloat16, compiled
slice layers, one forward and backward pass), a point at $\theta = 1$ reads about 800 sources, 95% of them virtual
cells: $1.6 \times 10^8$ pairs per layer, 250 times fewer than the dense read. The step takes 1.76 s and 194.5 GiB
against 0.475 s and 126 GiB for the same model without the boundary read, 3.7 times the step. At $\theta = 0.5$ a
point's list grows to about 6,900 sources, more than one GB300 holds at this size. The read adds about 1.2 million
parameters to ISLA's 69 million.

```{python}
#| label: para-arch-tree-cost-aerozoo
#| output: asis
import json
from pathlib import Path
_AZ = json.load(open(Path("..") / "results" / "campaigns_2026-10-09" / "aerozoo" / "aerozoo_reduction.json"))["cost"]
def _az_cost(arm):
    rows = [c for c in _AZ[arm]["full"].values() if c.get("step_seconds_median")]
    return (sum(c["step_seconds_median"] for c in rows) / len(rows), max(c["mem_gb_max"] for c in rows),
            rows[0]["n_parameters"] / 1e6)
(_ts, _tm, _tp), (_rs, _rm, _rp) = _az_cost("ISLA-DTT"), _az_cost("ISLA")
print(f"In training on AeroZoo (four GB300 ranks, bfloat16, 200,000 interior points and up to 50,000 faces per boundary "
      f"type per case, the final epoch's median step), ISLA-DTT steps in {_ts:.2f} s with {_tm:.0f} GB on its busiest rank "
      f"against ISLA's {_rs:.2f} s and {_rm:.0f} GB, {_ts / _rs:.1f} times the step, at {_tp:.1f} million parameters "
      f"against {_rp:.1f} million. Data: results/campaigns_2026-10-09/aerozoo/aerozoo_reduction.json (cost).")
```

::: {.callout-note collapse="true" title="The read-only task and the forest in the code"}
The passive points' measure bias, the read after each layer, and one point's walk of the forest.

```{python}
#| label: code-read-only
#| output: asis
listing("model.py", "ISLA._forward_tensors", start="if self.query_independent:", end="elif n_support")
listing("model.py", "ISLA._forward_tensors", start="first_read = len(self.blocks)", end="### The head frame builds")
listing("tree_read.py", "BoundaryForest.point_pairs")
```
:::
````

Notes for placement 1:

- The AeroZoo cell is optional. It reads the reduction JSON that exists in the main checkout but is not committed
  on this branch; drop it if the AeroZoo chapter's own cost paragraph is enough.
- The cost paragraph could instead be one row of the cost table in `06-using-isla.qmd` (`@sec-using-cost`), which
  STYLE calls the book's one cost-per-step table. The instrument differs from that table's (compiled slice layers,
  Volume task, 512 wide), so it would need its own row label.
- The `listing` markers were checked against `model.py` and `tree_read.py` on this branch.

Provenance: the class (`model.py` `ISLA` docstring and `_forward_tensors`; `kernel_read.py` `_GlobalKernelRead`,
`global_pair_invariants`; `tree_read.py` `BoundaryForest.build`, `point_pairs`, `canonical_frame`) and its tests
(`test_isla.py`, `test_isla_global_read.py`, `test_isla_tree_read.py`). Measurements: REPORT "E6 interim on AirfRANS"
(job 141408) and "E6 interim on the car" (job 142093), lanes `dtt_vol_{final,ps1024,is1024}_seed42/43` and
`af_{final,ps512,is512}_seed42/43` on `code_dtt14`; base rows from "E3 setup" (P, `dtt_vol_p_seed42/43`, job 113455;
I, `cl_r_vol_isla_seed42/43`, results/char/drivaerml_volume.json) and "E5 interim" (AirfRANS I and P); training at
the forest's $\theta$ from "Decision (2026-10-06 evening): train with the tree"; posed equal to native from
"Pose-free parity on AirfRANS" (job 146316) and "Pose-free parity on the car" (job 152583); cost from "Verification
and cost of the forest" (job 135330, `stats/bench13`) and "The read on ClusterTree" (6,900 sources at $\theta = 0.5$);
parameters from the AeroZoo reduction (70,311,186 against 69,098,627) and results/char/char_lanes_cost_2026-10-04.json
(ISLA on the Volume task, 69,078,633).

---

## Placement 2. `04-architecture.qmd`: one bullet of "What the pass never does"

Target: `## What the pass never does {#sec-arch-never}`, REPLACES the second bullet, whose first line is
"- **No positional encoding, no pairwise attention.** Tokens communicate". The bold lead is kept only to match the
sibling bullet of that list.

````qmd
- **No positional encoding, and no prediction point attends to another.** Tokens exchange information through
  the $M$ slice states, and the routing that decides which slice a token joins is a function of the token's state
  and its relation to the anchors. The one pairwise read in the class is the global boundary read of the read-only
  task (@sec-arch-global-read): a passive prediction point attends to boundary cells, and nothing flows from it to
  any token.
````

Provenance: `model.py` `_forward_tensors` (the read updates the points' states only; the boundary tokens and slices
are untouched).

---

## Placement 3. `04-architecture.qmd`: the switches table

Target: `## The switches {#sec-arch-switches}`. EDITS the lead sentence and ADDS three rows after the `interior` row.
The lead currently reads "...and a small set of arguments: the sizes, the declarations, the task, three cost knobs,
the routing-prototype choice and the spatial dimension."

````qmd
...and a small set of arguments: the sizes, the declarations, the task and the role of its prediction points, the
read-only task's boundary read, three cost knobs, the routing-prototype choice and the spatial dimension.

| `query_independent` (`False`) | modes | interior task only: `True` makes the prediction points passive (@sec-arch-passive-decoder), with no routing weight and no routing-mass parameter, so each prediction depends on its own point and the boundary alone |
| `kernel_read` (`None`), `kernel_read_radius` (0.01), `kernel_read_blocks` (`None`) | after each slice layer | read-only interior task only: `kernel_read="mean"` with `kernel_read_radius=None` adds the global boundary read after each of the last `kernel_read_blocks` layers, every layer when `None` (@sec-arch-global-read). A finite radius limits the read to the cells within that distance in units of $L$, and `"sum"` replaces the softmax by an unnormalized quadrature; neither is part of any configuration the book scores |
| `kernel_read_theta` (`None`) | after each slice layer | global read only: `None` evaluates it pair by pair, $Q \times N$ pairs per layer; a number $\theta$ evaluates it through the cluster forest at that opening angle (@sec-arch-tree-read), the dense read at 0. ISLA-DTT sets 1 and is trained at it |
````

Note: the radius-limited and `"sum"` reads exist in the class with no measurement in the chapters. Under the
standing rule that options are scaffolds (universalize or delete), ISLA Main or Peter may prefer to delete them
from the class rather than document them here.

Provenance: `model.py` `ISLA.__init__` signature and docstring on this branch.

---

## Placement 4. `07-drivaerml.qmd`: the read-only model on the Volume task

Target: under `## Ladders and controls {#sec-drivaerml-ladders}`, NEW subsection after
`### The interior rungs {#sec-interior-rungs}` and before `### Dependence on the query cloud {#sec-interior-companion}`.
Also ADDS one row to the table under `#### How each architecture treats an interior point` (given at the end of this
placement).

````qmd
### The read-only interior model {#sec-interior-read-only}

**On the DrivAerML Volume task ISLA-DTT, the query-independent ISLA, has 1.12 / 1.11 / 1.12 times ISLA's error on
pressure / velocity / eddy viscosity at the same configuration, and its error rises 1.3 to 3.1 times under the
uniform point cloud where ISLA's rises 6.6 to 7.9 times.** ISLA-DTT is ISLA with passive prediction points, which
read the boundary's slices at every layer and never write into them, so that a prediction depends on its own point
and the boundary alone (@sec-arch-passive-decoder), and with a global read of every boundary cell evaluated through
a cluster forest at $\theta = 1$ (@sec-arch-tree-read). Everything else is held equal: the reference configuration
(width 512, 12 layers, 512 slices, learning rate 3.75e-4), the recipe and inputs of @sec-interior-query-tokens, 435
training cars, 200,000 boundary cells and 200,000 interior points per car in training and scoring, 500 epochs, two
seeds, float32 on the 48 validation cars. ISLA-DTT carries the read's 1.2 million parameters on top of ISLA's 69
million. The middle row keeps the passive points and removes the boundary read.

| configuration | native mesh: p / v / nut | ÷ ISLA | uniform point cloud (÷ own native) | rotated and translated |
|---|---|---|---|---|
| ISLA (interacting prediction points) | 0.0394 / 0.0635 / 0.0876 | 1 | 0.311 / 0.417 / 0.583 (7.9x / 6.6x / 6.6x) | equal to native |
| ISLA, passive points, no boundary read | 0.0449 / 0.0834 / 0.147 | 1.14 / 1.31 / 1.68 | 0.0860 / 0.204 / 0.161 (1.9x / 2.5x / 1.10) | equal to native |
| ISLA-DTT | 0.0443 / 0.0707 / 0.0982 | 1.12 / 1.11 / 1.12 | 0.101 / 0.218 / 0.130 (2.3x / 3.1x / 1.32) | equal to native to the fourth digit |

: The DrivAerML Volume task, read-only controls at the reference configuration: interior pressure / velocity / eddy
viscosity relative L2, float32, 48 validation cars, seed means. Seeds 42 and 43: ISLA 0.0387 / 0.0619 / 0.0868 and
0.0400 / 0.0651 / 0.0884; passive points without a read 0.0454 / 0.0847 / 0.1745 and 0.0443 / 0.0822 / 0.1188;
ISLA-DTT 0.0446 / 0.0728 / 0.0997 and 0.0439 / 0.0685 / 0.0967, every ISLA-DTT seed above every ISLA seed on every
field. How to read this: ÷ ISLA above one means less accurate than ISLA; the uniform point cloud redraws the
boundary sample area-proportionally at inference (@sec-interior-results) and its factor is each row's own. Data:
results/char/drivaerml_volume.json (ISLA); research/dtt/REPORT.md, "E3 setup" (passive points) and "Pose-free
parity on the car" (ISLA-DTT). {#tbl-interior-read-only}

Without a boundary read the passive model trails ISLA by 14% on pressure, 31% on velocity and 68% on eddy
viscosity. The forest read takes back almost none of the pressure deficit and most of the other two: ISLA-DTT's
error is 0.99 / 0.85 / 0.67 of that model's.

#### What the boundary read supplies {#sec-interior-read-only-mechanism}

**Velocity improves with how much of the boundary a point reads, eddy viscosity mostly with the first few hundred
cells, and pressure with no read at all.** The instrument reads the boundary exactly but only part of it: each point
reads $K$ cells taken at an even stride along a Morton space-filling curve over the boundary, their measure fractions
renormalized over the subsample and the stride's offset random in training. It is a diagnostic form of the read,
not an option of the class; everything else is the protocol of @tbl-interior-read-only.

| boundary cells read per point | p | v | nut | v ÷ no read |
|---|---|---|---|---|
| none (passive points, no boundary read) | 0.0449 | 0.0834 | 0.147 | 1.00 |
| 256, exact | 0.0454 | 0.0839 | 0.105 | 1.01 |
| 1,024, exact | 0.0456 | 0.0790 | 0.101 | 0.95 |
| all 200,000, forest at $\theta = 1$ (ISLA-DTT) | 0.0443 | 0.0707 | 0.0982 | 0.85 |
| ISLA (interacting prediction points), for scale | 0.0394 | 0.0635 | 0.0876 | 0.76 |

: How much of the boundary a passive point reads, on the DrivAerML Volume task: interior relative L2, float32, 48
validation cars, reference configuration, two seeds (256 cells: 0.0458 / 0.0849 / 0.1042 and 0.0450 / 0.0829 /
0.1061; 1,024 cells: 0.0463 / 0.0807 / 0.1023 and 0.0449 / 0.0772 / 0.0997). How to read this: the first four rows
change only how many cells each point reads; the last row is the interacting model, not a rung. Data:
research/dtt/REPORT.md, "First rung on the car", "Last rung, the car at 1,024 cells" and "Pose-free parity on the
car". {#tbl-interior-read-ladder}

Velocity does not move at 256 cells, falls 5% at 1,024 and 15% with every cell. Pressure stays within 2% of the
no-read model at every rung. Eddy viscosity falls from 0.147 to 0.105 with 256 cells and to 0.098 with every cell.
More slices do not substitute for the read: doubling them moves neither the passive nor the interacting model
(@tbl-arch-read-depth). The 12% that separates ISLA-DTT from ISLA on pressure is therefore what interacting
prediction points supply to one another through the slices, which a query-independent model gives up by
construction. The 11% on velocity is that or a finer read than the car affords: at $\theta = 0.5$ a point's list
grows to about 6,900 sources, more than one GB300 holds at this size (@sec-arch-tree-read). On the airfoils, where
the exact read of every cell is affordable, the read-only model does not trail ISLA at all
(@sec-airfrans-read-only).
````

Row to ADD to the table under `#### How each architecture treats an interior point` (after the "ISLA with query
tokens" row):

````qmd
| ISLA-DTT (a control, @sec-interior-read-only) | a passive reader: it reads the boundary's slices at every layer without writing into them, and reads the boundary cells directly through the forest of @sec-arch-tree-read | as ISLA with query tokens, plus the relative geometry of every boundary cell or cluster it reads | no |
````

Optional sentence for ISLA Main, after the paragraph under the first table, if the chapter wants the
cross-architecture reading (numbers from results/char/drivaerml_volume.json and char_lanes_cost_2026-10-04.json):

````qmd
On the same 48 cars GeoTransolver width 256 reads 0.0487 / 0.1002 / 0.0838 (@tbl-drivaerml-volume-protocols), so
GeoTransolver ÷ ISLA-DTT is 1.10 / 1.42 / 0.85, at 28 million parameters against 70 million and with the inputs of
@sec-interior-task.
````

A figure this placement could take (not drafted): error ÷ the no-read model against boundary cells read per point
(0, 256, 1,024, 200,000; log axis with the zero rung set apart), one panel per field, ISLA-DTT in `BLUE_LIGHT` open
square dash-dot and ISLA's level as a reference line, fed by @tbl-interior-read-ladder's numbers or by LHR
`scoring/protocol/dtt_vol_{p,k256,k1024,tg1pf}_seed{42,43}/native/metrics.jsonl`.

Provenance: REPORT "E3 setup" (I native / area / posed; P, job 113455), "Pose-free parity on the car (2026-10-11
09:10 UTC, job 152583)" (ISLA-DTT, `dtt_vol_tg1pf_seed42/43`, `code_dtt16`; area seed mean), "First rung on the car"
(job 143308, `dtt_vol_k256_seed42/43`), "Last rung, the car at 1,024 cells" (job 148783, `dtt_vol_k1024_seed42/43`),
"E6 interim on the car" (job 142093, slice doubling), "The read on ClusterTree" (6,900 sources at $\theta = 0.5$).
ISLA's row and the GeoTransolver sentence: results/char/drivaerml_volume.json.

---

## Placement 5. `07-drivaerml.qmd`: dependence on the query cloud

Target: REPLACES the body of `### Dependence on the query cloud {#sec-interior-companion}` (lines 779 to 800), whose
first line is "**The interacting configuration's answer at a point depends on how many". The heading and anchor stay
(cited from 04, 06, 16 and 17).

````qmd
### Dependence on the query cloud {#sec-interior-companion}

**ISLA's answer at a point depends on the population of points requested with it; the read-only ISLA's does not,
to float32 roundoff.** The instrument holds the scored points fixed and changes the request around them. On each of
the 48 validation cars of the DrivAerML Volume task, the first 100,000 of the 200,000 native interior points are
scored in float32 under four requests: the native 200,000 (the control); the 100,000 alone; the native 200,000 plus
200,000 companions drawn uniformly in the interior's bounding box, a far-field-heavy population; and the native
200,000 plus 200,000 companions resampled from the native points with probability proportional to the inverse wall
distance, a wall-heavy one. The boundary sample is the same in every request, so every scored point keeps its own
built-in geometry, and only the other prediction points change.

| request around the same 100,000 scored points | ISLA: error ÷ control, p / v / nut | read-only ISLA: error ÷ control, p / v / nut |
|---|---|---|
| the 100,000 alone | 1.02 / 1.02 / 1.02 | 1.0000 / 1.0000 / 0.9999 to 1.0000 |
| native + 200,000 far-field companions | 24x / 15x / 12x | 1.0000 / 1.0000 / 0.9999 to 1.0000 |
| native + 200,000 wall-heavy companions | 4.3x / 4.3x / 3.5x | 1.0000 / 1.0000 / 0.9999 to 1.0000 |

: Query-population dependence on the DrivAerML Volume task: the error on fixed scored points under each request
divided by the same points' error in the native request, 48 validation cars, float32, seed means of the two ISLA
reference lanes (width 512, 512 slices) and of four read-only checkpoints at the same configuration (passive
prediction points, two without a boundary read and two with the tree read at $\theta = 1$). The largest change of
any read-only prediction over every car and request is 1.2e-4 relative L2. How to read this: a ratio of one means
the other points of the request do not matter; it says nothing about accuracy, which is @tbl-interior-read-only.
Data: results/char/qpop_2026-10-06.json (ISLA); research/dtt/REPORT.md, "E4 on the read-only reference" and "E4
verdict" (read-only). {#tbl-interior-query-population}

ISLA's 200,000 interacting points carry a learned share of the routing weight equal to 1.18 and 1.15 times the
whole boundary's measure (the two seeds), so the request weighs more than the boundary in the slice states, and a
population drawn by another law moves them. Halving the request at the training law costs 2%. ISLA is therefore
deployed at its training count and draw, and every interior number of the book states them
(@sec-using-checkpoint); training over a range of request sizes reduces the count dependence without removing it
(at the exploration budget of 10,000 cells, results/qcount_reduction_2026-09-11.json). The read-only ISLA moves by
float32 roundoff through the slice layers, as its construction requires (@sec-arch-passive-decoder), so ISLA-DTT can
be queried at any set of points, alone or in chunks; its accuracy on this task is @tbl-interior-read-only.
````

Provenance: results/char/qpop_2026-10-06.json (`mean_ratio_to_native_h`, `query_routing_mass_fraction_of_boundary`;
notebook @sec-nb-query-pop, job 1087747); REPORT "E4 on the read-only reference (job 113456)" (P,
`dtt_vol_p_seed42/43`, largest change 1.2e-4) and "E4 verdict (job 125332)" (the tree-read lanes
`dtt_vol_tg1_seed42/43`, largest change 1e-4). Those tree-read checkpoints use the earlier tree; the forest of
`code_dtt16` has the same per-point construction and its query independence is the unit test
`test_tree_read_is_exactly_query_independent_at_every_theta`, but no checkpoint of the final form has been through
this probe on DrivAerML. If the chapter must cite a measurement of ISLA-DTT itself, run the E4 probe on
`dtt_vol_tg1pf_seed42/43`, or cite the AeroZoo query-population protocol once it is scored. The 10,000-cell
sentence keeps the one result of the replaced text that 17's ledger row on query-count augmentation relies on.

---

## Placement 6. `09-airfrans.qmd`: the read-only model on the airfoils

Target: under `## Ladders and controls {#sec-airfrans-ladders}`, NEW subsection after
`### The training-airfoil ladder {#sec-airfrans-ladder}` and before
`### Outside the training range: the AirfRANS extrapolation splits {#sec-interior-extrapolation}`.

````qmd
### The read-only interior model {#sec-airfrans-read-only}

**On the AirfRANS interior task the read-only ISLA gives up nothing for its query independence: ISLA-DTT is 26% more
accurate than ISLA on pressure and 17% on velocity, level on eddy viscosity, and as accurate as the exact read of
every boundary cell.** ISLA-DTT is ISLA with passive prediction points, which read the boundary's slices at every
layer and never write into them, so that a prediction depends on its own point and the boundary alone
(@sec-arch-passive-decoder), and with a global read of every boundary cell evaluated through a cluster forest at
$\theta = 1$ (@sec-arch-tree-read). The four rows below share one configuration and protocol and differ only in the
prediction points' role and the boundary read; the dense read evaluates the same global read pair by pair, $10^8$
pairs per layer.

| configuration | p (seeds) | v (seeds) | nut (seeds) | ÷ ISLA: p / v / nut |
|---|---|---|---|---|
| ISLA (interacting prediction points) | 0.0350 (0.0332, 0.0368) | 0.0170 (0.0169, 0.0170) | 0.0314 (0.0326, 0.0301) | 1 |
| ISLA, passive points, no boundary read | 0.0328 (0.0299, 0.0357) | 0.0180 (0.0178, 0.0181) | 0.0346 (0.0325, 0.0366) | 0.94 / 1.06 / 1.10 |
| ISLA, passive points, dense global read | 0.0260 (0.0274, 0.0245) | 0.0135 (0.0137, 0.0133) | 0.0288 (0.0325, 0.0250) | 0.74 / 0.79 / 0.92 |
| ISLA-DTT (forest read at $\theta = 1$) | 0.0259 (0.0255, 0.0263) | 0.0141 (0.0135, 0.0146) | 0.0325 (0.0314, 0.0335) | 0.74 / 0.83 / 1.03 |

: The AirfRANS interior task, read-only controls: relative L2 per field on the 200 test airfoils of the full split,
float32, seed mean with seeds 42 and 43 in brackets. Every row is ISLA at 384 wide, 12 layers, 256 slices, 5e-4
(the earlier ISLA reference), trained on the 800 training airfoils for 500 epochs with 100,000 interior points and
the whole airfoil polyline (958 to 1,018 segments) as the boundary. Rotated and translated, every row's error is its
native error (ISLA-DTT to the fourth digit on pressure and velocity and within 0.3% on eddy viscosity), and the
uniform point cloud is the identity on this dataset (@sec-airfrans-measured). How to read this: ÷ ISLA above one
means less accurate than ISLA; these rows are a within-configuration control, and the standing of the reference
configurations is @tbl-airfrans-protocols. Data: research/dtt/REPORT.md, "E5 interim" (first three rows) and
"Pose-free parity on AirfRANS" (ISLA-DTT). {#tbl-airfrans-read-only}

The passive model without a boundary read is already level with ISLA here (0.94 / 1.06 / 1.10), where on the cars
it trails by 14% to 68% (@tbl-interior-read-only): the airfoil's interior field is recoverable from the boundary's
slices alone. The global read then removes a sixth to a quarter of the remaining error on every field, and the
forest keeps that gain: ISLA-DTT ÷ the dense read is 1.00 on pressure and 1.04 on velocity, and its eddy viscosity
lies inside the dense read's own seed range (0.0250 to 0.0325). The order of the cars' partial reads
(@sec-interior-read-only-mechanism) holds here with the rungs compressed: an exact read of 256 of the airfoil's
cells, taken at an even stride along a Morton curve, takes three fifths of the full read's velocity gain and all of
its eddy-viscosity gain (velocity 0.0153 against 0.0180 without a read and 0.0135 with every cell; eddy viscosity
0.0277), though at that stride the model trains on only four distinct subsamples.
````

Provenance: REPORT "E5 interim (2026-10-08 13:40 PDT)" (I, P and the dense read G on `code_dtt12`), "Pose-free parity
on AirfRANS (2026-10-10 18:30 UTC, job 146316)" (ISLA-DTT, `af_tg1pf_seed42/43`, `code_dtt16`; posed seed 42
0.0255 / 0.0135 / 0.0313, seed 43 equal to native), "First rung on the airfoil" (job 143086, `af_k256_seed42/43`,
stride 4). The configuration is PREREG E5 and E6 ("the AirfRANS recipe (384 / 256, lr 5e-4, 100,000 cells, 500
epochs)"); the boundary segment count is from REPORT "Lanes of the last two days".

---

## Passages in the current book that contradict these drafts

Quoted from this branch, which matches the main checkout for 04, 05, 06, 07, 09 and 16.

1. `04-architecture.qmd` 766 to 777 (replaced by placement 1): "The class offers no request-independent decode;
   what one would cost is the ledger row of @sec-principled-decisions." The class has `query_independent`, and 17
   has no such ledger row.
2. `04-architecture.qmd` 791 to 794 (replaced by placement 2): "**No positional encoding, no pairwise attention.**
   Tokens communicate only through the $M$ slice states". The global read is a pairwise read.
3. `04-architecture.qmd` 798 to 801 and the switches table (placement 3): the argument list omits
   `query_independent` and the `kernel_read*` arguments.
4. `07-drivaerml.qmd` 797 to 800 (replaced by placement 5): "A read-only decoder whose predictions are exactly
   query-independent is several times less accurate on every interior field, which is why the class offers none".
   Measured at the reporting protocol, the read-only model is 1.14 to 1.68 times ISLA's error on the cars and level
   on the airfoils, and ISLA-DTT 1.11 to 1.12 and 0.74 to 1.03.
5. `07-drivaerml.qmd` 669 to 674: "admitting the queries as interacting tokens, rather than decoding them read-only
   from the encoder's final states, closes most of ISLA's gap to GeoTransolver-volume on pressure". Reading the
   final states is the control of @tbl-arch-read-depth (1.37 to 1.72 times the matched-depth read's error), so this
   10,000-cell ordering conflates final-state reading with read-only reading.
6. `06-using-isla.qmd` 78 to 84: "The class offers no decode whose answer at a point is independent of the
   request; what one would cost is in @sec-principled-decisions." Also 376 to 378 ("Deploy at the training point
   count ... because the interacting decode's prediction at a point depends on the joint request"), true for ISLA
   but silent on ISLA-DTT as the choice when arbitrary point sets are needed.
7. `16-limits.qmd` 294 to 300: "The class has no evaluate-anywhere decode: the request-independent decoder that was
   built was several times less accurate on every interior field and was removed (@sec-arch-passive-decoder). What
   would move this: a query-independent decode within seed noise of the interacting one on either interior
   dataset." On AirfRANS ISLA-DTT is ahead of ISLA (0.74 / 0.83 / 1.03, at the earlier ISLA reference
   configuration).
8. `02-isla.qmd` 1098 to 1099: "A request-independent decode was built, priced and removed
   (@sec-arch-passive-decoder)."
9. `19-literature.qmd` 420 to 424: "Query-independent decoding by cross attention from the queries to a reduced
   anchor set (AB-UPT, UPT, Perceiver IO) is the passive decoder the program already priced at several-fold on
   every interior field."
10. `physicsnemo/experimental/models/isla/model.py`, `kernel_read_theta` docstring (code, but 04 renders the class as
    its source of truth): "A numerical tolerance, not a model constant: ``0`` is the dense read pair for pair, and the
    error vanishes as it falls." ISLA-DTT is trained at its $\theta$ (REPORT "Decision (2026-10-06 evening)"), so
    $\theta$ is a model constant.
11. The boundary-law mechanism. `07-drivaerml.qmd` 619 to 623 and `16-limits.qmd` 227 to 230 say the volume model's
    uniform-point-cloud sensitivity is entirely the queries' built-in geometry (16: "the slice states, anchors and
    routing mass carry none of it"). REPORT "E3 setup" says the opposite ("most of the interacting model's
    boundary-law sensitivity on the Volume task sits in the query tokens' participation in the slices, not in the
    built-in query geometry both models share"). Both can hold if the geometry channel is the source and the
    interacting points' writes into the slices amplify it (the read-only models share the channel and move 1.10 to
    3.1 times, against ISLA's 6.6 to 7.9). The drafts state only the measured factors and take no side.

Noticed in passing, not about these findings: `04-architecture.qmd` 10 to 11 says "$M = 256$ slices, the class's
default layer and slice counts" while the class default and 04's own switches table say 512; `07-drivaerml.qmd` 829
calls the boundary-law sensitivity "an open weakness with no measured mechanism" while 587 says "its cause is
measured (@tbl-vol-geo)"; `20-aerozoo.qmd` 797 describes ISLA-DTT as having the tree read "in place of the dense
one", where line 26 of the same file says "in place of the interacting prediction points".

## Figures these placements would take (not drafted)

- `04-architecture.qmd`: a third subgraph in @fig-isla-modes for the read-only task: boundary tokens through the 12
  layers; passive points reading each layer's slices with no arrow back, and, with the global read, the boundary
  cells through the forest; the head on the $Q$ points. Mermaid, no artifact.
- `04-architecture.qmd`: the forest on the running Joukowski airfoil for one prediction point at $\theta = 1$, cells
  read one by one near the point and virtual cells (centroid and mean normal) far from it, colored by quadrant tree.
  Drawn at render time from `BoundaryForest.build` and `point_pairs` on CPU with `palette.joukowski()`.
- `07-drivaerml.qmd`: the read ladder as a figure (described under placement 4).
