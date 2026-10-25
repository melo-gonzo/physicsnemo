# ISLA module goldens

Characterization goldens of ISLA's modules: the A/B harness that proves a refactor or
a library move bitwise, and the committed CI golden that guards against regressions.

- `cases.py`: the cases. Each builds its inputs on the CPU from seeded
  `torch.Generator`s (`build()`), by correctly rounded operations only (float64
  draws, `+ - * /` and `sqrt` one per op, a cast to float32), so that they are the
  same bits on x86 and Grace (`normalize`, `log`, `cos`, matrix products, QR and
  float32 `randn` are not; measured in job 377631), and runs the functions under
  test on a device under
  a variant (`run(inputs, device, variant)`: the reference, or the fast and fused
  kernels). Every function is reached through `TARGETS`, one line per function naming
  the module that owns it: a LIB-MOVE switch retargets that line, and the golden of
  the moved function must still pass.
- `record.py`: runs cases and writes a record, or regenerates the committed golden.
- `compare.py`: compares two records.
- `../test_isla_goldens.py` and `../data/goldens_v1.pt` (manifest
  `../data/goldens_v1.json`): the CI golden.

Groups: G-FRAME (`canonical_frame`, `frame_coordinates`, the skew-tie branch), G-SAMP
(`water_fill`, `pps_inclusion_probabilities`, `sample_boundary_cells`, `draw_inputs`),
G-FOREST (`BoundaryForest.build`, reference and fast; `node_means`,
`fused_node_means`), G-WALK (`point_pairs`, `frame_pairs(count_only=True)`, reference
and fast), G-SEG (`PairPlan.build`, `segment_softmax_read` eager and Triton), G-GEO
(`exact_boundary_geometry` with its search outputs and winding numbers, 2-D and 3-D,
the tied, on-wall, thin and degenerate sets; `run_start`, `canonical_cells`,
`closest_weights`), G-READ (`tree_read_walk`, `walk_range_pairs` over ranges,
`walk_points_range`, `walk_pair_counts`, `tree_pair_invariants`, `_KernelBasis`,
`_TreeKernelRead`, `fused_kernel_read`, the fused basis), G-BLOCKS (`_SliceBlock`
forward and read, `_RoutingLogits`, `_softmax_over_points`, `_relational_invariants`,
`_geo_region`, `fused_geo_region`), G-HEADS (`_isotropic_tensors`, `_levi_civita`,
`_OutputLayout`, `_assemble_output`, `_pseudovectors`, `_polar_complement`), G-MODEL
(`ISLA.forward` on small models, hidden 32, two layers, eight slices: a surface model; a
rich surface model with boundary fields of rank 0 to 2 on two conditions, a rank-1
field and a rank-0 global defining scales, outputs of rank 0 to 2 with odd ones of
rank 1 and 2, units, an output constant and `output_stats`; ISLA-DTT, with its
boundary reads live; ISLA-DTT in 2-D; ISLA-DTT drawing 512 tokens of about 2,000 cells
over two conditions; `query_chunks` over three ranges; float64 and float32; every
output and every parameter's gradient), G-PARTS (the same forward passes with spies on
the model's parts: each `_SliceBlock`'s output, the boundary reads' updates, the head
frame's `_RoutingLogits` prototypes and logits, `_QueryTokens`' input layer, and the
results of `cell_second_moments`, `rms_pairwise_distance`, `soft_log_length`,
`rms_moments` and `scale_factor` (the scales), `sample_boundary_cells` (kept cells and
weights), `_pseudovectors` and `_assemble_output`, call by call).
Every case records its inputs, outputs and the gradients of a fixed cotangent. The
models' parameters are one flat input (`in.params`, one hash), and the golden keeps 4
elements of each parameter's gradient (beside its summaries) to stay within budget.

## A/B: prove a change bitwise

Record the base and the head **each in its own process** (custom ops such as
`physicsnemo::isla_closest_candidates` register process-wide, so two versions in one
process compare one with itself), on the same machine, then compare:

```bash
G=test.experimental.models.isla.goldens
git checkout <base> && python -m $G.record --device cpu --out /tmp/a
git checkout <head> && python -m $G.record --device cpu --out /tmp/b
python -m $G.compare /tmp/a /tmp/b   # exit 0 PASS, 1 FAIL, 2 INCONCLUSIVE
```

`--groups G-GEO G-READ` and `--cases 'tree_read_*'` select cases. Each record prints the
file and md5 of every ISLA module it imported and the `physicsnemo::isla_*` ops
registered: check that each side ran the code you meant. CPU records run on one
thread (`--threads`). On CUDA (`--device cuda`, on a GB300 for the CUDA paths) each
case runs 5 times in-process (`--repeats`).

How compare.py reads the records:

- **CPU: bitwise**, every input, output and gradient.
- **CUDA, kernels auto: bitwise**, except:
  - the draw's kept cells are compared first, in every repeat of both sides; if any
    two differ the case is **INCONCLUSIVE**: the draw's float64 `cumsum` is not
    run-to-run deterministic on CUDA, so a kept cell can flip when a hit lands on a
    cell boundary. Rerun; never read it as a pass. An INCONCLUSIVE that repeats with
    each side's repeats bitwise (the kept cells differ only across the sides) is a
    FAIL: the change moved the draw. (On CPU the draw is deterministic and a kept
    difference fails.)
  - the draw's Horvitz-Thompson weights compare at 1e-15 relative, every pair of
    repeats within a side or across;
  - (DRAW-EXACT: the two rules above read today's CUDA draw, which is not
    run-to-run bitwise; once DRAW-EXACT makes it bitwise they can be dropped.)
  - the outputs a case declares as its `family` (float atomics: the reference read's
    node means, the Triton segment softmax's `dk`/`dv`, the exact geometry's vertex
    gradient) must lie within twice the run-to-run spread of the 5 repeats of either
    side (`compare.FAMILY_SLACK`: five runs estimate the spread from below; a family
    that repeats bitwise demands bitwise), and that spread must stay under a ceiling,
    `compare.FAMILY_RTOL` of `max|A|` (float32 1e-4, float64 1e-11; measured at most
    1.9e-5 and 4.6e-13 on GB300), or the case's own (`Case.family_rtol`), so that a
    head with noisy repeats cannot widen its own tolerance.
- A comparison of no case at all fails; `record.py` refuses unknown groups and
  `--cases` patterns that match nothing.
- A tensor that varies between the repeats of one side without being declared a
  family fails: the change made something non-deterministic.

## The CI golden

`test_isla_goldens.py` runs every case's CPU variants (and, with CUDA, its CUDA
variants) against `data/goldens_v1.pt`: inputs and integer outputs exact, float
outputs within `rtol * max|golden|` on 16 stored elements and three summaries over
every element within `sqrt(n) * rtol * max|golden|` (see the test's docstring and
`CPU_RTOL`/`CUDA_RTOL` for the bounds and their measurements). For the two large
bfloat16 gradients (`kernel_read_bf16`'s `grad.basis`, `segment_softmax_read_bf16`'s
`grad.ker`), whose summaries' slack exceeds their largest element, the golden also
keeps the sums over runs of 1,024 elements, each within `sqrt(1024) * rtol *
max|golden|`, half the largest element (`Case.chunks`). A missing golden fails.

Regenerate it only when a change is meant to move the numbers, and say so in the
commit:

```bash
python -m test.experimental.models.isla.goldens.record --regenerate goldens_v1.pt
```

It records on one CPU thread and writes the golden and its manifest (commit, whether
the code it depends on (the model, the modules `TARGETS` names, this harness) was
modified, the harness files' md5s, library versions, CPU, BLAS/LAPACK, module md5s, date).
The same command writes every other file of `data/`, named (or `--regenerate all`):
the goldens and reference checkpoints of `test_isla_release.py` (its `WRITERS`). A bare
`--regenerate` is refused: name `goldens_v1.pt` to regenerate this golden alone, so
that the older goldens (`isla_*_v1.pth`, which move by about 1e-6 when re-recorded)
stay as they are. Nothing else writes `data/`, and a missing file fails.
Committed data stays within its budget (350 KB for these goldens): floats are kept
as samples and summaries, integers and inputs as SHA-256.

Reading a CUDA failure of the CI golden: the CUDA variants are compared with the
CPU golden, so a difference there is CUDA against CPU (another summation order, the
fused kernels' own rounding); the A/B harness on a GB300 is what proves a CUDA change
bitwise.

## Exclusions

- **On-wall and tie directions at near-coplanar ties.** At a quad's diagonal, where
  two triangles' normals agree to rounding, the on-wall direction is a known defect
  (`test_on_wall_direction_at_quad_diagonals`, a strict xfail in
  test_isla_interior_geometry.py): it depends on the numbering. No golden holds it:
  the 3-D box-tie set keeps only the ties whose closest points lie inside the faces
  (not the corner diagonals or the centre, whose closest points lie on the faces'
  quad diagonals), the box-wall and zero-area sets keep their on-wall points on the
  box's edges and corners and the cells' centroids (not the points inside a side,
  where coplanar cells meet), the torus keeps generic queries only (not its core
  circle), and the thin cap (coplanar with its neighbours) records no direction
  (`u`, the gradients, `tied`, `tie_u`): its `d`, closest cell, sign, on-wall flag,
  weights and winding numbers stay.
- **Ill-posed outputs**: the closest point's weights on slivers 1e-13 off their line
  (x86 and Grace differ by 3e-5 on the same inputs); and, at on-wall queries, the
  winding number and the sign (`d = 0` there, the winding number is 1/2 up to
  rounding; they differ by up to 0.44 across platforms), recorded as 0.
- **Tie selections** (the box-tie sets): at a tie the reported closest cell, its
  weights and the gradients, which are taken through that cell (a selection from the
  subdifferential), differ between x86, Grace and GB300 on the same inputs and move
  under a one-ulp change of the vertices. They are not recorded; `d`, `u`, the tie
  flag and direction, the sign and the winding numbers are.
- **The closest cell at an edge or a vertex.** Where the closest point lies on a
  cell's edge or vertex (a weight within 16 eps of the queries' dtype of 0), every
  cell sharing it is at the same distance, and the reported cell and weights are the
  search's choice among them: GEO-TIEORDER's T1 (3a1a15d48) changes that choice at
  13 of the G-GEO cases, and nothing else in them beyond the golden's bounds. The
  cell is recorded as -1 and the weights as 0 there.
- **On-wall `u` at ties.** The on-wall queries at the vertices and edges of the
  sphere, circle, NACA section and the boxes' edges and corners are ties between
  cells whose normals are far apart; no golden holds an on-wall `u` at a tie of
  near-coplanar cells (above). GEO-TIEORDER at 3a1a15d48 leaves every recorded `u`,
  `d` and gradient within the golden's bounds (CPU, run against this golden). If a
  later change of the on-wall rule moves them, regenerate with
  `python -m test.experimental.models.isla.goldens.record --regenerate goldens_v1.pt`
  in its commit;
  reason: the on-wall direction at a tie is defined by that rule.
- **The draw's hits off the cells' boundaries.** Systematic sampling keeps the cell
  whose interval on the line holds each hit `start + m`; a hit within rounding of a
  boundary between two intervals would move with the summation order of the line
  (CUDA's scan, DRAW-EXACT's exact sums). Every recorded draw keeps its hits at least
  `cases.DRAW_MARGIN` = 1e-9 from every boundary (the float64 rounding of the line is
  at most 5e-10 here), checked by `test_the_draw_goldens_keep_off_the_cell_boundaries`
  (the kept cells are the same with the start moved by 1e-9 either way). The smallest
  margins measured: 5.9e-5 (`sample_boundary_cells_2d`, budget 128), 1.4e-4 (3-D).
  DRAW-EXACT at 02d5451fd passes every CPU golden unchanged.
- **The 3-D winding numbers' values** follow the tree's float32 opening decisions,
  which differ between x86, Grace and GB300 on the same inputs (by up to 4.4e-3 near
  the sphere, within the far-field approximation). The CI golden compares their
  sign, which is what reaches `d` (`out.search.sign`), not their values; the A/B
  harness records the values bitwise.
- **The thin cap's closest-point weights** compare at 1e-8 relative: they are
  conditioned by 1/h = 1e6 (measured at most 6.9e-12 across x86, Grace and GB300).
- **bfloat16 `dk`/`dv` on CUDA against the CPU golden**: the CPU golden is the eager
  read's, which adds those gradients up in bfloat16 (a reported defect:
  `_eager_segment_softmax_read` gathers before it casts); not compared there.
- **Gradients that are zero in exact arithmetic**: the basis MLP's last bias (a
  per-head constant on every pair's logit) and a slice block's `geo_logit.bias` (a
  constant over the slices) cancel in their softmaxes; what is computed is roundoff,
  and is not recorded.
