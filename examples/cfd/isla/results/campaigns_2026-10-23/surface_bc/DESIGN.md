# SURFACE-BC design: an ISLA surface task that reads the whole boundary

Lane SURFACE-BC of the ISLA program, written 2026-10-23 about 21:05 EDT. Base: `origin/psharpe/isla-whole-boundary`
(c1e07b7fd, ISLA 0.9.0). Peter has not yet chosen between (A) and (B) (supervisor's question, 20:34 EDT); this is the
supervisor's recommendation (A), built in a worktree while the answer is pending. Nothing here merges without Peter.

Rulings relied on: canonical ISLA reads every stored piece of the PDE domain's boundary, by name with its condition type
(Peter, 2026-10-23 17:08 EDT); boundary-only inputs, nothing precomputed, no caches across forward calls; keyword-only,
mesh-native forward; class defaults equal the recipe YAMLs; ISLA is unreleased, so no compatibility shims and no flag that
keeps the old path; DrivAerML lengths in the datapipe are in units of L_ref = 5 m.

## 1. The problem

An ISLA surface task (`interior=False`) predicts at every cell of every boundary in `boundary_meshes` (model.py, forward:
"A surface task's tokens are every cell"; the output Mesh's points are "the boundary token positions in token order").
So a condition that carries no outputs cannot be given to it: the output would include the floor's cells, which have no
targets. The DrivAerML case files store four domain meshes beside the vehicle (no_slip 722, slip 2,888, inlet 722, outlet
722 triangles, synthesized by physicsnemo-curator PR #58 from the documented tunnel), and the surface readers load
`boundaries/vehicle` only.

## 2. (A): which cells carry the outputs is declared; every named boundary is read

One behaviour for every surface task: the tokens are the cells of every given boundary (all conditions, as for the
interior task), and the outputs are predicted at the cells of the declared output boundaries only. There is no default
that predicts everywhere: a surface task always declares where its outputs live (DeepJEB++ declares all three of its
conditions, DrivAerML Surface declares `vehicle`).

### 2.1 Two API shapes

**Shape 1 (recommended): a keyword-only forward argument, `prediction_boundaries`.**

```python
model(
    boundary_meshes={"vehicle": v, "no_slip": f, "slip": s, "inlet": i, "outlet": o},
    prediction_boundaries=["vehicle"],
    global_data=gd,
)
```

- The surface-task analogue of `prediction_points`: `boundary_meshes` is the whole boundary for both tasks, and the
  second argument says where the outputs are, points for an interior task, named boundaries for a surface task.
- Required for a surface task (a non-empty sequence of distinct declared names, each present in `boundary_meshes`; a bare
  string is refused), refused for an interior task, exactly as `prediction_points` is the other way round.
- The output Mesh's points are the prediction boundaries' cell centroids, the boundaries in canonical (sorted) order, each
  boundary's cells in its mesh's order: the same rule as now, restricted to the named boundaries, so the recipe's
  alignment of predictions with the interior's targets (MeshToDomainMesh, SplitBoundaryByLabels' `reorder_interior`)
  is unchanged.
- Why it is principled: the docstring's own rule for what is a constructor argument is "different parameter sets" (the
  interior task owns the prediction-point input layer). Which cells the head reads owns no parameter: the head, the
  frame and every condition's input layer are the same whichever boundary carries the outputs. So it is data of the
  call, like `prediction_points`, not structure of the model.
- Consequence: the constructor and the state dict do not change. Every 0.9.0 checkpoint, the release test data
  (`isla_default_v1.mdlus`) included, loads as it is, and a body-only call with `prediction_boundaries=["vehicle"]`
  computes exactly what 0.9.0 computed. No version bump is needed (and none is made: the brief forbids it), and no
  shim exists.
- Recipe cost: `forward_kwargs` values are paths, so a literal list of names needs a resolver form. I add one:
  `{literal: <value>}` passes `<value>` through unresolved (`prediction_boundaries: {literal: [vehicle]}`).

**Shape 2: a constructor declaration, `ISLA(..., prediction_boundaries=["vehicle"])`** (beside `boundary_schemas`;
the schema grammar has no condition-level key, since a condition's mapping is a field schema, so a separate argument is
the clean form of "a declaration in the schemas").

- Pros: validated once at construction; the checkpoint records where the outputs live, so inference cannot predict on
  the floor by a config slip; no recipe resolver change (`model.prediction_boundaries: [vehicle]`).
- Cons: it is a constructor argument that owns no parameters; it has no meaningful class default (the names are the
  dataset's), so a surface model without it fails, and every stored 0.9.0 checkpoint (release test data and the
  program's trained DrivAerML Surface, HiLift and DeepJEB++ models) no longer constructs from its recorded arguments.
  Fixing that means regenerating the release data or a default that keeps the old path, the shim the rules forbid.
- One model then cannot be asked for its outputs on another boundary (say, wall shear on the floor if targets existed),
  which shape 1 allows.

**Recommendation: shape 1.** It follows the model's own rule for constructor versus forward arguments, it keeps every
checkpoint loadable without a shim, and it mirrors `prediction_points`. The recipe price is one small, general resolver
form. Alternative considered and rejected: splitting the mapping into `boundary_meshes` (predicted at) and a second
mapping of condition-only meshes; that gives the two tasks different meanings for `boundary_meshes`.

### 2.2 What changes, item by item

**Token draw.** None in the surface task, as now: it reads every cell it is given. The interior task's water-filled
budget (`boundary_tokens`, equal shares over the named boundaries by cell count, small boundaries kept whole) stays an
interior-task argument and a surface task still takes only its default. On DrivAerML Surface the dataset's subsample
draws the vehicle (200,000 cells, Horvitz-Thompson measures from `SubsampleMesh`), and the four domain meshes are read
whole: 205,054 tokens, of which 200,000 are outputs. If a condition-only boundary were ever large (a HiLift far field at
full resolution), the principled extension is the interior task's draw restricted to the condition-only boundaries (the
output boundary must stay whole, since every output cell is a token); not needed for any stored dataset now, so not
built.

**Geometry of the output cells.** Unchanged in kind: an output cell is a boundary token with its centroid, normal and
measure, and its condition's input layer. It gets no wall-distance channel: it lies on the boundary, so its distance to
the union is zero. The domain pieces reach it through the slice layers: they are tokens that route into the same
slices, so the anchors (measure-weighted mean positions and normals) and the slice states carry the floor and the walls,
and every vehicle token reads its distance and direction to those anchors (`dist`, `log dist`, `rel.n`, ...). The
distance from the underbody to the floor is therefore available as a relation to anchors the floor tokens form, not as
a stored or derived per-cell input. An explicit per-cell distance to the other conditions (the interior task's exact
closest point, from each output cell to the union of the condition-only boundaries) would be a new built-in channel with
new parameters; it is an option for later, not part of this change.

**The length unit.** `rms_pairwise_distance` over all given boundaries at their given resolution, as for the interior
task and on principle (one length of the PDE domain). On DrivAerML it grows from about 2.1 m (vehicle) to 59.5 m
(BOUNDARY-AUDIT section 4). Effects: the vehicle spans about 0.08 L instead of 2.3 L, so every vehicle-anchor `dist` is
28x smaller and `log dist` shifts by about -3.3; both enter learned linear maps, so the shift is learnable, but trained
weights do not transfer between the arms. The surface configs declare no `reynolds`, so no global input moves; were one
declared, z would shift by +1.45 (log10(59.5 / 2.11)). Alternative: L over the output boundaries only; rejected, since
the unit would then depend on which boundary carries outputs rather than on the domain.

**Measure weights.** The domain pieces hold 17,120 m^2 against the vehicle's 32-36 m^2 (99.8 percent of the measure),
and the routing reads `log_w = log(measure)`: measure-weighted prototypes, slice states and head-frame anchors are
dominated by the domain pieces wherever they share a slice with the vehicle. This is the quadrature of the boundary and
the same as on DrivAerML Volume since c1e07b7fd, so it is not changed; it is the first thing to look at if the
whole-boundary surface arm trains worse.

**Orientation.** The four domain meshes are wound into the fluid on all 484 files; the vehicle into the body.
The slice anchors average the normals of the tokens they hold, so one convention is needed once both kinds of token
share slices: the ISLA surface config appends `FlipBoundaryOrientation(vehicle)` (an ISLA-only recipe step, as the
Volume dataset does for every model), so every piece points into the fluid. The flip reverses the vehicle's normals and
leaves the cells, measures, centroids and targets as they are.

**The winding-number sign at on-wall points.** Not used by (A): a surface task computes no signed distance and no
winding number; every token is a cell with its own normal. It matters for (B) (section 3).

**Pose equivariance.** Exact, as now: positions enter only as differences, so a rotation, reflection or translation of
all five meshes together maps the outputs covariantly; the test moves the domain pieces with the vehicle. Moving the
vehicle alone relative to the floor is a different problem, and the outputs change (the test checks that the domain is
read). In the datapipe, every augmentation and the centering must therefore move the domain pieces with the vehicle,
which is why the reader returns a DomainMesh before any transform (section 2.3).

### 2.3 Recipe

- **Reader.** `MeshReaderWithGlobalData(domain_boundaries=[inlet, no_slip, outlet, slip])` loads the vehicle as now
  (subsampled to `sampling_resolution` cells in the read, global data merged from the case) and the named sibling
  boundaries of the same case file (`.../boundaries/<name>`) whole. It returns a DomainMesh whose interior is, until the
  terminal step, the vehicle surface, with the domain pieces as boundaries and the case's global data at the domain
  level. Every transform of the chain then acts on all meshes through `apply_to_domain`: the z rotation and the
  translation draw once and move all five; `CenterMesh` centres on the interior (the vehicle, as before) and moves all
  five; `NonDimensionalizeByMetadata` divides every mesh by L_ref; `SubsampleMesh` changes nothing (the vehicle is
  already at its size, the domain pieces are smaller). The vehicle's draw, centre and pose are those of the body-only
  reader, so the scored cells and their targets are the same.
- **Terminal.** `SurfaceMeshToDomainMesh` (recipe-local subclass of `MeshToDomainMesh`): on a DomainMesh it applies
  `MeshToDomainMesh` to the interior (centroid point cloud with the targets; the vehicle to `boundaries.vehicle`) and
  keeps the domain pieces and the domain-level global data. On a bare Mesh it is `MeshToDomainMesh`.
- **Datasets.** The four DrivAerML Surface dataset files (`drivaer_ml_surface`, `_posed`, `_pose`, `_1case`) use both,
  as the Volume dataset keeps its in-file boundaries for every model. Baselines read `interior` and
  `boundaries.vehicle` as before; the domain pieces are loaded and ignored by them (as GLOBE ignores them on Volume).
- **Model config.** `isla_surface.yaml` reads vehicle, no_slip, slip, inlet and outlet, each an empty schema named as
  in the file, and predicts at `vehicle`. `isla_surface_highlift.yaml` and `isla_surface_deepjebpp.yaml` declare their
  output boundaries (vehicle; fixed, free, loaded). The README's HiLift command names `isla_surface_highlift`.
- **Floor offset.** DRV-GROUND's `PlaceGroundUnderBody` (2.4 mm on 297 of 484 files; merged 22:15 EDT) needs the whole
  vehicle, whose lowest z the ground meets; a subsample's lowest z varies by draw. So the reader applies it
  (`whole_surface_transforms`) to the vehicle at full resolution, from its memory map, in raw metres, before the subsample
  and every pose; the transform may move only the domain boundaries (one that replaces the surface is refused). Built
  after the steer of 22:13 EDT (35a460b34).

## 3. (B): the surface task as an interior task at the vehicle's cell centres

`interior=True`, `prediction_points` = the vehicle's cell centroids, `boundary_meshes` = the five pieces at full
resolution. No model change, but:

- **A different model.** The interior task's parameters (prediction-point input layer, routing mass) and its
  query-set coupling (ISLA's prediction points interact; ISLA-DTT reads the boundary at 2-4x the step). Every surface
  checkpoint is retrained either way, but (B) also changes what a surface prediction depends on.
- **Duplicated points.** Each output location is both a boundary token (a drawn cell) and a prediction point at the
  same position: 200,000 extra tokens.
- **Degenerate geometry at on-wall points.** Every prediction point lies on the surface, so its built-in distance is 0
  or a rounding error of either sign: `s` and `asinh(s / 1e-6)` carry nothing, and only the direction remains. The sign
  is continuous through the surface by 0.8.0's encoding, but the closest-cell search at a centroid is a tie at shared
  edges near sharp features; with a winding-number sign (the robustness bundle under TIE-EXP's rule), a point on the
  surface has winding 1/2, and the DrivAerML union is open (7,425-7,497 open edges where the vehicle meets the floor),
  so the sign at the wheels' contact is ill-defined.
- **Full-resolution vehicle in every step.** The interior task's geometry and draw read the full surface (14.6-20.4 M
  triangles per case), so the reader stops subsampling the vehicle: about 0.6 GB read per sample instead of 200,000 cells.
- **Cost.** It is the DrivAerML Volume step's structure (200,000 points, 200,000 boundary tokens drawn from the
  full-resolution five pieces). Measured there (BOUNDARY-AUDIT section 4, 0.9.0, bfloat16, eager): 2,897 ms per step
  with the whole boundary (the Warp seed query takes 2.39 s of it), 513 ms without the seed, 140.6 GB peak, against the
  surface task's step measured in this lane (REPORT.md). After GEO-FAST's seed fix the projection is about 0.92 s.
- **Recipe.** New interior configs for the three surface datasets, and the target alignment moves to the
  prediction points (already the interior's order).

(A) keeps the surface task's meaning, adds 2.5 percent tokens on DrivAerML, and needs no geometry query.

## 4. Tests

Model (`test/experimental/models/isla/test_isla_surface_conditions.py`): outputs at the output boundary's cells only
(points equal its centroids, in order, count equal its cells); a domain boundary changes the prediction (it is read);
rotation, reflection and translation of all meshes together map the outputs covariantly; the argument's validation
(required for a surface task, refused for an interior task, unknown, absent, duplicate names, a bare string); a
body-only call equals the stored 0.9.0 release outputs (the release tests, which pass the argument). Every existing
surface forward call in the ISLA tests passes `prediction_boundaries`. Recipe: the resolver's literal form, the reader
(domain pieces loaded whole, the vehicle subsampled, the same draw as without them), the terminal, the config test for
five boundaries with predictions at the vehicle only.
