# PR 2 (recipe-extension): GeoPT pretraining as a Unified-Recipe flavor

This plan supersedes the parent plan
(`geopt-physicsnemo-engineering-plan.md`) §1.4 PR 2 + PR 3 and §1.3
module layout for the recipe-side of pretraining. **Round 1 + PR 2.5
demonstrated that pretraining and fine-tuning can be CLI flavors of the
same `examples/cfd/external_aerodynamics/unified_external_aero_recipe`
training entrypoint, not a separate `pretrain.py` + `finetune_drivaer.py`
script pair.** This document captures the rationale and the work plan.

## Why graft, not branch

The unified external-aero recipe (landed in PhysicsNeMo 25.08, lives in
`examples/cfd/external_aerodynamics/unified_external_aero_recipe/`)
already supplies every pretraining-loop ingredient:

| Pretraining requirement | Existing recipe affordance |
|---|---|
| `.pdmsh` reader → `DomainMesh` per sample | `DomainMeshReader` declared in dataset YAML's `pipeline.reader` |
| `forward_kwargs:` from DomainMesh to model.forward | `forward_kwargs.py:resolve_forward_kwargs` (declarative paths) |
| Hydra-injectable model / loss / optimizer / scheduler | every block is `_target_:` instantiated |
| Targets from `interior.point_data.<name>` | `forward_kwargs.py:extract_targets` |
| DDP / FSDP / mdlus checkpoints / LaunchLogger / AMP / `torch.compile` | all wired in `train.py` already |
| Loading a pretrained backbone before fine-tune | PR 2.5's `cfg.training.pretrained_backbone:` hook (commit `4afe62c5`) |

Adding a separate `pretrain.py` would duplicate this loop and force
us to keep two training entrypoints in sync. Adding a `finetune_drivaer.py`
shim around the recipe is now even less justified — PR 2.5's hook is
already a single Hydra override on the recipe's existing CLI.

The original parent plan §1.4 was written before the unified recipe
was as data-driven as it currently is. The recipe's own design has
caught up with the pretraining use case; we just need to point at it.

## What the workflow looks like

After this PR lands, GeoPT-style pretraining and fine-tuning are both
single CLI invocations of the same recipe:

```console
# Stage 1: pretraining on a .pdmsh corpus emitted by
# physicsnemo.experimental.pnm_pretraining.data.builder.
python src/train.py \
    model=transolver_pretrain \
    dataset=geopt_pretrain \
    training.num_epochs=200 \
    training.optimizer.lr=1e-3 \
    training.scheduler.step_size=100

# Stage 2: fine-tune from the pretrain ckpt onto DrivAerML.
python src/train.py \
    model=transolver_volume \
    dataset=drivaer_ml_volume \
    training.pretrained_backbone=runs/<pretrain_run_id>/checkpoints/Transolver.0.0.last.mdlus \
    training.pretrained_backbone.exclude_layers='[trajectory_head]'
```

No new entrypoint, no new training loop, no new logger setup.

## What lands

Five files, none of them duplicative of the recipe's existing logic:

### Recipe-side (`examples/cfd/external_aerodynamics/unified_external_aero_recipe/`)

1. **`src/loss.py`** — extend `LossType` to include `"rel_l2"`, add
   `_scalar_loss` / `_vector_loss` branches that compute relative L2
   (`‖pred − target‖₂ / ‖target‖₂`) per example, summed (matches
   GeoPT `utils/loss.py`'s default at training: `size_average=False`,
   `reduction=True`). The same loss kernel covers both scalar and
   vector targets (vector path treats per-point as a flat feature
   vector). ~30 LOC. No CLI changes.

2. **`datasets/geopt_pretrain.yaml`** — new dataset YAML.
   `DomainMeshReader` reads `*.pdmsh` files emitted by our
   `build_pretraining_sample`. No augmentations (the supervision is
   already geometry-fixed). Targets block declares the trajectory
   feature(s) — see schema follow-up below.

3. **`conf/model/transolver_pretrain.yaml`** — new model YAML.
   `forward_kwargs:` block points at `interior.points`,
   `interior.point_data.sdf`, `interior.point_data.normals_face_barycentric`
   (everything the backbone needs to encode a query point).
   `_target_:` is our new `TransolverPretrainBackbone` (next item).
   Sets `out_dim` to the flattened trajectory-feature width.

### Our-side (`physicsnemo/experimental/pnm_pretraining/`)

4. **`models/__init__.py` + `models/backbone.py`** — new subpackage
   defining `TransolverPretrainBackbone(physicsnemo.Module)`. Wraps
   `physicsnemo.models.transolver.Transolver`, replaces the Transolver
   output projection with a `TrajectoryHead` MLP. The head emits
   `(B, N, n_walks * n_steps * 3)` matching the dataset's `walks_supervise`
   target. Subclasses `physicsnemo.Module` so it gets `.mdlus`-archive
   serialization and is loadable by PR 2.5's `load_pretrained_backbone`
   (see `geopt-datagen-round1-plan.md` §8 row I17).

   **Importantly:** this wrapper does *not* use a forward hook — the
   parent plan §1.4 PR 2 proposed a hook to capture pre-output token
   hiddens, but for *training* a backbone-only model, replacing the
   output head outright is cleaner. The hook contract is still useful
   for a future "use a pretrained backbone with a different
   *architecture* head" path (e.g. attaching a regression head on top
   of a frozen pretrained backbone for fine-tuning); that comes in a
   later PR if needed.

   When the pretrain checkpoint is loaded into a *fine-tune* model
   (e.g. `transolver_volume`'s plain `physicsnemo.models.transolver.Transolver`),
   PR 2.5's `exclude_layers=["trajectory_head"]` drops the trajectory
   head; the backbone weights match by name and load. This is the
   round-trip the whole pipeline is built for.

### Round-1 follow-up (M3 schema reshape, improvement I16's resolution)

5. **`physicsnemo/experimental/pnm_pretraining/data/builder.py`** —
   reshape `walks_supervise` from M3's
   `(n_walks, n_points, n_steps, 3)` (in `interior.global_data` per
   I16) to point-major
   `(n_points, n_walks * n_steps * 3)` (in `interior.point_data`).
   Same for `walks_directions` and `walks_step_lengths` if the
   model needs them. The reshape is mechanical — no new geometry, no
   new walks, just a `.permute(1, 0, 2, 3).reshape(n_points, -1)` at
   write time. Updates the M3 round-trip test to check the new
   shape. Closes I16; the rationale ("each point owns its walk
   samples; *n_points* is the natural leading dim") is documented
   in the builder docstring.

## Loss (rel_L2) details

GeoPT's `utils/loss.py` `L2Loss(size_average=False)` at default
training (`d=2, p=2, reduction=True, size_average=False`):

```python
loss = sum_b ‖x_b − y_b‖_2 / ‖y_b‖_2
```

i.e. **per-example relative L2, summed over the batch**. The
unified recipe's existing loss interface is per-field
`_scalar_loss(pred_field, target_field, ...)` and per-field
`_vector_loss(pred_field, target_field, ...)`. We add `"rel_l2"`
to both:

- `_scalar_loss`: `‖pred − target‖₂ / max(‖target‖₂, eps)` per
  example, summed across batch dim. `eps=1e-8` to avoid division by
  zero on near-zero targets (GeoPT doesn't guard; on synthetic
  trajectory targets this can bite).
- `_vector_loss`: same formula, treating the per-point vector as a
  flat feature vector for the norm.

Adopting the existing recipe's per-batch-summing convention (not
per-batch-averaging) keeps the loss invariant under batch-size changes
matching GeoPT's behavior.

## Targets schema in the dataset YAML

Per the M3 schema reshape (item 5 above), the pretraining dataset's
`targets:` block looks like:

```yaml
targets:
  walks_supervise: vector
```

The `vector` type means "per-point, flat feature vector"; the model
emits `(B, N, F)` where `F = n_walks * n_steps * 3`. The dataset
declares `out_dim: F` via Hydra interpolation so the model YAML can
auto-set `Transolver.out_dim`.

For round-1 schema parity (e.g. M3's tiny config: `n_walks=4, n_steps=2`),
`F = 4 * 2 * 3 = 24`. For full GeoPT defaults (`n_walks=100, n_steps=3`),
`F = 900`. The model YAML uses Hydra interpolation
`out_dim: ${dataset.feature_dim}` (or the equivalent recipe-side
machinery — TBD by the subagent who writes the YAML; mirror how
`drivaer_ml_volume.yaml` exposes its target dims to the model YAML).

## Test plan

- **Recipe-side loss test:** add `"rel_l2"` cases to whichever existing
  loss-test file lives in `examples/cfd/external_aerodynamics/unified_external_aero_recipe/tests/`. Verify scalar and vector
  rel_L2 against analytic ground truth on tiny tensors.
- **Backbone-wrapper test:** in
  `test/experimental/pnm_pretraining/models/test_transolver_pretrain.py`,
  - construct a `TransolverPretrainBackbone(out_dim=F, ...)` and a
    plain `Transolver(out_dim=O, ...)` with the same backbone
    hyperparameters,
  - copy backbone weights from one to the other (mock
    `load_pretrained_backbone` with `exclude_layers=["trajectory_head"]`),
  - feed identical inputs, assert pre-projection latents match
    bit-exact (confirms the wrapper's backbone is unmodified).
- **End-to-end recipe smoke:** add a tiny Hydra integration test
  that runs `python src/train.py model=transolver_pretrain
  dataset=geopt_pretrain training.num_epochs=1 ...` against a
  micro-corpus of 4 synthesized `.pdmsh` files (built in-test with
  our M3 builder). Verify a checkpoint is written and the
  pretrained-backbone hook can subsequently load it into a
  fine-tune model. This exercises the full graft.
- **M3 schema-reshape test:** the existing `test_pdmsh_round_trip.py`
  is updated to expect point-major shapes; add an explicit assert
  that `walks_supervise.shape[0] == N+M`.

## Out of scope for this PR

- **Multi-backbone support** (GeoTransolver, GLOBE, DoMINO, FLARE
  pretraining wrappers). This PR ships *Transolver only* per the
  parent plan §1.2's "drive one pipeline first" decision. The
  `models/backbone.py` is structured so adding a `GeoTransolverPretrainBackbone`
  subclass is mechanical (parent plan §1.4 PR 2 says GeoTransolver
  + GLOBE adapters are F13-gated follow-ups; this PR honors that).
- **Real-data smoke** on ShapeNet — the recipe-side smoke test uses
  4 synthesized `.pdmsh` files built from the M3 conftest sphere.
  Real-data validation is a separate carry-forward.
- **AMP / static-capture optimization** — the recipe supports both
  via `cfg.training.precision` and `cfg.compile`; pretraining
  inherits without our intervention. We don't add new flags.
- **The reproduction study** (parent plan Part 2). That's PR 4 work
  per the parent plan.

## Decision: drop `pretrain.py` and `finetune_drivaer.py`

The parent plan §1.3 module layout calls for:

```
examples/cfd/external_aerodynamics/
  pnm_pretrain/
    config/
      shapenet_pretrain.yaml
      drivaer_ml_pretrain.yaml
    scripts/
      preprocess_shapenet.py
      pretrain.py
      finetune_drivaer.py
      eval_benchmark.py
```

This PR keeps **`preprocess_shapenet.py` and `eval_benchmark.py`**
(those are genuinely new tools) but drops `pretrain.py` and
`finetune_drivaer.py` in favor of the unified-recipe extension.
`shapenet_pretrain.yaml` becomes
`unified_external_aero_recipe/datasets/geopt_pretrain.yaml`
(possibly with a thin
`unified_external_aero_recipe/datasets/shapenet_geopt_pretrain.yaml`
that overrides specific corpus paths).
`drivaer_ml_pretrain.yaml` is unnecessary — fine-tuning from a
pretrain ckpt is one CLI override.

This is a real simplification of the parent plan, recorded as
**plan deviation D1** below.

## Plan deviations from the parent plan

### D1 — recipe-extension instead of separate scripts

Parent plan §1.4 PR 2 + PR 3 calls for `pretrain.py` and
`finetune_drivaer.py` wrappers. We replace both with Hydra config
overrides on the existing `unified_external_aero_recipe/src/train.py`.
Smaller code footprint, no new training-loop maintenance, full reuse
of recipe machinery. Documented in this PR's commit message and in
`geopt-datagen-round1-plan.md` §8 (new improvement I19, see below).

### D2 — drop the forward-hook adapter contract

Parent plan §1.4 PR 2 calls for `PnmPretrainingBackbone.encode_points()`
exposed via a forward hook on `physicsnemo/models/transolver/transolver.py:643-728`.
For *training* a backbone-only model, direct subclassing (replacing
the output head) is cleaner than hook-and-discard. The hook contract
is genuinely useful for "load a frozen pretrained backbone and attach
a different downstream head" workflows; we can add it later if a real
need arises.

### D3 — schema reshape (close I16)

Parent plan §1.2 schema (`interior.point_data.trajectory_features:
(N, τ+1, 3)`) is closer to point-major than M3's actual emission
ended up being. M3's I16 finding rationalized moving the walk arrays
to `interior.global_data` because of a TensorDict batch-size
invariant. **Now we reverse that decision** and reshape to
point-major `(n_points, n_walks * n_steps * 3)` so it satisfies the
TensorDict invariant, lives in `interior.point_data`, and is
consumable by the recipe's `extract_targets` without any recipe
changes. I16 stays in the catalog as "discovered, then unwound";
new improvement I19 records the reshape decision.

## New improvements to catalog

To be added to `geopt-datagen-round1-plan.md` §8:

| # | Improvement | GeoPT reference behavior | Our behavior | Lands in | Status |
|---|---|---|---|---|---|
| I19 | **Pretraining as a recipe flavor** | GeoPT ships separate `run.py` modes (`GeoPT_finetune`, `steady_cond`); finding D1 of the parent plan also notes pretraining itself is unreleased. | Pretraining and fine-tuning are CLI overrides on the same `unified_external_aero_recipe/src/train.py`. No new entrypoint. | PR 2 | planned |
| I20 | **Schema reshape: point-major walks_supervise** | N/A (GeoPT writes per-walk `.npy` files, not a structured DomainMesh). | `(n_points, n_walks * n_steps * 3)` in `interior.point_data`, satisfying the TensorDict per-point batch-size invariant and fitting the recipe's `extract_targets`. Closes I16 (M3 finding) by reversing it. | PR 2 | planned |

## Open questions

- **Q5.** Should `walks_directions` and `walks_step_lengths` also be
  reshaped to point-major and exposed as additional `forward_kwargs:`
  inputs (so the model can condition on the per-point velocity, à la
  GeoPT's `condition` tensor)? The parent plan §3.3 example kernel
  treats them as inputs *to* the walk, not features for the model.
  But GeoPT's released schema (`condition_{j}.npy = (N+M, 4)`)
  suggests the model is meant to *see* directions and step_lengths
  during pretraining (the model predicts `supervise` *given* the
  initial dynamics). PR 2 should resolve this by reading the GeoPT
  finetune script to confirm what the model's input signature looks
  like.

- **Q6.** What's the right default for `n_walks` in the dataset YAML?
  GeoPT's full default is 100 (10 base + 90 jittered, per I10). For a
  PR-2 smoke test we want something smaller (4–10 walks) so the test
  data is generatable in seconds. The dataset YAML can declare a
  default and let users override on the CLI.

- **Q7.** Does `cfg.training.pretrained_backbone:` need a
  `freeze_backbone:` flag? PR 2.5 doesn't expose one today. For
  fine-tuning runs that want to freeze the backbone and only train a
  head, we'd add `freeze_backbone: true` and disable grad on the
  matched parameters. Not required for the headline GeoPT recipe (the
  paper fine-tunes the whole model) but useful for ablations. Out of
  scope for this PR; tracked as a follow-up.
