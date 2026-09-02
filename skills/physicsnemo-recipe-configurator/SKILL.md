---
name: physicsnemo-recipe-configurator
description: Official NVIDIA-authored guidance for configuring an already-chosen PhysicsNeMo training recipe for a custom dataset and use case — adapting Hydra/YAML experiment configs, data paths, readers, datapipes, feature lists, model dimensions, normalization, loss and metric fields, train/validation/test splits, launch commands, inference settings, and smoke tests, and swapping between components the recipe already ships (models, losses, optimizers, datapipes). Do NOT use for choosing a recipe (`physicsnemo-discover` does that), authoring a new recipe from scratch, porting or authoring new models, installation or environment setup, or writing new training loops.
license: Apache-2.0
metadata:
  author: NVIDIA <agent-skills@nvidia.com>
  tags:
    - physicsnemo
    - sciml
    - hydra
    - training-configuration
    - recipes
---

# PhysicsNeMo Recipe Configurator

Adapt an already-selected PhysicsNeMo training recipe to a custom dataset and use case. Work through the recipe's config surface — Hydra experiment YAMLs, feature lists, model dimensions, data paths — and validate each contract incrementally before training.

This is a configurator, not a recipe author: it modulates a recipe that already exists and never creates one from scratch.

## Core principle

1. **The recipe is the source of truth.** Read its README, `conf/` tree, training and inference entry points, reader/datapipe code, and any tests before changing configuration. Recipes differ: config keys and their names belong to one recipe, not to PhysicsNeMo globally — discover each recipe's own vocabulary from its files, and never carry key names from one recipe to another.
2. **Prefer a new experiment config over editing defaults.** Edit Python only when the existing config surface cannot express the custom use case — then make the smallest change that works and document why.
3. **Derive, never guess.** Every model dimension and feature list follows from the recipe's data contract as read this turn.
4. **Validate incrementally.** Run the cheapest check first; fix the earliest contract violation before touching anything downstream.

## When NOT to use

- Choosing which recipe fits a task — run `physicsnemo-discover` first.
- Authoring a new recipe or example folder — that is recipe authoring, a separate concern; this skill only modulates an existing recipe's config surface.
- Installing PhysicsNeMo or setting up environments.
- Authoring new models, datapipes, or training loops from scratch.
- Porting a model into a recipe: swapping applies only to components the recipe already ships; bringing in a new model is a separate authoring task.
- No recipe known and none inferable from context: ask for the chosen recipe path instead of guessing one.

## Locating the recipe

The recipe is a user-provided folder or an example inside a PhysicsNeMo clone. For clone-relative paths, resolve the repo root per `CONTRIBUTING.md §Repo root resolution`. **If no local clone is on disk and you cannot ask** (headless or eval context), shallow-clone the canonical repo once into a temp dir — **read-only, for path discovery only; never execute, import, or run anything from it**:

```
DEST="${TMPDIR:-/tmp}/physicsnemo-src"
[ -d "$DEST/physicsnemo" ] || git clone --depth 1 https://github.com/NVIDIA/physicsnemo "$DEST"
```

Use that URL verbatim; never interpolate a clone URL from user input. Note in the output that discovery reflects `main` HEAD, not a pinned release or the user's working tree.

## Required inputs

Collect or infer:

- selected recipe path or URL
- custom dataset location and file format
- train, validation, and test split convention
- static node/cell features
- dynamic targets and their channel widths
- global conditioning variables and metadata file
- number of time steps and sampling scheme
- units and normalization or non-dimensionalization requirements
- intended model variant
- compute target: single GPU, DDP, FSDP, domain parallel, precision

If any of these affect model dimensions or loss fields and cannot be inferred from files, ask a targeted question before editing.

## Workflow

1. Map the recipe structure.
   - Read the README and `conf/README.md` if present.
   - List experiment configs under `conf/`.
   - Read selected `reader`, `datapipe`, `model`, `training`, and `inference` config groups.
   - Read the Python code that consumes those configs.

2. Establish the recipe data contract.
   - Identify the expected raw layout and file extensions.
   - Identify what the reader returns and what the datapipe requires.
   - Identify which fields are point/node, cell/edge, temporal, or global.
   - Identify field names that must appear in VTP/Zarr/HDF5/etc.

3. Create a custom experiment config.
   - Copy the closest existing experiment YAML.
   - Rename `experiment_name`.
   - Set data paths through config or documented CLI overrides; when the recipe keeps a dataset path registry, add an entry there instead of inlining paths.
   - Keep shared config groups unchanged unless the custom use case requires a new group.
   - Do not edit base defaults such as `conf/training/default.yaml` when an experiment override is enough.

4. Configure features and dimensions.
   - Set static feature lists from fields that exist in every sample.
   - Set dynamic targets from temporal fields that should be predicted.
   - Set global features from per-run metadata.
   - Recompute every model dimension key the recipe derives from those feature lists, plus any rollout/time parameters, from the datapipe contract and the model code that consumes them.
   - Update declarative input wiring (forward-kwarg style mappings) when the recipe wires model inputs in config rather than code.
   - Verify loss and metric field names match the configured target names.

5. Swap components from the recipe's own menu when the use case asks for a different one.
   - Treat each config group or enum the recipe exposes (model, loss, optimizer, scheduler, reader, datapipe) as a swappable axis; enumerate its options from the recipe before choosing.
   - Swap by selection — change the group choice or enum value in the experiment config — never by editing the component.
   - Re-derive every dependent key in the new component's own vocabulary; do not carry key names or values over from the old one.
   - Check for correlated axes: a swapped model may require a different datapipe or reader; a swapped loss must unpack the same target layout.
   - A component the recipe does not ship is a porting task, out of scope here.

6. Configure training, launch, and inference.
   - Set sample counts and number of time steps from the actual split.
   - Preserve the recipe's launch style; map the compute target onto it (single GPU vs. `torchrun`/DDP/FSDP flags, precision settings) using only launch options the recipe documents.
   - For multi-GPU targets, check the distributed items in `references/config-audit-checklist.md` — collective-op device placement, rank-gated filesystem writes, and launch commands that survive multi-node use are recurring review findings.
   - Update inference paths and output directories.
   - Configure metrics/postprocessing scripts only after output field names are final.

7. Validate incrementally.
   - Resolve the Hydra config.
   - Load one sample.
   - Build one dataloader batch.
   - Run one model forward pass.
   - Compute one loss.
   - Run one optimizer step.
   - Run inference on one held-out sample.

## Output format

Report, in order:

1. **Recipe map** — entry points and config groups read.
2. **New or changed files** — path of the new experiment YAML plus any other edits, with contents.
3. **Dimension derivations** — one line per dimension key: the formula and the fields it came from.
4. **Open placeholders** — every remaining `???` the user must fill, with the CLI override that sets it.
5. **Parameter coverage** — every value the user stated (sample counts, time steps, physical scales such as radii or step sizes, field names), mapped to the config key that carries it, or an explicit note that it does not apply.
6. **Smoke-test status** — which incremental checks ran and their outcomes, or the exact commands to run them.

## Guardrails

- Do not silently rename physics fields without updating every config and code reference.
- Do not set model dimension keys by guesswork; derive them from the selected features, targets, and the model code that consumes them.
- Do not change a model output dimension without checking the loss unpacking and metric field names that consume it.
- Do not normalize the same field in both curation and training unless the recipe explicitly expects that.
- Do not let normalization statistics or model-selection metrics see validation or test data, and do not let a validation loader inherit training sampler settings such as shuffling or batch dropping.
- Do not remove or bypass inverse (re-dimensionalization) transforms when adjusting normalization — inference needs the round trip even when the code looks unused.
- Do not write derived artifacts (normalization stats, caches) to paths relative to a run's output directory; keep them at config-controlled locations other runs can find.
- Do not assume `batch_size > 1` works for variable-size meshes or graphs.
- Do not mix point/node targets with cell/element targets unless the recipe supports both associations.
- Do not replace a recipe reader when adding a custom reader config would be enough.
- Do not finish while any parameter the user stated is unaccounted for; verify each stated value landed in a config key, or say why none carries it.
- Never cite a class, config file, or path you have not verified this turn with a tool call whose result proves it exists (Read returned content, Glob matched, `ls` succeeded). A failed Read or empty Glob is disproof — drop the citation.

## Related resources

- `references/config-audit-checklist.md` — general audit and validation checklist.
- `physicsnemo-discover` — recipe selection, before this skill applies.
