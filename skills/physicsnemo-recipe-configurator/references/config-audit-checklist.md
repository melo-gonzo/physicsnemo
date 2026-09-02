# Config Audit Checklist

Use this checklist when configuring an already-selected PhysicsNeMo recipe.

## 1. Recipe Map

Read or list these before editing:

- recipe `README.md`
- `conf/README.md` when present
- all top-level experiment YAMLs under `conf/`
- selected `conf/reader/*.yaml`
- selected `conf/datapipe/*.yaml`
- selected `conf/model/*.yaml`
- selected `conf/training/*.yaml`
- selected `conf/inference/*.yaml`
- training entry point
- inference entry point
- reader and datapipe implementation files
- tests or smoke-test scripts

Record which config file owns each setting. Prefer an experiment-level override for use-case-specific changes.

List the options available in each config group or enum — these are the recipe's swappable axes (model, loss, optimizer, scheduler, reader, datapipe).

## 2. Data Contract

For each sample, identify:

- raw file or store path
- sample ID convention
- number of time steps
- node/point count behavior: fixed or variable
- edge/cell connectivity source
- reference coordinates field
- temporal position or displacement fields
- static fields
- dynamic fields
- global metadata fields
- physical and sampling parameters the pipeline consumes (connectivity radii, neighbor counts, step sizes, resolutions)
- units and normalization or non-dimensionalization requirements

Verify that every configured feature name exists in every training and validation sample. If a feature is optional in the raw data, either remove it from the config or make the reader handle the missing case explicitly.

Split checks (recurring review findings):

- The split definition must control which files load, not just how many — a count-only split silently leaks train data into validation.
- Sample ordering must be numeric or explicit, never lexicographic (`run10` sorts before `run2`), or splits drift from the convention that generated them.

Cross-check before finishing: every numeric or named parameter the user stated must land in a config key, or be explicitly called out as not applicable. A stated value that silently falls back to a group default is a contract violation.

## 3. Reader and Datapipe

Read the datapipe constructor and reader call site. Confirm:

- what arguments the datapipe passes into the reader
- what tuple or object the reader must return
- whether global metadata is loaded by the reader or datapipe
- what keys are required in point/node data
- whether heavy preprocessing belongs in curation rather than `__getitem__`
- whether sample order, split order, and sample counts are deterministic

When writing a custom reader, match its signature to what the datapipe call site actually passes, and accept `**kwargs` so later recipe changes do not break it. Fail loudly: raise descriptive errors that list the available keys instead of letting bare `KeyError`s or silent fallbacks mask a bad config. If the reader adds imports, add them to the example's requirements file and confirm they install.

When a reader or datapipe change touches only one side of the contract, re-check the other side in the same pass — schema drift between them (renamed keys, changed return arity, stale `None`-checks) is the most common cross-file breakage reviewers catch. Exercise the validation and test paths too: bugs in stats loading or sample counting often only detonate on the split that was never run.

## 4. Dimension Derivation

Derive dimensions from the configured fields, in three steps:

1. Find where the model config's dimension keys are consumed in the model and rollout code.
2. Express each key as a function of the configured feature lists.
3. Recompute every affected key after any feature-list change.

The key names and formulas are recipe-specific — read them from the recipe, never from memory. The shapes they usually take:

- a global/conditioning width equal to the number of configured global features, or null/absent when the recipe disables them
- an input width summing the channel widths of whatever the config wires into the model input (coordinates, SDF-like scalars, normals, static features, a scalar time or other conditioning channel)
- a per-step output width summing the base target width and any extra per-point target widths
- a full-trajectory output width of (number of predicted steps) x (per-step width)
- widths the recipe interpolates from another config block instead of hard-coding — keep the interpolation, do not freeze a number

Derive each width by counting channels in the wiring the config declares, then confirm against the model/rollout code that consumes it.

## 5. Component Swaps

When swapping a component for another the recipe ships:

- Enumerate the sibling options in the config group, or the values the consuming code accepts for an enum key.
- Swap by changing the selection in the experiment config, never by editing the component itself.
- Re-derive every dimension and wiring key in the new component's own vocabulary; carry nothing over from the old one.
- Check correlated axes: a swapped model may require a different datapipe or reader; a swapped loss must unpack the same target layout.
- Verify loss and metric field names still match the configured targets.
- If the desired component is not configurable (a hard-coded loss, for example), that is the smallest-Python-change case — flag it and document why.
- A component the recipe does not ship is a porting task, not a swap; stop and say so.

## 6. Normalization and Splits

- Compute normalization statistics from the training split only; never let them see validation or test samples.
- Store stats and other derived artifacts at a config-controlled path that a separate inference or validation run can reach — never relative to one run's output directory.
- Normalize each field exactly once across curation and training; confirm which stage the recipe expects to own it.
- Keep the inverse (re-dimensionalization) path intact and confirm inference uses it; inverse transforms often look like dead code and are not.
- When reporting or configuring metrics, state whether values are in dimensional or non-dimensional space.
- Validation loaders must not inherit training sampler settings (`shuffle`, `drop_last`), and model selection or LR scheduling should key on validation metrics, not training loss.

## 7. Hydra Config Hygiene

- Add a new experiment YAML for the custom use case.
- Keep `_self_` in the defaults list if the recipe uses it.
- Keep unresolved required paths as `???` unless the user asked to bake paths into the config.
- Use CLI overrides for local paths when possible; never commit machine-local or cluster-specific paths, hosts, or usernames.
- Keep comments that explain formulas such as `out_dim`, and fix any comment the change makes stale — a comment contradicting its value is worse than no comment.
- Prefer keys the recipe derives automatically over hand-computed values; a config a user must do arithmetic to fill is a review finding.
- Remove config keys nothing consumes; a dead key misleads the next reader.
- Do not modify shared config groups for a single experiment unless the group is truly reusable.
- If a changed value, flag, or command appears in the recipe README, update the README in the same change — README/config drift is caught in review over and over.

## 8. Launch and Distributed Checks

For multi-GPU or multi-node compute targets:

- Collective operations require device tensors; a documented multi-GPU command that feeds CPU tensors to NCCL fails at runtime.
- Any rank-gated filesystem write (rank-0 `mkdir`, cache creation) needs a barrier before other ranks read, or each rank creates what it writes.
- Logged losses and metrics must be aggregated across ranks, not rank-local.
- Documented launch commands should survive multi-node use (no `--standalone`) and sample counts not divisible by world size.
- Check the recipe's seeding story: seeds set per rank, unseeded shuffles, or `cudnn.benchmark=True` all quietly undermine reproducibility claims.

## 9. Smoke Tests

Run the cheapest checks first:

1. Print the resolved Hydra config.
2. Instantiate the reader/datapipe.
3. Load one training sample.
4. Load one validation sample.
5. Run one dataloader batch.
6. Instantiate the model.
7. Run one forward pass.
8. Compute one loss.
9. Run one optimizer step.
10. Run inference on one test sample.

If a smoke test fails, fix the earliest contract violation. Do not tune model hyperparameters before the data contract is correct.
