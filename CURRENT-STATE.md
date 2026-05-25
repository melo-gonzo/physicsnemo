# Current state — `exp-geopt-datagen-r1`

**Read this first.** Single-file orientation for picking up the GeoPT
pretraining work in a fresh session. Everything else (plans, reports,
deferred findings) is linked from here.

## What this branch is

A working-branch worktree on top of PhysicsNeMo `main` that lands
GeoPT-style lifted geometric pretraining as an optional capability
in PhysicsNeMo, **end-to-end**:

1. **Data generation** (round 1 — M1, M2, M3): port GeoPT's
   FCPW/CPU pretraining data-generation pipeline to Warp + PhysicsNeMo
   `Mesh` / `DomainMesh`. Output is a `.pdmsh` corpus per geometry.
2. **Recipe consumer side** (PR 2.5): `load_pretrained_backbone`
   utility in `physicsnemo.utils.checkpoint` plus a Hydra hook in the
   unified external-aero recipe so any fine-tuning run can
   inherit a pretrained backbone.
3. **Recipe pretraining flavor** (PR 2): `rel_l2` loss + `WalkSampler`
   transform + `TransolverPretrainBackbone` wrapper + new dataset
   and model YAMLs. Pretraining is now a CLI flavor of the unified
   recipe (`python src/train.py model=transolver_pretrain
   dataset=geopt_pretrain ...`), not a separate script.

After this work, a user can:

```bash
# Stage 1 — pretrain on a .pdmsh corpus
python src/train.py model=transolver_pretrain dataset=geopt_pretrain \
    dataset_paths.geopt_pretrain=/path/to/corpus \
    training.num_epochs=200

# Stage 2 — fine-tune from the pretrain ckpt onto DrivAerML
python src/train.py model=transolver_volume dataset=drivaer_ml_volume \
    training.pretrained_backbone=runs/<pretrain_id>/checkpoints/...mdlus \
    training.pretrained_backbone.exclude_layers='[trajectory_head]'
```

## Status — paused 2026-05-23

- **Round 1 (data-gen):** GREEN with two yellow carry-forwards (G2
  ShapeNet-corpus leg, G4 H100 throughput rerun). All M1, M2, M3
  exit criteria met on this CPU host.
- **PR 2.5 (backbone-only loader + hook):** GREEN. 15 tests.
- **PR 2 (recipe pretraining flavor):** GREEN. End-to-end smoke
  verifies the full forward → loss → backward path.
- **Tier-1 review cleanup:** GREEN. 4/5 review fixes landed; 1
  abandoned with documented rationale (T1.4, see
  `reviews-deferred.md` §2).

**Test counts on this CPU host** (M2 macOS arm64, Python 3.13):

```
PNM_GEOPT_REF=/Users/carmelog/Projects/PhysicsNeMo/external-repos/GeoPT \
PNM_PR2_E2E_SMOKE=1 \
python -m pytest test/experimental/pnm_pretraining/ \
    test/utils/test_load_pretrained_backbone.py \
    examples/cfd/external_aerodynamics/unified_external_aero_recipe/tests/test_loss_equivalence.py \
    examples/cfd/external_aerodynamics/unified_external_aero_recipe/tests/test_pretraining_smoke.py
# → 78 passed, 2 skipped (DDP env-gated, PNM_M3_FULL_BENCH env-gated)
```

## Branch layout

```
exp-geopt-datagen-r1/
├── CURRENT-STATE.md                   ← you are here
├── geopt-datagen-round1-plan.md       ← round-1 master plan + progress log + improvements catalog
├── pr2-recipe-extension-plan.md       ← PR-2 plan (graft pretraining onto unified recipe)
├── reviews-deferred.md                ← deferred review findings, Tier-2 + Tier-3 verdicts
├── reports/
│   ├── m1-kernel-parity.md            ← M1 milestone report (kernel parity)
│   ├── m1-bvh-build-vs-query.md       ← M1 throughput bench
│   ├── m2-composite-parity.md         ← M2 milestone report (constrained walk parity)
│   └── m3-pdmsh-round-trip.md         ← M3 milestone report (.pdmsh round-trip)
├── physicsnemo/
│   ├── experimental/pnm_pretraining/  ← all new experimental code
│   │   ├── ops/
│   │   │   ├── mesh_ray_intersection.py    (M1: new FunctionSpec)
│   │   │   └── constrained_walk.py         (M2: GeoPT-specific composite)
│   │   ├── data/
│   │   │   ├── transforms.py               (M3: alignment + WalkSampler)
│   │   │   └── builder.py                  (M3: build_pretraining_sample)
│   │   └── models/
│   │       └── backbone.py                 (PR 2: TransolverPretrainBackbone + TrajectoryHead)
│   └── utils/checkpoint.py            ← PR 2.5: load_pretrained_backbone added (~310 net new LOC)
├── examples/cfd/external_aerodynamics/unified_external_aero_recipe/
│   ├── src/loss.py                    ← PR 2: rel_l2 LossType added
│   ├── src/train.py                   ← PR 2.5: cfg.training.pretrained_backbone hook
│   ├── conf/model/transolver_pretrain.yaml      ← PR 2: pretraining model template
│   ├── datasets/geopt_pretrain.yaml             ← PR 2: pretraining dataset config
│   ├── datasets/dataset_paths.yaml              ← PR 2: registers `geopt_pretrain` path key
│   └── tests/
│       ├── test_loss_equivalence.py             ← PR 2: rel_l2 test cases added
│       └── test_pretraining_smoke.py            ← PR 2: env-gated e2e smoke
├── test/experimental/pnm_pretraining/ ← unit tests for everything in experimental/pnm_pretraining
├── test/utils/test_load_pretrained_backbone.py  ← PR 2.5 unit tests
└── scripts/
    └── bench_bvh_build_vs_query.py              ← M1 throwaway throughput script
```

## Code surface

`git diff origin/main --shortstat` → **41 files changed,
10,670 insertions(+), 9 deletions(-)** across 16 commits.

By bucket (Python + YAML, non-blank-non-pure-comment SLOC):

| Category | SLOC | Notes |
|---|---:|---|
| `physicsnemo/` production | 2,538 | new experimental package + load_pretrained_backbone |
| `examples/` recipe | 130 | YAML configs + train.py hook + loss.py rel_l2 |
| `test/` (our tests) | 2,317 | unit tests for everything we built |
| `examples/.../tests/` | 364 | rel_l2 cases + e2e smoke |
| `scripts/` throwaway | 325 | BVH bench |
| **Total code SLOC** | **5,674** | + 1,767 lines of plan / report markdown |

Test-to-code ratio ≈ 0.86 (tests vs. production+recipe). Healthy.

## Geometry-direction convention (load-bearing — DON'T forget this)

Codified in `geopt-datagen-round1-plan.md` §A. **Every consumer of
the data-gen output must respect this**:

- `supervise = closest_point − query_point` (surface-pointing).
  Magnitude `|sdf|`; direction toward the nearest surface point.
  **This is the OPPOSITE sign from the GeoPT reference**
  (`GeoPT_PreTraining_Data.py:319` does `position − closest`).
  Parity tests negate before comparing.
- `signed_distance_field(use_sign_winding_number=True)` — `sdf > 0`
  means **outside** the geometry; `sdf < 0` means inside.
- Surface normals: outward unit, pointing into the fluid domain.
- Ray direction: caller-normalized unit vector, **not** auto-normalized
  by the kernel (matches `signed_distance_field`'s discipline).
- Aligned world frame: +X longitudinal, +Y vertical (mesh sits on
  Y=0 plane), +Z lateral. X-flip applied during alignment so
  conventional vehicle "front" is +X.

## Improvements catalogued over the GeoPT reference

`geopt-datagen-round1-plan.md` §8 — 20 entries (I1–I20). Most landed
during round 1 / PR 2 / PR 2.5. Status legend: `landed` / `in-progress`
/ `planned` / `deferred`.

Highlights:

- **I1** sign convention (surface-pointing supervise).
- **I2** winding-number sign-mode beats FCPW `contains()` even on
  watertight meshes (M1 measured 100% vs analytic; FCPW 86–94%).
- **I7** atomic `.pdmsh` write with preserve-on-failure semantics.
- **I12** three-tier parity-test infrastructure (analytic + trimesh +
  FCPW, vs GeoPT's zero parity tests in the released repo).
- **I14** composite-walk supervise is fp16-noise-level robust on
  non-watertight inputs (better than predicted).
- **I17** reusable `load_pretrained_backbone` (vs GeoPT's inlined
  finetune-script logic).
- **I19** pretraining as recipe flavor (no `pretrain.py`, no
  `finetune_drivaer.py`).

## Carry-forwards (do not block real-data work)

Three items in `reviews-deferred.md` §3 (Tier 2) and the round-1
plan §10 progress log's M1 closure section:

1. **ShapeNet subset.** 9 meshes (3 categories × 3 instances). Re-runs
   M1 G2 trimesh ray-cast cross-check and M2 GeoPT-reference parity
   on real meshes. Will exercise the General-variant transform's
   oversize-bbox safety hack on geometries that actually trigger it.
2. **H100 throughput rerun.** `scripts/bench_bvh_build_vs_query.py
   --device cuda` to close M1 G4 (100× speedup gate) and M3 G13 (60s
   wall-clock budget on full-config `.pdmsh` write).
3. **Op-level mesh-build hoist (I15).** Refactor
   `signed_distance_field_impl` and `constrained_walk_step` to
   accept a pre-built `wp.Mesh` handle. Builder-level hoist already
   landed in M3; op-level requires modifying the FunctionSpec API.

## Deferred review findings

`reviews-deferred.md` is the consolidated ledger of three review
passes (correctness / DRY / maintainability). Tier 1 (5 fixes)
landed in commit `05ac5489`. Tier 2 (6 items) deferred with
explicit triggers (multi-worker RNG, walk-emission order, builder
split, backbone ABC, fp32 round-off mismatch, under-collection
warning). Tier 3 (verified-skip) lists 7 reviewer claims that direct
source-read disproved or downgraded.

## How to resume in a fresh session

1. **Read** this file (`CURRENT-STATE.md`).
2. **Skim** `geopt-datagen-round1-plan.md` §A (geometry convention)
   and §8 (improvements catalog).
3. **Run the test suite** to confirm the branch is in the expected
   state (78 passed / 2 skipped — see "Status" above for the exact
   command).
4. **Look at** the most recent milestone report
   (`reports/m3-pdmsh-round-trip.md`) for the current end-to-end
   schema.
5. **Pick a next step:**
   - Real-data smoke (recommended): provision the ShapeNet subset
     (carry-forward 1), regenerate a `.pdmsh` corpus on real
     geometry, run the e2e smoke against it.
   - Op-level mesh-build hoist (I15 / Tier-2 item T2.4 in
     `reviews-deferred.md`).
   - Upstreaming PRs: split the 16 commits into 3 PRs per the parent
     plan's PR boundaries (data-gen / loader / recipe pretraining).
   - Continue Tier 2 review cleanup as opportunity arises.
6. **If you change the public schema** of the `.pdmsh` files emitted
   by `build_pretraining_sample`, you MUST update:
   - The schema dump in `reports/m3-pdmsh-round-trip.md`.
   - The dataset YAML's `targets:` block in
     `examples/.../datasets/geopt_pretrain.yaml`.
   - The model YAML's `forward_kwargs:` block in
     `examples/.../conf/model/transolver_pretrain.yaml`.
   - The progress log entry in `geopt-datagen-round1-plan.md` §10.
   - The improvements table in `geopt-datagen-round1-plan.md` §8 if
     it touches a numbered improvement.

## Optional dependencies installed (NOT in core `pyproject.toml`)

Documented in `geopt-datagen-round1-plan.md` §9. All four are kept
out of core `pyproject.toml` for round 1; the round-2 decision on
`trimesh` will follow when the builder needs it as a runtime dep:

- `scipy 1.17.1` — needed transitively by `trimesh.proximity`.
- `trimesh 3.23.5` — closest-point + ray-cast reference + OBJ I/O.
- `fcpw 1.2.0` — third-party numerical baseline for parity tests.
- `pyembree 0.1.12` — faster trimesh ray backend (downgrades
  trimesh 4.x → 3.x as a side effect).

Install for fresh dev env:

```bash
VIRTUAL_ENV=$VENV uv pip install scipy fcpw pyembree
# trimesh is installed transitively by pyembree
```

## Reference repos

- **GeoPT reference**: `external-repos/GeoPT/` (parent of this
  worktree, in the workspace). Read
  `data_generation/GeoPT_PreTraining_Data.py` (688 LOC) for the CPU
  reference data-gen we ported. Key lines:
  319 (sign convention), 333 (0.99 sticking factor), 344 (surface
  re-pin), 585-608 (walk-orchestration with BASE_WALKS=10,
  PERTURB_SIGMA=0.05).
- **Parent plan**: `geopt-physicsnemo-engineering-plan.md` (in the
  parent dir, `physicsnemo-core/` root). The original engineering
  plan that round-1 + PR 2.5 + PR 2 are executing. Some sections of
  this plan have been superseded by `pr2-recipe-extension-plan.md`
  (the recipe-grafted approach replaces parent plan §1.4 PR 2 + PR 3
  scripts).

## What WOULD invalidate this snapshot

- Anyone changing `physicsnemo/utils/checkpoint.py` upstream.
- Anyone changing the unified external-aero recipe's `forward_kwargs`
  or `extract_targets` contract upstream.
- Anyone changing PhysicsNeMo's `Mesh` / `DomainMesh` constructor
  signatures (the builder relies on dict→TensorDict coercion).
- Anyone changing the FunctionSpec API.

If any of those happen, the failure mode will be a test-suite
regression on first re-run; the diff will be obvious.
