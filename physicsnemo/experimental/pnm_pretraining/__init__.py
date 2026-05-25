# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experimental GeoPT-style pretraining utilities.

This subpackage hosts the components of a GeoPT-style lifted-pretraining
pipeline grafted onto PhysicsNeMo's unified external-aerodynamics
training recipe. APIs here are experimental and subject to change.

Subpackages
-----------
``ops``
    GPU-accelerated geometry primitives:

    - ``mesh_ray_intersection`` — ``MeshRayIntersection`` ``FunctionSpec``
      mirroring the existing ``SignedDistanceField`` template.
    - ``constrained_walk`` — fused Warp kernel implementing the
      sticking-boundary random walk that produces GeoPT-style
      lifted supervision targets, plus ``generate_walks`` for
      orchestrating the 10-base + 90-jittered walk layout.

``data``
    Dataset construction:

    - ``transforms.align_mesh_geopt_general`` + ``AlignmentRecord`` —
      port of GeoPT's General-variant ``transform_mesh`` with corrected
      named fields (replaces GeoPT's misleading ``(z_min, x_avg, y_avg,
      scale)`` tuple).
    - ``transforms.WalkSampler`` — ``MeshTransform`` that lifts one
      walk per sample from ``interior.global_data`` to
      ``interior.point_data`` so the recipe's ``extract_targets``
      consumes it natively.
    - ``builder.build_pretraining_sample`` — per-geometry orchestrator
      emitting one ``.pdmsh`` ``DomainMesh`` per call. Atomic write via
      ``save_pretraining_sample(..., atomic=True)``.

``models``
    Model wrappers:

    - ``backbone.TransolverPretrainBackbone`` — wraps
      ``physicsnemo.models.transolver.Transolver`` with a
      ``TrajectoryHead`` MLP for lifted-pretraining targets. Subclasses
      ``physicsnemo.Module`` for ``.mdlus`` serialization and
      compatibility with ``physicsnemo.utils.checkpoint.load_pretrained_backbone``.

Recipe integration
------------------
Pretraining and fine-tuning are CLI flavors of
``examples/cfd/external_aerodynamics/unified_external_aero_recipe/src/train.py``:

.. code-block:: bash

    # Stage 1 — pretrain on a .pdmsh corpus
    python src/train.py model=transolver_pretrain dataset=geopt_pretrain ...

    # Stage 2 — fine-tune from the pretrain checkpoint
    python src/train.py model=transolver_volume dataset=drivaer_ml_volume \
        training.pretrained_backbone=runs/<pretrain_id>/checkpoints/...mdlus \
        training.pretrained_backbone.exclude_layers='[trajectory_head]'

The ``training.pretrained_backbone:`` Hydra hook is provided by
``physicsnemo.utils.checkpoint.load_pretrained_backbone``.

Geometry-direction convention (load-bearing)
--------------------------------------------
``supervise = closest_point − query_point`` (surface-pointing), which
is the **opposite** sign convention from the GeoPT reference. See
``geopt-datagen-round1-plan.md`` §A for the full spec — every consumer
of the data-gen output (loss, head, validators) must respect it.

Documentation
-------------
- ``CURRENT-STATE.md`` (worktree root) — single-file orientation.
- ``geopt-datagen-round1-plan.md`` — round-1 plan, progress log,
  improvements catalog (I1–I20).
- ``pr2-recipe-extension-plan.md`` — PR-2 recipe-extension plan.
- ``reviews-deferred.md`` — review findings ledger.
- ``reports/m{1,2,3}-*.md`` — milestone reports.
"""
