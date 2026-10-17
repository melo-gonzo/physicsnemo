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

r"""ISLA: Invariant Slice Attention.

An O(d)/SE(d)-equivariant soft-slice transformer (any spatial dimension d >= 2;
the program's datasets are two- and three-dimensional) for steady boundary-value PDE
surrogates on the mesh-native API of the mesh models (GLOBE's): one case at a
time, the boundary goes in as a mapping of named
:class:`~physicsnemo.mesh.Mesh` objects (each a ``Mesh[d-1, d]``; its cells
are the tokens, with their centroids, normals and effective measures: every
cell for a surface task, a sample of the full-resolution cells drawn inside the
model for an interior task), the
problem's data as *named fields of declared tensor rank*: global inputs in a
TensorDict (a direction, a diffusivity, a Reynolds number), per-cell boundary
inputs in the meshes' ``cell_data`` (a prescribed traction, a Dirichlet
trace, a condition tag) and, for an interior task, per-prediction-point
inputs in the prediction Mesh's ``point_data`` (a material label, a source
term). A point-cloud ``Mesh`` comes out whose ``point_data`` holds fields of
declared rank: scalars, vectors, rank-2 tensors, up to rank 4, each exactly
covariant. The name states the design principle: the attention operates only
on invariants of the per-point vector set :math:`\{r_i, n_i, \hat g_1, \dots,
\hat g_K, v_{i1}, \dots\}` (scaled position, unit normal, unit global vectors,
the token's own tensor inputs), and the frame is re-attached only at the
heads, so exact rotation, reflection and translation covariance is paid once
at the network's edges instead of in every layer.

**Rank specs.** Inputs and outputs are declared in the field-schema grammar
shared by every mesh model (:mod:`physicsnemo.mesh.fields`): a mapping from a
field name to a field declaration, a :class:`~physicsnemo.mesh.fields.RankSpec`
or a mapping of values ``{"rank": r, "symmetric": bool, "parity": "even" |
"odd"}`` (``r`` = 0 scalar, 1 vector, 2 rank-2 tensor, ...); a mapping of
mappings is a nested group, and a flat name may contain ``"."`` for the
same nesting (``{"surface": {"pressure": {"rank": 0}}}`` and
``{"surface.pressure": {"rank": 0}}`` declare the same field). Each parsed
declaration is a :class:`~physicsnemo.mesh.fields.FieldSchema`. ``symmetric`` (rank
:math:`\ge 2`) means fully symmetric under index permutation and
``parity="odd"`` (outputs only) a pseudotensor that flips sign under a
reflection (a vorticity). One name declares one field; several independent
scalars are several rank-0 names. The same grammar serves the constructor's
``output_schema``, ``boundary_schema``, ``query_schema`` and ``global_schema``, and
the same dotted names address the fields in the meshes' ``cell_data`` and
``point_data`` and in ``global_data`` (``td["surface", "pressure"]``).

**Inputs of any rank enter through invariants.** A rank-1 boundary input
:math:`v_i` at a token contributes the seed scalars :math:`v_i \cdot \hat
n_i`, :math:`v_i \cdot \hat g_k`, :math:`|v_i|` and its dot products with
the token's other vector inputs, and, inside every slice layer where the
token-to-anchor direction :math:`\hat\rho_{is}` exists, the contraction
:math:`v_i \cdot \hat\rho_{is}` next to the existing :math:`\hat n_i \cdot
\hat\rho_{is}` and :math:`\hat g_k \cdot \hat\rho_{is}` (one shared helper
contracts every per-token vector with the anchor direction; the normal is
just the first of them). A rank-2 input :math:`T_i` contributes the seed
scalars :math:`a^{\mathsf T} T_i b` over the frame vectors :math:`a, b \in
\{\hat n_i, \hat g_k\}`, :math:`\operatorname{tr} T_i` and
:math:`\|T_i\|_F`, and in the layers :math:`\hat\rho^{\mathsf T} T_i
\hat\rho`, :math:`\hat\rho \cdot T_i \hat n_i` and :math:`\hat\rho \cdot
T_i \hat g_k` (the last two are contractions of the derived true vectors
:math:`T_i \hat n_i`, :math:`T_i \hat g_k` with the anchor direction, so they
ride on the same helper). Only true vectors appear on the input side, no
cross products, so every input invariant is invariant under improper
rotations as well; pseudotensor *inputs* are therefore refused. Rank-0
inputs are the scalars themselves. Inputs of rank above 2 are refused (the
contraction set grows combinatorially and no dataset of the program has
one).

**Outputs of any rank leave through an equivariant basis.** One linear head
on the normalized final token state emits invariant coefficients; a rank-
:math:`r` output is the contraction of those coefficients with tensor
products of :math:`r` vectors from the token's basis: the *true* vectors
:math:`t_1, \dots, t_m` (the global directions, the normal, the radial
direction from the token's soft anchor and the :math:`e_\theta`
complements; :math:`m = 2K + 3` in every dimension) for even parity, and
products with exactly one *pseudovector* factor for odd parity, the
pseudovectors being the Levi-Civita construction from the primary true vectors
:math:`\hat g_k, \hat n_i, \hat u_i`: their pairwise cross products in three
dimensions (:math:`p = \binom{K+2}{2}`), their perpendiculars in two
(:math:`p = K + 2`); odd parity is refused in four or more dimensions, where a
pseudovector needs :math:`d - 1` vectors. Plus the isotropic tensors of that
rank and parity (the identity at rank 2, the three products of Kronecker
deltas at rank 4, the Levi-Civita symbol, of rank :math:`d`, for odd parity).
A symmetric spec uses the symmetrized basis with one
coefficient per unordered multi-index. Coefficient counts are in the class
docstring; ranks above 4 are refused.

**The frame.** ISLA has no frame origin: positions enter only as
point-to-anchor differences, and the one length scale is the
measure-weighted root-mean-square pairwise distance of the sample, a
normalized two-point statistic that is invariant to the weights' overall
scale and converges under refinement of any boundary with a finite second
moment, fractal ones included. No centroid, no per-dataset reference length
and no external frame appears anywhere in the forward pass, so the frame
carries no information about where the mesher placed its cells.

**Routing.** Each block routes tokens to slices by bounded cosine logits:
:data:`ROUTING_TEMPERATURE` times the cosine between the normalized token
state and a prototype per slice, so every logit lies in :math:`[-T, T]` and
no block can route every point to one slice however its weights grow. The
prototypes are synthesized per sample: learned seeds attend over the token
states with a measure-weighted softmax, the attended term is centered across
the slices and smoothly bounded to a learned fraction of the seeds' spread,
and the result is added to the seeds (:class:`_RoutingLogits`). The
quadrature measure enters the routing as a log-space bias, so slice states
are measure-weighted (quadrature) means; with no measures the weights are
uniform.

**Surface and interior tasks.** ``interior=False`` (a surface task) predicts
at the boundary cells themselves. ``interior=True`` (an interior task)
predicts at the ``prediction_points`` given to ``forward`` (a point-cloud
``Mesh``, or a bare ``(N, d)`` tensor when ``query_schema`` declares no
fields): they join the encoder as interacting tokens next to the boundary
cells and the outputs are returned for them only. Per-prediction-point inputs
(declared in ``query_schema``, read from the prediction Mesh's ``point_data``)
are carried by those tokens; per-cell boundary inputs (declared in
``boundary_schema``) by the boundary cells, with zeros on the
prediction-point tokens. The prediction-point
tokens' total routing weight is a learned fraction of the total boundary
measure, split equally over them, so it is invariant to re-representing the
same measure (splitting a boundary cell in two changes nothing) and scales
like an area. With ``query_independent=True`` (ISLA-DTT) the prediction-point
tokens are passive instead: they carry no routing weight, so at every depth they
read the slices, anchors and prototypes the boundary formed and never write into
them, and after every block each one also reads every boundary cell
(:mod:`~physicsnemo.experimental.models.isla.tree_read`: a learned kernel of the
pair's relative geometry and an attention over the cells' states at that depth,
weighted by each cell's share of the boundary measure and evaluated through a
Barnes-Hut forest over the cells). The prediction at a point is then a function of
that point and the boundary alone (any request can be split into chunks with the
same result). The two tasks own different parameter sets (the interior task
adds the prediction-point embedding, type offset and routing mass), which is
why the choice is a constructor argument: a parameter that takes no part in
a forward pass stops DDP.

**Interior geometry.** A prediction-point token needs a direction (its
"normal" in every invariant) and a distance to the wall. Both are computed
inside the model from the boundary *surface*, the union of the cells of
``boundary_meshes`` at full resolution, never taken as inputs: for each
prediction point :math:`x_q` the exact closest point :math:`p_q` on the
surface's cells (point to face, never point to centroid) gives the distance
:math:`d_q = \pm |x_q - p_q|`, signed by the pseudo-normal of the closest
feature, so positive on the side the cells' normals point to, and the
direction :math:`u_q`: the cell's normal, oriented toward the query, when
:math:`p_q` is interior to a cell, else :math:`(x_q - p_q) / |x_q - p_q|`
(:func:`exact_boundary_geometry`). :math:`d_q / L` with its signed logarithm
enters the token's embedding as a built-in rank-0 channel ahead of the fields
declared in ``query_schema``. :math:`(d_q, u_q)` are functions of the surface
and its orientation alone: the same for every draw of the boundary tokens,
invariant to splitting a cell and converging as :math:`O(h^2)` under refinement
of the surface, like a stored signed-distance field computed from the same
surface. A dataset's stored signed-distance field or its gradient enter, if
wanted, as ordinary ``query_schema`` fields (rank 0 / rank 1) like any other
PDE data. The search is a Warp BVH in three dimensions (a torch custom op) and
an exact brute force over the segments in two (the interior task is
implemented in two and three dimensions); the closest point is rebuilt from its
cell's vertices, so both quantities are differentiable, and both are built from
differences of positions, so every covariance contract holds.

**Boundary tokens of the interior task.** The boundary tokens are a sample of
the same cells, drawn inside the forward pass
(:func:`~physicsnemo.experimental.models.isla.sampling.sample_boundary_cells`):
``boundary_tokens`` (default 200,000) tokens per case, shared over the named
boundaries by water filling (a boundary with fewer cells than its share is kept
whole and its unused share goes to the others), each cell drawn with
probability :math:`\pi_i` proportional to its measure (capped at one) by
systematic sampling along the Morton order of the cell centroids. The count is
exactly :math:`\min(\text{cells}, \text{boundary\_tokens})`, a function of the
cell counts only, so the draw keeps static shapes and traces into the one
whole-model graph; each kept cell's measure is its Horvitz-Thompson weight
:math:`a_i / \pi_i`, so every measure-weighted read is an unbiased quadrature
of the full surface, exact on a boundary kept whole; and the sample is spread
along the space-filling curve, so every part of the surface is represented. In
training the draw's start is uniform from the global torch RNG
(``torch.manual_seed`` reproduces it), in evaluation it is fixed, so inference
is deterministic. A surface task reads every cell it is given (its outputs are
predicted there) and samples none.

**Global inputs.** Rank-1 global inputs are the :math:`K` global directions:
each is normalized to a unit direction inside the model (a direction is what
the invariants consume; a physically meaningful magnitude belongs among the
rank-0 global inputs), enters the seed through :math:`\hat n_i \cdot \hat
g_k`, the layers through :math:`\hat g_k \cdot \hat\rho_{is}`, and the head
basis as :math:`\hat g_k` with the complements of :math:`(\hat u_i, \hat
g_k)`. Rank-0 global inputs are appended to every token's seed. Global
inputs of rank 2 are refused (broadcast the tensor to the boundary cells and
declare it as a boundary input instead).

**Kernels.** The per-layer geometry region has two exact implementations
of the same arithmetic: the eager PyTorch region, which rebuilds its
intermediates in backward instead of storing them, and the fused one (one
Triton kernel per direction that never materializes the (B, N, S, .) tensor;
CUDA, float32/float64 geometry, one to four per-point vectors counting the
global directions and the token's rank-1 inputs together, at most 1024
slices, no rank-2 inputs). At the first forward pass the model picks the
fused kernel when the inputs are on CUDA, Triton is importable and the
constraints hold, and the eager region otherwise, logging the choice once
(:meth:`ISLA.resolve_geo_kernel`; the ``geo_kernel`` attribute, ``"auto"``,
can be set to ``"eager"`` or ``"fused"`` to force one: an execution choice,
not a constructor argument, so a checkpoint does not store it and a loaded
model starts at ``"auto"``); at 200,000 cells the
fused kernel runs the training step in about a third of the eager time and
memory. The per-token
rank-2 contractions (rho^T T rho) are the standing kernel follow-up; until
then a model with rank-2 boundary or query inputs runs the eager region.

Every constructor argument and every ``forward`` input is keyword-only, so a
call reads as the recipe's ``forward_kwargs`` mapping does and models can be
swapped without positional bookkeeping. ``forward(boundary_meshes=...,
prediction_points=..., global_data=...)`` returns a point-cloud ``Mesh``; see
:meth:`ISLA.forward`.

**Data layout.** Inside ``forward`` the tokens stay TensorDicts: each boundary
mesh's cells (centroid, normal, measure and the declared ``cell_data`` fields)
and the prediction points are TensorDicts concatenated into one token state over
``(batch, token)``, the field inputs enter the invariants and the head's
coefficients leave as the output fields through cached, schema-ordered
:class:`~tensordict.PackLayout` objects, and the output ``Mesh`` is built from them.
With ``query_independent=False`` the whole model, ``Mesh`` inputs and output
included, compiles with ``torch.compile(fullgraph=True, dynamic=True)`` into one
graph for every boundary and query count: surface and interior tasks, boundary
inputs up to rank 2 (``test_isla_compile.py``). ISLA-DTT does not: its boundary
read sizes its pairs from the data and breaks a fullgraph trace, so it compiles
with ``compile_blocks``. The token axis itself stays dense where the arithmetic
needs it: every softmax over points spans all tokens at once (passive tokens
excluded by ``log_w = -inf``), the slice anchors are einsums over that axis, the
fused geometry kernel reads its ``(B, N, .)`` strides, and the boundary read
indexes the flattened boundary tokens; the slice blocks take plain tensors.
"""

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from jaxtyping import Float
from tensordict import TensorDict

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.experimental.models.isla.blocks import (
    _QueryTokens,
    _RoutingLogits,
    _SliceBlock,
    _softmax_over_points,
)
from physicsnemo.experimental.models.isla.geometry import (
    exact_boundary_geometry,
    rms_pairwise_distance,
)
from physicsnemo.experimental.models.isla.heads import (
    _assemble_output,
    _OutputLayout,
    _polar_complement,
    _pseudovectors,
)
from physicsnemo.experimental.models.isla.inputs import (
    _declare,
    _InputSet,
    _missing_fields,
    _pack,
    _schema_layout,
)
from physicsnemo.experimental.models.isla.sampling import sample_boundary_cells
from physicsnemo.experimental.models.isla.tree_read import (
    TREE_READ_THETA,
    _KernelBasis,
    _TreeKernelRead,
    kernel_geo_width,
    tree_read_pairs,
)
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import cell_measures
from physicsnemo.mesh.fields import FieldSchema, FieldSchemaLike

logger = logging.getLogger(__name__)

#: Numerical floor for norms and logarithms.
EPS = 1e-12
#: The interior task's default boundary-token budget per case (the reference
#: configuration's, DrivAerML Volume's 200,000 vehicle cells).
BOUNDARY_TOKENS = 200_000
#: The systematic draw's start in evaluation mode (the middle of its range), so
#: that inference is deterministic.
EVAL_START = 0.5


def _hint_token_axis(n, widths) -> None:
    """Tell torch.compile that ``k * n`` is a multiple of the token count ``n``.

    With dynamic shapes a token axis made of several inputs (boundary meshes,
    boundary and query tokens) has a length that is a sum of sizes, ``a + b``,
    and Inductor cannot prove that ``k * a + k * b`` divides by ``a + b``: its
    code generation fails with CantSplit where it tiles a flattened ``(k n,)``
    loop of a ``(B, n, k)`` tensor as ``(n, k)``. :func:`torch._check` records
    each (always true) fact as a guard, which those static checks then read.
    Skipped in eager mode, where every fact is trivially true; the test is
    ``is_compiling()`` since dynamo traces a SymInt as an int.

    The CantSplit failure is an Inductor bug in torch 2.12 and 2.13, fixed
    from torch 2.14.0 on. Its trigger in ISLA is an affine LayerNorm over a
    concatenated token axis whose output feeds both a dtype cast and a
    reduction over the hidden dimension (``norm_assign(h)`` into ``k_proj``
    and ``F.normalize``). Remove this function once the minimum torch is
    2.14.0, after checking that compiles across token counts still produce
    one graph: without these hints, torch 2.14 compiled a second graph when
    the token count doubled."""
    if torch.compiler.is_compiling():
        for k in widths:
            torch._check(k * n % n == 0)


def _log_count(n: int, like: torch.Tensor) -> float | torch.Tensor:
    """``log(n)`` of a token count; under torch.compile a 0-d tensor of ``like``'s
    dtype, since ``math.log`` of a shape would specialize the dynamic shapes on it."""
    if not torch.compiler.is_compiling():
        return math.log(n)
    return torch.full((), n, dtype=torch.float64, device=like.device).log().to(like.dtype)


def _tail(tokens: TensorDict, start) -> TensorDict:
    """The tokens of a ``(B, N)`` TensorDict from ``start`` on. Each leaf is sliced
    on its own: indexing the TensorDict (``tokens[:, start:]``) sizes the result
    through ``slice.indices``, which under torch.compile fixes a symbolic token
    count to its value (a recompile for every count)."""
    b, n = tokens.batch_size
    return tokens.apply(lambda t: t[:, start:], batch_size=[b, n - start])


class ISLA(Module):
    r"""ISLA (Invariant Slice Attention): invariant backbone, equivariant edges.

    A soft-slice transformer for steady boundary-value problems whose attention
    operates only on invariants of the per-point vector set and whose outputs of
    any rank are re-assembled in an equivariant basis at the heads, so exact
    rotation, reflection and translation covariance holds by construction (see
    the module docstring for the contracts). Every argument is keyword-only.

    **Rank specs.** ``output_schema``, ``boundary_schema``, ``query_schema`` and
    ``global_schema`` are :data:`~physicsnemo.mesh.fields.FieldSchemaLike`
    declarations, the grammar shared by every mesh model
    (:meth:`~physicsnemo.mesh.fields.FieldSchema.parse`): a field is a
    :class:`~physicsnemo.mesh.fields.RankSpec` or a mapping of values
    ``{"rank": r, "symmetric": False, "parity": "even"}``; a mapping of
    mappings is a nested group, and a flat name may contain ``"."`` for
    the same nesting (``{"surface": {"pressure": {"rank": 0}}}`` and
    ``{"surface.pressure": {"rank": 0}}`` declare the same field). Each
    parsed declaration is the :class:`~physicsnemo.mesh.fields.FieldSchema`
    attribute of the same name. ``symmetric`` means
    fully symmetric under index permutation (rank :math:`\ge 2` only);
    ``parity="odd"`` declares a pseudotensor output (outputs only). One name
    declares one field; several scalars are several rank-0 names.

    **Mesh API.** ``forward`` takes the boundary as a mapping of named
    :class:`~physicsnemo.mesh.Mesh` objects (one per boundary condition or
    part, each a ``Mesh[dim - 1, dim]`` whose cells have normals), the
    per-case data as a TensorDict, and, for an interior task, the prediction
    points as a point-cloud ``Mesh`` (or a bare ``(N, dim)`` tensor when
    ``query_schema`` declares no fields). It returns a point-cloud ``Mesh`` whose
    ``point_data`` holds the declared outputs. Details under ``forward``.

    **Execution.** The ``geo_kernel`` attribute (``"auto"``; set ``"eager"`` or
    ``"fused"`` to force one) selects how the per-layer geometry region is
    executed, not what it computes (:meth:`resolve_geo_kernel`). It is an
    execution choice, not a constructor argument: checkpoints do not store it,
    and a loaded model starts at ``"auto"``.

    Args:
        output_schema: Output fields, a rank spec per name. Each field is a leaf of
            the returned Mesh's ``point_data`` (nested per the dotted name): a
            rank-0 field has shape ``(N,)``, a rank-``r`` field ``(N, d, ...,
            d)`` with ``r`` trailing axes (``d`` the spatial dimension,
            ``dim``); a symmetric spec gives a symmetric tensor. Ranks up to
            :attr:`MAX_OUTPUT_RANK`. Each field's coefficients come from the
            shared linear head; with ``m = 2K + 3`` true basis vectors and
            ``p`` pseudovectors (``(K+2)(K+1)/2`` in three dimensions, ``K +
            2`` in two) the counts per field are ``1`` at rank 0; ``m`` (even)
            or ``p`` (odd) at rank 1; ``m^2 + 1`` (even, the identity
            included) or ``m(m+1)/2 + 1`` (even symmetric), ``2pm`` or ``pm``
            (odd) at rank 2. For ``K = 1`` (``m = 5, p = 3``): a vector costs
            5, a symmetric rank-2 tensor 16, a pseudovector 3.
        boundary_schema: Per-cell boundary fields, a rank spec per name with
            ranks 0 to 2 and even parity (a prescribed traction as
            ``{"rank": 1}``, a Dirichlet trace or a condition tag as
            ``{"rank": 0}``). One declaration
            applies to every boundary mesh: each mesh's ``cell_data`` must
            carry every declared field (per-boundary declarations are a
            follow-up). Their invariants widen the boundary tokens' seed;
            prediction-point tokens carry zeros there. Rank-1 and rank-2
            inputs also contract with the token-to-anchor direction in every
            slice layer (module docstring).
        query_schema: Per-prediction-point fields of the interior task
            (``interior=True`` required), a rank spec per name with ranks 0
            to 2 and even parity, read from the prediction Mesh's
            ``point_data`` (a material label, a source term, a dataset's
            stored signed distance or its gradient). Carried by the
            prediction-point tokens through their own embedding, with the
            same invariants as boundary inputs. Rank-0 entries are lengths in
            the frame's unit: each is divided by the length unit and enters
            as ``[s, sign(s) log(|s| + eps)]``; rank-1 and rank-2 entries are
            never scaled.
        global_schema: Per-case fields, a rank spec per name with rank 0
            (appended to every token's seed) or rank 1 (a global direction,
            normalized to unit length inside the model). The ``K`` rank-1
            entries, in declaration order, are the global directions of the
            geometry region and the head basis. Read from ``global_data`` by
            name; extra leaves there are ignored.
        interior: The task structure. ``False`` (default): a surface task,
            the outputs are predicted at the boundary cells themselves;
            ``forward`` takes no ``prediction_points`` and the model has no
            prediction-point parameters. ``True``: an interior task, the
            outputs are predicted at ``prediction_points``, which join the
            encoder as interacting tokens; ``forward`` requires them. The two
            are different parameter sets (a parameter without a gradient stops
            DDP), so the choice is made at construction.
        boundary_tokens: Interior task only: the number of boundary tokens per
            case (default 200,000, the reference configuration's). ``forward``
            takes each named boundary at full resolution and draws its tokens
            from the cells inside the model
            (:func:`~physicsnemo.experimental.models.isla.sampling.sample_boundary_cells`):
            the budget is shared over the named boundaries by water filling
            (equal shares, a boundary with fewer cells than its share kept
            whole and its unused share given to the others), the cells are
            drawn with probability proportional to their measure (capped at
            one) by systematic sampling along the Morton order of their
            centroids, and each kept cell carries its Horvitz-Thompson measure
            (its measure over its inclusion probability). The token count is
            exactly ``min(total cells, boundary_tokens)``; a boundary with all
            its cells kept is read as given. In training the draw's start is
            uniform from the global torch RNG (``torch.manual_seed`` reproduces
            it), in evaluation (``model.eval()``) it is fixed, so inference is
            deterministic. A surface task predicts at every cell it is given
            and samples none: it takes only the default.
        query_independent: Interior task only; the choice between ISLA and
            ISLA-DTT. ``False`` (default, ISLA): the prediction points interact,
            their total routing weight a learned fraction of the boundary
            measure, and a prediction depends on the other points of the
            request. ``True`` (ISLA-DTT): passive prediction points (zero
            routing weight, no routing-mass parameter); they pass through every
            block with the same weights and read its slices at matched depth,
            and after every block they also read every boundary cell (module
            docstring), so each prediction depends on its own point and the
            boundary only. The boundary read costs about two to four times the
            training step of ISLA and breaks the whole-model fullgraph compile
            (use ``compile_blocks``); rank-2 boundary or query inputs are not
            supported with it.
        hidden: Token width of the slice blocks. The defaults ``hidden=512``,
            ``n_layers=12`` and ``n_slices=512`` are the reference
            configuration (the most accurate under 80 GiB per GPU on DrivAerML
            Surface and the HiLift 210-wing rung at 200,000 cells); the
            reference training rate is 3.75e-4 (the width rule 1e-3 x 192 / hidden).
        n_layers: Number of slice-attention blocks.
        n_slices: Number of soft slices (data-adaptive anchors) per block.
        compile_blocks: Run every slice block through :func:`torch.compile`
            (default ``False``). The compiled function is the block's forward,
            built at the first forward pass, so parameter names, checkpoints and
            the eager path are unchanged; the fused geometry kernel stays a
            custom-op leaf. Compiles once per distinct token count (the
            reporting protocol's fixed sample compiles once); with a token count
            that changes every case, leave it off. On four GB300 ranks at
            200,000 cells it cut the training step by 19 to 35% and peak memory
            by 27% at bfloat16-roundoff differences (PERF-3, 2026-10-01).
        dim: Spatial dimension of the domain (default 3). Every boundary mesh
            is a ``Mesh[dim - 1, dim]`` and every prediction point has ``dim``
            coordinates. Every geometric quantity is built for ``dim``
            dimensions and the head's basis follows it (pseudovectors by the
            Levi-Civita construction: cross products in three dimensions,
            perpendiculars in two; odd-parity outputs are refused for ``dim >=
            4``). The fused Triton geometry kernel is written for three
            dimensions; other dimensions run the eager region. The
            interior task (its exact boundary geometry) is implemented in two
            and three dimensions. A
            two-dimensional problem is passed with ``dim=2`` (polyline
            boundaries) rather than embedded with a zero third coordinate.

    Output contract: a field declared with rank ``r`` and even parity
    transforms as a true rank-``r`` tensor under every orthogonal map of the
    inputs (``T -> R^{(x)r} T``; rotations and reflections alike) and is
    invariant under translations; an odd field picks up the extra sign
    ``det R``. Components in any frame are only ever a view of such a field.
    """

    @dataclass
    class MetaData(ModelMetaData):
        jit: bool = False
        amp: bool = True

    #: Largest output rank; the ordered basis grows as ``m**rank``.
    MAX_OUTPUT_RANK = 4
    #: Largest per-token input rank; see the module docstring.
    MAX_INPUT_RANK = 2

    ### 0.2.0 renamed the field declarations after the schemas they hold; 0.3.0 made
    ### the slice sublayer read the normalized state (same parameters, new forward);
    ### 0.4.0 removed the constructor arguments that only ever took one value or
    ### served ended experiments (same parameters and forward for every
    ### configuration it still builds); 0.5.0 gave the prediction points the exact
    ### geometry of the boundary surface in place of the nearest boundary token's
    ### and drew the interior task's boundary tokens inside the model from the
    ### full-resolution surface (boundary_tokens; same parameters, new forward).
    __model_checkpoint_version__ = "0.5.0"
    _RAW_READS = (
        " Its slice blocks were trained to pool and broadcast the raw residual stream; "
        "this version reads the normalized state there, so its outputs differ from "
        "the ones it was trained to give."
    )
    _REMOVED = (
        " Its removed constructor arguments (mlp_ratio, geo_checkpoint, geo_kernel, eps "
        "and the kernel_read options) are dropped; ISLA-DTT's kernel_read='mean' is "
        "query_independent=True."
    )
    _NEAREST_GEOMETRY = (
        " An interior model's prediction points were trained on the distance and "
        "direction to the nearest boundary token, with the boundary tokens drawn "
        "before the model; this version gives them the exact distance and direction "
        "to the full-resolution boundary surface and draws its boundary tokens from "
        "it (boundary_tokens), so its interior outputs differ from the ones it was "
        "trained to give."
    )
    __supported_model_checkpoint_version__ = {
        "0.1.0": (
            "Loading a 0.1.0 ISLA checkpoint: renaming its field declarations "
            "outputs, boundary_inputs, query_inputs and global_inputs to "
            "output_schema, boundary_schema, query_schema and global_schema."
            + _RAW_READS
            + _REMOVED
            + _NEAREST_GEOMETRY
        ),
        "0.2.0": "Loading a 0.2.0 ISLA checkpoint." + _RAW_READS + _REMOVED + _NEAREST_GEOMETRY,
        "0.3.0": "Loading a 0.3.0 ISLA checkpoint." + _REMOVED + _NEAREST_GEOMETRY,
        "0.4.0": "Loading a 0.4.0 ISLA checkpoint." + _NEAREST_GEOMETRY,
    }
    _RENAMED_ARGS = {
        "outputs": "output_schema",
        "boundary_inputs": "boundary_schema",
        "query_inputs": "query_schema",
        "global_inputs": "global_schema",
    }

    @classmethod
    def _backward_compat_arg_mapper(
        cls, version: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Rename the field declarations of a 0.1.0 checkpoint's arguments, and drop
        the arguments 0.4.0 removed where the checkpoint used the one value this
        version builds; any other value of a removed argument is an error naming it.
        """
        args = super()._backward_compat_arg_mapper(version, args)
        if version == "0.1.0":
            for old, new in cls._RENAMED_ARGS.items():
                if old in args:
                    args[new] = args.pop(old)

        def refuse(setting: str, reason: str):
            raise ValueError(
                f"this {version} ISLA checkpoint was built with {setting}, which ISLA "
                f"{cls.__model_checkpoint_version__} no longer has ({reason})"
            )

        ### Implementation choices with the same arithmetic.
        args.pop("geo_checkpoint", None)
        args.pop("geo_kernel", None)
        for name, value in (("mlp_ratio", 4), ("eps", EPS)):
            stored = args.pop(name, value)
            if stored != value:
                refuse(f"{name}={stored!r}", f"it is fixed at {value!r}")
        ### The boundary read: ISLA-DTT's form ("mean", no radius, the forest at
        ### theta 1 after every block) is query_independent=True; every other form ended.
        read = args.pop("kernel_read", None)
        radius = args.pop("kernel_read_radius", None)
        blocks = args.pop("kernel_read_blocks", None)
        theta = args.pop("kernel_read_theta", None)
        if read is None:
            if args.get("query_independent", False):
                refuse(
                    "query_independent=True and kernel_read=None",
                    "query_independent=True is ISLA-DTT, whose points also read the boundary cells",
                )
            return args
        if read != "mean":
            refuse(f"kernel_read={read!r}", "the boundary read is ISLA-DTT's 'mean' form")
        if radius is not None:
            refuse(f"kernel_read_radius={radius!r}", "the boundary read has no radius")
        if theta != TREE_READ_THETA:
            refuse(f"kernel_read_theta={theta!r}", f"the read runs through the forest at theta {TREE_READ_THETA}")
        if blocks not in (None, args.get("n_layers", 12)):
            refuse(f"kernel_read_blocks={blocks!r}", "the read follows every block")
        return args

    def __init__(
        self,
        *,
        output_schema: FieldSchemaLike,
        boundary_schema: FieldSchemaLike | None = None,
        query_schema: FieldSchemaLike | None = None,
        global_schema: FieldSchemaLike | None = None,
        interior: bool = False,
        query_independent: bool = False,
        boundary_tokens: int = BOUNDARY_TOKENS,
        hidden: int = 512,
        n_layers: int = 12,
        n_slices: int = 512,
        compile_blocks: bool = False,
        dim: int = 3,
    ) -> None:
        super().__init__(meta=self.MetaData())
        ### compile_blocks: run each slice block through torch.compile (built lazily at
        ### the first forward pass, on the block's bound forward so parameter names and
        ### checkpoints are unchanged). The blocks see only the (B, n_tokens, .) tensors,
        ### so a fixed token count per case compiles once; the mesh-facing forward,
        ### whose vertex count varies per case, stays eager. The fused geometry kernel
        ### is a registered custom op and stays a leaf. Measured 2026-10-01 (PERF-3,
        ### four GB300 ranks, 200,000 cells): step 221 -> 180 ms and peak memory
        ### 44.9 -> 32.9 GiB on DrivAerML Surface, 475 -> 308 ms and 88.5 -> 64.5 GiB
        ### on DrivAerML Volume, losses at bfloat16 roundoff of the eager path.
        self.compile_blocks = bool(compile_blocks)
        self._block_forwards: list | None = None
        ### The spatial dimension of the points. Every geometric quantity is built
        ### for d dimensions; the head's basis and the coefficient count depend on
        ### d (pseudovectors exist by the Levi-Civita construction: perpendiculars
        ### in two dimensions, cross products in three), which is why d is a
        ### constructor argument and forward checks the points.
        if int(dim) < 2:
            raise ValueError(f"dim must be at least 2, got {dim!r}")
        self.dim = int(dim)
        ### Every input and output is a named field of declared rank; the grammar
        ### is the mesh models' shared one (physicsnemo.mesh.fields), flattened to
        ### dotted names here.
        self.output_schema = _declare(
            "output_schema",
            output_schema,
            max_rank=self.MAX_OUTPUT_RANK,
            allow_odd=True,
        )
        if not self.output_schema:
            raise ValueError("output_schema must declare at least one field")
        self.boundary_schema = _declare(
            "boundary_schema",
            boundary_schema,
            max_rank=self.MAX_INPUT_RANK,
            allow_odd=False,
        )
        self.query_schema = _declare(
            "query_schema", query_schema, max_rank=self.MAX_INPUT_RANK, allow_odd=False
        )
        self.global_schema = _declare(
            "global_schema", global_schema, max_rank=1, allow_odd=False
        )
        ### The task structure: a surface task predicts at the boundary cells and
        ### has no prediction-point parameters; an interior task predicts at
        ### prediction points that join the encoder as tokens.
        self.interior = bool(interior)
        if self.query_schema and not self.interior:
            raise ValueError(
                f"query_schema {sorted(self.query_schema)!r} declares per-prediction-"
                "point fields of the interior task; declare them with interior=True"
            )
        if self.interior and self.dim not in (2, 3):
            raise NotImplementedError(
                "the interior task's built-in geometry (the exact distance to the boundary "
                f"surface) is implemented in two and three dimensions; got dim={self.dim}"
            )
        self.query_independent = bool(query_independent)
        if self.query_independent and not self.interior:
            raise ValueError(
                "query_independent=True concerns the prediction points of the interior "
                "task; build with interior=True (a surface task predicts at its input cells)"
            )
        if isinstance(boundary_tokens, bool) or int(boundary_tokens) != boundary_tokens or boundary_tokens < 1:
            raise ValueError(f"boundary_tokens must be a positive integer; got {boundary_tokens!r}")
        if not self.interior and boundary_tokens != BOUNDARY_TOKENS:
            raise ValueError(
                "boundary_tokens is the interior task's budget of boundary tokens; a surface "
                "task (interior=False) predicts at every boundary cell it is given and samples "
                f"none, so it takes the default ({BOUNDARY_TOKENS}); got {boundary_tokens!r}"
            )
        self.boundary_tokens = int(boundary_tokens)
        ### Checkpoints store the constructor arguments as JSON. A FieldSchema
        ### is a dict and serializes as it stands; recording the parsed schemas
        ### keeps the metadata JSON-safe whatever container the caller passed
        ### (a Hydra DictConfig, for instance).
        for key in ("output_schema", "boundary_schema", "query_schema", "global_schema"):
            self._args["__args__"][key] = getattr(self, key) or None
        ### The problem's global data are zero or more global DIRECTIONS (each
        ### enters as a unit direction: one seed term n.g_k per token, one
        ### direction cosine rel.g_k per point-anchor pair, and the head basis
        ### gains g_k with the spherical complements of (u, g_k)) and zero or more
        ### global SCALARS (PDE parameters; appended to every token's seed).
        self.global_vector_names = [
            n for n, s in self.global_schema.items() if s.rank == 1
        ]
        self.global_scalar_names = [
            n for n, s in self.global_schema.items() if s.rank == 0
        ]
        self.n_global_vectors = K = len(self.global_vector_names)
        self.n_global_scalars = len(self.global_scalar_names)
        ### Each kind's columns in schema order (cached layouts, see _global_fields).
        self._global_vector_layout = _schema_layout({k: (self.dim,) for k in self.global_vector_names})
        self._global_scalar_layout = _schema_layout({k: () for k in self.global_scalar_names})
        ### geo_kernel names the implementation of the per-layer geometry region,
        ### not its arithmetic. "eager" is the PyTorch region (its (B,N,S,.)
        ### intermediates rebuilt in backward); "fused" is the Triton kernel of
        ### geo_kernel.py, which reads the pre-logits once and writes the bias and
        ### mix once per direction, with the recompute built in (CUDA only). "auto"
        ### resolves at the first forward pass to "fused" wherever the fused kernel
        ### can run and to "eager" otherwise; the blocks read the resolved name.
        ### Writable: tests and debugging set "eager" or "fused" to force one.
        self.geo_kernel = "auto"
        self.n_slices = int(n_slices)
        ### The boundary cells' fields and the interior queries' fields, each set
        ### with its own invariant layout (_InputSet). Every token carries the
        ### UNION of the two block-level vector/tensor stacks (zeros where a class
        ### has no such field), so one geometry region serves boundary and query
        ### tokens alike. Rank-0 query fields are lengths in the frame's unit.
        n_frame = 1 + K  # {n_hat, g_1..g_K}
        self.bin = _InputSet(self.boundary_schema, n_frame, scaled_scalars=False, dim=self.dim)
        self.qin = _InputSet(self.query_schema, n_frame, scaled_scalars=True, dim=self.dim)
        self.n_block_vecs = self.bin.n_block_vecs + self.qin.n_block_vecs
        self.n_block_tens = self.bin.n_block_tens + self.qin.n_block_tens
        ### Seed width: the K invariants n.g_k (the only origin-free scalars of
        ### {n, g_k}; positions enter through the anchors), the boundary inputs'
        ### invariants and the global scalars.
        n_seed = K + self.bin.n_features + self.n_global_scalars
        ### With no global inputs and no other seed channel the seed is empty;
        ### every token then starts from one learned embedding and all
        ### separation comes from the slice blocks' relational anchors.
        self.constant_seed = n_seed == 0
        if self.constant_seed:
            n_seed = 1
        self.embed = nn.Sequential(
            nn.Linear(n_seed, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.blocks = nn.ModuleList(
            _SliceBlock(
                hidden,
                n_slices,
                n_global_vectors=K,
                n_block_vecs=self.n_block_vecs,
                n_block_tens=self.n_block_tens,
            )
            for _ in range(n_layers)
        )
        ### Head frame: per-point soft slice anchor for the radial basis vector.
        ### Built only when an output of rank >= 1 reads the token basis: a
        ### scalar-only model never touches the radial direction, and a parameter
        ### that receives no gradient stops DDP at the second step.
        self.needs_frame = any(spec.rank >= 1 for spec in self.output_schema.values())
        self.frame_assign = (
            _RoutingLogits(hidden, n_slices) if self.needs_frame else None
        )
        self.norm_out = nn.LayerNorm(hidden)
        ### One linear head emits every output's invariant coefficients; each
        ### output is contracted with tensor products of the token's basis vectors
        ### (_OutputLayout / _assemble_output). True vectors: g_1..g_K, n, rhat,
        ### e_th(rhat, n), e_th(rhat, g_k), m = 2K+3 in every dimension.
        ### Pseudovectors by the Levi-Civita construction from the primaries
        ### {g_k, n, rhat}: their perpendiculars in two dimensions (p = K + 2),
        ### their pairwise cross products in three (p = C(K+2, 2)); a pseudovector
        ### in d dimensions needs d - 1 vectors, so odd-parity outputs are refused
        ### for d >= 4 until a dataset needs them. For K = 1 in 3-D and outputs
        ### {scalar: 0, vector: 1} the head has width 1 + 5.
        self.n_true_basis = 3 + 2 * K
        if self.dim == 3:
            self.n_pseudo_basis = (K + 2) * (K + 1) // 2
        elif self.dim == 2:
            self.n_pseudo_basis = K + 2
        else:
            self.n_pseudo_basis = 0
            odd = [n for n, sp in self.output_schema.items() if sp.parity == "odd"]
            if odd:
                raise NotImplementedError(
                    f"odd-parity outputs {odd!r} in dim={self.dim}: a pseudovector needs "
                    f"{self.dim - 1} vectors (the Levi-Civita construction); only two and "
                    "three dimensions are implemented"
                )
        self.output_layouts = {
            name: _OutputLayout(spec, self.n_true_basis, self.n_pseudo_basis, self.dim)
            for name, spec in self.output_schema.items()
        }
        self.n_head_coeffs = sum(
            layout.n_coeffs for layout in self.output_layouts.values()
        )
        ### The head's columns field by field in schema order, unpacked into the
        ### output fields' coefficients (a rank-0 field is its one coefficient).
        self._coeff_layout = _schema_layout(
            {n: (lay.n_coeffs,) if lay.spec.rank else () for n, lay in self.output_layouts.items()}
        )
        ### Each layout's coefficient index as a (non-persistent) buffer, so it
        ### follows the model's device instead of being copied there every pass.
        for i, layout in enumerate(self.output_layouts.values()):
            if layout.index is not None:
                self.register_buffer(f"_output_index_{i}", layout.index, persistent=False)
        self.head = nn.Linear(hidden, self.n_head_coeffs)
        ### Prediction-point tokens (interior=True): the prediction points join the
        ### encoder as INTERACTING tokens, carrying the same {n_q, g_k} invariant
        ### seeds as the boundary tokens plus the parameters of _QueryTokens.
        ### Exactly SE(d)-covariant, not query-set independent (unless
        ### query_independent: passive tokens, no routing mass). Outputs are
        ### returned for these tokens only. The tokens' total routing weight is a
        ### learned fraction exp(qt.logw) of the TOTAL boundary measure, split
        ### equally over them: invariant to any re-representation of the same
        ### discrete measure (splitting a boundary cell in two changes nothing) and
        ### scaling like an area, so the similarity-gauge contract holds.
        ### Registered last, so the boundary-side parameter order is that of a
        ### surface model; a surface model (interior=False) has none of them, so
        ### every parameter it owns takes part in every forward pass.
        self.n_query_builtin = 2
        self.n_query_features = self.n_query_builtin + self.qin.n_features
        self.qt = (
            _QueryTokens(hidden, self.n_query_features, routing_mass=not self.query_independent)
            if self.interior
            else None
        )
        ### ISLA-DTT's boundary read (tree_read.py): with query_independent the passive
        ### prediction points also read every boundary cell after every block, through
        ### a Barnes-Hut forest at TREE_READ_THETA (the private _tree_read_theta, which
        ### tests set to 0 for the exact read). Registered after everything else, so
        ### every other model owns exactly the parameters it has without the read.
        self._tree_read_theta = TREE_READ_THETA
        self.kernel_basis = None
        self.kernel_reads = None
        if self.query_independent:
            if self.n_block_tens:
                raise NotImplementedError(
                    "query_independent=True with rank-2 boundary or query inputs (the "
                    "boundary read's pair contractions are not implemented)"
                )
            n_geo = kernel_geo_width(K, self.qin.n_block_vecs, self.bin.n_block_vecs)
            self.kernel_basis = _KernelBasis(n_geo)
            self.kernel_reads = nn.ModuleList(_TreeKernelRead(hidden) for _ in range(n_layers))
        ### The trailing widths k of the per-token tensors: every small width (the
        ### input features, frames and vector stacks) and every layer's (see
        ### _hint_token_axis).
        self._token_widths = tuple(
            sorted(
                set(range(1, max(32, self.embed[0].in_features) + 1))
                | {n_slices}
                | {f for m in self.modules() if isinstance(m, nn.Linear) for f in (m.in_features, m.out_features)}
                | {m.normalized_shape[-1] for m in self.modules() if isinstance(m, nn.LayerNorm)}
            )
        )

    @staticmethod
    def _dots(x, g_hat):
        """x (B, N, d) against the K unit global directions g_hat (B, N, K, d) -> (B, N, K).
        Multiply-and-sum rather than einsum so autocast leaves the geometry in fp32."""
        return (x[:, :, None, :] * g_hat).sum(-1)

    def _seed_invariants(self, n_hat, g_hat):
        """The per-token seed invariants: the K terms n.g_k (an empty tensor when
        K = 0). Positions have no origin, so they carry no seed scalar; they
        enter through the point-to-anchor invariants of the slice blocks."""
        return self._dots(n_hat, g_hat)

    def _with_global_scalars(self, inv, g_scalars, n_tokens: int):
        """Append the global scalar inputs (B, S) to every token's seed features;
        substitute the constant seed when the feature set is empty."""
        b = inv.shape[0]
        if g_scalars is not None and g_scalars.shape[-1]:
            inv = torch.cat(
                [inv, g_scalars[:, None, :].expand(b, n_tokens, -1).to(inv.dtype)],
                dim=-1,
            )
        if self.constant_seed:
            inv = inv.new_ones(b, n_tokens, 1)
        return inv

    def _token_seed(self, n_hat, g_hat, b_feat, g_scalars, n_tokens: int):
        """A token class's seed: [n.g_k, boundary-input invariants (zeros for tokens
        that are not boundary cells), global scalars]."""
        inv = self._seed_invariants(n_hat, g_hat)
        if self.bin.n_features:
            if b_feat is None:
                b_feat = inv.new_zeros(inv.shape[0], n_tokens, self.bin.n_features)
            inv = torch.cat([inv, b_feat.to(inv.dtype)], dim=-1)
        return self._with_global_scalars(inv, g_scalars, n_tokens)

    def _block_extras(self, b_vecs, b_tens, q_vecs, q_tens, b: int, n: int, like) -> dict:
        """The union block-level stacks of one token class, as its token fields:
        ``vecs`` [boundary-derived vectors, query-derived vectors] (B, N, P', d) and
        ``tens`` [boundary tensors, query tensors] (B, N, Q, d, d); zeros where the
        class has no such field, and neither when no token carries one."""
        if not (self.n_block_vecs or self.n_block_tens):
            return {}
        d = self.dim
        if b_vecs is None:
            b_vecs = like.new_zeros(b, n, self.bin.n_block_vecs, d)
            b_tens = like.new_zeros(b, n, self.bin.n_block_tens, d, d)
        if q_vecs is None:
            q_vecs = like.new_zeros(b, n, self.qin.n_block_vecs, d)
            q_tens = like.new_zeros(b, n, self.qin.n_block_tens, d, d)
        return {"vecs": torch.cat([b_vecs, q_vecs], 2), "tens": torch.cat([b_tens, q_tens], 2)}

    def resolve_geo_kernel(self, points: torch.Tensor) -> str:
        """The geometry-kernel implementation this forward pass will use.

        ``"eager"`` or ``"fused"`` when the ``geo_kernel`` attribute forces it;
        under the default (``"auto"``) the fused kernel when ``points`` is on CUDA, Triton is
        importable, the model has one to four per-point vectors (global directions
        plus rank-1 token inputs), at most 1024 slices and no rank-2 inputs, else the
        eager region. The
        decision is cached per device and logged once.
        """
        if self.geo_kernel != "auto":
            return self.geo_kernel
        key = (points.device.type, points.device.index)
        cached = getattr(self, "_geo_kernel_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            from .geo_kernel import HAS_TRITON, MAX_DIRECTIONS, MAX_SLICES
        except Exception:  # pragma: no cover - a broken triton install falls back
            HAS_TRITON, MAX_SLICES, MAX_DIRECTIONS = False, 0, 0
        has_token_inputs = bool(self.n_block_vecs or self.n_block_tens)
        ok = (
            points.is_cuda
            and HAS_TRITON
            and self.dim == 3  # the fused kernel is written for three components
            and self.n_global_vectors >= 1
            and self.n_global_vectors + self.n_block_vecs <= MAX_DIRECTIONS
            and self.n_slices <= MAX_SLICES
            and not self.n_block_tens
        )
        chosen = "fused" if ok else "eager"
        if torch.compiler.is_compiling():
            ### Inside a traced region: no cache write and no logging (a logger
            ### call breaks fullgraph=True, a module write adds a recompile guard).
            return chosen
        self._geo_kernel_cache = (key, chosen)
        logger.info(
            "ISLA geo_kernel=auto resolved to %r on %s (cuda=%s, triton=%s, global_vectors=%d, slices=%d, token_vector_inputs=%s)",
            chosen,
            points.device,
            points.is_cuda,
            HAS_TRITON,
            self.n_global_vectors,
            self.n_slices,
            has_token_inputs,
        )
        if points.is_cuda and HAS_TRITON and has_token_inputs and chosen == "eager":
            logger.warning(
                "ISLA: rank-1 or rank-2 boundary/query inputs are present, so the fused "
                "geometry kernel is not used (kernel follow-up); running the eager region."
            )
        return chosen

    def _block_callables(self):
        """The slice blocks as callables: the modules themselves, or their forwards
        under :func:`torch.compile` when ``compile_blocks`` is set (built once)."""
        if not self.compile_blocks or torch.compiler.is_compiling():
            return self.blocks
        if self._block_forwards is None:
            self._block_forwards = [torch.compile(block.forward) for block in self.blocks]
        return self._block_forwards

    def _apply_geo_kernel(self, points: torch.Tensor) -> str | None:
        """Write the resolved kernel name into the blocks, once per resolution.
        Under torch.compile the blocks are left untouched: the kernel is returned
        for the block calls instead (``None`` when the blocks already carry it)."""
        kernel = self.resolve_geo_kernel(points)
        if getattr(self, "_geo_kernel_applied", None) == kernel:
            return None
        if torch.compiler.is_compiling():
            return kernel
        for block in self.blocks:
            block.geo_kernel = kernel
        self._geo_kernel_applied = kernel
        return None

    def _validate_inputs(
        self, points, normals, measure_weights, query_points, surface_points=None, surface_cells=None
    ) -> None:
        """Shape and presence checks for :meth:`_forward_tensors`."""
        d = self.dim
        if self.bin or self.qin:
            raise ValueError(
                "the batched routine takes no per-token inputs; pass the meshes to forward"
            )
        if points.ndim != 3 or points.shape[-1] != d:
            raise ValueError(
                f"Expected points of shape (B, N, {d}) (this model has dim={d}) but got tensor of shape {tuple(points.shape)}"
            )
        b, n, _ = points.shape
        if normals.shape != points.shape:
            raise ValueError(
                f"Expected normals of shape {(b, n, d)} but got tensor of shape {tuple(normals.shape)}"
            )
        if measure_weights is not None and measure_weights.numel() != b * n:
            raise ValueError(
                f"Expected measure_weights of shape {(b, n)} but got tensor of shape {tuple(measure_weights.shape)}"
            )
        if query_points is None:
            if self.interior:
                raise ValueError("an interior model (interior=True) needs query_points")
            return
        if not self.interior:
            raise ValueError(
                "query_points given to a surface model (interior=False); build with interior=True"
            )
        if query_points.ndim != 3 or query_points.shape[0] != b or query_points.shape[-1] != d:
            raise ValueError(
                f"Expected query_points of shape ({b}, Q, {d}) but got tensor of shape {tuple(query_points.shape)}"
            )
        if surface_points is None or surface_cells is None:
            raise ValueError("query_points need the boundary surface: surface_points and surface_cells")

    def _global_rows(self, data: TensorDict, b: int, like) -> TensorDict:
        """The declared global fields of ``data`` over a batch of ``b`` cases, in
        ``like``'s dtype at their declared shapes (a single case's values, one row,
        shared across the batch)."""
        rows = TensorDict(batch_size=[b])
        for name, spec in self.global_schema.items():
            shape, width = spec.shape(self.dim), spec.numel(self.dim)
            t = data.get(FieldSchema.key(name))
            if not torch.compiler.is_compiling():
                if not isinstance(t, torch.Tensor) or t.numel() % width:
                    raise ValueError(
                        f"global_inputs[{name!r}]: expected {width} values per case, got "
                        f"{tuple(t.shape) if isinstance(t, torch.Tensor) else type(t).__name__}"
                    )
                if t.numel() // width not in (1, b):
                    raise ValueError(
                        f"global_inputs[{name!r}] gives {t.numel() // width} rows for a batch of {b}"
                    )
            rows.set(FieldSchema.key(name), t.reshape(-1, *shape).to(like.dtype).expand(b, *shape))
        return rows

    def _global_fields(self, global_rows: TensorDict, like):
        """The unit global directions (B, K, d) and the global scalars (B, S)
        (``None`` when none are declared) from the global fields over the batch
        (:meth:`_global_rows`), each kind's columns in schema order."""
        if self.n_global_vectors:
            gv = _pack(global_rows, self._global_vector_layout).unflatten(-1, (-1, self.dim))
            ### Each direction is normalized; the magnitude, if it means anything,
            ### is a rank-0 global input.
            g_unit = gv / gv.norm(dim=-1, keepdim=True).clamp_min(EPS)  # (B, K, d)
        else:
            g_unit = like.new_zeros(global_rows.batch_size[0], 0, self.dim)
        g_scalars = _pack(global_rows, self._global_scalar_layout) if self.n_global_scalars else None
        return g_unit, g_scalars

    ### -----------------------------------------------------------------------
    ### The mesh API
    ### -----------------------------------------------------------------------

    def _check_fields(self, specs, data: TensorDict, n: int, label: str) -> None:
        """Every declared field is a leaf of ``data`` (by dotted name) with the
        shape its rank prescribes over ``n`` entities; extra leaves are ignored."""
        missing = _missing_fields(specs, data)
        if missing:
            raise ValueError(f"{label} is missing the declared fields {missing!r}")
        for name, spec in specs.items():
            t = data.get(FieldSchema.key(name))
            expected = (n, *spec.shape(self.dim))
            if tuple(t.shape) != expected:
                raise ValueError(
                    f"{label}[{name!r}]: expected shape {expected} (rank {spec.rank}), "
                    f"got {tuple(t.shape)}"
                )

    def _validate_meshes(self, boundary_meshes, prediction_points) -> None:
        """Mesh-level checks for :meth:`forward`: dimensions, declared fields
        and the task structure (``interior``, ``query_independent``)."""
        d = self.dim
        if not isinstance(boundary_meshes, Mapping) or not boundary_meshes:
            raise ValueError("boundary_meshes must be a non-empty mapping of name to Mesh")
        for name, mesh in boundary_meshes.items():
            if not isinstance(mesh, Mesh):
                raise TypeError(
                    f"boundary_meshes[{name!r}] must be a physicsnemo.mesh.Mesh, got "
                    f"{type(mesh).__name__}"
                )
            if mesh.n_spatial_dims != d:
                raise ValueError(
                    f"boundary_meshes[{name!r}] has {mesh.n_spatial_dims} spatial dimensions; "
                    f"this model has dim={d}"
                )
            if mesh.n_manifold_dims != d - 1:
                raise ValueError(
                    f"boundary_meshes[{name!r}]: the boundary of a {d}-dimensional domain is a "
                    f"Mesh[{d - 1}, {d}] (its cells carry the normals); got a "
                    f"Mesh[{mesh.n_manifold_dims}, {d}]"
                )
            self._check_fields(
                self.boundary_schema,
                mesh.cell_data,
                mesh.n_cells,
                f"boundary_meshes[{name!r}].cell_data",
            )
        if prediction_points is None:
            if self.interior:
                raise ValueError(
                    "this model was built with interior=True and predicts at prediction "
                    "points; pass prediction_points (a point-cloud Mesh, or an (N, dim) "
                    "tensor when query_schema declares no fields)"
                )
            return
        if not self.interior:
            raise ValueError(
                "this model was built with interior=False (a surface task: predictions at "
                "the boundary cells), so prediction_points must be None; build with "
                "interior=True for an interior task"
            )
        self._validate_point_set(prediction_points, "prediction_points")
        if not any(mesh.n_cells for mesh in boundary_meshes.values()):
            raise ValueError(
                "boundary_meshes have no cells: the interior task's prediction points "
                "measure their distance to the boundary surface"
            )

    def _validate_point_set(self, pts, label: str) -> None:
        """An interior point set: a point-cloud Mesh carrying the ``query_schema``
        fields, or a bare ``(N, dim)`` tensor when none are declared."""
        d = self.dim
        if isinstance(pts, Mesh):
            if pts.n_spatial_dims != d:
                raise ValueError(
                    f"{label} has {pts.n_spatial_dims} spatial "
                    f"dimensions; this model has dim={d}"
                )
            self._check_fields(self.query_schema, pts.point_data, pts.n_points, f"{label}.point_data")
            return
        if not isinstance(pts, torch.Tensor) or pts.ndim != 2 or pts.shape[-1] != d:
            raise ValueError(
                f"{label} must be a point-cloud Mesh or a tensor of shape (N, {d}); "
                f"got {tuple(pts.shape) if isinstance(pts, torch.Tensor) else type(pts).__name__}"
            )
        if self.query_schema:
            raise ValueError(
                f"this model's query_schema declares {sorted(self.query_schema)!r}; pass "
                f"{label} as a Mesh whose point_data carries them"
            )

    def _select_global_data(self, global_data, device) -> TensorDict:
        """The declared leaves of ``global_data`` as a TensorDict (extra leaves
        dropped, a missing one an error); this is also what the output Mesh carries."""
        if global_data is None:
            td = TensorDict({}, batch_size=[], device=device)
        elif isinstance(global_data, TensorDict):
            td = global_data
        elif isinstance(global_data, Mapping):
            ### A plain mapping may name its leaves with dotted strings; TensorDict
            ### takes those as nested keys only in tuple form.
            td = TensorDict({}, batch_size=[], device=device)
            for name, value in global_data.items():
                td.set(FieldSchema.key(name) if isinstance(name, str) else name, value)
        else:
            raise TypeError(
                f"global_data must be a TensorDict or a mapping of name to tensor, got "
                f"{type(global_data).__name__}"
            )
        ### Skipped under torch.compile, where listing the leaves would guard on
        ### their order (the strict select below still fails on a missing leaf).
        if not torch.compiler.is_compiling():
            missing = _missing_fields(self.global_schema, td)
            if missing:
                raise ValueError(f"global_data is missing the declared fields {missing!r}")
        return td.select(*(FieldSchema.key(name) for name in self.global_schema))

    def forward(
        self,
        *,
        boundary_meshes: Mapping[str, Mesh],
        prediction_points: Mesh | Float[torch.Tensor, "n_points dim"] | None = None,
        global_data: TensorDict | Mapping[str, torch.Tensor] | None = None,
    ) -> Mesh:
        r"""Predict the declared output fields for one case.

        Args:
            boundary_meshes: The boundary, as named ``Mesh[dim - 1, dim]``
                objects (one per boundary condition or part; a polyline in two
                dimensions, a surface in three). Their cells are the boundary
                tokens, concatenated in sorted key order: a token's position is
                the cell centroid, its normal the cell normal, its quadrature
                measure the cell's effective measure
                (``cell_data["_effective_measure"]``) when present, else the
                cell's geometric measure. Every ``boundary_schema`` field is
                read from each mesh's ``cell_data`` by its (possibly dotted)
                name; other cell data are ignored. A surface task's tokens are
                every cell. An interior task takes the boundaries at full
                resolution: their cells are the surface of the prediction
                points' geometry, and its tokens are ``boundary_tokens`` of
                them drawn inside the model (module docstring), each kept cell
                with its Horvitz-Thompson measure.
            prediction_points: The interior task's prediction points
                (``interior=True``): a point-cloud ``Mesh`` whose ``points`` are
                the prediction points and whose ``point_data`` carries the
                ``query_schema`` fields, or a bare ``(N, dim)`` tensor when
                none are declared. The points join the encoder as interacting
                tokens (passive readers with ``query_independent``); each
                one's built-in geometry, its signed distance to the boundary
                surface (the cells of ``boundary_meshes``) and the direction
                from it, is computed inside the model from the exact closest
                point on the surface's cells (:func:`exact_boundary_geometry`)
                and is never an input. A
                dataset's stored signed-distance field or its gradient enter, if
                wanted, as ordinary ``query_schema`` fields (rank 0 / rank 1)
                like any other PDE data. ``None`` for a surface task
                (``interior=False``): the outputs are predicted at the boundary
                tokens.
            global_data: The case's per-case fields, a TensorDict (or mapping)
                with the ``global_schema`` fields as leaves (a rank-0 field a
                scalar, a rank-1 field a ``(dim,)`` vector); extra leaves are
                ignored, a missing one is an error. ``None`` when nothing is
                declared.

        Returns:
            A point-cloud ``Mesh`` whose ``points`` are the prediction points
            (interior) or the boundary token positions in token order
            (surface), whose ``point_data`` holds one leaf per declared output
            at its natural shape (``(N,)`` at rank 0, ``(N, dim)`` at rank 1,
            ``(N, dim, dim)`` at rank 2, nested per the dotted names), and
            whose ``global_data`` is the selected ``global_data`` passed
            through.
        """
        names = sorted(boundary_meshes) if isinstance(boundary_meshes, Mapping) else ()
        ### Validation at the API boundary (skipped under torch.compile): fail
        ### here with the offending mesh, not inside the network.
        if not torch.compiler.is_compiling():
            self._validate_meshes(boundary_meshes, prediction_points)
        ### The boundary cells: each mesh's, concatenated in name order.
        meshes = [boundary_meshes[name] for name in names]
        measures = [cell_measures(m) for m in meshes]
        boundary = torch.cat([
            self._token_inputs(
                m.cell_centroids, m.cell_data, self.boundary_schema,
                normals=m.cell_normals, measure=w,
            )
            for m, w in zip(meshes, measures)
        ])
        global_td = self._select_global_data(global_data, boundary["points"].device)
        query = None
        if prediction_points is None:
            out_points = boundary["points"]
        else:
            query = self._point_set(prediction_points)
            out_points = query["points"]
            ### The points' geometry from the boundary surface, the union of the
            ### named meshes' cells at full resolution, at the points' precision.
            q_points = query["points"]
            with torch.autocast(q_points.device.type, enabled=False):
                offsets = [0]
                for m in meshes[:-1]:
                    offsets.append(offsets[-1] + m.n_points)
                points = torch.cat([m.points.to(q_points.device, q_points.dtype) for m in meshes])
                cells = torch.cat([m.cells.to(q_points.device) + o for m, o in zip(meshes, offsets)])
                dist, direction = exact_boundary_geometry(q_points, points, cells, eps=EPS)
            query["wall_distance"], query["wall_direction"] = dist, direction
            ### The boundary tokens: the sample of the cells (sample_boundary_cells),
            ### each kept cell with its Horvitz-Thompson measure. A uniform start
            ### from the global torch RNG in training, a fixed one in evaluation; in
            ### the measures' precision (float64 inside), whatever the autocast.
            if self.training:
                start = torch.rand((), dtype=torch.float64, device=q_points.device)
            else:
                start = torch.full((), EVAL_START, dtype=torch.float64, device=q_points.device)
            with torch.autocast(q_points.device.type, enabled=False):
                kept, weights = sample_boundary_cells(
                    measures, [m.cell_centroids for m in meshes], self.boundary_tokens, start
                )
            boundary = boundary[kept]
            boundary["measure"] = weights
        ### One case: a batch axis of one for the batched machinery below.
        fields = self._forward_tokens(
            boundary[None],
            None if query is None else query[None],
            self._global_rows(global_td, 1, boundary["points"]),
        )
        return Mesh(
            points=out_points,
            point_data=fields[0].to(out_points.device),
            global_data=global_td,
        )

    @staticmethod
    def _token_inputs(points, data: TensorDict, schema, **geometry) -> TensorDict:
        """One mesh's tokens as network inputs: their ``points`` and ``geometry``
        and, under ``fields``, the ``schema`` fields of their ``data`` (the cell
        or point data)."""
        fields = data.select(*(FieldSchema.key(name) for name in schema))
        return TensorDict(points=points, fields=fields, **geometry, batch_size=points.shape[:1])

    def _point_set(self, pts) -> TensorDict:
        """An interior point set's tokens: a point-cloud Mesh or a bare tensor."""
        if isinstance(pts, Mesh):
            return self._token_inputs(pts.points, pts.point_data, self.query_schema)
        return self._token_inputs(pts, TensorDict(batch_size=pts.shape[:1]), {})

    def _tree_read_pairs(self, bnd: TensorDict, qry: TensorDict, q_wall, g_unit):
        """The boundary read's pairs (:mod:`tree_read`): each point's near cells and
        admitted clusters at the forest's opening angle, their kernel basis
        and log measures; ``bnd`` and ``qry`` are the boundary's and the points'
        tokens and ``q_wall`` ``(B, Q)`` the points' built-in wall distance (frame
        units)."""
        r, n_hat, vecs, log_w = bnd["r"], bnd["n_hat"], bnd.get("vecs", None), bnd["log_w"]
        q_r, q_nhat, q_vecs = qry["r"], qry["n_hat"], qry.get("vecs", None)
        nb_vecs = self.bin.n_block_vecs
        log_frac = log_w[..., 0] - torch.logsumexp(log_w[..., 0], dim=1, keepdim=True)
        return tree_read_pairs(
            self.kernel_basis,
            query_r=q_r,
            query_n=q_nhat,
            query_wall=q_wall,
            query_vecs=None if q_vecs is None else q_vecs[:, :, nb_vecs:],
            boundary_r=r,
            boundary_n=n_hat,
            boundary_vecs=None if vecs is None else vecs[:, :, :nb_vecs],
            log_frac=log_frac,
            g_unit=g_unit,
            theta=self._tree_read_theta,
            eps=EPS,
        )

    def _forward_tensors(
        self,
        *,
        points: Float[torch.Tensor, "batch tokens dim"],
        normals: Float[torch.Tensor, "batch tokens dim"],
        measure_weights: Float[torch.Tensor, "batch tokens"] | None = None,
        global_inputs: Mapping[str, torch.Tensor] | None = None,
        query_points: Float[torch.Tensor, "batch queries dim"] | None = None,
        surface_points: Float[torch.Tensor, "batch vertices dim"] | None = None,
        surface_cells: Float[torch.Tensor, "batch cells dim"] | None = None,
    ) -> TensorDict:
        """The batched network on bare token tensors, for a model without
        per-token inputs; the release tests pin it, and it runs the same
        :meth:`_forward_tokens` as :meth:`forward`.

        ``points``, ``normals`` ``(B, N, d)``; ``measure_weights`` ``(B, N)``
        (``None``: every token the same weight); ``global_inputs`` a flat
        ``{dotted name: tensor}`` mapping over the declared global fields
        (per-case values ``(B,)`` / ``(B, d)``, or one row shared across the
        batch); ``query_points`` ``(B, Q, d)`` for an interior model, with the
        boundary surface each case's geometry is computed from as
        ``surface_points`` ``(B, V, d)`` and ``surface_cells`` ``(B, F, d)``
        (vertex indices; :func:`exact_boundary_geometry`). The tokens are used as
        given: nothing is sampled here. Returns the outputs
        over the ``(B, N_out)`` tokens the head read.
        """
        if not torch.compiler.is_compiling():
            self._validate_inputs(points, normals, measure_weights, query_points, surface_points, surface_cells)
        fields = global_inputs or {}
        missing = [k for k in self.global_schema if k not in fields]
        extra = [k for k in fields if k not in self.global_schema]
        if missing or extra:
            raise ValueError(
                f"global_inputs fields do not match the declared set: missing {missing!r}, "
                f"unexpected {extra!r}"
            )
        b, n, _ = points.shape
        boundary = TensorDict(points=points, normals=normals, batch_size=[b, n])
        if measure_weights is not None:
            boundary["measure"] = measure_weights.reshape(b, n)
        query = None
        if query_points is not None:
            geometry = [
                exact_boundary_geometry(query_points[i], surface_points[i], surface_cells[i], eps=EPS)
                for i in range(b)
            ]
            query = TensorDict(
                points=query_points,
                wall_distance=torch.stack([g[0] for g in geometry]),
                wall_direction=torch.stack([g[1] for g in geometry]),
                batch_size=query_points.shape[:2],
            )
        data = TensorDict({FieldSchema.key(k): v for k, v in fields.items()}, batch_size=[])
        return self._forward_tokens(boundary, query, self._global_rows(data, b, points))

    def _forward_tokens(
        self,
        boundary: TensorDict,
        query: TensorDict | None,
        global_rows: TensorDict,
    ) -> TensorDict:
        """The batched network on the token inputs.

        ``boundary`` holds the boundary tokens over ``(B, N)``: ``points`` and
        ``normals`` ``(B, N, d)``, ``measure`` ``(B, N)`` (absent: every token the
        same weight) and the ``boundary_schema`` fields under ``fields``;
        ``query`` (interior) the prediction-point tokens over ``(B, Q)``:
        ``points``, the ``query_schema`` fields under ``fields`` and the built-in
        geometry ``wall_distance`` ``(B, Q)`` and ``wall_direction`` ``(B, Q, d)``
        (:func:`exact_boundary_geometry`);
        ``global_rows`` the global fields over ``(B,)`` (:meth:`_global_rows`).
        Returns the declared outputs over the ``(B, N_out)`` tokens the head reads.

        The token state is one TensorDict over ``(B, N)``, the boundary tokens
        then the prediction-point tokens, with fields ``h``, ``r``, ``n_hat``,
        ``g_hat``, ``log_w`` and the block-level ``vecs`` and ``tens``; the slice
        blocks take its tensors (module docstring, *Data layout*).
        """
        points = boundary["points"]
        b, n, _ = points.shape
        _hint_token_axis(n, self._token_widths)
        geo_kernel = self._apply_geo_kernel(points)
        K, dim = self.n_global_vectors, self.dim
        g_unit, g_scalars = self._global_fields(global_rows, points)
        g_hat = g_unit[:, None].expand(b, n, K, dim)

        ### Frame: no origin; the one length unit is the measure-weighted RMS
        ### pairwise distance, computed through the variance identity
        ### sum_ij w_i w_j |x_i-x_j|^2 = 2 sum_i w_i |x_i - c|^2 with c the
        ### weighted mean (used only inside this scalar; the frame keeps no
        ### centre). Normalized weights make it converge under refinement where
        ### the total measure diverges, and make it invariant to a weight rescale.
        measure = boundary.get("measure", None)
        gauge = rms_pairwise_distance(points, measure, EPS)
        normals = boundary["normals"]
        n_hat = normals / normals.norm(dim=-1, keepdim=True).clamp_min(EPS)

        if measure is None:
            ### No measures given: every token carries the same measure, i.e. the
            ### count-weighted quadrature.
            log_w = points.new_zeros(b, n, 1)
        else:
            log_w = torch.log(measure.to(points.dtype).clamp_min(EPS))[..., None]

        ### Boundary inputs: invariants against the token's frame {n, g_k}.
        if self.bin:
            frame = torch.cat([n_hat[:, :, None], g_hat], dim=2)  # (B, N, 1+K, d)
            b_feat, b_vecs, b_tens = self.bin.features(boundary["fields"], frame)
        else:
            b_feat, b_vecs, b_tens = None, None, None
        tokens = TensorDict(
            h=self.embed(self._token_seed(n_hat, g_hat, b_feat, g_scalars, n)),
            r=points / gauge,
            n_hat=n_hat,
            g_hat=g_hat,
            log_w=log_w,
            **self._block_extras(b_vecs, b_tens, None, None, b, n, points),
            batch_size=[b, n],
        )

        ### n_boundary counts the boundary tokens only: the kernel reads update
        ### the tokens appended below.
        n_boundary = n
        if query is not None:
            query_points = query["points"]
            nq = query_points.shape[1]
            ### The query's geometry from the boundary surface: the signed distance
            ### to it (the built-in channel) and the direction from it (the token's
            ### "normal" in every invariant), from the exact closest point.
            with torch.autocast(points.device.type, enabled=False):
                dist, q_nhat = query["wall_distance"], query["wall_direction"].to(points.dtype)
                s = (dist.to(points.dtype) / gauge.squeeze(-1))[..., None]  # (B, Q, 1), in frame units
                builtin = torch.cat([s, torch.sign(s) * torch.log(s.abs() + EPS)], -1)
            q_g = g_unit[:, None].expand(b, nq, K, dim)
            if self.qin:
                q_frame = torch.cat([q_nhat[:, :, None], q_g], dim=2)
                q_feat, q_vecs, q_tens = self.qin.features(
                    query["fields"], q_frame, gauge=gauge, eps=EPS
                )
                q_feat = torch.cat([builtin, q_feat], -1)
            else:
                q_feat, q_vecs, q_tens = builtin, None, None
            h_q = self.qt(self.embed(self._token_seed(q_nhat, q_g, None, g_scalars, nq)), q_feat)
            if self.query_independent:
                ### Passive queries: log-weight -inf leaves them out of every softmax
                ### over points (routing prototypes, anchors, slice states, head-frame
                ### anchors), so they read every block's slices and never write.
                q_logw = log_w.new_full((b, nq, 1), -math.inf)
            else:
                ### The query tokens' total routing weight is a learned fraction of the
                ### total boundary measure, split equally (see __init__): rescaling the
                ### geometry (areas x k^2) shifts every token's log-weight by the same
                ### 2 log k, so the similarity-gauge contract holds exactly.
                q_logw = (
                    self.qt.logw.to(log_w.dtype)
                    + torch.logsumexp(log_w, dim=1, keepdim=True)
                    - _log_count(nq, log_w)
                )
            q_tokens = TensorDict(
                h=h_q,
                r=query_points / gauge,
                n_hat=q_nhat,
                g_hat=q_g,
                log_w=q_logw.expand(b, nq, 1),
                **self._block_extras(None, None, q_vecs, q_tens, b, nq, points),
                batch_size=[b, nq],
            )
            if self.kernel_reads is not None:
                kernel_pairs = self._tree_read_pairs(tokens, q_tokens, s[..., 0].abs(), g_unit)
            tokens = torch.cat([tokens, q_tokens], dim=1)
            _hint_token_axis(n + nq, self._token_widths)

        h, log_w, r = tokens["h"], tokens["log_w"], tokens["r"]
        vecs, tens = tokens.get("vecs", None), tokens.get("tens", None)
        for i, block in enumerate(self._block_callables()):
            h = block(h, log_w, r, tokens["n_hat"], tokens["g_hat"], EPS, vecs, tens, geo_kernel=geo_kernel)
            if self.kernel_reads is not None:
                ### The points' residual update from the boundary cells' states at this
                ### depth; the boundary tokens (and so the slices) are untouched.
                h_b, h_p = h[:, :n_boundary], h[:, n_boundary:]
                delta = self.kernel_reads[i](
                    h_b.reshape(-1, h.shape[-1]), h_p.reshape(-1, h.shape[-1]), kernel_pairs
                )
                h = torch.cat([h_b, h_p + delta.view_as(h_p)], dim=1)
        tokens.set("h", h)

        ### The tokens the head reads: the prediction points in the interior mode,
        ### every token otherwise. The head frame builds its slice anchors from ALL
        ### tokens (r, log_w and h).
        out = tokens if query is None else _tail(tokens, n_boundary)
        coeffs = self._coeff_layout.unpack(self.head(self.norm_out(out["h"])))
        if not self.needs_frame:
            ### Scalar-only outputs are the coefficients themselves; no token basis.
            return coeffs
        ### Head frame: the radial basis vector is the direction from the token's
        ### soft slice anchor (measure-weighted mean of token positions), not
        ### from a centroid; translation covariant. The frame's prototypes are
        ### synthesized once from ALL tokens with their measures (the same
        ### quadrature that weights the anchors), so the frame is refinement
        ### invariant like the blocks; both routings read those prototypes.
        protos_f = self.frame_assign.prototypes(h, log_w)
        a_f = _softmax_over_points(self.frame_assign.logits(h, protos_f) + log_w)
        z_f = torch.einsum("bns,bnc->bsc", a_f, r)
        b_f = torch.softmax(self.frame_assign.logits(out["h"], protos_f), dim=-1)
        r_out = out["r"]
        u = r_out - torch.einsum("bns,bsc->bnc", b_f, z_f).to(r_out.dtype)
        r_hat = u / u.norm(dim=-1, keepdim=True).clamp_min(EPS)

        ### The token basis. True vectors: the global directions, the normal, the
        ### radial direction and the polar complements of (rhat, n) and each
        ### (rhat, g_k) (the vector's component orthogonal to rhat, left
        ### unnormalized so it vanishes smoothly where the pair is parallel; any
        ### dimension). Pseudovectors: the Levi-Civita construction from the
        ### primaries {g_k, n, rhat}: in three dimensions their pairwise cross
        ### products (for K = 1: g x n, g x rhat, n x rhat), in two their
        ### perpendiculars; both flip sign under a reflection while their
        ### coefficients do not.
        n_out, g_out = out["n_hat"], out["g_hat"]
        gs = [g_out[:, :, k] for k in range(K)]
        true_basis = gs + [n_out, r_hat, _polar_complement(r_hat, n_out)]
        for g in gs:
            true_basis.append(_polar_complement(r_hat, g))
        primaries = gs + [n_out, r_hat]
        pseudo_basis = _pseudovectors(primaries, dim)
        true_vecs = torch.stack(true_basis, dim=-2)  # (B, N, m, d)
        pseudo_vecs = (
            torch.stack(pseudo_basis, dim=-2)
            if pseudo_basis
            else true_vecs.new_zeros(*out.batch_size, 0, dim)
        )
        ### Each output of rank >= 1 from its coefficients, in place of them.
        for i, (name, layout) in enumerate(self.output_layouts.items()):
            if layout.spec.rank:
                key = FieldSchema.key(name)
                index = getattr(self, f"_output_index_{i}")
                coeffs.set(key, _assemble_output(coeffs.get(key), layout, true_vecs, pseudo_vecs, index))
        return coeffs
