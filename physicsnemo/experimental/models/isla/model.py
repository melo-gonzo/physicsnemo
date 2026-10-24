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
time, the boundary goes in as a mapping of :class:`~physicsnemo.mesh.Mesh`
objects keyed by boundary-condition name (each a ``Mesh[d-1, d]``; its cells
are the tokens, with their centroids, normals and effective measures: every
cell for a surface task, a sample of the full-resolution cells drawn inside the
model for an interior task), the
problem's data as *named fields of declared tensor rank*: global inputs in a
TensorDict (a direction, a diffusivity, a Reynolds number), per-cell boundary
inputs in the meshes' ``cell_data``, declared per condition (an inlet's
prescribed velocity, a loaded face's traction; a no-slip wall declares none);
an interior task's prediction points are read by their positions alone (their
geometry is computed inside the model). A point-cloud ``Mesh`` comes out whose ``point_data`` holds fields of
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
``output_schema``, each schema of ``boundary_schemas`` and ``global_schema``, and
the same dotted names address the fields in the meshes' ``cell_data`` and
in ``global_data`` (``td["surface", "pressure"]``).

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

**Boundary conditions by name.** ``boundary_schemas`` declares the boundary
conditions by name, each with its own per-cell fields (an empty schema for a
condition without any, a no-slip wall). A condition's identity is structural:
its tokens enter through the condition's own input layer, whose column on a
constant is its learned offset, and its vector and tensor inputs reach the
slice layers through its own linear maps onto shared channels (equivariant: a
linear combination with scalar weights commutes with every orthogonal map). No
tag field names a condition. The slice layers are shared: every condition's
tokens route into the same slices and read the same anchors, which is how the
conditions interact. ``forward`` takes any non-empty subset of the declared
names; the tokens are the given conditions' cells in canonical (sorted) order,
so the order of the input mapping changes nothing.

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
point-to-anchor differences, and the one length scale :math:`L` is the
measure-weighted root-mean-square pairwise distance of every cell of the given
boundaries at full resolution (computed before an interior task's token draw, so
it is exact and the same for every draw), a
normalized two-point statistic that is invariant to the weights' overall
scale and converges under refinement of any boundary with a finite second
moment, fractal ones included. No centroid, no per-dataset reference length
and no external frame appears anywhere in the forward pass, so the frame
carries no information about where the mesher placed its cells.

**Scales (opt-in).** A field of a condition's schema, of ``global_schema`` or of
``output_schema`` may declare, next to its rank spec, ``units``: exponents of
the three base scales, the length :math:`L` above, a velocity :math:`U` and a
density :math:`\rho` (``{"U": 2, "rho": 1}`` for a gauge pressure in Pa). A
rank-1 boundary field or one rank-1 global flagged ``velocity_scale: true`` is a
source: :math:`U = (\sum a |v|^2 / \sum a)^{1/2}`, the measure-weighted RMS speed
pooled over the boundary sources' full-resolution cells (a global source's
magnitude), and with boundary sources :math:`\hat g = \sum a v / |\sum a v|` is
appended to the global directions (a global source is a direction already); both
are computed per case in float64 before the draw, and an absent or zero source
is refused (in eager mode; torch.compile skips the data-dependent checks). Rank-0 globals with the ``role`` ``density`` and ``viscosity`` give
:math:`\rho` and :math:`\nu` and are not appended; with ``reynolds={"mean": m,
"std": s}`` the global scalar :math:`z = (\log_{10}(U L / \nu) - m) / s` is
appended. Every declared boundary field and rank-0 global is divided by its
units' :math:`\prod \text{scale}^{\text{exp}}` and by its constant ``scale`` (the
typical size of the dimensionless field, default 1) before it enters the invariants,
and every output with units is the head's dimensionless prediction times both, then
standardized by the constants ``output_stats`` (vector means zero). The length
:math:`L` is exact for the piecewise-flat boundary: each cell enters with its own
second moment about its centroid, so a subdivision of the cells changes nothing. The
keys are ISLA's own, popped before the shared grammar parses the field; with
none declared the model is the unscaled one. So a similarity of the problem
(lengths times :math:`s`, velocities :math:`c`, densities :math:`r`) leaves
every internal state unchanged and scales each output by its units.

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

**Surface and interior tasks.** Both tasks read the whole boundary: every cell of
every mesh in ``boundary_meshes`` is a token (the interior task's a sample of
them). ``interior=False`` (a surface task) predicts at the cells of the
``prediction_boundaries`` given to ``forward``, the names of the conditions that
carry the outputs (a vehicle's wall, with the tunnel's floor, walls, inlet and
outlet read as conditions only); its output points are those cells' centroids.
``interior=True`` (an interior task)
predicts at the ``prediction_points`` given to ``forward`` (a point-cloud
``Mesh``, whose ``points`` alone are read, or a bare ``(N, d)`` tensor): they
join the encoder as interacting tokens next to the boundary cells and the
outputs are returned for them only. Their only inputs are their positions and
the geometry the model computes for them; per-cell boundary inputs (declared per
condition in ``boundary_schemas``) are carried by the boundary cells of that
condition. The prediction-point
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
adds the prediction-point input layer and routing mass), which is
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
(:func:`exact_boundary_geometry`). The token reads :math:`s_q = d_q / L` and
:math:`\operatorname{asinh}(s_q / 10^{-6})` (:func:`soft_log_length`; logarithmic
away from the wall, linear through it and within :math:`10^{-6} L` of it) as its
two distance channels, and
:math:`\operatorname{sign}(d_q)\, u_q`
(with :math:`\operatorname{sign}(0) = +1`, the gradient of the signed distance) as
its direction. Both are continuous through the surface: a point on it, whose
distance is zero or a rounding error of either sign, gets the code of the points
just off it on either side. :math:`(d_q, u_q)` are functions of the surface
and its orientation alone: the same for every draw of the boundary tokens,
invariant to splitting a cell and converging as :math:`O(h^2)` under refinement
of the surface, like a stored signed-distance field computed from the same
surface, which the model therefore does not read. The search is a Warp BVH in three dimensions (a torch custom op) and
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
is deterministic. The draw reads the cells' measures and centroids in float64
from the input meshes, so it is the same at every model precision. A surface
task reads every cell it is given (its outputs are predicted at the cells of its
prediction boundaries, which must all be tokens) and samples none.

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
then a model with rank-2 boundary inputs runs the eager region.

Every constructor argument and every ``forward`` input is keyword-only, so a
call reads as the recipe's ``forward_kwargs`` mapping does and models can be
swapped without positional bookkeeping. ``forward(boundary_meshes=...,
prediction_points=..., global_data=...)`` (interior) and ``forward(boundary_meshes=...,
prediction_boundaries=..., global_data=...)`` (surface) return a point-cloud ``Mesh``;
see :meth:`ISLA.forward`.

**Data layout.** Inside ``forward`` the tokens stay TensorDicts: each boundary
condition's cells (centroid, normal, measure, and the invariants of its declared
``cell_data`` fields with its vectors and tensors, in a block layout over every
declared condition, so that the cells of all conditions, and the in-model
draw's sample of them, are one token set with static widths) and the
prediction points are TensorDicts concatenated into one token state over
``(batch, token)`` (completed to a multiple of :data:`TOKEN_ALIGNMENT` tokens by
padding tokens of no measure that nothing reads), the field inputs enter the invariants and the head's
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

import contextlib
import functools
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    cell_second_moments,
    exact_boundary_geometry,
    rms_pairwise_distance,
    soft_log_length,
)
from physicsnemo.experimental.models.isla.heads import (
    _assemble_output,
    _OutputLayout,
    _polar_complement,
    _pseudovectors,
)
from physicsnemo.experimental.models.isla.inputs import (
    _BoundaryCondition,
    _declare,
    _InputSet,
    _missing_fields,
    _pack,
    _schema_layout,
)
from physicsnemo.experimental.models.isla.sampling import draw_inputs, sample_boundary_cells
from physicsnemo.experimental.models.isla.scales import (
    check_output_stats,
    merged,
    scale_factor,
    split_scale_keys,
    velocity_moments,
)
from physicsnemo.experimental.models.isla.tree_read import (
    TREE_READ_THETA,
    TreeReadWalk,
    _KernelBasis,
    _TreeKernelRead,
    kernel_geo_width,
    tree_read_walk,
    walk_pair_count,
    walk_points_range,
    walk_range_pairs,
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
#: The token axis is completed to a multiple of this count by padding tokens
#: (:func:`_pad_tokens`). On CUDA in bfloat16 the slice blocks' GEMMs that reduce
#: over the token axis run on cuBLAS's fast kernels only when its rows span a
#: multiple of 16 bytes (8 tokens), and the LayerNorm backward's weight-gradient
#: reduction over more than 65,536 rows takes its aligned path only for a multiple
#: of its 32-row tile.
TOKEN_ALIGNMENT = 32


### Inductor options of the compiled slice blocks (compile_blocks). Mix-order
### reduction must stay off until the installed torch contains the fix for
### pytorch/pytorch#195845 (on main since 2026-09-29, not in 2.12 or 2.13). With it
### on, blocks in the eager geometry region add the padded rows of the last token
### tile to the per-slice sums whenever the token count is not a multiple of the tile
### height, and so compute a different function from the eager model. The recipe's
### whole-model compile options turn it off for the same reason.
_BLOCK_COMPILE_OPTIONS = {"triton.mix_order_reduction": False}


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


def _pad_tokens(tokens: TensorDict, alignment: int) -> TensorDict:
    """``tokens`` ``(B, N)`` followed by ``-N mod alignment`` padding tokens: every
    field zero and the log-weight ``-inf``. Like passive tokens they hold no measure
    in any softmax over points, so no other token reads them, and no output is read
    from them: every output and gradient is the unpadded one up to the summation
    order of the longer reductions."""
    b, n = tokens.batch_size
    pad = -n % alignment
    ### In eager mode an aligned count keeps the tokens as they are. A compiled graph
    ### always pads: a test of pad == 0 would guard the dynamic token count on it.
    if not torch.compiler.is_compiling() and pad == 0:
        return tokens

    ### F.pad rather than a concatenated block of pad tokens: under dynamic shapes
    ### a separate (B, pad, .) tensor makes torch.compile guard on pad == 0.
    def fill(v, value=0.0):
        return F.pad(v, (0, 0) * (v.dim() - 2) + (0, pad), value=value)

    padded = TensorDict({k: fill(v) for k, v in tokens.items()}, batch_size=[b, n + pad])
    padded.set("log_w", fill(tokens["log_w"], -math.inf))
    return padded


def _log_count(n: int, like: torch.Tensor) -> float | torch.Tensor:
    """``log(n)`` of a token count; under torch.compile a 0-d tensor of ``like``'s
    dtype, since ``math.log`` of a shape would specialize the dynamic shapes on it."""
    if not torch.compiler.is_compiling():
        return math.log(n)
    return torch.full((), n, dtype=torch.float64, device=like.device).log().to(like.dtype)


def _refuse_earlier_state_dict(module, state_dict, prefix, local_metadata, *args) -> None:
    """A load_state_dict pre-hook: a state dict of an earlier ISLA is refused with the
    versions named, on every loading path (``Module.load``, ``load_checkpoint``,
    ``load_state_dict``), not only ``Module.from_checkpoint``, which checks the
    recorded version. Two marks tell an earlier state dict: the parameter layout
    before 0.7.0 (the shared seed embedding ``embed.0``, the prediction points'
    ``qt.type_offset`` and ``qt.embed``), and, for 0.7.0 to 0.9.0, whose layout is
    the current one, the state-dict version torch records for every module
    (``ISLA._version``, 1 before 0.8.0, 8 for 0.8.0, 9 for 0.9.0, and 10 for the
    unmerged head-frame candidate 0.10.0). A state dict rebuilt without
    torch's metadata (a dict comprehension over it) carries no version and is read
    by its layout alone."""
    old = sorted(
        k[len(prefix) :]
        for k in state_dict
        if k.startswith(prefix) and k[len(prefix) :].startswith(("embed.0.", "qt.type_offset", "qt.embed."))
    )
    if old:
        raise RuntimeError(
            f"this ISLA state dict has the parameter layout of a checkpoint before version "
            f"0.7.0 ({', '.join(old[:3])}, ...): ISLA {module.__model_checkpoint_version__} "
            "declares its boundary conditions by name (boundary_schemas) and keeps no upgrade "
            "path from earlier checkpoints (0.6.0 and before); load it with the code that "
            "wrote it, or retrain"
        )
    version = local_metadata.get("version")
    if version is not None and version != type(module)._version:
        raise RuntimeError(
            f"this ISLA state dict was written before version {module.__model_checkpoint_version__} "
            f"(state-dict version {version}, now {type(module)._version}): its weights have the "
            "current layout but were trained on an earlier forward (before 0.8.0, the "
            "prediction points' geometry encoding sign(s) log(|s| + 1e-12) and the direction "
            "from the surface, in place of asinh(s / 1e-6) and sign(d) times it; before "
            "0.9.0, the length unit without each boundary cell's own second moment, and for "
            "the interior task measured on the drawn tokens in place of the full-resolution "
            "boundary; before 0.11.0, the geometry without least-norm tie directions, the "
            "winding-number sign of the wall distance, the float64 canonical-frame token "
            "draw, the on-wall rule and float64 cell normals; state-dict version 10 is the "
            "unmerged head-frame candidate 0.10.0), and ISLA keeps no upgrade path; "
            "load it with the code that wrote it, or retrain"
        )


class ISLA(Module):
    r"""ISLA (Invariant Slice Attention): invariant backbone, equivariant edges.

    A soft-slice transformer for steady boundary-value problems whose attention
    operates only on invariants of the per-point vector set and whose outputs of
    any rank are re-assembled in an equivariant basis at the heads, so exact
    rotation, reflection and translation covariance holds by construction (see
    the module docstring for the contracts). Every argument is keyword-only.

    **Rank specs.** ``output_schema``, each schema of ``boundary_schemas`` and
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

    **Mesh API.** ``forward`` takes the boundary as a mapping of
    :class:`~physicsnemo.mesh.Mesh` objects keyed by the declared
    boundary-condition names (each a ``Mesh[dim - 1, dim]`` whose cells have
    normals; any non-empty subset of the declared names), the
    per-case data as a TensorDict, and where the outputs are: for an interior
    task the prediction points as a point-cloud ``Mesh`` (its ``points``) or a
    bare ``(N, dim)`` tensor, for a surface task the names of the boundaries at
    whose cells they are predicted (``prediction_boundaries``). It returns a
    point-cloud ``Mesh`` whose ``point_data`` holds the declared outputs.
    Details under ``forward``.

    **Execution.** The ``geo_kernel`` attribute (``"auto"``; set ``"eager"`` or
    ``"fused"`` to force one) selects how the per-layer geometry region is
    executed, not what it computes (:meth:`resolve_geo_kernel`). The
    ``geometry_kernel`` attribute (``"auto"``; set ``"reference"`` or ``"fast"``) does
    the same for the search of the prediction points' boundary geometry
    (:meth:`resolve_geometry_kernel`), whose two implementations give the same
    outputs bitwise. The ``tree_read_kernel`` attribute does the same for
    ISLA-DTT's tree read: under ``"auto"`` each of its pieces (the forest build,
    the walk, the read with its node means) runs its fast kernel where that
    piece's requirements hold
    (:func:`~physicsnemo.experimental.models.isla.tree_build.resolve_tree_build`,
    ``tree_read.resolve_walk``, :meth:`resolve_tree_read_kernel`), each choice
    logged once, and its reference otherwise; ``"reference"`` runs every piece's
    reference. The forest and the pairs are the same either way, the read the
    same up to summation order. Each is an execution choice, not a constructor
    argument: checkpoints do not store it, and a loaded model starts at ``"auto"``.

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
        boundary_schemas: The boundary conditions by name, each with the
            per-cell fields it carries: a rank spec per field with ranks 0 to
            2 and even parity (an inlet's prescribed velocity as ``{"rank":
            1}``, an outlet's pressure as ``{"rank": 0}``), or an empty schema
            for a condition without fields (a no-slip wall: ``{}``). The
            names are fixed here, in canonical (sorted) order; a name is a
            non-empty string without ``"."``. ``forward`` takes any non-empty
            subset of them, each mesh carrying its own condition's fields.
            Each condition has its own input layer
            (``boundary_conditions[name].seed``), which maps the condition's
            seed ``[1, n.g_k, its fields' invariants, the global scalars]``
            to the token width ahead of the shared GELU and linear layer
            (``embed``); its first column, on the constant, is the
            condition's learned offset, so a condition's identity is in the
            parameters its tokens read even when it declares no field. A
            condition's rank-1 and rank-2 inputs (and the vectors derived
            from its rank-2 inputs) reach the slice layers through its own
            linear maps onto shared vector and tensor channels, as many as
            the widest condition has, where they contract with the
            token-to-anchor direction (module docstring); the slice layers
            themselves are shared by every condition.
        global_schema: Per-case fields, a rank spec per name with rank 0
            (appended to every token's seed) or rank 1 (a global direction,
            normalized to unit length inside the model). The ``K`` rank-1
            entries, in declaration order, are the global directions of the
            geometry region and the head basis. Read from ``global_data`` by
            name; extra leaves there are ignored.
        interior: The task structure. ``False`` (default): a surface task,
            the outputs are predicted at the cells of the boundaries ``forward``
            names in ``prediction_boundaries`` (every given boundary is read);
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
            deterministic. A surface task reads every cell it is given and
            samples none: it takes only the default.
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
            (use ``compile_blocks``); rank-2 boundary inputs are not
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
        reynolds: ``{"mean": m, "std": s}``, the standardization of
            ``log10(U L / nu)``, appended to the global scalars as ``z``
            (module docstring, *Scales*); required with a global of the role
            ``viscosity``, refused without one. ``None`` (default): no Reynolds
            number.
        output_stats: ``{name: {"mean", "std"}}`` over declared outputs (the
            format of ``NormalizeMeshFields.stats``; ``type`` is ignored): each
            output, after its units are restored, is returned as ``(y - mean) /
            std``, so the model predicts a dataset's standardized physical targets.
            The mean of a field of rank 1 or more must be 0. ``None`` (default):
            the outputs as the head gives them (in physical units when declared).
            The units, sources and roles are declared in the schemas: each field's
            mapping may carry ``units`` ({``"L"``, ``"U"``, ``"rho"``} to
            exponents), ``velocity_scale`` (rank 1, boundary or global),
            ``role`` (rank-0 global, ``"density"`` or ``"viscosity"``) and
            ``scale`` (an output's positive constant ``c``: ``y = c yhat prod
            scale^exp``). With any of them, or ``output_stats``, the head and
            the output assembly run in the parameters' precision under autocast.

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

    ### 0.7.0: boundary conditions by name (boundary_schemas, one input layer and
    ### vector and tensor maps per condition), the prediction points' own input layer.
    ### 0.8.0: the prediction points' geometry encoding is continuous through the
    ### surface (the built-in distance channel asinh(s / 1e-6) in place of
    ### sign(s) log(|s| + 1e-12), the direction sign(d) u; same parameters, new
    ### forward). 0.9.0: the interior task's length unit is the full-resolution
    ### boundary's RMS pairwise distance, exact and independent of the token draw (in
    ### place of its estimate from the drawn tokens; same parameters, new forward),
    ### and the opt-in scales (units, velocity_scale, the density and viscosity roles,
    ### reynolds, output_stats). 0.11.0: the geometry robustness bundle (least-norm
    ### tie directions, the winding-number sign of the wall distance, the float64
    ### canonical-frame token draw, the on-wall rule, float64 cell normals, and the
    ### cell-tree seed of the search radius; same parameters, new forward). 0.10.0
    ### is skipped: it names the unmerged head-frame candidate, whose checkpoints
    ### exist, so each version string names one model. No earlier checkpoint loads:
    ### ISLA is unreleased and keeps no upgrade paths.
    __model_checkpoint_version__ = "0.11.0"
    #: torch's state-dict version of the module (``nn.Module._version``, recorded in
    #: every state dict's metadata): 1 before 0.8.0, 8 for 0.8.0, 9 for 0.9.0 and 10
    #: for the head-frame candidate, so a state dict of 0.7.0 to 0.10.0, whose
    #: parameter layout is the current one, is told apart and refused on load
    #: (:func:`_refuse_earlier_state_dict`).
    _version = 11

    def __init__(
        self,
        *,
        output_schema: FieldSchemaLike,
        boundary_schemas: Mapping[str, FieldSchemaLike],
        global_schema: FieldSchemaLike | None = None,
        interior: bool = False,
        query_independent: bool = False,
        boundary_tokens: int = BOUNDARY_TOKENS,
        hidden: int = 512,
        n_layers: int = 12,
        n_slices: int = 512,
        compile_blocks: bool = False,
        dim: int = 3,
        reynolds: Mapping[str, float] | None = None,
        output_stats: Mapping[str, Mapping] | None = None,
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
        self._block_reads: list | None = None
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
        ### dotted names here. The ISLA-local keys of a field (units, velocity_scale,
        ### role; scales.py) are popped before the shared grammar parses it.
        output_schema, out_extras = split_scale_keys("output_schema", output_schema)
        self.output_schema = _declare(
            "output_schema",
            output_schema,
            max_rank=self.MAX_OUTPUT_RANK,
            allow_odd=True,
        )
        if not self.output_schema:
            raise ValueError("output_schema must declare at least one field")
        ### The boundary conditions by name, in canonical (sorted) order, each with the
        ### per-cell fields it alone carries (an empty schema is a condition without
        ### fields, a no-slip wall).
        if not isinstance(boundary_schemas, Mapping) or not boundary_schemas:
            raise ValueError(
                "boundary_schemas must be a non-empty mapping of boundary-condition name to "
                f"field schema; got {boundary_schemas!r}"
            )
        for name in boundary_schemas:
            if not isinstance(name, str) or not name or "." in name:
                raise ValueError(
                    f"boundary_schemas: a condition name is a non-empty string without '.'; got {name!r}"
                )
            ### A name keys the conditions' nn.ModuleDict, so it cannot be one of its
            ### attributes ("keys", "update", "training", ...).
            if hasattr(nn.ModuleDict(), name):
                raise ValueError(
                    f"boundary_schemas: the condition name {name!r} is an attribute of nn.ModuleDict, "
                    "which holds the conditions' parameters; choose another name"
                )
        split = {
            name: split_scale_keys(f"boundary_schemas[{name!r}]", boundary_schemas[name])
            for name in sorted(boundary_schemas)
        }
        b_extras = {name: extras for name, (_, extras) in split.items()}
        self.boundary_schemas = {
            name: _declare(
                f"boundary_schemas[{name!r}]",
                spec,
                max_rank=self.MAX_INPUT_RANK,
                allow_odd=False,
            )
            for name, (spec, _) in split.items()
        }
        global_schema, g_extras = split_scale_keys("global_schema", global_schema)
        self.global_schema = _declare(
            "global_schema", global_schema, max_rank=1, allow_odd=False
        )
        self._init_scales(out_extras, b_extras, g_extras, reynolds, output_stats)
        ### The task structure: a surface task predicts at the cells of the boundaries
        ### forward names and has no prediction-point parameters; an interior task
        ### predicts at prediction points that join the encoder as tokens.
        self.interior = bool(interior)
        if self.interior and self.dim not in (2, 3):
            raise NotImplementedError(
                "the interior task's built-in geometry (the exact distance to the boundary "
                f"surface) is implemented in two and three dimensions; got dim={self.dim}"
            )
        self.query_independent = bool(query_independent)
        if self.query_independent and not self.interior:
            raise ValueError(
                "query_independent=True concerns the prediction points of the interior "
                "task; build with interior=True (a surface task predicts at boundary cells)"
            )
        if isinstance(boundary_tokens, bool) or int(boundary_tokens) != boundary_tokens or boundary_tokens < 1:
            raise ValueError(f"boundary_tokens must be a positive integer; got {boundary_tokens!r}")
        if not self.interior and boundary_tokens != BOUNDARY_TOKENS:
            raise ValueError(
                "boundary_tokens is the interior task's budget of boundary tokens; a surface "
                "task (interior=False) reads every boundary cell it is given and samples "
                f"none, so it takes the default ({BOUNDARY_TOKENS}); got {boundary_tokens!r}"
            )
        self.boundary_tokens = int(boundary_tokens)
        ### Checkpoints store the constructor arguments as JSON. A FieldSchema
        ### is a dict and serializes as it stands; recording the parsed schemas
        ### keeps the metadata JSON-safe whatever container the caller passed
        ### (a Hydra DictConfig, for instance).
        ### The declarations are stored merged with their ISLA-local keys.
        self._args["__args__"]["output_schema"] = merged(self.output_schema, out_extras) or None
        self._args["__args__"]["global_schema"] = merged(self.global_schema, g_extras) or None
        self._args["__args__"]["boundary_schemas"] = {
            name: merged(schema, b_extras[name]) for name, schema in self.boundary_schemas.items()
        }
        self._args["__args__"]["reynolds"] = self.reynolds
        self._args["__args__"]["output_stats"] = self.output_stats
        ### The problem's global data are zero or more global DIRECTIONS (each
        ### enters as a unit direction: one seed term n.g_k per token, one
        ### direction cosine rel.g_k per point-anchor pair, and the head basis
        ### gains g_k with the spherical complements of (u, g_k)) and zero or more
        ### global SCALARS (PDE parameters; appended to every token's seed). With
        ### boundary velocity sources the derived direction g is the last direction;
        ### the density and viscosity roles are consumed, not appended, and the
        ### standardized log Reynolds number z is the last scalar.
        self.global_vector_names = [
            n for n, s in self.global_schema.items() if s.rank == 1
        ]
        self.global_scalar_names = [
            n for n, s in self.global_schema.items() if s.rank == 0 and n not in self._roles.values()
        ]
        self.n_global_vectors = K = len(self.global_vector_names) + bool(self._boundary_sources)
        self.n_global_scalars = len(self.global_scalar_names) + (self.reynolds is not None)
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
        ### geometry_kernel names the implementation of exact_boundary_geometry's search,
        ### not its arithmetic: "reference" defines it, "fast" gives the same outputs
        ### bitwise from a faster traversal. "auto" resolves at the first forward pass
        ### (resolve_geometry_kernel). Writable, as geo_kernel.
        self.geometry_kernel = "auto"
        self.n_slices = int(n_slices)
        ### Each boundary condition's fields, each set with its own invariant layout
        ### (_InputSet). A condition's block vectors and tensors reach the slice
        ### layers through its own linear maps onto shared channels, as many as the
        ### widest condition has (_BoundaryCondition); every token carries them
        ### (zeros for the prediction points), so one geometry region serves every token.
        n_frame = 1 + K  # {n_hat, g_1..g_K}
        inputs = {
            name: _InputSet(schema, n_frame, dim=self.dim)
            for name, schema in self.boundary_schemas.items()
        }
        self.n_boundary_vecs = max(i.n_block_vecs for i in inputs.values())
        self.n_boundary_tens = max(i.n_block_tens for i in inputs.values())
        self.n_block_vecs = self.n_boundary_vecs
        self.n_block_tens = self.n_boundary_tens
        ### A condition's seed: [1, n.g_k (the K origin-free scalars of {n, g_k};
        ### positions enter through the anchors), its fields' invariants, the global
        ### scalars], mapped to the token width by its own input layer and then by the
        ### shared GELU and linear layer (embed). A boundary token carries its seed in
        ### its condition's columns of one block layout over every declared condition,
        ### zeros elsewhere, so one product with the concatenated input layers encodes
        ### a mixed set of tokens whose conditions the in-model draw decides, and every
        ### condition's parameters take part in every pass (DDP), present or not.
        self.boundary_conditions = nn.ModuleDict(
            {
                name: _BoundaryCondition(
                    inp,
                    K + inp.n_features + self.n_global_scalars,
                    hidden,
                    self.n_boundary_vecs,
                    self.n_boundary_tens,
                )
                for name, inp in inputs.items()
            }
        )
        self.boundary_names = tuple(self.boundary_schemas)
        self._condition_index = {name: i for i, name in enumerate(self.boundary_names)}
        self._seed_columns = self._offsets(c.seed.in_features for c in self.boundary_conditions.values())
        self._vec_columns = self._offsets(i.n_block_vecs for i in inputs.values())
        self._tens_columns = self._offsets(i.n_block_tens for i in inputs.values())
        self.embed = nn.Sequential(nn.GELU(), nn.Linear(hidden, hidden))
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
        self.n_query_features = 2  # the built-in distance channels
        self.qt = (
            _QueryTokens(
                hidden,
                K + self.n_global_scalars + self.n_query_features,
                routing_mass=not self.query_independent,
            )
            if self.interior
            else None
        )
        ### ISLA-DTT's boundary read (tree_read.py): with query_independent the passive
        ### prediction points also read every boundary cell after every block, through
        ### a Barnes-Hut forest at TREE_READ_THETA (the private _tree_read_theta, which
        ### tests set to 0 for the exact read). Registered after everything else, so
        ### every other model owns exactly the parameters it has without the read.
        self._tree_read_theta = TREE_READ_THETA
        ### tree_read_kernel names the implementation of the tree read, as geo_kernel
        ### does the geometry region's: "reference" (the per-octant cluster trees,
        ### GLOBE's walk, the kernel term a matmul read by segment_softmax.py) or
        ### "auto", where the forest build (tree_build.py), the walk (tree_walk.py)
        ### and the read (fused_read.py: the kernel term computed inside the read,
        ### nothing per pair kept but the basis) each take their fast path where
        ### their own requirements hold. Writable.
        self.tree_read_kernel = "auto"
        self.kernel_basis = None
        self.kernel_reads = None
        if self.query_independent:
            if self.n_block_tens:
                raise NotImplementedError(
                    "query_independent=True with rank-2 boundary inputs (the "
                    "boundary read's pair contractions are not implemented)"
                )
            n_geo = kernel_geo_width(K, self.n_boundary_vecs)
            self.kernel_basis = _KernelBasis(n_geo)
            self.kernel_reads = nn.ModuleList(_TreeKernelRead(hidden) for _ in range(n_layers))
        ### A state dict of an earlier version fails with its version named on every
        ### loading path, not with a list of missing and unexpected keys.
        self.register_load_state_dict_pre_hook(_refuse_earlier_state_dict)
        ### The trailing widths k of the per-token tensors: every small width (the
        ### input features, frames and vector stacks) and every layer's (see
        ### _hint_token_axis).
        self._token_widths = tuple(
            sorted(
                set(range(1, max(32, self._seed_columns[-1]) + 1))
                | {n_slices}
                | {f for m in self.modules() if isinstance(m, nn.Linear) for f in (m.in_features, m.out_features)}
                | {m.normalized_shape[-1] for m in self.modules() if isinstance(m, nn.LayerNorm)}
            )
        )

    def _init_scales(self, out_extras, b_extras, g_extras, reynolds, output_stats) -> None:
        """The opt-in scales (module docstring, *Scales*) from the declarations'
        ISLA-local keys (:func:`~physicsnemo.experimental.models.isla.scales.split_scale_keys`),
        checked against the parsed schemas."""
        decl = {"output_schema": out_extras, "global_schema": g_extras}
        decl.update({f"boundary_schemas[{n!r}]": e for n, e in b_extras.items()})
        schemas = {"output_schema": self.output_schema, "global_schema": self.global_schema}
        schemas.update({f"boundary_schemas[{n!r}]": s for n, s in self.boundary_schemas.items()})
        used = set()
        for kind, extras in decl.items():
            for name, local in extras.items():
                rank = schemas[kind][name].rank
                if local.get("velocity_scale") and rank != 1:
                    raise ValueError(f"{kind}[{name!r}]: a velocity_scale source is a rank-1 field; got rank {rank}")
                if "role" in local and rank != 0:
                    raise ValueError(f"{kind}[{name!r}]: a {local['role']} is a rank-0 global; got rank {rank}")
                if "role" in local and "scale" in local:
                    raise ValueError(f"{kind}[{name!r}]: the {local['role']} is consumed, not an input; it takes no scale")
                if kind == "global_schema" and rank == 1 and not local.get("velocity_scale"):
                    if "units" in local or "scale" in local:
                        raise ValueError(
                            f"{kind}[{name!r}]: a rank-1 global that is not a velocity_scale source enters as a "
                            "direction only, so units and a scale have no effect on it; declare its magnitude as a "
                            "rank-0 global"
                        )
                if kind == "global_schema" and rank == 1 and "scale" in local:
                    raise ValueError(f"{kind}[{name!r}]: a velocity source's magnitude is U; it takes no scale")
                if "role" not in local:
                    used |= set(local.get("units", {}))
        self._output_units = {n: e["units"] for n, e in out_extras.items() if e.get("units")}
        self._output_scale = {n: e["scale"] for n, e in out_extras.items() if "scale" in e}
        self._global_units = {
            n: e["units"]
            for n, e in g_extras.items()
            if e.get("units") and "role" not in e and self.global_schema[n].rank == 0
        }
        self._global_scale = {n: e["scale"] for n, e in g_extras.items() if "scale" in e}
        self._boundary_units = {
            c: {n: e["units"] for n, e in extras.items() if e.get("units")} for c, extras in b_extras.items()
        }
        self._boundary_scale = {
            c: {n: e["scale"] for n, e in extras.items() if "scale" in e} for c, extras in b_extras.items()
        }
        self._boundary_sources = [
            (c, n) for c, extras in b_extras.items() for n, e in extras.items() if e.get("velocity_scale")
        ]
        self._global_sources = [n for n, e in g_extras.items() if e.get("velocity_scale")]
        self._roles = {}
        for n, e in g_extras.items():
            if "role" in e:
                if e["role"] in self._roles:
                    raise ValueError(f"global_schema: two fields have the role {e['role']!r}")
                self._roles[e["role"]] = n
        if self._boundary_sources and self._global_sources:
            raise ValueError(
                "the velocity_scale sources are rank-1 boundary fields or one rank-1 global, not both: "
                f"got {self._boundary_sources!r} and {self._global_sources!r}"
            )
        if len(self._global_sources) > 1:
            raise ValueError(
                f"at most one rank-1 global is a velocity_scale source; got {self._global_sources!r}"
            )
        has_source = bool(self._boundary_sources or self._global_sources)
        if "U" in used and not has_source:
            raise ValueError("units name U, but no field is a velocity_scale source")
        if "rho" in used and "density" not in self._roles:
            raise ValueError("units name rho, but no global has the role 'density'")
        if ("viscosity" in self._roles) != (reynolds is not None):
            raise ValueError(
                "the Reynolds number needs a global with the role 'viscosity' and its standardization "
                f"reynolds={{mean, std}}, both or neither; got reynolds={reynolds!r}"
            )
        if reynolds is not None:
            if not has_source:
                raise ValueError("reynolds needs a velocity_scale source (Re = U L / nu)")
            if (
                not isinstance(reynolds, Mapping)
                or set(reynolds) != {"mean", "std"}
                or not all(isinstance(v, Real) and not isinstance(v, bool) and math.isfinite(v) for v in reynolds.values())
                or not reynolds["std"] > 0
            ):
                raise ValueError(f"reynolds is {{mean, std}}, finite numbers with std > 0; got {reynolds!r}")
            reynolds = {"mean": float(reynolds["mean"]), "std": float(reynolds["std"])}
        self.reynolds = reynolds
        self.output_stats = check_output_stats(output_stats, self.output_schema)
        ### The scales are declared when any field has units, an input scale or is a
        ### source, or a role is.
        has_input_scale = bool(self._global_scale or any(self._boundary_scale.values()))
        self._scaled = bool(used or has_source or self._roles or has_input_scale)
        ### The head and the redimensionalization run in the parameters' precision
        ### under autocast whenever a physical output is assembled or standardized. The
        ### outputs' precision is bounded by the bf16 trunk either way; the fp32 head keeps
        ### bf16's rounding of the head's own product and of a nonzero output mean (an
        ### AirfRANS nut with mean / std 0.3) out of the outputs.
        self._fp32_head = self._scaled or self.output_stats is not None or bool(self._output_scale)

    @staticmethod
    def _offsets(widths) -> tuple[int, ...]:
        """The column offsets of consecutive blocks of the given ``widths``, one block
        per declared condition in canonical order: ``(0, w_0, w_0 + w_1, ..., total)``."""
        out = [0]
        for w in widths:
            out.append(out[-1] + int(w))
        return tuple(out)

    @staticmethod
    def _dots(x, g_hat):
        """x (B, N, d) against the K unit global directions g_hat (B, N, K, d) -> (B, N, K).
        Multiply-and-sum rather than einsum so autocast leaves the geometry in fp32."""
        return (x[:, :, None, :] * g_hat).sum(-1)

    @staticmethod
    def _in_block(x, cols: tuple[int, ...], i: int, dim: int):
        """``x`` placed in block ``i`` of the block layout ``cols`` along ``dim``
        (zeros in every other block)."""
        b, n = x.shape[:2]
        before = x.new_zeros(b, n, cols[i], *x.shape[dim + 1 :])
        after = x.new_zeros(b, n, cols[-1] - cols[i + 1], *x.shape[dim + 1 :])
        return torch.cat([before, x, after], dim)

    def _condition_tokens(
        self, name: str, points, normals, measure, fields: TensorDict, g_unit, g_scalars
    ) -> TensorDict:
        """One condition's boundary tokens over ``(B, n)``: their ``points``,
        ``normals`` and ``measure`` (``None``: none given) and, in the block layout
        over every declared condition (zeros in the other conditions' columns), the
        ``seed`` ``[1, n.g_k, the fields' invariants, global scalars]`` and the
        condition's block vectors ``vecs`` and rank-2 tensors ``tens`` (when any
        condition has them). ``fields`` holds the condition's declared fields over
        ``(B, n)``; ``g_unit`` ``(B, K, d)`` and ``g_scalars`` ``(B, S)`` or ``None``
        are the case's global inputs (:meth:`_global_fields`)."""
        i = self._condition_index[name]
        inputs = self.boundary_conditions[name].inputs
        b, n, d = points.shape
        n_hat = normals / normals.norm(dim=-1, keepdim=True).clamp_min(EPS)
        g_hat = g_unit[:, None].expand(b, n, -1, d).to(points.dtype)
        seed = [points.new_ones(b, n, 1), self._dots(n_hat, g_hat)]
        if inputs:
            feat, vecs, tens = inputs.features(fields, torch.cat([n_hat[:, :, None], g_hat], dim=2))
            seed.append(feat)
        if g_scalars is not None:
            seed.append(g_scalars[:, None, :].expand(b, n, -1).to(points.dtype))
        tokens = TensorDict(
            points=points,
            normals=normals,
            seed=self._in_block(torch.cat(seed, -1), self._seed_columns, i, 2),
            batch_size=[b, n],
        )
        if measure is not None:
            tokens["measure"] = measure
        if self._vec_columns[-1]:
            if not inputs.n_block_vecs:
                vecs = points.new_zeros(b, n, 0, d)
            tokens["vecs"] = self._in_block(vecs, self._vec_columns, i, 2)
        if self._tens_columns[-1]:
            if not inputs.n_block_tens:
                tens = points.new_zeros(b, n, 0, d, d)
            tokens["tens"] = self._in_block(tens, self._tens_columns, i, 2)
        return tokens

    def _boundary_channels(self, boundary: TensorDict):
        """The boundary tokens' initial state and their shared vector and tensor
        channels: the block layout's seed through the conditions' concatenated input
        layers and the shared GELU and linear layer, and each condition's block
        vectors and tensors through its linear maps (``None`` where the model has
        none). Every declared condition's parameters enter the products."""
        conditions = list(self.boundary_conditions.values())
        weight = torch.cat([c.seed.weight for c in conditions], dim=1)
        h = self.embed(torch.nn.functional.linear(boundary["seed"], weight))
        vecs = tens = None
        with torch.autocast(h.device.type, enabled=False):
            if self._vec_columns[-1]:
                maps = [c.vector_map for c in conditions if c.vector_map is not None]
                m = torch.cat(maps, dim=1).to(boundary["vecs"].dtype)
                vecs = torch.einsum("vp,bnpi->bnvi", m, boundary["vecs"])
            if self._tens_columns[-1]:
                maps = [c.tensor_map for c in conditions if c.tensor_map is not None]
                m = torch.cat(maps, dim=1).to(boundary["tens"].dtype)
                tens = torch.einsum("vp,bnpij->bnvij", m, boundary["tens"])
        return h, vecs, tens

    def _block_extras(self, b_vecs, b_tens, b: int, n: int, like) -> dict:
        """The block-level stacks of one token class, as its token fields: the
        boundary channels ``vecs`` (B, N, P', d) and ``tens`` (B, N, Q, d, d);
        zeros for the prediction points, which carry none, and neither when no
        token carries one."""
        if not (self.n_block_vecs or self.n_block_tens):
            return {}
        d = self.dim
        if b_vecs is None:
            b_vecs = like.new_zeros(b, n, self.n_boundary_vecs, d)
        if b_tens is None:
            b_tens = like.new_zeros(b, n, self.n_boundary_tens, d, d)
        return {"vecs": b_vecs, "tens": b_tens}

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
                "ISLA: rank-1 or rank-2 boundary inputs are present, so the fused "
                "geometry kernel is not used (kernel follow-up); running the eager region."
            )
        return chosen

    def resolve_tree_read_kernel(self, points: torch.Tensor) -> str:
        """The ISLA-DTT read's implementation this forward pass will use.

        ``"reference"`` when the ``tree_read_kernel`` attribute is ``"reference"``;
        under the default (``"auto"``) ``"fused"`` when ``points`` is on CUDA, not
        float64, and Triton is importable, else the reference read. Both compute the
        same read up to summation order. The decision is cached per device and dtype
        and logged once.
        """
        if self.tree_read_kernel == "reference":
            return "reference"
        if self.tree_read_kernel != "auto":
            raise ValueError(f"tree_read_kernel must be 'auto' or 'reference', not {self.tree_read_kernel!r}")
        key = (points.device.type, points.device.index, points.dtype)
        cached = getattr(self, "_tree_read_kernel_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            from .fused_read import HAS_TRITON
        except Exception:  # pragma: no cover - a broken triton install falls back
            HAS_TRITON = False
        ok = points.is_cuda and HAS_TRITON and points.dtype != torch.float64
        chosen = "fused" if ok else "reference"
        if torch.compiler.is_compiling():
            return chosen
        self._tree_read_kernel_cache = (key, chosen)
        logger.info(
            "ISLA tree_read_kernel=auto resolved to %r on %s (cuda=%s, triton=%s, dtype=%s)",
            chosen,
            points.device,
            points.is_cuda,
            HAS_TRITON,
            points.dtype,
        )
        return chosen

    def resolve_geometry_kernel(self, points: torch.Tensor) -> str:
        """The implementation of the boundary geometry's search this forward pass will
        use (``kernel`` of :func:`exact_boundary_geometry`).

        ``"reference"`` or ``"fast"`` when the ``geometry_kernel`` attribute forces it;
        under the default (``"auto"``) the fast search when ``points`` is on CUDA and
        Triton is importable (its weld and canonical-cell kernels), else the
        reference. The decision is cached per device and logged once.
        """
        if self.geometry_kernel != "auto":
            return self.geometry_kernel
        key = (points.device.type, points.device.index)
        cached = getattr(self, "_geometry_kernel_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        from .geometry_fast import HAS_TRITON

        chosen = "fast" if points.is_cuda and HAS_TRITON else "reference"
        if torch.compiler.is_compiling():
            return chosen
        self._geometry_kernel_cache = (key, chosen)
        logger.info(
            "ISLA geometry_kernel=auto resolved to %r on %s (cuda=%s, triton=%s)", chosen, points.device, points.is_cuda, HAS_TRITON
        )
        return chosen

    def _block_callables(self):
        """The slice blocks as callables: the modules themselves, or their forwards
        under :func:`torch.compile` when ``compile_blocks`` is set (built once)."""
        if not self.compile_blocks or torch.compiler.is_compiling():
            return self.blocks
        if self._block_forwards is None:
            self._block_forwards = [
                torch.compile(block.forward, options=_BLOCK_COMPILE_OPTIONS)
                for block in self.blocks
            ]
        return self._block_forwards

    def _block_read_callables(self):
        """The slice blocks' passive reads (:meth:`_SliceBlock.read`), compiled as
        :meth:`_block_callables` compiles their forwards."""
        if not self.compile_blocks or torch.compiler.is_compiling():
            return [block.read for block in self.blocks]
        if self._block_reads is None:
            self._block_reads = [
                torch.compile(block.read, options=_BLOCK_COMPILE_OPTIONS)
                for block in self.blocks
            ]
        return self._block_reads

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
        self, condition, points, normals, measure_weights, query_points, surface_points=None, surface_cells=None
    ) -> None:
        """Shape and presence checks for :meth:`_forward_tensors`."""
        d = self.dim
        if condition not in self.boundary_schemas:
            raise ValueError(
                f"condition {condition!r} is not one of the declared boundary conditions "
                f"{list(self.boundary_names)!r}"
            )
        if self.boundary_schemas[condition]:
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

    def _global_fields(self, global_rows: TensorDict, like, scales: dict | None = None):
        """The unit global directions (B, K, d) and the global scalars (B, S)
        (``None`` when none are declared) from the global fields over the batch
        (:meth:`_global_rows`), each kind's columns in schema order; with
        ``scales`` (:meth:`_case_scales`) the rank-0 globals divided by their
        units' scale, the derived direction ``g`` appended to the directions and
        the standardized log Reynolds number ``z`` to the scalars."""
        if self.global_vector_names:
            gv = _pack(global_rows, self._global_vector_layout).unflatten(-1, (-1, self.dim))
            ### Each direction is normalized; the magnitude, if it means anything,
            ### is a rank-0 global input.
            g_unit = gv / gv.norm(dim=-1, keepdim=True).clamp_min(EPS)  # (B, K, d)
        else:
            g_unit = like.new_zeros(global_rows.batch_size[0], 0, self.dim)
        g_scalars = _pack(global_rows, self._global_scalar_layout) if self.global_scalar_names else None
        if scales is None:
            return g_unit, g_scalars
        if self._global_units or self._global_scale:
            g_scalars = g_scalars / torch.stack(
                [
                    self._input_factor(self._global_units.get(n), self._global_scale.get(n), scales).to(like.dtype)
                    for n in self.global_scalar_names
                ]
            )
        if scales["g"] is not None:
            g_unit = torch.cat([g_unit, scales["g"].to(g_unit.dtype).expand(g_unit.shape[0], 1, -1)], 1)
        if scales["z"] is not None:
            z = scales["z"].to(like.dtype).expand(g_unit.shape[0], 1)
            g_scalars = z if g_scalars is None else torch.cat([g_scalars, z], -1)
        return g_unit, g_scalars

    @staticmethod
    def _gauge(meshes, measures) -> torch.Tensor:
        """The length unit ``(1, 1, 1)`` of the given boundary meshes: the RMS pairwise
        distance of their piecewise-flat cells, each cell with its centroid, measure and
        own second moment (:func:`rms_pairwise_distance`)."""
        return rms_pairwise_distance(
            torch.cat([m.cell_centroids for m in meshes])[None],
            torch.cat(measures)[None],
            EPS,
            torch.cat([cell_second_moments(m) for m in meshes])[None],
        )

    def _case_scales(self, names, meshes, measures, global_rows: TensorDict, gauge) -> dict:
        """The case's scales in float64 from its full-resolution boundary and its
        globals, before the token draw: ``L`` (the gauge), ``U`` (the
        measure-weighted RMS speed pooled over the boundary sources, or the global
        source's magnitude), ``rho`` (the density role), the derived direction
        ``g`` ``(1, 1, d)`` (``None`` unless the sources are boundary fields) and
        ``z`` ``(1, 1)`` (``None`` without ``reynolds``). An absent or zero source,
        or a non-positive density or viscosity, is refused in eager mode; under
        torch.compile these data-dependent checks are skipped (as every input check
        of ``forward`` is), so a compiled model reads such a case as NaN or inf."""
        check = not torch.compiler.is_compiling()
        scales = {"L": gauge.reshape(()).double(), "g": None, "z": None}
        if self._boundary_sources:
            present = dict(zip(names, zip(meshes, measures)))
            s0, s2, s1 = velocity_moments(
                (present[c][0].cell_data.get(FieldSchema.key(n)), present[c][1])
                for c, n in self._boundary_sources
                if c in present
            )
            if check and not (isinstance(s0, torch.Tensor) and bool(s0 > 0) and bool(s2 > 0)):
                raise ValueError(
                    f"the velocity_scale sources {self._boundary_sources!r} are absent from this case, "
                    "have no cells or are zero everywhere: U is undefined"
                )
            norm = s1.norm()
            if check and not bool(norm > 0):
                raise ValueError(
                    f"the velocity_scale sources {self._boundary_sources!r} cancel (sum of a v is zero): "
                    "the direction g is undefined"
                )
            scales["U"] = (s2 / s0).sqrt()
            scales["g"] = (s1 / norm)[None, None]
        elif self._global_sources:
            v = global_rows.get(FieldSchema.key(self._global_sources[0]))[0].double()
            scales["U"] = v.norm()
            if check and not bool(scales["U"] > 0):
                raise ValueError(f"the velocity_scale source {self._global_sources[0]!r} is zero: U is undefined")
        for role, key in (("density", "rho"), ("viscosity", "nu")):
            if role in self._roles:
                scales[key] = global_rows.get(FieldSchema.key(self._roles[role]))[0].double().reshape(())
                if check and not bool(scales[key] > 0):
                    raise ValueError(f"the {role} {self._roles[role]!r} must be positive; got {scales[key].item()}")
        if self.reynolds is not None:
            log_re = torch.log10(scales["U"] * scales["L"] / scales.pop("nu"))
            scales["z"] = ((log_re - self.reynolds["mean"]) / self.reynolds["std"])[None, None]
        return scales

    @staticmethod
    def _input_factor(units, c, scales) -> torch.Tensor:
        """An input's divisor: its units' scale times its constant ``scale``."""
        factor = scale_factor(units, scales)
        if factor is None:
            factor = scales["L"].new_ones(())
        return factor if c is None else factor * c

    def _nondimensional(self, condition: str, fields: TensorDict, scales: dict | None) -> TensorDict:
        """A condition's declared fields, each divided by its units' scale and then by
        its constant ``scale``."""
        if scales is None:
            return fields
        units, cs = self._boundary_units.get(condition, {}), self._boundary_scale.get(condition, {})
        names = [n for n in self.boundary_schemas[condition] if n in units or n in cs]
        if not names:
            return fields
        fields = fields.clone(False)
        for name in names:
            key = FieldSchema.key(name)
            t = fields.get(key)
            fields.set(key, t / self._input_factor(units.get(name), cs.get(name), scales).to(t.dtype))
        return fields

    def _redimensionalize(self, fields: TensorDict, scales: dict | None) -> TensorDict:
        """The outputs times their constant ``scale`` and their units' scale, then
        standardized by ``output_stats``, in the outputs' precision (the head's, see
        :meth:`_forward_tokens`)."""
        for name, spec in self.output_schema.items():
            factor = scale_factor(self._output_units.get(name), scales) if scales is not None else None
            c = self._output_scale.get(name)
            stats = (self.output_stats or {}).get(name)
            if factor is None and c is None and stats is None:
                continue
            key = FieldSchema.key(name)
            y = fields.get(key)
            if c is not None:
                y = y * c
            if factor is not None:
                y = y * factor.to(y.dtype)
            if stats is not None:
                y = (y - stats["mean"]) / stats["std"] if spec.rank == 0 else y / stats["std"]
            fields.set(key, y)
        return fields

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

    def _validate_meshes(self, boundary_meshes, prediction_points, prediction_boundaries) -> None:
        """Mesh-level checks for :meth:`forward`: dimensions, declared fields
        and the task structure (``interior``, ``query_independent``, where the
        outputs are)."""
        d = self.dim
        if not isinstance(boundary_meshes, Mapping) or not boundary_meshes:
            raise ValueError("boundary_meshes must be a non-empty mapping of name to Mesh")
        unknown = sorted(str(name) for name in boundary_meshes if name not in self.boundary_schemas)
        if unknown:
            raise ValueError(
                f"boundary_meshes names {unknown!r}, which are not declared boundary conditions; "
                f"this model declares {list(self.boundary_names)!r} (boundary_schemas)"
            )
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
                self.boundary_schemas[name],
                mesh.cell_data,
                mesh.n_cells,
                f"boundary_meshes[{name!r}].cell_data",
            )
        if prediction_points is None:
            if self.interior:
                raise ValueError(
                    "this model was built with interior=True and predicts at prediction "
                    "points; pass prediction_points (a point-cloud Mesh or an (N, dim) "
                    "tensor)"
                )
            self._validate_prediction_boundaries(prediction_boundaries, boundary_meshes)
            return
        if not self.interior:
            raise ValueError(
                "this model was built with interior=False (a surface task: predictions at "
                "the boundary cells), so prediction_points must be None; build with "
                "interior=True for an interior task"
            )
        if prediction_boundaries is not None:
            raise ValueError(
                "this model was built with interior=True and predicts at prediction_points, "
                "so prediction_boundaries must be None (a surface task names the boundaries "
                "it predicts on)"
            )
        self._validate_point_set(prediction_points, "prediction_points")
        if not any(mesh.n_cells for mesh in boundary_meshes.values()):
            raise ValueError(
                "boundary_meshes have no cells: the interior task's prediction points "
                "measure their distance to the boundary surface"
            )

    def _validate_prediction_boundaries(self, names, boundary_meshes) -> None:
        """A surface task's prediction boundaries: a non-empty sequence of distinct
        names of given boundaries (not a bare string)."""
        if names is None:
            raise ValueError(
                "this model was built with interior=False (a surface task) and predicts at "
                "the cells of named boundaries; pass prediction_boundaries, the names of "
                f"the given boundaries that carry the outputs (given: {sorted(boundary_meshes)!r})"
            )
        if isinstance(names, str) or not isinstance(names, (list, tuple)) or not names:
            raise ValueError(
                "prediction_boundaries must be a non-empty list or tuple of boundary names; "
                f"got {names!r}"
            )
        if len(set(names)) != len(names):
            raise ValueError(f"prediction_boundaries names a boundary twice: {list(names)!r}")
        absent = [n for n in names if n not in boundary_meshes]
        if absent:
            raise ValueError(
                f"prediction_boundaries names {absent!r}, which are not in boundary_meshes "
                f"({sorted(boundary_meshes)!r}); the outputs are predicted at given cells"
            )

    def _validate_point_set(self, pts, label: str) -> None:
        """An interior point set: a point-cloud Mesh (its ``points``) or a bare
        ``(N, dim)`` tensor."""
        d = self.dim
        if isinstance(pts, Mesh):
            if pts.n_spatial_dims != d:
                raise ValueError(
                    f"{label} has {pts.n_spatial_dims} spatial "
                    f"dimensions; this model has dim={d}"
                )
            return
        if not isinstance(pts, torch.Tensor) or pts.ndim != 2 or pts.shape[-1] != d:
            raise ValueError(
                f"{label} must be a point-cloud Mesh or a tensor of shape (N, {d}); "
                f"got {tuple(pts.shape) if isinstance(pts, torch.Tensor) else type(pts).__name__}"
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
        prediction_boundaries: list[str] | tuple[str, ...] | None = None,
        global_data: TensorDict | Mapping[str, torch.Tensor] | None = None,
    ) -> Mesh:
        r"""Predict the declared output fields for one case.

        Both tasks read the whole boundary in ``boundary_meshes``; where the outputs
        are is the second argument: ``prediction_points`` for an interior task,
        ``prediction_boundaries`` for a surface task (exactly one of them, as the
        task requires).

        Args:
            boundary_meshes: The boundary, as ``Mesh[dim - 1, dim]`` objects
                keyed by boundary-condition name (a polyline in two dimensions, a
                surface in three): any non-empty subset of the names declared in
                ``boundary_schemas``, in any order (an unknown name is a
                ``ValueError``; an absent condition, like one with no cells,
                contributes no tokens). Their cells are the boundary tokens, the
                conditions in canonical (sorted) order, each condition's cells in
                its mesh's order: a token's position is the cell centroid, its
                normal the cell normal, its quadrature measure the cell's
                effective measure (``cell_data["_effective_measure"]``) when
                present, else the cell's geometric measure. Each condition's
                ``boundary_schemas`` fields are read from its mesh's
                ``cell_data`` by their (possibly dotted) names; other cell data
                are ignored. A surface task's tokens are
                every cell of every given boundary, those of its prediction
                boundaries and those read as conditions only (a tunnel's floor,
                walls, inlet and outlet). An interior task takes the boundaries at full
                resolution: their cells are the surface of the prediction
                points' geometry, and its tokens are ``boundary_tokens`` of
                them drawn inside the model (module docstring), each kept cell
                with its Horvitz-Thompson measure.
            prediction_points: The interior task's prediction points
                (``interior=True``): a point-cloud ``Mesh`` whose ``points`` are
                the prediction points (its ``point_data`` is not read), or a bare
                ``(N, dim)`` tensor. The points join the encoder as interacting
                tokens (passive readers with ``query_independent``); each
                one's built-in geometry, its signed distance to the boundary
                surface (the cells of ``boundary_meshes``) and the direction
                from it, is computed inside the model from the exact closest
                point on the surface's cells (:func:`exact_boundary_geometry`)
                and is never an input. ``None`` for a surface task
                (``interior=False``).
            prediction_boundaries: The surface task's output boundaries
                (``interior=False``, required): a non-empty list or tuple of
                distinct names of given boundaries, at whose cells the outputs
                are predicted (DrivAerML Surface: ``["vehicle"]``, its floor,
                walls, inlet and outlet read as conditions only; a surface whose
                conditions together carry the outputs names them all). Their
                order does not matter. It owns no parameters: the head reads the
                same tokens whichever boundaries carry the outputs, only the rows
                it returns differ. The prediction is a quadrature over the cells
                given for these boundaries (the request), so a request must cover
                the body: on DrivAerML Surface, half a car's cells raise the
                pressure and wall-shear errors 10 to 18 times. A larger request,
                or the mean of several, lowers the error: the spread between
                requests falls about as one over the square root of their cell
                count. ``None`` for an interior task.
            global_data: The case's per-case fields, a TensorDict (or mapping)
                with the ``global_schema`` fields as leaves (a rank-0 field a
                scalar, a rank-1 field a ``(dim,)`` vector); extra leaves are
                ignored, a missing one is an error. ``None`` when nothing is
                declared.

        Returns:
            A point-cloud ``Mesh`` whose ``points`` are the prediction points
            (interior) or the cell centroids of the prediction boundaries
            (surface: those boundaries in sorted name order, each one's cells in
            its mesh's order), whose ``point_data`` holds one leaf per declared output
            at its natural shape (``(N,)`` at rank 0, ``(N, dim)`` at rank 1,
            ``(N, dim, dim)`` at rank 2, nested per the dotted names), and
            whose ``global_data`` is the selected ``global_data`` passed
            through.
        """
        boundary, query, g_unit, g_scalars, gauge, scales, out_points, global_td, out_rows = self._case_inputs(
            boundary_meshes, prediction_points, global_data, prediction_boundaries
        )
        fields = self._redimensionalize(
            self._forward_tokens(boundary, query, g_unit, g_scalars, gauge=gauge, out_rows=out_rows), scales
        )
        return Mesh(
            points=out_points,
            point_data=fields[0].to(out_points.device),
            global_data=global_td,
        )

    def query_chunks(
        self,
        *,
        boundary_meshes: Mapping[str, Mesh],
        prediction_points: Mesh | Float[torch.Tensor, "n_points dim"],
        global_data: TensorDict | Mapping[str, torch.Tensor] | None = None,
    ) -> "_QueryChunks":
        r"""ISLA-DTT's :meth:`forward` in chunks of prediction points, for a training
        step that holds the activations of one chunk of points at a time.

        Runs the boundary side of the forward pass once, from the same arguments:
        the case's inputs (the token draw included, which consumes the global RNG
        as :meth:`forward` does), the boundary tokens through every block and the
        forest (each chunk walks it for its own points). What the prediction points read of each block
        (its prototypes, anchors and slice states, the boundary read's keys and
        values) and the head frame's prototypes and anchors become leaves that
        accumulate gradients. The result ``chunks`` is then called as
        ``chunks(start, stop)``: the output Mesh of ``prediction_points[start:stop]``,
        equal to those rows of :meth:`forward` up to summation order (each point
        reads exactly what it reads among all of them), differentiable in the
        parameters and the leaves. After the chunks' backward passes,
        ``chunks.backward()`` carries the leaves' gradients through the boundary side
        once, so the parameters' gradients are those of the unchunked step whose
        loss is the sum of the chunks' (``chunks.n_points`` points in all).
        ``query_independent=True`` only: interacting points are not a function of
        each point alone.
        """
        if not self.query_independent:
            raise ValueError(
                "query_chunks needs query_independent=True: with interacting prediction points "
                "a prediction depends on every point of the request"
            )
        return _QueryChunks(self, self._case_inputs(boundary_meshes, prediction_points, global_data))

    def _case_inputs(self, boundary_meshes, prediction_points, global_data, prediction_boundaries=None) -> tuple:
        """One case's inputs to :meth:`_forward_tokens` from :meth:`forward`'s:
        ``(boundary, query, g_unit, g_scalars, gauge, scales)`` (the interior task's
        boundary tokens drawn, the prediction points with their geometry), then the
        output points, the selected global data and the surface task's ``out_rows``."""
        ### Validation at the API boundary (skipped under torch.compile): fail
        ### here with the offending mesh, not inside the network.
        if not torch.compiler.is_compiling():
            self._validate_meshes(boundary_meshes, prediction_points, prediction_boundaries)
        ### The given conditions in canonical order.
        names = [name for name in self.boundary_names if name in boundary_meshes]
        meshes = [boundary_meshes[name] for name in names]
        ### A surface task's outputs: the token rows of its prediction boundaries, as
        ### (start, stop) ranges in token order, adjacent conditions' ranges merged;
        ### None when every given boundary is one, so that the head reads the token
        ### state itself (decided by name: a test on the counts would guard on them).
        out_rows = None
        if prediction_points is None and not set(names) <= set(prediction_boundaries):
            out_rows, start = [], 0
            for name, m in zip(names, meshes):
                if name in prediction_boundaries:
                    if out_rows and out_rows[-1][1] == start:
                        out_rows[-1] = (out_rows[-1][0], start + m.n_cells)
                    else:
                        out_rows.append((start, start + m.n_cells))
                start += m.n_cells
        measures = [cell_measures(m) for m in meshes]
        like = meshes[0].points
        global_td = self._select_global_data(global_data, like.device)
        global_rows = self._global_rows(global_td, 1, like)
        ### The length unit, the measure-weighted RMS pairwise distance of the given
        ### boundaries at full resolution, each cell with its own second moment (the
        ### parallel-axis term), so it is exact for the piecewise-flat surface: the same
        ### for every draw of the tokens and every subdivision of the cells. With
        ### declared scales, the case's other scales (_case_scales).
        gauge = self._gauge(meshes, measures)
        scales = self._case_scales(names, meshes, measures, global_rows, gauge) if self._scaled else None
        g_unit, g_scalars = self._global_fields(global_rows, like, scales)
        ### The boundary cells, each condition's with its own inputs, concatenated in
        ### canonical order over a batch axis of one.
        boundary = torch.cat(
            [
                self._condition_tokens(
                    name,
                    m.cell_centroids[None],
                    m.cell_normals[None],
                    w[None],
                    self._nondimensional(
                        name, m.cell_data.select(*(FieldSchema.key(k) for k in self.boundary_schemas[name])), scales
                    )[None],
                    g_unit,
                    g_scalars,
                )
                for name, m, w in zip(names, meshes, measures)
            ],
            dim=1,
        )
        query = None
        if prediction_points is None:
            out_points = boundary["points"][0]
            if out_rows is not None:
                out_points = torch.cat([out_points[a:b] for a, b in out_rows])
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
                dist, direction = exact_boundary_geometry(
                    q_points, points, cells, kernel=self.resolve_geometry_kernel(q_points)
                )
            query["wall_distance"], query["wall_direction"] = dist, direction
            ### The boundary tokens: the sample of the cells (sample_boundary_cells),
            ### each kept cell with its Horvitz-Thompson measure. A uniform start
            ### from the global torch RNG in training, a fixed one in evaluation; in
            ### float64 from the input meshes whatever the model's precision or the
            ### autocast, the weights cast to the tokens' measures at the end.
            if self.training:
                start = torch.rand((), dtype=torch.float64, device=q_points.device)
            else:
                start = torch.full((), EVAL_START, dtype=torch.float64, device=q_points.device)
            with torch.autocast(q_points.device.type, enabled=False):
                draw = [draw_inputs(m) for m in meshes]
                kept, weights = sample_boundary_cells(
                    [a for a, _ in draw], [x for _, x in draw], self.boundary_tokens, start
                )
            boundary = boundary.apply(lambda t: t[:, kept], batch_size=[1, kept.shape[0]])
            boundary["measure"] = weights[None].to(boundary["measure"].dtype)
            query = query[None]
        return boundary, query, g_unit, g_scalars, gauge, scales, out_points, global_td, out_rows

    @staticmethod
    def _point_set(pts) -> TensorDict:
        """An interior point set's tokens: the points of a point-cloud Mesh or a bare
        tensor."""
        points = pts.points if isinstance(pts, Mesh) else pts
        return TensorDict(points=points, batch_size=points.shape[:1])

    def _tree_read_pairs(self, bnd: TensorDict, qry: TensorDict, q_wall, g_unit):
        """The boundary read's pairs (:mod:`tree_read`): each point's near cells and
        admitted clusters at the forest's opening angle, their kernel basis
        and log measures; ``bnd`` and ``qry`` are the boundary's and the points'
        tokens and ``q_wall`` ``(B, Q)`` the points' built-in wall distance (frame
        units)."""
        return walk_range_pairs(self.kernel_basis, self._tree_read_walk(bnd, qry, q_wall, g_unit))

    def _tree_read_walk(
        self, bnd: TensorDict, qry: TensorDict, q_wall, g_unit, walk_points: bool = True
    ) -> TreeReadWalk:
        """The forest and the points' lists of :meth:`_tree_read_pairs`, before the
        kernel basis
        (:func:`~physicsnemo.experimental.models.isla.tree_read.tree_read_walk`)."""
        r, n_hat, vecs, log_w = bnd["r"], bnd["n_hat"], bnd.get("vecs", None), bnd["log_w"]
        q_r, q_nhat = qry["r"], qry["n_hat"]
        log_frac = log_w[..., 0] - torch.logsumexp(log_w[..., 0], dim=1, keepdim=True)
        return tree_read_walk(
            query_r=q_r,
            query_n=q_nhat,
            query_wall=q_wall,
            boundary_r=r,
            boundary_n=n_hat,
            boundary_vecs=vecs,
            log_frac=log_frac,
            g_unit=g_unit,
            theta=self._tree_read_theta,
            eps=EPS,
            kernel=self.resolve_tree_read_kernel(q_r),
            walk=self.tree_read_kernel,
            build=self.tree_read_kernel,
            walk_points=walk_points,
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
        condition: str | None = None,
        cell_second_moments: Float[torch.Tensor, "batch tokens"] | None = None,
    ) -> TensorDict:
        """The batched network on bare token tensors, the tokens of one boundary
        condition without per-cell fields (``condition``, default the model's one
        declared condition); the release tests pin it, and it
        runs the same :meth:`_forward_tokens` as :meth:`forward`.

        ``points``, ``normals`` ``(B, N, d)``; ``measure_weights`` ``(B, N)``
        (``None``: every token the same weight); ``global_inputs`` a flat
        ``{dotted name: tensor}`` mapping over the declared global fields
        (per-case values ``(B,)`` / ``(B, d)``, or one row shared across the
        batch); ``query_points`` ``(B, Q, d)`` for an interior model, with the
        boundary surface each case's geometry is computed from as
        ``surface_points`` ``(B, V, d)`` and ``surface_cells`` ``(B, F, d)``
        (vertex indices; :func:`exact_boundary_geometry`); ``cell_second_moments``
        ``(B, N)`` the tokens' own second moments in the length unit
        (:func:`~physicsnemo.experimental.models.isla.geometry.cell_second_moments`;
        ``None``: the tokens are points). The tokens are used as
        given: nothing is sampled here. Returns the outputs
        over the ``(B, N_out)`` tokens the head read.
        """
        if condition is None and len(self.boundary_names) == 1:
            condition = self.boundary_names[0]
        if self._scaled:
            raise NotImplementedError(
                "the batched routine takes no boundary meshes, from which the declared scales are "
                "computed; pass the meshes to forward"
            )
        if not torch.compiler.is_compiling():
            self._validate_inputs(
                condition, points, normals, measure_weights, query_points, surface_points, surface_cells
            )
        fields = global_inputs or {}
        missing = [k for k in self.global_schema if k not in fields]
        extra = [k for k in fields if k not in self.global_schema]
        if missing or extra:
            raise ValueError(
                f"global_inputs fields do not match the declared set: missing {missing!r}, "
                f"unexpected {extra!r}"
            )
        b, n, _ = points.shape
        data = TensorDict({FieldSchema.key(k): v for k, v in fields.items()}, batch_size=[])
        g_unit, g_scalars = self._global_fields(self._global_rows(data, b, points), points)
        boundary = self._condition_tokens(
            condition,
            points,
            normals,
            None if measure_weights is None else measure_weights.reshape(b, n),
            TensorDict(batch_size=[b, n]),
            g_unit,
            g_scalars,
        )
        query = None
        if query_points is not None:
            geometry = [
                exact_boundary_geometry(
                    query_points[i], surface_points[i], surface_cells[i], kernel=self.resolve_geometry_kernel(query_points)
                )
                for i in range(b)
            ]
            query = TensorDict(
                points=query_points,
                wall_distance=torch.stack([g[0] for g in geometry]),
                wall_direction=torch.stack([g[1] for g in geometry]),
                batch_size=query_points.shape[:2],
            )
        gauge = rms_pairwise_distance(
            points,
            None if measure_weights is None else measure_weights.reshape(b, n),
            EPS,
            cell_second_moments,
        )
        return self._redimensionalize(self._forward_tokens(boundary, query, g_unit, g_scalars, gauge=gauge), None)

    def _forward_tokens(
        self,
        boundary: TensorDict,
        query: TensorDict | None,
        g_unit: torch.Tensor,
        g_scalars: torch.Tensor | None,
        gauge: torch.Tensor | None = None,
        out_rows: list[tuple[int, int]] | None = None,
    ) -> TensorDict:
        """The batched network on the token inputs.

        ``boundary`` holds the boundary tokens over ``(B, N)``: ``points`` and
        ``normals`` ``(B, N, d)``, ``measure`` ``(B, N)`` (absent: every token the
        same weight) and the conditions' inputs in the block layout, ``seed`` and
        the block ``vecs`` and ``tens`` (:meth:`_condition_tokens`);
        ``query`` (interior) the prediction-point tokens over ``(B, Q)``:
        ``points`` and the built-in geometry ``wall_distance`` ``(B, Q)`` and ``wall_direction`` ``(B, Q, d)``
        (:func:`exact_boundary_geometry`);
        ``g_unit`` ``(B, K, d)`` and ``g_scalars`` ``(B, S)`` (or ``None``) the
        global inputs (:meth:`_global_fields`); ``gauge`` ``(B, 1, 1)`` the length
        unit (``None``: measured on the given boundary tokens); ``out_rows`` the
        surface task's output tokens as ``(start, stop)`` ranges of the boundary
        tokens (``None``: every boundary token).
        Returns the declared outputs over the ``(B, N_out)`` tokens the head reads.

        The token state is one TensorDict over ``(B, N)``, the boundary tokens,
        the prediction-point tokens, then the padding tokens that complete ``N``
        to a multiple of :data:`TOKEN_ALIGNMENT` (:func:`_pad_tokens`), with fields ``h``, ``r``, ``n_hat``,
        ``g_hat``, ``log_w`` and the block-level ``vecs`` and ``tens``; the slice
        blocks take its tensors (module docstring, *Data layout*).
        """
        tokens, gauge, geo_kernel = self._boundary_tokens(boundary, g_unit, gauge)
        b, n = tokens.batch_size

        ### n_boundary counts the boundary tokens only: the kernel reads update
        ### the tokens appended below.
        n_boundary = n
        if query is not None:
            q_tokens, q_wall = self._query_tokens(query, tokens["log_w"], g_unit, g_scalars, gauge, boundary["points"])
            if self.kernel_reads is not None:
                kernel_pairs = self._tree_read_pairs(tokens, q_tokens, q_wall, g_unit)
            tokens = torch.cat([tokens, q_tokens], dim=1)
            _hint_token_axis(n + q_tokens.shape[1], self._token_widths)
        padded = _pad_tokens(tokens, TOKEN_ALIGNMENT)
        ### The end of the real tokens when padding follows them, None when nothing was
        ### padded: then every read below is that of the unpadded token set.
        n_real = None if padded is tokens else tokens.batch_size[1]
        tokens = padded
        _hint_token_axis(tokens.batch_size[1], self._token_widths)

        h, log_w, r = tokens["h"], tokens["log_w"], tokens["r"]
        vecs, tens = tokens.get("vecs", None), tokens.get("tens", None)
        for i, block in enumerate(self._block_callables()):
            h = block(h, log_w, r, tokens["n_hat"], tokens["g_hat"], EPS, vecs, tens, geo_kernel=geo_kernel)
            if self.kernel_reads is not None:
                ### The points' residual update from the boundary cells' states at this
                ### depth; the boundary tokens (and so the slices) are untouched.
                h_b, h_p = h[:, :n_boundary], h[:, n_boundary:n_real]
                delta = self.kernel_reads[i](
                    h_b.reshape(-1, h.shape[-1]), h_p.reshape(-1, h.shape[-1]), kernel_pairs
                )
                parts = [h_b, h_p + delta.view_as(h_p)]
                h = torch.cat(parts if n_real is None else [*parts, h[:, n_real:]], dim=1)
        tokens.set("h", h)

        ### The tokens the head reads: the prediction points in the interior mode,
        ### the prediction boundaries' cells in the surface mode, never the padding
        ### tokens. The head frame builds its slice anchors from ALL tokens (r, log_w and h).
        if query is not None:
            out = tokens[:, n_boundary:n_real]
        elif out_rows is not None:
            out = torch.cat([tokens[:, a:b] for a, b in out_rows], dim=1)
        else:
            out = tokens if n_real is None else tokens[:, :n_real]
        ### The frame is built after the head's coefficients: autograd accumulates the
        ### gradient of h over its consumers in the order they were built.
        return self._head(out, (lambda: self._frame_anchors(h, log_w, r)) if self.needs_frame else None)

    def _boundary_tokens(self, boundary: TensorDict, g_unit: torch.Tensor, gauge: torch.Tensor | None):
        """The boundary tokens' state over ``(B, N)`` (:meth:`_forward_tokens`), the
        length unit (measured on the tokens when ``gauge`` is ``None``) and the
        geometry kernel the blocks are called with."""
        points = boundary["points"]
        b, n, _ = points.shape
        _hint_token_axis(n, self._token_widths)
        geo_kernel = self._apply_geo_kernel(points)
        K, dim = self.n_global_vectors, self.dim
        g_hat = g_unit[:, None].expand(b, n, K, dim)

        ### Frame: no origin; the one length unit is the measure-weighted RMS
        ### pairwise distance, computed through the variance identity
        ### sum_ij w_i w_j |x_i-x_j|^2 = 2 sum_i w_i |x_i - c|^2 with c the
        ### weighted mean (used only inside this scalar; the frame keeps no
        ### centre). Normalized weights make it converge under refinement where
        ### the total measure diverges, and make it invariant to a weight rescale.
        ### forward passes the full-resolution boundary's (exact for every draw).
        measure = boundary.get("measure", None)
        if gauge is None:
            gauge = rms_pairwise_distance(points, measure, EPS)
        normals = boundary["normals"]
        n_hat = normals / normals.norm(dim=-1, keepdim=True).clamp_min(EPS)

        if measure is None:
            ### No measures given: every token carries the same measure, i.e. the
            ### count-weighted quadrature.
            log_w = points.new_zeros(b, n, 1)
        else:
            log_w = torch.log(measure.to(points.dtype).clamp_min(EPS))[..., None]

        ### Boundary inputs: each condition's seed through its input layer, its
        ### vectors and tensors through its maps onto the shared channels.
        h, b_vecs, b_tens = self._boundary_channels(boundary)
        tokens = TensorDict(
            h=h,
            r=points / gauge,
            n_hat=n_hat,
            g_hat=g_hat,
            log_w=log_w,
            **self._block_extras(b_vecs, b_tens, b, n, points),
            batch_size=[b, n],
        )
        return tokens, gauge, geo_kernel

    def _query_tokens(self, query: TensorDict, log_w, g_unit, g_scalars, gauge, like):
        """The prediction points' tokens over ``(B, Q)`` (:meth:`_forward_tokens`) and
        their built-in wall distance ``(B, Q)`` in frame units; ``log_w`` and ``like``
        are the boundary tokens' log weights and points."""
        b = like.shape[0]
        K, dim = self.n_global_vectors, self.dim
        query_points = query["points"]
        nq = query_points.shape[1]
        q_nhat, s, builtin = self._query_geometry(query, gauge, like)
        q_g = g_unit[:, None].expand(b, nq, K, dim)
        q_seed = [self._dots(q_nhat, q_g)]
        if g_scalars is not None:
            q_seed.append(g_scalars[:, None, :].expand(b, nq, -1).to(like.dtype))
        h_q = self.embed(self.qt.seed(torch.cat([*q_seed, builtin], -1)))
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
            **self._block_extras(None, None, b, nq, like),
            batch_size=[b, nq],
        )
        return q_tokens, s[..., 0].abs()

    @staticmethod
    def _query_geometry(query: TensorDict, gauge, like):
        """The prediction points' direction ``(B, Q, d)``, signed distance ``s``
        ``(B, Q, 1)`` in frame units and built-in channels ``(B, Q, 2)``."""
        ### The query's geometry from the boundary surface: the signed distance
        ### to it (the built-in channels s and asinh(s / 1e-6)) and the gradient
        ### of the signed distance, sign(d) u with sign(0) = +1 (the token's
        ### "normal" in every invariant), from the exact closest point. Both are
        ### continuous through the surface, so a point on it (d zero or a rounding
        ### error of either sign) gets the code of its neighbours on either side.
        with torch.autocast(like.device.type, enabled=False):
            dist = query["wall_distance"].to(like.dtype)
            u = query["wall_direction"].to(like.dtype)
            q_nhat = torch.where((dist >= 0)[..., None], u, -u)
            s = (dist / gauge.squeeze(-1))[..., None]  # (B, Q, 1), in frame units
            builtin = torch.cat([s, soft_log_length(s)], -1)
        return q_nhat, s, builtin

    def _frame_anchors(self, h, log_w, r):
        """The head frame's prototypes ``(B, S, hidden)`` and slice anchors ``(B, S, d)``
        from the tokens' states, log weights and positions."""
        ### The frame's prototypes are synthesized once from the tokens with their
        ### measures (the same quadrature that weights the anchors), so the frame is
        ### refinement invariant like the blocks; both routings read those prototypes.
        protos_f = self.frame_assign.prototypes(h, log_w)
        a_f = _softmax_over_points(self.frame_assign.logits(h, protos_f) + log_w)
        return protos_f, torch.einsum("bns,bnc->bsc", a_f, r)

    def _head(self, out: TensorDict, frame) -> TensorDict:
        """The declared outputs of the tokens ``out`` (their ``h``, ``r``, ``n_hat`` and
        ``g_hat``), with the head frame's ``(prototypes, anchors)``
        (:meth:`_frame_anchors`, or a callable returning them; ``None`` without outputs
        of rank 1 or more)."""
        K, dim = self.n_global_vectors, self.dim
        ### With declared scales or output_stats the head and the assembly run in the
        ### parameters' precision under autocast: bf16's relative step (4e-3) would be
        ### carried into the physical outputs (Pa pressure) by the redimensionalization.
        head_dtype = self.head.weight.dtype
        if self._fp32_head:
            head_ctx = lambda: torch.autocast(out["h"].device.type, enabled=False)  # noqa: E731
            h_out = out["h"].to(head_dtype)
        else:
            head_ctx, h_out = contextlib.nullcontext, out["h"]
        with head_ctx():
            coeffs = self._coeff_layout.unpack(self.head(self.norm_out(h_out)))
        if not self.needs_frame:
            ### Scalar-only outputs are the coefficients themselves; no token basis.
            return coeffs
        ### Head frame: the radial basis vector is the direction from the token's
        ### soft slice anchor (measure-weighted mean of token positions), not
        ### from a centroid; translation covariant.
        protos_f, z_f = frame() if callable(frame) else frame
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
        if self._fp32_head:
            true_vecs, pseudo_vecs = true_vecs.to(head_dtype), pseudo_vecs.to(head_dtype)
        ### Each output of rank >= 1 from its coefficients, in place of them.
        with head_ctx():
            for i, (name, layout) in enumerate(self.output_layouts.items()):
                if layout.spec.rank:
                    key = FieldSchema.key(name)
                    index = getattr(self, f"_output_index_{i}")
                    coeffs.set(key, _assemble_output(coeffs.get(key), layout, true_vecs, pseudo_vecs, index))
        return coeffs


class _QueryChunks:
    """The result of :meth:`ISLA.query_chunks`: ISLA-DTT's boundary side run once,
    what the prediction points read of it held as leaves, and the points' side by
    ranges of points (``chunks(start, stop)``), then ``chunks.backward()``."""

    def __init__(self, model: ISLA, inputs: tuple) -> None:
        boundary, query, g_unit, g_scalars, gauge, scales, out_points, global_td, _ = inputs
        self.model, self.query, self.g_unit, self.g_scalars = model, query, g_unit, g_scalars
        self.scales, self.out_points, self.global_td = scales, out_points, global_td
        self._roots, self._leaves = [], []
        tokens, self.gauge, self.geo_kernel = model._boundary_tokens(boundary, g_unit, gauge)
        self.like, self.log_w = boundary["points"], tokens["log_w"]
        q_nhat, s, _ = model._query_geometry(query, self.gauge, self.like)
        ### The forest now, each chunk's points walked by the chunk: the lists held are
        ### one chunk's, and the fast walk's array limit applies to a chunk's pairs.
        self.walk = model._tree_read_walk(
            tokens, {"r": query["points"] / self.gauge, "n_hat": q_nhat}, s[..., 0].abs(), g_unit, walk_points=False
        )
        ### The boundary side's token axis completed as forward completes its own
        ### (_pad_tokens): at an unaligned count cuBLAS reduces some token-axis bf16
        ### products, the slice anchors among them, with split-K partial sums rounded to bf16.
        n_cells = tokens.batch_size[1]
        tokens = _pad_tokens(tokens, TOKEN_ALIGNMENT)
        h, log_w, r = tokens["h"], tokens["log_w"], tokens["r"]
        vecs, tens = tokens.get("vecs", None), tokens.get("tens", None)
        self.slices, self.sources = [], []
        for i, block in enumerate(model._block_callables()):
            h, slices = block(
                h, log_w, r, tokens["n_hat"], tokens["g_hat"], EPS, vecs, tens, geo_kernel=self.geo_kernel, slices=True
            )
            self.slices.append(tuple(self._leaf(t) for t in slices))
            h_b = h[:, :n_cells].reshape(-1, h.shape[-1])
            self.sources.append(self._leaf(model.kernel_reads[i].sources(h_b, self.walk)))
        self.frame = (
            tuple(self._leaf(t) for t in model._frame_anchors(h, log_w, r)) if model.needs_frame else None
        )

    @property
    def n_points(self) -> int:
        return int(self.query.shape[1])

    @functools.cached_property
    def n_pairs(self) -> int:
        """The number of (point, source) pairs every prediction point reads, counted
        without making the lists."""
        return walk_pair_count(self.walk)

    def _leaf(self, t: torch.Tensor) -> torch.Tensor:
        leaf = t.detach().requires_grad_(t.requires_grad)
        self._roots.append(t)
        self._leaves.append(leaf)
        return leaf

    def __call__(self, start: int, stop: int) -> Mesh:
        """The output Mesh of the prediction points ``start`` to ``stop``."""
        m = self.model
        q_tokens, _ = m._query_tokens(
            self.query[:, start:stop], self.log_w, self.g_unit, self.g_scalars, self.gauge, self.like
        )
        pairs = walk_range_pairs(m.kernel_basis, walk_points_range(self.walk, start, stop))
        h = q_tokens["h"]
        vecs, tens = q_tokens.get("vecs", None), q_tokens.get("tens", None)
        for i, read in enumerate(m._block_read_callables()):
            h = read(
                h, self.slices[i], q_tokens["r"], q_tokens["n_hat"], q_tokens["g_hat"], EPS, vecs, tens,
                geo_kernel=self.geo_kernel,
            )  # fmt: skip
            delta = m.kernel_reads[i].read(self.sources[i], h.reshape(-1, h.shape[-1]), pairs)
            h = h + delta.view_as(h)
        q_tokens.set("h", h)
        fields = m._redimensionalize(m._head(q_tokens, self.frame), self.scales)
        points = self.out_points[start:stop]
        return Mesh(points=points, point_data=fields[0].to(points.device), global_data=self.global_td)

    def backward(self) -> None:
        """The boundary side's backward pass with the gradients the chunks' backward
        passes accumulated in the leaves; releases the boundary side's graph."""
        roots = [(t, leaf.grad) for t, leaf in zip(self._roots, self._leaves) if leaf.grad is not None]
        self._roots, self._leaves = [], []
        if roots:
            torch.autograd.backward([t for t, _ in roots], [g for _, g in roots])
