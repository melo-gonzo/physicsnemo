ISLA
====

ISLA (Invariant Slice Attention) is an O(d)/SE(d)-equivariant (any spatial
dimension d >= 2), measure-aware slice-attention surrogate for steady
boundary-value problems. A boundary sample
(positions, unit normals, quadrature measures) goes in together with the
problem's data as *named fields of declared tensor rank*: global inputs (a
direction, a PDE parameter), per-boundary-cell inputs declared per named
boundary condition (an inlet's prescribed velocity, a loaded face's traction; a
no-slip wall declares none) and, for an interior task, per-query
inputs (a source term, a material label). Named fields of declared rank come
out: scalars,
vectors, tensors up to rank 4, each exactly covariant. The attention operates
only on invariants of the per-point vector set, and the frame is re-attached
at the heads, so rotation, reflection and translation covariance hold by
construction.

ISLA lives in ``physicsnemo.experimental.models.isla`` and may change between
releases.

Rank specs
----------

``output_schema``, each schema of ``boundary_schemas`` and ``global_schema`` map a
field name to a field declaration in the grammar shared by every mesh model
(:meth:`physicsnemo.mesh.fields.FieldSchema.parse`): a
:class:`~physicsnemo.mesh.fields.RankSpec` or a mapping of values ``{"rank": r,
"symmetric": False, "parity": "even"}`` (``r`` = 0 scalar, 1 vector, 2 rank-2
tensor, ...); a mapping of mappings is a nested group, and a name may contain
``"."`` for the same nesting. ``symmetric`` declares a fully
symmetric tensor, ``parity="odd"`` a pseudotensor output (a vorticity). One
name declares one field; several independent scalars are several rank-0
names. Inputs take ranks 0 to 2 (global inputs 0 to 1) and even parity;
outputs take ranks 0 to 4. Each parsed declaration is the model's
:class:`~physicsnemo.mesh.fields.FieldSchema` attribute of the same name.

Contracts
---------

- **Rotation, reflection and translation covariance.** The backbone sees only
  invariants (inputs of rank 1 and 2 enter through their dot products and
  contractions with the token's own directions); an output of rank ``r`` is
  contracted from invariant coefficients and tensor products of ``r`` true
  basis vectors (plus isotropic tensors), or of one pseudovector and ``r - 1``
  true vectors for odd parity. The frame has no origin, so translation
  invariance needs no centering; the length unit is a measure-weighted
  integral of the sampled geometry.
- **Measure-aware aggregation.** Slice states are quadrature-weighted means, so
  the routing reads a sampled integral rather than a raw point population.

Boundary conditions by name
---------------------------

``boundary_schemas`` declares the boundary conditions by name, each with the
per-cell fields it alone carries (``{}`` for a condition without fields). The
names are fixed at construction, in sorted order. ``forward`` takes any
non-empty subset of them, in any order: an absent condition contributes no
tokens, an unknown name is an error. Each condition has its own input layer
(whose column on a constant is the condition's learned offset, so its identity
is structural even without fields) and its own equivariant linear maps from its
vector and tensor inputs onto the shared channels of the slice layers, which
every condition shares. No tag field names a condition.

Mesh API
--------

``forward`` follows the mesh-native contract of the mesh models (GLOBE's): one
case per call, no batch axis. The boundary is a mapping of
:class:`~physicsnemo.mesh.Mesh` objects keyed by declared boundary-condition
name (each a ``Mesh[n_spatial_dims - 1, n_spatial_dims]`` whose cells have normals); the per-case data is
a ``TensorDict`` or mapping. Both tasks read every boundary they are given;
where the outputs are is the second argument: for an interior task
(``interior=True``) the prediction points, a point-cloud ``Mesh``, of which only
the points are read, or a bare ``(N, n_spatial_dims)`` tensor; for a surface task the names
of the boundaries at whose cells the outputs are predicted
(``prediction_boundaries``), the other boundaries being read as conditions only.
The result is a point-cloud ``Mesh`` whose ``point_data`` holds the declared
outputs (on a surface task at the prediction boundaries' cells, those boundaries
in sorted name order). Every constructor argument
and every ``forward`` input is keyword-only. The reference configuration on a
surface task is the class defaults (``hidden=512``, ``n_layers=12``,
``n_slices=512``), as the recipe configures it:

.. code-block:: python

    model = ISLA(
        output_schema={"pressure": {"rank": 0}, "wss": {"rank": 1}},  # a scalar and a vector field
        boundary_schemas={"vehicle": {}, "floor": {}, "inlet": {}},  # conditions without per-cell fields
        global_schema={"U_inf_dir": {"rank": 1}},  # one global direction
    )
    out = model(
        boundary_meshes={"vehicle": surface, "floor": floor, "inlet": inlet},  # Mesh[2, 3] each
        prediction_boundaries=["vehicle"],          # the outputs live on the vehicle
        global_data={"U_inf_dir": direction},       # (3,)
    )
    out.point_data["pressure"]                      # (N,), one per vehicle cell
    out.point_data["wss"]                           # (N, 3)

A tensor-valued target is declared with its rank
(``output_schema={"stress": {"rank": 2, "symmetric": True}}`` gives ``(N, 3, 3)``
in three dimensions);
a prescribed boundary vector is a rank-1 input of its condition
(``boundary_schemas={"loaded": {"traction": {"rank": 1}}, "fixed": {}, "free": {}}``,
read from ``boundary_meshes["loaded"].cell_data["traction"]``). An interior task is built with ``interior=True``
and called with ``prediction_points=``. Its ``boundary_meshes`` are the
boundaries at full resolution, the only boundary input: each prediction
point's signed distance to that surface (positive on the side the cells'
normals point to) and the gradient of that distance are computed inside the
model from the exact closest point on its cells and are never an input; the
distance enters through its ratio to the frame's length and the ratio's
``asinh(s / 1e-6)``, so the encoding is continuous through the surface (a point
on it gets the code of its neighbours on either side). The
boundary tokens are drawn from the same cells inside the forward pass. The
budget is ``n_boundary_tokens`` per case (default 200,000), shared over the
named boundaries by water filling (a boundary with fewer cells than its share
is kept whole); within a boundary each cell is drawn with probability
proportional to its measure (capped at one) by systematic sampling along the
Morton order of the cell centroids, and carries its Horvitz-Thompson measure,
so the token count is exactly ``min(cells, n_boundary_tokens)`` (static shapes)
and every measure-weighted read is an unbiased quadrature
of the full surface. In training the draw's start comes from the global torch
RNG (``torch.manual_seed`` reproduces it); in evaluation (``model.eval()``) it
is fixed, so inference is deterministic. A surface task reads every cell it is
given and samples none. The interior task is ISLA-DTT: the prediction points
are passive readers of the slices and of the boundary cells, so each prediction
depends on its own point and the boundary alone and any request can be answered
in chunks.

The unified external-aerodynamics recipe
(``examples/cfd/external_aerodynamics/unified_external_aero_recipe``) documents
the reference configuration, its variants, and the data-to-model mapping.

.. autoclass:: physicsnemo.experimental.models.isla.model.ISLA
    :show-inheritance:
    :members:
    :exclude-members: forward

.. autofunction:: physicsnemo.experimental.models.isla.sampling.sample_boundary_cells

.. autoclass:: physicsnemo.mesh.fields.RankSpec
    :members:

.. autoclass:: physicsnemo.mesh.fields.FieldSchema
    :members:
