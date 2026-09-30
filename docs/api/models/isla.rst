ISLA
====

ISLA (Invariant Slice Attention) is an O(3)/SE(3)-equivariant, measure-aware
slice-attention surrogate for steady boundary-value problems. A boundary sample
(positions, unit normals, quadrature measures) goes in together with the
problem's data as *named fields of declared tensor rank*: global inputs (a
direction, a PDE parameter), per-boundary-cell inputs (a condition tag, a
prescribed traction, a prescribed stress) and, in the interior modes, per-query
inputs (the signed distance). Named fields of declared rank come out: scalars,
vectors, tensors up to rank 4, each exactly covariant. The attention operates
only on invariants of the per-point vector set, and the frame is re-attached
at the heads, so rotation, reflection and translation covariance hold by
construction.

ISLA lives in ``physicsnemo.experimental.models`` and may change between
releases.

Rank specs
----------

``outputs``, ``boundary_inputs``, ``query_inputs`` and ``global_inputs`` map a
field name to a field declaration in the grammar shared by every mesh model
(:meth:`physicsnemo.mesh.fields.FieldSchema.parse`): a
:class:`~physicsnemo.mesh.fields.RankSpec` or a mapping ``{"rank": r,
"symmetric": False, "parity": "even"}`` (``r`` = 0 scalar, 1 vector, 2 rank-2
tensor, ...); a mapping without a ``"rank"`` key is a nested group, and a name
may contain ``"."`` for the same nesting. ``symmetric`` declares a fully
symmetric tensor, ``parity="odd"`` a pseudotensor output (a vorticity). One
name declares one field; several independent scalars are several rank-0
names. Inputs take ranks 0 to 2 (global inputs 0 to 1) and even parity;
outputs take ranks 0 to 4. The parsed declarations are the model's
:class:`~physicsnemo.mesh.fields.FieldSchema` attributes ``output_specs``,
``boundary_specs``, ``query_specs`` and ``global_specs``.

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
- **Query independence (optional).** With ``query_independent=True`` queries are
  decoded by passive read blocks and a prediction at one point does not depend
  on which other points are queried.

Every constructor argument and every ``forward`` input is keyword-only. The
reference configuration:

.. code-block:: python

    model = ISLA(
        outputs={"pressure": {"rank": 0}, "wss": {"rank": 1}},  # a scalar and a vector field
        global_inputs={"freestream": {"rank": 1}},  # one global direction
        hidden=192,
        n_layers=12,
        n_slices=256,
        mlp_ratio=4,
        geo_checkpoint=True,
    )
    out = model(
        points=points,                              # (B, N, 3)
        normals=normals,                            # (B, N, 3)
        measure_weights=measure,                    # (B, N)
        global_inputs={"freestream": direction},    # (B, 3)
    )
    out["pressure"]                                 # (B, N)
    out["wss"]                                      # (B, N, 3)

A tensor-valued target is declared with its rank
(``outputs={"stress": {"rank": 2, "symmetric": True}}`` returns ``(B, N, 3,
3)``); a prescribed boundary vector is a rank-1 boundary input
(``boundary_inputs={"traction": {"rank": 1}}``, passed as
``boundary_inputs={"traction": (B, N, 3)}``). Interior queries are passed as ``query_points`` (with
``query_normals`` and ``query_inputs``); ``query_tokens=True`` admits them as
interacting tokens, ``query_independent=True`` decodes them passively, and
``support_tokens=True`` adds a per-case computational support
(``support_points``, ``support_normals``, ``support_inputs``).

The unified external-aerodynamics recipe
(``examples/cfd/external_aerodynamics/unified_external_aero_recipe``) documents
the reference configuration, its variants, and the data-to-model mapping.

.. autoclass:: physicsnemo.experimental.models.isla.model.ISLA
    :show-inheritance:
    :members:
    :exclude-members: forward

.. autoclass:: physicsnemo.mesh.fields.RankSpec
    :members:

.. autoclass:: physicsnemo.mesh.fields.FieldSchema
    :members:
