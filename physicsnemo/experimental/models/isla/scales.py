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

"""ISLA's own scales: the field declarations' ISLA-local keys and the per-case scales.

A field of a boundary condition's schema, of ``global_schema`` or of
``output_schema`` may carry, next to its rank spec, the keys of :data:`SCALE_KEYS`:
``units`` (exponents of named scales: the built-in length ``L`` and any scale a
field defines), ``defines: <name>`` (the field defines the scale ``name``: a rank-1
field its measure-weighted RMS magnitude and, on the boundary, the direction ``g``;
a rank-0 global its value, consumed rather than read as an input) and, for an
output, a boundary field or a rank-0 global input, ``scale`` (a positive constant
``c``, the typical size of the field made dimensionless by its units: an input enters
as ``x / (c prod scale^exp)``, and the head predicts ``y / (c prod scale^exp)``, both
of order one). :func:`split_scale_keys` pops them before
:meth:`~physicsnemo.mesh.fields.FieldSchema.parse`, so the shared
:class:`~physicsnemo.mesh.fields.RankSpec` grammar is untouched. Private to
:mod:`physicsnemo.experimental.models.isla`."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

import torch

from physicsnemo.mesh.fields import RankSpec

#: The built-in scale, the length unit of the boundary.
LENGTH = "L"
#: The key with which a field defines a scale (its name, the key's value).
DEFINES = "defines"
#: The ISLA-local keys of a field declaration.
SCALE_KEYS = ("units", DEFINES, "scale")


def _check_local(kind: str, name: str, local: Mapping) -> dict:
    """One field's ISLA-local keys, checked and in a JSON-safe form. Whether the
    scales its ``units`` name are defined is checked against every field
    (:func:`parse_scales`)."""
    where = f"{kind}[{name!r}]"
    out = {}
    if "units" in local:
        units = local["units"]
        if not isinstance(units, Mapping):
            raise TypeError(
                f"{where}: units must be a mapping of scale name to exponent; got {units!r}"
            )
        for k, e in units.items():
            if not isinstance(k, str):
                raise TypeError(f"{where}: units names a scale by a string; got {k!r}")
            if isinstance(e, bool) or not isinstance(e, Real) or not math.isfinite(e):
                raise TypeError(
                    f"{where}: the exponent of {k!r} must be a finite number; got {e!r}"
                )
        out["units"] = {
            k: (int(e) if float(e).is_integer() else float(e))
            for k, e in units.items()
            if e
        }
    if DEFINES in local:
        defined = local[DEFINES]
        if not isinstance(defined, str) or not defined:
            raise TypeError(
                f"{where}: {DEFINES} names a scale by a non-empty string; got {defined!r}"
            )
        if defined == LENGTH:
            raise ValueError(
                f"{where}: {LENGTH!r} is the built-in length unit; a field cannot define it"
            )
        if kind == "output_schema":
            raise ValueError(f"{where}: an output cannot define a scale")
        out[DEFINES] = defined
    if "scale" in local:
        c = local["scale"]
        if (
            isinstance(c, bool)
            or not isinstance(c, Real)
            or not (math.isfinite(c) and c > 0)
        ):
            raise ValueError(
                f"{where}: scale must be a positive finite number; got {c!r}"
            )
        out["scale"] = float(c)
    return out


def split_scale_keys(kind: str, specs):
    """``specs`` without the ISLA-local keys, and those keys by dotted field name.

    A field is a :class:`~physicsnemo.mesh.fields.RankSpec` or a mapping with a
    non-mapping ``rank``; any other mapping is a group, walked with its name as the
    prefix (the grammar's nested and dotted forms give the same dotted names).
    Everything else is left for :meth:`~physicsnemo.mesh.fields.FieldSchema.parse`
    to accept or refuse."""
    extras: dict[str, dict] = {}
    if not isinstance(specs, Mapping):
        return specs, extras

    def walk(group: Mapping, prefix: str) -> dict:
        out = {}
        for name, value in group.items():
            dotted = f"{prefix}.{name}" if prefix else name
            if isinstance(value, Mapping) and not isinstance(value, RankSpec):
                if "rank" in value and not isinstance(value["rank"], Mapping):
                    local = {k: value[k] for k in SCALE_KEYS if k in value}
                    if local:
                        extras[str(dotted)] = _check_local(kind, str(dotted), local)
                    value = {k: v for k, v in value.items() if k not in SCALE_KEYS}
                else:
                    value = walk(value, dotted)
            out[name] = value
        return out

    return walk(specs, ""), extras


def merged(schema, extras: Mapping):
    """The declaration a checkpoint stores: the parsed ``schema`` (``None`` when
    empty) with each field's ISLA-local keys merged back."""
    if not extras:
        return schema or None
    return {name: {**spec, **extras.get(name, {})} for name, spec in schema.items()}


def check_output_stats(stats, output_schema) -> dict | None:
    """``output_stats`` (``{name: {"mean", "std"}}``, the format of
    ``NormalizeMeshFields.stats``; a ``type`` key is ignored) as JSON-safe floats.
    Each name is a declared output, each std positive, and the mean of a field of
    rank 1 or more zero (a nonzero vector mean would break rotation covariance)."""
    if stats is None:
        return None
    if not isinstance(stats, Mapping):
        raise TypeError(
            f"output_stats must be a mapping of output name to {{mean, std}}; got {stats!r}"
        )
    out = {}
    for name, s in stats.items():
        if name not in output_schema:
            raise ValueError(
                f"output_stats[{name!r}]: not a declared output ({list(output_schema)!r})"
            )
        mean = torch.as_tensor(s["mean"], dtype=torch.float64)
        std = torch.as_tensor(s["std"], dtype=torch.float64)
        if std.numel() != 1 or not float(std) > 0:
            raise ValueError(
                f"output_stats[{name!r}]: std must be one positive number; got {s['std']!r}"
            )
        if output_schema[name].rank == 0:
            if mean.numel() != 1:
                raise ValueError(
                    f"output_stats[{name!r}]: a scalar's mean is one number; got {s['mean']!r}"
                )
        elif bool((mean != 0).any()):
            raise ValueError(
                f"output_stats[{name!r}]: the mean of a field of rank {output_schema[name].rank} must be 0 "
                f"(a nonzero mean breaks rotation covariance); got {s['mean']!r}"
            )
        out[name] = {
            "mean": float(mean.reshape(-1)[0]) if mean.numel() else 0.0,
            "std": float(std),
        }
    return out

@dataclass(frozen=True)
class DeclaredScales:
    """The scales the declarations define (:func:`parse_scales`), as the model reads
    them: per field the units and the constant ``scale`` of the outputs, the rank-0
    globals and each condition's fields; the rank-1 sources of the one RMS scale (the
    boundary's ``(condition, field)`` pairs or one global) and its name; the rank-0
    globals whose values are scales (scale -> field); the checked ``output_stats``;
    whether any scale is declared (``scaled``) and whether the head runs in the
    parameters' precision under autocast (``fp32_head``)."""

    output_units: dict
    output_scale: dict
    global_units: dict
    global_scale: dict
    boundary_units: dict
    boundary_scale: dict
    boundary_sources: list
    global_sources: list
    defined_values: dict
    rms_scale: str | None
    output_stats: dict | None
    scaled: bool
    fp32_head: bool


def parse_scales(
    out_extras, b_extras, g_extras, output_stats, output_schema, boundary_schemas, global_schema
) -> DeclaredScales:
    """The opt-in scales (the ISLA module docstring, *Scales*) from the declarations'
    ISLA-local keys (:func:`split_scale_keys`: ``out_extras``, ``g_extras`` and
    ``b_extras`` by condition), checked against the parsed schemas: which ranks may
    define a scale or carry units and a constant, one RMS scale at most, from boundary
    fields or one global, and every unit a defined scale; with ``output_stats``
    checked (:func:`check_output_stats`)."""
    decl = {"output_schema": out_extras, "global_schema": g_extras}
    decl.update({f"boundary_schemas[{n!r}]": e for n, e in b_extras.items()})
    schemas = {"output_schema": output_schema, "global_schema": global_schema}
    schemas.update({f"boundary_schemas[{n!r}]": s for n, s in boundary_schemas.items()})
    used = set()
    for kind, extras in decl.items():
        for name, local in extras.items():
            rank = schemas[kind][name].rank
            defined = local.get(DEFINES)
            if defined is not None and not (rank == 1 or (rank == 0 and kind == "global_schema")):
                raise ValueError(
                    f"{kind}[{name!r}]: a scale is defined by a rank-1 field or a rank-0 global; got rank {rank}"
                )
            if defined is not None and rank == 0 and ("units" in local or "scale" in local):
                raise ValueError(
                    f"{kind}[{name!r}]: the value of the scale {defined!r} is consumed, not an input; "
                    "it takes no units and no scale"
                )
            if kind == "global_schema" and rank == 1 and defined is None:
                if "units" in local or "scale" in local:
                    raise ValueError(
                        f"{kind}[{name!r}]: a rank-1 global that defines no scale enters as a "
                        "direction only, so units and a scale have no effect on it; declare its magnitude as a "
                        "rank-0 global"
                    )
            if kind == "global_schema" and rank == 1 and "scale" in local:
                raise ValueError(
                    f"{kind}[{name!r}]: a rank-1 global's magnitude is the scale {defined!r}; it takes no scale"
                )
            used |= set(local.get("units", {}))
    output_units = {n: e["units"] for n, e in out_extras.items() if e.get("units")}
    output_scale = {n: e["scale"] for n, e in out_extras.items() if "scale" in e}
    global_units = {
        n: e["units"]
        for n, e in g_extras.items()
        if e.get("units") and DEFINES not in e and global_schema[n].rank == 0
    }
    global_scale = {n: e["scale"] for n, e in g_extras.items() if "scale" in e}
    boundary_units = {
        c: {n: e["units"] for n, e in extras.items() if e.get("units")} for c, extras in b_extras.items()
    }
    boundary_scale = {
        c: {n: e["scale"] for n, e in extras.items() if "scale" in e} for c, extras in b_extras.items()
    }
    ### The rank-1 sources of the one RMS scale (boundary fields, pooled, or one
    ### global) and the rank-0 globals whose values are scales (consumed).
    boundary_sources = [
        (c, n) for c, extras in b_extras.items() for n, e in extras.items() if DEFINES in e
    ]
    global_sources = [
        n for n, e in g_extras.items() if DEFINES in e and global_schema[n].rank == 1
    ]
    defined_values = {}
    for n, e in g_extras.items():
        if DEFINES in e and global_schema[n].rank == 0:
            if e[DEFINES] in defined_values:
                raise ValueError(f"global_schema: two fields define the scale {e[DEFINES]!r}")
            defined_values[e[DEFINES]] = n
    rms = {b_extras[c][n][DEFINES] for c, n in boundary_sources}
    rms |= {g_extras[n][DEFINES] for n in global_sources}
    if len(rms) > 1:
        raise ValueError(f"at most one scale is defined by rank-1 fields; got {sorted(rms)!r}")
    rms_scale = next(iter(rms), None)
    if rms_scale in defined_values:
        raise ValueError(
            f"the scale {rms_scale!r} is defined by a rank-1 field and by the rank-0 global "
            f"{defined_values[rms_scale]!r}"
        )
    if boundary_sources and global_sources:
        raise ValueError(
            f"the sources of the scale {rms_scale!r} are rank-1 boundary fields or one rank-1 global, "
            f"not both: got {boundary_sources!r} and {global_sources!r}"
        )
    if len(global_sources) > 1:
        raise ValueError(
            f"at most one rank-1 global defines the scale {rms_scale!r}; got {global_sources!r}"
        )
    declared = {LENGTH, *defined_values, *rms}
    undefined = sorted(used - declared)
    if undefined:
        raise ValueError(f"units name {undefined!r}, which no field defines; the scales are {sorted(declared)!r}")
    has_source = bool(boundary_sources or global_sources)
    output_stats = check_output_stats(output_stats, output_schema)
    ### The scales are declared when any field has units or an input scale or
    ### defines a scale.
    has_input_scale = bool(global_scale or any(boundary_scale.values()))
    scaled = bool(used or has_source or defined_values or has_input_scale)
    ### The head and the redimensionalization run in the parameters' precision
    ### under autocast whenever a physical output is assembled or standardized. The
    ### outputs' precision is bounded by the bf16 trunk either way; the fp32 head keeps
    ### bf16's rounding of the head's own product and of a nonzero output mean (a
    ### mean / std of 0.3, for one) out of the outputs.
    fp32_head = scaled or output_stats is not None or bool(output_scale)
    return DeclaredScales(
        output_units=output_units,
        output_scale=output_scale,
        global_units=global_units,
        global_scale=global_scale,
        boundary_units=boundary_units,
        boundary_scale=boundary_scale,
        boundary_sources=boundary_sources,
        global_sources=global_sources,
        defined_values=defined_values,
        rms_scale=rms_scale,
        output_stats=output_stats,
        scaled=scaled,
        fp32_head=fp32_head,
    )


def scale_factor(units: Mapping | None, scales: Mapping):
    """``prod_k scales[k] ** units[k]`` (``None`` for a dimensionless field)."""
    if not units:
        return None
    factor = None
    for k, e in units.items():
        term = scales[k] ** e
        factor = term if factor is None else factor * term
    return factor


def rms_moments(sources):
    """``(sum a, sum a |v|^2, sum a v)`` in float64 over ``sources``, pairs of a
    per-cell vector field ``v (n, d)`` and the cells' measures ``a (n,)``."""
    s0 = s2 = s1 = 0.0
    for v, a in sources:
        v, a = v.double(), a.double()
        s0 = s0 + a.sum()
        s2 = s2 + (a * v.square().sum(-1)).sum()
        s1 = s1 + (a[:, None] * v).sum(0)
    return s0, s2, s1
