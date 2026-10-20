# GEO-CONT measurements of the current query-geometry encoding (2026-10-18, 21:45 EDT)

Branch psharpe/isla-geo-cont 04ecca138 (psharpe/isla-geo-exact b40537f9b + psharpe/isla ea397e04f merged).

Current encoding (model.py `_encode`): s = d / L with L = `rms_pairwise_distance` of the boundary tokens (the frame's length
unit); built-in channels [s, sign(s) log(|s| + 1e-12)]; declared rank-0 query fields (the stored `sdf`) get the same pair
(inputs.py, `scaled_scalars`). Direction u = exact_boundary_geometry's direction from the surface to the query
(the cell normal oriented toward the query; the pseudo-normal N when |x - p| <= 1e-12). With torch.sign(0) = 0 the log
channel is 0 at d = 0 exactly, while its limits are -/+27.6 at d -> 0+/0-.

## 1. Unit test, synthetic surfaces, float32 (test/experimental/models/isla/test_isla_query_encoding.py)

What the model ingests (forward hooks on the query-token embedding and the first block), max abs change. 96-gon (2-D) and
icosphere (642 vertices, 3-D), radius 1.3, off the origin; 4 random rigid motions (2 improper, translation ~2.6), scalings
1e-3 and 1e3; queries: the vertices, the cell centroids (on the surface but not vertices: d is rounding), and centroids
offset by +-delta L along the normal.

| | 2-D feature | 2-D direction | 3-D feature | 3-D direction |
|---|---|---|---|---|
| pose, vertices | 0 | 6e-7 | 0 | 1.4e-6 |
| pose, cell centroids (on-surface, d = rounding) | **36.8** | **2** (flip) | **41.9** | **2** (flip) |
| pose, +-1e-6 L | 0.16-0.20 | 1.4e-6 | 0.65-0.76 | 2.7e-6 |
| pose, +-1e-4 L | 1.8e-3 | | 5.7e-3 | |
| scale, cell centroids | 36.8 | 2 | 46.4 | 2 |
| across the surface, +1e-6 vs -1e-6 L | 28.5 | 2 | 28.6 | 2 |
| across, +-1e-4 L / +-1e-2 L | 19.1 / 9.9 | 2 | 19.1 / 9.9 | 2 |

## 2. Real data, LHR job 273927 (cpu-short, 1 min), probe/geo_probe.py, 6 cases each, native vs posed (CHAR-PROTOCOLS posed configs)

Full recipe pipelines of recipe_geo2 (cluster-only dataset configs), exact_boundary_geometry on the full boundary, float32.
Raw: probe/probe_273927.json, log probe/isla.gcont.probe-273927.log.

| | Tensile2d (plaid_tensile2d_2d) | lug (ginot_lug_volume) | AirfRANS (airfrans_uvolume_2d) |
|---|---|---|---|
| nodes per case | 6,143-6,209 | 6,640-11,077 | 165k-181k (200k cap; lanes use 100k) |
| L (gauge) | 1.42 | 0.054-0.068 | 0.41-0.43 |
| d > 0 / d = 0 / d < 0 (stored winding) | 0.92 / 0.08 / **0** | **0** / 0.32-0.39 / 0.61-0.68 | 0.994 / 0.006 / 0 |
| on-surface nodes (|d| <= 1e-6 L) | 494-496 | 2,524-3,616 | 908-1,018 |
| of which d == 0 exactly, native / posed | **all / all** | **all / all** | **all / all** |
| on-surface sign change or direction flip under pose | 0 | 0 | 0 |
| direction flips under pose, all nodes | 0 | 0-174 per case (0-2.3 %) | 0 |
| nodes with 0 < |s| < 1e-3 | none | none | 22 % (1 % quantile of |s| 5e-6) |
| current encoding: pose jump, on-surface / max any node | 0 / 2e-5 | 0 / 1e-5 | 0 / 0.11 (near-wall nodes) |
| asinh(s/1e-6): pose jump, max any node | 2e-5 | 1e-5 | 0.11 |
| asinh(s/1e-4): pose jump, max any node | 2e-5 | 1e-5 | 0.006 |
| stored sdf | 0 on the on-surface nodes, <= 0 inside | 0 on-surface, <= 0 inside | 0 on-surface, > 0 in the fluid |

Reading:
- On all three datasets the on-surface nodes are vertices of the boundary mesh, bitwise, in both frames, so exact geometry
  gives d = 0 exactly natively and posed. The rounding-sign mechanism of the A2 finding does not occur on this branch.
- The current encoding's discontinuity at d = 0 is real (section 1) but on this data it is a fixed, pose-independent
  "on-surface" code (log channel 0 on the surface, about -7..-1 one node in), not a pose instability.
- Orientation, from the data: Tensile2d's winding already puts d > 0 inside (no flip needed; its stored cell_data
  `normals` are outward but ISLA reads the winding); the lug's puts d < 0 inside (flip needed); AirfRANS d > 0 (none).
- The lug's remaining pose instability under exact geometry: 0-2.3 % of nodes per case whose direction reverses under
  the motion, off the surface (sign(d) u flips too): presumably mid-thickness ties between two opposite faces (the medial
  surface of the plate). Not checked node by node.
