# Draft for the ISLA book (GEO-TIEORDER, 2026-10-25). Lanes do not edit the book; the supervisor applies this.

## Notebook entry (18-notebook.qmd)

### 2026-10-25 — GEO-TIEORDER: the on-wall direction is a function of the mesh at near-equal normals; exact distance ties broken by position, and on the surface the least-norm point of the normals wherever they differ; joins ISLA 0.14.0 {#sec-nb-geo-tieorder}

- Defect (MAIN-MERGE, DEFECT_TORUS.md). At a planar quad's diagonal, u moved under renumbering at 51 of 1,152 torus
  diagonal midpoints, and float32 differed from float64 at 5 of 35.
- Two causes, both in `_closest_candidates_impl`:
  - Cells that share the closest point tie exactly in float64, and the closest cell was the least cell index.
  - On the surface the certified-distinct test chose between one normal and the normals' least-norm point, and its
    bound scales with the queries' precision.
- The same causes act at a fine polyline's vertices in 2-D (AirfRANS's airfoil) and off the wall at edge and vertex
  closest points (at float64 rounding).
- Rule (supervisor approval 00:50 EDT, DESIGN.md v2):
  - T1: exact distance ties are broken by the cells' canonical vertex positions, then parity.
  - T2: on the surface, u is the least-norm point of the hull of the normals of the cells holding the query wherever
    they differ, certified or not. The point lies within tau_k + max_j tau_j of each member and is the certified case's
    own value. Off the surface the certified rule stands.
  - No new constant.
- Reference and fast stay bitwise equal. Head 0f26e6db7, on psharpe/isla-geo-tieorder.
- float32 change (GB300 378186):
  - DrivAerML: on-wall u at 3.1k and 3.2k of 200k queries (median 2.8e-4, max 4.5e-3).
  - AirfRANS: 70-93 airfoil nodes per draw (max 1.6e-4).
  - Lug: rounding level. Tensile2d and AeroZoo: none.
  - d, the sign and the on-wall label: unchanged everywhere.
  - Renumbering moves and float32/float64 mismatches drop to 0 on every dump.
- HiLift (surface task, never calls the geometry):
  - The full-resolution wing (0.27 billion cells) runs out of memory on a GB300, in the base too.
  - On a 2.4M-cell slab, the base's u moved under renumbering at about 30 % of vertex and centroid queries; the head's
    at none.
  - u changes by more than 0.1 at about 1,000 of 50,000 queries, mostly at slivers and folds of the stored wing.
- Pose: DrivAerML's on-wall u becomes more stable (max |u_posed - R u| 1.4e-2 -> 3.1e-3; tie flips 630 -> 0). On the
  lug's and Tensile2d's planar regions the tie flag now flips (0 -> about 2,000 per lug case). The flag reaches only u's
  vertex gradient.
- Inference (377708): no metric moves by more than 2.7e-4 in log (DrivAerML), 6.5e-5 (AirfRANS) or 6.7e-7 (lug).
- Cost: the fast call +2 to 8 ms; training step +1 % on DrivAerML and at most +5 % on the lug.
- Suites: ISLA directory on CPU 476 passed; on GB300 734 passed, 1 skipped.

## Chapter changes

1. **05-forward-pass.qmd, the paragraph "No step of this geometry is left to the order of a search ..." (about line
   1574).** Replace its second and third sentences ("A point is tied only when ... do not make a tie.") with:
   "Off the surface a point is tied only when two of its tied features' directions are certified distinct: farther apart
   than the sum of bounds on their rounding, each $\sqrt 2$ times a bound on the sine of the computed direction's error
   (2 where that bound reaches 1). On the surface the direction is the least-norm point of the normals of the cells that
   hold the point wherever they differ, certified or not. That point lies within the normals' certified uncertainty and
   is the certified case's own value, so the direction does not depend on the cells' order or on the queries'
   precision. Cells at exactly the least distance are ordered by their positions, not their indices."
   Then add after "... the value is the least-norm point either way.":
   "From float32 values and their float64 copies, the on-wall direction is the same bitwise wherever both precisions
   hold the point by the same cells. The float32 band is the wider, so it may hold more cells (@sec-nb-geo-tieorder)."
   Source: scratch/isla-geo-tieorder-2026-10-25/RESULTS.md sections 2-3.
2. **21-design.qmd, the "Exact interior wall geometry" row (line 271).** Replace "with the least-norm point of the hull
   of the tied directions where two certified-distinct directions tie" with "with the least-norm point of the hull of
   the tied directions where two certified-distinct directions tie off the surface, and of the holding cells' normals
   wherever they differ on it, exact distance ties broken by the cells' positions".
3. **14-program-status.qmd.** Add a GEO-TIEORDER row: DONE 2026-10-25, branch psharpe/isla-geo-tieorder 0f26e6db7
   (unmerged; joins the 0.14.0 version commit). Question: "Is the on-wall direction a function of the input mesh at
   near-equal normals, and the same from float32 and float64 copies?" Predecessor: MAIN-MERGE's defect, GEO-BAND.
   Evidence: the per-dataset change, renumbering, precision, pose, inference and cost tables. Folder:
   scratch/isla-geo-tieorder-2026-10-25/.
4. No In-flight callout added or removed.
