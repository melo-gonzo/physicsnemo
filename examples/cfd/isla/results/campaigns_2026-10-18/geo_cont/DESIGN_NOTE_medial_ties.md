# Design note: the lug's medial-surface direction ties (not implemented; 2026-10-18)

**The defect.** A prediction point's direction is the unit gradient of its distance to the boundary: u = (x - p) / |x - p|,
where p is the closest surface point (or the closest cell's normal, oriented toward the point). On the medial surface, the
set of points with two or more closest surface points, u is not defined. The thin lug plate is the main case: a node on its
mid-thickness plane is equally far from the top and bottom faces, so u = +n or -n, and float32 rounding of the posed
coordinates picks which. With exact geometry, 0-2.3 % of a lug case's nodes reverse direction under the posed motion
(6 cases, LHR job 273927; the per-case instrument of the smokes, dirflip_probe.py, places them by the medial gap g).
ISLA reads u as the token's "normal" in odd invariants: n.g_k, rho.n, n.m (blocks.py) and the frame contractions of the
query inputs. So such a node's features change sign, and because the prediction points join the encoder as tokens, the
perturbation spreads to the slices. No encoding of d removes it, because d is continuous across the medial surface.

**Options**, each with what it would fix and what it would break:

1. **A sign-invariant direction feature, u ⊗ u (a nematic, or line field).** Replace the odd uses of u by even ones:
   (n.g)^2, (rho.n)^2, rank-2 contractions g^T (u u^T) g; or keep the direction only as a rank-2 query tensor u u^T.
   - Fixes: every antipodal tie (u vs -u) exactly, so the plate's mid-plane, where both faces are parallel.
   - Breaks or loses: the side information that u carries, i.e. which face is nearer. That is redundant only when the
     queries are on one side of every face. Ties at corners and non-parallel faces (directions at an angle other than
     180°) remain. ISLA's covariance machinery puts n in every frame and treats it as a polar vector; making it a line
     field changes the seed and block invariants, so it is a structural change that needs the full contract battery.
2. **Features from the two nearest faces.** Compute the closest point on the surface and the closest point on the part of
   the surface facing away from the first (e.g. the nearest cell with n.n1 < 0), and give the token (d1, d2, u1, u2) as a
   symmetric pair: d1 + d2 (local thickness), |d1 - d2| (distance to the medial surface), u1 + u2 and u1 u1^T + u2 u2^T.
   - Fixes: the mid-plane tie, with symmetric functions of the pair continuous where the label of "first" swaps. It also
     gives the model the local thickness, a physically meaningful quantity for a plate under bending or tension.
   - Breaks or costs: a second exact search per point, a definition of "the opposite face" that needs a threshold on n.n1
     (a constant to defend), and an ill-defined second face in thick or convex regions. It adds model content: new inputs
     and a new structure, so it needs the full battery.
3. **A smoothed direction**: the gradient of a soft-min distance, -tau log sum_i w_i exp(-|x - y_i| / tau) over the
   surface samples. Continuous everywhere; on the medial surface the two pulls cancel to a short vector.
   - Fixes: every tie, with continuity, in one construction.
   - Breaks: needs a length tau, a tuned constant unless it is tied to L with a resolution argument. The gradient is no
     longer unit length (its norm becomes a feature); it is costlier (a sum over nearby samples); and near the wall it
     differs from the exact normal by O(tau).
4. **Leave u, and report the residual pose sensitivity** as a known limit of point-wise directions on thin solids.

**Discriminating test**, if one is pursued: the posed/native ratio on the lug with the remaining encoding fixed,
CONTROL vs option 1 (the cheapest), same seeds. Predicted to remove most of the lug's excess posed error if the
reversing nodes carry it. Option 1's cost to native accuracy is measured on AirfRANS and Tensile2d, where no tie occurs.
