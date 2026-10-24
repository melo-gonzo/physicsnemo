# GEO-BAND phase A results (probes only; 2026-10-24, 10:15-10:45 EDT)

Code under test: `$T/code_geothin3` (geometry.py 7472ecf0 = 921472488's). Variants of that geometry.py are built by
`probes/variant.py` (md5 57ecefcd), each its own module with BOTH custom ops renamed; every log prints the op each call
ran (`isla_closest_candidates[_fast]_gb<V>`), the package's md5 and each variant's md5. Jobs 368478 (first constants) and
368608 (the derived constants), GB300 short QoS; tables `out/tables_368478.md`, `out/tables_368608.md` (reducer
`probes/reduce_dumps.py`), raw `out/probes{A,F}_*/`. Inputs: GEO-THIN's canonical dumps, which match the merged head
(921472488's model and test tree equal 6bac57cbd's, and GEO-THIN's dumps job 366938 ran at e94d91c86, the boundary merge,
after which no input pipeline changed): DrivAerML Volume whole boundary (2 draws x 200k), AirfRANS with its far field
(4 draws x 100k), AeroZoo (200k), the 20 lug cases, Tensile2d.

## The variants (switches of variant.py)

| name | frame of the arithmetic | band E_k | distinct-direction test | closest-point rebuild |
|---|---|---|---|---|
| base / V0 | as given | (u + 13 u64)(\|x\| + R_k) | bitwise | x - sum w v, queries' precision |
| V1F | as given | u (\|x\| + R_k) + 91 u64 (\|x\| + R_k) (C2 = 17 in 2-D) | bitwise | as base |
| V2F | centred on c | u (\|x\| + R_k) + 91 u64 (\|x'\| + R'_k) | bitwise | as base |
| V4F | as given | as V1F | certified (tau below) | as base |
| **VF** | centred | as V2F | certified | (x - v0) - sum_{i>0} w_i (v_i - v0) in float64, cast once |
| VFA | centred | u (\|x - c\| + R_k) + 91 u64 (\|x'\| + R'_k) (GEO-THIN's cA) | certified | float64 |

`x' = x - c`, `R'_k` about `c`; `c` the measure-weighted centroid of the unique cells (int64 fixed-point sums, so a
function of the set of cells, not their order), rounded to the queries' precision, subtracted in float64 (exact for
float32 inputs). Job 368478 also ran the first-pass constants (C3 = 80, C2 = 19; V1-V4, VALL, VALLA, V2B = cB) and
the pieces alone (V3 centring, V3r rebuild); their rows agree with the ones below to the counts shown there.

## Item 1: derived bounds (`DERIVATION.md`)

The arithmetic share is 89 units worst case at the current switch threshold (the accurate face foot 88, not 77; an edge
point 17.5; Ericson's unguarded point 14.5), so C3 = 91 with the centring and the second-order terms; 2-D C2 = 17. The
switch threshold needs no bound of its own: any threshold tau is certified once C >= tau + 88. K = 168 holds as a worst
case (second-order terms 3e-15 relative). Effect of the derived constants alone (V1F against base, native):

| dataset | float32: changed d / u / tie / on-wall / sign | float64: changed tie (max \|du\|) |
|---|---|---|
| DrivAerML whole (2 draws) | 0 / 0 / 0 / 0 / 0 | 20 (1.9e-4) |
| AirfRANS + far field (4) | 0 everywhere | 0 |
| AeroZoo | 0 everywhere | 3 (6.6e-6) |
| lug (20 cases) | 0 everywhere | 103 (7.7e-6) |
| Tensile2d | 0 everywhere | 0 |

## Item 2: pose-invariant bands

In float32 the storage term u32 (|x| + R_k) is 1 - 1.7e-7 of the band, so centring the arithmetic share (V2F) changes
no float32 tie or on-wall decision and leaves the lug's case 1884 as it was. End to end (DTT-POSE's probe_impl.py on
`code_dttpose_c3`, ISLA 0.12.0 forest, the variant's geometry; the lug's 24 posed test cases, AirfRANS 24 self-posed):

| geometry | lug posed / native rel. RMS: median, max (case) | AirfRANS pressure: median, max |
|---|---|---|
| base (V0) | 1.54e-7, 1.74e-5 (1884) | 5.90e-5, 9.93e-5 |
| V2 (centred arithmetic, first constants) | 1.53e-7, 1.74e-5 (1884) | - |
| VF | 1.55e-7, 1.74e-5 (1884) | 4.39e-5, 7.82e-5 |
| V2A / VALLA / VFA (query storage term about c) | 1.51e-7 / 1.52e-7 / 1.52e-7, max 3.0e-7 (877) | VFA 4.39e-5, 7.82e-5 |
| V2B (both storage terms about c) | 1.57e-7, 3.0e-7 | - |

Posed against native on the dumps (rotation R and translation t RMS radii of the boundary; float32; counts summed over
draws, `out/tables_368608.md`, section pose):

| dataset | t | base: \|du\| > 1e-3 (max), tie flips, on-wall flips | VF | VFA |
|---|---|---|---|---|
| DrivAerML whole | 0 | 180 (0.96), 776, 0 | 209, 272, 0 | 213, 202, 0 |
| DrivAerML whole | 1 | 3000 (0.999), 3492, 0 | 3706, 4199, 0 | 1353 (0.87), 1633, 0 |
| AirfRANS + far field | 0 | 78 (0.016), 6, 0 | 79, 20, 0 | 79, 19, 0 |
| AirfRANS + far field | 0.1 | 45,539 (0.96), 9,442, 813 | 46,326, 6,261, 813 | 46,498, 6,725, 2,030 |
| AirfRANS + far field | 1 | 161,342, 66,797, 16,077 | 164,955, 35,927, 16,077 | 164,310, 36,090, 9,207 |
| AeroZoo | 1 | 4 (0.013), 76, 0 | 4, 8, 0 | 1, 5, 0 |
| lug (20) | 1 | 0 (2.6e-5), 90,221, 0 | 0, 25, 0 | 0, 61, 0 |
| Tensile2d | 1 | 0 (4.9e-6), 231, 0 | 0, 0, 0 | 0, 0, 0 |

- What the storage term does under the pose, and where it binds. The term must grow with the translation, because the
  posed float32 inputs really are rounded more coarsely: a band edge then moves (case 1884's node 6997, DrivAerML's
  medial nodes at t = 1). It binds hardest on AirfRANS with its far field, whose boundary RMS radius is 160 chords: at
  t = 1 the storage rounding (about 1e-5 chords) exceeds the first nodes' distance from the airfoil (2e-6 chords), so
  16,077 near-wall nodes become on-wall (d = 0) in every variant; at t = 0.1, 813. That is the on-wall rule reading the
  input's own precision (their coordinates cannot tell them from the wall), not arithmetic error: it is the same in
  every variant, whatever the frame of the arithmetic. The scorer's AirfRANS
  pose (one RMS radius of the interior points, end-to-end table) is milder: residual 4e-5 to 1e-4.
- Rotation alone (t = 0) still flips u at band edges on DrivAerML (180 to 213 nodes per 400k, up to 0.96) under every
  variant: re-rounding moves the gap between two features by about the band itself, so no threshold rule is free of
  edge flips. Measuring a band about c only stops the band from moving; it does not stop the gap from moving.
- Why centring R_k too broke case 1909 (GEO-THIN), confirmed: V2B makes the vertex term u32 |v - c|, which is smaller
  than the vertices' actual storage rounding u32 |v| in a posed frame (|v| includes the translation). Exactly medial
  nodes (u = 0 natively) then see a re-rounding gap the band no longer covers, untie, and take u = +-n (|du| = 1): 170
  lug nodes over the 20 dump cases at t = 1 (job 368478). The query-centred cA keeps the vertex term absolute, which is
  why it survives there.
- cA (VFA) is not a bound on the query's rounding, and it changes native outputs where the boundary's centroid is far
  from the queries: AirfRANS's centroid is the far field's centre, so near-airfoil queries get |x - c| >> |x| and a
  wider band: natively 216 near-wall AirfRANS nodes become on-wall (d to 0, sign flips) and DrivAerML gains 12,296 ties
  over 2 draws with |du| up to 1 (change table below).

## Item 3: centred frame and the float64 rebuild

NEARWALL's far-from-origin compute rounding (float32 against float64 arithmetic on the same float32 input, the case
translated by T = 0.1 L and L; asinh(d / eps) changes, p99.9 / max; section far):

| dataset, T | base, eps 1e-9 L | VF | note |
|---|---|---|---|
| DrivAerML whole, L | 0.0156 / 0.066 | 1.9e-7 / 2.9e-7 | |
| AeroZoo, L | 7.5e-4 / 2.8e-3 | 1.9e-7 / 2.6e-7 | |
| lug, L | 6.4e-6 / 1.2e-5 | 1.3e-7 / 2.3e-7 | |
| AirfRANS + far field, 0.1 L, queries off the wall in both precisions | 0.29 / 0.90 | 1.8e-7 / 2.5e-7 | 4,529 queries on the wall in float32 only (the storage band; unchanged) |
| AirfRANS + far field, L, off the wall in both | 0.41 / 0.81 | 1.8e-7 / 2.5e-7 | 29,788 on the wall in float32 only |

The compute part came entirely from rebuilding `x - closest` from absolute coordinates in the queries' precision (V3r
alone gives the same rows); centring the search arithmetic (V3) does not change it. What remains on AirfRANS at 30
chords is NEARWALL's input part: queries within the float32 storage rounding of the wall are on the wall.

## Item 4: the certified distinct-direction test

Two directions are distinct when |c_i - c_j| > tau_i + tau_j; tau is sqrt(2) times a bound s on the sine of each
computed direction's error (a normal: the vertices' storage rounding u|v| through the certified cross product, plus the
float64 normal's own bound; a direction from a point: the band E_k and, on a face, Ericson's misplacement, over D_k;
tau = 2 where s >= 1). A tie needs a certified-distinct pair. Effect: tie flips under the pose fall from 90,221 to 25
(lug), 231 to 0 (Tensile2d), 76 to 8 (AeroZoo), 66,797 to 35,927 (AirfRANS t = 1); u moved by at most 2.6e-5 at those
nodes either way. On DrivAerML it moves the flips to a second threshold that also grows with the storage term (u
flips > 1e-3: 3,000 to 3,856 at t = 1, 180 to 242 at t = 0), each bounded by the certified direction uncertainty.

## Native change of each piece (against base; float32 rows; the recipe's precision)

| dataset | V2F (centring) d / u / tie | V4F (distinct) tie (max \|du\|) | VF all: d (max rel) / u (max) / tie / on-wall | VFA: tie / on-wall (max \|du\|) |
|---|---|---|---|---|
| DrivAerML whole (400k) | 12 / 20 (3.7e-4) / 0 | 7,932 (6.5e-3) | 266,536 (1.0e-3) / 83,175 (6.5e-3) / 7,932 / 0 | 20,228 / 0 (1) |
| AirfRANS + far field (400k) | 1 / 3 / 0 | 327 (6.2e-4) | 189,400 (7.0e-3) / 330 (6.2e-4) / 327 / 0 | 582 / 216 (6.5e-3) |
| AeroZoo (200k) | 0 | 20 (6.0e-5) | 184,680 (2.1e-3) / 1,946 (1.3e-4) / 20 / 0 | 28 / 0 (0.013) |
| lug (20 cases) | 1 / 1,746 (6.7e-7) / 0 | 1,305 (1.6e-5) | 6,615 (4.7e-6) / 7,163 (1.6e-5) / 1,305 / 0 | 1,284 / 0 |
| Tensile2d | 0 | 0 | 1,425 (3.7e-6) / 0 / 0 / 0 | 0 / 0 |

VF changes no winding sign and no on-wall decision on any dataset. Its d and u changes are the float64 rebuild (the old
value's own float32 cancellation, up to 0.7 % of d at AirfRANS's first nodes) and the certified tie test. Float64
queries: VF changes 141 DrivAerML ties (max |du| 1.7e-3), 43 lug, 3 AeroZoo, 0 AirfRANS and Tensile2d, and removes every
posed tie flip on DrivAerML (6 to 0) and AirfRANS.

## Fast path and cost

- Fast against reference: 816 rows (every variant, dataset, precision), every output bitwise equal.
- DrivAerML whole-boundary call (GB300, median of 5 after 3 warm-up, 2 draws): reference 473-475 -> 486-488 ms (VF,
  +13 ms, +2.8 %), fast 404-407 -> 419-420 ms (+14 ms, +3.5 %); VFA +25 ms.

## Open, not explained

- Lug, float64 queries posed in float64: 12-49 nodes per 20 cases change u by 1 under every variant (float32: 0). Not
  the recipe's precision; not chased.

## Phase C: the branch (psharpe/isla-geo-band 9e0d52e27, merged with psharpe/isla eba038f08 as 2e9cd7fda)

- Bitwise contract: the branch's geometry against variant VF of 921472488 (check_vf.py, job 369316, GB300): 106 of 106
  runs bitwise on every op output, d, u and both gradients (DrivAerML whole boundary 2 draws x 200k; AirfRANS with its far
  field 4 draws; AeroZoo; the 20 lug cases; Tensile2d; float32 and float64; reference and fast). So every phase A
  number for VF is the branch's.
- Fast against reference: fast_plugin 0 of 588,689 queries differ (507 calls), jobs 369316 and 369573.
- Tests: whole ISLA directory on GB300 with every CUDA test (369573, code_geoband2 = 2e9cd7fda): 548 passed, 1 skipped
  (ISLA_AF_POSED_SAMPLE not set), 0 failed. Geometry files forced: 304 passed, 1 skipped. Whole directory on cpu-short:
  407 passed (369318, 9e0d52e27) and 409 passed (369574, 2e9cd7fda), 0 failed. Recipe tests (369319): 551 passed, 2
  failed (test_campaign_c_hooks, E line identical to the base's in GEO-THIN's 368221).
- New tests failing at 921472488 (writer's run on 921472488's package): far from the origin, relative distance error
  0.0579 > 4 u32; posed planar star, 8 of 8 queries tied. The arithmetic-share and float64 band tests fail there for
  want of the new constants (AttributeError), as guards.
- Inference check (pre-registered, gated; INFER_CHECK.md, job 369317): preflight 36 of 36 O summaries equal TIE-EXP's
  registered ones; every |log(N / O)| <= 2.96e-5 (criterion 0.01), every posed / native change <= 4.4e-5 (0.01), largest
  per-case |log(N / O)| 5.2e-3 (Tensile2d s43 displacement_y). Criterion met: no retrain.
- Sensitivity, not gated (blinded lanes; both trained with 0.9.0's geometry): AF-WHOLE afwhole_af_isla_seed42 (epoch 197)
  N / O within 0.99994-1.00003 per field, posed / native moves <= 6.1e-5; FLOOR-DRV cap_drv_isla_fdF_s42 (step 2000) N /
  O within 0.99990-1.00014 on train and held-out.
- DrivAerML's translated-pose u flips, 3,000 (921472488) to 3,706 (branch) per 400k at one RMS radius, are band-edge
  flips, each within its certified direction uncertainty (supervisor 11:24: state this wherever the numbers appear).
