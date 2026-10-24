# HL-DUP: the constrained NN-crust rebuild of the root outline (DESIGN_HOLE.md, fe23f2fd), prototype result

2026-10-24, 10:30-10:50 EDT. Approved by the supervisor at 10:29 as a CPU-only scratch prototype. Code is in `lhr/`
(`hl_crust.py` 7051d249, plus the diagnostics `hl_crust_diag.py` f2308dea, `hl_orphan_order.py` ceecd168,
`hl_stl_root.py` 6b6a21cb and `stl_cap.py` 4fbfd16e). Records are in `out/crust/` (md5s at the end). LHR output is
only under `$T/bcsidecar/hilift_hole/crust/`. Jobs 368599, 368618, 368645, 368651, 368668, 368671 and 368682 (all
cpu-short), plus one 20 MB read on the login node. Total under 0.2 core-h of the 3 cap.

## Verdict

**The design fails its own validation. It builds no hole on any broken root, and it should not be used to generate
symmetry planes.** Every failure was caught by a check, so no wrong loop passed. The cause is the one the design
flagged: at the sharp trailing edges the outline's sampling violates NN-crust's condition, and the half-plane rule
breaks there.

## Results (`out/crust/crust.jsonl`, `crust_summary.json`)

| test | result |
|---|---|
| clean roots LHC001_AoA_10, _12 | the rebuild adds 0 edges; 1 loop; 612,136.61 in² against curator's 612,136.62 (Hausdorff 0.0002 in) |
| knock-out, the 5 broken roots' gap patterns, on each clean root (10) | rebuilt edge set exact in 10 of 10, but all 10 **fail** the registered transfer check (below) |
| knock-out, random runs (gap lengths from the broken roots, 34 % deleted), 20 per clean root | **14 of 40 exact**; 14 fail the half-plane check, 12 rebuild 1-2 wrong edges |
| the five broken roots | all 5 **fail** the first check: 16-17 outline points coincide in (x, z) |

- **Transfer.** 98.1-98.7 % of LHC001's 11,189 loop vertices have a broken-root vertex within 1e-3 in. The other
  142-271 lie up to 0.72-4.64 in from any. They cluster near the slat (x about 117, z about 168 in) and below the
  main element (x 938-951, z about 95 in), where the geometries differ. The transferred patterns are therefore
  approximate, and they pass only on the edge set.
- **Random-run failures.** These are the decisive ones.
  - Half-plane: all 14 at one vertex, the flap trailing-edge tip (2583.33, 725.29), a loose end with no Delaunay
    neighbour in the opposite half-plane.
  - Wrong edges: chords of about 100 in, with midpoints at (2407-2416, 257-259), next to the main element's
    trailing edge.
- **Sampling report** (the clean loop). eps = longer adjacent edge / local feature size: median 0.011, 99th
  percentile 0.25, maximum 8.2. 63 vertices (0.56 %) have eps of 1/3 or more, at the main trailing edge
  (2413.5, 309.4), the flap trailing edge (2583.3, 725.2) and (2562.0, 292.9). The largest turning angle is 106.8°,
  an interior angle under 90°, where the half-plane rule cannot hold.
- **Broken roots.** The coincident points are not copies. They are pairs of distinct stored vertices at bitwise-equal
  (x, z) with y = 0 against 2e-5 or 4.3e-4 in: 10-11 used+used pairs and 6-7 orphan+orphan pairs. As a diagnostic
  only (not the design), merging them exactly lets four roots reach the completion step, where they fail at
  (795.76, 99.14) with no opposite half-plane neighbour. LHC065 has a fixed-edge vertex of degree 3.
- **No order in the file.** Consecutive stored on-plane vertices are a median 780-1,000 in apart, so the storage
  order does not chain the orphans (`orphan_order.jsonl`).

## A finding beside the prototype: the source STL holds the root outline, and it is a dataset constant

41 case folders (LHC001-LHC005) also store the source surface, `geo_<case>.stl.pmsh` (334 k points). Its root is not
open: it is a planar cap of 966 triangles on 968 points in y = 0 (`stl_root.jsonl`, `stl_cap.jsonl`,
`stl_cap_all.jsonl`).

- **The cap is curator's hole.** On all 39 readable STLs the cap is one region of 612,127.85 ± 0.002 in², with area
  ratio 0.999986 and Hausdorff distance 0.061 in to curator's stored hole of LHC001_AoA_10.
- **It is the same everywhere.** The 39 caps fall into two coordinate md5 groups (20 and 19 cases) that agree to
  0.002 in² (last float32 bits). That holds across 4 geometries and every angle of attack, since the STLs are in the
  body frame.
- **Two STLs are incomplete.** geo_LHC004_AoA_8 has no `meta.json`, and the job stopped there before
  geo_LHC005_AoA_10.
- **Fit to the wing data.** Every on-plane CFD vertex of the five broken roots lies within 0.011 in of LHC001's
  outline (hole3). So the root section looks like a constant of the dataset, and its source outline is stored with the
  data.

This gives a parameter-free outline from the dataset's own source geometry, but it is stored for only 41 of 1,800
cases. Using it for the others assumes the root section is common to all 180 geometries. The evidence points that way
(4 geometries' STLs, and all broken roots' CFD vertices), but it is not proven for each case. It is a choice between
shared geometry and per-case data, which the supervisor and Peter must make; I recommend neither here.

## What would make a construction robust (not built)

- **A corner-aware reconstruction**, such as Gathan (Dey and Wenger 2001) or the conservative crust, validated by the
  same knock-out test. It still assumes dense sampling away from corners, and the trailing edges here have eps up to
  8. About 4-6 h, with uncertain success.
- **The STL cap as the outline**, with per-case verification that every on-plane CFD vertex and open edge lies on it
  (a check, not a fit). About 2-3 h for all 1,800 cases, about 30 core-h. Its legitimacy is the decision above.

## md5s

| file | md5 |
|---|---|
| out/crust/crust.jsonl | 5e223821de6e0961c2eb932e2721afdb |
| out/crust/crust_summary.json | 4181d1521d01efbe5491977b3b399ede |
| out/crust/crust_diag.json | 9112714c4ad4509d118e5879d3c660e6 |
| out/crust/crust_diag_xz.json | be91f69392c51c4bb78175ab53e67689 |
| out/crust/orphan_order.jsonl | 1b49da6bda7c2ec9059f1f9845dd1ae8 |
| out/crust/stl_root.jsonl | cc4ad74b56af0c72d780ae65afcc19e2 |
| out/crust/stl_cap.jsonl | f0e39eb35795ff2196e46a9db202fdb1 |
| out/crust/stl_cap_all.jsonl | 6b6fa9f030b0abe7697e0ac0f3b4924f |
| lhr/hl_crust.py | 7051d249c4b9dce9f2cb98ec220ce012 |
