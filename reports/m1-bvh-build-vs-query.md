# M1 — BVH build vs query throughput

Round-1 plan: `geopt-datagen-round1-plan.md` §3.2 (deliverable 5), exit criterion G4.
Device: `cpu`

> CUDA was requested but is unavailable on this host; benchmark ran on CPU. Absolute numbers are NOT the H100 target numbers in the plan; the ratios (build vs query, query vs end-to-end) and trends are what this run measures. FCPW unavailable (`fcpw` not installed); FCPW columns are `n/a`.

| mesh | n_tris | wp.Mesh build (ms) | warp query 1k (ms) | warp query 10k (ms) | warp query 100k (ms) | sdf end-to-end 100k (ms) | fcpw build (ms) | fcpw query 100k (ms) |
|------|--------|--------------------|--------------------|----------------------|----------------------|--------------------------|------------------|----------------------|
| sphere | 1024 | 0.132 ± 0.001 | 14.823 ± 0.015 | 151.741 ± 0.892 | 1541.253 ± 5.006 | 403.522 ± 2.416 | n/a | n/a |
| cube | 12 | 0.004 ± 0.000 | 0.277 ± 0.005 | 2.809 ± 0.011 | 27.973 ± 0.017 | 27.369 ± 0.054 | n/a | n/a |
| torus | 1600 | 0.223 ± 0.007 | 24.386 ± 0.042 | 245.466 ± 0.644 | 2483.840 ± 3.066 | 1878.446 ± 2.079 | n/a | n/a |

All cells: `median ± IQR/2` over the configured trials.
