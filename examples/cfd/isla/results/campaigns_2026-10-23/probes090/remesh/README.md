# REMESH-090 launcher (kernel/probe090_remesh, 2026-10-21)

Inference-only probe of ISLA 0.9.0's invariance to the discretization of its boundary, on the six REF-090 final
checkpoints (AirfRANS ISLA and ISLA-DTT, DrivAerML Volume ISLA). Built by the REMESH-PORT lane of PROBES-090
(repo scratch `scratch/isla-probes090-2026-10-21/remesh/`: FACTS_CHECKED.md, PREREG_SECTION.md, SMOKE.md).

Files
- `probe_remesh.py`: one checkpoint; all variants of its family per case from one read (infer.py's order); `--blind`
  writes only native metrics, the scorer comparison, non-degeneracy stats and timings. Docstring = the variant list.
- `probe_remesh.sbatch`: up to 4 runs per job (run i on GPU i), short QoS, node-local kernel caches; env RUNS, BLIND,
  N_CASES, CASE_CHUNK (+ --array), VARIANTS, OUT (default ./out).
- `reduce.py`: `python3 reduce.py <out> [--blind]` -> `<out>/reduced_{full,blind}.{md,json}`.
- `struct_check.py`: full-mode structure check that prints names and counts only (used on a smoke whose output was then
  deleted).
- `facts_dv.py`, `facts.sbatch`, `facts_dv.json`: raw-file facts (cpu job 309207): STL vs CFD vehicle, AirfRANS polylines.
- `drivaer_ml_volume_reference_stlbnd.yaml`: the dataset twin, also in `$T/recipe_p090rm/datasets/` (the recipe snapshot
  the probe runs: recipe_r090 + this file).

Smokes (blind, 3 cases): 309316/309317, 309384/309385, 309467/309468, final 309528 (AirfRANS) and 309565 (Volume) in
`smoke/`; full-path + case-split structure tests 309417 and 309529 (outputs deleted unread). Production: not submitted;
commands in PREREG_SECTION.md.
