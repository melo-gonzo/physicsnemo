# ANCHOR (E3b): interior writers drawn from the boundary, against interacting ISLA, passive points and ISLA-DTT, on AirfRANS

Copied from `~/gh/physicsnemo/scratch/isla-anchor-2026-10-18/`: pre-registration `PREREG.md`, jobs `JOBS.md`, the readout
`RESULTS.md`; `readout/` holds the machine-readable results (`scores.json` native and posed errors per arm and seed,
`verdict.json` the registered rule's inputs and outcome, `qpop_f32_reduced.*` and `qpop_f64_reduced.*` the
query-population probe, `cost_*.json` step times and memory) and the scripts that made them (`scores.py`, `verdict.py`,
`cost.py`).

Five arms, two seeds each (42, 43), AirfRANS uniform interior, 200 test cases at 100,000 points, one snapshot of the code
with the normalized slice reads and nearest-token query geometry (the previous input form): A interacting ISLA; B passive
prediction points (query-independent, the boundary the only writer); C ANCHOR, 50,000 boundary-drawn interior writers;
D ISLA-DTT; E 50,000 writers at the CFD mesh nodes (a diagnostic). Scoring jobs 290707-290715 and 290847 (LHR).
