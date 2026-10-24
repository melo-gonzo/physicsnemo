# ALIGN8: GPU-h at stake from token counts that are not multiples of 8 / 32 (2026-10-24)

Per-case counts are from exposure/exposure.tsv (child ALIGN8-EXPO, LHR cpu-short 362817, 362818). The classes are from
exposure/exposure_classes.tsv. N is fixed per case: the draws change which cells, never how many. "4-rank step" is the
slowest-rank class over 4 independent cases. Overheads are base / fix from results/repro_table.md (GB300, bf16; v3 363327,
v4 363671). Lane GPU-h are sacct actuals of the standings rows, summed over seeds. The harness step has no DDP or
dataloader, so each fraction is an upper estimate for a training step.

| dataset (standing rows) | path | lane GPU-h | 4-rank step class | overhead | at stake |
|---|---|---|---|---|---|
| AirfRANS ISLA (r090eag_af_isla 42/43) | eager | 178.3 | odd 0.949, even 0.051 | +8.9 % odd, +4.9 % even | about 14 |
| AirfRANS ISLA-DTT (r090eag_af_dtt 42/43) | eager | 220.8 | same | +6.9 % / +3.6 % | about 14 |
| AirfRANS + freestream (AF-WHOLE, canonical) | eager | about 400 (same lanes) | even 0.996 | +4.9 % (ISLA), +3.6 % (DTT) | about 16 |
| DrivAerML Surface whole boundary (205,054, if adopted) | compile_blocks / eager | 32.7 (r090_dr) | even 1.000 | +20 % | about 6 |
| lug ISLA (r090eag_lug 42/43) | eager | 156.9 | odd 0.66, even 0.32 | wall unresolved (launch-bound); GPU time +5 % | at most 8 |
| lug ISLA-DTT (r090fix_lug_dtt 42/43) | compile_blocks | 324.3 | same | same | at most 16 |
| Tensile2d (r090eag_t2dz 42-45) | eager | 51.4 | odd 0.95 | wall unresolved; GPU time +8.6 % | at most 4 |
| DeepJEB++ (r090_jb 42/43) | compile_blocks | 197.3 | even 0.99 | not resolved at 24,000 tokens | at most 10 |
| AeroZoo (all 7 splits), DrivAerML Volume (body or whole), DrivAerML Surface body, SHIFT-SUV | any | about 6,000 | aligned on every case (multiples of 32) | 0 | 0 |

Total for one two-seed set of the exposed standings rows: about 28 GPU-h measured (AirfRANS, current form), up to about
70 with the small-token upper bounds and DrivAerML Surface whole. That is 1-3 % of a 2,500-4,000 GPU-h rerun.

Counterfactual, the reason the fix matters: AeroZoo and DrivAerML are aligned only because their caps (200,000) and
budgets are multiples of 32 and every case has more cells than the budget. A count off by one there costs +32-38 %
(odd) or +20 % (even) per step in either path: about 1,100-2,200 GPU-h on REF-090's AeroZoo lanes alone.
