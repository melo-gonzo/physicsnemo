# STREAM-AF: the residual stream of trained ISLA checkpoints, raw slice reads against normalized reads

Data behind `fig-isla-stream` (book/03-isla.qmd). Copied from `~/gh/physicsnemo/scratch/isla-stream-af-2026-10-19/`
(the lane's own report is `STREAM_AF_REPORT.md`; `summary.txt` its key numbers, made by `src/summarize.py`).

| file | content |
|---|---|
| `stream_norms_airfrans.json` | AirfRANS uniform interior, 4 test cases (indices 9, 10, 12, 14), 100,000 cells, bf16 eager forward: raw-read `afu_r_isla_seed42/43`, normalized-read `afu_bn2_isla_seed42/43` (same code base and recipe as the raw arm, only the reads differ) and `afu_a2s_isla_seed42/43` (the standing's ISLA arm); 18 epochs each, 1 to 499 |
| `stream_norms_deepjebpp.json` | DeepJEB++ surface, 4 test cases (76-79, four load cases of one part), 200,000 cells: raw-read `cl_r_jeb_isla_seed42/43` (epochs 39, 40 only), normalized-read `jbn2_isla_seed42/43` (1, 9, 19, 29, 39, 40) |
| `src/` | the probe as run (`hresid_probe_af.py`, `hresid_probe_af2.py`), sbatch files, reducers, the DeepJEB++ lane manifest |

Schema: `runs.<run>.{arm, reads, lr, protocol, seed, code, epochs[...]}`, each epoch holding `epoch, n_tokens, stream_out`
(mean token norm after the last block), `embed_out`, `norm_in[12]` (mean token norm entering each block), `amp[12]`,
`amp_bcast[12]`, `amp_mlp[12]`, `log10_prod_amp`, `val_rel_l2` (4-case validation error, normalized units).

LHR jobs 288480 (raw and a2s), 288481 (bn2), 289830 (DeepJEB++), 2026-10-19. The per-checkpoint probe outputs (`out/`,
`out_jeb/`, 21 MB, the reducers' inputs) and the job logs stay in the scratch directory and are not copied.
