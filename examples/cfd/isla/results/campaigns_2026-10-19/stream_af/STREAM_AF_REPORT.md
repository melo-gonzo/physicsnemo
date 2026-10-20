# STREAM-AF (2026-10-19): the H-RESID residual stream re-measured off AeroZoo

Replaces the data behind `fig-isla-stream`: `results/renderings_2026-10-18/stream_norms.json`. That file held the AeroZoo
lanes, which are retired; commit e99aed240 cut it, and a copy is at `src/stream_norms_aerozoo_ref.json`. The new files keep
its exact schema (`protocol`, `sources`, `runs.<run>.{arm, reads, lr, protocol, seed, code, epochs[...]}`, with each epoch
holding `epoch, source_job, source_file, code, case_ids, n_tokens, stream_out, embed_out, norm_in[12], amp[12],
amp_bcast[12], amp_mlp[12], log10_prod_amp, val_rel_l2`). The book's figure code needs only new run keys.

| file | content |
|---|---|
| `stream_norms_airfrans.json` | AirfRANS uvolume 2-D: raw `afu_r_isla_seed42/43`, A2 `afu_bn2_isla_seed42/43`, A2-shipped `afu_a2s_isla_seed42/43`; 18 epochs each |
| `stream_norms_deepjebpp.json` | DeepJEB++ surface: raw `cl_r_jeb_isla_seed42/43` (epochs 39, 40 only), A2 `jbn2_isla_seed42/43` (1, 9, 19, 29, 39, 40) |
| `summary.txt` | the key numbers below, for all three datasets (`src/summarize.py`) |
| `src/` | the probe as run, sbatch files, reducers, manifest for DeepJEB++ |
| `out/`, `out_jeb/` | per-checkpoint probe jsons (the reducers' inputs, 21 MB); `sbatch_logs/` the job logs |

## How it was made
- Probe: H-RESID Instrument 1, `hresid_probe.py` (md5 514f3c6b..., the same file the AeroZoo jobs 224693/240803/240804/270135
  ran). Run here as `src/hresid_probe_af.py` (AirfRANS, md5 8d7cfc70...) and `src/hresid_probe_af2.py` (DeepJEB++, md5
  8347c0a1...). The only changes: the recipe dir, lane manifest and checkpoint root come from env vars, and
  `model.compile_blocks=` is dropped from the lane's extra so the forward runs eager, as it did on AeroZoo. Hooks-on vs
  hooks-off outputs were identical (max |diff| 0.0) on every lane.
- Reducers: `src/stream_reduce_af.py` and `src/stream_reduce_jeb.py` are `scripts/stream_reduce.py` with new SRC/RUNS.
- LHR jobs: **288480** (raw + a2s), **288481** (bn2), **289830** (DeepJEB++), 2026-10-19, short QoS, 1 node each, one
  lane per GPU. Each lane ran on its own snapshots, from its train.log and manifest:

| lanes | code | recipe | venv |
|---|---|---|---|
| afu_r_isla | code_v20 | recipe_v8 | .venv-recipe |
| afu_bn2_isla | code_v20a2 (code_v20 + the A2 reads; same blocks.py diff as code_v26 to code_v30) | recipe_v8 | .venv-recipe |
| afu_a2s_isla | code_a2s (shipped code; trained with compile_blocks=true, probed eager) | recipe_v11 | .venv-recipe |
| cl_r_jeb_isla (AGA jobs 1017653/4; mdlus copied read-only AGA to LHR, md5 checked) | code_v20 (the isla/*.py md5s match AGA's) | recipe_v8 | .venv-recipe |
| jbn2_isla | code_v20a2 | recipe_v8 | .venv-recipe |

- Cases: the first 4 of the val sampler, as on AeroZoo. AirfRANS `full_test` gave indices 9, 10, 12, 14; DeepJEB++
  `test_clean` gave 76-79, the four load cases of one part (012-079-IS02 dia/hor/tor/ver). Cells are each lane's
  training count: AirfRANS 100,000, DeepJEB++ 200,000 (AeroZoo used 200,000). bf16 forward.
- AirfRANS epochs: 1, 9, 19, 29, 39, 43, 49, 59, 69, 99, 149, 199, 249 (the AeroZoo A2 set), then 299-499, since the
  schedule runs 500 epochs. All of them were on disk.

## Key numbers (mean token norm after the last block, `stream_out`; depth factor = norm_in[11] / norm_in[0] at the last probe)

| dataset | arm | stream_out first probe to last | growth over training | max | depth factor | sum log10 amp |
|---|---|---|---|---|---|---|
| AeroZoo (retired) | raw | 7.4e3-9.1e3 to 3.1e11-1.5e12 (e249) | x4e7-1.7e8 | 1.5e12 | 5.7e7-2.0e8 | 9.7-11.1 |
| AeroZoo (retired) | A2 | 275-298 to 3.5e3 | x12 | 3.8e3 | 2.7-3.1 | 2.0 |
| AirfRANS | raw | 841-1451 to 2.6e4-2.9e4 (e499) | x20-31 | 3.0e4 | 158-160 | 2.8 |
| AirfRANS | A2 (bn2) | 196-234 to 802-824 | x3.5-4.1 | 826 | 4.9-5.6 | 1.0 |
| AirfRANS | A2 shipped (a2s) | 195-235 to 796-838 | x3.6-4.1 | 838 | 4.9-5.3 | 1.0 |
| DeepJEB++ | raw (e39, e40 only) | 1.8e16-1.5e17 (e39) to 7.8e15-6.4e17 (e40) | (no early checkpoints) | 6.4e17 | 3.9e13-1.2e15 | 14.5-17.2 |
| DeepJEB++ | A2 | 698-783 (e1) to 2.0e4-2.1e4 (e40) | x27-29 | 2.1e4 | 224-247 | 2.6-2.7 |

- The ratio of raw to A2 at the end of training: AeroZoo about 1e8-4e8; **AirfRANS 31-36**; DeepJEB++ about 4e11-3e13.
- On AirfRANS the raw-read stream grows in the deep blocks. At e499 seed 42, norm_in runs 39, 39, 63, 65, 81, 118, 172,
  304, 538, 1024, 2185, 6293, and per-block amp reaches 2.8 and 4.0 in blocks 10-11. A2's runs 78-439, with amp at most
  1.2 except blocks 0 and 11. The growth is geometric but mild. It is not the runaway seen on AeroZoo.
- Both AirfRANS arms stop growing at epoch 99: StepLR(step 100, gamma 0.1) under Muon, the same scheduler as AeroZoo's
  recipe_v9 default. stream_out is flat to 3 digits from e149 to e499.
- AirfRANS sanity read (val_rel_l2, normalized, 4 cases, e499): raw pressure 0.027-0.032 and velocity 0.019-0.021; A2
  0.020-0.027 and 0.017. The raw-read lanes still fit AirfRANS. On DeepJEB++ the raw lanes are broken at e39-40: von Mises
  0.99-1.02 and displacement 0.92-1.14, against A2's 0.36-0.38 and 0.14-0.15.

## How it differs from the AeroZoo original, and what limits the comparison
1. AirfRANS shows the same sign with a much smaller effect: raw/A2 about 35 at the end, against about 1e8 on AeroZoo. A
   caption or paragraph that speaks of a stream "growing without bound" would not hold on AirfRANS, and neither would the
   loss collapse. DeepJEB++ shows the runaway, larger even than AeroZoo's, but its raw lanes keep only epochs 39-40. So it
   can fill panel (b), the depth profile, but not panel (a), the series over training.
2. Raw and A2 lanes on AirfRANS: bn2 matches raw best (same code base and recipe_v8, only the reads differ). a2s is the
   shipped code on recipe_v11, but its stream_out agrees with bn2's to within 2% (depth factor within 6%).
3. Unlike AeroZoo, AirfRANS is 2-D, with 100,000 cells and 500 epochs. On AeroZoo the last probe was e249 out of 250 epochs.
4. A2 on DeepJEB++ is itself not small: x28 over 40 epochs, depth factor about 235, still rising at e40, where the
   40-epoch schedule never steps the rate down. That is about 80x AeroZoo A2's depth factor.
5. The DeepJEB++ cases are four load cases of one part (sampler order), so they are less varied than four different
   geometries.
