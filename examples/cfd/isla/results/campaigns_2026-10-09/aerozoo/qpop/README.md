# QUERY-POP on AeroZoo (recipe_v9)

`az_qpop.py` asks whether a trained model's prediction at a point depends on the *population* of prediction points
it is scored with. It is the DrivAerML probe of the ISLA-DTT fork (`dtt_qpop.py`, E4) carried to AeroZoo v1.2's
interior task (`datasets/aerozoo_volume.yaml`) and to three arms:

| arm    | model yaml                     | model overrides                                                                                              | library snapshot |
|--------|--------------------------------|--------------------------------------------------------------------------------------------------------------|------------------|
| `isla` | `isla_volume_aerozoo`          | none                                                                                                         | `$T/code_v26`    |
| `gt`   | `geotransolver_volume_aerozoo` | none                                                                                                         | `$T/code_v26`    |
| `dtt`  | `isla_volume_aerozoo`          | `++model.query_independent=true ++model.kernel_read=mean ++model.kernel_read_radius=null ++model.kernel_read_theta=1.0` | `$D/code_dtt16`  |

ISLA with the tree read (`dtt`) is query independent by construction: a point's walk sees the boundary tree and the
point alone. Its ratios must be 1.000 to float32 roundoff, which also checks the swap itself.

## What it scores

For each validation case of the `full` split (`full.val`, 386 cases), float32, it uses the lane's native checkpoint
(the latest, or `--epoch N`) and the recipe's reference draw. That draw is the reader's seeded 200,000 interior
points and 50,000 faces per boundary role, the same draw `infer.py` and the fleet scorer use (`training.seed` 42,
reader epoch 0). Every lane is scored on this draw, the seed-43 lanes included. Every metric is computed on the same scored set, the first 100,000 native points. The interior is
the row-shuffled copy, so these points are a random half. The conditions differ only in the query population:

| condition  | tokens the model sees                                                                     | scored points  |
|------------|-------------------------------------------------------------------------------------------|----------------|
| `native_h` | the native 200,000                                                                        | first 100,000  |
| `half`     | the first 100,000 only                                                                    | first 100,000  |
| `box`      | native 200,000 + 200,000 companions uniform in the interior's bounding box                | first 100,000  |
| `wall`     | native 200,000 + 200,000 companions resampled from the native points with p ∝ 1/\|sdf_wall\| | first 100,000  |
| `full`     | the native 200,000                                                                        | all 200,000    |

`native_h` is the control. `full` is what `infer.py` writes per case at the same checkpoint. The probe compares it
against the fleet scorer's float32 validation rows (`$T/p200k_fp32/<run>/<run>/metrics.jsonl`) when those exist at the
same epoch.

Output: `<out>/<condition>/metrics.jsonl` in `infer.py`'s format (a `config` row with the extra key
`qpop_condition`, one `infer_step` row per case, one `infer_summary` row) and `<out>/summary.json`. The summary
holds the per-condition means, `ratio_to_native_h` (per field, for `half`, `box` and `wall`) and
`prediction_deviation_max`. That is the largest relative L2 change, over cases, of each field's prediction on the
shared points against `native_h`'s, computed as the DrivAerML probe computes it. `prediction_deviation_max_all` is
the same, also maximized over fields. The summary also holds `sdf_native_max_rel_diff` and `sanity_full`.

## Choices

- **How a condition is built.** A swapped condition is a new case, not a new `prediction_points` argument. The
  interior point cloud is replaced, the pipeline's own two `ComputeSDFFromBoundary` transforms are applied to it, and
  the recipe's collate builds the forward kwargs. The transforms are the instances in the val dataset's chain, found
  by type: `sdf_wall` and `sdf_wall_normals` from `no_slip`, `sdf_slip` from `slip`. Building the case this way
  serves both arms with one code path:
  - ISLA reads only the points. It has no `query_schema` and computes its own geometry from the boundary tokens.
  - GeoTransolver reads the swapped set three times, as `geometry`, as `local_embedding` (points + `sdf_wall` +
    `sdf_wall_normals` + `sdf_slip`) and as `local_positions`, because its model yaml wires all three to
    `interior.*`.
- **SDF check.** On the native points, the recomputed SDF fields must equal the pipeline's to float32 roundoff. The
  probe asserts this on the first case (max |diff| / max(1, max |value|) ≤ 1e-6) and records the maximum over all
  cases in `sdf_native_max_rel_diff`.
- **Absent slip walls.** On a case without slip walls (0 faces), the pipeline sets `sdf_slip` to the diagonal of the
  interior points' bounding box. That value depends on the point set: it changes under `half`. The probe holds it at
  the native sample's value, so the native points' inputs are identical in every condition.
- **SDF source for `wall`.** AeroZoo's interior carries no stored `sdf`. The pipeline computes `sdf_wall`, the
  signed distance to the `no_slip` body, for both arms because the dataset yaml is shared. The near-wall draw uses
  it, the analogue of DrivAerML's distance to the car. Its magnitude is clamped at 1e-6, as on DrivAerML.
- **Companion measure.** Companion point data are zero except `_target_quadrature_measure`, which gets the mean
  native cell volume. Neither model reads that leaf, and the metric is the unweighted mean over the shared native
  points (`extract_target_measure` returns `None` for this interior), so the choice affects nothing scored.
- **Fixed draws.** Companion draws use a CPU generator seeded 2026, advanced case by case, as on DrivAerML.

## Run recipe

```
sbatch /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-10-06-isla-dtt/scoring/aerozoo/az_qpop.sbatch
```

It runs the six `az_full_*` lanes, all at once on one GB300 node (run i on GPU i mod 4), in the `normal` QOS with a
2 h limit. Output goes to `qpop/<run>/<condition>/metrics.jsonl`, with per-run logs in `qpop/logs/` and the job log in
`../../sbatch_logs/`. A lane whose `train.log` has no "Training completed" is a no-op unless `ALLOW_UNFINISHED=1`.

The launcher reads these environment variables:

- `RUNS`: the runs to score, space separated.
- `ARMS`: `isla`, `gt` or `dtt` for each run, in the order of `RUNS`. By default the arm comes from the run id.
- `AZ_QPOP_CASES`: score only the first N cases.
- `AZ_QPOP_EPOCH`: load `checkpoint.0.<N>`.
- `QPOP_OUT`: the output root.

For example, `RUNS="az_full_dtt_seed42 az_full_dtt_seed43" sbatch .../az_qpop.sbatch`.

Directly: `python az_qpop.py <run_id> <out_dir> <isla|gt|dtt> [--n-cases N] [--epoch N] [hydra override ...]`. Run
it from `$T/recipe_v9` with `PYTHONPATH=<code>:$T/recipe_v9/src` and `RANK=0 WORLD_SIZE=1 LOCAL_RANK=0
MASTER_ADDR=127.0.0.1 MASTER_PORT=<free>`.

## Smoke test (2026-10-11, `batch` / `normal`, one GB300 node)

All runs used checkpoint `checkpoint.0.1`, which holds the weights after training epoch 0, so the lanes' own epoch-0
validation is a reference. The run directories are `smoke/` (job 1182986, 1 case, three seed-42 lanes), `smoke4/`
(job 1194111, 4 cases) and `smoke12/` (job 1194150, 12 cases, all six lanes with the launcher's default GPU
assignment). Each run has five `metrics.jsonl` files with one `infer_step` row per case.

One case (`00000_case-0000000000000000000-a48a547f8d4c`), ratio to `native_h` (pressure / velocity / nut), with the
prediction deviation (max over fields):

| arm    | `native_h` p / v / nut | `half`                         | `box`                          | `wall`                         | deviation half / box / wall |
|--------|------------------------|--------------------------------|--------------------------------|--------------------------------|-----------------------------|
| `isla` | 0.1878 / 0.2338 / 0.3039 | 0.99791 / 1.00011 / 1.00126  | 1.62752 / 1.02154 / 1.40882    | 1.15867 / 1.10525 / 1.03869    | 1.4e-3 / 2.1e-1 / 1.6e-1    |
| `gt`   | 0.6714 / 0.3202 / 0.8459 | 1.03365 / 1.03805 / 1.03805  | 0.78184 / 0.82399 / 0.90108    | 0.87185 / 1.11078 / 0.99103    | 2.0e-1 / 5.1e-1 / 2.2e-1    |
| `dtt`  | 0.1844 / 0.2314 / 0.2653 | 1.00000 / 0.99999 / 1.00000  | 1.00001 / 1.00001 / 0.99987    | 1.00001 / 1.00001 / 0.99987    | 3.3e-5 / 5.2e-5 / 5.2e-5    |

- **Swap correctness (`dtt`).** Over all five fields and every swapped condition, max |ratio − 1| is 1.3e-4
  (1 case), 2.2e-5 (4 cases), and 1.4e-5 and 6.7e-6 (12 cases, seeds 42 and 43). The prediction deviation is at
  most 4.6e-4. `box` and `wall` give identical numbers: both have 400,000 tokens, so the residual follows the token
  count.
- **SDF check.** On the native points, the recomputed `sdf_wall`, `sdf_wall_normals` and `sdf_slip` equal the
  pipeline's exactly (diff 0.0) for every case of every run. The 4-case and 12-case runs include case `00008` (no
  slip faces), whose `half` condition exercises the held `sdf_slip`.
- **`full` against the lanes' own validation.** A training `val_step` row is the mean over the four ranks' samples
  at that step: `val_step 0` covers the first four validation cases, in bf16. The 4-case `full` mean agrees with
  each seed-42 lane's epoch-0 `val_step 0` to bf16 level (≤ 0.7%):

  | arm    | probe `full`, fp32, mean of 4 (p / v / nut) | lane `val_step 0`, epoch 0, bf16 |
  |--------|---------------------------------------------|----------------------------------|
  | `isla` | 0.46503 / 0.35959 / 0.66460                  | 0.46411 / 0.36016 / 0.66493      |
  | `gt`   | 0.76495 / 0.49100 / 0.64153                  | 0.76415 / 0.49075 / 0.63989      |
  | `dtt`  | 0.43695 / 0.38171 / 0.55122                  | 0.43646 / 0.38180 / 0.55501      |

- **Timing (12 cases, six runs on four GPUs).**
  - `isla` sharing its GPU with a `dtt` run: 5.8–8.5 s per case.
  - `gt` alone on its GPU: 3.7–4.5 s per case.
  - `dtt` sharing its GPU: 9–31 s per case for the first six cases while it warms up, then 4.7–6.3 s.

  The first case of a run also pays for model compilation. Scoring all 386 cases should take about 50 min for the
  slowest run, which fits the 2 h limit.
