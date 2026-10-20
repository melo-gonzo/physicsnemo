# GEO-CONT pre-registration: a query-geometry encoding that is continuous through the boundary (2026-10-18, written 23:00 EDT, before any lane was submitted)

**Status: approved by the supervisor on 2026-10-18 at about 23:30 EDT, with the amended rule below. The launch record is at the end.**

`T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0` (LHR).
Measurements behind this design: `MEASUREMENTS.md` (unit test, and the real-data probe from LHR job 273927).

## What was measured before the design (the reason this prereg differs from the brief)

- **On-surface nodes.** On Tensile2d (about 495 per case, 8 % of the nodes), the lug (2,524-3,616, 32-39 %) and AirfRANS
  (about 1,000, 0.6 %), every on-surface node is bitwise a vertex of the boundary mesh. This holds natively and under the
  posed motion, so the exact geometry gives d == 0 exactly in both frames. No on-surface node changes sign or direction
  under the pose, and the current encoding's pose jump on them is 0.
- **What that means for the brief.** The rounding-sign mechanism behind A2's pose sensitivity (nearest-token geometry)
  does not occur on the exact-geometry branch. The discontinuity of `sign(s) log(|s| + 1e-12)` at s = 0 is real: on
  synthetic off-vertex surface points it gives jumps of 37-46 under motion and scaling. On these data it is a fixed,
  pose-independent "on-surface" code.
- **Orientation, read from the data** (stored winding, which ISLA reads; 6 cases each):
  - Tensile2d: d > 0 / = 0 / < 0 on 0.92 / 0.08 / 0 of the nodes. The winding already points into the specimen, so no
    flip is needed. Its stored `cell_data.normals` point outward, but ISLA does not read them.
  - lug: 0 / 0.32-0.39 / 0.61-0.68. Wound outward, so the flip is needed.
  - AirfRANS: 0.994 / 0.006 / 0. No flip.
- **The lug's remaining pose instability under exact geometry.** 0-2.3 % of the nodes per case reverse direction under
  the motion. They are off the surface, presumably mid-thickness ties (the plate's medial surface). sign(d) u reverses on
  them too, so no encoding of d addresses them (`DESIGN_NOTE_medial_ties.md`).

## Question

Does the continuous encoding change ISLA's accuracy on Tensile2d, the GINOT lug and AirfRANS? It is principled: it is
continuous at d = 0, scale-free, pose-invariant and discretization-independent. Single-input exact-geometry design, same
recipe and seeds as the references. A null result adopts it. Also recorded, outside the rule: the first posed/native
ratios of the exact-geometry design on the two solids.

## Arms (12 lanes; `$T/kernel/gcont/lanes.tsv`)

| arm | code snapshot | SNAPSHOT_HEAD | MD5SUMS md5 (lines) | what it is |
|---|---|---|---|---|
| **CONTROL** (`gcont_ctl_*`) | `code_gctl` | `c9fec58bf` (md5 5b915fa6) | 8b350f04 (3,236) | psharpe/isla-geo-exact b40537f9b + psharpe/isla ea397e04f merged (04ecca138) + the measurement test (c9fec58bf). Its model.py is byte-identical to code_geo2's (0fa4de14) |
| **CONT** (`gcont_cont_*`) | `code_gcont` | `84aef1db4` (md5 1fef8859) | 478c0a4b (3,236) | CONTROL + the encoding: [s, asinh(s / 1e-6)] for the built-in distance and the stored-sdf channel, direction sign(d) u, checkpoint 0.6.0 |

- Both arms run on **`recipe_gcont`**: SNAPSHOT_HEAD md5 937b1231, MD5SUMS md5 72819f07 (171 lines).
  - It is `git archive 84aef1db4` of the recipe directory, plus recipe_geo2's cluster-only files (`dataset_paths.yaml`
    and the lug, Tensile2d, HiLift and DrivAerML-area dataset files).
  - The three `ginot_lug_volume{,_area,_posed}.yaml` files then get `FlipBoundaryOrientation` (boundaries: [surface],
    normal_fields: [normals]). It is the first transform in native and area; in posed it comes right after
    FixedRandomPose. md5s: 56d73aaa / d70a80b1 / a81345d3.
  - The recipe differs between the two commits only in two comment blocks (`isla_lug.yaml`, `isla_tensile2d.yaml`), so a
    shared recipe is functionally each commit's own.
- `code_gctl` vs `code_gcont` (`diff -rq`): model.py, inputs.py, geometry.py, the two regenerated goldens, two test files
  and the snapshot metadata. Nothing else.
- Each snapshot carries tensordict 0.15.0 (`tensordict/` and the dist-info, copied from code_geo2, without `__pycache__`).
  The venv is `.venv-recipe` (torch 2.12.0+cu130, warp 1.14.0).

**Rows.** Each row copies its reference row exactly, except idx, group (`gcont`), run id and code. The recipe comes from
site.env.

| dataset | reference rows | extras (identical to the reference) | cells | epochs |
|---|---|---|---|---|
| Tensile2d (`plaid_tensile2d_2d`, `isla_tensile2d`) | `a2r_t2dz_isla_seed42/43`, kernel/a2r/lanes.tsv | `train_split=train_400 val_split=val_100 +training.val_interval=10 model.hidden=512 model.n_slices=512 model.compile_blocks=true` | 20000 | 500 |
| lug (`ginot_lug_volume`, `isla_lug`) | `a2r_lug_isla_seed42/43` | `train_split=train val_split=test` + the same four | 20000 | 500 |
| AirfRANS uniform (`airfrans_uvolume_2d`, `isla_volume_airfrans`) | `afu_a2s_isla_seed42/43`, kernel/airfrans_u/lanes.tsv | `train_split=full_train val_split=full_test` + the same four | 100000 | 500 |

All rows: lr 3.75e-4, seeds 42 and 43, bfloat16, `compile=false` with `compile_blocks=true` (lanes.sbatch's command).

**Bundled against the A2 references** (recorded, not in the rule; the rule compares CONT with CONTROL only): the
single-input design (full-resolution boundary_meshes, the in-model draw), exact geometry, the lug flip, the psharpe/isla
code since the references, and checkpoint 0.5.0 (CONTROL) or 0.6.0 (CONT).

## Launcher (`$T/kernel/gcont/`, new; nothing shared is edited)

- `lanes.sbatch` 8192a90b and `ckpt_preflight.py` e34bd611 are byte copies of kernel/geo2's.
- `submit.sh` f6ca58dd is kernel/geo2's with the job-name prefix `isla.geo2.` changed to `isla.gcont.` (and the comment).
- `site.env` 6aa8150f is kernel/geo2's with `RECIPE` changed to recipe_gcont (and the comment), as in GEO2-AZ. A byte
  copy would train on recipe_geo2.
- `lanes.tsv` 140445ff, `score.sbatch` 8e238045 (a byte copy of kernel/a2r's; overrides through the environment, see
  JOBS.md).
- Chains: singleton `isla.gcont.<arm>-<dataset>-isla-seed<s>`, normal QoS, 03:59:00, `--segment=1`. Runs go to
  `$T/runs/gcont_*`.
- The AirfRANS rows run on this launcher, not kernel/airfrans_u's. Its lanes.sbatch is a copy of airfrans_u's: same
  training command, but QoS and wall from site.env (normal, 03:59 links) instead of 01:59. Link length changes only the
  chaining.
- Launch in two waves about 13 minutes apart: wave 1 is the seed-42 rows (0 2 4 6 8 10), wave 2 the seed-43 rows. The
  smokes warm the launcher's Inductor and Triton caches beforehand.
- **Acceptance after each lane's first link:**
  - at least 100 steps with finite loss;
  - no traceback and no "Error executing job";
  - `code=…/code_g{ctl,cont}` and `recipe=…/recipe_gcont` in the link-start line.
  A lane that crashes at step 0 and recovers in its chain-ahead link is accepted, as in GEO2-AZ.

## Metrics (each dataset's existing protocol)

- Relative L2 per field on the scored split: float32, the final checkpoint (epoch 500), infer_split=val, sampling
  resolution as trained. Splits: Tensile2d val_100 (100 cases), lug test (600), AirfRANS full_test (200).
- Three protocols: native, area and posed (`<dataset>_{area,posed}.yaml`).
- Scorer: kernel/gcont/score.sbatch, the A2-RERUN scorer. It runs the same infer.py command as the AF-UNIFORM scorer
  that scored afu_a2s.
- **Headline fields:**
  - Tensile2d: stress (Frobenius), displacement, plastic strain q.
  - lug: von Mises.
  - AirfRANS: velocity, pressure, nut.
- Ratio R = CONT ÷ CONTROL on native, per field, on seed means over 42 and 43. The per-seed ratios are reported beside it.

## RULE (the supervisor's, 2026-10-18) and its noise check

As set: **adopt** CONT into the single-input design if R ≤ 1.03 on every headline field of all three datasets (seed
means); **reject** if R ≥ 1.10 on any headline field; **otherwise Peter decides**. Posed/native is recorded for both arms
and is not in the rule.

**Noise check.** I estimated each field's single-seed log-error SD σ from the two seeds of its reference (A2-RERUN,
AF-UNIFORM a2s):
- Tensile2d: stress 0.14, displacement 0.017, q 0.16;
- lug: von Mises 0.10;
- AirfRANS: velocity 0.037, pressure 0.043, nut 0.061.

Under an exactly null CONT (Monte Carlo, lognormal seed noise, 2 seeds per arm), the rule as set gives
**adopt 0.08, reject 0.58, Peter 0.33**. The solids' seed spread decides it, not the encoding.

Proposed amendment (needs approval):
- The adopt and reject bars apply to the fields with σ below 7 %: Tensile2d displacement; AirfRANS velocity, pressure,
  nut.
- The noisy fields (Tensile2d stress and q, lug von Mises) are reported with both seeds and a flag when both CONT seeds
  lie outside the CONTROL seed range.
- Under the null this gives adopt 0.39, reject 0.08, Peter 0.53. With CONT 5 % worse everywhere: adopt 0.005,
  reject 0.40.
- Alternative: two more seeds (44, 45) on the solids, both arms, 8 lanes. The full rule then gives adopt 0.13,
  reject 0.41 under the null: still noise-dominated.

## Amendment before submission (2026-10-18, 23:30 EDT; approved by the supervisor; no lane submitted yet)

**Old rule** (the supervisor's first revision): adopt if R ≤ 1.03 on every headline field of all three datasets (seed
means); reject if R ≥ 1.10 on any headline field; otherwise Peter decides.

**Why it changed.** Single-seed log-error SDs σ, each estimated from its reference's two seeds (A2-RERUN a2r_t2dz and
a2r_lug; AF-UNIFORM afu_a2s), as |log e42 − log e43| / √2:

| field | Tensile2d stress | Tensile2d displacement | Tensile2d q | lug von Mises | AirfRANS velocity | AirfRANS pressure | AirfRANS nut |
|---|---|---|---|---|---|---|---|
| σ | 0.143 | 0.017 | 0.157 | 0.102 | 0.037 | 0.043 | 0.061 |

Monte Carlo of the old rule (lognormal seed noise, 2 seeds per arm, 200,000 draws): under an exactly null CONT,
adopt 0.08 / reject 0.58 / Peter 0.33.

**New rule.**
- **Decision fields:** the four with σ < 7 %: Tensile2d displacement; AirfRANS velocity, pressure and eddy viscosity
  (nut). R_f = CONT ÷ CONTROL on seed means; G = the geometric mean of the four R_f.
- **ADOPT** if G ≤ 1.01 and no R_f ≥ 1.10.
- **REJECT** if some R_f ≥ 1.10 with both seeds agreeing (each CONT seed ÷ the CONTROL seed mean ≥ 1.10 on that field),
  or if G ≥ 1.05.
- **Otherwise Peter decides.**
- Tensile2d stress, Tensile2d q and lug von Mises are reported per seed. Each is flagged when both CONT seeds fall
  outside CONTROL's two-seed range, and Peter judges the flagged fields.
- Posed/native is recorded for both arms and is not in the rule.

**Operating characteristics of the new rule** (same Monte Carlo, 400,000 draws, the same true effect on all four
fields):

| true CONT ÷ CONTROL | adopt | reject | Peter |
|---|---|---|---|
| 1.00 (null) | 0.67 | 0.04 | 0.29 |
| 1.02 | 0.32 | 0.12 | 0.56 |
| 1.05 | 0.03 | 0.52 | 0.44 |
| 1.10 | 0.00 | 0.99 | 0.01 |
| 0.95 | 0.99 | 0.00 | 0.01 |

**My outcome probabilities under the new rule:** adopt 0.60, reject 0.07, Peter 0.33. CONT is predicted null.

**Event guard** (as GEO2-AZ). If any lane's per-epoch validation loss passes twice its running minimum, I stop and report
before the rule is read.

## Predictions (mine, before any result)

- **CONT ÷ CONTROL, native:** about 1.00 on every field, inside each field's seed noise (95 % range about ±2σ: ±3-4 % on
  Tensile2d displacement, ±7-12 % on AirfRANS, ±20-30 % on the noisy solid fields).
  - Mechanism: CONT changes the encoding only off the surface. A one-sided monotone remap
    (`sign(s) log|s|` → `asinh(s/ε)`) differs from the old code by a constant offset of log(2/ε) ≈ 14.5 on interior nodes with
    s ≫ 1e-6 (all solid nodes; AirfRANS's first wall layers, s ~ 5e-6, see a smaller offset) and keeps φ(0) = 0 on the
    surface.
  - The direction is identical on all three datasets after the lug flip (ν = u where d ≥ 0).
  - The stored-sdf channel changes the same way.
- **Posed/native, both arms:**
  - Tensile2d within 1.005 on every field. No direction reverses and the boundary is kept whole. A2 gave 1.09, 1.095
    and 1.003.
  - AirfRANS 1.000 ± 0.002. A2: 1.00.
  - lug 1.02-1.6, central 1.15, from the 0-2.3 % medial-tie nodes (A2: 3.46). Equal between the arms within ±5 %.
- **CONTROL ÷ A2 reference** (recorded): AirfRANS about 0.93 (GEO1: 0.87-0.96); Tensile2d and lug about 1.0 (±seed noise).
- **Rule outcome** (as set): adopt 0.10, reject 0.50, Peter 0.40. With the amendment: adopt 0.40, reject 0.10,
  Peter 0.50.

## Instrument: direction reversals under the pose (both arms)

`smoke_gcont/dirflip_probe.py` (md5 0ba04a09) runs inside the CONTROL smoke job, on CPU, on the scored splits. For every
case it reports:
- the count and fraction of prediction points whose encoded direction reverses under the posed motion
  ((R^T u_posed) · u_native < 0, R from a Procrustes fit of the nodes). It is counted for both arms' encodings: u
  (CONTROL) and ν = sign(d) u (CONT).
- where they lie: |d|/L and the medial gap g = dist(x + |d| u) / |d|. g = 0 on the medial surface of a slab; for g < 2
  the local thickness is |d| (2 + g).

Results go to the smoke's `out_ctl_<job>/dirflip.json` and are summarized in section V.

## V. Smokes (before submission)

Jobs 274801 (CONTROL, ARM=ctl) and 274802 (CONT, afterany:274801). Short QoS, one node each, run in sequence so that the
two do not race on the new caches. Script: `smoke_gcont/smoke.sbatch` (md5 9fff5d44) with `train_probe.py`
(dce8d4ed). Each job:
- runs its arm's three seed-42 rows through lanes.sbatch's exact torchrun command, 20 training steps each;
- prints the query tokens' geometry features (the built-in pair and the stored-sdf pair) over the on-surface points;
- runs the arm's ISLA test suite on GPU;
- CONT only: recipe_gcont's test suite;
- CONTROL only: the instrument.

**Results.** Both jobs exited 0; logs are in `smoke/`.

- **Training smokes.** All six rows ran 20 steps with finite loss and no traceback (rc 0, 85-200 s each, compile
  included). Every link-start line names its arm's code snapshot and recipe_gcont. The probe reports ISLA version 0.5.0
  for CONTROL and 0.6.0 for CONT.
- **Signed distance with the lug flip:** d > 0 / = 0 / < 0 on 0.63-0.67 / 0.33-0.37 / 0 of the lug's queries. Tensile2d:
  0.94-0.96 / 0.04-0.06 / 0. AirfRANS: 0.994 / 0.0055 / 0.
- **Geometry features at the on-surface points** (forwards 1, 2 and 20 on rank 0; the built-in pair, then the stored-sdf
  pair):

  | | on-surface points (d == 0) | on-surface features, both arms | smallest \|log channel\| off the surface, CONTROL / CONT | log channel range, CONTROL / CONT |
  |---|---|---|---|---|
  | Tensile2d | 438-478 | exactly 0 (all four columns) | 0.77-1.14 / 9.45-9.54 | [-5.06, 0] / [0, 13.74] |
  | lug | 3,222-3,330 | exactly 0 | 1.79-2.18 / 10.70-10.90 | [-3.80, 0] / [0, 12.72] |
  | AirfRANS | 542-560 | exactly 0 | 3e-5-8e-5 / 1.99-2.11 | [-12.54, 2.13] / [0, 16.64] |

  - In CONTROL the on-surface code [0, 0] coincides with the log code of a point at s = 1 (on AirfRANS, some fluid
    points come within 3e-5 of it). In CONT it is the continuous end of a monotone code.
  - The stored-sdf pair is the same map applied to the stored field. On the solids it is negative inside, the built-in
    is positive.
- **ISLA suites on GPU** (GB300, from the snapshots): CONTROL 269 passed, 1 skipped; CONT 271 passed, 1 skipped. The
  skip is the posed-AirfRANS sample test, which needs a saved .pt.
- **recipe_gcont suite** (CONT job, CPU): 554 passed, 2 failed, 12 skipped. The two failures are
  `test_campaign_c_hooks.py::test_initialize_from_checkpoint_*`: DistributedManager reads Slurm variables inside the
  job. Local run of the same commit outside Slurm: 556 passed, 12 skipped.
- **Direction-reversal instrument** (`smoke/dirflip.json`; all scored cases, native vs posed, both encodings):
  - **Tensile2d** (100 cases, 0.97M nodes): 0 reversals, 0 sign changes.
  - **lug** (600 cases, 5.68M nodes): 45,492 reversals (0.80 %), in 331 of 600 cases; per case mean 0.87 %, median
    0.77 %, at most 4.0 %. The count is the same for u and sign(d) u.
    - **Every reversing node has medial gap g < 0.01**: it lies on the medial surface, equidistant from two faces.
    - Their |d| / L is 0.1-0.3 and the median local thickness is 0.37 L: the plate's mid-thickness.
    - 0 sign changes.
  - **AirfRANS** (200 cases): 0 reversals in 199 cases (pose fit at most 1.9e-6 L). Case 945 is an instrument artifact:
    it has more than the probe's 200,000-node cap, so native and posed subsample different nodes (pose fit 14.6 L, half
    the nodes "reverse"). It is excluded. The lanes use 100,000, so scoring is unaffected.

## Expected compute and finish

At the references' pace, from launch:
- Tensile2d: about 3.5 h (2 links);
- AirfRANS: about 10 h (3 links of 3:59);
- lug: about 13 h (4-5 links).
In total about 420 GPU-h (12 nodes). Launched about 00:30 EDT on Oct 19, the lanes finish about 04:00, 11:00 and
14:00 EDT, plus queue gaps. Scoring: 36 rows, 9 array tasks of score.sbatch (normal QoS, 4 GPUs each, under an hour
each).

## Launch record (added after submission; nothing above was changed by it)

- **Wave 1** (seed 42; idx 0 2 4 6 8 10) was submitted at 23:33 EDT: jobs 275257-275262. **Wave 2** (seed 43; idx 1 3 5 7
  9 11) followed at 23:46 EDT: jobs 275409-275414. Chain-ahead links queued themselves, singleton per lane.
- **Accepted at 00:12 EDT on Oct 19, all 12 lanes** (`smoke_gcont/accept.sh`). Each lane's first link:
  - has 264 to 9,624 steps logged, with finite loss;
  - has no traceback and no "Error executing job" (no crash at step 0 this time: the smokes had warmed the caches);
  - names code_gctl or code_gcont and recipe_gcont in its link-start line.
- **Pace at acceptance:** Tensile2d about 3.6 epochs per minute, so 500 epochs in about 2.3 h. Lug and AirfRANS about
  0.55 epochs per minute, so about 15 h plus the gaps between links. Expected finish: Tensile2d about 02:30 EDT, lug and
  AirfRANS about 15:00-17:00 EDT on Oct 19.
