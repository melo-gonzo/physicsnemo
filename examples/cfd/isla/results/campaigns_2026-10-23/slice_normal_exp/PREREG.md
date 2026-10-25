# SLICE-NORMAL-EXP: no-harm retrain check of the unnormalized slice mean normal (pre-registration, 2026-10-24)

Lane SLICE-NORMAL-EXP (research-lane, launched 2026-10-24 about 19:29 EDT by the supervisor 01a13c2c). Written
19:36-19:55 EDT 10-24, **before any job of this lane: nothing is submitted, no snapshot is built, and no checkpoint,
log or score of an arm exists.** `T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0`
(LHR); `K=$T/kernel/slicenormalx`.

Structure. Part A is SLICE-NORMAL's draft (`../isla-slice-normal-2026-10-24/PREREG_DRAFT.md`, md5 285ce8e1),
approved by the supervisor, copied verbatim. Part B records the supervisor's amendments A1-A5 (received with the launch
brief, 19:29 EDT 10-24), which govern where they differ from Part A. Part C registers how I implement A and B. Part D
holds my own predictions; Part E the cost, cap and calendar. Where Part C and Part A differ, Part C says so and why.

**What I have seen before registering (adds to A §2).** SLICE-NORMAL's RESULTS.md (md5 d8f83195) and FINAL_REPORT.md;
SLICES-ALL's PREREG, JOBS and smoke checks (its DR read exists; I did not open its scores); DTT-POSE-EXP's PREREG,
CHECKS and JOBS (no arm score of it exists); AF-WHOLE's PREREG and JOBS (no AF-WHOLE score exists; I opened none of its
checkpoints, logs' metrics or scores, and saw only `squeue` lines showing its four lanes running); the launchers,
row tables and scorers of kernel/afwhole, kernel/r090, kernel/r090fix; SURFACE-BC's REPORT, JOBS and cost table; the
dataset and model files of recipe_surfbc2. SLICE-NORMAL (not I) read AF-WHOLE's mid-training checkpoints for forward
statistics of |m_s| only (its RESULTS §2); I have not opened those records beyond its tables.

---

# Part A. The approved draft (verbatim, md5 285ce8e1)

Draft by lane SLICE-NORMAL, written 19:24-19:30 EDT 10-24, **before any job of this check: nothing is submitted, no
snapshot built, no arm checkpoint or score exists.** It is not approved. Adoption would change ISLA's architecture, so
it needs Peter's approval after the read. The design follows SLICES-ALL (PREREG md5 e0221626) and SLICES-DRV's §6
rule. `T` and `K` as in JOBS.md; the launcher would be `K=$T/kernel/slicenormalx`, with job names `isla.slicenormalx.*`.

## 1. Question

ISLA normalizes each slice's mean normal, `m_s / max(|m_s|, 1e-12)`. On trained checkpoints the median slice has
|m_s| = 0.04-0.18 (SLICE-NORMAL RESULTS.md §2, M). That gives a 1/|m| gain of 6-23 at the median slice, up to 1.5e4
in eager, and 1.5e9 when a compiled bf16 product rounds m to zero (FLAG). Option (a) leaves m unnormalized (gain 1, no
constant). **Does (a) harm ISLA-DTT (the interior form) or option A (the surface form) on any field?**

## 2. What I have seen before registering (disclosure)

- SLICE-NORMAL's own |m_s| measurements on the controls' checkpoints (forward statistics only, no scores of an arm).
- The controls' published scores: REF-090 (r090eag_af_dtt, r090_dr_isla), SLICES-ALL's DR read, AF-WHOLE's PREREG
  (no AF-WHOLE score exists yet), FINAL_RERUN_PLAN's costs.
- No arm of this check exists.

## 3. Arms and controls (two seeds, 42 and 43; 0.9.0 code, as SLICES-DRV and SLICES-ALL ran)

The change in every arm is one deleted line in `blocks.py` `_SliceBlock.forward` (`m_s = m_s / m_s.norm(...).clamp_min(eps)`).
blocks.py is the same file (md5 c97fe691) in code_r090, code_cmpaf_fix, code_surfbc and psharpe/isla b4b10d5a9.
New snapshots are copies plus that file, verified by md5, each with a SNAPSHOT_HEAD line:
`code_slicenormal_fix` (from code_cmpaf_fix) and `code_slicenormal_surfbc` (from code_surfbc). Each arm row equals
its control row except `run_id` and the code column; this is asserted on the resolved configs at the first link, as
SLICES-ALL does.

| arm | form, dataset | control | arm lanes | training path | scored |
|---|---|---|---|---|---|
| **AF** | ISLA-DTT, AirfRANS **with its far field** (the canonical boundary) | `afwhole_af_dtt_seed42/43` (AF-WHOLE's W lanes, `$T/kernel/afwhole/lanes.tsv` rows 2-3, recipe_afwhole; ending 10-25) | the same rows with code_slicenormal_fix | eager (`compile=false model.compile_blocks=false`, as the control) | AF-WHOLE's scorer, unchanged: eager, native / posed, 200 full_test cases, 100,000 cells |
| **DR-A** | option A, DrivAerML Surface with the tunnel pieces | **new** control lanes `slicenormalx_dra_ctl_seed42/43` on code_surfbc + recipe_surfbc2 (35a460b34), `isla_surface` reading all five boundaries and predicting at the vehicle | `slicenormalx_dra_m_seed42/43`, code_slicenormal_surfbc, otherwise byte-equal rows | eager for both, so no new compiled region needs certification | kernel/r090/score.sbatch rows with the snapshots changed: native / area / posed, 48 val cases, 200,000 cells |

- **AF fallback.** If AF-WHOLE's lanes fail their own registered checks, the control becomes `r090eag_af_dtt` (airfoil
  only, the ISLA-DTT standing; `kernel/r090eagaf` rows 2-3; s_c 0.0022 / 0.0395 / 0.0078), with arms on recipe_r090.
  This is decided and recorded before the AF lanes start.
- **Why new DR controls.** No option-A lane exists: 0.9.0's r090_dr_isla reads the vehicle only. Option A is the
  configuration in which slices span the closed tunnel box, which is the case at issue.
- **Optional, outside the core budget and calendar** (the supervisor's choice):
  - **LUG**: ISLA-DTT on the lug against r090fix_lug_dtt. Its slices are the most diffuse (median |m| 0.04-0.05), so
    (a) changes the most there. The arm must be compiled on the fix path like its control, which needs the block-wise
    certification of the changed region. About 2 x 150 GPU-h; it reads 10-27 at the earliest.
  - **AZ**: AeroZoo full ISLA-DTT against azc_full_dtt, where 2-3 % of slices sit below the bf16 level. About
    2 x 215 GPU-h; it reads after 10-27.

## 4. Smokes (one short-QoS job per arm, bad nodes excluded, `isla.slicenormalx.smoke`)

SLICES-ALL's smoke procedure: a control unit and an arm unit of 50 steps each, then a 2-epoch certification training.
The checks:
- rc 0, finite losses;
- the resolved-config diff shows only run_id and the code path;
- the geometry kernel lines match the control's;
- a forward check on 20 cases on the control's final checkpoint (AF) or the 2-epoch checkpoint (DR-A): the arm's
  `_SliceBlock` m_s equals the control's raw (unnormalized) m_s bitwise, which is the probe's mn64 path. It confirms that
  the arm runs the intended change and nothing else.

## 5. Rules (registered as written; SLICES-DRV §6 / SLICES-ALL R1)

Per arm and per field, on the native protocol. Fields: AF pressure, in-plane velocity (never velocity_z) and nu_t;
DR-A pressure and wall shear. CM is the scorer's case-mean relative L2 (float32, final epoch).
- `D = ln(mean_seeds CM(arm) / mean_seeds CM(control))`; `SE = sqrt((s_arm^2 + s_c^2) / 2)`, with s the pair's own
  two-seed sd (n - 1) of ln CM;
- **harms** if D > ln 1.10 and |D| > 2 SE; **helps** if D < ln 0.90 and |D| > 2 SE; otherwise **no change**;
- **sub-bar flag:** up if arm ÷ control > 1.03 with both arm seeds above both control seeds; down if < 0.97 with both
  below. A flag changes no label.

**Verdict.**
- **(a) is a candidate for adoption** if no field of AF or DR-A reads "harms".
- Up flags and "helps" labels are reported to Peter with the read. They do not block, since (a) is proposed on
  principle, not as a cost cut. SLICES-DRV's stricter "an up flag keeps the reference" was registered for a cost cut;
  if the supervisor prefers it, it is amended here before submission.
- A harm label keeps the reference, and (c) (RESULTS.md §4, the fallback) would be the next check.
- A lost or invalid arm makes the verdict "incomplete as registered", and I escalate. Adoption is Peter's.

**Beside, not verdicts:**
- posed for AF and DR-A, where it changes the input. Same runs, so not independent numbers.
- area: DR-A, which is non-degenerate; for AF, checked to equal native, then labelled degenerate.
- seed-matched ratios; per-case medians and win shares.
- cost: step time and peak memory, smoke and whole-run.
- **stability:** the optimizer-state spike scan of DTT-FAST (`spike/lhr/ckscan.py`, cpu-short) on arm and control
  checkpoints, reporting the largest AdamW jump per run and in the routing-path tensors (`assign.gain`, `norm_assign`).

## 6. Predictions (mine, before any result)

| id | prediction | p |
|---|---|---|
| P1 | both smokes pass on the first run | 0.80 |
| P2 | AF: no change on all three fields (harms on any 0.12, helps on any 0.10) | 0.78 |
| P3 | DR-A: no change on both fields (harms 0.07) | 0.88 |
| P4 | the verdict: candidate (no harm on any field) | 0.80 |
| P5 | an up flag on at least one field of either arm | 0.25 |
| P6 | the arm's largest routing-path AdamW jump below its control's, on both seeds of AF | 0.55 |
| P7 | arm ÷ control smoke median step within 0.98-1.02 | 0.90 |

## 7. Cost and calendar (E; per-lane costs from AF-WHOLE's PREREG and FINAL_RERUN_PLAN §3)

| item | lanes | GPU-h per lane | GPU-h | elapsed |
|---|---|---|---|---|
| AF arms (eager, about 1.0-1.25 x r090eag_af_dtt's 110 GPU-h) | 2 | 110-140 | 220-280 | 28-35 h |
| DR-A controls and arms (eager; compiled option A about 17 GPU-h, eager about 1.2-1.6x) | 4 | 20-27 | 82-109 | 5-7 h |
| smokes (2 short jobs) and scoring (AF 4 rows, DR-A 12 rows) | | | 10-15 | |
| **core total** | **6 lanes** | | **about 310-400 GPU-h** (cap proposed: 420) | |
| optional LUG / AZ | 2 / 2 | 150 / 215 | +300 / +430 | 38-43 h / 50+ h |

- **Calendar.** If approved by 10-25 morning and launched at 09:00 EDT, DR-A reads about 10-25 18:00. AF reads about
  10-26 16:00 to 10-27 02:00 (calendar 1.15 x elapsed, plus scoring and AF-WHOLE's own end on 10-25). Both read
  before the 10-27 freeze. The lug would read about 10-27 06:00-14:00.
- Six lanes run at once on six nodes, at most two submitted at a time and staggered. Short QoS is used only for the
  smokes and the final links (the watcher).

---

# Part B. Amendments before results (supervisor, received with the launch brief, 19:29 EDT 10-24)

**What was known when they were written:** no job of this check existed (no snapshot, smoke, lane, checkpoint or
score). Results of other experiments known then: SLICE-NORMAL's |m_s| measurements (RESULTS.md), the standings' scores
(REF-090), SLICES-ALL's DR read. No AF-WHOLE score existed. The supervisor approved the draft with these amendments; they
govern where Part A differs.

- **A1. LUG joins the core.** ISLA-DTT on the lug against `r090fix_lug_dtt`, two seeds, compiled per block like its
  control (`model.compile_blocks=true`). Its smoke includes the block-wise certification of the changed region by
  LANE_RULES 'Compiled regions' (Part C §C4). Scored on the fields the lug's standing scores: von Mises
  (`von_mises_l2`), its only field. The control's seeds are far apart (0.00221933 / 0.00183431, s_c 0.1347), so
  seed-matched ratios arm42 ÷ control42 and arm43 ÷ control43 are reported beside the unpaired rule, as DTT-POSE-EXP
  does. **The rule stays unpaired**, since the arms share no initialization with their controls (A §5 rule, unchanged).
- **A2. Launch now; do not wait for 10-25 morning.**
  - The AF arms use AF-WHOLE's rows (`$T/kernel/afwhole/lanes.tsv` rows 2-3) and its recipe snapshot
    (`$T/recipe_afwhole`). AF-WHOLE's control lanes are still training, normally.
  - **Blinding:** I open no AF-WHOLE score or checkpoint before AF-WHOLE's own read is in (about 10-25 afternoon to
    evening). My AF read waits for it. (Consequence for the AF smoke: the m_s forward check of A §4 runs on the AF arm's
    own 2-epoch certification checkpoint, not on the control's final checkpoint; Part C §C3.)
  - **The fallback decision is made now:** the AF control is AF-WHOLE's `afwhole_af_dtt_seed42/43`. If AF-WHOLE's lanes
    later fail their own registered checks, I tell the supervisor before anything else. The fallback, `r090eag_af_dtt`,
    would need new arm lanes on recipe_r090 (not launched without the supervisor).
- **A3.** Up flags and "helps" labels are reported, and they do not block. (a) is proposed on principle, not as a cost
  cut. **A "harms" on any field blocks.**
- **A4. Cap: 720 GPU-h** (the core 420 plus 300 for the lug). I ask before any job that would project the total past it.
- **A5. The verdict.** (a) is a **candidate for adoption** if no field of AF, DR-A or LUG reads "harms". Adoption
  changes the architecture, so it is Peter's decision after the read; it would go into 0.13.0 with DTT-POSE's forest.
  A lost or invalid arm makes the verdict "incomplete as registered" (escalated, never improvised). An arm still
  unread at my deadline is handed off (HANDOFF_SLICENORMALX.md); the verdict then waits for it.

---

# Part C. Implementation (registered before any job)

## C1. Snapshots (copies plus the one changed file, md5-verified, read-only, SNAPSHOT_HEAD line)

- `$T/code_slicenormal_fix` = copy of `$T/code_cmpaf_fix` (the AF and lug standings' code: 92d9d7800 + td0152 + COMPILE-AF
  model.py 45b24fb8), with `physicsnemo/experimental/models/isla/blocks.py` replaced by a copy from which exactly the
  line `        m_s = m_s / m_s.norm(dim=-1, keepdim=True).clamp_min(eps)` (line 306 of md5 c97fe691) is deleted.
  Nothing else: no docstring or comment edit (the brief: no change other than the one registered line).
- `$T/code_slicenormal_surfbc` = copy of `$T/code_surfbc` (SURFACE-BC's 129c66294 + td0152) with the same blocks.py.
- Proof recorded in JOBS.md: `diff` of the two blocks.py is that one line; `diff -rq` source vs snapshot lists only
  blocks.py, SNAPSHOT_HEAD and MD5SUMS; `md5sum -c` of the new MD5SUMS passes; `chmod -R a-w`.

## C2. Launchers and lane rows (`K=$T/kernel/slicenormalx`, one launcher directory per recipe)

| dir | recipe | rows (idx: run_id) | source rows | the only differences from the source row |
|---|---|---|---|---|
| `$K/af` | recipe_afwhole (site.env = kernel/afwhole's, exclude file path changed) | 0: slicenormalx_af_dtt_seed42, 1: ..._seed43 | kernel/afwhole/lanes.tsv rows 2-3 | idx, group `slicenormalx`, run_id, code `code_slicenormal_fix` |
| `$K/lug` | recipe_r090 (site.env = kernel/r090fix's with `--segment=1 --exclude=<file>`) | 0: slicenormalx_lug_dtt_seed42, 1: ..._seed43 | kernel/r090fix/lanes.tsv rows 0-1 | idx, group, run_id, code `code_slicenormal_fix` |
| `$K/dra` | recipe_surfbc2 (site.env = kernel/r090's with RECIPE and the exclude) | 0-1: slicenormalx_dra_ctl_seed42/43 (code_surfbc); 2-3: slicenormalx_dra_m_seed42/43 (code_slicenormal_surfbc) | kernel/r090/lanes.tsv rows 6-7 (r090_dr_isla) | idx, group, run_id, code; extra gains ` compile=false model.compile_blocks=false` (eager, A §3) |

- lanes.sbatch and ckpt_preflight.py are byte copies of kernel/afwhole's (8eecec8f / e34bd611, = r090's and r090fix's);
  submit.sh = afwhole's with the prefix `isla.slicenormalx.`; `exclude_nodes.txt` lists the five bad nodes
  (nvl72d015-T12, nvl72d058-T08, nvl72d052-T01, nvl72d057-T13, nvl72d043-T18). Normal QoS, 03:59:00 chained singleton
  links; short QoS only for smokes and (by the supervisor's watcher) final links.
- Byte-copy control launchers for the smokes' control units only: `$K/afc` (kernel/afwhole's lanes.sbatch, site.env,
  lanes.tsv, ckpt_preflight.py) and `$K/lugc` (kernel/r090fix's), with RUNS pointed at the smoke directory, so nothing
  writes into another lane's runs.
- Each arm's two lanes are submitted a few minutes apart (each launcher's Triton / Inductor cache is shared by its
  lanes; DTT-POSE-EXP lost a link to that NFS race).

## C3. Smokes (one short-QoS job per arm, `isla.slicenormalx.smoke`, 01:30:00, bad nodes excluded)

SLICES-ALL's smoke.sbatch (08936777) with paths, job name and arm parameters changed, plus stages for the checks below.
1. **Control unit and arm unit**, 50 training steps each through the lane launcher with NO_CHAIN and smoke_wrap_r090.py
   (32a2991e, byte copy), on one node: AF `afc:2` and `af:0`; LUG `lugc:0` and `lug:0`; DR-A `dra:0` and `dra:2`.
2. **Certification training:** the arm's seed-42 lane command for 2 epochs (writes the first checkpoint).
3. **m_s check** (`mcheck.py`, written after this registration, md5 in JOBS.md before it runs): the recipe's infer.py
   runs unchanged on the **arm** snapshot (eager, `compile=false model.compile_blocks=false`), 20 val cases, in
   `precision=bfloat16` and `float32`. The control snapshot's blocks.py is loaded beside the package under another
   module name (it registers no custom op; its relative imports resolve to the byte-identical package modules). At
   every block call the control file's `_SliceBlock.forward` is also run on the **same inputs and weights**, and the
   probe records (i) the control's raw product `einsum("bns,bnc->bsc", a, n_hat)` and the m_s it passes on, (ii) the
   arm's m_s passed to the geometry region (`_geo_region` or the fused kernel). Checkpoints: AF and DR-A the arm's 2-epoch
   certification checkpoint (AF-WHOLE's checkpoints are blinded, A2); LUG the control's final `r090fix_lug_dtt_seed42`.
4. **LUG only, block-wise certification** (C4).

**An arm's smoke passes if** (every item; otherwise that arm is not submitted and I report):
- both units rc 0, no Traceback, `nonfinite=[]`; the certification training reaches "Training completed";
- resolved configs: arm unit vs control unit differ only in `run_id` (and paths); the control unit equals the control's
  own `runs/<control>_seed42/resolved_config.yaml` except paths (AF: afwhole_af_dtt_seed42's config file, which is
  neither a score nor a checkpoint; LUG: r090fix_lug_dtt_seed42; DR-A: no prior run, so the control unit's config must
  list the five boundary meshes, `prediction_boundaries` [vehicle], `compile: false` and `compile_blocks: false`);
  the certification config equals the arm unit's except run_id and num_epochs; AF: AF-WHOLE's condition-1 guard
  (`check_cfg.py`: freestream in boundary_meshes and boundary_schemas, dataset md5 79210179) passes on the arm unit;
- compile path: `compile: false`; `compile_blocks` false (AF, DR-A) or true (LUG), as the control; LUG: Dynamo counters
  (unique graphs, recompiles, graph breaks) equal between the units;
- geometry kernel: every rank of the arm unit logs `ISLA geo_kernel=auto resolved to` the same kernel as the control
  unit;
- **m_s check:** on every case, block and precision, the arm's m_s is bitwise equal to the control's raw product on the
  same inputs (`torch.equal`), and the control's passed m_s differs from its raw product (the normalization ran there).
  This confirms that the arm runs the one intended change.
- LUG: C4 passes.

Beside: arm ÷ control median step and peak allocated memory (same node).

## C4. LUG: block-wise certification of the changed compiled region (LANE_RULES 'Compiled regions')

Instrument `cert_blocks.py` (written after this registration from DTT-FAST's cert_p2.py, the program's restated-rule
instrument, and COMPILE-AF's cmpaf_diag.py loader; md5 in JOBS.md before it runs). On the lug's first 2 val cases at the
training resolution (20,000 points, the scorer's loader), a float32 forward and a bf16-autocast forward of the eager
model capture each of the 12 blocks' inputs. Per block and precision (float32; bf16 autocast): the block in float64
eager (the reference; a float64 copy of the module and inputs), eager in the precision, the model's own compiled
callable (`_block_callables`, Inductor default casts, `_BLOCK_COMPILE_OPTIONS`) in the precision, and 8 eager draws
with the block's activation h and the output gradient each moved by one ulp (random signs, zeros kept). Forward
output plus backward for a fixed random output gradient. Tensor groups: the block output, the gradient of h, and the
gradient of each parameter tensor. Distance d = max |x - ref| / max |ref|.
- **Pass, per group:** d(compiled, f64) <= d(eager, f64) + allow, allow = the mean over the 8 draws of d(eager draw,
  eager) (the largest draw reported beside); in float64, compiled against eager <= 1e-12 (roundoff).
- **Runs** (one GPU each, in the LUG smoke job): (i) arm code on the arm's 2-epoch certification checkpoint;
  (ii) arm code on the control's final checkpoint r090fix_lug_dtt_seed42 (trained weights; the change adds no
  parameter, so it loads); (iii) control code (code_cmpaf_fix) on the same control checkpoint, the reference, beside.
- **Gate:** (i) and (ii) pass in every group. If either fails: if (iii) passes in the groups where (ii) fails, the
  change made the compiled region worse; the LUG lanes are not submitted and I report. If (iii) fails in the same
  groups, the bound, not the change, is at issue: I report the numbers and wait for the supervisor.

## C5. Event guard, scoring, checks before reading

- **Event guard (a stop, not a verdict):** an arm lane's validation loss above its running minimum by more than 4x after
  epoch 10 (AF, LUG; the controls' worst 1.9-3.05) or 2x (DR-A, both its lanes; r090_dr_isla's worst 1.10); any
  non-finite loss; a BLOCKED marker; a lane with no running or pending link. I report it; I resubmit only a
  duplicate-free restart after a node failure (keeping the oldest pending link), or a link lost to the NFS cache race
  (marker archived), as LANE_RULES say.
- **Scoring exactly as each control**, copies in `$K/<arm>` with job name `isla.slicenormalx.score`, roots `$K/<arm>` and
  `EVAL_ROOT=$T/evals_slicenormalx`, the five nodes excluded, normal QoS:
  - AF: kernel/afwhole/score.sbatch (bd40af29) with afwhole score_rows 8-10 and 12-14 (native, area, posed) with run_id
    and code changed. AF-WHOLE's guard (check_cfg.py) is asserted on every AF row's saved config before reading.
    The control's numbers are AF-WHOLE's own `$T/evals_afwhole/afwhole_af_dtt_seed4x/` trees, opened only after
    AF-WHOLE's read is in (A2).
  - LUG: kernel/r090fix/score.sbatch (3a5e4ca9) with r090fix score_rows 0-2 and 4-6 with run_id and code changed (eager,
    600 test cases, 200,000 cells). Control: REF-090's r090fix_lug_dtt scores (`$T/evals_r090fix`).
  - DR-A: kernel/r090/score.sbatch (a4a24f5e) with rows for the four DR-A runs, **native and posed**, from r090 rows
    24 and 26 with run_id, code and recipe changed and ` compile=false model.compile_blocks=false` appended (eager, as
    trained, so no uncertified compiled region scores the arm; recipe_surfbc2's isla_surface.yaml sets `compile: true`).
    **Area is not scored for DR-A** (a change from A §3): recipe_surfbc2's `drivaer_ml_surface_area.yaml` does not read
    the tunnel pieces, so the option-A model cannot run on it, and I make no change to the recipe.
- **Checks before reading** each row (a failed check makes it invalid and that arm "incomplete as registered"):
  `"epoch": 500` in its config row; 200 / 600 / 48 `infer_step` rows with the control's sample ids; SNAPSHOT_USED the
  arm's code; DATASET_USED as the control's per protocol; the score log's extra as the lane's.

## C6. The read

- The rule is A §5 per arm and field on the native protocol, with A3 and A5: AF pressure, in-plane velocity
  (`velocity_l2`, never velocity_z) and nut (`nut_l2`); DR-A pressure and wall shear (`wss_l2`); LUG von Mises
  (`von_mises_l2`). s_c is each control's own two-seed sd of ln CM (AF: AF-WHOLE's W-DTT scores; DR-A: the new control
  lanes; LUG: 0.1347).
- Resolution (stated now): LUG: 2 SE binds; a harm label needs arm ÷ control above about 1.21 (arm seeds equal) to 1.31
  (arm seeds spread like the control's); the up flag needs both arm seeds above 0.00221933. AF and DR-A: the ln 1.10
  bar binds unless a pair's seeds spread by more than about 13 %.
- Beside, not verdicts: posed (all three arms; same runs, no independent number); area for AF and LUG checked to equal
  native and labelled degenerate; seed-matched ratios (every field, A1 for the lug); case medians of arm ÷ control and
  the share of cases the arm wins; cost (smoke step and memory ratios, whole-run median train_step, GPU-h by sacct);
  stability: DTT-FAST's optimizer-state spike scan (`spike/lhr/ckscan.py`, cpu-short) on arm and control checkpoints if
  time allows (AF-WHOLE's only after its read).
- Slots: 3 arms x 2 arms x 2 seeds = 12 training runs, 8 of them new (AF 2, LUG 2, DR-A 4). Fields and protocols are
  readings of those runs (AF 3 fields, DR-A 2, LUG 1).
- Each arm's read goes to the supervisor through contact_supervisor as it lands (DR-A first).

---

# Part D. Predictions (mine, written before any job)

| id | prediction | p |
|---|---|---|
| D1 | DR-A smoke passes on its first run | 0.85 |
| D2 | AF smoke passes on its first run | 0.85 |
| D3 | LUG smoke passes on its first run, certification included | 0.65 |
| D4 | the m_s check is bitwise on every case, block and precision, in all three smokes (given the smoke runs) | 0.95 |
| D5 | AF: no change on all three fields (harms on any 0.12, helps on any 0.10) | 0.78 |
| D6 | DR-A: no change on both fields (harms on any 0.07) | 0.88 |
| D7 | LUG: no change on von Mises (harms 0.05, helps 0.07) | 0.88 |
| D8 | the verdict: candidate (no "harms" on any field of the three arms) | 0.80 |
| D9 | an up flag on at least one field of any arm | 0.30 |
| D10 | a down flag or a "helps" on at least one field of any arm | 0.30 |
| D11 | arm ÷ control smoke median step within 0.97-1.03 in all three smokes | 0.85 |
| D12 | LUG seed-matched ratios both within [0.80, 1.25] | 0.65 |
| D13 | every lane finishes with no event-guard stop | 0.85 |
| D14 | the LUG arm reads by 04:00 EDT 10-27 (else handed off) | 0.45 |

Why: the change keeps the direction of each slice's mean normal and only lets its length (median 0.04-0.18) scale the
two m-invariants; the two linear layers that read them (`geo_logit`, `geo_feat`) can absorb a scale where the slice is
coherent, so a retrained model should lose little. What it does lose is the unit-length direction of diffuse slices,
which RESULTS §2 shows is partly bf16 noise anyway. The lug, the most diffuse, changes the most, but its 21 % seed
spread hides anything under about 20-30 %. AF's far field adds 1,700 tokens with coherent normals; its slices' |m| was
no smaller than the airfoil-only standing's.

---

# Part E. Cost, cap and calendar

| item | basis | GPU-h |
|---|---|---|
| AF arm pair | AF-WHOLE's W-DTT smoke step 0.956 s ≈ r090eag_af_dtt's (110 GPU-h per lane) | 200-250 |
| LUG arm pair | r090fix_lug_dtt sacct 171 / 153 (with losses); DTT-POSE-EXP's lug arms price alike | 290-340 |
| DR-A, 4 lanes | compiled option A about 17 GPU-h per lane; eager about 1.2-1.6x | 80-110 |
| smokes (3 short jobs, the LUG one with the certification) | | 8-15 |
| scoring (AF 6 rows, LUG 6, DR-A 8) | | 10-15 |
| **total** | | **about 590-730 (central 660); cap 720 (A4)** |

The upper end touches the cap; the smokes re-price (arm GPU-h = control GPU-h x smoke step ratio) and I ask before any
job that would project past 720.

Calendar (E): smokes 10-24 20:30-22:30; DR-A lanes from about 22:00, read about 10-25 06:00-10:00; AF lanes from about
22:30, ending about 10-26 03:00-08:00, read after AF-WHOLE's read (10-25 evening) and the arms' scoring, about 10-26
morning; LUG lanes from about 23:00, ending about 10-26 18:00 to 10-27 08:00. Harness deadline about 06:00 EDT 10-27;
HANDOFF_SLICENORMALX.md at 04:00 EDT 10-27 if anything is unread. Registry row `^isla\.slicenormalx\.`
(expected_end 2026-10-27) added before the first job, removed after the last.

---

# Part F. Rulings and amendments after the smokes, before any arm result (written 20:24-20:40 EDT 10-24)

**What was known when this part was written:** the three smoke jobs' outputs (374067 DR-A, 374068 AF, 374069 LUG;
`smoke_checks.txt`, `smoke/lug/cert_compare.txt`): units' step times, memory and first-50-step losses, the m_s checks and
the LUG certification tables. The (a) DR-A and AF lanes had been submitted (19:5x-20:21) and had just started; no
validation loss, checkpoint or score of any arm lane had been read. No AF-WHOLE score or checkpoint was opened. ANCHOR-
DIST's RESULTS.md (e7cf6c10), PREREG_DRAFT.md (2d0e2257) and proto/blocks_ad.py (5dde38f5) were read for A6.

## F1. LUG certification ruling (supervisor, sent about 20:21 EDT)

The LUG smoke's units, certification training and m_s check passed; C4 failed for the arm and the standing's own code
alike. On the same control checkpoint r090fix_lug_dtt_seed42: arm code 62 of 1,200 groups fail, control code
(code_cmpaf_fix) 66 of 1,200, in the routing-path gradients (assign.*, norm_assign.*, geo_logit.weight); fp64 compiled
against eager reaches 1e-11 to 4e-11 relative on those gradients in both (bound 1e-12). `grad_geo_logit.bias` is excluded
from the counts: softmax shift invariance makes it identically zero, so its relative distance is roundoff over roundoff.
Under C4's registered branch ("the bound, not the change, is at issue") I asked; the ruling: **"C4 fails for the
standing's compiled region and the arm alike (control 66 of 1,200 groups, arm 62 of 1,200, in the routing-path gradients;
fp64 compiled against eager 1e-11 to 4e-11 relative), so the rule cannot certify compile_blocks on the lug; the failure
is not attributable to (a)."** The LUG arm trains on the compiled path like its control (like for like). LUG-S does the
same after its smoke; its C4 counts and maximum excess are reported beside the control's and (a)'s, not as a verdict.
For the freeze: the lug's compiled path fails the restated rule on the standing's own code; unless the final code
certifies, the lug trains eager in the final rerun. (Smoke-check note: the unit Dynamo `calls_captured` differ, 136 vs
130, the six ops the deleted line removes from the two graphs; unique graphs, recompiles and graph breaks are equal, which
is what C3 registered.)

## F2. A6: the combined arm (a)+(s) (supervisor, received about 20:02 EDT), sharing this lane's controls

- **Arms AF-S, DR-A-S, LUG-S**, two seeds each (42, 43), on new snapshots of the same bases with ANCHOR-DIST's
  `proto/blocks_ad.py` (md5 5dde38f5) as blocks.py: `$T/code_slicenormals_fix` (from code_cmpaf_fix; AF-S, LUG-S) and
  `$T/code_slicenormals_surfbc` (from code_surfbc; DR-A-S). Its diff against this lane's (a) file (c3e3573a) must be
  exactly the (s) change (ANCHOR-DIST's `proto/blocks_ad.diff`, md5 cd733b34): rho = sqrt(d^2 + sigma_s^2) with
  sigma_s^2 the slice's own second moment under the first-pass weights a, which replace eps as the last fixed argument of
  `_geo_region` and `_relational_invariants`; features rho and log rho in place of dist and log dist; rel / rho in place
  of rel / dist; the fused branch raises NotImplementedError.
- **Every other column equals the (a) arm's**, except as F3 says for DR-A-S. New launcher directories `$KX/afs`,
  `$KX/lugs`, `$KX/dras` (the (a) launchers are running and are not edited), each a copy of its (a) directory with
  lanes.tsv rows = the (a) rows with run_id (`slicenormalx_afs_dtt_seed4x`, `slicenormalx_lugs_dtt_seed4x`,
  `slicenormalx_dras_seed4x`) and code changed; score rows likewise.
- **Smokes as for (a)** (C3): control unit and S unit (50 steps), 2-epoch certification training, the m_s check (it also
  prints the arm file's md5 and its diff against the control file), and for LUG-S the C4 certification runs (i)-(iii).
  The smoke passes on C3's items; for LUG-S, C4 is reported under F1, not gated.
- **The verdict for (s):** (a)+(s) is a **candidate** if no field of AF-S, DR-A-S or LUG-S reads "harms" against the
  shared controls, with A §5's rule and flags (A3: up flags and "helps" reported, not blocking). **Beside, not a
  verdict:** the marginal effect of (s), (a)+(s) against (a), with the same D, SE and labels per field.
- Blinding as A2. **Cap: 1,440 GPU-h** for the whole lane (was 720). Launch as soon as the smokes pass, tonight; each
  read reported as it lands.
- The supervisor's decisions on ANCHOR-DIST's questions (recorded, not mine): sigma_s is allowed (the slice's own
  statistic per input from the model's own assignment, built as ISLA's length unit is built; Peter may veto at adoption);
  float32 anchors stay out; the controls are shared; LUG stays in. Adoption of either fix is Peter's decision after the read.

## F3. DR-A-S geometry kernel (supervisor, sent about 20:23 EDT)

The DR-A control and (a) arm resolve `geo_kernel=auto` to **'fused'** (smoke 374067, 4 of 4 ranks; ANCHOR-DIST's draft
assumed 'eager', which holds for AF and LUG only). blocks_ad.py has no fused branch, and `geo_kernel` is a model
attribute set in `ISLA.__init__`, not a constructor argument. Ruling: **for DR-A-S only**, `code_slicenormals_surfbc` also
changes model.py's `self.geo_kernel = "auto"` to `"eager"` (an execution choice: the fused kernel and the eager region
compute the same function up to summation order). So **DR-A-S runs the eager geometry region against fused controls and a
fused (a) arm.** Conditions:
1. DR-A-S's smoke measures the size of that execution difference at DR-A's shapes, on the (a) code without (s), from one
   checkpoint (the latest checkpoint of the (a) lane slicenormalx_dra_m_seed42 at smoke time): per block, geo_kernel
   'eager' against 'fused' on the same captured inputs, the block output and every parameter-gradient group (routing
   path named), in float32 and bf16, with float64 eager beside for scale; plus the whole-model output difference
   RMS(eager - fused) / RMS(target) per field on the same cases. Instrument `geo_exec.py`, md5 in JOBS.md before it runs.
   Reported beside the DR-A-S verdict as the measured bound of the execution confound.
2. For adoption: if (s) is adopted, the fused kernel needs (s) before the final rerun, or the surface tasks lose its
   speed; priced from the smokes' step times (the (a) unit's fused step against the S unit's eager step).
