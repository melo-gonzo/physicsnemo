# SLICES-ALL notebook draft and chapter changes (2026-10-26 11:00 EDT; for the supervisor and the second reader)

Sources: `scratch/isla-slices-all-2026-10-24/{RESULTS.md,reading_DR.md,reading_DR.json,reading_AF.md,reading_AF.json,reading_LUG.md,reading_LUG.json,PREREG.md}`.
Suggested campaign copy: `results/campaigns_2026-10-26/slices_all/` (RESULTS.md, the three reading_*.md and reading_*.json). The
book code below assumes a source key `slices_all`, registered as `slices_af` and `slices_drv` are. Line numbers move as the book
changes, so each change below names its chunk label or anchor.

## Notebook entry (18-notebook.qmd, after the SLICES-DRV entry)

```markdown
## 2026-10-26 — SLICES-ALL: 128 slices reads "no change" against 512 on every form, ISLA-DTT on AirfRANS (1.028 / 1.010 / 1.024) and the lug (0.966) and the interacting surface form on DrivAerML Surface (1.006 / 1.005), with no flag; 128 slices is a scope-freeze candidate for every form, at 0.53, 0.95 and 0.63 of the step {#sec-nb-slices-all}

**Question.** SLICES-AF and SLICES-DRV found that 128 slices cost interacting ISLA 0.9.0 nothing against 512 on AirfRANS and
DrivAerML Volume (@sec-nb-slices-af, @sec-nb-slices-drv). Since 2026-10-24 ISLA-DTT is the default interior form, and the
surface task uses the interacting form. Neither form, nor any solid-mechanics dataset, had run at 128 slices. Does 128
slices harm ISLA-DTT on AirfRANS or the lug, or the interacting surface form on DrivAerML Surface? At the freeze, the answer
sets the final rerun's slice count, about 2,560 against 3,380 GPU-h.

**Design (pre-registered, PREREG md5 e0221626; amendment A1 before any AirfRANS job, md5 982bf456).** Three arms, each its
control's lane with only `n_slices` 512 -> 128 and the run name changed: the same 0.9.0 code, recipe, launcher, compile path,
epochs, points, learning rate, bf16 and seeds 42 and 43, checked in the saved configs. The arms: ISLA-DTT on AirfRANS,
eager-trained, against the reference r090eag_af_dtt; ISLA-DTT on the lug, compiled on the fixed path, against
r090fix_lug_dtt; and interacting ISLA on DrivAerML Surface against the standing r090_dr_isla. On 0.9.0 code the surface task
reads only the vehicle, so the third arm tests the interacting surface form's slices, not option A's whole boundary. Each
arm was scored as its control: float32, the control's cases (200 / 600 / 48), native deciding, posed beside, area beside on
DrivAerML Surface and degenerate elsewhere. Rule per field: SLICES-DRV's, with the noise floor from each control's own seeds.
Harms if D = ln(arm ÷ control) > ln 1.10 and |D| > 2 SE. An up flag is a ratio above 1.03 with both arm seeds above both
control seeds. Verdict: 128 slices is a candidate for every form only if every field of every arm reads "no change" and no
up flag fires; otherwise 512 stays. Guards before any lane: configs equal apart from the two changes; the geometry kernel
resolved as the control's; for the compiled arms, a block-wise compiled-against-eager check at 5e-7 to 8e-7 (bar 1e-5). The
DrivAerML Surface eager re-score agreed with the compiled scores to 3e-6.

**Result: no change on every field, no flag. Verdict: candidate for every form.** Arm ÷ control, seed means of case-mean
relative L2:

| form, dataset | field | 512 slices (s42, s43) | 128 slices (s42, s43) | 128 ÷ 512 | 2 SE | label |
|---|---|---|---|---|---|---|
| ISLA-DTT, AirfRANS (eager) | pressure | 0.01241 (0.01243, 0.01239) | 0.01276 (0.01380, 0.01172) | 1.028 | 0.164 | no change |
| | in-plane velocity | 0.00717 (0.00737, 0.00697) | 0.00724 (0.00806, 0.00643) | 1.010 | 0.233 | no change |
| | eddy viscosity | 0.02060 (0.02071, 0.02048) | 0.02109 (0.02098, 0.02120) | 1.024 | 0.015 | no change |
| ISLA-DTT, lug (fix-trained) | von Mises | 0.002027 (0.002219, 0.001834) | 0.001958 (0.002062, 0.001854) | 0.966 | 0.218 | no change |
| interacting, DrivAerML Surface | pressure | 0.04113 (0.04116, 0.04111) | 0.04138 (0.04130, 0.04147) | 1.006 | 0.004 | no change |
| | wall shear | 0.06622 (0.06619, 0.06625) | 0.06653 (0.06649, 0.06658) | 1.005 | 0.002 | no change |

Posed reads the same labels on every field, and DrivAerML Surface's area protocol does too. Three cautions go with the
verdict. First, two shifts are seed-separated but under the flag: DrivAerML Surface +0.5-0.6 % on both fields and AirfRANS
eddy viscosity +2.4 %, with both 128-slice seeds above both 512-slice seeds. Second, resolution is uneven. On AirfRANS
pressure and velocity the 128-slice seeds spread 12-17 % (seed 42 the worst of the four runs, seed 43 the best), so 2 SE is
0.16 / 0.23 and a harm label needed a ratio above about 1.18 / 1.26. The lug's 512-slice seeds sit 21 % apart, so 2 SE is
0.22. Only DrivAerML Surface and AirfRANS eddy viscosity resolve near the 10 % bar. Third, every arm ran 0.9.0 code, not the
rerun's. On the lug, posed ÷ native is 1.81 / 1.65 at 128 slices against 1.53 / 1.98 at 512: 0.9.0's ISLA-DTT forest is
pose-dependent at either slice count.

**Cost.** Training step, whole run, 128 ÷ 512: 0.53 for ISLA-DTT on AirfRANS (eager, unfused geometry), 0.63 for the
interacting surface form on DrivAerML Surface (fused geometry; 0.49 in the same-node smoke), but 0.95 for ISLA-DTT on the lug
(1.04 in the smoke). Peak memory falls to 0.78-0.82 on all three. GPU-h per lane: 58 against 110, 11 against 16, and 143
against 153-171. On the lug, ISLA-DTT's step is dominated by work that does not scale with the slice count, so fewer slices
buy almost nothing there.

**Consequence.** By the registered rule, 128 slices is a scope-freeze candidate for every form tested: interacting ISLA
(AirfRANS, DrivAerML Volume), ISLA-DTT (AirfRANS, the lug) and the interacting surface form (DrivAerML Surface). Adoption is
the supervisor's decision at the freeze, reported to Peter. 512 stays the reference until then.

Predictions: no change on every arm (0.93 / 0.92 / 0.85) held; the joint verdict (0.50) held; no up flag fired (0.20 / 0.15
/ 0.20); the lug step ratio in [0.80, 0.97] (0.60) failed in the smoke (1.04; 0.95 whole run); DrivAerML Surface's in
[0.50, 0.75] (0.60) failed narrowly (0.49); AirfRANS's in [0.40, 0.70] (0.60) held.
Data: results/campaigns_2026-10-26/slices_all/. Cost 429 GPU-h (cap 500).
```

## Chapter changes

### 20-limits.qmd, the open question "A quarter of the slices" (`#sec-oq-slices`, chunk `para-oq-slices`, and its row in the open-questions table)

- **Remove the In-flight callout** at the end of `para-oq-slices`: the `S.in_flight_callout({... "replacement": "ISLA-DTT with
  128 slices on AirfRANS and the lug, and interacting ISLA with 128 slices on the DrivAerML Surface task," ...})` print.
- The registered question is answered: 128 slices reads no harm on every form. **Recommended:** remove the row "Does a quarter
  of the slices cost nothing on ISLA-DTT and on the surface task? (@sec-oq-slices)" from the open-questions table, and turn
  `para-oq-slices` into a settled note, kept under its anchor because 09, 12 and 19 cite it. Alternative: narrow the row to what
  stays open, "Does 128 slices hold on the rerun's code and on option A's whole boundary?". Its action would be "the final
  rerun's first lanes at 128 slices, read against the 0.9.0 128-slice arms", cost "within the rerun". I recommend the first,
  because the freeze carries this decision, not an open question.
- In the paragraph, replace the bold lead and the sentences from "These two datasets do not decide ..." to "Cost: not priced."
  with (numbers from `R.md_table("slices_all", "reading_<arm>.md")` native rows and RESULTS.md's cost lines):

  > **A quarter of the slices costs no measured accuracy on any form of ISLA, at a saving that depends on the form.**
  > [AirfRANS and DrivAerML Volume sentences, unchanged.] On ISLA-DTT, 128 slices against 512 reads 1.028 / 1.010 / 1.024 on
  > AirfRANS pressure / velocity / eddy viscosity and 0.966 on the lug's von Mises stress. On the DrivAerML Surface task,
  > interacting ISLA with 128 slices reads 1.006 / 1.005 on pressure / wall shear. No field moves by the same rule, and no
  > shift above 3% has both 128-slice seeds above both 512-slice seeds (@sec-nb-slices-all). Two shifts lie below that
  > threshold with the seeds separated: 0.5-0.6% on the surface task and 2.4% on AirfRANS eddy viscosity. The test resolves
  > only large effects on AirfRANS pressure and velocity, where the 128-slice seeds spread 12-17%, and on the lug, where the
  > 512-slice seeds sit 21% apart. The training step is 0.53 of the reference's for ISLA-DTT on AirfRANS, 0.63 on the surface
  > task and 0.95 for ISLA-DTT on the lug, whose step barely depends on the slice count. All of these ran ISLA 0.9.0. Whether
  > the reference takes 128 slices is the scope freeze's decision.

  Add `R.source('slices_all', 'RESULTS.md')` to the Data list, and an assert:
  `SLA = {a: {r_["field"]: r_ for r_ in R.md_table("slices_all", f"reading_{a}.md") if r_["protocol"] == "native"} for a in ("DR", "AF", "LUG")}`
  then `assert all(r_["label"] == "**no change**" and r_["flag"] == "none" for d in SLA.values() for r_ in d.values())`. (The
  label cell carries bold markers in reading_*.md; strip `*` if R.md_table does not.)

### 12-airfrans.qmd, the configuration bullets (chunk with the `- **ISLA**: width 512, ...` and `- **ISLA-DTT**: ...` bullets)

- ISLA bullet: replace "and is untested on ISLA-DTT (@sec-oq-slices)" with "and costs ISLA-DTT nothing measurable either
  (@sec-oq-slices)".
- ISLA-DTT bullet: after "(@sec-airfrans-dtt-reference)", add: "With 128 slices in place of 512, eager-trained, its test errors
  are 1.028 / 1.010 / 1.024 of this configuration's (two seeds each; no field moves by the registered rule) at 0.53 of the
  training step and 0.80 of the peak memory (@sec-nb-slices-all)." Source: `R.md_table("slices_all", "reading_AF.md")` native
  `arm ÷ control`; smoke cost from RESULTS.md's AF cost line.
- No standing number changes. If the freeze adopts 128 slices, this bullet's "512 slices" changes with every chapter's.

### 16-lug.qmd, the configuration sentence in "What is measured" (`#sec-lug-measured`)

After "ISLA width 512, 12 layers, 512 slices, learning rate 3.75e-4;" nothing changes in that sentence. Add one sentence
after the paragraph, or a clause in the ISLA-DTT passage of the same chapter (the one giving "ISLA-DTT ÷ ISLA ... on the von
Mises"):

> With 128 slices in place of 512, ISLA-DTT's von Mises error is 0.966 times its error at 512 slices (two seeds each, no
> change by the registered rule; the 512-slice seeds sit 21% apart, so only an effect above about 24% would register). Its
> training step falls only to 0.95, since on the lug little of the step scales with the slice count (@sec-nb-slices-all).

This is the first solid-mechanics reading at 128 slices. Source: reading_LUG.md, RESULTS.md (LUG cost).

### 09-drivaerml.qmd, the configuration bullets and `para-drivaerml-slices` (the surface chapter tested)

In `para-drivaerml-slices`, replace "and 128 slices is untested on ISLA-DTT, the surface task and the solids (@sec-oq-slices)"
with:

> on the Surface task (interacting ISLA, the vehicle only on this code) 128 slices reads 1.006 / 1.005 on pressure / wall
> shear, no change by the same rule, at 0.63 of the training step and 0.82 of the peak memory, and ISLA-DTT and the lug read
> no change too (@sec-oq-slices, @sec-nb-slices-all)

Add `R.source('slices_all', 'reading_DR.md')` to its Data line. The Surface task's standing is unchanged.

### The other surface chapters (10-hilift, 11-shift-suv, 17-deepjeb)

No change. None states a slice count outside the instrument lines ("ISLA at 384 wide, 256 slices ..."), and none was tested.
Optional, if the freeze adopts 128 slices: one clause where each states the reference configuration. Not needed now.

### 19-capacity.qmd, the drv-ladder paragraph and the AirfRANS band paragraph (the sentences citing @sec-oq-slices)

Optional. After "On the full dataset, two seeds each, 128 slices reads no change on every field either (...)", add "; so does
ISLA-DTT on AirfRANS and the lug, and the interacting surface form on DrivAerML Surface (@sec-nb-slices-all)". The capacity
reading does not change.

### 14-program-status.qmd, the ledger row `| SLICES-ALL | RUNNING ...`

Replace the status cell with:

`DONE 2026-10-26 (128 slices against 512 on three forms, two seeds each, each arm its control's lane with only n_slices changed,
on 0.9.0 code: ISLA-DTT on AirfRANS, eager, against r090eag_af_dtt, 1.028 / 1.010 / 1.024; ISLA-DTT on the lug, fix-trained, against
r090fix_lug_dtt, 0.966; interacting ISLA on DrivAerML Surface, vehicle only, against r090_dr_isla, 1.006 / 1.005; every field no change,
no up flag; registered verdict: 128 slices is a scope-freeze candidate for every form; step 0.53 / 0.95 / 0.63, memory 0.78-0.82; 429 GPU-h;
@sec-nb-slices-all)`

The rule cell can stay. Replace the last cell "to be recorded" with `@sec-nb-slices-all; results/campaigns_2026-10-26/slices_all/`.
If FINAL-RERUN's row or the freeze decisions cite the slice count as pending, they now cite this row.

### 18-notebook.qmd, the SLICES-DRV entry (`#sec-nb-slices-drv`)

Optional: after "The surface and solid-mechanics datasets were not run at 128 slices.", add "SLICES-ALL ran them; every form
reads no change (@sec-nb-slices-all)." Notebook entries are dated records, so this is only a forward pointer.
