Status: DONE

**Verdict: 128 slices is a scope-freeze candidate for every form.** All six fields of the three arms read "no change", and no up flag fired.

**PREREG and controls.** PREREG.md was written before any job: md5 e0221626, sent 12:41 EDT 10-24. Amendment A1, made before any AirfRANS job, brings it to **md5 982bf456**. A1 records the AirfRANS control, r090eag_af_dtt, and the cap of 500 GPU-h, from your reply sent about 00:13 EDT 10-25.
- DR: r090_dr_isla (kernel/r090 rows 6-7, code_r090, per-block compile, compiled scorer). The surface task on 0.9.0 reads the vehicle only.
- LUG: r090fix_lug_dtt (kernel/r090fix rows 0-1, code_cmpaf_fix, fixed compiled path, eager scorer).
- AF: r090eag_af_dtt (kernel/r090eagaf rows 2-3, eager). You named r090fix_af_dtt, but it is not the standing, so I held this arm from 12:41 until your choice at about 00:13.

**Launches (LHR, all bad nodes excluded).**
- Smokes: 369628 (LUG) and 369629 (DR), then 377354 (AF). All passed: configs differ only in run_id and n_slices; geometry kernel as the control's; block-wise d_max 4.7e-7 to 7.9e-7 (bar 1e-5).
- Lanes: DR 369773/369774, LUG 369768/369769 (10-link chains), AF 377564/377565 from the new dir `$T/kernel/slicesall_af`.
- Scorers: 371108, 371109 (DR eager guard), 388478, 404080, 404692. JOBS.md lists every link.
- No commits, since no code changed.

**Reads (native, arm ÷ control; RESULTS.md, reading_{DR,AF,LUG}.md).**
- DR, read 16:12 10-24: pressure 1.006, wall shear 1.005; no change, no flag. Posed and area read the same, and the eager guard passed at 3e-6.
- AF, read 15:30 10-25: pressure 1.028, velocity 1.010, nu_t 1.024; no change, no flag.
- LUG, read 10:52 10-26: von Mises 0.966; no change, no flag. Seed-matched 0.929 / 1.011, beside.

**What weighs on the freeze (how sure: the labels are firm, the resolution is uneven).**
- Shifts below the flag, with the seeds separated: DR +0.5-0.6 % and AF nu_t +2.4 %. These are small costs, but not zero.
- AF pressure and velocity resolve only large harms: the 128-slice seeds spread 12-17 %, so 2 SE is 0.16 / 0.23. The lug's 2 SE is 0.22. Only DR and AF nu_t resolve near the 10 % bar.
- The saving depends on the form. Whole-run step ratios: AF-DTT 0.53, DR 0.63, but the **lug 0.95** (smoke 1.04). Memory is 0.78-0.82 on all three.
- Everything ran on 0.9.0 code, and the surface arm did not test option A's whole boundary.
- On the lug, posed ÷ native is 1.81 / 1.65 at 128 slices, so the 0.9.0 forest's pose defect remains.

**GPU-h: 429.1** against the 500 cap: smokes 2.9, DR 21.8, LUG 286.6, AF 115.8, scorers 2.0 (`sacct_all.txt`).

**Notebook:** NOTEBOOK_DRAFT.md has the entry and the chapter changes for 20 (`#sec-oq-slices`: drop the in-flight callout and close the open-questions row), 12, 16, 09 (the surface chapter tested; 10, 11 and 17 unchanged), 19, the 14 ledger row and the 18 pointer.

**For you to decide.**
- Adopting 128 slices at the freeze is yours, reported to Peter. Weigh the lug's 5 % saving against the forms where the saving is large.
- Peter's rule to record in project memory: none new. Your own rule from this lane, already in effect: resend any request unanswered for an hour as a progress note that names the decision.
- Two departures from the brief: the DR and LUG smokes ran in short QoS (smoke.sbatch's header), against item 7; and the AF arm uses its own launcher dir, `kernel/slicesall_af`, so the running launcher was not edited.

**What is left, what stays.** Nothing is running, and the registry row is removed.
- On LHR: `$T/runs/slicesall_{dr,lug,af}_*`, about 125 GB per run, about 756 GB in all, mostly intermediate checkpoints (save_interval 2, as the controls). Trimming them to the final checkpoint is your call. Also `$T/kernel/slicesall*` (4.3 GB) and `$T/evals_slicesall*`.
- Locally: this folder, 4.4 MB.
