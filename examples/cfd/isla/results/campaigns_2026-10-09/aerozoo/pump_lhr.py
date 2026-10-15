#!/usr/bin/env python3
"""AZ-PUMP: the AEROZOO scoring pump for the LHR site (2026-10-14), installed as $T/kernel/aerozoo/pump_lhr.py.

Idempotent and safe to run every hour. For every lanes.tsv row whose LHR run has 'Training completed' in runs/<run>/train.log,
it submits whichever of the four scores is absent (no <OUT_ROOT>/<run>/.done), all through eval_bestval.sbatch (ROWS = up to four
lanes per job, one GPU each; sbatch -q normal --time 01:59:00, from $T/kernel/aerozoo):
  mode          checkpoint                                side  OUT_ROOT                     read locally by fetch_scores.sh as
  test          final  (CKPT_ROOT=$T/runs)                test  $T/p200k_fp32_test           scores/<run>/test
  val           final  (CKPT_ROOT=$T/runs)                val   $T/p200k_fp32                scores/<run>/val
  bestval       best   (CKPT_ROOT=$T/runs_bestval)        val   $T/p200k_fp32_bestval        scores/<run>/bestval
  bestval_test  best   (CKPT_ROOT=$T/runs_bestval)        test  $T/p200k_fp32_bestval_test   scores/<run>/bestval_test
Final checkpoint: infer.py calls physicsnemo.utils.load_checkpoint(path=<CKPT_ROOT>/<run>/checkpoints) without an epoch, which loads
the largest checkpoint.0.<E>.pt index (numeric sort), i.e. the final epoch's checkpoint (the trainer always saves the last epoch).
Best checkpoint: the val_summary epoch E of the minimum validation loss in runs/<run>/metrics.jsonl (the last record per epoch;
the earliest epoch on a tie) selects checkpoint E+1; runs_bestval/<run>/checkpoints holds absolute symlinks to checkpoint.0.(E+1).pt,
<Model>.0.(E+1).mdlus and norm_stats.pt. The pump builds that directory when it is absent (and writes PUMP_SELECTION beside it); an
existing directory is never changed: if it links another epoch than the protocol selects, the run's best-checkpoint modes are
reported as STALE and not submitted, and a WARN line names it when it was already scored (a directory built by hand mid-training).
The test side's selection is eval_test.sbatch's (eval_bestval.sbatch with BESTVAL_SIDE=test turns val_split=<split>.val into
<split>.test and passes infer_split=test); every lane's overrides (Hydra deletions ~model.*, ++model.* overrides) pass through
unchanged, exactly as the scorers handle them.
State: pump_state.tsv (run, mode, jobid, submit time), one line per submission. A (run, mode) whose last job is still queued or
running (squeue) is never resubmitted; a job that ended without the .done marker is resubmitted at most twice (three attempts),
after which the (run, mode) is reported as FAILED. A (run, mode) the pump never submitted whose <OUT_ROOT>/<run>.log was written in
the last two hours is left alone (BUSY: a scorer someone else started). A lock file keeps two pump runs from overlapping.
usage: python3 pump_lhr.py [--dry-run]     (prints one line per action and a summary; --dry-run submits, links and records nothing)
"""
import argparse, fcntl, getpass, glob, json, math, os, re, subprocess, sys, time

T = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"
MODES = {  # mode -> (BESTVAL_SIDE, CKPT_ROOT under T, OUT_ROOT under T)
    "test": ("test", "runs", "p200k_fp32_test"),
    "val": ("val", "runs", "p200k_fp32"),
    "bestval": ("val", "runs_bestval", "p200k_fp32_bestval"),
    "bestval_test": ("test", "runs_bestval", "p200k_fp32_bestval_test"),
}
ROWS_PER_JOB = 4
MAX_ATTEMPTS = 3                       # the first submission and at most two resubmissions
BUSY_SECONDS = 2 * 3600                # a foreign scorer's log written this recently means it may still be running
SBATCH = ["sbatch", "--parsable", "-q", "normal", "--time", "01:59:00"]


def lanes(path):
    rows = []
    with open(path) as f:
        head = f.readline().rstrip("\n").split("\t")
        for line in f:
            if line.strip():
                rows.append(dict(zip(head, line.rstrip("\n").split("\t"))))
    return rows


def read_state(path):
    jobs = {}
    if os.path.exists(path):
        for line in open(path):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3 and p[0] != "run":
                jobs.setdefault((p[0], p[1]), []).append(p[2])
    return jobs


def active_jobs():
    """Job ids of this user's queued or running jobs; None when squeue fails (then nothing is submitted)."""
    try:
        user = os.environ.get("USER") or getpass.getuser()
        out = subprocess.run(["squeue", "-h", "-u", user, "-o", "%i"], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return {j.split("_")[0] for j in out.stdout.split()}


def best_epoch(root, run):
    """(E, loss) of the minimum validation loss over runs/<run>/metrics.jsonl val_summary rows, or None."""
    p = f"{root}/runs/{run}/metrics.jsonl"
    if not os.path.exists(p):
        return None
    by_epoch = {}
    with open(p, errors="ignore") as f:
        for line in f:
            if '"val_summary"' not in line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            e, loss = rec.get("epoch"), rec.get("loss")
            if rec.get("phase") == "val_summary" and isinstance(e, int) and isinstance(loss, (int, float)):
                by_epoch[e] = loss          # the last record per epoch counts (a resumed link re-validates)
    good = sorted((e, l) for e, l in by_epoch.items() if math.isfinite(l))
    return min(good, key=lambda el: (el[1], el[0])) if good else None


def ckpt_indices(ckdir):
    return sorted(int(m.group(1)) for f in os.listdir(ckdir) if (m := re.fullmatch(r"checkpoint\.0\.(\d+)\.pt", f)))


def linked_epoch(ckdir):
    idx = ckpt_indices(ckdir) if os.path.isdir(ckdir) else []
    return idx[-1] if len(idx) == 1 else None


def best_links(root, run, dry_run, log):
    """Make sure runs_bestval/<run>/checkpoints exists and links the protocol's checkpoint. Returns None when the best-checkpoint
    modes may be submitted, else the reason they may not."""
    dest = f"{root}/runs_bestval/{run}"
    sel = best_epoch(root, run)
    if sel is None:
        return "no val_summary rows in metrics.jsonl"
    e, loss = sel
    ck = e + 1
    if os.path.isdir(f"{dest}/checkpoints"):
        have = linked_epoch(f"{dest}/checkpoints")
        if have != ck:
            return f"STALE runs_bestval/{run} links checkpoint {have}, the protocol selects {ck} (val epoch {e}, loss {loss:.5g})"
        return None
    src = f"{root}/runs/{run}/checkpoints"
    models = [os.path.basename(f) for f in glob.glob(f"{src}/*.0.{ck}.mdlus")]
    names = [f"checkpoint.0.{ck}.pt", "norm_stats.pt"] + models
    missing = [n for n in names if not os.path.exists(f"{src}/{n}")]
    if missing or len(models) != 1:
        return f"cannot link checkpoint {ck} (val epoch {e}): missing {missing or 'a single <Model>.0.' + str(ck) + '.mdlus'}"
    log(f"LINK {run}: runs_bestval/{run}/checkpoints -> checkpoint {ck} (val epoch {e}, loss {loss:.5g})"
        + (" [dry-run]" if dry_run else ""))
    if not dry_run:
        tmp = f"{root}/runs_bestval/.{run}.pump_tmp"
        os.makedirs(f"{tmp}/checkpoints", exist_ok=True)
        for n in names:
            if not os.path.lexists(f"{tmp}/checkpoints/{n}"):
                os.symlink(f"{src}/{n}", f"{tmp}/checkpoints/{n}")
        with open(f"{tmp}/PUMP_SELECTION", "w") as f:
            f.write(f"val_summary epoch {e} loss {loss!r} -> checkpoint {ck}; built by pump_lhr.py {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        os.rename(tmp, dest)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="print the submissions without making them")
    ap.add_argument("--root", default=T, help=argparse.SUPPRESS)   # the site root; tests point it at a mock tree
    a = ap.parse_args()
    root, dry = a.root, a.dry_run
    kernel = f"{root}/kernel/aerozoo"
    state_path, scorer = f"{kernel}/pump_state.tsv", f"{kernel}/eval_bestval.sbatch"
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    log = lambda s: print(s, flush=True)
    log(f"AZ-PUMP {stamp}{' DRY-RUN' if dry else ''} root={root}")
    lock = None
    if not dry:
        lock = open(f"{kernel}/pump_lhr.lock", "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            log("another pump run holds the lock; nothing done")
            return 0
    active = active_jobs()
    if active is None:
        log("ERROR squeue failed; nothing submitted")
        return 2
    jobs = read_state(state_path)
    counts = {"rows": 0, "no_lhr_run": 0, "training": 0, "completed": 0}
    per = {m: {k: 0 for k in ("scored", "waiting", "busy", "failed", "skipped", "pending")} for m in MODES}
    pending = {m: [] for m in MODES}
    for row in lanes(f"{kernel}/lanes.tsv"):
        counts["rows"] += 1
        idx, run, extra, epochs = row["idx"], row["run_id"], row.get("extra", ""), row.get("epochs", "")
        rdir = f"{root}/runs/{run}"
        if not os.path.isdir(rdir):
            counts["no_lhr_run"] += 1
            continue
        try:
            completed = "Training completed" in open(f"{rdir}/train.log", errors="ignore").read()
        except OSError:
            completed = False
        if not completed:
            counts["training"] += 1
            continue
        counts["completed"] += 1
        idxs = ckpt_indices(f"{rdir}/checkpoints") if os.path.isdir(f"{rdir}/checkpoints") else []
        if not idxs or (epochs.isdigit() and idxs[-1] != int(epochs)):
            log(f"WARN {run}: final checkpoint index {idxs[-1] if idxs else None} != epochs column {epochs}")
        links_reason = "unchecked"
        bdir = f"{root}/runs_bestval/{run}"
        if os.path.isdir(f"{bdir}/checkpoints") and not os.path.exists(f"{bdir}/PUMP_SELECTION"):
            links_reason = best_links(root, run, dry, log)   # a hand-built selection: check it against the protocol once trained
            if links_reason:
                log(f"WARN {run}: {links_reason}")
        for mode, (side, ckroot, outroot) in MODES.items():
            out = f"{root}/{outroot}/{run}"
            if os.path.exists(f"{out}/.done"):
                if "gave up" in open(f"{out}/.done", errors="ignore").read():
                    log(f"FAILED {run} {mode}: {outroot}/{run}/.done is a give-up marker; see {outroot}/{run}.log")
                    per[mode]["failed"] += 1
                else:
                    per[mode]["scored"] += 1
                continue
            mine = jobs.get((run, mode), [])
            if mine and mine[-1] in active:
                log(f"WAIT {run} {mode}: job {mine[-1]} queued or running (attempt {len(mine)})")
                per[mode]["waiting"] += 1
                continue
            if len(mine) >= MAX_ATTEMPTS:
                log(f"FAILED {run} {mode}: {len(mine)} jobs ({' '.join(mine)}) ended without .done; see {outroot}/{run}.log")
                per[mode]["failed"] += 1
                continue
            if not mine and os.path.exists(f"{out}.log") and time.time() - os.path.getmtime(f"{out}.log") < BUSY_SECONDS:
                log(f"BUSY {run} {mode}: {outroot}/{run}.log written {(time.time() - os.path.getmtime(f'{out}.log')) / 60:.0f} min "
                    f"ago by a scorer the pump did not submit; left alone")
                per[mode]["busy"] += 1
                continue
            if side == "test" and not re.search(r"val_split=[A-Za-z0-9_]+\.val", extra):
                log(f"SKIP {run} {mode}: no val_split=<split>.val override to turn into .test")
                per[mode]["skipped"] += 1
                continue
            if ckroot == "runs_bestval":
                if links_reason == "unchecked":
                    links_reason = best_links(root, run, dry, log)
                if links_reason:
                    log(f"SKIP {run} {mode}: {links_reason}")
                    per[mode]["skipped"] += 1
                    continue
            pending[mode].append((idx, run, len(mine) + 1))
            per[mode]["pending"] += 1
    submitted = 0
    for mode, (side, ckroot, outroot) in MODES.items():
        for i in range(0, len(pending[mode]), ROWS_PER_JOB):
            group = pending[mode][i:i + ROWS_PER_JOB]
            env = {"ROWS": " ".join(g[0] for g in group), "BESTVAL_SIDE": side, "CKPT_ROOT": f"{root}/{ckroot}",
                   "OUT_ROOT": f"{root}/{outroot}"}
            cmd = SBATCH + ["-J", f"isla.az.pump-{mode}", scorer]
            what = (f"{mode} rows {env['ROWS']} ({', '.join(f'{g[1]}#{g[2]}' for g in group)}): "
                    + " ".join(f"{k}='{v}'" for k, v in env.items()) + " " + " ".join(cmd))
            if dry:
                log(f"SUBMIT [dry-run] {what}")
                submitted += 1
                continue
            r = subprocess.run(cmd, cwd=kernel, env={**os.environ, **env}, capture_output=True, text=True)
            jid = r.stdout.strip().split(";")[0]
            if r.returncode != 0 or not jid.isdigit():
                log(f"ERROR sbatch failed for {what}: rc={r.returncode} {r.stderr.strip()}")
                continue
            new = not os.path.exists(state_path)
            with open(state_path, "a") as f:
                if new:
                    f.write("run\tmode\tjobid\tsubmitted\n")
                for g in group:
                    f.write(f"{g[1]}\t{mode}\t{jid}\t{time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
            log(f"SUBMIT job {jid} {what}")
            submitted += 1
    log(f"SUMMARY rows={counts['rows']} no_lhr_run={counts['no_lhr_run']} training={counts['training']} "
        f"completed={counts['completed']} jobs_{'planned' if dry else 'submitted'}={submitted}")
    for mode in MODES:
        log(f"SUMMARY {mode:12s} " + " ".join(f"{k}={v}" for k, v in per[mode].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
