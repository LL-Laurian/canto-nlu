"""Cross-validated learning-rate search and final test runs for the ModernBERT models, on Slurm.

    python slurm/hpo.py submit [--dry-run]    # submit every run that has not finished yet
    python slurm/hpo.py report                # CV tables, chosen learning rates, final test scores

`submit` is safe to re-run: it skips runs that have a finished log in logs/ or are still queued, and
resubmits failed ones. Once every fold of a model/task grid has finished, the next `submit` also
launches the final runs: the best learning rate by mean CV score and the original 2e-5, each with
every seed in SEEDS, trained on the full training split and scored on the test set.

Runs are submitted as one Slurm array per task (slurm/hpo_array.sh). Run from the repository root on
a login node: this script only reads logs and calls sbatch/squeue. Both commands take --models and
--tasks to restrict them, e.g. `--models mono --tasks pos deps`.
"""
import argparse
import collections
import getpass
import glob
import os
import re
import statistics
import subprocess
import time

MODELS = {"mono": "models/yue-modernbert-monolingual", "chinese": "models/chinese-modernbert",
          "transfer": "models/yue-modernbert-transfer"}
# Selection metric (as printed on run.py's "Final ..." line) and Slurm time limit per task
TASKS = {"ld": ("macro F1", "01:00:00"), "laj": ("macro F1", "01:00:00"), "pos": ("macro F1", "01:00:00"),
         "deps": ("LAS", "02:00:00")}
LEARNING_RATES = [2e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 8e-4]
BASELINE_LR = 2e-5
CV_FOLDS = 5
SEEDS = [42, 1, 2, 3, 4]
LEDGER = "logs/hpo_submitted.tsv"  # array job id, array index, run line: lets `submit` skip queued runs

# fold is None for a final (test set) run; CV runs always use seed 42
Run = collections.namedtuple("Run", "model task lr warmup wd fold seed")


def default_warmup(task):
    return 0.1 if task == "deps" else 0.0


def parse_run(model_dir, task, extra):
    """The Run that `slurm/finetune.sh MODEL_DIR TASK EXTRA` performs, with run.py's defaults filled in."""
    opts = dict(re.findall(r"--(\w+)=(\S+)", extra))
    name = os.path.basename(model_dir.rstrip("/"))
    model = next((k for k, v in MODELS.items() if os.path.basename(v) == name), name)
    fold = int(opts["fold"]) if "cv_folds" in opts else None
    return Run(model, task, float(opts.get("lr", BASELINE_LR)),
               float(opts.get("warmup_ratio", default_warmup(task))), float(opts.get("weight_decay", 0.0)),
               fold, 42 if fold is not None else int(opts.get("seed", 42)))


def run_line(run):
    """`slurm/finetune.sh` arguments for a Run, passing only the options that differ from run.py's defaults."""
    line = f"{MODELS[run.model]} {run.task} --lr={run.lr:g}"
    if run.warmup != default_warmup(run.task):
        line += f" --warmup_ratio={run.warmup:g}"
    if run.wd:
        line += f" --weight_decay={run.wd:g}"
    if run.fold is not None:
        return line + f" --cv_folds={CV_FOLDS} --fold={run.fold}"
    return line + f" --seed={run.seed}"


def collect():
    """{Run: metrics} from every finished log in logs/; for runs repeated, the latest job wins."""
    found = {}
    for path in glob.glob("logs/*.out"):
        text = open(path, encoding="utf-8", errors="replace").read()
        header = re.search(r"^model=(\S+) task=(\S+) extra args: ?(.*)$", text, re.M)
        job = re.search(r"-(\d+)(?:_(\d+))?\.out$", path)
        if not header or not job:
            continue
        run = parse_run(*header.groups())
        if run.task == "deps":
            metrics = {k: re.search(rf"^Final deps {k}: (\S+)", text, re.M) for k in ("LAS", "UAS")}
        else:
            metrics = {"macro F1": re.search(rf"^Final {run.task} test macro F1: (\S+)", text, re.M)}
        if not all(metrics.values()):
            continue  # failed, cancelled or still running
        order = (int(job.group(1)), int(job.group(2) or 0))
        if run not in found or order > found[run][0]:
            found[run] = (order, {k: float(m.group(1)) for k, m in metrics.items()})
    return {run: metrics for run, (_, metrics) in found.items()}


def cv_scores(results, model, task):
    """{(lr, warmup, wd): [score per fold]} for every configuration with all folds finished."""
    metric = TASKS[task][0]
    folds = collections.defaultdict(dict)
    for run, metrics in results.items():
        if run.model == model and run.task == task and run.fold is not None:
            folds[(run.lr, run.warmup, run.wd)][run.fold] = metrics[metric]
    return {cfg: [f[i] for i in range(CV_FOLDS)] for cfg, f in folds.items() if len(f) == CV_FOLDS}


def grid(model, task):
    return [Run(model, task, lr, default_warmup(task), 0.0, fold, 42)
            for lr in LEARNING_RATES for fold in range(CV_FOLDS)]


def queued_runs():
    """Run lines of this script's runs still in the queue, and how many of the user's queued jobs it did not submit."""
    queue = subprocess.run(["squeue", "-h", "-r", "-u", getpass.getuser(), "-o", "%i"],
                           capture_output=True, text=True, check=True).stdout.split()
    ledger = {}
    if os.path.exists(LEDGER):
        for line in open(LEDGER, encoding="utf-8"):
            job, index, run = line.rstrip("\n").split("\t")
            ledger[f"{job}_{index}"] = run
    ours = {job.split("_")[0] for job in ledger}
    foreign = [job for job in queue if job.split("_")[0] not in ours]
    return {ledger[job] for job in queue if job in ledger}, len(foreign)


def submit(args):
    results = collect()
    queued, foreign = queued_runs()
    if foreign and not args.force:
        raise SystemExit(f"{foreign} of your queued jobs were not submitted by this script, so their runs could be "
                         f"submitted twice. Wait for them to finish, or pass --force.")
    todo = collections.defaultdict(list)
    for model in args.models:
        for task in args.tasks:
            runs = grid(model, task)
            if all(run in results for run in runs):
                cv = cv_scores(results, model, task)
                best = max(cv, key=lambda cfg: statistics.mean(cv[cfg]))
                baseline = (BASELINE_LR, default_warmup(task), 0.0)
                runs = [Run(model, task, *cfg, None, seed) for cfg in dict.fromkeys([best, baseline])
                        for seed in SEEDS]
            for run in runs:
                if run not in results and run_line(run) not in queued:
                    todo[task].append(run_line(run))
    if not todo:
        print("Nothing to submit: every run has finished or is queued. See `python slurm/hpo.py report`.")
    os.makedirs("logs/hpo_runs", exist_ok=True)
    for task, lines in todo.items():
        print(f"{task}: {len(lines)} runs")
        if args.dry_run:
            print("  " + "\n  ".join(lines))
            continue
        runs_file = f"logs/hpo_runs/{task}-{time.strftime('%Y%m%d-%H%M%S')}.txt"
        with open(runs_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        job = subprocess.run(["sbatch", "--parsable", f"--array=0-{len(lines) - 1}", f"--time={TASKS[task][1]}",
                              f"--job-name=hpo-{task}", "slurm/hpo_array.sh", runs_file],
                             capture_output=True, text=True, check=True).stdout.strip().split(";")[0]
        with open(LEDGER, "a", encoding="utf-8") as f:
            f.writelines(f"{job}\t{i}\t{line}\n" for i, line in enumerate(lines))
        print(f"  submitted array job {job} ({runs_file})")


def mean_sd(scores):
    scores = [100 * s for s in scores]
    sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
    return f"{statistics.mean(scores):6.2f} ± {sd:4.2f}"


def report(args):
    results = collect()
    for model in args.models:
        for task in args.tasks:
            metric = TASKS[task][0]
            done = sum(run in results for run in grid(model, task))
            print(f"\n== {model} {task} ({metric}): {done}/{len(LEARNING_RATES) * CV_FOLDS} grid runs finished")
            cv = cv_scores(results, model, task)
            ranked = sorted(cv, key=lambda cfg: statistics.mean(cv[cfg]), reverse=True)
            for cfg in ranked:
                lr, warmup, wd = cfg
                note = " (baseline)" if cfg == (BASELINE_LR, default_warmup(task), 0.0) else ""
                print(f"   CV  lr={lr:<7g} warmup={warmup:<4g} wd={wd:<5g} {mean_sd(cv[cfg])}{note}")
            if ranked and ranked[0][0] == max(LEARNING_RATES):
                print("   note: the best learning rate is the largest in the grid; the optimum may lie higher")
            finals = collections.defaultdict(dict)
            for run, metrics in results.items():
                if run.model == model and run.task == task and run.fold is None:
                    finals[(run.lr, run.warmup, run.wd)][run.seed] = metrics
            for cfg, by_seed in sorted(finals.items()):
                scores = {k: [by_seed[s][k] for s in SEEDS if s in by_seed] for k in by_seed[next(iter(by_seed))]}
                n = len(next(iter(scores.values())))
                if n:
                    label = "baseline" if cfg == (BASELINE_LR, default_warmup(task), 0.0) else "tuned"
                    shown = ", ".join(f"{k} {mean_sd(v)}" for k, v in scores.items())
                    print(f"   TEST {label:<8} lr={cfg[0]:<7g} {shown} ({n}/{len(SEEDS)} seeds)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["submit", "report"])
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    parser.add_argument("--dry-run", action="store_true", help="submit: list the runs without submitting them")
    parser.add_argument("--force", action="store_true", help="submit: proceed despite queued jobs from elsewhere")
    args = parser.parse_args()
    submit(args) if args.command == "submit" else report(args)
