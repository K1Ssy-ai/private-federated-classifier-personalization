"""Launch one independent reproduction stage. Run summarize_runs.py to replay the archive."""
import argparse
import itertools
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent / "experiments"
MAIN = ROOT / "followup_20261104" / "unified_v2"
CHECKS = ROOT / "followup_20261105"
SEEDS = {"fashion": range(2030, 2035), "kmnist": range(2041, 2046), "cifar": range(2040, 2045)}


def run(script, mode=None, **options):
    command = [sys.executable, str(script)]
    for key, value in options.items():
        command.extend(["--" + key, str(value)])
    env = os.environ.copy()
    if mode is not None:
        env["BUDGET_MODE"] = mode
    subprocess.run(command, check=True, env=env)


def pilots():
    for dataset, seed in itertools.product(SEEDS, (9000, 9001)):
        run(MAIN / "shared_pilot.py", dataset=dataset, seed=seed, stage="pilot")
        run(MAIN / "pilot_adapt.py", dataset=dataset, seed=seed)
    run(MAIN / "freeze_selection.py")


def formal(dataset):
    selected = MAIN / "pilot" / dataset / "selection_frozen.json"
    if not selected.exists():
        selected.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT.parent / "published_settings" / (dataset + ".json"), selected)
    for alpha, epsilon, seed in itertools.product((0.1, 0.5, 1), (4, 8), SEEDS[dataset]):
        options = dict(dataset=dataset, alpha=alpha, epsilon=epsilon, seed=seed)
        run(MAIN / "shared_pilot.py", stage="formal", **options)
        run(MAIN / "formal_adapt.py", **options)
    for seed in (9000, 9001):
        run(MAIN / "shared_pilot.py", stage="pilot", dataset=dataset, seed=seed)
    run(CHECKS / "e1_histogram.py", dataset=dataset)


def clipping_and_whole_budget(dataset):
    if dataset == "kmnist":
        raise SystemExit("The threshold and whole-budget diagnostics use fashion and cifar.")
    for alpha, seed in itertools.product((0.1, 1), range(2030, 2035)):
        options = dict(dataset=dataset, seed=seed, alpha=alpha)
        if dataset == "cifar":
            run(CHECKS / "shared_run.py", mode="dual", stage="formal", epsilon=4, **options)
            run(CHECKS / "formal_cifar.py", epsilon=4, **options)
            run(CHECKS / "e3_hist_eval.py", alpha=alpha, seed=seed)
        run(CHECKS / "e2_clip.py", **options)
        run(CHECKS / "shared_run.py", mode="shared", stage="formal", epsilon=4, **options)
        run(CHECKS / "e3_evaluate.py", **options)


def timing(dataset):
    if os.name != "nt":
        raise SystemExit("The archived timing protocol samples Windows working-set memory.")
    for method, client in itertools.product(("B1", "B2", "B3", "B4"), (0, 7, 14)):
        run(CHECKS / "e4_efficiency.py", dataset=dataset, method=method, client=client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["pilots", "formal", "clipping", "timing", "kmnist-extension"])
    parser.add_argument("--dataset", choices=SEEDS)
    args = parser.parse_args()
    if args.stage == "pilots":
        pilots()
    elif args.stage == "kmnist-extension":
        run(ROOT / "prior_residual_pilot_v1" / "run_experiment.py")
        run(ROOT / "dp_class_scale_confirm_v1" / "run_confirm.py")
        run(ROOT / "bias_theory_check_v1" / "run_experiment.py")
        run(ROOT / "bias_theory_check_v1" / "complete_diagnostics.py")
    elif args.dataset is None:
        parser.error("Select --dataset for this stage.")
    else:
        {"formal": formal, "clipping": clipping_and_whole_budget, "timing": timing}[args.stage](args.dataset)
