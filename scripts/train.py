"""
Train model for a particular objective and optimizer on evry hyperparameter setting.
"""

import time
from joblib import Parallel, delayed
import sys
import argparse

# Create parser.
sys.path.append(".")
from src.utils.config import (
    L2_REG,
    L2_REG_LARGE,
    L2_REG_SMALL,
    L2_REG_XLARGE,
    LRS,
    SEEDS,
    N_EPOCHS,
)
from src.utils.training import (
    OptimizationError,
    compute_training_curve,
    format_time,
    find_best_optim_cfg,
    FAIL_CODE,
)
from src.utils.io import dict_to_list

# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=[
        "yacht",
        "energy",
        "simulated",
        "concrete",
        "civil_comments",
        "iwildcam",
        "iwildcam_std",
    ],
)
parser.add_argument(
    "--objective",
    type=str,
    required=True,
    choices=[
        "extremile",
        "superquantile",
        "esrm",
        "erm",
        "extremile_lite",
        "superquantile_lite",
        "esrm_lite",
        "extremile_hard",
        "superquantile_hard",
        "esrm_hard",
    ],
)
parser.add_argument(
    "--optimizer",
    type=str,
    required=True,
    choices=[
        "sgd",
        "srda",
        "lsaga",
        "lsvrg",
        "lsaga_uniform",
        "lsvrg_uniform",
        "slsvrg_l2",
        "slsvrg_neg_ent",
        "slsvrg_l2_rnd_check",
        "slsvrg_neg_ent_rnd_check",
    ],
)
parser.add_argument(
    "--loss", type=str, default="squared_error",
)
parser.add_argument(
    "--n_epochs", type=int, default=N_EPOCHS,
)
parser.add_argument(
    "--epoch_len", type=int, default=None,
)
parser.add_argument(
    "--l2_reg", type=str, required=True, choices=["small", "medium", "large", "xlarge"]
)
parser.add_argument("--parallel", type=int, default=1)
parser.add_argument("--n_jobs", type=int, default=-2)
args = parser.parse_args()

# Configure for input to trainers.
dataset = args.dataset
l2_regs = {
    "small": L2_REG_SMALL,
    "medium": L2_REG,
    "large": L2_REG_LARGE,
    "xlarge": L2_REG_XLARGE,
}
l2_reg = l2_regs[args.l2_reg]
model_cfg = {
    "objective": args.objective,
    "l2_reg": l2_reg,
    "loss": args.loss,
    "n_class": None,
}
optim_cfg = {"optimizer": args.optimizer, "lr": LRS, "epoch_len": args.epoch_len}
seeds = SEEDS
n_epochs = args.n_epochs
parallel = bool(args.parallel)

optim_cfgs = dict_to_list(optim_cfg)

config = {
    "dataset": dataset,
    "model_cfg": model_cfg,
    "optim_cfg": optim_cfg,
    "parallel": parallel,
    "seeds": seeds,
    "n_epochs": n_epochs,
    "epoch_len": args.epoch_len,
}

# Display.
print("-----------------------------------------------------------------")
for key in config:
    print(f"{key}:" + " " * (16 - len(key)), config[key])
print("-----------------------------------------------------------------")


# Run optimization.
def worker(optim):
    name, lr = optim["optimizer"], optim["lr"]
    diverged = False
    for seed in seeds:
        code = compute_training_curve(dataset, model_cfg, optim, seed, n_epochs)
        if code == FAIL_CODE:
            diverged = True
    if diverged:
        print(f"Optimizer '{name}' diverged at learning rate {lr}!")


tic = time.time()
if parallel:
    Parallel(n_jobs=args.n_jobs)(delayed(worker)(optim) for optim in optim_cfgs)
else:
    for optim in optim_cfgs:
        worker(optim)
toc = time.time()
print(f"Time:         {format_time(toc-tic)}.")

# Save best configuration.
find_best_optim_cfg(dataset, model_cfg, optim_cfgs, seeds)
