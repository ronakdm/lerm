"""
Rerun the train routine for the best hyperparameters of each model
and record bias, variance, and sorting error.
"""
import time
from joblib import Parallel, delayed
import os
import sys
import argparse
import pickle

sys.path.append(".")
from src.utils.io import var_to_str, get_path
from src.utils.config import L2_REG, SEEDS, N_EPOCHS
from src.utils.training import (
    compute_training_curve,
    format_time,
)

l2_reg = L2_REG

# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=["yacht", "energy", "simulated", "concrete", "civil_comments"],
)
parser.add_argument(
    "--objective",
    type=str,
    required=True,
    choices=["extremile", "superquantile", "esrm", "erm"],
)
parser.add_argument(
    "--optimizer",
    type=str,
    required=True,
    choices=["sgd", "srda", "lsaga", "lsvrg", "lsaga_uniform", "lsvrg_uniform"],
)
parser.add_argument(
    "--loss", type=str, default="squared_error",
)
parser.add_argument(
    "--n_epochs", type=int, default=N_EPOCHS,
)
parser.add_argument(
    "--l2_reg", type=float, default=L2_REG,
)
parser.add_argument("--parallel", type=int, default=1)
args = parser.parse_args()


# Configure for input to trainers.
dataset = args.dataset
l2_reg = args.l2_reg
model_cfg = {
    "objective": args.objective,
    "l2_reg": l2_reg,
    "loss": args.loss,
    "n_class": None,
}
# Get best learning rate.
path = get_path([dataset, var_to_str(model_cfg), args.optimizer])
optim = pickle.load(open(os.path.join(path, "best_cfg.p"), "rb"))
seeds = SEEDS
n_epochs = args.n_epochs
parallel = bool(args.parallel)
optim["epoch_len"] = 64

config = {
    "dataset": dataset,
    "model_cfg": model_cfg,
    "optim_cfg": optim,
    "parallel": parallel,
    "seeds": seeds,
    "n_epochs": n_epochs,
}

# Display.
print("-----------------------------------------------------------------")
print("Analyzing trajectory...")
for key in config:
    print(f"{key}:" + " " * (16 - len(key)), config[key])
print("-----------------------------------------------------------------")


# Run optimization.
def worker(seed):
    compute_training_curve(dataset, model_cfg, optim, seed, n_epochs, profile=True)


tic = time.time()
if parallel:
    Parallel(n_jobs=-2)(delayed(worker)(seed) for seed in seeds)
else:
    [worker(seed) for seed in seeds]
toc = time.time()
print(f"Time:         {format_time(toc-tic)}.")
