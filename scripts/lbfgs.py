"""
Run L-BFGS optimizer to get optimal value of spectral risk for a given dataset and regularizer.
Used to compute suboptimality of the optimizers assessed.
"""

import os
import sys
import numpy as np
import torch
from scipy.optimize import minimize
import pickle
import argparse

sys.path.append(".")
from src.utils.config import L2_REG, L2_REG_SMALL, L2_REG_LARGE, L2_REG_XLARGE
from src.utils.data import load_dataset
from src.utils.training import get_objective, OptimizationError
from src.utils.io import var_to_str, get_path

# Create parser.
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
    "--loss", type=str, default="squared_error",
)
parser.add_argument(
    "--l2_reg", type=str, required=True, choices=["small", "medium", "large", "xlarge"]
)
args = parser.parse_args()

dataset = args.dataset
l2_regs = {
    "small": L2_REG_SMALL,
    "medium": L2_REG,
    "large": L2_REG_LARGE,
    "xlarge": L2_REG_XLARGE,
}
model_cfg = {
    "objective": args.objective,
    "l2_reg": l2_regs[args.l2_reg],
    "loss": args.loss,
}

X_train, y_train, X_val, y_val = load_dataset(dataset)
if model_cfg["loss"] == "multinomial_cross_entropy":
    model_cfg["n_class"] = len(torch.unique(y_train))
else:
    model_cfg["n_class"] = None
objective = get_objective(model_cfg, X_train, y_train)


# Define function and Jacobian oracles.
def fun(w):
    return objective.get_batch_loss(torch.tensor(w, dtype=torch.float64)).item()


def jac(w):
    return (
        objective.get_batch_subgrad(
            torch.tensor(w, dtype=torch.float64, requires_grad=True)
        )
        .detach()
        .numpy()
    )


# Run optimizer.
init = np.zeros((objective.d,), dtype=np.float64)
if model_cfg["n_class"]:
    init = np.zeros((model_cfg["n_class"] * objective.d,), dtype=np.float64)
else:
    init = np.zeros((objective.d,), dtype=np.float64)
output = minimize(fun, init, method="L-BFGS-B", jac=jac)
if output.success:
    path = get_path([dataset, var_to_str(model_cfg)])
    f = os.path.join(path, "lbfgs_min_loss.p")
    pickle.dump(output.fun, open(f, "wb"))
else:
    raise OptimizationError(output.message)
