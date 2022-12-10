import torch
import sys

sys.path.append(".")
from src.utils.data import load_dataset
from src.utils.training import get_optimizer, get_objective
from src.utils.config import LRS

X_train, y_train, X_test, y_test = load_dataset("iwildcam")

objective = "erm"
l2_reg = 1.0
loss = "multinomial_cross_entropy"

model_cfg = {
    "objective": objective,
    "l2_reg": l2_reg,
    "loss": loss,
    "n_class": None,
}
if model_cfg["loss"] == "multinomial_cross_entropy":
    model_cfg["n_class"] = len(torch.unique(y_train))

train_obj = get_objective(model_cfg, X_train, y_train)
val_obj = get_objective(model_cfg, X_test, y_test)

optimizer_name = "sgd"
epoch_len = 1000
seed = 25

optim_cfg = {"optimizer": optimizer_name, "lr": 0.1, "epoch_len": epoch_len}
optimizer = get_optimizer(optim_cfg, train_obj, seed)

for i in range(3):
    optimizer.start_epoch()
    for _ in range(epoch_len):
        optimizer.step()
        print(train_obj.get_batch_loss(optimizer.weights).item())
    optimizer.end_epoch()
# optim_cfgs = dict_to_list({"optimizer": "osvrg", "lr": LRS,})

# find_best_optim_cfg(dataset, model_cfg, optim_cfgs, SEEDS)

# import sys
# import matplotlib.pyplot as plt
# import seaborn as sns

# sys.path.append(".")
# from src.utils.plotting import plot_traj
# from src.utils.config import SEEDS, L2_REG

# seeds = SEEDS
# l2_reg = L2_REG

# fig, ax = plt.subplots(1, 1, figsize=(5, 5))

# objective = "extremile"
# dataset = "iwildcam"
# optim = {"optimizer": "sgd", "lr": 0.001}
# n_epochs = 4
# seed = 1
# model_cfg = {"objective": objective, "l2_reg": l2_reg, "loss": "multinomial_cross_entropy"}

# compute_training_curve(dataset, model_cfg, optim, seed, n_epochs)
