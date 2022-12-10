#!/bin/bash

# Use to disable parallelism for time comparisons, otherwise leave commented.
# export MKL_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1
# export OMP_NUM_THREADS=1

for objective in esrm esrm_lite esrm_hard
do
    for dataset in yacht energy concrete simulated
    do
        python scripts/lbfgs.py --dataset $dataset --objective $objective --l2_reg medium
        for optim in sgd srda lsvrg lsvrg_uniform
        do
            python scripts/train.py --dataset $dataset --objective $objective --optimizer $optim --l2_reg medium
        done
    done
done