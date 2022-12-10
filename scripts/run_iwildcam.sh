#!/bin/bash

# Use to disable parallelism for time comparisons, otherwise leave commented.
# export MKL_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1
# export OMP_NUM_THREADS=1

for objective in erm extremile esrm
do
    python scripts/lbfgs.py --dataset iwildcam_std --objective $objective --loss multinomial_cross_entropy --l2_reg medium
    for optim in sgd srda lsvrg lsvrg_uniform
    do
        python scripts/train.py --dataset iwildcam_std --objective $objective --optimizer $optim --loss multinomial_cross_entropy --n_epochs 6400 --epoch_len 100 --l2_reg medium --n_jobs 8
    done
done