#!/bin/bash
for objective in erm superquantile extremile esrm
do
    for dataset in yacht energy simulated concrete
    do
        python scripts/profile_trajectory.py --dataset $dataset --objective $objective --optimizer lsvrg --n_epochs 64 --l2_reg 1
    done
done