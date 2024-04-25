#!/bin/bash

module load singularity

echo "Bayesian U-Net"

singularity exec --nv -w ../../../ubuntu_container/ python3 ../examples/main_cross_validation_Bayesian_U-Net.py

