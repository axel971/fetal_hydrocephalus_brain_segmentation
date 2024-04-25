#!/bin/bash

module load singularity

echo "Bayesian U-Net prediction"

singularity exec --nv -w ../../../ubuntu_container/ python3 ../examples/main_get_prediction_cross_validation_Bayesian_U-Net.py


