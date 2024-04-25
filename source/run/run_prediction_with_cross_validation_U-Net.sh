#!/bin/bash

module load singularity

echo "U-Net prediction"

singularity exec --nv -w ../../../ubuntu_container/ python3 ../examples/main_prediction_with_cross_validation_U-Net.py


