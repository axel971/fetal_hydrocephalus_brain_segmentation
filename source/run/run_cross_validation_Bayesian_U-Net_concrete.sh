#!/bin/bash

module load singularity

echo "Bayesian U-Net 3D spatial concrete dropout"

singularity exec --nv -w ../../../ubuntu_container/ python3 ../examples/main_cross_validation_Bayesian_U-Net_concrete.py

