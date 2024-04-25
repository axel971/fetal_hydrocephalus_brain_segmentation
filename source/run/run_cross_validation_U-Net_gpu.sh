#!/bin/bash

module load singularity
# sbatch -p gpua --gres=gpu run_cross_validation_U-Net_gpu.sh

echo "U-Net"

singularity exec --nv -w ../../../ubuntu_container/ python3 ../examples/main_cross_validation_U-Net.py


