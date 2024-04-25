#!/bin/bash

module load singularity
# sbatch -p gpua --gres=gpu run_cross_validation_U-Net_gpu.sh

echo "U-Net"

singularity exec --nv -w /home/axel/dev/ubuntu_container/ python3 /home/axel/dev/fetal_hydrocephalus_segmentation/source/examples/main_cross_validation_U-Net.py


