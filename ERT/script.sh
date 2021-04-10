#!/bin/bash
#SBATCH --account=soc-gpu-kp
#SBATCH --partition=soc-gpu-kp
#SBATCH --job-name=Roofline-Kingspeak-P100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --export=ALL
#SBATCH --qos=soc-gpu-kp

module load cuda
nvprof --print-gpu-summary ./ker
nvprof --kernels sobel_kernel_register_reuse --metrics all ./ker