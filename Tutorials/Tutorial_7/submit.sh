#!/bin/bash
#SBATCH --job-name=matmulpr
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

module load cuda

nvcc -O3 matmulpr.cu -o matmulpr

nsys profile --force-overwrite=true -o matmulpr_report ./matmulpr

nsys stats matmulpr_report.nsys-rep
