#!/bin/bash
#SBATCH --job-name=matmulpr
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

nsys profile \
    --stats=true \
    --trace=cuda,nvtx,osrt \
    -o matmul_profile \
    ./matmulpr