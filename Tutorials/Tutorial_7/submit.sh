#!/bin/bash
#SBATCH --job-name=matmulpr
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

/opt/ohpc/pub/apps/cuda/12.1/bin/nsys profile \
    --force-overwrite=true \
    -o matmul_profile \
    ./matmulpr

/opt/ohpc/pub/apps/cuda/12.1/bin/nsys stats matmul_profile.nsys-rep