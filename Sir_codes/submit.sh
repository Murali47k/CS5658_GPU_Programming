#!/bin/bash
#SBATCH -J vec_add
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -t 00:05:00

module unload gnu12
module load cuda/11.3

nvcc vecadd.cu -o vecadd

./vecadd 42 3
