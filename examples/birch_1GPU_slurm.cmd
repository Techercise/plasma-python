#!/bin/bash
#SBATCH -o frnn_1gpu_spock.out
#SBATCH -p batch
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive

module load rocm
module load cray-python
module load gcc
module load craype-accel-amd-gfx908
conda activate tf_rocm

srun python mpi_learn.py
