#!/bin/bash
#SBATCH -C gpu
#SBATCH -t 00:40:00
#SBATCH -G 1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH -c 8

module load cgpu
module load python
module load cuda
module load cudnn
module load gcc
module load openmpi
source activate test

srun -C gpu -t 00:40:00 -G 1 dlprof --nsys_opts="-d 200 -t cuda,nvtx,oshmem --mpi-impl=openmpi -s none" python mpi_learn.py
