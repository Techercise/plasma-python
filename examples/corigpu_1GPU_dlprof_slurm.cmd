#!/bin/bash
#SBATCH -C gpu
#SBATCH -t 03:00:00
#SBATCH -G 1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH -c 1

module load cgpu
module load python
module load cuda
module load cudnn
module load gcc
module load openmpi
source activate test

srun -C gpu -t 03:00:00 -G 1 dlprof --nsys_opts="-d 300 --capture-range cudaProfilerApi" python mpi_learn.py