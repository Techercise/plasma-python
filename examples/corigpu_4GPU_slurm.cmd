#!/bin/bash
#SBATCH -C gpu
#SBATCH -t 02:00:00
#SBATCH -G 4
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH -c 4
#SBATCH --exclusive

module load cgpu
module load python
module load cuda
module load cudnn
module load gcc
module load openmpi
source activate test

# rm /global/cscratch1/sd/$USER/model_checkpoints/*
# rm /global/cscratch1/sd/$USER/results/*
# rm /global/cscratch1/sd/$USER/csv_logs/*
# rm /global/cscratch1/sd/$USER/Graph/*
# rm /global/cscratch1/sd/$USER/normalization/*

srun -C gpu -t 03:00:00 -G 4 -N 1 --ntasks-per-node=4 --ntasks-per-socket=2 -c 4 --exclusive  python mpi_learn.py
