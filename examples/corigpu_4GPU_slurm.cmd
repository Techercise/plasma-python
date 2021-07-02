#!/bin/bash
#SBATCH -C gpu
#SBATCH -t 02:00:00
#SBATCH -G 4
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH -c 4
#SBATCH --mem-per-cpu=0

module load cgpu
module load python
source activate test
module load cuda
module load cudnn
module load gcc
module load openmpi

# rm /global/cscratch1/sd/$USER/model_checkpoints/*
# rm /global/cscratch1/sd/$USER/results/*
# rm /global/cscratch1/sd/$USER/csv_logs/*
# rm /global/cscratch1/sd/$USER/Graph/*
# rm /global/cscratch1/sd/$USER/normalization/*

mpirun -N 4 python mpi_learn.py
