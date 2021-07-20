#!/bin/bash
#SBATCH -t 03:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:1
#SBATCH --output=FRNN_Poplar_1GPU.output
#SBATCH --exclusive

module load python3
module load openmpi
module load rocm
module load gcc
module load craype-accel-amd-gfx908
conda activate tf_rocm

# remove checkpoints for a benchmark run
#rm /tigress/$USER/model_checkpoints/*
#rm /tigress/$USER/results/*
#rm /tigress/$USER/csv_logs/*
#rm /tigress/$USER/Graph/*
#rm /tigress/$USER/normalization/*

export MV2_USE_CUDA=1
export MV2_ENABLE_AFFINITY=0
export OMP_NUM_THREADS=1
srun -c ${OMP_NUM_THREADS} python mpi_learn.py

# Submit the job using sbatch -p amdMI100 binary_file
