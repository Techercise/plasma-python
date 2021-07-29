#!/bin/bash
#SBATCH -A CSC434_spock
#SBATCH -o frnn_1gpu_spock.out
#SBATCH -p batch
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive

export MPIR_CVAR_GPU_EAGER_DEVICE_MEM=0
export MPICH_GPU_SUPPORT_ENABLED=1

export HIPCC_COMPILE_FLAGS_APPEND="$HIPCC_COMPILE_FLAGS_APPEND -I${MPICH_DIR}/include -L${MPICH_DIR}/lib -lmpi -L/opt/cray/pe/mpich/8.1.4/gtl/lib -lmpi_gtl_hsa"

module swap PrgEnv-cray PrgEnv-gnu
module load rocm
module load cray-python
module load gcc
module load craype-accel-amd-gfx908
conda activate tf_rocm

srun python mpi_learn.py
