#!/bin/bash
#SBATCH -A dasrepo_g
#SBATCH --job-name=dh_cbo_lstm
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH --nodes=128
#SBATCH --gres=gpu:4
# #SBATCH --gpus-per-task=2


# User Configuration
INIT_SCRIPT=$PWD/load_modules.sh

SLURM_JOBSIZE=128
RANKS_PER_NODE=4

# Initialization of environment
source $INIT_SCRIPT

srun -N $SLURM_JOBSIZE -n $(( $SLURM_JOBSIZE * $RANKS_PER_NODE )) python evaluator_mpi.py


echo "Complete"
