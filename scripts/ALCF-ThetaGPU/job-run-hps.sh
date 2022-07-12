#!/bin/bash
#COBALT -q full-node
#COBALT -n 1
#COBALT -t 60
#COBALT -A $PROJECT_NAME
#COBALT --attrs filesystems=home,grand,eagle,theta-fs0
#COBALT -O job-run-hps

# Nodes Configuration
COBALT_JOBSIZE=1
RANKS_PER_NODE=8

# Initialization of environment
. /etc/profile
    # Tensorflow optimized for A100 with CUDA 11
module load conda/2022-07-01
    # Activate conda env
conda activate build/dhenv

# Execute python script
mpirun -x LD_LIBRARY_PATH -x PYTHONPATH -x PATH -n $(( $COBALT_JOBSIZE * $RANKS_PER_NODE )) -N $RANKS_PER_NODE --hostfile $COBALT_NODEFILE python search.py