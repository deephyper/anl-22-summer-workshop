#!/bin/bash
#BSUB -nnodes 4
#BSUB -W 2:00
#BSUB -q debug
#BSUB -P fus145
#BSUB -N
#BSUB -B

# User Configuration
INIT_SCRIPT=$PWD/load_modules.sh

NUM_NODES=4
RANKS_PER_NODE=6

# Initialization of environment
source $INIT_SCRIPT

jsrun -n $(( $NUM_NODES * $RANKS_PER_NODE )) -r1 -g6 -a1 -c42 -bpacked:42 python evaluator_mpi.py


echo "Complete"
