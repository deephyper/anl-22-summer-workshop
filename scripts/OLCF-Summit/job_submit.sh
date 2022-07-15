#!/bin/bash
#BSUB -nnodes 4
#BSUB -W 2:00
#BSUB -q debug
#BSUB -P fus145
#BSUB -J dh_cbo_lstm
#BSUB -N
#BSUB -B

. /etc/profile

# User Configuration
INIT_SCRIPT=$PWD/load_modules.sh

NUM_NODES=4
RANKS_PER_NODE=6

# Initialization of environment
source $INIT_SCRIPT
cd anl-22-summer-workshop/
jsrun -n $(( $NUM_NODES * $RANKS_PER_NODE )) -r6 -g1 -a1 -c1 python evaluator_mpi.py


echo "Complete"
