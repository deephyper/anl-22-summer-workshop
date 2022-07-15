#!/bin/bash

. /etc/profile
set -e

module load open-ce/1.5.0-py38-0
conda create -p dh --clone open-ce-1.5.0-py38-0 -y

# module load open-ce/1.5.2-py39-0
# conda create -p dh --clone open-ce-1.5.2-py39-0 -y

conda activate dh/

module load gcc

MPICC=mpicc pip install --force --no-cache-dir --no-binary=mpi4py mpi4py

# git clone https://github.com/mpi4py/mpi4py.git
# cd mpi4py/
# MPICC=mpicc python setup.py install

pip install deephyper==0.4.2
pip install matplotlib
