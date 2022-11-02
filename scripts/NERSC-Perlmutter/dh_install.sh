#!/bin/bash

# https://docs.neIrsc.gov/development/languages/python/using-python-perlmutter/#building-cuda-aware-mpi4py

# LOAD NERSC MODULES
module load PrgEnv-nvidia cudatoolkit python
module load cudnn/8.2.0
source /global/common/software/nersc/pm-2022q3/sw/python/3.9-anaconda-2021.11/etc/profile.d/conda.sh

# CREATING CONDA ENVIRONMENT
conda create -n dhenv python=3.9 -y
conda activate dhenv
conda install gxx_linux-64 gcc_linux-64

# BUILDING CUDA-AWARE MPI4PY
MPICC="cc -target-accel=nvidia80 -shared" CC=nvc CFLAGS="-noswitcherror" pip install --force --no-cache-dir --no-binary=mpi4py mpi4py

# INSTALLING OTHER DEPENDENCIES
pip install deephyper
pip install tensorflow==2.9.2
pip install kiwisolver
pip install cycler
pip install matplotlib
pip install progressbar2
pip install networkx[default]