#!/bin/bash

# Note the CUDA version from cudatoolkit (11.4)
module load PrgEnv-nvidia cudatoolkit python
module load cudnn/8.2.0
source /global/common/software/nersc/pm-2022q3/sw/python/3.9-anaconda-2021.11/etc/profile.d/conda.sh
conda activate dhenv
