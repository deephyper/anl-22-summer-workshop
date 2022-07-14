Perlmutter (NERSC)
******************

Perlmutter, a HPE Cray supercomputer at NERSC, is a heterogeneous system with both GPU-accelerated and CPU-only nodes. Phase 1 of the installation is made up of 12 GPU-accelerated cabinets housing over 1,500 nodes. Phase 2 adds 12 CPU cabinets with more than 3,000 nodes. Each GPU node of Perlmutter has 4x NVIDIA A100 GPUs. 


Connect to Perlmutter
=====================

For connecting to Perlmutter, check `documentation <https://docs.nersc.gov/systems/perlmutter/#connecting-to-perlmutter>`_. One can also configure SSH according to the `instructions <https://docs.nersc.gov/connect/mfa/#ssh-configuration-file-options>`_. To connect to Perlmutter via terminal, use:

.. code-block:: console

    $ ssh <username>@perlmutter-p1.nersc.gov


DeepHyper Installation
======================

After logging in Perlmutter, use the `installation script <https://github.com/deephyper/anl-22-summer-workshop/blob/main/scripts/NERSC-Perlmutter/dh_install.sh>`_ provided to install DeepHyper and the associated dependencies. Download the file and run ``source dh_install.sh`` on the terminal. 

The script first loads the Perlmutter modules, including cuDNN. 

.. code-block:: console

    $ module load PrgEnv-nvidia cudatoolkit python
    $ module load cudnn/8.2.0

Next, we create a conda environment and install DeepHyper. 

.. code-block:: console

    $ conda create -n dh python=3.9 -y
    $ conda activate dh
    $ conda install gxx_linux-64 gcc_linux-64


The crucial step is to install CUDA aware mpi4py, following the instructions given in the `mpi4py documentation <https://docs.nersc.gov/development/languages/python/using-python-perlmutter/#building-cuda-aware-mpi4py>`_

.. code-block:: console

    $ MPICC="cc -target-accel=nvidia80 -shared" CC=nvc CFLAGS="-noswitcherror" pip install --force --no-cache-dir --no-binary=mpi4py mpi4py

Finally we install deephyper and other packages. 

.. code-block:: console

    $ pip install deephyper==0.4.1
    $ pip install tensorflow
    $ pip install kiwisolver
    $ pip install cycler



Running the installed DeepHyper
===============================

Once DeepHyper is installed, one can use the deephyper after loading the modules and activating the conda environment. For the LSTM example for SST data, first copy and Paste the following scripts `load_modules.sh <https://github.com/deephyper/anl-22-summer-workshop/blob/main/scripts/NERSC-Perlmutter/load_modules.sh>`_, `common.py <https://github.com/deephyper/anl-22-summer-workshop/blob/main/scripts/NERSC-Perlmutter/common.py>`_, `evaluator_mpi.py <https://github.com/deephyper/anl-22-summer-workshop/blob/main/scripts/NERSC-Perlmutter/evaluator_mpi.py>`_,  `sst.py <https://github.com/deephyper/anl-22-summer-workshop/blob/main/scripts/NERSC-Perlmutter/sst.py>`_ and  `job_submit.sh <https://github.com/deephyper/anl-22-summer-workshop/blob/main/scripts/NERSC-Perlmutter/job_submit.sh>`_ on your folder on Perlmutter. 

Details of these scripts are shown in Part 3 of the workshop
 
 
Using Jupyter notebook on Perlmutter (Not demonstrated in the workshop)
====================================

NERSC also allows for launching jupyter kernel on Perlmutter. One can visit `jupyter.nersc.gov <https://jupyter.nersc.gov/>`_ and select Exclusive GPU node or a configurable GPU node (up to 4 GPU nodes, with 4 GPUs each). 
 
