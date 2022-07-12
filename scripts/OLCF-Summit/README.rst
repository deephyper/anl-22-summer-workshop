Summit (OLCF)
******************

`Summit <https://www.olcf.ornl.gov/summit/>`_ (or OLCF-4) is an IBM AC922 supercomputer located at Oak Ridge Leadership Computing Facility (OLCF). Launched in mid-2018, it is a 200 PFLOP system consisting of 4,608 nodes linked with EDR Infiniband. Each compute node contains 2x IBM POWER9 sockets and 6x NVIDIA V100 GPUs linked with NVLINK. It currently occupies spot 4 on the latest `Top500 <https://www.top500.org/system/179397/>`_ list (June 2022).

We hope to add a DeepHyper tutorial on OLCF Frontier in future iterations of this repository. 


Aside: POWER9 CPUs
======================

A distinguishing feature of the Summit CPUs is their use of a reduced instruction set computer (RISC) instruction set architecture versus the complex CISC x86 ISA that dominates HPC (e.g. the AMD CPUs on ThetaGPU and Perlmutter). One downside to this choice is that many library dependencies that have widely-available precompiled binaries for x86 do not offer precompiled binaries for POWER9, e.g. on the default Anaconda channels.

To ameliorate these pains, IBM, Summit's main vendor and leader of the OpenPOWER Foundation, maintained the"Watson Machine Learning Community Edition" (WML-CE) Anaconda channel for several years, which contained binaries for TensorFlow, PyTorch, and many other libraries. In 2019, the Power ISA was open-sourced and the OpenPOWER Foundation became a subsidiary of the Linux Foundation. 

In 2021, the Open Cognitive Environment Community Edition (Open-CE) effectively replaced WML-CE on `Summit <https://docs.olcf.ornl.gov/software/analytics/ibm-wml-ce.html>`_ and in the broader POWER9 `user community <https://community.ibm.com/community/user/hybriddatamanagement/blogs/christopher-sullivan/2021/06/16/open-cognitive-environment-open-ce-a-valuable-tool>`_. We will use the Open-CE modules on Summit as the basis for our DeepHyper installation.


Connect to Summit
=====================

For detailed instructions for connecting to OLCF resources for the first time, consult the `documentation <https://docs.olcf.ornl.gov/connecting/index.html#connecting-to-olcf>`_. A SecurID fob is required, and SSH multiplexing is disabled. From a terminal:

.. code-block:: console

    $ ssh <username>@summit.olcf.ornl.gov


DeepHyper Installation
======================

After logging in Perlmutter, use the `installation script <https://github.com/nesar/DeepHyperSwing/blob/main/saul/dh_install.sh>`_ provided to install DeepHyper and the associated dependencies. Download the file and run ``source dh_install.sh`` on the terminal. 

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

Once DeepHyper is installed, one can use the deephyper after loading the modules and activating the conda environment. For the LSTM example for SST data, first copy and Paste the following scripts `load_modules.sh <https://github.com/nesar/DeepHyperSwing/blob/main/saul/load_modules.sh>`_, `common.py <https://github.com/nesar/DeepHyperSwing/blob/main/saul/common.py>`_, `evaluator_mpi.py <https://github.com/nesar/DeepHyperSwing/blob/main/saul/evaluator_mpi.py>`_,  `sst.py <https://github.com/nesar/DeepHyperSwing/blob/main/saul/sst.py>`_ and  `job_submit.sh <https://github.com/nesar/DeepHyperSwing/blob/main/saul/job_submit.sh>`_ on your folder on Perlmutter. 


 
 
Using Jupyter notebook on Perlmutter
====================================

NERSC also allows for launching jupyter kernel on Perlmutter. One can visit `jupyter.nersc.gov <https://jupyter.nersc.gov/>`_ and select Exclusive GPU node or a configurable GPU node (up to 4 GPU nodes, with 4 GPUs each). 
 
