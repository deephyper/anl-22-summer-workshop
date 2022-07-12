ThetaGPU (ALCF)
******************

`ThetaGPU <https://www.alcf.anl.gov/theta>`_  is an extension of Theta and is comprised of 24 NVIDIA DGX A100 nodes at Argonne Leadership Computing Facility (ALCF). See the `documentation <https://argonne-lcf.github.io/ThetaGPU-Docs/>`_ of ThetaGPU from the Datascience group at Argonne National Laboratory for more information. The system documentation from the ALCF can be accessed `here <https://www.alcf.anl.gov/support-center/theta-gpu-nodes/getting-started-thetagpu>`_.

Connect to Theta
================

To connect to Theta via terminal, use:

.. code-block:: console

    $ ssh <username>@theta.alcf.anl.gov

Submitting jobs on ThetaGPU is then done from there using the command (make sure ``<submission_script>.sh`` is executable by performing ``chmod +x <submission_script>.sh``) :

.. code-block:: console

    $ qsub-gpu <submission_script>.sh


DeepHyper Installation
======================

As this procedure needs to be performed on ThetaGPU, we will directly execute it in this ``job-install-dhenv.sh`` submission script (replace the ``$PROJECT_NAME`` with the name of your project allocation, e-g: ``#COBALT -A datascience``):

.. literalinclude:: job-install-dhenv.sh
    :language: console
    :caption: **file**: ``job-install-dhenv.sh``

Submitting it with the following command :

.. code-block:: console
    
    $ qsub-gpu job-install-dhenv.sh

HPS Definition
==============

We will be using the problem from the `first notebook of the workshop <https://github.com/deephyper/anl-22-summer-workshop/blob/main/notebooks/1-Hyperparameter-Search.ipynb>`_, optimizing a surrogate geophysical model from data.

In this script we define the hyperparameter search space as well as run function, and feed it to a search instance using an MPI-based evaluator. 

.. literalinclude:: search.py
    :language: python
    :caption: **file**: ``search.py``

Executing the Search on ThetaGPU
================================

With the evaluator using MPI, we can simply use ``mpirun`` on ThetaGPU to launch it on all the gpus of every allocated node. This is what is done in this submission script (replace the ``$PROJECT_NAME`` with the name of your project allocation, e-g: ``#COBALT -A datascience``) :

.. literalinclude:: job-run-hps.sh
    :language: console
    :caption: **file**: ``job-run-hps.sh``

.. note::

    If you want to set the number of allocated nodes for the job to ``k``, make sure to change accordingly these two lines :
    
    .. code-block:: console

        #COBALT -n k
        COBALT_JOBSIZE=k