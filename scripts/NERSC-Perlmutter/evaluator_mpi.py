'''
The script for mpi-based evaluator

References:
## https://deephyper.readthedocs.io/en/latest/tutorials/tutorials/scripts/03_Evaluators/README.html
## https://github.com/deephyper/tutorials/blob/main/tutorials/scripts/03_Evaluators/ackley.py
## https://deephyper.readthedocs.io/en/latest/tutorials/tutorials/alcf/03_ThetaGPU_mpi/README.html
'''

if __name__ == "__main__":
    from deephyper.evaluator import Evaluator
    from sst import run
    from common import evaluate_and_plot
    
    print('Begin evaluator mpi')


    import mpi4py
    mpi4py.rc.initialize = False
    mpi4py.rc.threads = True
    mpi4py.rc.thread_level = "multiple"
    
    
    ###########################
    
    from mpi4py import MPI

    if not MPI.Is_initialized():
        MPI.Init_thread()
    
    gpu_per_node = 4 # $RANKS_PER_NODE
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print( '  size: ' + str(size)  +  '  rank: ' + str(rank))
    gpu_local_idx = rank % gpu_per_node
    node = int(rank / gpu_per_node)

    
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[gpu_local_idx], "GPU")
            tf.config.experimental.set_memory_growth(gpus[gpu_local_idx], True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(f"[r={rank}]: {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(f"{e}")

     ## GPU worker management
    
    if rank == 0:
        # Evaluator creation
        print("Creation of the Evaluator...")

    
    ###########################
    # import tensorflow as tf
    
    # available_gpus = tf.config.list_physical_devices("GPU")
    n_gpus = len(gpus)
    
    # if n_gpus > 1:
    #     n_gpus -= 1
    #     tf.config.set_visible_devices(available_gpus[-1], "GPU")
    #     gpus = tf.config.list_physical_devices("GPU")
        
    is_gpu_available = n_gpus > 0

    if is_gpu_available:
        print(f"{n_gpus} GPU{'s are' if n_gpus > 1 else ' is'} available.")
    else:
        print("No GPU available")
    
    ###########################
    
    print("Creation of the Evaluator...")
    
    with Evaluator.create(
        run,
        method="mpicomm",
    ) as evaluator:
        if evaluator is not None:
            evaluate_and_plot(evaluator, "mpi_evaluator")
