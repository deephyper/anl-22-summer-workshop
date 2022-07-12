import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gzip

import numpy as np

from utils import load_data_prepared

from deephyper.nas.metrics import r2, mse

import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.threads = True
mpi4py.rc.thread_level = "multiple"

from mpi4py import MPI

if not MPI.Is_initialized():
    MPI.Init_thread()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

gpu_per_node = 8
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
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(f"{e}") 

from deephyper.problem import HpProblem
from deephyper.search.hps import CBO
from deephyper.evaluator import Evaluator

n_components = 5
if gpu_local_idx == 0:
    load_data_prepared(
        n_components=n_components
    )


# Baseline LSTM Model
def build_and_train_model(config: dict, n_components: int = 5, verbose: bool = 0):
    tf.keras.utils.set_random_seed(42)

    default_config = {
        "lstm_units": 128,
        "activation": "tanh",
        "recurrent_activation": "sigmoid",
        "learning_rate": 1e-3,
        "batch_size": 64,
        "dropout_rate": 0,
        "num_layers": 1,
        "epochs": 20,
    }
    default_config.update(config)

    (X_train, y_train), (X_valid, y_valid), _, _ = load_data_prepared(
        n_components=n_components
    )

    layers = []
    for _ in range(default_config["num_layers"]):
        lstm_layer = tf.keras.layers.LSTM(
            default_config["lstm_units"],
            activation=default_config["activation"],
            recurrent_activation=default_config["recurrent_activation"],
            return_sequences=True,
        )
        dropout_layer = tf.keras.layers.Dropout(default_config["dropout_rate"])
        layers.extend([lstm_layer, dropout_layer])

    model = tf.keras.Sequential(
        [tf.keras.Input(shape=X_train.shape[1:])]
        + layers
        + [tf.keras.layers.Dense(n_components)]
    )

    if verbose:
        model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=default_config["learning_rate"])
    model.compile(optimizer, "mse", metrics=[])

    history = model.fit(
        X_train,
        y_train,
        epochs=default_config["epochs"],
        batch_size=default_config["batch_size"],
        validation_data=(X_valid, y_valid),
        verbose=verbose,
    ).history

    return model, history


def filter_failures(df):
    if df.objective.dtype != np.float64:
        df = df[~df.objective.str.startswith("F")]
        df = df.astype({"objective": float})
    return df


# Hyperparameter optimization with DeepHyper
    # Hyperparameter search space definition
problem = HpProblem()
problem.add_hyperparameter((10, 256), "units", default_value=128)
problem.add_hyperparameter(["sigmoid", "tanh", "relu"], "activation", default_value="tanh")
problem.add_hyperparameter(["sigmoid", "tanh", "relu"], "recurrent_activation", default_value="sigmoid")
problem.add_hyperparameter((1e-5, 1e-2, "log-uniform"), "learning_rate", default_value=1e-3)
problem.add_hyperparameter((2, 64), "batch_size", default_value=64)
problem.add_hyperparameter((0.0, 0.5), "dropout_rate", default_value=0.0)
problem.add_hyperparameter((1, 3), "num_layers", default_value=1)
problem.add_hyperparameter((10, 100), "epochs", default_value=20)

    # Definition of the function to optimize (configurable model to train)
def run(config):
    # important to avoid memory exploision
    tf.keras.backend.clear_session()
    
    _, history = build_and_train_model(config, n_components=n_components, verbose=0)

    return -history["val_loss"][-1]


    # Definition of an MPI Evaluator xecution of a Bayesian optimization search
if __name__ == "__main__":
    with Evaluator.create(
            run,
            method="mpicomm",
        ) as evaluator:
            if evaluator is not None:
                print(f"Creation of the Evaluator done with {evaluator.num_workers} worker(s)")

                # Search creation
                print("Creation of the search instance...")
                search = CBO(
                    problem,
                    evaluator,
                    initial_points=[problem.default_configuration],
                    log_dir="cbo-results",
                    random_state=42
                )
                print("Creation of the search done")

                # Search execution
                print("Starting the search...")
                results = search.search(timeout=540)
                print("Search is done")

                results.to_csv(os.path.join("cbo-results", "results.csv"))

                results = filter_failures(results)

                i_max = results.objective.argmax()
                best_config = results.iloc[i_max][:-4].to_dict()

                best_model, best_history = build_and_train_model(best_config, n_components=n_components, verbose=1)

                scores = {"MSE": mse, "R2": r2}

                (X_train, y_train), (X_valid, y_valid), (X_test, y_test), _ = load_data_prepared(
                    n_components=n_components
                )

                for metric_name, metric_func in scores.items():
                    print(f"Metric {metric_name}")
                    y_pred = best_model.predict(X_train)
                    score_train = np.mean(metric_func(y_train, y_pred).numpy())

                    y_pred = best_model.predict(X_valid)
                    score_valid = np.mean(metric_func(y_valid, y_pred).numpy())

                    y_pred = best_model.predict(X_test)
                    score_test = np.mean(metric_func(y_test, y_pred).numpy())

                    print(f"train: {score_train:.4f}")
                    print(f"valid: {score_valid:.4f}")
                    print(f"test : {score_test:.4f}")