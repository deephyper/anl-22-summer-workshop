import time
import numpy as np
from deephyper.problem import HpProblem
from deephyper.evaluator import profile
import gzip
import matplotlib.pyplot as plt
import tensorflow as tf
from data.utils import load_sst_data, load_data_prepared
from sklearn.decomposition import PCA
from deephyper.nas.metrics import r2


## Define the HP problem 

problem = HpProblem()
problem.add_hyperparameter((10, 256), "units", default_value=128)
# problem.add_hyperparameter(["sigmoid", "tanh", "relu"], "activation", default_value="tanh") ## This is not cuDNN compatible
# problem.add_hyperparameter(["sigmoid", "tanh", "relu"], "recurrent_activation", default_value="sigmoid") ## This is not cuDNN compatible
problem.add_hyperparameter((1e-5, 1e-2, "log-uniform"), "learning_rate", default_value=1e-3)
problem.add_hyperparameter((2, 64), "batch_size", default_value=64)
problem.add_hyperparameter((0.0, 0.5), "dropout_rate", default_value=0.0)
problem.add_hyperparameter((1, 3), "num_layers", default_value=1)

from common import RUN_SLEEP


def basic_sleep():
    time.sleep(RUN_SLEEP)


def cpu_bound():
    t = time.time()
    duration = 0
    while duration < RUN_SLEEP:
        sum(i * i for i in range(10**7))
        duration = time.time() - t


def IO_bound():
    with open("/dev/urandom", "rb") as f:
        t = time.time()
        duration = 0
        while duration < RUN_SLEEP:
            f.read(100)
            duration = time.time() - t


## Define the training model
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
    }
    default_config.update(config)

    (X_train, y_train), (X_valid, y_valid), _, _ = load_data_prepared(
        n_components=n_components
    )

    layers = []
    for _ in range(default_config["num_layers"]):
        
        # lstm_layer = tf.keras.layers.LSTM(
        #     default_config["lstm_units"],
        #     activation=default_config["activation"],
        #     recurrent_activation=default_config["recurrent_activation"],
        #     return_sequences=True,
        # ) ## This is not cuDNN compatible
        
        lstm_layer = tf.keras.layers.LSTM(
            default_config["lstm_units"],
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=True,
            recurrent_dropout=0.0,
            use_bias=True
        ) ## This is cuDNN compatible
        
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
        epochs=80,
        batch_size=default_config["batch_size"],
        validation_data=(X_valid, y_valid),
        verbose=verbose,
    ).history

    return model, history




@profile
def run(config):
    # important to avoid memory exploision
    tf.keras.backend.clear_session()
    
    n_components = 5
    
    _, history = build_and_train_model(config, n_components=n_components, verbose=0)

    return -history["val_loss"][-1]
