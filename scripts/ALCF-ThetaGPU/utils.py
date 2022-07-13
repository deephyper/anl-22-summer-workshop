import gzip
import os
import urllib.request

import numpy as np
import progressbar
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))


class URLRetrieveProgressBar():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            print(total_size)
            self.pbar=progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


def load_sst_data():
    """Load the STT raw data.

    Returns:
        train_data, test_data, mask: the training data, testing data and surface mask respectively.
    """

    files = {
        "mask": "mask.npy",
        "train": "sst_var_train.npy",
        "test": "sst_var_test.npy",
        }

    sources = {
        "mask": "https://anl.box.com/shared/static/jif3tza2h3gjiuuq2dmezg95bbe3i5th",
        "train": "https://anl.box.com/shared/static/65jcr8bzid3mus4grtfitjt8q8sdnfpo",
        "test": "https://anl.box.com/shared/static/oy9fg3fj1mthlub0jyfc8zvkc65rudw6"
    }

    files = {k:os.path.join(HERE, v) for k,v in files.items()}

    data = {}
    for name, fpath in files.items():

        if not(os.path.exists(fpath)):
            urllib.request.urlretrieve(sources[name], files[name], URLRetrieveProgressBar())
        
        data[name] = np.load(files[name], allow_pickle=True).data

    return data["train"], data["test"], data["mask"]


def prepare_as_seq2seq(data, input_horizon=8, output_horizon=8):

    total_size = data.shape[0] - (input_horizon + output_horizon)  # Limit of sampling
    input_seq = []
    output_seq = []

    for t in range(0, total_size):
        input_seq.append(data[t : t + input_horizon, :])
        output_seq.append(
            data[t + input_horizon : t + input_horizon + output_horizon, :]
        )

    X = np.asarray(input_seq)  # [Samples, timesteps, state length]
    y = np.asarray(output_seq)  # [Samples, timesteps, state length]

    return X, y


def load_data_prepared(n_components=5, input_horizon=8, output_horizon=8):

    cached_data = f"processed_data_{n_components}_{input_horizon}_{output_horizon}.npz"

    if not (os.path.exists(cached_data)):
        train_data, test_data, _ = load_sst_data()

        # flatten the data
        train_data_flat = train_data.reshape(train_data.shape[0], -1)
        test_data_flat = test_data.reshape(test_data.shape[0], -1)

        train_data_flat = np.concatenate(
            [train_data_flat, test_data_flat[:700]], axis=0
        )
        test_data_flat = test_data_flat[700:]

        # dimensionality reduction
        preprocessor = Pipeline(
            # [("pca", PCA(n_components=n_components)), ("standard", StandardScaler())]
            [("pca", PCA(n_components=n_components)), ("minmax", MinMaxScaler())]
        )
        train_data_reduc = preprocessor.fit_transform(train_data_flat)
        test_data_reduc = preprocessor.transform(test_data_flat)

        X_train, y_train = prepare_as_seq2seq(
            train_data_reduc, input_horizon, output_horizon
        )
        X_test, y_test = prepare_as_seq2seq(
            test_data_reduc, input_horizon, output_horizon
        )

        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # Save
        data = {
            "train": (X_train, y_train),
            "valid": (X_valid, y_valid),
            "test": (X_test, y_test),
            "preprocessor": preprocessor,
        }
        with gzip.GzipFile(cached_data, "w") as f:
            np.save(file=f, arr=data, allow_pickle=True)
    else:
        # Load (dict)
        with gzip.GzipFile(cached_data, "rb") as f:
            data = np.load(f, allow_pickle=True).item()

        X_train, y_train = data["train"]
        X_valid, y_valid = data["valid"]
        X_test, y_test = data["test"]
        preprocessor = data["preprocessor"]

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test), preprocessor
