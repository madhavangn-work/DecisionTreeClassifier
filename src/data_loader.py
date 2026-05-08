from pathlib import Path

import pandas as pd


def load_iris_features_and_labels(data_path: str | Path):
    data = pd.read_csv(data_path, header=None)
    features = data.drop(data.columns[4], axis=1).values
    labels = pd.factorize(data[4])[0]
    return features, labels
