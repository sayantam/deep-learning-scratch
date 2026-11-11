import os
import pickle
from types import SimpleNamespace

from numpy import ndarray


def prep_data(
    data: ndarray,
    target: ndarray,
    features: ndarray,
    test_size=0.3,
    random_state=80718,
):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    s = StandardScaler()
    data = s.fit_transform(data)
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=test_size, random_state=random_state
    )

    y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
    return SimpleNamespace(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        features=features,
    )


def boston_train_test_split():

    def load_boston():
        import numpy as np
        import pandas as pd

        data_url = "http://lib.stat.cmu.edu/datasets/boston"
        raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
        data_ = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target_ = raw_df.values[1::2, 2]
        feature_names = np.array(
            [
                "CRIM",
                "ZN",
                "INDUS",
                "CHAS",
                "NOX",
                "RM",
                "AGE",
                "DIS",
                "RAD",
                "TAX",
                "PTRATIO",
                "B",
                "LSTAT",
            ]
        )
        return SimpleNamespace(data=data_, target=target_, feature_names=feature_names)

    boston = load_boston()
    data = boston.data
    target = boston.target
    features = boston.feature_names
    return prep_data(data, target, features)


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def mnist_train_test_split():

    def to_namespace(mnist):
        data = mnist.data
        target = mnist.target.astype("int64")
        X_train, X_test = data[:60_000], data[60_000:]
        y_train, y_test = target[:60_000], target[60_000:]
        return SimpleNamespace(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

    def download():
        from sklearn.datasets import fetch_openml

        mnist = fetch_openml(
            "mnist_784", version=1, as_frame=False, parser="liac-arff"
        )
        with open(pkl_path, "wb") as f:
            pickle.dump(mnist, f)
        return to_namespace(mnist)

    pkl_path = os.path.join(MODULE_DIR, "mnist.pkl")
    # check if the pkl file exists
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            mnist = pickle.load(f)
            return to_namespace(mnist)
    return download()
