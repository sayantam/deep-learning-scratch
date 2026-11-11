import numpy as np
from numpy import ndarray


def mae(preds: ndarray, actuals: ndarray):
    """
    Compute mean absolute error.
    """
    return np.mean(np.abs(preds - actuals))


def rmse(preds: ndarray, actuals: ndarray):
    """
    Compute root mean squared error.
    """
    return np.sqrt(np.mean(np.power(preds - actuals, 2)))


def calc_accuracy_model(model, test_set, y_test):
    print(
        """The model validation accuracy is: {:.2f}%""".format(
            np.equal(np.argmax(model.forward(test_set), axis=1), y_test).sum()
            * 100.0
            / test_set.shape[0]
        )
    )
