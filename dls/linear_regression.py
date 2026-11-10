import numpy as np
from numpy import ndarray

from dls.foundations import permute_data, generate_batch


def forward_linear_regression(
    X_batch: ndarray, y_batch: ndarray, weights: dict[str, ndarray]
) -> tuple[float, dict[str, ndarray]]:
    """
    Forward pass for the step-by-step linear regression.
    :param X_batch:
    :param y_batch:
    :param weights:
    :return:
    """
    # assert batch sizes of X and y are equal
    assert X_batch.shape[0] == y_batch.shape[0]

    # assert that matrix multiplication can work
    assert X_batch.shape[1] == weights["W"].shape[0]

    # assert that B is simply a 1x1 ndarray
    assert weights["B"].shape == (1, 1)

    # compute the operations on the forward pass
    N = np.dot(X_batch, weights["W"])

    P = N + weights["B"]

    loss = np.mean(np.power(y_batch - P, 2))

    # save the information computed on the forward pass
    forward_info: dict[str, ndarray] = {
        "X": X_batch,
        "N": N,
        "P": P,
        "y": y_batch,
    }

    return loss, forward_info


def loss_gradients(
    forward_info: dict[str, ndarray], weights: dict[str, ndarray]
) -> dict[str, ndarray]:
    """
    Compute dLdW and dLdB for the step-by-step linear regression model.

    :param forward_info:
    :param weights:
    :return:
    """
    batch_size = forward_info["X"].shape[0]
    dLdP = 2 * (forward_info["P"] - forward_info["y"])
    dpDN = np.ones_like(forward_info["N"])
    dNdW = np.transpose(forward_info["X"], (1, 0))
    dPdB = np.ones_like(weights["B"])

    dLdN = dLdP * dpDN
    # need to use matrix multiplication here,
    # with dNdW on the left
    dLdW = np.dot(dNdW, dLdN)
    # need to sum along dimension representing the batch size
    dLdB = (dLdP * dPdB).sum(axis=0)

    return {
        "W": dLdW,
        "B": dLdB,
    }


def init_weights(n_in: int) -> dict[str, ndarray]:
    """
    Initialize weights on first forward pass of model.

    :param n_in:
    :return:
    """

    return {"W": np.random.randn(n_in, 1), "B": np.random.randn(1, 1)}


def train(
    X: ndarray,
    y: ndarray,
    n_iter: int = 1000,
    learning_rate: float = 0.01,
    batch_size: int = 100,
    return_losses: bool = False,
    return_weights: bool = False,
    seed: int = 1,
):
    """Train model for a certain number of epochs."""
    if seed:
        np.random.seed(seed)
    start = 0

    # initialize the weights
    weights = init_weights(X.shape[1])

    # permute data
    X, y = permute_data(X, y)

    if return_losses:
        losses = []

    for _ in range(n_iter):
        # generate batch
        if start >= X.shape[0]:
            X, y = permute_data(X, y)
            start = 0

        X_batch, y_batch = generate_batch(X, y, start, batch_size)
        start += batch_size

        # Train net using generated batch
        loss, forward_info = forward_linear_regression(X_batch, y_batch, weights)

        if return_losses:
            losses.append(loss)

        loss_grads = loss_gradients(forward_info, weights)
        for key in weights:
            weights[key] -= learning_rate * loss_grads[key]

    if return_weights:
        return losses, weights

    return None


def predict(X: ndarray, weights: dict[str, ndarray]):
    """Generate predictions from the step-by-step linear regression model."""
    N = np.dot(X, weights["W"])
    return N + weights["B"]
