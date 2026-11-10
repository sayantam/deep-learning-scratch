import numpy as np
from numpy import ndarray

from dls.foundations import sigmoid, permute_data, generate_batch


def init_weights(input_size: int, hidden_size: int) -> dict[str, ndarray]:
    return {
        "W1": np.random.randn(input_size, hidden_size),
        "B1": np.random.randn(1, hidden_size),
        "W2": np.random.randn(hidden_size, 1),
        "B2": np.random.randn(1, 1),
    }


def forward_loss(
    X: ndarray, y: ndarray, weights: dict[str, ndarray]
) -> tuple[dict[str, ndarray], float]:
    """
    Compute the forward pass and the loss for the step-by-step
    neural network model.

    :param X:
    :param y:
    :param weights:
    :return:
    """
    M1 = np.dot(X, weights["W1"])
    N1 = M1 + weights["B1"]
    O1 = sigmoid(N1)
    M2 = np.dot(O1, weights["W2"])
    P = M2 + weights["B2"]
    loss = np.mean((P - y) ** 2)
    forward_info = {
        "X": X,
        "M1": M1,
        "N1": N1,
        "O1": O1,
        "M2": M2,
        "P": P,
        "y": y,
    }
    return forward_info, loss


def loss_gradients(
    forward_info: dict[str, ndarray], weights: dict[str, ndarray]
) -> dict[str, ndarray]:
    """
    Compute the partial derivatives of the loss with respect to each of the parameters in the neural network.
    """
    dLdP = -(forward_info["y"] - forward_info["P"])

    dPdM2 = np.ones_like(forward_info["M2"])

    dLdM2 = dLdP * dPdM2

    dPdB2 = np.ones_like(weights["B2"])

    dLdB2 = (dLdP * dPdB2).sum(axis=0)

    dM2dW2 = np.transpose(forward_info["O1"], (1, 0))

    dLdW2 = np.dot(dM2dW2, dLdP)

    dM2dO1 = np.transpose(weights["W2"], (1, 0))

    dLdO1 = np.dot(dLdM2, dM2dO1)

    dO1dN1 = sigmoid(forward_info["N1"]) * (1 - sigmoid(forward_info["N1"]))

    dLdN1 = dLdO1 * dO1dN1

    dN1dB1 = np.ones_like(weights["B1"])

    dN1dM1 = np.ones_like(forward_info["M1"])

    dLdB1 = (dLdN1 * dN1dB1).sum(axis=0)

    dLdM1 = dLdN1 * dN1dM1

    dM1dW1 = np.transpose(forward_info["X"], (1, 0))

    dLdW1 = np.dot(dM1dW1, dLdM1)

    loss_gradients: dict[str, ndarray] = {
        "W2": dLdW2,
        "B2": dLdB2.sum(axis=0),
        "W1": dLdW1,
        "B1": dLdB1.sum(axis=0),
    }

    return loss_gradients


def predict(X: ndarray, weights: dict[str, ndarray]) -> ndarray:
    """
    Generate predictions from the step-by-step neural network model.
    """
    M1 = np.dot(X, weights["W1"])

    N1 = M1 + weights["B1"]

    O1 = sigmoid(N1)

    M2 = np.dot(O1, weights["W2"])

    P = M2 + weights["B2"]

    return P


def train(
    X_train: ndarray,
    y_train: ndarray,
    X_test: ndarray,
    y_test: ndarray,
    n_iter: int = 1000,
    test_every: int = 1000,
    learning_rate: float = 0.01,
    hidden_size=13,
    batch_size: int = 100,
    return_losses: bool = False,
    return_weights: bool = False,
    return_scores: bool = False,
    seed: int = 1,
):

    from sklearn.metrics import r2_score

    if seed:
        np.random.seed(seed)

    start = 0

    # Initialize weights
    weights = init_weights(X_train.shape[1], hidden_size=hidden_size)

    # Permute data
    X_train, y_train = permute_data(X_train, y_train)

    losses = []

    val_scores = []

    for i in range(n_iter):
        # Generate batch
        if start >= X_train.shape[0]:
            X_train, y_train = permute_data(X_train, y_train)
            start = 0

        X_batch, y_batch = generate_batch(X_train, y_train, start, batch_size)
        start += batch_size

        # Train net using generated batch
        forward_info, loss = forward_loss(X_batch, y_batch, weights)

        if return_losses:
            losses.append(loss)

        loss_grads = loss_gradients(forward_info, weights)
        for key in weights:
            weights[key] -= learning_rate * loss_grads[key]

        if return_scores:
            if i % test_every == 0 and i != 0:
                preds = predict(X_test, weights)
                val_scores.append(r2_score(preds, y_test))

    if return_weights:
        return losses, weights, val_scores

    return None
