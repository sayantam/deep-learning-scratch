import numpy as np
from numpy import ndarray
from scipy import special

from dls.dnn import Loss


def softmax(x: np.ndarray, axis=None) -> np.ndarray:
    return np.exp(x - special.logsumexp(x, axis=axis, keepdims=True))


class SoftmaxCrossEntropyLoss(Loss):
    """
    A Softmax "softly" maximizes the highest probability of the class (a vector),
    and reduces the lower probabilities of the other classes. This helps to create
    a steeper gradient, which can lead to faster convergence during training for
    classification tasks.

    Cross-entropy loss is a measure of the difference between the predicted probability
    distribution and the true distribution. It is commonly used in machine learning
    for classification tasks, where the goal is to predict the probability of each
    class for a given input.

    Combining softmax and cross-entropy loss produces steeper gradients than MSE.
    """

    def __init__(self, eps: float = 1e-9):
        super().__init__()
        self.eps = eps
        self.single_output = False
        self.softmax_preds: ndarray

    def do_input_grad(self) -> ndarray:
        return self.softmax_preds - self.target

    def do_output(self) -> float:
        # applying the softmax function to each row (observation)
        softmax_preds = softmax(self.prediction, axis=1)
        # clipping the softmax output to prevent numeric instability
        self.softmax_preds = np.clip(softmax_preds, self.eps, 1 - self.eps)
        # actual loss computation
        softmax_cross_entropy_loss = -1.0 * self.target * np.log(self.softmax_preds) - (
            1.0 - self.target
        ) * np.log(1 - self.softmax_preds)
        return np.sum(softmax_cross_entropy_loss)
