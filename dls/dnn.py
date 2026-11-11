from copy import deepcopy
from typing import Protocol, runtime_checkable

import numpy as np
from numpy import ndarray

from dls.foundations import permute_data
from dls.model_error import mae, rmse


@runtime_checkable
class OperationWithDropout(Protocol):
    def do_output_with_inference(self, inference: bool = False) -> ndarray: ...


class Operation:
    """
    Base class for an operation in a deep neural network (DNN).
    """

    def __init__(self):
        """
        Initializes an Operation instance.
        """
        self.inputs = None
        self.output = None
        self.input_grad = None

    def forward(self, inputs: ndarray, inference: bool = False):
        self.inputs = inputs
        if isinstance(self, OperationWithDropout):
            self.output = self.do_output_with_inference(inference)
        else:
            self.output = self.do_output()
        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        assert self.output.shape == output_grad.shape
        self.input_grad = self.do_input_grad(output_grad)
        assert self.input_grad.shape == self.inputs.shape
        return self.input_grad

    def do_output(self) -> ndarray:
        raise NotImplementedError

    def do_input_grad(self, output_grad: ndarray) -> ndarray:
        raise NotImplementedError


class ParamOperation(Operation):
    """
    Base class for an operation in a deep neural network (DNN) that involves parameters (weights and biases).
    """

    def __init__(self, params: ndarray):
        super().__init__()
        self.params = params
        self.params_grad = None

    def backward(self, output_grad: ndarray) -> ndarray:
        assert self.output.shape == output_grad.shape
        self.input_grad = self.do_input_grad(output_grad)
        self.params_grad = self.do_params_grad(output_grad)
        assert self.input_grad.shape == self.inputs.shape
        assert self.params_grad.shape == self.params.shape
        return self.input_grad

    def do_params_grad(self, output_grad: ndarray) -> ndarray:
        raise NotImplementedError

    def do_assert_params(self):
        pass


class Loss:
    """
    Base class for a loss function in a deep neural network (DNN).
    """

    def __init__(self):
        self.prediction: ndarray
        self.target: ndarray
        self.input_grad: ndarray

    def forward(self, prediction: ndarray, target: ndarray) -> float:
        """
        Computes the loss value
        :param prediction:
        :param target:
        :return:
        """
        assert prediction.shape == target.shape
        self.prediction = prediction
        self.target = target
        loss_value = self.do_output()
        return loss_value

    def backward(self) -> ndarray:
        """
        Computes the gradient of the loss with respect to the input to the loss function.
        :return:
        """
        self.input_grad = self.do_input_grad()
        assert self.prediction.shape == self.input_grad.shape
        return self.input_grad

    def do_input_grad(self) -> ndarray:
        raise NotImplementedError()

    def do_output(self) -> float:
        raise NotImplementedError()


class Optimizer:
    def __init__(
            self,
            learning_rate: float = 0.01,
            final_lr: float = 0,
            decay_type: str | None = None,
    ):
        """
        Base class for optimizers.

        :param learning_rate: learning rate
        :param final_lr: final learning rate
        :param decay_type: decay type
        """
        self.learning_rate = learning_rate
        self.final_lr = final_lr
        self.decay_type = decay_type
        self.decay_per_epoch: float

    def setup_decay(self, max_epoch: int):
        if self.decay_type is None:
            return
        if self.decay_type == "exponential":
            self.decay_per_epoch = np.power(
                self.final_lr / self.learning_rate, 1.0 / (max_epoch - 1)
            )
            return
        if self.decay_type == "linear":
            self.decay_per_epoch = (self.learning_rate - self.final_lr) / (
                    max_epoch - 1
            )

    def decay_learning_rate(self):
        if self.decay_type is None:
            return
        if self.final_lr <= 0.0:
            return
        if self.decay_type == "exponential":
            self.learning_rate *= self.decay_per_epoch
            return
        if self.decay_type == "linear":
            self.learning_rate -= self.decay_per_epoch

    def step_(self):
        pass


class WeightMultiply(ParamOperation):
    """
    Weight multiplication operation for a neural network.
    """

    def do_output(self) -> ndarray:
        return np.dot(self.inputs, self.params)

    def do_input_grad(self, output_grad: ndarray) -> ndarray:
        return np.dot(output_grad, self.params.T)

    def do_params_grad(self, output_grad: ndarray) -> ndarray:
        return np.dot(self.inputs.T, output_grad)


class BiasAdd(ParamOperation):
    """
    Compute bias addition.
    """

    def do_assert_params(self):
        assert self.params.shape[0] == 1

    def do_output(self) -> ndarray:
        return self.inputs + self.params

    def do_input_grad(self, output_grad: ndarray) -> ndarray:
        return np.ones_like(self.inputs) * output_grad

    def do_params_grad(self, output_grad: ndarray) -> ndarray:
        params_grad = np.ones_like(self.params) * output_grad
        return np.sum(params_grad, axis=0).reshape(1, params_grad.shape[1])


class Sigmoid(Operation):
    """
    Sigmoid activation function.
    """

    def do_output(self) -> ndarray:
        return 1.0 / (1.0 + np.exp(-1.0 * self.inputs))

    def do_input_grad(self, output_grad: ndarray) -> ndarray:
        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad


class Linear(Operation):
    """
    Linear activation function.
    """

    def do_output(self) -> ndarray:
        return self.inputs

    def do_input_grad(self, output_grad: ndarray) -> ndarray:
        return output_grad


class Layer:
    """
    A "layer" of neurons in a neural network.

    :param neurons: The number of neurons in the layer.
    """

    def __init__(self, neurons: int):
        self.neurons = neurons
        self.first = True
        self.params: list[ndarray] = []
        self.params_grad: list[ndarray] = []
        self.operations: list[Operation] = []
        self.inputs: ndarray
        self.output: ndarray

    def do_setup_layer(self, inputs: ndarray):
        raise NotImplementedError("do_setup_layer must be implemented by subclasses.")

    def forward(self, inputs: ndarray, inference: bool = False):
        if self.first:
            self.do_setup_layer(inputs)
            self.first = False
        self.inputs = inputs
        self.output = inputs
        for operation in self.operations:
            self.output = operation.forward(self.output, inference=inference)
        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        assert self.output is not None
        assert self.output.shape[0] == output_grad.shape[0]
        input_grad = output_grad
        for operation in reversed(self.operations):
            input_grad = operation.backward(input_grad)
        self.update_params_grad()
        return input_grad

    def update_params_grad(self):
        self.params_grad = []
        for operation in self.operations:
            if isinstance(operation, ParamOperation):
                self.params_grad.append(operation.params_grad)

    def update_params(self):
        self.params = []
        for operation in self.operations:
            if isinstance(operation, ParamOperation):
                self.params.append(operation.params)


class Dense(Layer):
    """
    Fully connected layer.
    """

    def __init__(
            self,
            neurons: int,
            activation: Operation | None = None,
            weight_init: str = "standard",
            dropout: float = 1.0,
    ):
        super().__init__(neurons)
        self.activation = activation or Sigmoid()
        self.weight_init = weight_init
        self.dropout = dropout
        self.seed: int

    def do_setup_layer(self, inputs: ndarray):
        if self.seed:
            np.random.seed(self.seed)
        num_in = inputs.shape[1]

        if self.weight_init == "glorot":
            scale = np.sqrt(
                2 / (num_in + self.neurons)
            )  # self.neurons is same as num_out
        else:
            scale = 1.0
        self.params = []
        # weights
        self.params.append(
            np.random.normal(
                loc=0.0,
                scale=scale,
                size=(num_in, self.neurons),
            )
        )
        # bias
        self.params.append(
            np.random.normal(loc=0.0, scale=scale, size=(1, self.neurons))
        )

        self.operations = [
            WeightMultiply(params=self.params[0]),
            BiasAdd(params=self.params[1]),
            self.activation,
        ]

        if self.dropout < 1.0:
            self.operations.append(Dropout(keep_prob=self.dropout))


class MeanSquaredError(Loss):
    def __init__(self, normalize: bool = False):
        super().__init__()
        self.normalize = normalize

    def do_output(self) -> float:
        if self.normalize:
            self.prediction = self.prediction / self.prediction.sum(
                axis=1, keepdims=True
            )
        loss = (
                np.sum(np.square(self.prediction - self.target)) / self.prediction.shape[0]
        )
        return loss

    def do_input_grad(self) -> ndarray:
        return 2 * (self.prediction - self.target) / self.prediction.shape[0]


class NeuralNetwork:
    def __init__(
            self,
            layers: list[Layer],
            loss: Loss,
            learning_rate: float = 0.01,
            seed: int = 1,
    ):
        self.layers = layers
        self.loss = loss
        self.seed = seed
        if seed:
            for layer in self.layers:
                layer.seed = self.seed

    def forward(self, x_batch: ndarray, inference: bool = False) -> ndarray:
        x_out = x_batch
        for layer in self.layers:
            x_out = layer.forward(x_out, inference=inference)
        return x_out

    def backward(self, loss_grad: ndarray):
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train_batch(self, x_batch: ndarray, y_batch: ndarray) -> float:
        """
        Passes data forward through the layers.
        Computes the loss.
        Passes data backward through the layers.

        :param x_batch:
        :param y_batch:
        :return:
        """
        predictions = self.forward(x_batch)
        loss = self.loss.forward(predictions, y_batch)
        self.backward(self.loss.backward())
        return loss

    def params(self):
        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        for layer in self.layers:
            yield from layer.params_grad


class NetworkAwareOptimizer(Protocol):
    net: NeuralNetwork


class SGD(Optimizer):
    def __init__(self, *args, **kwargs):
        """Stochastic Gradient Descent optimizer."""
        super().__init__(*args, **kwargs)
        self._net: NeuralNetwork

    @property
    def net(self):
        return self._net

    @net.setter
    def net(self, net_value):
        self._net = net_value

    def step_(self):
        """
        For each parameter, adjust in the appropriate direction, with the magnitude of the adjustment
        based on the learning rate.
        """
        assert self.net is not None, "Network must be set before calling step"
        for param, param_grad in zip(self.net.params(), self.net.param_grads()):
            param -= self.learning_rate * param_grad


class Trainer:
    """Trains a neural network"""

    def __init__(self, net: NeuralNetwork, optim: Optimizer):
        self.optim = optim
        self.init_net(net)
        self.best_loss = 1e9
        self.net: NeuralNetwork

    def init_net(self, net: NeuralNetwork):
        self.net = net
        if hasattr(self.optim, "net"):
            net_aware: NetworkAwareOptimizer | Optimizer = self.optim
            net_aware.net = self.net

    def generate_batch(self, X: ndarray, y: ndarray, size: int = 32) -> tuple[ndarray]:
        assert X.shape[0] == y.shape[0]
        N = X.shape[0]
        for ii in range(0, N, size):
            yield X[ii: ii + size], y[ii: ii + size]  # X_batch, y_batch

    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            X_test: ndarray,
            y_test: ndarray,
            epochs: int = 100,
            eval_every: int = 10,
            batch_size: int = 32,
            seed: int = 1,
            restart: bool = True,
            early_stopping: bool = True,
    ):
        """
        Fits the neural network on the training data for a certain number of
        epochs. Every "eval_every" epochs, it evaluates the neural network on
        the testing data.
        """
        # setup learning rate decay
        self.optim.setup_decay(max_epoch=epochs)

        np.random.seed(seed)
        if restart:
            for layer in self.net.layers:
                layer.first = True
            self.best_loss = 1e9
        last_model: NeuralNetwork | None = None
        for epoch in range(epochs):
            if (epoch + 1) % eval_every == 0:
                last_model = deepcopy(self.net)
            X_train, y_train = permute_data(X_train, y_train)
            # generate batches to train the network
            batch_generator = self.generate_batch(X_train, y_train, batch_size)
            for ii, (X_batch, y_batch) in enumerate(batch_generator):
                self.net.train_batch(X_batch, y_batch)
                self.optim.step_()
            # evaluate the loss every epoch interval
            if (epoch + 1) % eval_every == 0:
                test_preds = self.net.forward(X_test, inference=True)
                loss = self.net.loss.forward(prediction=test_preds, target=y_test)
                if early_stopping:
                    if loss < self.best_loss:
                        print(f"Epoch {epoch + 1}: loss = {loss:.3f}")
                        self.best_loss = loss
                    else:
                        print(
                            f"""Loss increased after epoch {epoch + 1}, final loss was {self.best_loss:.3f}, using the model from epoch {epoch + 1 - eval_every}"""
                        )
                        assert last_model is not None
                        self.init_net(last_model)
                        break
                else:
                    print(f"Validation loss after {epoch + 1} epochs is {loss:.3f}")
            # decay learning rate every epoch
            self.optim.decay_learning_rate()


class Dropout(Operation):
    def __init__(self, keep_prob: float = 0.8):
        super().__init__()
        self.keep_prob = keep_prob
        self.mask = None

    def do_output_with_inference(self, inference: bool = False) -> ndarray:
        if inference:
            return self.inputs * self.keep_prob
        else:
            self.mask = np.random.binomial(1, self.keep_prob, size=self.inputs.shape)
            return self.inputs * self.mask

    def do_input_grad(self, output_grad: ndarray) -> ndarray:
        return output_grad * self.mask


def eval_regression_model(model: NeuralNetwork, X_test: ndarray, y_test: ndarray):
    """
    Compute mae and rmse for a neural network.
    """
    preds = model.forward(X_test).reshape(-1, 1)
    print(f"Mean absolute error: {mae(preds, y_test):.2f}")
    print()
    print(f"Root mean squared error {rmse(preds, y_test):.2f}")
