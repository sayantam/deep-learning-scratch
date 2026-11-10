from typing import Callable, List

import numpy as np
from numpy import ndarray

# A Function takes in an ndarray as an argument and produces an ndarray
Array_Function = Callable[[ndarray], ndarray]

# A Chain is a list of functions
Chain = List[Array_Function]


def chain_length_2(chain: Chain, x: ndarray) -> ndarray:
    """
    Evaluates two functions in a row, in a "Chain".
    """
    assert len(chain) == 2, "Length of input 'chain' should be 2"

    f1 = chain[0]
    f2 = chain[1]

    return f2(f1(x))


def chain_length_3(chain: Chain, x: ndarray) -> ndarray:
    """
    Evaluates three functions in a row, in a "Chain".
    """
    assert len(chain) == 3, "Length of input 'chain' should be 3"

    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]

    return f3(f2(f1(x)))


def square(x: ndarray) -> ndarray:
    """
    Returns the square of an array.
    :param x:
    :return:
    """
    return np.power(x, 2)


def leaky_relu(x: ndarray) -> ndarray:
    """
    Applies the Leaky ReLU activation function.
    :param x:
    :return:
    """
    return np.maximum(0.2 * x, x)


def deriv(
    func: Callable[[ndarray], ndarray], input_: ndarray, delta: float = 0.001
) -> ndarray:
    """
    Evaluates the derivative of the given function at every element of the input array.
    :param func:
    :param input_:
    :param delta:
    :return:
    """
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)


def sigmoid(x: ndarray) -> ndarray:
    """
    Apply the sigmoid function to each element of the input array
    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def chain_deriv_2(chain: Chain, input_range: ndarray) -> ndarray:
    """
    Uses the chain rule to compute the derivative of two nested functions:
    d/dx [f2(f1(x))] = (df2/du)(f1(x)) * (df1/dx)(x), where u = f1(x)

    :param chain:
    :param input_range:
    :return:
    """
    assert len(chain) == 2, "This function requires 'Chain' objects to be of length 2"
    assert input_range.ndim == 1, "Function requires a 1-dim ndarray as input_range"

    f1 = chain[0]
    f2 = chain[1]
    # f1(x)
    f1_of_x = f1(input_range)
    # df1/du
    df1dx = deriv(f1, input_range)
    # df2/du(f1(x))
    df2du = deriv(f2, f1_of_x)
    return df2du * df1dx


def chain_deriv_3(chain: Chain, input_range: ndarray) -> ndarray:
    """
    Uses the chain rule to compute the derivative of three nested functions:
    d/dx [f3(f2(f1(x)))] = (df3/dv)(f2(f1(x))) * (df2/du)(f1(x)) * (df1/dx)(x), where u = f1(x), and v = f2(f1(x))

    :param chain:
    :param input_range:
    :return:
    """
    assert len(chain) == 3, "This function requires Chain object to have length 3"
    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]

    df1dx = deriv(f1, input_range)
    u = f1(input_range)
    df2du = deriv(f2, u)
    v = f2(u)
    df3dv = deriv(f3, v)
    return df3dv * df2du * df1dx


def plot_chain(ax, chain: Chain, input_range: ndarray, length: int = 2) -> None:
    """
    Plots a chain function - a function made up of
    multiple consecutive ndarray -> ndarray mappings -
    Across the input_range

    ax: matplotlib Subplot for plotting
    """

    assert input_range.ndim == 1, (
        "Function requires a 1 dimensional ndarray as input_range"
    )
    if length == 3:
        output_range = chain_length_3(chain, input_range)
    else:
        output_range = chain_length_2(chain, input_range)
    ax.plot(input_range, output_range)


def plot_chain_deriv(ax, chain: Chain, input_range: ndarray, length: int = 2) -> None:
    """
    Uses the chain rule to plot the derivative of a function consisting of two nested functions.

    ax: matplotlib Subplot for plotting
    """
    if length == 3:
        output_range = chain_deriv_3(chain, input_range)
    else:
        output_range = chain_deriv_2(chain, input_range)
    ax.plot(input_range, output_range)


def matmul_forward(X: ndarray, W: ndarray) -> ndarray:
    """
    Computes the forward pass of a matrix multiplication.
    :param X:
    :param W:
    :return:
    """
    assert X.shape[1] == W.shape[0], """
        For matrix multiplication, the number of columns in the first array should
        match the number of rows in the second; instead the number of columns in the
        first array is {0} and the number of rows in the second array is {1}.
        """.format(X.shape[1], W.shape[0])

    N = np.dot(X, W)
    return N


def matmul_backward_first(X: ndarray, W: ndarray) -> ndarray:
    """
    Computes the backward pass of a matrix multiplication.
    :param X:
    :param W:
    :return:
    """
    dNdX = np.transpose(W, (1, 0))
    return dNdX


def matrix_forward_extra(X: ndarray, W: ndarray, sigma: Array_Function) -> ndarray:
    """
    Computes the forward pass of a function involving matrix multiplication,
    one extra function.
    """
    assert X.shape[1] == W.shape[0]

    # matrix multiplication
    N = np.dot(X, W)

    # feeding the output of the matrix multiplication through sigma
    S = sigma(N)

    return S


def matrix_function_backward_1(
    X: ndarray, W: ndarray, sigma: Array_Function
) -> ndarray:
    """
    Computes the derivative of our matrix function with respect to
    the first element.
    :param X:
    :param W:
    :param sigma:
    :return:
    """
    assert X.shape[1] == W.shape[0]

    # matrix multiplication
    N = np.dot(X, W)

    # feeding the output of the matrix multiplication through sigma
    S = sigma(N)

    # backward calculation
    dSdN = deriv(sigma, N)

    dNdX = np.transpose(W, (1, 0))

    # multiply them together; since dNdX is 1x1 here, order doesn't matter
    return np.dot(dSdN, dNdX)


def matrix_function_forward_sum(X: ndarray, W: ndarray, sigma: Array_Function) -> float:
    """
    Computing the result of the forward pass of this function with
    input ndarrays X and W and function sigma.
    :param X:
    :param W:
    :param sigma:
    :return:
    """
    assert X.shape[1] == W.shape[0]

    # matrix multiplication
    N = np.dot(X, W)

    # feeding the output of the matrix multiplication through sigma
    S = sigma(N)

    # sum all the elements
    L = np.sum(S)

    return L


def matrix_function_backward_sum_1(
    X: ndarray, W: ndarray, sigma: Array_Function
) -> ndarray:
    """
    Compute derivative of matrix function with a sum with respect to the
    first matrix input.
    """
    assert X.shape[1] == W.shape[0]

    # matrix multiplication
    N = np.dot(X, W)

    # feeding the output of the matrix multiplication through sigma
    S = sigma(N)

    # sum all the elements
    L = np.sum(S)

    # note: I'll refer to the derivatives by their quantities here,
    # unlike the math, where we referred to their function names

    # dLdS - just 1s
    dLdS = np.ones_like(S)

    # dSdN
    dSdN = deriv(sigma, N)

    # dLdN
    dLdN = dLdS * dSdN

    # dNdX
    dNdX = np.transpose(W, (1, 0))

    # dLdX
    dLdX = np.dot(dLdN, dNdX)

    return dLdX


def permute_data(X: ndarray, y: ndarray):
    """
    Permute X and y, using the same permutation, along axis=0
    """
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]


Batch = tuple[ndarray, ndarray]


def generate_batch(
    X: ndarray, y: ndarray, start: int = 0, batch_size: int = 10
) -> Batch:
    """
    Generate batch from X and y, given a start position
    """
    assert X.ndim == y.ndim == 2, "X and Y must be 2 dimensional"

    if start + batch_size > X.shape[0]:
        batch_size = X.shape[0] - start

    X_batch, y_batch = X[start : start + batch_size], y[start : start + batch_size]

    return X_batch, y_batch
