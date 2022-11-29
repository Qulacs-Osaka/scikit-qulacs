from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.random import default_rng
from numpy.typing import NDArray
from sklearn.metrics import mean_squared_error

from skqulacs.circuit import create_qcl_ansatz
from skqulacs.qnn import QNNRegressor
from skqulacs.qnn.solver import Adam, Bfgs, Solver


def generate_noisy_data(
    x_min: float,
    x_max: float,
    x_shape: Tuple[int, int],
    function: Callable[[NDArray[np.float_]], NDArray[np.float_]],
    seed: Optional[int] = 0,
    mag_noise: float = 0.001,
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    """Generate random input data and its output

    Args:
        x_min: Minimum limit of random x value.
        x_max: Maximum limit of random x value.
        x_shape: Shape of x, (batch, features).
        function: Function which generates output from x. It takes 1D feature vector and returns 1D output vector.
        seed: Seed for random value.
        mag_noise: Noise amplitude to be added to output.
    """
    rng = default_rng(seed)
    x_train = rng.uniform(x_min, x_max, x_shape)
    y_train = np.array([function(x) for x in x_train])
    y_train += mag_noise * rng.random(y_train.shape)
    return x_train, y_train


def two_vars_two_outputs(x: NDArray[np.float_]) -> NDArray[np.float_]:
    return np.array([2.0 * x[0] + x[1], 1.5 * x[0] - 3.0 * x[1]])


@pytest.mark.parametrize(("solver", "maxiter"), [(Bfgs(), 30), (Adam(), 30)])
def test_noisy_two_vars_two_outputs(solver: Solver, maxiter: int) -> None:
    x_min = -0.5
    x_max = 0.5
    num_x = 50
    x_train, y_train = generate_noisy_data(
        x_min, x_max, (num_x, 2), two_vars_two_outputs
    )

    n_qubit = 4
    depth = 3
    time_step = 0.5
    circuit = create_qcl_ansatz(n_qubit, depth, time_step, 0)
    qnn = QNNRegressor(circuit, solver)
    qnn.fit(x_train, y_train, maxiter)

    x_test, y_test = generate_noisy_data(x_min, x_max, (num_x, 2), two_vars_two_outputs)
    y_pred = qnn.predict(x_test)
    loss = mean_squared_error(y_pred, y_test)
    assert loss < 0.11
    return x_test, y_test, y_pred


def sine_two_vars(x: NDArray[np.float_]) -> NDArray[np.float_]:
    return np.sin(np.pi * x[0] * x[1])


@pytest.mark.parametrize(("solver", "maxiter"), [(Bfgs(), 20), (Adam(), 20)])
def test_noisy_sine_two_vars(solver: Solver, maxiter: int) -> None:
    x_min = -0.5
    x_max = 0.5
    num_x = 50
    x_train, y_train = generate_noisy_data(x_min, x_max, (num_x, 2), sine_two_vars)

    n_qubit = 4
    depth = 3
    time_step = 0.5
    circuit = create_qcl_ansatz(n_qubit, depth, time_step, 0)
    qnn = QNNRegressor(circuit, solver)
    qnn.fit(x_train, y_train, maxiter)

    x_test, y_test = generate_noisy_data(x_min, x_max, (num_x, 2), sine_two_vars)
    y_pred = qnn.predict(x_test)
    loss = mean_squared_error(y_pred, y_test)
    assert loss < 0.1
    return x_test, y_test, y_pred


def sine(x: NDArray[np.float_]) -> NDArray[np.float_]:
    return np.sin(np.pi * x[0])


@pytest.mark.parametrize(
    ("solver", "maxiter"),
    [(Bfgs(), 20), (Adam(tolerance=2e-4, n_iter_no_change=8), 777)],
)
def test_noisy_sine(solver: Solver, maxiter: int) -> None:
    x_min = -1.0
    x_max = 1.0
    num_x = 50
    x_train, y_train = generate_noisy_data(x_min, x_max, (num_x, 1), sine)

    n_qubit = 3
    depth = 3
    time_step = 0.5
    circuit = create_qcl_ansatz(n_qubit, depth, time_step, 0)
    qnn = QNNRegressor(circuit, solver)
    qnn.fit(x_train, y_train, maxiter)

    x_test, y_test = generate_noisy_data(x_min, x_max, (num_x, 1), sine)
    y_pred = qnn.predict(x_test)
    loss = mean_squared_error(y_pred, y_test)
    assert loss < 0.03
    return x_test, y_test, y_pred


def main() -> None:
    x_test, y_test, y_pred = test_noisy_sine(Bfgs(), 50)
    plt.plot(x_test, y_test, "o", label="Test")
    plt.plot(x_test, y_pred, "o", label="Prediction")
    plt.legend()
    plt.show()
    #plt.savefig("test_qnn_regressor.jpg")


if __name__ == "__main__":
    main()
