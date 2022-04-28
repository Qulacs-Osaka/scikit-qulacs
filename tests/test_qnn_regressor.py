from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.random import default_rng
from numpy.typing import NDArray
from sklearn.metrics import mean_squared_error

from skqulacs.circuit import create_qcl_ansatz
from skqulacs.qnn import QNNRegressor
from skqulacs.qnn.solver import Adam, Bfgs, Solver


def sine_two_vars(x: List[float]) -> float:
    return np.sin(np.pi * x[0] * x[1])


def generate_noisy_sine_two_vars(
    x_min: float, x_max: float, num_x: int
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    rng = default_rng(0)
    x_train = np.array(
        [[rng.uniform(x_min, x_max), rng.uniform(x_min, x_max)] for _ in range(num_x)]
    )
    y_train = np.array([sine_two_vars(x) for x in x_train])
    mag_noise = 0.001
    y_train += mag_noise * rng.random(num_x)
    return x_train, y_train


@pytest.mark.parametrize(("solver", "maxiter"), [(Bfgs(), 20), (Adam(), 20)])
def test_noisy_sine_two_vars(solver: Solver, maxiter: int) -> None:
    x_min = -0.5
    x_max = 0.5
    num_x = 50
    x_train, y_train = generate_noisy_sine_two_vars(x_min, x_max, num_x)

    n_qubit = 4
    depth = 3
    time_step = 0.5
    circuit = create_qcl_ansatz(n_qubit, depth, time_step, 0)
    qnn = QNNRegressor(circuit, solver)
    qnn.fit(x_train, y_train, maxiter)

    x_test, y_test = generate_noisy_sine_two_vars(x_min, x_max, num_x)
    y_pred = qnn.predict(x_test)
    loss = mean_squared_error(y_pred, y_test)
    assert loss < 0.1
    return x_test, y_test, y_pred


def sine(x: float) -> float:
    return np.sin(np.pi * x)


def generate_noisy_sine(
    x_min: float, x_max: float, num_x: int
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    rng = default_rng(0)
    x_train = np.array([[rng.uniform(x_min, x_max)] for _ in range(num_x)])
    y_train = np.array([sine(x[0]) for x in x_train])
    mag_noise = 0.01
    y_train += mag_noise * rng.random(num_x)
    return x_train, y_train


@pytest.mark.parametrize(
    ("solver", "maxiter"),
    [(Bfgs(), 20), (Adam(tolerance=2e-4, n_iter_no_change=8), 777)],
)
def test_noisy_sine(solver: Solver, maxiter: int) -> None:
    x_min = -1.0
    x_max = 1.0
    num_x = 50
    x_train, y_train = generate_noisy_sine(x_min, x_max, num_x)

    n_qubit = 3
    depth = 3
    time_step = 0.5
    circuit = create_qcl_ansatz(n_qubit, depth, time_step, 0)
    qnn = QNNRegressor(circuit, solver)
    qnn.fit(x_train, y_train, maxiter)

    x_test, y_test = generate_noisy_sine(x_min, x_max, num_x)
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


if __name__ == "__main__":
    main()
