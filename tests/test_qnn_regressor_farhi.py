import pytest
from typing import List, Tuple
from skqulacs.circuit import create_farhi_circuit
import numpy as np
from numpy.random import default_rng
from skqulacs.qnn import QNNRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def sine_two_vars(x: List[float]) -> float:
    return np.sin(np.pi * x[0] * x[1])


def generate_noisy_sine_two_vars(
    x_min: float, x_max: float, num_x: int
) -> Tuple[List[List[float]], List[float]]:
    rng = default_rng(0)
    x_train = [
        [rng.uniform(x_min, x_max), rng.uniform(x_min, x_max)] for _ in range(num_x)
    ]
    # 2要素だと量子的な複雑さが足りず、　精度が悪いため、ダミーの2bitを加えて4bitにしている。
    y_train = [sine_two_vars(x) for x in x_train]
    mag_noise = 0.01
    y_train += mag_noise * rng.random(num_x)
    return x_train, y_train


@pytest.mark.parametrize(("solver", "maxiter"), [("BFGS", 30), ("Adam", 30)])
def test_noisy_sine_two_vars(solver: str, maxiter: int):
    x_min = -0.5
    x_max = 0.5
    num_x = 70
    x_train, y_train = generate_noisy_sine_two_vars(x_min, x_max, num_x)

    n_qubit = 4
    depth = 6
    circuit = create_farhi_circuit(n_qubit, depth, 0)
    qnn = QNNRegressor(n_qubit, circuit, solver)
    qnn.fit(x_train, y_train, maxiter)
    # BFGSじゃないなら600
    x_test, y_test = generate_noisy_sine_two_vars(x_min, x_max, num_x)
    y_pred = qnn.predict(x_test)
    loss = mean_squared_error(y_pred, y_test)
    assert loss < 0.1
    return x_test, y_test, y_pred


def sine(x: float) -> float:
    return np.sin(np.pi * x)


def generate_noisy_sine(
    x_min: float, x_max: float, num_x: int
) -> Tuple[List[List[float]], List[float]]:
    rng = default_rng(0)
    x_train = [[rng.uniform(x_min, x_max)] for _ in range(num_x)]
    y_train = [sine(x[0]) for x in x_train]
    mag_noise = 0.01
    y_train += mag_noise * rng.random(num_x)
    return x_train, y_train


@pytest.mark.parametrize(("solver", "maxiter"), [("BFGS", 30), ("Adam", 30)])
def test_noisy_sine(solver: str, maxiter: int):
    x_min = -1.0
    x_max = 1.0
    num_x = 50
    x_train, y_train = generate_noisy_sine(x_min, x_max, num_x)

    n_qubit = 4
    depth = 11
    circuit = create_farhi_circuit(n_qubit, depth, 0)
    qnn = QNNRegressor(n_qubit, circuit, solver)
    qnn.fit(x_train, y_train, maxiter)
    x_test, y_test = generate_noisy_sine(x_min, x_max, num_x)
    y_pred = qnn.predict(x_test)
    loss = mean_squared_error(y_pred, y_test)
    assert loss < 0.03
    return x_test, y_test, y_pred


def main():
    x_test, y_test, y_pred = test_noisy_sine("BFGS", 50)
    plt.plot(x_test, y_test, "o", label="Test")
    plt.plot(x_test, y_pred, "o", label="Prediction")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
