import pytest
from skqulacs.circuit.pre_defined import create_farhi_watle,create_farhi_circuit
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
    y_train = [sine_two_vars(x) for x in x_train]
    mag_noise = 0.02
    y_train += mag_noise * rng.random(num_x)
    return x_train, y_train

x_min = -1
x_max = 1
num_x = 70
x_train, y_train = generate_noisy_sine_two_vars(x_min, x_max, num_x)

n_qubit = 6
depth = 6
circuit = create_farhi_watle(n_qubit, depth, 0)
qnn = QNNRegressor(n_qubit, circuit, "BFGS")
qnn.fit(x_train, y_train, 29)
# BFGSじゃないなら600
x_test, y_test = generate_noisy_sine_two_vars(x_min, x_max, num_x)
y_pred = qnn.predict(x_test)
loss = mean_squared_error(y_pred, y_test)
print(loss)
assert loss < 0.1

