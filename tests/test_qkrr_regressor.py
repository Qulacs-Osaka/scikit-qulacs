import random

import numpy as np
from numpy.random import RandomState
from sklearn.metrics import mean_squared_error

from skqulacs.circuit import create_ibm_embedding_circuit
from skqulacs.qkrr import QKRR


def func_to_learn(x):
    return np.sin(x[0] * x[1] * 2)


def generate_noisy_sine(x_min: float, x_max: float, num_x: int):
    seed = 0
    random.seed(seed)
    random_state = RandomState(seed)
    x_train = []
    y_train = []
    for _ in range(num_x):
        xa = x_min + (x_max - x_min) * random.random()
        xb = x_min + (x_max - x_min) * random.random()
        x_train.append([xa, xb])
        y_train.append(func_to_learn([xa, xb]))
        mag_noise = 0.05
    y_train += mag_noise * random_state.randn(num_x)
    return x_train, y_train


def test_noisy_sine():
    x_min = -0.5
    x_max = 0.5
    num_x = 300
    num_test = 100
    x_train, y_train = generate_noisy_sine(x_min, x_max, num_x)
    x_test, y_test = generate_noisy_sine(x_min, x_max, num_test)
    n_qubit = 6
    circuit = create_ibm_embedding_circuit(n_qubit)
    qkrr = QKRR(circuit, n_iteration=100)
    print(qkrr.n_iteration)
    qkrr.fit(x_train, y_train)
    y_pred = qkrr.predict(x_test)
    print(y_pred)
    loss = mean_squared_error(y_pred, y_test)
    print(loss)
    assert loss < 0.008


test_noisy_sine()
