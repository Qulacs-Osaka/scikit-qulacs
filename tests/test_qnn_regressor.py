import numpy as np
from numpy.random import RandomState
from skqulacs.regressor import QNNRegressor
import matplotlib.pyplot as plt


def generate_noisy_sine(x_min: float, x_max: float, num_x: int):
    def func_to_learn(x):
        return np.sin(x * np.pi)

    seed = 0
    random_state = RandomState(seed)

    x_train = x_min + (x_max - x_min) * random_state.rand(num_x)
    y_train = func_to_learn(x_train)

    mag_noise = 0.05
    y_train += mag_noise * random_state.randn(num_x)
    return x_train, y_train


def test_noisy_sine():
    ########  パラメータ  #############
    n_qubit = 2  # qubitの数
    c_depth = 3  # circuitの深さ
    time_step = 0.50  # ランダムハミルトニアンによる時間発展の経過時間

    x_min = -1.0
    x_max = 1.0
    num_x = 50
    x_train, y_train = generate_noisy_sine(x_min, x_max, num_x)

    qnn = QNNRegressor(n_qubit, c_depth, time_step)
    _, theta = qnn.fit(x_train, y_train)

    x_list = np.arange(x_min, x_max, 0.02)
    y_pred = qnn.predict(theta, x_list)
