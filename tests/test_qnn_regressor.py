from typing import List
from skqulacs.circuit import LearningCircuit
import numpy as np
import random
from numpy.random import RandomState
from skqulacs.qnn import QNNRegressor
from skqulacs.qnn.qnnbase import _create_time_evol_gate
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def func_to_learn(x):
    return np.sin(x[0] * x[1] * np.pi)


def generate_noisy_sine(x_min: float, x_max: float, num_x: int):
    seed = 0
    random_state = RandomState(seed)

    x_train = []
    y_train = []
    for _ in range(num_x):
        xa = x_min + (x_max - x_min) * random.random()
        xb = x_min + (x_max - x_min) * random.random()
        x_train.append([xa, xb])
        y_train.append(func_to_learn([xa, xb]))
        # 2要素だと量子的な複雑さが足りず、　精度が悪いため、ダミーの2bitを加えて4bitにしている。
    mag_noise = 0.0005
    y_train += mag_noise * random_state.randn(num_x)
    return x_train, y_train


def create_circuit(n_qubit: int, c_depth: int, time_step: float) -> LearningCircuit:
    def preprocess_x(x: List[float], index: int):
        xa = x[index % len(x)]
        return min(1, max(-1, xa))

    circuit = LearningCircuit(n_qubit)
    for i in range(n_qubit):
        circuit.add_input_RX_gate(i, lambda x: np.arcsin(preprocess_x(x, i)))
        circuit.add_input_RZ_gate(i, lambda x: np.arccos(preprocess_x(x, i) * preprocess_x(x, i)))

    time_evol_gate = _create_time_evol_gate(n_qubit, time_step)
    for _ in range(c_depth):
        circuit.add_gate(time_evol_gate)
        for i in range(n_qubit):
            angle = 2.0 * np.pi * np.random.rand()
            circuit.add_parametric_RX_gate(i, angle)
            angle = 2.0 * np.pi * np.random.rand()
            circuit.add_parametric_RZ_gate(i, angle)
            angle = 2.0 * np.pi * np.random.rand()
            circuit.add_parametric_RX_gate(i, angle)
    return circuit


def test_noisy_sine():
    x_min = -0.5
    x_max = 0.5
    num_x = 100
    x_train, y_train = generate_noisy_sine(x_min, x_max, num_x)

    n_qubit = 4
    depth = 3
    time_step = 0.5
    circuit = create_circuit(n_qubit, depth, time_step)
    qnn = QNNRegressor(n_qubit, circuit)
    qnn.fit(x_train, y_train, maxiter=1000)

    x_test, y_test = generate_noisy_sine(x_min, x_max, num_x)
    y_pred = qnn.predict(x_test)
    loss = mean_squared_error(y_pred, y_test)
    # for i in range(len(x_test)):
    #     print([x_test[i][0], x_test[i][1], y_test[i], y_pred[i]])
    assert loss < 0.05


# 2要素のQNNを試してみる
# sin(x1*x2)をフィッティングさせる
def main():
    test_noisy_sine()

    qnn = QNNRegressor(circuit)
    _, theta = qnn.fit(x_train, y_train, maxiter=1000)


if __name__ == "__main__":
    main()
