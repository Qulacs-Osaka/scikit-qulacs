from typing import List, Tuple
from skqulacs.circuit import LearningCircuit
import numpy as np
from numpy.random import default_rng
from skqulacs.qnn import QNNRegressor
from skqulacs.qnn.qnnbase import _create_time_evol_gate
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def create_circuit(n_qubit: int, c_depth: int, time_step: float) -> LearningCircuit:
    def preprocess_x(x: List[float], index: int):
        xa = x[index % len(x)]
        return min(1, max(-1, xa))

    circuit = LearningCircuit(n_qubit)
    for i in range(n_qubit):
        circuit.add_input_RY_gate(i, lambda x: np.arcsin(preprocess_x(x, i)))
        circuit.add_input_RZ_gate(
            i, lambda x: np.arccos(preprocess_x(x, i) * preprocess_x(x, i))
        )

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
    mag_noise = 0.001
    y_train += mag_noise * rng.random(num_x)
    return x_train, y_train


def test_noisy_sine_two_vars():
    x_min = -0.5
    x_max = 0.5
    num_x = 50
    x_train, y_train = generate_noisy_sine_two_vars(x_min, x_max, num_x)

    n_qubit = 4
    depth = 3
    time_step = 0.5
    circuit = create_circuit(n_qubit, depth, time_step)
    qnn = QNNRegressor(n_qubit, circuit)
    qnn.fit(x_train, y_train, maxiter=1000)

    x_test, y_test = generate_noisy_sine_two_vars(x_min, x_max, num_x)
    y_pred = qnn.predict(x_test)
    loss = mean_squared_error(y_pred, y_test)
    assert loss < 0.1


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


def test_noisy_sine():
    x_min = -1.0
    x_max = 1.0
    num_x = 50
    x_train, y_train = generate_noisy_sine(x_min, x_max, num_x)

    n_qubit = 3
    depth = 3
    time_step = 0.5
    circuit = create_circuit(n_qubit, depth, time_step)
    qnn = QNNRegressor(n_qubit, circuit)
    qnn.fit(x_train, y_train, maxiter=500)

    x_test, y_test = generate_noisy_sine(x_min, x_max, num_x)
    y_pred = qnn.predict(x_test)
    loss = mean_squared_error(y_pred, y_test)
    assert loss < 0.1
    return x_test, y_test, y_pred


def main():
    x_test, y_test, y_pred = test_noisy_sine()
    plt.plot(x_test, y_test, "o", label="Test")
    plt.plot(x_test, y_pred, "o", label="Prediction")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
