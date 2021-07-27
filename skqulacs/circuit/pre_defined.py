from .circuit import LearningCircuit
from skqulacs.qnn.qnnbase import _create_time_evol_gate
from typing import List
import numpy as np
import random


def create_ansatz(n_qubit: int, c_depth: int, time_step: float) -> LearningCircuit:
    """Create a circuit used in this page: https://dojo.qulacs.org/ja/latest/notebooks/5.2_Quantum_Circuit_Learning.html

    Examples:
        >>> n_qubit = 4
        >>> circuit = create_ansatz(n_qubit, 3, 0.5)
        >>> qnn = QNNRegressor(n_qubit, circuit)
        >>> qnn.fit(x_train, y_train)
    """

    def preprocess_x(x: List[float], index: int):
        xa = x[index % len(x)]
        return min(1, max(-1, xa))

    circuit = LearningCircuit(n_qubit)
    for i in range(n_qubit):
        # Capture copy of i by `i=i`.
        # Without this, i in lambda is a reference to the i, so the lambda always
        # recognize i as n_qubit - 1.
        circuit.add_input_RY_gate(i, lambda x, i=i: np.arcsin(preprocess_x(x, i)))
        circuit.add_input_RZ_gate(
            i, lambda x, i=i: np.arccos(preprocess_x(x, i) * preprocess_x(x, i))
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


def create_farhi_circuit(
    n_qubit: int, c_depth: int, time_step: float
) -> LearningCircuit:
    def preprocess_x(x: List[float], index: int):
        xa = x[index % len(x)]
        return min(1, max(-1, xa))

    circuit = LearningCircuit(n_qubit)
    for i in range(n_qubit):
        circuit.add_input_RY_gate(i, lambda x, i=i: np.arcsin(preprocess_x(x, i)))
        circuit.add_input_RZ_gate(
            i, lambda x, i=i: np.arccos(preprocess_x(x, i) * preprocess_x(x, i))
        )

    zyu = list(range(n_qubit))
    for _ in range(c_depth):
        random.shuffle(zyu)
        # 今回の回路はdepthを多めにとったほうがいいかも
        # 最低でもn_qubitはほしいかも
        for i in range(0, n_qubit - 1, 2):
            anglex = 2.0 * np.pi * np.random.rand()
            angley = 2.0 * np.pi * np.random.rand()
            circuit.add_CNOT_gate(zyu[i + 1], zyu[i])
            circuit.add_parametric_RX_gate(zyu[i], anglex)
            circuit.add_parametric_RY_gate(zyu[i], angley)
            circuit.add_CNOT_gate(zyu[i + 1], zyu[i])
            anglex = 2.0 * np.pi * np.random.rand()
            angley = 2.0 * np.pi * np.random.rand()
            circuit.add_parametric_RY_gate(zyu[i], -angley)
            circuit.add_parametric_RX_gate(zyu[i], -anglex)
    return circuit
