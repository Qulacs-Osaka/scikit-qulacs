from .circuit import LearningCircuit
from ..qnn.qnnbase import _create_time_evol_gate
from typing import List, Optional
from numpy.random import default_rng
import numpy as np
import random


def create_ansatz(
    n_qubit: int, c_depth: int, time_step: float, seed: Optional[int] = None
) -> LearningCircuit:
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

    rng = default_rng(seed)
    time_evol_gate = _create_time_evol_gate(n_qubit, time_step)
    for _ in range(c_depth):
        circuit.add_gate(time_evol_gate)
        for i in range(n_qubit):
            angle = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RX_gate(i, angle)
            angle = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RZ_gate(i, angle)
            angle = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RX_gate(i, angle)
    return circuit


def create_farhi_circuit(
    n_qubit: int, c_depth: int, seed: Optional[int] = None
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
    rng = default_rng(seed)
    for _ in range(c_depth):
        rng.shuffle(zyu)
        # 今回の回路はdepthを多めにとったほうがいいかも
        # 最低でもn_qubitはほしいかも
        for i in range(0, n_qubit - 1, 2):
            angle_x = 2.0 * np.pi * rng.random()
            angle_y = 2.0 * np.pi * rng.random()
            circuit.add_CNOT_gate(zyu[i + 1], zyu[i])
            circuit.add_parametric_RX_gate(zyu[i], angle_x)
            circuit.add_parametric_RY_gate(zyu[i], angle_y)
            circuit.add_CNOT_gate(zyu[i + 1], zyu[i])
            angle_x = 2.0 * np.pi * rng.random()
            angle_y = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RY_gate(zyu[i], -angle_y)
            circuit.add_parametric_RX_gate(zyu[i], -angle_x)
    return circuit


def create_defqsv(n_qubit: int, tlotstep: int = 4) -> LearningCircuit:
    def preprocess_x(x: List[float], index: int) -> float:
        xa = x[index % len(x)]
        return xa

    circuit = LearningCircuit(n_qubit)
    for i in range(n_qubit):
        circuit.add_H_gate(i)

    for tlotkai in range(tlotstep):
        for i in range(n_qubit):
            j = (i + 1) % n_qubit
            circuit.add_input_RZ_gate(i, lambda x, i=i: preprocess_x(x, i) / tlotstep)
            circuit.add_CNOT_gate(i, j)
            circuit.add_input_RZ_gate(
                j,
                lambda x, i=i: (
                    (np.pi - preprocess_x(x, i)) * (np.pi - preprocess_x(x, j))
                )
                / tlotstep,
            )
            circuit.add_CNOT_gate(i, j)

    for i in range(n_qubit):
        circuit.add_H_gate(i)

    for tlotkai in range(tlotstep):
        for i in range(n_qubit):
            j = (i + 1) % n_qubit
            circuit.add_input_RZ_gate(i, lambda x, i=i: preprocess_x(x, i) / tlotstep)
            circuit.add_CNOT_gate(i, j)
            circuit.add_input_RZ_gate(
                j,
                lambda x, i=i: (
                    (np.pi - preprocess_x(x, i)) * (np.pi - preprocess_x(x, j))
                )
                / tlotstep,
            )
            circuit.add_CNOT_gate(i, j)
    return circuit
