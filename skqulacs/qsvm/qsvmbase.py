import numpy as np
from qulacs import QuantumState
from qulacs.gate import CNOT, RZ, H


def get_qvec(x, n_qubit, tlotstep):
    # xはデータ
    # n_qubit,tlotstepはそのままの意味
    data_state = QuantumState(n_qubit)
    data_state.set_zero_state()
    for a in range(n_qubit):
        H(a).update_quantum_state(data_state)
    for tlotkai in range(tlotstep):
        for a in range(n_qubit):
            RZ(a, x[a] / tlotstep).update_quantum_state(data_state)
            # aとa+1のゲートの交互作用
            b = (a + 1) % n_qubit
            CNOT(a, b).update_quantum_state(data_state)
            RZ(b, (np.pi - x[a]) * (np.pi - x[b]) / tlotstep).update_quantum_state(
                data_state
            )
            CNOT(a, b).update_quantum_state(data_state)
    for a in range(n_qubit):
        H(a).update_quantum_state(data_state)
    for tlotkai in range(tlotstep):
        for a in range(n_qubit):
            RZ(a, x[a] / tlotstep).update_quantum_state(data_state)
            # aとa+1のゲートの交互作用
            b = (a + 1) % n_qubit
            CNOT(a, b).update_quantum_state(data_state)
            RZ(b, (np.pi - x[a]) * (np.pi - x[b]) / tlotstep).update_quantum_state(
                data_state
            )
            CNOT(a, b).update_quantum_state(data_state)
    # 000の行のベクトルを取る
    return data_state
